import os
import logging
import concurrent.futures
from typing import List
import fitz  # PyMuPDF

from app.models.schemas import DocumentChunk, DocumentMetadata
from app.services.vision import generate_image_caption
from app.services.storage import StorageService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

class DocumentParser:
    def __init__(self):
        self.storage = StorageService()

    def _process_single_image(self, img_bytes: bytes, ext: str, image_counter: int, page_no: int, file_hash: str):
        """Writes the raw image bytes to disk, sends to Groq, and uploads to R2 concurrently."""
        img_filename = f"uploads/{file_hash}_img_{image_counter}.{ext}"
        
        # Save the raw image bytes extracted by PyMuPDF
        with open(img_filename, "wb") as f:
            f.write(img_bytes)
            
        logger.info(f"[VISION] Extracted Image {image_counter} from Page {page_no}.")
        
        caption = ""
        image_path_saved = None
        
        # 1. Send to Groq Vision API
        try:
            logger.info(f"[VISION] Sending Image {image_counter} to Groq Vision API...")
            caption = generate_image_caption(img_filename)
            logger.info(f"[VISION] Received Caption: {caption[:75]}...")
        except Exception as e:
            logger.error(f"[VISION] FAIL: Vision API crashed for Image {image_counter}. Error: {e}")
        
        # 2. Upload to Cloudflare R2
        try:
            logger.info(f"[STORAGE] Uploading Image {image_counter} to Cloudflare R2...")
            public_img_url = self.storage.upload_file(img_filename, f"images/{file_hash}_img_{image_counter}.{ext}")
            image_path_saved = public_img_url
            logger.info(f"[STORAGE] SUCCESS! Image {image_counter} URL: {public_img_url}")
        except Exception as e:
            logger.error(f"[STORAGE] FAIL: Cloudflare R2 Upload crashed for Image {image_counter}. Error: {e}")
        
        # Cleanup
        if os.path.exists(img_filename):
            os.remove(img_filename)
            
        return caption, image_path_saved

    def parse_document(self, file_path: str, source_file_name: str, file_hash: str) -> List[DocumentChunk]:
        logger.info(f"Starting PyMuPDF high-speed parsing for {source_file_name}...")
        
        # Open the document instantly with PyMuPDF
        doc = fitz.open(file_path)
        results = []
        image_counter = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_image = {}
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_no = page_num + 1
                
                # --- 1. EXTRACT TEXT BLOCKS ---
                # get_text("blocks") returns natural paragraph groupings, perfect for RAG!
                blocks = page.get_text("blocks")
                for block in blocks:
                    # Block type 0 is text
                    if block[6] == 0: 
                        text_content = block[4].strip()
                        if len(text_content) > 20: # Ignore tiny artifacts or empty strings
                            metadata = DocumentMetadata(
                                source_file=source_file_name,
                                file_hash=file_hash,
                                page_number=page_no,
                                chunk_type="text",
                                image_path=None
                            )
                            results.append(DocumentChunk(text=text_content, metadata=metadata))

                # --- 2. EXTRACT IMAGES ---
                image_list = page.get_images(full=True)
                for img in image_list:
                    xref = img[0] # The internal PDF reference for the image
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]
                    
                    # Ignore tiny images like icons or 1x1 tracking pixels
                    if len(image_bytes) > 5000: 
                        image_counter += 1
                        future = executor.submit(self._process_single_image, image_bytes, ext, image_counter, page_no, file_hash)
                        future_to_image[future] = page_no

            # --- 3. PROCESS VISION RESULTS ---
            for future in concurrent.futures.as_completed(future_to_image):
                page_no = future_to_image[future]
                try:
                    caption, image_path_saved = future.result()
                    if caption:
                        # Append the Groq caption as its own text chunk so Cohere embeds it!
                        final_text = f"[Visual Content on Page {page_no}]: {caption}"
                        metadata = DocumentMetadata(
                            source_file=source_file_name,
                            file_hash=file_hash,
                            page_number=page_no,
                            chunk_type="image",
                            image_path=image_path_saved
                        )
                        results.append(DocumentChunk(text=final_text, metadata=metadata))
                except Exception as exc:
                    logger.error(f"[SYSTEM] Image thread exception: {exc}")

        logger.info(f"PyMuPDF finished. Extracted {len(results)} chunks and processed {image_counter} images.")
        return results