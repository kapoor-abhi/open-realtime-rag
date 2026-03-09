#parser.py
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
        img_filename = f"uploads/{file_hash}_img_{image_counter}.{ext}"
        
        with open(img_filename, "wb") as f:
            f.write(img_bytes)
            
        logger.info(f"[VISION] Extracted Image {image_counter} from Page {page_no}.")
        
        caption = ""
        image_path_saved = None
        
        # 1. Groq Vision API with Fallback
        try:
            logger.info(f"[VISION] Sending Image {image_counter} to Groq Vision API...")
            caption = generate_image_caption(img_filename)
        except Exception as e:
            logger.error(f"[VISION] API failed for Image {image_counter}. Using fallback. Error: {e}")
            # NEW: Fallback caption ensures the image is still indexed and retrievable by the UI!
            caption = f"A diagram, chart, or visual element located on page {page_no}."
        
        # 2. Upload to Cloudflare R2
        try:
            logger.info(f"[STORAGE] Uploading Image {image_counter} to Cloudflare R2...")
            public_img_url = self.storage.upload_file(img_filename, f"images/{file_hash}_img_{image_counter}.{ext}")
            image_path_saved = public_img_url
        except Exception as e:
            logger.error(f"[STORAGE] Cloudflare upload failed: {e}")
        
        if os.path.exists(img_filename):
            os.remove(img_filename)
            
        return caption, image_path_saved

    def parse_document(self, file_path: str, source_file_name: str, file_hash: str) -> List[DocumentChunk]:
        logger.info(f"Starting Contextual Block parsing for {source_file_name}...")
        
        doc = fitz.open(file_path)
        results = []
        image_counter = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_image = {}
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_no = page_num + 1
                
                # --- 1. EXTRACT TEXT USING LAYOUT BLOCKS ---
                blocks = page.get_text("blocks")
                
                # NEW: Page 1 Aggregation for Title/Author context
                if page_no == 1:
                    page_1_text = "\n".join([b[4].strip() for b in blocks if b[6] == 0 and len(b[4].strip()) > 5])
                    if page_1_text:
                        contextualized_text = f"Source Document: {source_file_name}\nPage: {page_no}\nDocument Summary & Metadata:\n{page_1_text}"
                        metadata = DocumentMetadata(
                            source_file=source_file_name, file_hash=file_hash, page_number=page_no, chunk_type="text"
                        )
                        results.append(DocumentChunk(text=contextualized_text, metadata=metadata))
                else:
                    # Standard block extraction for all other pages
                    for block in blocks:
                        if block[6] == 0: 
                            text_content = block[4].strip()
                            if len(text_content) > 20: 
                                # Inject document identity to prevent cross-contamination
                                contextualized_text = f"Source Document: {source_file_name}\nPage: {page_no}\nContent:\n{text_content}"
                                metadata = DocumentMetadata(
                                    source_file=source_file_name, file_hash=file_hash, page_number=page_no, chunk_type="text"
                                )
                                results.append(DocumentChunk(text=contextualized_text, metadata=metadata))

                # --- 2. EXTRACT IMAGES ---
                image_list = page.get_images(full=True)
                for img in image_list:
                    xref = img[0] 
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]
                    
                    if len(image_bytes) > 5000: 
                        image_counter += 1
                        future = executor.submit(self._process_single_image, image_bytes, ext, image_counter, page_no, file_hash)
                        future_to_image[future] = page_no

            # --- 3. PROCESS VISION RESULTS ---
            for future in concurrent.futures.as_completed(future_to_image):
                page_no = future_to_image[future]
                try:
                    caption, image_path_saved = future.result()
                    if caption and image_path_saved:
                        final_text = f"Source Document: {source_file_name}\nPage: {page_no}\n[Visual Content]: {caption}"
                        metadata = DocumentMetadata(
                            source_file=source_file_name, file_hash=file_hash, page_number=page_no, chunk_type="image", image_path=image_path_saved
                        )
                        results.append(DocumentChunk(text=final_text, metadata=metadata))
                except Exception as exc:
                    logger.error(f"[SYSTEM] Image thread exception: {exc}")

        logger.info(f"PyMuPDF finished. Extracted {len(results)} layout-aware chunks and {image_counter} images.")
        return results