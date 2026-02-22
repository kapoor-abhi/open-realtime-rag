import os
import logging
from typing import List
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from app.models.schemas import DocumentChunk, DocumentMetadata
from app.services.vision import generate_image_caption
from app.services.storage import StorageService

# Setup transparent logging for the worker
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

class DocumentParser:
    def __init__(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True
        
        self.converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self.chunker = HierarchicalChunker()
        self.storage = StorageService()

    def parse_document(self, file_path: str, source_file_name: str) -> List[DocumentChunk]:
        logger.info(f"Starting Docling structural parsing for {source_file_name}...")
        conv_res = self.converter.convert(file_path)
        doc_chunks = self.chunker.chunk(dl_doc=conv_res.document)
        
        results = []
        image_counter = 0
        file_hash_prefix = file_path.split("/")[-1].split("_")[0]
        
        logger.info(f"Docling parsing complete. Found {len(list(doc_chunks))} total chunks. Beginning multimodal processing...")
        
        for chunk in doc_chunks:
            page_no = 1
            chunk_type = "text"
            image_path_saved = None
            caption = ""
            
            if chunk.meta and chunk.meta.doc_items:
                for item in chunk.meta.doc_items:
                    if hasattr(item, "prov") and item.prov:
                        page_no = item.prov[0].page_no
                        break
                        
                for item in chunk.meta.doc_items:
                    label_str = str(getattr(item, "label", "")).lower()
                    if "table" in label_str:
                        chunk_type = "table"
                        break
                    elif "picture" in label_str or "image" in label_str:
                        chunk_type = "image"
                        if hasattr(item, "get_image"):
                            img = item.get_image(conv_res.document)
                            if img:
                                image_counter += 1
                                img_filename = f"uploads/{file_hash_prefix}_img_{image_counter}.png"
                                img.save(img_filename)
                                logger.info(f"[VISION] Extracted Image {image_counter} from Page {page_no}.")
                                
                                # 1. Send to Llama 4 Scout for captioning
                                logger.info(f"[VISION] Sending Image {image_counter} to Llama-4-Scout via Groq...")
                                caption = generate_image_caption(img_filename)
                                logger.info(f"[VISION] Received Caption: {caption[:75]}...")
                                
                                # 2. Upload to Cloudflare R2
                                try:
                                    logger.info(f"[STORAGE] Uploading Image {image_counter} to Cloudflare R2...")
                                    public_img_url = self.storage.upload_file(img_filename, f"images/{file_hash_prefix}_img_{image_counter}.png")
                                    image_path_saved = public_img_url
                                    logger.info(f"[STORAGE] SUCCESS! Image {image_counter} public URL: {public_img_url}")
                                except Exception as e:
                                    logger.error(f"[STORAGE] FAIL: Cloudflare R2 Upload crashed for Image {image_counter}. Error: {e}")
                                    image_path_saved = None
                                
                                # Cleanup local file
                                if os.path.exists(img_filename):
                                    os.remove(img_filename)
                        break
            
            final_text = chunk.text
            if chunk_type == "image" and caption:
                final_text = f"[Image Caption: {caption}]\n{chunk.text}"
            
            metadata = DocumentMetadata(
                source_file=source_file_name,
                page_number=page_no,
                chunk_type=chunk_type,
                image_path=image_path_saved
            )
            results.append(DocumentChunk(text=final_text, metadata=metadata))
            
        logger.info(f"Finished processing. Total images processed & uploaded: {image_counter}")
        return results