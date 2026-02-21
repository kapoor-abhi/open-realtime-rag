from typing import List
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from app.models.schemas import DocumentChunk, DocumentMetadata

class DocumentParser:
    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = HierarchicalChunker()

    def parse_document(self, file_path: str, source_file_name: str) -> List[DocumentChunk]:
        conv_res = self.converter.convert(file_path)
        doc_chunks = self.chunker.chunk(dl_doc=conv_res.document)
        
        results = []
        for chunk in doc_chunks:
            page_no = 1
            chunk_type = "text"
            
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
                        break
            
            metadata = DocumentMetadata(
                source_file=source_file_name,
                page_number=page_no,
                chunk_type=chunk_type
            )
            results.append(DocumentChunk(text=chunk.text, metadata=metadata))
            
        return results