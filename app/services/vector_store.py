import uuid
import logging
from typing import List, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from app.models.schemas import DocumentChunk
from app.services.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self, client: AsyncQdrantClient, collection_name: str = "multirag_docs"):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = get_embedding_model()

    async def init_collection(self):
        exists = await self.client.collection_exists(self.collection_name)
        if not exists:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE
                )
            )

    async def upsert_chunks(self, chunks: List[DocumentChunk]):
        texts = [chunk.text for chunk in chunks]
        vectors = await self.embeddings.aembed_documents(texts)
        
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk.text, **chunk.metadata.model_dump()}
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    # NEW: Accept file_hash instead of source_file
    async def search(self, query: str, limit: int = 5, page_number: int = None, file_hash: str = None) -> List[dict]:
        logger.info(f"[RETRIEVAL] Generating Cohere embedding for query: '{query}'")
        
        query_embedding = await self.embeddings.aembed_query(query)
        
        filter_conditions = []
        if page_number is not None:
            filter_conditions.append(models.FieldCondition(key="page_number", match=models.MatchValue(value=page_number)))
        if file_hash is not None:
            filter_conditions.append(models.FieldCondition(key="file_hash", match=models.MatchValue(value=file_hash)))
            
        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

        try:
            # FIX: Change 'source_file' to 'file_hash' in the logging statement!
            logger.info(f"[RETRIEVAL] Querying Qdrant... (Filters: Page={page_number}, Hash={file_hash})")
            results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit
            )
        except Exception as e:
            if "Not found" in str(e) or "doesn't exist" in str(e):
                logger.warning("[RETRIEVAL] Collection not found. Returning empty list.")
                return []
            raise e
        
        # --- TRANSPARENCY LOGGING ---
        retrieved_chunks = []
        logger.info(f"\n{'='*60}\n[RETRIEVAL] FOUND {len(results)} CHUNKS FOR QUERY: '{query}'\n{'='*60}")
        
        for hit in results:
            chunk_data = {
                "text": hit.payload["text"],
                "page_number": hit.payload["page_number"],
                "source_file": hit.payload["source_file"],
                "chunk_type": hit.payload.get("chunk_type", "text"),
                "image_path": hit.payload.get("image_path")
            }
            retrieved_chunks.append(chunk_data)
            
            # Print exact scores, pages, and context snippets
            logger.info(f"-> Similarity Score: {hit.score:.4f} | Page: {chunk_data['page_number']} | Type: {chunk_data['chunk_type']}")
            logger.info(f"-> Content Preview: {chunk_data['text'][:150]}...\n{'-'*60}")
            
        return retrieved_chunks