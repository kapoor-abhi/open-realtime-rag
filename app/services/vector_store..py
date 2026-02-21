import uuid
from typing import List, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from app.models.schemas import DocumentChunk
from app.services.embeddings import get_embedding_model

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

    async def search(self, query: str, limit: int = 5, page_number: Optional[int] = None, source_file: Optional[str] = None) -> List[dict]:
        query_vector = await self.embeddings.aembed_query(query)
        
        must_conditions = []
        if source_file:
            must_conditions.append(models.FieldCondition(key="source_file", match=models.MatchValue(value=source_file)))
        if page_number is not None:
            must_conditions.append(models.FieldCondition(key="page_number", match=models.MatchValue(value=page_number)))
            
        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit
        )
        
        return [hit.payload for hit in results]