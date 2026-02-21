from langchain_cohere import CohereEmbeddings
from app.core.config import get_settings

def get_embedding_model() -> CohereEmbeddings:
    settings = get_settings()
    return CohereEmbeddings(
        cohere_api_key=settings.COHERE_API_KEY,
        model="embed-english-v3.0"
    )