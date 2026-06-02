from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DEEPSEEK_API_KEY:str
    DEEPSEEK_BASE_URL:str
    LLM_MODEL:str
    LLM_TIMEOUT:float
    LLM_TEMPERATURE: float = 0.1
    HF_TOKEN: str | None = None
    RAG_STORAGE_DIR: str = "storage/rag"
    RAG_CHROMA_DIR: str = "storage/chroma_db"
    RAG_COLLECTION_NAME: str = "default"
    RAG_EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    RAG_RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RAG_CHUNK_SIZE: int = 500
    RAG_CHUNK_OVERLAP: int = 80
    RAG_MIN_CHUNK_SIZE: int = 80
    RAG_RETRIEVE_TOP_K: int = 5
    RAG_RERANK_TOP_K: int = 3
    RAG_MAX_UPLOAD_MB: int = 20

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
