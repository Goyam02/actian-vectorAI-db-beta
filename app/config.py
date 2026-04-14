from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    vector_db_url: str = "localhost:50051"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    collection_name: str = "documents"
    default_top_k: int = 5
    chunk_size: int = 200
    chunk_overlap: int = 50
    
    azure_openai_endpoint: str | None = None
    azure_openai_key: str | None = None
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_deployment_name: str = "gpt-4o-mini"
    
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 200
    
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
