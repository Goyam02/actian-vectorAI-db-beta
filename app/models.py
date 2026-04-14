from pydantic import BaseModel, Field
from typing import Optional


class DocumentUpload(BaseModel):
    text: str = Field(..., description="Text content to index")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")
    id: Optional[str] = Field(default=None, description="Optional custom document ID")


class DocumentChunk(BaseModel):
    chunk_id: int
    text: str
    char_count: int
    source_id: Optional[int] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    score_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Minimum similarity score"
    )


class SearchResult(BaseModel):
    id: int
    score: float
    text: str
    metadata: dict
    document_id: Optional[str] = Field(default=None, description="Document identifier")


class RAGRequest(BaseModel):
    query: str = Field(..., description="User query")
    top_k: int = Field(default=5, ge=1, le=20, description="Context chunks to retrieve")
    use_llm: bool = Field(default=True, description="Generate answer with LLM")


class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchResult]
    context_used: str


class DocumentIndexResponse(BaseModel):
    indexed_count: int
    collection: str
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    vector_db_version: Optional[str] = None
    embedding_model: str
    collection_exists: bool
