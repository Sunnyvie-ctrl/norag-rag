from pydantic import BaseModel, Field
from typing import List, Optional

class ChunkMetadata(BaseModel):
    chunk_id: str
    document_name: str
    start_char: int
    end_char: int
    score: Optional[float] = None

class DocumentInfo(BaseModel):
    filename: str
    char_count: int
    chunk_count: int

class RAGQueryParams(BaseModel):
    question: str
    chunk_size: int = Field(default=800, ge=100, le=4000)
    overlap_percent: int = Field(default=10, ge=0, le=50)
    top_k: int = Field(default=5, ge=1, le=20)
    selected_docs: Optional[List[str]] = Field(default=None, description="Filter search to specific files")

class CitedChunk(BaseModel):
    chunk_id: str
    document_name: str
    excerpt: str
    similarity_score: float

class GroundedLLMResponse(BaseModel):
    """Pydantic schema passed directly to the LLM for structured output enforcement."""
    answer: str = Field(description="The response grounded strictly in the provided context excerpts.")
    is_answerable: bool = Field(description="False if context is insufficient or missing the required facts.")
    cited_chunk_ids: List[str] = Field(description="List of chunk_ids directly referenced to produce this answer.")
    reasoning: str = Field(description="Brief explanation of why the question can or cannot be answered.")

class RAGResponse(BaseModel):
    answer: str
    is_answerable: bool
    cited_chunks: List[CitedChunk]
    reasoning: str
    execution_time_ms: float