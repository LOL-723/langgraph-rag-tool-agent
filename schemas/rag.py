from pydantic import BaseModel


class RagChunk(BaseModel):
    id: str
    index: int
    content: str


class RagUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    chunks: list[RagChunk]


class RagSource(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str

