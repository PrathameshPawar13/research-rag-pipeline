from pydantic import BaseModel


class ChunkResult(BaseModel):
    text: str
    source: str
    score: float
    rerank_score: float | None = None
    chunk_index: int | None = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    chunks: list[ChunkResult]


class IngestRequest(BaseModel):
    arxiv_ids: list[str]


class IngestResponse(BaseModel):
    ingested: int
    papers: list[str]


class PaperInfo(BaseModel):
    id: str
    title: str
    authors: list[str]
    published: str
    summary: str


class StatusResponse(BaseModel):
    collection: str
    chunks_count: int
    papers_count: int
    papers: list[PaperInfo]
