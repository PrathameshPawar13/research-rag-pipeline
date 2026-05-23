from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams


_DENSE_DIM = 384


def get_client(path: str = ":memory:") -> QdrantClient:
    if path == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(path=path)


def create_collection(client: QdrantClient, collection_name: str = "papers") -> None:
    collections = client.get_collections().collections
    existing = {c.name for c in collections}

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=_DENSE_DIM, distance=Distance.COSINE),
        )


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            models.PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "source": chunk["metadata"]["source"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "total_chunks": chunk["metadata"]["total_chunks"],
                },
            )
        )

    client.upsert(collection_name=collection_name, points=points)


def search(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    top_k: int = 10,
) -> list[dict]:
    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    )
    return [
        {
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "score": hit.score,
            "chunk_index": hit.payload["chunk_index"],
        }
        for hit in result.points
    ]
