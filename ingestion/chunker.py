import re
from typing import TypedDict


class Chunk(TypedDict):
    text: str
    metadata: dict


def recursive_split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    separators: list[str] | None = None,
) -> list[str]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    if len(text) <= chunk_size:
        return [text]

    for sep in separators:
        if sep in text:
            chunks = text.split(sep)
            merged = []
            current = ""

            for chunk in chunks:
                piece = chunk + sep if sep != " " else chunk + " "
                if len(current) + len(piece) <= chunk_size:
                    current += piece
                else:
                    if current:
                        merged.append(current.strip())
                    current = piece

            if current:
                merged.append(current.strip())

            if merged:
                return merged

    return [text]


def chunk_document(text: str, source_id: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[Chunk]:
    raw_chunks = recursive_split_text(text, chunk_size, chunk_overlap)
    chunks: list[Chunk] = []

    for i, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "text": chunk_text.strip(),
            "metadata": {
                "source": source_id,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
            },
        })

    return chunks
