from ingestion.chunker import recursive_split_text, chunk_document


def test_recursive_split_text_basic():
    text = "This is a test. " * 100
    chunks = recursive_split_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 220 for c in chunks)


def test_recursive_split_text_short():
    text = "Short text."
    chunks = recursive_split_text(text, chunk_size=500)
    assert chunks == [text]


def test_chunk_document():
    text = "Word. " * 1000
    chunks = chunk_document(text, "test123", chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert chunks[0]["metadata"]["source"] == "test123"
    assert chunks[0]["metadata"]["chunk_index"] == 0
