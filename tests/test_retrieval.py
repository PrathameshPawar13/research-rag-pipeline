from retrieval.hybrid_search import hybrid_search, _tokenize, _compute_bm25_scores, _normalize


def test_tokenize():
    assert _tokenize("Hello World") == ["hello", "world"]
    assert _tokenize("") == []


def test_normalize():
    assert _normalize([4.0, 2.0, 0.0]) == [1.0, 0.5, 0.0]
    assert _normalize([]) == []


def test_hybrid_search_empty():
    result = hybrid_search("test", [], [])
    assert result == []


def test_hybrid_search():
    dense_results = [
        {"text": "The cat sat on the mat", "source": "doc1", "score": 0.9},
        {"text": "Dogs are great pets", "source": "doc2", "score": 0.5},
    ]
    documents = [r["text"] for r in dense_results]
    result = hybrid_search("cat mat", dense_results, documents, alpha=0.5)
    assert len(result) == 2
    assert "bm25_score" in result[0]
    assert "dense_score" in result[0]
