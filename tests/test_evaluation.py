from evaluation.evaluator import hit_rate, mean_reciprocal_rank, faithfulness_score


def test_hit_rate_perfect():
    retrieved = [["a", "b", "c"], ["b", "a", "c"]]
    relevant = ["a", "b"]
    assert hit_rate(retrieved, relevant, k=5) == 1.0


def test_hit_rate_miss():
    retrieved = [["a", "b", "c"], ["a", "c", "d"]]
    relevant = ["d", "b"]
    assert hit_rate(retrieved, relevant, k=5) == 0.5


def test_mrr():
    retrieved = [["a", "b", "c"], ["c", "a", "b"]]
    relevant = ["a", "b"]
    result = mean_reciprocal_rank(retrieved, relevant, k=10)
    assert 0.5 <= result <= 1.0


def test_faithfulness():
    answer = "The cat sat on the mat. The dog ran away."
    chunks = ["The cat sat on the mat quietly", "The dog ran fast"]
    score = faithfulness_score(answer, chunks)
    assert 0 <= score <= 1
