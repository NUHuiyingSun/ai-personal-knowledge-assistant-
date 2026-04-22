from __future__ import annotations

import pathlib
import statistics
import sys
import time

import pytest


# Project root configuration.
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INDEX_FILE = ROOT / "notes_index.faiss"
METADATA_FILE = ROOT / "notes_metadata.pkl"

# Benchmark configuration.
LATENCY_THRESHOLD_MS = 100.0
REPEAT_PER_QUERY = 5

# Warm-up queries to reduce one-time initialization noise.
WARMUP_QUERIES = [
    "terraform module organization",
    "python virtual environment setup",
    "kubernetes readiness probe",
]

# Representative benchmark queries for local semantic retrieval.
BENCH_QUERIES = [
    "how should I organize large terraform configurations to make them reusable",
    "what is the purpose of faiss in a rag pipeline",
    "how does mcp expose local notes to cursor",
    "how to reduce hallucination with local documentation retrieval",
    "what is the difference between liveness and readiness probes",
    "how to structure terraform modules for reuse",
]


@pytest.fixture(scope="session")
def server_module():
    """Import the MCP server module once for the entire test session."""
    import mcp_server

    return mcp_server


def _assert_dimension_consistency(server_module) -> None:
    """Validate that the loaded embedding model matches the FAISS index dimension."""
    # model_dim = server_module.model.get_sentence_embedding_dimension()
    model_dim = server_module.model.get_embedding_dimension()
    index_dim = server_module.index.d

    assert model_dim == index_dim, (
        f"Embedding dimension mismatch detected: model dimension = {model_dim}, "
        f"FAISS index dimension = {index_dim}. "
        "Use the same embedding model in build_index.py and mcp_server.py, "
        "then rebuild notes_index.faiss and notes_metadata.pkl."
    )


def _time_one_query(server_module, query: str) -> tuple[float, str]:
    """Measure retrieval latency for a single query in milliseconds."""
    start = time.perf_counter()
    result = server_module.search_notes(query)
    end = time.perf_counter()

    return (end - start) * 1000.0, result


def _p95(values: list[float]) -> float:
    """Compute the p95 latency from a list of measurements."""
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round(0.95 * (len(values) - 1)))
    return values[idx]


def test_index_artifacts_exist():
    """Verify that the persisted retrieval artifacts are present."""
    assert INDEX_FILE.exists(), f"Missing FAISS index file: {INDEX_FILE}"
    assert METADATA_FILE.exists(), f"Missing metadata file: {METADATA_FILE}"


def test_embedding_model_matches_faiss_index(server_module):
    """Verify that query embeddings are compatible with the persisted FAISS index."""
    _assert_dimension_consistency(server_module)


def test_search_notes_returns_context(server_module):
    """Verify that the retrieval function returns formatted context for a basic query."""
    _assert_dimension_consistency(server_module)

    result = server_module.search_notes("terraform module organization")

    assert isinstance(result, str), "search_notes must return a string."
    assert result.strip(), "search_notes returned an empty response."
    assert "Source:" in result, "Returned context is missing source annotations."
    assert "Content:" in result, "Returned context is missing content sections."


def test_retrieval_latency_under_100ms(server_module):
    """Validate that median retrieval latency remains below the target threshold."""
    _assert_dimension_consistency(server_module)

    for query in WARMUP_QUERIES:
        result = server_module.search_notes(query)
        assert result.strip(), f"Warm-up retrieval returned no content for query: {query!r}"

    latencies = []

    for query in BENCH_QUERIES:
        for _ in range(REPEAT_PER_QUERY):
            latency_ms, result = _time_one_query(server_module, query)
            assert result.strip(), f"Retrieval returned no content for query: {query!r}"
            latencies.append(latency_ms)

    median_ms = statistics.median(latencies)
    mean_ms = statistics.mean(latencies)
    p95_ms = _p95(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)

    print("\n=== Retrieval Benchmark Summary ===")
    print(f"Samples: {len(latencies)}")
    print(f"Min:     {min_ms:.2f} ms")
    print(f"Median:  {median_ms:.2f} ms")
    print(f"Mean:    {mean_ms:.2f} ms")
    print(f"P95:     {p95_ms:.2f} ms")
    print(f"Max:     {max_ms:.2f} ms")

    assert median_ms < LATENCY_THRESHOLD_MS, (
        f"Median retrieval latency {median_ms:.2f} ms exceeds "
        f"the threshold of {LATENCY_THRESHOLD_MS:.2f} ms."
    )