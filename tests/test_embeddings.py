"""Embedding response validation that never needs a provider request."""
import numpy as np
import pytest

from semantic_factors import embeddings
from semantic_factors.errors import EmbeddingError


def test_provider_rows_are_restored_to_input_order() -> None:
    """Require explicit provider indices to win over response order."""
    result = embeddings.parse_embeddings(
        payload={
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]}
            ]
        },
        expected=2,
        dims=2
    )

    assert np.array_equal(result, np.asarray(a=[[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))


@pytest.mark.parametrize(
    "rows",
    [
        ["not an object"],
        [{"index": 2, "embedding": [1.0, 0.0]}],
        [{"index": True, "embedding": [1.0, 0.0]}],
        [{"index": 0}],
        [{"index": 0, "embedding": [float("nan"), 0.0]}]
    ]
)
def test_malformed_provider_rows_are_rejected(rows: list) -> None:
    """Require bad indices and vectors to stay out of the content cache."""
    with pytest.raises(EmbeddingError):
        embeddings.parse_embeddings(payload={"data": rows}, expected=1, dims=2)
