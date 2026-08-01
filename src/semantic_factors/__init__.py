"""Build quantitative factors from NOSIBLE World by describing them in plain English.

    from semantic_factors import SemanticFactor

    factor = SemanticFactor(
        anchors=["Severe insect surges decimate agricultural yields, threatening food security."],
        pole_hi=["The infestation has spread and is getting worse"],
        pole_lo=["The infestation has subsided and is getting better"],
        filters={"region": "North America", "language": "en", "materiality": "high"}
    )
    factor.compute(weights="coverage", start_from="2015-01-01", api_key="nos_sk_...")
    factor.plot(annotate=True)
    factor.regress(tickers=["CTVA.US", "FMC.US", "ZTS.US"])

Documentation lives at https://docs.nosible.com.
"""
from semantic_factors.client import WorldClient
from semantic_factors.errors import (
    AccessDeniedError,
    ConfigurationError,
    CredentialError,
    EmbeddingError,
    NotComputedError,
    ReturnsUnavailableError,
    SemanticFactorError,
    WorldApiError,
)
from semantic_factors.factor import SemanticFactor
from semantic_factors.filters import FIELD_TYPES
from semantic_factors.logs import set_verbose
from semantic_factors.version import __version__
from semantic_factors.weights import WEIGHT_SCHEMES

__all__ = [
    "AccessDeniedError",
    "ConfigurationError",
    "CredentialError",
    "EmbeddingError",
    "FIELD_TYPES",
    "NotComputedError",
    "ReturnsUnavailableError",
    "SemanticFactor",
    "SemanticFactorError",
    "WEIGHT_SCHEMES",
    "WorldApiError",
    "WorldClient",
    "__version__",
    "set_verbose"
]
