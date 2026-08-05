"""Command-line definition and output-safety contracts."""

import pytest

from semantic_factors import cli
from semantic_factors.errors import SemanticFactorError


def test_cli_accepts_all_advertised_multi_pole_compute_keys() -> None:
    """Keep the JSON-spec vocabulary aligned with the public Python API."""
    assert "poles" in cli.DEFINITION_KEYS
    assert "polarity_weighted" in cli.COMPUTE_KEYS


@pytest.mark.parametrize("name", ["../escape", "sub/path", r"sub\path", ".."])
def test_factor_name_cannot_escape_the_cli_output_directory(tmp_path, name) -> None:
    """Reject path-bearing factor names before any artifact is written."""
    with pytest.raises(SemanticFactorError):
        cli.safe_output_stem(output_dir=str(tmp_path), factor_name=name)


def test_plain_factor_name_resolves_inside_the_cli_output_directory(tmp_path) -> None:
    """Preserve ordinary safe factor names."""
    result = cli.safe_output_stem(output_dir=str(tmp_path), factor_name="crop-risk")

    assert result == str(tmp_path.resolve() / "crop-risk")
