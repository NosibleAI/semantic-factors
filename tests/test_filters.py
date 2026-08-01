"""Filter normalisation and the vocabulary gate."""
import pytest

from semantic_factors import filters
from semantic_factors.errors import ConfigurationError


def test_nested_field_names_are_flattened() -> None:
    """
    Require the nested names that appear in an event record to resolve to flat field names.

    :return: None.
    """
    result = filters.normalise_filters(
        filters={
            "event.region": "North America",
            "event.language": "en",
            "signals.materiality": "high"
        }
    )

    assert result == {"region": "North America", "language": "en", "materiality": "high"}


def test_documented_aliases_resolve() -> None:
    """
    Require the API's published aliases to reach their canonical field.

    :return: None.
    """
    result = filters.normalise_filters(filters={"n_sources": {"gte": 50}, "iab1": "Business"})

    assert result == {"total_netlocs": {"gte": 50}, "iab_tier_1": "Business"}


def test_a_sequence_of_pairs_is_accepted() -> None:
    """
    Require the list-of-pairs shape to work as well as a mapping.

    :return: None.
    """
    result = filters.normalise_filters(filters=[("country", "Kenya"), ("language", "en")])

    assert result == {"country": "Kenya", "language": "en"}


def test_an_unknown_field_is_rejected_with_a_suggestion() -> None:
    """
    Require a near-miss field name to be caught locally and to name the field it resembles.

    A typo that reaches the server comes back as a bare 400 several seconds later.

    :return: None.
    """
    with pytest.raises(ConfigurationError) as error:
        filters.normalise_filters(filters={"materialty": "high"})

    assert "materiality" in str(error.value)


def test_an_operator_the_type_does_not_support_is_rejected() -> None:
    """
    Require a numeric operator on a string field to be caught before any request is made.

    :return: None.
    """
    with pytest.raises(ConfigurationError) as error:
        filters.normalise_filters(filters={"country": {"gte": 3}})

    assert "gte" in str(error.value)


def test_combinators_normalise_their_branches() -> None:
    """
    Require nested names inside and/or/not branches to be flattened too.

    :return: None.
    """
    result = filters.normalise_filters(
        filters={
            "or": [{"event.country": "Kenya"}, {"event.country": "Ethiopia"}],
            "not": {"signals.sentiment": "positive"}
        }
    )

    assert result == {
        "or": [{"country": "Kenya"}, {"country": "Ethiopia"}],
        "not": {"sentiment": "positive"}
    }


def test_geometry_clauses_pass_through_untouched() -> None:
    """
    Require bbox and geo_radius to survive normalisation unchanged.

    :return: None.
    """
    circle = {"lat": 25.6, "lng": 56.3, "km": 200}
    result = filters.normalise_filters(filters={"geo_radius": circle})

    assert result == {"geo_radius": circle}


def test_no_filters_is_an_empty_object() -> None:
    """
    Require None to mean "every event is eligible" rather than an error.

    :return: None.
    """
    assert filters.normalise_filters(filters=None) == {}


def test_every_alias_points_at_a_real_field() -> None:
    """
    Require the alias table to stay consistent with the field table.

    :return: None.
    """
    dangling = {
        alias: field
        for alias, field in filters.FIELD_ALIASES.items()
        if field not in filters.FIELD_TYPES
    }

    assert dangling == {}
