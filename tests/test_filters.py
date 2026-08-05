"""Filter normalisation and the vocabulary gate."""

import pytest

from semantic_factors import filters
from semantic_factors.errors import ConfigurationError

EXPECTED_ONTOLOGY_TOP3_FIELDS = {
    "iab_tier_1_top3",
    "iab_tier_2_top3",
    "iab_tier_3_top3",
    "iab_tier_4_top3",
    "iptc_genre_top3",
    "iptc_media_topic_l1_top3",
    "iptc_media_topic_l2_top3",
    "iptc_media_topic_l3_top3",
    "media_frame_top3",
    "ekman7_emotion_top3",
    "schema_org_event_top3",
    "gics_sector_top3",
    "gics_industry_group_top3",
    "gics_industry_top3",
    "gics_sub_industry_top3",
    "asset_class_main_class_top3",
    "asset_class_sub_class_top3",
    "sdgs_goal_top3",
    "sdgs_target_top3",
    "nosible_category_top3",
    "nosible_subcategory_top3",
    "nosible_event_top3",
    "icd11_chapter_top3",
    "icd11_block_top3",
    "emdat_disaster_group_top3",
    "emdat_disaster_subgroup_top3",
    "emdat_disaster_type_top3",
    "mitre_attack_tactic_top3",
    "mitre_attack_technique_top3",
    "mitre_attack_subtechnique_top3",
    "sportsml_sport_top3",
    "sportsml_event_type_top3",
    "plover_event_top3",
    "plover_mode_top3",
    "plover_context_top3",
}
EXPECTED_FIELDS_BY_TYPE = {
    "string": {
        "event_id", "title", "description", "language", "continent", "region", "country",
        "country_iso", "city", "coordinate_source", "sentiment", "materiality", "time_horizon",
        "iab_tier_1", "iab_tier_2", "iab_tier_3", "iab_tier_4", "iptc_genre",
        "iptc_media_topic_l1", "iptc_media_topic_l2", "iptc_media_topic_l3", "media_frame",
        "ekman7_emotion", "schema_org_event", "gics_sector", "gics_industry_group",
        "gics_industry", "gics_sub_industry", "asset_class_main_class", "asset_class_sub_class",
        "sdgs_goal", "sdgs_target", "nosible_category", "nosible_subcategory", "nosible_event",
        "icd11_chapter", "icd11_block", "emdat_disaster_group", "emdat_disaster_subgroup",
        "emdat_disaster_type", "mitre_attack_tactic", "mitre_attack_technique",
        "mitre_attack_subtechnique", "sportsml_sport", "sportsml_event_type", "plover_event",
        "plover_mode", "plover_context",
    },
    "date": {"date"},
    "number": {
        "lat", "lng", "total_coverage", "total_netlocs", "coverage_concentration_index",
        "materiality_score",
    },
    "boolean": {"has_tickers", "required_ontology_below_floor"},
    "string_set": {
        "entity_event", "entity_fac", "entity_gpe", "entity_language", "entity_law", "entity_loc",
        "entity_norp", "entity_org", "entity_person", "entity_product", "entity_work_of_art",
        "entity_any", "ticker", "isin", "figi", "lei", "wiki_qid",
    } | EXPECTED_ONTOLOGY_TOP3_FIELDS,
}
EXPECTED_FIELD_TYPES = {
    field: kind
    for kind, fields in EXPECTED_FIELDS_BY_TYPE.items()
    for field in fields
}


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


def test_filter_vocabulary_matches_the_world_schema_revision() -> None:
    """Pin the public client vocabulary to World's 109 authoritative canonical fields."""
    ontology_top3 = {name for name in filters.FIELD_TYPES if name.endswith("_top3")}
    string_sets = {name for name, kind in filters.FIELD_TYPES.items() if kind == "string_set"}

    assert len(filters.FIELD_TYPES) == 109
    assert filters.FIELD_TYPES == EXPECTED_FIELD_TYPES
    assert ontology_top3 == EXPECTED_ONTOLOGY_TOP3_FIELDS
    assert set(filters.ONTOLOGY_TOP3_FIELDS) == EXPECTED_ONTOLOGY_TOP3_FIELDS
    assert len(string_sets) == 52
    assert all(filters.FIELD_TYPES[name] == "string_set" for name in ontology_top3)


def test_top3_fields_accept_set_operators_and_reject_scalar_ordering() -> None:
    """Keep ontology candidate filters on the backend's string-set operator contract."""
    assert filters.normalise_filters(
        filters={"gics_sector_top3": {"all": ["Energy", "Industrials"]}}
    ) == {"gics_sector_top3": {"all": ["Energy", "Industrials"]}}

    with pytest.raises(ConfigurationError, match="gte"):
        filters.normalise_filters(filters={"gics_sector_top3": {"gte": "Energy"}})
