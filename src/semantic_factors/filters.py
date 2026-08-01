"""Filter normalisation and vocabulary.

NOSIBLE World exposes its event fields in nested JSON (``event.region``, ``signals.materiality``,
``coverage.total_netlocs``) but filters them by flat name (``region``, ``materiality``,
``total_netlocs``). People write the nested form because that is what they see in a record, so
this module accepts either and emits the flat form the API wants.

It also refuses unknown field names up front. A typo that reaches the server comes back as a
bare HTTP 400 several seconds later; caught here it comes back instantly, naming the closest
field that does exist.
"""
import difflib
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

from semantic_factors.errors import ConfigurationError

#: Prefixes people copy out of an event record that the filter vocabulary does not use.
NESTED_PREFIXES = ("event.", "signals.", "coverage.", "coordinate.", "ontology.", "extra.")

#: Boolean and set combinators the World filter DSL understands as top-level keys.
LOGICAL_KEYS = ("and", "or", "not", "bbox", "geo_radius")

#: Comparison operators, by the field type they are valid for.
OPERATORS = {
    "string": ("eq", "neq", "in", "not_in", "exists", "missing", "contains", "prefix"),
    "number": ("eq", "neq", "in", "not_in", "exists", "missing", "gt", "gte", "lt", "lte", "between"),
    "boolean": ("eq", "neq", "exists", "missing"),
    "date": ("eq", "neq", "in", "not_in", "gt", "gte", "lt", "lte", "between"),
    "string_set": ("eq", "in", "all", "not_in", "contains", "prefix")
}

#: Every filterable field, by canonical name, with the type whose operators it accepts.
FIELD_TYPES: Dict[str, str] = {
    "event_id": "string",
    "date": "date",
    "title": "string",
    "description": "string",
    "language": "string",
    "continent": "string",
    "region": "string",
    "country": "string",
    "country_iso": "string",
    "city": "string",
    "lat": "number",
    "lng": "number",
    "coordinate_source": "string",
    "total_coverage": "number",
    "total_netlocs": "number",
    "coverage_concentration_index": "number",
    "has_tickers": "boolean",
    "sentiment": "string",
    "materiality": "string",
    "materiality_score": "number",
    "time_horizon": "string",
    "required_ontology_below_floor": "boolean",
    "iab_tier_1": "string",
    "iab_tier_2": "string",
    "iab_tier_3": "string",
    "iab_tier_4": "string",
    "iptc_genre": "string",
    "iptc_media_topic_l1": "string",
    "iptc_media_topic_l2": "string",
    "iptc_media_topic_l3": "string",
    "media_frame": "string",
    "ekman7_emotion": "string",
    "schema_org_event": "string",
    "gics_sector": "string",
    "gics_industry_group": "string",
    "gics_industry": "string",
    "gics_sub_industry": "string",
    "asset_class_main_class": "string",
    "asset_class_sub_class": "string",
    "sdgs_goal": "string",
    "sdgs_target": "string",
    "nosible_category": "string",
    "nosible_subcategory": "string",
    "nosible_event": "string",
    "icd11_chapter": "string",
    "icd11_block": "string",
    "emdat_disaster_group": "string",
    "emdat_disaster_subgroup": "string",
    "emdat_disaster_type": "string",
    "mitre_attack_tactic": "string",
    "mitre_attack_technique": "string",
    "mitre_attack_subtechnique": "string",
    "sportsml_sport": "string",
    "sportsml_event_type": "string",
    "plover_event": "string",
    "plover_mode": "string",
    "plover_context": "string",
    "entity_event": "string_set",
    "entity_fac": "string_set",
    "entity_gpe": "string_set",
    "entity_language": "string_set",
    "entity_law": "string_set",
    "entity_loc": "string_set",
    "entity_norp": "string_set",
    "entity_org": "string_set",
    "entity_person": "string_set",
    "entity_product": "string_set",
    "entity_work_of_art": "string_set",
    "entity_any": "string_set",
    "ticker": "string_set",
    "isin": "string_set",
    "figi": "string_set",
    "lei": "string_set",
    "wiki_qid": "string_set"
}

#: Friendly names the API accepts, mapped onto their canonical field.
FIELD_ALIASES: Dict[str, str] = {
    "languages": "language",
    "geo_continent": "continent",
    "geo_region": "region",
    "countries": "country",
    "geo_country": "country",
    "geo_country_iso": "country_iso",
    "locality": "city",
    "lon": "lng",
    "longitude": "lng",
    "coverage": "total_coverage",
    "coverage_docs": "total_coverage",
    "coverage_sources": "total_netlocs",
    "n_sources": "total_netlocs",
    "coverage_hhi": "coverage_concentration_index",
    "ontology_below_floor": "required_ontology_below_floor",
    "iab1": "iab_tier_1",
    "iab2": "iab_tier_2",
    "iab3": "iab_tier_3",
    "iab4": "iab_tier_4",
    "genre": "iptc_genre",
    "iptc1": "iptc_media_topic_l1",
    "iptc2": "iptc_media_topic_l2",
    "iptc3": "iptc_media_topic_l3",
    "emotion": "ekman7_emotion",
    "nosible1": "nosible_category",
    "nosible2": "nosible_subcategory",
    "nosible3": "nosible_event",
    "icd1": "icd11_chapter",
    "icd2": "icd11_block",
    "emdat1": "emdat_disaster_group",
    "emdat2": "emdat_disaster_subgroup",
    "emdat3": "emdat_disaster_type",
    "mitre1": "mitre_attack_tactic",
    "mitre2": "mitre_attack_technique",
    "mitre3": "mitre_attack_subtechnique",
    "sportsml1": "sportsml_sport",
    "sportsml2": "sportsml_event_type",
    "plover1": "plover_event",
    "plover2": "plover_mode",
    "plover3": "plover_context",
    "gpes": "entity_gpe",
    "locations": "entity_loc",
    "orgs": "entity_org",
    "people": "entity_person",
    "products": "entity_product",
    "entities": "entity_any",
    "tickers": "ticker",
    "symbol": "ticker",
    "isins": "isin",
    "qid": "wiki_qid"
}

FilterInput = Union[None, Mapping[str, Any], Sequence[Tuple[str, Any]]]


def normalise_filters(filters: FilterInput) -> Dict[str, Any]:
    """
    Turn a user-supplied filter description into the flat form the World API expects.

    Accepts a mapping or a sequence of key/value pairs, tolerates the nested field names that
    appear in an event record, resolves documented aliases, and validates every field and
    operator before a request is made.

    :param filters: Mapping of field to value, a sequence of pairs, or None.
    :return: Validated filter object, empty when no filters were supplied.
    """
    if filters is None:
        return {}
    items = list(filters.items()) if isinstance(filters, Mapping) else list(filters)
    normalised: Dict[str, Any] = {}
    for key, value in items:
        if not isinstance(key, str):
            raise ConfigurationError(f"filter keys must be strings, got {type(key).__name__}")
        if key in LOGICAL_KEYS:
            normalised[key] = normalise_logical(
                key=key,
                value=value
            )
            continue
        field = canonical_field(name=key)
        normalised[field] = validate_condition(
            field=field,
            value=value
        )
    return normalised


def normalise_logical(key: str, value: Any) -> Any:
    """
    Normalise the branches of a combinator, leaving geometry clauses untouched.

    :param key: One of the logical or geometry keys.
    :param value: The combinator's operand.
    :return: Normalised operand.
    """
    if key in ("bbox", "geo_radius"):
        return value
    if key == "not":
        return normalise_filters(filters=value)
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        raise ConfigurationError(f"filter key {key!r} takes a list of filter objects")
    return [normalise_filters(filters=branch) for branch in value]


def canonical_field(name: str) -> str:
    """
    Resolve one field name, stripping a nested prefix and applying documented aliases.

    :param name: Field name as the caller wrote it.
    :return: Canonical flat field name.
    """
    cleaned = name.strip()
    for prefix in NESTED_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    cleaned = FIELD_ALIASES.get(cleaned, cleaned)
    if cleaned not in FIELD_TYPES:
        raise ConfigurationError(
            f"unknown filter field {name!r}{suggestion(name=cleaned)}. "
            f"See semantic_factors.filters.FIELD_TYPES for the {len(FIELD_TYPES)} filterable fields."
        )
    return cleaned


def validate_condition(field: str, value: Any) -> Any:
    """
    Check one field's condition against the operators its type allows.

    A scalar means equality and a list means membership, so only mapping values carry
    operators and only those need checking.

    :param field: Canonical field name.
    :param value: Scalar, list, or operator mapping.
    :return: The value, unchanged, once it is known to be valid.
    """
    if not isinstance(value, Mapping):
        return list(value) if isinstance(value, tuple) else value
    allowed = OPERATORS[FIELD_TYPES[field]]
    for operator in value:
        if operator not in allowed:
            raise ConfigurationError(
                f"operator {operator!r} is not valid for {field!r} "
                f"({FIELD_TYPES[field]}); allowed operators are {', '.join(allowed)}"
            )
    return dict(value)


def suggestion(name: str) -> str:
    """
    Offer the closest real field name, when one is close enough to be worth mentioning.

    :param name: The unrecognised field name, already stripped and unaliased.
    :return: A parenthesised suggestion, or an empty string.
    """
    candidates: List[str] = difflib.get_close_matches(
        word=name,
        possibilities=list(FIELD_TYPES) + list(FIELD_ALIASES),
        n=1,
        cutoff=0.7
    )
    return f" (did you mean {candidates[0]!r}?)" if candidates else ""
