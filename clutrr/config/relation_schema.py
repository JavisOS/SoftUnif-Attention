"""Shared CLUTRR relation vocabularies used across training and analysis scripts."""

RELATION_NAMES_20 = (
    "daughter",
    "sister",
    "son",
    "aunt",
    "father",
    "husband",
    "granddaughter",
    "brother",
    "nephew",
    "mother",
    "uncle",
    "grandfather",
    "wife",
    "grandmother",
    "niece",
    "grandson",
    "son-in-law",
    "father-in-law",
    "daughter-in-law",
    "mother-in-law",
)

RELATION_ID_MAP_20 = {name: idx for idx, name in enumerate(RELATION_NAMES_20)}
RELATION_ID_MAP_21_WITH_NOTHING = {
    **RELATION_ID_MAP_20,
    "nothing": len(RELATION_NAMES_20),
}

ID_RELATION_MAP_20 = {idx: name for name, idx in RELATION_ID_MAP_20.items()}
ID_RELATION_MAP_21_WITH_NOTHING = {
    idx: name for name, idx in RELATION_ID_MAP_21_WITH_NOTHING.items()
}

# Legacy scripts build transitivity triples on the 20 relation classes.
ALL_POSSIBLE_TRANSITIVES_20 = [
    (a, b, c) for a in range(20) for b in range(20) for c in range(20)
]
