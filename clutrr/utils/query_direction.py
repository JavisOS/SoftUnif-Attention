"""Audited query-direction transformations for CLUTRR kinship labels."""

from __future__ import annotations


GENDERED_GROUPS = {
    "father": ("male", "parent"),
    "mother": ("female", "parent"),
    "son": ("male", "child"),
    "daughter": ("female", "child"),
    "brother": ("male", "sibling"),
    "sister": ("female", "sibling"),
    "uncle": ("male", "pibling"),
    "aunt": ("female", "pibling"),
    "nephew": ("male", "nibling"),
    "niece": ("female", "nibling"),
    "grandfather": ("male", "grandparent"),
    "grandmother": ("female", "grandparent"),
    "grandson": ("male", "grandchild"),
    "granddaughter": ("female", "grandchild"),
    "husband": ("male", "spouse"),
    "wife": ("female", "spouse"),
    "father-in-law": ("male", "parent_in_law"),
    "mother-in-law": ("female", "parent_in_law"),
    "son-in-law": ("male", "child_in_law"),
    "daughter-in-law": ("female", "child_in_law"),
}

INVERSE_GROUP = {
    "parent": "child",
    "child": "parent",
    "sibling": "sibling",
    "pibling": "nibling",
    "nibling": "pibling",
    "grandparent": "grandchild",
    "grandchild": "grandparent",
    "spouse": "spouse",
    "parent_in_law": "child_in_law",
    "child_in_law": "parent_in_law",
}

RELATION_BY_GROUP_GENDER = {
    (group, gender): relation
    for relation, (gender, group) in GENDERED_GROUPS.items()
}


def parse_gender_map(text: str) -> dict[str, str]:
    genders = {}
    for item in (text or "").split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        name, gender = item.rsplit(":", 1)
        gender = gender.strip().lower()
        if gender in {"male", "female"}:
            genders[name.strip()] = gender
    return genders


def inverse_relation(relation: str, original_subject_gender: str | None) -> str | None:
    """Return the label after swapping the two query endpoints.

    CLUTRR labels describe the object relative to the subject. After reversing
    a query, the new relation's gender is therefore the original subject's.
    """

    if relation not in GENDERED_GROUPS:
        return None
    if original_subject_gender not in {"male", "female"}:
        return None
    _, group = GENDERED_GROUPS[relation]
    return RELATION_BY_GROUP_GENDER.get(
        (INVERSE_GROUP[group], original_subject_gender)
    )


def reverse_path_relations(
    relations: list[str],
    path_indices: list[int],
    all_names: list[str],
    genders: dict[str, str],
) -> list[str] | None:
    """Reverse an ordered path and invert every directed edge label."""

    if len(relations) != max(0, len(path_indices) - 1):
        return None
    reversed_relations = []
    for edge_index in range(len(relations) - 1, -1, -1):
        source_index = path_indices[edge_index]
        if not 0 <= source_index < len(all_names):
            return None
        source_gender = genders.get(all_names[source_index])
        reversed_relation = inverse_relation(relations[edge_index], source_gender)
        if reversed_relation is None:
            return None
        reversed_relations.append(reversed_relation)
    return reversed_relations
