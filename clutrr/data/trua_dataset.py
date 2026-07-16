import random

from clutrr.config.relation_schema import RELATION_ID_MAP_21_WITH_NOTHING as relation_id_map
from clutrr.data.clutrr_dataset import CLUTRRDataset, official_hop_count
from clutrr.utils.entity_alignment import align_entity_spans_to_tokens
from clutrr.utils.graph_reasoning import apply_bijective_map, parse_graph_and_path
from clutrr.utils.parsing import parse_pair_literal
from clutrr.utils.query_direction import (
    inverse_relation,
    parse_gender_map,
    reverse_path_relations,
)


class TruaClutrrDataset(CLUTRRDataset):
    def __init__(
        self,
        root,
        dataset,
        split,
        data_percentage=100,
        tokenizer=None,
        augment=False,
        augment_seed=None,
        reverse_query=False,
    ):
        super().__init__(root, dataset, split, data_percentage)
        self.tokenizer = tokenizer
        self.augment = augment
        self.augment_seed = augment_seed
        self.reverse_query = reverse_query
        self.split = split

        raw_rows = list(self.data)
        self.stats = {
            "split": split,
            "reverse_query": reverse_query,
            "raw_total": len(raw_rows),
            "valid_total": 0,
            "dropped_total": 0,
            "dropped_by_reason": {},
        }
        self.data = []

        for row_idx, row in enumerate(raw_rows):
            item, reason = self._build_item(row, row_idx=row_idx)
            if item is None:
                self.stats["dropped_total"] += 1
                self.stats["dropped_by_reason"][reason] = self.stats["dropped_by_reason"].get(reason, 0) + 1
                continue
            self.data.append(item)

        self.stats["valid_total"] = len(self.data)

    def _build_item(self, row, row_idx: int):
        if self.tokenizer is None:
            raise ValueError("TruaClutrrDataset requires a tokenizer for entity span alignment.")

        graph_info = parse_graph_and_path(row)
        if graph_info is None:
            return None, "graph_parse_failed"

        story_str = row[2]
        query = parse_pair_literal(row[3])
        if query is None:
            return None, "query_parse_failed"
        original_query = query
        original_target_relation = row[5]
        target_relation = original_target_relation
        all_names = graph_info["all_names"]
        path_indices = list(graph_info["path_node_indices"])
        path_rel_labels = list(graph_info.get("path_relation_labels", []))

        query_edge = parse_pair_literal(row[13])
        if query_edge is None:
            return None, "query_edge_parse_failed"
        sub_idx, obj_idx = query_edge
        if not isinstance(sub_idx, int) or not isinstance(obj_idx, int):
            return None, "query_edge_not_int"
        if not (0 <= sub_idx < len(all_names) and 0 <= obj_idx < len(all_names)):
            return None, "query_edge_out_of_range"

        if self.reverse_query:
            genders = parse_gender_map(row[14] if len(row) > 14 else "")
            reversed_target = inverse_relation(
                target_relation,
                genders.get(query[0]),
            )
            reversed_relations = reverse_path_relations(
                path_rel_labels,
                path_indices,
                all_names,
                genders,
            )
            if reversed_target is None:
                return None, "reverse_target_failed"
            if reversed_relations is None:
                return None, "reverse_path_relations_failed"
            query = (query[1], query[0])
            sub_idx, obj_idx = obj_idx, sub_idx
            target_relation = reversed_target
            path_indices = list(reversed(path_indices))
            path_rel_labels = reversed_relations

        alignments = align_entity_spans_to_tokens(story_str, all_names, self.tokenizer)

        node_spans = []
        valid_sample = True
        for name_idx, name in enumerate(all_names):
            res = alignments.get(name)
            if res and res["token_span"]:
                node_spans.append(res["token_span"])
            else:
                node_spans.append(None)
                if name_idx in path_indices:
                    valid_sample = False

        if not valid_sample:
            return None, "entity_alignment_failed"

        if target_relation in relation_id_map:
            target_id = relation_id_map[target_relation]
        else:
            target_id = relation_id_map["nothing"]

        item = {
            "story": story_str,
            "query": query,
            "query_indices": (sub_idx, obj_idx),
            "target_id": target_id,
            "path_indices": path_indices,
            "path_rel_labels": path_rel_labels,
            "node_spans": node_spans,
            "num_nodes": len(all_names),
            "hops": official_hop_count(row),
            "reference_path_hops": len(path_indices) - 1,
            "query_direction": "reversed" if self.reverse_query else "original",
            "original_query": original_query,
            "original_target_id": relation_id_map.get(
                original_target_relation,
                relation_id_map["nothing"],
            ),
            "label_changed_under_reverse": target_relation != original_target_relation,
        }

        if self.augment:
            rng = random.Random(self.augment_seed + row_idx) if self.augment_seed is not None else random
            names = list(all_names)
            if len(names) >= 2:
                shuffled = list(names)
                for _ in range(10):
                    rng.shuffle(shuffled)
                    if any(a != b for a, b in zip(names, shuffled)):
                        break
                mapping = {n: s for n, s in zip(names, shuffled)}

                aug_story = apply_bijective_map(story_str, mapping)
                aug_query = (mapping.get(query[0], query[0]), mapping.get(query[1], query[1]))
                aug_all_names = [mapping[n] for n in all_names]

                aug_alignments = align_entity_spans_to_tokens(aug_story, aug_all_names, self.tokenizer)
                aug_node_spans = []
                valid_aug = True
                for j, name in enumerate(aug_all_names):
                    res = aug_alignments.get(name)
                    if res and res["token_span"]:
                        aug_node_spans.append(res["token_span"])
                    else:
                        aug_node_spans.append(None)
                        if path_indices.count(j) > 0:
                            valid_aug = False
                if not valid_aug:
                    return None, "aug_entity_alignment_failed"

                item["aug_story"] = aug_story
                item["aug_query"] = aug_query
                item["aug_node_spans"] = aug_node_spans
            else:
                item["aug_story"] = story_str
                item["aug_query"] = query
                item["aug_node_spans"] = node_spans
        else:
            item["aug_story"] = story_str
            item["aug_query"] = query
            item["aug_node_spans"] = node_spans

        return item, None

    def __getitem__(self, i):
        return self.data[i]
