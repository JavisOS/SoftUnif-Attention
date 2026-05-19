import random

from clutrr.config.relation_schema import RELATION_ID_MAP_21_WITH_NOTHING as relation_id_map
from clutrr.data.clutrr_dataset import CLUTRRDataset
from clutrr.utils.entity_alignment import align_entity_spans_to_tokens
from clutrr.utils.graph_reasoning import apply_bijective_map, parse_graph_and_path
from clutrr.utils.parsing import parse_pair_literal


class TsraClutrrDataset(CLUTRRDataset):
    def __init__(
        self,
        root,
        dataset,
        split,
        data_percentage=100,
        tokenizer=None,
        augment=False,
        augment_seed=None,
    ):
        super().__init__(root, dataset, split, data_percentage)
        self.tokenizer = tokenizer
        self.augment = augment
        self.augment_seed = augment_seed
        self.split = split

        raw_rows = list(self.data)
        self.stats = {
            "split": split,
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
        self.stats.update(self._compute_mention_stats())

    def _compute_mention_stats(self):
        entity_total = 0
        entity_multi = 0
        mention_total = 0
        max_mentions = 0

        for item in self.data:
            for mentions in item.get("node_mentions", []):
                if not mentions:
                    continue
                entity_total += 1
                num_mentions = len(mentions)
                mention_total += num_mentions
                max_mentions = max(max_mentions, num_mentions)
                if num_mentions > 1:
                    entity_multi += 1

        avg_mentions = mention_total / entity_total if entity_total > 0 else 0.0
        multi_ratio = entity_multi / entity_total if entity_total > 0 else 0.0
        return {
            "entity_total": entity_total,
            "entity_multi": entity_multi,
            "mention_total": mention_total,
            "avg_mentions_per_entity": avg_mentions,
            "multi_mention_ratio": multi_ratio,
            "max_mentions_per_entity": max_mentions,
        }

    def _build_item(self, row, row_idx: int):
        if self.tokenizer is None:
            raise ValueError("TsraClutrrDataset requires a tokenizer for entity span alignment.")

        graph_info = parse_graph_and_path(row)
        if graph_info is None:
            return None, "graph_parse_failed"

        story_str = row[2]
        query = parse_pair_literal(row[3])
        if query is None:
            return None, "query_parse_failed"
        target_relation = row[5]
        all_names = graph_info["all_names"]
        path_rel_labels = graph_info.get("path_relation_labels", [])

        query_edge = parse_pair_literal(row[13])
        if query_edge is None:
            return None, "query_edge_parse_failed"
        sub_idx, obj_idx = query_edge
        if not isinstance(sub_idx, int) or not isinstance(obj_idx, int):
            return None, "query_edge_not_int"
        if not (0 <= sub_idx < len(all_names) and 0 <= obj_idx < len(all_names)):
            return None, "query_edge_out_of_range"

        alignments = align_entity_spans_to_tokens(story_str, all_names, self.tokenizer)

        node_spans = []
        node_mentions = []
        valid_sample = True
        for name_idx, name in enumerate(all_names):
            res = alignments.get(name)
            mention_spans = []
            if res:
                for occ in res.get("all_occurrences", []):
                    token_span = occ.get("token_span")
                    if token_span is not None:
                        mention_spans.append(tuple(token_span))
                if not mention_spans and res.get("token_span") is not None:
                    mention_spans.append(tuple(res["token_span"]))

            node_mentions.append(mention_spans)
            if mention_spans:
                node_spans.append(mention_spans[0])
            else:
                node_spans.append(None)
                if name_idx in graph_info["path_node_indices"]:
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
            "path_indices": graph_info["path_node_indices"],
            "path_rel_labels": path_rel_labels,
            "all_names": all_names,
            "node_spans": node_spans,
            "node_mentions": node_mentions,
            "num_nodes": len(all_names),
            "hops": len(graph_info["path_node_indices"]) - 1,
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
                aug_node_mentions = []
                valid_aug = True
                for j, name in enumerate(aug_all_names):
                    res = aug_alignments.get(name)
                    mention_spans = []
                    if res:
                        for occ in res.get("all_occurrences", []):
                            token_span = occ.get("token_span")
                            if token_span is not None:
                                mention_spans.append(tuple(token_span))
                        if not mention_spans and res.get("token_span") is not None:
                            mention_spans.append(tuple(res["token_span"]))

                    aug_node_mentions.append(mention_spans)
                    if mention_spans:
                        aug_node_spans.append(mention_spans[0])
                    else:
                        aug_node_spans.append(None)
                        if graph_info["path_node_indices"].count(j) > 0:
                            valid_aug = False
                if not valid_aug:
                    return None, "aug_entity_alignment_failed"

                item["aug_story"] = aug_story
                item["aug_query"] = aug_query
                item["aug_all_names"] = aug_all_names
                item["aug_node_spans"] = aug_node_spans
                item["aug_node_mentions"] = aug_node_mentions
            else:
                item["aug_story"] = story_str
                item["aug_query"] = query
                item["aug_all_names"] = all_names
                item["aug_node_spans"] = node_spans
                item["aug_node_mentions"] = node_mentions
        else:
            item["aug_story"] = story_str
            item["aug_query"] = query
            item["aug_all_names"] = all_names
            item["aug_node_spans"] = node_spans
            item["aug_node_mentions"] = node_mentions

        return item, None

    def __getitem__(self, i):
        return self.data[i]
