import torch

from clutrr.config.relation_schema import RELATION_ID_MAP_21_WITH_NOTHING as relation_id_map
from clutrr.models.backbones import is_decoder_only_model


class TruaBatchCollator:
    def __init__(self, tokenizer, device=None, model_type="roberta"):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.decoder_only = is_decoder_only_model(model_type)

    @staticmethod
    def _format_decoder_input(story, query_a, query_b):
        # Keep the raw story at character position 0 so token spans aligned on
        # the story alone remain valid after appending the directed query.
        return (
            f"{story}\n"
            f"Query subject: {query_a}\n"
            f"Query object: {query_b}\n"
            f"Question: What is the family relation from {query_a} to {query_b}?"
        )

    def __call__(self, batch):
        if not batch:
            return None

        stories = [b["story"] for b in batch]
        queries = [f"{b['query'][0]} and {b['query'][1]}" for b in batch]
        targets = torch.tensor([b["target_id"] for b in batch], dtype=torch.long)
        hops = torch.tensor([b["hops"] for b in batch], dtype=torch.long)
        query_indices = torch.tensor([b["query_indices"] for b in batch], dtype=torch.long)

        if self.decoder_only:
            prompts = [
                self._format_decoder_input(b["story"], b["query"][0], b["query"][1])
                for b in batch
            ]
            enc = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                add_special_tokens=True,
            )
        else:
            enc = self.tokenizer(
                stories,
                queries,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                add_special_tokens=True,
            )

        enc_aug = None
        if "aug_story" in batch[0]:
            aug_stories = [b["aug_story"] for b in batch]
            aug_queries = [f"{b['aug_query'][0]} and {b['aug_query'][1]}" for b in batch]
            if self.decoder_only:
                aug_prompts = [
                    self._format_decoder_input(b["aug_story"], b["aug_query"][0], b["aug_query"][1])
                    for b in batch
                ]
                enc_aug = self.tokenizer(
                    aug_prompts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
            else:
                enc_aug = self.tokenizer(
                    aug_stories,
                    aug_queries,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                    add_special_tokens=True,
                )

        max_path_len = max([len(b["path_indices"]) for b in batch])
        path_node_ids = torch.full((len(batch), max_path_len), -1, dtype=torch.long)

        max_rel_len = max([len(b.get("path_rel_labels", [])) for b in batch])
        path_rel_ids = torch.full((len(batch), max_rel_len), -1, dtype=torch.long)

        max_entities = max([b["num_nodes"] for b in batch])
        entity_spans = torch.full((len(batch), max_entities, 2), -1, dtype=torch.long)
        aug_entity_spans = torch.full((len(batch), max_entities, 2), -1, dtype=torch.long)

        for i, b in enumerate(batch):
            p_len = len(b["path_indices"])
            path_node_ids[i, :p_len] = torch.tensor(b["path_indices"], dtype=torch.long)

            rels = b.get("path_rel_labels", [])
            for j, rel in enumerate(rels):
                if j >= max_rel_len:
                    break
                path_rel_ids[i, j] = relation_id_map.get(rel, relation_id_map.get("nothing", 0))

            spans = b["node_spans"]
            for j, span in enumerate(spans):
                if span is not None:
                    entity_spans[i, j, 0] = span[0]
                    entity_spans[i, j, 1] = span[1]
            if "aug_node_spans" in b:
                aug_spans = b["aug_node_spans"]
                for j, span in enumerate(aug_spans):
                    if span is not None:
                        aug_entity_spans[i, j, 0] = span[0]
                        aug_entity_spans[i, j, 1] = span[1]
        return {
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "labels": targets,
            "hops": hops,
            "path_node_ids": path_node_ids,
            "path_rel_ids": path_rel_ids,
            "entity_spans": entity_spans,
            "query_indices": query_indices,
            "raw_batch": batch,
            "aug_input_ids": enc_aug.input_ids if enc_aug else None,
            "aug_attention_mask": enc_aug.attention_mask if enc_aug else None,
            "aug_entity_spans": aug_entity_spans if enc_aug else None,
        }
