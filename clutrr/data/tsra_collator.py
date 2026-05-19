import re

import torch

from clutrr.config.relation_schema import RELATION_ID_MAP_21_WITH_NOTHING as relation_id_map
from clutrr.models.backbones import is_decoder_only_model


class TsraBatchCollator:
    def __init__(self, tokenizer, device=None, model_type="roberta"):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.decoder_only = is_decoder_only_model(model_type)

    @staticmethod
    def _format_decoder_input(story, query_a, query_b):
        return f"Story: {story}\nQuestion: What is the relation between {query_a} and {query_b}?"

    @staticmethod
    def _find_literal_spans(text: str, pattern: str, *, char_base: int = 0) -> list[tuple[int, int]]:
        spans = []
        start_search = 0
        while True:
            start = text.find(pattern, start_search)
            if start == -1:
                break
            end = start + len(pattern)
            spans.append((char_base + start, char_base + end))
            start_search = end
        return spans

    @classmethod
    def _find_entity_char_spans(cls, text: str, name: str, *, char_base: int = 0) -> list[tuple[int, int]]:
        bracketed = cls._find_literal_spans(text, f"[{name}]", char_base=char_base)
        if bracketed:
            return [(start + 1, end - 1) for start, end in bracketed]

        pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)")
        return [(char_base + match.start(), char_base + match.end()) for match in pattern.finditer(text)]

    @staticmethod
    def _token_span_from_char_span(
        offsets,
        char_span: tuple[int, int],
        sequence_ids: list[int | None] | None = None,
        target_sequence_id: int | None = None,
    ) -> tuple[int, int] | None:
        char_start, char_end = char_span
        token_indices = []
        for token_idx, offset in enumerate(offsets):
            tok_start, tok_end = int(offset[0]), int(offset[1])
            if tok_start == tok_end:
                continue
            if target_sequence_id is not None:
                if sequence_ids is None or sequence_ids[token_idx] != target_sequence_id:
                    continue
            if tok_end > char_start and tok_start < char_end:
                token_indices.append(token_idx)

        if not token_indices:
            return None
        return token_indices[0], token_indices[-1] + 1

    @classmethod
    def _align_entity_mentions(
        cls,
        *,
        search_text: str,
        names: list[str],
        offsets,
        sequence_ids: list[int | None] | None = None,
        target_sequence_id: int | None = None,
        char_base: int = 0,
    ) -> list[list[tuple[int, int]]]:
        aligned = []
        for name in names:
            mention_spans = []
            for char_span in cls._find_entity_char_spans(search_text, name, char_base=char_base):
                token_span = cls._token_span_from_char_span(
                    offsets,
                    char_span,
                    sequence_ids=sequence_ids,
                    target_sequence_id=target_sequence_id,
                )
                if token_span is not None:
                    mention_spans.append(token_span)
            aligned.append(mention_spans)
        return aligned

    @staticmethod
    def _offsets_for_row(encoding, row_idx: int):
        offsets = encoding["offset_mapping"][row_idx]
        return offsets.tolist() if hasattr(offsets, "tolist") else offsets

    @staticmethod
    def _sequence_ids_for_row(encoding, row_idx: int):
        try:
            return encoding.sequence_ids(row_idx)
        except (AttributeError, ValueError):
            return None

    @staticmethod
    def _copy_mentions_to_tensors(mentions_by_entity, spans_tensor, mentions_tensor, row_idx: int, max_mentions: int):
        for entity_idx, mentions in enumerate(mentions_by_entity):
            if mentions:
                spans_tensor[row_idx, entity_idx, 0] = mentions[0][0]
                spans_tensor[row_idx, entity_idx, 1] = mentions[0][1]
            for mention_idx, span in enumerate(mentions[:max_mentions]):
                mentions_tensor[row_idx, entity_idx, mention_idx, 0] = span[0]
                mentions_tensor[row_idx, entity_idx, mention_idx, 1] = span[1]

    def _align_batch_mentions(self, batch, encoding, *, prompts=None, augmented: bool = False):
        aligned_batch = []
        for i, b in enumerate(batch):
            names_key = "aug_all_names" if augmented else "all_names"
            story_key = "aug_story" if augmented else "story"
            names = list(b.get(names_key) or [])
            if not names:
                old_mentions_key = "aug_node_mentions" if augmented else "node_mentions"
                aligned_batch.append([list(map(tuple, mentions)) for mentions in b.get(old_mentions_key, [])])
                continue

            offsets = self._offsets_for_row(encoding, i)
            if self.decoder_only:
                prompt = prompts[i]
                story = b[story_key]
                story_start = prompt.find(story)
                char_base = story_start if story_start >= 0 else 0
                aligned = self._align_entity_mentions(
                    search_text=story,
                    names=names,
                    offsets=offsets,
                    sequence_ids=None,
                    target_sequence_id=None,
                    char_base=char_base,
                )
            else:
                aligned = self._align_entity_mentions(
                    search_text=b[story_key],
                    names=names,
                    offsets=offsets,
                    sequence_ids=self._sequence_ids_for_row(encoding, i),
                    target_sequence_id=0,
                    char_base=0,
                )
            aligned_batch.append(aligned)
        return aligned_batch

    def __call__(self, batch):
        if not batch:
            return None

        stories = [b["story"] for b in batch]
        queries = [f"{b['query'][0]} and {b['query'][1]}" for b in batch]
        targets = torch.tensor([b["target_id"] for b in batch], dtype=torch.long)
        hops = torch.tensor([b["hops"] for b in batch], dtype=torch.long)
        query_indices = torch.tensor([b["query_indices"] for b in batch], dtype=torch.long)

        prompts = None
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
                return_offsets_mapping=True,
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
                return_offsets_mapping=True,
                add_special_tokens=True,
            )

        enc_aug = None
        aug_prompts = None
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
                    return_offsets_mapping=True,
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
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )

        aligned_mentions = self._align_batch_mentions(batch, enc, prompts=prompts, augmented=False)
        aligned_aug_mentions = (
            self._align_batch_mentions(batch, enc_aug, prompts=aug_prompts, augmented=True)
            if enc_aug is not None
            else []
        )

        max_path_len = max([len(b["path_indices"]) for b in batch])
        path_node_ids = torch.full((len(batch), max_path_len), -1, dtype=torch.long)

        max_rel_len = max([len(b.get("path_rel_labels", [])) for b in batch])
        path_rel_ids = torch.full((len(batch), max_rel_len), -1, dtype=torch.long)

        max_entities = max([b["num_nodes"] for b in batch])
        entity_spans = torch.full((len(batch), max_entities, 2), -1, dtype=torch.long)
        aug_entity_spans = torch.full((len(batch), max_entities, 2), -1, dtype=torch.long)

        max_mentions = max(
            max((len(mentions) for mentions in sample_mentions), default=0)
            for sample_mentions in aligned_mentions
        )
        max_mentions = max(max_mentions, 1)
        entity_mention_spans = torch.full((len(batch), max_entities, max_mentions, 2), -1, dtype=torch.long)

        if aligned_aug_mentions:
            aug_max_mentions = max(
                max((len(mentions) for mentions in sample_mentions), default=0)
                for sample_mentions in aligned_aug_mentions
            )
        else:
            aug_max_mentions = 1
        aug_max_mentions = max(aug_max_mentions, 1)
        aug_entity_mention_spans = torch.full(
            (len(batch), max_entities, aug_max_mentions, 2),
            -1,
            dtype=torch.long,
        )

        for i, b in enumerate(batch):
            p_len = len(b["path_indices"])
            path_node_ids[i, :p_len] = torch.tensor(b["path_indices"], dtype=torch.long)

            rels = b.get("path_rel_labels", [])
            for j, rel in enumerate(rels):
                if j >= max_rel_len:
                    break
                path_rel_ids[i, j] = relation_id_map.get(rel, relation_id_map.get("nothing", 0))

            self._copy_mentions_to_tensors(
                aligned_mentions[i],
                entity_spans,
                entity_mention_spans,
                i,
                max_mentions,
            )

            if enc_aug is not None:
                self._copy_mentions_to_tensors(
                    aligned_aug_mentions[i],
                    aug_entity_spans,
                    aug_entity_mention_spans,
                    i,
                    aug_max_mentions,
                )

        return {
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "labels": targets,
            "hops": hops,
            "path_node_ids": path_node_ids,
            "path_rel_ids": path_rel_ids,
            "entity_spans": entity_spans,
            "entity_mention_spans": entity_mention_spans,
            "query_indices": query_indices,
            "raw_batch": batch,
            "aug_input_ids": enc_aug.input_ids if enc_aug else None,
            "aug_attention_mask": enc_aug.attention_mask if enc_aug else None,
            "aug_entity_spans": aug_entity_spans if enc_aug else None,
            "aug_entity_mention_spans": aug_entity_mention_spans if enc_aug else None,
        }
