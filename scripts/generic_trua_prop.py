#!/usr/bin/env python3
"""Lightweight dataset loaders for the generic TRUA-Prop runner."""

from __future__ import annotations

import csv
import difflib
import json
import hashlib
import re
import unicodedata
from pathlib import Path


_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_ID_RE = re.compile(r"(?:triple|rule)\d+")


def _stable_limit(samples, limit):
    if limit is None or limit >= len(samples):
        return samples
    if limit <= 0:
        return []

    def stable_key(sample):
        identity = str(sample.get("id") or (sample.get("context", ""), sample.get("query", "")))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    return sorted(samples, key=stable_key)[:limit]


def _split_sentences(text: str):
    return [s.strip() for s in _SENT_RE.split(text.replace("$query$", " ")) if s.strip()]


PROOFWRITER_LABEL_NAMES = ("false", "true", "unknown")
PFOLIO_LABEL_NAMES = PROOFWRITER_LABEL_NAMES
BINARY_LABEL_NAMES = ("false", "true")


def _binary_label(answer):
    if isinstance(answer, bool):
        return int(answer)
    return int(str(answer).strip().lower() in {"true", "yes", "1", "entailment"})


def _proofwriter_label(answer):
    normalized = str(answer).strip().lower()
    labels = {"false": 0, "true": 1, "unknown": 2}
    if normalized not in labels:
        raise ValueError(f"Unsupported ProofWriter OWA answer: {answer!r}")
    return labels[normalized]


def _pfolio_label(answer):
    normalized = str(answer).strip().lower()
    labels = {
        "f": 0,
        "false": 0,
        "t": 1,
        "true": 1,
        "u": 2,
        "uncertain": 2,
        "unknown": 2,
    }
    if normalized in labels:
        return labels[normalized]
    correction = re.search(r"rui_comment:\s*([tfu])\s+is correct", normalized)
    if correction:
        return labels[correction.group(1)]
    if "should be true" in normalized:
        return 1
    line_labels = [labels[token] for token in re.findall(r"(?m)^\s*([tfu])\s*$", normalized)]
    return line_labels[-1] if line_labels else None


def _normalize_folio_text(value):
    value = unicodedata.normalize("NFKC", str(value or ""))
    value = value.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(value.split()).strip()


def _normalized_csv_row(row):
    return {" ".join(key.split()): value for key, value in row.items() if key is not None}


def _pfolio_csv_path(root: Path):
    if root.is_file():
        return root
    candidates = (
        root / "pfolio.csv",
        root / "LogicSpider Manual Annotation Collection - V1 - Proof Collection - All.csv",
    )
    selected = _first_existing(candidates)
    if selected is None:
        raise FileNotFoundError(f"P-FOLIO CSV not found under {root}")
    return selected


def _pfolio_split_path(root: Path, split: str):
    candidates = (
        root / f"folio_{split}.jsonl",
        root / f"folio-{split}.jsonl",
        root / f"folio_v2_{split}.jsonl",
        root / "FOLIO" / f"folio_{split}.jsonl",
        root / "folio" / f"folio_{split}.jsonl",
        root / "folio" / f"folio-{split}.jsonl",
        root / "folio" / f"folio_v2_{split}.jsonl",
    )
    selected = _first_existing(candidates)
    if selected is None:
        raise FileNotFoundError(f"Official FOLIO {split} split not found under {root}")
    return selected


def _official_folio_examples(root: Path):
    examples = []
    for split in ("train", "validation", "test"):
        with _pfolio_split_path(root, split).open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                item = json.loads(line)
                examples.append(
                    {
                        "example_id": int(item["example_id"]),
                        "story_id": str(item.get("story_id", "")),
                        "conclusion": str(item.get("conclusion", "")),
                        "label": _pfolio_label(item.get("label")),
                        "split": split,
                    }
                )
    examples.sort(key=lambda item: item["example_id"])
    observed_ids = [item["example_id"] for item in examples]
    if observed_ids != list(range(len(examples))):
        raise ValueError("Official FOLIO example_id values are not contiguous from zero")
    if any(item["label"] is None for item in examples):
        raise ValueError("Official FOLIO contains an unsupported label")
    return examples


def _pfolio_alignment_score(sample, official):
    sample_conclusion = _normalize_folio_text(sample["split_conclusion"]).lower()
    official_conclusion = _normalize_folio_text(official["conclusion"]).lower()
    similarity = difflib.SequenceMatcher(
        None, sample_conclusion, official_conclusion, autojunk=False
    ).ratio()
    score = 6.0 * similarity
    if sample_conclusion == official_conclusion:
        score += 4.0
    raw_label = _pfolio_label(sample["raw_truth"])
    if raw_label is not None:
        score += 1.0 if raw_label == official["label"] else -1.0
    if sample["story_id"] == official["story_id"]:
        score += 0.5
    return score, similarity


def _align_pfolio_examples(samples, official_examples):
    extra_count = len(samples) - len(official_examples)
    if extra_count < 0 or extra_count > 32:
        raise ValueError(
            "P-FOLIO/official FOLIO size mismatch is too large for ordered alignment: "
            f"{len(samples)} vs {len(official_examples)}"
        )
    official_count = len(official_examples)
    negative_infinity = float("-inf")
    scores = [
        [negative_infinity] * (extra_count + 1)
        for _ in range(official_count + 1)
    ]
    backpointers = {}
    scores[0][0] = 0.0
    for official_index in range(official_count + 1):
        for skipped in range(extra_count + 1):
            current_score = scores[official_index][skipped]
            if current_score == negative_infinity:
                continue
            sample_index = official_index + skipped
            if skipped < extra_count:
                candidate = current_score - 0.25
                if candidate > scores[official_index][skipped + 1]:
                    scores[official_index][skipped + 1] = candidate
                    backpointers[(official_index, skipped + 1)] = (
                        official_index,
                        skipped,
                        "skip",
                    )
            if official_index < official_count:
                match_score, _ = _pfolio_alignment_score(
                    samples[sample_index], official_examples[official_index]
                )
                candidate = current_score + match_score
                if candidate > scores[official_index + 1][skipped]:
                    scores[official_index + 1][skipped] = candidate
                    backpointers[(official_index + 1, skipped)] = (
                        official_index,
                        skipped,
                        "match",
                    )

    state = (official_count, extra_count)
    matched = []
    skipped_indices = []
    while state != (0, 0):
        previous_official, previous_skipped, action = backpointers[state]
        if action == "match":
            sample_index = previous_official + previous_skipped
            official = official_examples[previous_official]
            _, similarity = _pfolio_alignment_score(samples[sample_index], official)
            matched.append((sample_index, previous_official, similarity))
        else:
            skipped_indices.append(previous_official + previous_skipped)
        state = (previous_official, previous_skipped)
    matched.reverse()
    skipped_indices.reverse()
    if len(matched) != official_count or len(skipped_indices) != extra_count:
        raise ValueError("Incomplete ordered alignment between P-FOLIO and official FOLIO")
    return matched, skipped_indices


def _split_pfolio_premises(value):
    return [line.strip() for line in str(value or "").splitlines() if line.strip()]


def _parse_pfolio_references(value):
    references = []
    for token in re.findall(r"(?i)D\s*\d+|(?:P\s*)?\d+", str(value or "")):
        compact = re.sub(r"\s+", "", token).upper()
        references.append(compact if compact.startswith("D") else int(compact.lstrip("P")))
    return references


def _pfolio_leaf_evidence(steps, premise_count):
    if not steps:
        return set()
    cache = {}

    def resolve(step_id, active):
        if step_id in cache:
            return cache[step_id]
        if step_id in active:
            raise ValueError(f"Cyclic P-FOLIO derivation reference at {step_id}")
        if step_id not in steps:
            raise ValueError(f"Unknown P-FOLIO derivation reference: {step_id}")
        leaves = set()
        for reference in steps[step_id]:
            if isinstance(reference, int):
                if not 1 <= reference <= premise_count:
                    raise ValueError(
                        f"P-FOLIO premise reference {reference} exceeds {premise_count} premises"
                    )
                leaves.add(reference - 1)
            else:
                leaves.update(resolve(reference, active | {step_id}))
        cache[step_id] = leaves
        return leaves

    return resolve(next(reversed(steps)), set())


def load_pfolio_corpus(root: Path, return_audit=False):
    root = Path(root)
    official_examples = _official_folio_examples(root)
    raw_samples = []
    current = None
    story_premises = {}

    def finish_current():
        nonlocal current
        if current is None:
            return
        sentences = current["sentences"]
        if current["proof_error"]:
            evidence_indices = set()
            evidence_status = "malformed_reference"
        elif not current["steps"]:
            evidence_indices = set()
            evidence_status = "no_reference_proof"
        else:
            try:
                evidence_indices = _pfolio_leaf_evidence(current["steps"], len(sentences))
                evidence_status = "mapped"
            except ValueError:
                evidence_indices = set()
                evidence_status = "unmappable_reference"
        current["evidence_indices"] = evidence_indices
        current["evidence_status"] = evidence_status
        raw_samples.append(current)
        current = None

    with _pfolio_csv_path(root).open(encoding="utf-8-sig", newline="") as source:
        for row_index, raw_row in enumerate(csv.DictReader(source), start=2):
            row = _normalized_csv_row(raw_row)
            truth = str(row.get("Truth Value", "")).strip()
            original_premises = _split_pfolio_premises(row.get("Premises - NL", ""))
            corrected_premises = _split_pfolio_premises(row.get("Corrected Premises - NL", ""))
            original_conclusion = str(row.get("Conclusions - NL", "") or "").strip()
            corrected_conclusion = str(row.get("Corrected Conclusions - NL", "") or "").strip()
            is_example = bool(truth and (original_conclusion or corrected_conclusion))
            if is_example:
                finish_current()
                story_id = str(row.get("story_id", "")).strip()
                if original_premises or corrected_premises:
                    story_premises[story_id] = (original_premises, corrected_premises)
                elif story_id in story_premises:
                    original_premises, corrected_premises = story_premises[story_id]
                sentences = corrected_premises or original_premises
                query = corrected_conclusion or original_conclusion
                if not sentences or not query:
                    raise ValueError(f"Incomplete P-FOLIO example at CSV row {row_index}")
                current = {
                    "raw_index": len(raw_samples),
                    "story_id": story_id,
                    "sentences": sentences,
                    "query": query,
                    "raw_truth": truth,
                    "steps": {},
                    "proof_error": False,
                    "split_premises": original_premises or corrected_premises,
                    "split_conclusion": original_conclusion or corrected_conclusion,
                }
                continue

            step_id = re.sub(r"\s+", "", str(row.get("Derivation index", "") or "")).upper()
            if current is not None and step_id:
                if not re.fullmatch(r"D\d+", step_id):
                    current["proof_error"] = True
                    continue
                if step_id in current["steps"]:
                    current["proof_error"] = True
                    continue
                references = _parse_pfolio_references(row.get("Premises used", ""))
                if not references:
                    current["proof_error"] = True
                    continue
                current["steps"][step_id] = references
    finish_current()

    alignment, skipped_indices = _align_pfolio_examples(raw_samples, official_examples)
    samples = []
    for sample_index, official_index, similarity in alignment:
        raw_sample = raw_samples[sample_index]
        official = official_examples[official_index]
        label = _pfolio_label(raw_sample["raw_truth"])
        if label is None:
            raise ValueError(
                "Aligned P-FOLIO example has an unresolved truth value: "
                f"raw index {raw_sample['raw_index']}"
            )
        evidence_indices = raw_sample["evidence_indices"]
        evidence_status = raw_sample["evidence_status"]
        if label == 2:
            evidence_indices = set()
            evidence_status = "unknown_label"
        samples.append(
            {
                "id": f"pfolio:{official['example_id']}",
                "context": " ".join(raw_sample["sentences"]),
                "query": raw_sample["query"],
                "sentences": raw_sample["sentences"],
                "trace_labels": [
                    int(index in evidence_indices)
                    for index in range(len(raw_sample["sentences"]))
                ],
                "label": label,
                "depth": len(raw_sample["steps"]),
                "split": official["split"],
                "story_id": raw_sample["story_id"],
                "official_story_id": official["story_id"],
                "official_label": official["label"],
                "alignment_similarity": similarity,
                "evidence_status": evidence_status,
            }
        )
    if len(samples) != len(official_examples):
        raise ValueError("P-FOLIO alignment did not recover every official FOLIO example")
    audit = {
        "raw_examples": len(raw_samples),
        "official_examples": len(official_examples),
        "skipped_editorial_rows": [
            {
                "raw_index": raw_samples[index]["raw_index"],
                "story_id": raw_samples[index]["story_id"],
                "conclusion": raw_samples[index]["split_conclusion"],
                "truth_value": raw_samples[index]["raw_truth"],
            }
            for index in skipped_indices
        ],
        "alignment_similarity": {
            "minimum": min(sample["alignment_similarity"] for sample in samples),
            "mean": sum(sample["alignment_similarity"] for sample in samples) / len(samples),
            "below_0_5": sum(sample["alignment_similarity"] < 0.5 for sample in samples),
            "below_0_8": sum(sample["alignment_similarity"] < 0.8 for sample in samples),
        },
        "label_disagreements_with_folio": sum(
            sample["label"] != sample["official_label"] for sample in samples
        ),
    }
    return (samples, audit) if return_audit else samples


def load_pfolio(root: Path, split: str, limit=None):
    samples = [sample for sample in load_pfolio_corpus(root) if sample["split"] == split]
    return _stable_limit(samples, limit)


def _proof_ids(question):
    proofs = question.get("proofs") or ""
    if not proofs and question.get("proofsWithIntermediates"):
        proofs = " ".join(str(p.get("representation", "")) for p in question["proofsWithIntermediates"])
    return set(_ID_RE.findall(proofs))


def _first_existing(candidates):
    """Resolve one dataset variant without silently mixing multiple variants."""
    return next((path for path in candidates if path.exists()), None)


def _load_meta_depth_dirs(
    depth_dirs,
    split: str,
    limit=None,
    qdep_filter=None,
    answer_encoder=_binary_label,
    no_evidence_labels=(),
):
    qdep_filter = set(qdep_filter) if qdep_filter is not None else None
    samples = []
    for depth_dir in depth_dirs:
        path = depth_dir / f"meta-{split}.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            item = json.loads(line)
            facts = item.get("triples", {})
            rules = item.get("rules", {})
            sent_items = []
            for key, val in sorted(facts.items(), key=lambda kv: int(re.sub(r"\D", "", kv[0]) or 0)):
                sent_items.append((key, val.get("text", "")))
            for key, val in sorted(rules.items(), key=lambda kv: int(re.sub(r"\D", "", kv[0]) or 0)):
                sent_items.append((key, val.get("text", "")))
            sentences = [s for _, s in sent_items if s]
            context = " ".join(sentences) or item.get("theory", "")
            for qid, q in item.get("questions", {}).items():
                depth = int(q.get("QDep", item.get("maxD", -1)))
                if qdep_filter is not None and depth not in qdep_filter:
                    continue
                label = answer_encoder(q.get("answer"))
                ids = set() if label in no_evidence_labels else _proof_ids(q)
                trace = [1 if key in ids else 0 for key, _ in sent_items]
                samples.append(
                    {
                        "id": f"{item.get('id', '')}_{qid}",
                        "context": context,
                        "query": q.get("question", ""),
                        "sentences": sentences or _split_sentences(context),
                        "trace_labels": trace,
                        "label": label,
                        "depth": depth,
                    }
                )
    return _stable_limit(samples, limit)


def load_proofwriter(root: Path, depths, split: str, limit=None):
    depth_dirs = []
    for d in depths:
        candidates = [root / "OWA" / f"depth-{d}", root / "OWA" / f"depth-{d}ext-NatLang"]
        selected = _first_existing(candidates)
        if selected is not None:
            depth_dirs.append(selected)
    return _load_meta_depth_dirs(
        depth_dirs,
        split,
        limit,
        answer_encoder=_proofwriter_label,
        no_evidence_labels=(2,),
    )


def load_ruletaker_raw(root: Path, depth_dirs, split: str, limit=None, qdeps=None):
    dirs = []
    for d in depth_dirs:
        candidates = [root / f"depth-{d}", root / f"depth-{d}ext", root / f"depth-{d}ext-NatLang"]
        selected = _first_existing(candidates)
        if selected is not None:
            dirs.append(selected)
    return _load_meta_depth_dirs(dirs, split, limit, qdep_filter=qdeps)


def load_ruletaker_gfair(root: Path, split: str, limit=None):
    path = root / f"{split}.jsonl"
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            raw = item.get("context", "")
            if "$query$" in raw:
                context, query = raw.split("$query$", 1)
            else:
                context, query = raw, item.get("query", "")
            sentences = _split_sentences(context)
            rule_texts = {r[-1] for r in item.get("meta", {}).get("orig_rules", []) if isinstance(r, list) and r}
            rule_norm = {r.rstrip(".") for r in rule_texts}
            trace = [1 if s.rstrip(".") in rule_norm else 0 for s in sentences]
            if not any(trace) and sentences:
                trace = [1] + [0] * (len(sentences) - 1)
            m = re.search(r"-D(\d+)-", item.get("id", ""))
            samples.append(
                {
                    "id": item.get("id", ""),
                    "context": context.strip(),
                    "query": query.strip(),
                    "sentences": sentences,
                    "trace_labels": trace,
                    "label": _binary_label(item.get("answer")),
                    "depth": int(m.group(1)) if m else -1,
                }
            )
    return _stable_limit(samples, limit)


def _iter_pronto_examples(obj):
    if isinstance(obj, list):
        yield from obj
    elif isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, dict) and "test_example" in value:
                yield value["test_example"]
            elif isinstance(value, dict) and "question" in value:
                yield value


def load_prontoqa(root: Path, files, limit=None):
    samples = []
    for name in files:
        path = root / name
        if not path.exists():
            continue
        hop = int(name.split("hop", 1)[0]) if "hop" in name else -1
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in _iter_pronto_examples(data):
            context = item.get("question", "")
            query = item.get("query", "")
            chain = item.get("chain_of_thought") or item.get("proof") or []
            if isinstance(chain, str):
                chain = _split_sentences(chain)
            sentences = _split_sentences(context)
            chain_norm = {c.strip().rstrip(".").lower() for c in chain}
            trace = [1 if s.strip().rstrip(".").lower() in chain_norm else 0 for s in sentences]
            samples.append(
                {
                    "id": f"{name}:{len(samples)}",
                    "context": context,
                    "query": query,
                    "sentences": sentences,
                    "trace_labels": trace,
                    "label": _binary_label(item.get("answer", True)),
                    "depth": hop,
                }
            )
    return _stable_limit(samples, limit)
