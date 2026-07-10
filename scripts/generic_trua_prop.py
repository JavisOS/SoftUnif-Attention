#!/usr/bin/env python3
"""Lightweight dataset loaders for the generic TRUA-Prop runner."""

from __future__ import annotations

import json
import hashlib
import re
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


def _label(answer):
    if isinstance(answer, bool):
        return int(answer)
    return int(str(answer).strip().lower() in {"true", "yes", "1", "entailment"})


def _proof_ids(question):
    proofs = question.get("proofs") or ""
    if not proofs and question.get("proofsWithIntermediates"):
        proofs = " ".join(str(p.get("representation", "")) for p in question["proofsWithIntermediates"])
    return set(_ID_RE.findall(proofs))


def _load_meta_depth_dirs(depth_dirs, split: str, limit=None, qdep_filter=None):
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
                ids = _proof_ids(q)
                trace = [1 if key in ids else 0 for key, _ in sent_items]
                samples.append(
                    {
                        "id": f"{item.get('id', '')}_{qid}",
                        "context": context,
                        "query": q.get("question", ""),
                        "sentences": sentences or _split_sentences(context),
                        "trace_labels": trace,
                        "label": _label(q.get("answer")),
                        "depth": depth,
                    }
                )
    return _stable_limit(samples, limit)


def load_proofwriter(root: Path, depths, split: str, limit=None):
    depth_dirs = []
    for d in depths:
        candidates = [root / "OWA" / f"depth-{d}", root / "OWA" / f"depth-{d}ext-NatLang"]
        depth_dirs.extend([p for p in candidates if p.exists()])
    return _load_meta_depth_dirs(depth_dirs, split, limit)


def load_ruletaker_raw(root: Path, depth_dirs, split: str, limit=None, qdeps=None):
    dirs = []
    for d in depth_dirs:
        candidates = [root / f"depth-{d}", root / f"depth-{d}ext", root / f"depth-{d}ext-NatLang"]
        dirs.extend([p for p in candidates if p.exists()])
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
                    "label": _label(item.get("answer")),
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
                    "label": _label(item.get("answer", True)),
                    "depth": hop,
                }
            )
    return _stable_limit(samples, limit)
