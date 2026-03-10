import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import yaml
from openai import OpenAI
from tqdm import tqdm

from clutrr.config.defaults import DEFAULT_CLUTRR_DATASET, DEFAULT_CLUTRR_ROOT
from clutrr.config.relation_schema import RELATION_ID_MAP_21_WITH_NOTHING
from clutrr.utils.graph_reasoning import apply_bijective_map
from clutrr.utils.parsing import parse_pair_literal


MODEL_ALIASES = {
    "gemini3.0pro": "google/gemini-3-pro-preview",
    "gemini-3.0-pro": "google/gemini-3-pro-preview",
    "google/gemini-3.0-pro": "google/gemini-3-pro-preview",
}

EVAL_DEFAULTS = {
    "root": DEFAULT_CLUTRR_ROOT,
    "dataset": DEFAULT_CLUTRR_DATASET,
    "split": "test",
    "limit": 100,
    "model": "google/gemini-3-pro-preview",
    "base_url": "https://openrouter.ai/api/v1",
    "temperature": 0.0,
    "max_tokens": 256,
    "retries": 3,
    "sleep_seconds": 0.0,
    "output_dir": "outputs/gemini_eval",
    "raw_only": False,
    "resume": False,
}


def _extract_config_path(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args(argv)
    return args.config


def _load_yaml_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping/dict: {config_path}")
    return data


def _resolve_model_name(model_name: str) -> str:
    lowered = model_name.strip().lower()
    return MODEL_ALIASES.get(lowered, model_name)


def build_arg_parser(defaults=None):
    defaults = defaults or EVAL_DEFAULTS
    parser = argparse.ArgumentParser(
        prog="python -m clutrr.cli.eval_gemini_openai",
        description="Evaluate Gemini/OpenAI-compatible APIs on CLUTRR with TSRA-style metrics.",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path. CLI args override YAML values.")
    parser.add_argument("--root", type=str, default=defaults["root"], help="CLUTRR data root.")
    parser.add_argument("--dataset", type=str, default=defaults["dataset"], help="CLUTRR dataset folder.")
    parser.add_argument("--split", type=str, default=defaults["split"], choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=defaults["limit"], help="Max number of samples to evaluate.")
    parser.add_argument("--model", type=str, default=defaults["model"], help="Model name for OpenAI-compatible API.")
    parser.add_argument("--base_url", type=str, default=defaults["base_url"], help="OpenAI-compatible API base URL.")
    parser.add_argument("--api_key", type=str, default=None, help="API key. Prefer env OPENROUTER_API_KEY/OPENAI_API_KEY.")
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--max_tokens", type=int, default=defaults["max_tokens"])
    parser.add_argument("--retries", type=int, default=defaults["retries"])
    parser.add_argument("--sleep_seconds", type=float, default=defaults["sleep_seconds"])
    parser.add_argument("--output_dir", type=str, default=defaults["output_dir"])
    parser.add_argument(
        "--raw_only",
        action="store_true",
        default=bool(defaults["raw_only"]),
        help="Evaluate only original samples (standard accuracy) without renamed-consistency pass.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=bool(defaults["resume"]),
        help="Resume from existing predictions file (same run_name) if present.",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name for output files.")
    return parser


def parse_args():
    config_path = _extract_config_path()
    defaults = dict(EVAL_DEFAULTS)
    yaml_config = _load_yaml_config(config_path)
    unknown_keys = sorted(set(yaml_config.keys()) - set(defaults.keys()))
    if unknown_keys:
        raise ValueError(f"Unknown config keys in {config_path}: {', '.join(unknown_keys)}")
    defaults.update(yaml_config)
    parser = build_arg_parser(defaults=defaults)
    return parser.parse_args()


def _sort_key_for_split_file(path: str):
    base = os.path.basename(path)
    m = re.search(r"(\d+)\.(\d+)_", base)
    if m:
        return int(m.group(1)), int(m.group(2)), base
    return 9999, 9999, base


def _load_split_rows(root: str, dataset: str, split: str, limit: int | None):
    if limit is not None and limit <= 0:
        limit = None

    dataset_dir = os.path.join(root, dataset)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    file_names = sorted(
        (
            os.path.join(dataset_dir, fname)
            for fname in os.listdir(dataset_dir)
            if f"_{split}.csv" in fname
        ),
        key=_sort_key_for_split_file,
    )

    rows = []
    for file_name in file_names:
        with open(file_name, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    return rows
    return rows


def _extract_hop(row):
    try:
        task_name = row[10]
        return int(task_name.split(".")[-1])
    except Exception:
        return -1


def _extract_unique_names(story):
    names = re.findall(r"\[(.*?)\]", story)
    seen = set()
    ordered = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _build_shift_mapping(names):
    if len(names) < 2:
        return {}
    shifted = names[1:] + names[:1]
    return {src: dst for src, dst in zip(names, shifted)}


def _build_prompt(story, query):
    # IMPORTANT: CLUTRR labels are directional for query (A, B): predict relation of B to A.
    relations = [r for r in RELATION_ID_MAP_21_WITH_NOTHING.keys() if r != "nothing"]
    relation_text = ", ".join(relations)
    q_sub, q_obj = query
    return (
        "You are given one CLUTRR family reasoning sample.\n"
        "Choose exactly one relation label from the allowed set.\n"
        f"Allowed labels: {relation_text}\n\n"
        f"Story: {story}\n"
        f"Query pair: ({q_sub}, {q_obj})\n"
        f"Task: Predict the relation of {q_obj} TO {q_sub}.\n"
        "Answer with one label only."
    )


def _normalize_pred(text: str):
    if not text:
        return "nothing"

    s = text.strip().lower()
    s = s.replace('"', " ").replace("'", " ")
    s = re.sub(r"[^a-z0-9\-\s_]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    label_candidates = list(RELATION_ID_MAP_21_WITH_NOTHING.keys())
    aliases = {}
    for label in label_candidates:
        aliases[label] = label
        aliases[label.replace("-", " ")] = label
        aliases[label.replace("-", "_")] = label
        aliases[label.replace("-", "")] = label

    if s in aliases:
        return aliases[s]

    for alias, canonical in sorted(aliases.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, s):
            return canonical

    return "nothing"


def _extract_text_from_message(message):
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        chunks = [item.get("text", "") for item in content if isinstance(item, dict)]
        joined = " ".join(chunks).strip()
        if joined:
            return joined

    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning

    reasoning_details = getattr(message, "reasoning_details", None)
    if isinstance(reasoning_details, list):
        chunks = []
        for item in reasoning_details:
            if isinstance(item, dict):
                txt = item.get("text", "")
                if txt:
                    chunks.append(txt)
        joined = " ".join(chunks).strip()
        if joined:
            return joined

    return ""


def _call_model(client: OpenAI, model: str, prompt: str, temperature: float, max_tokens: int, retries: int):
    last_exc = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Return only one candidate relation label; no extra text."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if not resp.choices:
                return ""
            return _extract_text_from_message(resp.choices[0].message)
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"API call failed after {retries} retries: {last_exc}") from last_exc


def _effective_max_tokens(model_name: str, requested_max_tokens: int) -> int:
    lowered = model_name.strip().lower()
    if lowered.startswith("gpt-5") and requested_max_tokens < 1024:
        # GPT-5 reasoning responses can consume hidden tokens; low budgets often yield empty visible text.
        return 1024
    return requested_max_tokens


def _load_existing_progress(pred_path: Path):
    done_ids = set()
    stats = {
        "total": 0,
        "correct_base": 0,
        "correct_mod": 0,
        "consistent": 0,
        "consistent_and_correct": 0,
        "by_hop_total": {},
        "by_hop_correct": {},
    }
    if not pred_path.exists():
        return done_ids, stats, 0

    duplicate_rows_ignored = 0
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                x = json.loads(line)
            except Exception:
                continue

            sid = x.get("id")
            sid = str(sid) if sid is not None else None
            if sid is not None and sid in done_ids:
                duplicate_rows_ignored += 1
                continue
            if sid is not None:
                done_ids.add(sid)

            hop = x.get("hop", -1)
            stats["total"] += 1
            stats["by_hop_total"][hop] = stats["by_hop_total"].get(hop, 0) + 1

            if x.get("correct_base") is True:
                stats["correct_base"] += 1
                stats["by_hop_correct"][hop] = stats["by_hop_correct"].get(hop, 0) + 1
            if x.get("correct_mod") is True:
                stats["correct_mod"] += 1
            if x.get("consistent") is True:
                stats["consistent"] += 1
            if x.get("consistent") is True and x.get("correct_base") is True:
                stats["consistent_and_correct"] += 1

    return done_ids, stats, duplicate_rows_ignored


def evaluate(args):
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Set OPENROUTER_API_KEY (recommended) or --api_key.")

    resolved_model = _resolve_model_name(args.model)

    default_headers = {}
    if "openrouter.ai" in args.base_url:
        default_headers = {
            "HTTP-Referer": "https://github.com/",
            "X-Title": "TSRA-CLUTRR-Eval",
        }

    client = OpenAI(api_key=api_key, base_url=args.base_url, default_headers=default_headers)

    effective_max_tokens = _effective_max_tokens(resolved_model, args.max_tokens)
    if effective_max_tokens != args.max_tokens:
        print(
            f"[Info] Auto-adjust max_tokens for {resolved_model}: "
            f"{args.max_tokens} -> {effective_max_tokens}"
        )

    rows = _load_split_rows(args.root, args.dataset, args.split, args.limit)
    if not rows:
        raise RuntimeError("No rows loaded for evaluation.")

    total = 0
    correct_base = 0
    correct_mod = 0
    consistent = 0
    consistent_and_correct = 0
    by_hop_total = {}
    by_hop_correct = {}

    run_name = args.run_name or f"{resolved_model.replace('/', '_')}_{args.dataset}_{args.split}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{run_name}.predictions.jsonl"
    summary_path = output_dir / f"{run_name}.summary.json"

    done_ids = set()
    file_mode = "w"
    if args.resume and pred_path.exists():
        done_ids, prior, dup_ignored = _load_existing_progress(pred_path)
        total = prior["total"]
        correct_base = prior["correct_base"]
        correct_mod = prior["correct_mod"]
        consistent = prior["consistent"]
        consistent_and_correct = prior["consistent_and_correct"]
        by_hop_total = prior["by_hop_total"]
        by_hop_correct = prior["by_hop_correct"]
        file_mode = "a"
        print(f"[Info] Resume enabled: loaded {len(done_ids)} unique completed samples from {pred_path}")
        if dup_ignored > 0:
            print(f"[Info] Ignored {dup_ignored} duplicate rows already present in predictions file")

    with pred_path.open(file_mode, encoding="utf-8") as fw:
        for row in tqdm(rows, desc=f"Evaluating {resolved_model}"):
            story = row[2]
            query = parse_pair_literal(row[3])
            if query is None:
                continue
            target = row[5].strip().lower()
            hop = _extract_hop(row)
            sample_id = row[1] if len(row) > 1 else str(total)
            sample_id = str(sample_id)
            if sample_id in done_ids:
                continue

            prompt_base = _build_prompt(story, query)
            raw_base = _call_model(
                client,
                resolved_model,
                prompt_base,
                args.temperature,
                effective_max_tokens,
                args.retries,
            )
            pred_base = _normalize_pred(raw_base)

            pred_mod = None
            raw_mod = None
            is_correct_mod = None
            is_consistent = None
            if not args.raw_only:
                names = _extract_unique_names(story)
                mapping = _build_shift_mapping(names)
                story_mod = apply_bijective_map(story, mapping) if mapping else story
                query_mod = (mapping.get(query[0], query[0]), mapping.get(query[1], query[1]))

                prompt_mod = _build_prompt(story_mod, query_mod)
                raw_mod = _call_model(
                    client,
                    resolved_model,
                    prompt_mod,
                    args.temperature,
                    effective_max_tokens,
                    args.retries,
                )
                pred_mod = _normalize_pred(raw_mod)

            total += 1
            is_correct = pred_base == target
            if pred_mod is not None:
                is_correct_mod = pred_mod == target
                is_consistent = pred_base == pred_mod

            if is_correct:
                correct_base += 1
            if is_correct_mod:
                correct_mod += 1
            if is_consistent:
                consistent += 1
            if is_consistent and is_correct:
                consistent_and_correct += 1

            if hop not in by_hop_total:
                by_hop_total[hop] = 0
                by_hop_correct[hop] = 0
            by_hop_total[hop] += 1
            if is_correct:
                by_hop_correct[hop] += 1

            fw.write(
                json.dumps(
                    {
                        "id": sample_id,
                        "hop": hop,
                        "target": target,
                        "pred_base": pred_base,
                        "pred_mod": pred_mod,
                        "raw_base": raw_base,
                        "raw_mod": raw_mod,
                        "consistent": is_consistent,
                        "correct_base": is_correct,
                        "correct_mod": is_correct_mod,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            done_ids.add(sample_id)

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    overall = correct_base / total if total else 0.0
    renamed = (correct_mod / total if total else 0.0) if not args.raw_only else None
    consistency = (consistent / total if total else 0.0) if not args.raw_only else None
    consistent_correct = (consistent_and_correct / total if total else 0.0) if not args.raw_only else None

    short_corr = sum(by_hop_correct.get(h, 0) for h in [2, 3])
    short_tot = sum(by_hop_total.get(h, 0) for h in [2, 3])
    short_acc = short_corr / short_tot if short_tot else 0.0

    long_corr = sum(by_hop_correct.get(h, 0) for h in range(6, 15))
    long_tot = sum(by_hop_total.get(h, 0) for h in range(6, 15))
    long_acc = long_corr / long_tot if long_tot else 0.0

    hop_accuracy = {}
    for h in range(2, 11):
        hop_total = by_hop_total.get(h, 0)
        hop_correct = by_hop_correct.get(h, 0)
        hop_accuracy[str(h)] = {
            "accuracy": (hop_correct / hop_total) if hop_total > 0 else None,
            "correct": hop_correct,
            "total": hop_total,
        }

    summary = {
        "requested_model": args.model,
        "resolved_model": resolved_model,
        "evaluation_mode": "raw_only" if args.raw_only else "tsra_style",
        "base_url": args.base_url,
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": total,
        "max_tokens_requested": args.max_tokens,
        "max_tokens_effective": effective_max_tokens,
        "overall": overall,
        "renamed": renamed,
        "consistency": consistency,
        "consistent_and_correct": consistent_correct,
        "short_hop": short_acc,
        "long_hop": long_acc,
        "hop_accuracy_2_to_10": hop_accuracy,
        "predictions_path": str(pred_path),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    evaluate(parse_args())
