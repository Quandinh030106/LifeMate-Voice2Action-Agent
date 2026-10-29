"""Evaluation metric implementation for LifeMate (Zalo AI Challenge 2026).
Matches the official evaluation specifications:
- Exact state_hash match (otherwise score = 0)
- Tool-call matching with non-fuzzy exact match and fuzzy token-F1 > 0.6
- Precision, Recall, and F1 calculation
- Handles out-of-scope (empty calls) and missing information (sys.ask_missing_info)
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Free-text keys that use fuzzy matching (token-F1 > 0.6)
FREE_TEXT_KEYS = {
    "address", "body", "content", "dropoff", "expression",
    "label", "location", "name", "note", "pickup", "subject", "title"
}


def normalize_text_nfc(text: Any) -> str:
    """Lowercase and normalize to Unicode NFC."""
    if text is None:
        return ""
    s = str(text).lower()
    return unicodedata.normalize("NFC", s)


def get_word_tokens(text: str) -> List[str]:
    """Tokenize text into words/numbers."""
    normalized = normalize_text_nfc(text)
    return re.findall(r"\w+", normalized)


def token_f1(pred_text: Any, gold_text: Any) -> float:
    """Compute token-level F1 between two strings after NFC normalization."""
    pred_tokens = get_word_tokens(pred_text)
    gold_tokens = get_word_tokens(gold_text)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    # Count token frequencies
    pred_counts: Dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    gold_counts: Dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    common = 0
    for t, count in pred_counts.items():
        if t in gold_counts:
            common += min(count, gold_counts[t])

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def is_argument_match(
    pred_val: Any,
    gold_val: Any,
    arg_name: str,
    fuzzy_keys: Set[str] = FREE_TEXT_KEYS
) -> bool:
    """Check if argument matches: exact if non-fuzzy, token-F1 > 0.6 if fuzzy."""
    if arg_name in fuzzy_keys:
        return token_f1(pred_val, gold_val) > 0.6

    # Non-fuzzy argument: must be identical
    if isinstance(pred_val, str) and isinstance(gold_val, str):
        return normalize_text_nfc(pred_val).strip() == normalize_text_nfc(gold_val).strip()
    return pred_val == gold_val


def is_call_match(
    pred_call: Dict[str, Any],
    gold_call: Dict[str, Any],
    fuzzy_keys: Set[str] = FREE_TEXT_KEYS
) -> bool:
    """A submitted call matches a gold call when:
    1. Tool name is identical
    2. Every non-fuzzy argument is identical
    3. Every fuzzy argument has token-F1 > 0.6
    """
    if pred_call.get("name") != gold_call.get("name"):
        return False

    pred_args = pred_call.get("arguments") or {}
    gold_args = gold_call.get("arguments") or {}

    # All gold args must be satisfied
    all_keys = set(pred_args.keys()) | set(gold_args.keys())
    for k in all_keys:
        val_pred = pred_args.get(k)
        val_gold = gold_args.get(k)
        if not is_argument_match(val_pred, val_gold, k, fuzzy_keys):
            return False

    return True


def score_sample(
    pred_calls: List[Dict[str, Any]],
    pred_hash: str,
    gold_calls: List[Dict[str, Any]],
    gold_hash: str,
    fuzzy_keys: Set[str] = FREE_TEXT_KEYS
) -> Tuple[float, Dict[str, Any]]:
    """Compute score for a single sample according to ZAC 2026 LifeMate formula."""
    details = {
        "hash_match": pred_hash == gold_hash,
        "gold_calls_count": len(gold_calls),
        "pred_calls_count": len(pred_calls),
        "matched_calls": 0,
        "precision": 0.0,
        "recall": 0.0,
        "score": 0.0,
    }

    # Case 1: Out-of-scope sample (gold list is empty)
    if len(gold_calls) == 0:
        if len(pred_calls) == 0:
            details["score"] = 1.0
            return 1.0, details
        else:
            details["score"] = 0.0
            return 0.0, details

    # Case 2: State hash mismatch
    if pred_hash != gold_hash:
        details["score"] = 0.0
        return 0.0, details

    # Case 3: Match calls (order independent, maximum bipartite matching or greedy matching)
    matched_gold_indices = set()
    matched_count = 0

    for pred in pred_calls:
        for j, gold in enumerate(gold_calls):
            if j not in matched_gold_indices and is_call_match(pred, gold, fuzzy_keys):
                matched_gold_indices.add(j)
                matched_count += 1
                break

    M = matched_count
    G = len(gold_calls)
    U = len(pred_calls) - M

    details["matched_calls"] = M

    if M == 0:
        details["score"] = 0.0
        return 0.0, details

    precision = M / (M + U)
    recall = M / G
    f1 = 2 * precision * recall / (precision + recall)

    details["precision"] = precision
    details["recall"] = recall
    details["score"] = f1
    return f1, details


def evaluate_dataset(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    fuzzy_keys: Set[str] = FREE_TEXT_KEYS
) -> Dict[str, Any]:
    """Evaluate a complete submission against ground truth dataset."""
    pred_map = {item["id"]: item for item in predictions}

    total_samples = len(ground_truth)
    total_score = 0.0
    hash_matches = 0
    full_matches = 0

    sample_details = []

    for gt in ground_truth:
        qid = gt["id"]
        gold_calls = gt.get("label", {}).get("tool_calls", [])
        gold_hash = gt.get("gold_hash")

        pred = pred_map.get(qid, {"tool_calls": [], "state_hash": ""})
        pred_calls = pred.get("tool_calls", [])
        pred_hash = pred.get("state_hash", "")

        score, det = score_sample(pred_calls, pred_hash, gold_calls, gold_hash, fuzzy_keys)
        total_score += score
        if det["hash_match"]:
            hash_matches += 1
        if score == 1.0:
            full_matches += 1

        sample_details.append({"id": qid, **det})

    final_score = (100.0 / total_samples) * total_score if total_samples > 0 else 0.0

    return {
        "final_score": round(final_score, 4),
        "total_samples": total_samples,
        "hash_match_rate": round(hash_matches / total_samples * 100, 2) if total_samples > 0 else 0.0,
        "perfect_match_rate": round(full_matches / total_samples * 100, 2) if total_samples > 0 else 0.0,
        "samples": sample_details
    }
