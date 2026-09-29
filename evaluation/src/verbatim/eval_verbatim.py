import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


PUNCT_TOKENS = {",", ".", ":", ";", "!", "?"}


@dataclass
class ExampleResult:
    scenario: str
    stimid: int
    list_len: int
    prompt_len: int
    coarse_nll_full: float
    nll_list1: float
    nll_list2: float
    delta_list2_minus_list1: float
    x_del: float
    x_perc: float
    retrieval_success: int


def get_whitespace_token_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def spans_to_char_markers(text: str, token_spans: Sequence[Tuple[int, int]], word_markers: Sequence[int]) -> np.ndarray:
    if len(token_spans) != len(word_markers):
        raise ValueError(
            f"Token/marker length mismatch: {len(token_spans)} spans vs {len(word_markers)} markers"
        )

    char_markers = np.full(len(text), -1, dtype=np.int32)
    for (start, end), marker in zip(token_spans, word_markers):
        char_markers[start:end] = int(marker)
    return char_markers


def token_markers_from_offsets(offsets: Sequence[Tuple[int, int]], char_markers: np.ndarray) -> List[int]:
    markers: List[int] = []
    for start, end in offsets:
        if end <= start:
            markers.append(-1)
            continue

        span = char_markers[start:end]
        valid = span[span >= 0]
        if valid.size == 0:
            markers.append(-1)
            continue

        values, counts = np.unique(valid, return_counts=True)
        markers.append(int(values[np.argmax(counts)]))

    return markers


def is_punctuation_span(text: str, start: int, end: int) -> bool:
    if end <= start:
        return True
    value = text[start:end].strip()
    return (value in PUNCT_TOKENS) or (value == "")


def compute_example_scores(
    model,
    tokenizer,
    text: str,
    word_markers: Sequence[int],
    device: torch.device,
) -> Tuple[float, float, float]:
    encoded = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")

    if "offset_mapping" not in encoded:
        raise ValueError(
            "Tokenizer does not provide offset_mapping. Use a fast tokenizer checkpoint for this evaluation."
        )

    offsets = encoded["offset_mapping"][0].tolist()
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    token_spans = get_whitespace_token_spans(text)
    char_markers = spans_to_char_markers(text, token_spans, word_markers)
    token_markers = token_markers_from_offsets(offsets, char_markers)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]

    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).detach().cpu().numpy()

    pred_markers = token_markers[1:]
    pred_offsets = offsets[1:]

    full_vals: List[float] = []
    list1_vals: List[float] = []
    list2_vals: List[float] = []

    for idx, (marker, (start, end)) in enumerate(zip(pred_markers, pred_offsets)):
        if is_punctuation_span(text, start, end):
            continue

        full_vals.append(float(nll[idx]))
        if marker == 1:
            list1_vals.append(float(nll[idx]))
        elif marker == 3:
            list2_vals.append(float(nll[idx]))

    if len(full_vals) == 0:
        raise ValueError("No evaluable full-text tokens found.")

    if len(list1_vals) == 0 or len(list2_vals) == 0:
        raise ValueError("No evaluable list tokens found for marker==1 or marker==3.")

    return float(np.mean(full_vals)), float(np.mean(list1_vals)), float(np.mean(list2_vals))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate verbatim retrieval on preprocessed sce1-5 repeat data.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to preprocessed output hf_dataset directory.")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model checkpoint.")
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for future batching support.")
    parser.add_argument("--max-examples", type=int, default=0, help="If >0, evaluate only the first N examples.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Compute device.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where CSV/JSON results are written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    dataset = load_from_disk(str(args.dataset_dir))
    if args.max_examples > 0:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    results: List[ExampleResult] = []

    for ex in dataset:
        coarse_nll_full, nll_list1, nll_list2 = compute_example_scores(
            model=model,
            tokenizer=tokenizer,
            text=ex["text"],
            word_markers=ex["word_markers"],
            device=device,
        )

        x_del = (nll_list2 - nll_list1) / (nll_list2 + nll_list1)
        x_perc = (nll_list2 / nll_list1) * 100.0

        results.append(
            ExampleResult(
                scenario=ex["scenario"],
                stimid=int(ex["stimid"]),
                list_len=int(ex["list_len"]),
                prompt_len=int(ex["prompt_len"]),
                coarse_nll_full=coarse_nll_full,
                nll_list1=nll_list1,
                nll_list2=nll_list2,
                delta_list2_minus_list1=nll_list2 - nll_list1,
                x_del=float(x_del),
                x_perc=float(x_perc),
                retrieval_success=int(nll_list2 < nll_list1),
            )
        )
    df = pd.DataFrame([r.__dict__ for r in results])

    scenario_summary = (
        df.groupby("scenario", as_index=False)
        .agg(
            n_examples=("stimid", "count"),
            mean_coarse_nll_full=("coarse_nll_full", "mean"),
            mean_nll_list1=("nll_list1", "mean"),
            mean_nll_list2=("nll_list2", "mean"),
            mean_x_perc=("x_perc", "mean"),
            retrieval_success_rate=("retrieval_success", "mean"),
        )
        .sort_values("scenario")
    )
    scenario_summary.to_csv(args.output_dir / "verbatim_by_scenario.csv", index=False)

    breakdown: Dict[str, Dict[str, List[float]]] = {}
    for scenario, group in df.groupby("scenario"):
        breakdown[str(scenario)] = {
            "stimid": [int(x) for x in group["stimid"].tolist()],
            "coarse_nll_full": [float(x) for x in group["coarse_nll_full"].tolist()],
            "nll_list1": [float(x) for x in group["nll_list1"].tolist()],
            "nll_list2": [float(x) for x in group["nll_list2"].tolist()],
            "delta_list2_minus_list1": [float(x) for x in group["delta_list2_minus_list1"].tolist()],
            "x_del": [float(x) for x in group["x_del"].tolist()],
            "x_perc": [float(x) for x in group["x_perc"].tolist()],
            "retrieval_success": [int(x) for x in group["retrieval_success"].tolist()],
        }
    summary = {
        "model": args.model,
        "device": str(device),
        "num_examples": int(len(df)),
        "overall": {
            "mean_coarse_nll_full": float(df["coarse_nll_full"].mean()),
            "mean_nll_list1": float(df["nll_list1"].mean()),
            "mean_nll_list2": float(df["nll_list2"].mean()),
            "mean_x_perc": float(df["x_perc"].mean()),
            "retrieval_success_rate": float(df["retrieval_success"].mean()),
        },
        "details": breakdown,
    }

    with (args.output_dir / "verbatim_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
