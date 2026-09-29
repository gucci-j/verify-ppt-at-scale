import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Dict, List

from datasets import Dataset


DEFAULT_SCENARIOS = [1, 2, 3, 4, 5]


def parse_markers_file(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            markers = ast.literal_eval(row["markers"])
            records.append(
                {
                    "stimid": int(row["stimid"]),
                    "list_len": int(row["list_len"]),
                    "prompt_len": int(row["prompt_len"]),
                    "word_markers": markers,
                }
            )
    return records


def load_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh]


def build_examples(input_dir: Path, scenarios: List[int]) -> List[Dict]:
    examples: List[Dict] = []

    for scenario in scenarios:
        text_file = input_dir / f"categorized_lists_sce{scenario}_repeat.txt"
        marker_file = input_dir / f"categorized_lists_sce{scenario}_repeat_markers.txt"

        if not text_file.exists() or not marker_file.exists():
            raise FileNotFoundError(
                f"Missing input files for sce{scenario}: {text_file.name}, {marker_file.name}"
            )

        lines = load_lines(text_file)
        marker_rows = parse_markers_file(marker_file)

        if len(lines) != len(marker_rows):
            raise ValueError(
                f"Mismatched row count for sce{scenario}: {len(lines)} lines vs {len(marker_rows)} marker rows"
            )

        for line, marker_row in zip(lines, marker_rows):
            # Markers in rnn_input_files are word-level and should align with whitespace tokenization.
            n_words = len(line.split())
            if n_words != len(marker_row["word_markers"]):
                raise ValueError(
                    f"Token/marker length mismatch for sce{scenario}, stimid={marker_row['stimid']}: "
                    f"{n_words} words vs {len(marker_row['word_markers'])} markers"
                )

            examples.append(
                {
                    "scenario": f"sce{scenario}",
                    "condition": "repeat",
                    "stimid": marker_row["stimid"],
                    "list_len": marker_row["list_len"],
                    "prompt_len": marker_row["prompt_len"],
                    "text": line,
                    "word_markers": marker_row["word_markers"],
                }
            )

    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess verbatim retrieval data from rnn_input_files into a HF dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("external/verbatim-memory-in-NLMs/data/rnn_input_files"),
        help="Directory containing categorized_lists_sce{i}_repeat.txt and *_markers.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the processed dataset will be written.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated scenario ids to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = [int(s.strip()) for s in args.scenarios.split(",") if s.strip()]

    examples = build_examples(args.input_dir, scenarios)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_list(examples)
    dataset.save_to_disk(str(args.output_dir / "hf_dataset"))

    summary = {
        "num_examples": len(examples),
        "scenarios": sorted({ex["scenario"] for ex in examples}),
        "output_hf_dataset": str(args.output_dir / "hf_dataset"),
    }
    with (args.output_dir / "preprocess_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
