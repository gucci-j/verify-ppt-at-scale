"""Tokenize the Set task synthetic data (v2).

- Each generated document has *exactly* `target_total_ints` integers, which
  after SmolLM3 BPE produces *exactly* `2 * target_total_ints + 1` tokens
  (including a trailing space + EOS). With the default 2048 integers / doc
  the output is 4097 tokens per doc -- matching nanotron's
  `DatatroveFolderDataset` stride of `sequence_length + 1` so that
  one training sample == one document with zero leakage.
- Per-sample input length is variable: we sample one integer at a time and
  stop just before the running sum (input_len + unique_len) would overshoot
  `target_total_ints - 1`. Any residual budget is filled by repeating
  already-seen values in the input so that the unique-elements answer is
  unchanged.
- Default vocab_size = 999 so that every value in [1, 999] tokenizes as a
  single SmolLM3 BPE piece (4+ digit numbers split, so vocab >= 1000 is
  unsafe without also reasoning about the token count per integer).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer

import fsspec.compression
import zstandard
def decompress_zstd(fileobj, mode='rb', **kwargs):
    dctx = zstandard.ZstdDecompressor()
    return dctx.stream_reader(fileobj)
fsspec.compression.compr['zstd'] = decompress_zstd
fsspec.compression.compr['zstandard'] = decompress_zstd
print(f"Forced 'zstd' into registry. Current keys: {list(fsspec.compression.compr.keys())}")

DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_EOS_TOKEN = "<|end_of_text|>"
DEFAULT_SET_VOCAB_SIZE = 999
DEFAULT_SET_TOTAL_INTS = 2048   # -> 4097 SmolLM3 tokens per doc (= sequence_length + 1)
DEFAULT_SET_NUM_SENTENCES = 256_000
DEFAULT_SET_SEED = 0


def generate_set_sequence_fixed_total(rng, vocab_size, target_total_ints):
    """Generate one Set sample with exactly `target_total_ints` integers.

    Structure: input + [separator] + unique
    - input integers are drawn uniformly from [1, vocab_size)
    - separator == vocab_size
    - unique is the first-occurrence-order list of distinct input values
    - Phase 2 padding (repeating already-seen values in the input) is used to
      reach the exact target length without changing the unique list.
    """
    separator = vocab_size
    input_seq = []
    unique = []
    seen = set()
    target_sum = target_total_ints - 1   # input_len + unique_len

    # Phase 1: natural generation -- stop just before adding would overflow
    while True:
        x = int(rng.randint(1, vocab_size))
        is_new = x not in seen
        add = 2 if is_new else 1
        if len(input_seq) + len(unique) + add > target_sum:
            break
        input_seq.append(x)
        if is_new:
            seen.add(x)
            unique.append(x)

    # Phase 2: pad input with repeats of already-seen values
    remaining = target_sum - (len(input_seq) + len(unique))
    if remaining > 0:
        if not seen:
            raise RuntimeError("target_total_ints too small to seed any unique value")
        seen_arr = np.array(sorted(seen), dtype=np.int64)
        picks = rng.randint(0, len(seen_arr), size=remaining)
        input_seq.extend(int(seen_arr[i]) for i in picks)

    result = input_seq + [separator] + unique
    assert len(result) == target_total_ints, (
        f"length invariant violated: got {len(result)}, expected {target_total_ints}"
    )
    return result


def generate_set_txt_file(
    file_dir,
    vocab_size=DEFAULT_SET_VOCAB_SIZE,
    n=DEFAULT_SET_NUM_SENTENCES,
    target_total_ints=DEFAULT_SET_TOTAL_INTS,
    seed=DEFAULT_SET_SEED,
    force=False,
):
    """Write `n` set-v2 sentences (each exactly `target_total_ints` ints) and return the file path."""
    file_dir = Path(file_dir)
    file_dir.mkdir(parents=True, exist_ok=True)
    output_path = file_dir / f"set_v2_sequences_{vocab_size}_{target_total_ints}.txt"

    if output_path.exists() and not force:
        file_size = output_path.stat().st_size
        if file_size > 0:
            print(f"Found existing non-empty set_v2 file at {output_path} ({file_size:,} bytes), skipping generation.")
            return output_path
        else:
            print(f"Found existing set_v2 file at {output_path}, but it is EMPTY (0 bytes). Regenerating...")

    rng = np.random.RandomState(seed)
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_output_path.open("w", encoding="utf-8") as f:
        for _ in tqdm(range(n), desc="Generating set_v2"):
            result = generate_set_sequence_fixed_total(rng, vocab_size, target_total_ints)
            seq_str = " ".join(str(x) for x in result)
            # Trailing space adds one extra SmolLM3 token so the doc becomes
            # 2N+1 tokens (with EOS). This matches nanotron's DatatroveFolderDataset
            # stride of (sequence_length + 1), giving 1 sample == 1 doc alignment.
            f.write(f"{seq_str} \n")

    temp_output_path.replace(output_path)
    return output_path


def build_sources(set_text_path):
    if set_text_path is None:
        raise ValueError("set_text_path must be provided")

    return [
        {
            "alias": "set_v2",
            "dataset": "text",
            "dataset_options": {
                "data_files": str(set_text_path),
                "split": "train",
            },
            "text_fields": ["text"],
            "use_local": False,
        }
    ]


def predownload_selected_sources(sources, tokenizer, cache_root, workers, force=False):
    AutoTokenizer.from_pretrained(tokenizer)
    for source in sources:
        dataset = load_dataset(source["dataset"], **source["dataset_options"], cache_dir=cache_root)
        cache_file = dataset.cache_files[0]["filename"] if dataset.cache_files else "<no cache file>"
        print(f"Dataset ready for {source['alias']}: found {len(dataset)} rows at {cache_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a set-task v2 dataset in DataTrove tokenized-bytes format (each doc fits exactly in sequence_length).")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--logging-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--eos-token", default=DEFAULT_EOS_TOKEN)
    parser.add_argument("--tasks", type=int, default=128)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set-vocab-size", type=int, default=DEFAULT_SET_VOCAB_SIZE,
                        help="Must be <= 999 to keep every integer at <=3 digits (single SmolLM3 BPE piece).")
    parser.add_argument("--set-num-sentences", type=int, default=DEFAULT_SET_NUM_SENTENCES)
    parser.add_argument("--set-total-ints", type=int, default=DEFAULT_SET_TOTAL_INTS,
                        help="Total integers per doc. Default 2048 -> exactly 4097 SmolLM3 tokens per doc (= sequence_length + 1, so 1 sample == 1 doc).")
    parser.add_argument("--set-seed", type=int, default=DEFAULT_SET_SEED)
    parser.add_argument(
        "--set-file-dir",
        default=None,
        help="Directory where the generated set_v2 .txt file is written. Defaults to <cache-root>/set_v2.",
    )
    parser.add_argument(
        "--predownload",
        action="store_true",
        help="Download the selected dataset and tokenizer into --cache-root before tokenization.",
    )
    parser.add_argument(
        "--predownload-only",
        action="store_true",
        help="Exit after downloading the selected dataset and tokenizer into --cache-root.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require datasets, tokenizer, and stack-v2 blobs to resolve from --cache-root without network access.",
    )
    parser.add_argument("--force", action="store_true", help="Retokenize even if output metadata already exists.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.set_vocab_size > 999:
        raise SystemExit(
            f"--set-vocab-size={args.set_vocab_size} > 999: SmolLM3 splits 4+ digit numbers, "
            f"so per-doc token count would no longer be 2 * total_ints. Use <= 999."
        )

    set_dir = Path(args.set_file_dir) if args.set_file_dir else (Path(args.cache_root) / "set_v2")
    set_text_path = generate_set_txt_file(
        file_dir=set_dir,
        vocab_size=args.set_vocab_size,
        n=args.set_num_sentences,
        target_total_ints=args.set_total_ints,
        seed=args.set_seed,
        force=args.force,
    )

    sources = build_sources(set_text_path=set_text_path)

    if args.predownload or args.predownload_only:
        predownload_selected_sources(sources, args.tokenizer, args.cache_root, args.workers, force=args.force)
    if args.predownload_only:
        print("Finished predownloading selected sources and tokenizer.")
        return

    output_root = Path(args.output_root)
    logging_root = Path(args.logging_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logging_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(sources, indent=2), encoding="utf-8")

    for source in sources:
        output_folder = output_root / source["alias"]
        log_folder = logging_root / source["alias"]

        if source.get("use_local", False) is False:
            dataset_reference = source["dataset"]
            dataset_options = dict(source["dataset_options"])
            dataset_options.setdefault("cache_dir", os.environ.get("HF_DATASETS_CACHE", args.cache_root))

            pipeline = [
                HuggingFaceDatasetReader(
                    dataset=dataset_reference,
                    dataset_options=dataset_options,
                ),
                DocumentTokenizer(
                    output_folder=str(output_folder),
                    save_filename=source["alias"],
                    tokenizer_name_or_path=args.tokenizer,
                    eos_token=args.eos_token,
                    save_final_metadata=True,
                    batch_size=args.batch_size,
                ),
            ]
        else:
            output_files = [
                Path(args.cache_root) / "datasets--{}".format(source["alias"]) / f"train-*.parquet"
            ]
            dataset = load_dataset("parquet", data_files=[str(path) for path in output_files], split="train")
            dataset.save_to_disk(Path(args.cache_root) / "datasets--{}-processed".format(source["alias"]))
            dataset_reference = Path(args.cache_root) / "datasets--{}-processed".format(source["alias"])
            pipeline = [
                HuggingFaceDatasetReader(
                    dataset=dataset_reference,
                    load_from_disk=True,
                ),
                DocumentTokenizer(
                    output_folder=str(output_folder),
                    save_filename=source["alias"],
                    tokenizer_name_or_path=args.tokenizer,
                    eos_token=args.eos_token,
                    save_final_metadata=True,
                    batch_size=args.batch_size,
                ),
            ]

        print(f"Tokenizing {source['alias']} from {source['dataset']} to {output_folder}.")
        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=args.tasks,
            workers=args.workers,
            logging_dir=str(log_folder),
        )
        executor.run()


if __name__ == "__main__":
    main()
