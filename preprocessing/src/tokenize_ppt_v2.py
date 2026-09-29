"""Tokenize the k-Shuffle Dyck task synthetic data (v2).

- Each generated document has *exactly* `target_length` integers, which after
  SmolLM3 BPE produces *exactly* `2 * target_length + 1` tokens (including a
  trailing space + EOS). With the default 2048 integers / doc the output is
  4097 tokens per doc -- matching nanotron's `DatatroveFolderDataset` stride
  of `sequence_length + 1` so that one training sample == one document with
  zero leakage.
- The generation algorithm is unchanged from v1 (same generate_shuff_dyck).
  The only change is the default target_length: 4096 -> 2048.
- Default k is 64, identical to v1 and to the precedent pre-pretraining
  paper. With k=64 the value range is [0, 128), all <= 3 digits, so every
  integer tokenizes as a single SmolLM3 BPE piece (1 token bare, 2 tokens
  with leading space). Going beyond k=500 would break the 2-tokens-per-int
  invariant (4-digit closing brackets >= 1000 get split by SmolLM3 BPE).
"""

import argparse
import json
import os
import random
from pathlib import Path

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
DEFAULT_SHUFF_DYCK_K = 64
DEFAULT_SHUFF_DYCK_P_OPEN = 0.50
DEFAULT_SHUFF_DYCK_NUM_SENTENCES = 256_000
DEFAULT_SHUFF_DYCK_SEQ_LENGTH = 2048   # -> 4097 SmolLM3 tokens per doc (= sequence_length + 1)


def generate_shuff_dyck(k, max_length=2048, p_open=0.5, min_depth=1, max_depth=8):
    """Generate a k-shuffle Dyck sequence of length `max_length`.

    Identical to the v1 implementation in tokenize_ppt.py. Opening brackets are
    in [0, k); closing brackets are in [k, 2k) where closing of type i is i+k.
    """
    sequence = []
    counts = [0] * k

    if min_depth < 1:
        raise ValueError("min_depth must be at least 1")
    if k < 1:
        raise ValueError("k must be at least 1")

    for _ in range(min_depth):
        bracket = random.randint(0, k - 1)
        sequence.append(bracket)
        counts[bracket] += 1

    while len(sequence) < max_length:
        depth = sum(counts)

        if depth == 0:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
            continue

        if depth >= max_depth:
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1
            continue

        if random.random() < p_open and depth < max_depth:
            bracket = random.randint(0, k - 1)
            sequence.append(bracket)
            counts[bracket] += 1
        else:
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = random.choice(open_brackets)
            sequence.append(bracket + k)
            counts[bracket] -= 1

    return sequence


def generate_shuff_dyck_txt_file(
    file_dir,
    num_symbols=DEFAULT_SHUFF_DYCK_K,
    n=DEFAULT_SHUFF_DYCK_NUM_SENTENCES,
    target_length=DEFAULT_SHUFF_DYCK_SEQ_LENGTH,
    p=DEFAULT_SHUFF_DYCK_P_OPEN,
    force=False,
):
    """Write `n` k-shuffle Dyck sentences (each exactly target_length ints) to disk."""
    file_dir = Path(file_dir)
    file_dir.mkdir(parents=True, exist_ok=True)
    output_path = file_dir / f"shuff_dyck_v2_sequences_{num_symbols}_{p}_{target_length}.txt"

    if output_path.exists() and not force:
        file_size = output_path.stat().st_size
        if file_size > 0:
            print(f"Found existing non-empty shuff_dyck_v2 file at {output_path} ({file_size:,} bytes), skipping generation.")
            return output_path
        else:
            print(f"Found existing shuff_dyck_v2 file at {output_path}, but it is EMPTY (0 bytes). Regenerating...")

    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_output_path.open("w", encoding="utf-8") as f:
        for _ in tqdm(range(n), desc="Generating shuff_dyck_v2"):
            result = generate_shuff_dyck(num_symbols, target_length, p)
            assert len(result) == target_length, (
                f"length invariant violated: got {len(result)}, expected {target_length}"
            )
            dyck_str = " ".join(str(x) for x in result[:target_length])
            # Trailing space adds one extra SmolLM3 token so the doc becomes
            # 2N+1 tokens (with EOS). This matches nanotron's DatatroveFolderDataset
            # stride of (sequence_length + 1), giving 1 sample == 1 doc alignment.
            f.write(f"{dyck_str} \n")

    temp_output_path.replace(output_path)
    return output_path


def build_sources(shuff_dyck_text_path):
    if shuff_dyck_text_path is None:
        raise ValueError("shuff_dyck_text_path must be provided")

    return [
        {
            "alias": "shuff_dyck_v2",
            "dataset": "text",
            "dataset_options": {
                "data_files": str(shuff_dyck_text_path),
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
    parser = argparse.ArgumentParser(description="Build a k-Shuffle Dyck v2 dataset in DataTrove tokenized-bytes format (each doc fits exactly in sequence_length).")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--logging-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--eos-token", default=DEFAULT_EOS_TOKEN)
    parser.add_argument("--tasks", type=int, default=128)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shuff-dyck-k", type=int, default=DEFAULT_SHUFF_DYCK_K,
                        help="Must be <= 500 so that 2k-1 <= 999 stays within 3 digits (single SmolLM3 BPE piece).")
    parser.add_argument("--shuff-dyck-num-sentences", type=int, default=DEFAULT_SHUFF_DYCK_NUM_SENTENCES)
    parser.add_argument("--shuff-dyck-seq-length", type=int, default=DEFAULT_SHUFF_DYCK_SEQ_LENGTH,
                        help="Integers per doc. Default 2048 -> exactly 4097 SmolLM3 tokens per doc (= sequence_length + 1, so 1 sample == 1 doc).")
    parser.add_argument("--shuff-dyck-p-open", type=float, default=DEFAULT_SHUFF_DYCK_P_OPEN)
    parser.add_argument(
        "--shuff-dyck-file-dir",
        default=None,
        help="Directory where the generated Dyck .txt file is written. Defaults to <cache-root>/shuff_dyck_v2.",
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

    if args.shuff_dyck_k > 500:
        raise SystemExit(
            f"--shuff-dyck-k={args.shuff_dyck_k} > 500: closing bracket id 2k-1 would exceed 999, "
            f"and SmolLM3 splits 4+ digit numbers, breaking the 2-tokens-per-int invariant. Use <= 500."
        )

    shuff_dyck_dir = Path(args.shuff_dyck_file_dir) if args.shuff_dyck_file_dir else (Path(args.cache_root) / "shuff_dyck_v2")
    shuff_dyck_text_path = generate_shuff_dyck_txt_file(
        file_dir=shuff_dyck_dir,
        num_symbols=args.shuff_dyck_k,
        n=args.shuff_dyck_num_sentences,
        target_length=args.shuff_dyck_seq_length,
        p=args.shuff_dyck_p_open,
        force=args.force,
    )

    sources = build_sources(shuff_dyck_text_path=shuff_dyck_text_path)

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
