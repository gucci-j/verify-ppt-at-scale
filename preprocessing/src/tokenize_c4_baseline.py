import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader, ParquetReader
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer

import fsspec.compression
import zstandard
def decompress_zstd(fileobj, mode='rb', **kwargs):
    dctx = zstandard.ZstdDecompressor()
    return dctx.stream_reader(fileobj)
fsspec.compression.compr['zstd'] = decompress_zstd
fsspec.compression.compr['zstandard'] = decompress_zstd
print(f"Forced 'zstd' into registry. Current keys: {list(fsspec.compression.compr.keys())}")

TARGET_TOKENS = 100_000_000_000
DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_EOS_TOKEN = "<|end_of_text|>"
DEFAULT_PREDOWNLOAD_MAX_SAMPLES = 100_000
DEFAULT_PARQUET_ROWS_PER_FILE = 10_000
DEFAULT_PARQUET_BATCH_SIZE = 1_000

def build_sources():
    # Pre-pretraining baseline: a non-overlapping slice of C4 to match the
    # synthetic Set/Dyck pre-pretraining token budget (~1.05B tokens).
    #
    # We use STREAMING + .skip() + .take() instead of percentage slicing
    # (train[25%:26%]) because HF's percentage slicing on JSON datasets like
    # C4 forces a full download of all 1024 shards (~400GB) to count rows.
    # Streaming only fetches what we actually iterate over (a few GB).
    #
    # tokenize_c4.py uses train[:25%]; we skip the first 25% of rows
    # (91,217,223 rows = 25% of C4-en's 364,868,892 train rows) and take
    # enough docs for ~1.05B tokens (~2.5M docs at avg ~470 tok/doc).
    sources = [
        {
            "alias": "c4_baseline",
            "dataset": "allenai/c4",
            "dataset_options": {"name": "en", "split": "train"},
            "text_fields": ["text"],
            "streaming": True,
            "use_local": True,            # tokenize from the parquet we stream-write
            "stream_skip": 91_217_223,    # = 25% of 364,868,892 (C4-en train rows)
            "max_samples": 2_500_000,     # ~1.18B tokens, 1.12x margin over 1.05B
        },
    ]
    return sources


def _normalize_rows_for_schema(rows, schema):
    return [{field.name: row.get(field.name) for field in schema} for row in rows]


def sample_streaming_dataset_to_parquet(
    dataset,
    output_dir,
    num_samples=DEFAULT_PREDOWNLOAD_MAX_SAMPLES,
    file_prefix="train",
    rows_per_file=DEFAULT_PARQUET_ROWS_PER_FILE,
    batch_size=DEFAULT_PARQUET_BATCH_SIZE,
):
    # Ensure output directory exists and is empty of existing files with the same prefix
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_files = list(output_dir.glob(f"{file_prefix}-*.parquet"))
    if existing_files:
        print(f"Found existing sampled files in {output_dir} with prefix '{file_prefix}', skipping sampling.")
        total_rows = 0
        for file in existing_files:
            table = pq.read_table(file)
            total_rows += table.num_rows
        return existing_files, total_rows

    # Initialize state for writing Parquet files in batches
    file_index = 0
    rows_in_file = 0
    total_written = 0
    buffered_rows = []
    current_file_path = output_dir / f"{file_prefix}-{file_index:05d}.parquet"
    parquet_writer = None
    schema = None
    output_files = []

    # Helper function to flush buffered rows to a Parquet file
    def flush_buffer(writer, current_schema):
        nonlocal buffered_rows
        if not buffered_rows:
            return writer, current_schema

        if current_schema is None:
            table = pa.Table.from_pylist(buffered_rows)
            current_schema = table.schema
        else:
            table = pa.Table.from_pylist(_normalize_rows_for_schema(buffered_rows, current_schema), schema=current_schema)

        if writer is None:
            writer = pq.ParquetWriter(str(current_file_path), current_schema)
            output_files.append(current_file_path)

        writer.write_table(table)
        buffered_rows = []
        return writer, current_schema

    # Iterate through the streaming dataset, buffer rows, and write to Parquet files according to the specified batch size and rows per file limits
    try:
        for row in tqdm(dataset, desc="Sampling dataset", total=num_samples):
            buffered_rows.append(row)
            rows_in_file += 1
            total_written += 1

            if len(buffered_rows) >= batch_size or rows_in_file >= rows_per_file or total_written >= num_samples:
                parquet_writer, schema = flush_buffer(parquet_writer, schema)

            if rows_in_file >= rows_per_file:
                if parquet_writer is not None:
                    parquet_writer.close()
                parquet_writer = None
                schema = None
                file_index += 1
                rows_in_file = 0
                current_file_path = output_dir / f"{file_prefix}-{file_index:05d}.parquet"

            if total_written >= num_samples:
                break

        parquet_writer, schema = flush_buffer(parquet_writer, schema)
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    return output_files, total_written


def predownload_selected_sources(sources, tokenizer, cache_root, workers, force=False):
    AutoTokenizer.from_pretrained(tokenizer)
    for source in sources:
        if source.get("streaming", False):
            dataset = load_dataset(source["dataset"], **source["dataset_options"], streaming=True)
            stream_skip = source.get("stream_skip", 0)
            if stream_skip:
                print(f"  [{source['alias']}] skipping first {stream_skip:,} streamed rows...")
                dataset = dataset.skip(stream_skip)
            output_files, saved_rows = sample_streaming_dataset_to_parquet(
                dataset=dataset,
                output_dir=Path(cache_root) / "datasets--{}".format(source["alias"]),
                num_samples=source.get("max_samples", DEFAULT_PREDOWNLOAD_MAX_SAMPLES),
            )
            print(
                f"Dataset sample ready for {source['alias']}: saved {saved_rows:,} rows to {len(output_files)} Parquet files at {output_files[0].parent}"
            )
        else:
            dataset = load_dataset(source["dataset"], **source["dataset_options"], cache_dir=cache_root)
            print(f"Dataset ready for {source['alias']}: found {len(dataset)} rows at {dataset.cache_files[0]['filename']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a 100B-token SmolLM3 stage-1 dataset in DataTrove tokenized-bytes format.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--logging-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--eos-token", default=DEFAULT_EOS_TOKEN)
    parser.add_argument("--tasks", type=int, default=128)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--predownload-max-samples",
        type=int,
        default=DEFAULT_PREDOWNLOAD_MAX_SAMPLES,
        help="Maximum total number of streaming samples to download across sources.",
    )
    parser.add_argument(
        "--predownload",
        action="store_true",
        help="Download the selected datasets, tokenizer, and stack-v2 blobs into --cache-root before tokenization.",
    )
    parser.add_argument(
        "--predownload-only",
        action="store_true",
        help="Exit after downloading the selected datasets, tokenizer, and stack-v2 blobs into --cache-root.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require datasets, tokenizer, and stack-v2 blobs to resolve from --cache-root without network access.",
    )
    parser.add_argument("--force", action="store_true", help="Retokenize even if output metadata already exists.")
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Build the list of sources and allocate token budgets
    sources = build_sources()

    # Optionally predownload datasets, tokenizer, and stack-v2 blobs before tokenization
    # -> This is mandatory for HPCs whose compute nodes lack internet access
    # Streaming sources are ALWAYS materialized here (parquet on disk) since the
    # main loop's use_local=True branch expects the parquet to exist.
    needs_predownload = args.predownload or args.predownload_only or any(
        s.get("streaming", False) for s in sources
    )
    if needs_predownload:
        predownload_selected_sources(sources, args.tokenizer, args.cache_root, args.workers, force=args.force)
    if args.predownload_only:
        print("Finished predownloading selected sources and tokenizer.")
        return

    # Set up output and logging directories, and write the manifest file
    output_root = Path(args.output_root)
    logging_root = Path(args.logging_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logging_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(sources, indent=2), encoding="utf-8")
    
    # Initialize the tokenizer references
    for source in sources:
        output_folder = output_root / source["alias"]
        log_folder = logging_root / source["alias"]
        
        if source.get("use_local", False) is False:
            dataset_reference = source["dataset"]
            dataset_options = dict(source["dataset_options"])
            dataset_options.setdefault("cache_dir", os.environ["HF_DATASETS_CACHE"])

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
            # Read the parquet shards directly via DataTrove's ParquetReader.
            # Previously we did load_dataset("parquet") -> save_to_disk which
            # rewrote the entire dataset into HF Arrow format, doubling disk
            # usage (and hitting quota on small workspaces).
            parquet_dir = Path(args.cache_root) / "datasets--{}".format(source["alias"])
            text_key = (source.get("text_fields") or ["text"])[0]
            pipeline = [
                ParquetReader(
                    data_folder=str(parquet_dir),
                    glob_pattern="train-*.parquet",
                    recursive=False,
                    text_key=text_key,
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

        print(
            f"Tokenizing {source['alias']} from {source['dataset']} to {output_folder}."
        )
        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=args.tasks,
            workers=args.workers,
            logging_dir=str(log_folder),
        )
        executor.run()
        

if __name__ == "__main__":
    main()
