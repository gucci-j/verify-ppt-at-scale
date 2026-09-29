import argparse
import gzip
import json
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from botocore.exceptions import ClientError

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

TARGET_TOKENS = 100_000_000_000
DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_EOS_TOKEN = "<|end_of_text|>"
DEFAULT_PREDOWNLOAD_MAX_SAMPLES = 100_000
DEFAULT_PARQUET_ROWS_PER_FILE = 10_000
DEFAULT_PARQUET_BATCH_SIZE = 1_000

def build_sources():
    sources = [
        { # Required: 7.56B tokens - Available: 9.68B tokens
            "alias": "fineweb-edu",
            "weight": 0.333, # 0.378 (normalized weight)
            "dataset": "HuggingFaceFW/fineweb-edu",
            "dataset_options": {"name": "sample-10BT", "split": "train"},
            "text_fields": ["text"],
            "use_local": False
        },
        { # Required: 8.4B tokens - Available: 10B+ tokens
            "alias": "dclm",
            "weight": 0.37, # 0.420 (normalized weight)
            "dataset": "mlfoundations/dclm-baseline-1.0",
            "dataset_options": {"split": "train"},
            "text_fields": ["text"],
            "streaming": True,
            "use_local": True,
            "max_samples": 9_000_000,
        },
        { # Required: 454M tokens - Available: 6.5B tokens
            "alias": "pes2o",
            "weight": 0.02, # 0.0227 (normalized weight)
            "dataset": "allenai/dolmino-mix-1124",
            "dataset_options": {"name": "pes2o", "split": "train"},
            "text_fields": ["text"],
            "use_local": True,
            "streaming": True,
            "max_samples": 5_000_000,
        },
        { # Required: 22.8M tokens - Available: 3.6B tokens
            "alias": "wiki",
            "weight": 0.001, # 0.00114 (normalized weight)
            "dataset": "allenai/dolmino-mix-1124",
            "dataset_options": {"name": "wiki", "split": "train"},
            "text_fields": ["text"],
            "use_local": False,
        },
        { # Required: 91M tokens - Available: 1.2B tokens
            "alias": "stackexchange",
            "weight": 0.004, # 0.00455 (normalized weight)
            "dataset": "allenai/dolmino-mix-1124",
            "dataset_options": {"name": "stackexchange", "split": "train"},
            "text_fields": ["text"],
            "use_local": False,
        },
        { # Required: 228M tokens - Available: 443M tokens
            "alias": "infiwebmath",
            "weight": 0.01, # 0.0114 (normalized weight)
            "dataset": "HuggingFaceTB/finemath",
            "dataset_options": {"name": "infiwebmath-3plus", "split": "train"},
            "text_fields": ["text"],
            "streaming": True,
            "use_local": True,
            "max_samples": 300_000,
        },
        { # Required: 386M tokens - Available: 472M tokens
            "alias": "finemath",
            "weight": 0.017, # 0.0193 (normalized weight)
            "dataset": "HuggingFaceTB/finemath",
            "dataset_options": {"name": "finemath-3plus", "split": "train"},
            "text_fields": ["text"],
            "streaming": True,
            "use_local": True,
            "max_samples": 300_000,
        },
        { # Required: 568M tokens - Available: 626M tokens
            "alias": "stack-v2-Python",
            "weight": 0.025, # 0.0284 (normalized weight)
            "dataset": "bigcode/the-stack-v2",
            "dataset_options": {"name": "Python", "split": "train"},
            "source_type": "stack-v2",
            "streaming": True,
            "use_local": True,
            "max_samples": 400_000,
        },
        { # Required: 296M tokens - Available: 451M tokens
            "alias": "stack-v2-Java",
            "weight": 0.013, # 0.0148 (normalized weight)
            "dataset": "bigcode/the-stack-v2",
            "dataset_options": {"name": "Java", "split": "train"},
            "source_type": "stack-v2",
            "streaming": True,
            "use_local": True,
            "max_samples": 400_000,
        },
        { # Required: 296M tokens - Available: 723M tokens
            "alias": "stack-v2-JavaScript",
            "weight": 0.013, # 0.0148 (normalized weight)
            "dataset": "bigcode/the-stack-v2",
            "dataset_options": {"name": "JavaScript", "split": "train"},
            "source_type": "stack-v2",
            "streaming": True,
            "use_local": True,
        },
        { # Required: 159M tokens - Available: 1B+ tokens
            "alias": "stack-v2-C",
            "weight": 0.007, # 0.00795 (normalized weight)
            "dataset": "bigcode/the-stack-v2",
            "dataset_options": {"name": "C", "split": "train"},
            "source_type": "stack-v2",
            "streaming": True,
            "use_local": True,
            "max_samples": 2_000_000,
        },
        { # Required: 410M tokens - Available: 480M tokens
            "alias": "stack-v2-Cpp",
            "weight": 0.018, # 0.0205 (normalized weight)
            "dataset": "bigcode/the-stack-v2",
            "dataset_options": {"name": "C++", "split": "train"},
            "source_type": "stack-v2",
            "streaming": True,
            "use_local": True,
        },
        { # Required: 136M tokens - Available: 244M tokens
            "alias": "stack-v2-C-Sharp",
            "weight": 0.006, # 0.0068 (normalized weight)
            "dataset": "bigcode/the-stack-v2",
            "dataset_options": {"name": "C-Sharp", "split": "train"},
            "source_type": "stack-v2",
            "streaming": True,
            "use_local": True,
            "max_samples": 200_000,
        },
        { # Required: 136M tokens - Available: 164M tokens
            "alias": "stack-v2-PHP",
            "weight": 0.006, # 0.0068 (normalized weight)
            "dataset": "bigcode/the-stack-v2",
            "dataset_options": {"name": "PHP", "split": "train"},
            "source_type": "stack-v2",
            "streaming": True,
            "use_local": True,
        },
        { # Required: 68M tokens - Available: 109M tokens
            "alias": "stack-v2-TypeScript",
            "weight": 0.003, # 0.0034 (normalized weight)
            "dataset": "bigcode/the-stack-v2",
            "dataset_options": {"name": "TypeScript", "split": "train"},
            "source_type": "stack-v2",
            "streaming": True,
            "use_local": True,
        },
        { # Required: 22M tokens - Available: 80M tokens
            "alias": "stack-v2-Swift",
            "weight": 0.001, # 0.0011 (normalized weight)
            "dataset": "bigcode/the-stack-v2",
            "dataset_options": {"name": "Swift", "split": "train"},
            "source_type": "stack-v2",
            "streaming": True,
            "use_local": True,
        },
        { # Required: 90M tokens - Available: 2B+ tokens
            "alias": "stack-v2-SQL",
            "weight": 0.004, # 0.0045 (normalized weight)
            "dataset": "bigcode/the-stack-v2",
            "dataset_options": {"name": "SQL", "split": "train"},
            "source_type": "stack-v2",
            "streaming": True,
            "use_local": True,
        },
        { # Required: 18M tokens - Available: 67M tokens
            "alias": "stack-v2-Ruby",
            "weight": 0.0008, # 0.0009 (normalized weight)
            "dataset": "bigcode/the-stack-v2",
            "dataset_options": {"name": "Ruby", "split": "train"},
            "source_type": "stack-v2",
            "streaming": True,
            "use_local": True,
        },
        { # Required: 114M tokens - Available: 191M tokens
            "alias": "stack-v2-Markdown",
            "weight": 0.005, # 0.0057 (normalized weight)
            "dataset": "bigcode/the-stack-v2",
            "dataset_options": {"name": "Markdown", "split": "train"},
            "source_type": "stack-v2",
            "streaming": True,
            "use_local": True,
            "max_samples": 500_000,
        },
        { # Required: 136M tokens - Available: 2.1B tokens
            "alias": "stack-v2-HTML",
            "weight": 0.006, # 0.0068 (normalized weight)
            "dataset": "bigcode/the-stack-v2",
            "dataset_options": {"name": "HTML", "split": "train"},
            "source_type": "stack-v2",
            "streaming": True,
            "use_local": True,
            "max_samples": 200_000,
        },
        { # Required: 18M tokens - Available: 305M tokens
            "alias": "stack-v2-Rust",
            "weight": 0.0008, # 0.0009 (normalized weight)
            "dataset": "bigcode/the-stack-v2",
            "dataset_options": {"name": "Rust", "split": "train"},
            "source_type": "stack-v2",
            "streaming": True,
            "use_local": True,
            "max_samples": 500_000,
        },
        { #  Required: 11M tokens - Available: 388M tokens
            "alias": "stack-v2-Go",
            "weight": 0.0005, # 0.00057 (normalized weight)
            "dataset": "bigcode/the-stack-v2",
            "dataset_options": {"name": "Go", "split": "train"},
            "source_type": "stack-v2",
            "streaming": True,
            "use_local": True,
            "max_samples": 500_000,
        },
        { # Required: 16M tokens - Available: 120M tokens
            "alias": "stack-v2-Shell",
            "weight": 0.0007, # 0.0008 (normalized weight)
            "dataset": "bigcode/the-stack-v2",
            "dataset_options": {"name": "Shell", "split": "train"},
            "source_type": "stack-v2",
            "streaming": True,
            "use_local": True,
            "max_samples": 500_000,
        },
        { # Required: 136M tokens - Available: 1.3B tokens
            "alias": "kaggle",
            "weight": 0.006, # 0.0068 (normalized weight)
            "dataset": "HuggingFaceTB/issues-kaggle-notebooks",
            "dataset_options": {"name": "kaggle", "split": "train"},
            "text_fields": ["text"],
            "upstream_aliases": ["kaggle", "jupyter-scripts"],
            "use_local": False,
        },
        { # Required: 210M tokens - Available: 3.4B+ tokens
            "alias": "github-issues",
            "weight": 0.0092, # 0.0105 (normalized weight)
            "dataset": "HuggingFaceTB/issues-kaggle-notebooks",
            "dataset_options": {"name": "issues", "split": "train"},
            "text_fields": ["text"],
            "upstream_aliases": ["pull-requests", "github-issues"],
            "use_local": False,
        },
    ]

    # We drop the multilingual FineWeb2 shards and renormalize the surviving weights.
    total_weight = sum(source["weight"] for source in sources)
    for source in sources:
        source["normalized_weight"] = source["weight"] / total_weight
    return sources


@lru_cache(maxsize=1)
def get_software_heritage_client():
    import boto3
    return boto3.client("s3")


def download_stack_v2_content(row, stack_v2_content_root=None, allow_download=True):
    content = row.get("content")
    if isinstance(content, str) and content.strip():
        return content

    blob_id = row.get("blob_id")
    src_encoding = row.get("src_encoding") or "utf-8"
    if not blob_id:
        return None

    if stack_v2_content_root is not None:
        blob_path = Path(stack_v2_content_root) / blob_id[:2] / f"{blob_id}.txt"
        if blob_path.exists():
            return blob_path.read_text(encoding="utf-8")

    if not allow_download:
        raise RuntimeError(
            f"Missing cached stack-v2 blob {blob_id}. Run with --predownload on a node with internet access first."
        )
    try:
        response = get_software_heritage_client().get_object(
            Bucket="softwareheritage", 
            Key=f"content/{blob_id}",
            RequestPayer="requester"
        )
        compressed_bytes = response["Body"].read()
        with gzip.GzipFile(fileobj=BytesIO(compressed_bytes)) as handle:
            raw_bytes = handle.read()
        text = raw_bytes.decode(src_encoding, errors="replace")

        if stack_v2_content_root is not None:
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            blob_path.write_text(text, encoding="utf-8")
        return text
    except ClientError as e:
        if e.response['Error']['Code'] == "NoSuchKey":
            print(f"Warning: Missing blob for row. Skipping.")
            return None  # Return None so you can filter it out later
        else:
            raise e # Re-raise if it's a different error (like credentials/network)


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


def predownload_stack_v2_blobs(source, dataset_reference, cache_root, workers, force=False):
    blob_cache_root = Path(cache_root) / f"datasets--{source['alias']}"
    blob_cache_root.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("parquet", data_files=[str(path) for path in dataset_reference], split="train")
    processed = 0
    reused = 0
    downloaded = 0
    skipped = 0

    def process_row(row):
        nonlocal processed, downloaded, reused, skipped
        processed += 1
        
        content = row.get("content")
        if isinstance(content, str) and content.strip():
            skipped += 1
            row["text"] = content
            return row

        blob_id = row.get("blob_id")
        if not blob_id:
            skipped += 1
            return row

        blob_path = blob_cache_root / blob_id[:2] / f"{blob_id}.txt"
        if blob_path.exists() and not force:
            reused += 1
            row["text"] = blob_path.read_text(encoding="utf-8")
            return row

        text = download_stack_v2_content(row, stack_v2_content_root=blob_cache_root, allow_download=True)
        if text is not None:
            downloaded += 1
            row["text"] = text
        else:
            skipped += 1
            row["text"] = None
        
        return row

    dataset = dataset.map(
        process_row, 
        num_proc=workers,
        batched=False,
    )
    dataset = dataset.filter(lambda x: x['text'] is not None)

    print(
        f"Prepared stack-v2 blob cache for {source['alias']}: processed={processed:,} "
        f"downloaded={downloaded:,} reused={reused:,} skipped={skipped:,}"
    )
    
    return dataset


def predownload_selected_sources(sources, tokenizer, cache_root, workers, force=False):
    AutoTokenizer.from_pretrained(tokenizer)
    for source in sources:
        if source.get("streaming", False):
            dataset = load_dataset(source["dataset"], **source["dataset_options"], streaming=True)
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
        if source.get("source_type") == "stack-v2":
            dataset = predownload_stack_v2_blobs(source, output_files, cache_root, workers, force=force)
            dataset_dir = Path(cache_root) / f"datasets--{source['alias']}-processed"
            dataset.save_to_disk(str(dataset_dir))
            print(f"Saved processed dataset for {source['alias']} to {dataset_dir}")


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
    print(
        json.dumps(
            [
                {k: source[k] for k in ("alias", "weight", "normalized_weight", "dataset")}
                for source in sources
            ],
            indent=2,
        )
    )

    # Optionally predownload datasets, tokenizer, and stack-v2 blobs before tokenization
    # -> This is mandatory for HPCs whose compute nodes lack internet access
    if args.predownload or args.predownload_only:
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
            if source.get("source_type") != "stack-v2":
                # Load the dataset and save it as a `Dataset` object.
                output_files = [
                    Path(args.cache_root) / "datasets--{}".format(source["alias"]) / f"train-*.parquet"
                ]
                dataset = load_dataset("parquet", data_files=[str(path) for path in output_files], split="train")
                if source.get("alias") == "finemath" or source.get("alias") == "infiwebmath":
                    # Only retain the "text" field
                    dataset = dataset.map(
                        lambda x: {"text": x["text"]}, num_proc=args.workers, batched=False, remove_columns=dataset.column_names
                    )
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
