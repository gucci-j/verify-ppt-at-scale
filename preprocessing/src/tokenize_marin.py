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
DEFAULT_PREDOWNLOAD_MAX_SAMPLES = 100_000
DEFAULT_PARQUET_ROWS_PER_FILE = 500_000
DEFAULT_PARQUET_BATCH_SIZE = 25_000

def build_sources(dclm_alias="dclm_marin", dclm_max_samples=18_000_000):
    sources = [
        { # 6.1% total
            "alias": "starcoderdata_{}".format(data_dir),
            "dataset": "bigcode/starcoderdata",
            "dataset_options": {"split": "train", "data_dir": data_dir},
            "text_fields": ["content"],
            "use_local": False,
        }
        for data_dir in [
            "ada", # 0.034
            "agda", # 0.009
            "alloy", # 0.001
            "antlr", # 0.007
            "applescript", # 0.001
            "assembly", # 0.203
            "augeas", # 0
            "awk", # 0.003
            "batchfile", # 0.03
            "bluespec", # 0.004
            "c-sharp", # 5.823
            "c", # 7.027
            "clojure", # 0.06
            "cmake", # 0.059
            "coffeescript", # 0.083
            "common-lisp", # 0.183
            "cpp", # 6.379
            "css", # 0.391
            "cuda", # 0.073
            "dart", # 0.477
            "dockerfile", # 0.055
            "elixir", # 0.093
            "elm", # 0.039
            "emacs-lisp", # 0.053
            "erlang", # 0.091
            "f-sharp", # 0.08
            "fortran", # 0.232
            "git-commits-cleaned", # 4.172
            "github-issues-filtered-structured", # 7.093
            "glsl", # 0.052
            "go", # 3.101
            "groovy", # 0.119
            "haskell", # 0.291
            "html", # 3.828
            "idris", # 0.004
            "isabelle", # 0.01
            "java-server-pages", # 0.128
            "java", # 11.336
            "javascript", # 8.437
            "json", # 0.13
            "julia", # 0.171
            "jupyter-scripts-dedup-filtered", # 0.928
            "jupyter-structured-clean-dedup", # 0.782
            "kotlin", # 0.741
            "lean", # 0.012
            "literate-agda", # 0.001
            "literate-coffeescript", # 0.001
            "literate-haskell", # 0.007
            "lua", # 0.374
            "makefile", # 0.171
            "maple", # 0.001
            "markdown", # 9.77
            "mathematica", # 0.163
            "matlab", # 0
            "ocaml", # 0.134
            "pascal", # 0.219
            "perl", # 0.291
            "php", # 7.939
            "powershell", # 0.146
            "prolog", # 0.001
            "protocol-buffer", # 0.04
            "python", # 7.875
            "r", # 0.039
            "racket", # 0.004
            "restructuredtext", # 0.433
            "rmarkdown", # 0.008
            "ruby", # 0.888
            "rust", # 1.188
            "sas", # 0.016
            "scala", # 0.612
            "scheme", # 0.026
            "shell", # 0.403
            "smalltalk", # 0.076
            "solidity", # 0.111
            "sparql", # 0.005
            "sql", # 1.446
            "stan", # 0.001
            "standard-ml", # 0.025
            "stata", # 0.043
            "systemverilog", # 0.051
            "tcl", # 0.046
            "tcsh", # 0.003
            "tex", # 0.678
            "thrift", # 0.001
            "typescript", # 3.458
            "verilog", # 0
            "vhdl", # 0.123
            "visual-basic", # 0.185
            "xslt", # 0.007
            "yacc", # 0.014
            "yaml", # 0.13
            "zig", # 0.023
        ]
    ]
    sources += [
        { # 92.6%
          # alias + sample count come from CLI (--dclm-alias / --dclm-max-samples) so a
          # larger DCLM pool can be built under a NEW alias without touching this one.
          # Default 18M samples (~21.9B tokens, ~1216 tokens/sample). For a 100B-token
          # budget you need 0.926*100B = 92.6B (>=76.2M samples); 85M (~103B) adds margin.
            "alias": dclm_alias,
            "dataset": "mlfoundations/dclm-baseline-1.0",
            "dataset_options": {"split": "train"},
            "text_fields": ["text"],
            "streaming": True,
            "use_local": True,
            "max_samples": dclm_max_samples,
        },
        { # 1.3%
          # -> 0.013 *  29B / (29B + 15B + 11B) = 0.00685
          # ~28.1B tokens available
            "alias": "proof-pile-arxiv",
            "dataset": "aklein4/proof-pile-2-fixed",
            "dataset_options": {"name": "arxiv", "split": "train"},
            "text_fields": ["text"],
            "use_local": False,
        },
        { # -> 0.013 * 15B / (29B + 15B + 11B) = 0.00355
          # 12.5B tokens available
            "alias": "proof-pile-open-web-math",
            "dataset": "aklein4/proof-pile-2-fixed",
            "dataset_options": {"name": "open-web-math", "split": "train"},
            "text_fields": ["text"],
            "use_local": False,
        },
        { # -> 0.013 * 11B / (29B + 15B + 11B) = 0.00260
          # 8.3B tokens available
            "alias": "proof-pile-algebraic-stack",
            "dataset": "aklein4/proof-pile-2-fixed",
            "dataset_options": {"name": "algebraic-stack", "split": "train"},
            "text_fields": ["text"],
            "use_local": False,
        },
    ]
    return sources


def _normalize_rows_for_schema(rows, schema):
    return [{field.name: row.get(field.name) for field in schema} for row in rows]


def _make_type_nullable(dtype):
    if pa.types.is_struct(dtype):
        return pa.struct([
            pa.field(field.name, _make_type_nullable(field.type), nullable=True, metadata=field.metadata)
            for field in dtype
        ])
    if pa.types.is_list(dtype):
        value_field = dtype.value_field
        return pa.list_(pa.field(value_field.name, _make_type_nullable(value_field.type), nullable=True, metadata=value_field.metadata))
    if pa.types.is_large_list(dtype):
        value_field = dtype.value_field
        return pa.large_list(pa.field(value_field.name, _make_type_nullable(value_field.type), nullable=True, metadata=value_field.metadata))
    if pa.types.is_fixed_size_list(dtype):
        value_field = dtype.value_field
        return pa.list_(
            pa.field(value_field.name, _make_type_nullable(value_field.type), nullable=True, metadata=value_field.metadata),
            dtype.list_size,
        )
    if pa.types.is_map(dtype):
        return pa.map_(
            _make_type_nullable(dtype.key_type),
            _make_type_nullable(dtype.item_type),
            keys_sorted=dtype.keys_sorted,
        )
    return dtype


def _make_schema_nullable(schema):
    return pa.schema(
        [
            pa.field(field.name, _make_type_nullable(field.type), nullable=True, metadata=field.metadata)
            for field in schema
        ],
        metadata=schema.metadata,
    )


def sample_streaming_dataset_to_parquet(
    dataset,
    output_dir,
    num_samples=DEFAULT_PREDOWNLOAD_MAX_SAMPLES,
    file_prefix="train",
    rows_per_file=DEFAULT_PARQUET_ROWS_PER_FILE,
    batch_size=DEFAULT_PARQUET_BATCH_SIZE,
    selected_fields=None,
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
    selected_fields = selected_fields or ["text"]

    # Helper function to flush buffered rows to a Parquet file
    def flush_buffer(writer, current_schema):
        nonlocal buffered_rows
        if not buffered_rows:
            return writer, current_schema

        if current_schema is None:
            table = pa.Table.from_pylist(buffered_rows)
            current_schema = _make_schema_nullable(table.schema)
            table = table.cast(current_schema, safe=False)
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
            buffered_rows.append({field: row.get(field) for field in selected_fields})
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
        alias = source["alias"]
        processed_dir = Path(cache_root) / f"datasets--{alias}-processed"

        if not force and processed_dir.exists():
            print(f"Dataset {alias} already processed at {processed_dir}, skipping.")
            continue

        if source.get("streaming", False):
            dataset = load_dataset(source["dataset"], **source["dataset_options"], streaming=True)
            output_files, saved_rows = sample_streaming_dataset_to_parquet(
                dataset=dataset,
                output_dir=Path(cache_root) / f"datasets--{alias}",
                num_samples=source.get("max_samples", DEFAULT_PREDOWNLOAD_MAX_SAMPLES),
                selected_fields=source.get("text_fields", ["text"]),
            )
            print(
                f"Dataset sample ready for {alias}: saved {saved_rows:,} rows to {len(output_files)} Parquet files at {output_files[0].parent}"
            )
            if source.get("use_local", False):
                # Materialize the sampled Parquet into a `Dataset` saved on disk *here*,
                # during the single (non-arrayed) predownload job. The tokenize step then
                # only `load_from_disk` (read-only), so concurrent Slurm array tasks never
                # race on `save_to_disk` into the same `-processed` directory.
                dataset = load_dataset(
                    "parquet",
                    data_files=[str(path) for path in output_files],
                    split="train",
                )
                dataset.save_to_disk(str(processed_dir), num_proc=workers)
                print(f"Saved processed dataset for {alias} to {processed_dir}")
        else:
            dataset = load_dataset(source["dataset"], **source["dataset_options"], cache_dir=cache_root)
            if alias.startswith("starcoderdata_"):
                # Save to disk to avoid a loading error later
                dataset.save_to_disk(str(processed_dir), num_proc=workers)
            print(f"Dataset ready for {alias}: found {len(dataset)} rows at {dataset.cache_files[0]['filename']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a 100B-token SmolLM3 stage-1 dataset in DataTrove tokenized-bytes format.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--logging-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--eos-token", default=DEFAULT_EOS_TOKEN)
    parser.add_argument("--tasks", type=int, default=128)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--local-tasks",
        type=int,
        default=-1,
        help="How many of the total --tasks this process should run (-1 = all). Combine with --local-rank-offset to split a Slurm job array across machines.",
    )
    parser.add_argument(
        "--local-rank-offset",
        type=int,
        default=0,
        help="Rank of the first task to run in this process; tasks [offset, offset + local_tasks) are handled here.",
    )
    parser.add_argument(
        "--dclm-alias",
        default="dclm_marin",
        help="Output alias (folder name) for the DCLM source. Use a NEW alias to build a larger DCLM pool without overwriting the existing one.",
    )
    parser.add_argument(
        "--dclm-max-samples",
        type=int,
        default=18_000_000,
        help="Number of DCLM streaming samples to tokenize (~1216 tokens/sample). 18M~=21.9B tokens; 85M~=103B tokens.",
    )
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
    sources = build_sources(dclm_alias=args.dclm_alias, dclm_max_samples=args.dclm_max_samples)

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
        
        if source["alias"].startswith("starcoderdata_"):
            # For StarCoderData, we load the dataset from the local cache (which was saved during predownload) to avoid a loading error.
            dataset_reference = Path(args.cache_root) / "datasets--{}-processed".format(source["alias"])
            pipeline = [
                HuggingFaceDatasetReader(
                    dataset=dataset_reference,
                    text_key=source["text_fields"][0],
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
        elif source.get("use_local", False) is False:
            dataset_reference = source["dataset"]
            dataset_options = dict(source["dataset_options"])
            dataset_options.setdefault("cache_dir", os.environ["HF_DATASETS_CACHE"])
            print(f"Using remote dataset for {source['alias']}: {dataset_reference} with options {dataset_options}")

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
            # The `-processed` Dataset is built once during predownload (see
            # predownload_selected_sources). Here we only read it, so concurrent
            # Slurm array tasks never race on save_to_disk.
            dataset_reference = Path(args.cache_root) / "datasets--{}-processed".format(source["alias"])
            if not dataset_reference.exists():
                raise FileNotFoundError(
                    f"Processed dataset for {source['alias']} not found at {dataset_reference}. "
                    f"Run the predownload step (download_marin.sh) before tokenizing."
                )
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
            local_tasks=args.local_tasks,
            local_rank_offset=args.local_rank_offset,
            logging_dir=str(log_folder),
        )
        executor.run()
        

if __name__ == "__main__":
    main()
