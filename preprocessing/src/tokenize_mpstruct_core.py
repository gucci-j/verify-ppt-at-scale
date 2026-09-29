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


def decompress_zstd(fileobj, mode="rb", **kwargs):
    dctx = zstandard.ZstdDecompressor()
    return dctx.stream_reader(fileobj)


fsspec.compression.compr["zstd"] = decompress_zstd
fsspec.compression.compr["zstandard"] = decompress_zstd
print(f"Forced 'zstd' into registry. Current keys: {list(fsspec.compression.compr.keys())}")

DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_EOS_TOKEN = "<|end_of_text|>"
DEFAULT_K_STRUCT = 1
DEFAULT_K_DEP = 4
DEFAULT_USE_HEAD_DIVERSITY = True
DEFAULT_USE_COMPLEX_ARGS = False
# 260_000 documents (500 steps * 512 batch = 256k required, +4k safety buffer for shard boundaries)
DEFAULT_MPSTRUCT_NUM_SENTENCES = 260_000
# 2048 ints -> 4097 SmolLM3 BPE tokens per doc (2N+1 with trailing space + EOS, matching 1 sample == 1 doc)
DEFAULT_MPSTRUCT_SEQ_LENGTH = 2048
DEFAULT_SEED = 42
DEFAULT_FORMAT = "ids"


class EnhancedConstrainedGenerator:
    """MP-STRUCT CORE generator based on Mita et al. (ACL 2026).
    
    Generates Minimalist Program (MP) inspired structured sequences with:
    - Hierarchical structure (Merge)
    - Feature-based dependencies (Agree)
    - Long-distance displacement (Move)
    - Optional functional head diversity (HEAD_CP, HEAD_TP, HEAD_VP)
    """

    def __init__(
        self,
        k_struct=1,
        k_dep=4,
        use_head_diversity=True,
        use_complex_args=False,
        seed=42,
    ):
        self.rng = random.Random(seed)
        self.k_struct = k_struct
        self.k_dep = k_dep
        self.use_head_diversity = use_head_diversity
        self.use_complex_args = use_complex_args

        self.total_k = k_struct + k_dep
        self.bracket_vocab_size = self.total_k * 2

        self.DEP_MOVE = k_struct + 0
        self.DEP_AGR_A = k_struct + 1
        self.DEP_AGR_B = k_struct + 2
        self.DEP_SEL = k_struct + 3

        self.HEAD_CP_ID = self.bracket_vocab_size + 0
        self.HEAD_TP_ID = self.bracket_vocab_size + 1
        self.HEAD_VP_ID = self.bracket_vocab_size + 2

        self.LEAF_ID = self.bracket_vocab_size + 3

    def _open(self, id):
        return id

    def _close(self, id):
        return id + self.total_k

    def _wrap_struct(self, content_ids):
        s_id = 0
        return [self._open(s_id)] + content_ids + [self._close(s_id)]

    def _gen_local_pair(self):
        p1 = self._wrap_struct([self._open(self.DEP_SEL)])
        p2 = self._wrap_struct([self._close(self.DEP_SEL)])
        return p1 + p2

    def _get_head_token(self, layer_type):
        if not self.use_head_diversity:
            return []

        if layer_type == "CP":
            return [self.HEAD_CP_ID]
        elif layer_type == "TP":
            return [self.HEAD_TP_ID]
        elif layer_type == "VP":
            return [self.HEAD_VP_ID]
        return []

    def get_vocab(self):
        vocab = {}

        for x in range(self.bracket_vocab_size):
            total_k = self.total_k
            is_open = x < total_k
            base_id = x if is_open else x - total_k

            if base_id < self.k_struct:
                sym = f"[{base_id}" if is_open else f"]{base_id}"
            else:
                sym = f"({base_id}" if is_open else f"){base_id}"
            vocab[sym] = x

        vocab["HEAD_CP"] = self.HEAD_CP_ID
        vocab["HEAD_TP"] = self.HEAD_TP_ID
        vocab["HEAD_VP"] = self.HEAD_VP_ID

        vocab["LEAF"] = self.LEAF_ID

        return vocab

    def id_to_token(self, x):
        if x < self.bracket_vocab_size:
            total_k = self.total_k
            is_open = x < total_k
            base_id = x if is_open else x - total_k

            if base_id < self.k_struct:
                return f"[{base_id}" if is_open else f"]{base_id}"
            else:
                return f"({base_id}" if is_open else f"){base_id}"
        elif x == self.HEAD_CP_ID:
            return "HEAD_CP"
        elif x == self.HEAD_TP_ID:
            return "HEAD_TP"
        elif x == self.HEAD_VP_ID:
            return "HEAD_VP"
        elif x == self.LEAF_ID:
            return "LEAF"
        return "UNK"

    def generate_tree_sequence(self):
        has_move = self.rng.random() < 0.5
        agree_type = self.DEP_AGR_A if self.rng.random() < 0.5 else self.DEP_AGR_B

        args = []

        if has_move:
            subj_slot = [self._close(self.DEP_MOVE)]
        else:
            subj_slot = []

        if self.use_complex_args:
            rand_val = self.rng.random()
            if rand_val < 0.33:
                pass
            elif rand_val < 0.66:
                args.append(self._gen_local_pair())
            else:
                args.append(self._gen_local_pair())
                args.append(self._gen_local_pair())
        else:
            args.append(self._gen_local_pair())

        head_v_tokens = self._get_head_token("VP")
        head_v = self._wrap_struct(head_v_tokens)

        vp_inner = []
        elements = [head_v, self._wrap_struct(subj_slot)] + [self._wrap_struct(a) for a in args]
        self.rng.shuffle(elements)

        for e in elements:
            vp_inner += e

        vp_block = self._wrap_struct(vp_inner)

        head_t_tokens = self._get_head_token("TP")
        head_t_content = head_t_tokens + [self._close(agree_type)]
        head_t = self._wrap_struct(head_t_content)

        subj_content = self._gen_local_pair()
        subj_block = self._wrap_struct([self._open(agree_type)] + subj_content)

        if has_move:
            spec_tp = []
        else:
            spec_tp = subj_block

        tp_inner = []
        if spec_tp:
            tp_inner += spec_tp
        else:
            tp_inner += self._wrap_struct([])

        tp_inner += head_t
        tp_inner += vp_block

        tp_block = self._wrap_struct(tp_inner)

        head_c_tokens = self._get_head_token("CP")
        head_c_content = head_c_tokens
        if has_move:
            head_c_content += [self._open(self.DEP_MOVE)]

        head_c = self._wrap_struct(head_c_content)

        spec_cp = []
        if has_move:
            spec_cp = subj_block

        cp_inner = []
        if spec_cp:
            cp_inner += self._wrap_struct(spec_cp)
        else:
            cp_inner += self._wrap_struct([])

        cp_inner += head_c
        cp_inner += tp_block

        cp_block = self._wrap_struct(cp_inner)

        return cp_block


def generate_mpstruct_core_txt_file(
    file_dir,
    n=DEFAULT_MPSTRUCT_NUM_SENTENCES,
    target_length=DEFAULT_MPSTRUCT_SEQ_LENGTH,
    k_struct=DEFAULT_K_STRUCT,
    k_dep=DEFAULT_K_DEP,
    use_head_diversity=DEFAULT_USE_HEAD_DIVERSITY,
    use_complex_args=DEFAULT_USE_COMPLEX_ARGS,
    seed=DEFAULT_SEED,
    data_format=DEFAULT_FORMAT,
    force=False,
):
    """Write `n` MP-STRUCT CORE sequences to disk and return the text file path."""
    file_dir = Path(file_dir)
    file_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{data_format}_{n}_{target_length}_hd{int(use_head_diversity)}_ca{int(use_complex_args)}_s{seed}"
    output_path = file_dir / f"mpstruct_core_{tag}.txt"
    vocab_path = file_dir / "vocab.json"

    generator = EnhancedConstrainedGenerator(
        k_struct=k_struct,
        k_dep=k_dep,
        use_head_diversity=use_head_diversity,
        use_complex_args=use_complex_args,
        seed=seed,
    )

    # Save vocab.json
    vocab = generator.get_vocab()
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    if output_path.exists() and not force:
        file_size = output_path.stat().st_size
        if file_size > 0:
            print(f"Found existing non-empty MP-STRUCT CORE file at {output_path} ({file_size:,} bytes), skipping generation.")
            return output_path
        else:
            print(f"Found existing MP-STRUCT CORE file at {output_path}, but it is EMPTY (0 bytes). Regenerating...")

    # Fast check: if a larger raw file exists (e.g. target_length=4096), slice directly in seconds
    if not force:
        pattern = f"mpstruct_core_{data_format}_{n}_*_hd{int(use_head_diversity)}_ca{int(use_complex_args)}_s{seed}.txt"
        larger_candidates = sorted(file_dir.glob(pattern))
        for cand in larger_candidates:
            if cand.exists() and cand.stat().st_size > 0 and cand != output_path:
                try:
                    cand_parts = cand.stem.split("_")
                    cand_len = int(cand_parts[3])
                    if cand_len >= target_length:
                        print(f"Found existing raw MP-STRUCT file with longer sequence ({cand_len} >= {target_length}): {cand}")
                        print(f"Fast-slicing first {target_length} tokens per sequence to {output_path}...")
                        temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
                        with cand.open("r", encoding="utf-8") as in_f, temp_output_path.open("w", encoding="utf-8") as out_f:
                            for line in in_f:
                                tokens = line.strip().split()[:target_length]
                                out_f.write(" ".join(tokens) + " \n")
                        temp_output_path.replace(output_path)
                        print(f"Successfully sliced {n:,} sequences to {output_path}!")
                        return output_path
                except Exception:
                    pass

    print(f"Generating MP-STRUCT CORE text dataset ({tag})...")
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_output_path.open("w", encoding="utf-8") as f:
        for _ in tqdm(range(n), desc="Generating mpstruct_core"):
            line_ids = []
            while len(line_ids) < target_length:
                sent_ids = generator.generate_tree_sequence()
                line_ids.extend(sent_ids)

            line_ids = line_ids[:target_length]

            if data_format == "tokens":
                tokens = [generator.id_to_token(x) for x in line_ids]
                line_str = " ".join(tokens)
            else:
                line_str = " ".join(str(x) for x in line_ids)

            # Trailing space ensures consistent tokenization boundaries
            f.write(f"{line_str} \n")

    temp_output_path.replace(output_path)
    return output_path


def build_sources(mpstruct_core_text_path):
    if mpstruct_core_text_path is None:
        raise ValueError("mpstruct_core_text_path must be provided")

    return [
        {
            "alias": "mpstruct_core",
            "dataset": "text",
            "dataset_options": {
                "data_files": str(mpstruct_core_text_path),
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
    parser = argparse.ArgumentParser(description="Generate and tokenize MP-STRUCT CORE dataset in DataTrove format.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--logging-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--eos-token", default=DEFAULT_EOS_TOKEN)
    parser.add_argument("--tasks", type=int, default=128)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-sentences", "-n", type=int, default=DEFAULT_MPSTRUCT_NUM_SENTENCES)
    parser.add_argument("--seq-length", type=int, default=DEFAULT_MPSTRUCT_SEQ_LENGTH)
    parser.add_argument("--k-struct", type=int, default=DEFAULT_K_STRUCT)
    parser.add_argument("--k-dep", type=int, default=DEFAULT_K_DEP)
    parser.add_argument("--use-head-diversity", action="store_true", default=DEFAULT_USE_HEAD_DIVERSITY)
    parser.add_argument("--no-head-diversity", dest="use_head_diversity", action="store_false")
    parser.add_argument("--use-complex-args", action="store_true", default=DEFAULT_USE_COMPLEX_ARGS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--format", choices=["ids", "tokens"], default=DEFAULT_FORMAT)
    parser.add_argument(
        "--mpstruct-core-file-dir",
        default=None,
        help="Directory where generated MP-STRUCT CORE .txt is written. Defaults to <cache-root>/mpstruct_core.",
    )
    parser.add_argument(
        "--predownload",
        action="store_true",
        help="Download dataset and tokenizer into --cache-root before tokenization.",
    )
    parser.add_argument(
        "--predownload-only",
        action="store_true",
        help="Exit after downloading/generating dataset and tokenizer.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require datasets and tokenizer to resolve from cache without network access.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate/retokenize even if output exists.")
    return parser.parse_args()


def main():
    args = parse_args()

    mpstruct_dir = (
        Path(args.mpstruct_core_file_dir)
        if args.mpstruct_core_file_dir
        else (Path(args.cache_root) / "mpstruct_core")
    )
    mpstruct_text_path = generate_mpstruct_core_txt_file(
        file_dir=mpstruct_dir,
        n=args.num_sentences,
        target_length=args.seq_length,
        k_struct=args.k_struct,
        k_dep=args.k_dep,
        use_head_diversity=args.use_head_diversity,
        use_complex_args=args.use_complex_args,
        seed=args.seed,
        data_format=args.format,
        force=args.force,
    )

    sources = build_sources(mpstruct_core_text_path=mpstruct_text_path)

    if args.predownload or args.predownload_only:
        predownload_selected_sources(sources, args.tokenizer, args.cache_root, args.workers, force=args.force)
    if args.predownload_only:
        print("Finished predownloading/generating MP-STRUCT CORE sources and tokenizer.")
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
                Path(args.cache_root) / f"datasets--{source['alias']}" / "train-*.parquet"
            ]
            dataset = load_dataset("parquet", data_files=[str(path) for path in output_files], split="train")
            dataset.save_to_disk(Path(args.cache_root) / f"datasets--{source['alias']}-processed")
            dataset_reference = Path(args.cache_root) / f"datasets--{source['alias']}-processed"
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
