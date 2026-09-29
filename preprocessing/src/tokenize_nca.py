"""Tokenize the Neural Cellular Automata (NCA) pre-pretraining synthetic data.

GPU-accelerated implementation optimized for single-GPU
and scalable CPU fallback.

Based on:
- "Training Language Models via Neural Cellular Automata" (Lee et al., 2026, https://arxiv.org/abs/2603.10055)

Optimal configuration from the paper:
- Grid size: 12x12 (periodic boundary conditions)
- State alphabet (num_colors): 10 ({0, ..., 9})
- Patch size: 2x2 non-overlapping patches (10^4 = 10,000 vocabulary for patches)
- Grid delimiters: <grid> = 10000, </grid> = 10001 (total 38 tokens per grid timestep)
- Transition network: Conv3x3 (periodic wrap, 4 channels) -> Conv1x1 (16 channels, ReLU) -> Conv1x1 (10 logits)
- Stochasticity: temperature = 1e-4
- Warmup rollout: 10 steps (init_rollout_steps = 10)
- Step interval: dT = 1
- Complexity filter: GZIP compression ratio >= 0.50 (50%+ band)
- Diversity: 1 unique rule per trajectory simulation

Matched verify-ppt setting:
- 500 steps, 4096 seq len, 512 global batch size
- Number of sequences: 500 * 512 = 256,000 documents
- Target sequence length: 4096 tokens per document (108 grid timesteps sliced to 4096)
- Total tokens: 500 * 512 * 4096 = 1,048,576,000 tokens (~1.048B tokens)
"""

import argparse
import gzip
import io
import json
import math
import multiprocessing as mp
import os
import random
import time
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

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

# Defaults matching paper optimal configuration & verify-ppt recipe
DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_EOS_TOKEN = "<|end_of_text|>"

DEFAULT_NCA_GRID_SIZE = 12
DEFAULT_NCA_PATCH_SIZE = 2
DEFAULT_NCA_NUM_COLORS = 10
DEFAULT_NCA_TEMPERATURE = 1e-4
DEFAULT_NCA_IDENTITY_BIAS = 0.0
DEFAULT_NCA_INIT_ROLLOUT_STEPS = 10
DEFAULT_NCA_DT = 1
DEFAULT_NCA_FILTER_THRESHOLD = 0.50
DEFAULT_NCA_FILTER_UPPER_BOUND = 1.00

# verify-ppt setting: 500 steps, 512 global batch size, 4096 seq len (+4k safety buffer)
DEFAULT_NCA_NUM_SENTENCES = 260_000  # 500 * 512 = 256k + buffer
# 37 grid timesteps * 38 tokens/grid = 1,406 patch ints -> ~4096 SmolLM3 BPE tokens per doc (1 doc == 1 sample)
DEFAULT_NCA_SEQ_LENGTH = 1_406
DEFAULT_NCA_SEED = 0

# Presets optimized for A100 / H100 80GB GPUs
DEFAULT_GPU_BATCH_SIZE = 4_096
DEFAULT_CAND_BATCH_SIZE = 8_192

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# =====================================================================
# Batched GPU Vectorized NCA Engine
# =====================================================================

class BatchedNCAEngine:
    """High-performance batched NCA simulation running 100% on GPU using grouped convolutions."""

    def __init__(
        self,
        grid_size: int = DEFAULT_NCA_GRID_SIZE,
        num_colors: int = DEFAULT_NCA_NUM_COLORS,
        patch_size: int = DEFAULT_NCA_PATCH_SIZE,
        temperature: float = DEFAULT_NCA_TEMPERATURE,
        identity_bias: float = DEFAULT_NCA_IDENTITY_BIAS,
        device: Optional[torch.device] = None,
    ):
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.patch_size = patch_size
        self.temperature = temperature
        self.identity_bias = identity_bias

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.patches_h = grid_size // patch_size
        self.patches_w = grid_size // patch_size
        self.num_patches = self.patches_h * self.patches_w
        self.tokens_per_grid = 1 + self.num_patches + 1  # e.g., 38 for 12x12 grid with 2x2 patch

        self.start_token = num_colors ** (patch_size * patch_size)      # 10000
        self.end_token = num_colors ** (patch_size * patch_size) + 1    # 10001

        # Patch powers for tensorized patch encoding
        self.patch_powers = (
            self.num_colors ** torch.arange(patch_size * patch_size, dtype=torch.int64, device=self.device)
        )

    def sample_rule_weights(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample batch_size independent random NCA neural network rules matching Flax Lecun normal initialization in utils/nca.py.

        Flax default Conv initializer: lecun_normal (std = sqrt(1 / fan_in)), bias = 0.
        Conv1: 3x3 kernel, d_state (10) -> 4  (fan_in = 10 * 3 * 3 = 90)
        Conv2: 1x1 kernel, 4 -> 16            (fan_in = 4 * 1 * 1 = 4)
        Conv3: 1x1 kernel, 16 -> d_state (10) (fan_in = 16 * 1 * 1 = 16)
        """
        # Conv1: (4*B, 10, 3, 3), fan_in = 90
        w1 = torch.randn(batch_size * 4, self.num_colors, 3, 3, device=self.device) * math.sqrt(1.0 / (self.num_colors * 9))
        b1 = torch.zeros(batch_size * 4, device=self.device)

        # Conv2: (16*B, 4, 1, 1), fan_in = 4
        w2 = torch.randn(batch_size * 16, 4, 1, 1, device=self.device) * math.sqrt(1.0 / 4)
        b2 = torch.zeros(batch_size * 16, device=self.device)

        # Conv3: (10*B, 16, 1, 1), fan_in = 16
        w3 = torch.randn(batch_size * self.num_colors, 16, 1, 1, device=self.device) * math.sqrt(1.0 / 16)
        b3 = torch.zeros(batch_size * self.num_colors, device=self.device)

        return w1, b1, w2, b2, w3, b3

    def sample_init_state(self, batch_size: int) -> torch.Tensor:
        """Sample initial grid states matching NCA.init_state in utils/nca.py:

        init = repeat(jax.random.normal(rng, (n_groups, d_state)), "G D -> H W G D")
        state = categorical(rng, init)
        """
        # Sample per-rule base logit vector: (B, 1, 1, num_colors)
        init_logits = torch.randn(batch_size, 1, 1, self.num_colors, device=self.device)
        init_logits = init_logits.expand(batch_size, self.grid_size, self.grid_size, self.num_colors)

        # Gumbel-max sampling over init_logits
        u = torch.rand_like(init_logits).clamp_(1e-10, 1.0 - 1e-10)
        gumbel = -torch.log(-torch.log(u))
        state = (init_logits + gumbel).argmax(dim=-1)  # (B, H, W)
        return state

    def step_batch(
        self,
        state: torch.Tensor,  # (B, H, W) int64
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
        w3: torch.Tensor,
        b3: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one NCA step for all B simulations in parallel."""
        B = state.shape[0]

        # One-hot state: (B, 10, H, W)
        state_oh = F.one_hot(state, num_classes=self.num_colors).permute(0, 3, 1, 2).float()

        # Circular (periodic wrap) padding of 1 on spatial dims
        padded = F.pad(state_oh, (1, 1, 1, 1), mode="circular")  # (B, 10, H+2, W+2)

        # Reshape to (1, B*10, H+2, W+2) for grouped conv
        inp = padded.view(1, B * self.num_colors, self.grid_size + 2, self.grid_size + 2)

        # Conv1: (1, 4*B, H, W)
        h1 = F.conv2d(inp, w1, b1, groups=B)

        # Conv2 + ReLU: (1, 16*B, H, W)
        h2 = F.relu(F.conv2d(h1, w2, b2, groups=B))

        # Conv3 (logits): (1, 10*B, H, W) -> (B, 10, H, W)
        logits = F.conv2d(h2, w3, b3, groups=B).view(B, self.num_colors, self.grid_size, self.grid_size)

        if self.identity_bias > 0:
            logits = logits + state_oh * self.identity_bias

        if self.temperature > 1e-6:
            # Gumbel-max sampling for categorical distribution with temperature
            u = torch.rand_like(logits).clamp_(1e-10, 1.0 - 1e-10)
            gumbel = -torch.log(-torch.log(u))
            next_state = (logits / self.temperature + gumbel).argmax(dim=1)
        else:
            next_state = logits.argmax(dim=1)

        return next_state

    def rollout_batch(
        self,
        initial_state: torch.Tensor,  # (B, H, W)
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
        w3: torch.Tensor,
        b3: torch.Tensor,
        num_timesteps: int,
        warmup_steps: int = DEFAULT_NCA_INIT_ROLLOUT_STEPS,
        dt: int = DEFAULT_NCA_DT,
    ) -> torch.Tensor:
        """Run batched rollout for num_timesteps. Returns (B, num_timesteps, H, W)."""
        state = initial_state

        # Warmup rollout
        for _ in range(warmup_steps):
            state = self.step_batch(state, w1, b1, w2, b2, w3, b3)

        recorded = []
        step_count = 0
        while len(recorded) < num_timesteps:
            state = self.step_batch(state, w1, b1, w2, b2, w3, b3)
            step_count += 1
            if step_count % dt == 0:
                recorded.append(state)

        # Stack into (B, num_timesteps, H, W)
        return torch.stack(recorded, dim=1)

    def tokenize_rollout(self, rollout: torch.Tensor) -> torch.Tensor:
        """Tokenize (B, T, H, W) rollout into (B, T * tokens_per_grid) patch tokens."""
        B, T, H, W = rollout.shape
        n_h = self.patches_h
        n_w = self.patches_w
        p = self.patch_size

        # Reshape to (B, T, n_h, p, n_w, p) -> (B, T, n_h, n_w, p, p)
        reshaped = rollout.view(B, T, n_h, p, n_w, p).permute(0, 1, 2, 4, 3, 5)
        patches = reshaped.reshape(B, T, n_h * n_w, p * p)  # (B, T, num_patches, 4)

        # Patch token = sum_k patch[k] * 10^k
        patch_toks = (patches.to(torch.int64) * self.patch_powers).sum(dim=-1)  # (B, T, num_patches)

        # Add <grid> (10000) and </grid> (10001) delimiters
        start_tokens = torch.full((B, T, 1), self.start_token, dtype=torch.int64, device=self.device)
        end_tokens = torch.full((B, T, 1), self.end_token, dtype=torch.int64, device=self.device)

        full_grid_tokens = torch.cat([start_tokens, patch_toks, end_tokens], dim=-1)  # (B, T, 38)
        return full_grid_tokens.reshape(B, -1)  # (B, T * 38)


# =====================================================================
# Fast GZIP Complexity Calculation
# =====================================================================

def fast_zlib_complexity(byte_data: bytes) -> float:
    """Compute zlib / gzip compression ratio at level 9."""
    compressed = zlib.compress(byte_data, 9)
    return len(compressed) / max(len(byte_data), 1)


def _eval_gzip_ratios(byte_chunks: List[bytes]) -> List[float]:
    return [fast_zlib_complexity(b) for b in byte_chunks]


# =====================================================================
# Parallel Generator Pipeline
# =====================================================================

def generate_nca_dataset_gpu(
    output_path: Path,
    n_sentences: int = DEFAULT_NCA_NUM_SENTENCES,
    seq_length: int = DEFAULT_NCA_SEQ_LENGTH,
    grid_size: int = DEFAULT_NCA_GRID_SIZE,
    patch: int = DEFAULT_NCA_PATCH_SIZE,
    num_colors: int = DEFAULT_NCA_NUM_COLORS,
    temperature: float = DEFAULT_NCA_TEMPERATURE,
    identity_bias: float = DEFAULT_NCA_IDENTITY_BIAS,
    init_rollout_steps: int = DEFAULT_NCA_INIT_ROLLOUT_STEPS,
    dt: int = DEFAULT_NCA_DT,
    filter_threshold: float = DEFAULT_NCA_FILTER_THRESHOLD,
    filter_upper_bound: float = DEFAULT_NCA_FILTER_UPPER_BOUND,
    batch_size: int = DEFAULT_GPU_BATCH_SIZE,
    cand_batch_size: int = DEFAULT_CAND_BATCH_SIZE,
    seed: int = DEFAULT_NCA_SEED,
    num_threads: int = 8,
    force: bool = False,
) -> Path:
    """Generate n_sentences of NCA sequences using GPU batched simulation."""
    if output_path.exists() and not force:
        file_size = output_path.stat().st_size
        if file_size > 0:
            print(f"Found existing non-empty NCA text file at {output_path} ({file_size:,} bytes), skipping generation.")
            return output_path
        else:
            print(f"Found existing NCA text file at {output_path}, but it is EMPTY (0 bytes). Regenerating...")

    # Fast check: if a larger raw file exists (e.g. seq_length=4096), slice directly in seconds
    if not force:
        pattern = f"nca_sequences_g{grid_size}_p{patch}_c{num_colors}_thr{filter_threshold}_{n_sentences}_*.txt"
        larger_candidates = sorted(output_path.parent.glob(pattern))
        for cand in larger_candidates:
            if cand.exists() and cand.stat().st_size > 0 and cand != output_path:
                try:
                    cand_len = int(cand.stem.split("_")[-1])
                    if cand_len >= seq_length:
                        print(f"Found existing raw NCA file with longer sequence ({cand_len} >= {seq_length}): {cand}")
                        print(f"Fast-slicing first {seq_length} tokens per sequence to {output_path}...")
                        temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
                        with cand.open("r", encoding="utf-8") as in_f, temp_output_path.open("w", encoding="utf-8") as out_f:
                            for line in in_f:
                                tokens = line.strip().split()[:seq_length]
                                out_f.write(" ".join(tokens) + "\n")
                        temp_output_path.replace(output_path)
                        print(f"Successfully sliced {n_sentences:,} sequences to {output_path}!")
                        return output_path
                except Exception:
                    pass

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device} (CUDA: {torch.cuda.is_available()}, MPS: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()})")

    engine = BatchedNCAEngine(
        grid_size=grid_size,
        num_colors=num_colors,
        patch_size=patch,
        temperature=temperature,
        identity_bias=identity_bias,
        device=device,
    )

    tokens_per_grid = 1 + (grid_size // patch) * (grid_size // patch) + 1  # 38
    num_timesteps = int(math.ceil(seq_length / tokens_per_grid))          # ceil(4096 / 38) = 108

    print(
        f"Generating {n_sentences:,} NCA sequences | target_length={seq_length} "
        f"({num_timesteps} grid timesteps) | grid={grid_size}x{grid_size} | "
        f"patch={patch}x{patch} | gzip complexity band=[{filter_threshold:.2f}, {filter_upper_bound:.2f}] | "
        f"cand_batch={cand_batch_size}, rollout_batch={batch_size}..."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    start_time = time.time()
    generated_count = 0

    # Number of candidate rules to test in one GPU pass
    cand_batch_size = max(cand_batch_size, batch_size)

    with temp_output_path.open("w", encoding="utf-8") as out_f, tqdm(total=n_sentences, desc="Generating NCA sequences") as pbar:
        while generated_count < n_sentences:
            needed = n_sentences - generated_count

            # 1. Sample candidate rules on GPU
            with torch.no_grad():
                w1, b1, w2, b2, w3, b3 = engine.sample_rule_weights(cand_batch_size)
                s0 = engine.sample_init_state(cand_batch_size)

                # Test rollout for 10 steps to compute GZIP complexity
                test_rollout = engine.rollout_batch(
                    s0, w1, b1, w2, b2, w3, b3,
                    num_timesteps=10,
                    warmup_steps=init_rollout_steps,
                    dt=dt,
                )

                # Patch tokenize candidate rollouts on GPU (vocab size 10002 fits in uint16)
                B, T, H, W = test_rollout.shape
                n_h, n_w, p = engine.patches_h, engine.patches_w, engine.patch_size
                reshaped = test_rollout.view(B, T, n_h, p, n_w, p).permute(0, 1, 2, 4, 3, 5).reshape(B, T, n_h * n_w, p * p)
                cand_patch_tokens = (reshaped.to(torch.int64) * engine.patch_powers).sum(dim=-1).cpu().numpy().astype(np.uint16)

            # 2. Fast multi-threaded GZIP complexity check
            byte_data_list = [cand_patch_tokens[i].tobytes() for i in range(cand_batch_size)]
            ratios = _eval_gzip_ratios(byte_data_list)

            # 3. Find indices of accepted rules
            accepted_indices = [
                i for i, r in enumerate(ratios)
                if filter_threshold <= r <= filter_upper_bound
            ]

            if not accepted_indices:
                continue

            # Limit to needed count
            accepted_indices = accepted_indices[:min(len(accepted_indices), needed)]
            num_accepted = len(accepted_indices)

            # 4. Extract accepted weights and run full trajectory rollout on GPU
            idx_tensor = torch.tensor(accepted_indices, device=device, dtype=torch.int64)

            with torch.no_grad():
                # Slice grouped convolution weights for accepted rules
                # w1: (4*B, 10, 3, 3) -> indices: idx*4 .. idx*4 + 3
                w1_idx = (idx_tensor.unsqueeze(1) * 4 + torch.arange(4, device=device)).reshape(-1)
                b1_idx = w1_idx
                acc_w1 = w1[w1_idx]
                acc_b1 = b1[b1_idx]

                w2_idx = (idx_tensor.unsqueeze(1) * 16 + torch.arange(16, device=device)).reshape(-1)
                b2_idx = w2_idx
                acc_w2 = w2[w2_idx]
                acc_b2 = b2[b2_idx]

                w3_idx = (idx_tensor.unsqueeze(1) * num_colors + torch.arange(num_colors, device=device)).reshape(-1)
                b3_idx = w3_idx
                acc_w3 = w3[w3_idx]
                acc_b3 = b3[b3_idx]

                # Sample fresh initial grid states for the full rollout matching utils/nca.py
                acc_s0 = engine.sample_init_state(num_accepted)

                # Full rollout on GPU
                full_rollout = engine.rollout_batch(
                    acc_s0, acc_w1, acc_b1, acc_w2, acc_b2, acc_w3, acc_b3,
                    num_timesteps=num_timesteps,
                    warmup_steps=init_rollout_steps,
                    dt=dt,
                )

                # Patch tokenize and truncate to exact sequence length
                full_tokens = engine.tokenize_rollout(full_rollout)  # (num_accepted, num_timesteps * 38)
                sliced_tokens = full_tokens[:, :seq_length].cpu().numpy()

            # 5. Write lines to text file
            lines = [" ".join(map(str, row)) + "\n" for row in sliced_tokens]
            out_f.writelines(lines)

            generated_count += num_accepted
            pbar.update(num_accepted)

    # Atomically move temp file to destination
    temp_output_path.replace(output_path)

    elapsed = time.time() - start_time
    rate = generated_count / max(elapsed, 0.001)
    tokens_rate = (generated_count * seq_length) / max(elapsed, 0.001)
    print(
        f"\nDone! Generated {generated_count:,} NCA sequences in {elapsed:.2f}s "
        f"({rate:.1f} seq/s, {tokens_rate:,.0f} tokens/s)."
    )
    print(f"Output saved to: {output_path}")
    return output_path


# =====================================================================
# DataTrove Integration
# =====================================================================

def build_sources(nca_text_path: Path):
    if nca_text_path is None:
        raise ValueError("nca_text_path must be provided")

    return [
        {
            "alias": "nca",
            "dataset": "text",
            "dataset_options": {
                "data_files": str(nca_text_path),
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
    parser = argparse.ArgumentParser(
        description="Generate and tokenize NCA pre-pretraining synthetic dataset (GPU accelerated)."
    )
    parser.add_argument("--output-root", required=True, help="Destination directory for tokenized output shards.")
    parser.add_argument("--logging-root", required=True, help="Directory for DataTrove execution logs.")
    parser.add_argument("--cache-root", required=True, help="Directory for cache.")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="HuggingFace tokenizer name or path.")
    parser.add_argument("--eos-token", default=DEFAULT_EOS_TOKEN, help="EOS token string.")
    parser.add_argument("--tasks", type=int, default=128, help="Number of DataTrove tokenization tasks.")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers for tokenization.")
    parser.add_argument("--batch-size", type=int, default=1000, help="DocumentTokenizer batch size.")
    parser.add_argument("--gpu-batch-size", type=int, default=DEFAULT_GPU_BATCH_SIZE,
                        help=f"Rollout batch size on GPU (default: {DEFAULT_GPU_BATCH_SIZE} for A100/H100).")
    parser.add_argument("--cand-batch-size", type=int, default=DEFAULT_CAND_BATCH_SIZE,
                        help=f"Candidate filtering batch size on GPU (default: {DEFAULT_CAND_BATCH_SIZE} for A100/H100).")

    # NCA parameters
    parser.add_argument("--num-sentences", type=int, default=DEFAULT_NCA_NUM_SENTENCES,
                        help="Number of sequences (default: 256,000 for 500 steps * 512 global batch size).")
    parser.add_argument("--seq-length", type=int, default=DEFAULT_NCA_SEQ_LENGTH,
                        help="Sequence length in tokens (default: 4096).")
    parser.add_argument("--grid", type=int, default=DEFAULT_NCA_GRID_SIZE, help="NCA grid size H=W (default: 12).")
    parser.add_argument("--patch", type=int, default=DEFAULT_NCA_PATCH_SIZE, help="Patch size for tokenization (default: 2).")
    parser.add_argument("--num-colors", type=int, default=DEFAULT_NCA_NUM_COLORS, help="Alphabet size (default: 10).")
    parser.add_argument("--temperature", type=float, default=DEFAULT_NCA_TEMPERATURE, help="NCA transition softmax temperature.")
    parser.add_argument("--identity-bias", type=float, default=DEFAULT_NCA_IDENTITY_BIAS, help="Identity bias.")
    parser.add_argument("--init-rollout-steps", type=int, default=DEFAULT_NCA_INIT_ROLLOUT_STEPS, help="Initial rollout warmup steps.")
    parser.add_argument("--dt", type=int, default=DEFAULT_NCA_DT, help="Sampling timestep interval.")
    parser.add_argument("--filter-threshold", type=float, default=DEFAULT_NCA_FILTER_THRESHOLD,
                        help="Lower bound gzip compression ratio for rule filtering (default: 0.50).")
    parser.add_argument("--filter-upper-bound", type=float, default=DEFAULT_NCA_FILTER_UPPER_BOUND,
                        help="Upper bound gzip compression ratio (default: 1.00).")
    parser.add_argument("--seed", type=int, default=DEFAULT_NCA_SEED, help="Random seed.")
    parser.add_argument("--nca-file-dir", default=None,
                        help="Directory to save/load raw NCA .txt file. Defaults to <cache-root>/nca.")

    parser.add_argument("--predownload", action="store_true", help="Download dataset and tokenizer before tokenization.")
    parser.add_argument("--predownload-only", action="store_true", help="Exit after downloading without tokenizing.")
    parser.add_argument("--force", action="store_true", help="Regenerate even if files already exist.")
    return parser.parse_args()


def main():
    args = parse_args()

    nca_file_dir = Path(args.nca_file_dir) if args.nca_file_dir else (Path(args.cache_root) / "nca")
    output_txt_path = nca_file_dir / f"nca_sequences_g{args.grid}_p{args.patch}_c{args.num_colors}_thr{args.filter_threshold}_{args.num_sentences}_{args.seq_length}.txt"

    nca_text_path = generate_nca_dataset_gpu(
        output_path=output_txt_path,
        n_sentences=args.num_sentences,
        seq_length=args.seq_length,
        grid_size=args.grid,
        patch=args.patch,
        num_colors=args.num_colors,
        temperature=args.temperature,
        identity_bias=args.identity_bias,
        init_rollout_steps=args.init_rollout_steps,
        dt=args.dt,
        filter_threshold=args.filter_threshold,
        filter_upper_bound=args.filter_upper_bound,
        batch_size=args.gpu_batch_size,
        cand_batch_size=args.cand_batch_size,
        seed=args.seed,
        force=args.force,
    )

    sources = build_sources(nca_text_path=nca_text_path)

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

        print(f"Tokenizing {source['alias']} from {source['dataset']} to {output_folder}...")
        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=args.tasks,
            workers=args.workers,
            logging_dir=str(log_folder),
        )
        executor.run()

    print("Done! NCA data generation and tokenization completed successfully.")


if __name__ == "__main__":
    main()
