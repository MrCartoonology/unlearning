import os
import random
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
)
from unlearn.setup import safe_open


def create_random_chunks(cfg: dict) -> None:

    source_file = cfg["input_fname"]  # use same input as unlearn, wikipedia
    chunk_split = cfg["chunk_delim"]
    lines_per_chunk = cfg["retain_lines_per_chunk"]
    num_chunks = cfg["retain_num_chunks"]
    avoid_word = cfg["unlearn_word"]
    output_file = cfg["retain_output_fname"]
    cache = cfg["cache"]

    if cache and os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping...")
        return

    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file {source_file} does not exist")

    assert chunk_split.endswith("\n"), "chunk_split should end with a newline character"
    # Read all lines from the source file
    with open(source_file, "r") as f:
        lines = f.readlines()
    total_lines = len(lines)
    chunks = []

    # Sample chunks until we have the desired number
    while len(chunks) < num_chunks:
        start = random.randint(0, total_lines - lines_per_chunk)
        chunk = "".join(lines[start : start + lines_per_chunk])
        # If the chunk contains the avoid_word (any case), skip it
        if avoid_word.lower() in chunk.lower():
            continue
        chunks.append(chunk)

    # Write chunks to the output file, separated by "--"
    with safe_open(output_file, "w") as out_f:
        out_f.write(chunk_split.join(chunks))
    print(
        f"Wrote {len(chunks)} chunks to {output_file} with {lines_per_chunk} lines each, avoiding {avoid_word}"
    )


def make_retain_dset(cfg: dict, tokenizer: AutoTokenizer) -> Dataset:
    create_random_chunks(cfg=cfg)
    retain_raw_samples = (
        open(cfg["retain_output_fname"]).read().split(cfg["chunk_delim"])
    )
    retain_tokenized = [tokenizer.tokenize(sample) for sample in retain_raw_samples]
    retain_samples = [
        tokenizer.convert_tokens_to_string(
            token_chunk[0 : min(len(token_chunk), cfg["training_window_tokens"])]
        )
        for token_chunk in retain_tokenized
    ]
    return UnlabeledPrefixDataset(
        samples=retain_samples,
        unlabel_fraction=0.5,
        tokenizer=tokenizer,
    )


class UnlabeledPrefixDataset(Dataset):
    """The initial portion of each sample will not be labeled for prediction."""

    def __init__(self, samples, unlabel_fraction, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer
        self.unlabel_fraction = unlabel_fraction

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        sample_ids = self.tokenizer(sample, return_attention_mask=False)["input_ids"]
        start_predictions = int(self.unlabel_fraction * len(sample_ids))
        labels = [-100 for _ in range(start_predictions)] + sample_ids[
            start_predictions:
        ]
        input_ids = sample_ids
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "sample": sample,
        }
