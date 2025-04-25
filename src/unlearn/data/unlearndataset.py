import os
import re
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
)
from unlearn.setup import safe_open


def write_unlearn_chunks(cfg: dict) -> None:
    cache = cfg["cache"]
    unlearn_word = cfg["unlearn_word"]
    input_fname = cfg["input_fname"]
    context_lines = cfg["context_lines"]
    chunk_delim = cfg["chunk_delim"]
    unlearn_output_fname = cfg["unlearn_output_fname"]

    if cache and os.path.exists(unlearn_output_fname):
        print(f"Output file {unlearn_output_fname} already exists. Skipping...")
        return

    if not os.path.exists(input_fname):
        raise FileNotFoundError(f"Input file {input_fname} does not exist")

    input_lns = [ln for ln in open(input_fname, "r").readlines() if ln.strip()]
    matches = [idx for idx, ln in enumerate(input_lns) if unlearn_word in ln.lower()]

    chunk_lines = [(idx - context_lines, idx + context_lines + 1) for idx in matches]
    chunks = [
        "".join(input_lns[max(0, start) : min(len(input_lns), end)])
        for start, end in chunk_lines
    ]
    out = chunk_delim.join(chunks)

    with safe_open(unlearn_output_fname, "w") as f:
        f.write(out)
    print(
        f"Wrote {len(chunks)} chunks with case-insensitive {unlearn_word} to {unlearn_output_fname}"
    )


def make_unlearn_dset(cfg: dict, tokenizer: AutoTokenizer) -> Dataset:
    write_unlearn_chunks(cfg=cfg)

    unlearn_raw_samples = (
        open(cfg["unlearn_output_fname"]).read().split(cfg["chunk_delim"])
    )
    unlearn_keyword_split_tokenized_samples = make_keyword_split_tokenized_samples(
        raw_samples=unlearn_raw_samples,
        tokenizer=tokenizer,
        keyword=cfg["unlearn_word"],
    )

    unlearn_windowed_samples = make_keyword_windowed_samples(
        ksplit_tokenized=unlearn_keyword_split_tokenized_samples,
        tokenizer=tokenizer,
        approx_window_len=cfg["training_window_tokens"],
        limit=cfg["limit_window_samples"],
    )

    return KeywordSplitDataset(
        windowed_samples=unlearn_windowed_samples,
        tokenizer=tokenizer,
        max_dataset_tokens=cfg["max_dataset_tokens"],
    )


def make_keyword_split_tokenized_samples(
    raw_samples: List[str], tokenizer: AutoTokenizer, keyword: str
) -> List[Tuple[List[str], List[str], List[str]]]:
    ksplit = []
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    for sample in raw_samples:
        # just work with first match
        match = pattern.search(sample)
        if match:
            before_tokens = tokenizer.tokenize(sample[: match.start()])
            keyword_tokens = tokenizer.tokenize(sample[match.start(): match.end()])
            after_tokens = tokenizer.tokenize(sample[match.end():])
            ksplit.append((before_tokens, keyword_tokens, after_tokens))
    return ksplit


def make_keyword_windowed_samples(
    ksplit_tokenized: List[Tuple[List[str], List[str], List[str]]],
    tokenizer: AutoTokenizer,
    approx_window_len: int,
    limit: bool = False,
) -> List[Tuple[str, str]]:
    windowed = []
    for ksplit in ksplit_tokenized:
        before_tokens, keyword_tokens, after_tokens = ksplit
        first = max(0, len(before_tokens) - approx_window_len)
        first = backup_to_word_boundary(before_tokens, first)
        last = 0

        while first <= len(before_tokens) or last <= len(after_tokens):
            keyword_plus = tokenizer.convert_tokens_to_string(
                keyword_tokens + after_tokens[0:last]
            )
            windowed.append(
                (
                    tokenizer.convert_tokens_to_string(before_tokens[first:]),
                    keyword_plus,
                )
            )
            first += 1
            first = forward_to_word_boundary(before_tokens, first)
            last += 1
            last = forward_to_word_boundary(after_tokens, last)
            if limit:
                break
    return windowed


def backup_to_word_boundary(tokens: List[str], start: int) -> int:
    # Find the last space before the start index.
    # tokenizer will use characters like Ċ for a newline, or Ġ for a space. There will also be code tokens
    # like [[, we want to stop at code tokens, space or \n, but not have the window cut off words.

    while start > 0 and tokens[start][0].isalnum():
        start -= 1
    return start


def forward_to_word_boundary(tokens: List[str], start: int) -> int:
    while start < len(tokens) and tokens[start][0].isalnum():
        start += 1
    return start


class KeywordSplitDataset(Dataset):
    """each sample has a keyword and is split into text before and after/including the keyword.
    The Labels are for the keyword and onward"""

    def __init__(
        self, windowed_samples: List[Tuple[str, str]], tokenizer: AutoTokenizer, max_dataset_tokens: int
    ):
        self.windowed_samples = windowed_samples
        self.tokenizer = tokenizer
        self.max_dataset_tokens = max_dataset_tokens


    def __len__(self) -> int:
        return len(self.windowed_samples)

    def __getitem__(self, idx: int) -> dict:
        before, keyword_onward = self.windowed_samples[idx]
        before_ids = self.tokenizer(before, return_attention_mask=False)["input_ids"]
        n_before = len(before_ids)
        max_length = self.max_dataset_tokens - n_before
        keyword_onward_ids = self.tokenizer(
            keyword_onward, return_attention_mask=False, padding="max_length", truncation=True, max_length=max_length
        )["input_ids"]
        
        labels = [-100] * n_before + keyword_onward_ids
        input_ids = before_ids + keyword_onward_ids
        attention_mask = [1] * len(input_ids)
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        input_ids = input_ids.to('cpu')
        labels = labels.to('cpu')
        attention_mask = attention_mask.to('cpu')

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "before": before,
            "keyword_onward": keyword_onward,
        }