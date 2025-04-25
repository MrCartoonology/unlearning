import sys
import ipdb
import traceback
import torch
import yaml
from pathlib import Path
from typing import List, Tuple

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

from peft import get_peft_model, LoraConfig, TaskType


def debug_hook(type_, value, tb):
    traceback.print_exception(type_, value, tb)
    print("\n--- entering post-mortem debugger ---")
    ipdb.post_mortem(tb)


def load_config(path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    root_path = find_root()
    return replace_root_in_dict(d=cfg, root_path=root_path)


def find_root() -> str:
    # Get absolute path to current file
    current_file = Path(__file__).resolve()

    # Make sure we are where we think we are
    assert current_file.name == "setup.py", f"Unexpected file: {current_file.name}"
    assert (
        current_file.parent.name == "unlearn"
    ), f"Expected 'unlearn', got {current_file.parent.name}"
    assert (
        current_file.parent.parent.name == "src"
    ), f"Expected 'src', got {current_file.parent.parent.name}"

    return current_file.parent.parent.parent


def replace_root_in_dict(d: dict, root_path: str) -> dict:
    """Recursively replace '{{root}}' with root_path in all string values."""
    if isinstance(d, dict):
        return {k: replace_root_in_dict(v, root_path) for k, v in d.items()}
    elif isinstance(d, list):
        return [replace_root_in_dict(item, root_path) for item in d]
    elif isinstance(d, str):
        return d.replace("{{root}}", str(root_path))
    else:
        return d


def safe_open(file_path, mode="w"):
    """Ensure parent directory exists, then open the file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, mode)


class RunTracker(object):
    def __init__(self, cfg: dict):
        if cfg["drop_into_debugger_on_error"]:
            print("Setting up post-mortem debugger")
            sys.excepthook = debug_hook

        self.cfg = cfg
        self.tokenizer = None
        self.model = None
        self.training_args = None
        self.unlearn_dset = None
        self.retain_dset = None

        self.training_args = TrainingArguments(**cfg["training_args"])

        self.unlearn_dset = None
        self.retain_dset = None
        self.trainer = None

        self.unlearn_prompt = dict(
            prompt="What does supercalifragilisticexpialidocious mean?", temp=0.01
        )
        self.keep_prompt = dict(
            prompt="What does sqproctarineaiainsuguaypeidazionale mean? What is the definition of it?",
            temp=0.1,
        )


def load_tokenizer_and_base_model(
    cfg: dict,
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
    model = model.to("cpu")
    print("Base model device:", next(model.parameters()).device)
    return tokenizer, model


def identify_target_modules(model: PreTrainedModel) -> List[str]:
    # Identify the target modules for LoRA.
    # model                     | numer layers | example name
    # 'EleutherAI/gpt-neo-125M' | 12           | "transformer.h.8.attn.attention.v_proj",
    # "EleutherAI/gpt-j-6B"     | 28           |  "transformer.h.8.attn.q_proj"
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "q_proj" in name:
            target_modules.append(name)
        elif isinstance(module, torch.nn.Linear) and "v_proj" in name:
            target_modules.append(name)
    return target_modules


def load_peft_model(model: PreTrainedModel) -> PreTrainedModel:
    target_modules = identify_target_modules(model)

    # Create LoRA config and apply it to the model
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model = model.to("cpu")
    print("Fine tune model device:", next(model.parameters()).device)
    return model
