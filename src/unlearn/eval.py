import torch
from transformers import TrainerCallback
from unlearn.setup import RunTracker


def add_prompt_eval(res: RunTracker) -> None:
    trainer = res.trainer
    cfg = res.cfg["eval"]

    for name in cfg.keys():
        if cfg[name]["enable"]:
            prompt = cfg[name]["prompt"]
            temp = cfg[name]["temp"]
            arg = dict(prompt=prompt, temp=temp)
            trainer.add_callback(
                PromptEvaluationCallback(res.tokenizer, name, arg, max_length=100)
            )


def ask_prompt(model, tokenizer, prompt, temp, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    # Generate model output using the specified temperature
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, temperature=temp)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output_text = set(output_text.split("\n")).difference(set(prompt))
    output_text = " ".join(sorted(output_text))
    return output_text


class PromptEvaluationCallback(TrainerCallback):
    def __init__(self, tokenizer, name, prompt, max_length=100):
        self.tokenizer = tokenizer
        self.name = name
        self.prompt = prompt["prompt"]
        self.max_length = max_length
        self.temperature = prompt["temp"]

    def on_log(self, args, state, control, **kwargs):
        # Only evaluate at logging steps
        if state.global_step % args.logging_steps == 0:
            model = kwargs.get("model", None)
            if model is None:
                return
            device = model.device
            input_ids = self.tokenizer.encode(self.prompt, return_tensors="pt").to(
                device
            )
            # Generate model output using the specified temperature
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids, max_length=self.max_length, temperature=self.temperature
                )
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            output_text = set(output_text.split("\n")).difference(set(self.prompt))
            output_text = " ".join(sorted(output_text))
            print(f"\n{self.name} [Step {state.global_step}] Prompt evaluation:")
            print(f"  Input: {self.prompt}")
            print(f"  Output: {output_text}\n")
