import torch

import unlearn.data as data
import unlearn.setup as setup
import unlearn.trainers as trainers
import unlearn.eval as eval


CFG_FNAME = setup.find_root() / "config/config.yaml"


def run(cfg_fname=CFG_FNAME):
    torch.set_default_device("cpu")
    cfg = setup.load_config(cfg_fname)
    res = setup.RunTracker(cfg=cfg)
    res.tokenizer, res.model = setup.load_tokenizer_and_base_model(cfg=cfg)
    res.model = setup.load_peft_model(model=res.model)
    res.unlearn_dset = data.make_unlearn_dset(cfg=cfg, tokenizer=res.tokenizer)
    res.retain_dset = data.make_retain_dset(cfg=cfg, tokenizer=res.tokenizer)
    res.trainer = trainers.create_orthgrad_unlearn_trainer(res=res)
    eval.add_prompt_eval(res=res)
    res.trainer.train()


if __name__ == "__main__":
    run()
