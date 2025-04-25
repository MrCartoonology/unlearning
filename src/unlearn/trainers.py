from typing import List, Tuple  
from transformers import Trainer
from torch.utils.data import Dataset
import random
import torch
from torch.utils.tensorboard import SummaryWriter

from unlearn.setup import RunTracker


def create_orthgrad_unlearn_trainer(res: RunTracker) -> Trainer:
    """
    Create an OrthoGradUnlearnTrainer instance.

    Args:
        res (RunTracker): The RunTracker instance containing the configuration and model.

    Returns:
        Trainer: An instance of OrthoGradUnlearnTrainer.
    """
    return OrthoGradUnlearnTrainer(
        retain_dataset=res.retain_dset,
        cfg=res.cfg,
        model=res.model,
        args=res.training_args,
        train_dataset=res.unlearn_dset,
        eval_dataset=None,
    )


class OrthoGradUnlearnTrainer(Trainer):
    def __init__(self, retain_dataset: Dataset, cfg: dict, *args, **kwargs):
        """
        Initializes the OrthoGradUnlearnTrainer.
        """
        super().__init__(*args, **kwargs)
        self.retain_dataset = retain_dataset

        cfg = cfg["orthgrad_unlearn_trainer"]
        self.enable_proj = cfg["enable_proj"]
        self.num_retain_per_grad = cfg["num_retain_per_grad"]
        self.retain_dim = cfg["retain_dim"]
        self.retain_recompute_freq = cfg["retain_recompute_freq"]

        self.retain_gradients = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Return negative loss for unlearning, assume inputs come from unlearning dataset"""
        outputs = model(**inputs)
        loss = outputs.loss
        neg_loss = -loss  # Multiply loss by -1 to apply a negative gradient
        return (neg_loss, outputs) if return_outputs else neg_loss

    def compute_retain_gradients(self):
        """
        Compute gradients on the retain dataset.

        Returns:
            List of flattened gradient vectors (tensors).
        """
        indicies = list(range(len(self.retain_dataset)))
        random.shuffle(indicies)
        assert (
            len(indicies) >= self.retain_dim * self.num_retain_per_grad
        ), "Not enough samples in retain dataset"
        indicies_chunks = [
            indicies[start: start + self.num_retain_per_grad]
            for start in range(0, len(indicies), self.num_retain_per_grad)
        ]

        model = self.model
        model.eval()
        grads = []

        while len(grads) < self.retain_dim:
            grad_indices = indicies_chunks.pop()
            samples = [self.retain_dataset[idx] for idx in grad_indices]
            sample_grads = [get_flat_grad(model, sample) for sample in samples]
            grads.append(torch.stack(sample_grads).mean(dim=0))

        retain_grad_norms = [grad.norm() for grad in grads]
        med_grad_norm = torch.median(torch.tensor(retain_grad_norms))

        grads = [grad / norm for grad, norm in zip(grads, retain_grad_norms)]
        cond_num = compute_condition_number(grads)

        if not hasattr(self, 'tb_writer'):
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        self.tb_writer.add_scalar("retain_grads/cond_num", cond_num, self.state.global_step)
        self.tb_writer.add_scalar("retain_grads/median_norm", med_grad_norm, self.state.global_step)
        model.train()
        return grads

    def training_step(self, model, inputs, num_items_in_batch):
        """modify result of backward pass to modify training step to compute gradients on retain dataset, and do orthogonal
        projection.  Note _inner_training_loop calls this function - it calls the optimizer to update
         parameters, but training_step does the backward pass - so wee need to modify the
          backward pass output here.
        """
        # code for base class training_step https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L3698C5-L3701C7

        if not self.enable_proj:
            return Trainer.training_step(self, model, inputs, num_items_in_batch)
        
        inputs = self._prepare_inputs(inputs)

        if (
            self.state.global_step % self.retain_recompute_freq == 0
        ) or self.retain_gradients is None:
            self.retain_gradients = self.compute_retain_gradients()

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        loss = self.compute_loss(model, inputs)
        del inputs

        loss.backward()

        # Collect current model gradients
        model_grads = [
            param.grad.detach().clone()
            for param in model.parameters()
            if param.grad is not None
        ]

        # Apply the orthogonal projection to the gradients
        apply_orthogonal_projection(model_grads, self.retain_gradients)

        # Now copy projected gradients back to model parameters
        idx = 0
        for param in model.parameters():
            if param.grad is not None:
                param.grad.copy_(model_grads[idx])
                idx += 1

        return loss.detach()


def get_flat_grad(model, sample):
    sample = {
        k: v.unsqueeze(0).to(model.device)
        for k, v in sample.items()
        if torch.is_tensor(v)
    }
    model.zero_grad()
    outputs = model(**sample)
    loss = outputs.loss
    loss.backward()

    # Flatten gradients
    grad_list = []
    for param in model.parameters():
        if param.grad is not None:
            grad_list.append(param.grad.detach().view(-1))
    return torch.cat(grad_list)


def compute_condition_number(vectors: List[torch.Tensor]) -> float:
    # Stack vectors into a matrix
    X = torch.stack(vectors).T  # shape: (num_params, num_vectors)
    X -= X.mean(dim=0)  # Center the data
    S = torch.linalg.svdvals(X.cpu())  # Singular Value Decomposition
    if S.min() == 0:
        return -1.0
    return S.max() / S.min()


def apply_orthogonal_projection(model_gradients: List[torch.Tensor], retain_gradients: List[torch.Tensor]) -> None:
    """
    Apply orthogonal projection to model gradients to remove components in the span of retain_gradients.

    Args:
        model_gradients (list of torch.Tensor): Flattened current model gradients.
        retain_gradients (list of torch.Tensor): List of flattened retain gradients.
    """
    # Stack retain gradients into a matrix P (each column a vector)
    P = torch.stack(retain_gradients).T  # shape: (num_params, num_retain_vectors)

    # Flatten current model gradients into a single vector g
    g = torch.cat([g.view(-1) for g in model_gradients])

    # Compute projection
    # g_proj = g - P @ (P.T @ g)
    Pt_g = torch.matmul(P.T, g)
    projection = torch.matmul(P, Pt_g)
    g_proj = g - projection

    # Now assign projected gradients back to model parameters
    pointer = 0
    for param in model_gradients:
        numel = param.numel()
        param.copy_(g_proj[pointer: pointer + numel].view_as(param))
        pointer += numel
