from functools import partial
from typing import Dict

import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

from model import SeqModel


class CustomTrainer(Trainer):
    def compute_loss(self, model: SeqModel, inputs: Dict[str, torch.Tensor], return_outputs=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        outputs = model(inputs["features"].to(device, non_blocking=True))
        loss_fct = F.binary_cross_entropy_with_logits
        # loss_fct = partial(
        #     F.binary_cross_entropy_with_logits, pos_weight=torch.FloatTensor([10.0]).to(device)
        # )
        labels = inputs.get("labels").to(device, non_blocking=True)
        loss = loss_fct(outputs[labels != -100], labels[labels != -100].float())
        if return_outputs:
            return (loss, outputs)
        return loss

    def create_optimizer(self):
        model = self.model
        no_decay = []
        for n, m in model.named_modules():
            if isinstance(
                m,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.LayerNorm,
                    torch.nn.LayerNorm,
                    torch.nn.GroupNorm,
                ),
            ):
                no_decay.append(n)

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)
        outputs = outputs.float()
        outputs = nested_detach(outputs)
        return loss, outputs, inputs["labels"]
