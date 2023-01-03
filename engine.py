from typing import Dict

import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

from model import Model


class CustomTrainer(Trainer):
    def compute_loss(self, model: Model, inputs: Dict[str, torch.Tensor], return_outputs=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        outputs = model({k: v.to(device, non_blocking=True) for k, v in inputs.items()})
        loss_fct = F.binary_cross_entropy_with_logits
        labels = inputs.get("labels")
        loss = loss_fct(outputs.view(-1), labels.float())
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
        del inputs["images0"]
        del inputs["images1"]
        del inputs["boxes0"]
        del inputs["boxes1"]
        del inputs["features"]
        return loss, outputs, inputs["labels"]


def compute_metrics(eval_preds):
    predictions = torch.sigmoid(torch.from_numpy(eval_preds.predictions)).numpy()
    fbeta_score = pfbeta_torch(
        eval_preds.label_ids[0],
        predictions,
    )

    auc = roc_auc_score(eval_preds.label_ids, predictions)
    score = matthews_corrcoef(eval_preds.label_ids, predictions > 0.5)
    return {"pF1": fbeta_score, "AUC": auc, "matthews_corrcoef": score}


def pfbeta_torch(labels, preds, beta=1):
    preds = preds.clip(0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0.0
