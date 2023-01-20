from typing import Dict, Optional, List
import time
import math
import numpy as np

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import PredictionOutput, speed_metrics
from transformers.debug_utils import DebugOption
from torch.utils.data import Dataset


from model import Model
from samplers import ProportionalTwoClassesBatchSampler

class CustomTrainer(Trainer):
    # def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    #     return ProportionalTwoClassesBatchSampler(
    #         self.train_dataset.labels["contact"].values,
    #         self.args.per_device_train_batch_size,
    #         minority_size_in_batch=self.args.per_device_train_batch_size // 8,
    #         world_size=self.args.world_size,
    #         local_rank=self.args.local_rank
    #     )

    def compute_loss(self, model: Model, inputs: Dict, return_outputs=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        outputs = model(
                    inputs["images"].to(device), 
                    inputs["features"].to(device),
                    inputs["contact_ids"],
                )
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
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def prediction_step(
        self, model, inputs, prediction_loss_only=False, ignore_keys=None
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)
        outputs = outputs.float()
        outputs = nested_detach(outputs)
        del inputs["images"]
        del inputs["features"]
        return loss, outputs, inputs["labels"]

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        # torch.save(self.model.state_dict(), "debug_ckpt.pth")
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

def compute_metrics(eval_preds):
    try:
        predictions = torch.sigmoid(torch.from_numpy(eval_preds.predictions)).numpy()
        predictions = np.nan_to_num(predictions)
        fbeta_score = pfbeta_torch(
            eval_preds.label_ids[0],
            predictions,
        )

        auc = roc_auc_score(eval_preds.label_ids, predictions)
        score = matthews_corrcoef(eval_preds.label_ids, predictions > 0.5)
    except:
        import pdb;pdb.set_trace()
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
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0.0
