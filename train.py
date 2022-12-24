import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils.import_utils import is_torchdynamo_available

from data_args import DataArguments
from dataset import collate_fn, NFLDataset
from engine import CustomTrainer, compute_metrics, pfbeta_torch
from model import Model
from model_args import ModelArguments

torch.set_float32_matmul_precision("high")
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train"
                " from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        # transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(training_args.seed)
    data_folder = data_args.data_folder

    train_dataset = NFLDataset(
        csv_folder=data_folder,
        video_folder=os.path.join(data_folder, "train"),
        frames_folder=os.path.join(data_folder, "train_frames"),
        mode="train")
    
    val_dataset = NFLDataset(
        csv_folder=data_folder,
        video_folder=os.path.join(data_folder, "val"),
        frames_folder=os.path.join(data_folder, "val_frames"),
        mode="val")

    # Initialize trainer
    model = Model(model_args.model_name)
    if last_checkpoint is None and model_args.resume is not None:
        logger.info(f"Loading {model_args.resume} ...")
        checkpoint = torch.load(model_args.resume, "cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint.pop("state_dict")
        if "birads_linear.weight" in checkpoint:
            init_weight = checkpoint["birads_linear.weight"].mean(0, keepdim=True)
            init_bias = checkpoint["birads_linear.bias"].mean(0, keepdim=True)
            checkpoint = {
                k: v
                for k, v in checkpoint.items()
                if not k.startswith(("birads", "density", "finding"))
            }
            checkpoint["linear.weight"] = init_weight
            checkpoint["linear.bias"] = init_bias
        checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if is_torchdynamo_available():
        model = torch.compile(model)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        output = trainer.predict(val_dataset, metric_key_prefix="eval")
        metrics = output.metrics
        # torch.save(
        #     {
        #         "label_ids": output.label_ids[0],
        #         "patient_ids": output.label_ids[1],
        #         "laterality": output.label_ids[2],
        #         "predictions": output.predictions,
        #         "metrics": output.metrics,
        #     },
        #     os.path.join(training_args.output_dir, "output.pth"),
        # )
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info("*** Optimize pF1 ***")
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        predictions = torch.sigmoid(torch.from_numpy(output.predictions)).numpy()
        labels = output.label_ids[0]
        for threshold in thresholds:
            preds = predictions
            # preds[preds < threshold] = 0.0
            preds = preds > threshold
            pf1 = pfbeta_torch(labels, preds)
            scores.append(pf1)
        best_threshold = thresholds[np.argmax(scores)]
        best_score = np.max(scores)
        logger.info(f"Best pF1: {best_score} @ {best_threshold}")


if __name__ == "__main__":
    main()
