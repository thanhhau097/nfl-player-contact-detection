import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import transformers
from joblib import Parallel, delayed
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from data_args import DataArguments
from dataset import NFLDataset, collate_fn, read_image
from engine import CustomTrainer, compute_metrics, matthews_corrcoef
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
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        # transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(training_args.seed)
    data_folder = data_args.data_folder
    fold = data_args.fold

    labels_df = pd.read_csv(os.path.join(data_folder, "train_features.csv"))
    helmets = pd.read_csv(os.path.join(data_folder, "train_baseline_helmets_kfold.csv"))

    train_dataset = NFLDataset(
        labels_df[labels_df["fold"] != fold],
        helmets[helmets["fold"] != fold],
        video_folder=os.path.join(data_folder, "train"),
        frames_folder=os.path.join(data_folder, f"train_frames{'_heatmap' if data_args.use_heatmap else ''}", str(fold)),
        mode="train",
        size=data_args.size,
        num_frames=data_args.num_frames,
        frame_steps=data_args.frame_steps,
        img_height=data_args.img_height,
        img_width=data_args.img_width,
        use_heatmap=data_args.use_heatmap,
        heatmap_sigma=data_args.heatmap_sigma
    )

    val_dataset = NFLDataset(
        labels_df[labels_df["fold"] == fold],
        helmets[helmets["fold"] == fold],
        video_folder=os.path.join(data_folder, "train"),
        frames_folder=os.path.join(data_folder, f"val_frames{'_heatmap' if data_args.use_heatmap else ''}", str(fold)),
        mode="val",
        size=data_args.size,
        num_frames=data_args.num_frames,
        frame_steps=data_args.frame_steps,
        img_height=data_args.img_height,
        img_width=data_args.img_width,
        use_heatmap=data_args.use_heatmap,
        heatmap_sigma=data_args.heatmap_sigma
    )

    # # Pre-load images
    # train_dataset.paths2images.update(
    #     dict(
    #         Parallel(n_jobs=96, verbose=1)(
    #             delayed(read_image)(image_path)
    #             for image_path in train_dataset.image_paths[: data_args.num_cache]
    #         )
    #     )
    # )
    # val_dataset.paths2images.update(
    #     dict(
    #         Parallel(n_jobs=96, verbose=1)(
    #             delayed(read_image)(image_path)
    #             for image_path in val_dataset.image_paths
    #         )
    #     )
    # )

    # Initialize trainer
    # model = Model(model_args.model_name, in_chans=data_args.num_frames)
    model = Model(model_args.model_name)
    if last_checkpoint is None and model_args.resume is not None:
        logger.info(f"Loading {model_args.resume} ...")
        checkpoint = torch.load(model_args.resume, "cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint.pop("state_dict")
        checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

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
        logger.info("*** Optimize MCC ***")
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        predictions = torch.sigmoid(torch.from_numpy(output.predictions)).numpy()
        labels = output.label_ids
        for threshold in thresholds:
            preds = predictions
            preds = preds > threshold
            score = matthews_corrcoef(labels, preds)
            scores.append(score)
        best_threshold = thresholds[np.argmax(scores)]
        best_score = np.max(scores)
        logger.info(f"Best matthews_corrcoef: {best_score} @ {best_threshold}")


if __name__ == "__main__":
    main()
