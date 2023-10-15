import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import transformers
from joblib import Parallel, delayed
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from tqdm import tqdm

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

    val_dataset = NFLDataset(
        labels_df[labels_df["fold"] == fold],
        helmets[helmets["fold"] == fold],
        video_folder=os.path.join(data_folder, "train"),
        frames_folder=os.path.join('/data/hoanganh/', "frames"),
        mode="val",
        size=data_args.size,
        num_frames=data_args.num_frames,
        frame_steps=data_args.frame_steps,
        img_height=data_args.img_height,
        img_width=data_args.img_width,
        use_heatmap=data_args.use_heatmap,
        heatmap_sigma=data_args.heatmap_sigma
    )

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
    model.eval()

    test_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=32, pin_memory=True)
    feat_prob = {}
    print("Inference..")
    with torch.no_grad():
        tk = tqdm(test_loader, total=len(test_loader))
        for step, batch in enumerate(tk):
            output = model(batch["images"].to(device), batch["features"].to(device))
            output = output.sigmoid().cpu().numpy()
            contact_ids = batch["contact_ids"]
            for i in range(len(contact_ids)):
                feat_prob[contact_ids[i]] = output[i][0]

    print(f"Saving result to {model_args.model_name}_prob_fold{data_args.fold}.pth..")
    torch.save(feat_prob, f"{model_args.model_name}_prob_fold{data_args.fold}.pth")

if __name__ == "__main__":
    main()
