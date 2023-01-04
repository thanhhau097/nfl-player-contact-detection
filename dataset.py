import os
import random
import subprocess
from multiprocessing import Pool

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from tqdm import tqdm
from turbojpeg import TJPF_GRAY, TurboJPEG

turbo_jpeg = TurboJPEG()

USE_COLS = [
    "x_position_1",
    "y_position_1",
    "speed_1",
    "distance_1",
    "direction_1",
    "orientation_1",
    "acceleration_1",
    "sa_1",
    "x_position_2",
    "y_position_2",
    "speed_2",
    "distance_2",
    "direction_2",
    "orientation_2",
    "acceleration_2",
    "sa_2",
    "distance",
    "G_flug",
    "left_Sideline_10_1",
    "width_Sideline_10_1",
    "top_Sideline_10_1",
    "height_Sideline_10_1",
    "left_Sideline_10_2",
    "width_Sideline_10_2",
    "top_Sideline_10_2",
    "height_Sideline_10_2",
    "left_Endzone_10_1",
    "width_Endzone_10_1",
    "top_Endzone_10_1",
    "height_Endzone_10_1",
    "left_Endzone_10_2",
    "width_Endzone_10_2",
    "top_Endzone_10_2",
    "height_Endzone_10_2",
    "left_Sideline_50_1",
    "width_Sideline_50_1",
    "top_Sideline_50_1",
    "height_Sideline_50_1",
    "left_Sideline_50_2",
    "width_Sideline_50_2",
    "top_Sideline_50_2",
    "height_Sideline_50_2",
    "left_Endzone_50_1",
    "width_Endzone_50_1",
    "top_Endzone_50_1",
    "height_Endzone_50_1",
    "left_Endzone_50_2",
    "width_Endzone_50_2",
    "top_Endzone_50_2",
    "height_Endzone_50_2",
    "left_Sideline_100_1",
    "width_Sideline_100_1",
    "top_Sideline_100_1",
    "height_Sideline_100_1",
    "left_Sideline_100_2",
    "width_Sideline_100_2",
    "top_Sideline_100_2",
    "height_Sideline_100_2",
    "left_Endzone_100_1",
    "width_Endzone_100_1",
    "top_Endzone_100_1",
    "height_Endzone_100_1",
    "left_Endzone_100_2",
    "width_Endzone_100_2",
    "top_Endzone_100_2",
    "height_Endzone_100_2",
    "left_Sideline_500_1",
    "width_Sideline_500_1",
    "top_Sideline_500_1",
    "height_Sideline_500_1",
    "left_Sideline_500_2",
    "width_Sideline_500_2",
    "top_Sideline_500_2",
    "height_Sideline_500_2",
    "left_Endzone_500_1",
    "width_Endzone_500_1",
    "top_Endzone_500_1",
    "height_Endzone_500_1",
    "left_Endzone_500_2",
    "width_Endzone_500_2",
    "top_Endzone_500_2",
    "height_Endzone_500_2",
    "x_position_diff",
    "y_position_diff",
    "speed_diff",
    "distance_diff",
    "direction_diff",
    "orientation_diff",
    "acceleration_diff",
    "sa_diff",
    "left_Sideline_10_diff",
    "width_Sideline_10_diff",
    "top_Sideline_10_diff",
    "height_Sideline_10_diff",
    "left_Endzone_10_diff",
    "width_Endzone_10_diff",
    "top_Endzone_10_diff",
    "height_Endzone_10_diff",
    "left_Sideline_50_diff",
    "width_Sideline_50_diff",
    "top_Sideline_50_diff",
    "height_Sideline_50_diff",
    "left_Endzone_50_diff",
    "width_Endzone_50_diff",
    "top_Endzone_50_diff",
    "height_Endzone_50_diff",
    "left_Sideline_100_diff",
    "width_Sideline_100_diff",
    "top_Sideline_100_diff",
    "height_Sideline_100_diff",
    "left_Endzone_100_diff",
    "width_Endzone_100_diff",
    "top_Endzone_100_diff",
    "height_Endzone_100_diff",
    "left_Sideline_500_diff",
    "width_Sideline_500_diff",
    "top_Sideline_500_diff",
    "height_Sideline_500_diff",
    "left_Endzone_500_diff",
    "width_Endzone_500_diff",
    "top_Endzone_500_diff",
    "height_Endzone_500_diff",
    "x_position_prod",
    "y_position_prod",
    "speed_prod",
    "distance_prod",
    "direction_prod",
    "orientation_prod",
    "acceleration_prod",
    "sa_prod",
]


def run_video_cmd(cmd):
    print(cmd)
    if "Endzone2" not in cmd:
        subprocess.run(cmd, shell=True)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h.astype(np.float32)


def read_image(path: str):
    return turbo_jpeg.decode(open(path, "rb").read(), pixel_format=TJPF_GRAY)[:, :, 0]


def build_data_aug(mode: str):
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                A.Normalize(mean=[0.0], std=[1.0]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc"),
        )
    else:
        return A.Compose(
            [A.Normalize(mean=[0.0], std=[1.0]), ToTensorV2()],
            bbox_params=A.BboxParams(format="pascal_voc"),
        )


class NFLDataset(Dataset):
    def __init__(
        self,
        labels_df: pd.DataFrame,
        helmets: pd.DataFrame,
        video_folder: str,
        frames_folder: str,
        mode: str = "train",
        fold: int = 0,
        size: int = 256,
        num_frames: int = 13,
        frame_steps: int = 4,
        img_height=720,
        img_width=1280,
        use_heatmap=False,
        heatmap_sigma=128,
    ):
        self.labels = labels_df
        self.helmets = helmets
        self.video_folder = video_folder
        self.frames_folder = frames_folder
        self.mode = mode
        self.fold = fold
        self.size = size
        self.num_frames = num_frames
        self.frame_steps = frame_steps
        self.img_width = img_width
        self.img_height = img_height
        self.sigma = heatmap_sigma
        self.use_heatmap = use_heatmap
        self.heatmap = gaussian2D((self.img_height, self.img_width), self.sigma)

        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder, exist_ok=True)

        print("Processing CSV data")
        self.preprocess_csv()
        self.paths2images = {}

        self.aug = build_data_aug(mode)

    def preprocess_video(self):
        cmds = []
        for video in tqdm(self.helmets.video.unique()):
            if not os.path.exists(os.path.join(self.frames_folder, video)):
                os.makedirs(os.path.join(self.frames_folder, video), exist_ok=True)
            # "-q:v 2 -vf format=gray"
            cmds.append(
                f"ffmpeg -i {os.path.join(self.video_folder, video)} -q:v 2 -f image2 "
                f"{os.path.join(self.frames_folder, video, video)}_%04d.jpg -hide_banner "
                "-loglevel error"
            )

        with Pool(32) as p:
            print(p.map(run_video_cmd, cmds))

    def preprocess_csv(self):
        # self.frame = self.labels["frame"].values
        # self.players = self.labels[["nfl_player_id_1", "nfl_player_id_2"]].values
        # self.game_play = self.labels.game_play.values
        self.features = self.labels[USE_COLS + ["game_play", "frame"]].fillna(-1)
        self.game_play_frame = self.labels[["game_play", "frame"]].drop_duplicates().values

        if len(os.listdir(self.frames_folder)) == 0:
            print("Extracting frames from scratch ...")
            self.preprocess_video()

        self.video2helmets = {}
        helmets_new = self.helmets.set_index("video")
        for video in tqdm(self.helmets.video.unique()):
            self.video2helmets[video] = helmets_new.loc[video].reset_index(drop=True)

        print("Mapping videos to frames")
        video_start_end = (
            self.labels.groupby(["game_play"]).agg({"frame": ["min", "max"]}).reset_index().values
        )
        self.video2frames = {}
        self.image_paths = []
        for row in video_start_end:
            game_play, start_idx, end_idx = row
            for view in ["Endzone", "Sideline"]:
                video = game_play + f"_{view}.mp4"
                end_idx = max(
                    [
                        int(path.split(".")[1].split("_")[1])
                        for path in os.listdir(os.path.join(self.frames_folder, video))
                    ]
                )
                self.video2frames[video] = end_idx
                self.image_paths.extend(
                    [
                        os.path.join(self.frames_folder, video, f"{video}_{idx:04d}.jpg")
                        for idx in range(start_idx - 5, end_idx + 1)
                    ]
                )

    def __len__(self):
        # return len(self.labels)
        return len(self.game_play_frame)

    def __getitem__(self, idx):
        window = self.num_frames // 2 * self.frame_steps
        # frame = self.frame[idx]

        game_play, frame = self.game_play_frame[idx]
        labels = self.labels[
            (self.labels["game_play"] == game_play) & (self.labels["frame"] == frame)
        ]
        features = self.features[
            (self.features["game_play"] == game_play) & (self.features["frame"] == frame)
        ][USE_COLS].values.astype(np.float32)
        # temporally shift frame
        if self.mode == "train":
            frame = frame + random.randint(-5, 5)
        window_frames = np.arange(frame - window, frame + window + 1, self.frame_steps)

        pairs = labels[["nfl_player_id_1", "nfl_player_id_2"]].values
        imgs = []
        for view in ["Endzone", "Sideline"]:
            video = game_play + f"_{view}.mp4"
            img = []
            for f in range(frame - window, frame + window + 1, self.frame_steps):
                f = min(f, self.video2frames[video])
                img_path = os.path.join(self.frames_folder, video, f"{video}_{f:04d}.jpg")
                img.append(read_image(img_path))
            imgs.append(np.stack(img, -1))
        img_h, img_w = imgs[0].shape[:-1]

        pairs_bboxes = []

        for view in ["Endzone", "Sideline"]:
            video = game_play + f"_{view}.mp4"
            tmp = self.video2helmets[video]
            tmp = tmp.query("@frame-@window<=frame<=@frame+@window")
            players_bboxes = []
            for pair, label in zip(pairs, labels.contact.values):
                players = []
                for p in pair:
                    if p == "G":
                        players.append(p)
                    else:
                        players.append(int(p))
                tmp_players = tmp[tmp.nfl_player_id.isin(players)]
                tmp_frames = tmp_players.frame.values
                # Aggregate 2 players' boxes
                tmp_players = tmp_players.groupby("frame")[
                    ["left", "width", "top", "height"]
                ].mean()
                # Aggregate frames' boxes
                bboxes = []
                tmp_players_frame_dict = {}
                for i, r in tmp_players.iterrows():
                    tmp_players_frame_dict[i] = r.values

                for f in window_frames:
                    if f in tmp_frames:
                        x, w, y, h = tmp_players_frame_dict[f]
                        bboxes.append([x, w, y, h])
                    else:
                        bboxes.append([np.nan, np.nan, np.nan, np.nan])
                bboxes = pd.DataFrame(bboxes).interpolate(limit_direction="both").values
                frame_bbox = bboxes[window_frames == frame][0]
                # To xyxy
                if frame_bbox.sum() > 0:
                    x, w, y, h = frame_bbox
                    frame_bbox = [
                        np.clip(int(x + w / 2) - self.size // 2, 0, img_w),
                        np.clip(int(y + h / 2) - self.size // 2, 0, img_h),
                        np.clip(int(x + w / 2) + self.size // 2, 0, img_w),
                        np.clip(int(y + h / 2) + self.size // 2, 0, img_h),
                        label,
                    ]
                else:
                    frame_bbox = [0, 0, 1, 1, label]

                players_bboxes.append(frame_bbox)
            pairs_bboxes.append(np.array(players_bboxes))

        endzone = self.aug(image=imgs[0], bboxes=pairs_bboxes[0])
        sideline = self.aug(image=imgs[1], bboxes=pairs_bboxes[1])

        return {
            "images0": endzone["image"],
            "boxes0": torch.from_numpy(np.array(endzone["bboxes"])[:, :4]).float(),
            "images1": sideline["image"],
            "boxes1": torch.from_numpy(np.array(sideline["bboxes"])[:, :4]).float(),
            "features": torch.from_numpy(features),
            "labels": torch.from_numpy(labels.contact.values),
        }


def collate_fn(batch):
    images0, boxes0, images1, boxes1, features, labels = [], [], [], [], [], []

    for idx, item in enumerate(batch):
        batch_idx = torch.ones((len(item["boxes0"]), 1)) * idx
        images0.append(item["images0"])
        boxes0.append(torch.cat([item["boxes0"], batch_idx], 1))
        images1.append(item["images1"])
        boxes1.append(torch.cat([item["boxes1"], batch_idx], 1))
        features.append(item["features"])
        labels.append(item["labels"])

    return {
        "images0": torch.stack(images0),
        "boxes0": torch.cat(boxes0),
        "images1": torch.stack(images1),
        "boxes1": torch.cat(boxes1),
        "features": torch.cat(features),
        "labels": torch.cat(labels),
    }


if __name__ == "__main__":
    ds = NFLDataset(
        csv_folder="./data/",
        video_folder="./data/train",
        frames_folder="./data/train_frames",
        mode="train",
    )
