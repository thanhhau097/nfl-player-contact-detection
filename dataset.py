import os
import gc
import glob
import random
import subprocess
from multiprocessing import Pool

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
import subprocess
from turbojpeg import TurboJPEG, TJPF_GRAY
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

# from generate_features import USE_COLS
USE_COLS = ['x_position_1', 'y_position_1', 'speed_1', 'distance_1', 'direction_1', 'orientation_1', 'acceleration_1', 'sa_1', 'x_position_2', 'y_position_2', 'speed_2', 'distance_2', 'direction_2', 'orientation_2', 'acceleration_2', 'sa_2', 'distance', 'G_flug', 'left_Sideline_10_1', 'width_Sideline_10_1', 'top_Sideline_10_1', 'height_Sideline_10_1', 'left_Sideline_10_2', 'width_Sideline_10_2', 'top_Sideline_10_2', 'height_Sideline_10_2', 'left_Endzone_10_1', 'width_Endzone_10_1', 'top_Endzone_10_1', 'height_Endzone_10_1', 'left_Endzone_10_2', 'width_Endzone_10_2', 'top_Endzone_10_2', 'height_Endzone_10_2', 'left_Sideline_50_1', 'width_Sideline_50_1', 'top_Sideline_50_1', 'height_Sideline_50_1', 'left_Sideline_50_2', 'width_Sideline_50_2', 'top_Sideline_50_2', 'height_Sideline_50_2', 'left_Endzone_50_1', 'width_Endzone_50_1', 'top_Endzone_50_1', 'height_Endzone_50_1', 'left_Endzone_50_2', 'width_Endzone_50_2', 'top_Endzone_50_2', 'height_Endzone_50_2', 'left_Sideline_100_1', 'width_Sideline_100_1', 'top_Sideline_100_1', 'height_Sideline_100_1', 'left_Sideline_100_2', 'width_Sideline_100_2', 'top_Sideline_100_2', 'height_Sideline_100_2', 'left_Endzone_100_1', 'width_Endzone_100_1', 'top_Endzone_100_1', 'height_Endzone_100_1', 'left_Endzone_100_2', 'width_Endzone_100_2', 'top_Endzone_100_2', 'height_Endzone_100_2', 'left_Sideline_500_1', 'width_Sideline_500_1', 'top_Sideline_500_1', 'height_Sideline_500_1', 'left_Sideline_500_2', 'width_Sideline_500_2', 'top_Sideline_500_2', 'height_Sideline_500_2', 'left_Endzone_500_1', 'width_Endzone_500_1', 'top_Endzone_500_1', 'height_Endzone_500_1', 'left_Endzone_500_2', 'width_Endzone_500_2', 'top_Endzone_500_2', 'height_Endzone_500_2', 'x_position_diff', 'y_position_diff', 'speed_diff', 'distance_diff', 'direction_diff', 'orientation_diff', 'acceleration_diff', 'sa_diff', 'left_Sideline_10_diff', 'width_Sideline_10_diff', 'top_Sideline_10_diff', 'height_Sideline_10_diff', 'left_Endzone_10_diff', 'width_Endzone_10_diff', 'top_Endzone_10_diff', 'height_Endzone_10_diff', 'left_Sideline_50_diff', 'width_Sideline_50_diff', 'top_Sideline_50_diff', 'height_Sideline_50_diff', 'left_Endzone_50_diff', 'width_Endzone_50_diff', 'top_Endzone_50_diff', 'height_Endzone_50_diff', 'left_Sideline_100_diff', 'width_Sideline_100_diff', 'top_Sideline_100_diff', 'height_Sideline_100_diff', 'left_Endzone_100_diff', 'width_Endzone_100_diff', 'top_Endzone_100_diff', 'height_Endzone_100_diff', 'left_Sideline_500_diff', 'width_Sideline_500_diff', 'top_Sideline_500_diff', 'height_Sideline_500_diff', 'left_Endzone_500_diff', 'width_Endzone_500_diff', 'top_Endzone_500_diff', 'height_Endzone_500_diff', 'x_position_prod', 'y_position_prod', 'speed_prod', 'distance_prod', 'direction_prod', 'orientation_prod', 'acceleration_prod', 'sa_prod']


def run_video_cmd(cmd):
    print(cmd)
    if "Endzone2" not in cmd:
        subprocess.run(cmd, shell=True)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h.astype(np.float32)


def read_image(path: str):
    return path, np.array(Image.open(path).convert("L"))


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
        heatmap_sigma=128
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
        self.turbo_jpeg = TurboJPEG()

        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder, exist_ok=True)

        print("Processing CSV data")
        self.preprocess_csv()
        self.paths2images = {}

        self.train_aug = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                A.Normalize(mean=[0.0], std=[1.0]),
                ToTensorV2(),
            ]
        )

        self.valid_aug = A.Compose([A.Normalize(mean=[0.0], std=[1.0]), ToTensorV2()])

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
        self.frame = self.labels["frame"].values
        # feature_cols = [c + "_1" for c in USE_COLS]
        # feature_cols += [c + "_2" for c in USE_COLS]
        # feature_cols += ["distance"]
        # feature_cols += ["G_flug"]
        feature_cols = USE_COLS
        self.feature = self.labels[feature_cols].fillna(-1).values
        self.players = self.labels[["nfl_player_id_1", "nfl_player_id_2"]].values
        self.game_play = self.labels.game_play.values

        if len(os.listdir(self.frames_folder)) == 0:
            print("Extracting frames from scratch ...")
            self.preprocess_video()

        self.helmets = self.helmets.set_index("video")

        print("Mapping videos to frames")
        video_start_end = (
            self.labels.groupby(["game_play"])
            .agg({"frame": ["min", "max"]})
            .reset_index()
            .values
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
                        os.path.join(
                            self.frames_folder, video, f"{video}_{idx:04d}.jpg"
                        )
                        for idx in range(start_idx - 5, end_idx + 1)
                    ]
                )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        window = self.num_frames // 2 * self.frame_steps
        frame = self.frame[idx]

        if self.mode == "train":
            frame = frame + random.randint(-5, 5)

        players = []
        for p in self.players[idx]:
            if p == "G":
                players.append(p)
            else:
                players.append(int(p))

        imgs = []
        for view in ["Endzone", "Sideline"]:
            video = self.game_play[idx] + f"_{view}.mp4"

            # tmp = self.video2helmets[video]
            tmp = self.helmets.loc[video].reset_index()
            tmp = tmp.query("@frame-@window<=frame<=@frame+@window")
            tmp = tmp[
                tmp.nfl_player_id.isin(players)
            ]  # .sort_values(['nfl_player_id', 'frame'])
            tmp_frames = tmp.frame.values
            tmp = tmp.groupby("frame")[["left", "width", "top", "height"]].mean()

            bboxes = []
            for f in range(frame - window, frame + window + 1, 1):
                if f in tmp_frames:
                    x, w, y, h = tmp.loc[f][["left", "width", "top", "height"]]
                    bboxes.append([x, w, y, h])
                else:
                    bboxes.append([np.nan, np.nan, np.nan, np.nan])
            bboxes = pd.DataFrame(bboxes).interpolate(limit_direction="both").values
            bboxes = bboxes[:: self.frame_steps]

            if bboxes.sum() > 0:
                flag = 1
            else:
                flag = 0
            
            for i, f in enumerate(range(frame-window, frame+window+1, self.frame_steps)):
                if self.use_heatmap:
                    # using heatmap
                    img_new = np.zeros((self.img_height, self.img_width), dtype=np.float32)
                else:
                    # using crop
                    img_new = np.zeros((self.size, self.size), dtype=np.float32)

                if flag == 1 and f <= self.video2frames[video]:
                    # img = cv2.imread(os.path.join(self.frames_folder, video, f'{video}_{f:04d}.jpg'), 0)
                    img = self.paths2images.get(os.path.join(self.frames_folder, video, f'{video}_{f:04d}.jpg'))

                    if img is None:
                        with open(os.path.join(self.frames_folder, video, f'{video}_{f:04d}.jpg'), 'rb') as in_file:
                            img = self.turbo_jpeg.decode(in_file.read(), pixel_format=TJPF_GRAY)[:, : , 0]

                        if img.shape[0] != self.img_height and img.shape[1] != self.img_width:
                            img = cv2.resize(img, (self.img_width, self.img_height))

                    x, w, y, h = bboxes[i]
                    img = img[
                        int(y + h / 2)
                        - self.size // 2 : int(y + h / 2)
                        + self.size // 2,
                        int(x + w / 2)
                        - self.size // 2 : int(x + w / 2)
                        + self.size // 2,
                    ]
                    img_new[: img.shape[0], : img.shape[1]] = img


                    if self.use_heatmap:
                        # using heatmap
                        img_new = img * self.heatmap
                        # if is rgb: img_new = img * np.stack((self.heatmap,)*3, axis=-1)
                    else:
                        # using crop
                        img = img[int(y+h/2) - self.size // 2:int(y+h/2)+self.size // 2,int(x+w/2)-self.size // 2:int(x+w/2)+self.size // 2]
                        img_new[:img.shape[0], :img.shape[1]] = img

                imgs.append(img_new)

        feature = np.float32(self.feature[idx])

        img = np.array(imgs).transpose(1, 2, 0)
        if self.mode == "train":
            img = self.train_aug(image=img)["image"]
        else:
            img = self.valid_aug(image=img)["image"]

        label = np.float32(self.labels.contact.values[idx])

        return {"images": img, "features": feature, "labels": label}


def collate_fn(batch):
    images, labels, features = [], [], []

    for f in batch:
        images.append(f["images"])
        features.append(f["features"])
        labels.append(f["labels"])

    images = torch.stack(images)
    features = torch.as_tensor(np.array(features))
    labels = torch.as_tensor(np.array(labels))

    batch = {
        "images": images,
        "features": features,
        "labels": labels,
    }
    return batch


if __name__ == "__main__":
    ds = NFLDataset(
        csv_folder="./data/",
        video_folder="./data/train",
        frames_folder="./data/train_frames",
        mode="train",
    )
