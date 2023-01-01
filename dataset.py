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
import time

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
        self.heatmap = gaussian2D((self.img_height * 2, self.img_width * 2), self.sigma)

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
            if self.use_heatmap:
                cmds.append(
                    f"ffmpeg -i {os.path.join(self.video_folder, video)} -q:v 2 -s {self.img_width}x{self.img_height} -f image2 "
                    f"{os.path.join(self.frames_folder, video, video)}_%04d.jpg -hide_banner "
                    "-loglevel error"
                )
            else:
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
        self.feature = self.labels[feature_cols].fillna(-1).values.astype(np.float32)
        self.players = self.labels[["nfl_player_id_1", "nfl_player_id_2"]].values
        self.game_play = self.labels.game_play.values

        if len(os.listdir(self.frames_folder)) == 0:
            print("Extracting frames from scratch ...")
            self.preprocess_video()

        self.video2helmets = {}
        helmets_new = self.helmets.set_index("video")
        for video in tqdm(self.helmets.video.unique()):
            self.video2helmets[video] = helmets_new.loc[video].reset_index(drop=True)
        
        del helmets_new

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
        
        paths2image_file = f"_{self.mode}_{self.size}_paths2image.pth"
        if os.path.isfile(paths2image_file):
            print(f"Load paths2image from {paths2image_file}")
            self.paths2image = torch.load(paths2image_file)
        else:
            self.paths2image = {}
            for idx in tqdm(range(len(self.labels))):
                window = self.num_frames // 2 * self.frame_steps
                frame = self.frame[idx]
                for view in ["Endzone", "Sideline"]:
                    for i, f in enumerate(
                        range(frame - window, frame + window + 1, self.frame_steps)
                    ): 
                        video = self.game_play[idx] + f"_{view}.mp4"
                        path = os.path.join(self.frames_folder, video, f"{video}_{min(f, self.video2frames[video]):04d}.jpg")
                        if path not in self.paths2image:
                            self.paths2image[path] = cv2.resize(read_image(path), (self.size, self.size))
            print(f"Save paths2image to {paths2image_file}")
            torch.save(self.paths2image, paths2image_file)
            

    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        window = self.num_frames // 2 * self.frame_steps
        frame = self.frame[idx]

        # if self.mode == "train":
        #     frame = frame + random.randint(-5, 5)

        players = []
        for p in self.players[idx]:
            if p == "G":
                players.append(p)
            else:
                players.append(int(p))

        imgs = []
        # import time
        for view in ["Endzone", "Sideline"]:
            start = time.time()
            video = self.game_play[idx] + f"_{view}.mp4"
            for i, f in enumerate(
                range(frame - window, frame + window + 1, self.frame_steps)
            ):  
                path = os.path.join(self.frames_folder, video, f"{video}_{min(f, self.video2frames[video]):04d}.jpg")
                # if path not in self.paths2image:
                #     self.paths2image[path] = cv2.resize(read_image(path), (self.size, self.size))
                img_new = self.paths2image[path]
            # # tmp = self.helmets.loc[video].reset_index()
            # tmp = self.video2helmets[video]
            # # print("self.helmets.loc time", time.time() - start)
            # # start = time.time()
            # tmp = tmp[tmp['frame'].between(frame-window, frame+window)] # tmp.query("@frame-@window<=frame<=@frame+@window")
            # # print("tmp.query time", time.time() - start)
            # # start = time.time()

            # tmp = tmp[
            #     tmp.nfl_player_id.isin(players)
            # ]  # .sort_values(['nfl_player_id', 'frame'])
            # tmp_frames = tmp.frame.values
            # tmp = tmp.groupby("frame")[["left", "width", "top", "height"]].mean()
            # # print("tmp.groupby time", time.time() - start)
            # # start = time.time()

            # bboxes = []
            # for f in range(frame - window, frame + window + 1, 1):
            #     if f in tmp_frames:
            #         x, w, y, h = tmp.loc[f][["left", "width", "top", "height"]]
            #         bboxes.append([x, w, y, h])
            #     else:
            #         bboxes.append([np.nan, np.nan, np.nan, np.nan])
            # bboxes = pd.DataFrame(bboxes).interpolate(limit_direction="both").values
            # bboxes = bboxes[:: self.frame_steps]

            # if bboxes.sum() > 0:
            #     flag = 1
            # else:
            #     flag = 0
            
            # # print("box processing time", time.time() - start)
            # # start = time.time()

            # for i, f in enumerate(
            #     range(frame - window, frame + window + 1, self.frame_steps)
            # ):
            #     if self.use_heatmap:
            #         # using heatmap
            #         img_new = np.zeros(
            #             (self.img_height, self.img_width), dtype=np.float32
            #         )
            #     else:
            #         # using crop
            #         img_new = np.zeros((self.size, self.size), dtype=np.float32)

            #     if flag == 1 and f <= self.video2frames[video]:
            #         img = read_image(
            #             os.path.join(self.frames_folder, video, f"{video}_{f:04d}.jpg")
            #         )
            #         x, w, y, h = bboxes[i]

            #         if self.use_heatmap:
            #             if (img.shape[0] != self.img_height and img.shape[1] != self.img_width):
            #                 img = cv2.resize(img, (self.img_width, self.img_height))

            #             if self.img_height != 720 or self.img_width != 1280: # raw w and h is 1280 and 720
            #                 x = int(x * self.img_width / 1280)
            #                 y = int(y * self.img_height / 720)
            #                 w = int(w * self.img_width / 1280)
            #                 h = int(h * self.img_height / 720)
            #             # using heatmap
            #             # create new heatmap at a specific location
            #             center_x = int(x + w / 2)
            #             center_y = int(y + h / 2)
            #             heatmap = self.heatmap[self.img_height - center_y: 2 * self.img_height - center_y, self.img_width - center_x: 2 * self.img_width - center_x]
            #             img_new = (img * heatmap) # .astype(np.uint8)
            #             # if is rgb: img_new = img * np.stack((self.heatmap,)*3, axis=-1)
            #         else:
            #             # using crop
            #             img = img[
            #                 int(y + h / 2)
            #                 - self.size // 2 : int(y + h / 2)
            #                 + self.size // 2,
            #                 int(x + w / 2)
            #                 - self.size // 2 : int(x + w / 2)
            #                 + self.size // 2,
            #             ]
            #             img_new[: img.shape[0], : img.shape[1]] = img




                imgs.append(img_new)
            # print("image reading time", time.time() - start)
            # print("------------------------")

        # start = time.time()
        feature = torch.from_numpy(self.feature[idx])

        img = np.array(imgs).transpose(1, 2, 0).astype(np.uint8)
        # if self.mode == "train":
        #     img = self.train_aug(image=img)["image"]
        # else:
        #     img = self.valid_aug(image=img)["image"]
        img = self.valid_aug(image=img)["image"]

        label = self.labels.contact.values[idx]
        # print("augmentation time", time.time() - start)
        # print("--------------------------------------------------------------------------------")
        return {"images": img, "features": feature, "labels": label}


def collate_fn(batch):
    images, labels, features = [], [], []

    for f in batch:
        images.append(f["images"])
        features.append(f["features"])
        labels.append(f["labels"])

    images = torch.stack(images)
    features = torch.stack(features)
    labels = torch.as_tensor(labels)

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
