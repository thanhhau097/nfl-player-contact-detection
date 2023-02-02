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
from generate_features import USE_COLS

turbo_jpeg = TurboJPEG()


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
                # A.Resize(size, size),
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
            video_folder = os.path.join(self.frames_folder, video)
            if not os.path.exists(video_folder):
                os.makedirs(os.path.join(self.frames_folder, video), exist_ok=True)
            # "-q:v 2 -vf format=gray"
            if len(os.listdir(video_folder)) == 0:
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
        feature_cols = [c + "_1" for c in USE_COLS]
        feature_cols += [c + "_2" for c in USE_COLS]
        feature_cols += ["distance"]
        feature_cols += ["G_flug"]
        # feature_cols = USE_COLS
        self.feature = self.labels[feature_cols].fillna(-1).values.astype(np.float32)
        self.players = self.labels[["nfl_player_id_1", "nfl_player_id_2"]].values
        self.game_play = self.labels.game_play.values

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
        # paths2image_file = f"_{self.mode}_paths2image.pth"
        # # paths2image_file = f"_{self.mode}_origin_paths2image.pth"
        # if os.path.isfile(paths2image_file):
        #     print(f"Load paths2image from {paths2image_file}")
        #     self.paths2image = torch.load(paths2image_file)
        # else:
        #     self.paths2image = {}
        #     for idx in tqdm(range(len(self.labels))):
        #         window = self.num_frames // 2 * self.frame_steps
        #         frame = self.frame[idx]
        #         for view in ["Endzone", "Sideline"]:
        #             for i, f in enumerate(
        #                 range(frame - window - 5, frame + window + 1 + 5)
        #             ): 
        #                 video = self.game_play[idx] + f"_{view}.mp4"
        #                 path = os.path.join(self.frames_folder, video, f"{video}_{max(0, min(f, self.video2frames[video])):04d}.jpg")
        #                 if path not in self.paths2image:
        #                     img_org = read_image(path)
        #                     self.paths2image[path] = img_org
        #                     # cv2.imwrite(path + '_1013x1800', cv2.resize(img_org, (1800, 1013)))
        #     print(f"Save paths2image to {paths2image_file}")
            # torch.save(self.paths2image, paths2image_file)
            # print(f"Saved!")

        # self.paths2image = {}
        # for idx in tqdm(range(len(self.labels))):
        #     window = self.num_frames // 2 * self.frame_steps
        #     frame = self.frame[idx]
        #     for view in ["Sideline"]:
        #         for i, f in enumerate(
        #             range(frame - window - 5, frame + window + 1 + 5)
        #         ): 
        #             video = self.game_play[idx] + f"_{view}.mp4"
        #             path = os.path.join(self.frames_folder, video, f"{video}_{max(0, min(f, self.video2frames[video])):04d}.jpg")
        #             if path not in self.paths2image:
        #                 self.paths2image[path] = 1
        #                 img_org = read_image(path)
        #                 cv2.imwrite(path + '_1080x1920', cv2.resize(img_org, (1920, 1080)))

    
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
        # import time
        for view in ["Endzone", "Sideline"]:
            # start = time.time()
            video = self.game_play[idx] + f"_{view}.mp4"
            # tmp = self.helmets.loc[video].reset_index()
            tmp = self.video2helmets[video]
            # print("self.helmets.loc time", time.time() - start)
            # start = time.time()
            tmp = tmp[tmp['frame'].between(frame-window, frame+window)] # tmp.query("@frame-@window<=frame<=@frame+@window")
            tmp = tmp[
                tmp.nfl_player_id.isin(players)
            ]  # .sort_values(['nfl_player_id', 'frame'])
            tmp_0 = tmp.copy()
            tmp_1 = tmp.copy()
            tmp_frames = tmp.frame.values
            tmp = tmp.groupby("frame")[["left", "width", "top", "height"]].mean()

            # Helmet heatmap
            if "G" in players:
                if players[0] == "G":
                    tmp_0 = tmp_0[tmp_0.nfl_player_id.isin([players[1]])] \
                            .groupby("frame")[
                                ["left", "width", "top", "height", "x1_n", "y1_n", "x2_n", "y2_n"]
                                # ["left", "width", "top", "height"]
                                # ["x1_n", "y1_n", "x2_n", "y2_n"]
                            ] \
                            .mean()
                else:
                    tmp_0 = tmp_0[tmp_0.nfl_player_id.isin([players[0]])] \
                            .groupby("frame")[
                                ["left", "width", "top", "height", "x1_n", "y1_n", "x2_n", "y2_n"]
                                # ["left", "width", "top", "height"]
                                # ["x1_n", "y1_n", "x2_n", "y2_n"]
                            ] \
                            .mean()
            else:
                tmp_0 = tmp_0[tmp_0.nfl_player_id.isin([players[0]])] \
                        .groupby("frame")[
                            ["left", "width", "top", "height", "x1_n", "y1_n", "x2_n", "y2_n"]
                            # ["left", "width", "top", "height"]
                            # ["x1_n", "y1_n", "x2_n", "y2_n"]
                        ] \
                        .mean()
                tmp_1 = tmp_1[tmp_1.nfl_player_id.isin([players[1]])] \
                        .groupby("frame")[
                            ["left", "width", "top", "height", "x1_n", "y1_n", "x2_n", "y2_n"]
                            # ["left", "width", "top", "height"]
                            # ["x1_n", "y1_n", "x2_n", "y2_n"]
                        ] \
                        .mean()

            bboxes = []
            tmp_players_frame_dict = {}
            for i, r in tmp.iterrows():
                tmp_players_frame_dict[i] = r.values

            bboxes_0 = []
            tmp_players_frame_dict_0 = {}
            for i, r in tmp_0.iterrows():
                tmp_players_frame_dict_0[i] = r.values  
            # tmp_players_frame_dict_0_align = {}
            # for i, r in tmp.iterrows():
            #     if i in tmp_players_frame_dict_0:
            #         tmp_players_frame_dict_0_align[i] = tmp_players_frame_dict_0[i]
            if "G" not in players:
                bboxes_1 = []
                tmp_players_frame_dict_1 = {}
                for i, r in tmp_1.iterrows():
                    tmp_players_frame_dict_1[i] = r.values
                # tmp_players_frame_dict_1_align = {}
                # for i, r in tmp.iterrows():
                #     if i in tmp_players_frame_dict_1:
                #         tmp_players_frame_dict_1_align[i] = tmp_players_frame_dict_1[i]

            for f in range(frame - window, frame + window + 1, 1):
                if f in tmp_frames:
                    x, w, y, h = tmp_players_frame_dict[f]
                    bboxes.append([x, w, y, h])
                else:
                    bboxes.append([np.nan, np.nan, np.nan, np.nan])
                if f in tmp_players_frame_dict_0:
                    x0, w0, y0, h0, yl0_x1, yl0_y1, yl0_x2, yl0_y2 = tmp_players_frame_dict_0[f]
                    bboxes_0.append([x0, w0, y0, h0, yl0_x1, yl0_y1, yl0_x2, yl0_y2])
                    # x0, w0, y0, h0 = tmp_players_frame_dict_0[f]
                    # bboxes_0.append([x0, w0, y0, h0])
                else:
                    bboxes_0.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                if "G" not in players:
                    if f in tmp_players_frame_dict_1:
                        x1, w1, y1, h1, yl1_x1, yl1_y1, yl1_x2, yl1_y2 = tmp_players_frame_dict_1[f]
                        bboxes_1.append([x1, w1, y1, h1, yl1_x1, yl1_y1, yl1_x2, yl1_y2])
                        # x1, w1, y1, h1 = tmp_players_frame_dict_1[f]
                        # bboxes_1.append([x1, w1, y1, h1])
                    else:
                        bboxes_1.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

            bboxes = pd.DataFrame(bboxes).interpolate(limit_direction="both").values
            bboxes = bboxes[:: self.frame_steps]
            bboxes_0 = pd.DataFrame(bboxes_0).interpolate(limit_direction="both").values
            bboxes_0 = bboxes_0[:: self.frame_steps]

            if "G" not in players:
                bboxes_1 = pd.DataFrame(bboxes_1).interpolate(limit_direction="both").values
                bboxes_1 = bboxes_1[:: self.frame_steps]

            if bboxes.sum() > 0:
                flag = 1
            else:
                flag = 0

            for i, f in enumerate(
                range(frame - window, frame + window + 1, self.frame_steps)
            ):
                if (i+1)%4 == 0:
                    continue

                path = os.path.join(self.frames_folder, video, f"{video}_{max(0, min(f, self.video2frames[video])):04d}.jpg")
                img_new = np.zeros((self.size, self.size), dtype=np.float32)
                mask_new = np.zeros((self.size, self.size), dtype=np.float32)
                # maskyl_new = np.zeros((self.size, self.size), dtype=np.float32)
                if flag != 0:
                    # img_new = np.zeros((self.size, self.size), dtype=np.float32)

                    # img = self.paths2image[path]
                    img = read_image(path)
                    # img = read_image(path + '_1080x1920')
                    x, w, y, h = bboxes[i]
                    # x = x*1.5
                    # w = w*1.5
                    # y = y*1.5
                    # h = h*1.5

                    mask = np.zeros_like(img)
                    # maskyl = np.zeros_like(img)
                    x0, w0, y0, h0, yl0_x1, yl0_y1, yl0_x2, yl0_y2 = bboxes_0[i]
                    # x0, w0, y0, h0 = bboxes_0[i]
                    # x0 = x0*1.5
                    # w0 = w0*1.5
                    # y0 = y0*1.5
                    # h0 = h0*1.5
                    try:
                        mask[int(y0):int(y0+h0), int(x0):int(x0+w0)] = 1
                        # maskyl[max(0, int(yl0_y1)):max(0, int(yl0_y2)), max(0, int(yl0_x1)):max(0, int(yl0_x2))] = 1
                    except:
                        a = 1

                    if "G" not in players:
                        x1, w1, y1, h1, yl1_x1, yl1_y1, yl1_x2, yl1_y2 = bboxes_1[i]
                        # x1, w1, y1, h1 = bboxes_1[i]
                        # x1 = x1*1.5
                        # w1 = w1*1.5
                        # y1 = y1*1.5
                        # h1 = h1*1.5
                        try:
                            mask[int(y1):int(y1+h1), int(x1):int(x1+w1)] = 1
                            # maskyl[max(0, int(yl1_y1)):max(0, int(yl1_y2)), max(0, int(yl1_x1)):max(0, int(yl1_x2))] = 1
                        except:
                            a = 1
                                        

                    # using crop
                    img = img[
                        int(y + h / 2)
                        - self.size // 2 : int(y + h / 2)
                        + self.size // 2,
                        int(x + w / 2)
                        - self.size // 2 : int(x + w / 2)
                        + self.size // 2,
                    ]
                    mask = mask[
                        int(y + h / 2)
                        - self.size // 2 : int(y + h / 2)
                        + self.size // 2,
                        int(x + w / 2)
                        - self.size // 2 : int(x + w / 2)
                        + self.size // 2,
                    ]
                    # maskyl = maskyl[
                    #     int(y + h / 2)
                    #     - self.size // 2 : int(y + h / 2)
                    #     + self.size // 2,
                    #     int(x + w / 2)
                    #     - self.size // 2 : int(x + w / 2)
                    #     + self.size // 2,
                    # ]
                    img_new[: img.shape[0], : img.shape[1]] = img
                    mask_new[: mask.shape[0], : mask.shape[1]] = mask
                    # maskyl_new[: maskyl.shape[0], : maskyl.shape[1]] = maskyl
                imgs.append(img_new)
                imgs.append(img_new * mask_new)
                # imgs.append(img_new * maskyl_new)
            
            # print("image reading time", time.time() - start)
            # print("------------------------")

        # start = time.time()
        feature = torch.from_numpy(self.feature[idx])

        img = np.array(imgs).transpose(1, 2, 0).astype(np.uint8)
        if self.mode == "train":
            img = self.train_aug(image=img)["image"]
        else:
            img = self.valid_aug(image=img)["image"]
        # img = self.valid_aug(image=img)["image"]

        label = self.labels.contact.values[idx]
        contact_id = self.labels.contact_id.values[idx]
        # print("augmentation time", time.time() - start)
        # print("--------------------------------------------------------------------------------")
        return {"images": img, "features": feature, "labels": label, "contact_ids": contact_id}


def collate_fn(batch):
    images, labels, features, contact_ids = [], [], [], []

    for f in batch:
        images.append(f["images"])
        features.append(f["features"])
        labels.append(f["labels"])
        contact_ids.append(f["contact_ids"])

    images = torch.stack(images)
    features = torch.stack(features)
    labels = torch.as_tensor(labels)

    batch = {
        "images": images,
        "features": features,
        "labels": labels,
        "contact_ids": contact_ids,
    }
    return batch


if __name__ == "__main__":
    ds = NFLDataset(
        csv_folder="./data/",
        video_folder="./data/train",
        frames_folder="./data/train_frames",
        mode="train",
    )
