import os
import gc
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import random
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import subprocess
from turbojpeg import TurboJPEG, TJPF_GRAY


USE_COLS = [
    'x_position', 'y_position', 'speed', 'distance',
    'direction', 'orientation', 'acceleration', 'sa'
]


def expand_contact_id(df):
    """
    Splits out contact_id into seperate columns.
    """
    df["game_play"] = df["contact_id"].str[:12]
    df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
    df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
    df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
    return df


def create_features(df, tr_tracking, merge_col="step", use_cols=["x_position", "y_position"]):
    output_cols = []
    df_combo = (
        df.astype({"nfl_player_id_1": "str"})
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id",] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={c: c+"_1" for c in use_cols})
        .drop("nfl_player_id", axis=1)
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id"] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .drop("nfl_player_id", axis=1)
        .rename(columns={c: c+"_2" for c in use_cols})
        .sort_values(["game_play", merge_col, "nfl_player_id_1", "nfl_player_id_2"])
        .reset_index(drop=True)
    )
    output_cols += [c+"_1" for c in use_cols]
    output_cols += [c+"_2" for c in use_cols]
    
    if ("x_position" in use_cols) & ("y_position" in use_cols):
        index = df_combo['x_position_2'].notnull()
        
        distance_arr = np.full(len(index), np.nan)
        tmp_distance_arr = np.sqrt(
            np.square(df_combo.loc[index, "x_position_1"] - df_combo.loc[index, "x_position_2"])
            + np.square(df_combo.loc[index, "y_position_1"]- df_combo.loc[index, "y_position_2"])
        )
        
        distance_arr[index] = tmp_distance_arr
        df_combo['distance'] = distance_arr
        output_cols += ["distance"]
        
    df_combo['G_flug'] = (df_combo['nfl_player_id_2']=="G")
    output_cols += ["G_flug"]
    return df_combo, output_cols


def run_video_cmd(cmd):
    print(cmd)
    if 'Endzone2' not in cmd:
        subprocess.run(cmd, shell=True)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h.astype(np.float32)


class NFLDataset(Dataset):
    def __init__(self, csv_folder, video_folder, frames_folder, mode='train', cache=True, fold=0, size=256, num_frames=13, frame_steps=4, 
        img_height=720, img_width=1280, use_heatmap=False, heatmap_sigma=128
    ):
        self.csv_folder = csv_folder
        self.video_folder = video_folder
        self.frames_folder = frames_folder
        self.mode = mode
        self.cache = cache
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

        self.train_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.Normalize(mean=[0.], std=[1.]),
            ToTensorV2()
        ])

        self.valid_aug = A.Compose([
            A.Normalize(mean=[0.], std=[1.]),
            ToTensorV2()
        ])

    def preprocess_video(self):
        from multiprocessing import Pool

        cmds = []
        for video in tqdm(self.helmets.video.unique()):
            if not os.path.exists(os.path.join(self.frames_folder, video)):
                os.makedirs(os.path.join(self.frames_folder, video), exist_ok=True)
            cmds.append(f"ffmpeg -i {os.path.join(self.video_folder, video)} -q:v 2 -f image2 {os.path.join(self.frames_folder, video, video)}_%04d.jpg -hide_banner -loglevel error")
            
        with Pool(32) as p:
            print(p.map(run_video_cmd, cmds))

        # for video in tqdm(self.helmets.video.unique()):
        #     if 'Endzone2' not in video:
        #         subprocess.run(f"ffmpeg -i {os.path.join(self.video_folder, video)} -q:v 2 -f image2 {os.path.join(self.frames_folder, video)}_%04d.jpg -hide_banner -loglevel error", shell=True)

    def preprocess_csv(self):
        # split using fold in data
        labels_df = pd.read_csv(os.path.join(self.csv_folder, "train_labels.csv"))
        self.tracking = pd.read_csv(os.path.join(self.csv_folder, "train_player_tracking.csv"))
        self.helmets = pd.read_csv(os.path.join(self.csv_folder, "train_baseline_helmets.csv"))
        self.video_metadata = pd.read_csv(os.path.join(self.csv_folder, "train_video_metadata.csv"))

        if self.mode == "train":
            labels_df = labels_df[labels_df["fold"] != self.fold]
            self.labels = expand_contact_id(labels_df)
            self.tracking = self.tracking[self.tracking["fold"] != self.fold]
            self.helmets = self.helmets[self.helmets["fold"] != self.fold]
            self.video_metadata = self.video_metadata[self.video_metadata["fold"] != self.fold]
        elif self.mode == "val":
            labels_df = labels_df[labels_df["fold"] == self.fold]
            self.labels = expand_contact_id(labels_df)
            self.tracking = self.tracking[self.tracking["fold"] == self.fold]
            self.helmets = self.helmets[self.helmets["fold"] == self.fold]
            self.video_metadata = self.video_metadata[self.video_metadata["fold"] == self.fold]
        else:
            raise ValueError("Mode has to be in ['train', 'val']")

        print("Creating features...")
        df, feature_cols = create_features(self.labels, self.tracking, use_cols=USE_COLS)
        df = df.query('not distance>2').reset_index(drop=True)
        df['frame'] = (df['step']/10*59.94+5*59.94).astype('int') + 1

        self.df = df

        self.video2helmets = {}
        helmets_new = self.helmets.set_index('video')
        for video in tqdm(self.helmets.video.unique()):
            self.video2helmets[video] = helmets_new.loc[video].reset_index(drop=True)

        # TODO: add Helmet track Features
        CLUSTERS = [10, 50, 100, 500]

        def add_step_pct(df, cluster):
            df['step_pct'] = cluster * (df['step']-min(df['step']))/(max(df['step'])-min(df['step']))
            df['step_pct'] = df['step_pct'].apply(np.ceil).astype(np.int32)
            return df

        for cluster in CLUSTERS:
            self.df = self.df.groupby('game_play').apply(lambda x:add_step_pct(x, cluster))

            for helmet_view in ['Sideline', 'Endzone']:
                helmets = self.helmets.copy(deep=True)
                helmets.loc[helmets['view']=='Endzone2','view'] = 'Endzone'

                helmets.rename(columns = {'frame': 'step'}, inplace = True)
                helmets = helmets.groupby('game_play').apply(lambda x:add_step_pct(x, cluster))
                helmets = helmets[self.helmets['view']==helmet_view]

                helmets['helmet_id'] = helmets['game_play'] + '_' + helmets['nfl_player_id'].astype(str) + '_' + helmets['step_pct'].astype(str)

                helmets = helmets[['helmet_id', 'left', 'width', 'top', 'height']].groupby('helmet_id').mean().reset_index()
                for player_ind in [1, 2]:
                    self.df['helmet_id'] = self.df['game_play'] + '_' + self.df['nfl_player_id_'+str(player_ind)].astype(str) + \
                                            '_' + self.df['step_pct'].astype(str)

                    self.df = self.df.merge(helmets, how = 'left')
                    self.df.rename(columns = {i:i+'_'+helmet_view+'_'+str(cluster)+'_'+str(player_ind) for i in ['left', 'width', 'top', 'height']}, inplace = True)

                    del self.df['helmet_id']
                    gc.collect()

                    feature_cols += [i+'_'+helmet_view+'_'+str(cluster)+'_'+str(player_ind) for i in ['left', 'width', 'top', 'height']]
                del helmets
                gc.collect()

        cols = [i[:-2] for i in self.df.columns if i[-2:]=='_1' and i!='nfl_player_id_1']
        self.df[[i+'_diff' for i in cols]] = np.abs(self.df[[i+'_1' for i in cols]].values - self.df[[i+'_2' for i in cols]].values)
        feature_cols += [i+'_diff' for i in cols]

        cols = USE_COLS
        self.df[[i+'_prod' for i in cols]] = np.abs(self.df[[i+'_1' for i in cols]].values - self.df[[i+'_2' for i in cols]].values)
        feature_cols += [i+'_prod' for i in cols]

        self.frame = self.df.frame.values
        self.feature = self.df[feature_cols].fillna(-1).values
        self.players = self.df[['nfl_player_id_1','nfl_player_id_2']].values
        self.game_play = self.df.game_play.values

        print("Number of features", len(feature_cols))

        print("Extracting frames from video")
        if self.cache:
            print("Use cached frames from", self.frames_folder)
            if len(os.listdir(self.frames_folder)) == 0:
                print("Not found existing frames, extracting again...")
                self.preprocess_video()
        else:
            print("Extracting frames from scratch")
            self.preprocess_video()

        print("Mapping videos to frames")
        self.video2frames = {}
        for game_play in tqdm(self.video_metadata.game_play.unique()):
            for view in ['Endzone', 'Sideline']:
                video = game_play + f'_{view}.mp4'
                self.video2frames[video] = max(list(map(lambda x:int(x.split('_')[-1].split('.')[0]), glob.glob(os.path.join(os.path.abspath(self.frames_folder), video, f'{video}*')))))
            
    def __len__(self):
        return len(self.df)
    
    # @lru_cache(1024)
    # def read_img(self, path):
    #     return cv2.imread(path, 0)
   
    def __getitem__(self, idx):   
        window = self.num_frames // 2 * self.frame_steps
        frame = self.frame[idx]
        
        if self.mode == 'train':  # TODO: what is this?
            frame = frame + random.randint(-6, 6)

        players = []
        for p in self.players[idx]:
            if p == 'G':
                players.append(p)
            else:
                players.append(int(p))
        
        imgs = []

        for view in ['Endzone', 'Sideline']:
            video = self.game_play[idx] + f'_{view}.mp4'

            tmp = self.video2helmets[video]
            tmp = tmp.query('@frame-@window<=frame<=@frame+@window')
            tmp = tmp[tmp.nfl_player_id.isin(players)]#.sort_values(['nfl_player_id', 'frame'])
            tmp_frames = tmp.frame.values
            tmp = tmp.groupby('frame')[['left','width','top','height']].mean()

            bboxes = []
            for f in range(frame-window, frame+window+1, 1):
                if f in tmp_frames:
                    x, w, y, h = tmp.loc[f][['left','width','top','height']]
                    bboxes.append([x, w, y, h])
                else:
                    bboxes.append([np.nan, np.nan, np.nan, np.nan])
            bboxes = pd.DataFrame(bboxes).interpolate(limit_direction='both').values
            bboxes = bboxes[::self.frame_steps]

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
                    with open(os.path.join(self.frames_folder, video, f'{video}_{f:04d}.jpg'), 'rb') as in_file:
                        img = self.turbo_jpeg.decode(in_file.read(), pixel_format=TJPF_GRAY)[:, : , 0]

                    if img.shape[0] != self.img_height and img.shape[1] != self.img_width:
                        img = cv2.resize(img, (self.img_width, self.img_height))

                    x, w, y, h = bboxes[i]


                    if self.use_heatmap:
                        # using heatmap
                        img_new = img * self.heatmap
                        # if is rgb: img_new = img * np.stack((self.heatmap,)*3, axis=-1)
                    else:
                        # using crop
                        img = img[int(y+h/2) - self.size // 2:int(y+h/2)+self.size // 2,int(x+w/2)-self.size // 2:int(x+w/2)+self.size // 2].copy()
                        img_new[:img.shape[0], :img.shape[1]] = img

                imgs.append(img_new)

        feature = np.float32(self.feature[idx])

        img = np.array(imgs).transpose(1, 2, 0)
        if self.mode == "train":
            img = self.train_aug(image=img)["image"]
        else:
            img = self.valid_aug(image=img)["image"]

        label = np.float32(self.df.contact.values[idx])

        return {
            "images": img,
            "features": feature,
            "labels": label
        }


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
        mode="train"
    )