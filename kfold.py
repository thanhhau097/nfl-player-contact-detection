import os
import random

import pandas as pd


DATA_FOLDER = "./data"
kfold_folder = os.path.join(DATA_FOLDER, "kfold")

if not os.path.exists(kfold_folder):
    os.makedirs(kfold_folder, exist_ok=True)


# labels
print("Loading labels")
df = pd.read_csv(os.path.join(DATA_FOLDER, "train_labels.csv"))
df["game_key"] = pd.to_numeric(df["game_play"].str.split("_").str[0])
df["play_id"] = pd.to_numeric(df["game_play"].str.split("_").str[1])


random.seed(42)
game_keys = list(set(df.game_key.values))
random.shuffle(game_keys)

fold_dict = {}
N = 10
for i in range(15):
    keys = game_keys[i * N : (i + 1) * N]
    for k in keys:
        fold_dict[k] = i

df["fold"] = df['game_key'].map(fold_dict)
df.to_csv(os.path.join(kfold_folder, "train_labels.csv"))

# video metadata
print("Loading video metadata")
video_meta_data_df = pd.read_csv(os.path.join(DATA_FOLDER, "train_video_metadata.csv"))
video_meta_data_df["fold"] = video_meta_data_df["game_key"].map(fold_dict)
video_meta_data_df.to_csv(os.path.join(kfold_folder, "train_video_metadata.csv"))

# helmets
print("Loading helmets")
baseline_helmets_df = pd.read_csv(
    os.path.join(DATA_FOLDER, "train_baseline_helmets.csv")
)
baseline_helmets_df["fold"] = baseline_helmets_df["game_key"].map(fold_dict)
baseline_helmets_df.to_csv(os.path.join(kfold_folder, "train_baseline_helmets.csv"))

# player tracking
print("Loading player tracking")
player_tracking_df = pd.read_csv(os.path.join(DATA_FOLDER, "train_player_tracking.csv"))
player_tracking_df["fold"] = player_tracking_df["game_key"].map(fold_dict)
player_tracking_df.to_csv(os.path.join(kfold_folder, "train_player_tracking.csv"))
