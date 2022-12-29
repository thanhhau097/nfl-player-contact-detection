import os
import random
from typing import List

import numpy as np
import pandas as pd

USE_COLS = [
    "x_position",
    "y_position",
    "speed",
    "distance",
    "direction",
    "orientation",
    "acceleration",
    "sa",
]


def expand_contact_id(df: pd.DataFrame):
    """
    Splits out contact_id into seperate columns.
    """
    df["game_play"] = df["contact_id"].str[:12]
    df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
    df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
    df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
    return df


def create_features(
    df: pd.DataFrame,
    tr_tracking: pd.DataFrame,
    merge_col: str = "step",
    use_cols: List[str] = ["x_position", "y_position"],
):
    output_cols = []
    df_combo = (
        df.astype({"nfl_player_id_1": "str"})
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                [
                    "game_play",
                    merge_col,
                    "nfl_player_id",
                ]
                + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={c: c + "_1" for c in use_cols})
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
        .rename(columns={c: c + "_2" for c in use_cols})
        .sort_values(["game_play", merge_col, "nfl_player_id_1", "nfl_player_id_2"])
        .reset_index(drop=True)
    )
    output_cols += [c + "_1" for c in use_cols]
    output_cols += [c + "_2" for c in use_cols]

    if ("x_position" in use_cols) & ("y_position" in use_cols):
        index = df_combo["x_position_2"].notnull()

        distance_arr = np.full(len(index), np.nan)
        tmp_distance_arr = np.sqrt(
            np.square(
                df_combo.loc[index, "x_position_1"]
                - df_combo.loc[index, "x_position_2"]
            )
            + np.square(
                df_combo.loc[index, "y_position_1"]
                - df_combo.loc[index, "y_position_2"]
            )
        )

        distance_arr[index] = tmp_distance_arr
        df_combo["distance"] = distance_arr
        output_cols += ["distance"]

    df_combo["G_flug"] = df_combo["nfl_player_id_2"] == "G"
    output_cols += ["G_flug"]
    return df_combo, output_cols


def main(csv_folder: str):
    labels_df = pd.read_csv(os.path.join(csv_folder, "train_labels.csv"))
    tracking = pd.read_csv(os.path.join(csv_folder, "train_player_tracking.csv"))
    helmets = pd.read_csv(os.path.join(csv_folder, "train_baseline_helmets.csv"))
    # split fold
    labels_df["game_key"] = pd.to_numeric(labels_df["game_play"].str.split("_").str[0])
    random.seed(42)
    game_keys = list(set(labels_df.game_key.values))
    random.shuffle(game_keys)

    fold_dict = {}
    N = 10
    for i in range(15):
        keys = game_keys[i * N : (i + 1) * N]
        for k in keys:
            fold_dict[k] = i

    labels_df["fold"] = labels_df["game_key"].map(fold_dict)
    helmets["fold"] = helmets["game_key"].map(fold_dict)

    print("Creating features...")
    labels_df = expand_contact_id(labels_df)
    df, feature_cols = create_features(labels_df, tracking, use_cols=USE_COLS)
    df = df.query("not distance>2").reset_index(drop=True)
    df["frame"] = (df["step"] / 10 * 59.94 + 5 * 59.94).astype("int") + 1
    df[feature_cols].fillna(-1, inplace=True)

    features_csv_path = os.path.join(csv_folder, "train_features.csv")
    helmets_csv_path = os.path.join(csv_folder, "train_baseline_helmets_kfold.csv")
    df.to_csv(features_csv_path, index=False)
    helmets.to_csv(helmets_csv_path, index=False)
    print("Saved features df and helmets df")


if __name__ == "__main__":
    main("data/")
