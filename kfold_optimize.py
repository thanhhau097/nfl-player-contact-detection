import os
import torch
from typing import List
import torch

class Config:
    AUTHOR = "colum2131"

    NAME = "NFLC-" + "old-simple-xgb-baseline"

    COMPETITION = "nfl-player-contact-detection"

    seed = 42
    num_fold = 5
    
    # xgb_params = {
    #     'objective': 'binary:logistic',
    #     'eval_metric': 'auc',
    #     'learning_rate':0.03,
    #     'tree_method':'hist' if not torch.cuda.is_available() else 'gpu_hist'
    # }
    xgb_params = {
        "n_estimators": 5000,
        "learning_rate": 1e-2,
        "subsample": 0.7,
        "colsample_bytree": 0.85,
        "objective": "binary:logistic",
        "nthread": os.cpu_count(),
        'eval_metric': 'auc',
        'tree_method':'hist' if not torch.cuda.is_available() else 'gpu_hist'
    }


import os
import gc
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.optimize import minimize
import cv2
from glob import glob
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
)

import xgboost as xgb

import torch

if torch.cuda.is_available():
    import cupy 
    import cudf
    from cuml import ForestInference

def setup(cfg):
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # set dirs
    cfg.INPUT = f'data/'
    cfg.EXP = cfg.NAME
    cfg.OUTPUT_EXP = cfg.NAME
    cfg.SUBMISSION = './'
    cfg.DATASET = 'data/'

    cfg.EXP_MODEL = os.path.join(cfg.EXP, 'model')
    cfg.EXP_FIG = os.path.join(cfg.EXP, 'fig')
    cfg.EXP_PREDS = os.path.join(cfg.EXP, 'preds')

    # make dirs
    for d in [cfg.EXP_MODEL, cfg.EXP_FIG, cfg.EXP_PREDS]:
        os.makedirs(d, exist_ok=True)
        
    return cfg


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

def xgb_metric(preds, dmatrix):
   return "mcc", matthews_corrcoef(dmatrix.get_label(), preds > 0.5)

fit_params = {
    "verbose": 500,
    "early_stopping_rounds": 100
}
# xgboost code
def fit_xgboost(cfg, X, feat_cols, params, add_suffix=''):
    """
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate':0.01,
        'tree_method':'gpu_hist'
    }
    """
    
    oof_pred = np.zeros(len(X), dtype=np.float32)
    for fold in range(5):
        xgb_train = xgb.DMatrix(X[X.fold != fold][feat_cols], label=X[X.fold != fold]['contact'])
        xgb_valid = xgb.DMatrix(X[X.fold == fold][feat_cols], label=X[X.fold == fold]['contact'])
        evals = [(xgb_train,'train'),(xgb_valid,'eval')]
        params["scale_pos_weight"] = (X[X.fold != fold]['contact'] == 0).sum() / (X[X.fold != fold]['contact'] == 1).sum()
        model = xgb.train(
            params,
            xgb_train,
            num_boost_round=10_000,
            early_stopping_rounds=100,
            evals=evals,
            verbose_eval=100,
        )
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X[X.fold != fold][feat_cols].to_numpy(), X[X.fold != fold]['contact'].to_numpy(), 
            eval_set=[(X[X.fold == fold][feat_cols].to_numpy(), X[X.fold == fold]['contact'].to_numpy())], 
            **fit_params
        )

        model_path = os.path.join(cfg.EXP_MODEL, f'xgb_fold{fold}model')
        print(model_path)
        model.save_model(model_path)

        if not torch.cuda.is_available():
            model = xgb.Booster().load_model(model_path)
        else:
            model = ForestInference.load(model_path, output_class=True, model_type='xgboost')
        pred_i = model.predict_proba(X[X.fold == fold][feat_cols].to_numpy())[:, 1]
        oof_pred[X[X.fold == fold].index] = pred_i
        score = round(roc_auc_score(X[X.fold == fold]['contact'], pred_i), 5)
        print(f'Performance of the prediction: {score}\n')
        del model; gc.collect()

    np.save(os.path.join(cfg.EXP_PREDS, f'oof_pred_best{add_suffix}'), oof_pred)
    score = round(roc_auc_score(X['contact'], oof_pred), 5)
    print(f'All Performance of the prediction: {score}')
    return oof_pred

def pred_xgboost(X, feat_cols, add_suffix=''):
    models = glob(f'xgb_fold*{add_suffix}.model')
    if not torch.cuda.is_available():
        models = [xgb.Booster().load_model(model) for model in models]
    else:
        models = [ForestInference.load(model, output_class=True, model_type='xgboost') for model in models]
    preds = np.array([model.predict_proba(X[feat_cols])[:, 1] for model in models])
    preds = np.mean(preds, axis=0)
    return preds

cfg = setup(Config)

feature_cols = ['x_position_1', 'y_position_1', 'speed_1', 'distance_1', 'direction_1', 'orientation_1', 'acceleration_1', 'sa_1', 'x_position_2', 'y_position_2', 'speed_2', 'distance_2', 'direction_2', 'orientation_2', 'acceleration_2', 'sa_2', 'distance', 'G_flug', 'left_Sideline_10_1', 'width_Sideline_10_1', 'top_Sideline_10_1', 'height_Sideline_10_1', 'left_Sideline_10_2', 'width_Sideline_10_2', 'top_Sideline_10_2', 'height_Sideline_10_2', 'left_Endzone_10_1', 'width_Endzone_10_1', 'top_Endzone_10_1', 'height_Endzone_10_1', 'left_Endzone_10_2', 'width_Endzone_10_2', 'top_Endzone_10_2', 'height_Endzone_10_2', 'left_Sideline_50_1', 'width_Sideline_50_1', 'top_Sideline_50_1', 'height_Sideline_50_1', 'left_Sideline_50_2', 'width_Sideline_50_2', 'top_Sideline_50_2', 'height_Sideline_50_2', 'left_Endzone_50_1', 'width_Endzone_50_1', 'top_Endzone_50_1', 'height_Endzone_50_1', 'left_Endzone_50_2', 'width_Endzone_50_2', 'top_Endzone_50_2', 'height_Endzone_50_2', 'left_Sideline_100_1', 'width_Sideline_100_1', 'top_Sideline_100_1', 'height_Sideline_100_1', 'left_Sideline_100_2', 'width_Sideline_100_2', 'top_Sideline_100_2', 'height_Sideline_100_2', 'left_Endzone_100_1', 'width_Endzone_100_1', 'top_Endzone_100_1', 'height_Endzone_100_1', 'left_Endzone_100_2', 'width_Endzone_100_2', 'top_Endzone_100_2', 'height_Endzone_100_2', 'left_Sideline_500_1', 'width_Sideline_500_1', 'top_Sideline_500_1', 'height_Sideline_500_1', 'left_Sideline_500_2', 'width_Sideline_500_2', 'top_Sideline_500_2', 'height_Sideline_500_2', 'left_Endzone_500_1', 'width_Endzone_500_1', 'top_Endzone_500_1', 'height_Endzone_500_1', 'left_Endzone_500_2', 'width_Endzone_500_2', 'top_Endzone_500_2', 'height_Endzone_500_2', 'x_position_diff', 'y_position_diff', 'speed_diff', 'distance_diff', 'direction_diff', 'orientation_diff', 'acceleration_diff', 'sa_diff', 'left_Sideline_10_diff', 'width_Sideline_10_diff', 'top_Sideline_10_diff', 'height_Sideline_10_diff', 'left_Endzone_10_diff', 'width_Endzone_10_diff', 'top_Endzone_10_diff', 'height_Endzone_10_diff', 'left_Sideline_50_diff', 'width_Sideline_50_diff', 'top_Sideline_50_diff', 'height_Sideline_50_diff', 'left_Endzone_50_diff', 'width_Endzone_50_diff', 'top_Endzone_50_diff', 'height_Endzone_50_diff', 'left_Sideline_100_diff', 'width_Sideline_100_diff', 'top_Sideline_100_diff', 'height_Sideline_100_diff', 'left_Endzone_100_diff', 'width_Endzone_100_diff', 'top_Endzone_100_diff', 'height_Endzone_100_diff', 'left_Sideline_500_diff', 'width_Sideline_500_diff', 'top_Sideline_500_diff', 'height_Sideline_500_diff', 'left_Endzone_500_diff', 'width_Endzone_500_diff', 'top_Endzone_500_diff', 'height_Endzone_500_diff', 'x_position_prod', 'y_position_prod', 'speed_prod', 'distance_prod', 'direction_prod', 'orientation_prod', 'acceleration_prod', 'sa_prod']
print("Number of features", len(feature_cols))
print("Columns: ", feature_cols)


if torch.cuda.is_available():
    df= cudf.read_csv("xg_130feats.csv")
    df = df.to_pandas()
else:
    df = pd.read_csv("xg_130feats.csv")

print(f"All data: {len(df)}")

# for nnmodel in ["resnet50", "seresnext50_32x4d", "tf_efficientnetv2_s.in21k_ft_in1k"]:
for nnmodel in ["resnet50"]:
    data_full = {}
    print(f"Loading result of {nnmodel}")
    for kfold in range(5):
        data = torch.load(f"{nnmodel}_prob_fold{kfold}.pth")
        for k, v in data.items():
            data_full[k] = v
    probs = []
    # feats = []
    contact_ids = df.contact_id.to_list()
    for index in range(len(df)):
        contact_id = contact_ids[index]
        probs.append(data_full[contact_id])
        # feats.append(feat_full[contact_id])
    df[f'{nnmodel}_prob'] = probs
    feature_cols.append(f'{nnmodel}_prob')
            
print("Number of features final", len(feature_cols))

def func(x_list):
    score = matthews_corrcoef(df['contact'], oof_pred>x_list[0])
    return -score

oof_pred = df[f'{nnmodel}_prob'].to_list()
x0 = [0.5]
result = minimize(func, x0,  method="nelder-mead")
score = round(matthews_corrcoef(df['contact'], oof_pred>result.x[0]), 5)
best_thresh = round(result.x[0], 5)
print(f"Best MCC: {score} @ threshold", best_thresh)   

