import os
import torch
from typing import List
import torch

class Config:
    AUTHOR = "colum2131"

    NAME = "NFLC-" + "Exp001-simple-xgb-baseline"

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
        "subsample": 0.6,
        "colsample_bytree": 0.8,
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
        # xgb_train = xgb.DMatrix(X[X.fold != fold][feat_cols], label=X[X.fold != fold]['contact'])
        # xgb_valid = xgb.DMatrix(X[X.fold == fold][feat_cols], label=X[X.fold == fold]['contact'])
        # evals = [(xgb_train,'train'),(xgb_valid,'eval')]
        # params["scale_pos_weight"] = (X[X.fold != fold]['contact'] == 0).sum() / (X[X.fold != fold]['contact'] == 1).sum()
        # model = xgb.train(
        #     params,
        #     xgb_train,
        #     num_boost_round=10_000,
        #     early_stopping_rounds=100,
        #     evals=evals,
        #     verbose_eval=100,
        # )
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X[X.fold != fold][feat_cols].to_numpy(), X[X.fold != fold]['contact'].to_numpy(), 
            eval_set=[(X[X.fold == fold][feat_cols].to_numpy(), X[X.fold == fold]['contact'].to_numpy())], 
            **fit_params
        )

        model_path = os.path.join(cfg.EXP_MODEL, f'xgb_fold{fold}model')
        model.save_model(model_path)

        if not torch.cuda.is_available():
            model = xgb.Booster().load_model(model_path)
        else:
            model = ForestInference.load(model_path, output_class=True, model_type='xgboost')
    
        pred_i = model.predict_proba(X[X.fold == fold][feat_cols])[:, 1]
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

# if not torch.cuda.is_available():
#     df = pd.read_csv(os.path.join(cfg.INPUT, 'train_features.csv'))
#     helmets = pd.read_csv(os.path.join(cfg.INPUT, 'train_baseline_helmets_kfold.csv'))
    
# else:
#     df = cudf.read_csv(os.path.join(cfg.INPUT, 'train_features.csv'))
#     helmets = cudf.read_csv(os.path.join(cfg.INPUT, 'train_baseline_helmets_kfold.csv'))

# if torch.cuda.is_available():
#     df = df.to_pandas()
#     helmets = helmets.to_pandas()

# feature_cols = ['x_position_1', 'y_position_1', 'speed_1', 'distance_1', 'direction_1', 'orientation_1', 'acceleration_1', 'sa_1', 'x_position_2', 'y_position_2', 'speed_2', 'distance_2', 'direction_2', 'orientation_2', 'acceleration_2', 'sa_2', 'distance', 'G_flug']

# # add Helmet track Features
# CLUSTERS = [10, 50, 100, 500]

# def add_step_pct(df, cluster):
#     df['step_pct'] = cluster * (df['step']-min(df['step']))/(max(df['step'])-min(df['step']))
#     df['step_pct'] = df['step_pct'].apply(np.ceil).astype(np.int32)
#     return df

# for cluster in CLUSTERS:
#     df = df.groupby('game_play').apply(lambda x:add_step_pct(x, cluster))

#     for helmet_view in ['Sideline', 'Endzone']:
#         tmp_helmets = helmets.copy(deep=True)
#         tmp_helmets.loc[tmp_helmets['view']=='Endzone2','view'] = 'Endzone'

#         tmp_helmets.rename(columns = {'frame': 'step'}, inplace = True)
#         tmp_helmets = tmp_helmets.groupby('game_play').apply(lambda x:add_step_pct(x, cluster))
#         tmp_helmets = tmp_helmets[tmp_helmets['view']==helmet_view]

#         tmp_helmets['helmet_id'] = tmp_helmets['game_play'] + '_' + tmp_helmets['nfl_player_id'].astype(str) + '_' + tmp_helmets['step_pct'].astype(str)

#         tmp_helmets = tmp_helmets[['helmet_id', 'left', 'width', 'top', 'height']].groupby('helmet_id').mean().reset_index()
#         for player_ind in [1, 2]:
#             df['helmet_id'] = df['game_play'] + '_' + df['nfl_player_id_'+str(player_ind)].astype(str) + \
#                                     '_' + df['step_pct'].astype(str)

#             df = df.merge(tmp_helmets, how = 'left')
#             df.rename(columns = {i:i+'_'+helmet_view+'_'+str(cluster)+'_'+str(player_ind) for i in ['left', 'width', 'top', 'height']}, inplace = True)

#             del df['helmet_id']

#             feature_cols += [i+'_'+helmet_view+'_'+str(cluster)+'_'+str(player_ind) for i in ['left', 'width', 'top', 'height']]
#         del tmp_helmets

# cols = [i[:-2] for i in df.columns if i[-2:]=='_1' and i!='nfl_player_id_1']
# df[[i+'_diff' for i in cols]] = np.abs(df[[i+'_1' for i in cols]].values - df[[i+'_2' for i in cols]].values)
# feature_cols += [i+'_diff' for i in cols]

# cols = USE_COLS
# df[[i+'_prod' for i in cols]] = np.abs(df[[i+'_1' for i in cols]].values - df[[i+'_2' for i in cols]].values)
# feature_cols += [i+'_prod' for i in cols]
# df[feature_cols].fillna(-1, inplace=True)

feature_cols = ['x_position_1', 'y_position_1', 'speed_1', 'distance_1', 'direction_1', 'orientation_1', 'acceleration_1', 'sa_1', 'x_position_2', 'y_position_2', 'speed_2', 'distance_2', 'direction_2', 'orientation_2', 'acceleration_2', 'sa_2', 'distance', 'G_flug', 'left_Sideline_10_1', 'width_Sideline_10_1', 'top_Sideline_10_1', 'height_Sideline_10_1', 'left_Sideline_10_2', 'width_Sideline_10_2', 'top_Sideline_10_2', 'height_Sideline_10_2', 'left_Endzone_10_1', 'width_Endzone_10_1', 'top_Endzone_10_1', 'height_Endzone_10_1', 'left_Endzone_10_2', 'width_Endzone_10_2', 'top_Endzone_10_2', 'height_Endzone_10_2', 'left_Sideline_50_1', 'width_Sideline_50_1', 'top_Sideline_50_1', 'height_Sideline_50_1', 'left_Sideline_50_2', 'width_Sideline_50_2', 'top_Sideline_50_2', 'height_Sideline_50_2', 'left_Endzone_50_1', 'width_Endzone_50_1', 'top_Endzone_50_1', 'height_Endzone_50_1', 'left_Endzone_50_2', 'width_Endzone_50_2', 'top_Endzone_50_2', 'height_Endzone_50_2', 'left_Sideline_100_1', 'width_Sideline_100_1', 'top_Sideline_100_1', 'height_Sideline_100_1', 'left_Sideline_100_2', 'width_Sideline_100_2', 'top_Sideline_100_2', 'height_Sideline_100_2', 'left_Endzone_100_1', 'width_Endzone_100_1', 'top_Endzone_100_1', 'height_Endzone_100_1', 'left_Endzone_100_2', 'width_Endzone_100_2', 'top_Endzone_100_2', 'height_Endzone_100_2', 'left_Sideline_500_1', 'width_Sideline_500_1', 'top_Sideline_500_1', 'height_Sideline_500_1', 'left_Sideline_500_2', 'width_Sideline_500_2', 'top_Sideline_500_2', 'height_Sideline_500_2', 'left_Endzone_500_1', 'width_Endzone_500_1', 'top_Endzone_500_1', 'height_Endzone_500_1', 'left_Endzone_500_2', 'width_Endzone_500_2', 'top_Endzone_500_2', 'height_Endzone_500_2', 'x_position_diff', 'y_position_diff', 'speed_diff', 'distance_diff', 'direction_diff', 'orientation_diff', 'acceleration_diff', 'sa_diff', 'left_Sideline_10_diff', 'width_Sideline_10_diff', 'top_Sideline_10_diff', 'height_Sideline_10_diff', 'left_Endzone_10_diff', 'width_Endzone_10_diff', 'top_Endzone_10_diff', 'height_Endzone_10_diff', 'left_Sideline_50_diff', 'width_Sideline_50_diff', 'top_Sideline_50_diff', 'height_Sideline_50_diff', 'left_Endzone_50_diff', 'width_Endzone_50_diff', 'top_Endzone_50_diff', 'height_Endzone_50_diff', 'left_Sideline_100_diff', 'width_Sideline_100_diff', 'top_Sideline_100_diff', 'height_Sideline_100_diff', 'left_Endzone_100_diff', 'width_Endzone_100_diff', 'top_Endzone_100_diff', 'height_Endzone_100_diff', 'left_Sideline_500_diff', 'width_Sideline_500_diff', 'top_Sideline_500_diff', 'height_Sideline_500_diff', 'left_Endzone_500_diff', 'width_Endzone_500_diff', 'top_Endzone_500_diff', 'height_Endzone_500_diff', 'x_position_prod', 'y_position_prod', 'speed_prod', 'distance_prod', 'direction_prod', 'orientation_prod', 'acceleration_prod', 'sa_prod']
print("Number of features", len(feature_cols))
print("Columns: ", feature_cols)

# df.to_csv('xg_130feats.csv', index=False)

if torch.cuda.is_available():
    df= cudf.read_csv("xg_130feats.csv")
    df = df.to_pandas()
else:
    df = pd.read_csv("xg_130feats.csv")

print(f"All data: {len(df)}")

for nnmodel in ["resnet50", "tf_efficientnetv2_b0", "tf_efficientnetv2_b2"]:
    data_full = {}
    feat_full = {}
    print(f"Loading result of {nnmodel}")
    for kfold in range(5):
        data = torch.load(f"{nnmodel}_feat_prob_fold{kfold}.pth")
        for k, v in data.items():
            data_full[k] = v[0]
            feat_full[k] = v[1]
    probs = []
    feats = []
    contact_ids = df.contact_id.to_list()
    for index in range(len(df)):
        contact_id = contact_ids[index]
        probs.append(data_full[contact_id])
        feats.append(feat_full[contact_id])
    df[f'{nnmodel}_prob'] = probs
    feature_cols.append(f'{nnmodel}_prob')

    # columns = [f'{nnmodel}_feat{i}' for i in range(len(feats[0]))]
    # split = pd.DataFrame(feats, columns = columns)
    # df = pd.concat([df, split], axis=1)
    # feature_cols.extend(columns)
            

oof_pred = fit_xgboost(cfg, df, feature_cols, cfg.xgb_params, add_suffix="")


def func(x_list):
    score = matthews_corrcoef(df['contact'], oof_pred>x_list[0])
    return -score

x0 = [0.5]
result = minimize(func, x0,  method="nelder-mead")
print("score:", round(matthews_corrcoef(df['contact'], oof_pred>result.x[0]), 5))
print("threshold", round(result.x[0], 5))
gc.collect()