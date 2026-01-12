import os
import json
import cupy as cp
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, brier_score_loss, root_mean_squared_error
import joblib

HPARAMS = None

def grid_search_xgb(X, Y, device, binary, hparams):
    hparams = {
    'n_estimators': [20, 50, 100, 200],     # Number of boosting rounds
    'max_depth': [3, 4, 5, 6],             # Maximum depth of a tree
    'reg_lambda': [0.5, 1, 2, 5],          # L2 regularization term on weights
    }

    model_class = xgb.XGBClassifier if binary else xgb.XGBRegressor
    scoring = 'neg_brier_score' if binary else 'neg_mean_squared_error'
    m = model_class(
        tree_method='gpu_hist',
        device=device,
        verbosity=0,
    )
    grid_search = GridSearchCV(
        estimator=m, param_grid=hparams,
        scoring=scoring, cv=5, verbose=0,
    )

    X.columns = (
    X.columns.astype(str)  # ensure all are strings
    .str.replace('[', '(', regex=False)
    .str.replace(']', ')', regex=False)
    .str.replace('<', 'lt_', regex=False)
    .str.replace('>', 'gt_', regex=False))

    grid_search.fit(X, Y)
    return grid_search.best_params_

def train_estimate_effects(all_df, X, Y, W_cols, Z_cols, Y_binary,
        N, device, clip=0.0, K=5, bootstraps=30, sample_kwargs=None, 
        hparam_path=None, grid_search = grid_search_xgb):

    N = [len(all_df)]
    model_list = [
        'px_z', 'px_wz', 
        'y_zx0', 'y_zx1',
        'y_wvzx0', 'y_wvzx1',
        'mu2_ne', 'mu1_ne',
        'mu2_ne*', 'mu1_ne*'
    ]
    
    hp_dict = {}

    search_df = all_df.copy()

    aux_cols = {c: None for c in ['mu2_ne', 'mu2_ne*']}
    aux_df = pd.DataFrame(aux_cols, index=search_df.index)
    search_df = pd.concat([search_df, aux_df], axis=1)
    search_df = search_df.reset_index(drop=True)  
    search_df_x0 = search_df[search_df[X] == 0]
    search_df_x1 = search_df[search_df[X] == 1]
    idxs = np.arange(len(search_df))
    print('Fitting propensity hparams')
    for m in tqdm(model_list[:2]):
        if m in hp_dict.keys():
            continue
        cols = [z for z in Z_cols]
        cond = m.split('_')[1]
        if 'w' in cond:
            cols += W_cols
        hp = grid_search(search_df[cols], search_df[X], device, True, HPARAMS)
        hp_dict[m] = hp


    print('Fitting outcome hparams')
    if 'y_zx0' not in hp_dict.keys():
        hp_dict['y_zx0'] = grid_search(
            search_df_x0[Z_cols], search_df_x0[Y], device, Y_binary, HPARAMS)
    if 'y_zx1' not in hp_dict.keys():
        hp_dict['y_zx1'] = grid_search(
            search_df_x1[Z_cols], search_df_x1[Y], device, Y_binary, HPARAMS)
    if 'y_wvzx0' not in hp_dict.keys():
        hp_dict['y_wvzx0'] = grid_search(
            search_df_x0[Z_cols + W_cols], search_df_x0[Y], device, Y_binary, HPARAMS)
    if 'y_wvzx1' not in hp_dict.keys():
        hp_dict['y_wvzx1'] = grid_search(
            search_df_x1[Z_cols + W_cols], search_df_x1[Y], device, Y_binary, HPARAMS)

    print('Fitting nested outcome hparams')
    if 'mu2_zx0' not in hp_dict.keys():
        np.random.seed(1001)
        np.random.shuffle(idxs)
        mu, ns = np.array_split(idxs, 2)
        df_mu, df_ns = search_df.iloc[mu], search_df.iloc[ns]
        (_, df_mu_x0), (_, df_mu_x1) = df_mu.groupby(df_mu[X])
        (_, df_ns_x0), (_, df_ns_x1) = df_ns.groupby(df_ns[X])
        y_wvzx1 = fit(df_mu_x1[W_cols + Z_cols], df_mu_x1[Y], device, binary=Y_binary, **hp_dict['y_wvzx1'])
        df_ns_x0['mu2_ne'] = pred(df_ns_x0[W_cols + Z_cols], y_wvzx1, binary=Y_binary)
        hp = grid_search(df_ns_x0[Z_cols], df_ns_x0['mu2_ne'],device, False, HPARAMS)
        hp_dict['mu2_zx0'] = hp

    if all_df is not None and 'mu2_zx0*' not in hp_dict.keys():
        np.random.seed(1001)
        np.random.shuffle(idxs)
        mu, ns = np.array_split(idxs, 2)
        df_mu, df_ns = search_df.iloc[mu], search_df.iloc[ns]
        (_, df_mu_x0), (_, df_mu_x1) = df_mu.groupby(df_mu[X])
        (_, df_ns_x0), (_, df_ns_x1) = df_ns.groupby(df_ns[X])
        y_zx1 = fit(df_mu_x1[Z_cols], df_mu_x1[Y], device, binary=Y_binary, **hp_dict['y_zx1'])
        df_ns_x0['mu2_ne*'] = pred(df_ns_x0[Z_cols], y_zx1, binary=Y_binary)
        hp = grid_search(df_ns_x0[Z_cols], df_ns_x0['mu2_ne*'], device, False, HPARAMS)
        hp_dict['mu2_zx0*'] = hp

    with open(f'{hparam_path}/hparams_dimw{len(W_cols)}_dimz{len(Z_cols)}.pkl', 'w') as f:
        json.dump(hp_dict, f, indent=4)

    results_df = pd.DataFrame()
    aux_cols = {c: None for c in model_list}
    model_scores_df = pd.DataFrame()
    synthetic = False

    for n in N:
        all_idxs = np.arange(n)
        print('Sample size:', n)
        for i in tqdm(range(bootstraps)):
            folds = KFold(n_splits=K, shuffle=True, random_state=i)
            df = all_df.copy()
            aux_df = pd.DataFrame(aux_cols, index=df.index)
            df = pd.concat([df, aux_df], axis=1)
            df = df.reset_index(drop=True)
            for k_split, (tr, ts) in enumerate(folds.split(all_idxs)):
                df_tr, df_ts = df.iloc[tr], df.iloc[ts]
                (_, df_tr_x0), (_, df_tr_x1) = df_tr.groupby(df_tr[X])

                # Propensity models
                px_z = fit(df_tr[Z_cols], df_tr[X], device, **hp_dict['px_z'])
                px_wz = fit(df_tr[W_cols + Z_cols], df_tr[X], device, **hp_dict['px_wz'])
                df.loc[ts, 'px_z'] = pred(df_ts[Z_cols], px_z, clip=clip)
                df.loc[ts, 'px_wz'] = pred(df_ts[W_cols + Z_cols], px_wz, clip=clip)

                # Outcome models
                y_zx0 = fit(df_tr_x0[Z_cols], df_tr_x0[Y], device, binary=Y_binary, **hp_dict['y_zx0'])
                y_zx1 = fit(df_tr_x1[Z_cols], df_tr_x1[Y], device, binary=Y_binary, **hp_dict['y_zx1'])
                y_wvzx0 = fit(df_tr_x0[W_cols + Z_cols], df_tr_x0[Y], device, binary=Y_binary, **hp_dict['y_wvzx0'])
                y_wvzx1 = fit(df_tr_x1[W_cols + Z_cols], df_tr_x1[Y], device, binary=Y_binary, **hp_dict['y_wvzx1'])
                df.loc[ts, 'y_zx0'] = pred(df_ts[Z_cols], y_zx0, binary=Y_binary)
                df.loc[ts, 'y_zx1'] = pred(df_ts[Z_cols], y_zx1, binary=Y_binary)
                df.loc[ts, 'y_wvzx0'] = pred(df_ts[W_cols + Z_cols], y_wvzx0, binary=Y_binary)
                df.loc[ts, 'y_wvzx1'] = pred(df_ts[W_cols + Z_cols], y_wvzx1, binary=Y_binary)

                # Split df_tr into two equal parts mu and ns
                np.random.seed(i)
                np.random.shuffle(tr)
                mu, ns = np.array_split(tr, 2)
                df_mu, df_ns = df.iloc[mu], df.iloc[ns]
                (_, df_mu_x0), (_, df_mu_x1) = df_mu.groupby(df_mu[X])
                (_, df_ns_x0), (_, df_ns_x1) = df_ns.groupby(df_ns[X])

                # Regress Y on W, V, Z at level x1
                y_wvzx1_mu = fit(df_mu_x1[W_cols + Z_cols], df_mu_x1[Y], device, binary=Y_binary, **hp_dict['y_wvzx1'])
                df_ns_x0['mu2_ne'] = pred(df_ns_x0[W_cols + Z_cols], y_wvzx1_mu, binary=Y_binary)
                df.loc[ts, 'mu2_ne'] = pred(df_ts[W_cols + Z_cols], y_wvzx1_mu, binary=Y_binary)
                # Regress E[Y|W, V, Z, X=1] on Z at level x0
                mu1_zx0 = fit(df_ns_x0[Z_cols], df_ns_x0['mu2_ne'], device, binary=False, **hp_dict['mu2_zx0'])
                df.loc[ts, 'mu1_ne'] = pred(df_ts[Z_cols], mu1_zx0, binary=False)
                if Y_binary:
                    df.loc[ts, 'mu1_ne'] = np.clip(df.loc[ts, 'mu1_ne'], a_min=0, a_max=1)

                df_ns_x0['mu2_ne*'] = pred(df_ns_x0[Z_cols], y_zx1, binary=Y_binary)
                df.loc[ts, 'mu2_ne*'] = pred(df_ts[Z_cols], y_zx1, binary=Y_binary)
                # Regress E[Y|W, V, Z, X=1] on Z at level x0
                mu1_zx0_ = fit(df_ns_x0[Z_cols], df_ns_x0['mu2_ne*'], device, binary=False, **hp_dict['mu2_zx0*'])
                df.loc[ts, 'mu1_ne*'] = pred(df_ts[Z_cols], mu1_zx0_, binary=False)
                if Y_binary:
                    df.loc[ts, 'mu1_ne*'] = np.clip(df.loc[ts, 'mu1_ne*'], a_min=0, a_max=1)
 
                model_objects = {
                    'px_z': px_z,
                    'px_wz': px_wz,
                    'y_zx0': y_zx0,
                    'y_zx1': y_zx1,
                    'y_wvzx0': y_wvzx0,
                    'y_wvzx1': y_wvzx1,
                    'mu2_ne': y_wvzx1_mu,
                    'mu1_zx0': mu1_zx0,
                    'mu1_zx0_': mu1_zx0_,
                }

                # Save all models for this bootstrap/k_split
                file_path = os.path.join(hparam_path, f"models_boot{i}_split{k_split}.joblib")
                joblib.dump(model_objects, file_path)

            model_scores = compute_model_scores(df, X, Y, model_list)
            model_scores_df = pd.concat([model_scores_df, model_scores], ignore_index=True)

            results = compute_effects_from_df(df, X, Y, Y_binary, synthetic)
            results_df = pd.concat(
                [results_df, results], 
                ignore_index=True,
            )
    return results_df, model_scores_df

def compute_gt_effects(sample, n_mc, X, Y, seed, sample_kwargs):
    mc_df = sample(n=n_mc, seed=seed, **sample_kwargs)
    mc_df_x0 = mc_df[mc_df[X] == 0]
    mc_df_x1 = mc_df[mc_df[X] == 1]
    EYx0 = np.mean(mc_df[f'{Y}[x0]'])
    EYx1 = np.mean(mc_df[f'{Y}[x1]'])
    EYx1WVx0 = np.mean(mc_df[f'{Y}[x1WVx0]'])
    EYx1Vx0Wx1 = np.mean(mc_df[f'{Y}[x1Vx0Wx1]'])
    TV = np.mean(mc_df_x1[Y]) - np.mean(mc_df_x0[Y])
    TE = np.mean(mc_df[f'{Y}[x1]']) - np.mean(mc_df[f'{Y}[x0]'])
    NDE = np.mean(mc_df[f'{Y}[x1WVx0]']) - np.mean(mc_df[f'{Y}[x0]'])
    NIE = np.mean(mc_df[f'{Y}[x1]']) - np.mean(mc_df[f'{Y}[x1WVx0]'])
    expse_x1 = np.mean(mc_df_x1[Y])-np.mean(mc_df[f'{Y}[x1]'])
    EYx0_obs = np.mean(mc_df_x0[Y])
    EYx1_obs = np.mean(mc_df_x1[Y])
    expse_x0 = np.mean(mc_df_x0[Y])-np.mean(mc_df[f'{Y}[x0]'])
    SE = expse_x1-expse_x0
    gt_effects = (EYx0, EYx1, EYx1WVx0, EYx1Vx0Wx1, TV, TE, NDE, NIE, EYx0_obs, EYx1_obs, expse_x1, expse_x0, SE)
    return mc_df, gt_effects

def compute_model_scores(df, X, Y, aux_cols):
    list_perfs = {}
    for col in aux_cols:
        sub_df = df.loc[~(df[col].isna())]
        if col.split('_')[0] == 'px':
            auc = roc_auc_score(sub_df[X], sub_df[col])
            brier = brier_score_loss(sub_df[X], sub_df[col])
        elif col.split('_')[0] == 'mu1':
            auc = roc_auc_score(sub_df[Y], sub_df[col])
            brier = root_mean_squared_error(sub_df[Y], sub_df[col]) # RMSE
        else:
            auc = roc_auc_score(sub_df[Y], sub_df[col])
            brier = brier_score_loss(sub_df[Y], sub_df[col])
        list_perfs[f'{col}_auroc'] = auc
        list_perfs[f'{col}_brier'] = brier
    return pd.DataFrame([list_perfs])

def compute_effects_from_df(df, X, Y, Y_binary, synthetic):
    if synthetic:
        EYx0, EYx1, EYx1WVx0, EYx1Vx0Wx1, TV, TE, NDE, NIE, EYx0_obs, EYx1_obs, expse_x1, expse_x0, SE = gt_effects

    # do(x) nuisance parameters
    pi_x0 = (df[X] == 0) / (1 - df['px_z'])
    pi_x1 = (df[X] == 1) / (df['px_z'])
    pi_x0_sn = pi_x0 / np.mean(pi_x0)
    pi_x1_sn = pi_x1 / np.mean(pi_x1)
    Epi_x0 = np.mean(pi_x0)
    Epi_x1 = np.mean(pi_x1)

    # do(x) standard
    hat_Yx0 = pi_x0 * (df[Y] - df['y_zx0']) + df['y_zx0']
    hat_Yx1 = pi_x1 * (df[Y] - df['y_zx1']) + df['y_zx1']
    Estimated_EYx0 = np.mean(hat_Yx0)
    Estimated_EYx1 = np.mean(hat_Yx1)

    # do(x) self-normalized
    hat_Yx0_sn = pi_x0_sn * (df[Y] - df['y_zx0']) + df['y_zx0']
    hat_Yx1_sn = pi_x1_sn * (df[Y] - df['y_zx1']) + df['y_zx1']
    Estimated_EYx0_sn = np.mean(hat_Yx0_sn)
    Estimated_EYx1_sn = np.mean(hat_Yx1_sn)

    # NDE/NIE nuisance parameters
    pi2 = (df[X] == 1) * (1 - df['px_wz']) / (df['px_wz'] * (1 - df['px_z']))
    pi1 = (df[X] == 0) / (1 - df['px_z'])
    pi2_sn = pi2 / np.mean(pi2)
    pi1_sn = pi1 / np.mean(pi1)
    Epi2_ne = np.mean(pi2)
    Epi1_ne = np.mean(pi1)

    # NDE/NIE standard
    Estimated_EYx1WVx0_pi2Y = np.mean((pi2 * df[Y]))
    Estimated_EYx1WVx0_pi2mu2 = np.mean((pi2 * df['mu2_ne']))
    Estimated_EYx1WVx0_pi1mu2 = np.mean(pi1 * df['mu2_ne'])
    Estimated_EYx1WVx0_pi1mu1 = np.mean(pi1 * df['mu1_ne'])
    Estimated_EYx1WVx0_mu1 = np.mean(df['mu1_ne'])
    Estimated_EYx1WVx0 = (
        (Estimated_EYx1WVx0_pi2Y - Estimated_EYx1WVx0_pi2mu2) + 
        (Estimated_EYx1WVx0_pi1mu2 - Estimated_EYx1WVx0_pi1mu1) + 
        Estimated_EYx1WVx0_mu1
    )
    Estimated_NDE = Estimated_EYx1WVx0 - Estimated_EYx0
    Estimated_NIE = Estimated_EYx1 - Estimated_EYx1WVx0

    # NDE standard sub-estimates
    Estimated_NDE_pi2Y = Estimated_EYx1WVx0_pi2Y - Estimated_EYx0
    Estimated_NDE_pi2mu2 = Estimated_EYx1WVx0_pi2mu2 - Estimated_EYx0
    Estimated_NDE_pi1mu2 = Estimated_EYx1WVx0_pi1mu2 - Estimated_EYx0
    Estimated_NDE_pi1mu1 = Estimated_EYx1WVx0_pi1mu1 - Estimated_EYx0
    Estimated_NDE_mu1 = Estimated_EYx1WVx0_mu1 - Estimated_EYx0

    # NIE standard sub-estimates
    Estimated_NIE_pi2Y = Estimated_EYx1 - Estimated_EYx1WVx0_pi2Y
    Estimated_NIE_pi2mu2 = Estimated_EYx1 - Estimated_EYx1WVx0_pi2mu2
    Estimated_NIE_pi1mu2 = Estimated_EYx1 - Estimated_EYx1WVx0_pi1mu2
    Estimated_NIE_pi1mu1 = Estimated_EYx1 - Estimated_EYx1WVx0_pi1mu1
    Estimated_NIE_mu1 = Estimated_EYx1 - Estimated_EYx1WVx0_mu1

    # NDE/NIE self-normalized
    Estimated_EYx1WVx0_pi2Y_sn = np.mean((pi2_sn * df[Y]))
    Estimated_EYx1WVx0_pi2mu2_sn = np.mean((pi2_sn * df['mu2_ne']))
    Estimated_EYx1WVx0_pi1mu2_sn = np.mean(pi1_sn * df['mu2_ne'])
    Estimated_EYx1WVx0_pi1mu1_sn = np.mean(pi1_sn * df['mu1_ne'])
    Estimated_EYx1WVx0_sn = (
        (Estimated_EYx1WVx0_pi2Y_sn - Estimated_EYx1WVx0_pi2mu2_sn) + 
        (Estimated_EYx1WVx0_pi1mu2_sn - Estimated_EYx1WVx0_pi1mu1_sn) + 
        Estimated_EYx1WVx0_mu1
    )
    Estimated_NDE_sn = Estimated_EYx1WVx0_sn - Estimated_EYx0_sn
    Estimated_NIE_sn = Estimated_EYx1_sn - Estimated_EYx1WVx0_sn

    # NDE self-normalized sub-estimates
    Estimated_NDE_pi2Y_sn = Estimated_EYx1WVx0_pi2Y_sn - Estimated_EYx0_sn
    Estimated_NDE_pi2mu2_sn = Estimated_EYx1WVx0_pi2mu2_sn - Estimated_EYx0_sn
    Estimated_NDE_pi1mu2_sn = Estimated_EYx1WVx0_pi1mu2_sn - Estimated_EYx0_sn
    Estimated_NDE_pi1mu1_sn = Estimated_EYx1WVx0_pi1mu1_sn - Estimated_EYx0_sn
    Estimated_NDE_mu1_sn = Estimated_EYx1WVx0_mu1 - Estimated_EYx0_sn

    # NIE self-normalized sub-estimates
    Estimated_NIE_pi2Y_sn = Estimated_EYx1_sn - Estimated_EYx1WVx0_pi2Y_sn
    Estimated_NIE_pi2mu2_sn = Estimated_EYx1_sn - Estimated_EYx1WVx0_pi2mu2_sn
    Estimated_NIE_pi1mu2_sn = Estimated_EYx1_sn - Estimated_EYx1WVx0_pi1mu2_sn
    Estimated_NIE_pi1mu1_sn = Estimated_EYx1_sn - Estimated_EYx1WVx0_pi1mu1_sn
    Estimated_NIE_mu1_sn = Estimated_EYx1_sn - Estimated_EYx1WVx0_mu1

    # Spurious Estimates
    x0_only = df[df[X] == 0]
    x1_only = df[df[X] == 1]
    Estimated_EYx0_obs = np.mean(x0_only[Y])
    Estimated_EYx1_obs = np.mean(x1_only[Y])
    Estimated_ExpSE_x0 = Estimated_EYx0_obs-Estimated_EYx0
    Estimated_ExpSE_x1 = Estimated_EYx1_obs-Estimated_EYx1
    Estimated_SE = Estimated_ExpSE_x1-Estimated_ExpSE_x0

    # Self normalized Spurious estimates
    Estimated_ExpSE_x0_sn = Estimated_EYx0_obs-Estimated_EYx0_sn
    Estimated_ExpSE_x1_sn = Estimated_EYx1_obs-Estimated_EYx1_sn
    Estimated_SE_sn = Estimated_ExpSE_x1_sn-Estimated_ExpSE_x0_sn

    if synthetic:
        results = {
            'n': [len(df)],
            'EYx0': [EYx0],
            'EYx1': [EYx1],
            'EYx0_obs': [EYx0_obs], 
            'EYx1_obs': [EYx1_obs],
            'EYx1WVx0': [EYx1WVx0],
            'EYx1Vx0Wx1': [EYx1Vx0Wx1],
            'NDE': [NDE],
            'NIE': [NIE],
            'ExpSE_x0': [expse_x0],
            'ExpSE_x1': [expse_x1],
            'SE': [SE],
            # Estimates
            'Estimated_EYx0': [Estimated_EYx0],
            'Estimated_EYx1': [Estimated_EYx1],
            'Estimated_EYx0_obs': [Estimated_EYx0_obs],
            'Estimated_EYx1_obs': [Estimated_EYx1_obs],
            'Estimated_EYx0_sn': [Estimated_EYx0_sn],
            'Estimated_EYx1_sn': [Estimated_EYx1_sn],
            'Estimated_EYx1WVx0': [Estimated_EYx1WVx0],
            'Estimated_EYx1WVx0_sn': [Estimated_EYx1WVx0_sn],
            'Estimated_NDE': [Estimated_NDE],
            'Estimated_NIE': [Estimated_NIE],
            'Estimated_ExpSE_x0': [Estimated_ExpSE_x0],
            'Estimated_ExpSE_x1': [Estimated_ExpSE_x1],
            'Estimated_SE': [Estimated_SE],
            'Estimated_NDE_sn': [Estimated_NDE_sn],
            'Estimated_NIE_sn': [Estimated_NIE_sn],
            'Estimated_ExpSE_x0_sn': [Estimated_ExpSE_x0_sn],
            'Estimated_ExpSE_x1_sn': [Estimated_ExpSE_x1_sn],
            'Estimated_SE_sn': [Estimated_SE_sn],
            # Absolute errors
            'EYx0_Error': [Estimated_EYx0 - EYx0],
            'EYx1_Error': [Estimated_EYx1 - EYx1],
            'EYx0_obs_Error': [Estimated_EYx0_obs-EYx0_obs],
            'EYx1_obs_Error': [Estimated_EYx1_obs-EYx1_obs],
            'EYx0_sn_Error': [Estimated_EYx0_sn - EYx0],
            'EYx1_sn_Error': [Estimated_EYx1_sn - EYx1],
            'EYx1WVx0_Error': [Estimated_EYx1WVx0 - EYx1WVx0],
            'EYx1WVx0_sn_Error': [Estimated_EYx1WVx0_sn - EYx1WVx0],
            'NDE_Error': [Estimated_NDE - NDE],
            'NIE_Error': [Estimated_NIE - NIE],
            'ExpSE_X0_Error':[Estimated_ExpSE_x0-expse_x0],
            'ExpSE_X1_Error':[Estimated_ExpSE_x1-expse_x1],
            'SE_Error':[Estimated_SE-SE],
            'NDE_sn_Error': [Estimated_NDE_sn - NDE],
            'NIE_sn_Error': [Estimated_NIE_sn - NIE],
            'ExpSE_x1_sn_Error': [Estimated_ExpSE_x1_sn-expse_x1],
            'ExpSE_x0_sn_Error': [Estimated_ExpSE_x0_sn-expse_x0],
            'SE_sn_Error': [Estimated_SE_sn-SE],
            # Relative errors
            'EYx0_Error%': [rel_err(Estimated_EYx0, EYx0)],
            'EYx1_Error%': [rel_err(Estimated_EYx1, EYx1)],
            'EYx0_sn_Error%': [rel_err(Estimated_EYx0_sn, EYx0)],
            'EYx1_sn_Error%': [rel_err(Estimated_EYx1_sn, EYx1)],
            'EYx0_obs_Error%': [rel_err(Estimated_EYx0_obs, EYx0_obs)],
            'EYx1_obs_Error%': [rel_err(Estimated_EYx1_obs, EYx1_obs)],
            'EYx1WVx0_Error%': [rel_err(Estimated_EYx1WVx0, EYx1WVx0)],
            'EYx1WVx0_sn_Error%': [rel_err(Estimated_EYx1WVx0_sn, EYx1WVx0)],
            'NDE_Error%': [rel_err(Estimated_NDE, NDE)],
            'NIE_Error%': [rel_err(Estimated_NIE, NIE)],
            'ExpSE_X0_Error%':[rel_err(Estimated_ExpSE_x0, expse_x0)],
            'ExpSE_X1_Error%':[rel_err(Estimated_ExpSE_x1, expse_x1)],
            'SE_Error%':[rel_err(Estimated_SE, SE)],
            'NDE_sn_Error%': [rel_err(Estimated_NDE_sn, NDE)],
            'NIE_sn_Error%': [rel_err(Estimated_NIE_sn, NIE)],
            'ExpSE_X0_sn_Error%':[rel_err(Estimated_ExpSE_x0_sn, expse_x0)],
            'ExpSE_X1_sn_Error%':[rel_err(Estimated_ExpSE_x1_sn, expse_x1)],
            'SE_sn_Error%':[rel_err(Estimated_SE_sn, SE)],
            # Sub-estimates
            'Estimated_NDE_pi2Y': [Estimated_NDE_pi2Y],
            'Estimated_NDE_pi2mu2': [Estimated_NDE_pi2mu2],
            'Estimated_NDE_pi1mu2': [Estimated_NDE_pi1mu2],
            'Estimated_NDE_pi1mu1': [Estimated_NDE_pi1mu1],
            'Estimated_NDE_mu1': [Estimated_NDE_mu1],
            'Estimated_NIE_pi2Y': [Estimated_NIE_pi2Y],
            'Estimated_NIE_pi2mu2': [Estimated_NIE_pi2mu2],
            'Estimated_NIE_pi1mu2': [Estimated_NIE_pi1mu2],
            'Estimated_NIE_pi1mu1': [Estimated_NIE_pi1mu1],
            'Estimated_NIE_mu1': [Estimated_NIE_mu1],
            # Sub-estimates (SN)
            'Estimated_NDE_pi2Y_sn': [Estimated_NDE_pi2Y_sn],
            'Estimated_NDE_pi2mu2_sn': [Estimated_NDE_pi2mu2_sn],
            'Estimated_NDE_pi1mu2_sn': [Estimated_NDE_pi1mu2_sn],
            'Estimated_NDE_pi1mu1_sn': [Estimated_NDE_pi1mu1_sn],
            'Estimated_NDE_mu1_sn': [Estimated_NDE_mu1_sn],
            'Estimated_NIE_pi2Y_sn': [Estimated_NIE_pi2Y_sn],
            'Estimated_NIE_pi2mu2_sn': [Estimated_NIE_pi2mu2_sn],
            'Estimated_NIE_pi1mu2_sn': [Estimated_NIE_pi1mu2_sn],
            'Estimated_NIE_pi1mu1_sn': [Estimated_NIE_pi1mu1_sn],
            'Estimated_NIE_mu1_sn': [Estimated_NIE_mu1_sn],
            # Sub-estimate errors
            'NDE_pi2Y_Error': [Estimated_NDE_pi2Y - NDE],
            'NDE_pi2mu2_Error': [Estimated_NDE_pi2mu2 - NDE],
            'NDE_pi1mu2_Error': [Estimated_NDE_pi1mu2 - NDE],
            'NDE_pi1mu1_Error': [Estimated_NDE_pi1mu1 - NDE],
            'NDE_mu1_Error': [Estimated_NDE_mu1 - NDE],
            'NIE_pi2Y_Error': [Estimated_NIE_pi2Y - NIE],
            'NIE_pi2mu2_Error': [Estimated_NIE_pi2mu2 - NIE],
            'NIE_pi1mu2_Error': [Estimated_NIE_pi1mu2 - NIE],
            'NIE_pi1mu1_Error': [Estimated_NIE_pi1mu1 - NIE],
            'NIE_mu1_Error': [Estimated_NIE_mu1 - NIE],
            # Sub-estimate errors (SN)
            'NDE_pi2Y_sn_Error': [Estimated_NDE_pi2Y_sn - NDE],
            'NDE_pi2mu2_sn_Error': [Estimated_NDE_pi2mu2_sn - NDE],
            'NDE_pi1mu2_sn_Error': [Estimated_NDE_pi1mu2_sn - NDE],
            'NDE_pi1mu1_sn_Error': [Estimated_NDE_pi1mu1_sn - NDE],
            'NDE_mu1_sn_Error': [Estimated_NDE_mu1_sn - NDE],
            'NIE_pi2Y_sn_Error': [Estimated_NIE_pi2Y_sn - NIE],
            'NIE_pi2mu2_sn_Error': [Estimated_NIE_pi2mu2_sn - NIE],
            'NIE_pi1mu2_sn_Error': [Estimated_NIE_pi1mu2_sn - NIE],
            'NIE_pi1mu1_sn_Error': [Estimated_NIE_pi1mu1_sn - NIE],
            'NIE_mu1_sn_Error': [Estimated_NIE_mu1_sn - NIE],
            #Sub-estimate relative errors
            'NDE_pi2Y_Error%': [rel_err(Estimated_NDE_pi2Y, NDE)],
            'NDE_pi2mu2_Error%': [rel_err(Estimated_NDE_pi2mu2, NDE)],
            'NDE_pi1mu2_Error%': [rel_err(Estimated_NDE_pi1mu2, NDE)],
            'NDE_pi1mu1_Error%': [rel_err(Estimated_NDE_pi1mu1, NDE)],
            'NDE_mu1_Error%': [rel_err(Estimated_NDE_mu1, NDE)],
            'NIE_pi2Y_Error%': [rel_err(Estimated_NIE_pi2Y, NIE)],
            'NIE_pi2mu2_Error%': [rel_err(Estimated_NIE_pi2mu2, NIE)],
            'NIE_pi1mu2_Error%': [rel_err(Estimated_NIE_pi1mu2, NIE)],
            'NIE_pi1mu1_Error%': [rel_err(Estimated_NIE_pi1mu1, NIE)],
            'NIE_mu1_Error%': [rel_err(Estimated_NIE_mu1, NIE)],
            #Sub-estimate relative errors (SN)
            'NDE_pi2Y_sn_Error%': [rel_err(Estimated_NDE_pi2Y_sn, NDE)],
            'NDE_pi2mu2_sn_Error%': [rel_err(Estimated_NDE_pi2mu2_sn, NDE)],
            'NDE_pi1mu2_sn_Error%': [rel_err(Estimated_NDE_pi1mu2_sn, NDE)],
            'NDE_pi1mu1_sn_Error%': [rel_err(Estimated_NDE_pi1mu1_sn, NDE)],
            'NDE_mu1_sn_Error%': [rel_err(Estimated_NDE_mu1_sn, NDE)],
            'NIE_pi2Y_sn_Error%': [rel_err(Estimated_NIE_pi2Y_sn, NIE)],
            'NIE_pi2mu2_sn_Error%': [rel_err(Estimated_NIE_pi2mu2_sn, NIE)],
            'NIE_pi1mu2_sn_Error%': [rel_err(Estimated_NIE_pi1mu2_sn, NIE)],
            'NIE_pi1mu1_sn_Error%': [rel_err(Estimated_NIE_pi1mu1_sn, NIE)],
            'NIE_mu1_sn_Error%': [rel_err(Estimated_NIE_mu1_sn, NIE)],
            # Nuisance parameters
            'Epi_x0': [Epi_x0],
            'Epi_x1': [Epi_x1],
            'Epi2_ne': [Epi2_ne],
            'Epi1_ne': [Epi1_ne],
        }
    else:
        # do(x) biased
        pi_x0_biased = (df[X] == 0) / (1 - df['px_wz'])
        pi_x1_biased = (df[X] == 1) / (df['px_wz'])
        hat_Yx0_biased = pi_x0_biased * (df[Y] - df['y_wvzx0']) + df['y_wvzx0']
        hat_Yx1_biased = pi_x1_biased * (df[Y] - df['y_wvzx1']) + df['y_wvzx1']
        Estimated_EYx0_biased = np.mean(hat_Yx0_biased)
        Estimated_EYx1_biased = np.mean(hat_Yx1_biased)

        # Total Effect
        Estimated_TE = Estimated_EYx1 - Estimated_EYx0
        Estimated_TE_sn = Estimated_EYx1_sn - Estimated_EYx0_sn
        Estimated_TE_biased = Estimated_EYx1_biased - Estimated_EYx0_biased

        # NIE* nuisance parameters
        pi2_star = (df[X] == 1) * (1 - df['px_z']) / (df['px_z'] * (1 - df['px_z']))
        pi1_star = (df[X] == 0) / (1 - df['px_z'])
        pi2_star_sn = pi2_star / np.mean(pi2_star)
        pi1_star_sn = pi1_star / np.mean(pi1_star)

        # NDE/NIE* standard
        Estimated_EYx1WVx0_pi2Y_star = np.mean((pi2_star * df[Y]))
        Estimated_EYx1WVx0_pi2mu2_star = np.mean((pi2_star * df['mu2_ne*']))
        Estimated_EYx1WVx0_pi1mu2_star = np.mean(pi1_star * df['mu2_ne*'])
        Estimated_EYx1WVx0_pi1mu1_star = np.mean(pi1_star * df['mu1_ne*'])
        Estimated_EYx1WVx0_mu1_star = np.mean(df['mu1_ne*'])
        Estimated_EYx1WVx0_star = (
            (Estimated_EYx1WVx0_pi2Y_star - Estimated_EYx1WVx0_pi2mu2_star) + 
            (Estimated_EYx1WVx0_pi1mu2_star - Estimated_EYx1WVx0_pi1mu1_star) + 
            Estimated_EYx1WVx0_mu1_star
        )
        Estimated_NDE_star = Estimated_EYx1WVx0_star - Estimated_EYx0
        Estimated_NIE_star = Estimated_EYx1 - Estimated_EYx1WVx0_star

        # NDE/NIE* self-normalized
        Estimated_EYx1WVx0_pi2Y_star_sn = np.mean((pi2_star_sn * df[Y]))
        Estimated_EYx1WVx0_pi2mu2_star_sn = np.mean((pi2_star_sn * df['mu2_ne*']))
        Estimated_EYx1WVx0_pi1mu2_star_sn = np.mean(pi1_star_sn * df['mu2_ne*'])
        Estimated_EYx1WVx0_pi1mu1_star_sn = np.mean(pi1_star_sn * df['mu1_ne*'])
        Estimated_EYx1WVx0_star_sn = (
            (Estimated_EYx1WVx0_pi2Y_star_sn - Estimated_EYx1WVx0_pi2mu2_star_sn) + 
            (Estimated_EYx1WVx0_pi1mu2_star_sn - Estimated_EYx1WVx0_pi1mu1_star_sn) + 
            Estimated_EYx1WVx0_mu1_star
        )
        Estimated_NDE_star_sn = Estimated_EYx1WVx0_star_sn - Estimated_EYx0_sn
        Estimated_NIE_star_sn = Estimated_EYx1_sn - Estimated_EYx1WVx0_star_sn

        # do(x) conditional
        Estimated_cond_EYx0 = np.mean(pi_x0 * (df[Y] - df['y_zx0'])) + df['y_zx0']
        Estimated_cond_EYx1 = np.mean(pi_x1 * (df[Y] - df['y_zx1'])) + df['y_zx1']
        Estimated_cond_EYx0_sn = np.mean(pi_x0_sn * (df[Y] - df['y_zx0'])) + df['y_zx0']
        Estimated_cond_EYx1_sn = np.mean(pi_x1_sn * (df[Y] - df['y_zx1'])) + df['y_zx1']

     
        results = {
            # Estimates
            'Estimated_EYx0': [Estimated_EYx0],
            'Estimated_EYx1': [Estimated_EYx1],
            'Estimated_EYx0_sn': [Estimated_EYx0_sn],
            'Estimated_EYx1_sn': [Estimated_EYx1_sn],
            'Estimated_EYx1WVx0': [Estimated_EYx1WVx0],
            'Estimated_EYx1WVx0*': [Estimated_EYx1WVx0_star],
            'Estimated_EYx1WVx0_sn': [Estimated_EYx1WVx0_sn],
            'Estimated_EYx1WVx0*_sn': [Estimated_EYx1WVx0_star_sn],
            'Estimated_TE': [Estimated_TE],
            'Estimated_NDE': [Estimated_NDE],
            'Estimated_NIE': [Estimated_NIE],
            'Estimated_ExpSE_x1': [Estimated_ExpSE_x1],
            'Estimated_ExpSE_x0': [Estimated_ExpSE_x0],
            'Estimated_SE': [Estimated_SE],
            'Estimated_TE_sn': [Estimated_TE_sn],
            'Estimated_NDE_sn': [Estimated_NDE_sn],
            'Estimated_NIE_sn': [Estimated_NIE_sn],
            'Estimated_ExpSE_x1_sn': [Estimated_ExpSE_x1_sn],
            'Estimated_ExpSE_x0_sn': [Estimated_ExpSE_x0_sn],
            'Estimated_SE_sn': [Estimated_SE_sn],
            'Estimated_NDE*': [Estimated_NDE_star],
            'Estimated_NIE*': [Estimated_NIE_star],
            'Estimated_NDE*_sn': [Estimated_NDE_star_sn],
            'Estimated_NIE*_sn': [Estimated_NIE_star_sn],
            'Estimated_TE_biased': [Estimated_TE_biased],
            # Sub-estimates
            'Estimated_NDE_pi2Y': [Estimated_NDE_pi2Y],
            'Estimated_NDE_pi2mu2': [Estimated_NDE_pi2mu2],
            'Estimated_NDE_pi1mu2': [Estimated_NDE_pi1mu2],
            'Estimated_NDE_pi1mu1': [Estimated_NDE_pi1mu1],
            'Estimated_NDE_mu1': [Estimated_NDE_mu1],
            'Estimated_NIE_pi2Y': [Estimated_NIE_pi2Y],
            'Estimated_NIE_pi2mu2': [Estimated_NIE_pi2mu2],
            'Estimated_NIE_pi1mu2': [Estimated_NIE_pi1mu2],
            'Estimated_NIE_pi1mu1': [Estimated_NIE_pi1mu1],
            'Estimated_NIE_mu1': [Estimated_NIE_mu1],
            # Sub-estimates (SN)
            'Estimated_NDE_pi2Y_sn': [Estimated_NDE_pi2Y_sn],
            'Estimated_NDE_pi2mu2_sn': [Estimated_NDE_pi2mu2_sn],
            'Estimated_NDE_pi1mu2_sn': [Estimated_NDE_pi1mu2_sn],
            'Estimated_NDE_pi1mu1_sn': [Estimated_NDE_pi1mu1_sn],
            'Estimated_NDE_mu1_sn': [Estimated_NDE_mu1_sn],
            'Estimated_NIE_pi2Y_sn': [Estimated_NIE_pi2Y_sn],
            'Estimated_NIE_pi2mu2_sn': [Estimated_NIE_pi2mu2_sn],
            'Estimated_NIE_pi1mu2_sn': [Estimated_NIE_pi1mu2_sn],
            'Estimated_NIE_pi1mu1_sn': [Estimated_NIE_pi1mu1_sn],
            'Estimated_NIE_mu1_sn': [Estimated_NIE_mu1_sn],
            # Nuisance parameters
            'Epi_x0': [Epi_x0],
            'Epi_x1': [Epi_x1],
            'Epi2_ne': [Epi2_ne],
            'Epi1_ne': [Epi1_ne],
        }
    return pd.DataFrame(results)

def fit(X, Y, device, binary=True, **kwargs):
    X.columns = (
    X.columns.astype(str)  # ensure all are strings
    .str.replace('[', '(', regex=False)
    .str.replace(']', ')', regex=False)
    .str.replace('<', 'lt_', regex=False)
    .str.replace('>', 'gt_', regex=False))
    
    X = cp.asarray(X)
    Y = cp.asarray(Y)
    model_class = xgb.XGBClassifier if binary else xgb.XGBRegressor
    m = model_class(
        tree_method='hist',
        device=device,
        **kwargs
    ).fit(X, Y)
    return m

def pred(X, m, binary=True, clip=0.0, **kwargs):
    X.columns = (
    X.columns.astype(str)  # ensure all are strings
    .str.replace('[', '(', regex=False)
    .str.replace(']', ')', regex=False)
    .str.replace('<', 'lt_', regex=False)
    .str.replace('>', 'gt_', regex=False))
    
    X = cp.asarray(X)
    if binary:
        y = m.predict_proba(X)[:, 1]
    if not binary:
        y = m.predict(X)
    if clip:
        y = np.clip(y, a_min=clip, a_max=1-clip)
    return y


def rel_err(y_pred, y_true):
    if np.abs(y_true) < 1e-8:
        return None
    return 100 * (y_pred - y_true) / np.abs(y_true)


def test_estimate_effects(all_df, X, Y, W_cols, Z_cols, Y_binary,
                          device, clip=0.0, K=5, bootstraps=30,
                          model_path=None):
    """
    Load and run pre-trained models (from train_estimate_effects) 
    to compute effects and model scores on new data.
    """
    model_list = [
        'px_z', 'px_wz',
        'y_zx0', 'y_zx1',
        'y_wvzx0', 'y_wvzx1',
        'mu2_ne', 'mu1_ne',
        'mu2_ne*', 'mu1_ne*'
    ]

    results_df = pd.DataFrame()
    model_scores_df = pd.DataFrame()

    n = len(all_df)
    all_idxs = np.arange(n)
    synthetic = False

    # Preallocate auxiliary columns
    aux_cols = {c: None for c in model_list}
    df = all_df.copy()
    aux_df = pd.DataFrame(aux_cols, index=df.index)
    df = pd.concat([df, aux_df], axis=1).reset_index(drop=True)

    print('Running inference with pre-trained models...')
    for i in tqdm(range(bootstraps)):
        folds = KFold(n_splits=K, shuffle=True, random_state=i)

        for k_split, (_, ts) in enumerate(folds.split(all_idxs)):
            file_path = os.path.join(model_path, f"models_boot{i}_split{k_split}.joblib")
            if not os.path.exists(file_path):
                print(f"Skipping missing model file: {file_path}")
                continue

            # Load saved model dictionary
            model_objects = joblib.load(file_path)

            df_ts = df.iloc[ts]

            # Run predictions
            df.loc[ts, 'px_z'] = pred(df_ts[Z_cols], model_objects['px_z'], clip=clip)
            df.loc[ts, 'px_wz'] = pred(df_ts[W_cols + Z_cols], model_objects['px_wz'], clip=clip)

            df.loc[ts, 'y_zx0'] = pred(df_ts[Z_cols], model_objects['y_zx0'], binary=Y_binary)
            df.loc[ts, 'y_zx1'] = pred(df_ts[Z_cols], model_objects['y_zx1'], binary=Y_binary)
            df.loc[ts, 'y_wvzx0'] = pred(df_ts[W_cols + Z_cols], model_objects['y_wvzx0'], binary=Y_binary)
            df.loc[ts, 'y_wvzx1'] = pred(df_ts[W_cols + Z_cols], model_objects['y_wvzx1'], binary=Y_binary)

            # Nested components
            df.loc[ts, 'mu2_ne'] = pred(df_ts[W_cols + Z_cols], model_objects['mu2_ne'], binary=Y_binary)
            df.loc[ts, 'mu1_ne'] = pred(df_ts[Z_cols], model_objects['mu1_zx0'], binary=False)
            df.loc[ts, 'mu2_ne*'] = pred(df_ts[Z_cols], model_objects['y_zx1'], binary=Y_binary)
            df.loc[ts, 'mu1_ne*'] = pred(df_ts[Z_cols], model_objects['mu1_zx0_'], binary=False)

            if Y_binary:
                for col in ['mu1_ne', 'mu1_ne*']:
                    df.loc[ts, col] = np.clip(df.loc[ts, col], a_min=0, a_max=1)

        # Compute metrics/effects for this bootstrap
        model_scores = compute_model_scores(df, X, Y, model_list)
        model_scores_df = pd.concat([model_scores_df, model_scores], ignore_index=True)

        results = compute_effects_from_df(df, X, Y, Y_binary, synthetic)
        results_df = pd.concat([results_df, results], ignore_index=True)

    return results_df, model_scores_df
