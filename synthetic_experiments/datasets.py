import numpy as np
import pandas as pd

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def he_init(rng, fan_in, fan_out):
    scale = np.sqrt(2.0 / max(1, fan_in))
    W = rng.randn(fan_in, fan_out) * scale
    b = np.zeros((fan_out,))
    return W, b

def build_mlp(rng, in_dim, hidden_dims, out_dim):
    dims = [in_dim] + list(hidden_dims) + [out_dim]
    params = []
    for i in range(len(dims)-1):
        W, b = he_init(rng, dims[i], dims[i+1])
        params.append((W, b))
    return params

def mlp_forward(x, params, activation='tanh'):
    h = x
    for i, (W, b) in enumerate(params):
        h = h @ W + b
        if i < len(params)-1:
            if activation == 'tanh':
                h = np.tanh(h)
            elif activation == 'relu':
                h = np.maximum(h, 0)
            else:
                raise ValueError("unknown activation")
    return h  # final linear output (logits or continuous)


def create_data(dim_dict, lambda_dict, sigma_dict, n=150000, seed=31,
                dict_hidden_sizes = {}, dict_prebuilt_mlps = {},
                threshold = 0.5):
    """
    dict hidden sizes should include tuples for: 
        hidden_U_to_X
        hidden_UZ_to_W
        hidden_X_to_Y
        hidden_W_to_Y
        hidden_Z_to_Y
    if you want to pass in a prebuilt mlp, use the above key and the value can be the prebuilt NN
    in dict_prebuilt_mlps (otherwise leave as empty dict)
    """
    # dimensionality
    w_dim = dim_dict['W']
    z_dim = dim_dict['Z']

    # approximate causal effects
    lam_x = lambda_dict['X']
    lam_w = lambda_dict['W']
    lam_z = lambda_dict['Z']

    # scale of noise
    sigma_u = sigma_dict['U']
    sigma_x_noise = sigma_dict['X']
    sigma_w_noise = sigma_dict['W']
    sigma_y_noise = sigma_dict['Y']
    """
    Simulate data for DAG with X <-> Z (latent confounder U), X->W->Y, X->Y, Z->W, Z->Y.
    Returns:
      data: dict with arrays X (n,1), Z (n,z_dim), W (n,w_dim), y_prob (n,1), Y (n,1 if thresholded else None)
      counterfactuals: dict with Y_x0_prob, Y_x1_prob, Y_x0_Wx1_prob, Y_x1_Wx0_prob, W_x0, W_x1
      params: dict of networks and scalars for reproducibility
    Notes:
      - All probabilities returned are on [0,1] (no thresholding) unless return_thresholded=True.
      - To compute NDE/NIE use averages of these probabilities (see example).
    """
    rng = np.random.RandomState(seed)

    # ---- sample noise variables ----
    # U is shared latent parent causing correlation between X and Z
    np.random.seed(seed+1)
    U = np.random.randn(n, 1) * sigma_u  # (n,1)
    np.random.seed(seed+2)
    U_x_noise = np.random.randn(n, 1) * sigma_x_noise  # extra noise that influences X besides U
    np.random.seed(seed+3)
    U_w_noise = np.random.randn(n, w_dim) * sigma_w_noise
    np.random.seed(seed+4)
    U_y_noise = np.random.randn(n, 1) * sigma_y_noise

    # ---- build networks ----
    # U -> X 
    if 'net_U_to_X' in dict_prebuilt_mlps.keys():
        net_U_to_X = dict_prebuilt_mlps['net_U_to_X']
    else:
        net_U_to_X = build_mlp(rng, 1, dict_hidden_sizes['hidden_U_to_X'], 1)

    # generate Z as tanh (nonlinear function) of (U plus noise)
    # ensure variance stable by dividing by sqrt(z_dim)
    np.random.seed(seed+5)
    Z_noise = np.random.randn(n, z_dim) * 0.5
    Z = np.tanh(np.repeat(U, z_dim, axis=1) + Z_noise)
    print(f'Z Shape: {Z.shape}')

    # X is a binary variable generated from U (not from Z)
    logit_x = mlp_forward(U, net_U_to_X) + U_x_noise
    p_x = sigmoid(logit_x)
    X = (p_x > 0.5).astype(float)
    print(f'X Shape: {X.shape}')

    # ---- W | X, Z ----
    if 'net_XZ_to_W' in dict_prebuilt_mlps.keys():
        net_XZ_to_W = dict_prebuilt_mlps['net_XZ_to_W']
    else:
        net_XZ_to_W = build_mlp(rng, 1 + z_dim, dict_hidden_sizes['hidden_UZ_to_W'], w_dim)

    XZ = np.concatenate([X, Z], axis=1)
    W_det = mlp_forward(XZ, net_XZ_to_W)  # deterministic mapping
    # scale W to keep magnitude stable across w_dim
    W = W_det / np.sqrt(max(1, w_dim)) + U_w_noise
    print(f'W Shape: {W.shape}')

    # ---- Separable contributions into Y: gX(X), gW(W), gZ(Z) ----
    if 'net_X_to_Y' in dict_prebuilt_mlps.keys():
        net_X_to_Y = dict_prebuilt_mlps['net_X_to_Y']
    else:
        net_X_to_Y = build_mlp(rng, 1, dict_hidden_sizes['hidden_X_to_Y'], 1)

    if 'net_W_to_Y' in dict_prebuilt_mlps.keys():
        net_W_to_Y = dict_prebuilt_mlps['net_W_to_Y']
    else:
        net_W_to_Y = build_mlp(rng, w_dim, dict_hidden_sizes['hidden_W_to_Y'], 1)

    if 'net_Z_to_Y' in dict_prebuilt_mlps.keys():
        net_Z_to_Y = dict_prebuilt_mlps['net_Z_to_Y']
    else:
        net_Z_to_Y = build_mlp(rng, z_dim, dict_hidden_sizes['hidden_Z_to_Y'], 1)

    gX = mlp_forward(X, net_X_to_Y) / 1.0          # (n,1)
    gW = mlp_forward(W, net_W_to_Y) / np.sqrt(max(1, w_dim))  # normalize by sqrt(w_dim)
    gZ = mlp_forward(Z, net_Z_to_Y) / np.sqrt(max(1, z_dim))

    # Compose Y logit as a linear combination of these path contributions + noise
    logit_y = lam_x * gX + lam_w * gW + lam_z * gZ + U_y_noise

    # optional threshold to obtain binary Y (but prefer working with probabilities)
    Y = sigmoid(logit_y)
    if threshold is not None:
        Y = Y > threshold
    print(f'Y Shape: {Y.shape}')
    
    # ---- Counterfactual worlds ----
    # define do(X=0) and do(X=1) by forcing X and recomputing downstream nodes,
    # re-using the SAME W and Y exogenous noises (U_w_noise and U_y_noise)
    X0 = np.zeros_like(X)
    X1 = np.ones_like(X)

    # W under do(X=0/1)
    W_x0_det = mlp_forward(np.concatenate([X0, Z], axis=1), net_XZ_to_W) / np.sqrt(max(1, w_dim))
    W_x1_det = mlp_forward(np.concatenate([X1, Z], axis=1), net_XZ_to_W) / np.sqrt(max(1, w_dim))
    W_x0 = W_x0_det + U_w_noise
    W_x1 = W_x1_det + U_w_noise

    # Y probabilities in different combinations (re-using U_y_noise)
    def y_prob_with(X_in, W_in):
        gx = mlp_forward(X_in, net_X_to_Y)
        gw = mlp_forward(W_in, net_W_to_Y) / np.sqrt(max(1, w_dim))
        gz = gZ  # same as before (Z unchanged)
        logit = lam_x * gx + lam_w * gw + lam_z * gz + U_y_noise
        y_prob = sigmoid(logit)
        if threshold is not None:
            y_prob = y_prob > threshold
        return y_prob

    Y_x0_prob = y_prob_with(X0, W_x0)
    Y_x1_prob = y_prob_with(X1, W_x1)
    Y_x0_Wx1_prob = y_prob_with(X0, W_x1)  # X=0 but W set to what it would be under X=1
    Y_x1_Wx0_prob = y_prob_with(X1, W_x0)

    
    W_cols = [f'W{i+1}' for i in range(w_dim)] 
    Z_cols = [f'Z{i+1}' for i in range(z_dim)] 
    data_df = pd.DataFrame(columns = ['X', 'Y'])
    data_df['X'] = X.reshape(-1)
    data_df['Y'] = Y
    data_df[Z_cols] = Z
    data_df[W_cols] = W

    data_df['Y[x0]'] = Y_x0_prob
    data_df['Y[x1]'] = Y_x1_prob
    data_df['Y[x1Wx0]'] = Y_x1_Wx0_prob
    data_df['Y[x0Wx1]'] = Y_x0_Wx1_prob

    params = {
        'net_U_to_X': net_U_to_X,
        'net_XZ_to_W': net_XZ_to_W,
        'net_X_to_Y': net_X_to_Y,
        'net_W_to_Y': net_W_to_Y,
        'net_Z_to_Y': net_Z_to_Y,
        'lam_x': lam_x, 'lam_w': lam_w, 'lam_z': lam_z,
        'seed': seed, 'z_dim': z_dim, 'w_dim': w_dim
    }

    return data_df, params

def compute_data_effects(mc_df, X, Y):
    mc_df_x0 = mc_df[mc_df[X] == 0]
    mc_df_x1 = mc_df[mc_df[X] == 1]
    EYx0 = np.mean(mc_df[f'{Y}[x0]'])
    EYx1 = np.mean(mc_df[f'{Y}[x1]'])
    EYx1Wx0 = np.mean(mc_df[f'{Y}[x1Wx0]'])
    TV = np.mean(mc_df_x1[Y]) - np.mean(mc_df_x0[Y])
    TE = np.mean(mc_df[f'{Y}[x1]']) - np.mean(mc_df[f'{Y}[x0]'])
    NDE = np.mean(mc_df[f'{Y}[x1Wx0]']) - np.mean(mc_df[f'{Y}[x0]'])
    NIE = np.mean(mc_df[f'{Y}[x1]']) - np.mean(mc_df[f'{Y}[x1Wx0]'])
    EYx0_obs = np.mean(mc_df_x0[Y])
    EYx1_obs = np.mean(mc_df_x1[Y])
    expse_x0 = np.mean(mc_df_x0[Y])-np.mean(mc_df[f'{Y}[x0]'])
    expse_x1 = np.mean(mc_df_x1[Y])-np.mean(mc_df[f'{Y}[x1]'])
    SE = expse_x1-expse_x0
    gt_effects = (EYx0, EYx1, EYx1Wx0, TV, TE, NDE, NIE, EYx0_obs, EYx1_obs, expse_x0, expse_x1, SE)

    print(f'{Y}[x0]      : {EYx0:.4f}')
    print(f'{Y}[x1]      : {EYx1:.4f}')
    print(f'{Y}[x1Wx0]  : {EYx1Wx0:.4f}')
    print('*'*40)
    print(f'Total variation (TV)         : {TV:.4f}')
    print(f'Total effect (TE)            : {TE:.4f}')
    print(f'Natural direct effect (NDE)  : {NDE:.4f}')
    print(f'Natural indirect effect (NIE): {NIE:.4f}')
    print(f'Spurious Effect (SE)         : {SE:.4f}')
    print('*'*40)

    return gt_effects


    