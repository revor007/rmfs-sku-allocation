import pandas as pd
import numpy as np


def load_data(path_u, path_s, path_min_inv, G_scalar, path_max_cap):
    # load U and S
    U_df = pd.read_csv(path_u, sep=';', decimal=',', engine='python', index_col=0)
    S_df = pd.read_csv(path_s, sep=';', decimal=',', engine='python', index_col=0)

    # clean SKU labels
    U_df.index = U_df.index.astype(str).str.strip()
    U_df.columns = U_df.columns.astype(str).str.strip()
    S_df.index = S_df.index.astype(str).str.strip()
    S_df.columns = S_df.columns.astype(str).str.strip()

    # load minimum inventory
    g_df = pd.read_csv(path_min_inv, sep=';', decimal=',', engine='python')
    if 'item_code' in g_df.columns:
        g_df['item_code'] = g_df['item_code'].astype(str).str.strip()
        g_df = g_df.set_index('item_code')
    else:
        g_df = g_df.set_index(g_df.columns[0])
        g_df.index = g_df.index.astype(str).str.strip()

    if 'minimum_inventory' not in g_df.columns:
        raise KeyError("Column 'minimum_inventory' was not found in minimum inventory file.")

    # load max quantity per slot for each SKU
    p_df = pd.read_csv(path_max_cap, sep=None, engine='python')
    if 'item_code' in p_df.columns:
        p_df['item_code'] = p_df['item_code'].astype(str).str.strip()
        p_df = p_df.set_index('item_code')
    else:
        p_df = p_df.set_index(p_df.columns[0])
        p_df.index = p_df.index.astype(str).str.strip()

    if 'max_comp_number' not in p_df.columns:
        raise KeyError("Column 'max_comp_number' was not found in max capacity file.")

    # keep only common SKUs
    common_skus = sorted(set(U_df.index) & set(S_df.index) & set(g_df.index) & set(p_df.index))
    if len(common_skus) == 0:
        raise ValueError("No common SKUs were found across U, S, minimum inventory, and max capacity files.")

    # align all data
    U_df = U_df.reindex(index=common_skus, columns=common_skus)
    S_df = S_df.reindex(index=common_skus, columns=common_skus)
    g_df = g_df.reindex(index=common_skus)
    p_df = p_df.reindex(index=common_skus)

    # extract arrays
    U = U_df.to_numpy(dtype=np.float32)
    S = S_df.to_numpy(dtype=np.float32)
    g = np.ceil(g_df['minimum_inventory'].to_numpy()).astype(np.int32)
    p = np.floor(p_df['max_comp_number'].to_numpy()).astype(np.int32)

    if np.any(p <= 0):
        raise ValueError("All p values must be positive.")

    # compute PN and M
    required_slots = np.ceil(g / p).astype(np.int32)
    PN = len(common_skus)
    M = int(np.ceil(np.sum(required_slots) / G_scalar))

    # build G vector
    G = np.full(M, G_scalar, dtype=np.int32)

    return U, S, common_skus, G, g, p, M


def load_data_rmfs(path_u, path_s, path_min_inv, G_scalar, path_max_cap, lam=0.5):
    U, S, sku_codes, G, g, p, M = load_data(
        path_u, path_s, path_min_inv, G_scalar, path_max_cap
    )
    return U, S, sku_codes, G, g, p, lam, M