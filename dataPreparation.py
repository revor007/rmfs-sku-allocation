from pathlib import Path

import numpy as np
import pandas as pd

from experiment_context import find_column, load_experiment_context, normalize_item_code


def allocate_random_new_products(random_new_indices, stage_required_slots, p, g, G, random_seed):
    random_new_indices = np.asarray(random_new_indices, dtype=np.int32)
    allocation = {}
    remaining_capacity = np.asarray(G, dtype=np.int32).copy()
    feasible = True

    if random_new_indices.size == 0:
        return allocation, remaining_capacity, feasible

    rng = np.random.default_rng(random_seed)
    for idx in rng.permutation(random_new_indices):
        slots_needed = int(stage_required_slots[idx])
        used_pods = []
        used_slots = []

        if slots_needed > 0:
            for pod_idx in rng.permutation(remaining_capacity.size):
                if slots_needed <= 0:
                    break

                free_slots = int(remaining_capacity[pod_idx])
                if free_slots <= 0:
                    continue

                assigned_slots = min(slots_needed, free_slots)
                used_pods.append(int(pod_idx))
                used_slots.append(int(assigned_slots))
                remaining_capacity[pod_idx] -= assigned_slots
                slots_needed -= assigned_slots

        if slots_needed > 0:
            feasible = False

        pods = np.asarray(used_pods, dtype=np.int32)
        slots = np.asarray(used_slots, dtype=np.int32)
        quantity = np.zeros(slots.shape[0], dtype=np.int32)

        if pods.size > 0:
            order = np.argsort(pods, kind="stable")
            pods = pods[order]
            slots = slots[order]

        if slots.size > 0:
            target_quantity = int(max(0, g[idx]))
            if target_quantity == 0:
                quantity = slots * int(p[idx])
            else:
                remaining_quantity = target_quantity
                quantity_unsorted = np.zeros(slots.shape[0], dtype=np.int32)
                for pos, slot_count in enumerate(slots):
                    slot_capacity = int(slot_count) * int(p[idx])
                    assigned_quantity = min(remaining_quantity, slot_capacity)
                    quantity_unsorted[pos] = assigned_quantity
                    remaining_quantity -= assigned_quantity
                quantity = quantity_unsorted

        allocation[int(idx)] = {
            "pods": pods,
            "slots": slots,
            "quantity": quantity,
        }

    return allocation, remaining_capacity, feasible


def build_stage_allocation_metadata(sku_codes, g, p, G_scalar, path_stage1=None, random_seed=42):
    PN = len(sku_codes)
    sku_to_idx = {sku: idx for idx, sku in enumerate(sku_codes)}

    random_new_indices = set()
    common_groups = {}
    fallback_common_to_random = []

    if path_stage1 is not None and Path(path_stage1).exists():
        bc_df = pd.read_csv(path_stage1, sep=";", encoding="utf-8-sig", decimal=",")
        bc_df["new_product_code"] = bc_df["new_product_code"].map(normalize_item_code)
        bc_df["corresponding_historical_product"] = bc_df[
            "corresponding_historical_product"
        ].map(normalize_item_code)
        bc_df["allocation_type"] = bc_df["allocation_type"].astype(str).str.strip()

        for row in bc_df.itertuples(index=False):
            new_idx = sku_to_idx.get(row.new_product_code)
            hist_idx = sku_to_idx.get(row.corresponding_historical_product) if row.corresponding_historical_product else None
            if new_idx is None:
                continue

            if (
                row.allocation_type == "common_allocation"
                and hist_idx is not None
                and hist_idx != new_idx
            ):
                common_groups.setdefault(int(hist_idx), set()).add(int(new_idx))
            else:
                random_new_indices.add(int(new_idx))
                if row.allocation_type == "common_allocation":
                    fallback_common_to_random.append(row.new_product_code)

    common_groups = {
        int(hist_idx): sorted(followers)
        for hist_idx, followers in common_groups.items()
    }

    common_new_indices = sorted(
        {follower for followers in common_groups.values() for follower in followers}
    )
    random_new_indices = sorted(random_new_indices - set(common_new_indices))
    historical_indices = sorted(
        set(range(PN)) - set(random_new_indices) - set(common_new_indices)
    )

    effective_g = np.asarray(g, dtype=np.int32).copy()
    for hist_idx, followers in common_groups.items():
        effective_g[np.asarray(followers, dtype=np.int32)] = effective_g[int(hist_idx)]

    stage_required_slots = np.ceil(effective_g / p).astype(np.int32)
    historical_slots = stage_required_slots[np.asarray(historical_indices, dtype=np.int32)]
    average_historical_slots = int(max(1, np.ceil(historical_slots.mean()))) if historical_slots.size else 1

    if random_new_indices:
        stage_required_slots[np.asarray(random_new_indices, dtype=np.int32)] = average_historical_slots

    common_group_members = {}
    common_group_leaders = sorted(common_groups)
    group_follower_indices = []
    stage2_entities = []

    for hist_idx in common_group_leaders:
        members = np.asarray([hist_idx, *common_groups[hist_idx]], dtype=np.int32)
        equal_slots = int(stage_required_slots[members].max())
        stage_required_slots[members] = equal_slots
        common_group_members[int(hist_idx)] = members
        group_follower_indices.extend(common_groups[hist_idx])
        stage2_entities.append(
            {
                "leader_idx": int(hist_idx),
                "member_indices": members,
                "member_count": int(members.size),
                "slots_per_member": equal_slots,
                "total_load": int(equal_slots * members.size),
            }
        )

    singleton_historical_indices = sorted(set(historical_indices) - set(common_group_leaders))
    for hist_idx in singleton_historical_indices:
        slots_per_member = int(stage_required_slots[int(hist_idx)])
        stage2_entities.append(
            {
                "leader_idx": int(hist_idx),
                "member_indices": np.asarray([hist_idx], dtype=np.int32),
                "member_count": 1,
                "slots_per_member": slots_per_member,
                "total_load": slots_per_member,
            }
        )

    stage2_entities.sort(
        key=lambda entity: (-entity["total_load"], -entity["member_count"], entity["leader_idx"])
    )

    total_required_slots = int(stage_required_slots[np.asarray(random_new_indices, dtype=np.int32)].sum())
    total_required_slots += int(sum(entity["total_load"] for entity in stage2_entities))
    M = max(1, int(np.ceil(total_required_slots / G_scalar)))
    G = np.full(M, G_scalar, dtype=np.int32)

    fixed_random_allocation, remaining_stage2_capacity, random_stage_feasible = allocate_random_new_products(
        random_new_indices=random_new_indices,
        stage_required_slots=stage_required_slots,
        p=p,
        g=effective_g,
        G=G,
        random_seed=random_seed,
    )

    stage2_required_indices = np.asarray(
        sorted(set(historical_indices) | set(common_new_indices)),
        dtype=np.int32,
    )

    return {
        "effective_g": effective_g.astype(np.int32),
        "stage_required_slots": stage_required_slots.astype(np.int32),
        "historical_indices": np.asarray(historical_indices, dtype=np.int32),
        "common_new_indices": np.asarray(common_new_indices, dtype=np.int32),
        "random_new_indices": np.asarray(random_new_indices, dtype=np.int32),
        "common_group_leaders": np.asarray(common_group_leaders, dtype=np.int32),
        "common_group_members": common_group_members,
        "group_follower_indices": np.asarray(sorted(group_follower_indices), dtype=np.int32),
        "stage2_entities": stage2_entities,
        "stage2_required_indices": stage2_required_indices,
        "fixed_random_allocation": fixed_random_allocation,
        "remaining_stage2_capacity": remaining_stage2_capacity.astype(np.int32),
        "random_stage_feasible": bool(random_stage_feasible),
        "average_historical_slots": int(average_historical_slots),
        "fallback_common_to_random": fallback_common_to_random,
        "G": G,
        "M": int(M),
        "summary": {
            "historical_skus": int(len(historical_indices)),
            "common_new_skus": int(len(common_new_indices)),
            "random_new_skus": int(len(random_new_indices)),
            "common_groups": int(len(common_group_leaders)),
            "fallback_common_to_random": int(len(fallback_common_to_random)),
            "average_historical_slots": int(average_historical_slots),
            "total_required_slots": int(total_required_slots),
        },
    }


def load_data(path_u, path_min_inv, G_scalar, path_max_cap, path_stage1=None, random_seed=42):
    U_df = pd.read_csv(path_u, sep=";", decimal=",", engine="python", index_col=0)

    U_df.index = U_df.index.map(normalize_item_code)
    U_df.columns = U_df.columns.map(normalize_item_code)

    g_df = pd.read_csv(path_min_inv, sep=";", decimal=",", engine="python")
    
    g_code_col = find_column(g_df.columns, ["item_code"])
    g_value_col = find_column(g_df.columns, ["minimum_inventory"])
    g_df[g_code_col] = g_df[g_code_col].map(normalize_item_code)
    g_df = g_df.set_index(g_code_col)
    if g_value_col not in g_df.columns:
        raise KeyError("Column 'minimum_inventory' was not found in minimum inventory file.")

    p_df = pd.read_csv(path_max_cap, sep=None, engine="python")
    p_code_col = find_column(p_df.columns, ["item_code"])
    p_value_col = find_column(p_df.columns, ["max_fit", "max_comp_number"])
    p_df[p_code_col] = p_df[p_code_col].map(normalize_item_code)
    p_df = p_df.set_index(p_code_col)
    if p_value_col not in p_df.columns:
        raise KeyError("Column 'max_fit' or 'max_comp_number' was not found in max capacity file.")

    eligible_skus = set(load_experiment_context(Path(path_u).resolve().parent).eligible_skus)
    common_skus = sorted(
        set(U_df.index)
        & set(g_df.index)
        & set(p_df.index)
        & eligible_skus
    )
    if not common_skus:
        raise ValueError(
            "No common SKUs were found across U, minimum inventory, max capacity, "
            "and the cutoff-aligned eligible SKU universe."
        )

    U_df = U_df.reindex(index=common_skus, columns=common_skus)
    g_df = g_df.reindex(index=common_skus)
    p_df = p_df.reindex(index=common_skus)

    U = U_df.to_numpy(dtype=np.float32)
    g = np.ceil(g_df[g_value_col].to_numpy()).astype(np.int32)
    p = np.floor(pd.to_numeric(p_df[p_value_col], errors="coerce").to_numpy()).astype(np.int32)

    if np.any(p <= 0):
        raise ValueError("All p values must be positive.")
    if G_scalar <= 0:
        raise ValueError("G_scalar must be positive.")

    stage_meta = build_stage_allocation_metadata(
        sku_codes=common_skus,
        g=g,
        p=p,
        G_scalar=G_scalar,
        path_stage1=path_stage1,
        random_seed=random_seed,
    )

    effective_g = stage_meta["effective_g"].copy()
    G = stage_meta["G"].copy()
    M = int(stage_meta["M"])

    return U, common_skus, G, effective_g, p, M, stage_meta


def load_data_rmfs(path_u, path_min_inv, G_scalar, path_max_cap, lam=0.5, path_stage1=None, random_seed=42):
    U, sku_codes, G, g, p, M, stage_meta = load_data(
        path_u=path_u,
        path_min_inv=path_min_inv,
        G_scalar=G_scalar,
        path_max_cap=path_max_cap,
        path_stage1=path_stage1,
        random_seed=random_seed,
    )
    return U, sku_codes, G, g, p, lam, M, stage_meta
