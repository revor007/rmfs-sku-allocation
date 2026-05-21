import numpy as np
from scipy import sparse as sp

NEG_INF = -1e30


def validate_solution(X, Q, Y, problem):
    X = np.asarray(X, dtype=np.uint8)
    Q = np.asarray(Q, dtype=np.int32)
    Y = np.asarray(Y, dtype=np.int32)
    G = np.asarray(problem["G"], dtype=np.int32)
    p = np.asarray(problem["p"], dtype=np.int32)

    if np.any(Y.sum(axis=0) > G):
        return False
    if np.any(Y < 0) or np.any(Q < 0):
        return False
    if np.any(Y < X):
        return False
    if np.any(Q > p[:, None] * Y):
        return False

    stage_required_slots = np.asarray(problem["stage_required_slots"], dtype=np.int32)
    random_new_indices = np.asarray(problem["random_new_indices"], dtype=np.int32)
    if random_new_indices.size > 0:
        if np.any(Y[random_new_indices].sum(axis=1) < stage_required_slots[random_new_indices]):
            return False
        if np.any(X[random_new_indices].sum(axis=1) > stage_required_slots[random_new_indices]):
            return False

        for idx in random_new_indices:
            allocation = problem["fixed_random_allocation"].get(int(idx))
            if allocation is None:
                return False

            actual_pods = np.flatnonzero(Y[int(idx)])
            expected_pods = np.sort(allocation["pods"])
            if actual_pods.size != expected_pods.size:
                return False
            if not np.array_equal(actual_pods, expected_pods):
                return False
            if not np.array_equal(Y[int(idx), expected_pods], allocation["slots"]):
                return False

    stage2_required_indices = np.asarray(problem["stage2_required_indices"], dtype=np.int32)
    if stage2_required_indices.size > 0:
        if np.any(Y[stage2_required_indices].sum(axis=1) < stage_required_slots[stage2_required_indices]):
            return False
        if np.any(
            Q[stage2_required_indices].sum(axis=1)
            < np.asarray(problem["effective_g"], dtype=np.int32)[stage2_required_indices]
        ):
            return False

    for leader_idx in np.asarray(problem["common_group_leaders"], dtype=np.int32):
        members = problem["common_group_members"][int(leader_idx)]
        leader_x = X[int(leader_idx)]
        leader_y = Y[int(leader_idx)]
        for follower_idx in members[1:]:
            if not np.array_equal(X[int(follower_idx)], leader_x):
                return False
            if not np.array_equal(Y[int(follower_idx)], leader_y):
                return False

    return True


def RMFSobjective(pop, problem):
    X_eval = pop.get("position_bin", None)
    if X_eval is None:
        X_eval = np.round(np.asarray(pop["position"], dtype=np.float32))

    X = np.clip(np.asarray(X_eval), 0, 1).astype(np.uint8)
    Q = np.asarray(pop["quantity_num"], dtype=np.int32)
    Y = np.asarray(pop["compartment_num"], dtype=np.int32)

    feasible = bool(pop.get("is_feasible", True))
    if problem.get("ValidateFeasibility", True) and feasible:
        feasible = validate_solution(X, Q, Y, problem)

    if not feasible:
        pop["is_feasible"] = False
        pop["cost"] = NEG_INF
        return pop

    historical_indices = np.asarray(problem["historical_indices"], dtype=np.int32)
    if historical_indices.size == 0:
        pop["is_feasible"] = True
        pop["cost"] = 0.0
        return pop

    X_hist = sp.csr_matrix(X[historical_indices], dtype=np.float32)
    WX = problem["W_stage2_sparse"].dot(X_hist)
    objective = 0.5 * X_hist.multiply(WX).sum()

    pop["is_feasible"] = True
    pop["cost"] = float(objective)
    return pop


def decode_slot(Y, G):
    Y = np.asarray(Y, dtype=np.int32)
    G = np.asarray(G, dtype=np.int32)

    PN, M = Y.shape
    C = int(np.max(G))
    Z = np.zeros((PN, M, C), dtype=np.uint8)

    for m in range(M):
        next_slot = 0
        for i in range(PN):
            slots_used = int(Y[i, m])
            if slots_used <= 0:
                continue
            Z[i, m, next_slot:next_slot + slots_used] = 1
            next_slot += slots_used

    return Z


def seed_random_allocation(problem, Q, Y):
    for idx, allocation in problem["fixed_random_allocation"].items():
        pods = allocation["pods"]
        if pods.size == 0:
            continue
        Y[int(idx), pods] = allocation["slots"]
        Q[int(idx), pods] = allocation["quantity"]


def build_pod_order(row_binary, remaining_slots, required_capacity_per_pod):
    selected = np.flatnonzero(row_binary > 0)
    if selected.size > 0:
        if required_capacity_per_pod == 1:
            selected_capacity = remaining_slots[selected]
        else:
            selected_capacity = remaining_slots[selected] // required_capacity_per_pod

        eligible_selected = selected[selected_capacity > 0]
        if eligible_selected.size > 0:
            if required_capacity_per_pod == 1:
                order_metric = remaining_slots[eligible_selected]
            else:
                order_metric = remaining_slots[eligible_selected] // required_capacity_per_pod
            return eligible_selected[np.argsort(-order_metric, kind="stable")]

    if required_capacity_per_pod == 1:
        eligible_all = np.flatnonzero(remaining_slots > 0)
        order_metric = remaining_slots[eligible_all] if eligible_all.size > 0 else np.array([])
    else:
        eligible_all = np.flatnonzero(remaining_slots >= required_capacity_per_pod)
        order_metric = (
            remaining_slots[eligible_all] // required_capacity_per_pod
            if eligible_all.size > 0
            else np.array([])
        )

    if eligible_all.size == 0:
        return np.empty(0, dtype=np.int32)

    return eligible_all[np.argsort(-order_metric, kind="stable")]


def repair_solution(pop, problem):
    X_cont = np.asarray(pop["position"], dtype=np.float32)
    X = np.round(X_cont).astype(np.int8)
    X = np.clip(X, 0, 1).astype(np.float32)

    p = np.asarray(problem["p"], dtype=np.int32)
    effective_g = np.asarray(problem["effective_g"], dtype=np.int32)
    remaining_slots = np.asarray(problem["remaining_stage2_capacity"], dtype=np.int32).copy()

    PN, M = X.shape
    Q = np.zeros((PN, M), dtype=np.int32)
    Y = np.zeros((PN, M), dtype=np.int32)

    seed_random_allocation(problem, Q, Y)
    feasible = bool(problem["random_stage_feasible"])

    for entity in problem["stage2_entities"]:
        members = np.asarray(entity["member_indices"], dtype=np.int32)
        leader_idx = int(entity["leader_idx"])
        member_count = int(entity["member_count"])
        slots_per_member = int(entity["slots_per_member"])

        if slots_per_member <= 0:
            continue

        pod_order = build_pod_order(X[leader_idx], remaining_slots, member_count)
        if pod_order.size == 0:
            feasible = False
            continue

        remaining_equal_slots = slots_per_member
        remaining_quantity = effective_g[members].astype(np.int32).copy()

        for pod_idx in pod_order:
            if remaining_equal_slots <= 0:
                break

            shareable_slots = int(remaining_slots[pod_idx] // member_count)
            if shareable_slots <= 0:
                continue

            allocated_slots = min(remaining_equal_slots, shareable_slots)
            if allocated_slots <= 0:
                continue

            for member_pos, member_idx in enumerate(members):
                Y[int(member_idx), int(pod_idx)] = allocated_slots
                slot_capacity = int(allocated_slots) * int(p[int(member_idx)])
                allocated_quantity = min(int(remaining_quantity[member_pos]), slot_capacity)
                Q[int(member_idx), int(pod_idx)] = allocated_quantity
                remaining_quantity[member_pos] -= allocated_quantity

            remaining_slots[pod_idx] -= allocated_slots * member_count
            remaining_equal_slots -= allocated_slots

        if remaining_equal_slots > 0 or np.any(remaining_quantity > 0):
            feasible = False

    X_used = (Y > 0).astype(np.float32)

    pop["position_bin"] = X_used
    pop["quantity_num"] = Q
    pop["compartment_num"] = Y
    pop["is_feasible"] = feasible
    return pop


def build_default_stage_meta(g, p, G):
    PN = int(len(g))
    historical_indices = np.arange(PN, dtype=np.int32)
    stage_required_slots = np.ceil(
        np.asarray(g, dtype=np.float32) / np.asarray(p, dtype=np.float32)
    ).astype(np.int32)
    return {
        "effective_g": np.asarray(g, dtype=np.int32),
        "stage_required_slots": stage_required_slots,
        "historical_indices": historical_indices,
        "common_new_indices": np.empty(0, dtype=np.int32),
        "random_new_indices": np.empty(0, dtype=np.int32),
        "common_group_leaders": np.empty(0, dtype=np.int32),
        "common_group_members": {},
        "group_follower_indices": np.empty(0, dtype=np.int32),
        "stage2_entities": [
            {
                "leader_idx": int(idx),
                "member_indices": np.asarray([idx], dtype=np.int32),
                "member_count": 1,
                "slots_per_member": int(stage_required_slots[idx]),
                "total_load": int(stage_required_slots[idx]),
            }
            for idx in historical_indices
        ],
        "stage2_required_indices": historical_indices,
        "fixed_random_allocation": {},
        "remaining_stage2_capacity": np.asarray(G, dtype=np.int32),
        "random_stage_feasible": True,
        "summary": {
            "historical_skus": PN,
            "common_new_skus": 0,
            "random_new_skus": 0,
            "common_groups": 0,
            "fallback_common_to_random": 0,
            "average_historical_slots": 0,
            "total_required_slots": int(stage_required_slots.sum()),
        },
    }


def RMFSproblem(Dimensions, U, G, g, p, lam, stage_meta=None):
    U = np.asarray(U, dtype=np.float32)
    G = np.asarray(G, dtype=np.int32)
    g = np.asarray(g, dtype=np.int32)
    p = np.asarray(p, dtype=np.int32)

    if stage_meta is None:
        stage_meta = build_default_stage_meta(g, p, G)

    # Current experiment variant uses only the Jaccard matrix in the objective.
    W = U.copy()
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)

    historical_indices = np.asarray(stage_meta["historical_indices"], dtype=np.int32)
    if historical_indices.size > 0:
        W_stage2 = W[np.ix_(historical_indices, historical_indices)]
    else:
        W_stage2 = np.zeros((0, 0), dtype=np.float32)

    W_stage2_sparse = sp.csr_matrix(W_stage2)

    VarMin = np.zeros((Dimensions[0], Dimensions[1]), dtype=np.float32)
    VarMax = np.ones((Dimensions[0], Dimensions[1]), dtype=np.float32)

    for idx, allocation in stage_meta["fixed_random_allocation"].items():
        fixed_row = np.zeros(Dimensions[1], dtype=np.float32)
        fixed_row[allocation["pods"]] = 1.0
        VarMin[int(idx), :] = fixed_row
        VarMax[int(idx), :] = fixed_row

    follower_indices = np.asarray(stage_meta["group_follower_indices"], dtype=np.int32)
    if follower_indices.size > 0:
        VarMin[follower_indices, :] = 0.0
        VarMax[follower_indices, :] = 0.0

    problem = {
        "CostFunction": RMFSobjective,
        "RepairFunction": repair_solution,
        "nVar": Dimensions,
        "U": U,
        "G": G,
        "g": g,
        "p": p,
        "lam": lam,
        "W_stage2_sparse": W_stage2_sparse,
        "ValidateFeasibility": True,
        "VarMin": VarMin,
        "VarMax": VarMax,
        "StopCriterion": None,
        "Max/Min": 1,
        "effective_g": np.asarray(stage_meta["effective_g"], dtype=np.int32),
        "stage_required_slots": np.asarray(stage_meta["stage_required_slots"], dtype=np.int32),
        "historical_indices": historical_indices,
        "common_new_indices": np.asarray(stage_meta["common_new_indices"], dtype=np.int32),
        "random_new_indices": np.asarray(stage_meta["random_new_indices"], dtype=np.int32),
        "common_group_leaders": np.asarray(stage_meta["common_group_leaders"], dtype=np.int32),
        "common_group_members": stage_meta["common_group_members"],
        "group_follower_indices": follower_indices,
        "stage2_entities": stage_meta["stage2_entities"],
        "stage2_required_indices": np.asarray(stage_meta["stage2_required_indices"], dtype=np.int32),
        "fixed_random_allocation": stage_meta["fixed_random_allocation"],
        "remaining_stage2_capacity": np.asarray(stage_meta["remaining_stage2_capacity"], dtype=np.int32),
        "random_stage_feasible": bool(stage_meta["random_stage_feasible"]),
        "StageSummary": stage_meta.get("summary", {}),
        "ProblemStructure": {
            "position": None,
            "position_bin": None,
            "quantity_num": None,
            "compartment_num": None,
            "cost": None,
            "is_feasible": None,
        },
    }
    return problem
