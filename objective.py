# %%
# Import libraries
import numpy as np
from scipy import sparse as sp

NEG_INF = -1e30


# %%
# Define objective function and constraint for RMFS
def RMFSobjective(pop, problem):
    # Evaluate objective on decoded binary support when available.
    # This keeps MA movement continuous while feasibility/cost stay discrete.
    X_eval = pop.get('position_bin', None)
    if X_eval is None:
        X_eval = np.round(np.asarray(pop['position'], dtype=np.float32))
    X = np.clip(np.asarray(X_eval), 0, 1).astype(np.uint8)
    Q = np.asarray(pop['quantity_num'], dtype=np.int32)
    Y = np.asarray(pop['compartment_num'], dtype=np.int32)
    G = np.asarray(problem['G'], dtype=np.int32)
    g = np.asarray(problem['g'], dtype=np.int32)
    p = np.asarray(problem['p'], dtype=np.int32)

    feasible = (
        pop.get('is_feasible', True)
        and np.all(Y.sum(axis=0) <= G)
        and np.all(Q.sum(axis=1) >= g)
        and np.all(Q <= p[:, None] * Y)
        and np.all(Y >= X)
        and np.all(Q >= 0)
        and np.all(Y >= 0)
    )

    if problem.get('ValidateFeasibility', True) and not feasible:
        pop['cost'] = NEG_INF
        return pop

    # Sparse evaluation
    X_csr = sp.csr_matrix(X, dtype=np.float32)
    WX = problem['W_sparse'].dot(X_csr)

    objective = 0.5 * X_csr.multiply(WX).sum()
    pop['cost'] = float(objective)
    return pop


# %%
def decode_slot(Y, G):
    """
    Z[i, m, s] = 1 if slot s in pod m is occupied by SKU i.
    This is reporting-only and should be used after optimization ends.
    """
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


# %%
def repair_solution(pop, problem):
    # Keep continuous X for MA dynamics; derive a temporary binary decode for repair.
    X_cont = np.asarray(pop['position'], dtype=np.float32)
    X = np.round(X_cont).astype(np.int8)
    X = np.clip(X, 0, 1).astype(np.float32)

    g = np.asarray(problem['g'], dtype=np.int32)
    G = np.asarray(problem['G'], dtype=np.int32)
    p = np.asarray(problem['p'], dtype=np.int32)

    PN, M = X.shape
    Q = np.zeros((PN, M), dtype=np.int32)
    Y = np.zeros((PN, M), dtype=np.int32)

    # remaining_slots[m] = how many slots are still free in pod m
    remaining_slots = G.copy()
    feasible = True

    # Fill each SKU's demand across already-selected pods only
    for i in range(PN):
        demand = int(max(0, g[i]))
        if demand == 0:
            continue

        if p[i] <= 0:
            feasible = False
            continue

        selected = np.where(X[i, :] == 1)[0]
        if selected.size == 0:
            feasible = False
            continue

        # Prefer pods with more remaining free slots
        selected_order = selected[np.argsort(-remaining_slots[selected])]

        for m in selected_order:
            if demand <= 0:
                break

            if remaining_slots[m] <= 0:
                continue

            max_qty = int(remaining_slots[m]) * int(p[i])
            if max_qty <= 0:
                continue

            requested_qty = min(demand, max_qty)
            if requested_qty <= 0:
                continue

            old_slots = Y[i, m]
            prev_qty = int(Q[i, m])
            Q[i, m] += requested_qty
            new_slots = int(np.ceil(Q[i, m] / p[i]))
            extra_slots = new_slots - old_slots

            # Safety guard
            if extra_slots > remaining_slots[m]:
                extra_slots = int(remaining_slots[m])
                new_slots = old_slots + extra_slots
                Q[i, m] = min(Q[i, m], new_slots * int(p[i]))

            # Decrease remaining demand by the quantity actually allocated in this pod.
            actual_qty = int(max(0, Q[i, m] - prev_qty))

            Y[i, m] = new_slots
            remaining_slots[m] -= extra_slots
            demand -= actual_qty

        if demand > 0:
            feasible = False

    # Actual used support after decoding
    X_used = (Q > 0).astype(np.float32)

    # Do not overwrite continuous position; store decoded support separately.
    pop['position_bin'] = X_used
    pop['quantity_num'] = Q
    pop['compartment_num'] = Y
    pop['is_feasible'] = feasible
    return pop


# %%
# Create a problem builder
def RMFSproblem(Dimensions, U, S, G, g, p, lam):
    U = np.asarray(U, dtype=np.float32)
    S = np.asarray(S, dtype=np.float32)
    G = np.asarray(G, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)

    # Build reward
    W = (1.00 - lam) * U + lam * S

    # Enforce symmetry and zero diagonal
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)

    W_sparse = sp.csr_matrix(W)

    problem = {
        'CostFunction': RMFSobjective,
        'RepairFunction': repair_solution,
        'nVar': Dimensions,

        'U': U,
        'S': S,
        'G': G,
        'g': g,
        'p': p,
        'lam': lam,

        'W_sparse': W_sparse,
        'ValidateFeasibility': True,

        'VarMin': np.zeros((Dimensions[0], Dimensions[1]), dtype=np.float32),
        'VarMax': np.ones((Dimensions[0], Dimensions[1]), dtype=np.float32),
        'StopCriterion': None,
        'Max/Min': 1,
        'ProblemStructure': {
            'position': None,
            'position_bin': None,
            'quantity_num': None,
            'compartment_num': None,
            'cost': None,
        }
    }
    return problem