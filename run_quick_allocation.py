import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dataPreparation import load_data_rmfs
from initializer import initialization
from ma import MA
from objective import RMFSproblem


def export_best_solution(results_dir, run_name, best, sku_codes, pod_count):
    x_path = results_dir / f"X_{run_name}.csv"
    y_path = results_dir / f"Y_{run_name}.csv"
    z_path = results_dir / f"Z_{run_name}.csv"
    npz_path = results_dir / f"best_{run_name}.npz"
    csv_path = results_dir / f"best_{run_name}.csv"

    Q = np.asarray(best["quantity_num"], dtype=np.int32)
    p_arr = np.asarray(best["p"], dtype=np.int32)
    sku_arr = np.asarray(sku_codes, dtype=object)
    PN = len(sku_arr)
    M = int(pod_count)

    records = []
    for m in range(M):
        next_slot = 1
        for i in range(PN):
            qty_total = int(Q[i, m])
            if qty_total <= 0:
                continue

            full_slots = qty_total // p_arr[i]
            remainder = qty_total % p_arr[i]

            for _ in range(full_slots):
                records.append(
                    {
                        "pod": m + 1,
                        "slot": next_slot,
                        "item": sku_arr[i],
                        "quantity_in_that_slot": int(p_arr[i]),
                    }
                )
                next_slot += 1

            if remainder > 0:
                records.append(
                    {
                        "pod": m + 1,
                        "slot": next_slot,
                        "item": sku_arr[i],
                        "quantity_in_that_slot": int(remainder),
                    }
                )
                next_slot += 1

    z_table = pd.DataFrame(records)
    z_table.to_csv(z_path, index=False)
    pd.DataFrame(best["position"]).to_csv(x_path, index=False)
    pd.DataFrame(best["compartment_num"]).to_csv(y_path, index=False)
    np.savez(
        npz_path,
        X=best["position"],
        Y=best["compartment_num"],
        PN=PN,
        M=M,
        best_cost=best["cost"],
    )
    pd.DataFrame(
        {
            "best_cost": [best["cost"]],
            "PN": [PN],
            "M": [M],
        }
    ).to_csv(csv_path, index=False)

    return {
        "x_path": x_path,
        "y_path": y_path,
        "z_path": z_path,
        "npz_path": npz_path,
        "csv_path": csv_path,
        "rows": int(len(z_table)),
    }


def main():
    parser = argparse.ArgumentParser(description="Quick single-run FCGMA optimizer export.")
    parser.add_argument("--run-name", default="trial_no_sij_quick", help="Suffix for exported files.")
    parser.add_argument("--max-func-evals", type=int, default=3000)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--iter-print", type=int, default=5)
    parser.add_argument("--init-pop-size", type=int, default=60)
    parser.add_argument("--m-pop-size", type=int, default=30)
    parser.add_argument("--f-pop-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start = time.time()
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results_fcgma"
    results_dir.mkdir(exist_ok=True)

    path_U = base_dir / "jaccard_similarity_matrix.csv"
    path_S = base_dir / "same_cluster_matrix.csv"
    path_min_inv = base_dir / "minimum_inventory.csv"
    path_max_cap = base_dir / "max_comp_number.csv"
    path_stage1 = base_dir / "Clustering" / "bc-k-means-results.csv"

    G_scalar = 40
    print("[quick] loading cutoff-aligned data", flush=True)
    U, S, sku_codes, G, g, p, lam, M, stage_meta = load_data_rmfs(
        path_U,
        path_S,
        path_min_inv,
        G_scalar,
        path_max_cap,
        path_stage1=path_stage1,
        random_seed=args.seed,
    )
    PN = len(sku_codes)
    print(f"[quick] PN={PN}, M={M}, stage={stage_meta.get('summary', {})}", flush=True)

    problem = RMFSproblem([PN, M], U, S, G, g, p, lam, stage_meta=stage_meta)
    problem["InitPopSize"] = int(args.init_pop_size)
    problem["Seed"] = int(args.seed)
    problem["StopCriterion"] = "Function Evaluations"
    problem["ValidateFeasibility"] = True

    print("[quick] building initial population", flush=True)
    initialpop = initialization(problem)
    print("[quick] running MA", flush=True)
    results = MA(
        problem,
        int(args.iter_print),
        int(args.max_iter),
        int(args.max_func_evals),
        0,
        initialpop,
        int(args.m_pop_size),
        int(args.f_pop_size),
        a1=1.0,
        a2=1.5,
        a3=1.5,
        beta=2,
        dance=5,
        fl=1,
        dance_damp=0.99,
        fl_damp=0.99,
        nc=20,
        gmax=0.8,
        gmin=0.4,
        gamma=0.4,
        on_iter=None,
        on_position=None,
    )

    best = problem.get("LastBest")
    if best is None:
        raise RuntimeError("Optimizer finished without a LastBest solution.")

    best["p"] = np.asarray(problem["p"], dtype=np.int32)
    export_info = export_best_solution(results_dir, args.run_name, best, sku_codes, M)

    history = np.asarray(results[3], dtype=float)
    history_path = results_dir / f"objective_history_{args.run_name}.csv"
    pd.DataFrame(
        {
            "iteration": np.arange(len(history)),
            "best_objective": history,
        }
    ).to_csv(history_path, index=False)

    elapsed = time.time() - start
    print(
        f"[quick] done. best_cost={best['cost']:.6f}, elapsed={elapsed:.2f}s, "
        f"Z={export_info['z_path'].name}, rows={export_info['rows']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
