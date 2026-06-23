from __future__ import annotations

import contextlib
import importlib
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from shared_order_stream import ensure_shared_bootstrap_order_file


CSV_SEPARATOR = os.environ.get("FULL_POSTT_CSV_SEPARATOR", ";")
CSV_ENCODING = "utf-8-sig"
PROGRESS_ENABLED = os.environ.get("FULL_POSTT_ENABLE_PROGRESS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
PROGRESS_TICKS = max(1.0, float(os.environ.get("FULL_POSTT_PROGRESS_TICKS", "100")))
PROGRESS_SECONDS = max(1.0, float(os.environ.get("FULL_POSTT_PROGRESS_SECONDS", "30")))
RUN_COUNT_ENV = "FULL_POSTT_RUN_COUNT"
ORDER_MODE_ENV = "FULL_POSTT_ORDER_MODE"
BOOTSTRAP_BASE_SEED_ENV = "FULL_POSTT_BOOTSTRAP_BASE_SEED"
BOOTSTRAP_ARRIVAL_MODE_ENV = "FULL_POSTT_BOOTSTRAP_ARRIVAL_MODE"
BOOTSTRAP_N_ORDERS_ENV = "FULL_POSTT_BOOTSTRAP_N_ORDERS"
BOOTSTRAP_INCREMENT_EVERY_ENV = "FULL_POSTT_BOOTSTRAP_INCREMENT_EVERY"
BOOTSTRAP_SHARED_ORDER_PATH_ENV = "FULL_POSTT_SHARED_BOOTSTRAP_ORDER_PATH"
DIAGNOSTIC_CHECKPOINTS_ENV = "FULL_POSTT_DIAGNOSTIC_CHECKPOINTS"
CHECKPOINT_TICKS_ENV = "FULL_POSTT_CHECKPOINT_TICKS"
POD_LOCATION_MODE_ENV = "FULL_POSTT_POD_LOCATION_MODE"
POD_LOCATION_BASE_SEED_ENV = "FULL_POSTT_POD_LOCATION_BASE_SEED"
POD_LOCATION_FIXED_ENV = "FULL_POSTT_POD_LOCATION_FIXED"
POD_LOCATION_INCREMENT_EVERY_ENV = "FULL_POSTT_POD_LOCATION_INCREMENT_EVERY"
RUNTIME_POD_LOCATION_POLICY_ENV = "RMFS_RUNTIME_POD_LOCATION_POLICY"
RUNTIME_POD_LOCATION_SEED_ENV = "RMFS_RUNTIME_POD_LOCATION_SEED"


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", **kwargs)


def ensure_runtime_input_files(run_root: Path) -> None:
    data_dir = run_root / "data"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"
    legacy_input_dir = data_dir / "yohana" / "input"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if legacy_input_dir.exists():
        for legacy_file in legacy_input_dir.iterdir():
            if not legacy_file.is_file():
                continue

            target = input_dir / legacy_file.name
            if not target.exists():
                shutil.copy2(legacy_file, target)

    # Some clones only contain the older Yohana items_dictionary schema, which
    # lacks the newer `max_fit` column expected by the shared RMFS generator.
    # When that happens, rebuild the live input dictionary from the prepared
    # `data/output/items.csv` artifact instead.
    input_items_dictionary = input_dir / "items_dictionary.csv"
    output_items = output_dir / "items.csv"

    rebuild_items_dictionary = not input_items_dictionary.exists()
    if input_items_dictionary.exists():
        try:
            existing_columns = read_csv_auto(input_items_dictionary, nrows=0).columns
            rebuild_items_dictionary = "max_fit" not in existing_columns
        except Exception:
            rebuild_items_dictionary = True

    if rebuild_items_dictionary and output_items.exists():
        items_df = read_csv_auto(output_items)

        # Older writer variants sometimes persisted `item_id` as an unnamed index.
        unnamed_columns = [col for col in items_df.columns if str(col).startswith("Unnamed:")]
        if unnamed_columns:
            items_df = items_df.drop(columns=unnamed_columns, errors="ignore")

        items_df.to_csv(input_items_dictionary, index=False)


def load_input_summary(run_root: Path) -> dict[str, object]:
    summary_path = run_root / "data" / "output" / "cutoff_experiment_input_summary.csv"
    summary: dict[str, object] = {}

    if summary_path.exists():
        try:
            summary_df = read_csv_auto(summary_path)
            if {"metric", "value"}.issubset(summary_df.columns):
                for row in summary_df.itertuples(index=False):
                    summary[str(row.metric)] = row.value
        except Exception:
            summary = {}

    if "pods_used" not in summary:
        pods_path = run_root / "data" / "output" / "pods.csv"
        if pods_path.exists():
            try:
                pods_df = read_csv_auto(pods_path)
                if "pod_id" in pods_df.columns:
                    summary["pods_used"] = int(pods_df["pod_id"].nunique())
                summary["occupied_slots"] = int(len(pods_df))
            except Exception:
                pass

    if "physical_pods_available" not in summary:
        generated_pod_path = run_root / "data" / "output" / "generated_pod.csv"
        if generated_pod_path.exists():
            try:
                grid = pd.read_csv(generated_pod_path, header=None)
                summary["physical_pods_available"] = int((grid == 1).sum().sum())
            except Exception:
                pass

    return summary


def format_summary_value(value: object) -> str:
    if pd.isna(value):
        return "n/a"

    text = str(value).strip()
    try:
        numeric_value = float(text)
    except Exception:
        return text

    if numeric_value.is_integer():
        return str(int(numeric_value))
    return text


def print_input_summary(run_root: Path, replication_index: int, run_count: int) -> None:
    summary = load_input_summary(run_root)
    if not summary:
        return

    ordered_keys = [
        "historical_skus",
        "new_skus",
        "pods_used",
        "occupied_slots",
        "physical_pods_available",
    ]
    summary_parts = [
        f"{key}={format_summary_value(summary[key])}"
        for key in ordered_keys
        if key in summary
    ]
    summary_parts.append(f"run={replication_index}/{run_count}")
    print("[INPUT] " + " ".join(summary_parts), flush=True)


if len(sys.argv) < 5:
    script_name = Path(sys.argv[0]).name if sys.argv else "run_one_horizon.py"
    raise SystemExit(
        f"Usage: python {script_name} <run_dir> <output_csv> <label> <horizon_tick> [run_count]"
    )


run_dir = Path(sys.argv[1]).resolve()
output_csv = Path(sys.argv[2]).resolve()
label = sys.argv[3]
horizon_tick = float(sys.argv[4])
run_count = int(sys.argv[5]) if len(sys.argv) >= 6 else int(os.environ.get(RUN_COUNT_ENV, "1"))
order_mode = os.environ.get(ORDER_MODE_ENV, "fixed_actual").strip().lower()
bootstrap_base_seed = int(os.environ.get(BOOTSTRAP_BASE_SEED_ENV, "42"))
bootstrap_arrival_mode = os.environ.get(
    BOOTSTRAP_ARRIVAL_MODE_ENV,
    "empirical_interarrival",
).strip()
bootstrap_n_orders_raw = os.environ.get(BOOTSTRAP_N_ORDERS_ENV)
bootstrap_n_orders = (
    int(bootstrap_n_orders_raw)
    if bootstrap_n_orders_raw is not None and bootstrap_n_orders_raw.strip() != ""
    else None
)
bootstrap_increment_every_raw = os.environ.get(
    BOOTSTRAP_INCREMENT_EVERY_ENV,
    "",
).strip()
bootstrap_increment_every = (
    int(bootstrap_increment_every_raw)
    if bootstrap_increment_every_raw != ""
    else None
)
diagnostic_checkpoints_enabled = os.environ.get(
    DIAGNOSTIC_CHECKPOINTS_ENV,
    "0",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
checkpoint_ticks = max(1.0, float(os.environ.get(CHECKPOINT_TICKS_ENV, "20000")))
pod_location_mode = os.environ.get(POD_LOCATION_MODE_ENV, "identity").strip().lower()
pod_location_base_seed = int(os.environ.get(POD_LOCATION_BASE_SEED_ENV, "42"))
pod_location_fixed = os.environ.get(POD_LOCATION_FIXED_ENV, "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
pod_location_increment_every_raw = os.environ.get(
    POD_LOCATION_INCREMENT_EVERY_ENV,
    "",
).strip()
pod_location_increment_every = (
    int(pod_location_increment_every_raw)
    if pod_location_increment_every_raw != ""
    else None
)

if run_count <= 0:
    raise SystemExit("run_count must be a positive integer.")
if order_mode not in {"fixed_actual", "bootstrap_actual"}:
    raise SystemExit("FULL_POSTT_ORDER_MODE must be either 'fixed_actual' or 'bootstrap_actual'.")
if bootstrap_increment_every is not None and bootstrap_increment_every <= 0:
    raise SystemExit("FULL_POSTT_BOOTSTRAP_INCREMENT_EVERY must be a positive integer.")
if pod_location_mode not in {"identity", "shuffle"}:
    raise SystemExit("FULL_POSTT_POD_LOCATION_MODE must be either 'identity' or 'shuffle'.")
if pod_location_increment_every is not None and pod_location_increment_every <= 0:
    raise SystemExit("FULL_POSTT_POD_LOCATION_INCREMENT_EVERY must be a positive integer.")

ensure_runtime_input_files(run_dir)
output_csv.parent.mkdir(parents=True, exist_ok=True)
if output_csv.exists():
    output_csv.unlink()

os.chdir(run_dir)
if str(run_dir) not in sys.path:
    sys.path.insert(0, str(run_dir))

devnull = open(os.devnull, "w")


def append_result_frame(frame: pd.DataFrame) -> None:
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    frame.to_csv(
        output_csv,
        index=False,
        sep=CSV_SEPARATOR,
        encoding=CSV_ENCODING if write_header else "utf-8",
        mode="a",
        header=write_header,
    )


def prepare_order_stream(replication_index: int) -> tuple[str, int | None, Path | None]:
    os.environ[ORDER_MODE_ENV] = order_mode
    if order_mode != "bootstrap_actual":
        os.environ.pop(BOOTSTRAP_SHARED_ORDER_PATH_ENV, None)
        return order_mode, None, None

    if bootstrap_increment_every is not None:
        current_seed = bootstrap_base_seed + (
            (replication_index - 1) % bootstrap_increment_every
        )
    else:
        current_seed = bootstrap_base_seed + (replication_index - 1)
    shared_order_path = ensure_shared_bootstrap_order_file(
        run_root=run_dir,
        seed=current_seed,
        n_orders=bootstrap_n_orders,
        arrival_mode=bootstrap_arrival_mode,
    )
    os.environ[BOOTSTRAP_SHARED_ORDER_PATH_ENV] = str(shared_order_path)
    return order_mode, current_seed, shared_order_path


def prepare_pod_location_stream(replication_index: int) -> tuple[str, int | None]:
    if pod_location_mode != "shuffle":
        os.environ.pop(RUNTIME_POD_LOCATION_POLICY_ENV, None)
        os.environ.pop(RUNTIME_POD_LOCATION_SEED_ENV, None)
        return pod_location_mode, None

    if pod_location_fixed:
        current_seed = pod_location_base_seed
    elif pod_location_increment_every is not None:
        current_seed = pod_location_base_seed + (
            (replication_index - 1) // pod_location_increment_every
        )
    else:
        current_seed = pod_location_base_seed + (replication_index - 1)
    os.environ[RUNTIME_POD_LOCATION_POLICY_ENV] = "shuffle"
    os.environ[RUNTIME_POD_LOCATION_SEED_ENV] = str(current_seed)
    return pod_location_mode, current_seed


def load_simulation_module():
    last_error = None
    for _attempt in range(3):
        try:
            if "netlogo" in sys.modules:
                sim_module = importlib.reload(sys.modules["netlogo"])
            else:
                sim_module = importlib.import_module("netlogo")
            with contextlib.redirect_stdout(devnull):
                setup_result = sim_module.setup()
            if isinstance(setup_result, str) and "error" in setup_result.lower():
                raise RuntimeError(setup_result)
            return sim_module
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise last_error


def build_result_frame(
    *,
    warehouse,
    replication_index: int,
    active_order_mode: str,
    active_bootstrap_seed: int | None,
    active_pod_location_mode: str,
    active_pod_location_seed: int | None,
    elapsed: float,
    stopped_cleanly: bool,
    result_kind: str,
    checkpoint_target_tick: float | None,
) -> pd.DataFrame:
    if (
        hasattr(warehouse, "refreshSimulationHealth")
        and int(getattr(warehouse, "health_check_interval", 0)) > 0
    ):
        warehouse.refreshSimulationHealth(force_log=True)

    on_hold = int(
        sum(1 for o in warehouse.order_manager.unfinished_orders if getattr(o, "on_hold", False))
    )
    fulfilled = int(warehouse.orders_fulfilled)
    arrived = int(len(warehouse.order_manager.orders))
    delivered_order_lines = int(getattr(warehouse, "delivered_order_lines", 0))
    picked_units = int(getattr(warehouse, "total_picked_units", 0))
    pod_visits = int(warehouse.pod_visit_to_station)
    energy = float(warehouse.total_energy)
    fixed_energy = float(warehouse.total_fixed_load_energy)
    variable_energy = max(0.0, energy - fixed_energy)

    row = {
        "scenario": label,
        "result_kind": result_kind,
        "checkpoint_target_tick": checkpoint_target_tick,
        "checkpoint_interval_ticks": checkpoint_ticks if diagnostic_checkpoints_enabled else None,
        "replication": replication_index,
        "replications_total": run_count,
        "order_mode": active_order_mode,
        "bootstrap_seed": active_bootstrap_seed,
        "pod_location_mode": active_pod_location_mode,
        "pod_location_seed": active_pod_location_seed,
        "ticks_elapsed": float(warehouse._tick),
        "steps_elapsed": int(warehouse._step),
        "stopped_cleanly_before_horizon": int(stopped_cleanly),
        "arrived_orders_by_horizon": arrived,
        "fulfilled_orders": fulfilled,
        "fulfilled_over_arrived": (fulfilled / arrived) if arrived else 0.0,
        "throughput_orders_per_hour": (
            fulfilled / (float(warehouse._tick) / 60.0)
        )
        if float(warehouse._tick) > 0
        else 0.0,
        "on_hold_orders": on_hold,
        "unfinished_orders": int(len(warehouse.order_manager.unfinished_orders)),
        "job_queue_length": int(len(warehouse.job_queue)),
        "sku_queue_length": int(len(warehouse.sku_picking_queue)),
        "pod_visits": pod_visits,
        "delivered_order_lines": delivered_order_lines,
        "picked_units": picked_units,
        "delivered_order_lines_per_pod_visit": (
            delivered_order_lines / pod_visits
        )
        if pod_visits
        else 0.0,
        "picked_units_per_pod_visit": (picked_units / pod_visits) if pod_visits else 0.0,
        "replenishment_count": int(warehouse.replenishment_count),
        "replenishment_trips": int(warehouse.replenishment_trips),
        "health_status_final": getattr(warehouse, "health_status", "unknown"),
        "health_consistency_violations": int(
            getattr(warehouse, "health_consistency_violations", 0)
        ),
        "health_zombie_orders": int(
            getattr(warehouse, "health_zombie_order_count", 0)
        ),
        "health_pending_replenishment_count": int(
            getattr(warehouse, "health_pending_replenishment_count", 0)
        ),
        "health_aged_pending_replenishment_count": int(
            getattr(warehouse, "health_aged_pending_replenishment_count", 0)
        ),
        "health_oldest_pending_replenishment_age": int(
            getattr(warehouse, "health_oldest_pending_replenishment_age", 0)
        ),
        "health_last_progress_tick": int(
            getattr(warehouse, "health_last_progress_tick", 0)
        ),
        "health_progress_gap": max(
            0,
            int(float(warehouse._tick)) - int(getattr(warehouse, "health_last_progress_tick", 0)),
        ),
        "health_stalled_tick_count": int(
            getattr(warehouse, "health_stalled_tick_count", 0)
        ),
        "stop_and_go": int(warehouse.stop_and_go),
        "total_energy": energy,
        "total_fixed_load_energy": fixed_energy,
        "variable_energy": variable_energy,
        "energy_per_fulfilled_order": (energy / fulfilled) if fulfilled else 0.0,
        "fixed_energy_per_fulfilled_order": (
            fixed_energy / fulfilled
        )
        if fulfilled
        else 0.0,
        "variable_energy_per_delivered_line": (
            variable_energy / delivered_order_lines
        )
        if delivered_order_lines
        else 0.0,
        "energy_per_pod_visit": (energy / pod_visits) if pod_visits else 0.0,
        "variable_energy_per_pod_visit": (
            variable_energy / pod_visits
        )
        if pod_visits
        else 0.0,
        "wall_clock_seconds": elapsed,
    }

    if diagnostic_checkpoints_enabled:
        robots = list(getattr(warehouse.robot_manager, "getAllRobots", lambda: [])())
        robot_state_counts = Counter(
            str(getattr(robot, "current_state", "unknown") or "unknown")
            for robot in robots
        )
        available_robot_count = sum(
            1
            for robot in robots
            if (getattr(robot, "job", None) is None or getattr(robot.job, "is_finished", False))
            and getattr(robot, "current_state", None) == "idle"
        )
        row.update(
            {
                "diag_pending_replenishment_requests_current": int(
                    len(getattr(warehouse, "pending_replenishment_dispatches", []))
                ),
                "diag_blocked_replenishment_no_robot_total": int(
                    getattr(warehouse, "diag_blocked_replenishment_no_robot_total", 0)
                ),
                "diag_blocked_replenishment_pod_not_idle_total": int(
                    getattr(warehouse, "diag_blocked_replenishment_pod_not_idle_total", 0)
                ),
                "diag_blocked_replenishment_no_station_total": int(
                    getattr(warehouse, "diag_blocked_replenishment_no_station_total", 0)
                ),
                "diag_replenishment_trips_since_last_fulfillment_rise": int(
                    getattr(
                        warehouse,
                        "diag_replenishment_trips_since_last_fulfillment_rise",
                        0,
                    )
                ),
                "diag_last_fulfillment_rise_tick": int(
                    getattr(warehouse, "diag_last_fulfillment_rise_tick", 0)
                ),
                "diag_robot_state_idle": int(robot_state_counts.get("idle", 0)),
                "diag_robot_state_taking_pod": int(robot_state_counts.get("taking_pod", 0)),
                "diag_robot_state_delivering_pod": int(
                    robot_state_counts.get("delivering_pod", 0)
                ),
                "diag_robot_state_station_processing": int(
                    robot_state_counts.get("station_processing", 0)
                ),
                "diag_robot_state_returning_pod": int(
                    robot_state_counts.get("returning_pod", 0)
                ),
                "diag_robot_state_other": int(
                    sum(
                        count
                        for state, count in robot_state_counts.items()
                        if state
                        not in {
                            "idle",
                            "taking_pod",
                            "delivering_pod",
                            "station_processing",
                            "returning_pod",
                        }
                    )
                ),
                "diag_robot_idle_time_gt_20": int(
                    sum(1 for robot in robots if int(getattr(robot, "idle_time", 0)) > 20)
                ),
                "diag_robot_idle_time_gt_300": int(
                    sum(1 for robot in robots if int(getattr(robot, "idle_time", 0)) > 300)
                ),
                "diag_robot_idle_time_gt_1200": int(
                    sum(1 for robot in robots if int(getattr(robot, "idle_time", 0)) > 1200)
                ),
                "diag_robot_pod_carrying_congested": int(
                    sum(
                        1
                        for robot in robots
                        if getattr(robot, "current_state", None) in {"delivering_pod", "returning_pod"}
                        and int(getattr(robot, "idle_time", 0)) > 1200
                    )
                ),
                "diag_robot_available_count": int(available_robot_count),
            }
        )

    return pd.DataFrame([row])


def run_single_replication(replication_index: int) -> pd.DataFrame:
    active_order_mode, active_bootstrap_seed, shared_order_path = prepare_order_stream(
        replication_index
    )
    active_pod_location_mode, active_pod_location_seed = prepare_pod_location_stream(
        replication_index
    )

    print(
        "[START] "
        f"scenario={label} "
        f"run={replication_index}/{run_count} "
        f"order_mode={active_order_mode} "
        f"bootstrap_seed={active_bootstrap_seed if active_bootstrap_seed is not None else 'n/a'} "
        f"pod_location_mode={active_pod_location_mode} "
        f"pod_location_seed={active_pod_location_seed if active_pod_location_seed is not None else 'n/a'} "
        f"horizon_tick={horizon_tick:g} "
        f"run_dir={run_dir} "
        f"shared_order={shared_order_path if shared_order_path is not None else 'n/a'}",
        flush=True,
    )
    print_input_summary(run_dir, replication_index, run_count)

    sim = load_simulation_module()
    warehouse = sim.warehouse
    start = time.time()
    stopped_cleanly = False
    last_progress_tick = float(warehouse._tick)
    last_progress_time = start
    next_checkpoint_tick = checkpoint_ticks if diagnostic_checkpoints_enabled else None

    while float(warehouse._tick) < horizon_tick:
        warehouse.tick()
        current_tick = float(warehouse._tick)
        now = time.time()
        if PROGRESS_ENABLED and (
            (current_tick - last_progress_tick) >= PROGRESS_TICKS
            or (now - last_progress_time) >= PROGRESS_SECONDS
        ):
            elapsed_now = now - start
            print(
                "[PROGRESS] "
                f"scenario={label} "
                f"run={replication_index}/{run_count} "
                f"order_mode={active_order_mode} "
                f"tick={current_tick:.2f}/{horizon_tick:.2f} "
                f"step={int(warehouse._step)} "
                f"elapsed_s={elapsed_now:.1f}",
                flush=True,
            )
            last_progress_tick = current_tick
            last_progress_time = now
        while (
            diagnostic_checkpoints_enabled
            and next_checkpoint_tick is not None
            and next_checkpoint_tick < horizon_tick
            and current_tick >= next_checkpoint_tick
        ):
            checkpoint_result = build_result_frame(
                warehouse=warehouse,
                replication_index=replication_index,
                active_order_mode=active_order_mode,
                active_bootstrap_seed=active_bootstrap_seed,
                active_pod_location_mode=active_pod_location_mode,
                active_pod_location_seed=active_pod_location_seed,
                elapsed=now - start,
                stopped_cleanly=False,
                result_kind="checkpoint",
                checkpoint_target_tick=next_checkpoint_tick,
            )
            append_result_frame(checkpoint_result)
            print(
                "[CHECKPOINT] "
                f"scenario={label} "
                f"run={replication_index}/{run_count} "
                f"tick={current_tick:.2f} "
                f"target_tick={next_checkpoint_tick:.0f} "
                f"pending_repl={int(len(getattr(warehouse, 'pending_replenishment_dispatches', [])))} "
                f"blocked_no_robot={int(getattr(warehouse, 'diag_blocked_replenishment_no_robot_total', 0))} "
                f"blocked_pod_not_idle={int(getattr(warehouse, 'diag_blocked_replenishment_pod_not_idle_total', 0))} "
                f"available_robots={int(checkpoint_result.loc[0, 'diag_robot_available_count'])}",
                flush=True,
            )
            next_checkpoint_tick += checkpoint_ticks
        if warehouse.isSimulationComplete():
            stopped_cleanly = True
            break

    elapsed = time.time() - start
    result = build_result_frame(
        warehouse=warehouse,
        replication_index=replication_index,
        active_order_mode=active_order_mode,
        active_bootstrap_seed=active_bootstrap_seed,
        active_pod_location_mode=active_pod_location_mode,
        active_pod_location_seed=active_pod_location_seed,
        elapsed=elapsed,
        stopped_cleanly=stopped_cleanly,
        result_kind="final",
        checkpoint_target_tick=horizon_tick if diagnostic_checkpoints_enabled else None,
    )
    fulfilled = int(result.loc[0, "fulfilled_orders"])
    arrived = int(result.loc[0, "arrived_orders_by_horizon"])
    print(
        "[DONE] "
        f"scenario={label} "
        f"run={replication_index}/{run_count} "
        f"order_mode={active_order_mode} "
        f"bootstrap_seed={active_bootstrap_seed if active_bootstrap_seed is not None else 'n/a'} "
        f"pod_location_mode={active_pod_location_mode} "
        f"pod_location_seed={active_pod_location_seed if active_pod_location_seed is not None else 'n/a'} "
        f"tick={float(warehouse._tick):.2f} "
        f"step={int(warehouse._step)} "
        f"elapsed_s={elapsed:.1f} "
        f"fulfilled={fulfilled} "
        f"arrived={arrived} "
        f"output={output_csv}",
        flush=True,
    )
    return result


try:
    for replication_index in range(1, run_count + 1):
        result = run_single_replication(replication_index)
        append_result_frame(result)
finally:
    devnull.close()
