from __future__ import annotations
import contextlib, importlib, os, sys, time
from pathlib import Path
import pandas as pd

CSV_SEPARATOR = os.environ.get("FULL_POSTT_CSV_SEPARATOR", ";")
CSV_ENCODING = "utf-8-sig"

run_dir = Path(sys.argv[1])
output_csv = Path(sys.argv[2])
label = sys.argv[3]
horizon_tick = float(sys.argv[4])
os.chdir(run_dir)
if str(run_dir) not in sys.path:
    sys.path.insert(0, str(run_dir))

devnull = open(os.devnull, 'w')
last_error = None
sim = None
for attempt in range(3):
    try:
        if 'netlogo' in sys.modules:
            sim = importlib.reload(sys.modules['netlogo'])
        else:
            sim = importlib.import_module('netlogo')
        with contextlib.redirect_stdout(devnull):
            setup_result = sim.setup()
        if isinstance(setup_result, str) and 'error' in setup_result.lower():
            raise RuntimeError(setup_result)
        last_error = None
        break
    except Exception as exc:
        last_error = exc
        time.sleep(2)
else:
    devnull.close()
    raise last_error

warehouse = sim.warehouse
start = time.time()
stopped_cleanly = False
while float(warehouse._tick) < horizon_tick:
    warehouse.tick()
    if warehouse.isSimulationComplete():
        stopped_cleanly = True
        break
elapsed = time.time() - start
on_hold = int(sum(1 for o in warehouse.order_manager.unfinished_orders if getattr(o, 'on_hold', False)))
fulfilled = int(warehouse.orders_fulfilled)
arrived = int(len(warehouse.order_manager.orders))
delivered_order_lines = int(getattr(warehouse, 'delivered_order_lines', 0))
pod_visits = int(warehouse.pod_visit_to_station)
energy = float(warehouse.total_energy)
fixed_energy = float(warehouse.total_fixed_load_energy)
result = pd.DataFrame([{
    'scenario': label,
    'ticks_elapsed': float(warehouse._tick),
    'steps_elapsed': int(warehouse._step),
    'stopped_cleanly_before_horizon': int(stopped_cleanly),
    'arrived_orders_by_horizon': arrived,
    'fulfilled_orders': fulfilled,
    'fulfilled_over_arrived': (fulfilled / arrived) if arrived else 0.0,
    'throughput_orders_per_hour': (fulfilled / (float(warehouse._tick) / 60.0)) if float(warehouse._tick) > 0 else 0.0,
    'on_hold_orders': on_hold,
    'unfinished_orders': int(len(warehouse.order_manager.unfinished_orders)),
    'job_queue_length': int(len(warehouse.job_queue)),
    'sku_queue_length': int(len(warehouse.sku_picking_queue)),
    'pod_visits': pod_visits,
    'delivered_order_lines': delivered_order_lines,
    'delivered_order_lines_per_pod_visit': (delivered_order_lines / pod_visits) if pod_visits else 0.0,
    'replenishment_count': int(warehouse.replenishment_count),
    'replenishment_trips': int(warehouse.replenishment_trips),
    'stop_and_go': int(warehouse.stop_and_go),
    'total_energy': energy,
    'total_fixed_load_energy': fixed_energy,
    'energy_per_fulfilled_order': (energy / fulfilled) if fulfilled else 0.0,
    'fixed_energy_per_fulfilled_order': (fixed_energy / fulfilled) if fulfilled else 0.0,
    'wall_clock_seconds': elapsed,
}])
result.to_csv(output_csv, index=False, sep=CSV_SEPARATOR, encoding=CSV_ENCODING)
devnull.close()
