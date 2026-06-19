import os
import numpy as np
import pandas as pd

from lib.file import *
from lib.constant import *


ACTUAL_ORDER_TIME_SCALE_SECONDS = 60
ACTUAL_ORDER_ID_CANDIDATES = ["\u8ba2\u5355\u53f7", "order_id"]
ACTUAL_SKU_CANDIDATES = ["\u5546\u54c1\u7f16\u7801", "item_code"]
ACTUAL_QUANTITY_CANDIDATES = ["\u5546\u54c1\u6570\u91cf", "item_quantity", "quantity"]
ACTUAL_CREATED_TIME_CANDIDATES = ["\u521b\u5efa\u65f6\u95f4", "order_date", "created_at"]
ORDER_MODE_ENV = "FULL_POSTT_ORDER_MODE"
BOOTSTRAP_SHARED_ORDER_PATH_ENV = "FULL_POSTT_SHARED_BOOTSTRAP_ORDER_PATH"


def normalize_item_code(value):
    text = str(value).replace("\ufeff", "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text[:-2] if text.endswith(".0") else text


def normalize_column_name(name):
    return str(name).replace("\ufeff", "").strip()


def find_column(columns, candidates):
    normalized = {normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    raise KeyError(
        f"Could not find any of the expected columns {candidates}. "
        f"Available columns: {list(columns)}"
    )


def find_preprocessing_dir():
    base_path = os.path.abspath(PARENT_DIRECTORY)
    candidates = [
        os.path.join(base_path, "Preprocessing"),
        os.path.join(os.path.dirname(base_path), "Preprocessing"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError("Could not locate the 'Preprocessing' directory.")


def find_actual_order_data_path():
    prepared_cutoff_orders = os.path.join(PARENT_DIRECTORY, "data/input/cutoff_test_orders.csv")
    if os.path.exists(prepared_cutoff_orders):
        return prepared_cutoff_orders

    preprocessing_dir = find_preprocessing_dir()
    candidates = sorted(
        os.path.join(preprocessing_dir, name)
        for name in os.listdir(preprocessing_dir)
        if name.endswith("_final.csv") and name != "preprocessed_final.csv"
    )
    if not candidates:
        raise FileNotFoundError(
            f"No order data file ending with '_final.csv' was found in {preprocessing_dir}."
        )
    return candidates[0]


def read_csv_auto(path):
    frame = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    frame.columns = [normalize_column_name(col) for col in frame.columns]
    return frame


def parse_mixed_datetimes(series):
    raw = series.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    for fmt in ("%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        remaining = parsed.isna()
        if not remaining.any():
            break
        parsed.loc[remaining] = pd.to_datetime(raw.loc[remaining], format=fmt, errors="coerce")

    remaining = parsed.isna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(raw.loc[remaining], errors="coerce")

    return parsed


def build_item_lookup(items):
    item_lookup = items[["item_id", "item_code"]].copy()
    item_lookup["item_id"] = pd.to_numeric(item_lookup["item_id"], errors="coerce")
    item_lookup = item_lookup.dropna(subset=["item_id"]).copy()
    item_lookup["item_id"] = item_lookup["item_id"].astype(np.int32)
    item_lookup["item_code"] = item_lookup["item_code"].map(normalize_item_code)
    item_lookup = item_lookup[item_lookup["item_code"] != ""].copy()
    item_lookup = item_lookup.drop_duplicates(subset=["item_code"], keep="first")
    return item_lookup


def write_assign_order_from_generated(generated_order, assign_order_path):
    assign_order_df = generated_order.copy()
    assign_order_df["assigned_station"] = None
    assign_order_df["assigned_pod"] = None
    assign_order_df["status"] = -3
    assign_order_df.to_csv(assign_order_path, index=False)


def load_shared_bootstrap_orders():
    items_path = os.path.join(PARENT_DIRECTORY, "data/output/items.csv")
    generated_order_path = os.path.join(PARENT_DIRECTORY, "data/output/generated_order.csv")
    assign_order_path = os.path.join(PARENT_DIRECTORY, "data/input/assign_order.csv")
    shared_order_path = os.getenv(BOOTSTRAP_SHARED_ORDER_PATH_ENV)

    if not shared_order_path:
        raise ValueError(
            "bootstrap_actual mode requires FULL_POSTT_SHARED_BOOTSTRAP_ORDER_PATH to be set."
        )
    if not os.path.exists(shared_order_path):
        raise FileNotFoundError(
            f"Shared bootstrap order file was not found: {shared_order_path}"
        )

    items = read_csv_auto(items_path)
    shared_orders = read_csv_auto(shared_order_path)
    required_columns = {"order_id", "order_type", "item_code", "item_quantity", "order_arrival"}
    missing_columns = sorted(required_columns.difference(shared_orders.columns))
    if missing_columns:
        raise ValueError(
            f"Shared bootstrap order file is missing required columns: {missing_columns}"
        )

    shared_orders["item_code"] = shared_orders["item_code"].map(normalize_item_code)
    shared_orders["item_quantity"] = pd.to_numeric(shared_orders["item_quantity"], errors="coerce")
    shared_orders["order_arrival"] = pd.to_numeric(shared_orders["order_arrival"], errors="coerce")
    shared_orders["order_id"] = pd.to_numeric(shared_orders["order_id"], errors="coerce")
    shared_orders["order_type"] = pd.to_numeric(shared_orders["order_type"], errors="coerce").fillna(1)
    shared_orders = shared_orders.dropna(
        subset=["item_code", "item_quantity", "order_arrival", "order_id", "order_type"]
    ).copy()
    shared_orders = shared_orders[shared_orders["item_code"] != ""].copy()
    shared_orders["item_quantity"] = np.ceil(shared_orders["item_quantity"]).astype(np.int32)
    shared_orders = shared_orders[shared_orders["item_quantity"] > 0].copy()
    shared_orders["order_arrival"] = shared_orders["order_arrival"].round().astype(np.int64)
    shared_orders["order_id"] = shared_orders["order_id"].astype(np.int32)
    shared_orders["order_type"] = shared_orders["order_type"].astype(np.int32)

    item_lookup = build_item_lookup(items)
    generated_order = shared_orders.merge(item_lookup, on="item_code", how="left")

    missing_mask = generated_order["item_id"].isna()
    if missing_mask.any():
        missing_codes = sorted(
            generated_order.loc[missing_mask, "item_code"].astype(str).unique().tolist()
        )
        sample = ", ".join(missing_codes[:10])
        dropped_lines = int(missing_mask.sum())
        generated_order = generated_order.loc[~missing_mask].copy()
        print(
            f"Skipping {dropped_lines:,} bootstrap order lines across {len(missing_codes):,} SKUs "
            f"that are not present in RMFS items.csv. Sample missing codes: {sample}"
        )
        if generated_order.empty:
            raise ValueError(
                "No bootstrap order lines remain after filtering to SKUs present in RMFS items.csv."
            )

    if "source_order_id" not in generated_order.columns:
        generated_order["source_order_id"] = generated_order["order_id"].astype(str)

    generated_order["item_id"] = generated_order["item_id"].astype(np.int32)
    generated_order = generated_order[
        ["order_id", "order_type", "item_id", "item_quantity", "order_arrival", "source_order_id"]
    ].copy()
    generated_order.insert(0, "sequence_id", np.arange(len(generated_order), dtype=np.int64))
    generated_order.to_csv(generated_order_path, index=False)
    write_assign_order_from_generated(generated_order, assign_order_path)
    return generated_order


def generate_orders_from_actual_data():
    items_path = os.path.join(PARENT_DIRECTORY, "data/output/items.csv")
    generated_order_path = os.path.join(PARENT_DIRECTORY, "data/output/generated_order.csv")
    assign_order_path = os.path.join(PARENT_DIRECTORY, "data/input/assign_order.csv")

    items = read_csv_auto(items_path)
    actual_order_path = find_actual_order_data_path()
    raw_orders = read_csv_auto(actual_order_path)

    order_col = find_column(raw_orders.columns, ACTUAL_ORDER_ID_CANDIDATES)
    sku_col = find_column(raw_orders.columns, ACTUAL_SKU_CANDIDATES)
    quantity_col = find_column(raw_orders.columns, ACTUAL_QUANTITY_CANDIDATES)
    created_col = find_column(raw_orders.columns, ACTUAL_CREATED_TIME_CANDIDATES)

    raw_orders = raw_orders[[order_col, sku_col, quantity_col, created_col]].copy()
    raw_orders[sku_col] = raw_orders[sku_col].map(normalize_item_code)
    raw_orders[order_col] = raw_orders[order_col].astype(str).str.strip()
    quantity_series = raw_orders[quantity_col].astype(str).str.replace(",", ".", regex=False)
    raw_orders[quantity_col] = pd.to_numeric(quantity_series, errors="coerce")
    raw_orders[created_col] = parse_mixed_datetimes(raw_orders[created_col])
    raw_orders = raw_orders.dropna(subset=[created_col, quantity_col]).copy()
    if raw_orders.empty:
        raise ValueError(
            f"No valid order rows remained after parsing timestamps and quantities from {actual_order_path}."
        )

    item_lookup = build_item_lookup(items)
    raw_orders = raw_orders.merge(item_lookup, left_on=sku_col, right_on="item_code", how="left")

    missing_mask = raw_orders["item_id"].isna()
    if missing_mask.any():
        missing_codes = sorted(raw_orders.loc[missing_mask, sku_col].astype(str).unique().tolist())
        sample = ", ".join(missing_codes[:10])
        dropped_lines = int(missing_mask.sum())
        raw_orders = raw_orders.loc[~missing_mask].copy()
        print(
            f"Skipping {dropped_lines:,} order lines across {len(missing_codes):,} SKUs "
            f"that are not present in RMFS items.csv. Sample missing codes: {sample}"
        )
        if raw_orders.empty:
            raise ValueError(
                "No actual order lines remain after filtering to SKUs present in RMFS items.csv."
            )

    order_code_map = {
        original_order_id: idx
        for idx, original_order_id in enumerate(
            raw_orders[order_col].drop_duplicates().sort_values().tolist()
        )
    }
    raw_orders["source_order_id"] = raw_orders[order_col]
    raw_orders["order_id"] = raw_orders[order_col].map(order_code_map).astype(np.int32)
    raw_orders["item_id"] = raw_orders["item_id"].astype(np.int32)
    raw_orders["item_quantity"] = np.ceil(raw_orders[quantity_col]).astype(np.int32)
    raw_orders["order_type"] = 1

    first_timestamp = raw_orders[created_col].min()
    arrival_seconds = (
        (raw_orders[created_col] - first_timestamp).dt.total_seconds() / ACTUAL_ORDER_TIME_SCALE_SECONDS
    )
    raw_orders["order_arrival"] = arrival_seconds.round().astype(np.int64)

    generated_order = raw_orders[
        ["order_id", "order_type", "item_id", "item_quantity", "order_arrival", "source_order_id"]
    ].copy()
    generated_order.insert(0, "sequence_id", np.arange(len(generated_order), dtype=np.int64))
    generated_order.to_csv(generated_order_path, index=False)
    write_assign_order_from_generated(generated_order, assign_order_path)

    return generated_order

def get_random_quantity(quantity_range=[1, 12]):
    ## Generate a random quantity based on a normal distribution
    ## Even quantities are favored

    # Define the quantities from min to max
    min_qty = quantity_range[0]
    max_qty = quantity_range[1]
    numbers = np.arange(min_qty, max_qty + 1)

    # Generate a normal distribution centered around the mean of the range
    mean = np.mean(numbers)
    std_dev = np.std(numbers)
    normal_dist = np.exp(-((numbers - mean) ** 2) / (2 * std_dev ** 2))

    # Adjust the probabilities to favor even numbers
    adjusted_prob = np.array([prob * 2 if num % 2 == 0 else prob for num, prob in zip(numbers, normal_dist)])

    # Normalize the adjusted probabilities to sum to 1
    probabilities = adjusted_prob / adjusted_prob.sum()

    # Make a random choice using the defined probabilities
    random_qty = np.random.choice(numbers, p=probabilities)

    return random_qty

def gen_backlog(initial_order, total_requested_item, items_orders_class_configuration,quantity_range,dev_mode):

    items_path = os.path.join(PARENT_DIRECTORY, 'data/output/items.csv')
    orders_path = os.path.join(PARENT_DIRECTORY, 'data/output/generated_order.csv')
    items = pd.read_csv(items_path, index_col=False)
    order = pd.read_csv(orders_path, index_col=False)

    total_available_item = items.shape[0]
    if total_available_item >= total_requested_item:

        if total_available_item > total_requested_item:
            print("Total SKU is less than total items in the items.csv")
            print("Total SKU will be set to the total items in the items.csv")
            total_requested_item = total_available_item

        temp = 0
        # calculate cummulative the class_pod_conf value, to get the threshold to assign the item to the pod.
        for key, value in items_orders_class_configuration.items():
            if temp == 0:
                temp = value
            else:
                temp += value
                items_orders_class_configuration[key] = np.round(temp, 1)

        # sort the class_pod_conf based on the value
        items_orders_class_configuration = dict(sorted(items_orders_class_configuration.items(), key=lambda x: x[1]))

        # get class_pod_conf values into list as thresholds. The keys will be used to get the item based on the class
        thresholds = list(items_orders_class_configuration.values())
        keys = list(items_orders_class_configuration.keys())
        # print(keys, thresholds)

        orders_in_backlog = list(i * -1 for i in range(1, initial_order+1))
        items_in_order = np.random.geometric(p=0.3, size=initial_order)
        # print(orders_in_backlog, items_in_order)

        orders_backlog = pd.DataFrame(columns=[ 'order_id', 
                                               'order_type',
                                               'item_id', 
                                               'item_quantity', 
                                               'order_arrival'])
        
        for i, order in enumerate(orders_in_backlog):
            order_id = order
            order_type = 1
            order_duedate = 99999
            
            items_num = items_in_order[i]
            # print(f"Order {order_id} has {items_num} items")

            rand = np.random.rand(items_num)
            item_exist = list()
            # class_item = "A"
            for r in rand:
                if r <= thresholds[0]:
                    # print(f"  below {thresholds[0]}, then the class is {keys[0]}")
                    item_available = items.loc[(items["item_class"] == keys[0]) & (
                        ~items.index.isin(item_exist)), "item_order_frequency"]
                    #     ~items.index.isin(item_exist)), "total_order_freq"] # CHANGED from "item_order_frequency" to "total_order_freq"
                    # # class_item = keys[0]

                for l in range(len(thresholds) - 1):
                    if thresholds[l] < r <= thresholds[l + 1]:
                        # print(f"  between {thresholds[l]} and {thresholds[l + 1]}, then the class is {keys[l+1]}")
                        item_available = items.loc[(items["item_class"] == keys[l+1]) & (
                            ~items.index.isin(item_exist)), "item_order_frequency"]
                        #     ~items.index.isin(item_exist)), "total_order_freq"] # CHANGED from "item_order_frequency" to "total_order_freq"
                        # # class_item = keys[l+1]
                
                item_probability = (item_available / item_available.sum()).to_list()
                item_available = item_available.index.to_list()
                if len(item_available) > 0:
                    item_id = np.random.choice(item_available, p=item_probability)
                    qty = get_random_quantity(quantity_range=quantity_range)
                    order_arrival = 0
                    item_exist.append(item_id)
                    # print("    ", item_id, qty, class_item)

                    orders_backlog = pd.concat([orders_backlog, 
                                                pd.DataFrame({"order_id": [order_id],
                                                              "order_type": [order_type],
                                                              "item_id": [item_id],
                                                              "item_quantity": [qty],
                                                              "order_arrival" : [order_arrival]})
                                                ], axis=0)

        orders_backlog.sort_values(by="order_id", ascending=True, inplace=True)                    
        orders_backlog.reset_index(drop=True, inplace=True)
        orders_backlog.insert(loc=0, column="sequence_id", value=orders_backlog.index.to_list())
        orders_backlog_path = os.path.join(PARENT_DIRECTORY, 'data/input/generated_backlog.csv')
        orders_backlog.to_csv(orders_backlog_path, index=False)

        return orders_backlog

    else:
        print("Total SKU ("+str(total_requested_item)+") is more than total items in the items.csv ("+str(total_available_item)+")")
        print("Please provide a total SKU that is equal to or less than the total items in the items.csv")
        return None
    
# ["order_id", 'order_dum', 'order_type', "item", "qty", "facing", "due_date", 'station', 'pod_id', 'status', 'finish_time', 'date', 'time_gen']

def gen_order_arrival_time(order_cycle_time):
    
    # Define the total number of orders and the time period in minutes
    total_orders = order_cycle_time     # number of orders in a cycle (hour)
    time_period = 60                    # 60 minutes in an hour

############################## OLD CODE ##############################
    # # Calculate the average rate (lambda)
    # lambda_rate = total_orders / time_period

    # # Generate the number of orders per minute using a Poisson distribution
    # orders_per_minute = np.random.poisson(lambda_rate, size=time_period)

    # # Generate arrival times as integers
    # arrival_times = []
    # for minute, num_orders in enumerate(orders_per_minute):
    #     arrival_times.extend([minute] * num_orders)

    # # If there are more than total_orders (due to Poisson randomness), truncate the list
    # if len(arrival_times) >= total_orders:
    #     arrival_times = arrival_times[:total_orders]

    # # Sort the arrival times
    # arrival_times.sort()

    # # Print the arrival times
    # print(f"Order arrival times (in minutes): {arrival_times}")

    # return arrival_times

############################### NEW CODE ##############################

    # Calculate the average inter-arrival time (smooth Amazon-like distribution)
    average_inter_arrival = time_period / total_orders

    # Generate smooth exponential inter-arrival times (no gaps like Amazon)
    inter_arrival_times = np.random.exponential(average_inter_arrival, size=total_orders)
    
    # Convert to cumulative arrival times starting from 0
    arrival_times = np.cumsum(inter_arrival_times)
    
    # Scale to fit within the 60-minute window and convert to integer minutes
    if len(arrival_times) > 0:
        max_time = arrival_times[-1]
        arrival_times = (arrival_times / max_time) * (time_period - 1)
        arrival_times = np.round(arrival_times).astype(int)
    
    # Ensure we have exactly the right number of orders
    arrival_times = arrival_times[:total_orders]

    return arrival_times.tolist()
    
    

def gen_order(order_cycle_time,
              order_period_time,  
              order_start_arrival_time,
              total_requested_item, 
              items_orders_class_configuration,
              quantity_range,
              date, 
              dev_mode):

    items_path = os.path.join(PARENT_DIRECTORY, 'data/output/items.csv')
    items = pd.read_csv(items_path, index_col=False)    

    total_available_item = items.shape[0]

    if (total_available_item > total_requested_item) or (total_available_item == total_requested_item):
        print("Total SKU is less than total items in the items.csv")
        print("Total SKU will be set to the total items in the items.csv")
        total_requested_item = total_available_item

    if total_available_item >= total_requested_item:
        temp = 0
        # calculate cummulative the class_pod_conf value, to get the threshold to assign the item to the pod.
        for key, value in items_orders_class_configuration.items():
            if temp == 0:
                temp = value
            else:
                temp += value
                items_orders_class_configuration[key] = np.round(temp, 1)

        # sort the class_pod_conf based on the value
        items_orders_class_configuration = dict(sorted(items_orders_class_configuration.items(), key=lambda x: x[1]))

        # get class_pod_conf values into list as thresholds. The keys will be used to get the item based on the class
        thresholds = list(items_orders_class_configuration.values())
        keys = list(items_orders_class_configuration.keys())
        # print(keys, thresholds)

        arrival_times_list = list()
        last_arrival_time  = 0
        for i in range(1, order_period_time+1):
            arrival_times = gen_order_arrival_time(order_cycle_time=order_cycle_time)
            if i==1:
                index_start_arrival_time = np.where(np.array(arrival_times) > order_start_arrival_time)[0][0]

                arrival_times_list = arrival_times[index_start_arrival_time:-1] + [arrival_times[-1] + 1 + x for x in arrival_times[:index_start_arrival_time]]
                last_arrival_time = arrival_times[-1] + 1
            else:
                arrival_times_list = arrival_times_list + [last_arrival_time + x for x in arrival_times]
                last_arrival_time = arrival_times_list[-1]

        arrival_times_list = [60 * x for x in arrival_times_list] # convert to seconds
        orders = range(0, len(arrival_times_list))
        items_in_order = np.random.geometric(p=0.3, size=len(orders))

        database_order = pd.DataFrame(columns=['order_dum', 
                                               'order_type', 
                                               "item", 
                                               "qty", 
                                               "facing", 
                                               "due_date", 
                                               'station', 
                                               'pod_id', 
                                               'status', 
                                               'finish_time', 
                                               'date', 
                                               'time_gen'])
        for i, order in enumerate(orders):
            order_id = order
            order_type = 1
            order_duedate = 99999
            
            items_num = items_in_order[i]
            # print(f"Order {order_id} has {items_num} items")

            rand = np.random.rand(items_num)
            item_exist = list()
            # class_item = "A"
            for r in rand:
                if r <= thresholds[0]:
                    # print(f"  below {thresholds[0]}, then the class is {keys[0]}")
                    item_available = items.loc[(items["item_class"] == keys[0]) & (
                        ~items.index.isin(item_exist)), "item_order_frequency"]
                    #     ~items.index.isin(item_exist)), "total_order_freq"] # CHANGED from "item_order_frequency" to "total_order_freq"
                    # class_item = keys[0]

                for l in range(len(thresholds) - 1):
                    if thresholds[l] < r <= thresholds[l + 1]:
                        # print(f"  between {thresholds[l]} and {thresholds[l + 1]}, then the class is {keys[l+1]}")
                        item_available = items.loc[(items["item_class"] == keys[l+1]) & (
                            ~items.index.isin(item_exist)), "item_order_frequency"]
                        #     ~items.index.isin(item_exist)), "total_order_freq"] # CHANGED from "item_order_frequency" to "total_order_freq"
                        # class_item = keys[l+1]
                
                item_probability = (item_available / item_available.sum()).to_list()
                item_available = item_available.index.to_list()
                if len(item_available) > 0:
                    item_id = np.random.choice(item_available, p=item_probability)
                    qty = get_random_quantity(quantity_range=quantity_range)      
                    item_exist.append(item_id)
                    # print("    ", item_id, qty, class_item)

                    database_order = pd.concat([database_order, 
                                                pd.DataFrame({"order_dum": [order_id],
                                                              "order_type": [order_type],
                                                              "item": [item_id],
                                                              "qty": [qty],
                                                              "facing": [-1],
                                                              "due_date": [order_duedate],
                                                              "station": [-1],
                                                              "pod_id": [-1],
                                                              "status": [-3],
                                                              "finish_time": [-1],
                                                              "date": [date],
                                                              "time_gen": [arrival_times_list[i]]})
                                                ], axis=0)


        database_order.reset_index(drop=True, inplace=True)
        database_order.insert(loc=0, column="order_id", value=database_order.index.to_list())
        database_order_path = os.path.join(PARENT_DIRECTORY, 'data/output/generated_database_order.csv')
        database_order.to_csv(database_order_path, index=False)

        generated_order = database_order[["order_id", 'order_dum', 'order_type', "item", "qty", 'time_gen']].copy()
        generated_order.columns = ["sequence_id", 'order_id', 'order_type', "item_id", "item_quantity", 'order_arrival']
        
        generated_order_path = os.path.join(PARENT_DIRECTORY, 'data/output/generated_order.csv')
        generated_order.to_csv(generated_order_path, index=False)

        return database_order

    else:
        print("Total SKU ("+str(total_requested_item)+") is more than total items in the items.csv ("+str(total_available_item)+")")
        print("Please provide a total SKU that is equal to or less than the total items in the items.csv")
        return None

def config_orders(
    initial_order,
    total_requested_item,
    items_orders_class_configuration,
    quantity_range,
    order_cycle_time,
    order_period_time,
    order_start_arrival_time,
    date,
    sim_ver,
    dev_mode,
    use_actual_order_data=False,
):
    print(f"Config orders: sim_ver={sim_ver}, total_SKUs={total_requested_item:,}, initial_order={initial_order}")
    if use_actual_order_data:
        order_mode = os.getenv(ORDER_MODE_ENV, "fixed_actual").strip().lower()
        if order_mode == "bootstrap_actual":
            print("Generate orders from shared bootstrap actual-order stream...")
            generated_order = load_shared_bootstrap_orders()
        elif order_mode == "fixed_actual":
            print("Generate orders from actual 21-day order data...")
            generated_order = generate_orders_from_actual_data()
        else:
            raise ValueError(
                f"Unsupported FULL_POSTT_ORDER_MODE '{order_mode}'. "
                "Expected 'fixed_actual' or 'bootstrap_actual'."
            )
        print(f"    Actual order rows generated: {len(generated_order):,}")
        print(f"    Actual orders generated: {generated_order['order_id'].nunique():,}")
        return generated_order

    if sim_ver == 1:
        print("Generate database orders in simver 1...")

        database_order_path = os.path.join(PARENT_DIRECTORY, 'data/output/generated_database_order.csv')
        if not os.path.exists(database_order_path):
            print("    Generated database orders is not found. We will generate database orders:")
            orders = gen_order(order_cycle_time=order_cycle_time,order_period_time=order_period_time,order_start_arrival_time=order_start_arrival_time,total_requested_item=total_requested_item, items_orders_class_configuration=items_orders_class_configuration,quantity_range=quantity_range,date=date,dev_mode=dev_mode)
            print("Order:", orders)
            # print(" Table orders col:", orders.columns)
            # order_id_list = orders["order_dum"].unique().tolist()
            # print("    "+str(len(order_id_list))+" orders are generated.")
        else:
            print("    Generated database orders file is found. We will use the existing orders file.")    
            print("    If you want to reconfigure the orders, please delete the generated_order.csv file.") 

    elif sim_ver == 2:
        
        print("Generate backlog orders in simver 2...")

        print("Generate orders...")
        generated_order_path = os.path.join(PARENT_DIRECTORY, 'data/output/generated_order.csv')
        if not os.path.exists(generated_order_path):
            print("    Generated orders is not found. We will generate database orders:")
            orders = gen_order(order_cycle_time=order_cycle_time,
                               order_period_time=order_period_time,  
                               order_start_arrival_time=order_start_arrival_time,  
                               total_requested_item=total_requested_item, 
                               items_orders_class_configuration=items_orders_class_configuration,
                               quantity_range=quantity_range,
                               date=date,
                               dev_mode=dev_mode)
            order_id_list = orders["order_dum"].unique().tolist()
            print("    "+str(len(order_id_list))+" orders are generated.")
            print("    Generate orders is done. If you want to reconfigure the orders, please delete the generated_order.csv file.")

        else:
            print("    Generated orders file is found. We will use the existing orders file.")
            print("    If you want to reconfigure the orders, please delete the generated_order.csv file.")  

        backlogs_path = os.path.join(PARENT_DIRECTORY, 'data/input/generated_backlog.csv')
        backlog_generated = False
        if not os.path.exists(backlogs_path):
            
            print("    Generated backlog orders is not found. We will generate backlog orders.")
            backlogs = gen_backlog(initial_order=initial_order, total_requested_item=total_requested_item, 
                                   items_orders_class_configuration=items_orders_class_configuration,
                                   quantity_range=quantity_range,
                                   dev_mode=dev_mode)
            backlog_generated = True
        else:
            backlogs = pd.read_csv(backlogs_path, index_col=False)
            backlogs_id_list = backlogs["order_id"].unique().tolist()
            
            if initial_order == len(backlogs_id_list):
                print("    Initial order is the same as the number of orders in the backlog file.")
                print("    We will use the existing items file.")
                backlog_generated = True
            
            else:
                print("    Initial order is different from the number of orders in the backlog file.")
                print("    We will re-generate backlog orders using the new intial order.")
                backlogs = gen_backlog(initial_order=initial_order, total_requested_item=total_requested_item, 
                                       items_orders_class_configuration=items_orders_class_configuration,
                                       quantity_range=quantity_range,
                                       dev_mode=dev_mode)
                backlog_generated = True
        print("    Generate backlog orders is done. If you want to reconfigure the backlog orders, please delete the generated_backlog.csv file.")


    
        if backlog_generated:
            print("MERGING EUYY")
            order_path = os.path.join(PARENT_DIRECTORY, 'data/output/generated_order.csv')
            backlog_path = os.path.join(PARENT_DIRECTORY, 'data/input/generated_backlog.csv')
            csv_files = [backlog_path,order_path]
            dataframes = [pd.read_csv(file) for file in csv_files]
            merged_df = pd.concat(dataframes, ignore_index=True)
            merged_df['sequence_id'] = range(1, len(merged_df) + 1)
            os.remove(order_path)
            merged_df.to_csv(order_path, index=False)

            file_path = PARENT_DIRECTORY + "/data/input/assign_order.csv"
            assign_order_df = merged_df.copy()
            assign_order_df['assigned_station'] = None
            assign_order_df['assigned_pod'] = None
            assign_order_df['status'] = -3
            assign_order_df.to_csv(file_path, index=False)     
