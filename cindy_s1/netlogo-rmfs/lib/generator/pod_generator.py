import csv
import random
import os
from pathlib import Path
import numpy as np
import pandas as pd
from lib.file import *
from lib.constant import *


def generate_pod(pod_types=[4], pod_num=[177], total_sku=7024, 
            items_class_conf={"A": 0.023, "B": 0.245, "C": 0.732},
            items_pods_inventory_levels={"A": 0.4, "B": 0.4, "C": 0.4},
            items_warehouse_inventory_levels={"A": 0.4, "B": 0.4, "C": 0.4},
            items_pods_class_conf={"A": 0.0317, "B": 0.2461, "C": 0.7307},
            dev_mode=False):
        
    working_path = get_working_path(dev_mode)


# def generate_pod(pod_types=[0], pod_num=[409], total_sku=5798,
#             items_class_conf={0: 0.0772, 1: 0.4013, 2: 0.0107, 3: 0.0594, 4: 0.44, 5: 0.0115},
#             items_pods_inventory_levels={0: 0.4, 1: 0.5, 2: 0.6, 3: 0.4, 4: 0.5, 5: 0.6},
#             items_warehouse_inventory_levels={0: 0.4, 1: 0.5, 2: 0.6, 3: 0.4, 4: 0.5, 5: 0.6},
#             items_pods_class_conf={0: 0.0772, 1: 0.4013, 2: 0.0107, 3: 0.0594, 4: 0.44, 5: 0.0115},
#             dev_mode=False):
        
    get_working_path(dev_mode)
    
    config_items_slots(dev_mode=dev_mode)
    items_path = PARENT_DIRECTORY + "/data/output/items.csv"
    print("Configuring items...")
    
    if not os.path.exists(items_path):
        print("    Items configuration is not found. We will generate the items based on the class configuration.")

        print("Generating items with the following parameters:")
        print(f"  pod_types: {pod_types}")
        print(f"  total_sku: {total_sku}")
        print(f"  items_class_conf: {items_class_conf}")
        print(f"  items_pods_inventory_levels: {items_pods_inventory_levels}")
        print(f"  items_warehouse_inventory_levels: {items_warehouse_inventory_levels}")
        print(f"  dev_mode: {dev_mode}")

        items = gen_items(pod_types=pod_types, 
                total_sku=total_sku, 
                items_class_conf=items_class_conf,
                items_pods_inventory_levels=items_pods_inventory_levels,  
                items_warehouse_inventory_levels=items_warehouse_inventory_levels,
                select_option=0,
                dev_mode=dev_mode)       
    else:
        items = pd.read_csv(items_path, index_col=0)

        if items.shape[0] == total_sku:
            print("    Items already exist and the number of items is the same as the total SKU.")
            print("    We will use the existing items file.")
            items_flag = False

        elif items.shape[0] > total_sku:
            print("    Items already exist but the number of items is more than the total SKU.")
            print("    We will use the existing items file and select the items based on the total SKU and the class configuration.")
            for k, v in items_class_conf.items():
                items_class = items.loc[items["item_class"]==k]
                items_class = items_class.sort_values(by="item_order_frequency", ascending=False)
                items_class = items_class.head(int(total_sku*v))
                items = pd.concat([items, items_class])
                items = items.drop_duplicates(subset="item_code")
                if items.shape[0] == total_sku:
                    items.reset_index(drop=True, inplace=True)
                    break
            items.to_csv(items_path, index=True)
            items_flag = True
        else:
            print("    Items already exist but the number of items is less than the total SKU.")
            print("    We will generate the items based on number of items needed and the class configuration.")
            items = gen_items(pod_types=pod_types, 
                            total_sku=total_sku, 
                            items_class_conf=items_class_conf,
                            items_pods_inventory_levels=items_pods_inventory_levels,  
                            items_warehouse_inventory_levels=items_warehouse_inventory_levels,
                            select_option=0,
                            dev_mode=dev_mode)   
            items_flag = True
        print("    Items configuration is done. If you want to reconfigure the items, please delete the items.csv file.")
        print()

    if items is not None:

        pods_path = PARENT_DIRECTORY + "/data/output/pods.csv"
        
        print("Configuring pods and assigning items to the pods...")
        if not os.path.exists(pods_path):

            print("    Pods configuration is not found. We will generate the pods based on the configuration.")
            pods = gen_pods(pod_types=pod_types, pod_num=pod_num, dev_mode=dev_mode)
            pods = assign_items_to_pods(pods, items, items_pods_class_conf=items_pods_class_conf, dev_mode=dev_mode)
            pods_flag = True

        else: 
            pods = pd.read_csv(pods_path, index_col=False)
            pod_list = pods["pod_id"].unique().tolist()
            pod_type_list = pods["pod_type"].unique().tolist()

            flag_pod_num_per_type = True
            for i, pod_type in enumerate(pod_types):
                if pod_num[i] != len(pods.loc[pods["pod_type"]==pod_type, "pod_id"].unique().tolist()):
                    flag_pod_num_per_type = False
                    break
            
            flag_pod_type = True
            for pod_type in pod_type_list:
                if pod_type not in pod_types:
                    flag_pod_type = False
                    break
                    
            flag_pod_num = True
            if len(pod_list) != sum(pod_num):
                flag_pod_num = False

            if flag_pod_num_per_type and flag_pod_type and flag_pod_num:
                print("    Pods already exist and the number of pods, number of pod per pods type, and the pod type are the same as the configuration.")
                print("    We will use the existing pods file.")
                pods_flag = False
            else:
                print("    Pods already exist but the number of pods, number of pod per pods type, or the pod type is different from the configuration.")
                print("    We will generate the pods based on the configuration.")
                pods = gen_pods(pod_types=pod_types, pod_num=pod_num, dev_mode=dev_mode)
                pods = assign_items_to_pods(pods, items, items_pods_class_conf=items_pods_class_conf, dev_mode=dev_mode)
                pods_flag = True

        print("    Pods configuration is done. If you want to reconfigure the pods, please delete the pods.csv file.")
        print()
    else:     
        pods = None

    return     

def config_items_slots(dev_mode=False):
    working_path = get_working_path(dev_mode)

    print("Setting up item slot configuration based on Item and Pod dictionary (see data/v2/ folder)...")

    # check if file not exists
    items_slots_path = PARENT_DIRECTORY + "/data/output/items_slots_configuration.csv"
    if not os.path.exists(items_slots_path):

        pods_dictionary_path = working_path + "/data/input/pods_dictionary.csv"
        pods_dictionary = pd.read_csv(pods_dictionary_path, index_col=False)

        item_path = working_path + "/data/input/items_dictionary.csv"
        item_df = pd.read_csv(item_path, index_col=False)
        item_df = item_df[["item_code", "box_volume",
                        "item_volume", "number_of_item_in_a_box", "max_fit"]].copy()
        
        print("pod generator di config_items_slots:")

        # getl slot type from pod_dictionary
        items_slots_configuration = pd.DataFrame()
        slot_types = np.sort(pods_dictionary["slot_type"].unique())
        for slot_type in slot_types:
            # get slot volume
            slot_volume = pods_dictionary.loc[pods_dictionary["slot_type"]
                                    == slot_type, "slot_volume"].values[0]
            item_df["slot_type"] = slot_type
            item_df["slot_volume"] = slot_volume
            item_df["max_box_in_slot"] = (
                slot_volume / item_df["box_volume"]).astype(int)
            item_df["max_item_in_slot"] = item_df["max_fit"]

            items_slots_configuration = pd.concat(
                [items_slots_configuration, item_df], axis=0)

        items_slots_configuration.to_csv(items_slots_path, index=False)

    else:
        print("    Item slot configuration already exists. Delete the file to reconfigure.")
        print()

def gen_items(pod_types=[4], 
              total_sku=7024, 
              items_class_conf={"A": 0.023, "B": 0.245, "C": 0.732}, 
              items_pods_inventory_levels={"A": 0.4, "B": 0.4, "C": 0.4}, 
              items_warehouse_inventory_levels={"A": 0.4, "B": 0.4, "C": 0.4},
              select_option=1, 
              dev_mode=False):
        

        # Generate or select items based on class of items.
        # The items must align with the designated slot type within the pod.

        # get the working path
        working_path = get_working_path(dev_mode)

        # get item slot configuration
        items_slots_configuration_path = PARENT_DIRECTORY + "/data/output/items_slots_configuration.csv"
        items_slots_configuration = pd.read_csv(items_slots_configuration_path, index_col=False)

        # get pod configuration
        pods_dictionary_path = working_path + "/data/input/pods_dictionary.csv"
        pods_dictionary = pd.read_csv(pods_dictionary_path, index_col=False)

        if check_items_pods_feasibility(total_sku, pod_types, pods_dictionary):
            
            # get list of item id that can be stored in the pod type (and slot type)
            item_code_arr = list()
            for pod_type in pod_types:
                slot_types = pods_dictionary.loc[pods_dictionary["pod_type"]
                                        == pod_type, "slot_type"].unique()
                slot_types = np.sort(slot_types)

                # print(slot_types)
                for slot_type in slot_types:
                    # print(slot_type)
                    item_code = items_slots_configuration.loc[(items_slots_configuration["slot_type"] == slot_type), "item_code"]
                    item_code_arr = item_code_arr + item_code.to_list()

            # remove duplicates
            item_code_arr = np.unique(item_code_arr)

            # retrieve item details corresponding to the list of item IDs compatible with the pod's slot specifications
            item_path = working_path + "/data/input/items_dictionary.csv"
            item_df = pd.read_csv(item_path, index_col=False)
            print("item_df shape habis dibaca:", item_df.shape)

            item_df = item_df.loc[item_df["item_code"].isin(item_code_arr)].copy()
            print("item_df shape habis difilter:", item_df.shape)

            item_df = item_df[["item_code", "item_class", "item_order_frequency", "item_initial_quantity_inventory", "box_length", "box_width", "box_height", "box_volume", "box_weight", "number_of_item_in_a_box",
                            "item_volume", "item_unit", "max_fit"]].copy()
            print("item_df shape habis dicopy:", item_df.shape)

            # select items according to the class and its proportion
            items = pd.DataFrame()

            print("items_class_conf:", items_class_conf.items())

            for class_name, conf in items_class_conf.items():

                # get list of the item's code based on the class
                item = item_df.loc[item_df["item_class"] == class_name].copy()
                print("item shape before sort:", item.shape)
                item = item.sort_values(by="item_order_frequency", ascending=False)

                # KOMEN UTUH

                # # print(items_inventory_levels)
                item["item_pod_inventory_level"] = items_pods_inventory_levels[class_name]
                item["item_warehouse_inventory_level"] = items_warehouse_inventory_levels[class_name]

                # calculate the probability of order frequency for each item
                item_probability = item["item_order_frequency"] / \
                    item["item_order_frequency"].sum()

                # select option 1: select items based on the probability of order frequency for each class.
                # else: select items based on the top n-percent of the class ratio
                if select_option == 1:

                    # select items based on the probability of order frequency for each class.
                    item_code = np.random.choice(item["item_code"].to_list(), size=int(
                        total_sku*conf), p=item_probability, replace=False)
                    
                    print("Concat items:", len(item.loc[item["item_code"].isin(item_code)]))
                    items = pd.concat(
                        [items, item.loc[item["item_code"].isin(item_code)]])
                else:

                    # select items based on the top n-percent of the class ratio
                    item_top = item.head(round(total_sku*conf))
                    print("item_top shape:", item_top.shape)
                    items = pd.concat([items, item_top])

                #  KOMEN UTUH

            # save items selected to csv
            items.reset_index(drop=True, inplace=True)
            items.index.name = "item_id"

            items_weight = items["box_weight"] / items["number_of_item_in_a_box"]
            items.insert(11, "item_weight", items_weight.round(3))
            # items[["item_order_frequency", "number_of_item_in_a_box"]] = items[[
            #     "item_order_frequency", "number_of_item_in_a_box"]].astype(int)

            print("items final shape:", items.shape)

            items.to_csv(PARENT_DIRECTORY + "/data/output/items.csv", index=True)

        else:
            items = None

        return items

def gen_pods(pod_num, pod_types, dev_mode=True):
    # get the working path
    working_path = get_working_path(dev_mode)

    # get pod configuration
    pods_dictionary_path = working_path + "/data/input/pods_dictionary.csv"
    pods_dictionary = pd.read_csv(pods_dictionary_path, index_col=False)

    # generate pods based on the pod specification (types and the number of pods)
    counter_id = 0
    pods = pd.DataFrame()
    for i, p in enumerate(pod_types):

        pod_slot_num = pods_dictionary.loc[pods_dictionary["pod_type"] == p].shape[0]
        pod_slot_id = pods_dictionary.loc[pods_dictionary["pod_type"]
                                == p, "slot_sequence"].to_list()
        slot_type = pods_dictionary.loc[pods_dictionary["pod_type"]
                                == p, "slot_type"].to_list()       

        pod_type = [p] * pod_slot_num
        pod_face = pods_dictionary.loc[pods_dictionary["pod_type"] == p, "pod_face"].to_list()

        for _ in range(pod_num[i]):
            pod_id = [counter_id] * pod_slot_num
            counter_id += 1

            pod = pd.DataFrame({"pod_id": pod_id,
                                "pod_type": pod_type,
                                "slot_id": pod_slot_id,
                                "slot_type": slot_type,
                                "item": np.nan,
                                "unusedColumn1": 0,
                                "unusedColumn2": 0,
                                "unusedColumn3": 0,
                                "qty": np.nan,
                                "max_qty": np.nan,
                                'due_date': 99999,
                                'facing': pod_face,
                                'pick_ind': 0
                                })
            pods = pd.concat([pods, pod], axis=0)

    pods.reset_index(drop=True, inplace=True)
    return pods


def assign_items_to_pods(pods, items, items_pods_class_conf, class_slot_counts={"A": 12, "B": 21, "C": 7}, dev_mode=False):
    # =================================================================
    # Tahap 1: Persiapan & Kumpulin Data
    # =================================================================
    PARENT_DIRECTORY = "." # Ganti dengan path yang sesuai
    
    item_slot_configuration_path = PARENT_DIRECTORY + "/data/output/items_slots_configuration.csv"
    items_slots_configuration = pd.read_csv(item_slot_configuration_path, index_col=False)
    
    items_slots_configuration_selected = items_slots_configuration.loc[
        (items_slots_configuration["item_code"].isin(items["item_code"].unique())) &
        (items_slots_configuration["max_fit"] > 0)
    ]
    
    items["item_id"] = items.index
    items_slots_configuration_selected = items_slots_configuration_selected.merge(
        items.drop(columns=['box_volume', 'item_volume', 
                             'number_of_item_in_a_box', 'max_fit'], errors='ignore'),
        how='inner', on='item_code'
    )
    
    items_to_place = items_slots_configuration_selected[
        items_slots_configuration_selected['item_initial_quantity_inventory'] > 0
    ].sort_values(by=['item_class', 'item_id']).copy()

    # =================================================================
    # Tahap 2: Misi Utama - Alokasi MENYEBAR (Horizontal)
    # =================================================================
    
    # 1. Buat "Kolam Virtual" untuk setiap kelas
    print("INFO: Mempersiapkan 'kolam virtual' untuk setiap kelas slot...")
    all_class_slots_pools = {item_class: [] for item_class in class_slot_counts.keys()}
    
    empty_slots = pods[pods['item'].isnull()].copy()
    empty_slots_grouped_by_pod = empty_slots.groupby('pod_id')
    
    for pod_id, group in empty_slots_grouped_by_pod:
        slots_in_pod_indices = group.index.tolist()
        cursor = 0
        for item_class, count in class_slot_counts.items():
            # Ambil 'jatah' slot dari pod ini untuk kelas ini
            end_cursor = cursor + count
            slots_for_this_class = slots_in_pod_indices[cursor:end_cursor]
            all_class_slots_pools[item_class].extend(slots_for_this_class)
            cursor = end_cursor

    # 2. Loop per KELAS BARANG, bukan per POD
    for item_class, target_slots_count in class_slot_counts.items():
        if target_slots_count == 0:
            continue
            
        # Ambil kolam yang sesuai
        class_pool = all_class_slots_pools[item_class]
        items_of_this_class = items_to_place[items_to_place['item_class'] == item_class]
        
        print(f"\n--- Mengalokasikan untuk Kelas {item_class} ke dalam {len(class_pool)} slot yang tersedia ---")
        
        placed_in_pool_count = 0
        for _, item_config in items_of_this_class.iterrows():
            if placed_in_pool_count >= len(class_pool):
                print(f"WARNING: Kolam slot untuk kelas {item_class} sudah penuh.")
                break
            
            item_id_to_place = item_config['item_id']
            if item_id_to_place not in items_to_place['item_id'].values:
                continue

            slots_needed = np.ceil(item_config['item_initial_quantity_inventory'] / item_config['max_fit']).astype(int)

            # Cek apakah di 'kolam' ini masih ada cukup tempat
            if (len(class_pool) - placed_in_pool_count) >= slots_needed:
                # Ambil slot dari 'kolam'. Karena kolam ini isinya dari banyak pod,
                # alokasinya akan otomatis menyebar.
                slot_indices_for_this_item = class_pool[placed_in_pool_count : placed_in_pool_count + slots_needed]

                pods.loc[slot_indices_for_this_item, 'item'] = item_id_to_place
                pods.loc[slot_indices_for_this_item, 'qty'] = int(item_config['max_fit'])
                pods.loc[slot_indices_for_this_item, 'max_qty'] = int(item_config['max_fit'])
                pods.loc[slot_indices_for_this_item, 'item_weight'] = item_config['item_weight']
                pods.loc[slot_indices_for_this_item, 'total_item_weight'] = round(item_config['item_weight'] * int(item_config['max_fit']), 3)
                pods.loc[slot_indices_for_this_item, 'item_pod_inventory_level'] = item_config['item_pod_inventory_level']
                pods.loc[slot_indices_for_this_item, 'item_warehouse_inventory_level'] = item_config['item_warehouse_inventory_level']
                
                print(f"Item {item_id_to_place} (Kelas {item_class}) ditempatkan secara menyebar di {slots_needed} slot.")
                
                placed_in_pool_count += slots_needed
                items_to_place = items_to_place[items_to_place['item_id'] != item_id_to_place]
            else:
                print(f"INFO: Sisa slot di kolam Kelas {item_class} tidak cukup untuk item {item_id_to_place} (butuh {slots_needed}, sisa {len(class_pool) - placed_in_pool_count}).")

    # =================================================================
    # Tahap 3: SAPU BERSIH (Mop-up Phase)
    # =================================================================
    if not items_to_place.empty:
        print(f"\nINFO: Masih ada {len(items_to_place)} item belum teralokasi. Memulai tahap sapu bersih...")
        
        for index, item_config in items_to_place.copy().iterrows():
            current_empty_slots = pods[pods['item'].isnull()]
            if current_empty_slots.empty:
                print("ERROR: Slot di seluruh warehouse sudah habis, tapi item masih ada.")
                break

            item_id_to_place = item_config['item_id']
            max_fit = item_config['max_fit']
            inventory = item_config['item_initial_quantity_inventory']
            slots_needed = np.ceil(inventory / max_fit).astype(int)
            
            if len(current_empty_slots) >= slots_needed:
                slot_indices_for_this_item = current_empty_slots.head(slots_needed).index
                
                # Assign data ke dataframe 'pods'
                pods.loc[slot_indices_for_this_item, 'item'] = item_id_to_place
                pods.loc[slot_indices_for_this_item, 'qty'] = int(max_fit)
                pods.loc[slot_indices_for_this_item, 'max_qty'] = int(max_fit)
                pods.loc[slot_indices_for_this_item, 'item_weight'] = item_config['item_weight']
                pods.loc[slot_indices_for_this_item, 'total_item_weight'] = round(item_config['item_weight'] * int(max_fit), 3)
                pods.loc[slot_indices_for_this_item, 'item_pod_inventory_level'] = item_config['item_pod_inventory_level']
                pods.loc[slot_indices_for_this_item, 'item_warehouse_inventory_level'] = item_config['item_warehouse_inventory_level']
                
                # Hapus item dari daftar `items_to_place`
                items_to_place = items_to_place.drop(index)
                
                print(f"  -> Sapu Bersih: Item {item_id_to_place} ditempatkan di {slots_needed} slot.")
            else:
                print(f"  -> WARNING: Slot tidak cukup untuk item {item_id_to_place} di tahap sapu bersih. Butuh {slots_needed}, sisa {len(current_empty_slots)}.")

    # =================================================================
    # Tahap 4: Beberes dan Simpan Hasil
    # =================================================================
    pods.fillna({'item': -1, 'qty': 0, 'max_qty': 0}, inplace=True)
    pods[["item", "qty", "max_qty"]] = pods[["item", "qty", "max_qty"]].astype(int)
    
    pods.to_csv(PARENT_DIRECTORY + "/data/output/pods.csv", index=False)
    
    return pods

#=========================================================== OLD CODE BUT GOOD ==============================================================
# def assign_items_to_pods(pods, items, items_pods_class_conf, class_slot_counts={"A": 7, "B": 15, "C": 18}, dev_mode=False):
#     # =================================================================
#     # Tahap 1: Persiapan & Kumpulin Data
#     # =================================================================
#     PARENT_DIRECTORY = "." # Ganti dengan path yang sesuai
    
#     item_slot_configuration_path = PARENT_DIRECTORY + "/data/output/items_slots_configuration.csv"
#     items_slots_configuration = pd.read_csv(item_slot_configuration_path, index_col=False)
    
#     items_slots_configuration_selected = items_slots_configuration.loc[
#         (items_slots_configuration["item_code"].isin(items["item_code"].unique())) &
#         (items_slots_configuration["max_fit"] > 0)
#     ]
    
#     items["item_id"] = items.index
#     items_slots_configuration_selected = items_slots_configuration_selected.merge(
#         items.drop(columns=['box_volume', 'item_volume', 
#                              'number_of_item_in_a_box', 'max_fit'], errors='ignore'),
#         how='inner', on='item_code'
#     )
    
#     items_to_place = items_slots_configuration_selected[
#         items_slots_configuration_selected['item_initial_quantity_inventory'] > 0
#     ].sort_values(by=['item_class', 'item_id']).copy()

#     # =================================================================
#     # Tahap 2: Misi Utama - Alokasi per Pod Sesuai JATAH PASTI
#     # =================================================================
    
#     # Loop untuk setiap pod unik
#     for pod_id in pods['pod_id'].unique():
#         available_slots_in_pod = pods[(pods['pod_id'] == pod_id) & (pods['item'].isnull())]
        
#         if available_slots_in_pod.empty:
#             continue # Pod ini udah penuh, skip

#         # Langsung pake jatah pasti dari parameter
#         class_targets = class_slot_counts.copy()
        
#         filled_slots_indices = []
        
#         # Loop untuk setiap kelas: A, B, C
#         for item_class, target_slots_count in class_targets.items():
#             if target_slots_count == 0:
#                 continue

#             items_of_this_class = items_to_place[items_to_place['item_class'] == item_class]
            
#             current_available_slots = available_slots_in_pod.index.difference(filled_slots_indices)
#             slots_to_fill_for_this_class = current_available_slots[:target_slots_count]
            
#             placed_count = 0
#             for _, item_config in items_of_this_class.iterrows():
#                 if placed_count >= len(slots_to_fill_for_this_class):
#                     break 
                
#                 item_id_to_place = item_config['item_id']
#                 if item_id_to_place not in items_to_place['item_id'].values:
#                     continue 

#                 slots_needed = np.ceil(item_config['item_initial_quantity_inventory'] / item_config['max_fit']).astype(int)
                
#                 if (len(slots_to_fill_for_this_class) - placed_count) >= slots_needed:
#                     slot_indices_for_this_item = slots_to_fill_for_this_class[placed_count : placed_count + slots_needed]
                    
#                     pods.loc[slot_indices_for_this_item, 'item'] = item_id_to_place
#                     pods.loc[slot_indices_for_this_item, 'qty'] = int(item_config['max_fit'])
#                     pods.loc[slot_indices_for_this_item, 'max_qty'] = int(item_config['max_fit'])
#                     pods.loc[slot_indices_for_this_item, 'item_weight'] = item_config['item_weight']
#                     pods.loc[slot_indices_for_this_item, 'total_item_weight'] = round(item_config['item_weight'] * int(item_config['max_fit']), 3)
#                     pods.loc[slot_indices_for_this_item, 'item_pod_inventory_level'] = item_config['item_pod_inventory_level']
#                     pods.loc[slot_indices_for_this_item, 'item_warehouse_inventory_level'] = item_config['item_warehouse_inventory_level']

#                     print(f"Pod {pod_id}: Item {item_id_to_place} of class {item_class} placed in {slots_needed} slots.")
                    
#                     placed_count += slots_needed
#                     filled_slots_indices.extend(slot_indices_for_this_item)
#                     items_to_place = items_to_place[items_to_place['item_id'] != item_id_to_place]

#     # =================================================================
#     # Tahap 3: SAPU BERSIH (Mop-up Phase) - Untuk item yang belum kebagian tempat
#     # =================================================================
#     if not items_to_place.empty:
#         print(f"INFO: Masih ada {len(items_to_place)} item belum teralokasi. Memulai tahap sapu bersih...")
        
#         # Loop untuk setiap item 'gelandangan'
#         # Kita copy iterrows() biar aman saat modifikasi dataframe `items_to_place`
#         for index, item_config in items_to_place.copy().iterrows():
            
#             # Cek slot kosong terbaru di seluruh warehouse setiap loop
#             current_empty_slots = pods[pods['item'].isnull()]
#             if current_empty_slots.empty:
#                 print("ERROR: Slot di seluruh warehouse sudah habis, tapi item masih ada.")
#                 break 

#             item_id_to_place = item_config['item_id']
#             max_fit = item_config['max_fit']
#             inventory = item_config['item_initial_quantity_inventory']
#             slots_needed = np.ceil(inventory / max_fit).astype(int)
            
#             if len(current_empty_slots) >= slots_needed:
#                 # Ada tempat! Ambil slot kosong pertama yang ditemui
#                 slot_indices_for_this_item = current_empty_slots.head(slots_needed).index
                
#                 # Assign data ke dataframe 'pods'
#                 pods.loc[slot_indices_for_this_item, 'item'] = item_id_to_place
#                 pods.loc[slot_indices_for_this_item, 'qty'] = int(max_fit)
#                 pods.loc[slot_indices_for_this_item, 'max_qty'] = int(max_fit)
#                 pods.loc[slot_indices_for_this_item, 'item_weight'] = item_config['item_weight']
#                 pods.loc[slot_indices_for_this_item, 'total_item_weight'] = round(item_config['item_weight'] * int(max_fit), 3)
#                 pods.loc[slot_indices_for_this_item, 'item_pod_inventory_level'] = item_config['item_pod_inventory_level']
#                 pods.loc[slot_indices_for_this_item, 'item_warehouse_inventory_level'] = item_config['item_warehouse_inventory_level']
                
#                 # Hapus item dari daftar `items_to_place`
#                 items_to_place = items_to_place.drop(index)
                
#                 print(f"  -> Sapu Bersih: Item {item_id_to_place} ditempatkan di {slots_needed} slot.")
#             else:
#                 print(f"  -> WARNING: Slot tidak cukup untuk item {item_id_to_place} di tahap sapu bersih. Butuh {slots_needed}, sisa {len(current_empty_slots)}.")

#     # =================================================================
#     # Tahap 4: Beberes dan Simpan Hasil
#     # =================================================================
#     pods.fillna({'item': -1, 'qty': 0, 'max_qty': 0}, inplace=True)
#     pods[["item", "qty", "max_qty"]] = pods[["item", "qty", "max_qty"]].astype(int)
    
#     pods.to_csv(PARENT_DIRECTORY + "/data/output/pods.csv", index=False)
    
#     return pods


# ================================================ old code DEFAULT ==========================================================
# def assign_items_to_pods(pods, items, items_pods_class_conf, dev_mode=False):
#     # =================================================================
#     # Tahap 1: Persiapan & Kumpulin Data
#     # =================================================================
#     PARENT_DIRECTORY = "." # Ganti dengan path yang sesuai
    
#     item_slot_configuration_path = PARENT_DIRECTORY + "/data/output/items_slots_configuration.csv"
#     items_slots_configuration = pd.read_csv(item_slot_configuration_path, index_col=False)
    
#     items_slots_configuration_selected = items_slots_configuration.loc[
#         (items_slots_configuration["item_code"].isin(items["item_code"].unique())) &
#         (items_slots_configuration["max_fit"] > 0)
#     ]
    
#     items["item_id"] = items.index
#     items_slots_configuration_selected = items_slots_configuration_selected.merge(
#         items.drop(columns=['box_volume', 'item_volume',
#                             'number_of_item_in_a_box', 'max_fit']), # Drop kolom duplikat potensial
#         how='inner', on='item_code'
#     )
    
#     # =================================================================
#     # Tahap 2: Misi Utama - Logika Paling Simpel
#     # =================================================================
#     items_to_place = items_slots_configuration_selected[
#         items_slots_configuration_selected['item_initial_quantity_inventory'] > 0
#     ].copy()
    
#     # Loop untuk setiap item unik yang perlu ditempatkan
#     # Kita tidak perlu lagi mengurutkan atau mencari config terbaik
#     for item_id in items_to_place['item_id'].unique():
        
#         # Langsung ambil satu-satunya config untuk item ini
#         config = items_to_place[items_to_place['item_id'] == item_id].iloc[0]

#         # print("ini config bos:", config)
        
#         max_fit_value = config['max_fit']
#         inventory_to_place = config['item_initial_quantity_inventory']
        
#         # Hitung berapa slot yang dibutuhkan
#         slots_needed = np.ceil(inventory_to_place / max_fit_value).astype(int)
        
#         # Cari slot kosong MANAPUN yang tersedia
#         available_slots = pods.loc[pods['item'].isnull()]
        
#         if len(available_slots) < slots_needed:
#             print(f"Error: Slot tidak cukup untuk item_id {item_id}. Butuh {slots_needed}, tersedia {len(available_slots)}.")
#             continue
            
#         slots_to_fill = available_slots.head(slots_needed)
        
#         qty = int(max_fit_value)
#         item_weight = config['item_weight']
#         total_item_weight = round(item_weight * qty, 3)
#         item_inventory_level = config['item_pod_inventory_level']
#         item_global_level = config['item_warehouse_inventory_level']
        
#         # Assign item ke slot-slot yang terpilih
#         pods.loc[slots_to_fill.index, 'item'] = item_id
#         pods.loc[slots_to_fill.index, 'qty'] = qty
#         pods.loc[slots_to_fill.index, 'max_qty'] = qty
#         pods.loc[slots_to_fill.index, 'item_weight'] = item_weight
#         pods.loc[slots_to_fill.index, 'total_item_weight'] = total_item_weight
#         pods.loc[slots_to_fill.index, 'item_pod_inventory_level'] = item_inventory_level
#         pods.loc[slots_to_fill.index, 'item_warehouse_inventory_level'] = item_global_level

#     # =================================================================
#     # Tahap 3: Isi Sisa Slot Kosong (Logika ini tetap sama)
#     # =================================================================
#     # (Bagian ini tidak diubah karena masih relevan untuk mengisi sisa ruang)
#     pod_still_available = pods.loc[pods["item"].isnull(), "pod_id"].unique().tolist()
#     items = pd.merge(items, pd.DataFrame(list(items_pods_class_conf.items()), columns=['item_class', 'item_pod_class_ratio']), on='item_class', how='left')
    
#     for pod_id in pod_still_available:
#         # ... logika pengisian sisa slot tetap berjalan seperti sebelumnya ...
#         pass # Placeholder untuk menunjukkan logika lama tetap di sini

#     # =================================================================
#     # Tahap 4: Beberes dan Simpan Hasil
#     # =================================================================
#     pods.fillna({'item': -1, 'qty': 0, 'max_qty': 0}, inplace=True)
#     pods[["item", "qty", "max_qty"]] = pods[["item", "qty", "max_qty"]].astype(int)
    
#     pods.to_csv(PARENT_DIRECTORY + "/data/output/pods.csv", index=False)
    
#     return pods

def check_items_pods_feasibility(total_sku, pod_types, pods_dictionary):
        feasible = True
        for pod_type in pod_types:
            total_slot = pods_dictionary.loc[pods_dictionary["pod_type"] == pod_type].shape[0]
            if  total_slot > total_sku:
                print("Pod type", pod_type, "has more slot ("+str(total_slot)+") than total SKU ("+str(total_sku)+").")
                print("In this system we asume that each pod cannot store same item in multiple slot.")
                print("Please adjust the number of items before we can continue to generate the items and pods.")
                feasible = False
                break
        return feasible