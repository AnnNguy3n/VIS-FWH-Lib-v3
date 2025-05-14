import pandas as pd
import numpy as np
import os
import json
from colorama import Fore, Style
import time
import multiprocessing
from PySources import suppFunc
import datetime
import sys


def run_worker(lib_abs_path, generate_method, filter_name, worker_type, config_path, wait_before_run, timeout):
    """
    Run a worker process for a single task, with given parameters.
    """
    time.sleep(wait_before_run)

    command = f"{lib_abs_path}ExeFile/"
    command += suppFunc.generate_method[generate_method]["command"]
    command += suppFunc.filter_fields[filter_name]["command"]

    if worker_type == "CPU":
        command += "CPU.exe"
    elif worker_type == "GPU":
        command += "CUDA.exe"
    else:
        # Raise clear exception for unknown worker type
        raise ValueError(f"Unknown worker type: {worker_type}")

    command += f" {config_path}"

    print(Fore.LIGHTCYAN_EX + f"Run {worker_type} worker with input:",
          Fore.LIGHTMAGENTA_EX + f"; {lib_abs_path}; {generate_method}; {filter_name}; {worker_type}; {config_path}",
          Style.RESET_ALL)
    now = datetime.datetime.now()
    print(f"Time start: {now.hour}-{now.minute}-{now.second}")
    estimated_end = now + datetime.timedelta(minutes=timeout)
    print(f"Estimated time end: {estimated_end.hour}-{estimated_end.minute}-{estimated_end.second}")
    os.system(f"start /wait cmd /c {command}")


# ----------------- Utility functions for data and config preparation -----------------
def prepare_data_paths(data_path, warehouse_path, folder_formula):
    """
    Prepare data_full.xlsx and data_train.xlsx for a given dataset.
    Returns the data_train path and dataset name.
    """
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    warehouse_dataset_path = os.path.join(warehouse_path, dataset_name)
    formula_dataset_path = os.path.join(folder_formula, dataset_name)
    os.makedirs(warehouse_dataset_path, exist_ok=True)
    os.makedirs(formula_dataset_path, exist_ok=True)

    # Prepare data_full.xlsx
    try:
        data_full = pd.read_excel(os.path.join(warehouse_dataset_path, "data_full.xlsx"))
        print(Fore.GREEN + f"Read data_full: {warehouse_dataset_path}/data_full.xlsx", Style.RESET_ALL)
    except FileNotFoundError:
        data = pd.read_excel(data_path)
        data.to_excel(os.path.join(warehouse_dataset_path, "data_full.xlsx"), index=False)
        data.to_excel(os.path.join(formula_dataset_path, "data_full.xlsx"), index=False)
        data_full = data
        print(Fore.GREEN + f"Created data_full: {warehouse_dataset_path}/data_full.xlsx", Style.RESET_ALL)

    suppFunc.compare_dfs(pd.read_excel(data_path), data_full)

    # Prepare data_train.xlsx
    try:
        data_train = pd.read_excel(os.path.join(warehouse_dataset_path, "data_train.xlsx"))
        print(Fore.GREEN + f"Read data_train: {warehouse_dataset_path}/data_train.xlsx", Style.RESET_ALL)
    except FileNotFoundError:
        max_cycle = data_full["TIME"].max()
        data_train = data_full[data_full["TIME"] < max_cycle].reset_index(drop=True)
        data_train.index += 1
        for col in data_train.columns:
            data_train.loc[0, col] = "_NULL_" if data_train[col].dtype == "object" else 0
        data_train.loc[0, "TIME"] = max_cycle
        data_train.sort_index(inplace=True)
        data_train.to_excel(os.path.join(warehouse_dataset_path, "data_train.xlsx"), index=False)
        data_train.to_excel(os.path.join(formula_dataset_path, "data_train.xlsx"), index=False)
        print(Fore.GREEN + f"Created data_train: {warehouse_dataset_path}/data_train.xlsx", Style.RESET_ALL)

    return os.path.join(warehouse_dataset_path, "data_train.xlsx"), dataset_name


def generate_task_config(config_item, warehouse_path, folder_formula, lib_abs_path, timeout_per_task):
    """
    Generate config.txt for a task and return its path and relevant info.
    """
    data_path, dataset_name = prepare_data_paths(config_item["data_path"], warehouse_path, folder_formula)

    # Validate eval_method
    eval_method = config_item.get("eval_method")
    if eval_method not in [0, 1]:
        raise ValueError(f"Invalid eval_method = {eval_method}. Only 0 (Classic) or 1 (Root) are allowed.")

    # Determine eval_folder
    eval_folder = "Classic" if eval_method == 0 else "Root"

    # Update save folder path
    save_folder = os.path.join(warehouse_path, dataset_name, config_item["generate_method"], eval_folder, config_item["filter"])
    os.makedirs(save_folder, exist_ok=True)

    # Prepare task config lines
    num_strategy = config_item["num_strategy"]
    filter_fields = []
    for i in range(1, num_strategy + 1):
        filter_fields.extend([
            f"ValGeo{i}",
            f"GeoNgn{i}",
            f"ValHar{i}",
            f"HarNgn{i}"
        ])

    filter_field_str = ";".join(filter_fields)

    task_lines = [
        f"data_path = {data_path}",
        f"filter_field = {filter_field_str}",
        f"interest = {config_item['interest']}",
        f"valuearg_threshold = {config_item['valuearg_threshold']}",
        f"folder_save = {save_folder}",
        # f"eval_index = {config_item['eval_index']}",
        f"eval_threshold = {config_item['eval_threshold']}",
        # f"threshold_delta = {config_item['threshold_delta']}",
        f"eval_method = {eval_method}",     # Now added
        f"storage_size = {config_item.get('temp_storage_size', 1000)}",
        f"num_cycle = {config_item['num_cycle']}",
        f"lib_abs_path = {lib_abs_path}",
        f"timeout_in_minutes = {timeout_per_task}",
        f"num_strategy = {num_strategy}",
    ]

    config_path = os.path.join(save_folder, "config.txt")

    with open(config_path, "w") as f:
        f.write("\n".join(task_lines))
        print(Fore.GREEN + f"Created task_config: {config_path}", Style.RESET_ALL)

    print("\n".join(task_lines))
    print()

    return config_path, config_item['generate_method'], config_item['filter']


def get_worker_types(num_worker, worker_type):
    """
    Determine the list of worker types to use based on config.
    """
    if num_worker == 1:
        return [worker_type]
    if num_worker == 2:
        return ["GPU", "CPU"] if worker_type == "Hybrid" else ["GPU", "GPU"]
    return ["GPU"] * 3


if __name__ == "__main__":
    # Parse CLI arguments
    list_sys_args = [tuple(t_.split("=")) for t_ in sys.argv[1:]]
    dict_sys_args = {t_[0]:t_[1] for t_ in list_sys_args}
    print(dict_sys_args)
    with open("config.json", "r") as f:
        config = json.load(f)

    num_worker = config[0]["num_worker"]
    worker_type = config[0]["worker_type"]
    timeout_per_task = config[0]["timeout_per_task"]
    warehouse_path = config[0]["warehouse_path"]
    folder_formula = config[0]["folder_formula"]

    # Assertions for config sanity
    assert num_worker in [1, 2, 3]
    assert worker_type in ["GPU", "CPU", "Hybrid"]
    assert timeout_per_task >= 1
    assert not (num_worker == 1 and worker_type == "Hybrid")
    # assert not (num_worker == 3 and worker_type == "GPU")
    assert not (num_worker != 1 and worker_type == "CPU")

    lib_abs_path = __file__.replace("main.py", "")

    # Prepare all task configs
    list_config_path, list_generate_method, list_filter_name = [], [], []
    for task in config[1:]:
        config_path, generate_method, filter_name = generate_task_config(task, warehouse_path, folder_formula, lib_abs_path, timeout_per_task)
        list_config_path.append(config_path)
        list_generate_method.append(generate_method)
        list_filter_name.append(filter_name)

    # Determine worker types for this run
    list_worker_type = get_worker_types(num_worker, worker_type)

    n = len(list_config_path)
    while True:
        if "detail_only" in dict_sys_args.keys() and dict_sys_args["detail_only"] == "True":
            pass
        else:
            for i in range(n):
                list_worker_input = []
                for j in range(num_worker):
                    generate_method = list_generate_method[(i+j)%n]
                    filter_name = list_filter_name[(i+j)%n]
                    worker_type = list_worker_type[j]
                    config_path = list_config_path[(i+j)%n]

                    list_worker_input.append((
                        lib_abs_path,
                        generate_method,
                        filter_name,
                        worker_type,
                        config_path,
                        j,
                        timeout_per_task
                    ))

                pool = multiprocessing.Pool(processes=num_worker)
                pool.starmap(run_worker, list_worker_input)
                pool.close()
                pool.join()
                time.sleep(10)

        # Tong ket cong thuc o day
        ...
