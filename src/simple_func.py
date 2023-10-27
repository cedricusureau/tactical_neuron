import os


def make_results_folder(run_name):
    # Make subdirectory in results and figures directory if they don't exist
    if not os.path.exists(f"results/{run_name}"):
        os.makedirs(f"results/{run_name}")
    if not os.path.exists(f"figures/{run_name}"):
        os.makedirs(f"figures/{run_name}")
