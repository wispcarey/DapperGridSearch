import re
import os
import math
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from contextlib import contextmanager

# dapper
import dapper as dpr 
import dapper.da_methods as da
import dapper.mods as modelling
from dapper.tools.localization import nd_Id_localization
from dapper.mods.Lorenz96 import LPs

# customized
from grid_search_config import GRID_SEARCH_INFO

@contextmanager
def suppress_output():
    """
    Context manager to suppress stdout and stderr.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull  # Redirect stdout to null
            sys.stderr = devnull  # Redirect stderr to null
            yield
        finally:
            sys.stdout = old_stdout  # Restore stdout
            sys.stderr = old_stderr  # Restore stderr

def plot_heatmap_with_nan(data, x_list, y_list, save_path=None, img_title="Grid Search"):
    """
    Plots a heatmap with NaN values handled and saves it to a specified path.

    Parameters:
        data (numpy.ndarray): 2D array of data values (NaN values are allowed).
        x_list (list or numpy.ndarray): List of values for the x-axis (corresponding to rows of data).
        y_list (list or numpy.ndarray): List of values for the y-axis (corresponding to columns of data).
        save_path (str): File path to save the plot.

    Returns:
        None
    """
    # Validate dimensions
    if data.shape[0] != len(x_list):
        raise ValueError("The length of x_list must match the number of rows in data.")
    if data.shape[1] != len(y_list):
        raise ValueError("The length of y_list must match the number of columns in data.")

    # Replace NaN with a value greater than the maximum
    max_value = np.nanmax(data)
    nan_value = max_value * 1.1  # Set NaN to 1.1 times the max value
    data_with_nan_replaced = np.where(np.isnan(data), nan_value, data)

    # Create grid edges (for pcolormesh)
    x_edges = np.linspace(x_list[0] - (x_list[1] - x_list[0]) / 2, 
                          x_list[-1] + (x_list[1] - x_list[0]) / 2, len(x_list) + 1)
    y_edges = np.linspace(y_list[0] - (y_list[1] - y_list[0]) / 2, 
                          y_list[-1] + (y_list[1] - y_list[0]) / 2, len(y_list) + 1)

    # Use the 'jet' colormap
    cmap = plt.cm.jet

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    mesh = plt.pcolormesh(y_edges, x_edges, data_with_nan_replaced, cmap=cmap, shading='auto')

    # Add a colorbar
    cbar = plt.colorbar(mesh)
    cbar.set_label("Values", fontsize=15)

    # Adjust colorbar ticks to include NaN
    cbar_ticks = np.linspace(np.nanmin(data), max_value, num=6)  # Generate ticks for original data range
    cbar_ticks = np.append(cbar_ticks, nan_value)  # Add NaN as the last tick
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{tick:.2f}" for tick in cbar_ticks[:-1]] + ["NaN"])  # Add "NaN" label
    cbar.ax.tick_params(labelsize=15)

    # Explicitly set ticks and labels
    plt.xticks(ticks=y_list, 
               labels=[f"{y_list[0]:.0e}"] + [f"{val:.0f}" for val in y_list[1:]], 
               fontsize=15,
               rotation=45)
    plt.yticks(ticks=x_list, 
               labels=[f"{val:.2f}" for val in x_list], 
               fontsize=15)

    # Label axes
    plt.xlabel("Localization Radius", fontsize=15)
    plt.ylabel("Inflation Factors", fontsize=15)

    # Set title
    plt.title(img_title, fontsize=18)

    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')  # Save plot to file
    plt.close()  # Close the plot to release memory

def plot_simple(x, y, save_path=None, img_title="Infl Search"):
    """
    Plots a simple 2D line plot for two arrays, handling NaN values.

    Parameters:
        x (list or numpy.ndarray): Array of x-axis values.
        y (list or numpy.ndarray): Array of y-axis values.
        save_path (str): File path to save the plot. If None, plot is shown instead.
        img_title (str): Title of the plot. Default is "Infl Search".

    Returns:
        None
    """
    # Convert inputs to numpy arrays if they aren't already
    x = np.array(x)
    y = np.array(y)

    # Check if y contains NaN values
    if np.all(np.isnan(y)):  # All values are NaN
        y = np.zeros_like(y)  # Replace all NaN with 0
        special_ticks = {0: "NaN"}  # Add 0 as a tick labeled "NaN"
        max_value = 0  # No valid max value
    elif np.any(np.isnan(y)):  # Some values are NaN
        max_non_nan = np.nanmax(y)  # Get maximum non-NaN value
        special_value = max_non_nan * 1.3
        nan_indices = np.where(np.isnan(y))  # Indices of NaN values
        y[nan_indices] = special_value  # Replace NaN with 1.3 * max
        special_ticks = {special_value: "NaN"}  # Add special value to ticks
        max_value = max_non_nan
    else:
        special_ticks = {}  # No NaN, no special ticks
        max_value = np.max(y)

    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')  # Line with markers

    # Add labels and title
    plt.title(img_title, fontsize=18)
    plt.xlabel("Inflation Factors", fontsize=15)
    plt.ylabel("RMSE", fontsize=15)

    # Customize y-ticks to only show up to slightly beyond max_value
    if special_ticks:
        max_tick_value = max_value * 1.05  # Allow a bit of padding beyond max value
        current_ticks = plt.yticks()[0]  # Get current y-ticks
        filtered_ticks = [tick for tick in current_ticks if tick <= max_tick_value]  # Filter ticks
        filtered_ticks = np.append(filtered_ticks, list(special_ticks.keys()))  # Add special ticks
        plt.yticks(
            filtered_ticks,
            labels=[
                f"{tick:.2f}" if tick not in special_ticks else special_ticks[tick]
                for tick in filtered_ticks
            ],
            fontsize=15
        )
    else:
        plt.yticks(fontsize=15)

    # Customize x-ticks
    plt.xticks(fontsize=15, rotation=45, ha='right')

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')  # Save plot to file
    else:
        plt.show()  # Show plot interactively
    plt.close()  # Close the plot to release memory

def create_HMM(dataset, sigma_y):
    """
    Create a Hidden Markov Model (HMM) based on the specified dataset.

    Parameters:
    -----------
    dataset : str
        Name of the dataset ('ks', 'lorenz96', 'lorenz63').
    sigma_y : float
        Observation noise standard deviation.

    Returns:
    --------
    HMM : modelling.HiddenMarkovModel
        The created Hidden Markov Model for the specified dataset.
    """
    if dataset == 'ks':
        from dapper.mods.KS import Model, Tplot
        
        KS = Model(dt=GRID_SEARCH_INFO[dataset]["dt"])
        Nx = KS.Nx

        tseq = modelling.Chronology(
            dt=GRID_SEARCH_INFO[dataset]["dt"],
            dto=GRID_SEARCH_INFO[dataset]["dto"],
            Ko=GRID_SEARCH_INFO[dataset]["ko"],
            BurnIn=1000,
            Tplot=Tplot
        )

        Dyn = {
            "M": Nx,
            "model": KS.step,
            "linear": KS.dstep_dx,
            "noise": 0,
        }

        X0 = modelling.GaussRV(mu=KS.x0, C=1)

        jj = GRID_SEARCH_INFO[dataset]["obs_inds"]
        Obs = modelling.partial_Id_Obs(Nx, jj)
        Obs["noise"] = sigma_y ** 2
        Obs["localizer"] = nd_Id_localization((Nx,), (4,), jj)

        HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

    elif dataset == 'lorenz96':
        from dapper.mods.Lorenz96 import Tplot, dstep_dx, step, x0
        
        tseq = modelling.Chronology(
            dt=GRID_SEARCH_INFO[dataset]["dt"],
            dto=GRID_SEARCH_INFO[dataset]["dto"],
            Ko=GRID_SEARCH_INFO[dataset]["ko"],
            Tplot=Tplot,
            BurnIn=2 * Tplot
        )
        
        Nx = 40
        x0 = x0(Nx)

        Dyn = {
            "M": Nx,
            "model": step,
            "linear": dstep_dx,
            "noise": 0,
        }

        X0 = modelling.GaussRV(mu=x0, C=0.1)

        jj = GRID_SEARCH_INFO[dataset]["obs_inds"]
        Obs = modelling.partial_Id_Obs(Nx, jj)
        Obs["noise"] = sigma_y ** 2
        Obs["localizer"] = nd_Id_localization((Nx,), (2,), jj)

        HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

    elif dataset == 'lorenz63':
        from dapper.mods.Lorenz63 import LPs, Tplot, dstep_dx, step, x0

        tseq = modelling.Chronology(
            dt=GRID_SEARCH_INFO[dataset]["dt"],
            dto=GRID_SEARCH_INFO[dataset]["dto"],
            Ko=GRID_SEARCH_INFO[dataset]["ko"],
            Tplot=Tplot,
            BurnIn=4 * Tplot
        )
        
        Nx = len(x0)

        Dyn = {
            "M": Nx,
            "model": step,
            "linear": dstep_dx,
            "noise": 0,
        }

        X0 = modelling.GaussRV(mu=x0, C=1)

        jj = GRID_SEARCH_INFO[dataset]["obs_inds"]
        Obs = modelling.partial_Id_Obs(Nx, jj)
        Obs["noise"] = sigma_y ** 2

        HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return HMM, tseq



def main(grid_search_info):
    def _process_trial(trial_ind, tseq):
        # with suppress_output():
        for xp in xps:
            xp.seed = trial_ind + 1
        
        with suppress_output():
            save_as = xps.launch(HMM, liveplots=False, save_as=f"noname_{trial_ind}")
            states, _ = HMM.simulate()
            obs_states = states[tseq.kko]
            obs_states_rms = np.mean(np.sqrt(np.mean(obs_states ** 2)))

        averages = dpr.stats.tabulate_avrgs([C.avrgs for C in xps])
        
        
        try:
            # Find the first key that contains 'rmsea'
            rmse_a_key = [key for key in averages.keys() if 'rmse.a' in key][0]
        except Exception as e:
            # If not found, print available keys
            print("Exception message:", e)
            print("Available keys:", averages.keys())

        try:
            # Find the first key that contains 'rmva'
            rmv_a_key = [key for key in averages.keys() if 'rmv.a' in key][0]
        except Exception as e:
            # If not found, log error details and set rmv_a_key to rmse_a_key
            print("Exception message:", e)
            print(f"Error trial: {method_name} on {dataset} with sigma_y = {sigma_y}, ensemble size = {N}")
            print("Available keys:", averages.keys())
            rmv_a_key = rmse_a_key



        rmse_mean, rmse_std, rmv_mean, rmv_std = [], [], [], []

        for entry in averages[rmse_a_key]:
            clean_entry = entry.replace('␣', '').strip()
            if 'nan' in clean_entry.lower():
                rmse_mean.append(float('nan'))
                rmse_std.append(float('nan'))                                  
            else:
                match = re.match(r'([0-9.]+)\s*±\s*([0-9.]+)', clean_entry)
                if match:
                    rmse_mean.append(float(match.group(1)))
                    rmse_std.append(float(match.group(2)))
        
        # Convert lists to numpy arrays for efficient computation
        rmse_mean = np.array(rmse_mean)
        rmse_std = np.array(rmse_std)
                    
        for entry in averages[rmv_a_key]:
            clean_entry = entry.replace('␣', '').strip()
            if 'nan' in clean_entry.lower():
                rmv_mean.append(float('nan'))
                rmv_std.append(float('nan'))
            else:
                match = re.match(r'([0-9.]+)\s*±\s*([0-9.]+)', clean_entry)
                if match:
                    rmv_mean.append(float(match.group(1)))
                    rmv_std.append(float(match.group(2)))
        
        # Convert rmv_mean and rmv_std to numpy arrays
        rmv_mean = np.array(rmv_mean)
        rmv_std = np.array(rmv_std)
        
        rrmse_mean = rmse_mean / obs_states_rms
        rrmse_std = rmse_std / obs_states_rms
        
        return rmse_mean, rmse_std, rmv_mean, rmv_std, rrmse_mean, rrmse_std


    def _combine_results(results):
        len_data = len(results[0])
        
        result_record = np.zeros((GRID_SEARCH_INFO[dataset]["num_trials"], len_data, K))

        for trial_ind, data in enumerate(results):
            for i in range(len_data):
                result_record[trial_ind, i, :] = data[i]
        
        valid_val_count = np.sum(~np.isnan(result_record), axis=0)
        safe_valid_val_count = np.where(valid_val_count == 0, 1, valid_val_count)
        
        # Store original result_record with NaNs
        result_record_original = result_record.copy()
        
        # Replace NaNs with zeros for mean calculation
        result_record[np.isnan(result_record)] = 0
        result_avg = np.sum(result_record, axis=0) / safe_valid_val_count
        result_avg[valid_val_count == 0] = np.nan
        
        # Calculate standard deviation using numpy's nanstd function
        result_std = np.nanstd(result_record_original, axis=0, ddof=1)
        
        return result_avg, result_std, valid_val_count[0, :]

    def _check_trial_processed(dataset, method_name, N, sigma_y):
        file_path = os.path.join("save", f"benchmarks_{dataset}.csv")

        if not os.path.exists(file_path):
            return False
    
        df = pd.read_csv(file_path)
        matching_rows = df[(df['method'] == method_name) & (df['N'] == N) & (df['sigma_y'] == sigma_y)]
        if not matching_rows.empty:
            return True
        else: 
            return False

    for dataset, info in grid_search_info.items():
        for sigma_y in info['sigma_y']:
            HMM, tseq = create_HMM(dataset, sigma_y)            

            for method_name in info['methods']:

                N_list = info['N_list']
        
                for N in N_list:
                    # Check the type of infl_list and loc_rad_list
                    if method_name == "LETKF" and 'letkf_infl_list' in info:
                        # Use letkf_infl_list if available and method_name is "LETKF"
                        if isinstance(info['letkf_infl_list'], list):
                            infl_list = info['letkf_infl_list']
                        elif isinstance(info['letkf_infl_list'], dict):
                            infl_list = info['letkf_infl_list'].get(N, [])
                    else:
                        # Fallback to infl_list
                        if isinstance(info['infl_list'], list):
                            infl_list = info['infl_list']
                        elif isinstance(info['infl_list'], dict):
                            infl_list = info['infl_list'].get(N, [])
                    
                    if method_name == 'LETKF':
                        if isinstance(info['loc_rad_list'], list):
                            loc_rad_list = info['loc_rad_list']
                        elif isinstance(info['loc_rad_list'], dict):
                            loc_rad_list = info['loc_rad_list'].get(N, [])
                    else:
                        loc_rad_list = []
                        
                    trial_processed = _check_trial_processed(dataset, method_name, N, sigma_y)

                    xps = dpr.xpList()

                    for infl in infl_list:
                        if loc_rad_list:
                            for loc_rad in loc_rad_list:
                                xps += da.LETKF(N=N, infl=infl, loc_rad=loc_rad)
                        else:
                            if method_name == 'iEnKS':
                                xps += da.iEnKS("Sqrt", N=N , infl=infl, Lag=2)
                            if method_name == 'iEnKF':
                                xps += da.iEnKS("Sqrt", N=N , infl=infl, Lag=1)
                            elif method_name == 'EnKF_PertObs':
                                xps += da.EnKF("PertObs", N=N, infl=infl)
                            elif method_name == 'EnKF_Sqrt':
                                xps += da.EnKF("Sqrt", N=N, infl=infl)

                    if trial_processed:
                        print(f"Grid search reults exists for {method_name} on {dataset} with sigma_y = {sigma_y}, ensemble size {N}.")
                        data_save_path = os.path.join('save', 'data', f'{dataset}_{method_name}_{N}_{sigma_y}_results.npz')
                        saved_data = np.load(data_save_path)

                        result_avg = saved_data['result_avg']
                        result_std = saved_data['result_std']
                        valid_val_count = saved_data['valid_val_count']
                        infl_list = saved_data['infl_list'].tolist()
                        loc_rad_list = saved_data['loc_rad_list'].tolist()
                    else:
                        print(f"Start grid search for {method_name} on {dataset} with sigma_y = {sigma_y}, ensemble size {N}.")
                        
                        K = len(xps)
                        
                        results = Parallel(n_jobs=-1)(delayed(_process_trial)(trial_ind, tseq) for trial_ind in range(GRID_SEARCH_INFO[dataset]["num_trials"]))
                        result_avg, result_std, valid_val_count = _combine_results(results)

                        data_save_path = os.path.join('save', 'data', f'{dataset}_{method_name}_{N}_{sigma_y}_results.npz')
                        np.savez(data_save_path, 
                                 result_avg=result_avg, 
                                 result_std=result_std,
                                 valid_val_count=valid_val_count, 
                                 infl_list=infl_list, 
                                 loc_rad_list=loc_rad_list)

                    if loc_rad_list:
                        rmse_avg_grid = result_avg[4, :].reshape(len(infl_list), len(loc_rad_list))
                        img_save_path = os.path.join('save', 'figures', f'{dataset}_{method_name}_{N}_{sigma_y}_gridsearch.png')
                        img_title = f"{method_name} Grid Search with Ensemble Size {N}"
                        plot_heatmap_with_nan(rmse_avg_grid, infl_list, loc_rad_list, save_path=img_save_path, img_title=img_title)
                    else:
                        rmse_avg_grid = result_avg[4, :]
                        img_save_path = os.path.join('save', 'figures', f'{dataset}_{method_name}_{N}_{sigma_y}_gridsearch.png')
                        img_title = f"{method_name} Parameter Search with Ensemble Size {N}"
                        plot_simple(infl_list, rmse_avg_grid, save_path=img_save_path, img_title=img_title)
                        
                    exp_info, _, _ = xps.prep_table()

                    # Check if the entire result_avg[4, :] is NaN
                    if np.all(np.isnan(result_avg[4, :])):
                        print("Error: All elements in result_avg[4, :] are NaN. Unable to determine the best RMSE index.")
                        best_rmse_ind = 0  # Assign a default value or handle this case appropriately
                    else:
                        best_rmse_ind = np.nanargmin(result_avg[4, :])  # Find index of minimum non-NaN value
                    
                    
                    if loc_rad_list:
                        best_loc_rad = exp_info['loc_rad'][best_rmse_ind]
                    else:
                        best_loc_rad = ""
                    
                    if valid_val_count[best_rmse_ind] < GRID_SEARCH_INFO[dataset]["num_trials"]:
                        nan_exist = True
                    else:
                        nan_exist = False

                    # Define lists of metrics and their types
                    metric_names = ["rmse", "rmse_std", "rmv_mean", "rmv_std", "rrmse_mean", "rrmse_std"]
                    
                    # Initialize an empty dictionary for DataFrame
                    df_dict = {
                        "method": [method_name],
                        "N": [N],
                        "sigma_y": [sigma_y],
                        "best_loc_rad": [best_loc_rad],
                        "best_infl": [exp_info['infl'][best_rmse_ind]],
                        "nan_exist": nan_exist
                    }

                    # Add metrics data to dictionary dynamically
                    for i, metric in enumerate(metric_names):
                        df_dict[f"{metric}"] = [result_avg[i, best_rmse_ind]]
                        df_dict[f"{metric}_dstd"] = [result_std[i, best_rmse_ind]]
                    
                    df = pd.DataFrame(df_dict)

                    csv_file = os.path.join("save", f"benchmarks_{dataset}.csv")
                    
                    if not trial_processed:
                        csv_file = os.path.join("save", f"benchmarks_{dataset}.csv")

                        directory = os.path.dirname(csv_file)
                        if not os.path.exists(directory):
                            os.makedirs(directory)  # Create the directory if it doesn't exist

                        if os.path.isfile(csv_file):
                            df.to_csv(csv_file, mode='a', header=False, index=False)
                        else:
                            df.to_csv(csv_file, mode='w', index=False)

                    # print("Results saved to:", csv_file)

if __name__ == "__main__":
    main(GRID_SEARCH_INFO)