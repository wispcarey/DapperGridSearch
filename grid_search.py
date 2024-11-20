import dapper as dpr  # noqa: I001
import dapper.da_methods as da
import re
import pandas as pd
import os
import math

import numpy as np
from joblib import Parallel, delayed

import dapper.mods as modelling
from dapper.tools.localization import nd_Id_localization
from dapper.mods.Lorenz96 import LPs
from dapper.tools.progressbar import disable_progbar

from grid_search_config import GRID_SEARCH_INFO

def main(grid_search_info):
    def _process_trial(trial_ind):
        for xp in xps:
            xp.seed = trial_ind + 1
            
        save_as = xps.launch(HMM, liveplots=False, save_as=f"noname_{trial_ind}")

        averages = dpr.stats.tabulate_avrgs([C.avrgs for C in xps])
        exp_info, _, _ = xps.prep_table()

        rmse_a_key = [key for key in averages.keys() if key.startswith('rmse.a')][0]
        rmv_a_key = [key for key in averages.keys() if key.startswith('rmv.a')][0]  

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
        
        return rmse_mean, rmse_std, rmv_mean, rmv_std

    def _combine_results(results):
        result_record = np.zeros((GRID_SEARCH_INFO[dataset]["num_trials"], 4, K))

        for trial_ind, data in enumerate(results):
            for i in range(4):
                result_record[trial_ind, i, :] = data[i]
        
        nan_val_count = np.sum(~np.isnan(result_record), axis=0)
        safe_nan_val_count = np.where(nan_val_count == 0, 1, nan_val_count)
        result_record[np.isnan(result_record)] = 0
        result_avg = np.sum(result_record, axis=0) / safe_nan_val_count
        result_avg[nan_val_count == 0] = np.nan

        return result_avg

    for dataset, info in grid_search_info.items():
        for sigma_y in info['sigma_y']:
            if dataset == 'ks':
                from dapper.mods.KS import Model, Tplot
                
                KS = Model(dt=GRID_SEARCH_INFO[dataset]["dt"])
                Nx = KS.Nx

                # nRepeat=10
                tseq = modelling.Chronology(GRID_SEARCH_INFO[dataset]["dt"], 
                                            dto=GRID_SEARCH_INFO[dataset]["dto"], 
                                            Ko=GRID_SEARCH_INFO[dataset]["ko"], 
                                            BurnIn=1000, 
                                            Tplot=Tplot)

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
                
                tseq = modelling.Chronology(GRID_SEARCH_INFO[dataset]["dt"], 
                                            dto=GRID_SEARCH_INFO[dataset]["dto"], 
                                            Ko=GRID_SEARCH_INFO[dataset]["ko"], 
                                            Tplot=Tplot,
                                            BurnIn=2 * Tplot)
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
                tseq = modelling.Chronology(GRID_SEARCH_INFO[dataset]["dt"], 
                                            dto=GRID_SEARCH_INFO[dataset]["dto"], 
                                            Ko=GRID_SEARCH_INFO[dataset]["ko"], 
                                            Tplot=Tplot, 
                                            BurnIn=4 * Tplot)
                Nx = len(x0)

                Dyn = {
                    "M": Nx,
                    "model": step,
                    "linear": dstep_dx,
                    "noise": 0,
                }

                X0 = modelling.GaussRV(mu=x0,C=1)

                jj = GRID_SEARCH_INFO[dataset]["obs_inds"]
                Obs = modelling.partial_Id_Obs(Nx, jj)
                Obs["noise"] = sigma_y ** 2  # modelling.GaussRV(C=CovMat(2*eye(Nx)))

                HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
            
            print(f"Start grid search for {dataset} with sigma_y = {sigma_y}.")

            for method_name in info['methods']:
                # method_name = "iEnKS"

                N_list = info['N_list']
                infl_list = info['infl_list']

                if method_name == 'LETKF':
                    loc_rad_list = info['loc_rad_list']
                else:
                    loc_rad_list = []
                for N in N_list:
                    xps = dpr.xpList()

                    for infl in infl_list:
                        if loc_rad_list:
                            for loc_rad in loc_rad_list:
                                xps += da.LETKF(N=N, infl=infl, loc_rad=loc_rad)
                        else:
                            if method_name == 'iEnKS':
                                xps += da.iEnKS("Sqrt", N=N , infl=infl, Lag=2)
                            elif method_name == 'EnKF_PertObs':
                                xps += da.EnKF("PertObs", N=N, infl=infl)
                            elif method_name == 'EnKF_Sqrt':
                                xps += da.EnKF("Sqrt", N=N, infl=infl)
                    
                    K = len(xps)
                    
                    results = Parallel(n_jobs=-1)(delayed(_process_trial)(trial_ind) for trial_ind in range(GRID_SEARCH_INFO[dataset]["num_trials"]))
                    result_avg = _combine_results(results)
                    exp_info, _, _ = xps.prep_table()

                    best_rmse_ind = np.argmin(result_avg[0,:])
                    
                    if loc_rad_list:
                        best_loc_rad = exp_info['loc_rad'][best_rmse_ind]
                    else:
                        best_loc_rad = ""

                    df = pd.DataFrame({
                        "method": [method_name],
                        "N": [N],
                        "sigma_y": [sigma_y],
                        "best_loc_rad": [best_loc_rad],
                        "best_infl": [exp_info['infl'][best_rmse_ind]],
                        "rmse_mean": [result_avg[0, best_rmse_ind]],
                        "rmse_std": [result_avg[1, best_rmse_ind]],
                        "rmv_mean": [result_avg[2, best_rmse_ind]],  
                        "rmv_std": [result_avg[3, best_rmse_ind]]     
                    })

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
    disable_progbar=True
    main(GRID_SEARCH_INFO)