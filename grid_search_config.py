import torch
import numpy as np

GRID_SEARCH_INFO = {}

#################### ks
KS_INFL = {5: [1 + 0.1*i for i in range(21)],
           10: [1 + 0.1*i for i in range(21)],
           15: [1 + 0.05*i for i in range(21)],
           20: [1 + 0.05*i for i in range(21)],
           40: [1 + 0.03*i for i in range(11)],
           60: [1 + 0.02*i for i in range(11)],
           100: [1 + 0.01*i for i in range(11)],
           }

KS_INFL_LETKF = {5: [1 + 0.03*i for i in range(11)],
           10: [1 + 0.03*i for i in range(11)],
           15: [1 + 0.03*i for i in range(11)],
           20: [1 + 0.02*i for i in range(11)],
           40: [1 + 0.02*i for i in range(11)],
           60: [1 + 0.015*i for i in range(11)],
           100: [1 + 0.01*i for i in range(11)],
           }

KS_LOC = {5: [0.001,] + [i for i in range(1,11,1)],
           10: [0.001,] + [i for i in range(2,21,2)],
           15: [0.001,] + [i for i in range(2,21,2)],
           20: [0.001,] + [i for i in range(2,21,2)],
           40: [0.001,] + [i for i in range(3,31,3)],
           60: [0.001,] + [i for i in range(3,31,3)],
           100: [0.001,] + [i for i in range(5,51,5)],
           }

GRID_SEARCH_INFO["ks"] = {
       "sigma_y":[1, 0.7],  
       "methods":["LETKF", "EnKF_Sqrt", "EnKF_PertObs", "iEnKS"],
       "N_list":[5,10,15,20,40,60,100],
       "infl_list":KS_INFL,
       "letkf_infl_list":KS_INFL_LETKF,
       "loc_rad_list":KS_LOC,
       "dt": 0.25,
       "dto": 1,
       "ko": 2000,
       "obs_inds": np.arange(0,128,8),
       "num_trials": 64,
       }


###################### L96
L96_INFL = {5: [1 + 0.02*i for i in range(21)],
           10: [1 + 0.02*i for i in range(21)],
           15: [1 + 0.02*i for i in range(21)],
           20: [1 + 0.01*i for i in range(21)],
           40: [1 + 0.01*i for i in range(21)],
           60: [1 + 0.01*i for i in range(11)],
           100: [1 + 0.01*i for i in range(11)],
           }

L96_INFL_LETKF = {5: [1 + 0.02*i for i in range(21)],
           10: [1 + 0.02*i for i in range(21)],
           15: [1 + 0.02*i for i in range(21)],
           20: [1 + 0.01*i for i in range(21)],
           40: [1 + 0.01*i for i in range(21)],
           60: [1 + 0.01*i for i in range(11)],
           100: [1 + 0.01*i for i in range(11)],
           }

L96_LOC = {5: [0.001,] + [i for i in range(1,11,1)],
           10: [0.001,] + [i for i in range(1,11,1)],
           15: [0.001,] + [i for i in range(1,11,1)],
           20: [0.001,] + [i for i in range(1,11,1)],
           40: [0.001,] + [i for i in range(2,21,2)],
           60: [0.001,] + [i for i in range(2,21,2)],
           100: [0.001,] + [i for i in range(3,31,3)],
           }

GRID_SEARCH_INFO["lorenz96"] = {
       "sigma_y":[1, 0.7], 
       "methods":["LETKF", "EnKF_Sqrt", "EnKF_PertObs", "iEnKS"],
       "N_list":[5,10,15,20,40,60,100],
       "infl_list":L96_INFL,
       "letkf_infl_list":L96_INFL_LETKF,
       "loc_rad_list":L96_LOC,
       "dt": 0.03,
       "dto": 0.15,
       "ko": 1500,
       "obs_inds": np.arange(0,40,4),
       "num_trials": 64,
       }

# ####################### L63
L63_INFL = {5: [1 + 0.05*i for i in range(11)],
           10: [1 + 0.02*i for i in range(11)],
           15: [1 + 0.01*i for i in range(11)],
           20: [1 + 0.01*i for i in range(11)],
           40: [1 + 0.01*i for i in range(11)],
           60: [1 + 0.01*i for i in range(11)],
           100: [1 + 0.01*i for i in range(11)],
           }
GRID_SEARCH_INFO["lorenz63"] = {
       "sigma_y":[1, 0.7], 
       "methods":["EnKF_Sqrt", "EnKF_PertObs", "iEnKS"],
       "N_list":[5,10,15,20,40,60,100],
       "infl_list":L63_INFL,
       "loc_rad_list":[],
       "dt": 0.03,
       "dto": 0.15,
       "ko": 1500,
       "obs_inds": np.array([0]),
       "num_trials": 64,
       }
