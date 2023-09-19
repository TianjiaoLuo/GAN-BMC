from typing import Dict, Any

HYPERPARAMETERS: Dict[str, Any] = {
    "training_iterations": 100000,
    "batch_size": 128,
    "lr": .1,
    "in_scale": 0.6,
    "r1_w": 0.2,
    "r2_w": 0.2,
    "gp_w": 0.25,
    "dra_w": 0.1,
    "rlc_af": 1.,
    "rlc_ar": 1.,
    "rlc_w": 0.15,
    "rou1" : 0.001,
    "rou2" : 0.01,
    "beta" : 1
}
