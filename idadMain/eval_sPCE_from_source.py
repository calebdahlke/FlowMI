import os
import math
import argparse
import pickle
from collections import defaultdict

import pandas as pd

import torch
import pyro

import mlflow

from experiment_tools.pyro_tools import auto_seed
from experiment_tools.output_utils import get_mlflow_meta
from estimators.mi import PriorContrastiveEstimation, NestedMonteCarloEstimation
from location_finding import HiddenObjects
from pharmacokinetic import Pharmacokinetic
from neural.modules import LazyFn


def make_data_source(fname, T, device="cuda", n=1):
    with open(fname, "rb") as f:
        data = pickle.load(f)

    sample = defaultdict(list)
    latent_name = "log_theta" if "pharmaco" in fname else "theta"
    for history in data["loop"]:
        sample[latent_name].append(history["theta"])

        for i in range(T):
            sample[f"y{i+1}"].append(history[f"y{i+1}"])
            sample[f"xi{i+1}"].append(history[f"xi{i+1}"])

        if len(sample[latent_name]) == n:
            record = {k: torch.cat(v).to(device) for k, v in sample.items()}
            yield record
            sample = defaultdict(list)


def eval_from_source(
    path_to_artifact, num_experiments_to_perform, num_inner_samples, seed, device,
):
    n = 1
    seed = auto_seed(seed)
    with open(path_to_artifact, "rb") as f:
        data = pickle.load(f)
    meta = data["meta"]

    if meta["model"] == "location_finding" or "locfin" in path_to_artifact:
        K, p = meta["K"], meta["p"]
        design_dim = (1, p)
        model_instance = HiddenObjects(
            design_net=LazyFn(
                lambda *args: None, prototype=torch.ones(design_dim, device=device),
            ),
            theta_loc=torch.zeros((K, p), device=device),
            theta_covmat=torch.eye(p, device=device),
            noise_scale=meta["noise_scale"] * torch.ones(1, device=device),
            p=p,
            K=K,
            T=num_experiments_to_perform[0],
        )

    elif meta["model"] == "pharmacokinetic" or "pharmaco" in path_to_artifact:
        design_dim = 1
        model_instance = Pharmacokinetic(
            design_net=LazyFn(
                lambda *args: None, prototype=torch.ones(design_dim, device=device),
            ),
            T=num_experiments_to_perform[0],
            theta_loc=torch.tensor([1, 0.1, 20], device=device).log(),
            theta_covmat=torch.eye(3, device=device) * 0.05,
        )
    else:
        raise ValueError("Unknown model.")

    EIGs_mean = pd.DataFrame(columns=["lower", "upper"])
    EIGs_se = pd.DataFrame(columns=["lower", "upper"])

    for t_exp in num_experiments_to_perform:
        data_source = make_data_source(
            fname=path_to_artifact, T=t_exp, device=device, n=n
        )

        model_instance.T = t_exp

        loss_upper = NestedMonteCarloEstimation(
            model_instance.model, n, num_inner_samples, data_source=data_source
        )
        auto_seed(seed)
        EIG_proxy_upper = torch.tensor(
            [-loss_upper.loss() for _ in range(meta["num_histories"] // n)]
        )

        data_source = make_data_source(
            fname=path_to_artifact, T=t_exp, device=device, n=n
        )
        loss_lower = PriorContrastiveEstimation(
            model_instance.model, n, num_inner_samples, data_source=data_source
        )
        auto_seed(seed)
        EIG_proxy_lower = torch.tensor(
            [-loss_lower.loss() for _ in range(meta["num_histories"] // n)]
        )

        EIGs_mean.loc[t_exp, "lower"] = EIG_proxy_lower.mean().item()
        EIGs_mean.loc[t_exp, "upper"] = EIG_proxy_upper.mean().item()
        EIGs_se.loc[t_exp, "lower"] = EIG_proxy_lower.std().item() / math.sqrt(
            meta["num_histories"] // n
        )
        EIGs_se.loc[t_exp, "upper"] = EIG_proxy_upper.std().item() / math.sqrt(
            meta["num_histories"] // n
        )

    print("EIG mean\n", EIGs_mean)
    print("EIG se\n", EIGs_se)

    return EIGs_mean, EIGs_se


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation") 
    parser.add_argument("--path-to-artifact", default="mlruns/533010734123778362/b5ca5c8f1d2749ac9404d6ba32f4f4ae/artifacts/results_locfin_vi.pickle", type=str)#
    parser.add_argument("--device", default="cpu", type=str)#"cuda""cpu"
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-inner-samples", default=int(5e5), type=int)
    parser.add_argument("--num-experiments-to-perform", nargs="+", default=[10])#
    args = parser.parse_args()
    args.num_experiments_to_perform = [
        int(x) if x else x for x in args.num_experiments_to_perform
    ]
    ## Gaus
    # +d, +hxY (3.462949 +- 0.437588) "mlruns/145551772296899500/1315d87cf4654d9eb82982c9a89108ef/artifacts/results_locfin_mm_vi.pickle"
    # +d, -hXY (5.491775 +- 0.508812) "mlruns/145551772296899500/454d2c7369594af994d1570666e738c8/artifacts/results_locfin_mm_vi.pickle"
    #          (4.394956 +- 0.703709) "mlruns/145551772296899500/61ac7f199b0b4ee9bb9ab10ca73730a1/artifacts/results_locfin_mm_vi.pickle"
    # -d, +hXY (4.326183 +- 0.617077) "mlruns/145551772296899500/83af62b67ae1419fb8f0fa4e36ffbd8f/artifacts/results_locfin_mm_vi.pickle"
    #          (4.736244 +- 0.820297) "mlruns/145551772296899500/6704efea40ff4c7caaca57320854e00f/artifacts/results_locfin_mm_vi.pickle"
    
    ### HPC 
    # First Flow: (9142530) "mlruns/145551772296899500/8b48bee030f74309a6a5987ea8c596c7/artifacts/results_locfin_mm_vi.pickle"
    # All Flow:   (9142100) "mlruns/145551772296899500/87b14e6370b44452aea5d8209c2b3168/artifacts/results_locfin_mm_vi.pickle"
    # Gauss:      (9142569) "mlruns/145551772296899500/27a8b20b22b14097876e1563670b49c8/artifacts/results_locfin_mm_vi.pickle"

    ## 256 Sample
    #Flow loc fin (lr = .0005 : good):   "mlruns/145551772296899500/73788b2ec75045158610d237fd80dcfc/artifacts/results_locfin_mm_vi.pickle"
    # (lr = .0005,lr_f =.003)  bad       "mlruns/145551772296899500/924c21cf68cc4d11afe597a5611f3e58/artifacts/results_locfin_mm_vi.pickle"
    # bins=128 bound=10 layer=1 [64]
    # (lr = .0005,lr_f =.003)  bad       "mlruns/145551772296899500/924c21cf68cc4d11afe597a5611f3e58/artifacts/results_locfin_mm_vi.pickle"
    # bins=128 bound=10 layer=1 [64]
    # (lr = .0005,lr_f =.003)            "mlruns/145551772296899500/13ba457b202440f19ad0646da92968ed/artifacts/results_locfin_mm_vi.pickle"
    # bins=256 bound=10 layer=1 [64]
    ##512 Sample
    # (lr = .0005,lr_f =.003)            "mlruns/145551772296899500/56f0bca5d958438aa3056e5d0df7343b/artifacts/results_locfin_mm_vi.pickle"
    # bins=128 bound=10 layer=1 [64]
    # (lr = .0005,lr_f =.003)            "mlruns/145551772296899500/c7b9ab7ba4e548df9a0f9a127a9eccd5/artifacts/results_locfin_mm_vi.pickle"
    # bins=128 bound=10 layer=1 [32,64]
    # (lr = .0005,lr_f =.003)  bad       "mlruns/145551772296899500/c6fff83602544e7e86149ca5f5937180/artifacts/results_locfin_mm_vi.pickle"
    # bins=256 bound=10 layer=1 [64]
    
    
    ## 1024 Sample
    # (lr = .0005, lr_f = 0.003: good):  "mlruns/145551772296899500/ed753f59aad14d509da1fb321f93599d/artifacts/results_locfin_mm_vi.pickle"
    # (lr = .001, lr_f = 0.003: bad):    "mlruns/145551772296899500/19ca85401b97416da7af1a3a0f8a2145/artifacts/results_locfin_mm_vi.pickle"
    
    ## 256 Sample
    #Gaus loc fin (with paper params):   "mlruns/145551772296899500/4e669cbb5f824a08b1a6771eae0fceea/artifacts/results_locfin_mm_vi.pickle"
    #                                    "mlruns/145551772296899500/608d64c3c59a4ace896374505992279f/artifacts/results_locfin_mm_vi.pickle"
    #             (lr = .003 : bad):     "mlruns/145551772296899500/763302cce9de4476a85661738449875b/artifacts/results_locfin_mm_vi.pickle"
    #             (lr = .0005 : good):   "mlruns/145551772296899500/86d43764d5ce445892f0f22f58d60e9e/artifacts/results_locfin_mm_vi.pickle"
    #                                    "mlruns/145551772296899500/18c58556671948c68f813aed6543dc2f/artifacts/results_locfin_mm_vi.pickle"
    #             (lr = .0001 : good):   "mlruns/145551772296899500/528333aec7644547ba85e795e634cc72/artifacts/results_locfin_mm_vi.pickle"
    #             (lr = 0.00005 : good): "mlruns/145551772296899500/b69ea1513d1a49d8b9534406b5143e40/artifacts/results_locfin_mm_vi.pickle"
    # .0001 and .00005 likely too small
    ## 512 sample
    #              (lr = .0005)          "mlruns/145551772296899500/32ca6aa4d2984417aad358fa91067bf2/artifacts/results_locfin_mm_vi.pickle"  
    ## 1024 Sample
    #             (lr = .005 : bad):     "mlruns/145551772296899500/5243148b447949e78b9d98d591d9aea3/artifacts/results_locfin_mm_vi.pickle"
    #             (lr = .001 : bad):     "mlruns/145551772296899500/508f1aea115742598e5d4841b551480c/artifacts/results_locfin_mm_vi.pickle"
    #                                    "mlruns/145551772296899500/af6e74c39c4e44a8b9c0eab9c33c71a6/artifacts/results_locfin_mm_vi.pickle"
    #             (lr = .0005 : meh):    "mlruns/145551772296899500/07a25f677f584322bf2537581a14ca9a/artifacts/results_locfin_mm_vi.pickle"
    #                                    "mlruns/145551772296899500/b3659d71a69145b2b5d2801b1d7ed7f9/artifacts/results_locfin_mm_vi.pickle"
    #             (lr = .0001 : meh):    "mlruns/145551772296899500/f4e314fbb594410eb4fda2c31e6dfd1e/artifacts/results_locfin_mm_vi.pickle"
    eval_from_source(
        path_to_artifact=args.path_to_artifact,
        num_experiments_to_perform=args.num_experiments_to_perform,
        num_inner_samples=args.num_inner_samples,
        seed=args.seed,
        device=args.device,
    )

################################################

# 1 runs, 10 decision, flow, 2000 steps (20 min for 10 decision)
# mlruns/4/1ad525c05d044ec0a304b0aabf5b4022/artifacts/results_locfin_mm_vi.pickle
#
# 1 runs, 10 decision, flow, 5000 steps
# mlruns/4/8ad323fb560e45128cbc984667da61b6/artifacts/results_locfin_mm_vi.pickle
#
# 1 runs, 10 decision, MM Gauss, 2000 steps (1 min for 10 decision)
# mlruns/4/60a2c93390904acab6368923e62bff07/artifacts/results_locfin_mm_vi.pickle
# mlruns/4/daafba4cb01c40c9aaeffffb89da37f4/artifacts/results_locfin_mm_vi.pickle (3.06)
#
# 1 runs, 10 decision, MM Gauss, 5000 steps (2.2 min for 10 decision)
# mlruns/4/1e1b0d723ff7498aa4f6a60b77fff3e3/artifacts/results_locfin_mm_vi.pickle
#

# 3 runs, 10 decision, flow, 5000 samples (50 min for 10 decision)
# mlruns/4/f22a441b86ad4b379d7666a8e3cf5f99/artifacts/results_locfin_mm_vi.pickle
#
# 3 runs, 10 decision, MM Gauss, 5000 steps (1.5 min for 10 decision)
# mlruns/4/5d06eff273e44e83a5df802bb5385d25/artifacts/results_locfin_mm_vi.pickle
#