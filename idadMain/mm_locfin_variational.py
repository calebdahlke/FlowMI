import os
import pickle
import argparse
import time
import torch
from torch import nn
# from pyro.infer.util import torch_item
import pyro
import pyro.distributions as dist
from tqdm import trange
import mlflow
import copy
from pyro.nn import AutoRegressiveNN
from neural.baselines import BatchDesignBaseline#, DesignBaseline

from experiment_tools.pyro_tools import auto_seed
from oed.design import OED

from pyro import poutine
from pyro.poutine.util import prune_subsample_sites

# import datetime
# import math
# import subprocess
# from functools import lru_cache
# import time

# from torch.distributions import constraints
# from torch.distributions import transform_to

# import pyro.contrib.gp as gp
# from pyro.contrib.util import rmv

import pyro.distributions.transforms as T


# from pyro.util import torch_isnan, torch_isinf
# def is_bad(a):
#     return torch_isnan(a) or torch_isinf(a)
from simulations import HiddenObjectsVar
from flow_estimator_pyro import MomentMatchMarginalPosterior,SplineFlow, IdentityTransform, RealNVP

def optimise_design(
    posterior_loc,
    posterior_cov,
    flow_theta,
    flow_obs,
    train_flow,
    run_flow,
    experiment_number,
    noise_scale,
    p,
    num_sources,
    device,
    batch_size,
    num_steps,
    lr,
    annealing_scheme=None,
):
    design_init = (
        torch.distributions.Normal(0.0, 0.01)
        if experiment_number == 0
        else torch.distributions.Normal(0.0, 1.0)
    )
    design_net = BatchDesignBaseline(
        T=1, design_dim=(1, p), design_init=design_init
    ).to(device)


    ho_model = HiddenObjectsVar(
        design_net=design_net,
        theta_loc=posterior_loc,
        theta_covmat=posterior_cov,
        flow_theta = flow_theta,
        T=1,
        p=p,
        K=num_sources,
        noise_scale=noise_scale * torch.ones(1, device=device),
    )
    
    if run_flow:
        dim_x = num_sources*p
        dim_y = 1
        if flow_theta == None:
            fX = SplineFlow(dim_x, count_bins=8, bounds=5, device=device).to(device)
            # fX = RealNVP(dim_x, num_blocks=3, dim_hidden=hidden//2,device=device).to(device)
        else:
            fX = copy.deepcopy(flow_theta)
        if flow_obs == None:
            gY = SplineFlow(dim_y, count_bins=8, bounds=5, device=device).to(device)
            # gY = RealNVP(dim_y, num_blocks=3, dim_hidden=hidden//2,device=device).to(device)
        else:
            gY = copy.deepcopy(flow_obs)
    else:
        fX = IdentityTransform()
        gY = IdentityTransform()


    ### Set-up loss ###
    mi_loss_instance = MomentMatchMarginalPosterior(
        model=ho_model.model,
        batch_size=batch_size,
        flow_x=fX,
        flow_y=gY,
        train_flow=train_flow,
        device=device
    )
    
    ### Set-up optimiser ###
    optimizer = torch.optim.Adam
    # Annealed LR. Set gamma=1 if no annealing required
    annealing_freq, patience, factor = annealing_scheme
    scheduler = pyro.optim.ReduceLROnPlateau(
        {
            "optimizer": optimizer,
            "optim_args": {"lr": lr},
            "factor": factor,
            "patience": patience,
            "verbose": False,
        }
    )
    oed = OED(optim=scheduler, loss=mi_loss_instance)
    
    ### Optimise ###
    loss_history = []
    num_steps_range = trange(0, num_steps + 0, desc="Loss: 0.000 ")
    for i in num_steps_range:
        loss = oed.step()
        # Log every 100 losses -> too slow (and unnecessary to log everything)
        if i % 100 == 0:
            num_steps_range.set_description("Loss: {:.3f} ".format(loss))#mi_loss_instance.loss().detach().numpy()[0]
            loss_eval = oed.evaluate_loss()
            # mlflow.log_metric(f"loss_{experiment_number}", loss_eval, step=i)

        # Check if lr should be decreased every 200 steps.
        # patience=5 so annealing occurs at most every 1.2K steps
        if i % annealing_freq == 0:
            scheduler.step(loss_eval)
            # store design paths

    return ho_model, mi_loss_instance


def main_loop(
    run,  # number of rollouts
    mlflow_run_id,
    device,
    T,
    train_flow_every_step,
    run_flow,
    noise_scale,
    num_sources,
    p,
    batch_size,
    num_steps,
    lr,
    annealing_scheme,
):
    pyro.clear_param_store()
    theta_loc = torch.zeros((1, num_sources*p), device=device)
    theta_covmat = torch.eye(num_sources*p, device=device)
    flow_theta = None
    flow_obs = None
    train_flow = True
    prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)

    # sample true param
    true_theta = prior.sample(torch.Size([1]))
    time_per_design = []
    # flow_dist_so_far = []
    mus_so_far = []
    sigmas_so_far = []
    flow_theta_so_far = []
    flow_obs_so_far = []
    posterior_loc_so_far = []
    posterior_cov_so_far = []
    designs_so_far = []
    observations_so_far = []

    # Set posterior equal to the prior
    posterior_loc = theta_loc.reshape(-1)
    posterior_cov = torch.eye(p * num_sources, device=device)

    for t in range(0, T):
        t_start = time.time()
        print(f"Step {t + 1}/{T} of Run {run + 1}")
        pyro.clear_param_store()
        ho_model, mi_loss_instance = optimise_design(
            posterior_loc,
            posterior_cov,
            flow_theta,
            flow_obs,
            train_flow=train_flow,
            run_flow=run_flow,
            experiment_number=t,
            noise_scale=noise_scale,
            p=p,
            num_sources=num_sources,
            device=device,
            batch_size=batch_size,
            num_steps=num_steps,
            lr=lr,
            annealing_scheme=annealing_scheme,
        )
        
        
        ################################ CHECK ON THESE ###################################################
        with torch.no_grad():
            
            if t>0:
                trans_true_theta,_ = flow_theta.forward(true_theta[0])
            else:
                trans_true_theta = true_theta
            design, observation = ho_model.forward(theta=trans_true_theta)
            mux = mi_loss_instance.mu[:p * num_sources].detach()
            muy = mi_loss_instance.mu[p * num_sources:].detach()
            Sigmaxx = mi_loss_instance.Sigma[:p * num_sources,:p * num_sources].detach()
            Sigmaxy = mi_loss_instance.Sigma[:p * num_sources,p * num_sources:].detach()
            Sigmayy = mi_loss_instance.Sigma[p * num_sources:,p * num_sources:].detach()
            obs, _ = mi_loss_instance.gY.forward(observation[0])
            posterior_loc = (mux + torch.matmul(Sigmaxy,torch.linalg.solve(Sigmayy,(obs-muy))).flatten())
            max_posterior = mi_loss_instance.fX.reverse(posterior_loc)
            # print(true_theta)
            # print(posterior_loc)
            # print(mi_loss_instance.fX.reverse(posterior_loc))
            posterior_cov = Sigmaxx-torch.matmul(Sigmaxy,torch.linalg.solve(Sigmayy,Sigmaxy.T))
            flow_theta = mi_loss_instance.fX
            flow_obs = mi_loss_instance.gY
        t_end = time.time()
        run_time = t_end-t_start

        
        mus_so_far.append(mi_loss_instance.mu.detach().clone().numpy())
        sigmas_so_far.append(mi_loss_instance.Sigma.detach().clone().numpy())
        flow_theta_so_far.append(copy.deepcopy(mi_loss_instance.fX))
        flow_obs_so_far.append(copy.deepcopy(mi_loss_instance.gY))

        posterior_loc_so_far.append(posterior_loc.numpy())
        posterior_cov_so_far.append(posterior_cov.numpy())
        time_per_design.append(run_time)
        designs_so_far.append(design[0])
        observations_so_far.append(observation[0])
        if not train_flow_every_step:
            train_flow = False

    print(f"Fitted posterior: mean = {posterior_loc}, cov = {posterior_cov}")
    print("True theta = ", true_theta.reshape(-1))

    data_dict = {}
    for i, xi in enumerate(designs_so_far):
        data_dict[f"xi{i + 1}"] = xi.cpu()
    for i, y in enumerate(observations_so_far):
        data_dict[f"y{i + 1}"] = y.cpu()
    data_dict["theta"] = true_theta.reshape((num_sources, p)).cpu()
    
    extra_data = {}
    extra_data["mu"] = mus_so_far
    extra_data["sigmas"] = sigmas_so_far
    extra_data["flow_theta"] = flow_theta_so_far
    extra_data["flow_obs"] = flow_obs_so_far
    extra_data["posterior_loc"] = posterior_loc_so_far
    extra_data["posterior_cov"] = posterior_cov_so_far
    extra_data["design_time"] = time_per_design
    extra_data["designs"] = designs_so_far
    extra_data["observations"] = observations_so_far

    return data_dict, extra_data


def main(
    seed,
    mlflow_experiment_name,
    num_histories,
    device,
    T,
    train_flow_every_step,
    run_flow,
    p,
    num_sources,
    noise_scale,
    batch_size,
    num_steps,
    lr,
):
    pyro.clear_param_store()
    seed = auto_seed(seed)
    pyro.set_rng_seed(seed)

    mlflow.set_experiment(mlflow_experiment_name)
    # Log everything
    mlflow.log_param("seed", seed)
    mlflow.log_param("p", p)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("lr", lr)
    mlflow.log_param("num_histories", num_histories)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("noise_scale", noise_scale)
    mlflow.log_param("num_sources", num_sources)
    annealing_scheme = [100, 5, 0.8]
    mlflow.log_param("annealing_scheme", str(annealing_scheme))

    meta = {
        "model": "location_finding",
        "p": p,
        "K": num_sources,
        "noise_scale": noise_scale,
        "num_histories": num_histories,
    }
    extra_meta = {
        "model": "location_finding",
        "p": p,
        "K": num_sources,
        "noise_scale": noise_scale,
        "num_histories": num_histories,
        "train_flow_every_step": train_flow_every_step,
        "run_flow": run_flow,
        "seed": seed
    }

    results_vi = {"loop": [], "seed": seed, "meta": meta}
    extra_vi = {"loop":[],"meta":extra_meta}
    for i in range(num_histories):
        results, extra_data = main_loop(
            run=i,
            mlflow_run_id=mlflow.active_run().info.run_id,
            device=device,
            T=T,
            train_flow_every_step=train_flow_every_step,
            run_flow=run_flow,
            noise_scale=noise_scale,
            num_sources=num_sources,
            p=p,
            batch_size=batch_size,
            num_steps=num_steps,
            lr=lr,
            annealing_scheme=annealing_scheme,
        )
        results_vi["loop"].append(results)
        extra_vi["loop"].append(extra_data)

    # Log the results dict as an artifact
    if not os.path.exists("./mlflow_outputs"):
        os.makedirs("./mlflow_outputs")
    with open("./mlflow_outputs/results_locfin_mm_vi.pickle", "wb") as f:
        pickle.dump(results_vi, f)
    mlflow.log_artifact("mlflow_outputs/results_locfin_mm_vi.pickle")
    print("Done.")
    ml_info = mlflow.active_run().info
    path_to_artifact = "mlruns/{}/{}/artifacts/results_locfin_mm_vi.pickle".format(
        ml_info.experiment_id, ml_info.run_id
    )
    print("Path to artifact - use this when evaluating:\n", path_to_artifact)
    
    t = time.localtime()
    run_id = time.strftime("%Y%m%d%H%M%S", t)
    path_to_artifact = "./experiment_outputs/loc_fin/{}".format(run_id)
    if not os.path.exists("./experiment_outputs/loc_fin"):
        os.makedirs("./experiment_outputs/loc_fin")
    with open(path_to_artifact, "wb") as f:
        pickle.dump(extra_vi, f)
    # --------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VI baseline: Location finding with MM M+P"
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--physical-dim", default=2, type=int)
    parser.add_argument(
        "--num-histories", help="Number of histories/rollouts", default=2, type=int#128
    )
    parser.add_argument("--num-experiments", default=10, type=int)  # 10
    parser.add_argument("--batch-size", default=256, type=int)#512,1024
    parser.add_argument("--device", default="cpu", type=str)#"cuda"
    parser.add_argument(
        "--mlflow-experiment-name", default="locfin_mm_variational", type=str
    )
    parser.add_argument("--lr", default=0.005, type=float)#0.005
    parser.add_argument("--num-steps", default=5000, type=int)#
    parser.add_argument("--train-flow-every-step", default=False, type=bool)
    parser.add_argument("--run-flow", default=True, type=bool)
    
    args = parser.parse_args()

    main(
        seed=args.seed,
        mlflow_experiment_name=args.mlflow_experiment_name,
        num_histories=args.num_histories,
        device=args.device,
        T=args.num_experiments,
        train_flow_every_step= args.train_flow_every_step,
        run_flow = args.run_flow,
        p=args.physical_dim,
        num_sources=2,
        noise_scale=0.5,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
    )

