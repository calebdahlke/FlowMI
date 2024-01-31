import os
import pickle
import argparse
from collections import OrderedDict

import torch
from torch import nn
import pyro
import pyro.distributions as dist
from tqdm import trange
import mlflow

import time
import torchsde    
from epidemic_simulate_data import SIR_SDE

from neural.modules import Mlp
from neural.critics import CriticBA
from neural.baselines import BatchDesignBaseline
from neural.aggregators import ConcatImplicitDAD

from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from estimators.bb_mi import BarberAgakov

from epidemic import Epidemic
from epidemic_simulate_data import solve_sir_sdes
import copy

from simulations import solve_sir_sdes_var, EpidemicVar
# from mm_locfin_variational import IdentityTransform, RealNVP, MomentMatchMarginalPosterior, SplineFlow
from flow_estimator_pyro import MomentMatchMarginalPosterior, SplineFlow, IdentityTransform, RealNVP


def optimise_design(
    simdata,
    previous_design,
    posterior_loc,
    posterior_cov,
    flow_theta,
    flow_obs,
    train_flow,
    run_flow,
    experiment_number,
    device,
    batch_size,
    num_steps,
    lr,
    lr_critic,
    annealing_scheme=None,
):

    design_init = torch.distributions.Uniform(-3.0, 3.0)
    design_net = BatchDesignBaseline(T=1, design_dim=(1, 1), design_init=design_init).to(
        device
    )

    epidemic = EpidemicVar(#Epidemic2(
        design_net=design_net,
        T=1,
        design_transform="iid",
        simdata=simdata,
        lower_bound=previous_design.to(device),
        upper_bound=torch.tensor(100.0 - 1e-2, device=device),
    )

    ### Set up model networks ###
    n = 1  # output dim/number of samples per design
    design_dim = (n, 1) #design_dim = 1  # design is t (time)
    latent_dim = 2  #
    observation_dim = n

    if run_flow:
        dim_x = latent_dim
        dim_y = observation_dim ### Try Quadratic and test bin/bound layers try 12
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
        model=epidemic.model,
        batch_size=batch_size,
        flow_x=fX,
        flow_y=gY,
        train_flow=train_flow,
        device=device
    )
    

    ### Set-up optimiser ###
    optimizer = torch.optim.Adam
    
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
            num_steps_range.set_description("Loss: {:.3f} ".format(loss))
            loss_eval = oed.evaluate_loss()
        if i % annealing_freq == 0:
            scheduler.step(loss_eval)

    return epidemic, mi_loss_instance

def main_loop(
    run,  # number of rollouts
    mlflow_run_id,
    device,
    T,
    train_flow_every_step,
    run_flow,
    batch_size,
    num_steps,
    lr,
    lr_critic,
    annealing_scheme=None,
    true_theta=None,
):
    pyro.clear_param_store()
    SIMDATA = torch.load("data/sir_sde_data.pt", map_location=device)
    latent_dim = 2
    theta_loc = torch.tensor([0.5, 0.1], device=device).log()
    theta_covmat = torch.eye(2, device=device) * 0.5 ** 2
    flow_theta = None#IdentityTransform()
    flow_obs = None#IdentityTransform()
    train_flow = True
    prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)

    if true_theta is None:
        # true_theta = prior.sample(torch.Size([1])).exp()
        true_theta = torch.tensor([[.33,.11]])#Middle R0#torch.tensor([[.7,.075]])#HighR0 #torch.tensor([[.15,.15]])#Low R0#
    time_per_design = []
    mus_so_far = []
    sigmas_so_far = []
    flow_theta_so_far = []
    flow_obs_so_far = []
    posterior_loc_so_far = []
    posterior_cov_so_far = []
    design_trans_so_far = []
    designs_so_far = []
    observations_so_far = []

    # Set posterior equal to the prior
    posterior_loc = theta_loc
    posterior_cov = theta_covmat
    # posterior_scale = torch.sqrt(theta_covmat.diag())
    
    previous_design = torch.tensor(0.0, device=device)  # no previous design
    for t in range(0, T):
        t_start = time.time()
        print(f"Step {t + 1}/{T} of Run {run + 1}")
        pyro.clear_param_store()
        if t > 0:
            # update previous design so the lower bound gets updated!
            # e.forward method of Epidemic designs/obs stuff in lists
            previous_design = design_transformed[0].reshape(-1)
            ## pre-simulate data using the posterior as the prior!
            epidemic._remove_data()
            del SIMDATA
            SIMDATA = solve_sir_sdes_var(#solve_sir_sdes2(
                num_samples=5000,#5000
                device=device,
                grid=10000,#10000
                save=False,
                savegrad=False,
                theta_loc=posterior_loc,#.log().reshape(-1)
                theta_covmat=posterior_cov,
                flows_theta=flow_theta
            )
            SIMDATA = {
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in SIMDATA.items()
            }
  
        epidemic, mi_loss_instance = optimise_design(
            simdata=SIMDATA,
            previous_design=previous_design,
            posterior_loc=posterior_loc,
            posterior_cov=posterior_cov,
            flow_theta = flow_theta,
            flow_obs = flow_obs,
            train_flow=train_flow,
            run_flow=run_flow,
            experiment_number=t,
            device=device,
            batch_size=batch_size,
            num_steps=num_steps,
            lr=lr,
            lr_critic=lr_critic,
            annealing_scheme=annealing_scheme,
        )   
        ################################ CHECK ON THESE ###################################################
        with torch.no_grad():
            if t>0:
                trans_true_theta,_ = flow_theta.forward(torch.log(true_theta))
            else:
                trans_true_theta = torch.log(true_theta)
            theta, design_untransformed, observation = epidemic.forward(epidemic.theta_to_index(theta=torch.exp(trans_true_theta)))#true_theta
            design_transformed = epidemic.transform_designs(design_untransformed[0])
            mux = mi_loss_instance.mu[:latent_dim].detach()
            muy = mi_loss_instance.mu[latent_dim:].detach()
            Sigmaxx = mi_loss_instance.Sigma[:latent_dim,:latent_dim].detach()
            Sigmaxy = mi_loss_instance.Sigma[:latent_dim,latent_dim:].detach()
            Sigmayy = mi_loss_instance.Sigma[latent_dim:,latent_dim:].detach()
            obs, _ = mi_loss_instance.gY.forward(observation[0])
            # obs = observation[0]
            posterior_loc = (mux + torch.matmul(Sigmaxy,torch.linalg.solve(Sigmayy,(obs-muy))).flatten())
            max_posterior = mi_loss_instance.fX.reverse(posterior_loc).exp()
            # print(true_theta)#flow_theta.reverse
            # print(posterior_loc)#
            # print(mi_loss_instance.fX.reverse(posterior_loc).exp())
            posterior_cov = Sigmaxx-torch.matmul(Sigmaxy,torch.linalg.solve(Sigmayy,Sigmaxy.T))
            flow_theta = mi_loss_instance.fX
            flow_obs = mi_loss_instance.gY
        ###################################################################################################
        t_end = time.time()
        run_time = t_end-t_start
        
        mus_so_far.append(mi_loss_instance.mu.detach().clone().numpy())
        sigmas_so_far.append(mi_loss_instance.Sigma.detach().clone().numpy())
        flow_theta_so_far.append(copy.deepcopy(mi_loss_instance.fX))
        flow_obs_so_far.append(copy.deepcopy(mi_loss_instance.gY))
        posterior_loc_so_far.append(posterior_loc.numpy())
        posterior_cov_so_far.append(posterior_cov.numpy())
        time_per_design.append(run_time)
        design_trans_so_far.append(design_transformed)
        designs_so_far.append(design_untransformed[0])
        observations_so_far.append(observation[0])
        if not train_flow_every_step:
            train_flow = False

    print(f"Final posterior: mean = {posterior_loc}, sd = {posterior_cov}")
    print(f"True theta = {true_theta}")

    data_dict = {}
    for i, xi in enumerate(designs_so_far):
        data_dict[f"xi{i + 1}"] = xi.cpu()
    for i, y in enumerate(observations_so_far):
        data_dict[f"y{i + 1}"] = y.cpu()
    data_dict["theta"] = true_theta.cpu()
    
    extra_data = {}
    extra_data["mu"] = mus_so_far
    extra_data["sigmas"] = sigmas_so_far
    extra_data["flow_theta"] = flow_theta_so_far
    extra_data["flow_obs"] = flow_obs_so_far
    extra_data["posterior_loc"] = posterior_loc_so_far
    extra_data["posterior_cov"] = posterior_cov_so_far
    extra_data["design_time"] = time_per_design
    extra_data["designs_trans"] = design_trans_so_far
    extra_data["designs_untrans"] = designs_so_far
    extra_data["observations"] = observations_so_far

    return data_dict, extra_data


def main(
    seed,
    mlflow_experiment_name,
    num_loop,
    device,
    T,
    train_flow_every_step,
    run_flow,
    batch_size,
    num_steps,
    lr,
    lr_critic,
):
    pyro.clear_param_store()
    seed = auto_seed(seed)
    pyro.set_rng_seed(seed)

    mlflow.set_experiment(mlflow_experiment_name)
    # Log everything
    mlflow.log_param("seed", seed)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("lr", lr)
    mlflow.log_param("lr_critic", lr_critic)
    mlflow.log_param("num_loop", num_loop)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("batch_size", batch_size)
    annealing_scheme = [500, 5, 0.96]
    mlflow.log_param("annealing_scheme", str(annealing_scheme))

    if not os.path.exists("./mlflow_outputs"):
        os.makedirs("./mlflow_outputs")
    extra_meta = {
        "model": "SIR_epidemic",
        "train_flow_every_step": train_flow_every_step,
        "run_flow": run_flow,
        "seed": seed
    }
    results_vi = {"loop": [], "seed": seed}
    extra_vi = {"loop":[],"meta":extra_meta}
    for i in range(num_loop):
        results, extra_data = main_loop(
            run=i,
            mlflow_run_id=mlflow.active_run().info.run_id,
            device=device,
            T=T,
            train_flow_every_step=train_flow_every_step,
            run_flow=run_flow,
            batch_size=batch_size,
            num_steps=num_steps,
            lr=lr / (i + 1),
            lr_critic=lr_critic / (i + 1),
            annealing_scheme=annealing_scheme,
        )
        results_vi["loop"].append(results)
        extra_vi["loop"].append(extra_data)

    # Log the results dict as an artifact
    with open("./mlflow_outputs/results_sir_vi.pickle", "wb") as f:
        pickle.dump(results_vi, f)
    mlflow.log_artifact("mlflow_outputs/results_sir_vi.pickle")
    print("Done.")

    t = time.localtime()
    run_id = time.strftime("%Y%m%d%H%M%S", t)
    path_to_artifact = "./experiment_outputs/SIR/{}".format(run_id)
    if not os.path.exists("./experiment_outputs/SIR"):
        os.makedirs("./experiment_outputs/SIR")
    with open(path_to_artifact, "wb") as f:
        pickle.dump(extra_vi, f)
    # --------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VI baseline: SIR model")
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-loop", default=1, type=int)#32
    parser.add_argument("--batch-size", default=512, type=int)  #512 == T
    parser.add_argument("--num-experiments", default=5, type=int)  # == T
    parser.add_argument("--device", default="cpu", type=str)#"cuda"
    parser.add_argument(
        "--mlflow-experiment-name", default="epidemic_variational", type=str
    )
    parser.add_argument("--lr-design", default=0.01, type=float)
    parser.add_argument("--lr-critic", default=0.001, type=float)
    parser.add_argument("--num-steps", default=5000, type=int)#5000

    args = parser.parse_args()

    main(
        seed=args.seed,
        num_loop=args.num_loop,
        device=args.device,
        batch_size=args.batch_size,
        T=args.num_experiments,
        lr=args.lr_design,
        lr_critic=args.lr_critic,
        num_steps=args.num_steps,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )

#######################################################
# with torch.no_grad():
#     import numpy as np
#     import scipy
#     from matplotlib.pyplot import colorbar, pcolor, show, scatter
#     x = np.linspace(0.01,1,100)
#     y = np.linspace(0.01,.2,100)
#     X, Y = np.meshgrid(x, y)
    
#     fX, logJac = mi_loss_instance.fX.forward(torch.from_numpy(np.log(np.vstack((X.flatten(),Y.flatten())).T)).float())#.numpy()
#     # gY = np.exp(mi_loss_instance.gY.reverse(torch.from_numpy(np.log(y))).numpy())
#     # xx, yy = np.meshgrid(fX[:,0], fX[:,1])
#     # points = np.stack((xx, yy), axis=-1)
#     points = fX.reshape((100,100,2))
#     Z = scipy.stats.multivariate_normal.pdf(points, posterior_loc, posterior_cov)*torch.exp(logJac.reshape((100,100))).numpy()
#     pcolor(X, Y, Z)
#     scatter(true_theta.numpy()[0][0],true_theta.numpy()[0][1], color='red', marker='x')
#     scatter(mi_loss_instance.fX.reverse(posterior_loc).exp().numpy()[0],mi_loss_instance.fX.reverse(posterior_loc).exp().numpy()[1], color='green', marker='x')
#     colorbar()
#     show()
    
# with torch.no_grad():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     XY = np.random.multivariate_normal(posterior_loc,posterior_cov,5000)
#     fX = mi_loss_instance.fX.reverse(torch.from_numpy((XY)).float()).exp()
#     plt.scatter(fX[:,0], fX[:,1], s=10, c='black', alpha=0.5)
#     plt.scatter(true_theta.numpy()[0][0],true_theta.numpy()[0][1], color='red', marker='x')
#     plt.scatter(mi_loss_instance.fX.reverse(posterior_loc).exp().numpy()[0],mi_loss_instance.fX.reverse(posterior_loc).exp().numpy()[1], color='green', marker='x')
#     # Add labels and title
#     plt.xlabel('X0-axis label')
#     plt.ylabel('X1-axis label')
#     plt.xlim(0,1)
#     plt.ylim(0, .2)
#     plt.title('Scatter Plot of X and Y')

#     # Show the plot
#     plt.show()   
    

# with torch.no_grad():
#     import numpy as np
#     import scipy
#     from matplotlib.pyplot import colorbar, pcolor, show, scatter
#     x = np.linspace(0.01,1,100)
#     y = np.linspace(0.01,.2,100)
#     X, Y = np.meshgrid(x, y)
    
#     fX, logJac = mi_loss_instance.fX.forward(torch.from_numpy(np.log(np.vstack((X.flatten(),Y.flatten())).T)).float())#.numpy()
#     # gY = np.exp(mi_loss_instance.gY.reverse(torch.from_numpy(np.log(y))).numpy())
#     # xx, yy = np.meshgrid(fX[:,0], fX[:,1])
#     # points = np.stack((xx, yy), axis=-1)
#     points = fX.reshape((100,100,2))
#     Z = scipy.stats.multivariate_normal.pdf(points, mux, Sigmaxx)*torch.exp(logJac.reshape((100,100))).numpy()
#     pcolor(X, Y, Z)
#     scatter(true_theta.numpy()[0][0],true_theta.numpy()[0][1], color='red', marker='x')
#     scatter(mi_loss_instance.fX.reverse(mux).exp().numpy()[0],mi_loss_instance.fX.reverse(mux).exp().numpy()[1], color='green', marker='x')
#     colorbar()
#     show()
    
# with torch.no_grad():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     XY = np.random.multivariate_normal(mux,Sigmaxx,5000)
#     fX = mi_loss_instance.fX.reverse(torch.from_numpy((XY)).float()).exp()
#     plt.scatter(fX[:,0], fX[:,1], s=10, c='black', alpha=0.5)
#     plt.scatter(true_theta.numpy()[0][0],true_theta.numpy()[0][1], color='red', marker='x')
#     plt.scatter(mi_loss_instance.fX.reverse(mux).exp().numpy()[0],mi_loss_instance.fX.reverse(mux).exp().numpy()[1], color='green', marker='x')
#     # Add labels and title
#     plt.xlabel('X0-axis label')
#     plt.ylabel('X1-axis label')
#     plt.xlim(0,1)
#     plt.ylim(0, .2)
#     plt.title('Scatter Plot of X and Y')

#     # Show the plot
#     plt.show()