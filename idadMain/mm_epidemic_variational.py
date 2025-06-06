import os
import pickle
import argparse

import torch
import pyro
from tqdm import trange
import mlflow

import time
from neural.baselines import BatchDesignBaseline

from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
import copy

from simulations import solve_sir_sdes_var, EpidemicVar
from flow_estimator_pyro import MomentMatchMarginalPosterior, SplineFlow, IdentityTransform, InitFlowToIdentity
from joblib import Parallel, delayed
from pyro.infer.util import torch_item

def optimise_design(
    simdata,
    previous_design,
    flow_theta,
    flow_obs,
    train_flow,
    run_flow,
    device,
    batch_size,
    num_steps,
    lr_design,
    lr_flow,
    annealing_scheme=None,
):

    # design_init = torch.distributions.Uniform(-3.0, 3.0)
    design_init = torch.distributions.Uniform(-3, -2.9)
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

    if run_flow:
        dim_x = 2
        dim_y = 1 ### Try Quadratic and test bin/bound layers try 12
        if flow_theta == None:
            # fX = SplineFlow(dim_x, count_bins=16, bounds=3, device=device).to(device)
            # fX = InitFlowToIdentity(dim_x, fX, bounds = 3, device=device)
            # fX = RealNVP(dim_x, num_blocks=3, dim_hidden=hidden//2,device=device).to(device)
            fX = SplineFlow(dim_x,n_flows=1,hidden_dims=[8,8], count_bins=128, bounds=4,order = 'linear', device=device)
        else:
            fX = copy.deepcopy(flow_theta)
        if flow_obs == None:
            # gY = SplineFlow(dim_y, count_bins=256, bounds=3, device=device).to(device)
            # gY = InitFlowToIdentity(dim_y, gY, bounds = 3, device=device)
            # gY = RealNVP(dim_y, num_blocks=3, dim_hidden=hidden//2,device=device).to(device)
            gY = SplineFlow(dim_y,count_bins=128, bounds=5,order = 'linear', device=device)
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
        prev_flow_theta = flow_theta,
        train_flow=train_flow,
        device=device
    )

    ### Set-up optimiser ###
    optimizer_design = torch.optim.Adam(epidemic.design_net.designs)
    annealing_freq, factor = annealing_scheme
    # scheduler_design = pyro.optim.ExponentialLR(
    #     {
    #         "optimizer": optimizer_design,
    #         "optim_args": {"lr": lr_design},
    #         "gamma" : factor,
    #         "verbose": False,
    #     }
    # )
    scheduler_design = pyro.optim.ReduceLROnPlateau(
        {
            "optimizer": optimizer_design,
            "optim_args": {"lr": lr_design},
            "factor": .8,
            "patience": 2,
            "verbose": False,
        }
    )
    
    if run_flow and train_flow:
        optimizer_flow = torch.optim.Adam(list(mi_loss_instance.fX.parameters())+list(mi_loss_instance.gY.parameters()))
        annealing_freq, factor = annealing_scheme
        # scheduler_flow = pyro.optim.ExponentialLR(
        #     {
        #         "optimizer": optimizer_flow,
        #         "optim_args": {"lr": lr_flow},
        #         "gamma" : factor,
        #         "verbose": False,
        #     }
        # )
        scheduler_flow = pyro.optim.ReduceLROnPlateau(
            {
                "optimizer": optimizer_flow,
                "optim_args": {"lr": lr_flow},
                "factor": .8,
                "patience": 2,
                "verbose": False,
            }
        )
    
    min_loss = torch.inf
    j=0
    loss_history = []
    num_steps_range = trange(0, num_steps + 0, desc="Loss: 0.000 ")
    for i in num_steps_range:
        optimizer_design.zero_grad()
        negMI = mi_loss_instance.differentiable_loss()
        negMI.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(design_net.parameters(), 1.0)
        optimizer_design.step()
        if run_flow and train_flow:
            optimizer_flow.zero_grad()
            # # Log Likelihood Optimization
            (mi_loss_instance.hXY).backward()
            # Foster Bound Optimization
            # (mi_loss_instance.hX + mi_loss_instance.hXY - mi_loss_instance.hY).backward()
            # (mi_loss_instance.hX+mi_loss_instance.hY).backward()
            optimizer_flow.step()
            
        if i % 100 == 0:
            num_steps_range.set_description("Loss: {:.3f} ".format(torch_item(negMI)))
            loss_eval = mi_loss_instance.loss()#*args, **kwargs)
            loss_history.append(loss_eval)
            
        # if i % annealing_freq == 0 and not i == 0:
        #     scheduler_design.step()
        #     if run_flow and train_flow:
        #         scheduler_flow.step()
        if i % 250 ==0 and not i == 0:
            with torch.no_grad():
                if min_loss < mi_loss_instance.hXY:
                    j+=1
                    if j>=4:
                        break
                else:
                    min_loss = 1*mi_loss_instance.hXY
                    j=0
            scheduler_design.step(mi_loss_instance.hXY)
            if run_flow and train_flow:
                scheduler_flow.step(mi_loss_instance.hXY)
            # scheduler.step(loss_eval)

    return epidemic, mi_loss_instance, loss_history

def main_loop(
    run,
    path_to_extra_data,
    device,
    T,
    train_flow_every_step,
    run_flow,
    batch_size,
    num_steps,
    lr_design,
    lr_flow,
    annealing_scheme,
    true_theta = None
):
    pyro.clear_param_store()
    SIMDATA = torch.load("data/sir_sde_data.pt", map_location=device)
    latent_dim = 2
    theta_loc = torch.tensor([0.5, 0.1], device=device).log()
    theta_covmat = torch.eye(2, device=device) * 0.5 ** 2
    flow_theta = None
    flow_obs = None
    train_flow = True
    prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)

    if true_theta is None:
        true_theta = prior.sample(torch.Size([1])).exp()
        # true_theta = torch.tensor([[.7,.075]], device=device)#HighR0 #torch.tensor([[.33,.11]], device=device)#Middle R0#torch.tensor([[.15,.15]], device=device)#Low R0#

    designs_so_far = []
    observations_so_far = []

    # Set posterior equal to the prior
    posterior_loc = theta_loc
    posterior_cov = theta_covmat
    
    previous_design = torch.tensor(0.0, device=device)
    for t in range(0, T):
        t_start = time.time()
        t_sim = t_start
        print(f"Step {t + 1}/{T} of Run {run + 1}")
        pyro.clear_param_store()
        if t > 0:
            # update previous design so the lower bound gets updated!
            # e.forward method of Epidemic designs/obs stuff in lists
            previous_design = design_transformed[0].reshape(-1)
            ## pre-simulate data using the posterior as the prior!
            epidemic._remove_data()
            del SIMDATA
            SIMDATA = solve_sir_sdes_var(
                num_samples=5000,#5000,#100000
                device=device,
                grid=100,#10000
                save=False,
                savegrad=False,
                theta_loc=posterior_loc,
                theta_covmat=posterior_cov,
                flows_theta=flow_theta
            )
            SIMDATA = {
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in SIMDATA.items()
            }
            t_sim = time.time()

        epidemic, mi_loss_instance, loss_history  = optimise_design(
            simdata=SIMDATA,
            previous_design=previous_design,
            flow_theta = flow_theta,
            flow_obs = flow_obs,
            train_flow=train_flow,
            run_flow=run_flow,
            device=device,
            batch_size=batch_size,
            num_steps=num_steps,
            lr_design=lr_design,
            lr_flow=lr_flow,
            annealing_scheme=annealing_scheme,
        )   
        ################################ CHECK ON THESE ###################################################
        with torch.no_grad():
            if t>0:
                trans_true_theta,_ = flow_theta.forward(torch.log(true_theta))
            else:
                trans_true_theta = torch.log(true_theta)
            theta, design_untransformed, observation = epidemic.forward(epidemic.theta_to_index(theta=torch.exp(trans_true_theta)))
            design_transformed = epidemic.transform_designs(design_untransformed[0])
            mux = mi_loss_instance.mu[:latent_dim].detach()
            muy = mi_loss_instance.mu[latent_dim:].detach()
            Sigmaxx = mi_loss_instance.Sigma[:latent_dim,:latent_dim].detach()
            Sigmaxy = mi_loss_instance.Sigma[:latent_dim,latent_dim:].detach()
            Sigmayy = mi_loss_instance.Sigma[latent_dim:,latent_dim:].detach()
            obs, _ = mi_loss_instance.gY.forward(observation[0])
            posterior_loc = (mux + torch.matmul(Sigmaxy,torch.linalg.solve(Sigmayy,(obs-muy))).flatten())
            max_posterior = mi_loss_instance.fX.reverse(posterior_loc).exp()
            posterior_cov = Sigmaxx-torch.matmul(Sigmaxy,torch.linalg.solve(Sigmayy,Sigmaxy.T))
            flow_theta = mi_loss_instance.fX
            flow_obs = mi_loss_instance.gY
        ###################################################################################################
        t_end = time.time()
        run_time = t_end-t_start
        
        designs_so_far.append(design_untransformed[0].detach().clone().cpu())
        observations_so_far.append(observation[0].cpu())
        
        extra_data = {}
        extra_data["mu"] = mi_loss_instance.mu.detach().clone().cpu().numpy()
        extra_data["sigmas"] = mi_loss_instance.Sigma.detach().clone().cpu().numpy()
        extra_data["flow_theta"] = copy.deepcopy(mi_loss_instance.fX).cpu()
        extra_data["flow_obs"] = copy.deepcopy(mi_loss_instance.gY).cpu()
        extra_data["posterior_loc"] = posterior_loc.cpu().numpy()
        extra_data["posterior_cov"] = posterior_cov.cpu().numpy()
        extra_data["total_time"] = run_time
        extra_data["simulation_time"] = t_sim - t_start
        extra_data["designs_trans"] = design_transformed.detach().clone().cpu()
        extra_data["designs_untrans"] = design_untransformed[0].detach().clone().cpu()
        extra_data["observations"] = observation[0].cpu()
        extra_data["theta"] = true_theta.cpu()
        
        path_to_run = path_to_extra_data + '/Run{}'.format(run)
        path_to_step = path_to_run + '/Step{}.pickle'.format(t)
        path_to_loss = path_to_run +'/Loss{}.pickle'.format(t)
        if not os.path.exists(path_to_run):
            os.makedirs(path_to_run)
        with open(path_to_step, "wb") as f:
            pickle.dump(extra_data, f)
        with open(path_to_loss, "wb") as f:
            pickle.dump(loss_history, f)
        del extra_data

        print(f"Final posterior: mean = {max_posterior}, sd = {posterior_cov}")
        print(f"True theta = {true_theta}")
        
        if not train_flow_every_step:
            train_flow = False

    data_dict = {}
    for i, xi in enumerate(designs_so_far):
        data_dict[f"xi{i + 1}"] = xi.cpu()
    for i, y in enumerate(observations_so_far):
        data_dict[f"y{i + 1}"] = y.cpu()
    data_dict["theta"] = true_theta.cpu()

    return data_dict


def main(
    seed,
    mlflow_experiment_name,
    num_loop,
    num_parallel,
    device,
    T,
    train_flow_every_step,
    run_flow,
    batch_size,
    num_steps,
    lr_design,
    lr_flow,
    annealing_scheme,
    true_theta = None,
):

    pyro.clear_param_store()
    seed = auto_seed(seed)
    pyro.set_rng_seed(seed)

    mlflow.set_experiment(mlflow_experiment_name)
    # Log everything
    mlflow.log_param("seed", seed)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("lr_design", lr_design)
    mlflow.log_param("lr_flow", lr_flow)
    mlflow.log_param("num_loop", num_loop)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("annealing_scheme", str(annealing_scheme))

    if not os.path.exists("./mlflow_outputs"):
        os.makedirs("./mlflow_outputs")
    
    t = time.localtime()
    extra_data_id = time.strftime("%Y%m%d%H%M%S", t)
    path_to_extra_data = "./experiment_outputs/SIR/{}".format(extra_data_id)
    print(path_to_extra_data)
    if not os.path.exists(path_to_extra_data):
        os.makedirs(path_to_extra_data)
                
    results_vi = {"loop": [], "seed": seed}  
    results = Parallel(n_jobs=num_parallel)(delayed(main_loop)(run=i,
                        path_to_extra_data =path_to_extra_data,
                        device=device,
                        T=T,
                        train_flow_every_step=train_flow_every_step,
                        run_flow=run_flow,
                        batch_size=batch_size,
                        num_steps=num_steps,
                        lr_design=lr_design,
                        lr_flow=lr_flow,
                        annealing_scheme=annealing_scheme,
                        true_theta = true_theta,
                    ) for i in range(num_loop))
    for i in range(num_loop):
        results_vi["loop"].append(results[i])

    # Log the results dict as an artifact
    with open("./mlflow_outputs/results_sir_mm_vi.pickle", "wb") as f:
        pickle.dump(results_vi, f)
    mlflow.log_artifact("mlflow_outputs/results_sir_mm_vi.pickle")
    
    ml_info = mlflow.active_run().info
    path_to_artifact = "mlruns/{}/{}/artifacts/results_sir_mm_vi.pickle".format(
        ml_info.experiment_id, ml_info.run_id
    )
    with open("./"+path_to_artifact, "wb") as f:
        pickle.dump(results_vi, f)
    print("Path to artifact - use this when evaluating:\n", path_to_artifact)

    
    extra_meta = {
        "model": "SIR_epidemic",
        "train_flow_every_step": train_flow_every_step,
        "run_flow": run_flow
    }
    path_to_extra_meta =path_to_extra_data + '/extra_meta.pickle'
    with open(path_to_extra_meta, "wb") as f:
        pickle.dump(extra_meta, f)
        
    print(path_to_extra_data)
    print("Done.")
    # --------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VI baseline: SIR model")
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-loop", default=32, type=int)#32
    parser.add_argument(
        "--num-parallel", help="Number of loops to run parallel", default=1, type=int
    )
    parser.add_argument("--batch-size", default=512, type=int)  #512 == T
    parser.add_argument("--num-experiments", default=5, type=int)  # == T
    parser.add_argument("--device", default="cuda", type=str)#"cuda""cpu"
    parser.add_argument(
        "--mlflow-experiment-name", default="epidemic_variational", type=str
    )
    parser.add_argument("--lr-design", default=0.005, type=float)#0.01
    parser.add_argument("--lr-flow", default=0.005, type=float)
    parser.add_argument("--annealing-scheme", nargs="+", default=[500,.95], type=float)
    parser.add_argument("--num-steps", default=5000, type=int)#5000
    parser.add_argument("--train-flow-every-step", default=False, type=bool)
    parser.add_argument("--run-flow", default=False, type=bool)
    
    # parser.add_argument("--true-theta", default=True, type=bool)
    args = parser.parse_args()

    main(
        seed=args.seed,
        mlflow_experiment_name=args.mlflow_experiment_name,
        num_loop=args.num_loop,
        num_parallel=args.num_parallel,
        device=args.device,
        T=args.num_experiments,
        train_flow_every_step= args.train_flow_every_step,
        run_flow = args.run_flow,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr_design=args.lr_design,
        lr_flow=args.lr_flow,
        annealing_scheme = args.annealing_scheme,
        # true_theta = args.true_theta
    )

#######################################################
# with torch.no_grad():
#     import numpy as np
#     import scipy
#     from matplotlib.pyplot import colorbar, pcolor, show, scatter
#     x = np.linspace(0.01,1,100)
#     y = np.linspace(0.01,.2,100)
#     X, Y = np.meshgrid(x, y)
    
#     fX, logJac = mi_loss_instance.fX.forward(torch.from_numpy(np.log(np.vstack((X.flatten(),Y.flatten())).T)).float().to(mux.device))#.numpy()
#     # gY = np.exp(mi_loss_instance.gY.reverse(torch.from_numpy(np.log(y))).numpy())
#     # xx, yy = np.meshgrid(fX[:,0], fX[:,1])
#     # points = np.stack((xx, yy), axis=-1)
#     points = fX.reshape((100,100,2)).cpu()
#     Z = scipy.stats.multivariate_normal.pdf(points, posterior_loc.cpu(), posterior_cov.cpu())*torch.exp(logJac.reshape((100,100)).cpu()).numpy()
#     pcolor(X, Y, Z)
#     scatter(true_theta.cpu().numpy()[0][0],true_theta.cpu().numpy()[0][1], color='red', marker='x')
#     scatter(mi_loss_instance.fX.reverse(posterior_loc).exp().cpu().numpy()[0],mi_loss_instance.fX.reverse(posterior_loc).exp().cpu().numpy()[1], color='green', marker='x')
#     colorbar()
#     show()
    
# with torch.no_grad():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     XY = np.random.multivariate_normal(posterior_loc.cpu(),posterior_cov.cpu(),5000)
#     fX = mi_loss_instance.fX.reverse(torch.from_numpy((XY)).float().to(mux.device)).exp()
#     plt.scatter(fX[:,0].cpu(), fX[:,1].cpu(), s=10, c='black', alpha=0.5)
#     plt.scatter(true_theta.cpu().numpy()[0][0],true_theta.cpu().numpy()[0][1], color='red', marker='x')
#     plt.scatter(mi_loss_instance.fX.reverse(posterior_loc).exp().cpu().numpy()[0],mi_loss_instance.fX.reverse(posterior_loc).exp().cpu().numpy()[1], color='green', marker='x')
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
    
#     fX, logJac = mi_loss_instance.fX.forward(torch.from_numpy(np.log(np.vstack((X.flatten(),Y.flatten())).T)).to(mux.device).float())#.numpy()
#     # gY = np.exp(mi_loss_instance.gY.reverse(torch.from_numpy(np.log(y))).numpy())
#     # xx, yy = np.meshgrid(fX[:,0], fX[:,1])
#     # points = np.stack((xx, yy), axis=-1)
#     points = fX.reshape((100,100,2)).cpu()
#     Z = scipy.stats.multivariate_normal.pdf(points, mux.cpu(), Sigmaxx.cpu())*torch.exp(logJac.reshape((100,100))).cpu().numpy()
#     pcolor(X, Y, Z)
#     scatter(true_theta.cpu().numpy()[0][0],true_theta.cpu().numpy()[0][1], color='red', marker='x')
#     scatter(mi_loss_instance.fX.reverse(mux).exp().cpu().numpy()[0],mi_loss_instance.fX.reverse(mux).exp().cpu().numpy()[1], color='green', marker='x')
#     colorbar()
#     show()
    
# with torch.no_grad():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     XY = np.random.multivariate_normal(mux.cpu(),Sigmaxx.cpu(),5000)#,device = mux.device
#     fX = mi_loss_instance.fX.reverse(torch.from_numpy((XY)).float().to(mux.device)).exp()
#     plt.scatter(fX[:,0].cpu(), fX[:,1].cpu(), s=10, c='black', alpha=0.5)
#     plt.scatter(true_theta.cpu().numpy()[0][0],true_theta.cpu().numpy()[0][1], color='red', marker='x')
#     plt.scatter(mi_loss_instance.fX.reverse(mux).exp().cpu().numpy()[0],mi_loss_instance.fX.reverse(mux).exp().cpu().numpy()[1], color='green', marker='x')
#     # Add labels and title
#     plt.xlabel('X0-axis label')
#     plt.ylabel('X1-axis label')
#     plt.xlim(0,1)
#     plt.ylim(0, .2)
#     plt.title('Scatter Plot of X and Y')

#     # Show the plot
#     plt.show()