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

from mm_locfin_variational import IdentityTransform, RealNVP, MomentMatchMarginalPosterior, SplineFlow

##############################################################################################
############################### Alternative Epidemic Model ###################################
##############################################################################################
from oed.primitives import observation_sample, latent_sample, compute_design
import pandas as pd
from epidemic import SIR_SDE_Simulator
class Epidemic2(nn.Module):

    """
    Class for the SDE-based SIR model. This version loads in pre-simulated data
    and then access observations corresponding to the emitted design.
    """

    def __init__(
        self,
        design_net,
        T,
        design_transform="iid",
        simdata=None,
        lower_bound=torch.tensor(1e-2),
        upper_bound=torch.tensor(100.0 - 1e-2),
    ):

        super().__init__()#Epidemic2, self

        self.p = 2  # dim of latent
        self.design_net = design_net
        self.T = T  # number of experiments
        self.SIMDATA = simdata
        loc = torch.tensor([0.5, 0.1]).log().to(simdata["ys"].device)
        covmat = torch.eye(2).to(simdata["ys"].device) * 0.5 ** 2
        self._prior_on_log_theta = torch.distributions.MultivariateNormal(loc, covmat)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if design_transform == "ts":
            self.transform_designs = self._transform_designs_increasing
        elif design_transform == "iid":
            self.transform_designs = self._transform_designs_independent
        else:
            raise ValueError

    def simulator(self, xi, theta, batch_data):
        # extract data from global dataset
        sim_sir = SIR_SDE_Simulator.apply
        y = sim_sir(xi, batch_data, theta.device)

        return y

    def _get_batch_data(self, indices):
        batch_data = {
            "ys": self.SIMDATA["ys"][:, indices],
            "prior_samples": self.SIMDATA["prior_samples"][indices, :],
            "ts": self.SIMDATA["ts"],
            "dt": self.SIMDATA["dt"],
        }
        return batch_data

    def _transform_designs_increasing(self, xi_untransformed, xi_prev):
        xi_prop = nn.Sigmoid()(xi_untransformed)
        xi = xi_prev + xi_prop * (self.upper_bound - xi_prev)
        return xi

    def _transform_designs_independent(self, xi_untransformed, xi_prev=None):
        xi_prop = nn.Sigmoid()(xi_untransformed)
        xi = self.lower_bound + xi_prop * (self.upper_bound - self.lower_bound)
        return xi

    def _remove_data(self):
        self.SIMDATA = None

    def theta_to_index(self, theta):
        theta_expanded = theta.unsqueeze(1).expand(
            theta.shape[0], self.SIMDATA["prior_samples"].shape[0], theta.shape[1]
        )
        norms = torch.linalg.norm(
            self.SIMDATA["prior_samples"] - theta_expanded, dim=-1
        )
        closest_indices = norms.min(-1).indices
        assert closest_indices.shape[0] == theta.shape[0]
        return closest_indices

    def model(self):
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)

        device = self.SIMDATA["prior_samples"].device
        prior_on_index = dist.Categorical(
            torch.ones(self.SIMDATA["num_samples"], device=device)
        )

        ################################################################################
        # Sample theta
        ################################################################################
        # conditioning should be on the indices:

        indices = pyro.sample("indices", prior_on_index)
        batch_data = self._get_batch_data(indices)

        # helper to 'sample' theta
        def get_theta():
            return batch_data["prior_samples"].log()

        theta = latent_sample("theta", get_theta)
        theta = theta.exp()

        y_outcomes = []
        xi_designs = []

        # at t=0 set last design equal to the lower bound
        xi_prev = self.lower_bound

        for t in range(self.T):
            ####################################################################
            # Get a design xi
            ####################################################################
            xi_untransformed = compute_design(
                f"xi{t + 1}", self.design_net.lazy(*zip(xi_designs, y_outcomes))
            )
            # squeeze the first dim (corrresponds to <n>)
            xi = self.transform_designs(
                xi_untransformed=xi_untransformed.squeeze(1), xi_prev=xi_prev,
            )

            ####################################################################
            # Sample y
            ####################################################################
            y = observation_sample(
                f"y{t + 1}", self.simulator, xi=xi, theta=theta, batch_data=batch_data
            )

            ####################################################################
            # Update history
            ####################################################################
            y_outcomes.append(y)
            xi_designs.append(xi_untransformed)  #! pass untransformed

            xi_prev = xi  # set current design as previous for next loop

        del batch_data  # delete manually just in case
        return theta, xi_designs, y_outcomes

    def forward(self, indices):
        """ Run the policy for a given index (corresponding to a latent theta) """
        self.design_net.eval()

        def conditioned_model():
            # indices = self.theta_to_index(theta)
            with pyro.plate_stack("expand_theta_test", [indices.shape[0]]):
                # condition on "theta" (ie the corresponding indices)
                return pyro.condition(self.model, data={"indices": indices})()

        with torch.no_grad():
            theta, designs, observations = conditioned_model()

        return theta, designs, observations

    def eval(self, theta=None, verbose=False):
        """
        Run policy and produce a df with output
        """
        self.design_net.eval()
        # can't do more than one in this form since we (in all likelihood)
        # have one realisation per theta
        n_trace = 1
        if theta is None:
            theta = self._prior_on_log_theta.sample(torch.Size([1])).exp()
            indices = self.theta_to_index(theta)
        else:
            indices = self.theta_to_index(theta)

        output = []
        theta, designs, observations = self.forward(indices)
        for i in range(n_trace):
            run_xis = []
            run_ys = []

            xi_prev = self.lower_bound
            if verbose:
                print("Example run")
                print(f"*True Theta: {theta[i]}*")

            for t in range(self.T):
                xi_untransformed = designs[t][i].detach().cpu()
                xi = self.transform_designs(
                    xi_untransformed=xi_untransformed.squeeze(0), xi_prev=xi_prev,
                )
                xi_prev = xi
                run_xis.append(xi.cpu().reshape(-1))
                y = observations[t][i].detach().cpu().item()
                run_ys.append(y)

                if verbose:
                    print(f"xi{t + 1}: {run_xis[-1][0].data}  y{t + 1}: {y}")

            run_df = pd.DataFrame(torch.stack(run_xis).numpy())
            run_df.columns = [f"xi_{i}" for i in range(xi.shape[0])]
            run_df["observations"] = run_ys
            run_df["order"] = list(range(1, self.T + 1))
            run_df["run_id"] = i + 1
            output.append(run_df)

        return pd.concat(output), theta.cpu().numpy()

def solve_sir_sdes2(
    num_samples,
    device,
    grid=10000,
    savegrad=False,
    save=False,
    filename="sir_sde_data.pt",
    theta_loc=None,
    theta_covmat=None,
    flows_theta= None,
):
    ####### Change priors here ######
    if theta_loc is None or theta_covmat is None:
        theta_loc = torch.tensor([0.5, 0.1], device=device).log()
        theta_covmat = torch.eye(2, device=device) * 0.5 ** 2

    prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)
    params_Z = prior.sample(torch.Size([num_samples]))#.exp()
    with torch.no_grad():
        params_theta = flows_theta.reverse(params_Z)
        params = params_theta.exp()
    #################################

    T0, T = 0.0, 100.0  # initial and final time
    GRID = grid  # time-grid

    population_size = 500.0
    initial_infected = 2.0  # initial number of infected

    ## [non-infected, infected]
    y0 = torch.tensor(
        num_samples * [[population_size - initial_infected, initial_infected]],
        device=device,
    )  # starting point
    ts = torch.linspace(T0, T, GRID, device=device)  # time grid

    sde = SIR_SDE(
        population_size=torch.tensor(population_size, device=device), params=params,
    ).to(device)

    start_time = time.time()
    ys = torchsde.sdeint(sde, y0, ts)  # solved sde
    end_time = time.time()
    # return ys0, ys1
    print("Simulation Time: %s seconds" % (end_time - start_time))

    save_dict = dict()
    idx_good = torch.where(ys[:, :, 1].mean(0) >= 1)[0]

    save_dict["prior_samples"] = params[idx_good].cpu()
    save_dict["ts"] = ts.cpu()
    save_dict["dt"] = (ts[1] - ts[0]).cpu()  # delta-t (time grid)
    # drop 0 as it's not used (saves space)
    save_dict["ys"] = ys[:, idx_good, 1].cpu()

    # grads can be calculated in backward pass (saves space)
    if savegrad:
        # central difference
        grads = (ys[2:, ...] - ys[:-2, ...]) / (2 * save_dict["dt"])
        save_dict["grads"] = grads[:, idx_good, :].cpu()

    # meta data
    save_dict["N"] = population_size
    save_dict["I0"] = initial_infected
    save_dict["num_samples"] = save_dict["prior_samples"].shape[0]

    if save:
        print("Saving data.", end=" ")
        torch.save(save_dict, f"data/{filename}")

    print("DONE.")
    return save_dict
##############################################################################################
##############################################################################################

def optimise_design(
    simdata,
    previous_design,
    posterior_loc,
    posterior_cov,
    flow_theta,
    flow_obs,
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

    epidemic = Epidemic2(
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

    hidden = 64#None#
    if hidden == None:
        fX = IdentityTransform()
        gY = IdentityTransform()
    else:
        dim_x = latent_dim
        dim_y = observation_dim ### Try Quadratic and test bin/bound layers try 12
        if flow_theta == None:
            fX = SplineFlow(dim_x, count_bins=8, bounds=5, device=device).to(device)
        else:
            fX = copy.deepcopy(flow_theta)
        if flow_obs == None:
            gY = SplineFlow(dim_y, count_bins=8, bounds=5, device=device).to(device)
        else:
            gY = copy.deepcopy(flow_obs)
        # fX = RealNVP(dim_x, num_blocks=3, dim_hidden=hidden//2,device=device).to(device)
        # gY = RealNVP(dim_y, num_blocks=3, dim_hidden=hidden//2,device=device).to(device)
    
    ### Set-up loss ###
    mi_loss_instance = MomentMatchMarginalPosterior(
        model=epidemic.model,
        batch_size=batch_size,
        flow_x=fX,
        flow_y=gY,
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
    prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)

    if true_theta is None:
        # true_theta = prior.sample(torch.Size([1])).exp()
        true_theta = torch.tensor([[.33,.11]])#Middle R0#torch.tensor([[.7,.075]])#HighR0 #torch.tensor([[.15,.15]])#Low R0#

    design_trans_so_far = []
    designs_so_far = []
    observations_so_far = []

    # Set posterior equal to the prior
    posterior_loc = theta_loc
    posterior_cov = theta_covmat
    # posterior_scale = torch.sqrt(theta_covmat.diag())
    
    previous_design = torch.tensor(0.0, device=device)  # no previous design
    for t in range(0, T):
        print(f"Step {t + 1}/{T} of Run {run + 1}")
        pyro.clear_param_store()
        if t > 0:
            # update previous design so the lower bound gets updated!
            # e.forward method of Epidemic designs/obs stuff in lists
            previous_design = design_transformed[0].reshape(-1)
            ## pre-simulate data using the posterior as the prior!
            epidemic._remove_data()
            del SIMDATA
            SIMDATA = solve_sir_sdes2(
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
            print(true_theta)#flow_theta.reverse
            print(posterior_loc)#
            print(mi_loss_instance.fX.reverse(posterior_loc).exp())
            posterior_cov = Sigmaxx-torch.matmul(Sigmaxy,torch.linalg.solve(Sigmayy,Sigmaxy.T))
            flow_theta = mi_loss_instance.fX
            flow_obs = mi_loss_instance.gY
        ###################################################################################################
  
        
        design_trans_so_far.append(design_transformed)
        designs_so_far.append(design_untransformed[0])
        observations_so_far.append(observation[0])

    print(f"Final posterior: mean = {posterior_loc}, sd = {posterior_cov}")
    print(f"True theta = {true_theta}")

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
    device,
    T,
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

    results_vi = {"loop": [], "seed": seed}
    for i in range(num_loop):
        results = main_loop(
            run=i,
            mlflow_run_id=mlflow.active_run().info.run_id,
            device=device,
            T=T,
            batch_size=batch_size,
            num_steps=num_steps,
            lr=lr / (i + 1),
            lr_critic=lr_critic / (i + 1),
            annealing_scheme=annealing_scheme,
        )
        results_vi["loop"].append(results)

    # Log the results dict as an artifact
    with open("./mlflow_outputs/results_sir_vi.pickle", "wb") as f:
        pickle.dump(results_vi, f)
    mlflow.log_artifact("mlflow_outputs/results_sir_vi.pickle")
    print("Done.")

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
with torch.no_grad():
    import numpy as np
    import scipy
    from matplotlib.pyplot import colorbar, pcolor, show, scatter
    x = np.linspace(0.01,1,100)
    y = np.linspace(0.01,.2,100)
    X, Y = np.meshgrid(x, y)
    
    fX, logJac = mi_loss_instance.fX.forward(torch.from_numpy(np.log(np.vstack((X.flatten(),Y.flatten())).T)).float())#.numpy()
    # gY = np.exp(mi_loss_instance.gY.reverse(torch.from_numpy(np.log(y))).numpy())
    # xx, yy = np.meshgrid(fX[:,0], fX[:,1])
    # points = np.stack((xx, yy), axis=-1)
    points = fX.reshape((100,100,2))
    Z = scipy.stats.multivariate_normal.pdf(points, posterior_loc, posterior_cov)*torch.exp(logJac.reshape((100,100))).numpy()
    pcolor(X, Y, Z)
    scatter(true_theta.numpy()[0][0],true_theta.numpy()[0][1], color='red', marker='x')
    scatter(mi_loss_instance.fX.reverse(posterior_loc).exp().numpy()[0],mi_loss_instance.fX.reverse(posterior_loc).exp().numpy()[1], color='green', marker='x')
    colorbar()
    show()
    
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
    

with torch.no_grad():
    import numpy as np
    import scipy
    from matplotlib.pyplot import colorbar, pcolor, show, scatter
    x = np.linspace(0.01,1,100)
    y = np.linspace(0.01,.2,100)
    X, Y = np.meshgrid(x, y)
    
    fX, logJac = mi_loss_instance.fX.forward(torch.from_numpy(np.log(np.vstack((X.flatten(),Y.flatten())).T)).float())#.numpy()
    # gY = np.exp(mi_loss_instance.gY.reverse(torch.from_numpy(np.log(y))).numpy())
    # xx, yy = np.meshgrid(fX[:,0], fX[:,1])
    # points = np.stack((xx, yy), axis=-1)
    points = fX.reshape((100,100,2))
    Z = scipy.stats.multivariate_normal.pdf(points, mux, Sigmaxx)*torch.exp(logJac.reshape((100,100))).numpy()
    pcolor(X, Y, Z)
    scatter(true_theta.numpy()[0][0],true_theta.numpy()[0][1], color='red', marker='x')
    scatter(mi_loss_instance.fX.reverse(mux).exp().numpy()[0],mi_loss_instance.fX.reverse(mux).exp().numpy()[1], color='green', marker='x')
    colorbar()
    show()
    
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