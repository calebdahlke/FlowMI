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
from flow_estimator_pyro import MomentMatchMarginalPosterior, SplineFlow, IdentityTransform, RealNVP, VariationalMutualInformationOptimizer, cov


from oed.primitives import observation_sample, latent_sample, compute_design
import pandas as pd
from epidemic import SIR_SDE_Simulator
from pyro import poutine
from pyro.poutine.util import prune_subsample_sites
from joblib import Parallel, delayed

class EpidemicVarNoResample(nn.Module):

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
        theta_loc = None,
        theta_covmat = None,
        flow_theta = None,
        lower_bound=torch.tensor(1e-2),
        upper_bound=torch.tensor(100.0 - 1e-2),
    ):

        super().__init__()#Epidemic2, self

        self.p = 2  # dim of latent
        self.design_net = design_net
        self.T = T  # number of experiments
        self.SIMDATA = simdata
        # Set prior:
        self.theta_loc = theta_loc if theta_loc is not None else torch.tensor([0.5, 0.1]).log().to(simdata["ys"].device)
        self.theta_covmat = theta_covmat if theta_covmat is not None else torch.eye(2).to(simdata["ys"].device) * 0.5 ** 2
        self.flow_theta = flow_theta if flow_theta is not None else IdentityTransform() #reverse
        self._prior_on_log_theta = dist.MultivariateNormal(
            self.theta_loc, self.theta_covmat
        )
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

        # with torch.no_grad():
        #     theta, designs, observations = conditioned_model()
        theta, designs, observations = conditioned_model()
        return theta, designs, observations

    def eval(self, theta=None, verbose=False):
        """
        Run policy and produce a df with output
        """
        self.design_net.eval()
        # can't do more than one in this form since we (in all likelihood)
        # have one realisation per theta
        # n_trace = 1
        if theta is None:
            with torch.no_grad():
                theta_z = self._prior_on_log_theta.sample(torch.Size([512]))#512
                theta = self.flow_theta.reverse(theta_z).exp()
                indices = self.theta_to_index(theta)
        else:
            indices = self.theta_to_index(theta)

        output = []
        theta_exp, designs, observations = self.forward(indices)
        with torch.no_grad():
            theta = theta_exp.log()
        return theta, (designs[0], observations[0])#(latents, *zip(designs, observations))#
        # for i in range(n_trace):
        #     run_xis = []
        #     run_ys = []

        #     xi_prev = self.lower_bound
        #     if verbose:
        #         print("Example run")
        #         print(f"*True Theta: {theta[i]}*")

        #     for t in range(self.T):
        #         xi_untransformed = designs[t][i].detach().cpu()
        #         xi = self.transform_designs(
        #             xi_untransformed=xi_untransformed.squeeze(0), xi_prev=xi_prev,
        #         )
        #         xi_prev = xi
        #         run_xis.append(xi.cpu().reshape(-1))
        #         y = observations[t][i].detach().cpu().item()
        #         run_ys.append(y)

        #         if verbose:
        #             print(f"xi{t + 1}: {run_xis[-1][0].data}  y{t + 1}: {y}")

        #     run_df = pd.DataFrame(torch.stack(run_xis).numpy())
        #     run_df.columns = [f"xi_{i}" for i in range(xi.shape[0])]
        #     run_df["observations"] = run_ys
        #     run_df["order"] = list(range(1, self.T + 1))
        #     run_df["run_id"] = i + 1
        #     output.append(run_df)

        # return pd.concat(output), theta.cpu().numpy()
    
class MomentMatchMarginalPosteriorNoResample(VariationalMutualInformationOptimizer):
    def __init__(self, model,sampler, batch_size, flow_x, flow_y,train_flow,device, **kwargs):
        super().__init__(
            model=model, batch_size=batch_size
        )
        self.sampler = sampler
        self.mu = 0
        self.Sigma = 0
        self.hX = 0
        self.hX_Y = 0
        self.fX = flow_x
        self.gY = flow_y
        self.train_flow = train_flow
        self.pi_const = 2*torch.acos(torch.zeros(1)).to(device)
        self.e_const = torch.exp(torch.tensor([1])).to(device)

    def differentiable_loss(self, *args, **kwargs):
        if self.train_flow:
            if hasattr(self.fX, "parameters"): #and not isinstance(self.fX, IdentityTransform):
                #! this is required for the pyro optimizer
                pyro.module("flow_x_net", self.fX)
            if hasattr(self.gY, "parameters"): #and not isinstance(self.gY, IdentityTransform):
                #! this is required for the pyro optimizer
                pyro.module("flow_y_net", self.gY)
                
        # sample from design
        # latents1, *history1 = self._get_data(args, kwargs)
        
        latents, *history = self.sampler()
        
        # model_v = self._vectorized(
        #         self.sampler, self.batch_size, name="outer_vectorization",
        #         data=data
        #     )
        # trace = poutine.trace(self.sampler, graph_type="flat").get_trace(*args, **kwargs)
        # trace = prune_subsample_sites(trace)
        # designs = [
        #     node["value"]
        #     for node in trace.nodes.values()
        #     if node.get("subtype") == "design_sample"
        # ]
        # observations = [
        #     node["value"]
        #     for node in trace.nodes.values()
        #     if node.get("subtype") == "observation_sample"
        # ]
        # latents = [
        #     node["value"]
        #     for node in trace.nodes.values()
        #     if node.get("subtype") == "latent_sample"
        # ]
        # latents = torch.cat(latents, axis=-1)
        # history = [(torch.cat(designs, axis=-1),torch.cat(observations, axis=-1))]
        
        
        dim_lat = latents.shape[1]
        dim_obs = history[0][1].shape[1]
        
        # # if self.train_flow:
        # if hasattr(self.fX, "parameters"):
        #     #! this is required for the pyro optimizer
        #     pyro.module("flow_x_net", self.fX)
        # if hasattr(self.gY, "parameters"):
        #     #! this is required for the pyro optimizer
        #     pyro.module("flow_y_net", self.gY)
        
        mufX, logDetJfX = self.fX.forward(latents)
        mugY, logDetJgY = self.gY.forward(history[0][1])
        # with torch.no_grad():
        #     mufX1, logDetJfX1 = self.fX.forward(torch.ones(dim_lat))
        #     print(mufX1)
        #     print(logDetJfX1)
        # compute loss
        data = torch.cat([mufX,mugY],axis=1)
        
        Sigma = cov(data)+1e-4*torch.eye(dim_lat+dim_obs).to(latents.device)
        self.hX = .5*torch.log(torch.linalg.det(Sigma[:dim_lat,:dim_lat]))+(dim_lat/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX)
        hY = .5*torch.log(torch.linalg.det(Sigma[dim_lat:,dim_lat:]))+(dim_obs/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJgY)
        hXY = .5*torch.log(torch.linalg.det(Sigma))+((dim_lat+dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX)-torch.mean(logDetJgY)
        self.hX_Y = hXY-hY
        hY_X = hXY-self.hX
        
        # save optimal parameters for decision
        self.mu = torch.mean(data,axis=0)
        self.Sigma = Sigma
        return self.hX+self.hX_Y+hY_X+hY#-torch.detach(2*self.hX_Y+hY_X+hY)

    def loss(self, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using the BA lower bound == -EIG
        """
        return self.hX-self.hX_Y

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

    epidemic = EpidemicVarNoResample(
        design_net=design_net,
        T=1,
        design_transform="iid",
        simdata=simdata,
        theta_loc = posterior_loc,
        theta_covmat = posterior_cov,
        flow_theta = flow_theta,
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
    mi_loss_instance = MomentMatchMarginalPosteriorNoResample(
        model=epidemic.model,
        sampler=epidemic.eval,
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
        true_theta = torch.tensor([[.7,.075]], device=device)#HighR0 #torch.tensor([[.15,.15]], device=device)#Low R0#torch.tensor([[.33,.11]], device=device)#Middle R0#
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
        
        mus_so_far.append(mi_loss_instance.mu.detach().clone().cpu().numpy())
        sigmas_so_far.append(mi_loss_instance.Sigma.detach().clone().cpu().numpy())
        flow_theta_so_far.append(copy.deepcopy(mi_loss_instance.fX).cpu())
        flow_obs_so_far.append(copy.deepcopy(mi_loss_instance.gY).cpu())

        posterior_loc_so_far.append(posterior_loc.cpu().numpy())
        posterior_cov_so_far.append(posterior_cov.cpu().numpy())
        time_per_design.append(run_time)
        design_trans_so_far.append(design_transformed.detach().clone().cpu())
        designs_so_far.append(design_untransformed[0].detach().clone().cpu())
        observations_so_far.append(observation[0].cpu())
        
        if not train_flow_every_step:
            train_flow = False

    print(f"Final posterior: mean = {max_posterior}, sd = {posterior_cov}")#posterior_loc
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

    return [data_dict, extra_data]


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
    # for i in range(num_loop):
    #     results, extra_data = main_loop(
    #         run=i,
    #         mlflow_run_id=mlflow.active_run().info.run_id,
    #         device=device,
    #         T=T,
    #         train_flow_every_step=train_flow_every_step,
    #         run_flow=run_flow,
    #         batch_size=batch_size,
    #         num_steps=num_steps,
    #         lr=lr / (i + 1),
    #         lr_critic=lr_critic / (i + 1),
    #         annealing_scheme=annealing_scheme,
    #     )
    #     results_vi["loop"].append(results)
    #     extra_vi["loop"].append(extra_data)
        
    results = Parallel(n_jobs=num_loop)(delayed(main_loop)(run=i,
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
                    ) for i in range(num_loop))
    for i in range(num_loop):
        results_vi["loop"].append(results[i][0])
        extra_vi["loop"].append(results[i][1])

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
    parser.add_argument("--device", default="cpu", type=str)#"cuda""cpu"
    parser.add_argument(
        "--mlflow-experiment-name", default="epidemic_variational", type=str
    )
    parser.add_argument("--lr-design", default=0.01, type=float)
    parser.add_argument("--lr-critic", default=0.001, type=float)
    parser.add_argument("--num-steps", default=5000, type=int)#5000
    parser.add_argument("--train-flow-every-step", default=True, type=bool)
    parser.add_argument("--run-flow", default=True, type=bool)
    args = parser.parse_args()

    main(
        seed=args.seed,
        num_loop=args.num_loop,
        device=args.device,
        batch_size=args.batch_size,
        T=args.num_experiments,
        train_flow_every_step= args.train_flow_every_step,
        run_flow = args.run_flow,
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