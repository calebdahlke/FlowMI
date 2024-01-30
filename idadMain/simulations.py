from oed.primitives import observation_sample, latent_sample, compute_design
import pandas as pd
import torch
from torch import nn
import pyro
import pyro.distributions as dist
from flow_estimator_pyro import IdentityTransform

from oed.primitives import observation_sample, latent_sample, compute_design
import pandas as pd
from epidemic import SIR_SDE_Simulator
import time
import torchsde    
from epidemic_simulate_data import SIR_SDE
##############################################################################################
################################# Alternative Hidden Model ###################################
##############################################################################################
class HiddenObjectsVar(nn.Module):
    """Location finding example"""

    def __init__(
        self,
        design_net,
        base_signal=0.1,  # G-map hyperparam
        max_signal=1e-4,  # G-map hyperparam
        theta_loc=None,  # prior on theta mean hyperparam
        theta_covmat=None,  # prior on theta covariance hyperparam
        flow_theta = None,
        noise_scale=None,  # this is the scale of the noise term
        p=1,  # physical dimension
        K=1,  # number of sources
        T=2,  # number of experiments
    ):
        super().__init__()
        self.design_net = design_net
        self.base_signal = base_signal
        self.max_signal = max_signal
        # Set prior:
        self.theta_loc = theta_loc if theta_loc is not None else torch.zeros(K*p)
        self.theta_covmat = theta_covmat if theta_covmat is not None else torch.eye(K*p)
        self.flow_theta = flow_theta if flow_theta is not None else IdentityTransform() #reverse
        self.theta_prior = dist.MultivariateNormal(
            self.theta_loc, self.theta_covmat
        )
        # Observations noise scale:
        self.noise_scale = noise_scale if noise_scale is not None else torch.tensor(1.0)
        self.n = 1  # samples per design=1
        self.p = p  # dimension of theta (location finding example will be 1, 2 or 3).
        self.K = K  # number of sources
        self.T = T  # number of experiments

    def forward_map(self, xi, theta):
        """Defines the forward map for the hidden object example
        y = G(xi, theta) + Noise.
        """
        # two norm squared
        sq_two_norm = (xi - theta).pow(2).sum(axis=-1)
        # add a small number before taking inverse (determines max signal)
        sq_two_norm_inverse = (self.max_signal + sq_two_norm).pow(-1)
        # sum over the K sources, add base signal and take log.
        mean_y = torch.log(self.base_signal + sq_two_norm_inverse.sum(-1, keepdim=True))
        return mean_y

    def model(self):
        if hasattr(self.design_net, "parameters"):
            #! this is required for the pyro optimizer
            pyro.module("design_net", self.design_net)

        ########################################################################
        # Sample latent variables theta
        ########################################################################
        theta = latent_sample("theta", self.theta_prior)
        with torch.no_grad():
            theta = self.flow_theta.reverse(theta)#.flatten(-1)
        theta = theta.reshape((len(theta),self.K,self.p))
        y_outcomes = []
        xi_designs = []

        # T-steps experiment
        for t in range(self.T):
            ####################################################################
            # Get a design xi; shape is [batch size x self.n x self.p]
            ####################################################################
            xi = compute_design(
                f"xi{t + 1}", self.design_net.lazy(*zip(xi_designs, y_outcomes))
            )
            ####################################################################
            # Sample y at xi; shape is [batch size x 1]
            ####################################################################
            mean = self.forward_map(xi, theta)
            sd = self.noise_scale
            y = observation_sample(f"y{t + 1}", dist.Normal(mean, sd).to_event(1))

            ####################################################################
            # Update history
            ####################################################################
            y_outcomes.append(y)
            xi_designs.append(xi)

        return theta, xi_designs, y_outcomes

    def forward(self, theta):
        """Run the policy for a given theta"""
        self.design_net.eval()

        def conditioned_model():
            with pyro.plate_stack("expand_theta_test", [theta.shape[0]]):
                # condition on theta
                return pyro.condition(self.model, data={"theta": theta})()

        with torch.no_grad():
            theta, designs, observations = conditioned_model()
        self.design_net.train()
        return designs, observations

    def eval(self, n_trace=3, theta=None, verbose=True):
        """run the policy, print output and return it in a dataframe"""
        self.design_net.eval()

        if theta is None:
            theta = self.theta_prior.sample(torch.Size([n_trace]))
            # theta = self.flow_theta.reverse(theta)
        else:
            theta = theta.unsqueeze(0).expand(n_trace, *theta.shape)
            # dims: [n_trace * number of thetas given, shape of theta]
            theta = theta.reshape(-1, *theta.shape[2:])

        designs, observations = self.forward(theta)
        output = []
        true_thetas = []

        for i in range(n_trace):
            if verbose:
                print("\nExample run {}".format(i + 1))
                print(f"*True Theta: {theta[i].cpu()}*")
            run_xis = []
            run_ys = []
            # Print optimal designs, observations for given theta
            for t in range(self.T):
                xi = designs[t][i].detach().cpu().reshape(-1)
                run_xis.append(xi)
                y = observations[t][i].detach().cpu().item()
                run_ys.append(y)
                if verbose:
                    print(f"xi{t + 1}: {xi},   y{t + 1}: {y}")
            run_df = pd.DataFrame(torch.stack(run_xis).numpy())
            run_df.columns = [f"xi_{i}" for i in range(self.p)]
            run_df["observations"] = run_ys
            run_df["order"] = list(range(1, self.T + 1))
            run_df["run_id"] = i + 1
            output.append(run_df)

        self.design_net.train()
        return pd.concat(output), theta.cpu().numpy()
##############################################################################################
##############################################################################################


##############################################################################################
############################### Alternative Epidemic Model ###################################
##############################################################################################

class EpidemicVar(nn.Module):

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

def solve_sir_sdes_var(
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
