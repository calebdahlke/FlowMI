import os
import pickle
import argparse

import torch
from torch import nn
from pyro.infer.util import torch_item
import pyro
import pyro.distributions as dist
from tqdm import trange
import mlflow

from neural.baselines import BatchDesignBaseline, DesignBaseline

from experiment_tools.pyro_tools import auto_seed
from oed.design import OED

from location_finding import HiddenObjects
from pyro import poutine
from pyro.poutine.util import prune_subsample_sites

import argparse
import datetime
import math
import subprocess
import pickle
from functools import lru_cache
import time

import torch
from torch.distributions import constraints
from torch.distributions import transform_to

import pyro
import pyro.distributions as dist
import pyro.contrib.gp as gp
from pyro.contrib.util import rmv

import pyro.distributions.transforms as T


from pyro.util import torch_isnan, torch_isinf
def is_bad(a):
    return torch_isnan(a) or torch_isinf(a)

##############################################################################################
################################# Alternative Hidden Model ###################################
##############################################################################################
from oed.primitives import observation_sample, latent_sample, compute_design
import pandas as pd
class HiddenObjects2(nn.Module):
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
######################################### Flows ##############################################
##############################################################################################
from neural.modules import LazyDelta
class LazyNN(nn.Module):
    def __init__(self, design_dim):
        super().__init__()
        self.register_buffer("prototype", torch.zeros(design_dim))

    def lazy(self, x):
        def delayed_function():
            return self.forward(x)

        lazy_delta = LazyDelta(
            delayed_function, self.prototype, event_dim=self.prototype.dim()
        )
        return lazy_delta

class RealNVP(LazyNN):##DesignBaseline ##nn.Module
    def __init__(self, dim_input, num_blocks=5, dim_hidden=256,device = 'cuda'): #cpu
        super().__init__(dim_input)#, RealNVP, self
        
        self.dim_input = dim_input
        self.num_blocks = num_blocks
        self.dim_hidden = dim_hidden#dim_input#

        self.scale_net = nn.ModuleList([self._scale_block() for _ in range(num_blocks)])
        self.translation_net = nn.ModuleList([self._translation_block() for _ in range(num_blocks)])
        mask = torch.ones(dim_input).to(device)
        mask[int(dim_input / 2):] = 0
        mask.requires_grad = False
        self.mask = mask

    def _scale_block(self):
        return nn.Sequential(
            nn.Linear(self.dim_input, self.dim_hidden),#
            nn.ReLU(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.ReLU(),
            nn.Linear(self.dim_hidden, self.dim_input),#
            nn.Tanh()
        )

    def _translation_block(self):
        return nn.Sequential(
            nn.Linear(self.dim_input, self.dim_hidden),# 
            nn.ReLU(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.ReLU(),
            nn.Linear(self.dim_hidden, self.dim_input)# 
        )

    def forward(self, x):
        log_det_J = torch.zeros(x.size(0), device=x.device)

        for i in range(self.num_blocks):
            if i%2==0:
                maski = 1- self.mask
            else:
                maski = self.mask

            s = maski * self.scale_net[i](x * (1 - maski))
            s = torch.tanh(s)#torch.tanh()
            t = maski * self.translation_net[i](x * (1 - maski))

            x = x*torch.exp(s) + t

            log_det_J += torch.sum(s, dim=1)

        return x, log_det_J

    def reverse(self, z):
        # log_det_J = torch.zeros(z.size(0), device=z.device)
        for i in reversed(range(self.num_blocks)):
            if i%2==0:
                maski = 1- self.mask
            else:
                maski = self.mask

            s = maski * self.scale_net[i](z * (1 - maski))
            s = torch.tanh(s)
            t = maski * self.translation_net[i](z * (1 - maski))
            
            z = (z - t) * torch.exp(-s)

            # log_det_J += torch.sum(-s, dim=1)
        return z

    def sample(self, num_samples=1):
        z = torch.randn((num_samples, self.dim_input), device=self.device)
        samples = self.reverse(z)
        return samples


class IdentityTransform(nn.Module):
    def __init__(self):
        super(IdentityTransform, self).__init__()

    def forward(self, x):
        log_det_J = torch.zeros(x.size(0), device=x.device)
        return x, log_det_J

    def reverse(self, z):
        return z
    
class SplineFlow(LazyNN):##DesignBaseline ##nn.Module
    def __init__(self, dim_input, count_bins=8, bounds=3,device = 'cuda'): #cpu
        super().__init__(dim_input)#, RealNVP, self
        
        self.dim_input = dim_input
        self.countbins = count_bins
        self.bounds = bounds
        
        if dim_input == 1:
            self.spline_transform = T.Spline(dim_input, count_bins=count_bins, bound=bounds)
        else:     # spline_coupling
            # spl1 = spline_autoregressive1(dim_input,hidden_dims=[dim_input * 4+1, dim_input * 4+1], count_bins=count_bins, bound=bounds, order='linear')
            # per1 = T.permute(dim_input)#T.Permute(torch.arange(dim_input, dtype=torch.long).flip(0))
            # spl2 =spline_autoregressive1(dim_input,hidden_dims=[dim_input * 4+1, dim_input * 4+1], count_bins=count_bins, bound=bounds, order='linear')
            # # per2 = T.Permute(torch.arange(dim_input, dtype=torch.long).flip(0)),per2
            # self.spline_transform = T.ComposeTransform([spl1,per1,spl2],cache_size=0)
            self.spline_transform = spline_autoregressive1(dim_input,hidden_dims=[64, 32], count_bins=count_bins, bound=bounds, order='quadratic')
        #y, lofdet = spline_transform.spline_op(-2*torch.ones(2))
        #z = spline_transform.inv(y)

    def forward(self, x):
        z = self.spline_transform(x)
        logDet = self.spline_transform.log_abs_det_jacobian(x, z)
        return z, logDet

    def reverse(self, z):
        x = self.spline_transform.inv(z)
        # logDet = self.spline_transform._cache_log_detJ
        return x

from pyro.nn import AutoRegressiveNN
def spline_autoregressive1(input_dim, hidden_dims=None, count_bins=8, bound=3.0, order='linear'):
    r"""
    A helper function to create an
    :class:`~pyro.distributions.transforms.SplineAutoregressive` object that takes
    care of constructing an autoregressive network with the correct input/output
    dimensions.

    :param input_dim: Dimension of input variable
    :type input_dim: int
    :param hidden_dims: The desired hidden dimensions of the autoregressive network.
        Defaults to using [3*input_dim + 1]
    :type hidden_dims: list[int]
    :param count_bins: The number of segments comprising the spline.
    :type count_bins: int
    :param bound: The quantity :math:`K` determining the bounding box,
        :math:`[-K,K]\times[-K,K]`, of the spline.
    :type bound: float
    :param order: One of ['linear', 'quadratic'] specifying the order of the spline.
    :type order: string

    """

    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]
    # T.SplineCoupling
    if order=='quadratic':
        param_dims = [count_bins, count_bins, count_bins - 1]
    else:
        param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
    arn = AutoRegressiveNN(input_dim, hidden_dims, param_dims=param_dims)
    return T.SplineAutoregressive(input_dim, arn, count_bins=count_bins, bound=bound, order=order)

##############################################################################################
##############################################################################################


##############################################################################################
###################################### Marg + Post MM ########################################
##############################################################################################
def cov(X):
    D = X.shape[0]
    mean = torch.mean(X, dim=0)
    X = X - mean
    return 1/(D-1) * X.transpose(-1, -2) @ X


class VariationalMutualInformationOptimizer(object):
    def __init__(
        self, model, batch_size, data_source=None
    ):
        self.model = model
        self.batch_size = batch_size
        self.data_source = data_source

    def _vectorized(self, fn, *shape, name="vectorization_plate"):
        """
        Wraps a callable inside an outermost :class:`~pyro.plate` to parallelize
        MI computation over `num_particles`, and to broadcast batch shapes of
        sample site functions in accordance with the `~pyro.plate` contexts
        within which they are embedded.
        :param fn: arbitrary callable containing Pyro primitives.
        :return: wrapped callable.
        """

        def wrapped_fn(*args, **kwargs):
            with pyro.plate_stack(name, shape):
                return fn(*args, **kwargs)

        return wrapped_fn

    def get_primary_rollout(self, args, kwargs, graph_type="flat", detach=False):
        """
        sample data: batch_size number of examples -> return trace
        """
        if self.data_source is None:
            model_v = self._vectorized(
                self.model, self.batch_size, name="outer_vectorization"
            )
        else:
            data = next(self.data_source)
            model = pyro.condition(
                self._vectorized(model, self.batch_size, name="outer_vectorization"),
                data=data,
            )

        trace = poutine.trace(model_v, graph_type=graph_type).get_trace(*args, **kwargs)
        if detach:
            # what does the detach do?
            trace.detach_()
        trace = prune_subsample_sites(trace)
        return trace

    def _get_data(self, args, kwargs, graph_type="flat", detach=False):
        # esample a trace and xtract the relevant data from it
        trace = self.get_primary_rollout(args, kwargs, graph_type, detach)
        designs = [
            node["value"]
            for node in trace.nodes.values()
            if node.get("subtype") == "design_sample"
        ]
        observations = [
            node["value"]
            for node in trace.nodes.values()
            if node.get("subtype") == "observation_sample"
        ]
        latents = [
            node["value"]
            for node in trace.nodes.values()
            if node.get("subtype") == "latent_sample"
        ]
        latents = torch.cat(latents, axis=-1)
        return (latents, *zip(designs, observations))


# class MomentMatchMarginalPosterior(VariationalMutualInformationOptimizer):
#     def __init__(self, model, batch_size, **kwargs):
#         super().__init__(
#             model=model, batch_size=batch_size
#         )
#         self.mu = 0
#         self.Sigma = 0
#         self.hX = 0
#         self.hX_Y = 0      

#     def differentiable_loss(self, *args, **kwargs):
#         # sample from design
#         latents, *history = self._get_data(args, kwargs)
#         # print(history[0][0][0]) ### Check Design Value
#         # set constants
#         pi_const = 2*torch.acos(torch.zeros(1))
#         e_const = torch.exp(torch.tensor([1]))
#         dim_lat = latents.flatten(-2).shape[1]#latents.shape[1]#
#         dim_obs = history[0][1].shape[1]
        
#         #compute loss
#         data = torch.cat([latents.flatten(-2),history[0][1]],axis=1)#torch.cat([latents,history[0][1]],axis=1)#
        
#         Sigma = cov(data)
#         self.hX = .5*torch.log(torch.linalg.det(Sigma[:dim_lat,:dim_lat]))+(dim_lat/2)*(torch.log(2*pi_const*e_const))
#         hY = .5*torch.log(torch.linalg.det(Sigma[dim_lat:,dim_lat:]))+(dim_obs/2)*(torch.log(2*pi_const*e_const))
#         hXY = .5*torch.log(torch.linalg.det(Sigma))+((dim_lat+dim_obs)/2)*(torch.log(2*pi_const*e_const))
#         self.hX_Y = hXY-hY
        
#         # save optimal parameters for decision
#         self.mu = torch.mean(data,axis=0)
#         self.Sigma = Sigma
#         # print(self.hX+self.hX_Y)
#         return (self.hX+self.hX_Y)

#     def loss(self, *args, **kwargs):
#         """
#         :returns: returns an estimate of the mutual information
#         :rtype: float
#         Evaluates the MI lower bound using the BA lower bound == -EIG
#         """
#         return self.hX-self.hX_Y

class MomentMatchMarginalPosterior(VariationalMutualInformationOptimizer):
    def __init__(self, model, batch_size, flow_x, flow_y,device, **kwargs):
        super().__init__(
            model=model, batch_size=batch_size
        )
        self.mu = 0
        self.Sigma = 0
        self.hX = 0
        self.hX_Y = 0
        self.fX = flow_x
        self.gY = flow_y
        self.pi_const = 2*torch.acos(torch.zeros(1)).to(device)
        self.e_const = torch.exp(torch.tensor([1])).to(device)
        # self.I = torch.eye()
        # if hidden == None:
        #     self.fX = IdentityTransform()
        #     self.gY = IdentityTransform()
        # else:
        #     self.fX = RealNVP(dim_x, num_blocks=2, dim_hidden=hidden//2)
        #     self.gY = RealNVP(dim_y, num_blocks=2, dim_hidden=hidden//2)

    def differentiable_loss(self, *args, **kwargs):
        # sample from design
        latents, *history = self._get_data(args, kwargs)
        # theta, xi_designs, y_outcomes = self.model
        
        
        dim_lat = latents.shape[1]#latents.flatten(-2).shape[1]#
        dim_obs = history[0][1].shape[1]
        
        if hasattr(self.fX, "parameters"):
            #! this is required for the pyro optimizer
            pyro.module("flow_x_net", self.fX)
        if hasattr(self.gY, "parameters"):
            #! this is required for the pyro optimizer
            pyro.module("flow_y_net", self.gY)
        
        mufX, logDetJfX = self.fX.forward(latents)#self.fX.forward(latents.flatten(-2))#
        mugY, logDetJgY = self.gY.forward(history[0][1])#self.gY(history[0][1])#
        
        #compute loss
        data = torch.cat([mufX,mugY],axis=1)#torch.cat([latents.flatten(-2),history[0][1]],axis=1)#torch.cat([latents,history[0][1]],axis=1)#
        
        Sigma = cov(data)+1e-4*torch.eye(dim_lat+dim_obs).to(latents.device)
        self.hX = .5*torch.log(torch.linalg.det(Sigma[:dim_lat,:dim_lat]))+(dim_lat/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX)
        hY = .5*torch.log(torch.linalg.det(Sigma[dim_lat:,dim_lat:]))+(dim_obs/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJgY)
        hXY = .5*torch.log(torch.linalg.det(Sigma))+((dim_lat+dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX)-torch.mean(logDetJgY)
        self.hX_Y = hXY-hY
        
        # save optimal parameters for decision
        self.mu = torch.mean(data,axis=0)
        self.Sigma = Sigma
        # print(self.hX+self.hX_Y)
        return (self.hX+self.hX_Y)

    def loss(self, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using the BA lower bound == -EIG
        """
        return self.hX-self.hX_Y

##############################################################################################
##############################################################################################


def optimise_design(
    posterior_loc,
    posterior_cov,
    flow_theta,
    flow_obs,
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
    # new_mean = posterior_loc.reshape(num_sources, p)
    # new_covmat = torch.cat(
    #     [
    #         posterior_cov[:p,:p].unsqueeze(0),posterior_cov[p:,p:].unsqueeze(0)
    #     ]
    # )
    new_mean = posterior_loc
    new_covmat = posterior_cov
    ho_model = HiddenObjects2(
        design_net=design_net,
        # Normal family -- new prior is stil MVN but with different params
        theta_loc=new_mean,
        theta_covmat=new_covmat,
        flow_theta = flow_theta,
        T=1,
        p=p,
        K=num_sources,
        noise_scale=noise_scale * torch.ones(1, device=device),
    )
    hidden = 128#None#
    if hidden == None:
        fX = IdentityTransform()
        gY = IdentityTransform()
    else:
        dim_x = num_sources*p
        dim_y = 1
        fX = SplineFlow(dim_x, count_bins=32, bounds=4, device=device).to(device)
        gY = SplineFlow(dim_y, count_bins=32, bounds=4, device=device).to(device)
        # fX = RealNVP(dim_x, num_blocks=3, dim_hidden=hidden//2,device=device).to(device)
        # gY = RealNVP(dim_y, num_blocks=3, dim_hidden=hidden//2,device=device).to(device)
    
    ### Set-up loss ###
    # mi_loss_instance = MomentMatchMarginalPosterior(
    #     model=ho_model.model,
    #     batch_size=batch_size,
    #     # prior_entropy=ho_model.theta_prior.entropy(),
    # )
    mi_loss_instance = MomentMatchMarginalPosterior(
        model=ho_model.model,
        batch_size=batch_size,
        flow_x=fX,
        flow_y=gY,
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
    noise_scale,
    num_sources,
    p,
    batch_size,
    num_steps,
    lr,
    annealing_scheme,
):
    pyro.clear_param_store()

    # theta_loc = torch.zeros((num_sources, p), device=device)
    # theta_covmat = torch.eye(p, device=device)
    theta_loc = torch.zeros((1, num_sources*p), device=device)
    theta_covmat = torch.eye(num_sources*p, device=device)
    flow_theta = IdentityTransform()
    flow_obs = IdentityTransform()
    prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)

    # sample true param
    true_theta = prior.sample(torch.Size([1]))

    designs_so_far = []
    observations_so_far = []

    # Set posterior equal to the prior
    posterior_loc = theta_loc.reshape(-1)  # check if needs to be reshaped.
    posterior_cov = torch.eye(p * num_sources, device=device)

    for t in range(0, T):
        print(f"Step {t + 1}/{T} of Run {run + 1}")
        pyro.clear_param_store()
        ho_model, mi_loss_instance = optimise_design(
            posterior_loc,
            posterior_cov,
            flow_theta,
            flow_obs,
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
            design, observation = ho_model.forward(theta=trans_true_theta)#true_theta
            mux = mi_loss_instance.mu[:p * num_sources].detach()
            muy = mi_loss_instance.mu[p * num_sources:].detach()
            Sigmaxx = mi_loss_instance.Sigma[:p * num_sources,:p * num_sources].detach()
            Sigmaxy = mi_loss_instance.Sigma[:p * num_sources,p * num_sources:].detach()
            Sigmayy = mi_loss_instance.Sigma[p * num_sources:,p * num_sources:].detach()
            obs, _ = mi_loss_instance.gY.forward(observation[0])
            # obs = observation[0]
            posterior_loc = (mux + torch.matmul(Sigmaxy,torch.linalg.solve(Sigmayy,(obs-muy))).flatten())
            print(true_theta)#flow_theta.reverse
            print(posterior_loc)
            print(mi_loss_instance.fX.reverse(posterior_loc))
            test2, hold = mi_loss_instance.fX.forward(posterior_loc.reshape(1,4))
            print(test2)
            posterior_cov = Sigmaxx-torch.matmul(Sigmaxy,torch.linalg.solve(Sigmayy,Sigmaxy.T))
            flow_theta = mi_loss_instance.fX
            flow_obs = mi_loss_instance.gY
        ###################################################################################################
        
        designs_so_far.append(design[0])
        observations_so_far.append(observation[0])#obs

    print(f"Fitted posterior: mean = {posterior_loc}, cov = {posterior_cov}")
    print("True theta = ", true_theta.reshape(-1))

    data_dict = {}
    for i, xi in enumerate(designs_so_far):
        data_dict[f"xi{i + 1}"] = xi.cpu()
    for i, y in enumerate(observations_so_far):
        data_dict[f"y{i + 1}"] = y.cpu()
    data_dict["theta"] = true_theta.reshape((num_sources, p)).cpu()

    return data_dict


def main(
    seed,
    mlflow_experiment_name,
    num_histories,
    device,
    T,
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
    results_vi = {"loop": [], "seed": seed, "meta": meta}
    for i in range(num_histories):
        results = main_loop(
            run=i,
            mlflow_run_id=mlflow.active_run().info.run_id,
            device=device,
            T=T,
            noise_scale=noise_scale,
            num_sources=num_sources,
            p=p,
            batch_size=batch_size,
            num_steps=num_steps,
            lr=lr,
            annealing_scheme=annealing_scheme,
        )
        results_vi["loop"].append(results)

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
    # --------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VI baseline: Location finding with MM M+P"
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--physical-dim", default=2, type=int)
    parser.add_argument(
        "--num-histories", help="Number of histories/rollouts", default=1, type=int#128
    )
    parser.add_argument("--num-experiments", default=10, type=int)  # == T
    parser.add_argument("--batch-size", default=1024, type=int)#512
    parser.add_argument("--device", default="cpu", type=str)#"cuda"
    parser.add_argument(
        "--mlflow-experiment-name", default="locfin_mm_variational", type=str
    )
    parser.add_argument("--lr", default=0.005, type=float)#0.005
    parser.add_argument("--num-steps", default=2000, type=int)#5000
    
    args = parser.parse_args()

    main(
        seed=args.seed,
        mlflow_experiment_name=args.mlflow_experiment_name,
        num_histories=args.num_histories,
        device=args.device,
        T=args.num_experiments,
        p=args.physical_dim,
        num_sources=2,
        noise_scale=0.5,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
    )
    
    
########################################################################
#################### Maybe for later use ###############################
########################################################################
# def gp_opt_w_history(loss_fn, num_steps, time_budget, num_parallel, num_acquisition, lengthscale,model, design_dim):

#     if time_budget is not None:
#         num_steps = 100000000000
    
#     est_loss_history = []
#     xi_history = []
#     t = time.time()
#     wall_times = []
#     run_times = []
#     X = .01 + 4.99 * torch.rand((num_parallel, num_acquisition, design_dim))
#     with torch.no_grad():
#             model.design_net.designs[0][0]=X.flatten()
#     y = loss_fn(X)

#     # GPBO
#     y = y.detach().clone()
#     kernel = gp.kernels.Matern52(input_dim=1, lengthscale=torch.tensor(lengthscale),
#                                  variance=torch.tensor(1.))
#     constraint = torch.distributions.constraints.interval(1e-2, 5.)
#     noise = torch.tensor(0.5).pow(2)

#     def gp_conditional(Lff, Xnew, X, y):
#         KXXnew = kernel(X[0], Xnew[0])#########################################################
#         LiK = torch.triangular_solve(KXXnew, Lff, upper=False)[0]
#         Liy = torch.triangular_solve(y.unsqueeze(-1), Lff, upper=False)[0]
#         mean = rmv(LiK.transpose(-1, -2), Liy.squeeze(-1))
#         KXnewXnew = kernel(Xnew[0])
#         var = (KXnewXnew - LiK.transpose(-1, -2).matmul(LiK)).diagonal(dim1=-2, dim2=-1)
#         return mean, var

#     def acquire(X, y, sigma, nacq):
#         Kff = kernel(X[0])##################################
#         Kff += noise * torch.eye(Kff.shape[-1])
#         Lff = Kff.cholesky(upper=False)
#         Xinit = .01 + 4.99 * torch.rand((num_parallel, nacq, design_dim))
#         unconstrained_Xnew = transform_to(constraint).inv(Xinit).detach().clone().requires_grad_(True)
#         minimizer = torch.optim.LBFGS([unconstrained_Xnew], max_eval=20)

#         def gp_ucb1():
#             minimizer.zero_grad()
#             Xnew = transform_to(constraint)(unconstrained_Xnew)
#             mean, var = gp_conditional(Lff, Xnew, X, y)
#             ucb = -(mean + sigma * var.clamp(min=0.).sqrt())
#             ucb[is_bad(ucb)] = 0.
#             loss = ucb.sum()
#             torch.autograd.backward(unconstrained_Xnew,
#                                     torch.autograd.grad(loss, unconstrained_Xnew))
#             return loss

#         minimizer.step(gp_ucb1)
#         X_acquire = transform_to(constraint)(unconstrained_Xnew).detach().clone()
#         y_expected, _ = gp_conditional(Lff, X_acquire, X, y)

#         return X_acquire, y_expected

#     def find_gp_max(X, y, n_tries=100):
#         X_star = torch.zeros(num_parallel, 1, design_dim)
#         y_star = torch.zeros(num_parallel, 1)
#         for j in range(n_tries):  # Cannot parallelize this because sometimes LBFGS goes bad across a whole batch
#             X_star_new, y_star_new = acquire(X, y, 0, 1)
#             y_star_new[is_bad(y_star_new)] = 0.
#             mask = y_star_new > y_star
#             y_star[mask, ...] = y_star_new[mask, ...]
#             X_star[mask, ...] = X_star_new[mask, ...]

#         return X_star.squeeze(), y_star.squeeze()

#     for i in range(num_steps):
#         X_acquire, _ = acquire(X, y, 2, num_acquisition)
#         y_acquire = loss_fn(X_acquire.flatten()).detach().clone()
#         X = torch.cat([X, X_acquire], dim=-2)
#         y = torch.cat([y, y_acquire], dim=-1)
#         run_times.append(time.time() - t)

#         if time_budget and time.time() - t > time_budget:
#             break

#     final_time = time.time() - t

#     for i in range(1, len(run_times)+1):

#         if i % 10 == 0:
#             s = num_acquisition * i
#             X_star, y_star = find_gp_max(X[:, :s, :], y[:s])#y[:, :s]
#             print(X_star)
#             est_loss_history.append(y_star)
#             xi_history.append(X_star.detach().clone())
#             wall_times.append(run_times[i-1])

#     # Record the final GP max
#     X_star, y_star = find_gp_max(X, y)
#     xi_history.append(X_star.detach().clone())
#     wall_times.append(final_time)

#     est_loss_history = torch.stack(est_loss_history)
#     xi_history = torch.stack(xi_history)
#     wall_times = torch.tensor(wall_times)

#     return xi_history, est_loss_history, wall_times


##############################################################################################
########################### Optimization Functions ###########################################
##############################################################################################
# def get_git_revision_hash():
#     return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

# def opt_eig_loss_w_history(design, loss_fn, num_samples, num_steps, optim, time_budget):
#     if time_budget is not None:
#         num_steps = 100000000000
#     params = None
#     est_loss_history = []
#     xi_history = []
#     t = time.time()
#     wall_times = []
#     for step in range(num_steps):
#         if params is not None:
#             pyro.infer.util.zero_grads(params)
#         with poutine.trace(param_only=True) as param_capture:
#             agg_loss, loss = loss_fn(design, num_samples, evaluation=True)
#         params = set(site["value"].unconstrained()
#                      for site in param_capture.trace.nodes.values())
#         if torch.isnan(agg_loss):
#             raise ArithmeticError("Encountered NaN loss in opt_eig_ape_loss")
#         agg_loss.backward(retain_graph=True)
#         if step % 200 == 0:
#             est_loss_history.append(loss)
#             wall_times.append(time.time() - t)
#             xi_history.append(pyro.param('xi').detach().clone())
#         optim(params)
#         optim.step()
#         print(pyro.param("xi"))
#         if time.time() - t > time_budget:
#             break

#     xi_history.append(pyro.param('xi').detach().clone())
#     wall_times.append(time.time() - t)

#     est_loss_history = torch.stack(est_loss_history)
#     xi_history = torch.stack(xi_history)
#     wall_times = torch.tensor(wall_times)

#     return xi_history, est_loss_history, wall_times
##############################################################################################
##############################################################################################






########################################################################
########################################################################