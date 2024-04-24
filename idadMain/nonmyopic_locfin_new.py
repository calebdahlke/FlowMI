######################### JOINT VARIATIONAL ESTIMATOR TWO STEP SAMPLE (NON-POLICY) ##########################
import os
import pickle
import argparse
import time
import torch
import pyro
from tqdm import trange
import mlflow
import copy

from neural.baselines import DesignBaseline
from torch import nn

from oed.primitives import observation_sample, latent_sample, compute_design
import pandas as pd
import torch
from torch import nn
import pyro
import pyro.distributions as dist
from flow_estimator_pyro import IdentityTransform

from experiment_tools.pyro_tools import auto_seed
from joblib import Parallel, delayed
from simulations import HiddenObjectsVar
from flow_estimator_pyro import MomentMatchMarginalPosterior,SplineFlow, IdentityTransform,InitFlowToIdentity, VariationalMutualInformationOptimizer, cov
from eval_sPCE_from_source import eval_from_source
from pyro.infer.util import torch_item
from oed.design import OED

class OEDMixed:
    def __init__(self, optim_design, optim_flow, loss, **kwargs):
        self.optim_design = optim_design
        self.optim_flow = optim_flow
        self.loss = loss
        super().__init__(**kwargs)
        
    def evaluate_loss(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float
        Evaluate the loss function.
        """
        with torch.no_grad():
            loss = self.loss.loss(*args, **kwargs)
            return torch_item(loss)

    def step(self, *args, **kwargs):
        # Define a closure for LBFGS optimizer
        def closure():
            self.optim_design.zero_grad()
            neg_loss = -self.loss.differentiable_loss(*args, **kwargs)
            # loss.backward()
            return neg_loss
        
        # Run LBFGS optimization step for design_net parameters
        neg_loss = self.optim_design.step(closure)
        
        if self.optim_flow:
            with pyro.poutine.trace(param_only=True) as param_capture:
                loss = self.loss.differentiable_loss(*args, **kwargs)
                loss.backward()
            
            params = set(
                site["value"].unconstrained() for site in param_capture.trace.nodes.values()
            )

            self.optim_flow(params)
            
            pyro.infer.util.zero_grads(params)
        else:
            loss = -neg_loss
        return torch_item(loss)

class MomentMatchMarginalPosteriorNonMyopic(VariationalMutualInformationOptimizer):
    def __init__(self, model,future_model, batch_size, flow_x, flow_y, flow_x_future, flow_y_future,sample_future,device, **kwargs):
        super().__init__(
            model=model, batch_size=batch_size
        )
        self.mu = 0
        self.Sigma = 0
        self.dim_lat = 0
        self.dim_obs = 0
        self.future_model = future_model
        self.fX = flow_x
        self.gY = flow_y
        self.fX_future = flow_x_future
        self.gY_future = flow_y_future
        self.sample_future = sample_future
        self.hXY = 0
        self.hXY_future = 0
        self.pi_const = 2*torch.acos(torch.zeros(1, device=device))#.to(device)
        self.e_const = torch.exp(torch.tensor([1], device=device))#.to(device)
        # self.zero_vec = torch.zeros(4).to(device)

    def differentiable_loss(self, *args, **kwargs):
        # sample from design
        latents, *history = self._get_data(args, kwargs)

        self.dim_lat = latents.shape[1]
        self.dim_obs = history[0][1].shape[1]

        fX_sample, logDetJfX = self.fX.forward(latents)
        gY_sample, logDetJgY = self.gY.forward(history[0][1][:,:1])

        data = torch.cat([fX_sample,gY_sample],axis=1)
        
        Sigma = cov(data)+1e-5*torch.eye(self.dim_lat+self.dim_obs, device=latents.device)
        self.cond_num = torch.linalg.cond(Sigma)
        
        self.hXY = .5*torch.log(torch.linalg.det(Sigma))+(self.dim_lat+self.dim_obs/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX)-torch.mean(logDetJgY)
        hY = .5*torch.log(torch.linalg.det(Sigma[:self.dim_lat,:self.dim_lat]))+(self.dim_obs/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJgY)
        hX = .5*torch.log(torch.linalg.det(Sigma[:self.dim_lat,:self.dim_lat]))+(self.dim_lat/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX)
        Sigmayyyx = torch.linalg.solve(Sigma[self.dim_lat:,self.dim_lat:],Sigma[self.dim_lat:,:self.dim_lat])
        Sigma_post = Sigma[:self.dim_lat,:self.dim_lat] - torch.matmul(Sigma[:self.dim_lat,self.dim_lat:],Sigmayyyx)
        mux = torch.mean(fX_sample,axis=0)
        muy = torch.mean(gY_sample,axis=0)
        # self.hXY_future = 0
        # hY_future = 0
        # hX_future = 0
        # if self.sample_future:
        #     for i in range(latents.shape[0]):
        #         mu_post = mux-torch.matmul(Sigmayyyx.T,gY_sample[i]-muy)
        #         post_dist = dist.MultivariateNormal(mu_post, Sigma_post)
        #         latents_future, obs_future = self.future_model(post_dist, self.fX,16)
        #         fX_future_sample, logDetJfX_future = self.fX_future.forward(latents_future)
        #         gY_future_sample, logDetJgY_future = self.gY_future.forward(obs_future[0])
        #         data_future = torch.cat([fX_future_sample,gY_future_sample],axis=1)
        #         Sigma_future = cov(data_future)+1e-5*torch.eye(self.dim_lat+self.dim_obs, device=latents.device)
        #         self.hXY_future += .5*torch.log(torch.linalg.det(Sigma_future))+((self.dim_lat+self.dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX_future)-torch.mean(logDetJgY_future)
        #         hY_future += .5*torch.log(torch.linalg.det(Sigma_future[self.dim_lat:,self.dim_lat:]))+((self.dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJgY_future)
        #         hX_future += .5*torch.log(torch.linalg.det(Sigma_future[:self.dim_lat,:self.dim_lat]))+((self.dim_lat)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX_future)
        
        # if self.sample_future:
        #     Sigma_hold = 1e-5*torch.eye(self.dim_lat+self.dim_obs, device=latents.device)
        #     for i in range(latents.shape[0]):
        #         mu_post = mux-torch.matmul(Sigmayyyx.T,gY_sample[i]-muy)
        #         post_dist = dist.MultivariateNormal(mu_post, Sigma_post)
        #         latents_future, obs_future = self.future_model(post_dist, self.fX,16)
        #         fX_future_sample, logDetJfX_future = self.fX_future.forward(latents_future)
        #         gY_future_sample, logDetJgY_future = self.gY_future.forward(obs_future[0])
        #         data_future = torch.cat([fX_future_sample,gY_future_sample],axis=1)
        #         Sigma_hold += cov(data_future)#+1e-5*torch.eye(self.dim_lat+self.dim_obs, device=latents.device)
        #     Sigma_future = (1/latents.shape[0])*Sigma_hold
        #     self.hXY_future = .5*torch.log(torch.linalg.det(Sigma_future))+((self.dim_lat+self.dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX_future)-torch.mean(logDetJgY_future)
        #     hY_future = .5*torch.log(torch.linalg.det(Sigma_future[self.dim_lat:,self.dim_lat:]))+((self.dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJgY_future)
        #     hX_future = .5*torch.log(torch.linalg.det(Sigma_future[:self.dim_lat,:self.dim_lat]))+((self.dim_lat)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX_future)
        self.hXY_future = 0
        hY_future = 0
        hX_future = 0
        Sigma_hold = 0
        if self.sample_future:
            mu_post = ((mux-torch.matmul(Sigmayyyx.T,(gY_sample-muy).T).T).T).repeat(1,64).T
            post_dist = dist.MultivariateNormal(0*mux, Sigma_post)
            latents_future, obs_future = self.future_model(post_dist,mu_post, self.fX,len(gY_sample)*64)
            fX_future_sample, logDetJfX_future = self.fX_future.forward(latents_future)
            gY_future_sample, logDetJgY_future = self.gY_future.forward(obs_future[0])
            data_future = torch.cat([fX_future_sample,gY_future_sample],axis=1)
            for k in range(len(gY_sample)):
                Sigma_hold += torch.linalg.slogdet(cov(data_future[k::len(gY_sample)]))[1]
            Sigma_future = Sigma_hold/len(gY_sample)
            self.hXY_future = .5*Sigma_future+((self.dim_lat+self.dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX_future)-torch.mean(logDetJgY_future)
            hY_future = .5*Sigma_future+((self.dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJgY_future)
            hX_future = .5*Sigma_future+((self.dim_lat)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX_future)
            # self.hXY_future = .5*torch.log(torch.linalg.det(Sigma_future))+((self.dim_lat+self.dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX_future)-torch.mean(logDetJgY_future)
            # hY_future = .5*torch.log(torch.linalg.det(Sigma_future[self.dim_lat:,self.dim_lat:]))+((self.dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJgY_future)
            # hX_future = .5*torch.log(torch.linalg.det(Sigma_future[:self.dim_lat,:self.dim_lat]))+((self.dim_lat)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX_future)
        # save optimal parameters for decision
        self.mu = torch.mean(data,axis=0)
        self.Sigma = Sigma
        return self.hXY-hX-hY+(self.hXY_future-hX_future-hY_future)#(1/latents.shape[0])*(self.hXY_future-hX_future-hY_future)

    def loss(self, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using the BA lower bound == -EIG
        """
        loss_to_constant = torch_item(self.differentiable_loss(*args, **kwargs))
        return loss_to_constant

class BatchDesignBaselineNonMyopic(DesignBaseline):
    """
    Batch design baseline: learns T constants.

    - If trained with InfoNCE bound, this is the SG-BOED static baseline.
    - If trained with the NWJ bound, this is the MINEBED static baselines.
    """

    def __init__(
        self,
        T,
        design_dim,
        output_activation=nn.Identity(),
        design_init=torch.distributions.Normal(0, 0.5),#torch.zeros((1,1)),
    ):
        super().__init__(design_dim)
        self.designs = nn.ParameterList(
            [
                nn.Parameter(design_init.sample())
                for i in range(T)
            ]
        )
        self.output_activation = output_activation

    def forward(self, *design_obs_pairs):
        j = len(design_obs_pairs)
        return self.output_activation(self.designs[j])
    
class HiddenObjectsNonMyopic(nn.Module):
    """Location finding example"""

    def __init__(
        self,
        design_net,
        design_net_future,
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
        self.design_net_future = design_net_future
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
        self.prev_design = torch.zeros(self.design_net.designs[0].size(),device=self.design_net.designs[0].device)

    def forward_map(self, xi, theta):
        """Defines the forward map for the hidden object example
        y = G(xi, theta) + Noise.
        """
        mean_y = torch.zeros(xi.size()[:2],device=xi.device)
        # two norm squared
        for i in range(xi.size()[1]):
            sq_two_norm = (xi[:,i,:].unsqueeze(1) - theta).pow(2).sum(axis=-1)
            # add a small number before taking inverse (determines max signal)
            sq_two_norm_inverse = (self.max_signal + sq_two_norm).pow(-1)
            # sum over the K sources, add base signal and take log.
            mean_y[:,i] = torch.log(self.base_signal + sq_two_norm_inverse.sum(-1))#, keepdim=True
        return mean_y

    def model(self):
        if hasattr(self.design_net, "parameters"):
            #! this is required for the pyro optimizer
            pyro.module("design_net", self.design_net)
        
        ########################################################################
        # Sample latent variables theta
        ########################################################################
        theta1 = latent_sample("theta", self.theta_prior)
        with torch.no_grad():
            theta = self.flow_theta.reverse(theta1)#.flatten(-1)
            if torch.any(theta.isnan()):
                theta[torch.unique(torch.where(torch.isnan(theta))[0])] = self.flow_theta.reverse(theta1[torch.unique(torch.where(torch.isnan(theta))[0])])
            if torch.any(theta.isnan()):
                print('NaN')
                # theta[torch.where(torch.isnan(theta))] = self.flow_theta.reverse(1.0001*theta1[torch.where(torch.isnan(theta))])
                theta[torch.unique(torch.where(torch.isnan(theta))[0])] = self.flow_theta.reverse(1.0001*theta1[torch.unique(torch.where(torch.isnan(theta))[0])])
        theta = theta.reshape((len(theta),self.K,self.p))
        if torch.any(theta.isnan()):
            print('NaN')
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
            if torch.any(xi.isnan()):
                print('NaN')
            self.prev_design = 1*xi[0]
            # xi = xi.reshape(xi.shape[0], int(xi.shape[2]/self.p), self.p)
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
    
    def _future_model(self, post,mu_post, post_flow,num_samples):
        theta = post.sample((1,num_samples))[0]
        # with torch.no_grad():
        theta += mu_post
        theta = post_flow.reverse(theta)
        theta = theta.reshape((len(theta),self.K,self.p))
        y_outcomes = []
        xi_designs = []

        # T-steps experiment
        for t in range(self.T):
            ####################################################################
            # Get a design xi; shape is [batch size x self.n x self.p]
            ####################################################################
            xi = compute_design(
                f"xi{t + 2}", self.design_net_future.lazy(*zip(xi_designs, y_outcomes))
            )
            xi = xi*torch.ones((len(theta),1,xi.shape[1]),device=xi.device)
            if torch.any(xi.isnan()):
                print('NaN')
            # self.prev_design = 1*xi[0]
            # xi = xi.reshape(xi.shape[0], int(xi.shape[2]/self.p), self.p)
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
        return theta.flatten(-2), y_outcomes

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

def optimise_design(
    posterior_loc,
    posterior_cov,
    flow_theta,
    flow_obs,
    sample_future,
    design_init,
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
    annealing_scheme,
):
    # design_init_dist = (
    #     torch.distributions.Normal(design_init, 0.01)
    #     if experiment_number == 0
    #     else torch.distributions.Normal(design_init, 1.0)
    # )
    design_init_dist = torch.distributions.Normal(design_init,.1)#0.01
    design_net = BatchDesignBaselineNonMyopic(
        T=1, design_dim=(1, p), design_init=design_init_dist
    ).to(device)
    
    design_net_future = BatchDesignBaselineNonMyopic(
        T=1, design_dim=(1, p), design_init=design_init_dist
    ).to(device)
    ho_model = HiddenObjectsNonMyopic(
        design_net=design_net,
        design_net_future=design_net_future,
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
            # flowx_loss = torch.tensor(torch.nan)
            # init_lr_x = .005
            # while torch.isnan(flowx_loss):
            #     fX = SplineFlow(dim_x,n_flows=1,hidden_dims=[32], count_bins=128, bounds=4,order = 'quadratic', device=device)
            #     fX, flowx_loss= InitFlowToIdentity(dim_x, fX, bounds=4,lr=init_lr_x,device=device)
            #     init_lr_x *= .5
            fX = SplineFlow(dim_x,n_flows=1,hidden_dims=[32], count_bins=128, bounds=4,order = 'quadratic', device=device)
        else:
            fX = copy.deepcopy(flow_theta)
        if flow_obs == None:
            # flowy_loss = torch.tensor(torch.nan)
            # init_lr_y = .005
            # while torch.isnan(flowy_loss):
            #     gY = SplineFlow(1, count_bins=128, bounds=5,order = 'quadratic', device=device)#SplineFlow(1, count_bins=256, bounds=3, device=device).to(device)
            #     gY, flowy_loss = InitFlowToIdentity(1, gY, bounds=5,lr=init_lr_y,device=device)
            #     init_lr_y *= .5
            gY = SplineFlow(1, count_bins=128, bounds=5,order = 'quadratic', device=device)#SplineFlow(1, count_bins=256, bounds=3, device=device).to(device)
        else:
            gY = copy.deepcopy(flow_obs)
        if sample_future:
            fX_future = SplineFlow(dim_x,n_flows=1,hidden_dims=[32], count_bins=128, bounds=4,order = 'quadratic', device=device)
            gY_future = SplineFlow(1, count_bins=128, bounds=5,order = 'quadratic', device=device)
        else:
            fX_future = IdentityTransform()
            gY_future = IdentityTransform()
            
    else:
        fX = IdentityTransform()
        gY = IdentityTransform()
        fX_future = IdentityTransform()
        gY_future = IdentityTransform()

    ### Set-up loss ###
    mi_loss_instance = MomentMatchMarginalPosteriorNonMyopic(
        model=ho_model.model,
        future_model=ho_model._future_model,
        batch_size=batch_size,
        flow_x=fX,
        flow_y=gY,
        flow_x_future=fX_future,
        flow_y_future=gY_future,
        prior_flow = flow_theta,
        sample_future=sample_future,
        device=device
    )
    lr_design = .005
    lr_flow = .005
    ### Set-up optimiser ###
    optimizer_design = torch.optim.Adam(list(ho_model.design_net.designs.parameters())+list(ho_model.design_net_future.designs.parameters()))#ho_model.design_net.designs)
    annealing_freq, factor = annealing_scheme
    scheduler_design = pyro.optim.ExponentialLR(
        {
            "optimizer": optimizer_design,
            "optim_args": {"lr": lr_design},
            "gamma" : factor,
            "verbose": False,
        }
    )
    
    if run_flow and train_flow:
        optimizer_flow = torch.optim.Adam(list(mi_loss_instance.fX.parameters())+list(mi_loss_instance.gY.parameters())+list(mi_loss_instance.fX_future.parameters())+list(mi_loss_instance.gY_future.parameters()))
        annealing_freq, factor = annealing_scheme
        scheduler_flow = pyro.optim.ExponentialLR(
            {
                "optimizer": optimizer_flow,
                "optim_args": {"lr": lr_flow},
                "gamma" : factor,
                "verbose": False,
            }
        )
    
    ### Optimise ###
    design_prev = 1*design_net.designs[0]
    print(design_prev)
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
            (mi_loss_instance.hXY+mi_loss_instance.hXY_future).backward()
            # Foster Bound Optimization
            # (mi_loss_instance.hX + mi_loss_instance.hXY - mi_loss_instance.hY).backward()
            # (mi_loss_instance.hX+mi_loss_instance.hY).backward()
            optimizer_flow.step()
            
        if i % 100 == 0:
            num_steps_range.set_description("Loss: {:.3f} ".format(torch_item(negMI)))
            loss_eval = mi_loss_instance.loss()#*args, **kwargs)
            loss_history.append(loss_eval)
            
        if i % 500 ==0 and not i == 0:
            with torch.no_grad():
                design_current = design_net.designs[0]
                design_diff = torch.max((design_current-design_prev).pow(2).sum(axis=1).pow(.5))
                # with torch.no_grad():
                #     hold, toss = mi_loss_instance.fX.forward(posterior_loc)
                #     print(hold)
                print(design_current)
                print(design_net_future.designs[0])
                # design_prev = 1*design_current
                # print(design_diff)
                if design_diff < 1e-3:
                    j+=1
                    if j>=2:
                        break
                else:
                    design_prev = 1*design_current
                    j=0
    return ho_model, mi_loss_instance, loss_history

def main_loop(
    run,
    path_to_extra_data,
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
    true_theta = prior.sample(torch.Size([1]))#torch.tensor([[[-0.3281,  0.2271, -0.0320,  0.9442]]], device=device)#torch.tensor([[[-1.3281,  1.2271, 1.0320,  -0.9442]]], device=device)#
    designs_so_far = []
    observations_so_far = []

    # Set posterior equal to the prior
    posterior_loc = theta_loc.reshape(-1)
    posterior_cov = torch.eye(p * num_sources, device=device)

    design_init = torch.zeros((1,p),device=device, dtype=torch.float32)
    sample_future = True
    for t in range(0, T):
        if t == T-1:
            sample_future = False
        t_start = time.time()
        print(f"Step {t + 1}/{T} of Run {run + 1}")
        pyro.clear_param_store()
        ho_model, mi_loss_instance, loss_history = optimise_design(
            posterior_loc,
            posterior_cov,
            flow_theta,
            flow_obs,
            sample_future,
            design_init,
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
        
        with torch.no_grad():
            if t>0:
                trans_true_theta,_ = flow_theta.forward(true_theta[0])
            else:
                trans_true_theta = true_theta
            design, observation = ho_model.forward(theta=trans_true_theta)

            mux = mi_loss_instance.mu[:p * num_sources].detach()
            muy = mi_loss_instance.mu[p * num_sources:p * num_sources+1].detach()
            Sigmaxx = mi_loss_instance.Sigma[:p * num_sources,:p * num_sources].detach()
            Sigmaxy = mi_loss_instance.Sigma[:p * num_sources,p * num_sources:p * num_sources+1].detach()
            Sigmayy = mi_loss_instance.Sigma[p * num_sources:p * num_sources+1,p * num_sources:p * num_sources+1].detach()
            
            obs, _ = mi_loss_instance.gY.forward(observation[0][0][0].unsqueeze(0).unsqueeze(0))
            posterior_loc = (mux + torch.matmul(Sigmaxy,torch.linalg.solve(Sigmayy,(obs-muy))).flatten())
            max_posterior = mi_loss_instance.fX.reverse(posterior_loc)
            posterior_cov = Sigmaxx-torch.matmul(Sigmaxy,torch.linalg.solve(Sigmayy,Sigmaxy.T))
            flow_theta = mi_loss_instance.fX
            flow_obs = mi_loss_instance.gY
            
            design_init = 0*ho_model.design_net_future.forward().detach()
            prev_obs = observation[0][0][0]
        t_end = time.time()
        run_time = t_end-t_start

        designs_so_far.append(design[0][0][0].unsqueeze(0).unsqueeze(0).detach().clone().cpu())
        observations_so_far.append(observation[0][0][0].unsqueeze(0).unsqueeze(0).cpu())
        
        # extra_data = {}
        # extra_data["mu"] = mi_loss_instance.mu.detach().clone().cpu().numpy()
        # extra_data["sigmas"] = mi_loss_instance.Sigma.detach().clone().cpu().numpy()
        # extra_data["flow_theta"] = copy.deepcopy(mi_loss_instance.fX).cpu()
        # extra_data["flow_obs"] = copy.deepcopy(mi_loss_instance.gY).cpu()
        # extra_data["posterior_loc"] = posterior_loc.cpu().numpy()
        # extra_data["posterior_cov"] = posterior_cov.cpu().numpy()
        # extra_data["total_time"] = run_time
        # extra_data["design"] = design[0].detach().clone().cpu()
        # extra_data["observations"] = observation[0].cpu()
        
        # path_to_run = path_to_extra_data + '/Run{}'.format(run)
        # path_to_step = path_to_run + '/Step{}.pickle'.format(t)
        # path_to_loss = path_to_run +'/Loss{}.pickle'.format(t)
        # if not os.path.exists(path_to_run):
        #     os.makedirs(path_to_run)
        # with open(path_to_step, "wb") as f:
        #     pickle.dump(extra_data, f)
        # with open(path_to_loss, "wb") as f:
        #     pickle.dump(loss_history, f)
        # del extra_data
        
        if not train_flow_every_step:
            train_flow = False
        print(design)
        print(designs_so_far)
        print(observations_so_far)
        print(f"Fit post mean = {max_posterior}")#, cov = {posterior_cov}")
        print("True theta     = ", true_theta.reshape(-1))

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
    num_parallel,
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
    annealing_scheme,
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
    mlflow.log_param("annealing_scheme", str(annealing_scheme))

    meta = {
        "model": "location_finding",
        "p": p,
        "K": num_sources,
        "noise_scale": noise_scale,
        "num_histories": num_histories,
    }
    
    t = time.localtime()
    extra_data_id = time.strftime("%Y%m%d%H%M%S", t)
    path_to_extra_data = "./experiment_outputs/loc_fin/{}".format(extra_data_id)
    # if not os.path.exists(path_to_extra_data):
    #     os.makedirs(path_to_extra_data)
    print(path_to_extra_data)

    results_vi = {"loop": [], "seed": seed, "meta": meta}
    
    results = Parallel(n_jobs=num_parallel)(delayed(main_loop)(run=i,
                            path_to_extra_data =path_to_extra_data,
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
                        ) for i in range(num_histories))
    for i in range(num_histories):
        results_vi["loop"].append(results[i])

    if not os.path.exists("./mlflow_outputs"):
        os.makedirs("./mlflow_outputs")
    with open("./mlflow_outputs/results_locfin_mm_vi.pickle", "wb") as f:
        pickle.dump(results_vi, f)
    mlflow.log_artifact("mlflow_outputs/results_locfin_mm_vi.pickle")
    
    ml_info = mlflow.active_run().info
    path_to_artifact = "mlruns/{}/{}/artifacts/results_locfin_mm_vi.pickle".format(
        ml_info.experiment_id, ml_info.run_id
    )
    with open("./"+path_to_artifact, "wb") as f:
        pickle.dump(results_vi, f)
    print("Path to artifact - use this when evaluating:\n", path_to_artifact)
    
    # extra_meta = {
    #     "train_flow_every_step": train_flow_every_step,
    #     "run_flow": run_flow,
    #     "ml_experiment_id":ml_info.experiment_id,
    #     "ml_run_id":ml_info.run_id
    # }
    # path_to_extra_meta =path_to_extra_data + '/extra_meta.pickle'
    # with open(path_to_extra_meta, "wb") as f:
    #     pickle.dump(extra_meta, f)
    # print(path_to_extra_data)
    # print("Done.")
    # print("Evaluating Results")
    eval_from_source(
        path_to_artifact=path_to_artifact,
        num_experiments_to_perform=[T],
        num_inner_samples=int(5e5),
        seed=-1,
        device='cpu',
    )
    # --------------------------------------------------------------------------
    
    
# torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VI baseline: Location finding with MM M+P"
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--physical-dim", default=2, type=int)
    parser.add_argument(
        "--num-histories", help="Number of histories/rollouts", default=1, type=int
    )
    parser.add_argument(
        "--num-parallel", help="Number of histories to run parallel", default=1, type=int
    )
    parser.add_argument("--num-experiments", default=10, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--mlflow-experiment-name", default="locfin_mm_variational", type=str
    )
    parser.add_argument("--lr", default=.005, type=float)
    parser.add_argument("--annealing-scheme", nargs="+", default=[500,.9], type=float)#[250,.8]
    parser.add_argument("--num-steps", default=5000, type=int)
    parser.add_argument("--train-flow-every-step", default=True, type=bool)
    parser.add_argument("--run-flow", default=True, type=bool)
    
    args = parser.parse_args()
    main(
        seed=args.seed,
        mlflow_experiment_name=args.mlflow_experiment_name,
        num_histories=args.num_histories,
        num_parallel=args.num_parallel,
        device=args.device,
        T=args.num_experiments,
        train_flow_every_step= args.train_flow_every_step,
        run_flow = args.run_flow,
        p=args.physical_dim,
        num_sources=2,
        noise_scale=0.5,#0.01,#
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        annealing_scheme = args.annealing_scheme,
    )