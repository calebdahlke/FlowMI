################# JOINT VARIATIONAL ESTIMATOR POLICY ###################################
import os
import pickle
import argparse
import time
import torch
import pyro
from tqdm import trange
import mlflow
import copy
import math

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
from flow_estimator_pyro import MomentMatchMarginalPosterior,SplineFlow, IdentityTransform,InitFlowToIdentity, VariationalMutualInformationOptimizer, cov, LazyNN
from eval_sPCE_from_source import eval_from_source
from pyro.infer.util import torch_item
from oed.design import OED
import pyro.distributions.transforms as Tdist
import numpy as np

from neural.modules import Mlp, SelfAttention
from neural.aggregators import PermutationInvariantImplicitDAD, ConcatImplicitDAD
from neural.baselines import BatchDesignBaseline

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
            neg_loss = self.loss.differentiable_loss(*args, **kwargs)
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

class IndependentSplines(LazyNN):
    def __init__(self,dim,count_bins,bounds,order = 'linear',device='cuda'):
        super().__init__(dim)
        self.dim = dim
        self.spline_transform = Tdist.Spline(dim, count_bins=count_bins, bound=bounds, order=order).to(device=device)
    
    def forward(self,x):
        z = self.spline_transform(x)
        logDet = self.spline_transform.log_abs_det_jacobian(x, z)
        return z, torch.sum(logDet,axis=1)#logDet
    
    def reverse(self,z):
        x = self.spline_transform.reverse(z)
        return x
        

class MomentMatchMarginalPosteriorNonMyopic(VariationalMutualInformationOptimizer):
    def __init__(self, model, batch_size, flow_x, flow_y,prev_flow_theta, train_flow,device, **kwargs):
        super().__init__(
            model=model, batch_size=batch_size
        )
        self.mu = 0
        self.Sigma = 0
        self.dim_lat = 0
        self.dim_obs = 0
        self.fX = flow_x
        self.gY = flow_y
        self.flow_theta = prev_flow_theta if prev_flow_theta is not None else IdentityTransform()
        self.train_flow = train_flow
        self.cond_num = 0
        self.pi_const = 2*torch.acos(torch.zeros(1, device=device))#.to(device)
        self.e_const = torch.exp(torch.tensor([1], device=device))#.to(device)
        # self.zero_vec = torch.zeros(4).to(device)

    def differentiable_loss(self, *args, **kwargs):
        latents, *history = self._get_data(args, kwargs)
        with torch.no_grad():
            latents = self.flow_theta.reverse(latents)#.flatten(-1)
            
        obs_sample = history[0][1]
        for j in range(1,len(history)):
            obs_sample = torch.hstack((obs_sample,history[j][1]))
            
        self.dim_lat = latents.shape[1]
        self.dim_obs = obs_sample.shape[1]
        
        mufX, logDetJfX = self.fX.forward(latents)
        mugY, logDetJgY = self.gY.forward(obs_sample)
        
        data = torch.cat([mufX,mugY],axis=1)
        
        Sigma = cov(data)+1e-5*torch.eye(self.dim_lat+self.dim_obs, device=latents.device)
        self.cond_num = torch.linalg.cond(Sigma)
        if self.cond_num >400:
            cond_hold = 1

        # ############################# Derived non myopic ##########################################################
        # self.hXY = 0
        # self.hY = 0
        # sign, logdetSx  = torch.linalg.slogdet(Sigma[:self.dim_lat,:self.dim_lat])
        # if sign < 0:
        #     print("negative det")
        # self.hX = self.dim_obs*(.5*logdetSx+(self.dim_lat/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX))
        # sign, logdetS  = torch.linalg.slogdet(Sigma)
        # if sign < 0:
        #     print("negative det")
        # self.hXY = .5*logdetS +((self.dim_lat+self.dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX)-torch.mean(logDetJgY)
        
        # sign, logdetSy  = torch.linalg.slogdet(Sigma[self.dim_lat:self.dim_lat+1,self.dim_lat:self.dim_lat+1])
        # if sign < 0:
        #     print("negative det")
        # self.hY = .5*logdetSy +((1)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJgY)
        
        # ### Initial Guess ##################
        # # for i in range(self.dim_obs-1):
        # #     Sytxy1t1 = torch.matmul(Sigma[self.dim_lat+i+1:self.dim_lat+i+2,:self.dim_lat],torch.linalg.solve(Sigma[:self.dim_lat,:self.dim_lat],Sigma[:self.dim_lat,self.dim_lat:self.dim_lat+i+1]))
        # #     Scondy = Sigma[self.dim_lat+i+1:self.dim_lat+i+2,self.dim_lat+i+1:self.dim_lat+i+2]-torch.matmul(Sytxy1t1,torch.linalg.solve(Sigma[self.dim_lat:self.dim_lat+1+i,self.dim_lat:self.dim_lat+1+i],Sytxy1t1.T))
        # #     sign, logdetScondy  = torch.linalg.slogdet(Scondy)
        # #     if sign < 0:
        # #         print("negative det")
        # #     self.hY += .5*logdetScondy + ((1)/2)*(torch.log(2*self.pi_const*self.e_const))
        
        ### Fixed Factor #################
        # for i in range(self.dim_obs-1):
        #     Syy = Sigma[self.dim_lat+i+1:self.dim_lat+i+2,self.dim_lat+i+1:self.dim_lat+i+2]
        #     Sxh = Sigma[:self.dim_lat,self.dim_lat:self.dim_lat+1+i]
        #     Sxx = Sigma[:self.dim_lat,:self.dim_lat]
        #     Syx = Sigma[self.dim_lat+i+1:self.dim_lat+i+2,:self.dim_lat]
        #     Shh = Sigma[self.dim_lat:self.dim_lat+1+i,self.dim_lat:self.dim_lat+1+i]
        #     Shy = Sigma[self.dim_lat:self.dim_lat+1+i,self.dim_lat+i+1:self.dim_lat+i+2]
        #     Syxh = torch.matmul(Syx,torch.linalg.solve(Sxx,Sxh))
        #     A = torch.linalg.solve(Shh,Syxh.T)
        #     Scondy = Syy-torch.matmul(Syxh,A)
        #     Trace_term = torch.matmul(torch.linalg.solve(Scondy,A.T),(Shy-Syxh.T))
        #     cond_num_trace = torch.linalg.cond(Trace_term)
        #     cond_num_y = torch.linalg.cond(Scondy)
        #     if cond_num_trace > 400:
        #         cond_hold = 1
        #         # print('Large Trace Cond')
        #     if cond_num_y > 400:
        #         cond_hold = 1
        #         # print('Large y Cond')
        #     sign, logdetScondy  = torch.linalg.slogdet(Scondy)
        #     if sign < 0:
        #         print("negative det")
        #     self.hY += .5*logdetScondy + ((1)/2)*(torch.log(2*self.pi_const*self.e_const))#-torch.trace(Trace_term)
            
        ###############################################################################################################################################
        # sign, logdetS = torch.linalg.slogdet(Sigma)
        # if sign < 0:
        #     print("negative det")
        # self.hXY = self.dim_obs*(.5*logdetS +((self.dim_lat+self.dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX)-torch.mean(logDetJgY))
        # self.hY = 0
        # self.hX = 0
        # for i in range(self.dim_obs):
        #     sign, logdetSyi = torch.linalg.slogdet(Sigma[self.dim_lat:self.dim_lat+1+i,self.dim_lat:self.dim_lat+1+i])
        #     if sign < 0:
        #         print("negative det")
        #     self.hY += .5*logdetSyi +((1+i)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJgY)
            
        #     idx = torch.cat((torch.arange(self.dim_lat,device=latents.device),self.dim_lat+i+1+torch.arange(self.dim_obs-(i+1),device=latents.device)))
        #     sign, logdetSxyi = torch.linalg.slogdet(Sigma[idx,:][:,idx])
        #     if sign < 0:
        #         print("negative det")
        #     self.hX += .5*logdetSxyi +((self.dim_lat+self.dim_obs-(1+i))/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX)
        
        # self.hY = 0
        # self.hX = 0
        # sign, logdetSy1 = torch.linalg.slogdet(Sigma[self.dim_lat:self.dim_lat+1,self.dim_lat:self.dim_lat+1])
        # sign, logdetSx = torch.linalg.slogdet(Sigma[:self.dim_lat,:self.dim_lat])
        # sign, logdetSxy1 = torch.linalg.slogdet(Sigma[:self.dim_lat+1,:self.dim_lat+1])
        # loss = .5*logdetSx+.5*logdetSy1-.5*logdetSxy1
        # Sigma_x_inv = torch.linalg.inv(Sigma[:self.dim_lat,:self.dim_lat])
        # for i in range(self.dim_obs-1):
        #     Sigma_ytyt = Sigma[self.dim_lat+1+i:self.dim_lat+2+i,self.dim_lat+1+i:self.dim_lat+2+i]
        #     Sigma_ytx= Sigma[self.dim_lat+1+i:self.dim_lat+2+i,:self.dim_lat]
        #     Sigma_yt1yt1= Sigma[self.dim_lat:self.dim_lat+1+i,self.dim_lat:self.dim_lat+1+i]
        #     Sigma_yt1x= Sigma[self.dim_lat:self.dim_lat+1+i,:self.dim_lat]
        #     # Sigma_ytyt1 = Sigma[self.dim_lat+1+i:self.dim_lat+2+i,self.dim_lat:self.dim_lat+1+i]
        #     Sigmaytcondx = Sigma_ytyt-torch.matmul(Sigma_ytx,torch.matmul(Sigma_x_inv,Sigma_ytx.T))
        #     Sigmaytxyt1 = torch.matmul(Sigma_ytx,torch.matmul(Sigma_x_inv,Sigma_yt1x.T))
        #     Sigmaytcondxcondyt1 = Sigma_ytyt-torch.matmul(Sigmaytxyt1,torch.linalg.solve(Sigma_yt1yt1,Sigmaytxyt1.T))
        #     # Sigmaytcondyt1 = Sigma_ytyt-torch.matmul(Sigma_ytyt1,torch.linalg.solve(Sigma_yt1yt1,Sigma_ytyt1.T))
        #     self.hY += -.5*torch.linalg.slogdet(Sigmaytcondx)[1] +.5*torch.linalg.slogdet(Sigmaytcondxcondyt1)[1]#+.5*torch.trace(torch.linalg.solve(Sigmaytcondxcondyt1,Sigmaytcondyt1.T))
        
        sign, logdetS = torch.linalg.slogdet(Sigma)
        if sign < 0:
            print("negative det")
        self.hXY = .5*logdetS +((self.dim_lat+self.dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX)-torch.mean(logDetJgY)

        sign, logdetSy = torch.linalg.slogdet(Sigma[self.dim_lat:,self.dim_lat:])
        if sign < 0:
            print("negative det")
        self.hY = .5*logdetSy +((self.dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJgY)
        
        sign, logdetSx = torch.linalg.slogdet(Sigma[:self.dim_lat,:self.dim_lat])
        if sign < 0:
            print("negative det")
        self.hX = .5*logdetSx +((self.dim_lat)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX)
        
        
        # CondSigma = 1*Sigma
        # for i in range(self.dim_obs):
        #     idx = torch.cat((torch.arange(self.dim_lat,device=latents.device),self.dim_lat+1+torch.arange(self.dim_obs-(i+1),device=latents.device)))
        #     Sigma_yy = CondSigma[self.dim_lat:self.dim_lat+1,self.dim_lat:self.dim_lat+1]
        #     Sigma_yx= CondSigma[self.dim_lat:self.dim_lat+1,:][:,idx]
        #     Sigma_xx= CondSigma[idx,:][:,idx]
        #     CondSigma = Sigma_xx-torch.matmul(Sigma_yx.T,torch.linalg.solve(Sigma_yy,Sigma_yx))
            
        # sign, logdetSpost = torch.linalg.slogdet(CondSigma)
        # if sign < 0:
        #     print("negative det")
        # self.hY = .5*logdetSpost +((self.dim_lat)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX)
        ################################################################################################
        self.mu = torch.mean(data,axis=0)
        self.Sigma = Sigma
        return -self.hX-self.hY#self.hXY-self.hX-self.hY#MI+Loss#

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
        design_init=torch.distributions.Normal(0, 0.5),
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
        # self.prev_design = torch.zeros(self.design_net.designs[0].size(),device=self.design_net.designs[0].device)

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
            # if torch.any(theta.isnan()):
            #     print('NaN')
            #     theta[torch.unique(torch.where(torch.isnan(theta))[0])] = self.flow_theta.reverse(1.0001*theta1[torch.unique(torch.where(torch.isnan(theta))[0])])
        theta = theta.reshape((len(theta),self.K,self.p))
        # if torch.any(theta.isnan()):
        #     print('NaN')
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

def optimise_design(
    posterior_loc,
    posterior_cov,
    flow_theta,
    flow_obs,
    prev_obs,
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
    T = 1 if experiment_number == 9 else 2#int(10-experiment_number)#len(design_init)
    
    n = 1  # batch dim
    design_dim = (n, p)
    observation_dim = n
    hidden_dim = 512
    encoding_dim = 64
    design_arch = "sum"#"attention"#None#"static"#

    hist_encoder_HD = [64, hidden_dim]
    # design emitter hidden dimensions
    des_emitter_HD = [hidden_dim // 2, encoding_dim]
    
    if design_arch == "static":
        des_emitter_HD = []
    # mlflow.log_param(
    #     "design_arch_full", f"{hist_encoder_HD}, {design_arch}, {des_emitter_HD}"
    # )
    if design_arch == "static":
        # batch design baseline
        design_net = BatchDesignBaseline(T=T, design_dim=design_dim,).to(device)
    else:
        history_encoder = Mlp(
            input_dim=[*design_dim, observation_dim],
            hidden_dim=hist_encoder_HD,
            output_dim=encoding_dim,
            name="policy_history_encoder",
        )
        design_emitter = Mlp(
            input_dim=encoding_dim,
            hidden_dim=des_emitter_HD,
            output_dim=design_dim,
            name="policy_design_emitter",
        )
        # Design net: takes pairs [design, observation] as input
        design_net = PermutationInvariantImplicitDAD(
            history_encoder,
            design_emitter,
            empty_value=torch.zeros(design_dim).to(device),
            self_attention_layer=SelfAttention(encoding_dim, encoding_dim)
            if design_arch == "attention"
            else None,
        ).to(device)
    
    ho_model = HiddenObjectsNonMyopic(
        design_net=design_net,
        theta_loc=posterior_loc,
        theta_covmat=posterior_cov,
        flow_theta = flow_theta,
        T=T,
        p=p,
        K=num_sources,
        noise_scale=noise_scale * torch.ones(1, device=device),
    )
    
    if run_flow:
        dim_x = num_sources*p
        dim_y = T
        if flow_theta == None:
            # flowx_loss = torch.tensor(torch.nan)
            # init_lr_x = .005
            # while torch.isnan(flowx_loss):
            #     fX = SplineFlow(dim_x,n_flows=1,hidden_dims=[64], count_bins=128, bounds=6,order = 'quadratic', device=device)
            #     fX, flowx_loss= InitFlowToIdentity(dim_x, fX, bounds=6,lr=init_lr_x,device=device)
            #     init_lr_x *= .5
            fX = SplineFlow(dim_x,n_flows=1,hidden_dims=[64], count_bins=128, bounds=4,order = 'quadratic', device=device)
        else:
            fX = copy.deepcopy(flow_theta)
        if flow_obs == None:
            # flowy_loss = torch.tensor(torch.nan)
            # init_lr_y = .005
            # while torch.isnan(flowy_loss):
            #     gY = IndependentSplines(T, count_bins=128, bounds=5,order = 'quadratic', device=device)
            #     gY, flowy_loss = InitFlowToIdentity(T, gY, bounds=5,lr=init_lr_y,device=device)
            #     init_lr_y *= .5
            
            gY = IndependentSplines(T, count_bins=128, bounds=5,order = 'quadratic', device=device)
            # gY = SplineFlow(T,n_flows=1,hidden_dims=[64], count_bins=128, bounds=4,order = 'quadratic', device=device)
        else:
            gY = IndependentSplines(T, count_bins=128, bounds=5,order = 'quadratic', device=device)
            with torch.no_grad():
                gY.spline_transform.unnormalized_widths.copy_(flow_obs.spline_transform.unnormalized_widths[1:,:])
                gY.spline_transform.unnormalized_heights.copy_(flow_obs.spline_transform.unnormalized_heights[1:,:])
                gY.spline_transform.unnormalized_derivatives.copy_(flow_obs.spline_transform.unnormalized_derivatives[1:,:])
                # gY.spline_transform.unnormalized_lambdas.copy_(flow_obs.spline_transform.unnormalized_lambdas[1:,:])
                
    else:
        fX = IdentityTransform()
        gY = IdentityTransform()

    ### Set-up loss ###
    mi_loss_instance = MomentMatchMarginalPosteriorNonMyopic(
        model=ho_model.model,
        batch_size=batch_size,
        flow_x=fX,
        flow_y=gY,
        prev_flow_theta=flow_theta,
        train_flow=train_flow,
        device=device
    )
    
    lr_design = .005
    lr_flow = .005
    ### Set-up optimiser ###
    optimizer_design = torch.optim.Adam(list(ho_model.design_net.parameters()))#ho_model.design_net.designs)
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
        optimizer_flow = torch.optim.Adam(list(mi_loss_instance.fX.parameters())+list(mi_loss_instance.gY.parameters()))
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
    # design_prev = 1*design_net.designs[0]
    # print(design_prev)
    # j=0
    loss_history = []
    num_steps_range = trange(0, num_steps + 0, desc="Loss: 0.000 ")
    for i in num_steps_range:
        optimizer_design.zero_grad()
        negMI = mi_loss_instance.differentiable_loss()
        negMI.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(design_net.parameters(), 1.0)
        # optimizer_design.step()
        if run_flow and train_flow:
            optimizer_flow.zero_grad()
            # # Log Likelihood Optimization
            (mi_loss_instance.hXY).backward()
            # Foster Bound Optimization
            # (mi_loss_instance.hX + mi_loss_instance.hXY - mi_loss_instance.hY).backward()
            optimizer_flow.step()
        else:
            (mi_loss_instance.hXY).backward()
        optimizer_design.step()
        
        
        if i % 100 == 0:
            num_steps_range.set_description("Loss: {:.3f} ".format(torch_item(negMI)))
            loss_eval = mi_loss_instance.loss()#*args, **kwargs)
            loss_history.append(loss_eval)

        if i % annealing_freq == 0 and not i == 0:
            scheduler_design.step()
            if run_flow and train_flow:
                scheduler_flow.step()
            # scheduler.step(loss_eval)
        # if i % 500 ==0 and not i == 0:
        #     with torch.no_grad():
        #         design_current = design_net.designs[0]
        #         print(design_current)
        #         print(mi_loss_instance.hY)
        #         design_diff = torch.max((design_current-design_prev).pow(2).sum(axis=1).pow(.5))
        #         if design_diff < 1e-2:
        #             j+=1
        #             if j>=2:
        #                 break
        #         else:
        #             design_prev = 1*design_current
        #             j=0      
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
    true_theta = torch.tensor([[[-0.3281,  0.2271, -0.0320,  0.9442]]], device=device)#torch.tensor([[[ 1.3236,  3.7343, -0.5865,  0.5617]]], device=device)#prior.sample(torch.Size([1]))#
    designs_so_far = []
    observations_so_far = []

    # Set posterior equal to the prior
    posterior_loc = theta_loc.reshape(-1)
    posterior_cov = torch.eye(p * num_sources, device=device)

    design_init = torch.zeros((T,p),device=device, dtype=torch.float32)
    prev_obs = 0
    for t in range(0, T):
        t_start = time.time()
        print(f"Step {t + 1}/{T} of Run {run + 1}")
        pyro.clear_param_store()
        ho_model, mi_loss_instance, loss_history = optimise_design(
            posterior_loc,
            posterior_cov,
            flow_theta,
            flow_obs,
            prev_obs,
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
            ell = 0
                    
            mux = mi_loss_instance.mu[:p * num_sources].detach()
            muy = mi_loss_instance.mu[p * num_sources+ell:p * num_sources+ell+1].detach()
            
            Sigmaxx = mi_loss_instance.Sigma[:p * num_sources,:p * num_sources].detach()
            Sigmayy =  mi_loss_instance.Sigma[p * num_sources+ell:p * num_sources+ell+1,p * num_sources+ell:p * num_sources+ell+1].detach()
            
            Sigmaxy = mi_loss_instance.Sigma[:p * num_sources,p * num_sources+ell:p * num_sources+ell+1].detach()
            
            obs, _ = mi_loss_instance.gY.forward(torch.hstack(observation))
            posterior_loc = (mux + torch.matmul(Sigmaxy,torch.linalg.solve(Sigmayy,(obs[0][ell]-muy))).flatten())
            max_posterior = mi_loss_instance.fX.reverse(posterior_loc)
            posterior_cov = Sigmaxx-torch.matmul(Sigmaxy,torch.linalg.solve(Sigmayy,Sigmaxy.T))
            
            flow_theta = mi_loss_instance.fX
            flow_obs = mi_loss_instance.gY
            
            design_init = design[1:]#0*design[0][0][1:]#torch.cat((design[0][0][:ell],design[0][0][ell+1:]))#
            prev_obs = observation[0][0][ell]
            
            #### Plot From Trans Mean Value    
            with torch.no_grad():
                import numpy as np
                import scipy
                import matplotlib.pyplot as plt
                x = np.linspace(-2.5,2.5,100)
                y = np.linspace(-2.5,2.5,100)
                X, Y = np.meshgrid(x, y)
                fig, axs = plt.subplots(2, 2)
                ######### Prior on source 1 ###########################################################
                fX, logJac = mi_loss_instance.fX.forward(torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),mux[2].cpu().numpy()*np.ones(np.shape(X.flatten())),mux[3].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
                points = fX.reshape((100,100,4))
                Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mux.cpu().numpy(), Sigmaxx.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
                axs[0, 0].pcolor(X, Y, Z)
                axs[0, 0].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][1].cpu().numpy(), color='red', marker='x',label = 'True')
                axs[0, 0].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
                ######### Prior on source 2 ###########################################################
                fX, logJac = mi_loss_instance.fX.forward(torch.from_numpy((np.vstack((mux[0].cpu().numpy()*np.ones(np.shape(X.flatten())),mux[1].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float().to(device=device))
                points = fX.reshape((100,100,4))
                Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mux.cpu().numpy(), Sigmaxx.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
                axs[0, 1].pcolor(X, Y, Z)
                axs[0, 1].scatter(true_theta[0][0][2].cpu().numpy(),true_theta[0][0][3].cpu().numpy(), color='red', marker='x',label = 'True')
                axs[0, 1].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
                
                ######### Posterior on source 1 ###########################################################
                fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),max_posterior[2].cpu().numpy()*np.ones(np.shape(X.flatten())),max_posterior[3].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
                points = fX.reshape((100,100,4))
                Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc.cpu().numpy(), posterior_cov.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
                axs[1, 0].pcolor(X, Y, Z)
                axs[1, 0].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][1].cpu().numpy(), color='red', marker='x',label = 'True')
                axs[1, 0].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
                ######### Posterior on source 2 ###########################################################
                fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((max_posterior[0].cpu().numpy()*np.ones(np.shape(X.flatten())),max_posterior[1].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float().to(device=device))
                points = fX.reshape((100,100,4))
                Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc.cpu().numpy(), posterior_cov.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
                axs[1, 1].pcolor(X, Y, Z)
                axs[1, 1].scatter(true_theta[0][0][2].cpu().numpy(),true_theta[0][0][3].cpu().numpy(), color='red', marker='x',label = 'True')
                axs[1, 1].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
                axs[0, 0].title.set_text('Source 1')
                axs[0, 1].title.set_text('Source 2')
                axs[0, 0].set(ylabel='Prior')
                axs[1, 0].set(ylabel='Posterior')
                plt.show()
        t_end = time.time()
        run_time = t_end-t_start

        designs_so_far.append(design[0][0][ell].unsqueeze(0).unsqueeze(0).detach().clone().cpu())
        observations_so_far.append(observation[0][0][ell].unsqueeze(0).unsqueeze(0).cpu())
        
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
        # print(design)
        print(designs_so_far)
        print(observations_so_far)
        print(f"Fit mean  = {max_posterior}, cov = {posterior_cov}")#")#
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
    if not os.path.exists(path_to_extra_data):
        os.makedirs(path_to_extra_data)
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
    print("Evaluating Results")
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
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--mlflow-experiment-name", default="locfin_mm_variational", type=str
    )
    parser.add_argument("--lr", default=.005, type=float)
    parser.add_argument("--annealing-scheme", nargs="+", default=[500,.9], type=float)
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
    
            # ## Plot From True
            # with torch.no_grad():
            #     import numpy as np
            #     import scipy
            #     import matplotlib.pyplot as plt
            #     x = np.linspace(-2.5,2.5,100)
            #     y = np.linspace(-2.5,2.5,100)
            #     X, Y = np.meshgrid(x, y)
            #     fig, axs = plt.subplots(2, 2)
            #     ######### Prior on source 1 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),true_theta[0][0][2].cpu().numpy()*np.ones(np.shape(X.flatten())),true_theta[0][0][3].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mux.cpu().numpy(), Sigmaxx.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[0, 0].pcolor(X, Y, Z)
            #     axs[0, 0].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][1].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[0, 0].scatter(design[0][0][ell].detach().clone().cpu()[0],design[0][0][ell].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            #     ######### Prior on source 2 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((true_theta[0][0][0].cpu().numpy()*np.ones(np.shape(X.flatten())),true_theta[0][0][1].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mux.cpu().numpy(), Sigmaxx.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[0, 1].pcolor(X, Y, Z)
            #     axs[0, 1].scatter(true_theta[0][0][2].cpu().numpy(),true_theta[0][0][3].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[0, 1].scatter(design[0][0][ell].detach().clone().cpu()[0],design[0][0][ell].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
                
            #     ######### Posterior on source 1 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),true_theta[0][0][2].cpu().numpy()*np.ones(np.shape(X.flatten())),true_theta[0][0][3].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc.cpu().numpy(), posterior_cov.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[1, 0].pcolor(X, Y, Z)
            #     axs[1, 0].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][1].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[1, 0].scatter(design[0][0][ell].detach().clone().cpu()[0],design[0][0][ell].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            #     ######### Posterior on source 2 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((true_theta[0][0][0].cpu().numpy()*np.ones(np.shape(X.flatten())),true_theta[0][0][1].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc.cpu().numpy(), posterior_cov.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[1, 1].pcolor(X, Y, Z)
            #     axs[1, 1].scatter(true_theta[0][0][2].cpu().numpy(),true_theta[0][0][3].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[1, 1].scatter(design[0][0][ell].detach().clone().cpu()[0],design[0][0][ell].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            #     axs[0, 0].title.set_text('Source 1')
            #     axs[0, 1].title.set_text('Source 2')
            #     axs[0, 0].set(ylabel='Prior')
            #     axs[1, 0].set(ylabel='Posterior')
            #     plt.show()
            # ## Plot From Mean
            # with torch.no_grad():
            #     import numpy as np
            #     import scipy
            #     import matplotlib.pyplot as plt
            #     x = np.linspace(-2.5,2.5,100)
            #     y = np.linspace(-2.5,2.5,100)
            #     X, Y = np.meshgrid(x, y)
            #     fig, axs = plt.subplots(2, 2)
            #     ######### Prior on source 1 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),mux[2].cpu().numpy()*np.ones(np.shape(X.flatten())),mux[3].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mux.cpu().numpy(), Sigmaxx.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[0, 0].pcolor(X, Y, Z)
            #     axs[0, 0].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][1].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[0, 0].scatter(design[0][0][ell].detach().clone().cpu()[0],design[0][0][ell].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            #     ######### Prior on source 2 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((mux[0].cpu().numpy()*np.ones(np.shape(X.flatten())),mux[1].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mux.cpu().numpy(), Sigmaxx.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[0, 1].pcolor(X, Y, Z)
            #     axs[0, 1].scatter(true_theta[0][0][2].cpu().numpy(),true_theta[0][0][3].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[0, 1].scatter(design[0][0][ell].detach().clone().cpu()[0],design[0][0][ell].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
                
            #     ######### Posterior on source 1 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),posterior_loc[2].cpu().numpy()*np.ones(np.shape(X.flatten())),posterior_loc[3].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc.cpu().numpy(), posterior_cov.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[1, 0].pcolor(X, Y, Z)
            #     axs[1, 0].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][1].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[1, 0].scatter(design[0][0][ell].detach().clone().cpu()[0],design[0][0][ell].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            #     ######### Posterior on source 2 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((posterior_loc[0].cpu().numpy()*np.ones(np.shape(X.flatten())),posterior_loc[1].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc.cpu().numpy(), posterior_cov.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[1, 1].pcolor(X, Y, Z)
            #     axs[1, 1].scatter(true_theta[0][0][2].cpu().numpy(),true_theta[0][0][3].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[1, 1].scatter(design[0][0][ell].detach().clone().cpu()[0],design[0][0][ell].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            #     axs[0, 0].title.set_text('Source 1')
            #     axs[0, 1].title.set_text('Source 2')
            #     axs[0, 0].set(ylabel='Prior')
            #     axs[1, 0].set(ylabel='Posterior')
            #     plt.show()
            # ## Plot From mean cross
            # with torch.no_grad():
            #     import numpy as np
            #     import scipy
            #     import matplotlib.pyplot as plt
            #     x = np.linspace(-2.5,2.5,100)
            #     y = np.linspace(-2.5,2.5,100)
            #     X, Y = np.meshgrid(x, y)
            #     fig, axs = plt.subplots(2, 2)
            #     ######### Prior on source 1 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((mux[0].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten(),mux[3].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mux.cpu().numpy(), Sigmaxx.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[0, 0].pcolor(X, Y, Z)
            #     axs[0, 0].scatter(true_theta[0][0][1].cpu().numpy(),true_theta[0][0][2].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[0, 0].scatter(design[0][0][ell].detach().clone().cpu()[0],design[0][0][ell].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            #     ######### Prior on source 2 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((X.flatten(),mux[1].cpu().numpy()*np.ones(np.shape(X.flatten())),mux[2].cpu().numpy()*np.ones(np.shape(X.flatten())),Y.flatten())).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mux.cpu().numpy(), Sigmaxx.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[0, 1].pcolor(X, Y, Z)
            #     axs[0, 1].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][3].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[0, 1].scatter(design[0][0][ell].detach().clone().cpu()[0],design[0][0][ell].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
                
            #     ######### Posterior on source 1 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((posterior_loc[0].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten(),posterior_loc[3].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc.cpu().numpy(), posterior_cov.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[1, 0].pcolor(X, Y, Z)
            #     axs[1, 0].scatter(true_theta[0][0][1].cpu().numpy(),true_theta[0][0][2].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[1, 0].scatter(design[0][0][ell].detach().clone().cpu()[0],design[0][0][ell].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            #     ######### Posterior on source 2 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((X.flatten(),posterior_loc[1].cpu().numpy()*np.ones(np.shape(X.flatten())),posterior_loc[2].cpu().numpy()*np.ones(np.shape(X.flatten())),Y.flatten())).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc.cpu().numpy(), posterior_cov.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[1, 1].pcolor(X, Y, Z)
            #     axs[1, 1].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][3].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[1, 1].scatter(design[0][0][ell].detach().clone().cpu()[0],design[0][0][ell].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            #     axs[0, 0].title.set_text('Source 1')
            #     axs[0, 1].title.set_text('Source 2')
            #     axs[0, 0].set(ylabel='Prior')
            #     axs[1, 0].set(ylabel='Posterior')
            #     plt.show()