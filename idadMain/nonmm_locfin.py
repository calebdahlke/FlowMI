############################## NEURAL VARIATIONAL ESTIMATOR POLICY ###############################
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

from neural.modules import Mlp
from neural.critics import CriticBA
from neural.baselines import BatchDesignBaseline
from neural.aggregators import ConcatImplicitDAD

from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from estimators.bb_mi import BlackBoxMutualInformation

from location_finding import HiddenObjects

from flow_estimator_pyro import SplineFlow, IdentityTransform, InitFlowToIdentity,VariationalMutualInformationOptimizer, cov
import copy
from oed.primitives import observation_sample, latent_sample, compute_design
import pandas as pd
from eval_sPCE_from_source import eval_from_source
from joblib import Parallel, delayed

from neural.modules import Mlp, SelfAttention
from neural.aggregators import PermutationInvariantImplicitDAD, ConcatImplicitDAD
from neural.baselines import BatchDesignBaseline

class HiddenObjectsTest(nn.Module):
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
        self.theta_loc = theta_loc if theta_loc is not None else torch.zeros((K, p))
        self.theta_covmat = theta_covmat if theta_covmat is not None else torch.eye(p)
        self.flow_theta = flow_theta if flow_theta is not None else IdentityTransform()
        self.theta_prior = dist.MultivariateNormal(
            self.theta_loc, self.theta_covmat
        ).to_event(1)
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
        # if hasattr(self.design_net, "parameters"):
        #     #! this is required for the pyro optimizer
        #     pyro.module("design_net", self.design_net)

        ########################################################################
        # Sample latent variables theta
        ########################################################################
        theta = latent_sample("theta", self.theta_prior)
        with torch.no_grad():
            theta = self.flow_theta.reverse(theta.reshape(len(theta),-1))#
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

class FlowEstimator(BlackBoxMutualInformation):
    def __init__(self, model, batch_size, critic,critic_future, flow_x_prior, flow_x_post,flow_x_post_future, prev_flow_theta, train_flow, device, **kwargs):
        super().__init__(
            model=model, critic=critic, batch_size=batch_size, num_negative_samples=0
        )
        self.mu_prior = 0
        self.Sigma_prior = 0
        self.dim_lat = 0
        # self.dim_obs = 0
        self.hX_post = 0
        self.hX_prior = 0
        self.hX_post_future = 0
        self.critic_future = critic_future
        self.fX_prior = flow_x_prior
        self.fX_post = flow_x_post
        self.fX_post_future = flow_x_post_future
        # self.gY = flow_y
        self.flow_theta = prev_flow_theta if prev_flow_theta is not None else IdentityTransform()
        self.train_flow = train_flow
        self.pi_const = 2*torch.acos(torch.zeros(1, device=device))
        self.e_const = torch.exp(torch.tensor([1], device=device))

    def differentiable_loss(self, *args, **kwargs):
        # pyro.module("critic_net", self.critic)  # !!
        latents, *history = self._get_data(args, kwargs)
        K = latents.shape[1]
        p = latents.shape[2]
        with torch.no_grad():
            latents = self.flow_theta.reverse(latents.reshape(len(latents),-1))
            
        self.dim_lat = latents.shape[1]
        
        latents_fX_post, logDetJfX_post = self.fX_post.forward(latents)
        log_probs_q = self.critic(latents_fX_post.reshape((len(latents),K,p)), history[0])
        self.hX_post = -log_probs_q.mean()-logDetJfX_post.mean()
        
        latents_fX_prior, logDetJfX_prior = self.fX_prior.forward(latents)
        Sigma_prior = torch.cov(latents_fX_prior.T)
        sign, logdetSx_prior  = torch.linalg.slogdet(Sigma_prior)
        if sign < 0:
            print("negative det")
        self.hX_prior = .5*logdetSx_prior+(self.dim_lat/2)*(torch.log(2*self.pi_const*self.e_const))-logDetJfX_prior.mean()
        
        if self.critic_future == None:
            self.hX_post_future = self.hX_post_future
            loss = -2*self.hX_prior
        else:
            latents_fX_post_future, logDetJfX_post_future = self.fX_post_future.forward(latents)
            log_probs_q_future = self.critic_future(latents_fX_post_future.reshape((len(latents),K,p)), *history)
            self.hX_post_future = -log_probs_q_future.mean()-logDetJfX_post_future.mean()
            loss = -1*(self.hX_post+2*self.hX_prior)
        
        self.mu_prior = torch.mean(latents_fX_prior,axis=0)
        self.Sigma_prior = Sigma_prior
        return loss#-1*(self.hX_post+2*self.hX_prior)

    def loss(self, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using the BA lower bound == -EIG
        """
        loss_to_constant = torch_item(self.differentiable_loss(*args, **kwargs))
        return loss_to_constant


def optimise_design_and_critic(
    posterior_loc,
    posterior_scale,
    flow_prior_theta,
    flow_post_theta,
    obs_critic,
    train_flow,
    run_flow,
    experiment_number,
    noise_scale,
    p,
    num_sources,
    device,
    batch_size,
    num_steps,
    lr_design,
    lr_flow,
    annealing_scheme=None,
):

    T = int(10-experiment_number)#1 if experiment_number == 9 else 2##len(design_init)
    
    n = 1  # batch dim
    design_dim = (n, p)
    observation_dim = n
    hidden_dim = 512
    encoding_dim = 64#128#
    design_arch = "sum"#"attention"#None#"static"#

    hist_encoder_HD = [64, hidden_dim]
    # design emitter hidden dimensions
    des_emitter_HD = [hidden_dim // 2, encoding_dim]
    
    if design_arch == "static":
        des_emitter_HD = []
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
    
    new_mean = posterior_loc.reshape(num_sources, p)
    new_covmat = torch.cat(
        [
            torch.diag(x).unsqueeze(0)
            for x in (posterior_scale ** 2).reshape(num_sources, p)
        ]
    )
    ho_model = HiddenObjectsTest(
        design_net=design_net,
        # Normal family -- new prior is stil MVN but with different params
        theta_loc=new_mean,
        theta_covmat=new_covmat,
        flow_theta=flow_post_theta,
        T=T,
        p=p,
        K=num_sources,
        noise_scale=noise_scale * torch.ones(1, device=device),
    )

    ### Set up model networks ###
    latent_dim = (num_sources, p)
    if obs_critic== None:
        hist_encoder_HD = [64, hidden_dim]#[hidden_dim, hidden_dim]#
        hist_enc_critic_head_HD = [
            hidden_dim // 2,
            hidden_dim,
        ]
        ###### CRITIC NETWORKS #######
        ## history encoder
        critic_pre_pool_history_encoder = Mlp(
            input_dim=[*design_dim, observation_dim],
            hidden_dim=hist_encoder_HD,
            output_dim=encoding_dim,
        )
        critic_history_enc_head = Mlp(
            input_dim=encoding_dim,
            hidden_dim=hist_enc_critic_head_HD,
            output_dim=encoding_dim,
        )
        critic_history_encoder = ConcatImplicitDAD(
            encoder_network=critic_pre_pool_history_encoder,
            emission_network=critic_history_enc_head,
            T=1,
            empty_value=torch.ones(design_dim).to(device),
        )
        # critic_history_encoder = PermutationInvariantImplicitDAD(
        #     critic_pre_pool_history_encoder,
        #     critic_history_enc_head,
        #     empty_value=torch.zeros(design_dim).to(device),
        #     self_attention_layer=SelfAttention(encoding_dim, encoding_dim)
        #     if design_arch == "attention"
        #     else None,
        # ).to(device)
        critic_net = CriticBA(
            history_encoder_network=critic_history_encoder, latent_dim=latent_dim
        ).to(device)
    else:
        critic_net = obs_critic
        
    if T>1:
        hist_encoder_HD_future = [64, hidden_dim]#[hidden_dim, hidden_dim]#
        hist_enc_critic_head_HD_future = [
            hidden_dim // 2,
            hidden_dim,
        ]
        ###### CRITIC NETWORKS #######
        ## history encoder
        critic_pre_pool_history_encoder_future = Mlp(
            input_dim=[*design_dim, observation_dim],
            hidden_dim=hist_encoder_HD_future,
            output_dim=encoding_dim,
        )
        critic_history_enc_head_future = Mlp(
            input_dim=encoding_dim,
            hidden_dim=hist_enc_critic_head_HD_future,
            output_dim=encoding_dim,
        )
        # critic_history_encoder = ConcatImplicitDAD(
        #     encoder_network=critic_pre_pool_history_encoder,
        #     emission_network=critic_history_enc_head,
        #     T=T,
        #     empty_value=torch.ones(design_dim).to(device),
        # )
        critic_history_encoder_future = PermutationInvariantImplicitDAD(
            critic_pre_pool_history_encoder_future,
            critic_history_enc_head_future,
            empty_value=torch.zeros(design_dim).to(device),
            self_attention_layer=SelfAttention(encoding_dim, encoding_dim)
            if design_arch == "attention"
            else None,
        ).to(device)
        critic_net_future = CriticBA(
            history_encoder_network=critic_history_encoder_future, latent_dim=latent_dim
        ).to(device)
    else:
        critic_net_future = None
    
    if run_flow:
        dim_x = num_sources*p
        if flow_post_theta == None:#flow_prior_theta == None:
            # flowx_loss = torch.tensor(torch.nan)
            # init_lr_x = .005
            # while torch.isnan(flowx_loss):
            #     fX_prior = SplineFlow(dim_x,n_flows=1,hidden_dims=[32,64], count_bins=256, bounds=4,order = 'quadratic', device=device)
            #     fX_prior, flowx_loss= InitFlowToIdentity(dim_x, fX_prior, bounds=4,lr=init_lr_x,device=device)
            #     init_lr_x *= .5
            fX_prior = SplineFlow(dim_x,n_flows=1,hidden_dims=[32,64], count_bins=256, bounds=4,order = 'quadratic', device=device)
        else:
            # fX_prior = copy.deepcopy(flow_prior_theta)
            fX_prior = copy.deepcopy(flow_post_theta)
        if flow_post_theta == None:
            # flowx_loss = torch.tensor(torch.nan)
            # init_lr_x = .005
            # while torch.isnan(flowx_loss):
            #     fX_post = SplineFlow(dim_x,n_flows=1,hidden_dims=[32,64], count_bins=256, bounds=4,order = 'quadratic', device=device)
            #     fX_post, flowx_loss= InitFlowToIdentity(dim_x, fX_post, bounds=4,lr=init_lr_x,device=device)
            #     init_lr_x *= .5
            fX_post = SplineFlow(dim_x,n_flows=1,hidden_dims=[32,64], count_bins=256, bounds=4,order = 'quadratic', device=device)
        else:
            fX_post = copy.deepcopy(flow_post_theta)
        if T>1:
            # flowx_loss = torch.tensor(torch.nan)
            # init_lr_x = .005
            # while torch.isnan(flowx_loss):
            #     fX_post = SplineFlow(dim_x,n_flows=1,hidden_dims=[32,64], count_bins=256, bounds=4,order = 'quadratic', device=device)
            #     fX_post, flowx_loss= InitFlowToIdentity(dim_x, fX_post, bounds=4,lr=init_lr_x,device=device)
            #     init_lr_x *= .5
            fX_post_future = SplineFlow(dim_x,n_flows=1,hidden_dims=[32,64], count_bins=256, bounds=4,order = 'quadratic', device=device)
        else:
            fX_post_future = IdentityTransform()
    else:
        fX_prior = IdentityTransform()
        fX_post = IdentityTransform()
        fX_post_future = IdentityTransform()
    ### Set-up loss ###
    mi_loss_instance = FlowEstimator(
        model=ho_model.model,
        batch_size=batch_size,
        critic=critic_net,
        critic_future = critic_net_future,
        flow_x_prior = fX_prior,
        flow_x_post = fX_post,
        flow_x_post_future = fX_post_future,
        prev_flow_theta=flow_post_theta,
        train_flow = train_flow,
        device = device,
    )

    ### Set-up optimiser ###

    optimizer_design = torch.optim.Adam(list(ho_model.design_net.parameters()))#list(ho_model.design_net.parameters())+list(critic_net.parameters())+list(critic_net_future.parameters()))#
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
        if T>1:
            optimizer_flow = torch.optim.Adam(list(mi_loss_instance.fX_prior.parameters())+list(mi_loss_instance.fX_post.parameters())+list(mi_loss_instance.fX_post_future.parameters())+list(critic_net.parameters())+list(critic_net_future.parameters()))#+list(critic_net.parameters()))
        else:
            optimizer_flow = torch.optim.Adam(list(mi_loss_instance.fX_prior.parameters())+list(mi_loss_instance.fX_post.parameters())+list(mi_loss_instance.fX_post_future.parameters())+list(critic_net.parameters()))
        annealing_freq, factor = annealing_scheme
        scheduler_flow = pyro.optim.ExponentialLR(
            {
                "optimizer": optimizer_flow,
                "optim_args": {"lr": lr_flow},
                "gamma" : factor,
                "verbose": False,
            }
        )
    
    ## Optimise ###
    # design_prev = 1*design_net.designs[0]
    # j=0
    loss_history = []
    num_steps_range = trange(0, num_steps + 0, desc="Loss: 0.000 ")
    for i in num_steps_range:
        optimizer_design.zero_grad()
        # if run_flow and train_flow:
        #     optimizer_flow.zero_grad()
        negMI = mi_loss_instance.differentiable_loss()
        negMI.backward(retain_graph=True)
        if run_flow and train_flow:
            optimizer_flow.zero_grad()
            if T>1:
                (mi_loss_instance.hX_post_future+mi_loss_instance.hX_post+mi_loss_instance.hX_prior).backward()
            else:
                (mi_loss_instance.hX_post+mi_loss_instance.hX_prior).backward()
            optimizer_flow.step()
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
        #         design_diff = torch.max((design_current-design_prev).pow(2).sum(axis=1).pow(.5))
        #         # print(design_current)
        #         if design_diff < 1e-2:
        #             j+=1
        #             if j>=2:
        #                 break
        #         else:
        #             design_prev = 1*design_current
        #             j=0

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
    lr_design,
    lr_flow,
    annealing_scheme,
):
    pyro.clear_param_store()
    flow_prior_theta = None
    flow_post_theta = None
    critic_net = None
    train_flow = True
    theta_loc = torch.zeros((num_sources, p), device=device)
    theta_covmat = torch.eye(p, device=device)
    prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)

    # sample true param
    true_theta = torch.tensor([[[-0.3281,  0.2271], [-0.0320,  0.9442]]], device=device)#prior.sample(torch.Size([1]))#torch.tensor([[[-0.6240, -1.1926], [-0.3522, -0.7592]]], device=device)#

    designs_so_far = []
    observations_so_far = []

    # Set posterior equal to the prior
    posterior_loc = theta_loc.reshape(-1)  # check if needs to be reshaped.
    posterior_scale = torch.ones(p * num_sources, device=device)

    for t in range(0, T):
        print(f"Step {t + 1}/{T} of Run {run + 1}")
        pyro.clear_param_store()
        ho_model, mi_loss_instance = optimise_design_and_critic(
            posterior_loc,
            posterior_scale,
            flow_prior_theta,
            flow_post_theta,
            critic_net,
            train_flow,
            run_flow,
            experiment_number=t,
            noise_scale=noise_scale,
            p=p,
            num_sources=num_sources,
            device=device,
            batch_size=batch_size,
            num_steps=num_steps,
            lr_design = lr_design,
            lr_flow = lr_flow,
            annealing_scheme=annealing_scheme,
        )      
        with torch.no_grad(): 
            if t>0:
                trans_true_theta,_ = flow_post_theta.forward(true_theta.reshape(len(true_theta),-1))
                trans_true_theta = trans_true_theta.reshape(true_theta.size())
            else:
                trans_true_theta = true_theta
        
        design, observation = ho_model.forward(theta=trans_true_theta)
        
        posterior_loc, posterior_scale = mi_loss_instance.critic.get_variational_params(
            *zip(design[0], observation[0])
        )
        
        posterior_loc, posterior_scale = (
            posterior_loc.unsqueeze(0).detach(),
            posterior_scale.unsqueeze(0).detach(),
        )
        
        ## Fixes deepcopy issue?
        with torch.no_grad():
            mu_prior_trans,_ = mi_loss_instance.fX_prior(mi_loss_instance.mu_prior)
            mu_post_trans,_ = mi_loss_instance.fX_post(posterior_loc)
            
        flow_prior_theta = mi_loss_instance.fX_prior
        flow_post_theta = mi_loss_instance.fX_post
        critic_net = mi_loss_instance.critic
            
        designs_so_far.append(design[0])
        observations_so_far.append(observation[0])
        
        print(designs_so_far)
        print(observations_so_far)
        print(f"Fitted posterior: mean = {posterior_loc}, sd = {posterior_scale}")
        print("True theta = ", true_theta.reshape(-1))
        
        #### Plot From True Theta
        with torch.no_grad():
            import numpy as np
            import scipy
            import matplotlib.pyplot as plt
            x = np.linspace(-2.5,2.5,100)
            y = np.linspace(-2.5,2.5,100)
            X, Y = np.meshgrid(x, y)
            fig, axs = plt.subplots(2, 2)
            ######### Prior on source 1 ###########################################################
            fX, logJac = mi_loss_instance.fX_prior.forward(torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),true_theta[0][1][0].cpu().numpy()*np.ones(np.shape(X.flatten())),true_theta[0][1][1].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
            points = fX.reshape((100,100,4))
            Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mi_loss_instance.mu_prior.cpu().numpy(), mi_loss_instance.Sigma_prior.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            axs[0, 0].pcolor(X, Y, Z)
            axs[0, 0].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][1].cpu().numpy(), color='red', marker='x',label = 'True')
            axs[0, 0].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            ######### Prior on source 2 ###########################################################
            fX, logJac = mi_loss_instance.fX_prior.forward(torch.from_numpy((np.vstack((true_theta[0][0][0].cpu().numpy()*np.ones(np.shape(X.flatten())),true_theta[0][0][1].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float().to(device=device))
            points = fX.reshape((100,100,4))
            Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mi_loss_instance.mu_prior.cpu().numpy(), mi_loss_instance.Sigma_prior.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            axs[0, 1].pcolor(X, Y, Z)
            axs[0, 1].scatter(true_theta[0][1][0].cpu().numpy(),true_theta[0][1][1].cpu().numpy(), color='red', marker='x',label = 'True')
            axs[0, 1].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            
            ######### Posterior on source 1 ###########################################################
            fX, logJac = flow_post_theta.forward(torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),true_theta[0][1][0].cpu().numpy()*np.ones(np.shape(X.flatten())),true_theta[0][1][1].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
            points = fX.reshape((100,100,4))
            Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc[0].cpu().numpy(), torch.diag(posterior_scale[0]).cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            axs[1, 0].pcolor(X, Y, Z)
            axs[1, 0].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][1].cpu().numpy(), color='red', marker='x',label = 'True')
            axs[1, 0].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            ######### Posterior on source 2 ###########################################################
            fX, logJac = flow_post_theta.forward(torch.from_numpy((np.vstack((true_theta[0][0][0].cpu().numpy()*np.ones(np.shape(X.flatten())),true_theta[0][0][1].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float().to(device=device))
            points = fX.reshape((100,100,4))
            Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc[0].cpu().numpy(), torch.diag(posterior_scale[0]).cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            axs[1, 1].pcolor(X, Y, Z)
            axs[1, 1].scatter(true_theta[0][1][0].cpu().numpy(),true_theta[0][1][1].cpu().numpy(), color='red', marker='x',label = 'True')
            axs[1, 1].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            axs[0, 0].title.set_text('Source 1')
            axs[0, 1].title.set_text('Source 2')
            axs[0, 0].set(ylabel='Prior')
            axs[1, 0].set(ylabel='Posterior')
            plt.show()
            
        # #### Plot From Mean Value
        # with torch.no_grad():
        #     import numpy as np
        #     import scipy
        #     import matplotlib.pyplot as plt
        #     x = np.linspace(-2.5,2.5,100)
        #     y = np.linspace(-2.5,2.5,100)
        #     X, Y = np.meshgrid(x, y)
        #     fig, axs = plt.subplots(2, 2)
        #     ######### Prior on source 1 ###########################################################
        #     fX, logJac = mi_loss_instance.fX_prior.forward(torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),mi_loss_instance.mu_prior[2].cpu().numpy()*np.ones(np.shape(X.flatten())),mi_loss_instance.mu_prior[3].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
        #     points = fX.reshape((100,100,4))
        #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mi_loss_instance.mu_prior.cpu().numpy(), mi_loss_instance.Sigma_prior.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
        #     axs[0, 0].pcolor(X, Y, Z)
        #     axs[0, 0].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][1].cpu().numpy(), color='red', marker='x',label = 'True')
        #     axs[0, 0].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
        #     ######### Prior on source 2 ###########################################################
        #     fX, logJac = mi_loss_instance.fX_prior.forward(torch.from_numpy((np.vstack((mi_loss_instance.mu_prior[0].cpu().numpy()*np.ones(np.shape(X.flatten())),mi_loss_instance.mu_prior[1].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float().to(device=device))
        #     points = fX.reshape((100,100,4))
        #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mi_loss_instance.mu_prior.cpu().numpy(), mi_loss_instance.Sigma_prior.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
        #     axs[0, 1].pcolor(X, Y, Z)
        #     axs[0, 1].scatter(true_theta[0][1][0].cpu().numpy(),true_theta[0][1][1].cpu().numpy(), color='red', marker='x',label = 'True')
        #     axs[0, 1].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            
        #     ######### Posterior on source 1 ###########################################################
        #     fX, logJac = flow_post_theta.forward(torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),posterior_loc[0,2].cpu().numpy()*np.ones(np.shape(X.flatten())),posterior_loc[0,3].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
        #     points = fX.reshape((100,100,4))
        #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc[0].cpu().numpy(), torch.diag(posterior_scale[0]).cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
        #     axs[1, 0].pcolor(X, Y, Z)
        #     axs[1, 0].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][1].cpu().numpy(), color='red', marker='x',label = 'True')
        #     axs[1, 0].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
        #     ######### Posterior on source 2 ###########################################################
        #     fX, logJac = flow_post_theta.forward(torch.from_numpy((np.vstack((posterior_loc[0,0].cpu().numpy()*np.ones(np.shape(X.flatten())),posterior_loc[0,1].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float().to(device=device))
        #     points = fX.reshape((100,100,4))
        #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc[0].cpu().numpy(), torch.diag(posterior_scale[0]).cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
        #     axs[1, 1].pcolor(X, Y, Z)
        #     axs[1, 1].scatter(true_theta[0][1][0].cpu().numpy(),true_theta[0][1][1].cpu().numpy(), color='red', marker='x',label = 'True')
        #     axs[1, 1].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
        #     axs[0, 0].title.set_text('Source 1')
        #     axs[0, 1].title.set_text('Source 2')
        #     axs[0, 0].set(ylabel='Prior')
        #     axs[1, 0].set(ylabel='Posterior')
        #     plt.show()
            
        # #### Plot From Trans Mean Value    
        # with torch.no_grad():
        #     import numpy as np
        #     import scipy
        #     import matplotlib.pyplot as plt
        #     x = np.linspace(-2.5,2.5,100)
        #     y = np.linspace(-2.5,2.5,100)
        #     X, Y = np.meshgrid(x, y)
        #     fig, axs = plt.subplots(2, 2)
        #     ######### Prior on source 1 ###########################################################
        #     fX, logJac = mi_loss_instance.fX_prior.forward(torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),mu_prior_trans[2].cpu().numpy()*np.ones(np.shape(X.flatten())),mu_prior_trans[3].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
        #     points = fX.reshape((100,100,4))
        #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mi_loss_instance.mu_prior.cpu().numpy(), mi_loss_instance.Sigma_prior.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
        #     axs[0, 0].pcolor(X, Y, Z)
        #     axs[0, 0].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][1].cpu().numpy(), color='red', marker='x',label = 'True')
        #     axs[0, 0].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
        #     ######### Prior on source 2 ###########################################################
        #     fX, logJac = mi_loss_instance.fX_prior.forward(torch.from_numpy((np.vstack((mu_prior_trans[0].cpu().numpy()*np.ones(np.shape(X.flatten())),mu_prior_trans[1].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float().to(device=device))
        #     points = fX.reshape((100,100,4))
        #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mi_loss_instance.mu_prior.cpu().numpy(), mi_loss_instance.Sigma_prior.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
        #     axs[0, 1].pcolor(X, Y, Z)
        #     axs[0, 1].scatter(true_theta[0][1][0].cpu().numpy(),true_theta[0][1][1].cpu().numpy(), color='red', marker='x',label = 'True')
        #     axs[0, 1].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            
        #     ######### Posterior on source 1 ###########################################################
        #     fX, logJac = flow_post_theta.forward(torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),mu_post_trans[0,2].cpu().numpy()*np.ones(np.shape(X.flatten())),mu_post_trans[0,3].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
        #     points = fX.reshape((100,100,4))
        #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc[0].cpu().numpy(), torch.diag(posterior_scale[0]).cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
        #     axs[1, 0].pcolor(X, Y, Z)
        #     axs[1, 0].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][1].cpu().numpy(), color='red', marker='x',label = 'True')
        #     axs[1, 0].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
        #     ######### Posterior on source 2 ###########################################################
        #     fX, logJac = flow_post_theta.forward(torch.from_numpy((np.vstack((mu_post_trans[0,0].cpu().numpy()*np.ones(np.shape(X.flatten())),mu_post_trans[0,1].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float().to(device=device))
        #     points = fX.reshape((100,100,4))
        #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc[0].cpu().numpy(), torch.diag(posterior_scale[0]).cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
        #     axs[1, 1].pcolor(X, Y, Z)
        #     axs[1, 1].scatter(true_theta[0][1][0].cpu().numpy(),true_theta[0][1][1].cpu().numpy(), color='red', marker='x',label = 'True')
        #     axs[1, 1].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
        #     axs[0, 0].title.set_text('Source 1')
        #     axs[0, 1].title.set_text('Source 2')
        #     axs[0, 0].set(ylabel='Prior')
        #     axs[1, 0].set(ylabel='Posterior')
        #     plt.show()
        
        
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
    lr_design,
    lr_flow,
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
    mlflow.log_param("lr_design", lr_design)
    mlflow.log_param("lr_flow", lr_flow)
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
    results_vi = {"loop": [], "seed": seed, "meta": meta}
    # for i in range(num_histories):
    #     results = main_loop(
    #         run=i,
    #         mlflow_run_id=mlflow.active_run().info.run_id,
    #         device=device,
    #         T=T,
    #         train_flow_every_step=train_flow_every_step,
    #         run_flow=run_flow,
    #         noise_scale=noise_scale,
    #         num_sources=num_sources,
    #         p=p,
    #         batch_size=batch_size,
    #         num_steps=num_steps,
    #         lr_design = lr_design,
    #         lr_flow = lr_flow,
    #         annealing_scheme=annealing_scheme,
    #     )
    #     results_vi["loop"].append(results)
    
    results = Parallel(n_jobs=num_parallel)(delayed(main_loop)(run=i,
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
                            lr_design=lr_design,
                            lr_flow=lr_flow,
                            annealing_scheme=annealing_scheme,
                        ) for i in range(num_histories))
    for i in range(num_histories):
        results_vi["loop"].append(results[i])

    # Log the results dict as an artifact
    if not os.path.exists("./mlflow_outputs"):
        os.makedirs("./mlflow_outputs")
    with open("./mlflow_outputs/results_locfin_vi.pickle", "wb") as f:
        pickle.dump(results_vi, f)
    mlflow.log_artifact("mlflow_outputs/results_locfin_vi.pickle")
    print("Done.")
    ml_info = mlflow.active_run().info
    path_to_artifact = "mlruns/{}/{}/artifacts/results_locfin_vi.pickle".format(
        ml_info.experiment_id, ml_info.run_id
    )
    with open("./"+path_to_artifact, "wb") as f:
        pickle.dump(results_vi, f)
    print("Path to artifact - use this when evaluating:\n", path_to_artifact)
    
    eval_from_source(
        path_to_artifact=path_to_artifact,
        num_experiments_to_perform=[T],
        num_inner_samples=int(5e5),
        seed=-1,
        device='cpu',
    )
    # --------------------------------------------------------------------------


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
    parser.add_argument("--lr-design", default=.005, type=float)
    parser.add_argument("--lr-flow", default=.005, type=float)
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
        noise_scale=0.5,#0.00001,#
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr_design=args.lr_design,
        lr_flow=args.lr_flow,
        annealing_scheme = args.annealing_scheme,
    )