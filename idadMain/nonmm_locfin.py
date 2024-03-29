import os
import pickle
import argparse
import time
import torch
import pyro
from tqdm import trange
import mlflow
import copy
from neural.baselines import BatchDesignBaseline

from experiment_tools.pyro_tools import auto_seed
from oed.design import OED

from joblib import Parallel, delayed

from simulations import HiddenObjectsVar
from flow_estimator_pyro import SplineFlow, IdentityTransform, InitFlowToIdentity,VariationalMutualInformationOptimizer, cov
from eval_sPCE_from_source import eval_from_source
from pyro.infer.util import torch_item

class MomentMatchMarginalPosteriorTest(VariationalMutualInformationOptimizer):
    def __init__(self, model, batch_size, flow_x_prior, flow_x_post, flow_y,prev_flow_theta,train_flow,device, **kwargs):
        super().__init__(
            model=model, batch_size=batch_size
        )
        self.mu_post = 0#torch.zeros(5, device=device)
        self.Sigma_post = 0#torch.eye(5, device=device)
        self.mu_prior = 0
        self.Sigma_prior = 0
        self.dim_lat = 0
        self.dim_obs = 0
        self.hX = 0
        self.hY = 0
        self.hXY = 0
        self.fX_prior = flow_x_prior
        self.fX_post = flow_x_post
        self.gY = flow_y
        self.flow_theta = prev_flow_theta if prev_flow_theta is not None else IdentityTransform()
        self.train_flow = train_flow
        self.pi_const = 2*torch.acos(torch.zeros(1, device=device))
        self.e_const = torch.exp(torch.tensor([1], device=device))
    
    def differentiable_loss(self, *args, **kwargs):
        latents, *history = self._get_data(args, kwargs)
        with torch.no_grad():
            latents = self.flow_theta.reverse(latents)
        self.dim_lat = latents.shape[1]
        self.dim_obs = history[0][1].shape[1]
        
        mufX_prior, logDetJfX_prior = self.fX_prior.forward(latents)
        mufX_post, logDetJfX_post = self.fX_post.forward(latents)
        mugY, logDetJgY = self.gY.forward(history[0][1])

        data_post = torch.cat([mufX_post,mugY],axis=1)
        
        Sigma_post = torch.cov(data_post.T)
        
        Sigma_prior = torch.cov(mufX_prior.T)
        
        sign, logdetS_post  = torch.linalg.slogdet(Sigma_post)
        if sign < 0:
            print("negative det")
        self.hXY = .5*logdetS_post +((self.dim_lat+self.dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX_post)-torch.mean(logDetJgY)
        
        sign, logdetSy  = torch.linalg.slogdet(Sigma_post[self.dim_lat:,self.dim_lat:])
        if sign < 0:
            print("negative det")
        self.hY = .5*logdetSy +(self.dim_obs/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJgY)

        sign, logdetSx_prior  = torch.linalg.slogdet(Sigma_prior)
        if sign < 0:
            print("negative det")
        self.hX = .5*logdetSx_prior+(self.dim_lat/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX_prior)

        self.mu_post = torch.mean(data_post,axis=0)
        self.Sigma_post = Sigma_post
        self.mu_prior = torch.mean(mufX_prior,axis=0)
        self.Sigma_prior = Sigma_prior
        return (self.hXY-self.hY-self.hX)

    def loss(self, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using the BA lower bound == -EIG
        """
        loss_to_constant = torch_item(self.differentiable_loss(*args, **kwargs))
        return loss_to_constant

def optimise_design(
    posterior_loc,
    posterior_cov,
    flow_prior_theta,
    flow_post_theta,
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
    lr_design,
    lr_flow,
    annealing_scheme=None,
):
    design_init = (
        torch.distributions.Normal(0.0, 0.01)
        if experiment_number == 0
        else torch.distributions.Normal(0.0, 1.0)
    )
    # design_init = torch.distributions.Normal(0.0, 0.01)
    design_net = BatchDesignBaseline(
        T=1, design_dim=(1, p), design_init=design_init
    ).to(device)


    ho_model = HiddenObjectsVar(
        design_net=design_net,
        theta_loc=posterior_loc,
        theta_covmat=posterior_cov,
        flow_theta = flow_prior_theta,
        T=1,
        p=p,
        K=num_sources,
        noise_scale=noise_scale * torch.ones(1, device=device),
    )
    
    if run_flow:
        dim_x = num_sources*p
        dim_y = 1
        if flow_prior_theta == None:
            # flowx_loss = torch.tensor(torch.nan)
            # init_lr_x = .005
            # while torch.isnan(flowx_loss):
            #     fX = SplineFlow(dim_x,n_flows=1,hidden_dims=[32], count_bins=64, bounds=4,order = 'quadratic', device=device)
            #     fX, flowx_loss= InitFlowToIdentity(dim_x, fX, bounds=4,lr=init_lr_x,device=device)
            #     init_lr_x *= .5
            fX_prior = SplineFlow(dim_x,n_flows=1,hidden_dims=[32], count_bins=128, bounds=4,order = 'quadratic', device=device)
        else:
            # fX_prior = copy.deepcopy(flow_prior_theta)
            fX_prior = copy.deepcopy(flow_post_theta)
        if flow_post_theta == None:
            # flowx_loss = torch.tensor(torch.nan)
            # init_lr_x = .005
            # while torch.isnan(flowx_loss):
            #     fX = SplineFlow(dim_x,n_flows=1,hidden_dims=[32], count_bins=64, bounds=4,order = 'quadratic', device=device)
            #     fX, flowx_loss= InitFlowToIdentity(dim_x, fX, bounds=4,lr=init_lr_x,device=device)
            #     init_lr_x *= .5
            fX_post = SplineFlow(dim_x,n_flows=1,hidden_dims=[32], count_bins=128, bounds=4,order = 'quadratic', device=device)
        else:
            fX_post = copy.deepcopy(flow_post_theta)
        if flow_obs == None:
            # flowy_loss = torch.tensor(torch.nan)
            # init_lr_y = .005
            # while torch.isnan(flowy_loss):
            #     gY = SplineFlow(dim_y,count_bins=64, bounds=5,order = 'quadratic', device=device)
            #     gY, flowy_loss = InitFlowToIdentity(dim_y, gY, bounds=5,lr=init_lr_y,device=device)
            #     init_lr_y *= .5
            gY = SplineFlow(dim_y,count_bins=128, bounds=5,order = 'quadratic', device=device)
        else:
            gY = copy.deepcopy(flow_obs)
            
    else:
        fX_prior = IdentityTransform()
        fX_post = IdentityTransform()
        gY = IdentityTransform()

    ### Set-up loss ###
    mi_loss_instance = MomentMatchMarginalPosteriorTest(
        model=ho_model.model,
        batch_size=batch_size,
        flow_x_prior=fX_prior,
        flow_x_post= fX_post,
        flow_y=gY,
        prev_flow_theta=flow_prior_theta,
        train_flow=train_flow,
        device=device
    )
    
    ### Set-up optimiser ###
    optimizer_design = torch.optim.Adam(ho_model.design_net.designs)
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
        optimizer_flow = torch.optim.Adam(list(mi_loss_instance.fX_prior.parameters())+list(mi_loss_instance.fX_post.parameters())+list(mi_loss_instance.gY.parameters()))
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
    j=0
    loss_history = []
    num_steps_range = trange(0, num_steps + 0, desc="Loss: 0.000 ")
    for i in num_steps_range:
        optimizer_design.zero_grad()
        # optimizer_flow.zero_grad()
        negMI = mi_loss_instance.differentiable_loss()
        # negMI.backward(retain_graph=True)
        # optimizer_design.step()
        if run_flow and train_flow:
            optimizer_flow.zero_grad()
            # # Log Likelihood Optimization
            # mi_loss_instance.hXY.backward()
            # # Foster Bound Optimization
            (mi_loss_instance.hX + mi_loss_instance.hXY-mi_loss_instance.hY).backward()
            # (2*mi_loss_instance.hXY-mi_loss_instance.hY).backward()
            optimizer_flow.step()
        
        
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
    lr_design,
    lr_flow,
    annealing_scheme,
):
    pyro.clear_param_store()
    theta_loc = torch.zeros((1, num_sources*p), device=device)
    theta_covmat = torch.eye(num_sources*p, device=device)
    flow_prior_theta = None
    flow_post_theta = None
    flow_obs = None
    train_flow = True
    prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)

    # sample true param
    true_theta = torch.tensor([[[-0.3281,  0.2271, -0.0320,  0.9442]]], device=device)#prior.sample(torch.Size([1]))#torch.tensor([[[-0.1306, -1.6480,  0.3361,  0.9620]]], device=device)#
    designs_so_far = []
    observations_so_far = []

    # Set posterior equal to the prior
    posterior_loc = theta_loc.reshape(-1)
    posterior_cov = torch.eye(p * num_sources, device=device)

    for t in range(0, T):
        t_start = time.time()
        print(f"Step {t + 1}/{T} of Run {run + 1}")
        pyro.clear_param_store()
        ho_model, mi_loss_instance, loss_history = optimise_design(
            posterior_loc,
            posterior_cov,
            flow_prior_theta,
            flow_post_theta,
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
            lr_design=lr_design,
            lr_flow=lr_flow,
            annealing_scheme=annealing_scheme,
        )
        
        with torch.no_grad():
            if t>0:
                trans_true_theta,_ = flow_post_theta.forward(true_theta[0])
            else:
                trans_true_theta = true_theta
            design, observation = ho_model.forward(theta=trans_true_theta)
            mux = mi_loss_instance.mu_post[:p * num_sources].detach()
            muy = mi_loss_instance.mu_post[p * num_sources:].detach()
            Sigmaxx = mi_loss_instance.Sigma_post[:p * num_sources,:p * num_sources].detach()
            Sigmaxy = mi_loss_instance.Sigma_post[:p * num_sources,p * num_sources:].detach()
            Sigmayy = mi_loss_instance.Sigma_post[p * num_sources:,p * num_sources:].detach()
            obs, _ = mi_loss_instance.gY.forward(observation[0])
            posterior_loc = (mux + torch.matmul(Sigmaxy,torch.linalg.solve(Sigmayy,(obs-muy))).flatten())
            max_posterior = mi_loss_instance.fX_post.reverse(posterior_loc)
            posterior_cov = (Sigmaxx-torch.matmul(Sigmaxy,torch.linalg.solve(Sigmayy,Sigmaxy.T)))
            flow_prior_theta = mi_loss_instance.fX_prior
            flow_post_theta = mi_loss_instance.fX_post
            flow_obs = mi_loss_instance.gY
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
            #     axs[0, 0].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            #     ######### Prior on source 2 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((true_theta[0][0][0].cpu().numpy()*np.ones(np.shape(X.flatten())),true_theta[0][0][1].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mux.cpu().numpy(), Sigmaxx.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[0, 1].pcolor(X, Y, Z)
            #     axs[0, 1].scatter(true_theta[0][0][2].cpu().numpy(),true_theta[0][0][3].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[0, 1].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
                
            #     ######### Posterior on source 1 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),true_theta[0][0][2].cpu().numpy()*np.ones(np.shape(X.flatten())),true_theta[0][0][3].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc.cpu().numpy(), posterior_cov.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[1, 0].pcolor(X, Y, Z)
            #     axs[1, 0].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][1].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[1, 0].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            #     ######### Posterior on source 2 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((true_theta[0][0][0].cpu().numpy()*np.ones(np.shape(X.flatten())),true_theta[0][0][1].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc.cpu().numpy(), posterior_cov.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[1, 1].pcolor(X, Y, Z)
            #     axs[1, 1].scatter(true_theta[0][0][2].cpu().numpy(),true_theta[0][0][3].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[1, 1].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            #     axs[0, 0].title.set_text('Source 1')
            #     axs[0, 1].title.set_text('Source 2')
            #     axs[0, 0].set(ylabel='Prior')
            #     axs[1, 0].set(ylabel='Posterior')
            #     plt.show()
            # ## Plot From Mean
            with torch.no_grad():
                import numpy as np
                import scipy
                import matplotlib.pyplot as plt
                x = np.linspace(-2.5,2.5,100)
                y = np.linspace(-2.5,2.5,100)
                X, Y = np.meshgrid(x, y)
                fig, axs = plt.subplots(2, 2)
                ######### Prior on source 1 ###########################################################
                fX, logJac = mi_loss_instance.fX_prior.forward(torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),mi_loss_instance.mu_prior[2].cpu().numpy()*np.ones(np.shape(X.flatten())),mi_loss_instance.mu_prior[3].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
                points = fX.reshape((100,100,4))
                Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mi_loss_instance.mu_prior.cpu().numpy(), mi_loss_instance.Sigma_prior.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
                axs[0, 0].pcolor(X, Y, Z)
                axs[0, 0].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][1].cpu().numpy(), color='red', marker='x',label = 'True')
                axs[0, 0].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
                ######### Prior on source 2 ###########################################################
                fX, logJac = mi_loss_instance.fX_prior.forward(torch.from_numpy((np.vstack((mi_loss_instance.mu_prior[0].cpu().numpy()*np.ones(np.shape(X.flatten())),mi_loss_instance.mu_prior[1].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float().to(device=device))
                points = fX.reshape((100,100,4))
                Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mi_loss_instance.mu_prior.cpu().numpy(), mi_loss_instance.Sigma_prior.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
                axs[0, 1].pcolor(X, Y, Z)
                axs[0, 1].scatter(true_theta[0][0][2].cpu().numpy(),true_theta[0][0][3].cpu().numpy(), color='red', marker='x',label = 'True')
                axs[0, 1].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
                
                ######### Posterior on source 1 ###########################################################
                fX, logJac = flow_post_theta.forward(torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),posterior_loc[2].cpu().numpy()*np.ones(np.shape(X.flatten())),posterior_loc[3].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
                points = fX.reshape((100,100,4))
                Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc.cpu().numpy(), posterior_cov.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
                axs[1, 0].pcolor(X, Y, Z)
                axs[1, 0].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][1].cpu().numpy(), color='red', marker='x',label = 'True')
                axs[1, 0].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
                ######### Posterior on source 2 ###########################################################
                fX, logJac = flow_post_theta.forward(torch.from_numpy((np.vstack((posterior_loc[0].cpu().numpy()*np.ones(np.shape(X.flatten())),posterior_loc[1].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float().to(device=device))
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
            #     axs[0, 0].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            #     ######### Prior on source 2 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((X.flatten(),mux[1].cpu().numpy()*np.ones(np.shape(X.flatten())),mux[2].cpu().numpy()*np.ones(np.shape(X.flatten())),Y.flatten())).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), mux.cpu().numpy(), Sigmaxx.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[0, 1].pcolor(X, Y, Z)
            #     axs[0, 1].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][3].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[0, 1].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
                
            #     ######### Posterior on source 1 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((posterior_loc[0].cpu().numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten(),posterior_loc[3].cpu().numpy()*np.ones(np.shape(X.flatten())))).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc.cpu().numpy(), posterior_cov.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[1, 0].pcolor(X, Y, Z)
            #     axs[1, 0].scatter(true_theta[0][0][1].cpu().numpy(),true_theta[0][0][2].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[1, 0].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            #     ######### Posterior on source 2 ###########################################################
            #     fX, logJac = flow_theta.forward(torch.from_numpy((np.vstack((X.flatten(),posterior_loc[1].cpu().numpy()*np.ones(np.shape(X.flatten())),posterior_loc[2].cpu().numpy()*np.ones(np.shape(X.flatten())),Y.flatten())).T)).float().to(device=device))
            #     points = fX.reshape((100,100,4))
            #     Z = scipy.stats.multivariate_normal.pdf(points.detach().cpu().numpy(), posterior_loc.cpu().numpy(), posterior_cov.cpu().numpy())*np.exp(logJac.reshape((100,100)).detach().cpu().numpy())
            #     axs[1, 1].pcolor(X, Y, Z)
            #     axs[1, 1].scatter(true_theta[0][0][0].cpu().numpy(),true_theta[0][0][3].cpu().numpy(), color='red', marker='x',label = 'True')
            #     axs[1, 1].scatter(design[0][0][0].detach().clone().cpu()[0],design[0][0][0].detach().clone().cpu()[1], color='green', marker='x',label = 'Design')
            #     axs[0, 0].title.set_text('Source 1')
            #     axs[0, 1].title.set_text('Source 2')
            #     axs[0, 0].set(ylabel='Prior')
            #     axs[1, 0].set(ylabel='Posterior')
            #     plt.show()
            
        t_end = time.time()
        run_time = t_end-t_start

        designs_so_far.append(design[0].detach().clone().cpu())
        observations_so_far.append(observation[0].cpu())
        
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

        print(designs_so_far)
        print(observations_so_far)
        print(f"Fit mean  = {max_posterior}, cov = {torch.diag(posterior_cov)}")
        print("True theta = ", true_theta.reshape(-1))
    print(f"Fitted posterior: mean = {max_posterior}, cov = {posterior_cov}")
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
    
    t = time.localtime()
    extra_data_id = time.strftime("%Y%m%d%H%M%S", t)
    path_to_extra_data = "./experiment_outputs/loc_fin/{}".format(extra_data_id)
    # if not os.path.exists(path_to_extra_data):
    #     os.makedirs(path_to_extra_data)
    # print(path_to_extra_data)

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
                            lr_design=lr_design,
                            lr_flow=lr_flow,
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
    parser.add_argument("--batch-size", default=1024, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--mlflow-experiment-name", default="locfin_mm_variational", type=str
    )
    parser.add_argument("--lr-design", default=.005, type=float)
    parser.add_argument("--lr-flow", default=.0005, type=float)
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