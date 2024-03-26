import torch
from torch import nn
import pyro
from pyro.nn import AutoRegressiveNN, DenseNN
from pyro import poutine
from pyro.poutine.util import prune_subsample_sites
import pyro.distributions.transforms as T
from neural.modules import LazyDelta
# import zuko
import time
# from numba import cuda
# import numba
from pyro.infer.util import torch_item
import copy
##############################################################################################
######################################### Flows ##############################################
##############################################################################################

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

class RealNVP(LazyNN):
    def __init__(self, dim_input, num_blocks=5, dim_hidden=256,device = 'cuda'): #'cuda''cpu'
        super().__init__(dim_input)
        
        self.dim_input = dim_input
        self.num_blocks = num_blocks
        self.dim_hidden = dim_hidden

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
            s = torch.tanh(s)
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
    
class SplineFlow(nn.Module):#LazyNN):
    def __init__(self, dim_input, n_flows=1,hidden_dims=[64], count_bins=8, bounds=4,order = 'linear',device = 'cuda'):
        # super().__init__(dim_input)
        super(SplineFlow, self).__init__()
        self.dim_input = dim_input
        self.countbins = count_bins
        self.bounds = bounds
        
        if dim_input == 1:
            self.spline_transform = T.Spline(dim_input, count_bins=count_bins, bound=bounds, order=order).to(device)
        else:
            self.spline_transform = spline_autoregressive1(dim_input,n_flows=n_flows,hidden_dims=hidden_dims, count_bins=count_bins, bound=bounds, order=order, device=device)

    def forward(self, x):
        z = self.spline_transform(x)
        logDet = self.spline_transform.log_abs_det_jacobian(x, z)
        return z, logDet
    
    def reverse(self, z):
        x = self.spline_transform.inv(z)
        # logDet = self.spline_transform.log_abs_det_jacobian(z, x)
        return x#, logDet
    
    # def forward(self, x):
    #     z = self.spline_transform.inv(x)
    #     logDet = self.spline_transform.log_abs_det_jacobian(x, z)
    #     return z, logDet
    
    # def reverse(self, z):
    #     x = self.spline_transform(z)
    #     # logDet = self.spline_transform.log_abs_det_jacobian(z, x)
    #     return x#, logDet
    
# @torch.compile
def spline_autoregressive1(input_dim, n_flows = 1, hidden_dims=None, count_bins=8, bound=4.0, order='linear',device = 'cuda'):
    if hidden_dims is None:
        hidden_dims = [64]

    if order=='quadratic':
        param_dims = [count_bins, count_bins, count_bins - 1]
    else:
        param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
        
    arns = nn.ModuleList([AutoRegressiveNN(input_dim,
            hidden_dims,
            param_dims=param_dims) for _ in range(n_flows)])
    
    #### Autoregressive Flows (Slow inverse, More Accurate)
    nfs = [T.SplineAutoregressive(input_dim, arns[0], count_bins=count_bins, bound=bound, order=order)]
    for i in range(n_flows-1):
        # nfs.append(T.Permute(torch.arange(input_dim, dtype=torch.long).flip(0).to(device)))
        nfs.append(T.SplineAutoregressive(input_dim, arns[i], count_bins=count_bins, bound=bound, order=order))

    
    # split_dim = input_dim // 2

    # nns = nn.ModuleList([DenseNN(
    #     split_dim,
    #     hidden_dims,
    #     param_dims=[
    #         (input_dim - split_dim) * count_bins,
    #         (input_dim - split_dim) * count_bins,
    #         (input_dim - split_dim) * (count_bins - 1),
    #         (input_dim - split_dim) * count_bins,
    #     ],
    # ) for _ in range(n_flows)])
    # #### Coupling Flows (Fast inverse, Less Accurate)
    # nfs = [T.SplineCoupling(input_dim, split_dim, nns[0], count_bins=count_bins, bound=bound, order=order)]
    # for i in range(n_flows-1):
    #     nfs.append(T.Permute(torch.arange(input_dim, dtype=torch.long).flip(0)))
    #     nfs.append(T.SplineCoupling(input_dim, split_dim, nns[0], count_bins=count_bins, bound=bound, order=order))
           
    return T.ComposeTransformModule(nfs).to(device)#T.ComposeTransform(nfs,cache_size=0)


def InitFlowToIdentity(dim, flow,bounds = 5,lr=.005, device = 'cpu'):
    ## takes in a flow and trains it to approximate the linear function, this is equivalent
    ## to the underlying flow being a Gaussian
    tol = 1e-2#1#
    optimizer = torch.optim.Adam(flow.spline_transform.parameters(),lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = .75,patience = 1)
    i = 0
    max_loss = 2*tol
    while max_loss>tol and i<10000:
        optimizer.zero_grad()
        sample = torch.distributions.Uniform(-bounds,bounds).sample((1024,dim)).to(device)
        flow_sample, log_det = flow.forward(sample)
        loss = torch.mean(torch.norm(torch.abs(sample-flow_sample),dim=1)) +torch.mean(torch.abs(log_det))
        loss.backward()
        optimizer.step()
        if i % 100 == 0 and not i == 0:
            scheduler.step(loss)
        with torch.no_grad():
            sample = torch.distributions.Uniform(-bounds,bounds).sample((1024,dim)).to(device)
            flow_sample, log_det = flow.forward(sample)
            max_loss = torch.max(torch.norm(torch.abs(sample-flow_sample),dim=1)) +torch.max(torch.abs(log_det))
            # print(max_loss)
        i+=1
    print(max_loss)
    return flow,max_loss


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

class MomentMatchMarginalPosterior(VariationalMutualInformationOptimizer):
    def __init__(self, model, batch_size, flow_x, flow_y,train_flow,device, **kwargs):
        super().__init__(
            model=model, batch_size=batch_size
        )
        self.mu = 0
        self.Sigma = 0
        self.dim_lat = 0
        self.dim_obs = 0
        self.hX = 0
        self.hX_Y = 0
        self.fX = flow_x
        self.gY = flow_y
        self.train_flow = train_flow
        self.pi_const = 2*torch.acos(torch.zeros(1, device=device))
        self.e_const = torch.exp(torch.tensor([1], device=device))
        self.grad_free_flow_x = None
        self.grad_free_flow_y = None
    
    def differentiable_loss(self, *args, **kwargs):
        # self.grad_free_flow_x = copy.deepcopy(self.fX)
        # self.grad_free_flow_y = copy.deepcopy(self.gY)
        if self.train_flow:
            if hasattr(self.fX, "parameters"):
                #! this is required for the pyro optimizer
                pyro.module("flow_x_net", self.fX)
            if hasattr(self.gY, "parameters"):
                #! this is required for the pyro optimizer
                pyro.module("flow_y_net", self.gY)

        latents, *history = self._get_data(args, kwargs)
        # latents, *history = self.model()

        self.dim_lat = latents.shape[1]
        self.dim_obs = history[0][1].shape[1]
        
        mufX, logDetJfX = self.fX.forward(latents)
        mugY, logDetJgY = self.gY.forward(history[0][1])

        data = torch.cat([mufX,mugY],axis=1)
        
        Sigma = torch.cov(data.T)#cov(data)+1e-5*torch.eye(self.dim_lat+self.dim_obs, device=latents.device)

        #################################################### Maximum Likelihood Bound ###############################################################
        sign, logdetS  = torch.linalg.slogdet(Sigma)
        if sign < 0:
            print("negative det")
        Loss = .5*logdetS +((self.dim_lat+self.dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX)-torch.mean(logDetJgY)
        # MI = 0
        if self.train_flow:
            if hasattr(self.fX, "spline_transform"):
                self.fX.spline_transform.requires_grad_(False)
            if hasattr(self.gY, "spline_transform"):
                self.gY.spline_transform.requires_grad_(False)
        sign, logdetSx  = torch.linalg.slogdet(Sigma[:self.dim_lat,:self.dim_lat])
        if sign < 0:
            print("negative det")
        hX = .5*logdetSx+(self.dim_lat/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX)
        
        sign, logdetSy  = torch.linalg.slogdet(Sigma[self.dim_lat:,self.dim_lat:])
        if sign < 0:
            print("negative det")
        hY = .5*logdetSy +(1/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJgY)

        MI = -hY#-hX
        if self.train_flow:
            if hasattr(self.fX, "spline_transform"):
                self.fX.spline_transform.requires_grad_(True)
            if hasattr(self.gY, "spline_transform"):
                self.gY.spline_transform.requires_grad_(True)
        ##################################################################################################################################
                
        ###################################### Traditional Bound #########################################################################        
        # sign, logdetSy  = torch.linalg.slogdet(Sigma[self.dim_lat:,self.dim_lat:])
        # if sign < 0:
        #     print("negative det")
        # hY = .5*logdetSy +(1/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJgY)
        
        # sign, logdetSx  = torch.linalg.slogdet(Sigma[:self.dim_lat,:self.dim_lat])
        # if sign < 0:
        #     print("negative det")
        # hX = .5*logdetSx+(self.dim_lat/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX)
        
        # sign, logdetS  = torch.linalg.slogdet(Sigma)
        # if sign < 0:
        #     print("negative det")
        # Loss = .5*logdetS +((self.dim_lat+self.dim_obs)/2)*(torch.log(2*self.pi_const*self.e_const))-torch.mean(logDetJfX)-torch.mean(logDetJgY)-hY+hX
        # if self.train_flow:
        #     if hasattr(self.fX, "spline_transform"):
        #         self.fX.spline_transform.requires_grad_(False)
        #     if hasattr(self.gY, "spline_transform"):
        #         self.gY.spline_transform.requires_grad_(False)
        # MI = -2*hX
        # if self.train_flow:
        #     if hasattr(self.fX, "spline_transform"):
        #         self.fX.spline_transform.requires_grad_(True)
        #     if hasattr(self.gY, "spline_transform"):
        #         self.gY.spline_transform.requires_grad_(True)
        ##########################################################################################################################
        
        # save optimal parameters for decision
        self.mu = torch.mean(data,axis=0)
        self.Sigma = Sigma
        # hXY = self.compute_joint_entropy(latents,history[0][1])
        # hX, hY = self.compute_marginal_entropies(latents,history[0][1])
        
        return MI+Loss

    def loss(self, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using the BA lower bound == -EIG
        """
        loss_to_constant = torch_item(self.differentiable_loss(*args, **kwargs))
        return loss_to_constant

##############################################################################################
##############################################################################################

class MomentMatchPosterior(VariationalMutualInformationOptimizer):
    def __init__(self, model, batch_size, flow_x, flow_y,train_flow,device, **kwargs):
        super().__init__(
            model=model, batch_size=batch_size
        )
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
            if hasattr(self.fX, "parameters"):
                #! this is required for the pyro optimizer
                pyro.module("flow_x_net", self.fX)
            if hasattr(self.gY, "parameters"):
                #! this is required for the pyro optimizer
                pyro.module("flow_y_net", self.gY)
        # sample from design
        latents, *history = self._get_data(args, kwargs)
        
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
        return self.hX+self.hX_Y+hY_X+hY-torch.detach(2*self.hX_Y+hY_X+hY)

    def loss(self, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using the BA lower bound == -EIG
        """
        return self.hX-self.hX_Y