from typing import Optional, Tuple, Callable, List, Iterable
from jaxtyping import Array
import numpy as np
from abc import abstractmethod, abstractproperty

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pydantic
from numpy.typing import ArrayLike

import bmi.estimators.neural._estimators as _estimators
from bmi.estimators.neural._training_log import TrainingLog
from bmi.estimators.neural._types import BatchedPoints
from bmi.interface import BaseModel, EstimateResult, IMutualInformationPointEstimator
from bmi.utils import ProductSpace
from bmi.estimators.neural._basic_training import get_batch
from functools import partial

from flowjax.distributions import Normal, Transformed
from flowjax.flows import masked_autoregressive_flow
from flowjax.bijections import RationalQuadraticSpline

import matplotlib.pyplot as plt


class NFModel(eqx.Module):
    @abstractmethod
    def __init__(self):
        return NotImplemented

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: Array) -> Tuple[Array, Array]:
        return NotImplemented

    @abstractmethod
    def inverse(self, x: Array) -> Tuple[Array, Array]:
        return NotImplemented


    
class SplineFlow(eqx.Module):
    spline_transform: List
    def __init__(self, 
                 dim_input:int, 
                 n_flows:int, 
                 nn_width:int, 
                 nn_depth:int , 
                 knots:int, 
                 interval:int, 
                 key: jax.random.PRNGKeyArray, **kwargs):#

        # self.spline_transform = [MaskedAutoregressive(
        #     key=key,
        #     transformer=RationalQuadraticSpline(knots=knots,interval=interval),
        #     dim=dim_input,
        #     cond_dim=None,#dim_input//2,#
        #     nn_width=nn_width,
        #     nn_depth=nn_depth,
        #     nn_activation=jax.nn.relu,
        # )]
        
        self.spline_transform = [masked_autoregressive_flow(key,
                                base_dist=Normal(jnp.zeros(dim_input)),
                                transformer=RationalQuadraticSpline(knots=knots, 
                                                                    interval=interval),
                                flow_layers=n_flows,
                                nn_width= nn_width,
                                nn_depth=nn_depth,
                                invert = False,
                                ).bijection]
       
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        return self.forward(x)

    def forward(self, x: Array) -> Tuple[Array, Array]:
        # z_norm = jax.nn.sigmoid(x)
        # z = 2*z_norm-1
        # z, log_det_df = self.spline_transform[0].transform_and_log_det(z)
        # log_det_df += jnp.log(jnp.abs(2*jnp.sum(jax.nn.sigmoid(z)*(1-jax.nn.sigmoid(z)),axis=0)))
        # # log_det_df += jnp.log(jnp.abs(2*jnp.sum(z_norm*(1-z_norm),axis=0)))
        z, log_det_df = self.spline_transform[0].transform_and_log_det(x)  
        return z, log_det_df
    
    def inverse(self, z: Array) -> Tuple[Array, Array]:
        x, log_det_df = self.spline_transform[0].inverse_and_log_det(z)
        return x, log_det_df,

class IdentityFlow(eqx.Module):
    def __init__(self, **kwargs):
        hold = 1        
       
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        return self.forward(x)

    def forward(self, x: Array) -> Tuple[Array, Array]:
        z = x 
        log_det_df = jnp.zeros(len(x))
        return z, log_det_df
    
    def inverse(self, z: Array) -> Tuple[Array, Array]:
        x = z 
        log_det_df = jnp.zeros(len(z))
        return x, log_det_df,


class CriticParams(eqx.Module):
    obs_encoder_network: List
    head_layer_mean: List
    head_layer_sd: List
    dropout_mean: eqx.nn.Dropout
    dropout_sd: eqx.nn.Dropout
    dropout_encoder: eqx.nn.Dropout
    dropout_key: jax.random.PRNGKeyArray
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        key: jax.random.PRNGKeyArray
    ):
        # keys = jax.random.split(key, 6)
        # self.obs_encoder_network = [eqx.nn.Linear(dim_y, 512,key=keys[0]),eqx.nn.Linear(512, 2*dim_x,key=keys[1])]
        # self.head_layer_mean = [eqx.nn.Linear(2*dim_x, 512,key=keys[2]), eqx.nn.Linear(512, dim_x,key=keys[3])]
        # self.head_layer_sd = [eqx.nn.Linear(2*dim_x, 512,key=keys[4]),eqx.nn.Linear(512, dim_x,key=keys[5])]
        
        # keys = jax.random.split(key, 6)
        # self.obs_encoder_network = [eqx.nn.Linear(dim_y, 512,key=keys[0]),eqx.nn.Linear(512, 2*dim_x,key=keys[1])]
        # self.head_layer_mean = [eqx.nn.Linear(dim_y, 512,key=keys[2]), eqx.nn.Linear(512, dim_x,key=keys[3])]
        # self.head_layer_sd = [eqx.nn.Linear(dim_y, 512,key=keys[4]),eqx.nn.Linear(512, dim_x,key=keys[5])]
        key_layers, key_dropout = jax.random.split(key, 2)
        keys = jax.random.split(key_layers, 9)
        self.obs_encoder_network = [eqx.nn.Linear(dim_y, 64,key=keys[0]),eqx.nn.Linear(64, 512,key=keys[1]),eqx.nn.Linear(512, 2*dim_x,key=keys[2])]
        self.head_layer_mean = [eqx.nn.Linear(dim_y, 512,key=keys[3]),eqx.nn.Linear(512, 512,key=keys[4]), eqx.nn.Linear(512, dim_x,key=keys[5])]
        self.head_layer_sd = [eqx.nn.Linear(dim_y, 512,key=keys[6]),eqx.nn.Linear(512, 512,key=keys[7]),eqx.nn.Linear(512, dim_x,key=keys[8])]
        self.dropout_mean = eqx.nn.Dropout(0.5)
        self.dropout_sd = eqx.nn.Dropout(0.5)
        self.dropout_encoder = eqx.nn.Dropout(0.5)
        self.dropout_key = key_dropout
        # keys = jax.random.split(key, 12)
        # self.obs_encoder_network = [eqx.nn.Linear(dim_y, 64,key=keys[0]),eqx.nn.Linear(64, 512,key=keys[1]),eqx.nn.Linear(512, 2*dim_x,key=keys[2]),eqx.nn.Linear(2*dim_x, 64,key=keys[9]),eqx.nn.Linear(64, 512,key=keys[10]),eqx.nn.Linear(512, 2*dim_x,key=keys[11])]
        # self.head_layer_mean = [eqx.nn.Linear(dim_y, 512,key=keys[3]),eqx.nn.Linear(512, 512,key=keys[4]), eqx.nn.Linear(512, dim_x,key=keys[5])]
        # self.head_layer_sd = [eqx.nn.Linear(dim_y, 512,key=keys[6]),eqx.nn.Linear(512, 512,key=keys[7]),eqx.nn.Linear(512, dim_x,key=keys[8])]
    def __set_key__(self,key):
        object.__setattr__(self, 'dropout_key', key)   
           
    def get_variational_params(self, y: Array) -> Tuple[Array, Array]:
        # for layer in self.obs_encoder_network[:-1]:
        #         y = jax.nn.relu(layer(y))
        # y_encoding = self.obs_encoder_network[-1](y)
        # mean = self.head_layer_mean[1](jax.nn.relu(self.head_layer_mean[0](y_encoding)))
        # sd = 1e-5 + jax.nn.softplus(self.head_layer_sd[1](jax.nn.relu(self.head_layer_sd[0](y_encoding))))
        key_mean, key_sd, key_encoder = jax.random.split(self.dropout_key, 3)
        key_encoders = jax.random.split(key_encoder, len(self.obs_encoder_network[:-1]))
        key_means = jax.random.split(key_mean, len(self.head_layer_mean[1:-1])+1)
        key_sds = jax.random.split(key_sd, len(self.head_layer_sd[1:-1])+1)
        # i=0
        # for layer in self.obs_encoder_network[:-1]:
        #         y = jax.nn.relu(layer(y))
        #         # y = self.dropout_encoder(y,key = key_encoders[i])
        #         # i+=1
        # y_encoding = self.obs_encoder_network[-1](y)
        y_encoding = y
        mean = jax.nn.relu(self.head_layer_mean[0](y_encoding))
        mean = self.dropout_mean(mean,key = key_means[0])
        i=1
        for layer in self.head_layer_mean[1:-1]:
            mean = jax.nn.relu(layer(mean))
            mean = self.dropout_mean(mean,key = key_means[i])
            i+=1
        # mean = self.dropout_mean(mean,key = key_mean)
        mean = self.head_layer_mean[-1](mean)
        sd = jax.nn.relu(self.head_layer_sd[0](y_encoding))
        sd = self.dropout_sd(sd,key = key_sds[0])
        i=1
        for layer in self.head_layer_sd[1:-1]:
            sd = jax.nn.relu(layer(sd))
            sd = self.dropout_sd(sd,key = key_sds[i])
            i+=1
        # sd = self.dropout_sd(sd,key = key_sd)
        sd = 1e-5+jax.nn.softplus(self.head_layer_sd[-1](sd))
        # sd = self.head_layer_sd[-1](sd)
        return mean, sd

    def forward(self, x: Array, y: Array) -> Array:
        x_flat = x#.flatten(-2)
        mean, sd = self.get_variational_params(y)
        log_probs_q = (jax.scipy.stats.norm.logpdf(x_flat,loc=mean, scale=sd).sum(axis=-1))#jax.scipy.stats.multivariate_normal.logpdf(x_flat,mean=mean, cov=jax.numpy.diag(sd))#(jax.scipy.stats.norm.logpdf(x_flat,loc=mean, scale=sd).sum(axis=-1))
        return log_probs_q
#####################################################################################################################################
@eqx.filter_jit
def _neural_variational_value(
    model: eqx.Module,
    xs: jnp.ndarray,
    ys: jnp.ndarray):
    n_sample, dim_x = xs.shape
    toss, dim_y = ys.shape
    fX_prior_vmap = jax.vmap(model._fX_prior, in_axes=(0,))
    fX_post_vmap = jax.vmap(model._fX_post, in_axes=(0,))
    critic_vmap = jax.vmap(model._critic_net.forward, in_axes=(0,0))
    xs_prior, logDetJfX_prior = fX_prior_vmap(xs)
    xs_post, logDetJfX_post = fX_post_vmap(xs)
    # xs_prior = xs
    # xs_post = xs
    log_probs_q = critic_vmap(xs_post,ys)
    
    hX_post = -log_probs_q.mean()-logDetJfX_post.mean()
    
    Sigma_prior = jnp.cov(xs_prior.T) + 1e-5*jnp.eye(dim_x)
    sign, logdetSx_prior  = jnp.linalg.slogdet(Sigma_prior)
    hX_prior = 0.5 * logdetSx_prior + dim_x / 2 * (1 + jnp.log(2 * jnp.pi)) - logDetJfX_prior.mean() 
    return hX_prior-hX_post

# @eqx.filter_value_and_grad
# @jax.value_and_grad
@eqx.filter_value_and_grad
def _neural_variational_loss_function(
    model: eqx.Module,
    xs: jnp.ndarray,
    ys: jnp.ndarray):

    n_sample, dim_x = xs.shape
    _, dim_y = ys.shape
    fX_prior_vmap = jax.vmap(model._fX_prior, in_axes=(0,))
    fX_post_vmap = jax.vmap(model._fX_post, in_axes=(0,))
    critic_vmap = jax.vmap(model._critic_net.forward, in_axes=(0,0))
    xs_prior, logDetJfX_prior = fX_prior_vmap(xs)
    xs_post, logDetJfX_post = fX_post_vmap(xs)
    # xs_prior = xs
    # xs_post = xs
    log_probs_q = critic_vmap(xs_post,ys)
    
    hX_post = -log_probs_q.mean()-logDetJfX_post.mean()
    
    Sigma_prior = jnp.cov(xs_prior.T) + 1e-5*jnp.eye(dim_x)
    sign, logdetSx_prior  = jnp.linalg.slogdet(Sigma_prior)
    hX_prior = 0.5 * logdetSx_prior + dim_x / 2 * (1 + jnp.log(2 * jnp.pi)) - logDetJfX_prior.mean() 
    return hX_post+hX_prior#+jax.lax.stop_gradient(-2*hX_post)


def _neural_variational_value_neg_grad(
    model: eqx.Module,
    xs: jnp.ndarray,
    ys: jnp.ndarray):
    value, neg_grad = _neural_variational_loss_function(model, xs, ys)
    return value, neg_grad
    
def neural_variational_training(
    rng: jax.random.PRNGKeyArray,
    model: eqx.Module,
    xs: BatchedPoints,
    ys: BatchedPoints,
    xs_test: Optional[BatchedPoints] = None,
    ys_test: Optional[BatchedPoints] = None,
    batch_size: Optional[int] = 256,
    test_every_n_steps: int = 250,
    max_n_steps: int = 2_000,
    early_stopping: bool = False,
    learning_rate: float = 0.1,
    verbose: bool = False
) -> tuple[TrainingLog, eqx.Module]:

    xs_test = xs_test if xs_test is not None else xs
    ys_test = ys_test if ys_test is not None else ys

    # # initialize the optimizer
    # optimizer = optax.adam(learning_rate=learning_rate)
    
    # Exponential decay of the learning rate.
    scheduler = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=500,
        decay_rate=0.9)

    # Combining gradient transforms using `optax.chain`.
    optimizer = optax.chain(
        # optax.clip_by_global_norm(2.0),  # Clip by the gradient by the global norm.
        optax.scale_by_adam(),  # Use the updates from adam.
        optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
        # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
        optax.scale(-1.0)
    )
    
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    # cpus = jax.devices("cpu")
    # gpus = jax.devices("gpu")
    # compile the training step   flows.affine_coupling_g.base_dist.scale
    @eqx.filter_jit#(device=gpus[0])
    def step(
        *,
        model,
        opt_state,
        xs: BatchedPoints,
        ys: BatchedPoints):
        
        value, neg_grad = _neural_variational_value_neg_grad(
            model,
            xs=xs,
            ys=ys)      
        updates, opt_state = optimizer.update(neg_grad, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, value
    
    ## Compile Model So Timing is accurate
    model1, opt_state1, mi_train1 = step(
            model = model,
            opt_state = opt_state,
            xs=jnp.ones((batch_size,xs.shape[1])),
            ys=jnp.ones((batch_size,ys.shape[1])))
    
    model1, opt_state1, mi_train1 = step(
            model = model1,
            opt_state = opt_state1,
            xs=jnp.ones((batch_size,xs.shape[1])),
            ys=jnp.ones((batch_size,ys.shape[1])))
    
    mi_test1 = _neural_variational_value(
                model = model1,
                xs=xs_test, 
                ys=ys_test
                )
    
    # main training loop
    training_log = TrainingLog(
        max_n_steps=max_n_steps, 
        early_stopping=early_stopping, 
        verbose=verbose)
    
    keys = jax.random.split(rng, max_n_steps)
    for n_step, key in enumerate(keys, start=1):
        key_sample, key_test = jax.random.split(key)
        # model._critic_net.key_dropout = key_test
        model._critic_net.__set_key__(key_test)
        # sample
        batch_xs, batch_ys = get_batch(xs, ys, key_sample, batch_size)

        # run step
        model, opt_state, mi_train = step(
            model = model,
            opt_state = opt_state,
            xs=batch_xs,
            ys=batch_ys)

        # logging train
        training_log.log_train_mi(n_step, mi_train)

        # logging test
        if n_step % test_every_n_steps == 0 or n_step==1:
            model = eqx.nn.inference_mode(model)
            mi_test = _neural_variational_value(
                model = model,
                xs=xs_test, 
                ys=ys_test
                )
            # test_loss, _ = _neural_variational_loss_function(
            #     model = model,
            #     xs=xs_test, 
            #     ys=ys_test
            #     )
            training_log.log_test_mi(n_step, mi_test)
            model = eqx.nn.inference_mode(model, value=False)
            
        # early stop?
        if training_log.early_stop():
            break

    training_log.finish()
    return training_log, model

class NeuralVariationalParams(BaseModel):
    batch_size: pydantic.PositiveInt
    max_n_steps: pydantic.PositiveInt
    train_test_split: Optional[pydantic.confloat(gt=0.0, lt=1.0)]
    test_every_n_steps: pydantic.PositiveInt
    learning_rate: pydantic.PositiveFloat
    dim_x: pydantic.PositiveInt
    dim_y: pydantic.PositiveInt
    use_flow: bool
    flow_layers: pydantic.PositiveInt
    nn_width: pydantic.PositiveInt
    nn_depth: pydantic.PositiveInt
    knots: pydantic.PositiveInt
    interval: pydantic.PositiveInt
    standardize: bool
    seed: int
    
class NeuralVariationalModel(eqx.Module):
    _fX_prior: eqx.Module
    _fX_post: eqx.Module
    _critic_net: eqx.Module
    def __init__(
        self,
        dim_x:int, 
        dim_y:int,
        use_flow: bool,
        n_flows:int, 
        nn_width:int, 
        nn_depth:int , 
        knots:int, 
        interval:int, 
        key: jax.random.PRNGKeyArray
    ) -> None:
        key_init, _ = jax.random.split(key, 2)
        key_init_prior, key_init_post, key_init_critic = jax.random.split(key_init, 3)
        if use_flow:
            self._fX_prior = SplineFlow(dim_input=dim_x, n_flows=n_flows, nn_width=nn_width, nn_depth=nn_depth , knots=knots, interval=interval, key = key_init_prior)
            self._fX_post = SplineFlow(dim_input=dim_x, n_flows=n_flows, nn_width=nn_width, nn_depth=nn_depth , knots=knots, interval=interval, key = key_init_post)
        else:
            self._fX_prior = IdentityFlow()
            self._fX_post = IdentityFlow()
        self._critic_net = CriticParams(dim_x=dim_x,dim_y=dim_y,key=key_init_critic)


class NeuralVariationalEstimator(IMutualInformationPointEstimator):
    def __init__(
        self,
        batch_size: int =  _estimators._DEFAULT_BATCH_SIZE,
        max_n_steps: int = _estimators._DEFAULT_N_STEPS,
        train_test_split: Optional[float] = _estimators._DEFAULT_TRAIN_TEST_SPLIT,
        test_every_n_steps: int = _estimators._DEFAULT_TEST_EVERY_N,
        learning_rate: float = _estimators._DEFAULT_LEARNING_RATE,
        dim_x: int = 1,
        dim_y: int =1,
        use_flow: bool = False,
        flow_layers: int = 1,
        nn_width: int = 32,#64
        nn_depth: int = 2,
        knots: int = 128,#128
        interval: int = 8,#10
        standardize: bool = _estimators._DEFAULT_STANDARDIZE,
        verbose: bool = _estimators._DEFAULT_VERBOSE,
        seed: int = _estimators._DEFAULT_SEED,
    ) -> None:
        self._params = NeuralVariationalParams(
            batch_size=batch_size,
            max_n_steps=max_n_steps,
            train_test_split=train_test_split,
            test_every_n_steps=test_every_n_steps,
            learning_rate=learning_rate,
            standardize=standardize,
            seed=seed,
            dim_x = dim_x,
            dim_y = dim_y,
            use_flow=use_flow,
            flow_layers = flow_layers, 
            nn_width = nn_width,
            nn_depth = nn_depth,
            knots = knots,
            interval = interval)
        
        self._verbose = verbose
        self._training_log: Optional[TrainingLog] = None

        key = jax.random.PRNGKey(self._params.seed)
        self.variational_model = NeuralVariationalModel(
            dim_x=self._params.dim_x, 
            dim_y=self._params.dim_y,
            use_flow=self._params.use_flow,
            n_flows=self._params.flow_layers, 
            nn_width=self._params.nn_width, 
            nn_depth=self._params.nn_depth, 
            knots=self._params.knots, 
            interval=self._params.interval, 
            key=key)


    def parameters(self) -> NeuralVariationalParams:
        return self._params

    def estimate_with_info(self, x: ArrayLike, y: ArrayLike) -> EstimateResult:
        key = jax.random.PRNGKey(self._params.seed)
        _ , key_estimate = jax.random.split(key, 2)
        key_init, key_split, key_fit = jax.random.split(key_estimate, 3)

        # standardize the data, note we do so before splitting into train/test
        space = ProductSpace(x, y, standardize=self._params.standardize)
        xs, ys = jnp.asarray(space.x), jnp.asarray(space.y)
        
        # split
        xs_train, xs_test, ys_train, ys_test = _estimators.train_test_split(
            xs, ys, train_size=self._params.train_test_split, key=key_split
        )

        training_log, model = neural_variational_training(
            rng= key_fit,
            model = self.variational_model,
            xs=xs_train,
            ys=ys_train,
            xs_test=xs_test,
            ys_test=ys_test,
            batch_size=self._params.batch_size,
            test_every_n_steps=self._params.test_every_n_steps,
            max_n_steps=self._params.max_n_steps,
            learning_rate=self._params.learning_rate,
            verbose=self._verbose,
        )
        self.variational_model = model
        
        return EstimateResult(
            mi_estimate=training_log.final_mi,
            additional_information=training_log.additional_information,
        )

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        return self.estimate_with_info(x,y).mi_estimate
    
    def plot_estimator(
        self,
        x: ArrayLike,
        y: ArrayLike):
        space = ProductSpace(x, y, standardize=self._params.standardize)
        xs, ys = jnp.asarray(space.x), jnp.asarray(space.y)
        fX_vmap = jax.vmap(self.variational_model._fX_prior, in_axes=(0,))
        fXs, logDetJfX = fX_vmap (xs)
        Mu = jnp.mean(fXs,axis=0)
        Sigma = jnp.cov(fXs.T)
        
        n_sample, dim_x = xs.shape

        if dim_x == 2:
            X = np.random.multivariate_normal(Mu[:dim_x],Sigma[:dim_x,:dim_x],2500)
            fX_inv_vmap = jax.vmap(self.variational_model._fX_prior.inverse, in_axes=(0,))
            fX, logDetJfX = fX_inv_vmap(X)
            
            plt.subplot(1, 2, 1) # row 1, col 2 index 1
            plt.scatter(xs[:,0], xs[:,1])
            plt.title("True Samples")
            plt.xlabel('X1')
            plt.ylabel('X2')

            plt.subplot(1, 2, 2) # index 2
            plt.scatter(fX[:,0], fX[:,1])
            plt.title("Flow Samples")
            plt.xlabel('X1')
            # plt.ylabel('X2')

            plt.show()
##################################################################################################################################
@eqx.filter_jit
def _joint_variational_value(
    model: eqx.Module,
    xs: jnp.ndarray,
    ys: jnp.ndarray):
    n_sample, dim_x = xs.shape
    _, dim_y = ys.shape
    fX_vmap = jax.vmap(model._fX, in_axes=(0,))
    gY_vmap = jax.vmap(model._gY, in_axes=(0,))
    fXs, logDetJfX = fX_vmap (xs)
    gYs, logDetJgY = gY_vmap(ys)
    fXs_gYs = jnp.concatenate((fXs, gYs), axis=1)
    
    Sigma = jnp.cov(fXs_gYs.T)
    
    sign, logdetS  = jnp.linalg.slogdet(Sigma)
    hXY = 0.5 * logdetS + (dim_x+dim_y) / 2 * (1 + jnp.log(2 * jnp.pi)) - logDetJfX.mean() - logDetJgY.mean()
    
    sign, logdetSx  = jnp.linalg.slogdet(Sigma[:dim_x, :dim_x])
    hX = 0.5 * logdetSx + dim_x / 2 * (1 + jnp.log(2 * jnp.pi)) - logDetJfX.mean()#(1 / n_sample) * jnp.sum(logDetJfX)
    
    sign, logdetSy  = jnp.linalg.slogdet(Sigma[dim_x:, dim_x:])
    hY = 0.5 * logdetSy + dim_y / 2 * (1 + jnp.log(2 * jnp.pi)) - logDetJgY.mean()
    return hX+hY-hXY

@eqx.filter_value_and_grad
def _joint_variational_loss_function(
    model: eqx.Module,
    xs: jnp.ndarray,
    ys: jnp.ndarray):
    n_sample, dim_x = xs.shape
    _, dim_y = ys.shape
    fX_vmap = jax.vmap(model._fX, in_axes=(0,))
    gY_vmap = jax.vmap(model._gY, in_axes=(0,))
    fXs, logDetJfX = fX_vmap (xs)
    gYs, logDetJgY = gY_vmap(ys)
    fXs_gYs = jnp.concatenate((fXs, gYs), axis=1)
    
    Sigma = jnp.cov(fXs_gYs.T)
    
    sign, logdetS  = jnp.linalg.slogdet(Sigma)
    hXY = 0.5 * logdetS + (dim_x+dim_y) / 2 * (1 + jnp.log(2 * jnp.pi)) - logDetJfX.mean() - logDetJgY.mean()
    
    sign, logdetSx  = jnp.linalg.slogdet(Sigma[:dim_x, :dim_x])
    hX = 0.5 * logdetSx + dim_x / 2 * (1 + jnp.log(2 * jnp.pi)) - logDetJfX.mean()#(1 / n_sample) * jnp.sum(logDetJfX)
    
    sign, logdetSy  = jnp.linalg.slogdet(Sigma[dim_x:, dim_x:])
    hY = 0.5 * logdetSy + dim_y / 2 * (1 + jnp.log(2 * jnp.pi)) - logDetJgY.mean()
    return hXY+jax.lax.stop_gradient(hX+hY-2*hXY)#+hX-hY#+jax.lax.stop_gradient(-2*hX_post)

def _joint_variational_value_neg_grad(
    model: eqx.Module,
    xs: jnp.ndarray,
    ys: jnp.ndarray):
    value, neg_grad = _joint_variational_loss_function(model, xs, ys)
    return value, neg_grad
    
def joint_variational_flow_training(
    rng: jax.random.PRNGKeyArray,
    model: eqx.Module,
    xs: BatchedPoints,
    ys: BatchedPoints,
    xs_test: Optional[BatchedPoints] = None,
    ys_test: Optional[BatchedPoints] = None,
    batch_size: Optional[int] = 256,
    test_every_n_steps: int = 250,
    max_n_steps: int = 2_000,
    early_stopping: bool = False,
    learning_rate: float = 0.1,
    verbose: bool = False
) -> tuple[TrainingLog, eqx.Module]:

    xs_test = xs_test if xs_test is not None else xs
    ys_test = ys_test if ys_test is not None else ys

    # # initialize the optimizer
    # optimizer = optax.adam(learning_rate=learning_rate)
    
    # Exponential decay of the learning rate.
    scheduler = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=500,
        decay_rate=0.9)

    # Combining gradient transforms using `optax.chain`.
    optimizer = optax.chain(
        # optax.clip_by_global_norm(5.0),  # Clip by the gradient by the global norm.
        optax.scale_by_adam(),  # Use the updates from adam.
        optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
        # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
        optax.scale(-1.0)
    )
    
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # compile the training step   flows.affine_coupling_g.base_dist.scale
    @eqx.filter_jit
    def step(
        *,
        model,
        opt_state,
        xs: BatchedPoints,
        ys: BatchedPoints):
        
        value, neg_grad = _joint_variational_value_neg_grad(
            model,
            xs=xs,
            ys=ys)
        updates, opt_state = optimizer.update(neg_grad, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, value
    
    ## Compile Model So Timing is accurate
    model1, opt_state1, mi_train1 = step(
            model = model,
            opt_state = opt_state,
            xs=jnp.ones((batch_size,xs.shape[1])),
            ys=jnp.ones((batch_size,ys.shape[1])))
    
    model1, opt_state1, mi_train1 = step(
            model = model1,
            opt_state = opt_state1,
            xs=jnp.ones((batch_size,xs.shape[1])),
            ys=jnp.ones((batch_size,ys.shape[1])))
    
    mi_test1 = _joint_variational_value(
                model = model1,
                xs=xs_test, 
                ys=ys_test
                )
        
    # main training loop
    training_log = TrainingLog(
        max_n_steps=max_n_steps, 
        early_stopping=early_stopping, 
        verbose=verbose)
    
    keys = jax.random.split(rng, max_n_steps)
    for n_step, key in enumerate(keys, start=1):
        key_sample, key_test = jax.random.split(key)

        # sample
        batch_xs, batch_ys = get_batch(xs, ys, key_sample, batch_size)

        # run step
        model, opt_state, mi_train = step(
            model = model,
            opt_state = opt_state,
            xs=batch_xs,
            ys=batch_ys)

        # logging train
        training_log.log_train_mi(n_step, mi_train)

        # logging test
        if n_step % test_every_n_steps == 0 or n_step==1:
            mi_test = _joint_variational_value(
                model = model,
                xs=xs_test, 
                ys=ys_test
                )
            training_log.log_test_mi(n_step, mi_test)
            
        # early stop?
        if training_log.early_stop():
            break

    training_log.finish()
    return training_log, model

def joint_variational_gauss_training(
    model: eqx.Module,
    xs: BatchedPoints,
    ys: BatchedPoints,
    xs_test: Optional[BatchedPoints] = None,
    ys_test: Optional[BatchedPoints] = None,
    max_n_steps: int = 2_000,
    early_stopping: bool = False,
    verbose: bool = False,
) -> tuple[TrainingLog, eqx.Module]:

    xs_test = xs_test if xs_test is not None else xs
    ys_test = ys_test if ys_test is not None else ys
    
    training_log = TrainingLog(
        max_n_steps=max_n_steps, 
        early_stopping=early_stopping, 
        verbose=verbose)
    
    mi_test = _joint_variational_value(
        model = model,
        xs=xs_test, 
        ys=ys_test
        )
    training_log.log_test_mi(0, mi_test)

    training_log.finish()
    return training_log, model
    
class JointVariationalParams(BaseModel):
    batch_size: pydantic.PositiveInt
    max_n_steps: pydantic.PositiveInt
    train_test_split: Optional[pydantic.confloat(gt=0.0, lt=1.0)]
    test_every_n_steps: pydantic.PositiveInt
    learning_rate: pydantic.PositiveFloat
    dim_x: pydantic.PositiveInt
    dim_y: pydantic.PositiveInt
    use_flow: bool
    flow_layers: pydantic.PositiveInt
    nn_width: pydantic.PositiveInt
    nn_depth: pydantic.PositiveInt
    knots: pydantic.PositiveInt
    interval: pydantic.PositiveInt
    standardize: bool
    seed: int
    
class JointVariationalModel(eqx.Module):
    _fX: eqx.Module
    _gY: eqx.Module
    def __init__(
        self,
        dim_x:int, 
        dim_y:int,
        use_flow: bool,
        n_flows:int, 
        nn_width:int, 
        nn_depth:int , 
        knots:int, 
        interval:int, 
        key: jax.random.PRNGKeyArray
    ) -> None:
        key_init, _ = jax.random.split(key, 2)
        key_init_fX, key_init_gY = jax.random.split(key_init, 2)
        if use_flow:
            self._fX = SplineFlow(dim_input=dim_x, n_flows=n_flows, nn_width=nn_width, nn_depth=nn_depth , knots=knots, interval=interval, key = key_init_fX)
            self._gY = SplineFlow(dim_input=dim_y, n_flows=n_flows, nn_width=nn_width, nn_depth=nn_depth , knots=knots, interval=interval, key = key_init_gY)
        else:
            self._fX = IdentityFlow()
            self._gY = IdentityFlow()
    
class JointVariationalEstimator(IMutualInformationPointEstimator):
    def __init__(
        self,
        batch_size: int =  _estimators._DEFAULT_BATCH_SIZE,
        max_n_steps: int = _estimators._DEFAULT_N_STEPS,
        train_test_split: Optional[float] = _estimators._DEFAULT_TRAIN_TEST_SPLIT,
        test_every_n_steps: int = _estimators._DEFAULT_TEST_EVERY_N,
        learning_rate: float = _estimators._DEFAULT_LEARNING_RATE,
        dim_x: int = 1,
        dim_y: int =1,
        use_flow: bool = False,
        flow_layers: int = 1,
        nn_width: int = 32,
        nn_depth: int = 2,
        knots: int = 128,
        interval: int = 8,
        standardize: bool = _estimators._DEFAULT_STANDARDIZE,
        verbose: bool = _estimators._DEFAULT_VERBOSE,
        seed: int = _estimators._DEFAULT_SEED,
    ) -> None:
        self._params = JointVariationalParams(
            batch_size=batch_size,
            max_n_steps=max_n_steps,
            train_test_split=train_test_split,
            test_every_n_steps=test_every_n_steps,
            learning_rate=learning_rate,
            standardize=standardize,
            seed=seed,
            use_flow=use_flow,
            dim_x = dim_x,
            dim_y = dim_y,
            flow_layers = flow_layers, 
            nn_width = nn_width,
            nn_depth = nn_depth,
            knots = knots,
            interval = interval)
        
        self._verbose = verbose
        self._training_log: Optional[TrainingLog] = None
        key = jax.random.PRNGKey(self._params.seed)
        self.variational_model = JointVariationalModel(
            dim_x=self._params.dim_x, 
            dim_y=self._params.dim_y,
            use_flow=self._params.use_flow,
            n_flows=self._params.flow_layers, 
            nn_width=self._params.nn_width, 
            nn_depth=self._params.nn_depth, 
            knots=self._params.knots, 
            interval=self._params.interval, 
            key=key)


    def parameters(self) -> JointVariationalParams:
        return self._params

    def estimate_with_info(self, x: ArrayLike, y: ArrayLike) -> EstimateResult:
        key = jax.random.PRNGKey(self._params.seed)
        _ , key_estimate = jax.random.split(key, 2)
        key_init, key_split, key_fit = jax.random.split(key_estimate, 3)

        # standardize the data, note we do so before splitting into train/test
        space = ProductSpace(x, y, standardize=self._params.standardize)
        xs, ys = jnp.asarray(space.x), jnp.asarray(space.y)

        # split
        xs_train, xs_test, ys_train, ys_test = _estimators.train_test_split(
            xs, ys, train_size=self._params.train_test_split, key=key_split
        )
        if self._params.use_flow == True:
            training_log, model = joint_variational_flow_training(
                rng= key_fit,
                model = self.variational_model,
                xs=xs_train,
                ys=ys_train,
                xs_test=xs_test,
                ys_test=ys_test,
                batch_size=self._params.batch_size,
                test_every_n_steps=self._params.test_every_n_steps,
                max_n_steps=self._params.max_n_steps,
                learning_rate=self._params.learning_rate,
                verbose=self._verbose,
            )
        else:
            training_log, model = joint_variational_gauss_training(
                model = self.variational_model,
                xs=xs_train,
                ys=ys_train,
                xs_test=xs_test,
                ys_test=ys_test,
            )
            
        self.variational_model = model

        return EstimateResult(
            mi_estimate=training_log.final_mi,
            additional_information=training_log.additional_information,
        )

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        return self.estimate_with_info(x,y).mi_estimate
    
    def plot_estimator(
        self,
        x: ArrayLike,
        y: ArrayLike):
        space = ProductSpace(x, y, standardize=self._params.standardize)
        xs, ys = jnp.asarray(space.x), jnp.asarray(space.y)

        fX_vmap = jax.vmap(self.variational_model._fX, in_axes=(0,))
        gY_vmap = jax.vmap(self.variational_model._gY, in_axes=(0,))
        fXs, logDetJfX = fX_vmap (xs)
        gYs, logDetJgY = gY_vmap(ys)
        fX_gYs = jnp.concatenate((fXs, gYs), axis=1)
        Mu = jnp.mean(fX_gYs,axis=0)
        Sigma = jnp.cov(fX_gYs.T)
        
        n_sample, dim_x = xs.shape
        if dim_x == 1:
            XY = np.random.multivariate_normal(Mu,Sigma,2500)
            fX_inv_vmap = jax.vmap(self.variational_model._fX.inverse, in_axes=(0,))
            gY_inv_vmap = jax.vmap(self.variational_model._gY.inverse, in_axes=(0,))
            fX, logDetJfX = fX_inv_vmap(XY[:,:dim_x])
            gY, logDetJgY = gY_inv_vmap(XY[:,dim_x:])
            
            plt.subplot(1, 2, 1) # row 1, col 2 index 1
            plt.scatter(xs[:2500], ys[:2500])
            plt.title("True Samples")
            plt.xlabel('X')
            plt.ylabel('Y')

            plt.subplot(1, 2, 2) # index 2
            plt.scatter(fX, gY)
            plt.title("Flow Samples")
            plt.xlabel('X')
            plt.ylabel('Y')

            plt.show()
        if dim_x == 2:
            X = np.random.multivariate_normal(Mu[:dim_x],Sigma[:dim_x,:dim_x],2500)
            fX_inv_vmap = jax.vmap(self.variational_model._fX.inverse, in_axes=(0,))
            fX, logDetJfX = fX_inv_vmap(X)
            
            plt.subplot(1, 2, 1) # row 1, col 2 index 1
            plt.scatter(xs[:,0], xs[:,1])
            plt.title("True Samples")
            plt.xlabel('X1')
            plt.ylabel('X2')

            plt.subplot(1, 2, 2) # index 2
            plt.scatter(fX[:,0], fX[:,1])
            plt.title("Flow Samples")
            plt.xlabel('X1')
            # plt.ylabel('X2')

            plt.show()
            
        if dim_x == 1:

            import scipy
            xtest = np.linspace(-5,5,2000)
            # fxtest, logDetJfX, gYtest, logDetJgY = flows_vmap(xtest, xtest)
            fXtest, logDetJfX = fX_vmap(xtest.reshape(-1,1))
            gYtest, logDetJgY = gY_vmap(xtest.reshape(-1,1))

            fmarg = scipy.stats.multivariate_normal.pdf(fXtest, Mu[:dim_x], Sigma[:dim_x,:dim_x])*np.exp(logDetJfX).T
            gmarg = scipy.stats.multivariate_normal.pdf(gYtest, Mu[dim_x:], Sigma[dim_x:,dim_x:])*np.exp(logDetJgY).T
            plt.plot(xtest, fmarg.flatten())
            # Add labels and title
            plt.xlabel('X-axis label')
            plt.ylabel('p(x)-axis label')
            plt.title('x marginal')
            # Show the plot
            plt.show()

            plt.plot(xtest, gmarg.flatten())
            # Add labels and title
            plt.xlabel('Y-axis label')
            plt.ylabel('p(y)-axis label')
            plt.title('y marginal')
            # Show the plot
            plt.show()


###### OLD CODE CAN TOSS ONCE OTHER RESULTS ARE WORKING ##################
# ########################### Templates ##################################
# class NFModel(eqx.Module):

#     """
#     Base class for normalizing flow models.
    
#     This is an abstract template that should not be directly used.
#     """

#     @abstractmethod
#     def __init__(self):
#         return NotImplemented

#     def __call__(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
#         """
#         Forward pass of the model.
        
#         Args:
#             x (Array): Input data.

#         Returns:
#             Tuple[Array, Array]: Output data and log determinant of the Jacobian.
#         """
#         return self.forward(x,y)

#     @abstractmethod
#     def forward(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
#         """
#         Forward pass of the model.
        
#         Args:
#             x (Array): Input data.
            
#         Returns:
#             Tuple[Array, Array]: Output data and log determinant of the Jacobian."""
#         return NotImplemented

#     @abstractmethod
#     def inverse(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
#         """
#         Inverse pass of the model.

#         Args:
#             x (Array): Input data.
            
#         Returns:
#             Tuple[Array, Array]: Output data and log determinant of the Jacobian."""
#         return NotImplemented

#     @abstractproperty
#     def n_features_x(self) -> int:
#         return NotImplemented
    
#     @abstractproperty
#     def n_features_y(self) -> int:
#         return NotImplemented

#     def save_model(self, path: str):
#         eqx.tree_serialise_leaves(path+".eqx", self)

#     def load_model(self, path: str) -> eqx.Module:
#         return eqx.tree_deserialise_leaves(path+".eqx", self)
    
#     # @abstractmethod
#     # def log_prob(self, x: Array) -> Array:
#     #     return NotImplemented
    
#     # @abstractmethod
#     # def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> Array:
#     #     return NotImplemented
    
# class Bijection(eqx.Module):

#     """
#     Base class for bijective transformations.
    
#     This is an abstract template that should not be directly used."""

#     @abstractmethod
#     def __init__(self):
#         return NotImplemented

#     def __call__(self, x: Array) -> Tuple[Array, Array]:
#         return self.forward(x)

#     @abstractmethod
#     def forward(self, x: Array) -> Tuple[Array, Array]:
#         return NotImplemented

#     @abstractmethod
#     def inverse(self, x: Array) -> Tuple[Array, Array]:
#         return NotImplemented
    
    
    
# ############################# Real NVP Structure ##############################

# class MLP(eqx.Module):
#     """Multilayer perceptron.

#     Args:
#         shape (Iterable[int]): Shape of the MLP. The first element is the input dimension, the last element is the output dimension.
#         key (jax.random.PRNGKey): Random key.

#     Attributes:
#         layers (List): List of layers.
#         activation (Callable): Activation function.
#         use_bias (bool): Whether to use bias.        
#     """
#     layers: List

#     def __init__(self, shape: Iterable[int], key: jax.random.PRNGKey, scale: float = 1e-4, activation: Callable = jax.nn.relu, use_bias: bool = True):
#         self.layers = []
#         for i in range(len(shape) - 2):
#             key, subkey1, subkey2 = jax.random.split(key, 3)
#             layer = eqx.nn.Linear(shape[i], shape[i + 1], key=subkey1, use_bias=use_bias)
#             weight = jax.random.normal(subkey2, (shape[i + 1], shape[i]))*jnp.sqrt(scale/shape[i])
#             layer = eqx.tree_at(lambda l: l.weight, layer, weight)
#             self.layers.append(layer)
#             self.layers.append(activation)
#         key, subkey = jax.random.split(key)
#         self.layers.append(eqx.nn.Linear(shape[-2], shape[-1], key=subkey, use_bias=use_bias))

#     def __call__(self, x: Array):
#         for layer in self.layers:
#             x = layer(x)
#         return x
    
#     @property
#     def n_input(self) -> int:
#         return self.layers[0].in_features
    
#     @property
#     def n_output(self) -> int:
#         return self.layers[-1].out_features

#     @property
#     def dtype(self) -> jnp.dtype:
#         return self.layers[0].weight.dtype
    
# class AffineCoupling(eqx.Module):
#     """
#     Affine coupling layer. 
#     (Defined in the RealNVP paper https://arxiv.org/abs/1605.08803)
#     We use tanh as the default activation function.

#     Args:
#         n_features: (int) The number of features in the input.
#         n_hidden: (int) The number of hidden units in the MLP.
#         mask: (ndarray) Alternating mask for the affine coupling layer.
#         dt: (float) Scaling factor for the affine coupling layer.
#     """
#     _mask: Array
#     scale_MLP: eqx.Module
#     translate_MLP: eqx.Module
#     dt: float = 1

#     def __init__(self, n_features: int, n_hidden: int, mask:Array, key: jax.random.PRNGKey, dt: float = 1, scale: float = 1e-4):
#         self._mask = mask
#         self.dt = dt
#         key, scale_subkey, translate_subkey = jax.random.split(key, 3)
#         features = [n_features, 2*n_hidden, n_hidden, n_features]
#         self.scale_MLP = MLP(features, key=scale_subkey, scale=scale)
#         self.translate_MLP = MLP(features, key=translate_subkey, scale=scale)

#     @property
#     def mask(self):
#         return jax.lax.stop_gradient(self._mask)

#     @property
#     def n_features(self):
#         return self.scale_MLP.n_input

#     def __call__(self, x: Array):
#         return self.forward(x)

#     def forward(self, x: Array) -> Tuple[Array, Array]:
#         """ From latent space to data space

#         Args:
#             x: (Array) Latent space.

#         Returns:
#             outputs: (Array) Data space.
#             log_det: (Array) Log determinant of the Jacobian.
#         """
#         s = self.mask * self.scale_MLP(x * (1 - self.mask))
#         s = jnp.tanh(s) * self.dt
#         t = self.mask * self.translate_MLP(x * (1 - self.mask)) * self.dt
        
#         # Compute log determinant of the Jacobian
#         log_det = s.sum()

#         # Apply the transformation
#         outputs = (x + t) * jnp.exp(s)
#         return outputs, log_det

#     def inverse(self, x: Array) -> Tuple[Array, Array]:
#         """ From data space to latent space

#         Args:
#             x: (Array) Data space.

#         Returns:
#             outputs: (Array) Latent space.
#             log_det: (Array) Log determinant of the Jacobian. 
#         """
#         s = self.mask * self.scale_MLP(x * (1 - self.mask))
#         s = jnp.tanh(s) * self.dt
#         t = self.mask * self.translate_MLP(x * (1 - self.mask)) * self.dt
#         log_det = -s.sum()
#         outputs = x * jnp.exp(-s) - t
#         return outputs, log_det

# ############################## MM Flow Structures ##############################
# class RealNVP(NFModel):
#     """
#     RealNVP mode defined in the paper https://arxiv.org/abs/1605.08803.
#     MLP is needed to make sure the scaling between layers are more or less the same.

#     Args:
#         n_layer: (int) The number of affine coupling layers.
#         n_features: (int) The number of features in the input.
#         n_hidden: (int) The number of hidden units in the MLP.
#         dt: (float) Scaling factor for the affine coupling layer.

#     Properties:
#         data_mean: (ndarray) Mean of Gaussian base distribution
#         data_cov: (ndarray) Covariance of Gaussian base distribution
#     """
#     affine_coupling_f: List[AffineCoupling]
#     affine_coupling_g: List[AffineCoupling]
#     _n_features_x: int
#     _n_features_y: int

#     @property
#     def n_features_x(self):
#         return self._n_features_x
    
#     @property
#     def n_features_y(self):
#         return self._n_features_y


#     def __init__(self,
#                 n_features_x: int,
#                 n_features_y: int,
#                 n_layer: int,
#                 n_hidden: int,
#                 key: jax.random.PRNGKey,
#                 **kwargs):
        
#         self._n_features_x = n_features_x
#         affine_coupling_f = []
#         for i in range(n_layer):
#             mask = np.ones(n_features_x)
#             mask[int(n_features_x / 2):] = 0
#             if i % 2 == 0:
#                 mask = 1 - mask
#             mask = jnp.array(mask)
#             affine_coupling_f.append(AffineCoupling(n_features_x,n_hidden,mask,key))
#         self.affine_coupling_f = affine_coupling_f
        
#         self._n_features_y = n_features_y
#         affine_coupling_g = []
#         for i in range(n_layer):
#             mask = np.ones(n_features_y)
#             mask[int(n_features_y / 2):] = 0
#             if i % 2 == 0:
#                 mask = 1 - mask
#             mask = jnp.array(mask)
#             affine_coupling_g.append(AffineCoupling(n_features_y,n_hidden,mask,key))
#         self.affine_coupling_g = affine_coupling_g
        
#     def __call__(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
#         return self.forward(x,y)

#     def forward(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
#         log_det_df = 0
#         log_det_dg = 0
#         for i in range(len(self.affine_coupling_f)):
#             x, log_det_dfi = self.affine_coupling_f[i](x)
#             log_det_df += log_det_dfi
            
#             y, log_det_dgi = self.affine_coupling_g[i](y)
#             log_det_dg += log_det_dgi      
#         return x, log_det_df, y, log_det_dg
    
#     def inverse(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
#         """ From latent space to data space"""
#         log_det_df = 0
#         log_det_dg = 0
#         for i in reversed(range(len(self.affine_coupling_f))):
#             x, log_det_dfi = self.affine_coupling_f[i].inverse(x)
#             log_det_df += log_det_dfi
            
#             y, log_det_dgi = self.affine_coupling_g[i].inverse(y)
#             log_det_dg += log_det_dgi
#         return x, log_det_df, y, log_det_dg

# class MaskedSplineFlows(NFModel):
#     """
#     RealNVP mode defined in the paper https://arxiv.org/abs/1605.08803.
#     MLP is needed to make sure the scaling between layers are more or less the same.

#     Args:
#         n_layer: (int) The number of affine coupling layers.
#         n_features: (int) The number of features in the input.
#         n_hidden: (int) The number of hidden units in the MLP.
#         dt: (float) Scaling factor for the affine coupling layer.

#     Properties:
#         data_mean: (ndarray) Mean of Gaussian base distribution
#         data_cov: (ndarray) Covariance of Gaussian base distribution
#     """
#     affine_coupling_f: Transformed
#     affine_coupling_g: Transformed
#     _n_features_x: int
#     _n_features_y: int

#     @property
#     def n_features_x(self):
#         return self._n_features_x
    
#     @property
#     def n_features_y(self):
#         return self._n_features_y
    
#     # @property
#     # def meanx(self):
#     #     return jax.lax.stop_gradient(self.affine_coupling_f.base_dist.loc)
    
#     # @property
#     # def meany(self):
#     #     return jax.lax.stop_gradient(self.affine_coupling_g.base_dist.loc)
    
#     # @property
#     # def scalex(self):
#     #     return jax.lax.stop_gradient(self.affine_coupling_f.base_dist.scale)
    
#     # @property
#     # def scaley(self):
#     #     return jax.lax.stop_gradient(self.affine_coupling_g.base_dist.scale)


#     def __init__(self,
#                 n_features_x: int,
#                 n_features_y: int,
#                 key: jax.random.PRNGKey,
#                 flow_layers: int = 8,
#                 nn_width: int = 50,
#                 nn_depth: int = 1,
#                 knots: int = 8,
#                 interval: int = 4,
#                 **kwargs):
#         f_subkey, g_subkey = jax.random.split(key, 2)
        
#         # Args:
#         # key: Array,
#         # *,
#         # base_dist: AbstractDistribution,
#         # transformer: AbstractBijection | None = None,
#         # cond_dim: int | None = None,
#         # flow_layers: int = 8,
#         # nn_width: int = 50,
#         # nn_depth: int = 1,
#         # nn_activation: Callable = jnn.relu,
#         # invert: bool = True,
        
        
#         self._n_features_x = n_features_x
#         self.affine_coupling_f = masked_autoregressive_flow(f_subkey,
#                                                             base_dist=Normal(jnp.zeros(n_features_x)),
#                                                             transformer=RationalQuadraticSpline(knots=knots, 
#                                                                                                 interval=interval),
#                                                             flow_layers=flow_layers,
#                                                             nn_width= nn_width,
#                                                             nn_depth=nn_depth,
#                                                             invert = False,
#                                                             )
        
#         self._n_features_y = n_features_y 
#         self.affine_coupling_g = masked_autoregressive_flow(g_subkey,
#                                                             base_dist=Normal(jnp.zeros(n_features_y)),
#                                                             transformer=RationalQuadraticSpline(knots=knots, 
#                                                                                                 interval=interval),
#                                                             flow_layers=flow_layers,
#                                                             nn_width= nn_width,
#                                                             nn_depth=nn_depth,
#                                                             invert = False,
#                                                             )
        
#     def __call__(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
#         return self.forward(x,y)

#     def forward(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
#         x, log_det_df = self.affine_coupling_f.bijection.transform_and_log_det(x)
           
#         y, log_det_dg = self.affine_coupling_g.bijection.transform_and_log_det(y)     
#         return x, log_det_df, y, log_det_dg
    
#     def inverse(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
#         x, log_det_df = self.affine_coupling_f.bijection.inverse_and_log_det(x)
            
#         y, log_det_dg = self.affine_coupling_g.bijection.inverse_and_log_det(y) 
#         return x, log_det_df, y, log_det_dg


# @eqx.filter_jit
# def _FlowMP_value(
#     flows: MaskedSplineFlows,
#     xs: jnp.ndarray,
#     ys: jnp.ndarray):
#     n_sample, dim_x = xs.shape
#     _, dim_y = ys.shape
#     flows_vmap = jax.vmap(flows, in_axes=(0, 0))
#     fX, logDetJfX, gY, logDetJgY = flows_vmap(xs, ys)
#     fX_gY = jnp.concatenate((fX, gY), axis=1)
    
#     Sigma = jnp.cov(fX_gY.T)

#     hX = 0.5 * jnp.linalg.slogdet(Sigma[:dim_x, :dim_x])[1] + dim_x / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / n_sample) * jnp.sum(logDetJfX)#logDetJfX.shape[0]
    
#     hXY = 0.5 * jnp.linalg.slogdet(Sigma)[1] + (dim_x+dim_y) / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / n_sample) * jnp.sum(logDetJfX)#- (1 / n_sample) * jnp.sum(logDetJgY)
#     hY = 0.5 * jnp.linalg.slogdet(Sigma[dim_x:, dim_x:])[1] + dim_y / 2 * (1 + jnp.log(2 * jnp.pi))# - (1 / n_sample) * jnp.sum(logDetJgY)
#     hX_Y = hXY-hY
    
#     value = hX - hX_Y
#     return value

# @eqx.filter_value_and_grad
# def loss_function_FlowMargPost(flows, xs, ys):
#     n_sample, dim_x = xs.shape
#     _, dim_y = ys.shape
#     flows_vmap = jax.vmap(flows, in_axes=(0, 0))
#     fX, logDetJfX, gY, logDetJgY = flows_vmap(xs, ys)
#     fX_gY = jnp.concatenate((fX, gY), axis=1)
    
#     Sigma = jnp.cov(fX_gY.T) + 1e-4*jnp.eye(dim_x+dim_y)
    
#     hX = 0.5 * jnp.linalg.slogdet(Sigma[:dim_x, :dim_x])[1] + dim_x / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / n_sample) * jnp.sum(logDetJfX)
    
#     hXY = 0.5 * jnp.linalg.slogdet(Sigma)[1] + (dim_x+dim_y) / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / n_sample) * jnp.sum(logDetJfX)- (1 / n_sample) * jnp.sum(logDetJgY)
#     hY = 0.5 * jnp.linalg.slogdet(Sigma[dim_x:, dim_x:])[1] + dim_y / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / (n_sample)) * jnp.sum(logDetJgY)
#     hX_Y = hXY-hY
#     hY_X = hXY-hX
    
    
#     loss = hX + hX_Y + hY_X + hY
#     return loss+jax.lax.stop_gradient(hX-hX_Y-loss)


# def _FlowMargPost_value_neg_grad(
#     flows: MaskedSplineFlows,
#     xs: jnp.ndarray,
#     ys: jnp.ndarray):
#     # Define functions to compute gradients using JAX's autograd
#     value, neg_grad = loss_function_FlowMargPost(flows, xs, ys)
#     return value, neg_grad


# @eqx.filter_jit
# def _FlowP_value(
#     flows: MaskedSplineFlows,
#     xs: jnp.ndarray,
#     ys: jnp.ndarray):
#     n_sample, dim_x = xs.shape
#     _, dim_y = ys.shape
#     flows_vmap = jax.vmap(flows, in_axes=(0, 0))
#     fX, logDetJfX, gY, logDetJgY = flows_vmap(xs, ys)
#     fX_gY = jnp.concatenate((fX, gY), axis=1)
    
#     Sigma = jnp.cov(fX_gY.T)

#     mux = jnp.zeros(dim_x)
#     Sigmax = jnp.eye(dim_x)
#     logmarg = jax.scipy.stats.multivariate_normal.logpdf(xs, mux, Sigmax)
#     hX = -jnp.mean(logmarg)
    
#     hXY = 0.5 * jnp.linalg.slogdet(Sigma)[1] + (dim_x+dim_y) / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / n_sample) * jnp.sum(logDetJfX)#- (1 / n_sample) * jnp.sum(logDetJgY)
#     hY = 0.5 * jnp.linalg.slogdet(Sigma[dim_x:, dim_x:])[1] + dim_y / 2 * (1 + jnp.log(2 * jnp.pi))# - (1 / n_sample) * jnp.sum(logDetJgY)
#     hX_Y = hXY-hY
    
#     value = hX - hX_Y
#     return value

# @eqx.filter_value_and_grad
# def loss_function_FlowPost(flows, xs, ys):
#     n_sample, dim_x = xs.shape
#     _, dim_y = ys.shape
#     flows_vmap = jax.vmap(flows, in_axes=(0, 0))
#     fX, logDetJfX, gY, logDetJgY = flows_vmap(xs, ys)
#     fX_gY = jnp.concatenate((fX, gY), axis=1)
    
#     Sigma = jnp.cov(fX_gY.T) + 1e-4*jnp.eye(dim_x+dim_y)
    
#     mux = jnp.zeros(dim_x)
#     Sigmax = jnp.eye(dim_x)
#     logmarg = jax.scipy.stats.multivariate_normal.logpdf(xs, mux, Sigmax)
#     # hX = 0.5 * jnp.linalg.slogdet(Sigma[:dim_x, :dim_x])[1] + dim_x / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / n_sample) * jnp.sum(logDetJfX)
    
#     hXY = 0.5 * jnp.linalg.slogdet(Sigma)[1] + (dim_x+dim_y) / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / n_sample) * jnp.sum(logDetJfX)- (1 / n_sample) * jnp.sum(logDetJgY)
#     hY = 0.5 * jnp.linalg.slogdet(Sigma[dim_x:, dim_x:])[1] + dim_y / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / (n_sample)) * jnp.sum(logDetJgY)
#     hX_Y = hXY-hY
    
    
#     loss = hX_Y
#     return loss +jax.lax.stop_gradient(-jnp.mean(logmarg)-2*hX_Y)

# def _FlowPost_value_neg_grad(
#     flows: MaskedSplineFlows,
#     xs: jnp.ndarray,
#     ys: jnp.ndarray):
#     # Define functions to compute gradients using JAX's autograd
#     value, neg_grad = loss_function_FlowPost(flows, xs, ys)
#     return value, neg_grad

# def Flow_training(
#     rng: jax.random.PRNGKeyArray,
#     flows: NFModel,
#     xs: BatchedPoints,
#     ys: BatchedPoints,
#     xs_test: Optional[BatchedPoints] = None,
#     ys_test: Optional[BatchedPoints] = None,
#     batch_size: Optional[int] = 256,
#     test_every_n_steps: int = 250,
#     max_n_steps: int = 2_000,
#     early_stopping: bool = False,
#     learning_rate: float = 0.1,
#     verbose: bool = False,
#     MargPost_loss:  bool = True
# ) -> tuple[TrainingLog, eqx.Module]:
#     """Basic training loop for MINE.

#     Args:
#         rng: random key
#         critic: critic to be trained
#         xs: samples of X, shape (n_points, dim_x)
#         ys: paired samples of Y, shape (n_points, dim_y)
#         xs_test: samples of X used for computing test MI, shape (n_points_test, dim_x),
#           if None will reuse xs
#         ys_test: paired samples of Y used for computing test MI, shape (n_points_test, dim_y),
#           if None will reuse ys
#         batch_size: batch size
#         test_every_n_steps: step intervals at which the training checkpoint should be saved
#         max_n_steps: maximum number of steps
#         early_stopping: whether training should stop early when test MI stops growing
#         learning_rate: learning rate to be used
#         verbose: print info during training

#     Returns:
#         training log
#         trained critic
#     """
#     xs_test = xs_test if xs_test is not None else xs
#     ys_test = ys_test if ys_test is not None else ys

#     # initialize the optimizer
#     optimizer = optax.adam(learning_rate=learning_rate)
#     opt_state = optimizer.init(eqx.filter(flows, eqx.is_array))

#     # compile the training step   flows.affine_coupling_g.base_dist.scale
#     @eqx.filter_jit
#     def step(
#         *,
#         flows,
#         opt_state,
#         MargPost_loss,
#         xs: BatchedPoints,
#         ys: BatchedPoints):
        
#         if MargPost_loss:
#             value, neg_grad = _FlowMargPost_value_neg_grad(
#                 flows=flows,
#                 xs=xs,
#                 ys=ys)
#         else:
#             value, neg_grad = _FlowPost_value_neg_grad(
#                 flows=flows,
#                 xs=xs,
#                 ys=ys)
        
#         updates, opt_state = optimizer.update(neg_grad, opt_state)
#         flows = eqx.apply_updates(flows, updates)
#         return flows, opt_state, value
        
#     # main training loop
#     training_log = TrainingLog(
#         max_n_steps=max_n_steps, 
#         early_stopping=early_stopping, 
#         verbose=verbose)
    
#     keys = jax.random.split(rng, max_n_steps)
#     for n_step, key in enumerate(keys, start=1):
#         key_sample, key_test = jax.random.split(key)

#         # sample
#         batch_xs, batch_ys = get_batch(xs, ys, key_sample, batch_size)

#         # run step
#         flows, opt_state, mi_train = step(
#             flows=flows,
#             opt_state = opt_state,
#             MargPost_loss= MargPost_loss,
#             xs=batch_xs,
#             ys=batch_ys)

#         # logging train
#         training_log.log_train_mi(n_step, mi_train)

#         # logging test
#         if n_step % test_every_n_steps == 0:
#             if MargPost_loss:
#                 mi_test = _FlowMP_value(
#                     flows=flows, xs=xs_test, ys=ys_test
#                 )
#             else:
#                 mi_test = _FlowP_value(
#                     flows=flows, xs=xs_test, ys=ys_test
#                 )
#             training_log.log_test_mi(n_step, mi_test)
            
#         # early stop?
#         if training_log.early_stop():
#             break

#     training_log.finish()
#     return training_log, flows


# class FlowParams(BaseModel):
#     batch_size: pydantic.PositiveInt
#     max_n_steps: pydantic.PositiveInt
#     train_test_split: Optional[pydantic.confloat(gt=0.0, lt=1.0)]
#     test_every_n_steps: pydantic.PositiveInt
#     learning_rate: pydantic.PositiveFloat
#     standardize: bool
#     seed: int
#     flow_layers: pydantic.PositiveInt
#     nn_width: pydantic.PositiveInt
#     nn_depth: pydantic.PositiveInt
#     knots: pydantic.PositiveInt
#     interval: pydantic.PositiveInt

# class FlowMargPostEstimator(IMutualInformationPointEstimator):
#     def __init__(
#         self,
#         batch_size: int =  _estimators._DEFAULT_BATCH_SIZE,
#         max_n_steps: int = _estimators._DEFAULT_N_STEPS,
#         train_test_split: Optional[float] = _estimators._DEFAULT_TRAIN_TEST_SPLIT,
#         test_every_n_steps: int = _estimators._DEFAULT_TEST_EVERY_N,
#         learning_rate: float = _estimators._DEFAULT_LEARNING_RATE,
#         flow_layers: int = 8,#8
#         nn_width: int = 64,#64
#         nn_depth: int = 1,#1
#         knots: int = 8,#8
#         interval: int = 4,#4
#         standardize: bool = _estimators._DEFAULT_STANDARDIZE,
#         verbose: bool = _estimators._DEFAULT_VERBOSE,
#         seed: int = _estimators._DEFAULT_SEED,
#     ) -> None:
#         self._params = FlowParams(
#             batch_size=batch_size,
#             max_n_steps=max_n_steps,
#             train_test_split=train_test_split,
#             test_every_n_steps=test_every_n_steps,
#             learning_rate=learning_rate,
#             standardize=standardize,
#             seed=seed,
#             flow_layers = flow_layers, 
#             nn_width = nn_width,
#             nn_depth = nn_depth,
#             knots = knots,
#             interval = interval)
        
#         self._verbose = verbose
#         self._training_log: Optional[TrainingLog] = None

#         # After the training we will store the trained
#         # critic function here
#         self._trained_flows = None

#     @property
#     def trained_flows(self) -> Optional[eqx.Module]:
#         """Returns the critic function from the end of the training.

#         Note:
#           1. You need to train the model by estimating mutual information,
#             otherwise `None` is returned.
#           2. Note that the critic can have different meaning depending on
#             the function used.
#         """
#         return self._trained_flows

#     def parameters(self) -> FlowParams:
#         return self._params

#     def _create_flows(self, dim_x: int, dim_y: int, key: jax.random.PRNGKeyArray) -> MaskedSplineFlows:#RealNVP:#
#         return MaskedSplineFlows(n_features_x = dim_x, 
#                                  n_features_y = dim_y, 
#                                  key=key, 
#                                  flow_layers = self._params.flow_layers, 
#                                  nn_width = self._params.nn_width,
#                                  nn_depth = self._params.nn_depth,
#                                  knots = self._params.knots,
#                                  interval = self._params.interval,)
    
#     def estimate_with_info(self, x: ArrayLike, y: ArrayLike) -> EstimateResult:
#         key = jax.random.PRNGKey(self._params.seed)
#         key_init, key_split, key_fit = jax.random.split(key, 3)

#         # standardize the data, note we do so before splitting into train/test
#         space = ProductSpace(x, y, standardize=self._params.standardize)
#         xs, ys = jnp.asarray(space.x), jnp.asarray(space.y)

#         # split
#         xs_train, xs_test, ys_train, ys_test = _estimators.train_test_split(
#             xs, ys, train_size=self._params.train_test_split, key=key_split
#         )

#         # initialize critic
#         _flows = self._create_flows(dim_x=space.dim_x, dim_y=space.dim_y, key=key_init)

#         training_log, trained_flows = Flow_training(
#             rng=key_fit,
#             flows = _flows,
#             xs=xs_train,
#             ys=ys_train,
#             xs_test=xs_test,
#             ys_test=ys_test,
#             batch_size=self._params.batch_size,
#             test_every_n_steps=self._params.test_every_n_steps,
#             max_n_steps=self._params.max_n_steps,
#             learning_rate=self._params.learning_rate,
#             verbose=self._verbose,
#             MargPost_loss=True,
#         )
        
#         self._trained_flows = trained_flows

#         return EstimateResult(
#             mi_estimate=training_log.final_mi,
#             additional_information=training_log.additional_information)

#     def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
#         return self.estimate_with_info(x, y).mi_estimate

#     def plot_estimator(
#         self,
#         x: ArrayLike,
#         y: ArrayLike):
#         space = ProductSpace(x, y, standardize=self._params.standardize)
#         xs, ys = jnp.asarray(space.x), jnp.asarray(space.y)
#         flows_vmap = jax.vmap(self.trained_flows, in_axes=(0, 0))
#         fXs, logDetJfX, gYs, logDetJgY = flows_vmap(xs, ys)
#         fX_gYs = jnp.concatenate((fXs, gYs), axis=1)
#         Mu = jnp.mean(fX_gYs,axis=0)
#         Sigma = jnp.cov(fX_gYs.T)
        
#         n_sample, dim_x = xs.shape
#         XY = np.random.multivariate_normal(Mu,Sigma,1000)
#         flows_inv_vmap = jax.vmap(self.trained_flows.inverse, in_axes=(0, 0))
#         fX, logDetJfX, gY, logDetJgY  = flows_inv_vmap(XY[:,:dim_x], XY[:,dim_x:])
        
#         plt.subplot(1, 2, 1) # row 1, col 2 index 1
#         plt.scatter(xs[:1000], ys[:1000])
#         plt.title("True Samples")
#         plt.xlabel('X')
#         plt.ylabel('Y')

#         plt.subplot(1, 2, 2) # index 2
#         plt.scatter(fX, gY)
#         plt.title("Flow Samples")
#         plt.xlabel('X')
#         plt.ylabel('Y')

#         plt.show()


# class FlowPostEstimator(IMutualInformationPointEstimator):
#     def __init__(
#         self,
#         batch_size: int =  _estimators._DEFAULT_BATCH_SIZE,
#         max_n_steps: int = _estimators._DEFAULT_N_STEPS,
#         train_test_split: Optional[float] = _estimators._DEFAULT_TRAIN_TEST_SPLIT,
#         test_every_n_steps: int = _estimators._DEFAULT_TEST_EVERY_N,
#         learning_rate: float = _estimators._DEFAULT_LEARNING_RATE,
#         flow_layers: int = 2,
#         nn_width: int = 50,
#         nn_depth: int = 1,
#         knots: int = 8,
#         interval: int = 4,
#         standardize: bool = _estimators._DEFAULT_STANDARDIZE,
#         verbose: bool = _estimators._DEFAULT_VERBOSE,
#         seed: int = _estimators._DEFAULT_SEED,
#     ) -> None:
#         self._params = FlowParams(
#             batch_size=batch_size,
#             max_n_steps=max_n_steps,
#             train_test_split=train_test_split,
#             test_every_n_steps=test_every_n_steps,
#             learning_rate=learning_rate,
#             standardize=standardize,
#             seed=seed,
#             flow_layers = flow_layers, 
#             nn_width = nn_width,
#             nn_depth = nn_depth,
#             knots = knots,
#             interval = interval)
        
#         self._verbose = verbose
#         self._training_log: Optional[TrainingLog] = None

#         # After the training we will store the trained
#         # critic function here
#         self._trained_flows = None

#     @property
#     def trained_flows(self) -> Optional[eqx.Module]:
#         """Returns the critic function from the end of the training.

#         Note:
#           1. You need to train the model by estimating mutual information,
#             otherwise `None` is returned.
#           2. Note that the critic can have different meaning depending on
#             the function used.
#         """
#         return self._trained_flows

#     def parameters(self) -> FlowParams:
#         return self._params

#     def _create_flows(self, dim_x: int, dim_y: int, key: jax.random.PRNGKeyArray) -> MaskedSplineFlows:#RealNVP:#
#         return MaskedSplineFlows(n_features_x = dim_x, 
#                                  n_features_y = dim_y, 
#                                  key=key, 
#                                  flow_layers = self._params.flow_layers, 
#                                  nn_width = self._params.nn_width,
#                                  nn_depth = self._params.nn_depth,
#                                  knots = self._params.knots,
#                                  interval = self._params.interval,)
    
#     def estimate_with_info(self, x: ArrayLike, y: ArrayLike) -> EstimateResult:
#         key = jax.random.PRNGKey(self._params.seed)
#         key_init, key_split, key_fit = jax.random.split(key, 3)

#         # standardize the data, note we do so before splitting into train/test
#         space = ProductSpace(x, y, standardize=self._params.standardize)
#         xs, ys = jnp.asarray(space.x), jnp.asarray(space.y)

#         # split
#         xs_train, xs_test, ys_train, ys_test = _estimators.train_test_split(
#             xs, ys, train_size=self._params.train_test_split, key=key_split
#         )

#         # initialize critic
#         _flows = self._create_flows(dim_x=space.dim_x, dim_y=space.dim_y, key=key_init)

#         training_log, trained_flows = Flow_training(
#             rng=key_fit,
#             flows = _flows,
#             xs=xs_train,
#             ys=ys_train,
#             xs_test=xs_test,
#             ys_test=ys_test,
#             batch_size=self._params.batch_size,
#             test_every_n_steps=self._params.test_every_n_steps,
#             max_n_steps=self._params.max_n_steps,
#             learning_rate=self._params.learning_rate,
#             verbose=self._verbose,
#             MargPost_loss= False,
#         )
        
#         self._trained_flows = trained_flows

#         return EstimateResult(
#             mi_estimate=training_log.final_mi,
#             additional_information=training_log.additional_information)

#     def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
#         return self.estimate_with_info(x, y).mi_estimate


# class MMParams(BaseModel):
#     train_test_split: Optional[pydantic.confloat(gt=0.0, lt=1.0)]
#     standardize: bool
#     seed: int
    
# class MargPostEstimator(IMutualInformationPointEstimator):
#     def __init__(
#         self,
#         train_test_split: Optional[float] = _estimators._DEFAULT_TRAIN_TEST_SPLIT,
#         standardize: bool = _estimators._DEFAULT_STANDARDIZE,
#         seed: int = _estimators._DEFAULT_SEED,
#     ) -> None:
#         self._params = MMParams(
#             train_test_split=train_test_split,
#             standardize=standardize,
#             seed=seed,)
        
#         self._training_log: Optional[TrainingLog] = None

#         # After the training we will store the trained
#         # critic function here
#         self._trained_flows = None

#     @property
#     def trained_flows(self) -> Optional[eqx.Module]:
#         """Returns the critic function from the end of the training.

#         Note:
#           1. You need to train the model by estimating mutual information,
#             otherwise `None` is returned.
#           2. Note that the critic can have different meaning depending on
#             the function used.
#         """
#         return self._trained_flows

#     def parameters(self) -> FlowParams:
#         return self._params
    
#     def estimate_with_info(self, x: ArrayLike, y: ArrayLike) -> EstimateResult:
#         key = jax.random.PRNGKey(self._params.seed)
#         key_init, key_split, key_fit = jax.random.split(key, 3)

#         # standardize the data, note we do so before splitting into train/test
#         space = ProductSpace(x, y, standardize=self._params.standardize)
#         xs, ys = jnp.asarray(space.x), jnp.asarray(space.y)

#         # split
#         xs_train, xs_test, ys_train, ys_test = _estimators.train_test_split(
#             xs, ys, train_size=self._params.train_test_split, key=key_split
#         )

#         data = {}
#         # data['x_train'] = xs_train
#         # data['y_train'] = ys_train
#         # data['x_test'] = xs_test
#         # data['y_test'] = ys_test
        
#         XY = np.concatenate((xs_train, ys_train), axis=1)
#         Sigma = np.cov(XY.T)
#         dim_x = xs_train.shape[1]
#         hX = 0.5 * np.linalg.slogdet(Sigma[:dim_x, :dim_x])[1] + dim_x / 2 * (1 + np.log(2 * np.pi))
#         hXY = 0.5 * np.linalg.slogdet(Sigma)[1] + (2*dim_x) / 2 * (1 + np.log(2 * np.pi))
#         hY = 0.5 * np.linalg.slogdet(Sigma[dim_x:, dim_x:])[1] + dim_x / 2 * (1 + np.log(2 * np.pi))
#         hX_Y = hXY-hY
#         value = hX - hX_Y 
        
#         return EstimateResult(
#             mi_estimate=value,
#             additional_information=data)

#     def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
#         return self.estimate_with_info(x, y).mi_estimate


# # ########### Plotting Function ######################################################

# # def plot_final(
# #     flows,
# #     xs: BatchedPoints,
# #     ys: BatchedPoints):
# #     flows_vmap = jax.vmap(flows, in_axes=(0, 0))
# #     fXs, logDetJfX, gYs, logDetJgY = flows_vmap(xs, ys)
# #     fX_gYs = jnp.concatenate((fXs, gYs), axis=1)
# #     Mu = jnp.mean(fX_gYs,axis=0)
# #     Sigma = jnp.cov(fX_gYs.T)
    
# #     n_sample, dim_x = xs.shape
# #     XY = np.random.multivariate_normal(Mu,Sigma,10000)
# #     flows_inv_vmap = jax.vmap(flows.inverse, in_axes=(0, 0))
# #     fX, logDetJfX, gY, logDetJgY  = flows_inv_vmap(XY[:,:dim_x], XY[:,dim_x:])
    
# #     if dim_x == 1:
# #         plt.scatter(fX, gY, s=10, c='black', alpha=0.5) 
# #         # Add labels and title
# #         plt.xlabel('X-axis label')
# #         plt.ylabel('Y-axis label')
# #         plt.title('Scatter Plot of X and Y')
# #         # Show the plot
# #         plt.show()
# #         import scipy
# #         xtest = np.linspace(-5,5,2000)
# #         fxtest, logDetJfX, gYtest, logDetJgY = flows_vmap(xtest, xtest)

# #         fmarg = scipy.stats.multivariate_normal.pdf(fxtest, Mu[:dim_x], Sigma[:dim_x,:dim_x])*np.exp(logDetJfX).T
# #         gmarg = scipy.stats.multivariate_normal.pdf(gYtest, Mu[dim_x:], Sigma[dim_x:,dim_x:])*np.exp(logDetJgY).T
# #         plt.plot(xtest, fmarg.flatten())
# #         # Add labels and title
# #         plt.xlabel('X-axis label')
# #         plt.ylabel('p(x)-axis label')
# #         plt.title('x marginal')
# #         # Show the plot
# #         plt.show()

# #         plt.plot(xtest, gmarg.flatten())
# #         # Add labels and title
# #         plt.xlabel('Y-axis label')
# #         plt.ylabel('p(y)-axis label')
# #         plt.title('y marginal')
# #         # Show the plot
# #         plt.show()
        
    
# #     if dim_x == 2:
# #         # xtest = (np.linspace(-4,4,100)*np.ones((2,100))).T
# #         # actualX = xtest[:,0] + 0.4 * np.sin(1.0 * xtest[:,0]) + 0.2 * np.sin(1.7 * xtest[:,0] + 1) + 0.03 * np.sin(3.3 * xtest[:,0] - 2.5)
# #         # actualY = xtest[:,0] - 0.4 * np.sin(0.4 * xtest[:,0]) + 0.17 * np.sin(1.3 * xtest[:,0] + 3.5) + 0.02 * np.sin(4.3 * xtest[:,0] - 2.5)
# #         # fxtest, logDetJfX, gYtest, logDetJgY = flows_inv_vmap(xtest, xtest)
        
# #         # plt.plot(xtest[:,0], fxtest[:,0],label='fX0')
# #         # plt.plot(xtest[:,1], fxtest[:,1],label='fX1')
# #         # plt.plot(xtest[:,1], actualX,label='True Transform')
# #         # plt.plot(xtest[:,1], -actualX,label='Negative True')
# #         # # Add labels and title
# #         # plt.xlabel('X-axis label')
# #         # plt.ylabel('f(x)-axis label')
# #         # plt.title('Learned X Flow')
# #         # plt.legend()
# #         # # Show the plot
# #         # plt.show()
        
# #         # plt.plot(xtest[:,0], gYtest[:,0],label='gY0')
# #         # plt.plot(xtest[:,1], gYtest[:,1],label='gY1')
# #         # plt.plot(xtest[:,1], actualY,label='True Transform')
# #         # plt.plot(xtest[:,1], -actualY,label='Negative True')
# #         # # Add labels and title
# #         # plt.xlabel('Y-axis label')
# #         # plt.ylabel('g(y)-axis label')
# #         # plt.title('Learned Y Flow')
# #         # plt.legend()
# #         # # Show the plot
# #         # plt.show()
# #         # fX[:,0][:,0], fX[:,0]  #fX[:,0][:,1], fX[:,1]
# #         plt.scatter(fX[:,0], fX[:,1], s=10, c='black', alpha=0.5)
# #         # Add labels and title
# #         plt.xlabel('X0-axis label')
# #         plt.ylabel('X1-axis label')
# #         plt.title('Scatter Plot of X and Y')

# #         # Show the plot
# #         plt.show()
# #         #fX[:,0][:,0], fX[:,0] 
# #         plt.scatter(fX[:,0], gY[:,0], s=10, c='black', alpha=0.5)
# #         # Add labels and title
# #         plt.xlabel('X0-axis label')
# #         plt.ylabel('Y0-axis label')
# #         plt.title('Scatter Plot of X and Y')

# #         # Show the plot
# #         plt.show()
# #         #fX[:,0][:,1], fX[:,1]
# #         plt.scatter(fX[:,1], gY[:,1], s=10, c='black', alpha=0.5)
# #         # Add labels and title
# #         plt.xlabel('X1-axis label')
# #         plt.ylabel('Y1-axis label')
# #         plt.title('Scatter Plot of X and Y')

# #         # Show the plot
# #         plt.show()
        
# #         plt.scatter(gY[:,0], gY[:,1], s=10, c='black', alpha=0.5)
# #         # Add labels and title
# #         plt.xlabel('Y0-axis label')
# #         plt.ylabel('Y1-axis label')
# #         plt.title('Scatter Plot of X and Y')

# #         # Show the plot
# #         plt.show()    