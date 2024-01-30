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

########################### Templates ##################################
class NFModel(eqx.Module):

    """
    Base class for normalizing flow models.
    
    This is an abstract template that should not be directly used.
    """

    @abstractmethod
    def __init__(self):
        return NotImplemented

    def __call__(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
        """
        Forward pass of the model.
        
        Args:
            x (Array): Input data.

        Returns:
            Tuple[Array, Array]: Output data and log determinant of the Jacobian.
        """
        return self.forward(x,y)

    @abstractmethod
    def forward(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
        """
        Forward pass of the model.
        
        Args:
            x (Array): Input data.
            
        Returns:
            Tuple[Array, Array]: Output data and log determinant of the Jacobian."""
        return NotImplemented

    @abstractmethod
    def inverse(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
        """
        Inverse pass of the model.

        Args:
            x (Array): Input data.
            
        Returns:
            Tuple[Array, Array]: Output data and log determinant of the Jacobian."""
        return NotImplemented

    @abstractproperty
    def n_features_x(self) -> int:
        return NotImplemented
    
    @abstractproperty
    def n_features_y(self) -> int:
        return NotImplemented

    def save_model(self, path: str):
        eqx.tree_serialise_leaves(path+".eqx", self)

    def load_model(self, path: str) -> eqx.Module:
        return eqx.tree_deserialise_leaves(path+".eqx", self)
    
    # @abstractmethod
    # def log_prob(self, x: Array) -> Array:
    #     return NotImplemented
    
    # @abstractmethod
    # def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> Array:
    #     return NotImplemented
    
class Bijection(eqx.Module):

    """
    Base class for bijective transformations.
    
    This is an abstract template that should not be directly used."""

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
    
    
    
############################# Real NVP Structure ##############################

class MLP(eqx.Module):
    """Multilayer perceptron.

    Args:
        shape (Iterable[int]): Shape of the MLP. The first element is the input dimension, the last element is the output dimension.
        key (jax.random.PRNGKey): Random key.

    Attributes:
        layers (List): List of layers.
        activation (Callable): Activation function.
        use_bias (bool): Whether to use bias.        
    """
    layers: List

    def __init__(self, shape: Iterable[int], key: jax.random.PRNGKey, scale: float = 1e-4, activation: Callable = jax.nn.relu, use_bias: bool = True):
        self.layers = []
        for i in range(len(shape) - 2):
            key, subkey1, subkey2 = jax.random.split(key, 3)
            layer = eqx.nn.Linear(shape[i], shape[i + 1], key=subkey1, use_bias=use_bias)
            weight = jax.random.normal(subkey2, (shape[i + 1], shape[i]))*jnp.sqrt(scale/shape[i])
            layer = eqx.tree_at(lambda l: l.weight, layer, weight)
            self.layers.append(layer)
            self.layers.append(activation)
        key, subkey = jax.random.split(key)
        self.layers.append(eqx.nn.Linear(shape[-2], shape[-1], key=subkey, use_bias=use_bias))

    def __call__(self, x: Array):
        for layer in self.layers:
            x = layer(x)
        return x
    
    @property
    def n_input(self) -> int:
        return self.layers[0].in_features
    
    @property
    def n_output(self) -> int:
        return self.layers[-1].out_features

    @property
    def dtype(self) -> jnp.dtype:
        return self.layers[0].weight.dtype
    
class AffineCoupling(eqx.Module):
    """
    Affine coupling layer. 
    (Defined in the RealNVP paper https://arxiv.org/abs/1605.08803)
    We use tanh as the default activation function.

    Args:
        n_features: (int) The number of features in the input.
        n_hidden: (int) The number of hidden units in the MLP.
        mask: (ndarray) Alternating mask for the affine coupling layer.
        dt: (float) Scaling factor for the affine coupling layer.
    """
    _mask: Array
    scale_MLP: eqx.Module
    translate_MLP: eqx.Module
    dt: float = 1

    def __init__(self, n_features: int, n_hidden: int, mask:Array, key: jax.random.PRNGKey, dt: float = 1, scale: float = 1e-4):
        self._mask = mask
        self.dt = dt
        key, scale_subkey, translate_subkey = jax.random.split(key, 3)
        features = [n_features, 2*n_hidden, n_hidden, n_features]
        self.scale_MLP = MLP(features, key=scale_subkey, scale=scale)
        self.translate_MLP = MLP(features, key=translate_subkey, scale=scale)

    @property
    def mask(self):
        return jax.lax.stop_gradient(self._mask)

    @property
    def n_features(self):
        return self.scale_MLP.n_input

    def __call__(self, x: Array):
        return self.forward(x)

    def forward(self, x: Array) -> Tuple[Array, Array]:
        """ From latent space to data space

        Args:
            x: (Array) Latent space.

        Returns:
            outputs: (Array) Data space.
            log_det: (Array) Log determinant of the Jacobian.
        """
        s = self.mask * self.scale_MLP(x * (1 - self.mask))
        s = jnp.tanh(s) * self.dt
        t = self.mask * self.translate_MLP(x * (1 - self.mask)) * self.dt
        
        # Compute log determinant of the Jacobian
        log_det = s.sum()

        # Apply the transformation
        outputs = (x + t) * jnp.exp(s)
        return outputs, log_det

    def inverse(self, x: Array) -> Tuple[Array, Array]:
        """ From data space to latent space

        Args:
            x: (Array) Data space.

        Returns:
            outputs: (Array) Latent space.
            log_det: (Array) Log determinant of the Jacobian. 
        """
        s = self.mask * self.scale_MLP(x * (1 - self.mask))
        s = jnp.tanh(s) * self.dt
        t = self.mask * self.translate_MLP(x * (1 - self.mask)) * self.dt
        log_det = -s.sum()
        outputs = x * jnp.exp(-s) - t
        return outputs, log_det

############################## MM Flow Structures ##############################
class RealNVP(NFModel):
    """
    RealNVP mode defined in the paper https://arxiv.org/abs/1605.08803.
    MLP is needed to make sure the scaling between layers are more or less the same.

    Args:
        n_layer: (int) The number of affine coupling layers.
        n_features: (int) The number of features in the input.
        n_hidden: (int) The number of hidden units in the MLP.
        dt: (float) Scaling factor for the affine coupling layer.

    Properties:
        data_mean: (ndarray) Mean of Gaussian base distribution
        data_cov: (ndarray) Covariance of Gaussian base distribution
    """
    affine_coupling_f: List[AffineCoupling]
    affine_coupling_g: List[AffineCoupling]
    _n_features_x: int
    _n_features_y: int

    @property
    def n_features_x(self):
        return self._n_features_x
    
    @property
    def n_features_y(self):
        return self._n_features_y


    def __init__(self,
                n_features_x: int,
                n_features_y: int,
                n_layer: int,
                n_hidden: int,
                key: jax.random.PRNGKey,
                **kwargs):
        
        self._n_features_x = n_features_x
        affine_coupling_f = []
        for i in range(n_layer):
            mask = np.ones(n_features_x)
            mask[int(n_features_x / 2):] = 0
            if i % 2 == 0:
                mask = 1 - mask
            mask = jnp.array(mask)
            affine_coupling_f.append(AffineCoupling(n_features_x,n_hidden,mask,key))
        self.affine_coupling_f = affine_coupling_f
        
        self._n_features_y = n_features_y
        affine_coupling_g = []
        for i in range(n_layer):
            mask = np.ones(n_features_y)
            mask[int(n_features_y / 2):] = 0
            if i % 2 == 0:
                mask = 1 - mask
            mask = jnp.array(mask)
            affine_coupling_g.append(AffineCoupling(n_features_y,n_hidden,mask,key))
        self.affine_coupling_g = affine_coupling_g
        
    def __call__(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
        return self.forward(x,y)

    def forward(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
        log_det_df = 0
        log_det_dg = 0
        for i in range(len(self.affine_coupling_f)):
            x, log_det_dfi = self.affine_coupling_f[i](x)
            log_det_df += log_det_dfi
            
            y, log_det_dgi = self.affine_coupling_g[i](y)
            log_det_dg += log_det_dgi      
        return x, log_det_df, y, log_det_dg
    
    def inverse(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
        """ From latent space to data space"""
        log_det_df = 0
        log_det_dg = 0
        for i in reversed(range(len(self.affine_coupling_f))):
            x, log_det_dfi = self.affine_coupling_f[i].inverse(x)
            log_det_df += log_det_dfi
            
            y, log_det_dgi = self.affine_coupling_g[i].inverse(y)
            log_det_dg += log_det_dgi
        return x, log_det_df, y, log_det_dg

class MaskedSplineFlows(NFModel):
    """
    RealNVP mode defined in the paper https://arxiv.org/abs/1605.08803.
    MLP is needed to make sure the scaling between layers are more or less the same.

    Args:
        n_layer: (int) The number of affine coupling layers.
        n_features: (int) The number of features in the input.
        n_hidden: (int) The number of hidden units in the MLP.
        dt: (float) Scaling factor for the affine coupling layer.

    Properties:
        data_mean: (ndarray) Mean of Gaussian base distribution
        data_cov: (ndarray) Covariance of Gaussian base distribution
    """
    affine_coupling_f: Transformed
    affine_coupling_g: Transformed
    _n_features_x: int
    _n_features_y: int

    @property
    def n_features_x(self):
        return self._n_features_x
    
    @property
    def n_features_y(self):
        return self._n_features_y
    
    # @property
    # def meanx(self):
    #     return jax.lax.stop_gradient(self.affine_coupling_f.base_dist.loc)
    
    # @property
    # def meany(self):
    #     return jax.lax.stop_gradient(self.affine_coupling_g.base_dist.loc)
    
    # @property
    # def scalex(self):
    #     return jax.lax.stop_gradient(self.affine_coupling_f.base_dist.scale)
    
    # @property
    # def scaley(self):
    #     return jax.lax.stop_gradient(self.affine_coupling_g.base_dist.scale)


    def __init__(self,
                n_features_x: int,
                n_features_y: int,
                key: jax.random.PRNGKey,
                flow_layers: int = 8,
                nn_width: int = 50,
                nn_depth: int = 1,
                knots: int = 8,
                interval: int = 4,
                **kwargs):
        f_subkey, g_subkey = jax.random.split(key, 2)
        
        # Args:
        # key: Array,
        # *,
        # base_dist: AbstractDistribution,
        # transformer: AbstractBijection | None = None,
        # cond_dim: int | None = None,
        # flow_layers: int = 8,
        # nn_width: int = 50,
        # nn_depth: int = 1,
        # nn_activation: Callable = jnn.relu,
        # invert: bool = True,
        
        
        self._n_features_x = n_features_x
        self.affine_coupling_f = masked_autoregressive_flow(f_subkey,
                                                            base_dist=Normal(jnp.zeros(n_features_x)),
                                                            transformer=RationalQuadraticSpline(knots=knots, 
                                                                                                interval=interval),
                                                            flow_layers=flow_layers,
                                                            nn_width= nn_width,
                                                            nn_depth=nn_depth,
                                                            invert = False,
                                                            )
        
        self._n_features_y = n_features_y 
        self.affine_coupling_g = masked_autoregressive_flow(g_subkey,
                                                            base_dist=Normal(jnp.zeros(n_features_y)),
                                                            transformer=RationalQuadraticSpline(knots=knots, 
                                                                                                interval=interval),
                                                            flow_layers=flow_layers,
                                                            nn_width= nn_width,
                                                            nn_depth=nn_depth,
                                                            invert = False,
                                                            )
        
    def __call__(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
        return self.forward(x,y)

    def forward(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
        x, log_det_df = self.affine_coupling_f.bijection.transform_and_log_det(x)
           
        y, log_det_dg = self.affine_coupling_g.bijection.transform_and_log_det(y)     
        return x, log_det_df, y, log_det_dg
    
    def inverse(self, x: Array, y: Array) -> Tuple[Array, Array, Array, Array]:
        x, log_det_df = self.affine_coupling_f.bijection.inverse_and_log_det(x)
            
        y, log_det_dg = self.affine_coupling_g.bijection.inverse_and_log_det(y) 
        return x, log_det_df, y, log_det_dg


@eqx.filter_jit
def _FlowMP_value(
    flows: MaskedSplineFlows,
    xs: jnp.ndarray,
    ys: jnp.ndarray):
    n_sample, dim_x = xs.shape
    _, dim_y = ys.shape
    flows_vmap = jax.vmap(flows, in_axes=(0, 0))
    fX, logDetJfX, gY, logDetJgY = flows_vmap(xs, ys)
    fX_gY = jnp.concatenate((fX, gY), axis=1)
    
    Sigma = jnp.cov(fX_gY.T)

    hX = 0.5 * jnp.linalg.slogdet(Sigma[:dim_x, :dim_x])[1] + dim_x / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / n_sample) * jnp.sum(logDetJfX)#logDetJfX.shape[0]
    
    hXY = 0.5 * jnp.linalg.slogdet(Sigma)[1] + (dim_x+dim_y) / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / n_sample) * jnp.sum(logDetJfX)#- (1 / n_sample) * jnp.sum(logDetJgY)
    hY = 0.5 * jnp.linalg.slogdet(Sigma[dim_x:, dim_x:])[1] + dim_y / 2 * (1 + jnp.log(2 * jnp.pi))# - (1 / n_sample) * jnp.sum(logDetJgY)
    hX_Y = hXY-hY
    
    value = hX - hX_Y
    return value

@eqx.filter_value_and_grad
def loss_function_FlowMargPost(flows, xs, ys):
    n_sample, dim_x = xs.shape
    _, dim_y = ys.shape
    flows_vmap = jax.vmap(flows, in_axes=(0, 0))
    fX, logDetJfX, gY, logDetJgY = flows_vmap(xs, ys)
    fX_gY = jnp.concatenate((fX, gY), axis=1)
    
    Sigma = jnp.cov(fX_gY.T) + 1e-4*jnp.eye(dim_x+dim_y)
    
    hX = 0.5 * jnp.linalg.slogdet(Sigma[:dim_x, :dim_x])[1] + dim_x / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / n_sample) * jnp.sum(logDetJfX)
    
    hXY = 0.5 * jnp.linalg.slogdet(Sigma)[1] + (dim_x+dim_y) / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / n_sample) * jnp.sum(logDetJfX)- (1 / n_sample) * jnp.sum(logDetJgY)
    hY = 0.5 * jnp.linalg.slogdet(Sigma[dim_x:, dim_x:])[1] + dim_y / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / (n_sample)) * jnp.sum(logDetJgY)
    hX_Y = hXY-hY
    hY_X = hXY-hX
    
    
    loss = hX + hX_Y + hY_X + hY
    return loss+jax.lax.stop_gradient(hX-hX_Y-loss)


def _FlowMargPost_value_neg_grad(
    flows: MaskedSplineFlows,
    xs: jnp.ndarray,
    ys: jnp.ndarray):
    # Define functions to compute gradients using JAX's autograd
    value, neg_grad = loss_function_FlowMargPost(flows, xs, ys)
    return value, neg_grad


@eqx.filter_jit
def _FlowP_value(
    flows: MaskedSplineFlows,
    xs: jnp.ndarray,
    ys: jnp.ndarray):
    n_sample, dim_x = xs.shape
    _, dim_y = ys.shape
    flows_vmap = jax.vmap(flows, in_axes=(0, 0))
    fX, logDetJfX, gY, logDetJgY = flows_vmap(xs, ys)
    fX_gY = jnp.concatenate((fX, gY), axis=1)
    
    Sigma = jnp.cov(fX_gY.T)

    mux = jnp.zeros(dim_x)
    Sigmax = jnp.eye(dim_x)
    logmarg = jax.scipy.stats.multivariate_normal.logpdf(xs, mux, Sigmax)
    hX = -jnp.mean(logmarg)
    
    hXY = 0.5 * jnp.linalg.slogdet(Sigma)[1] + (dim_x+dim_y) / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / n_sample) * jnp.sum(logDetJfX)#- (1 / n_sample) * jnp.sum(logDetJgY)
    hY = 0.5 * jnp.linalg.slogdet(Sigma[dim_x:, dim_x:])[1] + dim_y / 2 * (1 + jnp.log(2 * jnp.pi))# - (1 / n_sample) * jnp.sum(logDetJgY)
    hX_Y = hXY-hY
    
    value = hX - hX_Y
    return value

@eqx.filter_value_and_grad
def loss_function_FlowPost(flows, xs, ys):
    n_sample, dim_x = xs.shape
    _, dim_y = ys.shape
    flows_vmap = jax.vmap(flows, in_axes=(0, 0))
    fX, logDetJfX, gY, logDetJgY = flows_vmap(xs, ys)
    fX_gY = jnp.concatenate((fX, gY), axis=1)
    
    Sigma = jnp.cov(fX_gY.T) + 1e-4*jnp.eye(dim_x+dim_y)
    
    mux = jnp.zeros(dim_x)
    Sigmax = jnp.eye(dim_x)
    logmarg = jax.scipy.stats.multivariate_normal.logpdf(xs, mux, Sigmax)
    # hX = 0.5 * jnp.linalg.slogdet(Sigma[:dim_x, :dim_x])[1] + dim_x / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / n_sample) * jnp.sum(logDetJfX)
    
    hXY = 0.5 * jnp.linalg.slogdet(Sigma)[1] + (dim_x+dim_y) / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / n_sample) * jnp.sum(logDetJfX)- (1 / n_sample) * jnp.sum(logDetJgY)
    hY = 0.5 * jnp.linalg.slogdet(Sigma[dim_x:, dim_x:])[1] + dim_y / 2 * (1 + jnp.log(2 * jnp.pi)) - (1 / (n_sample)) * jnp.sum(logDetJgY)
    hX_Y = hXY-hY
    
    
    loss = hX_Y
    return loss +jax.lax.stop_gradient(-jnp.mean(logmarg)-2*hX_Y)

def _FlowPost_value_neg_grad(
    flows: MaskedSplineFlows,
    xs: jnp.ndarray,
    ys: jnp.ndarray):
    # Define functions to compute gradients using JAX's autograd
    value, neg_grad = loss_function_FlowPost(flows, xs, ys)
    return value, neg_grad

def Flow_training(
    rng: jax.random.PRNGKeyArray,
    flows: NFModel,
    xs: BatchedPoints,
    ys: BatchedPoints,
    xs_test: Optional[BatchedPoints] = None,
    ys_test: Optional[BatchedPoints] = None,
    batch_size: Optional[int] = 256,
    test_every_n_steps: int = 250,
    max_n_steps: int = 2_000,
    early_stopping: bool = False,
    learning_rate: float = 0.1,
    verbose: bool = False,
    MargPost_loss:  bool = True
) -> tuple[TrainingLog, eqx.Module]:
    """Basic training loop for MINE.

    Args:
        rng: random key
        critic: critic to be trained
        xs: samples of X, shape (n_points, dim_x)
        ys: paired samples of Y, shape (n_points, dim_y)
        xs_test: samples of X used for computing test MI, shape (n_points_test, dim_x),
          if None will reuse xs
        ys_test: paired samples of Y used for computing test MI, shape (n_points_test, dim_y),
          if None will reuse ys
        batch_size: batch size
        test_every_n_steps: step intervals at which the training checkpoint should be saved
        max_n_steps: maximum number of steps
        early_stopping: whether training should stop early when test MI stops growing
        learning_rate: learning rate to be used
        verbose: print info during training

    Returns:
        training log
        trained critic
    """
    xs_test = xs_test if xs_test is not None else xs
    ys_test = ys_test if ys_test is not None else ys

    # initialize the optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(eqx.filter(flows, eqx.is_array))

    # compile the training step   flows.affine_coupling_g.base_dist.scale
    @eqx.filter_jit
    def step(
        *,
        flows,
        opt_state,
        MargPost_loss,
        xs: BatchedPoints,
        ys: BatchedPoints):
        
        if MargPost_loss:
            value, neg_grad = _FlowMargPost_value_neg_grad(
                flows=flows,
                xs=xs,
                ys=ys)
        else:
            value, neg_grad = _FlowPost_value_neg_grad(
                flows=flows,
                xs=xs,
                ys=ys)
        
        updates, opt_state = optimizer.update(neg_grad, opt_state)
        flows = eqx.apply_updates(flows, updates)
        return flows, opt_state, value
        
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
        flows, opt_state, mi_train = step(
            flows=flows,
            opt_state = opt_state,
            MargPost_loss= MargPost_loss,
            xs=batch_xs,
            ys=batch_ys)

        # logging train
        training_log.log_train_mi(n_step, mi_train)

        # logging test
        if n_step % test_every_n_steps == 0:
            if MargPost_loss:
                mi_test = _FlowMP_value(
                    flows=flows, xs=xs_test, ys=ys_test
                )
            else:
                mi_test = _FlowP_value(
                    flows=flows, xs=xs_test, ys=ys_test
                )
            training_log.log_test_mi(n_step, mi_test)
            
        # early stop?
        if training_log.early_stop():
            break

    training_log.finish()
    return training_log, flows


class FlowParams(BaseModel):
    batch_size: pydantic.PositiveInt
    max_n_steps: pydantic.PositiveInt
    train_test_split: Optional[pydantic.confloat(gt=0.0, lt=1.0)]
    test_every_n_steps: pydantic.PositiveInt
    learning_rate: pydantic.PositiveFloat
    standardize: bool
    seed: int
    flow_layers: pydantic.PositiveInt
    nn_width: pydantic.PositiveInt
    nn_depth: pydantic.PositiveInt
    knots: pydantic.PositiveInt
    interval: pydantic.PositiveInt

class FlowMargPostEstimator(IMutualInformationPointEstimator):
    def __init__(
        self,
        batch_size: int =  _estimators._DEFAULT_BATCH_SIZE,
        max_n_steps: int = _estimators._DEFAULT_N_STEPS,
        train_test_split: Optional[float] = _estimators._DEFAULT_TRAIN_TEST_SPLIT,
        test_every_n_steps: int = _estimators._DEFAULT_TEST_EVERY_N,
        learning_rate: float = _estimators._DEFAULT_LEARNING_RATE,
        flow_layers: int = 8,
        nn_width: int = 50,
        nn_depth: int = 1,
        knots: int = 8,
        interval: int = 4,
        standardize: bool = _estimators._DEFAULT_STANDARDIZE,
        verbose: bool = _estimators._DEFAULT_VERBOSE,
        seed: int = _estimators._DEFAULT_SEED,
    ) -> None:
        self._params = FlowParams(
            batch_size=batch_size,
            max_n_steps=max_n_steps,
            train_test_split=train_test_split,
            test_every_n_steps=test_every_n_steps,
            learning_rate=learning_rate,
            standardize=standardize,
            seed=seed,
            flow_layers = flow_layers, 
            nn_width = nn_width,
            nn_depth = nn_depth,
            knots = knots,
            interval = interval)
        
        self._verbose = verbose
        self._training_log: Optional[TrainingLog] = None

        # After the training we will store the trained
        # critic function here
        self._trained_flows = None

    @property
    def trained_flows(self) -> Optional[eqx.Module]:
        """Returns the critic function from the end of the training.

        Note:
          1. You need to train the model by estimating mutual information,
            otherwise `None` is returned.
          2. Note that the critic can have different meaning depending on
            the function used.
        """
        return self._trained_flows

    def parameters(self) -> FlowParams:
        return self._params

    def _create_flows(self, dim_x: int, dim_y: int, key: jax.random.PRNGKeyArray) -> MaskedSplineFlows:#RealNVP:#
        return MaskedSplineFlows(n_features_x = dim_x, 
                                 n_features_y = dim_y, 
                                 key=key, 
                                 flow_layers = self._params.flow_layers, 
                                 nn_width = self._params.nn_width,
                                 nn_depth = self._params.nn_depth,
                                 knots = self._params.knots,
                                 interval = self._params.interval,)
    
    def estimate_with_info(self, x: ArrayLike, y: ArrayLike) -> EstimateResult:
        key = jax.random.PRNGKey(self._params.seed)
        key_init, key_split, key_fit = jax.random.split(key, 3)

        # standardize the data, note we do so before splitting into train/test
        space = ProductSpace(x, y, standardize=self._params.standardize)
        xs, ys = jnp.asarray(space.x), jnp.asarray(space.y)

        # split
        xs_train, xs_test, ys_train, ys_test = _estimators.train_test_split(
            xs, ys, train_size=self._params.train_test_split, key=key_split
        )

        # initialize critic
        _flows = self._create_flows(dim_x=space.dim_x, dim_y=space.dim_y, key=key_init)

        training_log, trained_flows = Flow_training(
            rng=key_fit,
            flows = _flows,
            xs=xs_train,
            ys=ys_train,
            xs_test=xs_test,
            ys_test=ys_test,
            batch_size=self._params.batch_size,
            test_every_n_steps=self._params.test_every_n_steps,
            max_n_steps=self._params.max_n_steps,
            learning_rate=self._params.learning_rate,
            verbose=self._verbose,
            MargPost_loss=True,
        )
        
        self._trained_flows = trained_flows

        return EstimateResult(
            mi_estimate=training_log.final_mi,
            additional_information=training_log.additional_information)

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        return self.estimate_with_info(x, y).mi_estimate


class FlowPostEstimator(IMutualInformationPointEstimator):
    def __init__(
        self,
        batch_size: int =  _estimators._DEFAULT_BATCH_SIZE,
        max_n_steps: int = _estimators._DEFAULT_N_STEPS,
        train_test_split: Optional[float] = _estimators._DEFAULT_TRAIN_TEST_SPLIT,
        test_every_n_steps: int = _estimators._DEFAULT_TEST_EVERY_N,
        learning_rate: float = _estimators._DEFAULT_LEARNING_RATE,
        flow_layers: int = 8,
        nn_width: int = 50,
        nn_depth: int = 1,
        knots: int = 8,
        interval: int = 4,
        standardize: bool = _estimators._DEFAULT_STANDARDIZE,
        verbose: bool = _estimators._DEFAULT_VERBOSE,
        seed: int = _estimators._DEFAULT_SEED,
    ) -> None:
        self._params = FlowParams(
            batch_size=batch_size,
            max_n_steps=max_n_steps,
            train_test_split=train_test_split,
            test_every_n_steps=test_every_n_steps,
            learning_rate=learning_rate,
            standardize=standardize,
            seed=seed,
            flow_layers = flow_layers, 
            nn_width = nn_width,
            nn_depth = nn_depth,
            knots = knots,
            interval = interval)
        
        self._verbose = verbose
        self._training_log: Optional[TrainingLog] = None

        # After the training we will store the trained
        # critic function here
        self._trained_flows = None

    @property
    def trained_flows(self) -> Optional[eqx.Module]:
        """Returns the critic function from the end of the training.

        Note:
          1. You need to train the model by estimating mutual information,
            otherwise `None` is returned.
          2. Note that the critic can have different meaning depending on
            the function used.
        """
        return self._trained_flows

    def parameters(self) -> FlowParams:
        return self._params

    def _create_flows(self, dim_x: int, dim_y: int, key: jax.random.PRNGKeyArray) -> MaskedSplineFlows:#RealNVP:#
        return MaskedSplineFlows(n_features_x = dim_x, 
                                 n_features_y = dim_y, 
                                 key=key, 
                                 flow_layers = self._params.flow_layers, 
                                 nn_width = self._params.nn_width,
                                 nn_depth = self._params.nn_depth,
                                 knots = self._params.knots,
                                 interval = self._params.interval,)
    
    def estimate_with_info(self, x: ArrayLike, y: ArrayLike) -> EstimateResult:
        key = jax.random.PRNGKey(self._params.seed)
        key_init, key_split, key_fit = jax.random.split(key, 3)

        # standardize the data, note we do so before splitting into train/test
        space = ProductSpace(x, y, standardize=self._params.standardize)
        xs, ys = jnp.asarray(space.x), jnp.asarray(space.y)

        # split
        xs_train, xs_test, ys_train, ys_test = _estimators.train_test_split(
            xs, ys, train_size=self._params.train_test_split, key=key_split
        )

        # initialize critic
        _flows = self._create_flows(dim_x=space.dim_x, dim_y=space.dim_y, key=key_init)

        training_log, trained_flows = Flow_training(
            rng=key_fit,
            flows = _flows,
            xs=xs_train,
            ys=ys_train,
            xs_test=xs_test,
            ys_test=ys_test,
            batch_size=self._params.batch_size,
            test_every_n_steps=self._params.test_every_n_steps,
            max_n_steps=self._params.max_n_steps,
            learning_rate=self._params.learning_rate,
            verbose=self._verbose,
            MargPost_loss= False,
        )
        
        self._trained_flows = trained_flows

        return EstimateResult(
            mi_estimate=training_log.final_mi,
            additional_information=training_log.additional_information)

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        return self.estimate_with_info(x, y).mi_estimate


class MMParams(BaseModel):
    train_test_split: Optional[pydantic.confloat(gt=0.0, lt=1.0)]
    standardize: bool
    seed: int
    
class MargPostEstimator(IMutualInformationPointEstimator):
    def __init__(
        self,
        train_test_split: Optional[float] = _estimators._DEFAULT_TRAIN_TEST_SPLIT,
        standardize: bool = _estimators._DEFAULT_STANDARDIZE,
        seed: int = _estimators._DEFAULT_SEED,
    ) -> None:
        self._params = MMParams(
            train_test_split=train_test_split,
            standardize=standardize,
            seed=seed,)
        
        self._training_log: Optional[TrainingLog] = None

        # After the training we will store the trained
        # critic function here
        self._trained_flows = None

    @property
    def trained_flows(self) -> Optional[eqx.Module]:
        """Returns the critic function from the end of the training.

        Note:
          1. You need to train the model by estimating mutual information,
            otherwise `None` is returned.
          2. Note that the critic can have different meaning depending on
            the function used.
        """
        return self._trained_flows

    def parameters(self) -> FlowParams:
        return self._params
    
    def estimate_with_info(self, x: ArrayLike, y: ArrayLike) -> EstimateResult:
        key = jax.random.PRNGKey(self._params.seed)
        key_init, key_split, key_fit = jax.random.split(key, 3)

        # standardize the data, note we do so before splitting into train/test
        space = ProductSpace(x, y, standardize=self._params.standardize)
        xs, ys = jnp.asarray(space.x), jnp.asarray(space.y)

        # split
        xs_train, xs_test, ys_train, ys_test = _estimators.train_test_split(
            xs, ys, train_size=self._params.train_test_split, key=key_split
        )

        data = {}
        data['x_train'] = xs_train
        data['y_train'] = ys_train
        data['x_test'] = xs_test
        data['y_test'] = ys_test
        
        XY = np.concatenate((xs_train, ys_train), axis=1)
        Sigma = np.cov(XY.T)
        dim_x = xs_train.shape[1]
        hX = 0.5 * np.linalg.slogdet(Sigma[:dim_x, :dim_x])[1] + dim_x / 2 * (1 + np.log(2 * np.pi))
        hXY = 0.5 * np.linalg.slogdet(Sigma)[1] + (2*dim_x) / 2 * (1 + np.log(2 * np.pi))
        hY = 0.5 * np.linalg.slogdet(Sigma[dim_x:, dim_x:])[1] + dim_x / 2 * (1 + np.log(2 * np.pi))
        hX_Y = hXY-hY
        value = hX - hX_Y 
        
        return EstimateResult(
            mi_estimate=value,
            additional_information=data)

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        return self.estimate_with_info(x, y).mi_estimate


# ########### Plotting Function ######################################################

# def plot_final(
#     flows,
#     xs: BatchedPoints,
#     ys: BatchedPoints):
#     flows_vmap = jax.vmap(flows, in_axes=(0, 0))
#     fXs, logDetJfX, gYs, logDetJgY = flows_vmap(xs, ys)
#     fX_gYs = jnp.concatenate((fXs, gYs), axis=1)
#     Mu = jnp.mean(fX_gYs,axis=0)
#     Sigma = jnp.cov(fX_gYs.T)
    
#     n_sample, dim_x = xs.shape
#     XY = np.random.multivariate_normal(Mu,Sigma,10000)
#     flows_inv_vmap = jax.vmap(flows.inverse, in_axes=(0, 0))
#     fX, logDetJfX, gY, logDetJgY  = flows_inv_vmap(XY[:,:dim_x], XY[:,dim_x:])
    
#     if dim_x == 1:
#         plt.scatter(fX, gY, s=10, c='black', alpha=0.5) 
#         # Add labels and title
#         plt.xlabel('X-axis label')
#         plt.ylabel('Y-axis label')
#         plt.title('Scatter Plot of X and Y')
#         # Show the plot
#         plt.show()
#         import scipy
#         xtest = np.linspace(-5,5,2000)
#         fxtest, logDetJfX, gYtest, logDetJgY = flows_vmap(xtest, xtest)

#         fmarg = scipy.stats.multivariate_normal.pdf(fxtest, Mu[:dim_x], Sigma[:dim_x,:dim_x])*np.exp(logDetJfX).T
#         gmarg = scipy.stats.multivariate_normal.pdf(gYtest, Mu[dim_x:], Sigma[dim_x:,dim_x:])*np.exp(logDetJgY).T
#         plt.plot(xtest, fmarg.flatten())
#         # Add labels and title
#         plt.xlabel('X-axis label')
#         plt.ylabel('p(x)-axis label')
#         plt.title('x marginal')
#         # Show the plot
#         plt.show()

#         plt.plot(xtest, gmarg.flatten())
#         # Add labels and title
#         plt.xlabel('Y-axis label')
#         plt.ylabel('p(y)-axis label')
#         plt.title('y marginal')
#         # Show the plot
#         plt.show()
        
    
#     if dim_x == 2:
#         # xtest = (np.linspace(-4,4,100)*np.ones((2,100))).T
#         # actualX = xtest[:,0] + 0.4 * np.sin(1.0 * xtest[:,0]) + 0.2 * np.sin(1.7 * xtest[:,0] + 1) + 0.03 * np.sin(3.3 * xtest[:,0] - 2.5)
#         # actualY = xtest[:,0] - 0.4 * np.sin(0.4 * xtest[:,0]) + 0.17 * np.sin(1.3 * xtest[:,0] + 3.5) + 0.02 * np.sin(4.3 * xtest[:,0] - 2.5)
#         # fxtest, logDetJfX, gYtest, logDetJgY = flows_inv_vmap(xtest, xtest)
        
#         # plt.plot(xtest[:,0], fxtest[:,0],label='fX0')
#         # plt.plot(xtest[:,1], fxtest[:,1],label='fX1')
#         # plt.plot(xtest[:,1], actualX,label='True Transform')
#         # plt.plot(xtest[:,1], -actualX,label='Negative True')
#         # # Add labels and title
#         # plt.xlabel('X-axis label')
#         # plt.ylabel('f(x)-axis label')
#         # plt.title('Learned X Flow')
#         # plt.legend()
#         # # Show the plot
#         # plt.show()
        
#         # plt.plot(xtest[:,0], gYtest[:,0],label='gY0')
#         # plt.plot(xtest[:,1], gYtest[:,1],label='gY1')
#         # plt.plot(xtest[:,1], actualY,label='True Transform')
#         # plt.plot(xtest[:,1], -actualY,label='Negative True')
#         # # Add labels and title
#         # plt.xlabel('Y-axis label')
#         # plt.ylabel('g(y)-axis label')
#         # plt.title('Learned Y Flow')
#         # plt.legend()
#         # # Show the plot
#         # plt.show()
#         # fX[:,0][:,0], fX[:,0]  #fX[:,0][:,1], fX[:,1]
#         plt.scatter(fX[:,0], fX[:,1], s=10, c='black', alpha=0.5)
#         # Add labels and title
#         plt.xlabel('X0-axis label')
#         plt.ylabel('X1-axis label')
#         plt.title('Scatter Plot of X and Y')

#         # Show the plot
#         plt.show()
#         #fX[:,0][:,0], fX[:,0] 
#         plt.scatter(fX[:,0], gY[:,0], s=10, c='black', alpha=0.5)
#         # Add labels and title
#         plt.xlabel('X0-axis label')
#         plt.ylabel('Y0-axis label')
#         plt.title('Scatter Plot of X and Y')

#         # Show the plot
#         plt.show()
#         #fX[:,0][:,1], fX[:,1]
#         plt.scatter(fX[:,1], gY[:,1], s=10, c='black', alpha=0.5)
#         # Add labels and title
#         plt.xlabel('X1-axis label')
#         plt.ylabel('Y1-axis label')
#         plt.title('Scatter Plot of X and Y')

#         # Show the plot
#         plt.show()
        
#         plt.scatter(gY[:,0], gY[:,1], s=10, c='black', alpha=0.5)
#         # Add labels and title
#         plt.xlabel('Y0-axis label')
#         plt.ylabel('Y1-axis label')
#         plt.title('Scatter Plot of X and Y')

#         # Show the plot
#         plt.show()    