import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from _flow_estimator import JointVariationalEstimator, NeuralVariationalEstimator
import jax

class GaussianMixtureModel:
    def __init__(self, means, covariances, weights=None):
        self.means = means
        self.covariances = covariances
        self.k = len(means)
        self.weights = weights if weights is not None else [1.0 / self.k] * self.k

    def pdf(self, x):
        return sum(w * multivariate_normal.pdf(x, mean=m, cov=cov) for m, cov, w in zip(self.means, self.covariances, self.weights))
    
    def evaluate_marginal(self, points, dim=0):
        marginal_values = np.zeros(len(points))
        for i in range(self.k):
            mean = self.means[i][dim]
            cov = self.covariances[i][dim][dim]
            weight = self.weights[i]
            marginal_values += weight * multivariate_normal.pdf(points, mean=mean, cov=cov)
        return marginal_values

    def sample(self, n_samples):
        samples = []
        component_choices = np.random.choice(range(self.k), size=n_samples, p=self.weights)
        for choice in component_choices:
            samples.append(np.random.multivariate_normal(self.means[choice], self.covariances[choice]))
        return np.array(samples)

    def plot_marginal(self):
        x = np.linspace(-2.5, 2.5, 300)
        y = np.linspace(-2.5, 1.5, 300)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        
        Z = np.sum([w * multivariate_normal.pdf(pos, mean=m, cov=cov) for m, cov, w in zip(self.means, self.covariances, self.weights)], axis=0)
        levels = np.linspace(0,.6,50)
        plt.rcParams.update({'font.size': 45})
        plt.figure(figsize=(10, 10))
        cnt=plt.contourf(X, Y, Z, levels=levels, cmap='viridis')
        for c in cnt.collections:
            c.set_edgecolor("face")
        plt.title('True: p(x,y)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        plt.savefig('TrueJointExample.pdf')
        # plt.colorbar()
        # plt.show()
        
        # marginal_x = np.sum(Z, axis=1)
        # marginal_y = np.sum(Z, axis=0)
        
        # plt.figure(figsize=(10, 6))
        # plt.plot(x, marginal_x, label='Marginal X')
        # plt.plot(y, marginal_y, label='Marginal Y')
        # plt.title('Marginal Distributions')
        # plt.xlabel('Value')
        # plt.ylabel('Density')
        # plt.legend()
        # plt.show()
    
    def plot_conditional(self, given_x=None, given_y=None):
        x = np.linspace(-2.5, 2.5, 300)
        y = np.linspace(-2.5, 1.5, 300)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        
        Z = np.sum([w * multivariate_normal.pdf(pos, mean=m, cov=cov) for m, cov, w in zip(self.means, self.covariances, self.weights)], axis=0)
        
        if given_x is not None:
            conditional_y = Z[:, np.argmin(np.abs(x - given_x))]
            plt.figure(figsize=(10, 6))
            plt.plot(y, conditional_y, label=f'P(Y|X={given_x})')
            plt.title(f'Conditional Distribution P(Y|X={given_x})')
            plt.xlabel('Y')
            plt.ylabel('Density')
            plt.legend()
            plt.show()
        
        if given_y is not None:
            conditional_x = Z[np.argmin(np.abs(y - given_y)), :]
            plt.figure(figsize=(10, 6))
            plt.plot(x, conditional_x, label=f'P(X|Y={given_y})')
            plt.title(f'Conditional Distribution P(X|Y={given_y})')
            plt.xlabel('X')
            plt.ylabel('Density')
            plt.legend()
            plt.show()

# Example usage:[-1.25, -1.25], ,[0, 0] 
means = [[1.25, 0],[0, 0],[0, -1.25],[-1.25, 0]]#[[1, 0],[.5, np.sqrt(3)/2],[-1, 0],[-.5, -np.sqrt(3)/2],[-.5, np.sqrt(3)/2],[.5, -np.sqrt(3)/2]]#
covariances = [[[0.1, 0], [0, 0.1]],[[0.1, 0], [0, 0.1]],[[0.1, 0], [0, 0.1]],[[0.1, 0], [0, 0.1]]]
weights = None#[0.3, 0.35, 0.35]#

gmm = GaussianMixtureModel(means, covariances, weights)
gmm.plot_marginal()
# gmm.plot_conditional(given_x=1)
# gmm.plot_conditional(given_y=0)

samples = gmm.sample(150000)#
# plt.scatter(samples[:, 0], samples[:, 1], s=5)
# plt.title('Samples from GMM')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()
X = samples[:, 0].reshape(-1, 1)
Y = samples[:, 1].reshape(-1, 1)
levels = np.linspace(0,.65,50)#.45

plot_x = np.linspace(-2.5, 2.5, 300)
plot_y = np.linspace(-2.5, 1.5, 300)
plot_X, plot_Y = np.meshgrid(plot_x, plot_y)
plot_pos = np.dstack((plot_X, plot_Y))
true_y_marg = gmm.evaluate_marginal(plot_Y.flatten(), dim=1)

JointGauss = JointVariationalEstimator(dim_x=1,dim_y=1,use_flow=False,batch_size = 256, max_n_steps=15000, learning_rate=.005,test_every_n_steps=250,train_test_split=.5)
JointGaussResults = JointGauss.estimate_with_info(X,Y)

JVG_mean = samples.mean(axis=0)
JVG_sigma = np.cov(samples.T)
plot_JVG = multivariate_normal.pdf(plot_pos, mean=JVG_mean, cov=JVG_sigma)
plt.figure(figsize=(10, 10))
cnt = plt.contourf(plot_X, plot_Y, plot_JVG, levels=levels, cmap='viridis')
# This is the fix for the white lines between contour levels
for c in cnt.collections:
    c.set_edgecolor("face")
plt.title('JVG: q(x,y)')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.savefig('JVGJointExample.pdf')
# plt.colorbar()
# plt.show()

NeuralGauss = NeuralVariationalEstimator(dim_x=1,dim_y=1,use_flow=False,batch_size = 256, max_n_steps=15000, learning_rate=.005,test_every_n_steps=250,train_test_split=.5)
NeuralGaussResults = NeuralGauss.estimate_with_info(X,Y)
NVG_critic_vmap = jax.vmap(NeuralGauss.variational_model._critic_net.forward, in_axes=(0,0))
NVG_fX_post_vmap = jax.vmap(NeuralGauss.variational_model._fX_post, in_axes=(0,))
NVGfX, NVGlogDetJfX = NVG_fX_post_vmap(plot_X.flatten().reshape(-1,1))
NVG_cond = np.exp(NVG_critic_vmap(NVGfX,plot_Y.flatten().reshape(-1,1))+NVGlogDetJfX.flatten())
plot_NVG = (true_y_marg*NVG_cond).reshape(plot_X.shape)
plt.figure(figsize=(10, 10))
cnt = plt.contourf(plot_X, plot_Y, plot_NVG, levels=levels, cmap='viridis')
for c in cnt.collections:
    c.set_edgecolor("face")
plt.title('NVG: p(y)q(x|y)')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
# plt.colorbar()
plt.savefig('NVGJointExample.pdf')
# plt.show()

JointFlow = JointVariationalEstimator(dim_x=1,dim_y=1,use_flow=True,batch_size = 256, max_n_steps=15000, learning_rate=.005,test_every_n_steps=250,train_test_split=.5)
JointFlowResults = JointFlow.estimate_with_info(X,Y)

JVF_fX_vmap = jax.vmap(JointFlow.variational_model._fX, in_axes=(0,))
JVF_gy_vmap = jax.vmap(JointFlow.variational_model._gY, in_axes=(0,))
JVF_X, _ = JVF_fX_vmap(X)
JVF_Y, _ = JVF_fX_vmap(Y)
JVF_samples = np.hstack((JVF_X,JVF_Y))
JVF_mean = JVF_samples.mean(axis=0)
JVF_sigma = np.cov(JVF_samples.T)
JVF_plot_X, JVFlogDetJfX = JVF_fX_vmap(plot_X.flatten().reshape(-1,1))
JVF_plot_Y, JVFlogDetJgY = JVF_fX_vmap(plot_Y.flatten().reshape(-1,1))

plot_JVF = np.exp(multivariate_normal.logpdf(np.hstack((JVF_plot_X,JVF_plot_Y)), mean=JVF_mean, cov=JVF_sigma)+JVFlogDetJfX.flatten()+JVFlogDetJgY.flatten()).reshape(plot_X.shape)
plt.figure(figsize=(10, 10))
cnt = plt.contourf(plot_X, plot_Y, plot_JVF, levels=levels, cmap='viridis')
# This is the fix for the white lines between contour levels
for c in cnt.collections:
    c.set_edgecolor("face")
plt.title('JVF: q(x,y)')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.savefig('JVFJointExample.pdf')
# plt.colorbar()
# plt.show()


NeuralFlow = NeuralVariationalEstimator(dim_x=1,dim_y=1,use_flow=True,batch_size = 256, max_n_steps=15000, learning_rate=.005,test_every_n_steps=500,train_test_split=.5)
NeuralFlowResults = NeuralFlow.estimate_with_info(X,Y)

NVF_critic_vmap = jax.vmap(NeuralFlow.variational_model._critic_net.forward, in_axes=(0,0))
NVF_fX_post_vmap = jax.vmap(NeuralFlow.variational_model._fX_post, in_axes=(0,))
NVFfX, NVFlogDetJfX = NVF_fX_post_vmap(plot_X.flatten().reshape(-1,1))
NVF_cond = np.exp(NVF_critic_vmap(NVFfX,plot_Y.flatten().reshape(-1,1))+NVFlogDetJfX.flatten())
plot_NVF = (true_y_marg*NVF_cond).reshape(plot_X.shape)
plt.figure(figsize=(10, 10))
cnt = plt.contourf(plot_X, plot_Y, plot_NVF, levels=levels, cmap='viridis')
for c in cnt.collections:
    c.set_edgecolor("face")
plt.title('NVF: p(y)q(x|y)')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
# plt.colorbar()
plt.savefig('NVFJointExample.pdf')
# plt.show()

print("done")

# plt.figure(figsize=(10, 6))
# plt.contourf(plot_X, plot_Y, Z, levels=50, cmap='viridis')
# plt.title('Joint Distribution')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.colorbar()
# plt.show()