import bmi
from _flow_estimator import FlowMargPostEstimator
import numpy as np
import jax
import scipy
from matplotlib.pyplot import colorbar, pcolor, show, scatter
import matplotlib.pyplot as plt
import jax.numpy as jnp
from bmi.utils import ProductSpace

from scipy.ndimage.filters import gaussian_filter
def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x.flatten(), y.flatten(), bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

num_samples = 20000
seed = 42
batch_size = 256
num_steps = 1000
lr = .003
test_every_n_steps = 1000
train_test_split = .8

task = bmi.benchmark.BENCHMARK_TASKS['1v1-bimodal-0.75']
X, Y = task.sample(num_samples, seed=seed)
n_sample, dim_x = X.shape

# plt.scatter(X, Y, s=10, c='black', alpha=0.5) 
# # Add labels and title
# plt.xlabel('X-axis label')
# plt.ylabel('Y-axis label')
# plt.title('Scatter Plot of X and Y')
# # Show the plot
# plt.show()


# Points = np.hstack((X,Y))
# mu = np.mean(Points,axis=0)
# Sigma = np.cov(Points.T)
# XY = np.random.multivariate_normal(mu,Sigma,10000)
# plt.scatter(XY[:,:dim_x], XY[:,dim_x:], s=10, c='black', alpha=0.5) 
# # Add labels and title
# plt.xlabel('X-axis label')
# plt.ylabel('Y-axis label')
# plt.title('Scatter Plot of X and Y')
# # Show the plot
# plt.show()

space = ProductSpace(X, Y, standardize=True)
X, Y = jnp.asarray(space.x), jnp.asarray(space.y)

# plt.scatter(X, Y, s=10, c='black', alpha=0.5) 
# # Add labels and title
# plt.xlabel('X-axis label')
# plt.ylabel('Y-axis label')
# plt.title('Scatter Plot of X and Y')
# # Show the plot
# plt.show()

img, extent = myplot(X, Y, 32)
plt.imshow(img, extent=extent, origin='lower')#, cmap=cm.jet
plt.title("True: $I(X,Y)=%0.3f$"% task.mutual_information)
plt.xlim([np.min(X), np.max(X)])
plt.ylim([np.min(Y), np.max(Y)])
plt.show()
print(task.mutual_information)

Points = np.hstack((X,Y))
mu = np.mean(Points,axis=0)
Sigma = np.cov(Points.T)
hX = 0.5 * np.linalg.slogdet(Sigma[:dim_x, :dim_x])[1] + dim_x / 2 * (1 + np.log(2 * np.pi))
hXY = 0.5 * np.linalg.slogdet(Sigma)[1] + (2*dim_x) / 2 * (1 + np.log(2 * np.pi))
hY = 0.5 * np.linalg.slogdet(Sigma[dim_x:, dim_x:])[1] + dim_x / 2 * (1 + np.log(2 * np.pi))
hX_Y = hXY-hY
value = hX - hX_Y 
XY = np.random.multivariate_normal(mu,Sigma,20000)
# plt.scatter(XY[:,:dim_x], XY[:,dim_x:], s=10, c='black', alpha=0.5) 
# # Add labels and title
# plt.xlabel('X-axis label')
# plt.ylabel('Y-axis label')
# plt.title('Scatter Plot of X and Y')
# # Show the plot
# plt.show()

img, extent = myplot(XY[:,:dim_x], XY[:,dim_x:], 32)
plt.imshow(img, extent=extent, origin='lower')#, cmap=cm.jet
plt.title("Variational: $I_{m+p}(X,Y)=%0.3f$"%value)
plt.xlim([np.min(X), np.max(X)])
plt.ylim([np.min(Y), np.max(Y)])
plt.show()
print(value)


MMFlow = FlowMargPostEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split,flow_layers=3,knots=16)#max_n_steps=int(100*(i+1))
MMFLowResults = MMFlow.estimate_with_info(X,Y)

flows_vmap = jax.vmap(MMFlow.trained_flows, in_axes=(0, 0))
fX, logdetf, gY, logdetg = flows_vmap(X,Y)
Flow_Points = np.hstack((fX,gY))
mu_flow = np.mean(Flow_Points,axis=0)
Sigma_flow = np.cov(Flow_Points.T)
    
XY = np.random.multivariate_normal(mu_flow,Sigma_flow,20000)
flows_inv_vmap = jax.vmap(MMFlow.trained_flows.inverse, in_axes=(0, 0))
fX, logDetJfX, gY, logDetJgY  = flows_inv_vmap(XY[:,:dim_x], XY[:,dim_x:])

# plt.scatter(fX, gY, s=10, c='black', alpha=0.5) 
# # Add labels and title
# plt.xlabel('X-axis label')
# plt.ylabel('Y-axis label')
# plt.title('Scatter Plot of X and Y')
# # Show the plot
# plt.show()

img, extent = myplot(fX, gY, 32)
plt.imshow(img, extent=extent, origin='lower')#, cmap=cm.jet
plt.title("Variational: $I_{m+p}(X,Y)=%0.3f$"%MMFLowResults.mi_estimate)
plt.xlim([np.min(X), np.max(X)])
plt.ylim([np.min(Y), np.max(Y)])
plt.show()
print(MMFLowResults.mi_estimate)

# # x_plot = np.linspace(-5, 10,500)
# # y_plot = np.linspace(-7.5, 7.5,500)
# x_plot = np.linspace(-3, 2,500)
# y_plot = np.linspace(-2.5, 2.5,500)
# # x = np.linspace(0.01,np.max([1, true_theta[i][0]+.2]),100)
# # y = np.linspace(0.01,np.max([.2, true_theta[i][1]+.05]),100)
# X_plot, Y_plot = np.meshgrid(x_plot, y_plot)

# flows_vmap = jax.vmap(MMFlow.trained_flows, in_axes=(0, 0))
# fX, logdetf, gY, logdetg = flows_vmap(X,Y)
# Flow_Points = np.hstack((fX,gY))
# mu_flow = np.mean(Flow_Points,axis=0)
# Sigma_flow = np.cov(Flow_Points.T)
# flows_vmap_inv = jax.vmap(MMFlow.trained_flows.inverse, in_axes=(0, 0))
# fX_plot,logdetJfX_plot, gY_plot,logdetJgY_plot = flows_vmap(X_plot.flatten().reshape(-1,1),Y_plot.flatten().reshape(-1,1))
# Flow_Points_plot = np.hstack((fX_plot,gY_plot))
# points = Flow_Points_plot.reshape((500,500,2))
# Z = scipy.stats.multivariate_normal.pdf(points, mu_flow, Sigma_flow)*np.exp(logdetJfX_plot.reshape((500,500)))*np.exp(logdetJgY_plot.reshape((500,500)))
# pcolor(X_plot, Y_plot, Z)


