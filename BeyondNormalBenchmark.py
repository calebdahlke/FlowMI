import bmi
from _flow_estimator import MMFLowEstimator
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import bmi.estimators.neural._estimators as _estimators
from bmi.estimators.neural._training_log import TrainingLog
import pydantic

import bmi.estimators.external.r_estimators as r_estimators
import bmi.estimators.external.julia_estimators as julia_estimators

import bmi.benchmark.tasks.multinormal as multinormal

TASKS = ['1v1-normal-0.75',                             #0
'normal_cdf-1v1-normal-0.75',                           #1
'1v1-additive-0.1',                                     #2
'1v1-additive-0.75',                                    #3
'1v1-bimodal-0.75',                                     #4
'wiggly-1v1-normal-0.75',                               #5
'half_cube-1v1-normal-0.75',                            #6
'student-identity-1-1-1',                               #7
'asinh-student-identity-1-1-1',                         #8
'swissroll_x-normal_cdf-1v1-normal-0.75',               #9
'multinormal-dense-2-2-0.5',                            #10
'multinormal-dense-3-3-0.5',                            #11
'multinormal-dense-5-5-0.5',                            #12
'multinormal-dense-25-25-0.5',                          #13
'multinormal-dense-50-50-0.5',                          #14
'multinormal-sparse-2-2-2-2.0',                         #15
'multinormal-sparse-3-3-2-2.0',                         #16
'multinormal-sparse-5-5-2-2.0',                         #17
'student-identity-2-2-1',                               #18
'student-identity-2-2-2',                               #19
'student-identity-3-3-2',                               #20
'student-identity-3-3-3',                               #21
'student-identity-5-5-2',                               #22
'student-identity-5-5-3',                               #23
'normal_cdf-multinormal-sparse-3-3-2-2.0',              #24
'normal_cdf-multinormal-sparse-5-5-2-2.0',              #25
'normal_cdf-multinormal-sparse-25-25-2-2.0',            #26
'half_cube-multinormal-sparse-25-25-2-2.0',             #27
'spiral-multinormal-sparse-3-3-2-2.0',                  #28
'spiral-multinormal-sparse-5-5-2-2.0',                  #29
'spiral-multinormal-sparse-25-25-2-2.0',                #30
'spiral-normal_cdf-multinormal-sparse-3-3-2-2.0',       #31
'spiral-normal_cdf-multinormal-sparse-5-5-2-2.0',       #32
'spiral-normal_cdf-multinormal-sparse-25-25-2-2.0',     #33
'asinh-student-identity-2-2-1',                         #34
'asinh-student-identity-3-3-2',                         #35
'asinh-student-identity-5-5-2']                         #36

run_list = [9]#[10,14,18,24,31,34]#
n_tasks = len(run_list) #len(TASKS)

n_runs = 1
n_samples = 10000#30000#30000#35000#
learning_rate = .003#.001#.0003
max_n_steps = 10000#250#
test_every_n_steps = 50
train_test_split = .85 # % of samples dedicated to training.
batch_size = 256#train_test_split*n_samples#512#


True_value = np.zeros((n_tasks, n_runs))
Gauss_value = np.zeros((n_tasks, n_runs))
cca_value = np.zeros((n_tasks, n_runs))
ksg_value = np.zeros((n_tasks, n_runs))
MMFlow_value = np.zeros((n_tasks, n_runs))
mine_value = np.zeros((n_tasks, n_runs))
InfoNCE_value = np.zeros((n_tasks, n_runs))
NWJ_value = np.zeros((n_tasks, n_runs))
DV_value = np.zeros((n_tasks, n_runs))

method_names = ['True',  'CCA', 'KSG','GaussMM', 'MMFlow', 'MINE', 'InfoNCE', 'NWJ', 'DV']

task_names = []

for j in range(n_tasks):
    # task = bmi.benchmark.BENCHMARK_TASKS[TASKS[j]]#['spiral-multinormal-sparse-3-3-2-2.0']#['multinormal-dense-5-5-0.5']#['student-identity-2-2-1']#['spiral-multinormal-sparse-25-25-2-2.0']#['student-identity-5-5-2']#
    
    task = bmi.benchmark.BENCHMARK_TASKS[TASKS[run_list[j]]]
    
    task_names.append(task.name)
    
    print(f"Task {task.name} with dimensions {task.dim_x} and {task.dim_y}")
    print(f"Ground truth mutual information: {task.mutual_information}")#:.2f

    np.random.seed(42)#23

    for i in range(n_runs):
        True_value[j,i] = task.mutual_information
        
        X, Y = task.sample(n_samples, seed=i)#1000,42
        
        XY = np.concatenate((X, Y), axis=1)
        Sigma = np.cov(XY.T)
        dim_x = X.shape[1]
        hX = 0.5 * np.linalg.slogdet(Sigma[:dim_x, :dim_x])[1] + dim_x / 2 * (1 + np.log(2 * np.pi))
        hXY = 0.5 * np.linalg.slogdet(Sigma)[1] + (2*dim_x) / 2 * (1 + np.log(2 * np.pi))
        hY = 0.5 * np.linalg.slogdet(Sigma[dim_x:, dim_x:])[1] + dim_x / 2 * (1 + np.log(2 * np.pi))
        hX_Y = hXY-hY
        value = hX - hX_Y 
        Gauss_value[j,i] = value
        print(f"Gaussian Assumption mutual information: {value}")   
        
        cca = bmi.estimators.CCAMutualInformationEstimator()
        cca_value[j,i] = cca.estimate(X, Y)
        
        ksg = bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,))#10
        ksg_value[j,i] = ksg.estimate(X, Y)
        
        
        MMFlow = MMFLowEstimator(batch_size = batch_size, max_n_steps=max_n_steps, learning_rate=learning_rate,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)#max_n_steps=int(100*(i+1))
        MMFLowResults = MMFlow.estimate_with_info(X,Y)
        MMFlow_value[j,i] = np.array(MMFLowResults.additional_information['test_history'])[-1,1]#MMFLowResults.mi_estimate
        
        mine = bmi.estimators.MINEEstimator(batch_size = batch_size, max_n_steps=max_n_steps, learning_rate=learning_rate,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
        mineResults = mine.estimate_with_info(X,Y)
        mine_value[j,i] = mineResults.mi_estimate
        
        InfoNCE = bmi.estimators.InfoNCEEstimator(batch_size = batch_size, max_n_steps=max_n_steps, learning_rate=learning_rate,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
        InfoNCEResults = InfoNCE.estimate_with_info(X,Y)
        InfoNCE_value[j,i] = InfoNCEResults.mi_estimate
        
        NWJ = bmi.estimators.NWJEstimator(batch_size = batch_size, max_n_steps=max_n_steps, learning_rate=learning_rate,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
        NWJResults = NWJ.estimate_with_info(X,Y)
        NWJ_value[j,i] = NWJResults.mi_estimate
        
        DV = bmi.estimators.DonskerVaradhanEstimator(batch_size = batch_size, max_n_steps=max_n_steps, learning_rate=learning_rate,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
        DVResults = DV.estimate_with_info(X,Y)
        DV_value[j,i] = DVResults.mi_estimate
        
    if np.shape(X)[1] == 1:
        plt.scatter(X, Y, s=10, c='blue', alpha=0.5)  # You can customize the color, size, and transparency (alpha) as needed
        # Add labels and title
        plt.xlabel('X-axis label')
        plt.ylabel('Y-axis label')
        plt.title('Scatter Plot of X and Y')

        # Show the plot
        plt.show()
        
    if np.shape(X)[1] == 2:
        plt.scatter(X[:,0], X[:,1], s=10, c='blue', alpha=0.5)

        # Add labels and title
        plt.xlabel('X0-axis label')
        plt.ylabel('X1-axis label')
        plt.title('Scatter Plot of X and Y')

        # Show the plot
        plt.show()

        plt.scatter(X[:,0], Y[:,0], s=10, c='blue', alpha=0.5)

        # Add labels and title
        plt.xlabel('X0-axis label')
        plt.ylabel('Y0-axis label')
        plt.title('Scatter Plot of X and Y')

        # Show the plot
        plt.show()

        plt.scatter(X[:,1], Y[:,1], s=10, c='blue', alpha=0.5)

        # Add labels and title
        plt.xlabel('X1-axis label')
        plt.ylabel('Y1-axis label')
        plt.title('Scatter Plot of X and Y')

        # Show the plot
        plt.show()

        plt.scatter(Y[:,0], Y[:,1], s=10, c='blue', alpha=0.5)

        # Add labels and title
        plt.xlabel('Y0-axis label')
        plt.ylabel('Y1-axis label')
        # plt.yscale()
        plt.title('Scatter Plot of X and Y')

        # Show the plot
        plt.show()

    plt.plot(np.array(MMFLowResults.additional_information['test_history'])[:,0],task.mutual_information*np.ones(len(np.array(MMFLowResults.additional_information['test_history'])[:,1])),label='True', color='black')

    plt.plot(np.array(MMFLowResults.additional_information['training_history'])[:,0],np.array(MMFLowResults.additional_information['training_history'])[:,1],label='MMFlow Train', color='blue', linestyle='dashed', alpha = .5)
    plt.plot(np.array(MMFLowResults.additional_information['test_history'])[:,0],np.array(MMFLowResults.additional_information['test_history'])[:,1],label='MMFlow Test', color='blue')

    plt.plot(np.array(mineResults.additional_information['training_history'])[:,0],np.array(mineResults.additional_information['training_history'])[:,1],label='MINE Train', color='orange', linestyle='dashed', alpha = .5)
    plt.plot(np.array(mineResults.additional_information['test_history'])[:,0],np.array(mineResults.additional_information['test_history'])[:,1],label='MINE Test', color='orange')

    # plt.plot(np.array(InfoNCEResults.additional_information['training_history'])[:,0],np.array(InfoNCEResults.additional_information['training_history'])[:,1],label='InfoNCE Train', color='green', linestyle='dashed', alpha = .5)
    # plt.plot(np.array(InfoNCEResults.additional_information['test_history'])[:,0],np.array(InfoNCEResults.additional_information['test_history'])[:,1],label='InfoNCE Test', color='green')


    # Add labels and title
    plt.xlabel('Training Step')
    plt.ylabel('MI approx')
    plt.title('Training MI')
    plt.legend()
    plt.ylim(0, None)

    # Show the plot
    plt.show()
        
    # print(f"Ground truth mutual information: {task.mutual_information}")#:.2f

    # print(f"Gaussian Assumption mutual information: {value/n_runs}")
        
    # print(f"Estimate by CCA: {cca_mean/n_runs}")#:.2f

    # print(f"Estimate by KSG: {ksg_mean/n_runs:.2f}")

    # print(f"Estimate by MMFlow: {MMFlow_mean/n_runs}")#:.2f

    # print(f"Estimate by MINE: {mine_mean/n_runs:.2f}")

    # print(f"Estimate by InfoNCE: {InfoNCE_mean/n_runs:.2f}")

True_mean = np.mean(True_value,axis=1)
Gauss_mean = np.mean(Gauss_value,axis=1)
cca_mean = np.mean(cca_value,axis=1)
ksg_mean = np.mean(ksg_value,axis=1)
MMFlow_mean = np.mean(MMFlow_value,axis=1)
mine_mean = np.mean(mine_value,axis=1)
InfoNCE_mean = np.mean(InfoNCE_value,axis=1)

Means = np.vstack((True_mean,Gauss_mean,cca_mean,ksg_mean,MMFlow_mean,mine_mean,InfoNCE_mean))
Means_round = Means.round(decimals=3)

fig, ax = plt.subplots()

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

# Create the data
data = np.random.rand(10, 5)

# Create the table
ax.table(cellText=Means_round, colLabels=task_names, rowLabels=method_names, loc='center')

fig.tight_layout()

plt.show()

###################################################################################
# plt.plot(np.array(MMFLowResults.additional_information['test_history'])[:,0],task.mutual_information*np.ones(len(np.array(MMFLowResults.additional_information['test_history'])[:,1])),label='True', color='black')

# plt.plot(np.array(MMFLowResults.additional_information['training_history'])[:,0],np.array(MMFLowResults.additional_information['training_history'])[:,1],label='MMFlow Train', color='blue', linestyle='dashed', alpha = .5)
# plt.plot(np.array(MMFLowResults.additional_information['test_history'])[:,0],np.array(MMFLowResults.additional_information['test_history'])[:,1],label='MMFlow Test', color='blue')

# plt.plot(np.array(mineResults.additional_information['training_history'])[:,0],np.array(mineResults.additional_information['training_history'])[:,1],label='MINE Train', color='orange', linestyle='dashed', alpha = .5)
# plt.plot(np.array(mineResults.additional_information['test_history'])[:,0],np.array(mineResults.additional_information['test_history'])[:,1],label='MINE Test', color='orange')

# # plt.plot(np.array(InfoNCEResults.additional_information['training_history'])[:,0],np.array(InfoNCEResults.additional_information['training_history'])[:,1],label='InfoNCE Train', color='green', linestyle='dashed', alpha = .5)
# # plt.plot(np.array(InfoNCEResults.additional_information['test_history'])[:,0],np.array(InfoNCEResults.additional_information['test_history'])[:,1],label='InfoNCE Test', color='green')

# # plt.plot(np.array(DVResults.additional_information['training_history'])[:,0],np.array(DVResults.additional_information['training_history'])[:,1],label='DV Train', color='red', linestyle='dashed', alpha = .5)
# # plt.plot(np.array(DVResults.additional_information['test_history'])[:,0],np.array(DVResults.additional_information['test_history'])[:,1],label='DV Test', color='red')


# # Add labels and title
# plt.xlabel('Training Step')
# plt.ylabel('MI approx')
# plt.title('McAllester High-MI Gaussian')
# plt.legend()
# plt.ylim(0, None)

# # Show the plot
# plt.show()