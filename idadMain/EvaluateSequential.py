import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import argparse
import numpy as np
import pandas as pd
import pickle
import os
import scipy
from matplotlib.pyplot import colorbar, pcolor, show, scatter
import torch
from eval_sPCE_from_source import eval_from_source
ESTIMATOR_COLORS = {
    "TrueMI": "rgba(0,0,0,1.0)", # Black
    "FlowP": "rgba(204,0,0,1.0)", # Red
    "FlowMP": "rgba(0,0,153,1.0)", # Blue
    "MINE": "rgba(0,204,204,1.0)", # Light Blue
    "InfoNCE": "rgba(0,153,0,1.0)", # Green
    "NWJ": "rgba(255,0,127,1.0)", # Pink
    "DV": "rgba(204,102,0,1.0)", # Orange
    "KSG": "rgba(102,51,0,1.0)", # Brown
    "CCA": "rgba(96,96,96,1.0)", # Grey
}

def plotMcAlester(path_to_artifact):
    with open(path_to_artifact, "rb") as input_file:
        data_dict = pickle.load(input_file)
    num_steps = data_dict["Experiment"][9]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = np.linspace(0,num_steps-1,num_steps),
                                y = data_dict["TrueMI"]*np.ones(num_steps),
                                name = 'True', 
                                mode='lines',
                                line = dict(color=ESTIMATOR_COLORS['TrueMI'],
                                            width = 3)))
    for index, key in enumerate(data_dict):
        if key not in ["Experiment","TrueMI"]:
            [Results, best_arg] = data_dict[key] 
            fig.add_trace(go.Scatter(x = np.array(Results[3].additional_information['training_history'])[:,0]-1,
                                    y = np.array(Results[3].additional_information['training_history'])[:,1],
                                    name = key+' Train', 
                                    line = dict(color=ESTIMATOR_COLORS[key][:-4]+"0.5)", # add opacity 
                                                dash = 'dash',
                                                width = 3)))
    for index, key in enumerate(data_dict):
        if key not in ["Experiment","TrueMI"]:
            [Results, best_arg] = data_dict[key] 
            fig.add_trace(go.Scatter(x = np.array(Results[3].additional_information['test_history'])[:,0]-1,
                                    y = np.array(Results[3].additional_information['test_history'])[:,1],
                                    name = key+' Test', 
                                    mode='lines',
                                    line = dict(color=ESTIMATOR_COLORS[key],
                                                width = 3)))
            # Add labels and title
    fig.update_layout(xaxis_title='Gradient Steps',
                        yaxis_title='MI Value',
                        plot_bgcolor='white',
                        # legend=dict(#orientation="h",
                        #             yanchor="top",
                        #             y=0.35,
                        #             xanchor="left",
                        #             x=0.01,),
                        font=dict(size=12),)
    fig.update_xaxes(mirror=True,
                        ticks='outside',
                        showline=True,
                        linecolor='black',
                        gridcolor='lightgrey',
                        automargin=True)
    fig.update_yaxes(mirror=True,
                        ticks='outside',
                        showline=True,
                        linecolor='black',
                        gridcolor='lightgrey',
                        automargin=True)
    # fig.tight_layout()
    fig.show()
    
    pio.write_image(fig,path_to_artifact+'.pdf')
    print("Done.")

def plotBMI(path_to_artifact):
    data_mean = {}
    data_std = {}
    data_mean_err = {}
    data_mean_rel_err = {}
    data_mean_end = {}
    meta_file = path_to_artifact + '/meta'
    with open(meta_file , "rb") as input_file:
        meta_dict = pickle.load(input_file)
    for i in range(len(meta_dict['task_list'])):# × 
        task_name = bmi.benchmark.BENCHMARK_TASKS[meta_dict['task_list'][i]].name
        task_file = path_to_artifact + '/' +task_name
        if os.path.exists(task_file):
            with open(task_file , "rb") as input_file:
                task_dict = pickle.load(input_file)
            task_mean = {}
            task_std = {}
            task_mean_err = {}
            task_mean_rel_err = {}
            task_mean_end = {}
            task_mean['True MI'] = task_dict[meta_dict['task_list'][i]]
            task_std['True MI'] = 0
            task_mean_err['True MI'] = 0
            task_mean_rel_err['True MI'] = 0
            task_mean_end['True MI'] = task_dict[meta_dict['task_list'][i]]
            for j, method_name in enumerate(meta_dict['method_names']):
                if method_name in ['CCA','KSG']:
                    task_mean[method_name] = np.mean(task_dict[method_name])
                    task_std[method_name] = np.std(task_dict[method_name])
                    task_mean_err[method_name] = np.mean(task_dict[method_name]) - task_dict[meta_dict['task_list'][i]]
                    task_mean_rel_err[method_name] = (np.mean(task_dict[method_name]) - task_dict[meta_dict['task_list'][i]])/task_dict[meta_dict['task_list'][i]]
                    task_mean_end[method_name] = np.mean(task_dict[method_name])
                elif method_name in ['MPGauss']:
                    max_mi_est = []
                    for k in range(len(task_dict[method_name])):
                        max_mi_est.append(task_dict[method_name][k].mi_estimate)
                    task_mean[method_name] = np.mean(max_mi_est)
                    task_std[method_name] = np.std(max_mi_est)
                    task_mean_err[method_name] = np.mean(max_mi_est) - task_dict[meta_dict['task_list'][i]]
                    task_mean_rel_err[method_name] = (np.mean(max_mi_est) - task_dict[meta_dict['task_list'][i]])/task_dict[meta_dict['task_list'][i]]
                    task_mean_end[method_name] = np.mean(max_mi_est)
                elif method_name in ['FlowMP']:
                    max_mi_est = []
                    end_mi_est = []
                    for k in range(len(task_dict[method_name])):
                        max_mi_est.append(task_dict[method_name][k].mi_estimate)
                        end_mi_est.append(task_dict[method_name][0].additional_information['test_history'][-1][1])
                    task_mean[method_name] = np.mean(end_mi_est)
                    task_std[method_name] = np.std(max_mi_est)
                    task_mean_err[method_name] = (np.mean(end_mi_est) - task_dict[meta_dict['task_list'][i]])
                    task_mean_rel_err[method_name] = (np.mean(end_mi_est) - task_dict[meta_dict['task_list'][i]])/task_dict[meta_dict['task_list'][i]]
                    task_mean_end[method_name] = np.mean(max_mi_est)      
                else:
                    max_mi_est = []
                    end_mi_est = []
                    for k in range(len(task_dict[method_name])):
                        max_mi_est.append(task_dict[method_name][k].mi_estimate)
                        end_mi_est.append(task_dict[method_name][0].additional_information['test_history'][-1][1])
                    task_mean[method_name] = np.mean(max_mi_est)
                    task_std[method_name] = np.std(max_mi_est)
                    task_mean_err[method_name] = (np.mean(max_mi_est) - task_dict[meta_dict['task_list'][i]])
                    task_mean_rel_err[method_name] = (np.mean(max_mi_est) - task_dict[meta_dict['task_list'][i]])/task_dict[meta_dict['task_list'][i]]
                    task_mean_end[method_name] = np.mean(end_mi_est)      
            data_mean[task_name] = task_mean
            data_std[task_name] = task_std
            data_mean_err[task_name] = task_mean_err
            data_mean_rel_err[task_name] = task_mean_rel_err
            data_mean_end[task_name] = task_mean_end

    df_mean = pd.DataFrame(data_mean)
    df_mean_err = pd.DataFrame(data_mean_rel_err)        
    vcenter = 0
    # vmin, vmax = y.min(), y.max() 
    normalize = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=df_mean_err.min().min(), vmax=df_mean_err.max().max())
    colormap = cm.seismic#cm.bwr
    
    ax = sns.heatmap(data=df_mean_err, annot=df_mean,norm=normalize, cmap=colormap, cbar=False, fmt=".2f")#, annot=True
    ax.set(xlabel="", ylabel="")
    loc, labels = plt.xticks()
    ax.set_xticklabels(labels, rotation=60,ha='right')
    ax.figure.tight_layout()
    # plt.xticks(rotation=30)
    plt.show()
    # first = path_to_artifact.index('/')+1
    # second = path_to_artifact.index('/',first,len(path_to_artifact))
    # plot_name = path_to_artifact[second+1:]
    plt.savefig(path_to_artifact+'/plot.pdf')
    print("Done.")

def plotLocationFinding(path_to_artifact):
    path_to_artifact = 'C:/Users/Caleb/OneDrive/Documents/GitHub/FlowMI/' +path_to_artifact
    path_to_meta = path_to_artifact + '/extra_meta.pickle'
    # with open(path_to_meta, "rb") as input_file:
    #     meta_info = pickle.load(input_file)
    # num_runs = 0
    # # Iterate directory
    # for path in os.listdir(path_to_artifact):
    #     # check if current path is a file
    #     if os.path.isfile(os.path.join(path_to_artifact, path)):
    #         num_runs += 1
    num_runs = len(next(os.walk(path_to_artifact))[1])
    designs = np.zeros((2,num_runs))
    times = np.zeros(num_runs)
    for i in range(num_runs):
        path_to_run = path_to_artifact + '/Run{}'.format(i)
        with open(path_to_artifact+'/extra_meta.pickle', "rb") as input_file:
            extra_meta = pickle.load(input_file)
        path_to_ml = "mlruns/{}/{}/artifacts/results_locfin_mm_vi.pickle".format(
        extra_meta['ml_experiment_id'], extra_meta['ml_run_id'])
        with open(path_to_ml, "rb") as input_file:
            ml_info = pickle.load(input_file)
        true_theta = ml_info['loop'][0]['theta']
        # x = np.linspace(-3.5,3.5,100)
        # y = np.linspace(-3.5,3.5,100)
        # X, Y = np.meshgrid(x, y)
        for j in range(10):
            path_to_step = path_to_run+ '/Step{}.pickle'.format(j)
            with open(path_to_step, "rb") as input_file:
                step_info = pickle.load(input_file)
            times[i]+=step_info['total_time']
    mean_time = times.mean()
    std_time = times.std()
    print(mean_time)
    print(std_time)
    eval_from_source(
    path_to_artifact=path_to_ml,
    num_experiments_to_perform=[10],
    num_inner_samples=int(5e5),
    seed=-1,
    device='cpu',
    )
            # fig, axs = plt.subplots(2, 2)
            # post_mean = step_info['posterior_loc']
            # post_cov = step_info['posterior_cov']
            # prior_mean = step_info['mu'][:len(post_mean)]
            # prior_cov = step_info['sigmas'][:len(post_mean),:len(post_mean)]
            # flow_theta = step_info['flow_theta']
            # ######### Prior on source 1 ###########################################################
            # fX, logJac = step_info['flow_theta'](torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),true_theta[1,0].numpy()*np.ones(np.shape(X.flatten())),true_theta[1,1].numpy()*np.ones(np.shape(X.flatten())))).T)).float())
            # points = fX.reshape((100,100,4))
            # Z = scipy.stats.multivariate_normal.pdf(points.detach().numpy(), prior_mean, prior_cov)*np.exp(logJac.reshape((100,100)).detach().numpy())
            # axs[0, 0].pcolor(X, Y, Z)
            # axs[0, 0].scatter(true_theta[0,0].numpy(),true_theta[0,1].numpy(), color='red', marker='x',label = 'True')
            # # axs[k, l].scatter(true_theta[i][0],true_theta[i][1], color='red', marker='x')
            # axs[0, 0].scatter(step_info['design'][0][0].numpy()[0],step_info['design'][0][0].numpy()[1], color='green', marker='x',label = 'Design')
            # ######### Prior on source 2 ###########################################################
            # fX, logJac = step_info['flow_theta'](torch.from_numpy((np.vstack((true_theta[0,0].numpy()*np.ones(np.shape(X.flatten())),true_theta[0,1].numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float())
            # points = fX.reshape((100,100,4))
            # Z = scipy.stats.multivariate_normal.pdf(points.detach().numpy(), prior_mean, prior_cov)*np.exp(logJac.reshape((100,100)).detach().numpy())
            # axs[0, 1].pcolor(X, Y, Z)
            # axs[0, 1].scatter(true_theta[1,0].numpy(),true_theta[1,1].numpy(), color='red', marker='x',label = 'True')
            # # axs[k, l].scatter(true_theta[i][0],true_theta[i][1], color='red', marker='x')
            # axs[0, 1].scatter(step_info['design'][0][0].numpy()[0],step_info['design'][0][0].numpy()[1], color='green', marker='x',label = 'Design')
            
            # ######### Posterior on source 1 ###########################################################
            # fX, logJac = step_info['flow_theta'](torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),true_theta[1,0].numpy()*np.ones(np.shape(X.flatten())),true_theta[1,1].numpy()*np.ones(np.shape(X.flatten())))).T)).float())
            # points = fX.reshape((100,100,4))
            # Z = scipy.stats.multivariate_normal.pdf(points.detach().numpy(), post_mean, post_cov)*np.exp(logJac.reshape((100,100)).detach().numpy())
            # axs[1, 0].pcolor(X, Y, Z)
            # axs[1, 0].scatter(true_theta[0,0].numpy(),true_theta[0,1].numpy(), color='red', marker='x',label = 'True')
            # # axs[k, l].scatter(true_theta[i][0],true_theta[i][1], color='red', marker='x')
            # axs[1, 0].scatter(step_info['design'][0][0].numpy()[0],step_info['design'][0][0].numpy()[1], color='green', marker='x',label = 'Design')
            # ######### Posterior on source 2 ###########################################################
            # fX, logJac = step_info['flow_theta'](torch.from_numpy((np.vstack((true_theta[0,0].numpy()*np.ones(np.shape(X.flatten())),true_theta[0,1].numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float())
            # points = fX.reshape((100,100,4))
            # Z = scipy.stats.multivariate_normal.pdf(points.detach().numpy(), post_mean, post_cov)*np.exp(logJac.reshape((100,100)).detach().numpy())
            # axs[1, 1].pcolor(X, Y, Z)
            # axs[1, 1].scatter(true_theta[1,0].numpy(),true_theta[1,1].numpy(), color='red', marker='x',label = 'True')
            # # axs[k, l].scatter(true_theta[i][0],true_theta[i][1], color='red', marker='x')
            # axs[1, 1].scatter(step_info['design'][0][0].numpy()[0],step_info['design'][0][0].numpy()[1], color='green', marker='x',label = 'Design')
            # axs[0, 0].title.set_text('Source 1')
            # axs[0, 1].title.set_text('Source 2')
            # axs[0, 0].set(ylabel='Prior')
            # axs[1, 0].set(ylabel='Posterior')
            # plt.legend()#loc='center left', bbox_to_anchor=(1, 0.5)
            # fig.suptitle('Step {}'.format(j))
            # plt.tight_layout()
            # plt.savefig(path_to_run+'/plot{}.pdf'.format(j))
            # plt.close()
        # for j in range(10):
        #     path_to_step = path_to_run+ '/Step{}.pickle'.format(j)
        #     with open(path_to_step, "rb") as input_file:
        #         step_info = pickle.load(input_file)
        #     path_to_loss = path_to_run+ '/Loss{}.pickle'.format(j)
        #     with open(path_to_loss, "rb") as input_file:
        #         loss_info = pickle.load(input_file)
        #     designs[:,j] = step_info['design'][0][0].numpy()
        #     plt.plot(np.arange(len(loss_info))*100,loss_info,label = 'Decision {}'.format(j))
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.tight_layout()
        # plt.savefig(path_to_run+'/plot.pdf')
        # plt.close()

def plotSIRresults(path_to_artifact):
    path_to_artifact = 'C:/Users/Caleb/OneDrive/Documents/GitHub/FlowMI/' +path_to_artifact
    path_to_meta = path_to_artifact + '/extra_meta.pickle'
    # with open(path_to_meta, "rb") as input_file:
    #     meta_info = pickle.load(input_file)
    # num_runs = 0
    # # Iterate directory
    # for path in os.listdir(path_to_artifact):
    #     # check if current path is a file
    #     if os.path.isfile(os.path.join(path_to_artifact, path)):
    #         num_runs += 1
    num_runs = len(next(os.walk(path_to_artifact))[1])
    designs = np.zeros((num_runs,5))
    # true_theta = [[0.3266, 0.1686],[0.6851, 0.0612],[0.4029, 0.2700], [0.3720, 0.0322], [0.4267, 0.0781],[0.3144, 0.2316],[0.5334, 0.1013],[0.6305, 0.0993],[0.8173, 0.1684],[0.3931, 0.0642],[0.5435, 0.0782]]
    # true_theta = [[0.5078, 0.0800],[0.2857, 0.0821], [0.7736, 0.0882], [0.5033, 0.0445], [0.3973, 0.0660],[0.5968, 0.0907], [0.4240, 0.0450],[0.7852, 0.0690],[0.5240, 0.0680],
    #               [0.3213, 0.0559],[0.5508, 0.1212],[0.4822, 0.1405],[0.3953, 0.1215],[0.8154, 0.1019],[1.5404, 0.0687],[0.3839, 0.1347],[0.6085, 0.0843],[0.4127, 0.0578],[0.2072, 0.0599],[1.0484, 0.1661]]
    for i in range(num_runs):
        path_to_run = path_to_artifact + '/Run{}'.format(i)
        fig, axs = plt.subplots(2, 3)
        for j in range(5):
            path_to_step = path_to_run+ '/Step{}.pickle'.format(j)
            with open(path_to_step, "rb") as input_file:
                step_info = pickle.load(input_file)
            designs[i,j] = step_info['designs_trans'][0][0].numpy()[0]
            x = np.linspace(0.01,np.max([1, step_info['theta'][0].numpy()[0]+.2]),100)
            y = np.linspace(0.01,np.max([.2, step_info['theta'][0].numpy()[1]+.05]),100)
            # x = np.linspace(0.01,np.max([1, true_theta[i][0]+.2]),100)
            # y = np.linspace(0.01,np.max([.2, true_theta[i][1]+.05]),100)
            X, Y = np.meshgrid(x, y)
            latent_dim = 2
            mux = step_info['mu'][:latent_dim]
            muy = step_info['mu'][latent_dim:]
            Sigmaxx = step_info['sigmas'][:latent_dim,:latent_dim]
            Sigmaxy = step_info['sigmas'][:latent_dim,latent_dim:]
            Sigmayy = step_info['sigmas'][latent_dim:,latent_dim:]
            obs, _ = step_info['flow_obs'](step_info['observations'])
            # obs = observation[0]
            posterior_loc = (mux + np.matmul(Sigmaxy,np.linalg.solve(Sigmayy,(obs.detach().numpy()-muy))).flatten())
            if not np.isnan(posterior_loc[0]):
                max_posterior = step_info['flow_theta'].reverse(torch.from_numpy(posterior_loc)).exp()
                posterior_cov = Sigmaxx-np.matmul(Sigmaxy,np.linalg.solve(Sigmayy,Sigmaxy.T))
                fX, logJac = step_info['flow_theta'](torch.from_numpy(np.log(np.vstack((X.flatten(),Y.flatten())).T)).float())#.numpy()torch.from_numpy
                points = fX.reshape((100,100,2))
                Z = scipy.stats.multivariate_normal.pdf(points.detach().numpy(), posterior_loc, posterior_cov)*np.exp(logJac.reshape((100,100)).detach().numpy())
                if j<3:
                    k = 0
                    l =j
                else:
                    k=1
                    l = j-3
                axs[k, l].pcolor(X, Y, Z)
                axs[k, l].scatter(step_info['theta'][0].numpy()[0],step_info['theta'][0].numpy()[1], color='red', marker='x',label = 'True')
                # axs[k, l].scatter(true_theta[i][0],true_theta[i][1], color='red', marker='x')
                axs[k, l].scatter(max_posterior[0].detach().numpy(),max_posterior[1].detach().numpy(), color='green', marker='x',label = 'Pred')
                # axs[k,l].title.set_text('d=%.2f  I=%.2f'%(step_info['designs_trans'][0][0].numpy()[0],step_info['observations'][0].numpy()[0]))
                axs[k,l].title.set_text('Obs. Infected=%.2f'%(step_info['observations'][0].numpy()[0]))
                if j == 4:
                    axs[k, l].legend(loc="lower right")
            # colorbar()
        axs[1, 2].scatter(designs[i,:],np.arange(5))
        axs[1, 2].set(xlabel='Time', ylabel='Decision')
        fig.tight_layout()
        # title = "Sim Time: {:.2f}  |  Maximization Time: {:.2f}  |  Total Time: {:.2f}".format(step_info['simulation_time'],step_info['total_time']-step_info['simulation_time'],step_info['total_time'])
        # plt.suptitle(title)
        plt.savefig(path_to_run+'/plot.pdf')
            
def main(path_to_artifact):
    fig = go.Figure(go.Scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16]))
    # fig.show()
    # fig.write_image("random.pdf")
    first = path_to_artifact.index('/')+1
    second = path_to_artifact.index('/',first,len(path_to_artifact))
    experiment_name = path_to_artifact[first:second]
    if experiment_name == 'BMI':
        plotBMI(path_to_artifact)
    # if data_dict["Experiment"][0] == "McAllester_experiment":
    if experiment_name == 'LargeMI':
        plotMcAlester(path_to_artifact)
    if experiment_name == 'loc_fin':
        plotLocationFinding(path_to_artifact)
    if experiment_name == 'SIR':
        plotSIRresults(path_to_artifact)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--path-to-artifact", default="experiment_outputs/loc_fin/20240520132743", type=str)
    ################## location finding ###################################################
    ################## fullrun ########################
    ##### JVG ##############
    ## True Prior
    ## "experiment_outputs/loc_fin/20240520195707" (Time: [ +- ]  EIG: [ +- ]) (3214627) "mlruns/145551772296899500/c7060deeb88e48148ce95cd2c1373660/artifacts/results_locfin_mm_vi.pickle"
    # "experiment_outputs/loc_fin/20240521022230" (Time: [577.687 +- 25.528]  EIG: [3.895481 +- 0.180756]) (3215127) (slower decay rate) "mlruns/145551772296899500/62ccfcabe68248219a4d0d7f4c17826e/artifacts/results_locfin_mm_vi.pickle"
    
    ## False Prior
    ## "experiment_outputs/loc_fin/20240520153133" (Time: [ +- ]  EIG: [ +- ]) (3213161) "mlruns/145551772296899500/fb7a8a03e2bb4fc58fdd48da0157b0a9/artifacts/results_locfin_mm_vi.pickle"
    # "experiment_outputs/loc_fin/20240521054906" (Time: [594.194 +- 24.918]  EIG: [3.478048 +- 0.170425]) (3215132) (slower decay rate) "mlruns/145551772296899500/a3032bbd9d844e68b828280cf93a62fb/artifacts/results_locfin_mm_vi.pickle"
    
    ##### NVG ##############
    ## True Prior
    ## "experiment_outputs/loc_fin/20240520190123" (Time: [ +- ]  EIG: [ +- ]) (3214325) "mlruns/145551772296899500/b72bee0b33d4468986cfaad239f35f47/artifacts/results_locfin_vi.pickle"
    # "experiment_outputs/loc_fin/20240521022335" (Time: [467.358 +- 12.817]  EIG: [4.074939 +- 0.165542]) (3215129) (slower decay rate) "mlruns/145551772296899500/b8a7ba3646c54fb69c16355d35a742f2/artifacts/results_locfin_vi.pickle"
    # "experiment_outputs/loc_fin/20240521165632" (Time: [624.707 +- 18.271]  EIG: [4.550186 +- 0.175854]) (3216149) (original network) "mlruns/145551772296899500/e7e89330c9fe431f8b6f7717eb3deb37/artifacts/results_locfin_vi.pickle"
    
    ## False Prior
    ## "experiment_outputs/loc_fin/20240520153048" (Time: [ +- ]  EIG: [ +- ]) (3213158) "mlruns/145551772296899500/a73607108c3c4e5f88dd30109c94c6ad/artifacts/results_locfin_vi.pickle"
    ## "experiment_outputs/loc_fin/20240521063344" (Time: [ +- ]  EIG: [ +- ]) (3215133) (slower decay rate) "mlruns/145551772296899500/4aa5dd55deb348ad935ec091436b36bd/artifacts/results_locfin_vi.pickle"
    # "experiment_outputs/loc_fin/20240521185827" (Time: [557.548 +- 17.562]  EIG: [3.992774 +- 0.172565]) (3216416) (original network) "mlruns/145551772296899500/1568bbd5a06f45b89327bad3dbc403b9/artifacts/results_locfin_vi.pickle"
    
    ##### JVF ##############
    ## True Prior
    # "experiment_outputs/loc_fin/20240520153548" (Time: [3790.747 +- 116.514]  EIG: [4.146697 +- 0.186421]) (3213215) "mlruns/145551772296899500/b35a83c9d2084278b521ad36c05bf9f9/artifacts/results_locfin_mm_vi.pickle"
    # "experiment_outputs/loc_fin/20240521022213" (Time: [3469.818 +- 118.225]  EIG: [4.280338 +- 0.17536]) (3215126) (slower decay rate) "mlruns/145551772296899500/2b589d45ca1c4a0991dd972ffcbfb3fa/artifacts/results_locfin_mm_vi.pickle"
    
    ## False Prior
    # "experiment_outputs/loc_fin/20240520152944" (Time: [3728.630 +- 111.319]  EIG: [3.935164 +- 0.196292]) (3213156) "mlruns/145551772296899500/929ae41b9f3a4a6eacc6f2f29ca90712/artifacts/results_locfin_mm_vi.pickle"
    # "experiment_outputs/loc_fin/20240521023331" (Time: [3685.134 +- 107.365]  EIG: [3.777488 +- 0.179521]) (3215130) (slower decay rate) "mlruns/145551772296899500/b98364d6df8640dcb5d694a7f7b2be6f/artifacts/results_locfin_mm_vi.pickle"
    
    ##### NVF ##############
    ## True Prior
    # "experiment_outputs/loc_fin/20240520181046" (Time: [2792.940 +- 84.300]  EIG: [4.755394 +- 0.193923]) (3214098) "mlruns/145551772296899500/9158cc9acf364503ae7dc15ad6dbf2dc/artifacts/results_locfin_vi.pickle"
    # "experiment_outputs/loc_fin/20240521022307" (Time: [3073.663 +- 81.928]  EIG: [5.106241 +- 0.197699]) (3215128) (slower decay rate) "mlruns/145551772296899500/702d8911c27249d796a0ec25083e7a40/artifacts/results_locfin_vi.pickle"
    #R "experiment_outputs/loc_fin/20240521104114" (Time: [2956.524 +- 94.549]  EIG: [5.111974 +-  0.184681]) (3215326) (original network) "mlruns/145551772296899500/bc6fe7d110524805803bd866b027db56/artifacts/results_locfin_vi.pickle"
    
    ## False Prior
    # "experiment_outputs/loc_fin/20240520153010" (Time: [2881.031 +- 83.221]  EIG: [4.380203 +- 0.185274]) (3213157) "mlruns/145551772296899500/a741868894be4583b72b230ee39746e9/artifacts/results_locfin_vi.pickle"
    # "experiment_outputs/loc_fin/20240521023356" (Time: [2998.682 +- 85.785]  EIG: [4.214872 +- 0.178373]) (3215131) (slower decay rate) "mlruns/145551772296899500/2719721cd40741eda21f9f401bbac01d/artifacts/results_locfin_vi.pickle"
    ### "" (3215327) (original network) 
    
    ################## Early Stopping ########################
    ##### JVG ##############
    ## True Prior
    ## "experiment_outputs/loc_fin/20240519162648" (Time: [ +- ]  EIG: [ +- ]) (3210177) "mlruns/145551772296899500/6f788395a94c4937853d89839e4f1712/artifacts/results_locfin_mm_vi.pickle"
    # "experiment_outputs/loc_fin/20240519233857" (Time: [ \pm ]  EIG: [3.775533 +- 0.195]) (3210255) "mlruns/145551772296899500/88ce96eb434d42f0ac5e0fb7018ae34f/artifacts/results_locfin_mm_vi.pickle"
    #SSSS "experiment_outputs/loc_fin/20240520112207" (Time: [286.629 \pm 35.306]  EIG: [3.556697 \pm 0.19287]) (3210388) "mlruns/145551772296899500/4daef82038544a95b121e3baeaf5ffbb/artifacts/results_locfin_mm_vi.pickle"
    
    ## False Prior
    ##SSSS "experiment_outputs/loc_fin/20240520024635" (Time: [ \pm ]  EIG: [3.585144 \pm 0.149617]) (3210266) "mlruns/145551772296899500/e3eb072226c045f0bdd2301d3f6230ae/artifacts/results_locfin_mm_vi.pickle"
    ## "experiment_outputs/loc_fin/20240520123941" (Time: [ \pm ]  EIG: [3.773447 \pm 0.179452])(3210416) "mlruns/145551772296899500/df98d2fa64ea47dab7b59bf187665868/artifacts/results_locfin_mm_vi.pickle"
    
    ##### NVG ##############
    ## True Prior
    # "experiment_outputs/loc_fin/20240519193056" (Time: [182.873 \pm 20.669]  EIG: [3.830811 \pm 0.177804]) (3210216) "mlruns/145551772296899500/cd8e0f1e41fc4999b1315d68bec433f1/artifacts/results_locfin_vi.pickle"
    # SSS"experiment_outputs/loc_fin/20240519210154" (Time: [185.906 \pm 21.418]  EIG: [4.607234 \pm 0.192131]) (3210229) "mlruns/145551772296899500/ea1fde1fce0d484abdfa31d4b6511211/artifacts/results_locfin_vi.pickle"
    # "experiment_outputs/loc_fin/20240520111814" (Time: [182.566 \pm 20.625]  EIG: [4.381109 \pm 0.181779]) (3210383) "mlruns/145551772296899500/8450216ceac7475aa83c2df60146f04c/artifacts/results_locfin_vi.pickle"
    
    ## False Prior
    ## SSSS"experiment_outputs/loc_fin/20240520021351" (Time: [ \pm ]  EIG: [ 3.994429\pm 0.202681]) (3210265) "mlruns/145551772296899500/1e52163efe0443c4842db136fc993239/artifacts/results_locfin_vi.pickle"
    ## "experiment_outputs/loc_fin/20240520123113" (3210414) "mlruns/145551772296899500/4807db276e0a47dc872b7d76e587fb01/artifacts/results_locfin_vi.pickle"
    
    ##### JVF ##############
    ## True Prior
    # "experiment_outputs/loc_fin/20240519165801" (Time: [2362.777 \pm 271.661]  EIG: [4.315582 \pm 0.174397]) (3210184) "mlruns/145551772296899500/78369b957a354953a4118c59cd145186/artifacts/results_locfin_mm_vi.pickle"
    # "experiment_outputs/loc_fin/20240519191615" (Time: [2337.478 \pm 237.500]  EIG: [4.146299 \pm 0.169304]) (3210212) "mlruns/145551772296899500/e61150bf0d93405bb9c9932d62fb15eb/artifacts/results_locfin_mm_vi.pickle"
    # "experiment_outputs/loc_fin/20240519231127" (Time: [2001.140 \pm 225.670]  EIG: [4.09795 \pm 0.160508]) (3210252) "mlruns/145551772296899500/e285729a049c48969dc36dffe8bc6e7d/artifacts/results_locfin_mm_vi.pickle"
    # SSSS"experiment_outputs/loc_fin/20240520112250" (Time: [1730 \pm 207]  EIG: [4.30 \pm 0.19]) (3210390) "mlruns/145551772296899500/ea1d69be5dd846ae8ce8693ee24aec2f/artifacts/results_locfin_mm_vi.pickle"
    
    ## False Prior
    # "experiment_outputs/loc_fin/20240520002925" (Time: [2004.165 \pm 211.417]  EIG: [3.84275 \pm 0.184771]) (3210259) "mlruns/145551772296899500/208b943d523f4b658aab57a457fcfece/artifacts/results_locfin_mm_vi.pickle"
    # "experiment_outputs/loc_fin/20240520034258" (Time: [1962.742 \pm 221.455]  EIG: [ \pm ]) (3210267) "mlruns/145551772296899500/e033ba9b08f94485a1df4c0f36d93317/artifacts/results_locfin_mm_vi.pickle"
    # SSSS"experiment_outputs/loc_fin/20240520132743" (Time: [1730 \pm 207]  EIG: [ 3.849171\pm  0.185747]) (3210417) "mlruns/145551772296899500/95b9089ed8864eeebea76da55d2b1c27/artifacts/results_locfin_mm_vi.pickle"
    
    ##### NVF ##############
    ## True Prior
    # "experiment_outputs/loc_fin/20240519193116" (Time: [1302.222 \pm 159.831]  EIG: [4.500601 \pm 0.162272]) (3210217) "mlruns/145551772296899500/b7575100d8424aa584c76f0a263e3742/artifacts/results_locfin_vi.pickle"
    # "experiment_outputs/loc_fin/20240519210300" (Time: [1267.083 \pm 147.285]  EIG: [4.654656 \pm 0.176478]) (3210230) "mlruns/145551772296899500/8dc93ba9a2374927a43df57639f5e74a/artifacts/results_locfin_vi.pickle"
    #SSSS "experiment_outputs/loc_fin/20240519225257" (Time: [1253.626 \pm 155.220]  EIG: [5.104215 \pm 0.197219]) (3210249) "mlruns/145551772296899500/b9ade313113b49b98292a47e8be2f194/artifacts/results_locfin_vi.pickle"
    # "experiment_outputs/loc_fin/20240520111935" (Time: [1218.018 \pm 152.688]  EIG: [5.331629 \pm 0.183142]) (3210384) "mlruns/145551772296899500/84c28b189a734bcc9e82fd7e6d2480ae/artifacts/results_locfin_vi.pickle"
    
    ## False Prior
    #SSSS "experiment_outputs/loc_fin/20240520052531" (Time: [1261.625 \pm 158.008]  EIG: [4.748 \pm 0.18342]) (3210260) "mlruns/145551772296899500/be9c8719ecee4fb385e368b9157b35fd/artifacts/results_locfin_vi.pickle"
    # "experiment_outputs/loc_fin/20240520123155" (3210415) "mlruns/145551772296899500/4e21122212f241fe97a8c6652a67e162/artifacts/results_locfin_vi.pickle"
    
    args = parser.parse_args()

    main(path_to_artifact=args.path_to_artifact)