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
    for i in range(num_runs):
        path_to_run = path_to_artifact + '/Run{}'.format(i)
        with open(path_to_artifact+'/extra_meta.pickle', "rb") as input_file:
            extra_meta = pickle.load(input_file)
        path_to_ml = "mlruns/{}/{}/artifacts/results_locfin_mm_vi.pickle".format(
        extra_meta['ml_experiment_id'], extra_meta['ml_run_id'])
        with open(path_to_ml, "rb") as input_file:
            ml_info = pickle.load(input_file)
        true_theta = ml_info['loop'][0]['theta']
        x = np.linspace(-3.5,3.5,100)
        y = np.linspace(-3.5,3.5,100)
        X, Y = np.meshgrid(x, y)
        for j in range(10):
            fig, axs = plt.subplots(2, 2)
            path_to_step = path_to_run+ '/Step{}.pickle'.format(j)
            with open(path_to_step, "rb") as input_file:
                step_info = pickle.load(input_file)

            post_mean = step_info['posterior_loc']
            post_cov = step_info['posterior_cov']
            prior_mean = step_info['mu'][:len(post_mean)]
            prior_cov = step_info['sigmas'][:len(post_mean),:len(post_mean)]
            flow_theta = step_info['flow_theta']
            ######### Prior on source 1 ###########################################################
            fX, logJac = step_info['flow_theta'](torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),true_theta[1,0].numpy()*np.ones(np.shape(X.flatten())),true_theta[1,1].numpy()*np.ones(np.shape(X.flatten())))).T)).float())
            points = fX.reshape((100,100,4))
            Z = scipy.stats.multivariate_normal.pdf(points.detach().numpy(), prior_mean, prior_cov)*np.exp(logJac.reshape((100,100)).detach().numpy())
            axs[0, 0].pcolor(X, Y, Z)
            axs[0, 0].scatter(true_theta[0,0].numpy(),true_theta[0,1].numpy(), color='red', marker='x',label = 'True')
            # axs[k, l].scatter(true_theta[i][0],true_theta[i][1], color='red', marker='x')
            axs[0, 0].scatter(step_info['design'][0][0].numpy()[0],step_info['design'][0][0].numpy()[1], color='green', marker='x',label = 'Design')
            ######### Prior on source 2 ###########################################################
            fX, logJac = step_info['flow_theta'](torch.from_numpy((np.vstack((true_theta[0,0].numpy()*np.ones(np.shape(X.flatten())),true_theta[0,1].numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float())
            points = fX.reshape((100,100,4))
            Z = scipy.stats.multivariate_normal.pdf(points.detach().numpy(), prior_mean, prior_cov)*np.exp(logJac.reshape((100,100)).detach().numpy())
            axs[0, 1].pcolor(X, Y, Z)
            axs[0, 1].scatter(true_theta[1,0].numpy(),true_theta[1,1].numpy(), color='red', marker='x',label = 'True')
            # axs[k, l].scatter(true_theta[i][0],true_theta[i][1], color='red', marker='x')
            axs[0, 1].scatter(step_info['design'][0][0].numpy()[0],step_info['design'][0][0].numpy()[1], color='green', marker='x',label = 'Design')
            
            ######### Posterior on source 1 ###########################################################
            fX, logJac = step_info['flow_theta'](torch.from_numpy((np.vstack((X.flatten(),Y.flatten(),true_theta[1,0].numpy()*np.ones(np.shape(X.flatten())),true_theta[1,1].numpy()*np.ones(np.shape(X.flatten())))).T)).float())
            points = fX.reshape((100,100,4))
            Z = scipy.stats.multivariate_normal.pdf(points.detach().numpy(), post_mean, post_cov)*np.exp(logJac.reshape((100,100)).detach().numpy())
            axs[1, 0].pcolor(X, Y, Z)
            axs[1, 0].scatter(true_theta[0,0].numpy(),true_theta[0,1].numpy(), color='red', marker='x',label = 'True')
            # axs[k, l].scatter(true_theta[i][0],true_theta[i][1], color='red', marker='x')
            axs[1, 0].scatter(step_info['design'][0][0].numpy()[0],step_info['design'][0][0].numpy()[1], color='green', marker='x',label = 'Design')
            ######### Posterior on source 2 ###########################################################
            fX, logJac = step_info['flow_theta'](torch.from_numpy((np.vstack((true_theta[0,0].numpy()*np.ones(np.shape(X.flatten())),true_theta[0,1].numpy()*np.ones(np.shape(X.flatten())),X.flatten(),Y.flatten())).T)).float())
            points = fX.reshape((100,100,4))
            Z = scipy.stats.multivariate_normal.pdf(points.detach().numpy(), post_mean, post_cov)*np.exp(logJac.reshape((100,100)).detach().numpy())
            axs[1, 1].pcolor(X, Y, Z)
            axs[1, 1].scatter(true_theta[1,0].numpy(),true_theta[1,1].numpy(), color='red', marker='x',label = 'True')
            # axs[k, l].scatter(true_theta[i][0],true_theta[i][1], color='red', marker='x')
            axs[1, 1].scatter(step_info['design'][0][0].numpy()[0],step_info['design'][0][0].numpy()[1], color='green', marker='x',label = 'Design')
            axs[0, 0].title.set_text('Source 1')
            axs[0, 1].title.set_text('Source 2')
            axs[0, 0].set(ylabel='Prior')
            axs[1, 0].set(ylabel='Posterior')
            plt.legend()#loc='center left', bbox_to_anchor=(1, 0.5)
            fig.suptitle('Step {}'.format(j))
            plt.tight_layout()
            plt.savefig(path_to_run+'/plot{}.pdf'.format(j))
            plt.close()
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
    parser.add_argument("--path-to-artifact", default="experiment_outputs/SIR/20240418100716", type=str)
    ### Loc FIn
    # mlruns/145551772296899500/d3fceeecf790419793e42374817f23f6/artifacts/results_locfin_mm_vi.pickle
    # experiment_outputs/loc_fin/20240229231620
    
    # Gauss SIR
    # (random init)
    # experiment_outputs/SIR/20240418091526
    # experiment_outputs/SIR/20240418092544
    # (fixed init)
    # experiment_outputs/SIR/20240418095723
    
    # Flow SIR
    # (random init) experiment_outputs/SIR/20240418100915
    # experiment_outputs/SIR/20240418092033
    # (fixed init) experiment_outputs/SIR/20240418100716
    #
    
    # First Flow: (9142547)  "experiment_outputs/SIR/20240219152500"
    # Decision:   (9145135)  "experiment_outputs/SIR/20240219210049"
    # lr = 0.01: (9145413)   "experiment_outputs/SIR/20240219222404"
    
    # All Flow: (9142101)    "experiment_outputs/SIR/20240219150912"
    # Decision: (9143766)    "experiment_outputs/SIR/20240219193513"
    # lr = 0.01: (9145525)   "experiment_outputs/SIR/20240219230339"
    
    # Fixed
    # First Flow: (9160456)  "experiment_outputs/SIR/20240221114200"
    #             (9182528)  "experiment_outputs/SIR/20240223135736"
    # (lr=.0005)  (9183084)  "experiment_outputs/SIR/20240223161220"
    
    # All Flow:   (9160369)  "experiment_outputs/SIR/20240221104357"
    #             (9182526)  "experiment_outputs/SIR/20240223135622"
    # (lr=.0005)  (9183150)  "experiment_outputs/SIR/20240223163918"
    
    # Gauss Sir:  (9182530)  "experiment_outputs/SIR/20240223135828"
    # (lr=.0005)  (9183082)  "experiment_outputs/SIR/20240223161048"
    
    # "experiment_outputs/loc_fin/20240211104331"
    
    ##
    # 2 flow (9192562) "experiment_outputs/SIR/20240226003537"
    # norm (9192532) "experiment_outputs/SIR/20240226001242"
    
    # (9199605) "experiment_outputs/SIR/20240226101311"
    
    #####
    # Allflow "experiment_outputs/SIR/20240225203634" (9190377) Any Init lr=.0005 lr_f=.0001 
    # Allflow "experiment_outputs/SIR/20240225203541" (9190376) Any Init lr=.001 lr_f=.0001 
    # Allflow "experiment_outputs/SIR/20240225203940" (9190389) Start Init lr=.0005 lr_f=.0001
    # Allflow "experiment_outputs/SIR/20240225203740" (9190378) Start Init lr=.001 lr_f=.0001 
    
    # (9209688) "experiment_outputs/SIR/20240226190011"
    # (9209676) "experiment_outputs/SIR/20240226185655"
    
    
    
    ######
    # (9212651) (lr = 1e-5, lr_f = 5e-3)  "experiment_outputs/SIR/20240227100145" NaN in samples
    # (9212682) (lr = 1e-6, lr_f = 5e-3)  "experiment_outputs/SIR/20240227100644"
    # (9212781) (lr = 1e-5, Gauss)        "experiment_outputs/SIR/20240227102152"
    # (9215546) (lr = 1e-5, lr_f = 5e-4)  "experiment_outputs/SIR/20240227191654"
    # (9215696) (lr = 1e-5, Gauss)        "experiment_outputs/SIR/20240227200703"
    args = parser.parse_args()

    main(path_to_artifact=args.path_to_artifact)