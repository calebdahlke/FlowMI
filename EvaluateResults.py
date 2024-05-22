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
import bmi

# ESTIMATOR_COLORS = {
#     "TrueMI": "rgba(0,0,0,1.0)", # Black
#     "FlowP": "rgba(204,0,0,1.0)", # Red
#     "FlowMP": "rgba(0,0,153,1.0)", # Blue
#     "MINE": "rgba(0,204,204,1.0)", # Light Blue
#     "InfoNCE": "rgba(0,153,0,1.0)", # Green
#     "NWJ": "rgba(255,0,127,1.0)", # Pink
#     "DV": "rgba(204,102,0,1.0)", # Orange
#     "KSG": "rgba(102,51,0,1.0)", # Brown
#     "CCA": "rgba(96,96,96,1.0)", # Grey
# }

ESTIMATOR_COLORS = {
    "TrueMI": "rgba(0,0,0,1.0)", # Black
    "JointGauss": "rgba(204,0,0,1.0)", # Red
    "NeuralGauss": "rgba(204,102,0,1.0)", # Orange
    "JointFlow": "rgba(127, 0, 255,1.0)", # Violet
    "NeuralFlow": "rgba(0,0,153,1.0)", # Blue
    "MINE": "rgba(0,204,204,1.0)", # Light Blue
    "InfoNCE": "rgba(0,153,0,1.0)", # Green
    "NWJ": "rgba(255,0,127,1.0)", # Pink
    "DV": "rgba(102,51,0,1.0)", # Brown
}

ESTIMATOR_NAME = {
    "TrueMI": "True", # Black
    "JointGauss": "JVG", # Red
    "NeuralGauss": "NVG", # Orange
    "JointFlow": "JVF", # Violet
    "NeuralFlow": "NVF", # Blue
    "MINE": "MINE", # Light Blue
    "InfoNCE": "InfoNCE", # Green
    "NWJ": "NWJ", # Pink
    "DV": "DV", # Brown
}
ESTIMATOR_GROUP = {
    "TrueMI": "group1", # Black
    "JointGauss": "group2", # Red
    "NeuralGauss": "group3", # Orange
    "JointFlow": "group2", # Violet
    "NeuralFlow": "group3", # Blue
    "MINE": "group4", # Light Blue
    "InfoNCE": "group4", # Green
    "NWJ": "group5", # Pink
    "DV": "group5", # Brown
}

def plotMcAlester(path_to_artifact):
    with open(path_to_artifact, "rb") as input_file:
        data_dict = pickle.load(input_file)
    num_steps = data_dict["Experiment"][9]
    fig = go.Figure()
    log_limit = np.log(data_dict['Experiment'][7])
    fig.add_trace(go.Scatter(x = np.linspace(0,num_steps-1,num_steps),
                                y = data_dict["TrueMI"]*np.ones(num_steps),
                                legendgroup="group1",
                                name = 'True', 
                                mode='lines',
                                line = dict(color=ESTIMATOR_COLORS['TrueMI'],
                                            width = 3)))
    fig.add_trace(go.Scatter(x = np.linspace(0,num_steps-1,num_steps),
                                y = log_limit*np.ones(num_steps),
                                legendgroup="group1",
                                name = 'ln N', 
                                mode='lines',
                                line = dict(color="rgba(0,0,0,.5)",
                                            width = 3,
                                            dash='dash')))
    # for index, key in enumerate(data_dict):
    #     if key not in ["Experiment","TrueMI","JointGauss"]:
    #         [Results, best_arg] = data_dict[key] 
    #         fig.add_trace(go.Scatter(x = np.array(Results[0].additional_information['training_history'])[:,0]-1,#Results[3]
    #                                 y = np.array(Results[0].additional_information['training_history'])[:,1],#Results[3]
    #                                 name = key+' Train', 
    #                                 line = dict(color=ESTIMATOR_COLORS[key][:-4]+"0.5)", # add opacity 
    #                                             dash = 'dash',
    #                                             width = 3)))
    for index, key in enumerate(data_dict):
        if key in ["JointGauss"]:
            [Results, best_arg] = data_dict[key]
            fig.add_trace(go.Scatter(x = np.linspace(0,num_steps-1,num_steps),
                            y = Results[0].mi_estimate*np.ones(num_steps),
                            legendgroup=ESTIMATOR_GROUP[key],
                            name = ESTIMATOR_NAME[key],#key+' Test', 
                            mode='lines',
                            line = dict(color=ESTIMATOR_COLORS[key],
                                        width = 3)))
        if key not in ["Experiment","TrueMI","JointGauss"]:
            [Results, best_arg] = data_dict[key]
            fig.add_trace(go.Scatter(x = np.array(Results[0].additional_information['test_history'])[:,0]-1,#Results[3]
                                    y = np.array(Results[0].additional_information['test_history'])[:,1],#Results[3]
                                    legendgroup=ESTIMATOR_GROUP[key],
                                    name = ESTIMATOR_NAME[key],#key+' Test', 
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
                        font=dict(size=20),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=.02,
                            xanchor="right",
                            x=.99
                        ))
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
        # if task_name == 'Student-t 1 × 1 (dof=1)':
        #     print('hello')
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
            meta_dict['method_names'] = ['MINE', 'InfoNCE', 'NWJ', 'DV', 'CCA', 'KSG','JointGauss','JointFlow','NeuralGauss','NeuralFlow' ]#
            for j, method_name in enumerate(meta_dict['method_names']):
                # if method_name in ['CCA','KSG']:
                #     task_mean[method_name] = np.mean(task_dict[method_name])
                #     task_std[method_name] = np.std(task_dict[method_name])
                #     task_mean_err[method_name] = np.mean(task_dict[method_name]) - task_dict[meta_dict['task_list'][i]]
                #     task_mean_rel_err[method_name] = (np.mean(task_dict[method_name]) - task_dict[meta_dict['task_list'][i]])/task_dict[meta_dict['task_list'][i]]
                #     task_mean_end[method_name] = np.mean(task_dict[method_name])
                # elif method_name in ['JointGauss']:
                #     max_mi_est = []
                #     for k in range(len(task_dict[method_name])):
                #         max_mi_est.append(task_dict[method_name][k].mi_estimate)
                #     task_mean[method_name] = np.mean(max_mi_est)
                #     task_std[method_name] = np.std(max_mi_est)
                #     task_mean_err[method_name] = np.mean(max_mi_est) - task_dict[meta_dict['task_list'][i]]
                #     task_mean_rel_err[method_name] = (np.mean(max_mi_est) - task_dict[meta_dict['task_list'][i]])/task_dict[meta_dict['task_list'][i]]
                #     task_mean_end[method_name] = np.mean(max_mi_est)
                if method_name in ['NeuralGauss','NeuralFlow']:#elif method_name in ['NeuralGauss','JointFlow','NeuralFlow']:
                    max_mi_est = []
                    end_mi_est = []
                    for k in range(len(task_dict[method_name])):
                        max_mi_est.append(task_dict[method_name][k].mi_estimate)
                        end_mi_est.append(task_dict[method_name][0].additional_information['test_history'][-1][1])
                    task_mean[method_name] = np.mean(max_mi_est)#np.mean(end_mi_est)
                    task_std[method_name] = np.std(max_mi_est)
                    task_mean_err[method_name] = (np.mean(max_mi_est) - task_dict[meta_dict['task_list'][i]])#(np.mean(end_mi_est) - task_dict[meta_dict['task_list'][i]])
                    task_mean_rel_err[method_name] = (np.mean(max_mi_est) - task_dict[meta_dict['task_list'][i]])/task_dict[meta_dict['task_list'][i]]#(np.mean(end_mi_est) - task_dict[meta_dict['task_list'][i]])/task_dict[meta_dict['task_list'][i]]
                    task_mean_end[method_name] = np.mean(max_mi_est)      
                    if np.mean(max_mi_est)<-2:
                        task_mean[method_name] = -np.inf 
                        task_mean_rel_err[method_name] =   -np.inf
                # else:
                #     max_mi_est = []
                #     end_mi_est = []
                #     for k in range(len(task_dict[method_name])):
                #         max_mi_est.append(task_dict[method_name][k].mi_estimate)
                #         end_mi_est.append(task_dict[method_name][0].additional_information['test_history'][-1][1])
                #     task_mean[method_name] = np.mean(max_mi_est)
                #     task_std[method_name] = np.std(max_mi_est)
                #     task_mean_err[method_name] = (np.mean(max_mi_est) - task_dict[meta_dict['task_list'][i]])
                #     task_mean_rel_err[method_name] = (np.mean(max_mi_est) - task_dict[meta_dict['task_list'][i]])/task_dict[meta_dict['task_list'][i]]
                #     task_mean_end[method_name] = np.mean(end_mi_est)   
                #     if np.mean(max_mi_est)<-2:
                #         task_mean[method_name] = -np.inf 
                #         task_mean_rel_err[method_name] =   -np.inf
            data_mean[task_name] = task_mean
            data_std[task_name] = task_std
            data_mean_err[task_name] = task_mean_err
            data_mean_rel_err[task_name] = task_mean_rel_err
            data_mean_end[task_name] = task_mean_end

    df_mean = pd.DataFrame(data_mean)
    df_mean_err = pd.DataFrame(data_mean_rel_err)#data_mean_err)#
    # df_mean_err=(df_mean_err)/df_mean_err.std()        
    vcenter = 0
    # vmin, vmax = y.min(), y.max() 
    normalize = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=-1, vmax=1)#vmin=df_mean_err.min().min(), vmax=df_mean_err.max().max())#
    colormap = cm.seismic#cm.RdBu#cm.coolwarm#bwr#
    plt.figure(figsize=(15,6))
    ax = sns.heatmap(data=df_mean_err, annot=df_mean,norm=normalize, cmap=colormap, cbar=False, fmt=".2f", xticklabels=True, yticklabels=True)#, annot=True
    ax.set(xlabel="", ylabel="")
    loc, labels = plt.xticks()
    ax.set_xticklabels(labels, rotation=60,ha='right')#
    row = 0
    for idx in np.arange(df_mean.size).reshape(df_mean.shape)[row, :]:
        ax.texts[idx].set_fontweight('bold')
        
    # min_indices = df_mean_err.iloc[1:].abs().idxmin()
    
    ax.axhline(y=1, color='black', linewidth=1)  # Line between first and second rows
    ax.axhline(y=7, color='black', linewidth=1)  # Line between fifth and sixth rows
    ax.axhline(y=9, color='black', linewidth=1)  # Line between fifth and sixth rows
    ax.figure.tight_layout()
    # plt.xticks(rotation=30)
    # plt.show()
    # first = path_to_artifact.index('/')+1
    # second = path_to_artifact.index('/',first,len(path_to_artifact))
    # plot_name = path_to_artifact[second+1:]
    plt.savefig(path_to_artifact+'/BigBMITable.pdf')
    # pio.write_image(ax,path_to_artifact+'/plot_2.pdf')
    plt.show()
    
    
    Small_List = ['Bivariate normal 1 × 1', 'Normal CDF @ Bivariate normal 1 × 1', 'Uniform 1 × 1 (additive noise=0.75)', 'Bimodal 1 × 1', 'Wiggly @ Bivariate normal 1 × 1',
     'Swiss roll 2 × 1', 'Multinormal 3 × 3 (dense)', 'Multinormal 3 × 3 (2-pair)', 'Normal CDF @ Multinormal 3 × 3 (2-pair)', 
     'Normal CDF @ Multinormal 25 × 25 (2-pair)', 'Half-cube @ Multinormal 25 × 25 (2-pair)', 'Spiral @ Multinormal 3 × 3 (2-pair)', 
     'Spiral @ Multinormal 25 × 25 (2-pair)', 'Spiral @ Normal CDF @ Multinormal 3 × 3 (2-pair)', 'Spiral @ Normal CDF @ Multinormal 25 × 25 (2-pair)',
     'Student-t 2 × 2 (dof=1)','Student-t 3 × 3 (dof=2)','Asinh @ Student-t 2 × 2 (dof=1)', 'Asinh @ Student-t 3 × 3 (dof=2)']#, 'Student-t 5 × 5 (dof=2)','Asinh @ Student-t 5 × 5 (dof=2)'
    df_mean_small = df_mean[Small_List]
    df_mean_err_small = df_mean_err[Small_List]
    plt.figure(figsize=(9,6))
    ax = sns.heatmap(data=df_mean_err_small, annot=df_mean_small,norm=normalize, cmap=colormap, cbar=False, fmt=".2f", xticklabels=True, yticklabels=True)#, annot=True
    ax.set(xlabel="", ylabel="")
    loc, labels = plt.xticks()
    ax.set_xticklabels(labels, rotation=60,ha='right')#
    row = 0
    for idx in np.arange(df_mean_small.size).reshape(df_mean_small.shape)[row, :]:
        ax.texts[idx].set_fontweight('bold')
        
    # min_indices = df_mean_err.iloc[1:].abs().idxmin()
    
    ax.axhline(y=1, color='black', linewidth=1)  # Line between first and second rows
    ax.axhline(y=7, color='black', linewidth=1)  # Line between fifth and sixth rows
    ax.axhline(y=9, color='black', linewidth=1)  # Line between fifth and sixth rows
    ax.figure.tight_layout()
    
    # plt.xticks(rotation=30)
    # plt.show()
    # first = path_to_artifact.index('/')+1
    # second = path_to_artifact.index('/',first,len(path_to_artifact))
    # plot_name = path_to_artifact[second+1:]
    plt.savefig(path_to_artifact+'/SmallBMITable.pdf')
    # pio.write_image(ax,path_to_artifact+'/plot_2.pdf')
    plt.show()
    
    print("Done.")

def plotLocationFinding(path_to_artifact):
    with open(path_to_artifact, "rb") as input_file:
        data_dict = pickle.load(input_file)
    test = 10

def main(path_to_artifact):
    fig = go.Figure(go.Scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16]))
    # fig.show()
    fig.write_image("random.pdf")
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--path-to-artifact", default="experiment_outputs/BMI/20240521225601", type=str)
    # McAllester HPC: "experiment_outputs/LargeMI/20240130210847" "large network 10000 total sample"
    #                 "experiment_outputs/LargeMI/20240412004044" "large network 10000 total sample"
    #                 "experiment_outputs/BMI/20240521132933" (3215522) "with timing"
    #                 "experiment_outputs/BMI/20240521225601" (swap X and Y)
    # BMI HPC:  experiment_outputs/BMI/20240513200457  - small network 1000 sample per dimension ([8,1] 32)
    #           experiment_outputs/BMI/20240514100652  - large network 10000 total sample ([8,8] 128)
    #           experiment_outputs/BMI/20240516125116  - large network 1000 sample per dimension ([8,8] 128) random seed: 42
    #           experiment_outputs/BMI/20240518094248  - large network 1000 sample per dimension medium network ([16] 64)
    #           experiment_outputs/BMI/20240519012247  - large network 1000 sample per dimension ([8,8] 128) random seed: 1234
    args = parser.parse_args()

    main(path_to_artifact=args.path_to_artifact)
    
    # if np.shape(X)[1] == 1:
    #     plt.scatter(X, Y, s=10, c='blue', alpha=0.5)  # You can customize the color, size, and transparency (alpha) as needed
    #     # Add labels and title
    #     plt.xlabel('X-axis label')
    #     plt.ylabel('Y-axis label')
    #     plt.title('Scatter Plot of X and Y')

    #     # Show the plot
    #     plt.show()
        
    # if np.shape(X)[1] == 2:
    #     plt.scatter(X[:,0], X[:,1], s=10, c='blue', alpha=0.5)

    #     # Add labels and title
    #     plt.xlabel('X0-axis label')
    #     plt.ylabel('X1-axis label')
    #     plt.title('Scatter Plot of X and Y')

    #     # Show the plot
    #     plt.show()

    #     plt.scatter(X[:,0], Y[:,0], s=10, c='blue', alpha=0.5)

    #     # Add labels and title
    #     plt.xlabel('X0-axis label')
    #     plt.ylabel('Y0-axis label')
    #     plt.title('Scatter Plot of X and Y')

    #     # Show the plot
    #     plt.show()

    #     plt.scatter(X[:,1], Y[:,1], s=10, c='blue', alpha=0.5)

    #     # Add labels and title
    #     plt.xlabel('X1-axis label')
    #     plt.ylabel('Y1-axis label')
    #     plt.title('Scatter Plot of X and Y')

    #     # Show the plot
    #     plt.show()

    #     plt.scatter(Y[:,0], Y[:,1], s=10, c='blue', alpha=0.5)

    #     # Add labels and title
    #     plt.xlabel('Y0-axis label')
    #     plt.ylabel('Y1-axis label')
    #     # plt.yscale()
    #     plt.title('Scatter Plot of X and Y')

    #     # Show the plot
    #     plt.show()

    # plt.plot(np.array(MMFLowResults.additional_information['test_history'])[:,0],task.mutual_information*np.ones(len(np.array(MMFLowResults.additional_information['test_history'])[:,1])),label='True', color='black')

    # plt.plot(np.array(MMFLowResults.additional_information['training_history'])[:,0],np.array(MMFLowResults.additional_information['training_history'])[:,1],label='MMFlow Train', color='blue', linestyle='dashed', alpha = .5)
    # plt.plot(np.array(MMFLowResults.additional_information['test_history'])[:,0],np.array(MMFLowResults.additional_information['test_history'])[:,1],label='MMFlow Test', color='blue')

    # plt.plot(np.array(mineResults.additional_information['training_history'])[:,0],np.array(mineResults.additional_information['training_history'])[:,1],label='MINE Train', color='orange', linestyle='dashed', alpha = .5)
    # plt.plot(np.array(mineResults.additional_information['test_history'])[:,0],np.array(mineResults.additional_information['test_history'])[:,1],label='MINE Test', color='orange')

    # # plt.plot(np.array(InfoNCEResults.additional_information['training_history'])[:,0],np.array(InfoNCEResults.additional_information['training_history'])[:,1],label='InfoNCE Train', color='green', linestyle='dashed', alpha = .5)
    # # plt.plot(np.array(InfoNCEResults.additional_information['test_history'])[:,0],np.array(InfoNCEResults.additional_information['test_history'])[:,1],label='InfoNCE Test', color='green')


    # # Add labels and title
    # plt.xlabel('Training Step')
    # plt.ylabel('MI approx')
    # plt.title('Training MI')
    # plt.legend()
    # plt.ylim(0, None)

    # # Show the plot
    # plt.show()