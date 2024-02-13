import plotly.graph_objects as go
import argparse
import numpy as np
import pickle
import os

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

def plotMcAlester(data_dict,path_to_artifact):
    
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
            fig.add_trace(go.Scatter(x = np.array(Results[best_arg].additional_information['training_history'])[:,0]-1,
                                    y = np.array(Results[best_arg].additional_information['training_history'])[:,1],
                                    name = key+' Train', 
                                    line = dict(color=ESTIMATOR_COLORS[key][:-4]+"0.5)", # add opacity 
                                                dash = 'dash',
                                                width = 3)))
    for index, key in enumerate(data_dict):
        if key not in ["Experiment","TrueMI"]:
            [Results, best_arg] = data_dict[key] 
            fig.add_trace(go.Scatter(x = np.array(Results[best_arg].additional_information['test_history'])[:,0]-1,
                                    y = np.array(Results[best_arg].additional_information['test_history'])[:,1],
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
                        font=dict(size=40),)
    fig.update_xaxes(mirror=True,
                        ticks='outside',
                        showline=True,
                        linecolor='black',
                        gridcolor='lightgrey')
    fig.update_yaxes(mirror=True,
                        ticks='outside',
                        showline=True,
                        linecolor='black',
                        gridcolor='lightgrey')
    fig.write_image(path_to_artifact+'.pdf')
    fig.show()
    print("Done.")

def plotBMI(data_dict,path_to_artifact):
    means = np.zeros((len(data_dict['task_list']),len(data_dict['method_names'])))
    data_dict['method_names'].reverse()
    for i, method in enumerate(data_dict['method_names']):
        for j, task in enumerate(data_dict['task_list']):
            means[j, i] = format(np.mean(data_dict['task_data'][j][method]),'.2f')

    # Add a row for true values
    true_values = np.array([format(data_dict['task_data'][i][task],'.2f') for i, task in enumerate(data_dict['task_list'])])
    means = np.hstack((means,true_values.reshape(-1,1)))
    Methods = data_dict['method_names']
    Methods.append('True')

    # Define layout
    fig = go.Figure(data=[go.Table(#values=[Methods] + [means[j] for j in range(len(data_dict['task_list']))]
        cells=dict(values = means,
                fill_color='lavender',
                align='left'))
    ])

    # Define layout
    fig.update_layout(
                    autosize=True,
                    )
    fig.for_each_trace(lambda t: t.update(header_fill_color = 'rgba(0,0,0,0)'))
    # Create annotations for rotated method names (column headers)
    annotations = []
    col_title = ['']+data_dict['task_list']
    for i, task in enumerate(col_title):
        x = i / (len(col_title))  # Normalize positions
        y = .55
        annotations.append(
            go.layout.Annotation(
                x=x,
                y=y,
                text=task,
                showarrow=False,
                textangle=305,
                xanchor="center",
                font=dict(
                    size=12,
                )
            )
        )
    row_title = ['']+Methods    
    for i, method in enumerate(row_title):
        x = -.01  # Normalize positions
        y = 1-i / (4.85*len(row_title))
        annotations.append(
            go.layout.Annotation(
                x=x,
                y=y,
                text=method,
                showarrow=False,
                textangle=0,
                xanchor="center",
                font=dict(
                    size=12,
                )
            )
        )

    # Add created annotations and update layout
    fig.update_layout(annotations=annotations, margin=dict(t=0, b=0,l=50,r=0))#

    fig.show()
    #pandas and seaborn
    print("Done.")
    

def main(path_to_artifact):

    with open(path_to_artifact, "rb") as input_file:
        data_dict = pickle.load(input_file)
    first = path_to_artifact.index('/')+1
    second = path_to_artifact.index('/',first,len(path_to_artifact))
    experiment_name = path_to_artifact[first:second]
    if experiment_name == 'BMI':
        plotBMI(data_dict,path_to_artifact)
    # if data_dict["Experiment"][0] == "McAllester_experiment":
    if experiment_name == 'LargeMI':
        plotMcAlester(data_dict,path_to_artifact)
    if experiment_name == 'loc_fin':
        print("not yet implimented")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--path-to-artifact", default="experiment_outputs/BMI/20240130012752", type=str)
    #McAllester HPC: experiment_outputs/LargeMI/20240130210847
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