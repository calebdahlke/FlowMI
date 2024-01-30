import plotly.graph_objects as go
import argparse
import numpy as np
import pickle
import os

ESTIMATOR_COLORS = {
    "TrueMI": "rgb(0,0,0,1.0)", # Black
    "FlowP": "rgb(204,0,0,1.0)", # Red
    "FlowMP": "rgb(0,0,153,1.0)", # Blue
    "MINE": "rgb(0,204,204,1.0)", # Light Blue
    "InfoNCE": "rgb(0,153,0,1.0)", # Green
    "NWJ": "rgb(255,0,127,1.0)", # Pink
    "DV": "rgb(204,102,0,1.0)", # Orange
    "KSG": "rgb(102,51,0,1.0)", # Brown
    "CCA": "rgb(96,96,96,1.0)", # Grey
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

def main(path_to_artifact):

    with open(path_to_artifact, "rb") as input_file:
        data_dict = pickle.load(input_file)
    if data_dict["Experiment"][0] == "McAllester_experiment":
        plotMcAlester(data_dict,path_to_artifact)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--path-to-artifact", default="experiment_outputs/LargeMI/20240130004239", type=str)
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