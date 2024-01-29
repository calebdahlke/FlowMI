import plotly.graph_objects as go
import argparse
import numpy as np
import pickle
import os

ESTIMATOR_COLORS = {
    "TrueMI": "rgb(0,0,0,1.0)", # Black
    "FlowP": "rgb(181,29,20,1.0)", # Red
    "FlowMP": "rgb(64,83,211,1.0)", # Blue
    "MINE": "rgb(221,179,16,1.0)", # Orange
    "InfoNCE": "rgb(0,178,93,1.0)", # Green
    "NWJ": "rgb(251,73,176,1.0)", # Pink
    "DV": "rgb(0,190,255,1.0)", # Light Blue
    "KSG": "rgb(68,53,0,1.0)", # Brown
    "CCA": "rgb(202,202,202,1.0)", # Grey
}

def plotMcAlester(data_dict):
    
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

    fig.show()
    return

def main(path_to_artifact):

    with open(path_to_artifact, "rb") as input_file:
        data_dict = pickle.load(input_file)
    if data_dict["Experiment"][0] == "McAllester_experiment":
        plotMcAlester(data_dict)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--path-to-artifact", default="experiment_outputs/LargeMI/20240129114952", type=str)
    args = parser.parse_args()

    main(path_to_artifact=args.path_to_artifact)