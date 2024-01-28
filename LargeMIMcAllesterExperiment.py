import numpy as np
import bmi
from _flow_estimator import MMFLowEstimator
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import argparse

def main(
    mlflow_experiment_name,
    seed,
    device,
    dim,
    rho,
    num_runs,
    N,
    train_test_split,
    batch_size,
    test_every_n_steps,
    lr,
    num_steps,
    Method_Names,
    ):
    RunFlowP = RunFlowMP = RunMINE = RunInfoNCE = RunNWJ = RunDV = False
    for i in range(len(Method_Names)):
        if Method_Names[i] == 'FlowP':
            FlowPResults = []
            FlowP_best = 0
            FlowP_best_arg = 0
            RunFlowP = True
        elif Method_Names[i] == 'FlowM+P':
            FlowMPResults = []
            FlowMP_best = np.Inf
            FlowMP_best_arg = 0
            RunFlowMP = True
        elif Method_Names[i] == 'MINE':
            mineResults = []
            mine_best = -np.Inf
            mine_best_arg = 0
            RunMINE = True
        elif Method_Names[i] == 'InfoNCE':
            InfoNCEResults = []
            InfoNCE_best = -np.Inf
            InfoNCE_best_arg = 0
            RunInfoNCE = True
        elif Method_Names[i] == 'NWJ':
            NWJResults = []
            NWJ_best = -np.Inf
            NWJ_best_arg = 0
            RunNWJ = True
        elif Method_Names[i] == 'DV':
            DVResults = []
            DV_best = -np.Inf
            DV_best_arg = 0
            RunDV = True
        else:
            print("Method '%s' is unknown and will not be run. Please include only the following:\n \
                'FlowP'\n 'FlowM+P'\n 'MINE'\n 'InfoNCE'\n \
                'NWJ' 'DV'\n"%Method_Names[i])
    
    sigma = np.eye(2*dim)
    sigma[:dim,dim:]=sigma[dim:,:dim]= rho*np.eye(dim)    
    True_MI = -(dim/2)*np.log((1-rho**2))
    for i in range(num_runs):
        np.random.seed(seed+i)
        sample = np.random.multivariate_normal(np.zeros(2*dim),sigma, N)
        X = sample[:,:dim]
        Y = sample[:,dim:]     
        
        if RunFlowMP:
            FlowMP = MMFLowEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)#max_n_steps=int(100*(i+1))
            FlowMPResults.append(FlowMP.estimate_with_info(X,Y))
            if np.array(FlowMPResults[-1].additional_information['test_history'])[-1,1] < FlowMP_best:
                FlowMP_best = np.abs(True_MI -np.array(FlowMPResults[-1].additional_information['test_history'])[-1,1])#MMFLowResults.mi_estimate
                FlowMP_best_arg = i
        
        if RunMINE:
            mine = bmi.estimators.MINEEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
            mineResults.append(mine.estimate_with_info(X,Y))
            if np.array(mineResults[-1].additional_information['test_history'])[-1,1] > mine_best:
                mine_best = np.array(mineResults[-1].additional_information['test_history'])[-1,1]
                mine_best_arg = i
        
        if RunInfoNCE:
            InfoNCE = bmi.estimators.InfoNCEEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
            InfoNCEResults.append(InfoNCE.estimate_with_info(X,Y))
            if np.array(InfoNCEResults[-1].additional_information['test_history'])[-1,1] > InfoNCE_best:
                InfoNCE_best = np.array(InfoNCEResults[-1].additional_information['test_history'])[-1,1]
                InfoNCE_best_arg = i
                
        if RunNWJ:
            NWJ = bmi.estimators.NWJEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
            NWJResults.append(NWJ.estimate_with_info(X,Y))
            if np.array(NWJResults[-1].additional_information['test_history'])[-1,1] > NWJ_best:
                NWJ_best = np.array(NWJResults[-1].additional_information['test_history'])[-1,1]
                NWJ_best_arg = i
        
        if RunDV:
            DV = bmi.estimators.DonskerVaradhanEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
            DVResults.append(DV.estimate_with_info(X,Y))
            if np.array(DVResults[-1].additional_information['test_history'])[-1,1] > DV_best:
                DV_best = np.array(DVResults[-1].additional_information['test_history'])[-1,1]
                DV_best_arg = i

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = np.linspace(0,num_steps-1,num_steps),
                                y = True_MI*np.ones(num_steps),
                                name = 'True', 
                                mode='lines',
                                line = dict(color='rgba(0, 0, 0, 1)',
                                            width = 3)))
    if RunFlowMP:
        fig.add_trace(go.Scatter(x = np.array(FlowMPResults[FlowMP_best_arg].additional_information['training_history'])[:,0]-1,
                                y = np.array(FlowMPResults[FlowMP_best_arg].additional_information['training_history'])[:,1],
                                name = 'MMFlow Train', 
                                line = dict(color='rgba(200, 0, 0, .5)', 
                                            dash = 'dash',
                                            width = 3)))
        fig.add_trace(go.Scatter(x = np.array(FlowMPResults[FlowMP_best_arg].additional_information['test_history'])[:,0]-1,
                                y = np.array(FlowMPResults[FlowMP_best_arg].additional_information['test_history'])[:,1],
                                name = 'MMFlow Test', 
                                mode='lines',
                                line = dict(color='rgba(200, 0, 0, 1)',
                                            width = 3)))

    if RunMINE:
        fig.add_trace(go.Scatter(x = np.array(mineResults[mine_best_arg].additional_information['training_history'])[:,0]-1,
                                    y = np.array(mineResults[mine_best_arg].additional_information['training_history'])[:,1],
                                    name = 'MINE Train', 
                                    line = dict(color='rgba(0, 200, 0, .5)', 
                                            dash = 'dash',
                                            width = 3)))
        fig.add_trace(go.Scatter(x = np.array(mineResults[mine_best_arg].additional_information['test_history'])[:,0]-1,
                                    y = np.array(mineResults[mine_best_arg].additional_information['test_history'])[:,1],
                                    name = 'MINE Test', 
                                    mode='lines',
                                    line = dict(color='rgba(0, 200, 0, 1)', 
                                            width = 3)))
        
    if RunInfoNCE:
        fig.add_trace(go.Scatter(x = np.array(InfoNCEResults[InfoNCE_best_arg].additional_information['training_history'])[:,0]-1,
                                    y = np.array(InfoNCEResults[InfoNCE_best_arg].additional_information['training_history'])[:,1],
                                    name = 'InfoNCE Train', 
                                    line = dict(color='rgba(0, 0, 200, .5)', 
                                            dash = 'dash',
                                            width = 3)))
        fig.add_trace(go.Scatter(x = np.array(InfoNCEResults[InfoNCE_best_arg].additional_information['test_history'])[:,0]-1,
                                    y = np.array(InfoNCEResults[InfoNCE_best_arg].additional_information['test_history'])[:,1],
                                    name = 'InfoNCE Test', 
                                    mode='lines',
                                    line = dict(color='rgba(0, 0, 200, 1)',
                                            width = 3)))
    
    if RunNWJ:
        fig.add_trace(go.Scatter(x = np.array(NWJResults[NWJ_best_arg].additional_information['training_history'])[:,0]-1,
                                    y = np.array(NWJResults[NWJ_best_arg].additional_information['training_history'])[:,1],
                                    name = 'NWJ Train', 
                                    line = dict(color='rgba(0, 200, 200, .5)', 
                                            dash = 'dash',
                                            width = 3)))
        fig.add_trace(go.Scatter(x = np.array(NWJResults[NWJ_best_arg].additional_information['test_history'])[:,0]-1,
                                    y = np.array(NWJResults[NWJ_best_arg].additional_information['test_history'])[:,1],
                                    name = 'NWJ Test', 
                                    mode='lines',
                                    line = dict(color='rgba(0, 200, 200, 1)',
                                            width = 3)))

    if RunDV:
        fig.add_trace(go.Scatter(x = np.array(DVResults[DV_best_arg].additional_information['training_history'])[:,0]-1,
                                    y = np.array(DVResults[DV_best_arg].additional_information['training_history'])[:,1],
                                    name = 'DV Train', 
                                    line = dict(color='rgba(200, 0, 200, .5)', 
                                            dash = 'dash',
                                            width = 3)))
        fig.add_trace(go.Scatter(x = np.array(DVResults[DV_best_arg].additional_information['test_history'])[:,0]-1,
                                    y = np.array(DVResults[DV_best_arg].additional_information['test_history'])[:,1],
                                    name = 'DV Test', 
                                    mode='lines',
                                    line = dict(color='rgba(200, 0, 200, 1)', 
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Large MI estimation experiment"
    )
    parser.add_argument(
        "--mlflow-experiment-name", default="McAllester_experiment", type=str
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cpu", type=str)#"cuda"
    
    parser.add_argument("--dim", default=32, type=int)
    parser.add_argument("--rho", default=.9, type=int)
    
    parser.add_argument("--num-runs", default=1, type=int)
    
    parser.add_argument("--num-samples", default=50000, type=int)
    parser.add_argument("--train_test_split", default=.85, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--test_every_n_steps", default=25, type=int)
    
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num-steps", default=2000, type=int)
    
    Method_Names = ['InfoNCE','NWJ','DV'] # 'FlowP','FlowM+P','MINE',
    parser.add_argument("--method_names", default=Method_Names, type=list)
    

    args = parser.parse_args()

    main(
        mlflow_experiment_name = args.mlflow_experiment_name,
        seed = args.seed,
        device = args.device,
        dim = args.dim,
        rho = args.rho,
        num_runs = args.num_runs,
        N = args.num_samples,
        train_test_split = args.train_test_split,
        batch_size = args.batch_size,
        test_every_n_steps = args.test_every_n_steps,
        lr = args.lr,
        num_steps = args.num_steps,
        Method_Names = args.method_names,
    )