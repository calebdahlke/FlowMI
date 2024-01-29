import numpy as np
import bmi
from _flow_estimator import MMFLowEstimator

import pickle
import os
import argparse
import time

def main(
    experiment_name,
    seed,
    device,
    dim,
    rho,
    num_runs,
    num_samples,
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
    data_dict = {}
    data_dict["Experiment"] = [experiment_name,seed, dim, rho, num_runs, 
                               num_samples, train_test_split,
                               batch_size, lr, num_steps]
    sigma = np.eye(2*dim)
    sigma[:dim,dim:]=sigma[dim:,:dim]= rho*np.eye(dim)    
    True_MI = -(dim/2)*np.log((1-rho**2))
    data_dict["TrueMI"] = True_MI
    for i in range(num_runs):
        np.random.seed(seed+i)
        sample = np.random.multivariate_normal(np.zeros(2*dim),sigma, num_samples)
        X = sample[:,:dim]
        Y = sample[:,dim:]     
        
        if RunFlowMP:
            FlowMP = MMFLowEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)#max_n_steps=int(100*(i+1))
            FlowMPResults.append(FlowMP.estimate_with_info(X,Y))
            if np.array(FlowMPResults[-1].additional_information['test_history'])[-1,1] < FlowMP_best:
                FlowMP_best = np.abs(True_MI -np.array(FlowMPResults[-1].additional_information['test_history'])[-1,1])#MMFLowResults.mi_estimate
                FlowMP_best_arg = i
            if i == num_runs-1:
                data_dict["FlowMP"] = [FlowMPResults, FlowMP_best_arg]
        
        if RunMINE:
            mine = bmi.estimators.MINEEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
            mineResults.append(mine.estimate_with_info(X,Y))
            if np.array(mineResults[-1].additional_information['test_history'])[-1,1] > mine_best:
                mine_best = np.array(mineResults[-1].additional_information['test_history'])[-1,1]
                mine_best_arg = i
            if i == num_runs-1:
                data_dict["MINE"] = [mineResults, mine_best_arg]
        
        if RunInfoNCE:
            InfoNCE = bmi.estimators.InfoNCEEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
            InfoNCEResults.append(InfoNCE.estimate_with_info(X,Y))
            if np.array(InfoNCEResults[-1].additional_information['test_history'])[-1,1] > InfoNCE_best:
                InfoNCE_best = np.array(InfoNCEResults[-1].additional_information['test_history'])[-1,1]
                InfoNCE_best_arg = i
            if i == num_runs-1:
                data_dict["InfoNCE"] = [InfoNCEResults, InfoNCE_best_arg]
                
        if RunNWJ:
            NWJ = bmi.estimators.NWJEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
            NWJResults.append(NWJ.estimate_with_info(X,Y))
            if np.array(NWJResults[-1].additional_information['test_history'])[-1,1] > NWJ_best:
                NWJ_best = np.array(NWJResults[-1].additional_information['test_history'])[-1,1]
                NWJ_best_arg = i
            if i == num_runs-1:
                data_dict["NWJ"] = [NWJResults, NWJ_best_arg]
        
        if RunDV:
            DV = bmi.estimators.DonskerVaradhanEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
            DVResults.append(DV.estimate_with_info(X,Y))
            if np.array(DVResults[-1].additional_information['test_history'])[-1,1] > DV_best:
                DV_best = np.array(DVResults[-1].additional_information['test_history'])[-1,1]
                DV_best_arg = i
            if i == num_runs-1:
                data_dict["DV"] = [DVResults, DV_best_arg]

    # Pickle the results
    t = time.localtime()
    run_id = time.strftime("%Y%m%d%H%M%S", t)
    path_to_artifact = "./experiment_outputs/LargeMI/{}".format(run_id)
    if not os.path.exists("./experiment_outputs/LargeMI"):
        os.makedirs("./experiment_outputs/LargeMI")
    with open(path_to_artifact, "wb") as f:
        pickle.dump(data_dict, f)
    print("Done.")
    print("Path to artifact - use this when evaluating:\n", path_to_artifact[2:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Large MI estimation experiment"
    )
    parser.add_argument(
        "--experiment-name", default="McAllester_experiment", type=str
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cpu", type=str)#"cuda"
    
    parser.add_argument("--dim", default=2, type=int)#32
    parser.add_argument("--rho", default=.9, type=int)
    
    parser.add_argument("--num-runs", default=1, type=int)
    
    parser.add_argument("--num-samples", default=20000, type=int)
    parser.add_argument("--train_test_split", default=.85, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--test_every_n_steps", default=25, type=int)
    
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num-steps", default=100, type=int)
    
    Method_Names = ['FlowM+P','MINE','InfoNCE','NWJ','DV'] # 'FlowP',
    parser.add_argument("--method_names", default=Method_Names, type=list)
    

    args = parser.parse_args()

    main(
        experiment_name = args.experiment_name,
        seed = args.seed,
        device = args.device,
        dim = args.dim,
        rho = args.rho,
        num_runs = args.num_runs,
        num_samples = args.num_samples,
        train_test_split = args.train_test_split,
        batch_size = args.batch_size,
        test_every_n_steps = args.test_every_n_steps,
        lr = args.lr,
        num_steps = args.num_steps,
        Method_Names = args.method_names,
    )