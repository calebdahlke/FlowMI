import numpy as np
import bmi
# from _flow_estimator import FlowPostEstimator, FlowMargPostEstimator
from joblib import Parallel, delayed
import pickle
import os
import argparse
import time
from _flow_estimator import NeuralVariationalEstimator,JointVariationalEstimator

def evaluate_parallel(i,
    seed,
    device,
    dim,
    sigma,
    num_samples,
    train_test_split,
    batch_size,
    test_every_n_steps,
    lr,
    num_steps,
    Method_Names,
    ):
    results = {}
    # np.random.seed(seed+i)
    # sample = np.random.multivariate_normal(np.zeros(2*dim),sigma, num_samples)
    # X = sample[:,:dim]
    # Y = sample[:,dim:]
    task = bmi.benchmark.BENCHMARK_TASKS['multinormal-dense-25-25-0.5']
    X, Y = task.sample(num_samples, seed=seed+i)
    dim = task.dim_x
    
    # if 'FlowP' in Method_Names:
    #     # FlowP = FlowPostEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)#max_n_steps=int(100*(i+1))
    #     FlowP = NeuralVariationalEstimator(dim_x=dim,dim_y=dim,batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
    #     # FlowP = JointVariationalEstimator(dim_x=dim,dim_y=dim,batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
    #     FlowPResults = FlowP.estimate_with_info(X,Y)
    #     results["FlowP"] = FlowPResults
    
    # if 'FlowMP' in Method_Names:
    #     # FlowMP = FlowMargPostEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)#max_n_steps=int(100*(i+1))
    #     FlowMP = JointVariationalEstimator(dim_x=dim,dim_y=dim,batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
    #     FlowMPResults = FlowMP.estimate_with_info(X,Y)
    #     results["FlowMP"] = FlowMPResults
    if 'JointGauss' in Method_Names:
        JointGauss = JointVariationalEstimator(dim_x=dim,dim_y=dim,use_flow=False,batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
        JointGaussResults = JointGauss.estimate_with_info(X,Y)
        results['JointGauss'] = JointGaussResults
        
    if 'NeuralGauss' in Method_Names:
        NeuralGauss = NeuralVariationalEstimator(dim_x=dim,dim_y=dim,use_flow=False,batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
        NeuralGaussResults = NeuralGauss.estimate_with_info(X,Y)
        results['NeuralGauss'] = NeuralGaussResults
        
    if 'JointFlow' in Method_Names:
        JointFlow = JointVariationalEstimator(dim_x=dim,dim_y=dim,use_flow=True,batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
        JointFlowResults = JointFlow.estimate_with_info(X,Y)
        results['JointFlow'] = JointFlowResults
        
    if 'NeuralFlow' in Method_Names:
        NeuralFlow = NeuralVariationalEstimator(dim_x=dim,dim_y=dim,use_flow=True,batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
        NeuralFlowResults = NeuralFlow.estimate_with_info(X,Y)
        results['NeuralFlow'] = NeuralFlowResults
    
    if 'MINE' in Method_Names:
        mine = bmi.estimators.MINEEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
        mineResults = mine.estimate_with_info(X,Y)
        results["MINE"] = mineResults
    
    if 'InfoNCE' in Method_Names:
        InfoNCE = bmi.estimators.InfoNCEEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
        InfoNCEResults = InfoNCE.estimate_with_info(X,Y)
        results["InfoNCE"] = InfoNCEResults
            
    if 'NWJ' in Method_Names:
        NWJ = bmi.estimators.NWJEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
        NWJResults = NWJ.estimate_with_info(X,Y)
        results["NWJ"] = NWJResults
    
    if 'DV' in Method_Names:
        DV = bmi.estimators.DonskerVaradhanEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
        DVResults = DV.estimate_with_info(X,Y)
        results["DV"] = DVResults
    return results

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
    
    data_dict = {}
    data_dict["Experiment"] = [experiment_name,seed, dim, rho, num_runs, 
                               num_samples, train_test_split,
                               batch_size, lr, num_steps]
    sigma = np.eye(2*dim)
    sigma[:dim,dim:]=sigma[dim:,:dim]= rho*np.eye(dim)    
    True_MI = -(dim/2)*np.log((1-rho**2))
    data_dict["TrueMI"] = True_MI
    
    results = Parallel(n_jobs=num_runs)(delayed(evaluate_parallel)(i,
                                                seed,
                                                device,
                                                dim,
                                                sigma,
                                                num_samples,
                                                train_test_split,
                                                batch_size,
                                                test_every_n_steps,
                                                lr,
                                                num_steps,
                                                Method_Names
                                                ) for i in range(num_runs))
    
    for k in results[0].keys():
        runs = list(d[k] for d in results)
        arg_best_run = np.argmax(np.min(True_MI-runs[i][-1]) for i in range(len(runs))) 
        data_dict[k] = [runs, arg_best_run]
        
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
    parser.add_argument("--device", default="cpu", type=str)
    
    parser.add_argument("--dim", default=15, type=int)#32
    parser.add_argument("--rho", default=.95, type=int)
    
    parser.add_argument("--num-runs", default=1, type=int)#10
    
    parser.add_argument("--num-samples", default=75000, type=int)#
    parser.add_argument("--train_test_split", default=.5, type=float)#.8
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--test_every_n_steps", default=50, type=int)
    
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num-steps", default=3000, type=int)#10000
    
    Methods = ['JointGauss','NeuralGauss','JointFlow','NeuralFlow','MINE']#'FlowP','FlowMP','MINE','InfoNCE','NWJ','DV']#, 'CCA', 'KSG'
    # Methods = ['JointGauss','NeuralGauss','JointFlow','NeuralFlow', 'MINE', 'InfoNCE', 'NWJ', 'DV']
    parser.add_argument("--method_names", default=Methods, type=list)
    

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