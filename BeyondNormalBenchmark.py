import bmi
from _flow_estimator import MargPostEstimator, FlowMargPostEstimator
import argparse
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import time
import os
import pickle
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

def evaluate_parallel_run(i,
                        seed,
                        task,
                        num_samples,
                        train_test_split,
                        batch_size,
                        num_steps,
                        lr,
                        test_every_n_steps,
                        method_names):
    X, Y = task.sample(num_samples, seed=seed+i)
    
    run = {}
    if 'FlowMP' in method_names:
        MMFlow = FlowMargPostEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)#max_n_steps=int(100*(i+1))
        MMFLowResults = MMFlow.estimate_with_info(X,Y)
        run['FlowMP'] = MMFLowResults#np.array(MMFLowResults.additional_information['test_history'])[-1,1]
    
    if 'MPGauss' in method_names:
        MPGauss = MargPostEstimator(train_test_split=train_test_split)
        MPGaussResults = MPGauss.estimate_with_info(X,Y)
        run['MPGauss']= MPGaussResults#MPGaussResults.mi_estimate
    
    if 'MINE' in method_names:
        mine = bmi.estimators.MINEEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
        mineResults = mine.estimate_with_info(X,Y)
        run['MINE'] = mineResults #mineResults.mi_estimate
    
    if 'InfoNCE' in method_names:
        InfoNCE = bmi.estimators.InfoNCEEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
        InfoNCEResults = InfoNCE.estimate_with_info(X,Y)
        run['InfoNCE'] = InfoNCEResults#InfoNCEResults.mi_estimate
    
    if 'NWJ' in method_names:
        NWJ = bmi.estimators.NWJEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
        NWJResults = NWJ.estimate_with_info(X,Y)
        run['NWJ'] = NWJResults#NWJResults.mi_estimate
    
    if 'DV' in method_names:
        DV = bmi.estimators.DonskerVaradhanEstimator(batch_size = batch_size, max_n_steps=num_steps, learning_rate=lr,test_every_n_steps=test_every_n_steps,train_test_split=train_test_split)
        DVResults = DV.estimate_with_info(X,Y)
        run['DV'] = DVResults#DVResults.mi_estimate
        
    if 'CCA' in method_names:
        cca = bmi.estimators.CCAMutualInformationEstimator()
        run['CCA'] = cca.estimate(X, Y)
    
    if 'KSG' in method_names:
        ksg = bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,))#10
        run['KSG'] = ksg.estimate(X, Y)
        
    return run

def evaluate_parallel_task(j,
                            seed,
                            task_list,
                            num_runs,
                            num_samples,
                            train_test_split,
                            batch_size,
                            num_steps,
                            lr,
                            test_every_n_steps,
                            method_names):

    np.random.seed(seed+j)
    
    task = bmi.benchmark.BENCHMARK_TASKS[task_list[j]]
    runs = {}
    runs[task_list[j]] = task.mutual_information
    with parallel_backend("loky", inner_max_num_threads=num_runs):
        results = Parallel(n_jobs=num_runs)(delayed(evaluate_parallel_run)(i,
                                                    seed,
                                                    task,
                                                    num_samples,
                                                    train_test_split,
                                                    batch_size,
                                                    num_steps,
                                                    lr,
                                                    test_every_n_steps,
                                                    method_names
                                                    ) for i in range(num_runs))
    
    # results = []
    # for i in range(num_runs):
    #     resulti = evaluate_parallel_run(i,
    #                                     seed,
    #                                     task,
    #                                     num_samples,
    #                                     train_test_split,
    #                                     batch_size,
    #                                     num_steps,
    #                                     lr,
    #                                     test_every_n_steps,
    #                                     method_names)
    #     results.append(resulti)
        
    for k in results[0].keys():
        runs[k] = list(d[k] for d in results)
        
    return runs

def main(experiment_name,
        seed,
        device,
        task_list,
        num_runs,
        num_samples,
        train_test_split,
        batch_size,
        test_every_n_steps,
        lr,
        num_steps,
        method_names):
    
    num_tasks = len(task_list)
    
    experiments = {}
    experiments['task_list'] = task_list
    experiments['method_names'] = method_names
    task_data = Parallel(n_jobs=num_tasks)(delayed(evaluate_parallel_task)(j,
                                                seed,
                                                task_list,
                                                num_runs,
                                                num_samples,
                                                train_test_split,
                                                batch_size,
                                                num_steps,
                                                lr,
                                                test_every_n_steps,
                                                method_names
                                                ) for j in range(num_tasks))
    experiments['task_data'] = task_data
    # Pickle the results
    t = time.localtime()
    # run_id = time.strftime("%Y%m%d%H%M%S", t)
    run_id = experiment_name
    path_to_artifact = "./experiment_outputs/BMI/{}".format(run_id)
    if not os.path.exists("./experiment_outputs/BMI"):
        os.makedirs("./experiment_outputs/BMI")
    with open(path_to_artifact, "wb") as f:
        pickle.dump(experiments, f)

    print("Done.")
    print("Path to artifact - use this when evaluating:\n", path_to_artifact[2:])  
        


if __name__ == "__main__":
    Trans1D1 = ['1v1-normal-0.75',                             #0
    'normal_cdf-1v1-normal-0.75',                           #1
    '1v1-additive-0.1',                                     #2
    '1v1-additive-0.75',                                    #3
    '1v1-bimodal-0.75',                                     #4
    'wiggly-1v1-normal-0.75',                               #5
    'half_cube-1v1-normal-0.75',                            #6
    'student-identity-1-1-1',                               #7
    'asinh-student-identity-1-1-1',                         #8
    'swissroll_x-normal_cdf-1v1-normal-0.75',               #9
    ] #30min * 10 task *10 runs (parallel 100)
    DenseNorm2 = ['multinormal-dense-2-2-0.5',                  #10
    'multinormal-dense-3-3-0.5',                            #11
    'multinormal-dense-5-5-0.5',                            #12
    'multinormal-dense-25-25-0.5',                          #13
    ] #(25,25,50,100) min 4 task * 10 runs (parallel 40)
    LargeNorm3 = ['multinormal-dense-50-50-0.5',                #14
    ] #150 min 1 task * 10 runs (parallel 10)
    SparseNorm4 = [
    'multinormal-sparse-2-2-2-2.0',                         #15
    'multinormal-sparse-3-3-2-2.0',                         #16
    'multinormal-sparse-5-5-2-2.0',                         #17
    ] #(25,25,50) min 3 task * 10 runs (parallel 30)
    StudentT5 = [
    'student-identity-2-2-1',                               #18
    'student-identity-2-2-2',                               #19
    'student-identity-3-3-2',                               #20
    'student-identity-3-3-3',                               #21
    'student-identity-5-5-2',                               #22
    'student-identity-5-5-3',                               #23
    ] #(25,25,25,25,50,50) min 6 task 10 run (parallel 60)
    SparseUnif6 = [
    'normal_cdf-multinormal-sparse-3-3-2-2.0',              #24
    'normal_cdf-multinormal-sparse-5-5-2-2.0',              #25
    'normal_cdf-multinormal-sparse-25-25-2-2.0',            #26
    ] #(25,50,120) min 3 task 10 run (parallel 30)
    TransNorm7 = [
    'spiral-multinormal-sparse-3-3-2-2.0',                  #28
    'spiral-multinormal-sparse-5-5-2-2.0',                  #29
    'spiral-multinormal-sparse-25-25-2-2.0',                #30
    ] #(25,50,120) min 3 task 10 run (parallel 30)
    SpiralUnif8 = [
    'spiral-normal_cdf-multinormal-sparse-3-3-2-2.0',       #31
    'spiral-normal_cdf-multinormal-sparse-5-5-2-2.0',       #32
    'spiral-normal_cdf-multinormal-sparse-25-25-2-2.0',     #33
    ] #(25,50,120) min 3 task 10 run (parallel 30)
    AsinhST9 = [
    'half_cube-multinormal-sparse-25-25-2-2.0',             #27
    'asinh-student-identity-2-2-1',                         #34
    'asinh-student-identity-3-3-2',                         #35
    'asinh-student-identity-5-5-2'                          #36
    ]#(120,25,25,50) min 4 task 10 run (parallel 40)
    parser = argparse.ArgumentParser(
        description="Benchmark from Beyond Normal paper"
    )
    parser.add_argument(
        "--experiment-name", default="BenchmarkMI", type=str
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cpu", type=str)#"cuda"
    
    parser.add_argument("--task_list", default=Trans1D1, type=list)#TASKS
    
    parser.add_argument("--num_runs", default=3, type=int)#10
    
    parser.add_argument("--num_samples", default=10000, type=int)
    parser.add_argument("--train_test_split", default=.8, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--test_every_n_steps", default=100, type=int)#1000
    
    parser.add_argument("--lr", default=0.003, type=float)
    parser.add_argument("--num_steps", default=200, type=int)#10000
    
    Method_Names = ['FlowMP', 'MPGauss', 'MINE', 'InfoNCE', 'NWJ', 'DV', 'CCA', 'KSG']
    # Method_Names = ['MPGauss']
    parser.add_argument("--method_names", default=Method_Names, type=list)
    

    args = parser.parse_args()
    
    main(
        experiment_name = args.experiment_name,
        seed = args.seed,
        device = args.device,
        task_list = args.task_list,
        num_runs = args.num_runs,
        num_samples = args.num_samples,
        train_test_split = args.train_test_split,
        batch_size = args.batch_size,
        test_every_n_steps = args.test_every_n_steps,
        lr = args.lr,
        num_steps = args.num_steps,
        method_names = args.method_names,)