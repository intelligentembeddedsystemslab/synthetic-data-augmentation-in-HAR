from settings import get_args
import preprocessing
from datasets import SensorDataset
import torch
from training import train_model, model_eval
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from models import create
from utils.utils import paint, makedir
from utils.utils_plot import plot_confusion
import json
import pickle
from copy import deepcopy
import shutil
import time
import sys
from datetime import timedelta
from subprocess import run

#from ray import tune
#from ray.tune.schedulers import ASHAScheduler
#import torchvision
#from torchview import draw_graph



def train_holdout(args, dataset_train, dataset_val, config_model, num_loops):
    results = []
    for i in range(num_loops):
        print(f"Starting execution loop number {i}")
        #  create HAR models
        if torch.cuda.is_available():
            model = create(args.model, config_model).to(device=args.device)
        else:
            model = create(args.model, config_model)
                           
        print(f"Number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        #pytroch 2.0 is used, therefore compile model
        #model = torch.compile(model)
        #torch.set_float32_matmul_precision('medium')
        #visualization of archiecture:
        #model_graph = draw_graph(model, input_size=(1, 120, 18), expand_nested=True)
        #model_graph.visual_graph.render()

        results.append(train_model(model, dataset_train, dataset_val, args, verbose=True))
    scores = []
    for dic in results:
        scores.append(np.max(dic["test_fm"]))
    print(f"All fm scores: {scores} \nMean fm score: {np.mean(scores)}")


def LOPO_evaluation(args, config_model, config_dataset):
    results = {}
    #loop through all participants and train model with train set containing all except the current test participant
    start_time = time.time()

    for participant in args.participants:
        print("--" * 50)
        print(f"Starting LOPO evaluation loop with test user: {participant}")
    
        train_x, train_y, test_x, test_y = preprocessing.partition_dataset(args, participant)
        #interpolate data
        train_x = preprocessing.interpolate(train_x)
        test_x = preprocessing.interpolate(test_x)
        #sort so overlap of sliding window contains the same activity as often as possible
        test_x, test_y = preprocessing.sort_by_activity(test_x, test_y)
        train_x, train_y = preprocessing.sort_by_activity(train_x, train_y)

        print(f"Size of train_set: {np.shape(train_x)}")
        print(f"Size of test_set: {np.shape(test_x)}")
        print("Filter, scale and apply sliding window:")

        #filter and scale:
        train_x = preprocessing.lowpass(args, train_x)
        test_x = preprocessing.lowpass(args, test_x)
        train_x, _, test_x = preprocessing.scale(train_x, pd.DataFrame(), test_x, args)

        #plt.plot(train_x[:,-3:], label=" Right Thigh Gyro")
        #plt.plot(train_y, label=" Label")
        #plt.legend()
        #plt.show()
        
        #use sliding window
        train_data, train_target = preprocessing.sliding_window(train_x, train_y, args.window, args.stride, args.window_scheme)
        test_data, test_target = preprocessing.sliding_window(test_x, test_y, args.window, args.stride, args.window_scheme)
        #save data
        print(f"Size of train_set: {np.shape(train_data)}")
        print(f"Size of test_set: {np.shape(test_data)}")
        print("Saving files...")


        makedir(args.path_processed)
        np.savez_compressed(os.path.join(args.path_processed, "train.npz"), data=train_data, target=train_target)
        np.savez_compressed(os.path.join(args.path_processed, "val.npz"), data=test_data, target=test_target)

        #change experiment so logs/figures for all folds get saved
        config_model["experiment"] = args.experiment + "LOPO_" + participant

        dataset = SensorDataset(**config_dataset, prefix="train")
        dataset_val = SensorDataset(**config_dataset, prefix="val")
        if torch.cuda.is_available():
            model = create(args.model, config_model).to(device=args.device)
        else:
            model = create(args.model, config_model)
        
        #uncomment if pytroch 2.0 compile should be used
        #model = torch.compile(model)
        results[participant] = train_model(model, dataset, dataset_val, args, verbose=True)
        best_fm = max(results[participant]["test_fm"])
        print(f"best f1-score result for this user: {best_fm}")

    elapsed = round(time.time() - start_time)
    elapsed = str(timedelta(seconds=elapsed))

    epoch_means = {"train_loss": [], "train_acc": [],"train_fm":[],"train_fw":[],"test_loss":[],"test_acc":[],"test_fm":[],"test_fw":[], }

    #calculate average over each test participant for every epoch
    for i in range(args.epochs):
        epoch = {"train_loss": [], "train_acc": [],"train_fm":[],"train_fw":[],"test_loss":[],"test_acc":[],"test_fm":[],"test_fw":[], }
        for participant in args.participants:
            epoch["train_loss"].append(results[participant]["train_loss"][i])
            epoch["train_acc"].append(results[participant]["train_acc"][i])
            epoch["train_fm"].append(results[participant]["train_fm"][i])
            epoch["train_fw"].append(results[participant]["train_fw"][i])

            epoch["test_loss"].append(results[participant]["test_loss"][i])
            epoch["test_acc"].append(results[participant]["test_acc"][i])
            epoch["test_fm"].append(results[participant]["test_fm"][i])
            epoch["test_fw"].append(results[participant]["test_fw"][i])


        epoch_means["train_loss"].append(np.mean(epoch["train_loss"]))
        epoch_means["train_acc"].append(np.mean(epoch["train_acc"]))
        epoch_means["train_fm"].append(np.mean(epoch["train_fm"]))
        epoch_means["train_fw"].append(np.mean(epoch["train_fw"]))

        epoch_means["test_loss"].append(np.mean(epoch["test_loss"]))
        epoch_means["test_acc"].append(np.mean(epoch["test_acc"]))
        epoch_means["test_fm"].append(np.mean(epoch["test_fm"]))
        epoch_means["test_fw"].append(np.mean(epoch["test_fw"]))

    best_metrics = []
    best_accuracy = []
    final_fm = []
    final_acc = []
    for dic in results.values():
        final_fm.append(dic["test_fm"][-1])
        final_acc.append(dic["test_acc"][-1])
        best_metrics.append(np.max(dic["test_fm"]))
        best_accuracy.append(np.max(dic["test_acc"]))
    print(f"Best fm from each participant: {best_metrics} \nMean best fm score: {np.mean(best_metrics)}")
    print(f"Macro F1 score for each paricipant at last epoch: {final_fm} \nMean last fm: {np.mean(final_fm)}")


    #create total confusion matrix:
    y_true, y_pred = [], []
    for participant in args.participants:
        y_true = y_true + results[participant]["y_true"].tolist()
        y_pred = y_pred + results[participant]["y_pred"].tolist()


    # Plot accuracies
    #set plot parameters
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize']=14
    plt.rcParams['ytick.labelsize']=14
    fig = plt.figure(figsize=(10,8))
    ax = fig.subplots()
    ax.plot(epoch_means["test_fm"], label='Test' )
    ax.plot(epoch_means["train_fm"], label='Train')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r"Macro $\mathbf{F_1}$-score")
    ax.legend()
    ax.grid(True)
    #plt.show()
    fig.savefig(os.path.join(model.path_visuals, "Macro_F1-score_per_epoch.png"))
    #reset parameter again for confusion matrix
    plt.rcParams.update(plt.rcParamsDefault)


    #save dict if experiment is defined
    if args.save_results:
        path = os.path.join(args.path_results, args.experiment)
        makedir(path)
        with open(os.path.join(path, "results.txt"), 'w') as file:
            file.write(f"Parameter for this experiment: {args} \n \n")
            file.write(f"Elapsed time for whole LOPO run: {elapsed} \n \n")
            file.write(f"Best Fm from each participant: {best_metrics} \nMean best Fm score: {np.mean(best_metrics)} \n \n") #best fm for every participant
            file.write(f"Best accuracy from each participant: {best_accuracy} \nMean best accuracy: {np.mean(best_accuracy)} \n \n") #best acc for every participant
            file.write(f"Macro Fm score for each paricipant at last epoch: {final_fm} \nMean last Fm: {np.mean(final_fm)} \n \n")
            file.write(f"Accuracy for each paricipant at last epoch: {final_acc} \nMean last acc: {np.mean(final_acc)} \n \n")
            file.write(f"Best mean Fm for all participants was in epoch {np.argmax(epoch_means['test_fm'])} with an Fm of {np.max(epoch_means['test_fm'])} \n \n")
            file.write(f"Epoch mean dictionary: \n ")
            file.write(json.dumps(epoch_means)) #save all averages over all participants per epoch
        fig.savefig(os.path.join(path, "Macro_F1-score_per_epoch.pdf"))
        #save all results as pkl:
        #with open(os.path.join(path, "result_dict.pkl"), 'wb') as file:
        #    pickle.dump(results, file)
        plot_confusion(y_true, y_pred, os.path.join(path, "Confusion_matrix"), args.epochs -1, True, class_map=args.class_map)


def print_cuda_status():
    print('Python VERSION:', sys.version)
    print('pyTorch VERSION:', torch.__version__)
    print('CUDA VERSION', torch.version.cuda)
    print('CUDNN VERSION:', torch.backends.cudnn.version())
    print('Number CUDA Devices:', torch.cuda.device_count())
    print('Devices:')
    run(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print ('Current cuda device ', torch.cuda.current_device())

#returns index of gpu with most available memory
def get_free_gpu():
    output = run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv"], capture_output=True, text=True).stdout
    memory_available = [int(x.split(' ')[0]) for x in output.splitlines()[1:]]
    return np.argmax(memory_available)


def main():

    # get experiment arguments
    args, config_dataset, config_model = get_args()

    #select GPU to train on if available
    if torch.cuda.is_available():
        #limit number of CPU threads to avoid CPU oversubscription
        print(f"number of utilised threads: {np.floor(os.cpu_count()/torch.cuda.device_count()).astype(int)}")
        torch.set_num_threads(np.floor(os.cpu_count()/torch.cuda.device_count()).astype(int))
        
        print_cuda_status()
        if args.device is None:
            args.device = torch.device(get_free_gpu())
            print(f"free gpu: {args.device}")
        else:
            args.device = torch.device(args.device)
        config_model["cuda"] = args.device

    
    if args.model == "AttendDiscriminate" and not args.center_loss:
        print(paint(f"Model Attend and Discriminate has been selected without center_loss(and maybe mixup)", "warning"))

    #preprocess data according to arguments given
    preprocessing.preprocess_pipeline(args)

    if args.train_mode:

        if args.validation == "Holdout":
            #  create HAR datasets
            dataset_train = SensorDataset(**config_dataset, prefix="train")
            dataset_val = SensorDataset(**config_dataset, prefix="val")
            train_holdout(args, dataset_train, dataset_val, config_model, args.num_loops)
        elif args.validation == "LOPO":
            LOPO_evaluation(args, config_model, config_dataset)
            shutil.rmtree(args.path_lopo)

    #remove the processed data again
    if args.train_mode:
        shutil.rmtree(args.path_processed)
        
    else:
        dataset_test = SensorDataset(**config_dataset, prefix="test")
        config_model["experiment"] = "inference"
        if torch.cuda.is_available():
            model = create(args.model, config_model).to(device=args.device)
        else:
            model = create(args.model, config_model)
                           
        print(f"Number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        model_eval(model, dataset_test, args, False)


if __name__ == "__main__":
    main()
