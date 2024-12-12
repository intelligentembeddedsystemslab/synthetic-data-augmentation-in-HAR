
import os
import time
import random
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from datetime import timedelta

from utils.utils_mixup import mixup_data, MixUpLoss
from utils.utils_centerloss import compute_center_loss, get_center_delta
from utils.utils_plot import plot_confusion


if __package__ is None or __package__ == '':
    # uses current directory visibility
    from utils.utils import  paint, AverageMeter
    from utils.utils_pytorch import init_weights_orthogonal

else:
    # uses current package visibility
    from .utils import  paint, AverageMeter
    from .utils.utils_pytorch import init_weights_orthogonal

"""
Implementation based on https://github.com/AdelaideAuto-IDLab/Attend-And-Discriminate
"""

train_on_gpu = torch.cuda.is_available()  # Check for cuda


def train_model(model, dataset, dataset_val, args, verbose=False):
    """
    Train model for a number of epochs.

    :param model: A pytorch model
    :param dataset: A SensorDataset containing the data to be used for training the model.
    :param dataset_val: A SensorDataset containing the data to be used for validation of the model.
    :param args: A dict containing config options for the training.
    Required keys:
                    'batch_size': int, number of windows to process in each batch (default 256)
                    'optimizer': str, optimizer function to use. Options: 'Adam' or 'RMSProp'. Default 'Adam'.
                    'lr': float, maximum initial learning rate. Default 0.001.
                    'lr_step': int, interval at which to decrease the learning rate. Default 10.
                    'lr_decay': float, factor by which to  decay the learning rate. Default 0.9.
                    'init_weights': str, How to initialize weights. Options 'orthogonal' or None. Default 'orthogonal'.
                    'epochs': int, Total number of epochs to train the model for. Default 300.
                    'print_freq': int, How often to print loss during each epoch if verbose=True. Default 100.

    :param verbose:
    :return:
    """
    if verbose:
        print("Starting Loop with the following arguments:")
        print(args)
        print(paint("Running HAR training loop ..."))

    loader = DataLoader(dataset, args.batch_size, shuffle=True, pin_memory=True)
    loader_val = DataLoader(dataset_val, args.batch_size, shuffle=False, pin_memory=True)



    if args.init_weights == "orthogonal":
        if verbose:
            print(paint("[-] Initializing weights (orthogonal)..."))
        model.apply(init_weights_orthogonal)


    if args.finetune:
        #weight averaging over all saved checkpoints:
        sd_list = []
        for i, file in enumerate(os.listdir(os.path.join(os.getcwd(), "models", "Finetune", args.model, args.dataset))):
            print("Loading checkpoint: " + file)
            checkpoint = torch.load(os.path.join(os.getcwd(), "models", "Finetune", args.model, args.dataset, file))
            model.load_state_dict(checkpoint["model_state_dict"])

        args.num_class = len(args.activity_list) #set num_class back to correct value for dataset
        if args.model == "TransformHAR":
            model.classifier = torch.nn.Linear(args.input_dim * args.filter_num, args.num_class) #reset final fully connected layer
            center_size = int(args.input_dim * args.filter_num)

        else:
            model.classifier = torch.nn.Linear(args.hidden_dim, args.num_class) #reset final fully connected layer
            center_size = int(args.hidden_dim)

        #create center buffer with correct num_class:
        if torch.cuda.is_available():
            model.register_buffer(
                "centers", (torch.randn(args.num_class, center_size).to(device=args.device))
            )
        else:
            model.register_buffer(
                "centers", (torch.randn(args.num_class, center_size))
            )
        model.to(device=args.device)


    params = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=args.lr)
    elif args['optimizer'] == "RMSprop":
        optimizer = optim.RMSprop(params, lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step , gamma=args.lr_decay
    )
    
    if train_on_gpu:
        print(paint(f"CUDA available, training on GPU: {args.device}"))
        criterion = nn.CrossEntropyLoss(reduction="mean").to(device=args.device)
    else:
        print(paint("No GPU available, training on CPU..."))
        criterion = nn.CrossEntropyLoss(reduction="mean")


    history = {"train_loss": [], "train_acc": [],"train_fm":[],"train_fw":[],"test_loss":[],"test_acc":[],"test_fm":[],"test_fw":[], "y_true":[], "y_pred":[], }
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    metric_best = 0.0
    start_time = time.time()


    for epoch in range(args.epochs):
        if verbose:
            print("--" * 50)
            print("[-] Learning rate: ", optimizer.param_groups[0]["lr"])
        train_one_epoch(model, loader, criterion, optimizer, args, verbose)
        loss, acc, fm, fw, _ , _ = eval_one_epoch(model, loader, criterion, args, epoch)

        start_inf = time.time()
        loss_val, acc_val, fm_val, fw_val, y_true, y_pred = eval_one_epoch(model, loader_val, criterion, args, epoch)
        inf_time = round(time.time() - start_inf)


        if verbose:
            print(
                paint(
                    f"[-] Epoch {epoch}/{(args.epochs - 1)}"
                    f"\tTrain loss: {loss:.2f} \tacc: {100 * acc:.2f}(%)\tfm: {100 * fm:.2f}(%)\tfw: {100 * fw:.2f}"
                    f"(%)\t"
                )
            )

            print(
                paint(
                    f"[-] Epoch {epoch}/{(args.epochs - 1)}"
                    f"\tVal loss: {loss_val:.2f} \tacc: {100 * acc_val:.2f}(%)\tfm: {100 * fm_val:.2f}(%)"
                    f"\tfw: {100 * fw_val:.2f}(%)"
                    f"\t Best fm so far: {100 * metric_best:.4f}(%)"
                )
            )


        #save stats for LOPO:
        history["train_loss"].append(loss)
        history["train_acc"].append(acc)
        history["train_fm"].append(fm)
        history["train_fw"].append(fw)

        history["test_loss"].append(loss_val)
        history["test_acc"].append(acc_val)
        history["test_fm"].append(fm_val)
        history["test_fw"].append(fw_val) 


        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "random_rnd_state": random.getstate(),
            "numpy_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
        }

        metric = fm_val
        if metric >= metric_best:
            if verbose:
                print(paint(f"[*] Saving checkpoint... ({metric_best}->{metric})", "blue"))
            metric_best = metric
       #     torch.save(
        #      checkpoint, os.path.join(model.path_checkpoints, "checkpoint_best.pth")
        #    )

        if  epoch == args.epochs - 1: #save results for confusion matrix of last epoch
            history["y_true"] = y_true
            history["y_pred"] = y_pred

        scheduler.step()

       # if epoch == args.epochs -1:
        #    torch.save(
         #     checkpoint, os.path.join(model.path_checkpoints, "checkpoint_last.pth")
          #  )


    elapsed = round(time.time() - start_time)
    elapsed = str(timedelta(seconds=elapsed))
    if verbose:
        print(paint(f"Finished HAR training loop (h:m:s): {elapsed}"))
        print(paint("--" * 50, "blue"))
    
    return history


def train_one_epoch(model, loader, criterion, optimizer, args, verbose=False):
    losses = AverageMeter("Loss")
    model.train()

    for batch_idx, (data, target, idx) in enumerate(loader):
        if train_on_gpu:
            data = data.to(device=args.device)
            target = target.view(-1).to(device=args.device)
        else:
            target = target.view(-1)

        if args.center_loss:
            centers = model.centers

        if args.mixup:
            data, y_a_y_b_lam = mixup_data(data, target, args.device, args.alpha)

        z, logits = model(data)

        if args.mixup:
            criterion = MixUpLoss(criterion)
            loss = criterion(logits, y_a_y_b_lam)
        else:
            loss = criterion(logits, target)

        if args.center_loss:
            center_loss = compute_center_loss(z, centers, target)
            loss = loss + args.beta * center_loss
        else:
            loss = criterion(logits, target)
            
        losses.update(loss.item(), data.shape[0])

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        if args.center_loss:
            center_deltas = get_center_delta(z.data, centers, target, args.lr_cent, args.device)
            model.centers = centers - center_deltas

        if verbose:
            if batch_idx % args.print_freq == 0:
                print(f"[-] Batch {batch_idx}/{len(loader)}\t Loss: {str(losses)}")
        
        if args.mixup:
            criterion = criterion.get_old()


def eval_one_epoch(model, loader, criterion, args, epoch):
    losses = AverageMeter("Loss")
    y_true, y_pred = [], []
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target, idx) in enumerate(loader):
            if train_on_gpu:
                data = data.to(device=args.device)
                target = target.to(device=args.device)

            z, logits = model(data)
            loss = criterion(logits, target.view(-1))
            losses.update(loss.item(), data.shape[0])
            probabilities = nn.Softmax(dim=1)(logits)
            _, predictions = torch.max(probabilities, 1)

            y_pred.append(predictions.cpu().numpy().reshape(-1))
            y_true.append(target.cpu().numpy().reshape(-1))

    # append invalid samples at the beginning of the test sequence
    if loader.dataset.prefix == "test":
        ws = data.shape[1] - 1
        samples_invalid = [y_true[0][0]] * ws
        y_true.append(samples_invalid)
        y_pred.append(samples_invalid)

    y_true = np.concatenate(y_true, 0)
    y_pred = np.concatenate(y_pred, 0)


    acc = metrics.accuracy_score(y_true, y_pred)
    fm = metrics.f1_score(y_true, y_pred, average="macro")
    fw = metrics.f1_score(y_true, y_pred, average="weighted")


    if epoch % 10 == 0 or epoch == args.epochs - 1 or not args.train_mode: #plot confusion matrix every 10th or last training epoch
        plot_confusion(y_true, y_pred, os.path.join(model.path_visuals, "cm" ,loader.dataset.prefix), epoch, True, class_map=args.class_map)
    return losses.avg, acc, fm, fw, y_true, y_pred


def model_eval(model, dataset_test, args, return_results):
    print(paint("Running HAR evaluation loop ..."))

    loader_test = DataLoader(dataset_test, args.batch_size, shuffle=False, pin_memory=True)

    if train_on_gpu:
        criterion = nn.CrossEntropyLoss(reduction="mean").to(device=args.device)
    else:
        criterion = nn.CrossEntropyLoss(reduction="mean")

    print("[-] Loading checkpoint ...")

    path_checkpoint = os.path.join(model.path_checkpoints, "checkpoint_best.pth")

    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion.load_state_dict(checkpoint["criterion_state_dict"])

    start_time = time.time_ns()

    loss_test, acc_test, fm_test, fw_test, y_true, y_pred = eval_one_epoch(
        model, loader_test, criterion, args, 0
    )

    print(
        paint(
            f"[-] Test loss: {loss_test:.2f}"
            f"\tacc: {100 * acc_test:.2f}(%)\tfm: {100 * fm_test:.2f}(%)\tfw: {100 * fw_test:.2f}(%)"
        )
    )

    elapsed = round(time.time_ns() - start_time)
    elapsed = pd.Timedelta(elapsed, "ns")
    print(paint(f"[Finished HAR evaluation loop (ms): {elapsed.microseconds / 1e3}"))
    
    if return_results:
        return acc_test, fm_test, fw_test, elapsed

