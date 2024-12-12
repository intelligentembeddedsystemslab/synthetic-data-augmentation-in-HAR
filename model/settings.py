import sys
import argparse
import datetime
import os
import random

__all__ = ["get_args"]
"""
Implementation based on https://github.com/AdelaideAuto-IDLab/Attend-And-Discriminate
"""

def get_args():

    parser = argparse.ArgumentParser(
        description="HAR dataset, model and optimization arguments. Most options are only available for the mosurf dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # get HAR arguments
    parser.add_argument("--experiment", default=None, help="experiment name")
    parser.add_argument(
        "--train_mode", action="store_true", help="execute code in training mode"
    )
    parser.add_argument(
        "--dataset",
        default="mosurf",
        type=str,
        choices=["mosurf", "realworld", "pamap2"],
        help="HAR dataset",
    )
    parser.add_argument(
        "--synthetic_data",  action="store_true", default=False, help="use the synthetic data from MoCap as augmentation, requires the data (not published yet)"
    )
    parser.add_argument(
        "--noise_augmentation",   action="store_true", default=False, help="use noisy variations of the training data as augmentation"
    )
    parser.add_argument(
        "--warp_augmentation",   action="store_true", default=False, help="use over/undersampled variations of the training data as augmentation"
    )
    parser.add_argument(
        "--vertices", default="har5", type=str, choices=["har5", "har9", "har9_2", "har9RW", "har9PAMAP"], help="set of vertices for synthetic data augmentation"
    )
    parser.add_argument(
        "--activities", default="standard8", type=str,choices=["all", "standard8", "standard7", "splitted10"], help="list/number of activities to classify, only valid for mosurf dataset"
        )
    parser.add_argument(
        "--finetune", action="store_true", default=False, help="instead of choosing model and initialize weights, use stored model(see path) and finetune"
    )
    parser.add_argument(
        "--synthetic_training", action="store_true", default=False, help="removes all non-synthetic data from the training set"
    )
    parser.add_argument("--validation", default="LOPO", type=str, choices=["LOPO", "Holdout"], help="validation strategy used for the model")
    parser.add_argument("--imus", default="mars3", type=str, choices=["all", "mars3", "blub3", "watch+phone2", "realworld7","blub10", "blub5", "blub2", "blub7", "pamap2_3"], help="number and location of imus used in the model, only valid for mosurf dataset")
    parser.add_argument("--window", default=120, type=int, help="sliding window size")
    parser.add_argument("--stride", default=60, type=int, help="sliding window stride")
    parser.add_argument("--window_scheme", default="max", type=str, choices=["last", "max"], help="scheme to determine label of a window")
    parser.add_argument(
        "--stride_test", default=1, type=int, help="set to 1 for sample-wise prediction"
    )
    parser.add_argument(
        "--num_participants", default=19, choices=[3, 5, 6, 7, 9, 11, 12, 15, 19], type=int, help="number of participants used from dataset"
    )
    parser.add_argument(
        "--model", default="DeepConvLSTM", choices=["AttendDiscriminate", "DeepConvLSTM", "TransformHAR"],  type=str, help="HAR architecture"
    )
    parser.add_argument(
        "--epochs", default=200, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--load_epoch", default=0, type=int, help="epoch to resume training"
    )
    parser.add_argument(
        "--downsample", default=2, type=int, help="factor for downsampling data(MoSurf dataset is recorded at 100Hz)"
    )
    parser.add_argument(
        "--lowpass_cutoff", default=10, type=int, help="cutoff frequency of lowpass filter used on dataset"
    )
    parser.add_argument(
        "--scaler", default="Standard", choices=["Standard", "MinMax", "MaxAbs", "Robust", "Normalizer", "Quantile", "Power"], type=str,  help="sklearn scaler to use on data, wil be fitted only on training data and applied to train + val + test"
    )
    parser.add_argument(
        "--mixup", action="store_true", default=False, help="use mixup data augmentation"
    )
    parser.add_argument(
        "--center_loss", action="store_true", default=False,  help="use center loss criterium"
    )
    parser.add_argument(
        "--batch_size", default=256, type=int,  help="batch size for training"
    )
    parser.add_argument(
        "--device", type=int, default=None, help="Index of CUDA device to train on, leave empty for automatic assignment"
    )
    parser.add_argument("--print_freq", default=40, type=int)
    parser.add_argument("--num_loops", default=1, type=int, help="number of training loops when using holdout validation")

    args = parser.parse_args()

    #select imus for mosurf:
    match args.imus:
        case "blub10":
            args.imu_list = ["Head", "Upper Thoracic", "Left Forearm", "Right Upper Arm", "Lower Thoracic", "Pelvis", 
            "Left Shank",  "Left Foot", "Right Thigh", "Right Foot"]
        case "realworld7":
            args.imu_list = ["Head", "Upper Thoracic", "Left Upper Arm", "Left Forearm", "Pelvis", "Left Thigh", "Left Shank"] # closest ones to placement used in realworld dataset
        case "blub7":
            args.imu_list = ["Head", "Left Forearm", "Right Upper Arm", "Pelvis", "Left Shank", "Left Foot", "Right Thigh"] 
        case "blub5":
            args.imu_list = ["Head", "Left Forearm", "Pelvis",  "Left Foot", "Right Thigh"]
        case "pamap2_3":
            args.imu_list = ["Upper Thoracic","Right Forearm", "Right Shank"] #closest one to placements used in pamap2 dataset
        case "mars3":
            args.imu_list = ["Head", "Left Forearm", "Right Thigh"] #same 3 as in MARS paper
        case "blub3":
            args.imu_list = ["Left Forearm", "Pelvis", "Left Foot"] 
        case "watch+phone2":
            args.imu_list = ["Left Forearm", "Pelvis"] 
        case "blub2":
            args.imu_list = ["Left Forearm", "Right Thigh"] #alternative watch + smartphone
        case _: #default: use all imus
            args.imu_list = ["Head", "Upper Thoracic", "Left Upper Arm", "Left Forearm", "Right Upper Arm", "Right Forearm", "Lower Thoracic", "Pelvis",
                    "Left Thigh", "Left Shank", "Left Foot", "Right Thigh", "Right Shank", "Right Foot"]
    if args.dataset == "realworld":
        args.imus = "realworld7"
        args.imu_list = ["head", "chest" , "upperarm", "forearm" , "waist" , "thigh", "shin"]
    elif args.dataset == "pamap2":
        args.imus = "pamap2"
        args.imu_list = ["chest", "wrist", "ankle"]
    elif args.dataset == "wisdm":
        args.imus = "wisdm"
        args.imu_list = ["pocket"] 

    #select activities for mosurf:
    match args.activities:
        case "all": #all recorded activities
            args.activity_list = ["getting_up_floor.csv", "jumping.csv", "picking_up_pen.csv", "shelve_ordering.csv", "stair_climbing_up.csv", 
                                  "stair_climbing_down.csv", "walking.csv", "timed_up_go.csv", "cervical_flex.csv",
                                  "cervical_rot.csv", "frontal_flex.csv", "lateral_flex.csv", "lumbar_rot.csv", "lumbar_rot_head.csv", "spagat.csv", "static.csv" ]
        case "standard8":
            args.activity_list = ["getting_up_floor.csv", "jumping.csv", "picking_up_pen.csv", "shelve_ordering.csv", "stair_climbing_up.csv", 
                                  "stair_climbing_down.csv", "walking.csv", "timed_up_go.csv"]
        case "splitted10":
            args.activity_list = ["getting_up_floor.csv", "jumping.csv", "picking_up_pen.csv", "shelve_ordering.csv", "stair_climbing_up.csv", 
                                  "stair_climbing_down.csv", "walking.csv", "sitting_down.csv", "static.csv", "getting_up_chair.csv"]
        case "standard7": 
            args.activity_list = ["getting_up_floor.csv", "jumping.csv", "picking_up_pen.csv", "shelve_ordering.csv", "stair_climbing_up.csv", 
                                  "stair_climbing_down.csv", "walking.csv"]

        #a bit hardcoded selection to simulate smaller training set
    args.participants = []
    if args.dataset == "mosurf":
        participants =  ["AMOAS01", "AMOAS02", "AMOAS03", "AMOAS04", "AMOAS05", "AMOAS06", "AMOAS07", "AMOAS08", 
                         "AMOAS09", "AMOAS010", "AMOAS011", "AMOAS012", "AMOAS013", "AMOAS014", "AMOAS015", 
                         "AMOAS016", "AMOAS017", "AMOAS018", "AMOAS019"]

        args.participants = random.sample(participants, args.num_participants)
        args.participants.sort(key=lambda x: int(x[-2:])) #sort participants according to number

    elif args.dataset == "realworld":
        if args.num_participants == 19: #change default value to realworld participants number
            args.num_participants = 15
        participants = ["proband1", "proband2", "proband3", "proband4", "proband5", "proband6",
                              "proband7", "proband8", "proband9", "proband10", "proband11", "proband12", "proband13", "proband14", "proband15"]
        args.participants = random.sample(participants, args.num_participants)
        args.participants.sort(key=lambda x: int(x[7:])) #sort participants according to number

    elif args.dataset == "pamap2":
        if args.num_participants == 19: #change default value to pamap2 participants number
            args.num_participants = 8
        participants = ["subject101", "subject102", "subject103", "subject104", "subject105",
                 "subject106", "subject107", "subject108" ]
        args.participants = random.sample(participants, args.num_participants)
        args.participants.sort(key=lambda x: int(x[-3:])) #sort participants according to number

    elif args.dataset == "wisdm":
        if args.num_participants == 19: #change default value to wisdm participants number
            args.num_participants = 36
        participants = [f"participant_{x}" for x in range(1,37)]
        args.participants = random.sample(participants, args.num_participants)
        args.participants.sort(key=lambda x: int(x[12:])) #sort participants according to number

    if args.experiment is None:
        args.experiment = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M") 
        args.save_results = False
    else:
        args.save_results = True

    # get HAR dataset arguments     
    if args.dataset == "mosurf":
        args.num_class = len(args.activity_list)
        args.input_dim = len(args.imu_list) * 6
        class_map = [w[:-4] for w in args.activity_list] #remove the .csv from the names and the underscores
        args.class_map = [w.replace('_', ' ') for w in class_map]
    elif args.dataset == "realworld":
        args.activity_list = ["jumping", "lying", "running", "sitting", "standing", "walking", "climbing_down", "climbing_up" ]
        args.num_class = len(args.activity_list)
        if args.finetune:
            args.num_class = 10 #hardcode in case of pretrain because related mosurf dataset has 10 acivities, need to set this to import weights later
        args.input_dim = len(args.imu_list) * 6
        args.class_map  = [w.replace('_', ' ') for w in args.activity_list]
    elif args.dataset == "pamap2":
        args.activity_list = ["lying", "sitting", "standing", "ironing", "vacuum_cleaning", "climbing_up", 
                              "climbing_down", "walking", "nordic_walking" , "cycling", "running", "rope_jumping"]
        args.num_class = len(args.activity_list)
        if args.finetune:
            args.num_class = 10 #hardcode in case of pretrain because related mosurf dataset has 10 acivities, need to set this to import weights later
        args.input_dim = len(args.imu_list) * 6
        args.class_map = [w.replace('_', ' ') for w in args.activity_list]
    elif args.dataset == "wisdm":
        args.activity_list  = ["Jogging", "Walking", "Upstairs", "Downstairs", "Sitting", "Standing"]
        args.num_class = len(args.activity_list)
        if args.finetune:
            args.num_class = 10 #hardcode in case of pretrain because related mosurf dataset has 10 acivities, need to set this to import weights later
        args.input_dim = len(args.imu_list) * 3
        args.class_map = args.activity_list
    else:
        print(f"[!] Unknown HAR dataset: {args.dataset}")
        sys.exit(0)
    
     
    if args.synthetic_data:
        augmentation = "synthetic_" + args.vertices
    elif args.noise_augmentation:
        augmentation = "noise_augmented"
    elif args.warp_augmentation:
        augmentation = "warp_augmented"
    elif not args.dataset == "mosurf" and args.finetune: #distinguish between pretrained if dataset is not mosurf
        augmentation = "pretrained"
    else:
        augmentation = "none"
    #args.path_data = os.path.join(".", "dataset", args.dataset + ".mat")
    if args.activities == "splitted10":
        args.path_raw = os.path.join(".", "data", args.dataset, "raw_splitted")
    else:
        args.path_raw = os.path.join(".", "data", args.dataset, "raw")
    if args.activities == "splitted10":
        args.path_synthetic = os.path.join(".", "data", args.dataset, "synthetic_splitted_" + args.vertices)
    else:
        args.path_synthetic = os.path.join(".", "data", args.dataset, "synthetic_" + args.vertices)
    args.path_processed = os.path.join(".", "data", args.dataset, "processed", args.imus, args.model,augmentation, args.experiment)
    args.path_lopo = os.path.join(".", "data", args.dataset, "lopo", args.imus, args.model, augmentation, args.experiment)
    args.path_results = os.path.join(".", "saved_data", args.dataset, args.validation, args.model, augmentation, args.imus, str(args.num_participants) + "_participants")
 

    # get HAR optimization arguments
    args.weighted_sampler = False
    
    args.optimizer = "Adam"
    args.clip_grad = 0
    args.lr = 0.001
    args.lr_decay = 0.9
    args.lr_step = 10
    args.alpha = 0.8
    args.lr_cent = 0.001

    args.n_head = 8
    args.d_feedforward = 128
    args.sa_div = 2
        
    
    if args.dataset == "mosurf":
        args.init_weights = "orthogonal"
        args.beta = 0.3
        args.dropout = 0.25 #0.25
        args.dropout_rnn = 0.5 #0.5
        args.dropout_cls = 0.5 #0
    else:
        args.init_weights = "orthogonal"
        args.beta = 0.3
        args.dropout = 0.25
        args.dropout_rnn = 0.5
        args.dropout_cls = 0.5


    # get HAR model arguments
    if args.model == "AttendDiscriminate":
        if args.dataset == "mosurf":
            args.filter_num, args.filter_size = 64, 2 #2
            args.enc_num_layers = 6 #6
            args.enc_is_bidirectional = False
            args.hidden_dim = 256 #256
            args.activation = "ReLU"
            args.sa_div = 2 #2
        elif  args.dataset == "realworld":
            args.filter_num, args.filter_size = 64, 5
            args.enc_num_layers = 2
            args.enc_is_bidirectional = False
            args.hidden_dim = 128 
            args.activation = "ReLU"
            args.sa_div = 1
            if args.finetune:
                args.epochs = 2 
                args.lr = 0.00008 
                args.batch_size = 128
            else:
                args.epochs = 2 
                args.lr = 0.0003 
                args.batch_size = 64   
        elif  args.dataset == "pamap2":
            args.filter_num, args.filter_size = 64, 2
            args.enc_num_layers = 6
            args.enc_is_bidirectional = False
            args.hidden_dim = 256 
            args.activation = "ReLU"
            args.sa_div = 2
            if args.finetune:
                args.epochs = 10
                args.batch_size = 128
                args.lr = 0.0001 
            else:
                args.epochs = 20
                args.batch_size = 128
                args.lr = 0.0001   
        
    if args.model == "TransformHAR":
        if args.dataset == "mosurf":
            args.filter_num, args.filter_size = 16, 1
            args.enc_num_layers = 8
            args.enc_is_bidirectional = False
            args.hidden_dim = 32
            args.activation = "ReLU"
            args.lr = 0.00004
            args.dropout = 0.1
            args.dropout_rnn = 0.1
            args.dropout_cls = 0.25
        elif  args.dataset == "realworld":
            args.filter_num, args.filter_size = 16, 1
            args.enc_num_layers = 8
            args.enc_is_bidirectional = False
            args.hidden_dim = 32
            args.activation = "ReLU"
            args.dropout = 0.1
            args.dropout_rnn = 0.1
            args.dropout_cls = 0.5
            if args.finetune:
                args.batch_size = 128
                args.lr = 0.000001
                args.epochs = 6
            else:
                args.batch_size = 128
                args.epochs = 3
                args.lr = 0.00005  
        elif  args.dataset == "pamap2":
            args.filter_num, args.filter_size = 16, 1
            args.enc_num_layers = 8
            args.enc_is_bidirectional = False
            args.hidden_dim = 32
            args.activation = "ReLU"
            args.dropout = 0.1
            args.dropout_rnn = 0.1
            args.dropout_cls = 0.5
            if args.finetune:
                args.epochs = 10
                args.batch_size = 128
                args.lr = 0.00005 
            else:
                args.epochs = 10
                args.batch_size = 128
                args.lr = 0.0001     

    if args.model == "DeepConvLSTM":
        if args.dataset == "mosurf":
            args.filter_num, args.filter_size = 16, 4
            args.enc_num_layers = 2
            args.enc_is_bidirectional = False
            args.hidden_dim = 128
            args.activation = "ReLU"
        elif  args.dataset == "realworld":
            args.filter_num, args.filter_size = 16, 4 #32
            args.enc_num_layers = 2
            args.enc_is_bidirectional = False
            args.hidden_dim = 128 #512
            args.activation = "ReLU"
            if args.finetune:
                args.epochs = 4
                args.lr = 0.00008
                args.batch_size = 256
            else:
                args.epochs = 2
                args.lr = 0.0002
                args.batch_size = 256  
        elif  args.dataset == "pamap2":
            args.filter_num, args.filter_size = 16, 4 #32
            args.enc_num_layers = 2
            args.enc_is_bidirectional = False
            args.hidden_dim = 128 #512
            args.activation = "ReLU"
            if args.finetune:
                args.epochs = 7
                args.lr = 0.0005
                args.batch_size = 128
            else:
                args.epochs = 5
                args.lr = 0.0008
                args.batch_size = 256    
        elif  args.dataset == "wisdm":
            args.filter_num, args.filter_size = 16, 4 #32
            args.enc_num_layers = 2
            args.enc_is_bidirectional = False
            args.hidden_dim = 128 #512
            args.activation = "ReLU"
            if args.finetune:
                args.epochs = 7
                args.lr = 0.0005
                args.batch_size = 128
            else:
                args.epochs = 5
                args.lr = 0.0008
                args.batch_size = 256       

    
    # set dataset and model arguments
    config_dataset = {
        "dataset": args.dataset,
        "window": args.window,
        "stride": args.stride,
        "stride_test": args.stride_test,
        "path_processed": args.path_processed,
    }

    #set model config
    config_model = {
            "model": args.model,
            "dataset": args.dataset,
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,  
            "filter_num": args.filter_num, 
            "filter_size": args.filter_size, 
            "enc_num_layers": args.enc_num_layers, 
            "enc_is_bidirectional": args.enc_is_bidirectional,
            "dropout": args.dropout,
            "dropout_rnn": args.dropout_rnn,
            "dropout_cls": args.dropout_cls,
            "activation": args.activation,
            "sa_div": args.sa_div, 
            "num_class": args.num_class,
            "n_head": args.n_head, 
            "d_feedforward": args.d_feedforward, 
            "train_mode": args.train_mode,
            "experiment": args.experiment,
        }
    
    return args, config_dataset, config_model


if __name__ == "__main__":
    get_args()
