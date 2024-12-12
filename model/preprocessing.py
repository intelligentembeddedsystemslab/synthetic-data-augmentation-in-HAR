
import sys
import os
import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn import utils as skutils
import glob
from scipy import signal
from matplotlib import pyplot as plt
import utils.utils as utils


from settings import get_args
from utils.utils_pamap2 import read_pamap2
from utils.utils_mosurf import read_mosurf, create_mosurf_dataframes
from utils.utils_realworld import read_realworld

__all__ = ["preprocess_pipeline", "partition_dataset", "scale", "lowpass", "sliding_window", "sort_by_activity", "interpolate"]


def add_synthetic_data(args, participants):
    #adds the synthetic data as additional particiant, only posible for the MoSurf dataset

    for name, data in participants.items():
        dir = os.path.join(args.path_synthetic, name)
        paths = []
        valid_paths = []
        if not os.path.exists(dir): #skip if participant has no synthetic data
            continue
        print(f"Adding additional synthetic data for participant {name}")
        for entry in os.scandir(dir): #every vertex
            paths += (glob.glob(os.path.join(entry.path, "*.csv")))
        for file in paths:
            if os.path.basename(file) in args.activity_list:
                valid_paths.append(file)
        data_x, data_y = create_mosurf_dataframes(args, valid_paths, synthetic=True)

        data_x.columns = data[0].columns #copy column names from real data for simpler handling 
        participants[name] = [pd.concat([data[0], data_x], ignore_index=True), pd.concat([data[1], data_y], ignore_index=True)] # append synthetic data to the end of the df of real measurements
        participants[name] = sort_by_activity(participants[name][0], participants[name][1]) #optional: sort so overlap of sliding window contains the same activity as often as possible
    return participants


def select_imus(data_x, imus):
    #returns dataframe with only selected imus
    # to be called after seperatio into data_x and data_y
    columns= []
    for col in data_x.columns:
        for imu in imus:
            if imu in col: 
                columns.append(col)
        if col == 'synthetic':
            columns.append(col) #also keep synthetic laber to delete synthetic data in test/val set later
        if col == 'augmented':
            columns.append(col) #same with (noise) augmented
    return data_x[data_x.columns.intersection(columns)]

def interpolate(data_x):
    print(f" Interpolating {data_x.isna().values.sum()} NaN values...")
    # Perform linear interpolation
    data_x =  data_x.interpolate()
    
    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0
    return data_x

def lowpass(args, data_x):
    #lowpass filter data with given cutoff in args
    b, a = signal.butter(4, args.lowpass_cutoff /((100 / args.downsample)/2), "low") #create filter depending on sample rate(and therefore Nyquist freq.) and cutoff freq.
    data_x_filter = signal.filtfilt(b,a,data_x.to_numpy().transpose())
    data_x = pd.DataFrame(data_x_filter.transpose(), columns=data_x.columns)
    return data_x

def scale(train_x, val_x, test_x, args):
    match args.scaler:
        case "MinMax":
            scaler = sklearn.preprocessing.MinMaxScaler()
        case "MaxAbs":
            scaler = sklearn.preprocessing.MaxAbsScaler()
        case "Standard":
            scaler = sklearn.preprocessing.StandardScaler()
        case "Robust":
            scaler = sklearn.preprocessing.RobustScaler()
        case "Normalizer":
            scaler = sklearn.preprocessing.Normalizer()
        case "Quantile":
            scaler = sklearn.preprocessing.QuantileTransformer()
        case "Power":
            scaler = sklearn.preprocessing.PowerTransformer()
    print(f"Used scaler: {scaler}")
    #fit scaler only on training data:
    scaler.fit(train_x)
    #apply data to all sets:
    train_x = scaler.transform(train_x)
    if not val_x.empty:
        val_x = scaler.transform(val_x)
    if not test_x.empty:
        test_x = scaler.transform(test_x)

    return train_x, val_x, test_x

def sliding_window(x, y, window, stride, scheme="max"):
    data, target = [], []
    start = 0
    y = np.ravel(y)
    while start + window < x.shape[0]:
        end = start + window
        x_segment = x[start:end]
        if scheme == "last":
            # last scheme: : last observed label in the window determines the segment annotation
            y_segment = y[start:end][-1]
        elif scheme == "max":
            # max scheme: most frequent label in the window determines the segment annotation
            y_segment = np.argmax(np.bincount(y[start:end]))
        data.append(x_segment)
        target.append(y_segment)
        start += stride
    return data, target


def partition_dataset(args, test_participant):
    #partition dataset for LOPO
    train_x = np.empty( [0, args.input_dim], dtype=float )
    train_y = np.empty( [0], dtype=int )

    test_x  = np.empty( [0,  args.input_dim], dtype=float )
    test_y  = np.empty( [0], dtype=int )

    for participant in args.participants:
        df = pd.read_pickle(os.path.join(args.path_lopo, f"{participant}_x.pkl"))
        data_y = pd.read_pickle(os.path.join(args.path_lopo, f"{participant}_y.pkl"))
        df["activityID"] = data_y

        if participant == test_participant:
            df.drop(df[df["synthetic"] == 1].index, inplace = True) #remove synthetic data from test set
            df = df.drop(columns=["synthetic"])
            df.drop(df[df["augmented"] == 1].index, inplace = True) #remove noise augmented data from test set
            df = df.drop(columns=["augmented"])
            test_x  = np.concatenate( (test_x, df.iloc[:,:-1].to_numpy()), axis=0 )
            test_y  = np.concatenate( (test_y, df.iloc[:,-1].to_numpy()), axis=0 )
                
        else:
            if not args.synthetic_data:
                df.drop(df[df["synthetic"] == 1].index, inplace = True) #just to be sure that no synthetic data from another run is still saved
            if args.synthetic_training:
                df.drop(df[df["synthetic"] == 0].index, inplace = True) #remove non-synthetic data
                df.reset_index(inplace=True, drop=True)
            df = df.drop(columns=["synthetic"])  #remove marker for synthetic data  
            df = df.drop(columns=["augmented"])  #remove marker for noise augmented data       
            train_x  = np.concatenate( (train_x, df.iloc[:,:-1].to_numpy()), axis=0 )
            train_y  = np.concatenate( (train_y, df.iloc[:,-1].to_numpy()), axis=0 )
    train_x = pd.DataFrame(train_x)
    test_x = pd.DataFrame(test_x)     

    return  train_x, train_y, test_x, test_y

def sort_by_activity(data_x, data_y):
    data_x["activityID"] = data_y #add label again 
    data_x['colFromIndex'] = data_x.index # add index by column to differentiate between participants
    data_x = data_x.sort_values(['activityID', 'colFromIndex']) #sort by activity first and then by partcipant
    data_x = data_x.iloc[:,:-1] #remove (old) index
    data_x = data_x.reset_index(drop=True) #set new index
    
    #seperate x and y yet again
    data_y = data_x.iloc[:,-1]
    data_x = data_x.iloc[:,:-1]
    return [data_x, data_y]


def preprocess_pipeline(args):
    # read csv files of individual participants into dataframes:
    if args.dataset == "mosurf":
        participants = read_mosurf(args)
    elif args.dataset == "realworld":
        participants = read_realworld(args)
    elif args.dataset == "pamap2":
        participants = read_pamap2(args)

    if args.synthetic_data:
        #append synthetic augmentations to dataframe of corresponding participant
        participants = add_synthetic_data(args, participants)

    #select imu which will be used in experiment
    for participant in participants.values():
        participant[0] = select_imus(participant[0], args.imu_list)
    
    #save subjects for further processing inside the LOPO loop
    if args.validation == "LOPO":
        utils.makedir(args.path_lopo)
        for name, data in participants.items():
            #save as dataframe because need to differentiate real from synthetic data later
            data[0].to_pickle(os.path.join(args.path_lopo, name + "_x.pkl"))
            data[1].to_pickle(os.path.join(args.path_lopo, name + "_y.pkl"))
            #np.savez_compressed(os.path.join(args.path_lopo, name + ".npz"), x=data[0], y=data[1])

    #partition and process for Holdout
    elif args.validation == "Holdout":
        train_x, train_y, val_x, val_y, test_x, test_y = ([] for i in range(6))
        #split participants into respective groups
        if args.dataset == "mosurf":
            test_participants = ["AMOAS01", "AMOAS015"]
            val_participants = ["AMOAS017", "AMOAS014"]
        elif args.dataset == "realworld":
            test_participants = ["proband1"] 
            val_participants = ["proband3"] 
        elif args.dataset == "pamap2":
            test_participants = ["subject106"]
            val_participants = ["subject105"]
        for name, data in participants.items():
            if name in test_participants: 
                test_x.append(data[0])
                test_y.append(data[1])
            elif name in val_participants:
                val_x.append(data[0])
                val_y.append(data[1])
            else:
                train_x.append(data[0])
                train_y.append(data[1])

        train_x= pd.concat(train_x, ignore_index=True)
        train_y= pd.concat(train_y, ignore_index=True)
        train_x = interpolate(train_x)
        train_x, train_y = sort_by_activity(train_x, train_y) #optional: sort so overlap of sliding window contains the same activity as often as possible
        if args.synthetic_training:
            train_x["activityID"] = train_y #add y values to remove synthetic data
            train_x.drop(train_x[train_x["synthetic"] == 0].index, inplace = True) #remove non.synthetic data 
            train_x.reset_index(inplace=True, drop=True)
            train_y = train_x.iloc[:,-1] 
            train_x = train_x.iloc[:,:-1] #remove y again

        

        val_x= pd.concat(val_x,ignore_index=True)
        val_y= pd.concat(val_y,ignore_index=True)
        val_x = interpolate(val_x)
        val_x["activityID"] = val_y #add y values to remove synthetic data
        val_x.drop(val_x[val_x["synthetic"] == 1].index, inplace = True) #remove synthetic data from validation set
        val_x.drop(val_x[val_x["augmented"] == 1].index, inplace = True) #remove augmented data 
        val_x.reset_index(inplace=True, drop=True)
        val_y = val_x.iloc[:,-1] 
        val_x = val_x.iloc[:,:-1] #remove y again


        test_x= pd.concat(test_x,ignore_index=True)
        test_y= pd.concat(test_y,ignore_index=True)
        test_x = interpolate(test_x)
        test_x["activityID"] = test_y #add y values to remove synthetic data
        test_x.drop(test_x[test_x["synthetic"] == 1].index, inplace = True) #remove synthetic data from test set
        test_x.drop(test_x[test_x["augmented"] == 1].index, inplace = True) #remove augmented data 
        test_x.reset_index(inplace=True, drop=True)
        test_y = test_x.iloc[:,-1] 
        test_x = test_x.iloc[:,:-1] #remove y again

        #remove synthetic and augmented data marker:
        train_x = train_x.drop(columns=["synthetic"])
        val_x = val_x.drop(columns=["synthetic"])        
        test_x = test_x.drop(columns=["synthetic"])
        train_x = train_x.drop(columns=["augmented"])
        val_x = val_x.drop(columns=["augmented"])        
        test_x = test_x.drop(columns=["augmented"])

        print(f"Size of train_set: {np.shape(train_x)}")
        print(f"Size of val_set: {np.shape(val_x)}")
        print(f"Size of test_set: {np.shape(test_x)}")
        print("filter, scale and apply sliding window:")
        #filter and scale:
        train_x = lowpass(args, train_x)
        val_x = lowpass(args, val_x)
        test_x = lowpass(args, test_x)
        train_x, val_x, test_x = scale(train_x, val_x, test_x, args)


        #use sliding window#
        train_data, train_target = sliding_window(train_x, train_y.to_numpy(), args.window, args.stride, args.window_scheme)
        val_data, val_target = sliding_window(val_x, val_y.to_numpy(), args.window, args.stride, args.window_scheme)
        test_data, test_target = sliding_window(test_x, test_y.to_numpy(), args.window, args.stride, args.window_scheme)
        #data_test_sample_wise, target_test_sample_wise = sliding_window(test_x, test_y.to_numpy(), args.window, args.stride_test, args.window_scheme)
        #save data
        print(f"Size of train_set: {np.shape(train_data)}")
        print(f"Size of val_set: {np.shape(val_data)}")
        print(f"Size of test_set: {np.shape(test_data)}")
        print("saving files...")

        # Check if the directory exists
        utils.makedir(args.path_processed)
        np.savez_compressed(os.path.join(args.path_processed, "train.npz"), data=train_data, target=train_target)
        np.savez_compressed(os.path.join(args.path_processed, "val.npz"), data=val_data, target=val_target)
        np.savez_compressed(os.path.join(args.path_processed, "test.npz"), data=test_data, target=test_target)
        #np.savez_compressed(os.path.join(args.path_processed, "test_sample_wise.npz"), data=data_test_sample_wise, target=target_test_sample_wise)




    

def main():

    # get experiment arguments
    args, _, _ = get_args()
    preprocess_pipeline(args)



if __name__ == "__main__":
    main()