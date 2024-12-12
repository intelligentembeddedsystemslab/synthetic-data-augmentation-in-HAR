
import pandas as pd
from  utils.utils_augment import augment_data
import os
import glob

__all__ = ["read_mosurf", "create_mosurf_dataframes"]


#create one large dataset out of all data for one participant
#create activity ids and add to dataset, getting_up_floor = 1, jumping = 2 ...
def create_mosurf_dataframes(args, files, synthetic=False, augmentation=False):
    data_frames = []
    for idx, activity in enumerate(args.activity_list):
        for path in files:
            if os.path.basename(path) != activity:
                continue
            
            if not synthetic:
                df = pd.read_csv(path, skiprows=4)
                df.drop(df[df["Noraxon MyoMotion.Sync,On"] == 0].index, inplace = True) #remove non relevant data(no sync for activity)
                df = df.drop(columns=["Time,s", "Noraxon MyoMotion.Sync,On"]) #drop unnecessary columns before activity starts
                df.drop(columns=[x for x in df.columns if "Mag" in x], inplace=True)#remove Mag values
                df = df.rolling(window=args.downsample, step=args.downsample, min_periods=1).mean().reset_index(drop=True)   #downsample as specified
                df["synthetic"] = 0
                df["augmented"] = 0
                if augmentation:
                    augmentations = augment_data(args, df, 5) #create data augmentations as noisy copies of original
                    df = pd.concat(augmentations, ignore_index=True)
            else:
                df = pd.read_csv(path) #synthetic data has no time or Mag colum and no header like the files for the measured data
                df = df.rolling(window=args.downsample, step=args.downsample, min_periods=1).mean().reset_index(drop=True)   #downsample as specified
                df["synthetic"] = 1
                df["augmented"] = 0
            df["activityID"] = idx #add activity id for whole file
            data_frames.append(df)
           
        
    dataset_frame = pd.concat(data_frames, ignore_index=True)
    
    data_x = dataset_frame.iloc[:,:-1]
    data_y = dataset_frame.iloc[:,-1]
    return [data_x, data_y]

def read_mosurf(args):
    participants = {}
    for entry in os.scandir(args.path_raw): #every participant
        if(os.path.basename(entry) not in args.participants):
            continue
        valid_paths = []
        subfolder = os.path.join(entry, "IMU_measured")
        all_files = glob.glob(os.path.join(subfolder, "*.csv")) #collect all csv files
        for file in all_files:
            if os.path.basename(file) in args.activity_list:
                valid_paths.append(file) #save path if activity matches specified ones given as argument
        print(f"Number of valid paths: {len(valid_paths)}")
        #append dataframes to list of all participants
        participants[os.path.basename(entry)] = create_mosurf_dataframes(args, valid_paths, augmentation=True if args.noise_augmentation or args.warp_augmentation else False)
    return participants

