
import pandas as pd
from  utils.utils_augment import augment_data
import os

__all__ = ["get_pamap2_columns", "read_pamap2"]

def get_pamap2_columns():
    colNames = ["timestamp", "activityID","heartrate"]

    IMUwrist = ['wristTemperature', 
            'wrist X accel', 'wrist Y accel', 'wrist Z accel', 
            'wristAcc6_1', 'wristAcc6_2', 'wristAcc6_3', 
            'wrist X gyro', 'wrist Y gyro', 'wrist Z gyro', 
            'wristMagne1', 'wristMagne2', 'wristMagne3',
            'wristOrientation1', 'wristOrientation2', 'wristOrientation3', 'wristOrientation4']

    IMUchest = ['chestTemperature', 
            'chest X accel', 'chest Y accel', 'chest Z accel', 
            'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3', 
            'chest X gyro', 'chest Y gyro', 'chest Z gyro', 
            'chestMagne1', 'chestMagne2', 'chestMagne3',
            'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4']

    IMUankle = ['ankleTemperature', 
            'ankle X accel', 'ankle Y accel', 'ankle Z accel', 
            'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3', 
            'ankle X gyro', 'ankle Y gyro', 'ankle Z gyro', 
            'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
            'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']

    columns = colNames + IMUwrist + IMUchest + IMUankle  #all columns in one list
    colums_reordered = colNames + IMUchest + IMUwrist + IMUankle
    return columns, colums_reordered


def create_pamap2_dataframes(args, file, augmentation=False):
    df = pd.read_table(file, header=None, sep='\s+')
    columns, colums_reordered = get_pamap2_columns()
    df.columns = columns
    df = df[colums_reordered]
    df.drop(columns=[x for x in df.columns if "Mag" in x], inplace=True) #remove mag data
    df.drop(columns=[x for x in df.columns if "Acc6" in x], inplace=True) #remove data from thw worse accelerometer data
    df.drop(columns=[x for x in df.columns if "Temp" in x], inplace=True) #remove temperature
    df.drop(columns=[x for x in df.columns if "Orientation" in x], inplace=True) #remove orientation data
    df = df.drop(df[df.activityID == 0].index) #removal of any row of activity 0 as it is transient activity which it is not used
    df.drop(columns=["heartrate", "timestamp"], inplace=True) #remove heartrate and timestamp as it is not used
    df.apply(pd.to_numeric) #convert all values to numeric values
    id_map = dict([(1,0), (2,1), (3,2), (17,3), (16,4), (12,5), (13,6), (4,7), (7,8), (6,9), (5,10), (24,11)])
    df["activityID"] = df["activityID"].map(lambda x: id_map[x]) #map activitiy id to a more logical number series :)
    activityID = df["activityID"]
    activityID = activityID[::args.downsample] #seperate activityID downsampling from the data to avoid values being casted as floats
    df.reset_index(drop = True, inplace = True)
    df.drop(columns=["activityID"], inplace = True)
    activityID.reset_index(drop = True, inplace = True)
    df = df.rolling(window=args.downsample, step=args.downsample, min_periods=1).mean().reset_index(drop=True)   #downsample 
    df["synthetic"] = 0
    df["augmented"] = 0
    df["activityID"] = activityID
    if augmentation:
        augmentations = augment_data(args, df, 5) #create data augmentations as noisy copies of original
        df = pd.concat(augmentations, ignore_index=True)
    print(f"Number of samples: {len(df)}")
    data_x = df.iloc[:,:-1]
    data_y = df.iloc[:,-1]
    return [data_x, data_y]


def read_pamap2(args):
    participants = {}
    for entry in os.scandir(args.path_raw): #every participant
        subject_name = os.path.basename(entry).split('.')[0]
        if(subject_name not in args.participants):
            continue
        print(f"Reading data from: {subject_name}")
        #append dataframes to list of all participants
        participants[subject_name] = create_pamap2_dataframes(args, entry, augmentation=True if args.noise_augmentation or args.warp_augmentation else False) 
    return participants