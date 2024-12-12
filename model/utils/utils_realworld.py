import pandas as pd
from  utils.utils_augment import augment_data
import os
import glob
import numpy as np

__all__ = ["read_realworld"]


def resample_realword(df):
    #needs timestamp as index of df to work
    resampled_df = df.resample("20ms").mean().interpolate()#resample to 50Hz
    resampled_df.reset_index(inplace=True)
    return resampled_df

def reindex_dataframe(df):
    #redo the timestamp without weird jumps/breaks that dont fit to the data
    df.reset_index(inplace=True)
    df["attr_time"] = df["attr_time"].astype(np.int64) // 10**6 # convert timestamp in ms
    start_time =  df["attr_time"].iloc[0]
    end_time =  df["attr_time"].iloc[-1]
    df["attr_time"] = np.linspace(start_time, end_time, len(df), dtype = int)

    df['attr_time'] = pd.to_datetime(df['attr_time'], unit='ms') #convert back into timestamp
    df.set_index('attr_time', inplace=True)
    return df


def combine_realworld_sensors(args, files, participant): 
    #basically the create_dataframes for the realworld dataset while handling all the problems from the smartphone imus
    data_frames = []
    if participant in ["proband4", "proband7", "proband14"]:
        #these participant have the stair climbing activities split and the sensor are not synchronized so you cant just concat into 1 file
        activities = ["jumping", "lying", "running", "sitting", "standing", "walking", "climbing_down_1", "climbing_down_2", "climbing_down_3", "climbing_up_1", "climbing_up_2", "climbing_up_3"]
    else:
        activities = args.activity_list

    for idx, activity in enumerate(activities): #every activity
        imus = [] #dataframes from each imu for this activity
        for imu in args.imu_list: #already check this here so the columns in the dataframe are always in the same order
            acc = pd.DataFrame
            gyro = pd.DataFrame
            for path in files:
                if imu not in path or activity.replace('_','') not in path.replace('_',''):
                    continue
                if "acc" in path:
                    acc = pd.read_csv(path)
                elif "Gyro" in path:
                    gyro = pd.read_csv(path)

            if  acc.empty or gyro.empty:
                print(f"Data not complete, skipping imu {imu} for activity {activity}")
                continue

            acc['attr_time'] = pd.to_datetime(acc['attr_time'], unit='ms')
            acc.set_index('attr_time', inplace=True)
            acc.drop(columns= [ "id"], inplace=True)
            acc = reindex_dataframe(acc) 

            gyro['attr_time'] = pd.to_datetime(gyro['attr_time'], unit='ms')
            gyro.set_index('attr_time', inplace=True)
            gyro.drop(columns= [ "id"], inplace=True)
            gyro = reindex_dataframe(gyro)

            acc.columns = [ f"{imu} X accel", f"{imu} Y accel", f"{imu} Z accel"]
            gyro.columns = [  f"{imu} X gyro", f"{imu} Y gyro", f"{imu} Z gyro"]

            combined = pd.concat([acc, gyro], axis=1)

            imus.append(combined)
        if len(imus) > 1:
            df = pd.concat(imus, axis=1)
            df = resample_realword(df)
            df.drop(columns= ["attr_time"], inplace=True)
            df["synthetic"] = 0
            df["augmented"] = 0

            # give splitted activities correct label
            if "climbing_down_" in activity:
                df["activityID"] = 6
            elif "climbing_up_" in activity:
                df["activityID"] = 7
            else:
                df["activityID"] = idx

            data_frames.append(df)
        else:
            print(f"No data for activity {activity} found")
    dataset_frame = pd.concat(data_frames, ignore_index=True)

    data_x = dataset_frame.iloc[:,:-1]
    data_y = dataset_frame.iloc[:,-1]
    return [data_x, data_y]


def read_realworld(args):
    participants = {}
    for entry in os.scandir(args.path_raw): #every participant
        if(entry.name not in args.participants):
            continue
        print(f"Reading data from: {entry.name}")
        valid_paths = []
        subfolder = os.path.join(entry, "data")
        valid_paths = glob.glob(os.path.join(subfolder, "*.csv")) #collect all csv files
        print(f"Number of valid paths for participant {entry.name}: {len(valid_paths)}")

        participants[os.path.basename(entry)] = combine_realworld_sensors(args, valid_paths, os.path.basename(entry)) #append dataframes to list of all Paricipants

    return participants