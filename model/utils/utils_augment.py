
import pandas as pd
import numpy as np
import random

__all__ = ["add_noise", "add_warped", "augment_data"]


def add_noise(df, acc_noise_var, gyro_noise_var):
    """
        Simple noise augmentation implementation from https://github.com/DavideBuffelli/TrASenD/blob/master/
    """
    for col in df.columns[:-2]:
        if 'Accel' in col:
            df[col] = df[col].add(np.random.normal(0, df[col].std() * acc_noise_var, df.shape[0]))
        elif 'Gyro' in col:
            df[col] = df[col].add(np.random.normal(0, df[col].std() * gyro_noise_var, df.shape[0]))
    df['augmented'] = 1
    return df
    
def add_warped(args, df):
    """
        Simple warp augmentation that over or undersamples the data
    """
    #reset index
    df.reset_index(inplace=True)
    df.insert(0, 'TimeStamp', pd.to_datetime('now').replace(microsecond=0)) #add timestamp for resampling
    start_time =  df["TimeStamp"].iloc[0]
    start_time = df["TimeStamp"].iloc[0]
    end_time = start_time + pd.to_timedelta(len(df) * args.downsample * 10, unit='ms') # add timestamps depending on sample frequency
    df["TimeStamp"] = pd.date_range(start_time, end_time, len(df))
    df = df.set_index('TimeStamp')
    ms = random.randint(10,40) #random resample frequency between half and double of original one
    resampled_df = df.resample(f"{ms}L").mean().interpolate() 
    resampled_df.reset_index(inplace=True)
    resampled_df.drop(columns= [ "TimeStamp", "index"], inplace=True)
    resampled_df['augmented'] = 1
    return resampled_df

    
def augment_data(args, data_frame, num_augmentations, acc_noise_var=0.5, gyro_noise_var=0.2):
    augmentations = [data_frame]
    for _ in range(num_augmentations):
        df_copy = data_frame.copy()
        if args.noise_augmentation:
            augmentations.append(add_noise(df_copy, acc_noise_var, gyro_noise_var))
        if args.warp_augmentation:
            augmentations.append(add_warped(args, df_copy))
    return augmentations