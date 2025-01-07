## **Abstract:**
We investigate personalised, combined biomechanical dynamics models and human surface models to synthesise IMU sensor time series data and to improve Human Activity Recognition (HAR) model performance for activities of daily living (ADLs). We analyse two model training scenarios: (1) data fusion of synthetic and measurement IMU data to train HAR models directly, and (2) pretraining HAR models with synthetic and measured IMU data and subsequent transfer learning with public benchmark datasets. Furthermore, we analyse how the synthetic IMU data helps in configurations with scarce measurement data, by limiting the number of participants and IMUs in both training scenarios. We evaluate three state-of-the-art HAR models to determine the benefit of our approach. Depending on the HAR model, synthetic data increased the macroF1 score on average by $8$\% for configurations with reduced data and by up to 7.5% for transfer learning. In the transfer learning scenario, combining synthetic data with measurement data during pretraining outperformed the results obtained by pretraining with measurement data only, by an average of 5.2% across the public datasets. Our results show that the IMU data synthesis approach improves performance across all HAR models in both training scenarios. Performance improvements exceeded those of noise augmentation and measurement data only. The largest performance improvements were found when original measurement data was scarce. We conclude that augmenting HAR models with synthetic IMU data obtained from the combination of biomechanical dynamics models and human surface models provides clear performance gains for HAR and a versatile approach to accurately reflect actual human movements.
![alt text](https://github.com/intelligentembeddedsystemslab/synthetic-data-augmentation-in-HAR/blob/main/SynHAR_methods.png?raw=true)
FIGURE 1: Method overview. Personalised, biomechanically validated human surface models were created
and various inertial measurement unit (IMU) sensors were attached. The models were co-simulated to
synthesise acceleration and gyroscope data. Synthesised data were used to augment deep-learning models
in two model training scenarios: (1) direct data fusion with measurement data (MoSurf), and (2) pretraining
with synthetic data before transfer learning.

## **Setup** ##

### Requirements ###
Please check the enviroments.yml for required packages. You can create a corresponding Conda environment as follows:
```
conda env create -f environment.yml
```

### Datasets ###
We use the MoSurf (not (yet) publicly available), [RealWorld](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/) and [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) dataset.
Check section **B. DATASETS** in our paper for more information about what data was used. In addition to that, the axes of the RealWorld dataset were adjusted to match the synthetic data using utils/RealWorld_change_axis.ipynb.

### Checkpoints ###
The checkpoints obtained by pretraining on the MoSurf dataset with and without synthetic data can be downloaded [here](https://drive.google.com/file/d/1ToUin5PjPGel0LJj4JZ7_xJh7AxGGgwJ/view?usp=drive_link).
Select the checkpoints you want and insert them in the corresponding location, e.g.:
```
/synthetic-data-augmentation-in-HAR/model/models/Finetune/AttendDiscriminate/realworld/checkpoint_synthetic_realworld.pth
```
## **How to run** ##

To train a model with the corresponding HAR dataset, run:
```
cd model/
python main.py --dataset [pamap2,realworld,mosurf] --model [DeepConvLSTM, AttendDiscriminate, TransformHAR] --train_mode
```
This will start a LOPO evaluation of the respective dataset with the chosen model architecture.
After successful evaluation results will be saved in the model/saved_data folder. Use the --experiment argument to change the name under which the results should be saved.

To finetune the checkpoints that were pre-trained on the MoSurf dataset, run the training command with the --finetune argument:
```
python main.py --dataset [pamap2,realworld,mosurf] --model [DeepConvLSTM, AttendDiscriminate, TransformHAR] --train_mode --pretrain
```

All arguments of the main.py can be viewed by executing:
```
python main.py -h

usage: main.py [-h] [--experiment EXPERIMENT] [--train_mode] [--dataset {mosurf,realworld,pamap2}] [--synthetic_data] [--noise_augmentation] [--warp_augmentation]
               [--vertices {har5,har9,har9_2,har9RW,har9PAMAP}] [--activities {all,standard8,standard7,splitted10}] [--finetune] [--synthetic_training]
               [--validation {LOPO,Holdout}] [--imus {all,mars3,blub3,watch+phone2,realworld7,blub10,blub5,blub2,blub7,pamap2_3}] [--window WINDOW]
               [--stride STRIDE] [--window_scheme {last,max}] [--stride_test STRIDE_TEST] [--num_participants {3,5,6,7,9,11,12,15,19}]
               [--model {AttendDiscriminate,DeepConvLSTM,TransformHAR}] [--epochs EPOCHS] [--load_epoch LOAD_EPOCH] [--downsample DOWNSAMPLE]
               [--lowpass_cutoff LOWPASS_CUTOFF] [--scaler {Standard,MinMax,MaxAbs,Robust,Normalizer,Quantile,Power}] [--mixup] [--center_loss]
               [--batch_size BATCH_SIZE] [--device DEVICE] [--print_freq PRINT_FREQ] [--num_loops NUM_LOOPS]

HAR dataset, model and optimization arguments. Most options are only available for the mosurf dataset

options:
  -h, --help            show this help message and exit
  --experiment EXPERIMENT
                        experiment name (default: None)
  --train_mode          execute code in training mode (default: False)
  --dataset {mosurf,realworld,pamap2}
                        HAR dataset (default: mosurf)
  --synthetic_data      use the synthetic data from MoCap as augmentation, requires the data (not published yet) (default: False)
  --noise_augmentation  use noisy variations of the training data as augmentation (default: False)
  --warp_augmentation   use over/undersampled variations of the training data as augmentation (default: False)
  --vertices {har5,har9,har9_2,har9RW,har9PAMAP}
                        set of vertices for synthetic data augmentation (default: har5)
  --activities {all,standard8,standard7,splitted10}
                        list/number of activities to classify, only valid for mosurf dataset (default: standard8)
  --finetune            instead of choosing model and initialize weights, use stored model(see path) and finetune (default: False)
  --synthetic_training  removes all non-synthetic data from the training set (default: False)
  --validation {LOPO,Holdout}
                        validation strategy used for the model (default: LOPO)
  --imus {all,mars3,blub3,watch+phone2,realworld7,blub10,blub5,blub2,blub7,pamap2_3}
                        number and location of imus used in the model, only valid for mosurf dataset (default: mars3)
  --window WINDOW       sliding window size (default: 120)
  --stride STRIDE       sliding window stride (default: 60)
  --window_scheme {last,max}
                        scheme to determine label of a window (default: max)
  --stride_test STRIDE_TEST
                        set to 1 for sample-wise prediction (default: 1)
  --num_participants {3,5,6,7,9,11,12,15,19}
                        number of participants used from dataset (default: 19)
  --model {AttendDiscriminate,DeepConvLSTM,TransformHAR}
                        HAR architecture (default: DeepConvLSTM)
  --epochs EPOCHS       number of training epochs (default: 200)
  --load_epoch LOAD_EPOCH
                        epoch to resume training (default: 0)
  --downsample DOWNSAMPLE
                        factor for downsampling data(MoSurf dataset is recorded at 100Hz) (default: 2)
  --lowpass_cutoff LOWPASS_CUTOFF
                        cutoff frequency of lowpass filter used on dataset (default: 10)
  --scaler {Standard,MinMax,MaxAbs,Robust,Normalizer,Quantile,Power}
                        sklearn scaler to use on data, wil be fitted only on training data and applied to train + val + test (default: Standard)
  --mixup               use mixup data augmentation (default: False)
  --center_loss         use center loss criterium (default: False)
  --batch_size BATCH_SIZE
                        batch size for training (default: 256)
  --device DEVICE       Index of CUDA device to train on, leave empty for automatic assignment (default: None)
  --print_freq PRINT_FREQ
  --num_loops NUM_LOOPS
                        number of training loops when using holdout validation (default: 1)
```
Please be aware, that not all arguments may be compatible with each other.

## **Citation**

If you find our work useful in your research, please consider citing:
```
@article{SynHAR,
  title = {SynHAR: Augmenting Human Activity Recognition with Synthetic Inertial Sensor Data Generated from Human Surface Models},
  author = {Uhlenberg, Lena and Ole Haeusler, Lars and Amft, Oliver},
  date = {2024},
  journaltitle = {IEEE Access},
  volume = {12},
  pages = {194839--194858},
  doi = {10.1109/ACCESS.2024.3513477}
}
```

