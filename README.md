# Deep Learning Method For Forest Fire Prediction

## Introduction

## Run
```bash
git clone https://github.com/lycheel1/Deep-Learning-Method-for-Forest-Fire-Prediction.git
```

Create a PyCharm Project by openning the clonned repository. Currently this project can only run under PyCharm refactoring tools. 

## Code Base Structure

### Data
Data preparation has already been done. To fetch the processed `.npy` files, run the following command.

These `.npy` files are 2D numpy arrays of geological data, and can be further categorized into two sub-groups: 2D and 1D. 2D represents the Rasterized featrues, where 1D represents Tebular Features.

2D Data consists of 5 channels (in order):
- Burned Areas of the Day
- Cumulative Burned Areas from day 0 to day i
- Fuel
- Elevation
- Fire Weather Index (fwi)

Whereas 1D Data consists of three channels (in order):
- Agency (e.g. national parks, provincial territories etc.)
- Temporal Feature
- Fire Cause

### Data Preparation
`data_prep` folder includes the original code of processing raw data.
All the processed data can be retrived via `TODO: add a database for our training data`. These data are in the format of `.npy`, which means that you can
read it and add transformation for furthur training tasks.

### Model Training
The main entry of Model Training is under `model_training/execution`, where a lists of different variation of model architectures are implemented. These implementations are mostly built upon `model_training/WF01_Trainer.py`, where the basic architecture of U-net is defined. You can directly run any of the training scripts, or implement your own archtecture by adding a new file.

### Saving Models and Performance Metrics

You can save the model via the following scripts:

```python
trainer1 = WF01_Trainer({your data}) # See WF01_Trainer init function for initialization parameters
trainer.K_fold_train_and_val(f'{save_path}/nn_save/kfold/',
                                  f'{network_name}{net_suffix}')
trainer1.save_metrics(f'{save_path}',
                        f'kfold_metrics{net_suffix}.csv')
```

Once you run the model training script, with the above lines at the end, you should see the saved models and model performance metrics in the corresponding
destinations.

## Experimentations

### MultiClass
Currently, our 2D channels only includes two burning related channels: current burning areas and cumulative burned areas, and we're only predicting the next-day cumulative burned area. We can incorporate more features into the data to potentially make the model more well-rounded.

During forest fires, if a area was burned and no longer burning, it means that this area will never be burned again since the "fuel" of this area for fire to occur has been used up. We can add this information implicitly into the model by adding one more feature of burnt out areas. That is, areas that were on fire but no longer burning. 

Another potential improvement is by adding another layer of prediction: next-day burn areas. That is, separating the next-day cumulative burning areas into new burnt areas and the rest cumulative burned area. 
