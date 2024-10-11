# the idea of this script
# 1. select a target or multiple features from the validation_results.csv, and specify the range of epochs to compute the mean and std on
# 2. compute the mean of the selected features for all models under /checkpoints (the csv file is under /checkpoints/model_name/validation_results.csv)
# 3. compute the std similarly
# 4. plot the mean and std of the selected features for all models
# 5. save the plot to /checkpoints/mean_std_plot_{feature_name}.png
# 6. pretty print the mean and std of the selected features for all models

import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt

def compute_mean_std(feature_name, start_epoch, end_epoch, best_direction='min'):
    # get all the models
    models = os.listdir('checkpoints')
    models = [model for model in models if os.path.isdir(f'checkpoints/{model}')]

    # get the mean and std for the selected feature
    mean = []
    std = []
    for model in models:
        # read the validation_results.csv
        df = pd.read_csv(f'checkpoints/{model}/validation_results.csv')
        df = df[(df['epoch'] >= start_epoch) & (df['epoch'] <= end_epoch)]
        mean.append(df[feature_name].mean())
        std.append(df[feature_name].std())

    # plot the mean and std, and put the smallest in green if best_direction is min, otherwise put the largest in green
    mean = np.array(mean)
    std = np.array(std)
    if best_direction == 'min':
        best_idx = np.argmin(mean)
    else:
        best_idx = np.argmax(mean)
    
    # Set figure size to make y-axis larger
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(models, mean, std, fmt='o', color='b', label='mean')
    plt.errorbar(models[best_idx], mean[best_idx], std[best_idx], fmt='o', color='g', label='best')

    plt.xlabel('Model')
    plt.ylabel(feature_name)
    # smaller title font size
    plt.title(f'Mean and Std of {feature_name} from epoch {start_epoch} to epoch {end_epoch}', fontsize=10)
    plt.xticks(rotation=75)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'checkpoints/mean_std_plot_{feature_name}.png')
    plt.show()

    # pretty print the mean and std
    for i in range(len(models)):
        print(f'{models[i]}: mean={mean[i]}, std={std[i]}')
        
if __name__ == '__main__':
    feature_name = sys.argv[1]
    start_epoch = int(sys.argv[2])
    end_epoch = int(sys.argv[3])

    compute_mean_std(feature_name, start_epoch, end_epoch)

