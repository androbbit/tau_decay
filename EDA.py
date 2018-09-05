#This script produces plots of features separated in signal/background to explore the data


# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# File system manangement
import os

# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

#Read the data file
df = pd.read_csv('data/training.csv')

features = df.keys()
#features = features.delete(-3) #delete Label
features = features.delete(0) #delete Event Id


print(df.value_counts())

for feature in features:
    if (feature == 'signal') | (feature == 'production'): continue
    sns.kdeplot(df.loc[df['signal'] == 1 , feature], label = 'Signal')
    sns.kdeplot(df.loc[df['signal'] == 0 , feature], label = 'Background')
    plt.xlabel(feature)
    plt.ylabel('Density')
    figTitle = 'plots/KDE/'+feature+'.png'
    plt.savefig(figTitle)
    plt.close()

