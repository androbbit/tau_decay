
# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

from hep_ml.gradientboosting import UGradientBoostingClassifier
from hep_ml.losses import BinFlatnessLossFunction

#sklearn train/test split function
from sklearn.model_selection import train_test_split

# File system manangement
import os
import math

# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

#def accuracy_fn(y_pred, y_val):
#    true_positive = (y_pred == y_val).sum()
 #   true = y_val.sum()
 #   print(true_positive/true)
#    return true_positive/true



def add_features(df):
    # features used by the others on Kaggle
    df['NEW_FD_SUMP']=df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])
    df['NEW5_lt']=df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3
    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    df['flight_dist_sig2'] = (df['FlightDistance']/df['FlightDistanceError'])**2
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    df['NEW_IP_dira'] = df['IP']*df['dira']
    df['p0p2_ip_ratio']=df['IP']/df['IP_p0p2']
    df['p1p2_ip_ratio']=df['IP']/df['IP_p1p2']
    df['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)
    df['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
    df['iso_min'] = df.loc[:, ['isolationa', 'isolationb', 'isolationc','isolationd', 'isolatione', 'isolationf']].min(axis=1)
    df['NEW_iso_abc'] = df['isolationa']*df['isolationb']*df['isolationc']
    df['NEW_iso_def'] = df['isolationd']*df['isolatione']*df['isolationf']
    df['NEW_pN_IP'] = df['p0_IP']+df['p1_IP']+df['p2_IP']
    df['NEW_pN_p']  = df['p0_p']+df['p1_p']+df['p2_p']
    df['NEW_IP_pNpN'] = df['IP_p0p2']*df['IP_p1p2']
    df['NEW_pN_IPSig'] = df['p0_IPSig']+df['p1_IPSig']+df['p2_IPSig']
    df['NEW_FD_LT']=df['FlightDistance']/df['LifeTime']
    return df



df = pd.read_csv('data/training.csv')

print("Add features")
df = add_features(df)


print("Eliminate features")
features_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal',
              'SPDhits','CDF1', 'CDF2', 'CDF3',
              'isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt',
              'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta',
              'isolationa', 'isolationb', 'isolationc', 'isolationd', 'isolatione', 'isolationf',
              'p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT',
              'p0_IP', 'p1_IP', 'p2_IP',
              'IP_p0p2', 'IP_p1p2',
              'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof',
              'p0_IPSig', 'p1_IPSig', 'p2_IPSig',
              'DOCAone', 'DOCAtwo', 'DOCAthree']

features = list(f for f in df.columns if f not in features_out)


print("Split train/test")
train, test = train_test_split(df,test_size = 0.33)

X_train = train[features]
y_train = train['signal']

X_val = test[features]
y_val = test['signal']


loss = BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0 , fl_coefficient=15, power=2)
ugbc = UGradientBoostingClassifier(loss=loss, n_estimators=550,
                                 max_depth=6,
                                 learning_rate=0.15,
                                 train_features=features,
                                 subsample=0.7,
                                 random_state=123)
ugbc.fit(train[features+['mass']], train['signal'])
pred_raw = ugbc.predict(test[features])
#print(pred_raw)
pred = pd.DataFrame(data={'signal':pred_raw})
#print(pred.head(5))
#accuracy_fn(pred,y_val)
#print(pred_raw.sum())
print(((pred_raw == y_val) & y_val ).sum())
print(y_val.sum())
#print(pred['signal'].sum())
#print((pred['signal']==test['signal']).sum())
