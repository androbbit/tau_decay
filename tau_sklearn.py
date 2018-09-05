
# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

#sklearn boosted tree classifiers
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#sklearn train/test split function
from sklearn.model_selection import train_test_split

# File system manangement
import os
import math

# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns


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

clf = GradientBoostingClassifier(n_estimators = 550, learning_rate=0.15)
clf.fit(X_train, y_train)
result = clf.predict(X_val)

print(((result == y_val) & y_val ).sum())
print(y_val.sum())


#Define lists to save results for later-plotting
score_ne_cv = np.zeros(10)
score_ne_self = np.zeros(10)

n_estimators = range(100,1100,100)




#Vary number of estimators from 10 to 200
for i in range(10):
    clf = GradientBoostingClassifier(n_estimators=(i+1)*100, learning_rate=0.15)
    clf.fit(X_train, y_train)
    result_self = clf.predict(X_train)
    result_cv = clf.predict(X_val)

    score_ne_self[i] = clf.score(X_train,y_train)
    score_ne_cv[i] = clf.score(X_val,y_val)


#Plot socres as a function of the number of estimators
plt.plot(n_estimators, score_ne_cv, label='Test Data')
plt.plot(n_estimators, score_ne_self,  label='Training Data')
plt.xlabel('Number of Estimators')
plt.ylabel('Score')
plt.legend(loc='best')
plt.savefig('score_vs_nestimators.png')
plt.close()



