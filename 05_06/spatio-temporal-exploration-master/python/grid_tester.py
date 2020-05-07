###
##Start with the noraml model training as in 5_Machine_Learning.ipynb
###

#Import the tools for machine learning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#read in data for machine learning training and testing
all_data_in_file = pd.read_csv('testcase_1_deposit_feature_and_label.csv')
labels = all_data_in_file.iloc[:,-1]
data = all_data_in_file.iloc[:,:-1]

#choose classifier
classifier = 'RFC' #random forest
#classifier = 'SVC' #support vector Classification 

#preprocess features and split data for training and testing
data = preprocessing.scale(data)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=1)

if classifier == 'RFC':
    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    #n_estimators use between 64-128 doi: 10.1007/978-3-642-31537-4_13
    clf = RandomForestClassifier(n_estimators=128, n_jobs=1,class_weight=None)
elif classifier == 'SVC':
    clf = SVC(probability=True,class_weight=None, gamma='auto')


###Then Read in the testing points 
#Read in the data that you have coregistered
MLgrid = numpy.loadtxt("./coreg_output/0_vector_subStats.out",delimiter=',')

#Remove the rows that failed to coregister
MLgrid=MLgrid[~np.isnan(MLgrid).any(axis=1)] 
MLgrid=MLgrid[~(MLgrid==0).all(axis=1)] 

#Apply the trained ML to our gridded data to determine the probabilities at each of the points
pRF=numpy.array(clf.predict_proba(preprocessing.scale(MLgrid[:,params])))

print("Done Testing target point!")