import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import roc_auc_score

def run(df):
    #print(sent)
    #get df already preprocessed
    #Split based on targets X,y
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #get list of models
    models = [SVC(),RandomForestClassifier(),XGBClassifier(),CategoricalNB()]
    for clf in models:
        #For each model
        #print model name
        print('#######################################')
        print(f'Model IS: {clf}')
        print('#######################################')
        #train
        clf.fit(X_train,y_train)
        preds = clf.predict(X_test)
        #test
        #print accuracy, aoc for both train and test
        print(f'Accuracy Scores ARE: {clf.score(X_test,y_test)} for test set AND {clf.score(X_train,y_train)} for train')
        print(f'ROC SCORES ARE: {roc_auc_score(y_test,preds,multi_class='ovr')} for test set')
        #confusion matrix plot
    print('THE END')

#df = pd.read_csv('Zenteiq/inputs/dataset.csv')
#run(df)
