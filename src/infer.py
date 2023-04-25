import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
) 

def run(df):
    target_dict = {
    'Dropout':0,
    'Graduate':1,
    }
    df['Target'] = df['Target'].map(target_dict)
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    clf = RandomForestClassifier()
    
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    
    report = pd.DataFrame.from_dict(classification_report(y_test, y_pred, output_dict = True))
    print(report)
    
    return disp

