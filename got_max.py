# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:00 2019

@author: Max Li
Purpose: Predicting isAlive on Game of Thrones dataset

"""

####################################################
# 1) Initial Data Set-up
####################################################



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score # k-folds cross validation


got = pd.read_excel('Data/GOT_character_predictions.xlsx', index_col = 0)
got.sort_index(inplace=True)

### Replace some of the mis-spelled entries in culture columns
got['culture'].replace(['ironmen'], ['ironborn'], inplace = True)
got['culture'].replace(['braavos'], ['braavosi'], inplace = True)
got['culture'].replace(['dornishmen', 'dorne'], 'dornish', inplace = True)
got['culture'].replace(['ghiscaricari'], ['ghiscari'], inplace = True)
got['culture'].replace(['vale'], ['valemen'], inplace = True)
got['culture'].replace(['riverlands'], ['rivermen'], inplace = True)
got['culture'].replace(['the reach', 'reachmen'], 'reach', inplace = True)
got['culture'].replace(['westerman'], ['westermen'], inplace = True)


## Get some basic information of the dtaframe
print(got.info())
print(got.describe())
print(got.columns)

### Missing values

missing_cols = ['title', 'culture','dateOfBirth', 'mother',
                'father', 'heir','house', 'spouse', 'isAliveMother',
                'isAliveFather', 'isAliveHeir','isAliveSpouse', 'age']

## Create a missing value columnn for flag
for col in got:
    if got[col].isnull().any():
        got['m_'+col] = got[col].isnull().astype(int)

got.columns


################# Feature Engineering ###################


#List of columns to be dropped before model fitting
drop_cols = []
## Drop name column for unconsistent data and lots of unique values
got.drop('name', axis = 1, inplace = True)

####### title ###########
got['title'] = got['title'].fillna("Unknown")
got['title'].isnull().any()

####### culture ###########
got['culture'] = got['culture'].fillna("Unknown")
got['culture'].isnull().any()
## DateOfBirth
# Drop date of birth for missing values 

got.drop('dateOfBirth', axis = 1, inplace = True)

## Drop 'mother' for many missing value
got.drop('mother', axis = 1, inplace = True)
## Drop 'father' for many missing value
got.drop('father', axis = 1, inplace = True)
## Drop 'heir' for many missing value
got.drop('heir', axis = 1, inplace = True)
## Drop 'spouse' for many missing value
got.drop('spouse', axis = 1, inplace = True)

####### house ###########
got['house'] = got['house'].fillna("Unknown")


## Delte isAliveMother for lots of missing values
got.drop('isAliveMother', axis = 1, inplace = True)

## Delte isAliveFather for lots of missing values
got.drop('isAliveFather', axis = 1, inplace = True)

## Delte isAliveHeir for lots of missing values
got.drop('isAliveHeir', axis = 1, inplace = True)

## Delte isAliveSpouse for lots of missing values
got.drop('isAliveSpouse', axis = 1, inplace = True)

## Found two age for rows 1685 and 1689 wiht values --277980 and -298001
## shall impute median age at the abnove two values and all missing data
## replace negatiev age with NaN
got.loc[got.index[got['age'] == -277980].tolist()[0], 'age'] = np.nan
got.loc[got.index[got['age'] == -298001].tolist()[0], 'age'] = np.nan

## Replace all NaN with median agea
birth_fill = got['age'].median()

##Fill in NA in age column with median
got['age'] = got['age'].fillna(birth_fill)

got.isnull().any().any()
## Set outlier flag for Age variable
# print(got['age'].quantile([0.05, 0.93]))
# print(got['age'].max())
# print(got['age'].min())

got['o_age'] = 0
age_up_outlier = 98
age_low_outlier = 8

## Impute age based on first book apapearance era
for count in range(1, got.shape[0]+1):    
    if got.loc[count, 'age'] < age_low_outlier:
        got.loc[count, 'o_age'] = -1
    elif got.loc[count, 'age'] > age_up_outlier:
        got.loc[count, 'o_age'] = 1
got['o_age']

got.isnull().any().any()

## Set outlier flag for numDeadRelations
got['o_numDeadRelations'] = 0
numDeadRelations_up_outlier = 7
numDeadRelations_low_outlier = 0
## Impute age based on first book apapearance era
for count in range(1, got.shape[0]+1):    
    if got.loc[count, 'numDeadRelations'] < numDeadRelations_low_outlier:
        got.loc[count, 'o_numDeadRelations'] = -1
    elif got.loc[count, 'numDeadRelations'] > numDeadRelations_up_outlier:
        got.loc[count, 'o_numDeadRelations'] = 1
got.isnull().any().any()

## Set outlier flag for popularity
# print(got['popularity'].quantile([0.10, 0.98]))
got['o_popularity'] = 0
popularity_up_outlier = 0.71
popularity_low_outlier = 0.0067
for count in range(1, got.shape[0]+1):    
    if got.loc[count, 'popularity'] < popularity_low_outlier:
        got.loc[count, 'o_popularity'] = -1
    elif got.loc[count, 'popularity'] > popularity_up_outlier:
        got.loc[count, 'o_popularity'] = 1

got.isnull().any().any()
# Cerate a copy of imputed value dataframe
got_OLS = got.copy(deep = True)

## Create dummy variables for title column data, drop original column
title_dummies = pd.get_dummies(list(got_OLS['title']), prefix = 'title', drop_first = True)
title_dummies.index = np.arange(1, len(title_dummies)+1)
got_OLS.drop('title', axis = 1, inplace = True)


## Create dummy variables for culture column data, drop original column
culture_dummies = pd.get_dummies(list(got_OLS['culture']), prefix = 'culture', drop_first = True)
culture_dummies.index = np.arange(1, len(culture_dummies)+1)
got_OLS.drop('culture', axis = 1, inplace = True)

## Create dummy variables for house column data, drop original column
house_dummies = pd.get_dummies(list(got_OLS['house']), prefix = 'house', drop_first = True)
house_dummies.index = np.arange(1, len(house_dummies)+1)
got_OLS.drop('house', axis = 1, inplace = True)

## Concatenate dummy variables
got_OLS = pd.concat(
        [got_OLS.loc[:,:],
            title_dummies,
            culture_dummies,
            house_dummies
         ], axis = 1)
got_OLS.isnull().any()
got_OLS.isnull().any().any()

## Seperate predictor and dependent variables
X = got_OLS.drop('isAlive', axis = 1)
y = got_OLS['isAlive']


## Train and test data split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, 
                                                    test_size = 0.1,
                                                    random_state = 508)
lm = LogisticRegression()
lm.fit(X_train, y_train)

## Predict for the train set data
predictions = lm.predict(X_test)

# pd.DataFrame(lm.coef_, index = X.columns)
## Prediction report
print("Classification report: \n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

######### Cross Validation ################

cv_lr_3 = cross_val_score(lm,
                          X,
                          y,
                          cv = 3)

print("Cross validation score of LR: ", (pd.np.mean(cv_lr_3).round(3)))



##################################
####### Random Forest ############
##################################

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pydot

## Creating tree model
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

## Predictions using tree model
predictions = dtree.predict(X_test)

## Tree model accuracy
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

cv_DTree_3 = cross_val_score(dtree,
                          X,
                          y,
                          cv = 3)

print("Cross validation score of RFC: ", (pd.np.mean(cv_DTree_3)).round(3))


## Random forest model
rfc = RandomForestClassifier(n_estimators=200)
## Fit random forest model
rfc.fit(X_train, y_train)

## Prediction of Random Forest model
predictions = rfc.predict(X_test)
df = pd.DataFrame(predictions)
df.to_excel("result_RFC.xlsx")
## Accuracy oif random forest
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

######### Cross Validation ################
cv_RandFor_3 = cross_val_score(rfc,
                          X,
                          y,
                          cv = 3)

print("Cross validation score of RFC: ", (pd.np.mean(cv_RandFor_3)).round(3))
