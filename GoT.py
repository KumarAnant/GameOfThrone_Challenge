import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


import os
os.chdir('E:/OneDrive/Hult/Machine Learning/Assignments/Assignment - 2')
got = pd.read_excel('Data/GOT_character_predictions.xlsx', index_col = 0)

## Get some basic information of the dtaframe
print(got.info())
print(got.describe())
print(got.columns)

# Check for missing values in the dataset
plt.figure(figsize=(40, 20))
sns.set(font_scale=2)
sns.heatmap(got.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.set_style('whitegrid')
plt.show()
## Interpretation of the above graph: Missing values in following columns
                        # title
                        # culture
                        # dateOfBirth
                        # mother
                        # father
                        # heir
                        # house
                        # spouse
                        # isAliveMother
                        # isAliveFather
                        # isAliveHeir
                        # isAliveSpouse
                        # age
missing_cols = ['title', 'culture','dateOfBirth', 'mother',
                'father', 'heir','house', 'spouse', 'isAliveMother',
                'isAliveFather', 'isAliveHeir','isAliveSpouse', 'age']

# Create a blank dataframe for missing summary data
data = np.array([np.arange(len(missing_cols))]*4).T
missing_summary = pd.DataFrame(data, columns=['Data Type', '% Missing', 'Unique Values', '% Unique values'], index=missing_cols)

## Create a missing value columnn for flag
for col in got:
    if got[col].isnull().any():
        got['m_'+col] = got[col].isnull().astype(int)



## Calculate % missing values in dataframe
for cols in missing_cols:
    missing_summary.loc[cols, 'Data Type'] = got[cols].dtype
    missing_summary.loc[cols, '% Missing'] = (got[cols].isnull().sum() * 100 / len(got[cols])).round(1)    
    missing_summary.loc[cols, 'Unique Values'] = (len(got[cols].unique()))
    missing_summary.loc[cols, '% Unique values'] = int((len(got[cols].unique()))/(len(got[cols].dropna()))*100)
print(missing_summary)

## Create a variable for book number
got['bookNo'] = 10000*got['book1_A_Game_Of_Thrones'] + 1000*got['book2_A_Clash_Of_Kings'] + 100*got['book3_A_Storm_Of_Swords'] + 10*got['book4_A_Feast_For_Crows'] + got['book5_A_Dance_with_Dragons']
sns.boxplot(hue = 'bookNo', y = 'age', date = got)


## Function ot check if a value passed is null to set it as "Unknown" in dataframe
def checkNUll(val):
    if val == val:
        return val
    else:
        return "Unknown"

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#*********** Feature Engineering *********
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

#List of columns to be dropped before model fitting
drop_cols = []

####### title ###########
## Check distribution of unique culture
print(got['title'].value_counts())

## column title is 51% missing with 28% unique records, missing 
## value wil be replaced with 'Unknown'
got['title'] = got['title'].apply(lambda X: checkNUll(X))
print(got['title'].isnull().any())


## column culture is 65% missing with 9% unique records, 
## missing value can replaced with 'Unknown' tag
got['culture'] = got['culture'].apply(lambda X: checkNUll(X))
print(got['culture'].isnull().any())

## DateOfBirth being overall 77% missing and also that another
## column of Age is already available, the BirthDay column can be dropped
drop_cols.append('dateOfBirth')

## column mother is 85% missing and it is a qualitative datta
## can not be imputed, hence be dropped
drop_cols.append('mother')

## column father is 80% missing and it is a qualitative datta
## can not be imputed, hence be dropped
drop_cols.append('father')

## column heir is almost 100% missing and it is a qualitative datta
## can not be imputed, hence be dropped
drop_cols.append('heir')



# selection = ['Ser', 'Stonehelm', 'Prince']
# count = 0
# for count in range(0, got.shape[0]):
#     if got.loc[count, 'title'] not in selection or got.loc[count, 'title'] is 'NaN':
#         got.loc[count, 'title'] = 'OtherABC'
# print(got['title'])
