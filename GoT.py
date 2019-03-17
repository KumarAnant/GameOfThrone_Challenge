import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


## Calculate % missing values in dataframe
for cols in missing_cols:
    missing_summary.loc[cols, 'Data Type'] = got[cols].dtype
    missing_summary.loc[cols, '% Missing'] = (got[cols].isnull().sum() * 100 / len(got[cols])).round(1)    
    missing_summary.loc[cols, 'Unique Values'] = (len(got[cols].unique()))
    missing_summary.loc[cols, '% Unique values'] = (len(got[cols].unique()))/(len(got[cols].dropna()))*100
print(missing_summary)

## Title is 51% missing and 28% unique records, missing value wil be imputed with 'Unknown'
## and flag column will be created








# selection = ['Ser', 'Stonehelm', 'Prince']
# count = 0
# for count in range(0, got.shape[0]):
#     if got.loc[count, 'title'] not in selection or got.loc[count, 'title'] is 'NaN':
#         got.loc[count, 'title'] = 'OtherABC'
# print(got['title'])
