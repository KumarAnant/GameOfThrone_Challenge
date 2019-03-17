import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


import os
os.chdir('E:/OneDrive/Hult/Machine Learning/Assignments/Assignment - 2')
got = pd.read_excel('Data/GOT_character_predictions.xlsx', index_col = 0)
got.sort_index(inplace=True)

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

## Create a variable for book number. If the character did not appear in any of the book, he/she must have
## appeared in book number 6
got['bookNo'] = got['book1_A_Game_Of_Thrones'] + \
                got['book2_A_Clash_Of_Kings'] + \
                got['book3_A_Storm_Of_Swords'] + \
                got['book4_A_Feast_For_Crows'] + \
                got['book5_A_Dance_with_Dragons']
got['bookNo'] = got['bookNo'].apply(lambda X: 6 if X == 0 else X)

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
## Check distribution of unique title
## column title is 51% missing with 28% unique records, missing 
## value wil be replaced with 'Unknown'

title_list = got['title'].value_counts()
min_acceptable_title_count = 10
## Anything less than the accepted count will be treated 
## as "Other" title, to reduct variability of the dataframe. 
## Save the transofmed data in column mod_title
got['mod_tile'] = " "
for count in range(1, got.shape[0]+1):
    if (got.loc[count, 'title'] != got.loc[count, 'title']):
        got.loc[count, 'mod_title'] = "Unknown"
        # print(count, got.loc[count, 'T_title'])
    elif (title_list.loc[got.loc[count, 'title']] < min_acceptable_title_count):
        got.loc[count, 'mod_title'] = "Other"
        # print(count, got.loc[count, 'T_title'])
    else:
        got.loc[count, 'mod_title'] = got.loc[count, 'title']
print(got['mod_title'].isnull().any())


## column culture is 65% missing with 9% unique records, 
## missing value can replaced with 'Unknown' tag
culture_list = got['culture'].value_counts()
min_acceptable_culture_count = 10
## Anything less than the accepted count (10) will be treated 
## as "Other" culture, to reduct variability of the dataframe. 
got['mod_culture'] = " "
for count in range(1, got.shape[0]+1):
    if (got.loc[count, 'culture'] != got.loc[count, 'culture']):
        got.loc[count, 'mod_culture'] = "Unknown"
        # print(count, got.loc[count, 'T_title'])
    elif (culture_list.loc[got.loc[count, 'culture']] < min_acceptable_culture_count):
        got.loc[count, 'mod_culture'] = "Other"
        # print(count, got.loc[count, 'T_title'])
    else:
        got.loc[count, 'mod_culture'] = got.loc[count, 'culture']
print(got['mod_culture'].isnull().any())


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

#### Imputing house data for some characters, source online various
houses = {   'House Baratheon':  ['Tommen Baratheon', 
                                'Joffrey Baratheon',
                                'Stannis Baratheon'],
            'House Durrandon':  ['Arrec Durrandon'],
            'House Florent':    ['Omer Florent'],
            'House Greyjoy':    ['Euron Greyjoy',
                                'Balon Greyjoy'],
            'House Hoare':      ['Harrag Hoare'],
            'House Lannister':  ['Genna Lannister',
                                'Lancel V Lannister'],
            'House Martell':    ['Elia Martell'],
            'House Mudd':       ['Tristifer IV Mudd',
                                'Tristifer V Mudd'],
            'House Stark':      ['Bessa [Winterfell',
                                'Benjen Stark [Bitter]',
                                'Brandon Stark [Burner]',
                                'Brandon Stark [Shipwright]',
                                'Benjen Stark [Sweet]',
                                'Torrhen Stark',
                                'Robb Stark',
                                'Theon Stark'],
            'House Targaryen':  ['Aerys I Targaryen',
                                'Aegon I Targaryen',
                                'Aenys I Targaryen',
                                'Viserys I Targaryen',
                                'Baelor I Targaryen',
                                'Maegor I Targaryen',
                                'Aegon II Targaryen',
                                'Daeron II Targaryen',
                                'Viserys II Targaryen',
                                'Jaehaerys II Targaryen',
                                'Aegon III Targaryen',
                                'Aegon IV Targaryen',
                                'Alysanne Targaryen',
                                'Aegon V Targaryen'],
            'House Tyrell':     ['Medwick Tyrell',
                                'Normund Tyrell'],
            'House Woodwright': ['Lucantine Woodwright',
                                'Portifer Woodwright'],
            'Kemmett Pyke':     ['Kemmett Pyke']
            }

nrow = 0
for nrow in np.arange(1, got.shape[0]):
    if(got.loc[nrow, 'house'] != got.loc[nrow, 'house']):
        for house, names in houses.items():
            if got.loc[nrow, 'name'] in names:
                got.loc[nrow, 'house'] = house
                break
            else:
                got.loc[nrow, 'house'] = 'Unknown'
            
got.loc[212, 'house']
for house, names in houses.items():
            if got.loc[6, 'name'] in names:
                got.loc[6, 'house'] = house
                print(got.loc[6, 'house'])



# selection = ['Ser', 'Stonehelm', 'Prince']
# count = 0
# for count in range(0, got.shape[0]):
#     if got.loc[count, 'title'] not in selection or got.loc[count, 'title'] is 'NaN':
#         got.loc[count, 'title'] = 'OtherABC'
# print(got['title'])
