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
got['mod_title'] = " "
for count in range(1, got.shape[0]+1):
    ## Impute values in new column for missing values in original column
    if (got.loc[count, 'title'] != got.loc[count, 'title']):
        got.loc[count, 'mod_title'] = "Unknown"
    # Impute values for columns with less than accetable freuency counts
    elif (title_list.loc[got.loc[count, 'title']] < min_acceptable_title_count):
        got.loc[count, 'mod_title'] = "Other"
    # Copy all other values from original column
    else:
        got.loc[count, 'mod_title'] = got.loc[count, 'title']
print(got['mod_title'].isnull().any())
print('Total ', len(got['mod_title'].unique()),' titles')

## column culture is 65% missing with 9% unique records, 
## missing value can replaced with 'Unknown' tag
culture_list = got['culture'].value_counts()
min_acceptable_culture_count = 10
## Anything less than the accepted count (10) will be treated 
## as "Other" culture, to reduct variability of the dataframe. 
got['mod_culture'] = " "
for count in range(1, got.shape[0]+1):
    ## Impute values in new column for missing values in original column
    if (got.loc[count, 'culture'] != got.loc[count, 'culture']):
        got.loc[count, 'mod_culture'] = "Unknown"
    # Impute values for columns with less than accetable freuency counts
    elif (culture_list.loc[got.loc[count, 'culture']] < min_acceptable_culture_count):
        got.loc[count, 'mod_culture'] = "Other"
    # Copy all other values from original column
    else:
        got.loc[count, 'mod_culture'] = got.loc[count, 'culture']
print(got['mod_culture'].isnull().any())
print('Total ', len(got['mod_culture'].unique()),' cultures')

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

## House is 22% missing with 348 unique values
#### Imputing house data for some characters, sourced from various online
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

# Impute data in house column based on name of the character in name column
# Also impute house for missing values as "unknown"
for nrow in np.arange(1, got.shape[0]+1):
    if(got.loc[nrow, 'house'] != got.loc[nrow, 'house']):
        for house, names in houses.items():
            if got.loc[nrow, 'name'] in names:
                got.loc[nrow, 'house'] = house
                break
            else:
                got.loc[nrow, 'house'] = 'Unknown'

## Check the missing value status

for cols in missing_cols:
    missing_summary.loc[cols, 'Data Type'] = got[cols].dtype
    missing_summary.loc[cols, '% Missing'] = (got[cols].isnull().sum() * 100 / len(got[cols])).round(1)    
    missing_summary.loc[cols, 'Unique Values'] = (len(got[cols].unique()))
    missing_summary.loc[cols, '% Unique values'] = int((len(got[cols].unique()))/(len(got[cols].dropna()))*100)
print(missing_summary)



house_list = got['house'].value_counts()
min_acceptable_house_count = 15
## Anything less than the accepted count will be treated 
## as "Other" house, to reduct variability of the dataframe. 
## Save the transofmed data in column mod_house
got['mod_house'] = " "
for count in range(1, got.shape[0]+1):
    ## Impute values in new column for missing values in original column
    if (got.loc[count, 'house'] != got.loc[count, 'house']):
        got.loc[count, 'mod_house'] = "Unknown"
    # Impute values for columns with less than accetable freuency counts
    elif (house_list.loc[got.loc[count, 'house']] < min_acceptable_house_count):
        got.loc[count, 'mod_house'] = "Other"
    # Copy all other values from original column
    else:
        got.loc[count, 'mod_house'] = got.loc[count, 'house']
print(got['mod_house'].isnull().any())
print('Total ', len(got['mod_house'].unique()),' houses')

## Spouse has 86% missing with 255 (92%) unique values
## The spouse which is name, may anot have correlation
## with model, so better to drop the colkumn
drop_cols.append('spouse')


## Create a variable for book number. If the character 
## did not appear in any of the book, he/she must have
## appeared in book number 6 or subsequent books
got['appearanceCount'] = got['book1_A_Game_Of_Thrones'] + \
                got['book2_A_Clash_Of_Kings'] + \
                got['book3_A_Storm_Of_Swords'] + \
                got['book4_A_Feast_For_Crows'] + \
                got['book5_A_Dance_with_Dragons']
got['appearanceCount'] = got['appearanceCount'].apply(lambda X: 1 if X == 0 else X)

plt.figure(figsize=(30, 18))
sns.countplot(x = got['appearanceCount'])
plt.xlabel("Number of books hero appeared in")
plt.ylabel("Hero Count")
plt.title("Lead character apperance frequency")
plt.show()


got['firstBook'] = np.nan
## Create a variable for the sequence of books a character has appaeared in
got['bookSequence'] =   got['book1_A_Game_Of_Thrones']*10000 + \
                        got['book2_A_Clash_Of_Kings']*2000 + \
                        got['book3_A_Storm_Of_Swords']*300 +\
                        got['book4_A_Feast_For_Crows']*40 +\
                        got['book5_A_Dance_with_Dragons']*5
got['bookSequence'] = got['bookSequence'].apply(lambda X: 6 if X == 0 else X)


## Create a varaiable the first book a character has appaeared in
# Function to create first and last book
def lastBook(X):
    if X >= 10**0 & X < 10**1:
        return X%10
    elif X >= 10**1 & X < 10**2:
        return X%100
    elif X >= 10**2 & X < 10**3:
        return X%1000
    elif X >= 10**3 & X < 10**4:
        return X%1000
    elif X >= 10**4 & X < 10**5:
        return X%10000
    elif X >= 10**5 & X < 10**6:
        return X%100000

def firstBook(X):
    if ((X >= 10**5) & (X < 10**6)):
        return int(X/100000)
    elif (X >= 10**4) & (X < 10**5):
        return int(X/10000)
    elif (X >= 10**3) & (X < 10**4):
        return int(X/1000)
    elif (X >= 10**2) & (X < 10**3):
        return int(X/100)
    elif (X >= 10**1) & (X < 10**2):
        return int(X/10)
    elif (X >= 10**0) & (X < 10**1):
        return int(X/10**0)

## create column for first and last book of the character
got['firstBook'] = got['bookSequence'].apply(lambda X: firstBook(X))

got['lastBook'] = got['bookSequence'].apply(lambda X: lastBook(X))

## Frequency of characters based on first and last book
plt.figure(figsize=(30, 18))
sns.countplot(x = got['firstBook'])
plt.xlabel("First book character has appeared in")
plt.ylabel("Character Count")
plt.title("Lead character first book apperance frequency")
plt.show()

plt.figure(figsize=(30, 18))
sns.countplot(x = got['lastBook'])
plt.xlabel("Last book character has appeared in")
plt.ylabel("Character Count")
plt.title("Lead character last book apperance frequency")
plt.show()



## isAliveMother has 98.9% missing value. Let us drop in first iteration
drop_cols.append('isAliveMother')


## isAliveFather has 98.7% missing value. Let us drop in first iteration
drop_cols.append('isAliveFather')

## isAliveHeir has 98.7% missing value. Let us drop in first iteration
drop_cols.append('isAliveHeir')

## isAliveHeir has 98.7% missing value. Let us drop in first iteration
drop_cols.append('isAliveSpouse')

## age is a continuous variable and has 77% missing value. 
## It might prove be a strongly correlated variable and needs 
## to be imputed

## Found two age for rows 1685 and 1689 wiht values --277980 and -298001
## shall impute median age at the abnove two values and all missing data
## replace negatiev age with NaN
got.loc[got.index[got['age'] == -277980].tolist()[0], 'age'] = np.nan
got.loc[got.index[got['age'] == -298001].tolist()[0], 'age'] = np.nan


plt.figure(figsize=(30, 18))
# sns.countplot(x = got['firstBook'], hue=)
sns.scatterplot(x = got['firstBook'], y = got['age'])
sns
plt.xlabel("First book character has appeared in")
plt.ylabel("Character Count")
plt.title("Lead character first book apperance frequency")
plt.show()

plt.figure(figsize=(30, 18))
sns.countplot(x = got['lastBook'])
plt.xlabel("Last book character has appeared in")
plt.ylabel("Character Count")
plt.title("Lead character last book apperance frequency")
plt.show()



# selection = ['Ser', 'Stonehelm', 'Prince']
# count = 0
# for count in range(0, got.shape[0]):
#     if got.loc[count, 'title'] not in selection or got.loc[count, 'title'] is 'NaN':
#         got.loc[count, 'title'] = 'OtherABC'
# print(got['title'])
