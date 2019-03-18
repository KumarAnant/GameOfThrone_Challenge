import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


import os
os.chdir('E:/OneDrive/Hult/Machine Learning/Assignments/Assignment - 2')
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


########################
# Correlation Heatmap
########################
# Using palplot to view a color scheme
sns.set(font_scale=2)
sns.palplot(sns.color_palette('coolwarm', 20))
fig, ax = plt.subplots(figsize=(30,20)) 
got2 = got.corr().round(1)
sns.heatmap(got2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)
plt.title("Correlation Heat Map")
plt.show()



## Get some basic information of the dtaframe
print(got.info())
print(got.describe())
print(got.columns)

# Check for missing values in the dataset
plt.figure(figsize=(40, 20))
sns.set(font_scale=2)
sns.heatmap(got.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.set_style('whitegrid')
plt.title("Missing Values Heat Map")
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
min_acceptable_title_count = 15
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
print("Missing value: ", got['mod_title'].isnull().any())
print('Total ', len(got['mod_title'].unique()),' titles')

## Once mod_title is ready, add title column in remove list
drop_cols.append('title')

## column culture is 65% missing with 9% unique records, 
## missing value can replaced with 'Unknown' tag
culture_list = got['culture'].value_counts()
min_acceptable_culture_count = 15
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
print("Missing value: ", got['mod_culture'].isnull().any())
print('Total ', len(got['mod_culture'].unique()),' cultures')

## Once mod_culture is ready, add title column in remove list
drop_cols.append('culture')

## Drop the name variable
drop_cols.append('name')
## DateOfBirth being overall 77% missing and also that another
## column of Age is already available, the BirthDay column can be dropped
drop_cols.append('dateOfBirth')
drop_cols.append('m_dateOfBirth')
## column mother is 85% missing and it is a qualitative datta
## can not be imputed, hence be dropped
drop_cols.append('mother')
drop_cols.append('m_mother')
## column father is 80% missing and it is a qualitative datta
## can not be imputed, hence be dropped
drop_cols.append('father')
drop_cols.append('m_father')
## column heir is almost 100% missing and it is a qualitative datta
## can not be imputed, hence be dropped
drop_cols.append('heir')
drop_cols.append('m_heir')
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
print("Missing value: ", got['mod_house'].isnull().any())
print('Total ', len(got['mod_house'].unique()),' houses')

## Once mod_house is ready, add title column in remove list
drop_cols.append('house')

## Spouse has 86% missing with 255 (92%) unique values
## The spouse which is name, may anot have correlation
## with model, so better to drop the column
drop_cols.append('spouse')
drop_cols.append('m_spouse')

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
def getBookSeq(book1, book2, book3, book4, book5):
    bookSeq = 0
    if(book1 != 0):
        bookSeq = bookSeq*10 + book1
    if(book2 != 0):
        bookSeq = bookSeq*10 + book2*2
    if(book3 != 0):
        bookSeq = bookSeq*10 + book3*3
    if(book4 != 0):
        bookSeq = bookSeq*10 + book4*4
    if(book5 != 0):
        bookSeq = bookSeq*10 + book5*5
    if((book1 + book2 + book3 + book4 + book5)  == 0):
        bookSeq = 6
    return int(bookSeq)


got['bookSequence'] = np.nan
for count in range(1, got.shape[0]+1):
    got.loc[count, 'bookSequence'] =   getBookSeq(  got.loc[count, 'book1_A_Game_Of_Thrones'],
                                                    got.loc[count, 'book2_A_Clash_Of_Kings'],
                                                    got.loc[count, 'book3_A_Storm_Of_Swords'],
                                                    got.loc[count, 'book4_A_Feast_For_Crows'],
                                                    got.loc[count, 'book5_A_Dance_with_Dragons'])

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
        return int(X)

# Create columns for sequence of book character has appear in
got['bookSequence']

## create column for first and last book of the character
got['firstBook'] = got['bookSequence'].apply(lambda X: firstBook(X))
got['lastBook'] = got['bookSequence'].apply(lambda X: lastBook(int(X)))



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
drop_cols.append('m_isAliveMother')

## isAliveFather has 98.7% missing value. Let us drop in first iteration
drop_cols.append('isAliveFather')
drop_cols.append('m_isAliveFather')
## isAliveHeir has 98.7% missing value. Let us drop in first iteration
drop_cols.append('isAliveHeir')
drop_cols.append('m_isAliveFather')
## isAliveHeir has 98.7% missing value. Let us drop in first iteration
drop_cols.append('isAliveSpouse')
drop_cols.append('m_isAliveSpouse')
## age is a continuous variable and has 77% missing value. 
## It might prove be a strongly correlated variable and needs 
## to be imputed

## Found two age for rows 1685 and 1689 wiht values --277980 and -298001
## shall impute median age at the abnove two values and all missing data
## replace negatiev age with NaN
got.loc[got.index[got['age'] == -277980].tolist()[0], 'age'] = np.nan
got.loc[got.index[got['age'] == -298001].tolist()[0], 'age'] = np.nan

## Replace all NaN with median agea
birth_fill = got['age'].median()

## Plot boxplot to see distribution of median age for characters appearing
## first in variousdifferent volumes of book
plt.figure(figsize=(30, 20))
sns.boxenplot(y = 'age', x = 'firstBook', data = got)
plt.title("Age distribution Boxplot")
plt.show()

# Creat a list to save median age per book as start of character appearance
med_age = np.arange(6)
## For book volume 1, median age. Population size = 133
med_age[0] = (got[got['firstBook'] == 1]['age'].dropna().median())
## For book volume 2, median age. Population size = 134
med_age[1] = (got[got['firstBook'] == 2]['age'].dropna().median())
## For book volume 3, median age. Population size = 43
med_age[1] = (got[got['firstBook'] == 3]['age'].dropna().median())
## For book volume 4, median age. Population size = 43
med_age[3] = (got[got['firstBook'] == 4]['age'].dropna().median())
## For book volume 5, median age. Population size = 13
med_age[4] = (got[got['firstBook'] == 5]['age'].dropna().median())
## For book volume 6, median age. Population size = 65
med_age[5] = (got[got['firstBook'] == 6]['age'].dropna().median())

## Set outlier flag for Age variable
# print(got['age'].quantile([0.05, 0.93]))
# print(got['age'].max())
# print(got['age'].min())

got['o_age'] = 0
age_up_outlier = 98
age_low_outlier = 5

## Impute age based on first book apapearance era
for count in range(1, got.shape[0]+1):    
    if ((got.loc[count, 'age']) != (got.loc[count, 'age'])):
        got.loc[count, 'age'] = med_age[got.loc[count, 'firstBook']-1]
    elif got.loc[count, 'age'] < age_low_outlier:
        got.loc[count, 'o_age'] = -1
    elif got.loc[count, 'age'] > age_up_outlier:
        got.loc[count, 'o_age'] = 1


## numDeadRelations
## Set outlier flag for numDeadRelations
# print(got['numDeadRelations'].quantile([0.10, 0.99]))
got['o_numDeadRelations'] = 0
numDeadRelations_up_outlier = 7
numDeadRelations_low_outlier = 0


## Set outlier flag for popularity
# print(got['popularity'].quantile([0.10, 0.98]))
got['o_popularity'] = 0
popularity_up_outlier = 0.71
popularity_low_outlier = 0.0067



## Impute age based on first book apapearance era
for count in range(1, got.shape[0]+1):    
    if got.loc[count, 'numDeadRelations'] < numDeadRelations_low_outlier:
        got.loc[count, 'o_numDeadRelations'] = -1
    elif got.loc[count, 'numDeadRelations'] > numDeadRelations_up_outlier:
        got.loc[count, 'o_numDeadRelations'] = 1
    if got.loc[count, 'popularity'] < popularity_low_outlier:
        got.loc[count, 'o_popularity'] = -1
    elif got.loc[count, 'popularity'] > popularity_up_outlier:
        got.loc[count, 'o_popularity'] = 1


########################
# Creat dummie variables for discrete data
########################

# Cerate a copy of imputed value dataframe
got_OLS = got.copy(deep = True)
## Drop all unnecessary columns, listed in drop_cols column
for cols in drop_cols:
    print("\n\nDeleting..", cols)
    if cols in got_OLS.columns:
        print("Deleted .. ", cols)
        got_OLS.drop(cols, axis = 1, inplace = True)

## Create dummy variables for title column data, drop original column
title_dummies = pd.get_dummies(list(got_OLS['mod_title']), prefix = 'title', drop_first = True)
title_dummies.index = np.arange(1, len(title_dummies)+1)
got_OLS.drop('mod_title', axis = 1, inplace = True)


## Create dummy variables for culture column data, drop original column
culture_dummies = pd.get_dummies(list(got_OLS['mod_culture']), prefix = 'culture', drop_first = True)
culture_dummies.index = np.arange(1, len(culture_dummies)+1)
got_OLS.drop('mod_culture', axis = 1, inplace = True)

## Create dummy variables for house column data, drop original column
house_dummies = pd.get_dummies(list(got_OLS['mod_house']), prefix = 'house', drop_first = True)
house_dummies.index = np.arange(1, len(house_dummies)+1)
got_OLS.drop('mod_house', axis = 1, inplace = True)


## Concatenate dummy variables
got_OLS = pd.concat(
        [got_OLS.loc[:,:],
            title_dummies,
            culture_dummies,
            house_dummies
         ], axis = 1)


## Seperate predictor and dependent variables
X = got_OLS.drop('isAlive', axis = 1)
X = X.loc[:, :]
y = got_OLS['isAlive']
y = y.loc[:]


## Train and test data split with stratification
X_test, X_train, y_test, y_train = train_test_split(X, y, stratify = y, test_size = 0.3)
lm = LogisticRegression()
lm.fit(X_train, y_train)

## Predict for the train set data
predictions = lm.predict(X_test)

# pd.DataFrame(lm.coef_, index = X.columns)
## Prediction report
print("Classification report: \n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))


###################################
######### KNN Model ###############
###################################

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

## Making data ready for KNN model
## Brining the target column at the end of dataframe
df_isAlive = got_OLS.pop('isAlive')
got_OLS['isAlive'] = df_isAlive

## Scaling the data to normalize
scaler = StandardScaler()
scaler.fit(got_OLS.drop(['isAlive'], axis = 1))
scaled_feature = scaler.transform(got_OLS.drop('isAlive', axis = 1))
got_feat = pd.DataFrame(scaled_feature, columns=got_OLS.columns[:-1])

## Test and train split of data
X = got_feat
y = got['isAlive']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 508)

## KNN Model intiatiate with N value - 1
knn = KNeighborsClassifier(n_neighbors=1)

# Fit model
knn.fit(X_train, y_train)

## Predict data using model fro test dataset
pred = knn.predict(X_test)

## Model accuracy for N = 1
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

## Find value of N for best accuracy
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(y_test != pred_i))

## Plot the accuracy graph
plt.figure(figsize=(20, 15))
plt.ylabel('Error Rate')
plt.xlabel('N-Neighbour Value')
plt.title("KNN Neighour Optimum Level")
plt.plot(range(1, 40), error_rate, color = 'blue', 
                        linestyle = 'dashed', 
                        marker = 'o', 
                        markerfacecolor = 'red', 
                        markersize = 10)

## Accuracy is best for N = 22.

## Build model for N = 22
## KNN Model intiatiate with N value - 1
knn = KNeighborsClassifier(n_neighbors=22)

# Fit model
knn.fit(X_train, y_train)

## Predict data using model fro test dataset
pred = knn.predict(X_test)

## Model accuracy for N = 22
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))



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

## Random forest model
rfc = RandomForestClassifier(n_estimators=200)
## Fit random forest model
rfc.fit(X_train, y_train)

## Prediction of Random Forest model
predictions = rfc.predict(X_test)

## Accuracy oif random forest
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


#############################################
###### Support vector classifier ############
#############################################
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001]}

grid = GridSearchCV(SVC(), param_grid=param_grid, verbose=3)
grid.fit(X_train, y_train)

grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))

