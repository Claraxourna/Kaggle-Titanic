## Import useful libraries
import numpy as np # linear algrebra
import pandas as pd # data processing
from pandas import Series,DataFrame
import numpy as np 
import matplotlib.pyplot as plt # plots
import seaborn as sns  # statistical data visualization

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

## Get data frame
titanic_test = pd.read_csv("../input/test.csv")
titanic_train = pd.read_csv("../input/train.csv")

# Preview & Describe
print(titanic_train.head())
print(titanic_train.describe())
print(titanic_train.info())

## Convert Sex into a bolean column
titanic_test.loc[titanic_test['Sex'] == 'female','Sex'] = 1
titanic_test.loc[titanic_test['Sex'] == 'male','Sex'] = 0
titanic_train.loc[titanic_train['Sex'] == 'female','Sex'] = 1
titanic_train.loc[titanic_train['Sex'] == 'male','Sex'] = 0

## Embarked column
# Visualization
sns.countplot('Embarked', data = titanic_train)
sns.factorplot('Embarked','Survived', data = titanic_train)

# Replace missing values with S
titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")

# Convert into numeric values
titanic_train.loc[titanic_train["Embarked"] == "S","Embarked"] = 0
titanic_train.loc[titanic_train["Embarked"] == "C","Embarked"] = 1
titanic_train.loc[titanic_train["Embarked"] == "Q","Embarked"] = 2

titanic_test.loc[titanic_test["Embarked"] == "S","Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C","Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q","Embarked"] = 2

## Fare
print(titanic_train["Fare"].describe()) # no missing values 
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

# convert from float to int
titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare']    = test_df['Fare'].astype(int)

# get fare for survived & didn't survive passengers 
fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] == 0]
fare_survived     = titanic_df["Fare"][titanic_df["Survived"] == 1]

# get average and std for fare of survived/not survived passengers
avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])

# plot
titanic_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))

avgerage_fare.index.names = std_fare.index.names = ["Survived"]
avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
#modif on github
