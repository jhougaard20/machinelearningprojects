import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

print(test.head())
print(train.head())

print(train.info())
# print(train.shape)
# print(test.shape)

print(train.describe().transpose())

# passenger ID is not needed, it's basically redundant with an index
test.drop('PassengerId',axis=1,inplace=True)
train.drop('PassengerId',axis=1,inplace=True)

# Exploratory Data Analysis

plt.figure(figsize=(10,6))
sns.heatmap(train.isnull(),cmap='viridis',cbar=False)

plt.figure(figsize=(10,6))
sns.heatmap(test.isnull(),cmap='viridis',cbar=False)

plt.figure(figsize=(10,6))
train['Age'].plot.hist(bins=40)

plt.figure(figsize=(10,6))
sns.distplot(train['Age'].dropna(),bins=40)

plt.figure(figsize=(10,6))
sns.distplot(train['Fare'],bins=30)

print(train['Fare'].max())
print(train['Fare'].mean())

plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass',y='Fare',data=train,palette='rainbow')

plt.figure(figsize=(8,5))
sns.countplot(x='Survived',data=train,hue='Sex',palette="RdBu")

plt.figure(figsize=(8,5))
sns.countplot(x='Survived',data=train,hue='Pclass')

plt.figure(figsize=(8,5))
sns.countplot(x='SibSp',data=train)

print(train.isnull().sum())
print(test.isnull().sum())

# so we need to figure out what to do with our missing data
print(train.corr())

print(train.corr()['Age'].sort_values()) # most correlated to Pclass so we can fill in age based on the average age of people
# in the various Pclasses

print(test.corr()['Age'].sort_values())

plt.figure(figsize=(12,6))
sns.countplot('Pclass',data=train)

print(train.groupby('Pclass')['Age'].mean()) # get the average age from people in the various groups)
print(test.groupby('Pclass')['Age'].mean())

# so let's fill in the null values in the dataframes by putting in the average age of everyone else based on class


def fill_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 39
        elif Pclass == 2:
            return 29
        elif Pclass == 3:
            return 25

    else:
        return Age


train['Age'] = train[['Age','Pclass']].apply(fill_age,axis=1)
test['Age'] = test[['Age','Pclass']].apply(fill_age,axis=1)

print(train.isnull().sum())
print(test.isnull().sum())

plt.figure(figsize=(12,8))
sns.heatmap(train.corr(),cmap='viridis',annot=True)

plt.figure(figsize=(12,8))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)

test.dropna(inplace=True)
train.dropna(inplace=True)

# create categorical variables using pandas.  ML algorithm can't take in a string, so you have to make it binary,
# creating a dummy variable

sex = pd.get_dummies(train['Sex'],drop_first=True)
sex_test = pd.get_dummies(test['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
embark_test = pd.get_dummies(test['Embarked'],drop_first=True)

train = pd.concat([train,sex,embark],axis=1)
test = pd.concat([test,sex_test,embark_test],axis=1)

train.drop(['Sex','Embarked','Ticket','Name'],axis=1,inplace=True)
test.drop(['Sex','Embarked','Ticket','Name'],axis=1,inplace=True)

print(train.head())

from sklearn.model_selection import train_test_split

X = train.drop('Survived',axis=1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)
preds = logreg.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print('Model Test w/ Train Data','\n',confusion_matrix(y_test,preds),'\n',classification_report(y_test,preds))

test_predictions = logreg.predict(test)
test_predictions = pd.DataFrame(test_predictions,columns=['Survived'])
print(test_predictions.head())

plt.figure()
sns.countplot(test_predictions['Survived'])
plt.show()

