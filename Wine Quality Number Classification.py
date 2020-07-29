# Classifying the quality of red and white wines (separately) using Decision Tree and Random Forest Classifiers
# Random Forest outperforms decision trees consistenly by at least 10%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

dfr = pd.read_csv('winequality-red.csv')  # dataframe for red wine
dfw = pd.read_csv('winequality-white.csv')  # dataframe for white wine

# print(dfr.head())
# print(dfw.head())
#
# print(dfr.columns, '\n', dfw.columns)

dfcols = ["fixed acidity",
           "volatile acidity",
           "citric acid",
           "residual sugar",
           "chlorides",
           "free sulfur dioxide",
           "total sulfur dioxide",
           "density",
           "pH",
           "sulphates",
           "alcohol",
           "quality"]

# The next little bit is changing the format of the data so that the columns are separated and the values are assigned
# to their proper columns. The initial data comes as one single column and one list of numbers which you can't work
# with

def separatew(dflist):
    return [ float(x) for x in dfw['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";' \
                                   '"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";' \
                                   '"alcohol";"quality"'][dflist].split(";")]


dfwhite = []
for i in range(1,len(dfw)):
    item = separatew(i)
    dfwhite.append(item)

dfw = pd.DataFrame(data=dfwhite,columns=dfcols)


def separater(dflist):
    return [ float(x) for x in dfr['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";' \
                                   '"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";' \
                                   '"alcohol";"quality"'][dflist].split(";")]

dfred = []
for i in range(1,len(dfr)):
    item = separater(i)
    dfred.append(item)

dfr = pd.DataFrame(data=dfred,columns=dfcols)

print(dfr.head())
print(dfw.head())

# lets do a little exploratory data analysis, we'll do all the analysis and classification for red wine and then
# try it on the white wine.

plt.figure(figsize=(8,5))
sns.countplot(x='quality',data=dfr)

plt.figure(figsize=(12,8))
sns.heatmap(data=dfr.corr(),annot=True, cmap='viridis')
plt.ylim(12,0)

plt.figure(figsize=(8,5))
plt.hist(x='quality',data=dfr)

sns.scatterplot(x=dfr['alcohol'], y=dfr['quality'])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = dfr.drop('quality', axis=1)
y = dfr['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier

dtreer = DecisionTreeClassifier()
dtreer.fit(X_train,y_train)
predsr = dtreer.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print('Decision Tree Results, Red Wine','\n',classification_report(y_test,predsr),'\n',confusion_matrix(y_test,predsr))

# 60% accuracy if just using the numbers to classify the wine quality and decision tree
# lets try random forest to see if it works any better

from sklearn.ensemble import RandomForestClassifier

rfcr = RandomForestClassifier(n_estimators=100)
rfcr.fit(X_train,y_train)
rfc_predsr = rfcr.predict(X_test)

print('Random Forest Results, Red Wine','\n',classification_report(y_test,rfc_predsr),'\n',confusion_matrix(y_test,rfc_predsr))

# 69% with random forest classifier, so much better.

# Lets look at white wine


plt.figure(figsize=(8,5))
sns.countplot(x='quality',data=dfw)

plt.figure(figsize=(12,8))
sns.heatmap(data=dfw.corr(),annot=True, cmap='viridis')
plt.ylim(12,0)

plt.figure(figsize=(8,5))
plt.hist(x='quality',data=dfw)

Xw = dfw.drop('quality',axis=1)
yw = dfw['quality']

X_trainw, X_testw, y_trainw, y_testw = train_test_split(Xw, yw, test_size=0.25, random_state=101)

X_trainw = scaler.fit_transform(X_trainw)
X_testw = scaler.transform(X_testw)

dtreew = DecisionTreeClassifier()
dtreew.fit(X_trainw,y_trainw)
predsw = dtreew.predict(X_testw)

print('Decision Tree Results, White Wine','\n',classification_report(y_testw,predsw),'\n',confusion_matrix(y_testw,predsw))

# Bad, 60% much better than 44 which is what we got the first time

rfcw = RandomForestClassifier(n_estimators=100)
rfcw.fit(X_trainw,y_trainw)
rfc_predsw = rfcw.predict(X_testw)

print('Random Forest Results, White Wine','\n',classification_report(y_testw,rfc_predsw),'\n',confusion_matrix(y_testw,rfc_predsw))

# 69% with 100, 200 and 300 estimators so it's pretty consistenly getting 69% classification accuracy
# ran the white wine test data on the red wine model and vice versa but the accuracy was pretty much the same
# for all of them, right around 69%




