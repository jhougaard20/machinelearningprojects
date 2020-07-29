# This classification changed up a little from predicting quality of wine in a number, instead predicts
# whether the wine is "good quality" or "bad quality" as good quality wine being any wine that was given a
# quality score of 6 or better (out of 10)

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

dfr['color'] = 'red'
dfw['color'] = 'white'

df = pd.concat([dfr,dfw],axis=0,ignore_index=True)
print(df.head())

plt.figure(figsize=(8,5))
sns.countplot(x='quality',data=df,hue='color')

plt.figure(figsize=(12,8))
sns.heatmap(data=df.corr(),cmap='viridis',annot=True)
plt.ylim(12,0)

plt.figure(figsize=(12,6))
sns.boxplot(x='quality',y='alcohol',data=df,hue='color')
# seems like there's a relative correlation that the higher the alcohol content, the higher the quality

print(df.corr()['quality'].sort_values())

plt.figure(figsize=(12,6))
sns.boxplot(x='quality',y='density',data=df,palette='rainbow')
# kind of a trend that the lower the density the higher the quality but
# the difference in density is so minimal I can't imagine anyone would be able to tell
# maybe it feels lighter? Who tf knows
plt.show()

print(df.isna().sum()) # not missing any data

df['goodquality'] = [1 if x >= 6 else 0 for x in df['quality']]

# print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN didn't really work here, <50% accuracy trying random forest
from sklearn.ensemble import RandomForestClassifier

X = df.drop(['goodquality','quality','color'],axis=1)
y = df['goodquality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rfc = RandomForestClassifier(n_estimators=150)
rfc.fit(X_train,y_train)
preds = rfc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print('Binary Classification Reports','\n',confusion_matrix(y_test,preds),'\n',classification_report(y_test,preds))

# woah, significantly better at 83% if 6 or higher is deemed "good" wine, that's about as high as I can get it
# definitely more accurate at predicting if the wine is "good quality" and messed up more where it said "bad" wine
# was "good quality"

# gonna try to see if it can predict the color of the wine

X_c = df.drop(['color','quality','goodquality'],axis=1)
y_c = df['color']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_c, y_c, test_size=0.25, random_state=101)

Xc_train = scaler.fit_transform(Xc_train)
Xc_test = scaler.transform(Xc_test)

rfcc = RandomForestClassifier(n_estimators=200)
rfcc.fit(Xc_test,yc_test)
predsc = rfcc.predict(Xc_test)

print('Color Classification Reports','\n',confusion_matrix(yc_test,predsc),'\n',classification_report(yc_test,predsc))
# well I stand corrected, it predicted perfectly the color of the wine which I feel like I did something wrong but
# wow




