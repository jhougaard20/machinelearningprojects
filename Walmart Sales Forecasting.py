import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

features = pd.read_csv('features.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
stores = pd.read_csv('stores.csv')

train['Split'] = 'Train'
test['Split'] = 'Test'

print(stores.head())
print(features.head())
print(train.head())
print(test.head())

mylist = [stores['Type'] for x in stores['Store']][0]
print(stores['Store'][44])

store_type = []
i = 0
for item in train['Store']:

    if item == stores['Store'][i]:

        store_type.append(mylist[i])
    else:
        i += 1

stype = pd.DataFrame(store_type)

df = pd.concat([train,stype],axis=1)
df = df.rename(columns={0:'Type'})
df['Type'].fillna(value='B')

storetest_type = []
i = 0
for item in test['Store']:

    if item == stores['Store'][i]:

        storetest_type.append(mylist[i])
    else:
        i += 1

testtype = pd.DataFrame(storetest_type)

dftest = pd.concat([test,testtype],axis=1)

dftest = dftest.rename(columns={0:'Type'})

dftest['Type'].fillna(value='B',inplace=True)


def add_new_train_col(col_name, col_wanted):
    new_col = []
    i = 0
    for item in train[col_name]:

        if item == stores[col_name][i]:

            new_col.append(stores[col_wanted][i])
        else:
            i += 1

    new_col = pd.DataFrame(new_col)
    new_col.rename(columns={0: col_wanted}, inplace=True)
    return new_col


def add_new_test_col(col_name, col_wanted):
    new_test_col = []
    i = 0
    for item in test[col_name]:

        if item == stores[col_name][i]:

            new_test_col.append(stores[col_wanted][i])
        else:
            i += 1
    new_test_col = pd.DataFrame(new_test_col)
    new_test_col.rename(columns={0: col_wanted}, inplace=True)
    return new_test_col


store_size_col = add_new_train_col('Store','Size')
df = pd.concat([df,store_size_col],axis=1)

df['Size'].fillna(value=118221,inplace=True)
store_size_test_col = add_new_test_col('Store','Size')

dftest = pd.concat([dftest,store_size_test_col],axis=1)
dftest['Size'].fillna(value=118221,inplace=True)

df = pd.concat([df,dftest],axis=0,sort=True)

print(df.isnull().sum()*(100/len(df)))
print(df.info())

plt.figure(figsize=(6,4))
df.corr()['Weekly_Sales'].drop('Weekly_Sales').plot(kind='bar')

print(df.isnull().sum())

df['Type'].fillna(value='A',inplace=True)

print(df.isnull().sum())

plt.figure(figsize=(8,5))
sns.countplot(x='Type',data=df)

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='viridis', annot=True,
            linewidths=.5, cbar_kws={"shrink": .5})
plt.ylim(5,0)

print(df.loc[df['Weekly_Sales'] >240000,"Date"].value_counts())

print(df.select_dtypes(include=['bool','object']).columns)

# Make sure we can later recognize what a dummy once belonged to
df['Type'] = 'Type_' + df['Type'].map(str)
df['Store'] = 'Store_' + df['Store'].map(str)
df['Dept'] = 'Dept_' + df['Dept'].map(str)
df['IsHoliday'] = 'IsHoliday_' + df['IsHoliday'].map(str)

from datetime import datetime
from datetime import timedelta

df['DateType'] = [datetime.strptime(date, '%Y-%m-%d').date() for date in df['Date'].astype(str).values.tolist()]
df['Month'] = [date.month for date in df['DateType']]
df['Month'] = 'Month_' + df['Month'].map(str)
Month_dummies = pd.get_dummies(df['Month'] )

type_dummies = pd.get_dummies(data=df['Type'],drop_first=True)
store_dummies = pd.get_dummies(data=df['Store'],drop_first=True)
dept_dummies = pd.get_dummies(data=df['Dept'],drop_first=True)
holiday_dummies = pd.get_dummies(data=df['IsHoliday'],drop_first=True)

df['Black_Friday'] = np.where((df['DateType']==datetime(2010, 11, 26).date()) |
                              (df['DateType']==datetime(2011, 11, 25).date()), 'yes', 'no')
df['Pre_christmas'] = np.where((df['DateType']==datetime(2010, 12, 23).date()) |
                               (df['DateType']==datetime(2010, 12, 24).date()) |
                               (df['DateType']==datetime(2011, 12, 23).date()) |
                               (df['DateType']==datetime(2011, 12, 24).date()), 'yes', 'no')
df['Black_Friday'] = 'Black_Friday_' + df['Black_Friday'].map(str)
df['Pre_christmas'] = 'Pre_christmas_' + df['Pre_christmas'].map(str)
Black_Friday_dummies = pd.get_dummies(df['Black_Friday'] )
Pre_christmas_dummies = pd.get_dummies(df['Pre_christmas'] )

df = pd.concat([df,holiday_dummies,Pre_christmas_dummies,Black_Friday_dummies,type_dummies],axis=1)

medians = pd.DataFrame({'Median Sales' :df.loc[df['Split']=='Train'].groupby(by=['Type','Dept','Store','Month','IsHoliday'])['Weekly_Sales'].median()}).reset_index()
print(medians.head(-5))

# Merge by type, store, department and month
df = df.merge(medians, how = 'outer', on = ['Type','Dept','Store','Month','IsHoliday'])
# Fill NA
df['Median Sales'].fillna(df['Median Sales'].loc[df['Split']=='Train'].median(), inplace=True)

# Create a key for easy access
df['Key'] = df['Type'].map(str)+df['Dept'].map(str)+df['Store'].map(str)+df['Date'].map(str)+df['IsHoliday'].map(str)

print(df.head())

print(df[df['Split']=='Train']['Weekly_Sales'].isnull().sum())

# Forecast the difference between the median sales and the weekly sales
df['Difference'] = df['Median Sales'] - df['Weekly_Sales']

print(df.columns)

df['Sales_Diff'] = df['Difference']
df.drop('Difference',axis=1,inplace=True)

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.ylim(11,0)

selector = ['Size',
            'Pre_christmas_no',
            'Pre_christmas_yes',
            'Black_Friday_no',
            'Black_Friday_yes',
            'Type_B',
            'Type_C',
            'Median Sales',
            'Sales_Diff',
            'IsHoliday_True']

plt.figure(figsize=(8, 5))
df.corr()['Weekly_Sales'].drop('Weekly_Sales').plot(kind='bar')

print(df[selector].describe().transpose())

train = df[df['Split'] == 'Train']
test = df[df['Split'] == 'Test']

test.drop('Sales_Diff',axis=1,inplace=True)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy import sparse

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation

from tensorflow.keras.optimizers import Adam

X = train[selector].values
y = train['Weekly_Sales'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rfreg = RandomForestRegressor(n_estimators=100)

print(X_test.shape)

# We'll do a random forest regression first and then a deep learning algorithm to predict weekly sales.
# Some of the examples I have seen online are predicting the difference between median and weekly sales, I am planning
# on attempting both

rfreg.fit(X_train,y_train)
rf_pred_ws = rfreg.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import regularizers

print('Mean Absolute Error: ',np.round(mean_absolute_error(y_test, rf_pred_ws),3),'\n','Mean Squared Error: ',
      np.round(mean_squared_error(y_test,rf_pred_ws),2),'\n',
      'Root Mean Squared Error: ',np.round(np.sqrt(mean_squared_error(y_test,rf_pred_ws)),2))

plt.figure(figsize=(10,6))
sns.scatterplot(y_test,rf_pred_ws,)
plt.xlabel('Y True')
plt.ylabel('Y Predicted')
plt.title('Random Forest Predictor of Weekly Sales')
# This is pretty accurate which seems fishy to me... oh well!

rf_diff = (y_test - rf_pred_ws).mean()
print(rf_diff)

ann_model = Sequential()

ann_model.add(Dense(20, activation='relu',
                    kernel_regularizer = regularizers.l2(0.01)))
ann_model.add(Dropout(0.4))
ann_model.add(Dense(10, activation='relu',
                   kernel_regularizer = regularizers.l2(0.01)))
ann_model.add(Dropout(0.4))
# ann_model.add(Dense(5, activation='relu'))
# ann_model.add(Dropout(0.3))

# BINARY CLASSIFICATION, YES OR NO, 1 OR 0, SO THE ACTIVATION FUNCTION MUST BE SIGMOID
ann_model.add(Dense(1, activation='linear'))

ann_model.compile(loss='mean_absolute_error',optimizer='adam')

# Choose whatever number of layers/neurons you want.

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

# Remember to compile()

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',mode='min', verbose=1, patience=25)
# tracking validation loss which is what we want to minimize

# ann_model.fit(x=X_train,y=y_train,epochs=1000,validation_data=(X_test,y_test),
#           callbacks=[early_stop],
#           batch_size=3000,verbose=0)

dropout_losses = pd.DataFrame(ann_model.history.history)
plt.figure(figsize=(8,5))
dropout_losses.plot()
plt.xlabel('Epochs')
plt.ylabel('loss')

preds_neural = ann_model.predict(X_test)

preds_nn = []
for item in preds_neural:
    preds_nn.append(item)

plt.figure(figsize=(10,6))
sns.scatterplot(y_test,preds_nn)
plt.xlabel('Y True')
plt.ylabel('Y Predicted Neural')
plt.title('ANN Predictor of Weekly Sales')

print('Mean Absolute Error: ',np.round(mean_absolute_error(y_test, preds_neural),3),'\n','Mean Squared Error: ',
      np.round(mean_squared_error(y_test,preds_neural),2),'\n',
      'Root Mean Squared Error: ',np.round(np.sqrt(mean_squared_error(y_test,preds_neural)),2))

print(train['Weekly_Sales'].describe())

print(train['Weekly_Sales'].median())

test['Sales_Diff'] = np.zeros(len(test))

rf_test_preds = rfreg.predict(test[selector])
rsquared_rf = np.corrcoef(y_test,rf_pred_ws)

# preds_test_neural = ann_model.predict(test)









