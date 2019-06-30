
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import math
import matplotlib
import seaborn as sns


# In[25]:


# read data from csv file
original = pd.read_csv('autos.csv', sep=',', header=0, encoding='cp1252')
#original = original.head(10000)
# Check data statistics
original.describe()

# boxplot for quantity data
plotData = [original['price']]
plt.boxplot(plotData)
plt.show()

plotData = [original['kilometer']]
plt.boxplot(plotData)
plt.show()

plotData = [original['monthOfRegistration']]
plt.boxplot(plotData)
plt.show()

plotData = [original['yearOfRegistration'],original['powerPS']]
plt.boxplot(plotData)
plt.show()

plotData = [original['postalCode']]
plt.boxplot(plotData)
plt.show()

plotData = [original['nrOfPictures']]
plt.boxplot(plotData)
plt.show()


# In[26]:


# In[33]:
# Restructuring -> Vertical Decomposition
# drop low impact data on price

# seller, offerType, abtest -> has one or two data.
# nrOfPictures -> has only 0.
# dataCrawled, lastSeen, dataCreated -> information about data crawling
# postalCode -> postal information
original.drop(['seller', 'offerType', 'abtest', 'dateCrawled', 'nrOfPictures', 'lastSeen', 'postalCode', 'dateCreated'], axis='columns', inplace=True)


# In[27]:


# In[34]:
# Cleaning dirty data



# Remove unusable data(redundant data)
data = original.drop_duplicates(['name','price','vehicleType','yearOfRegistration'
                         ,'gearbox','powerPS','model','kilometer','monthOfRegistration','fuelType'
                         ,'notRepairedDamage'])

# Remove the outliers and wrong data
# if price < 100 -> wrong data
# if price > 150000 -> the outliers
data = data[
        (data.yearOfRegistration <= 2016)
      & (data.yearOfRegistration >= 1950)
      & (data.price >= 100)
      & (data.price <= 150000)
      & (data.powerPS >= 10)
      & (data.powerPS <= 500)]


# In[35]:
# Check null data
data.isnull().sum()


# In[36]:
# Missing Data

# 'model' ->  very high impact data on price -> essential for price prediction
# --> need to drop null data in 'model' column
data = data[data.model.notnull()]

# Replace null with others
# Replace with the most value according to model type.
for i in range(len(data.model.unique())) :
    data["vehicleType"].fillna(data.vehicleType[data["model"]==data.model.unique()[i]].value_counts().head(1).index[0], inplace=True)
    data["fuelType"].fillna(data.fuelType[data["model"]==data.model.unique()[i]].value_counts().head(1).index[0], inplace=True)
    data["gearbox"].fillna(data.gearbox[data["model"]==data.model.unique()[i]].value_counts().head(1).index[0], inplace=True)

# null in 'notRepairedDamage' is one of the deciding factors for price prediction.
# -> replace null with 'not-declared'
data["notRepairedDamage"].fillna(value='not-declared', inplace=True)

# Check whether the misting data is processed well.
print("\n\n** After **")
print(data.isnull().sum())

print("\n\n** Size **")
print(len(data))


# In[15]:


# In[37]:
# columns that value is string
labels = ['gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']
les = {}


# label encoder
# encode string columns to 0~k-1 labels
# and add to exist dataframe --> feature creation
for l in labels:
    les[l] = preprocessing.LabelEncoder()
    les[l].fit(data[l].astype(str))
    tr = les[l].transform(data[l].astype(str))
    data.loc[:, l + '_encode'] = pd.Series(tr, index=data.index)

# extract only necessary columns to analysis
labeled = data[ ['price'
                        ,'yearOfRegistration'
                        ,'powerPS'
                        ,'kilometer'
                        ,'monthOfRegistration']
                    + [x+"_encode" for x in labels]]


# In[16]:


# In[38]:
# correlation
# calculate the correlation matrix
corr = labeled.corr()
# adjust matrix size to (12, 10)
plt.subplots( figsize =( 12 , 10 ) )
# plot the correlation heatmap
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# In[17]:


# In[39]:
# Split X and Y
# Y --> price column
# X --> data without price column
Y = labeled['price']
X = labeled.drop(['price'], axis='columns', inplace=False)

# standardization
# using StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scale = scaler.fit_transform(X)
X_scale = pd.DataFrame(X_scale, columns = X.columns)
print(X_scale)
print(Y)


# In[23]:


# In[40]:


# Divide data into training data and test data (70% training data, 30% test data)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123)

from sklearn.tree import DecisionTreeRegressor


decisionTree = DecisionTreeRegressor().fit(X_train, y_train) # make decisionTreeRegressor model for predict dependent value
predictDecision = decisionTree.predict(X_test) # predict dependent value to test data and assign 'predictDecision'
score = decisionTree.score(X_test, y_test) # accuracy for result of predictition
print("Decision Tree Score : ",score)





# In[45]:

# Using ensemble learning for regression
# ==> RandomForestRegressor
# n_estimators: The number of trees in the forest
# oob_score: out of bagging score
randomForest = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=123)
randomForest.fit(X_train, y_train)


# In[47]:

# predict data using X_test
predicted = randomForest.predict(X_test)
y_predict = pd.Series(predicted) # make series of pandas

# print test data, predict data
print("\nTEST")
print(y_test)
print("PREDICT")
print(y_predict)


# In[19]:


# Resets index to compare original test data with predicted data
y_test = y_test.reset_index(drop=True)
y_predict = y_predict.reset_index(drop=True)

# Print the data using the index in order that the values are less different.
temp = (y_test - y_predict).abs().sort_values().index
for i in temp:
    print("Original: ", y_test.iloc[i])
    print("Predicted: ", y_predict.iloc[i])
    print()

# print out of bag score
# Testing with samples not used in training.
print(f'Out-of-bag score estimate: {randomForest.oob_score_:.3}')


# In[20]:


# MAPE(mean absolute percentage error)
# : measure of prediction accuracy of a forecasting method
def MAPE(actual, predict):

    sum_actuals = sum_errors = 0

    for actual_val, predict_val in zip(actual, predict):
        abs_error = actual_val - predict_val # calculate error
        if abs_error < 0:
            abs_error = abs_error * -1

        sum_errors = sum_errors + abs_error
        sum_actuals = sum_actuals + actual_val

    # calculate and print MAPE
    mean_abs_percent_error = sum_errors / sum_actuals
    print("MAPE: ")
    print(mean_abs_percent_error)
    # calculate accuracy
    print("Accuracy using MAPE: ")
    print(1-mean_abs_percent_error)


# In[15]:
MAPE(y_test, y_predict)

