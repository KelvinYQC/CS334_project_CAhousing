#%%
import pandas as pd
import numpy as np
import sklearn.linear_model
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, \
    RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import uniform, randint
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import r2_score


import xgboost as xgb
from sklearn.metrics import mean_absolute_error
#%%

# load the dataset
from sklearn.tree import DecisionTreeRegressor

dat = pd.read_csv("housing.csv")
sns.set(font_scale=0.5)
correlation = dat.corr()
sns.heatmap(correlation, xticklabels=dat.columns, yticklabels=dat.columns, cmap="YlGnBu", annot=True, fmt='.2f')
plt.xticks(rotation=20)
plt.yticks(rotation=45)
plt.savefig('heatmap.png', dpi=900)
plt.show()
#%%
# drop variables
print(dat.columns)
dat = dat.drop(columns=['households','total_rooms'])
#%%
dat.total_bedrooms.fillna(dat.total_bedrooms.median(), inplace=True)
print(dat.isnull().sum())

#%%
# # encode categorical
dat = pd.get_dummies(dat)
print(dat.head())

#%%
# encoder =LabelEncoder()
# dat['ocean_proximity']=encoder.fit_transform(dat['ocean_proximity'])

#%%
X = dat.drop("median_house_value", axis = 1)
y = dat['median_house_value']
#%%
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#%%
from sklearn.linear_model import LinearRegression

result = list()
MLA_compare = pd.DataFrame()
modelList = [
    DecisionTreeRegressor(),
    GradientBoostingRegressor(),
    RandomForestRegressor(),
    LinearRegression()
]
def modeling(models, X_train,y_train, X_test, y_test):
    row_index = 0
    for classifier in models:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        classifier_r2_score = r2_score(y_test, y_pred)
        result.append(( classifier.__class__.__name__, classifier_r2_score*100))
        row_index+=1

modeling(modelList, X_train,y_train, X_test, y_test)

# tune models
