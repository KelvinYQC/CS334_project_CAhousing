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




#%%
X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.3, random_state=42)
#%%
xgb_model = xgb.XGBRegressor()
parameters = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]
    }
#passing the scoring function in the GridSearchCV
clf = GridSearchCV(GradientBoostingClassifier(), parameters,refit=False,cv=5)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

#%%

#%%
pipe = Pipeline([('transformer', StandardScaler()),
                ('gradientBoost', GradientBoostingClassifier())
                ])
params = {
    'knn__n_neighbors':[3,5,7,8,100]
}

search = GridSearchCV(estimator = pipe,
                      param_grid = params,
                      cv = 5,
                      return_train_score= True)
search.fit(X_train,y_train)





#%%
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test)
y_mean = np.mean(y_train)
import multiprocessing
n_cpus_avaliable = multiprocessing.cpu_count()
print(f'We\'ve got {n_cpus_avaliable} CPUs to work with.')
xgb_params = {
    'eta':  0.05,
    'max_depth': 8,
    'subsample': 0.80,
    'objective':  'reg:linear',
    'eval_metric': 'rmse', # root mean square error
    'base_score':  y_mean,
    'nthread': n_cpus_avaliable
}
model = xgb.train(xgb_params, dtrain, num_boost_round=1648)
xgb_pred = model.predict(dtest)
test_mse = np.mean(((xgb_pred - y_test)**2))
test_rmse = np.sqrt(test_mse)
r_square = 1-(test_rmse/y_test.var())
print(f'Final test Rsquare: {r_square} with 1648 prediction rounds used')
#%%
print(y_test.var())
