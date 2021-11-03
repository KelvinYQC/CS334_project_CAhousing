#%%
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

X_train = pd.read_csv("X_train.csv")
y_train= pd.read_csv("y_train.csv")
X_test= pd.read_csv("X_test.csv")
y_test= pd.read_csv("y_test.csv")
y_test = y_test.to_numpy()
y_test = y_test.reshape(y_test.shape[0],)
y_train = y_train.to_numpy()
y_train = y_train.reshape(y_train.shape[0],)
#%%
# decision tree
tunedResult = list()
dt = DecisionTreeRegressor()
hyperparameters_dt = [{"splitter":["best","random"],
                       'max_depth': range(1, 100, 5),
                       'min_samples_leaf': range(1, 1000,50),
                       "max_features":["auto","log2","sqrt",None]}]

clf = GridSearchCV(dt, hyperparameters_dt, cv=5)
best_model = clf.fit(X_train, y_train)
print("decision tree tuned result")
print(best_model.best_estimator_)
turnedmax_depth = best_model.best_estimator_.get_params()['max_depth']
turnedmax_features = best_model.best_estimator_.get_params()['max_features']
tunredmin_samples_leaf = best_model.best_estimator_.get_params()['min_samples_leaf']


dt_classifier = DecisionTreeRegressor(max_depth = turnedmax_depth,
                                      max_features=turnedmax_features,
                                      min_samples_leaf=tunredmin_samples_leaf)
clf = dt_classifier.fit(X_train, y_train)
y_pred = clf.predict(X_test)
classifier_r2_score_dt = r2_score(y_test, y_pred)
print(classifier_r2_score_dt)

tunedResult.append((clf.__class__.__name__, classifier_r2_score_dt))
#%%
# RandomForestRegressor
rd = RandomForestRegressor()

hyperparameters_rd = [{'bootstrap': [True, False],
     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,500, None],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': range(1, 1000, 100),
     'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
                      ]

clf = GridSearchCV(rd, hyperparameters_rd, cv=5)
best_model = clf.fit(X_train, y_train)
print("random forest tuned result")
print(best_model.best_estimator_)
#%%
turnedmax_depth = best_model.best_estimator_.get_params()['max_depth']
turnedmax_features = best_model.best_estimator_.get_params()['max_features']
tunredmin_samples_leaf = best_model.best_estimator_.get_params()['min_samples_leaf']
tunredmin_samples_split = best_model.best_estimator_.get_params()['min_samples_split']
tunredn_estimators = best_model.best_estimator_.get_params()['n_estimators']


#%%
rd_classifier = RandomForestRegressor(max_depth = turnedmax_depth,
                                      max_features=turnedmax_features,
                                      min_samples_leaf=tunredmin_samples_leaf,
                                      min_samples_split=tunredmin_samples_split,
                                      n_estimators=tunredn_estimators
                                      )

clf = rd_classifier.fit(X_train, y_train)
y_pred = clf.predict(X_test)
classifier_r2_score_rd = r2_score(y_test, y_pred)
print(classifier_r2_score_rd)

tunedResult.append((clf.__class__.__name__, classifier_r2_score_rd))

#%%
# gradient boost
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
# #passing the scoring function in the GridSearchCV
clf = GridSearchCV(GradientBoostingClassifier(), parameters,refit=False,cv=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
classifier_r2_score_gb = r2_score(y_test, y_pred)
print(classifier_r2_score_gb)
tunedResult.append((clf.__class__.__name__, classifier_r2_score_gb))
