# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")
y_test = y_test.to_numpy()
y_test = y_test.reshape(y_test.shape[0], )
y_train = y_train.to_numpy()
y_train = y_train.reshape(y_train.shape[0], )
# %%
# decision tree
tunedResult = list()
parameterResult = list()
dt = DecisionTreeRegressor()
hyperparameters_dt = [{"splitter": ["best", "random"],
                       #'max_depth': range(1, 80, 5),
                        'max_depth': range(1, 10, 5),
                       'min_samples_leaf': range(1, 700, 40),
                       "max_features": ["auto", "log2", "sqrt", None]}]

dt_gridSearch = GridSearchCV(dt, hyperparameters_dt, cv=10)
best_model_dt = dt_gridSearch.fit(X_train, y_train)
print("decision tree tuned result")
print(best_model_dt.best_estimator_)
parameterResult.append((dt_gridSearch.__class__.__name__, best_model_dt.best_estimator_))

turnedmax_depth = best_model_dt.best_estimator_.get_params()['max_depth']
turnedmax_features = best_model_dt.best_estimator_.get_params()['max_features']
tunredmin_samples_leaf = best_model_dt.best_estimator_.get_params()['min_samples_leaf']

dt_classifier = DecisionTreeRegressor(max_depth=turnedmax_depth,
                                      max_features=turnedmax_features,
                                      min_samples_leaf=tunredmin_samples_leaf)
clf_dt = dt_classifier.fit(X_train, y_train)
y_pred = clf_dt.predict(X_test)
classifier_r2_score_dt = r2_score(y_test, y_pred)
print((clf_dt.__class__.__name__, classifier_r2_score_dt))
tunedResult.append((clf_dt.__class__.__name__, classifier_r2_score_dt))
# %%
# RandomForestRegressor
rd = RandomForestRegressor()
hyperparameters_rd = [{'bootstrap': [True, False],
                       'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, None],
                       'max_features': ['auto', 'sqrt'],
                       'min_samples_leaf': range(1, 500, 50),
                       'n_estimators': [200, 500, 800, 1200, 1400, 1700, 2000, 3000, 4000]}
                      ]

clf = GridSearchCV(rd, hyperparameters_rd, cv=10)
best_model = clf.fit(X_train, y_train)
print("random forest tuned result")
print(best_model.best_estimator_)
parameterResult.append((clf.__class__.__name__, best_model.best_estimator_))

turnedmax_depth = best_model.best_estimator_.get_params()['max_depth']
turnedmax_features = best_model.best_estimator_.get_params()['max_features']
tunredmin_samples_leaf = best_model.best_estimator_.get_params()['min_samples_leaf']
tunredmin_samples_split = best_model.best_estimator_.get_params()['min_samples_split']
tunredn_estimators = best_model.best_estimator_.get_params()['n_estimators']

rd_classifier = RandomForestRegressor(max_depth=turnedmax_depth,
                                      max_features=turnedmax_features,
                                      min_samples_leaf=tunredmin_samples_leaf,
                                      min_samples_split=tunredmin_samples_split,
                                      n_estimators=tunredn_estimators
                                      )

clf_rd = rd_classifier.fit(X_train, y_train)
y_pred = clf_rd.predict(X_test)
classifier_r2_score_rd = r2_score(y_test, y_pred)
print((clf_rd.__class__.__name__, classifier_r2_score_rd))

tunedResult.append((clf_rd.__class__.__name__, classifier_r2_score_rd))

# %%
# gradient boost
parameters = {
    "loss": ["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth": [3, 5, 8],
    "max_features": ["log2", "sqrt"],
    "criterion": ["friedman_mse", "mae"],
    "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    'n_estimators': [10, 50, 100, 200, 500],
}
# #passing the scoring function in the GridSearchCV
clf = GridSearchCV(GradientBoostingClassifier(), parameters, refit=False, cv=10)
best_model = clf.fit(X_train, y_train)
print("gradient boost tuned result")
print(best_model.best_estimator_)
parameterResult.append((clf.__class__.__name__, best_model.best_estimator_))



loss = best_model.best_estimator_.get_params()['loss']
learning_rate = best_model.best_estimator_.get_params()['learning_rate']
min_samples_split = best_model.best_estimator_.get_params()['min_samples_split']
min_samples_leaf = best_model.best_estimator_.get_params()['min_samples_leaf']
max_depth = best_model.best_estimator_.get_params()['max_depth']
max_features = best_model.best_estimator_.get_params()['max_features']
criterion = best_model.best_estimator_.get_params()['criterion']
subsample = best_model.best_estimator_.get_params()['subsample']
n_estimators = best_model.best_estimator_.get_params()['n_estimators']

gb_classifier_clf = GradientBoostingClassifier(loss=loss,
                                               learning_rate=learning_rate,
                                               min_samples_split=min_samples_split,
                                               min_samples_leaf=min_samples_leaf,
                                               max_depth=max_depth,
                                               max_features=max_features,
                                               criterion=criterion,
                                               subsample=subsample,
                                               n_estimators=n_estimators
                                               )
clf_gb = gb_classifier_clf.fit(X_train, y_train)
y_pred = clf_gb.predict(X_test)
classifier_r2_score_gb = r2_score(y_test, y_pred)
print((clf_gb.__class__.__name__, classifier_r2_score_gb))
tunedResult.append((clf_gb.__class__.__name__, classifier_r2_score_gb))
# %%
parameters = {
    'learning_rate': [0.001, 0.01, 0.3, 0.5, 0.7, 1],
    'max_depth': [3, 5, 7, 10, 20, 30, 50],
    'min_child_weight': [1, 3, 5, 7, 10, 20, 50, 100],
    'subsample': [0.3, 0.5, 0.7],
    'colsample_bytree': [0.5, 0.7],
    'n_estimators': [100, 200, 500],
    'objective': ['reg:squarederror', 'reg:squaredlogerror']
}
# #passing the scoring function in the GridSearchCV
clf = GridSearchCV(xgb.XGBRegressor(), parameters, refit=False, cv=10)

best_model = clf.fit(X_train, y_train)
print("XGBoost tuned result")
print(best_model.best_estimator_)
parameterResult.append((clf.__class__.__name__, best_model.best_estimator_))


learning_rate = best_model.best_estimator_.get_params()['learning_rate']
max_depth = best_model.best_estimator_.get_params()['max_depth']
min_child_weight = best_model.best_estimator_.get_params()['min_child_weight']
subsample = best_model.best_estimator_.get_params()['subsample']
colsample_bytree = best_model.best_estimator_.get_params()['colsample_bytree']
n_estimators = best_model.best_estimator_.get_params()['n_estimators']
objective = best_model.best_estimator_.get_params()['objective']

XGB_clf = xgb.XGBRegressor(learning_rate=learning_rate,
                           max_depth=max_depth,
                           min_child_weight=min_child_weight,
                           subsample=subsample,
                           colsample_bytree=colsample_bytree,
                           n_estimators=n_estimators,
                           objective=objective
                           )

clf_xgb = XGB_clf.fit(X_train, y_train)
y_pred = clf_xgb.predict(X_test)
classifier_r2_score_xgb = r2_score(y_test, y_pred)
print((clf_xgb.__class__.__name__, classifier_r2_score_xgb))
tunedResult.append((clf_xgb.__class__.__name__, classifier_r2_score_xgb))
# %%
pd.DataFrame(tunedResult).to_csv('Tuned_Performance_Result.csv', index=False)
pd.DataFrame(parameterResult).to_csv('Tuned_Parameter_Result.csv', index=False)

