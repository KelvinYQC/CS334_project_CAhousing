#%%
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

X_train = pd.read_csv("X_train.csv")
y_train= pd.read_csv("y_train.csv")
X_test= pd.read_csv("X_test.csv")
y_test= pd.read_csv("y_test.csv")
#%%
# y_test.reval(y_test.shape[0],)
#%%

#%%

MLA_compare = pd.DataFrame()
modelList = [
    DecisionTreeRegressor(),
    GradientBoostingRegressor(),
    RandomForestRegressor(),
    LinearRegression()]

def modeling(models, X_train,y_train, X_test, y_test):
    result = list()
    row_index = 0
    for classifier in models:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        classifier_r2_score = r2_score(y_test, y_pred)
        result.append(( classifier.__class__.__name__, classifier_r2_score*100))
        row_index+=1
    return result



performance = modeling(modelList, X_train,y_train, X_test, y_test)
print(performance)
# tune models









#%%
# xgb_model = xgb.XGBRegressor()
# parameters = {
#     "loss":["deviance"],
#     "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
#     "min_samples_split": np.linspace(0.1, 0.5, 12),
#     "min_samples_leaf": np.linspace(0.1, 0.5, 12),
#     "max_depth":[3,5,8],
#     "max_features":["log2","sqrt"],
#     "criterion": ["friedman_mse",  "mae"],
#     "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
#     "n_estimators":[10]
#     }
# #passing the scoring function in the GridSearchCV
# clf = GridSearchCV(GradientBoostingClassifier(), parameters,refit=False,cv=5)
#
# clf.fit(X_train, y_train)
#
# y_pred = clf.predict(X_test)

