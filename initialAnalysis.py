#%%
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb


X_train = pd.read_csv("X_train.csv")
y_train= pd.read_csv("y_train.csv")
X_test= pd.read_csv("X_test.csv")
y_test= pd.read_csv("y_test.csv")
#%%
y_test = y_test.to_numpy()
y_test = y_test.reshape(y_test.shape[0],)
y_train = y_train.to_numpy()
y_train = y_train.reshape(y_train.shape[0],)

#%%
result = list()
MLA_compare = pd.DataFrame()
modelList = [
    DecisionTreeRegressor(),
    GradientBoostingRegressor(),
    RandomForestRegressor(),
    LinearRegression(),
    xgb.XGBRegressor()]

def modeling(models, X_train,y_train, X_test, y_test):
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


