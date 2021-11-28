# %%

import pandas as pd
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")
y_test = y_test.to_numpy()
y_test = y_test.reshape(y_test.shape[0], )
y_train = y_train.to_numpy()
y_train = y_train.reshape(y_train.shape[0], )

model = XGBClassifier(learning_rate=0.01,
                      max_depth=10,
                      colsample_bytree = 0.7,
                      n_estimators=500)
model.fit(X_train, y_train)
#%%
import seaborn as sns

ax = plot_importance(model)
sns.set(font_scale=0.5)
pyplot.yticks(rotation=45)
pyplot.title("Variable Importance from XGBoost", fontsize=10)

pyplot.savefig("Variable Importance from XGBoost",dpi=500)

pyplot.show()
#%%