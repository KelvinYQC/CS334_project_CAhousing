# extract coefficient from simple regression model
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
# load datasets
X_train = pd.read_csv("../X_train.csv")
y_train = pd.read_csv("../y_train.csv")
X_test = pd.read_csv("../X_test.csv")
y_test = pd.read_csv("../y_test.csv")
y_test = y_test.to_numpy()
y_test = y_test.reshape(y_test.shape[0], )
y_train = y_train.to_numpy()
y_train = y_train.reshape(y_train.shape[0], )

# run the classifier to get coefficient
classifier = LinearRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
classifier_r2_score = r2_score(y_test, y_pred)
# find coefficient
cdf = pd.DataFrame(classifier.coef_, X_test.columns, columns=['Coefficients'])
print(cdf)
# export to csv
cdf.to_csv('regressionCoeffcient.csv')
