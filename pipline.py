#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
#%%
# load the dataset
dat = pd.read_csv("housing.csv")
#%%
sns.set(font_scale=0.7)
sns.heatmap(heart_dat.corr(), annot=True, cmap='RdYlBu', fmt='.1f')
plt.savefig('heatmap.png', dpi=900)
plt.show()

# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# construct the pipline
pipeline = make_pipeline(StandardScaler(),
                         RandomForestClassifier(criterion='gini', n_estimators=50, max_depth=2, random_state=1))
#
# Fit the pipeline
#
pipeline.fit(X_train, y_train)
#
# Score the model
#
print('Model Accuracy: %.3f' % pipeline.score(X_test, y_test))