#%%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder


def correlationAnalysis(dat, name):
    sns.set(font_scale=0.5)
    correlation = dat.corr()
    sns.heatmap(correlation, xticklabels=dat.columns, yticklabels=dat.columns, cmap="YlGnBu", annot=True, fmt='.2f')
    plt.xticks(rotation=20)
    plt.yticks(rotation=45)
    plt.title("%s" % name, fontsize = 10)
    plt.savefig("%s.png" % name, dpi=500)

    #plt.show()
    plt.close()


def variableImputation(dat):

    dat = pd.get_dummies(dat)
    dat.rename(columns={'ocean_proximity_<1H OCEAN': 'Ocean',
                        'ocean_proximity_INLAND': 'Inland',
                        'ocean_proximity_ISLAND': 'Island',
                        'ocean_proximity_NEAR BAY': 'Near_Bay',
                        'ocean_proximity_NEAR OCEAN': 'Near_Ocean',
                        'median_house_value': 'outcome',

                        }, inplace=True)

    return dat



def locationVar(dat, city_lat_long, city_pop_data):
    combined = city_lat_long.merge(city_pop_data, left_on='Name', right_on='City', how='left')
    median_pop = combined['pop_april_1990'].median()
    big_city = combined[combined["pop_april_1990"] > median_pop]
    #big_city = combined
    results = list()
    for i in range(dat.shape[0]):
        distance = float('inf')
        houseLocation = (dat.iloc[i,1], dat.iloc[i,0])
        result = "none"
        for j in range(big_city.shape[0]):
            cityLocation =(big_city.iloc[j,1], big_city.iloc[j,2])
            newDist = geodesic(houseLocation, cityLocation)
            if newDist < distance:
                distance = newDist
                result = big_city.iloc[j,0]
        results.append(result)
    print("city classfication done")
    dat['city'] = results
    encoder = LabelEncoder()
    dat['city'] = encoder.fit_transform(dat['city'])

    return dat

def split(dat):
    X = dat.drop(["outcome"], axis=1)
    y = dat['outcome']
    # train test spli
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test




def main():
    dat = pd.read_csv("Datasets/housing.csv")
    city_lat_long = pd.read_csv('Datasets/cal_cities_lat_long.csv')
    city_pop_data = pd.read_csv('Datasets/cal_populations_city.csv')


    correlationAnalysis(dat,"Correlation for Initial Dataset")
    print("Correlation for original dataset")
    dat_noNA = variableImputation(dat)
    print("NA dropping and impuing Done")
    correlationAnalysis(dat_noNA, "Correlation for Imputed Dataset")
    print("Correlation for imputed dataset")
    dat_location = locationVar(dat_noNA, city_lat_long, city_pop_data)
    print("Location Done")


    correlationAnalysis(dat_location, "Correlation for Engineered Dataset")


    X_train, X_test, y_train, y_test = split(dat_location)
    scaler = StandardScaler()
    scaler.fit(X_train)
    xtrain_scaled = scaler.transform(X_train)
    xtest_scaled = scaler.transform(X_test)

    scaled_features_train = pd.DataFrame(xtrain_scaled, index=X_train.index, columns=X_train.columns)
    scaled_features_test = pd.DataFrame(xtest_scaled, index=X_test.index, columns=X_test.columns)
    correlationAnalysis(scaled_features_train, "Correlation for Engineered X Train Dataset")
    correlationAnalysis(scaled_features_test, "Correlation for Engineered X Test Dataset")

    print("Exporting to CSV")
    scaled_features_train.to_csv('X_train.csv', index=False)
    scaled_features_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

if __name__ == "__main__":
    main()
