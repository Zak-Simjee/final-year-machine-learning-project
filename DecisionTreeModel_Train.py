import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from joblib import dump
from sklearn.model_selection import train_test_split
from ray.util.joblib import register_ray
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import parallel_backend
#Importing required packages


register_ray()
#Setting ray to be used with parallel_backend

data = pd.read_csv("magpie_supercon.csv")
#Loading data from Supercon


X_train, X_test, y_train, y_test = train_test_split(data.drop(labels=["name", "Tc"], axis=1), data["name"], test_size=0.2, random_state=5050)
#Splitting data to train and test

X_scaler = MinMaxScaler()
y_scaler = OneHotEncoder(handle_unknown="ignore")
#Defining scalers for data


X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)
y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).toarray()
y_test = y_scaler.transform(y_test.values.reshape(-1, 1)).toarray()
#Applying scalers to both train and test data


clf = DecisionTreeRegressor(random_state=5050)
#setting Decsion Tree regressor to variable "clf"


with parallel_backend("ray"):
    clf.fit(X_train, y_train)
#Fitting the regressor to the training data using ray for parallelisation


print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
#Printing Training score and Test accuracy score

dump(clf, "DecisionTreeModel.joblib")
#Saving the model to model.joblib for later use

