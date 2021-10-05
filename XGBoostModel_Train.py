import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
#Importing required packages


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


clf = MultiOutputRegressor(XGBRegressor(n_estimators=10, n_jobs=-1, tree_method='gpu_hist', gpu_id=0, random_state=5050), n_jobs=-1)
#setting "multi-output" XGBoost regressor to variable "clf"


clf.fit(X_train, y_train)
#Fitting the regressor to the training data


print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
#Printing Training score and Test accuracy score

dump(clf, "XGBoostModel.joblib")
#Saving the model to XGBoostModel.joblib for later use