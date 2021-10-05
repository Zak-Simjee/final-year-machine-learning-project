import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import load
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
#Importing required packages

clf = load("XGBoostModel.joblib")
#loading the XGBoost model

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


print("Training accuracy score = "+str(clf.score(X_train, y_train)))
print("Testing accuracy score = "+str(r2_score(y_test, clf.predict(X_test))))
print("MSE = "+str(mean_squared_error(y_test, clf.predict(X_test))))
print("RMSE = "+str(mean_squared_error(y_test, clf.predict(X_test), squared=False)))
print("MAE = "+str((mean_absolute_error(y_test, clf.predict(X_test)))))
#Printing out Accuracy and error metrics