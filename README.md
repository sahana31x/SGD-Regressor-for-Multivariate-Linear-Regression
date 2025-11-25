# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import necessary libraries and dataset. 
2.Extract relevant features and target variables from the dataset. 
3.Split the dataset into training and testing subsets. 
4.Standardize the training and testing data using scalers. 
5.Initialize the SGDRegressor and wrap it with MultiOutputRegressor. 
6.Train the model on the standardized training data. 
7.Make predictions on the test data. 
8.Inverse transform the predictions and actual target values. 
9.Evaluate the model using the Mean Squared Error metric.
10.Display the Mean Squared Error and sample predictions.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SAHANA G
RegisterNumber: 25018306

  import numpy as np
  from sklearn.datasets import fetch_california_housing
  from sklearn.linear_model import SGDRegressor
  from sklearn.multioutput import MultiOutputRegressor
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import mean_squared_error
  from sklearn.preprocessing import StandardScaler

  data = fetch_california_housing()

  X= data.data[:, :3] #features: 'Medinc','housage','averooms'
  Y=np.column_stack((data.target,data.data[:, 6]))
  x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

  scaler_x = StandardScaler()
  scaler_y = StandardScaler()
  x_train = scaler_x.fit_transform(x_train)
  x_test = scaler_x.transform(x_test)
  y_train = scaler_y.fit_transform(y_train)
  y_test = scaler_y.transform(y_test)

  sgd = SGDRegressor(max_iter = 1000,tol = 1e-3)

  multi_output_sgd = MultiOutputRegressor(sgd)

  multi_output_sgd.fit(x_train,y_train)

  y_pred = multi_output_sgd.predict(x_test)

  y_pred = scaler_y.inverse_transform(y_pred)
  y_test = scaler_y.inverse_transform(y_test)

  mse = mean_squared_error(y_test,y_pred)
  print("Mean Squared Error:",mse)

  print("\npredictions:\n",y_pred[:5])
```

## Output:

<img width="533" height="247" alt="Screenshot (134)" src="https://github.com/user-attachments/assets/e22400a7-19cf-477e-9470-bf64923e246d" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
