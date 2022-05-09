import pandas as pd
import numpy as np
#usiamo il dataset Boston
boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", sep='\s+',
usecols=[5,13], names=["RM", "MEDV"])
boston.head()

x = boston.drop("MEDV", axis=1).values
y = boston["MEDV"].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)

#validità del modello, errore quadratico medio

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)

#coefficente di determinazione [mse], è una funzione di punteggio non di costo
#piu vicino siamo a 1 migliore è il nostro modello

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
