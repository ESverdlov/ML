import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the data
cols = 11
nRows = 15000
dataAll  = pd.read_csv('https://raw.githubusercontent.com/ESverdlov/ML/main/Data/PolynomialData.csv', usecols=np.arange(1,cols+1), nrows = nRows)
dataTrain = dataAll[:round(2*nRows/3)]
dataTest = dataAll[round(2*nRows/3):]

attributes = []
for i in range(cols - 1):
  attributes.append(str(i))

poly_features = PolynomialFeatures(degree=1, include_bias=False)
dataTrain_poly = poly_features.fit_transform(dataTrain[attributes])

lin_reg = LinearRegression()
lin_reg.fit(dataTrain_poly, dataTrain[str(cols - 1)])

dataTest_poly = poly_features.fit_transform(dataTest[attributes])

predict = lin_reg.predict(dataTest_poly)
Errors = np.zeros(len(predict))
RealValues = dataTest[str(cols - 1)].to_numpy()

mult = np.std(RealValues)
eps = 10**(-10)
Inf = 10**(10)
for i in range(len(predict)):
  val = RealValues[i]
  if np.abs(predict[i] - val) < eps:
    Errors[i] = 0
  elif mult < eps:
    Errors[i] = Inf
  else:
    e = np.abs((predict[i]-val)**2/mult)
    Errors[i] = e

plt.plot(RelErrors)
plt.show()
plt.plot(predict)
plt.show()
