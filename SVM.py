import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load the data
cols = 11
nRows = 15000
dataAll  = pd.read_csv('https://raw.githubusercontent.com/ESverdlov/ML/main/Data/SVMRegression.csv', usecols=np.arange(1,cols+1), nrows = nRows)
dataTrain = dataAll[:round(2*nRows/3)]
dataTest = dataAll[round(2*nRows/3):]



attributes = []
for i in range(cols - 1):
  attributes.append(str(i))


svm_reg = make_pipeline(StandardScaler(), SVR(kernel="rbf",epsilon = 0.1, C = 1))
X = dataTrain[attributes]
y = dataTrain[str(cols - 1)]


svm_reg.fit(X, y)

predict = svm_reg.predict(dataTest[attributes])
Errors = np.zeros(len(predict))
RealValues = dataTest[str(cols - 1)].to_numpy()
eps = 10**(-10) 
Inf = 10**(10)

mult = np.std(RealValues)
for i in range(len(predict)):
  val = RealValues[i]
  if np.abs(predict[i] - val) < eps:
    Errors[i] = 0
  elif mult < eps:
    Errors[i] = Inf
  else:
    e = np.abs((predict[i]-val)**2/mult)
    Errors[i] = e

plt.plot(Errors)
plt.show()
plt.plot(predict)
plt.show()
