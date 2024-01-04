import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA

#Load
dataAll = pd.read_csv('https://raw.githubusercontent.com/ESverdlov/ML/main/Data/PCA_apple_stock.csv')
[rows, cols] = dataAll.shape

data = dataAll.drop(columns = 'Date')
data = data - data.min() + sys.float_info.epsilon
dataLog = np.log(data)

#Find PCA
pcaAll = PCA(n_components = cols - 1)
pcaAll.fit_transform(dataLog)
x = np.zeros(pcaAll.n_components_)
y = np.zeros(pcaAll.n_components_)
sum = 0
for i in range(1, pcaAll.n_components_ + 1):
  x[i - 1] = i
  sum += pcaAll.explained_variance_ratio_[i - 1]
  y[i - 1] = sum
plt.plot(x,y,color='green', marker='o', linestyle='dashed', linewidth=1)
plt.grid(True)
plt.show()
print('Features(log):\n')
print(pcaAll.feature_names_in_,'\n')
print('PCA components:\n')
pca95 = PCA(n_components = 0.95)
dataLog_reduced = pca95.fit_transform(dataLog)
print(pcaAll.components_,'\n')
print('Reduced set:\n')

colsLog = dataLog_reduced.shape[1]
result = np.exp(dataLog_reduced)
print(result)
result_save = pd.DataFrame(data = result)
result_save
result_save.to_csv('Reduced.csv')


