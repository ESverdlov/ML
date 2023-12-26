from math import isnan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import export_graphviz
from graphviz import Source


#Load data
def Fill_NaN(datas):
  data = []
  for el in datas:
    if (not np.isnan(el)):
      data = np.append(data, el)
  val = np.median(data)
  for i in range(len(datas)):
    if np.isnan(datas[i]):
      datas[i] = val

dataAll = pd.read_csv('https://raw.githubusercontent.com/ESverdlov/ML/main/Data/ObesityDataSet.csv')
dataAll = dataAll.sample(frac=1, random_state = 0)

#Replace

#Gender
Gender_map = {'Female' : 0, 'Male' : 1}
dataAll['Gender'] = dataAll['Gender'].map(Gender_map)
Fill_NaN(dataAll['Gender'])

#family_history_with_overweight
history_map = {'no': 0, 'yes' : 1}
dataAll['family_history_with_overweight'] = dataAll['family_history_with_overweight'].map(history_map)
Fill_NaN(dataAll['family_history_with_overweight'])

#FAVC
FAVC_map = {'no': 0, 'yes' : 1}
dataAll['FAVC'] = dataAll['FAVC'].map(FAVC_map)
Fill_NaN(dataAll['FAVC'])


#CAEC
CAEC_map = {'Always': 0, 'Frequently': 1, 'no': 2, 'Sometimes': 3}
dataAll['CAEC'] = dataAll['CAEC'].map(CAEC_map)
Fill_NaN(dataAll['CAEC'])


#SMOKE
SMOKE_map = {'no': 0, 'yes' : 1}
dataAll['SMOKE'] = dataAll['SMOKE'].map(SMOKE_map)
Fill_NaN(dataAll['SMOKE'])

#SCC
SCC_map = {'no': 0, 'yes' : 1}
dataAll['SCC'] = dataAll['SCC'].map(SCC_map)
Fill_NaN(dataAll['SCC'])


#CALC
SCC_map = {'Always': 0, 'Frequently': 1, 'no': 2, 'Sometimes': 3}
dataAll['CALC'] = dataAll['CALC'].map(SCC_map)
Fill_NaN(dataAll['CALC'])


#MTRANS
MTRANS_map = {'Automobile': 0, 'Bike': 1, 'Motorbike': 2, 'Public_Transportation': 3, 'Walking': 4}
dataAll['MTRANS'] = dataAll['MTRANS'].map(MTRANS_map)
Fill_NaN(dataAll['MTRANS'])

#NObeyesdad
NObeyesdad_map = {'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2, 'Overweight_Level_II': 3,
                  'Obesity_Type_I': 4, 'Obesity_Type_II': 5, 'Obesity_Type_III': 6}
dataAll['NObeyesdad'] = dataAll['NObeyesdad'].map(NObeyesdad_map)

Fill_NaN(dataAll['NObeyesdad'])

rows = int(dataAll.shape[0] * 0.7)
cols = dataAll.shape[1]

dataAll_array = dataAll.values

XY_Train = dataAll_array[:rows]
XY_Test = dataAll_array[rows:]

y_Train = np.zeros(rows)
X_Train = np.zeros([rows, cols - 1])
for i in range(len(XY_Train)):
  y_Train[i] = XY_Train[i][cols - 1]
  for j in range(cols - 1):
    X_Train[i][j] = XY_Train[i][j]

y_Test = np.zeros(dataAll.shape[0] - rows)
X_Test = np.zeros([dataAll.shape[0] - rows, cols-1])
for i in range(len(XY_Test)):
  y_Test[i] = XY_Test[i][cols - 1]
  for j in range(cols - 1):
    X_Test[i][j] = XY_Test[i][j]

#Decision tree
clf = DecisionTreeClassifier(max_depth=7, random_state=0)
clf.fit(X_Train, y_Train)

y_pred = clf.predict(X_Test)

#Confusion matrices

display_labels = ["Insufficient\nWeight", "Normal\nWeight", "Overweight\nLevel I",
                  "Overweight\nLevel II", "Obesity\nType I", "Obesity\nType II",
                  "Obesity\nType III"]
fig, ax = plt.subplots(figsize=(22,10))
plt.rc('font', size=10)
ax.set_title("Numbers")

ConfusionMatrixDisplay.from_predictions(y_Test, y_pred, ax = ax, display_labels = display_labels)
                                            
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(22,10))
plt.rc('font', size=14)
ax[0].set_title("Confusion Matrix. On the diagonal there are recalls.")
ConfusionMatrixDisplay.from_predictions(y_Test, y_pred,
                                            values_format = ".0%", normalize = "true",
                                            display_labels = display_labels, ax = ax[0], colorbar = False)


ax[1].set_title("Confusion Matrix. On the diagonal there are precisions.")
ConfusionMatrixDisplay.from_predictions(y_Test, y_pred,
                                           values_format = ".0%", normalize = "pred",
                                           display_labels = display_labels, ax = ax[1], colorbar = False)
plt.show()
#Output
feature_names = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC',
                  'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']
export_graphviz(
  clf,
  out_file = "Obesity.dot",
  class_names = display_labels,
  feature_names = feature_names,
  rounded = True,
  filled = True
)
Source.from_file("Obesity.dot")