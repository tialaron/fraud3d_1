import streamlit as st
import numpy as np # библиотека для работы с массивами данных
import pandas as pd # библиотека для анализа и обработки данных
import matplotlib.pyplot as plt # из библиотеки для визуализации данных
#import seaborn as sns

#from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split # модуль для разбивки выборки на тренировочную/тестовую
from sklearn.preprocessing import StandardScaler # модуль для стандартизации данных
from tensorflow.keras.models import Sequential,load_model

st.title("3D отображение внутренней работы автокодировщика")
'Вашему вниманию предлагается модель распределения транзакций на нормальные (синие) и мошеннические (оранжевые).'
'Распределение проводилось с помощью заранее обученного автокодировщика с 3 нейронами внутри скрытого среднего слоя.'
'Именно поэтому у нас график трехмерный.Создание автокодировщика с более мощным скрытым слоем может разграничить транзакции более точно.'
df = pd.read_csv("H:\Pythonprojects\\frauds3d\\venv\creditcard1.csv") # читаем базу

# Удаляем столбец со временем
data_12 = df.drop(['Time'], axis=1)
# Нормализуем столбец с суммой транзакции
data_12['Amount'] = StandardScaler().fit_transform(data_12['Amount'].values.reshape(-1, 1))

frauds = data_12[data_12.Class == 1] # записываем мошеннические операции
normal = data_12[data_12.Class == 0] # записываем нормальные операции

# Удаляем класс в нормальном наборе данных
X_normal = normal.drop(['Class'], axis=1)
# Удаляем класс в мошеннических операциях
X_frauds = frauds.drop(['Class'], axis=1)

# Преобразуем данные в массивы numpy
X_normal_arr = X_normal.values
X_frauds_arr = X_frauds.values
#Загружаем сеть
model = load_model('H:\Pythonprojects\\frauds3d\\venv\\autocoder2_3d.h5')
#Из первых обученных слоев этой сети создаем другую
hid_rep = Sequential()
hid_rep.add(model.layers[0])
hid_rep.add(model.layers[1])
hid_rep.add(model.layers[2])
hid_rep.add(model.layers[3])
hid_rep.add(model.layers[4])

norm_hid_rep = hid_rep.predict(X_normal_arr[:3000])
fraud_hid_rep = hid_rep.predict(X_frauds_arr)

norm_hid_rep_set = pd.DataFrame(norm_hid_rep, columns=['x','y','z'])
fraud_hid_rep_set = pd.DataFrame(fraud_hid_rep, columns=['x','y','z'])
norm_hid_rep_set['class'] = 'normal'
fraud_hid_rep_set['class'] = 'fraud'


#Дальше пойдет отображение графика
azim = st.sidebar.slider("azim", 0, 90, 30, 1)
elev = st.sidebar.slider("elev", 0, 360, 240, 1)

fig = plt.figure()
ax = fig.gca(projection='3d')
x_n = norm_hid_rep_set.x
y_n = norm_hid_rep_set.y
z_n = norm_hid_rep_set.z
x_f = fraud_hid_rep_set.x
y_f = fraud_hid_rep_set.y
z_f = fraud_hid_rep_set.z
ax.scatter3D(x_n,y_n,z_n)
ax.scatter3D(x_f,y_f,z_f)
#ax.voxels(voxels, facecolors=colors, edgecolor='k')
ax.view_init(azim, elev)

st.pyplot(fig)
