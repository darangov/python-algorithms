'''
# Test libraries' versions:
# Cambio
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
'''


# Cargo Librerias
import pandas
from pandas.tools.plotting import scatter_matrix #Matriz de dispersion
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

"""
Formas de cargar datasets
# Load CSV (using python)
import csv
import numpy
filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x).astype('float')
print(data.shape)



# Load CSV with Numpy
import numpy
filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename, 'rt')
data = numpy.loadtxt(raw_data, delimiter=",")
print(data.shape)


# Load CSV from URL using NumPy
from numpy import loadtxt
from urllib.request import urlopen
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
raw_data = urlopen(url)
dataset = loadtxt(raw_data, delimiter=",")
print(dataset.shape)

# Load CSV using Pandas
import pandas
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(filename, names=names)
print(data.shape)


# Load CSV using Pandas from URL
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
print(data.shape)

"""

# Dimensiones del DataSet
# shape-Tamano
print(dataset.shape)
# Primeros 20 registros del dataset
print(dataset.head(20))
# Resumen estadistico del dataset
print(dataset.describe())
# Distribucion por clases
print(dataset.groupby('class').size())

#Visualizacion de la data Box y Whisker plots
color = dict(boxes = 'DarkGreen', whiskers = 'DarkOrange', medians = 'Red', caps = 'Gray')

# Uni variable
dataset.plot(kind='box', subplots='True', layout=(2,2), sharex='False', sharey='False', color = color)
plt.show()

# Univariable tipo Histograma
dataset.hist()
plt.show()

# Diagramas Multivariable
# Scatter Plot Matrix - Grafico de Dispersion
scatter_matrix(dataset)
plt.show()
 
# Divido y separo el datset
# 80% para entrenamiento y validacion del modelo y se
# deja un 20% para Pruebas (datos virgenes)
array = dataset.values # corresponde al 100% de los datos
X = array[:,0:4] # Todas las filas y columnas desde la 0 hasta la posicion 3
Y = array[:,4] # Todas las filas y solo para la columna en posicion 4
validation_size = 0.20 # 20% del dataset para validacion
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size,random_state = seed)
# Funcion model_selection.train_test_split: 
# Split arrays or matrices into random train and test subsets

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
# Chequeo de algoritmos
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluar cada modelo. Se aplica Cross Validation al 80% de los datos que corresponde al 
# entrenamiento, de esta manera se entrena y valida el modelo de manera simultanea, sin
# tocar aun la porcion destinada para pruebas (datos virgenes, sin usar)
# Cross Validation tiene la caracteristica de que simultaneamente entrena y prueba.
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    # Cross Validation con el dataset de entrenamiento - train
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results) # results es una lista, con append le agrego el valor
    # que existe dentro del parentesis - vble cv_results - al final de la lista results.
    
    # Imprimo el cross validation de cada modelo para los k definidos (1. corresponde a 1.000..)
    #print("Modelo: " + name)
    #print("{}".format(cv_results)) #vble dento del () se imprime en formato de texto imprimible dentro de {}
    print("------------------------------")
    names.append(name)
    #resultado = model_selection()
    #msg = "%s: %f (%f) %s" % (name, cv_results.mean(), cv_results.std(), "{}".format(cv_results))
    msg = "%s: %f (%f)" % ("Modelo " + name, cv_results.mean(), cv_results.std())
    print(msg)
    # Imprimo el cross validation de cada modelo para los k definidos (1. corresponde a 1.000..)
    #print("Modelo: " + name)
    print("Exactitud por k-fold: {}".format(cv_results)) #vble dentro del () se imprime en formato de texto imprimible dentro de {}
    print("------------------------------")
    # Accuracy (Exactitud) son sinonimos de Score

# We can also create a plot of the model evaluation results and 
# compare the spread and the mean accuracy of each model. 
# There is a population of accuracy measures for each algorithm because
# each algorithm was evaluated 10 times (10 fold cross validation).

# Compare Algorithm
fig = plt.figure()
fig.suptitle('Comparativo de Algoritmos')
ax = fig.add_subplot(111)
# These are subplot grid parameters encoded as a single integer. 
# For example, "111" means "1x1 grid, first subplot" and "234" means "2x3 grid, 4th subplot".
#Alternative form for add_subplot(111) is add_subplot(1, 1, 1)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
# simbolos con + en grafico son los outliers

# 6 - Make predictions on validation dataset
# Luego de determinar el mejor modelo de seis, para el caso asumo que 
# fue KNN, para mi caso ejecutado fue SVN
 
#Imprime 120 registros que corresponden al 80% de la data orientada al Entrenamiento y Validacion - 
#valores en X - medidas
print("X_train: {}".format(X_train)) 
print("------------------------------")

#Imprime 120 registros que corresponden al 80% de la data - orientada al Entrenamiento y Validacion - 
# valores en Y - Clases asociadas a los valores de X_train - Clase a la que corresponde cada medicion
print("Y_train: {}".format(Y_train)) 
print("------------------------------")

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print "Exactitud: " 
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))