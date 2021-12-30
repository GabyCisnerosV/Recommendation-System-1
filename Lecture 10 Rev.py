# -*- coding: utf-8 -*-
"""
Data Preprocessing and Machine Learning
Examples from the lecture

@author: Manuel Lopez-Ibanez <manuel.lopez-ibanez@manchester.ac.uk>

This file is meant to be executed line by line and not all at once.
Use the IPython console and F9 in Spyder.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor

path = 'C:/Users/gabri/Dropbox/Gaby/UoM Msc Data Science/Semester 1/Programming in Python for Business Analytics/Scripts/Revisions/Lecture 10 Rev/'
# Set a seed so the results are somewhat replicable
np.random.seed(1)






#==============================================================================
# Gaussian Processes
#==============================================================================

from sklearn.gaussian_process import GaussianProcessRegressor




#Esta es mi funcion. En la vida normal no se cual es
def f(x):
    """The function to predict."""
    return x * np.sin(x)




# ----------------------------------------------------------------------
#Esto corre e nuevo asi hagas manual o funcion siempre para resetear X
#  Create 1-column, n rows 2D matrix
# Estos son los puntos que se van a ir aumentando cada vez
X = np.transpose([[1., 3., 5., 6., 7., 8.]])
X


# Mesh the input space for evaluations of the real function, the prediction and
# its stdandard deviation
# 1-column, n rows 2D matrix
x_test = np.linspace(0, 10, 1000)[:, np.newaxis]  
#Return evenly spaced numbers over a specified interval. De 0 al 10, 1000 numeros
x_test
fx_test = f(x_test)
fx_test

# ----------------------------------------------------------------------

#Manual
#Vas cambiando el numero de la fila 66

X = X[:5, :]        # 1 Fit to data using Maximum Likelihood Estimation of the parameters
                    # Si pones 1, entra el numero 1. Si pones 2 entra el primero y el segundo y asi
X

y = f(X).ravel()    #2 Observations: convert to 1D vector
y

                    #3 FIT: Instantiate a Gaussian Process model
gp = GaussianProcessRegressor()
gp.fit(X, y)

                    #4 PREDICT: Make the prediction on the whole x-axis (ask for std as well)
y_pred, sigma = gp.predict(x_test, return_std=True)
y_pred
sigma
#sigma es apra el intervalo de conianza
# 95% confidence interval
low = y_pred - 1.96 * sigma
high= y_pred + 1.96 * sigma 


                    #5 Plot the function, the prediction and the 95% confidence interval based on
plt.figure()
plt.plot(x_test, fx_test, 'r:', label=u'$f(x)$')                        #FUNCION ROJA
plt.plot(X, y, 'r.', markersize=10, label=u'Observations')              #PUNTO ROJO
plt.plot(x_test, y_pred, 'b-', label=u'Prediction')                     #LINEA AZUL
plt.fill(np.concatenate([x_test, x_test[::-1]]),                        #INTERVALO DE CONFIANZA
         np.concatenate([low, high[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')                                                       #LABEL X
plt.ylabel('$y=f(x)$')                                                  #LABEL Y
plt.ylim(-10, 20)                                                       #LIMITES DE Y
plt.legend(loc='upper left')                                            #LEGEND
plt.tight_layout()
plt.savefig("gp-" + str(2) + ".png")
plt.show()

#Ahora cambias por todos los que estan en X: 1, 2, 5, 6 



# ----------------------------------------------------------------------
#Ahora hecho funcion
def gp_fit_and_plot(k, X_orig, x_test, fx_test):
    # Fit to data using Maximum Likelihood Estimation of the parameters
    X = X_orig[:k, :]
    
    y = f(X).ravel() # Observations: convert to 1D vector
    # Instantiate a Gaussian Process model
    gp = GaussianProcessRegressor()
    gp.fit(X, y)
    # Make the prediction on the whole x-axis (ask for std as well)
    y_pred, sigma = gp.predict(x_test, return_std=True)
    # 95% confidence interval
    low = y_pred - 1.96 * sigma
    high= y_pred + 1.96 * sigma 
    
    # Plot the function, the prediction and the 95% confidence interval based on
    # the std
    plt.figure()
    plt.plot(x_test, fx_test, 'r:', label=u'$f(x)$')
    plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
    plt.plot(x_test, y_pred, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([x_test, x_test[::-1]]),
             np.concatenate([low, high[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$y=f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("gp-" + str(k) + ".png")
    #plt.show()



for k in [1,2,5,6]:
    gp_fit_and_plot(k, X, x_test, fx_test)















#==============================================================================
# Random Forest
#==============================================================================
np.random.seed(42)
df = pd.read_csv(path+"card.csv")
df.info()

#Queremos saber cuantos hacen default

print(df['default'].value_counts())


#Split the data into test and train con stratify tambien
X_train, X_test, y_train, y_test = train_test_split(
        df.drop('default', axis=1),
        df['default'],
        test_size = 0.20,
        stratify = df['default'])

from sklearn.ensemble import RandomForestClassifier



#-----------------------------------------------------------------------------

#Haciendo sin cross validation, solo 1 con stratify

#Haciendo el fit y prediction
# n_estimators: number of trees
forest = RandomForestClassifier(n_estimators = 5)
forest.fit(X_train, y_train)
forest.score(X_train, y_train)
#0.9709166666666667
forest.score(X_test, y_test) #Este es el que me sirve para ver que tal, el de test data
#0.7896666666666666


#Sacando los attributes o features importantes
importances = forest.feature_importances_
indices = np.argsort(-importances)
# Create a dataframe just for pretty printing
df_imp = pd.DataFrame(dict(feature=X_train.columns[indices],
                           importance = importances[indices]))
df_imp.head()

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(importances)), df_imp['importance'], color="b", align="center")
plt.xticks(range(len(importances)), df_imp['feature'], rotation=90)
plt.tight_layout()
plt.show()





#-----------------------------------------------------------------------------

#Haciendo con cross validation, 3 modelos viendo cual es el mejor
#Este le hiciste tu apra probar

uno = RandomForestClassifier(n_estimators = 4)
dos = RandomForestClassifier(n_estimators = 5)
tres = RandomForestClassifier(n_estimators = 6)

scores1 = cross_val_score(uno, X_train, y_train, cv=5)
scores2 = cross_val_score(dos, X_train, y_train, cv=5) 
scores3 = cross_val_score(tres, X_train, y_train, cv=5) 

#Comprobado que funciona esta funcion para saber intrvalos de confianza
for i in [scores1,scores2,scores3]:
    print("{}-fold CV score: {:.2f} (+/- {:.2f})"
      .format(len(i), i.mean(), i.std() * 2))
    
#Es mejor el 3ero
tres.fit(X_train, y_train)
tres.score(X_test, y_test)

#Sacando los attributes o features importantes
importances = tres.feature_importances_
indices = np.argsort(-importances)
# Create a dataframe just for pretty printing
df_imp = pd.DataFrame(dict(feature=X_train.columns[indices],
                           importance = importances[indices]))
df_imp.head()

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(importances)), df_imp['importance'], color="b", align="center")
plt.xticks(range(len(importances)), df_imp['feature'], rotation=90)
plt.tight_layout()
plt.show()












# =============================================================================
# Metrics
# =============================================================================
forest.score(X_test, y_test)

from sklearn.metrics import confusion_matrix
y_test_pred = forest.predict(X_test)
conf_mat = confusion_matrix(y_test, y_test_pred, labels=[0,1])
print(conf_mat)

# (TP + TN) / (TP + TN + FP + FN)
print("Accuracy:", (conf_mat[0,0] + conf_mat[1,1]) / np.sum(conf_mat))
# TP / (TP + FP)
print("Precision:", conf_mat[1,1] / np.sum(conf_mat[:,1]))
# TP / (TP + FN)
print("Recall:", conf_mat[1,1] / np.sum(conf_mat[1,:]))

from sklearn.metrics import accuracy_score, precision_score,\
                            recall_score, f1_score
print("Accuracy:", accuracy_score(y_test, y_test_pred)) # Es el mismo que te dice en score
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred))
print("F1 score:", f1_score(y_test, y_test_pred))

















#==============================================================================
# Random Forest multi-class
#==============================================================================

np.random.seed(42)
nursery = pd.read_csv(path+"nursery-data.csv")
nursery.info()

#funcion solo apra describir dataset
def my_describe_dataframe(df):
    for col in df.columns:
        if df.dtypes[col] == object:
            # Convert to list to print in one-line
            print(col + " : " + str(df[col].unique().tolist()))
        else:
            # Convert to dictionary to print in one-line but keep names
            print(df[col].describe().to_dict())
            
my_describe_dataframe(nursery)

#Solo para contar cuantos de cada clase
nursery.groupby("Application_Status")["Application_Status"].count()




# Encode labels as integers
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
nursery_enc = nursery.copy()

nursery_enc
nursery.select_dtypes(include=object).columns #Columnas del tipo object

for col in nursery.select_dtypes(include=object).columns:
    label_encoders[col] = LabelEncoder()
    nursery_enc[col] = label_encoders[col].fit_transform(nursery[col])
  
    
  
    
    
# Train/test split
X_train, X_test, y_train, y_test = \
    train_test_split(nursery_enc.drop('Application_Status', axis=1),
                     nursery_enc['Application_Status'],
                     test_size = 0.20,
                     stratify = nursery_enc['Application_Status'])


# Fit and train model
from sklearn.ensemble import RandomForestClassifier
# n_estimators: number of trees
forest = RandomForestClassifier(n_estimators = 5)
forest.fit(X_train, y_train)
forest.score(X_test, y_test) #0.9467592592592593 es muy bueno

#Top columnas importantes
importances = forest.feature_importances_
indices = np.argsort(-importances)
# Create a dataframe just for pretty printing
df_imp = pd.DataFrame(dict(feature=X_train.columns[indices],
                           importance = importances[indices]))
df_imp.head()


# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(importances)), df_imp['importance'], color="b", align="center")
plt.xticks(range(len(importances)), df_imp['feature'], rotation=90)
plt.tight_layout()
plt.show()


#Function de confusion matrix
# From https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
from sklearn.utils.multiclass import unique_labels
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title ='Confusion matrix, without normalization'
    
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the code
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    # ... and label them with the respective list entries
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    return ax
    

#Te hago acuerdo del score
forest.score(X_test, y_test)

from sklearn.metrics import confusion_matrix
y_test_pred = forest.predict(X_test)
conf_mat = confusion_matrix(y_test, y_test_pred)
print(conf_mat)

y_classes_names = label_encoders['Application_Status'].classes_
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_test_pred, classes=y_classes_names)

print(nursery['Application_Status']
      .value_counts()) #hay solo 2 de recommend, por eso no aparece en la amtrix

from sklearn.metrics import classification_report
print(
  classification_report(
      label_encoders['Application_Status'].inverse_transform(y_test),
      label_encoders['Application_Status'].inverse_transform(y_test_pred)
      )
  )











# =============================================================================
# Grid Search
# =============================================================================
np.random.seed(42)

from sklearn.datasets import load_boston
boston = load_boston()
print(boston.DESCR)

from sklearn.model_selection import train_test_split

# 1. Split train/test
X_train, X_test, y_train, y_test = \
    train_test_split(boston.data, boston.target, test_size = 0.10)
    
# Standardization (z-score normalization)
from sklearn.preprocessing import StandardScaler
# We fit the scaler on the train data and apply to both train and test
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural network con un setting especifico
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=10)

# K-fold cross-validation
scores = cross_val_score(mlp, X_train_scaled, y_train, cv=10)
print("Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Refit using all training data
mlp.fit(X_train_scaled, y_train)
mlp.score(X_test_scaled, y_test)

# Parameters and values to tune
param_grid = dict(hidden_layer_sizes= [10, 25, (5,5), (10,10)],
                  solver = ['lbfgs','sgd'])
                     

from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(mlp, param_grid, cv=10, n_jobs=2)
# Run the search
gs.fit(X_train_scaled, y_train)
print("Best parameters found:", gs.best_params_)
print("Mean CV score of best parameters:", gs.best_score_)
# Before calculating the score, the model is refit using all training data.
print("Score of best parameters on test data:",
      gs.score(X_test_scaled, y_test))

means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
params = gs.cv_results_['params']
for mean, std, param in zip(means, stds, params):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, param))












# =============================================================================
# RandomSearch
# =============================================================================
np.random.seed(42)

from timeit import default_timer as timer

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: ", i)
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: ", results['params'][candidate])
            print("")

import scipy.stats as sp
# sp.randint generates a random number generator
param_dist = dict(hidden_layer_sizes = sp.randint(5,10,(1,1)),
                  solver = ['lbfgs','sgd'])


from sklearn.model_selection import RandomizedSearchCV
n_iter_search = 10
random_search = RandomizedSearchCV(
        MLPRegressor(), param_distributions=param_dist,
        n_iter=n_iter_search, cv=10, n_jobs = 2)

start = timer()
random_search.fit(X_train_scaled, y_train)
print("RandomizedSearchCV took {:.2f} seconds for {} candidate"
      " parameter settings.".format(timer() - start, n_iter_search))

report(random_search.cv_results_)

print("Best parameters set found:", random_search.best_params_)
print("Mean CV score of best parameters:", random_search.best_score_)
# Before calculating the score, the model is refit using all training data.
print("Score of best parameters on test data:",
      random_search.score(X_test_scaled, y_test))















# =============================================================================
# Example of hyper-parameter optimization: Random Search
# =============================================================================
df = pd.read_csv(path+"card.csv")
df.info()

print(df['default'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
        df.drop('default', axis=1),
        df['default'],
        test_size = 0.20,
        stratify = df['default'])

from sklearn.ensemble import RandomForestClassifier
# n_estimators: number of trees
forest = RandomForestClassifier(n_estimators = 5)
# K-fold cross-validation
scores = cross_val_score(forest, X_train, y_train, cv=10)
print("%i-fold CV Score: %0.2f (+/- %0.2f)"
      % (len(scores), scores.mean(), scores.std() * 2))


#Si quiero cambiar el scorer dentro de la cross validation
from sklearn.metrics import make_scorer, f1_score
f1_scorer = make_scorer(f1_score)

# K-fold cross-validation
scores = cross_val_score(forest, X_train, y_train, cv=10, scoring = f1_scorer)
print("%i-fold CV F1-score: %0.2f (+/- %0.2f)"
      % (len(scores), scores.mean(), scores.std() * 2))




#Empiezas a hacer el random search
import scipy.stats as sp

# sp.randint generates a discrete random number generator (RNG)
# sp.uniform generates a continuous uniform RNG
param_dist = dict(n_estimators = sp.randint(5,20, 1),
                  # None is the default value and means maximum depth
                  max_depth = [10, 50, 100, None])              

from sklearn.model_selection import RandomizedSearchCV
n_iter_search = 10 #numero de modelos que va a probar

randsearch = RandomizedSearchCV(
        forest, param_distributions = param_dist,
        n_iter = n_iter_search, cv = 10, n_jobs = 2)

randsearch.fit(X_train, y_train)
randsearch.cv_results_
print("Best parameters set found:", randsearch.best_params_)
print("Mean score of best parameters:", randsearch.best_score_)

from sklearn.metrics import make_scorer, f1_score
f1_scorer = make_scorer(f1_score)
randsearch = RandomizedSearchCV(
        forest, param_distributions = param_dist,
        n_iter = n_iter_search, cv = 10, n_jobs = 2,
        scoring = f1_scorer)
randsearch.fit(X_train, y_train)
print("Best parameters set found:", randsearch.best_params_)
print("Mean F1-score of best parameters:", randsearch.best_score_)















# =============================================================================
# Pipelines
# =============================================================================

# The above examples are not completely correct 
# because we fitted the StandardScaler using
# the whole training data, however, when doing cross-validation only part of
# the training data is used as training data, the rest is used for validation.
np.random.seed(42)

from sklearn.datasets import load_boston
boston = load_boston()
print(boston.DESCR)

from sklearn.model_selection import train_test_split

# 1. Split train/test
X_train, X_test, y_train, y_test = \
    train_test_split(boston.data, boston.target, test_size = 0.10)
    
# Standardization (z-score normalization)
from sklearn.preprocessing import StandardScaler
# We fit the scaler on the train data and apply to both train and test
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.neural_network import MLPRegressor

import scipy.stats as sp
# sp.randint generates a random number generator
param_dist = dict(hidden_layer_sizes = sp.randint(5,10,(1,1)),
                  solver = ['lbfgs','sgd'])

from sklearn.model_selection import RandomizedSearchCV
n_iter_search = 10
random_search = RandomizedSearchCV(
        MLPRegressor(), param_distributions=param_dist,
        n_iter=n_iter_search, cv=10, n_jobs=2)

random_search.fit(X_train_scaled, y_train)
print("Best parameters set found:", random_search.best_params_)
print("Mean CV score of best parameters:", random_search.best_score_)



from sklearn.pipeline import Pipeline
pipeline = Pipeline([('stdscale', StandardScaler()),
                     ('mlp',  MLPRegressor())])
print(pipeline.named_steps)

print(pipeline.named_steps['stdscale'])

# Prefix hyper-parameters with mlp__
pipe_param_dist = dict(mlp__hidden_layer_sizes = sp.randint(5,10,(1,1)),
                       mlp__solver = ['lbfgs','sgd'])

random_search = RandomizedSearchCV(pipeline,
                                   param_distributions=pipe_param_dist,
                                   n_iter=n_iter_search, cv=10, n_jobs = 2,
                                   random_state=42)
random_search.fit(X_train, y_train)
print("Best parameters set found:", random_search.best_params_)
print("Mean CV score of best parameters:", random_search.best_score_)

# Quiz

from sklearn.metrics import accuracy_score, precision_score,\
                            recall_score, f1_score
                            
y_true = [0,1,0,1,1,0,1,1,1,1]
y_pred = [0,1,1,0,1,1,0,0,1,1]
confusion_matrix(y_true, y_pred)
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 score:", f1_score(y_true, y_pred))