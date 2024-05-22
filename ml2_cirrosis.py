
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.model_selection import KFold,LeaveOneOut,StratifiedKFold

#logistic

import matplotlib
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn import preprocessing, model_selection, linear_model
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_predict
from pandas.plotting import scatter_matrix
from sklearn.metrics import accuracy_score,make_scorer
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_colwidth', None)

# Cargar el csv
def load():
  path="./cirrhosis.csv"
  df=pd.read_csv(path,delimiter=',')
  return df

# Analisis exploratorio
# Graficar una variables
def plotVariable(data):
  print(data.info())
  print(data.describe())
  plt.figure(figsize=(15,5))
  plt.plot(data['Status'])
  plt.title('Status.', fontsize=15)
  plt.ylabel('Status')
  plt.show()

# Grafico de barras variables categoricas
def analisisCategoricas(df):
  varCategoricas = df.select_dtypes(exclude=np.number).columns
  print("Categoricas: ", varCategoricas)
  n_vars = len(varCategoricas)
  n_cols = 3
  n_rows = (n_vars + n_cols - 1) // n_cols  if  (n_vars + n_cols - 1) // n_cols > 0 else 1
  fig, axis = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
  index = 0
  for i in range(n_rows):
    for j in range(n_cols):
      if index < n_vars:
        ax = sns.countplot(x=varCategoricas[index], data=df, ax=axis[i][j])
        if varCategoricas[index] in ['job']:
          for item in ax.get_xticklabels():
            item.set_rotation(15)
        for p in ax.patches:
          height = p.get_height()
          ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/len(df)*100),
            ha="center")
        index += 1
      else:
        axis[i, j].set_visible(False)
  plt.tight_layout()
  plt.show()

def plotCategorica(datos):
  valores_unicos, recuentos = np.unique(datos, return_counts=True)

  # Crear el gráfico de barras
  plt.bar(valores_unicos, recuentos)

  # Agregar etiquetas y título
  plt.xlabel('Valor')
  plt.ylabel('Recuento')
  plt.title('Recuento de valores')

  # Mostrar el gráfico
  plt.show()

# Histograma y diagrama de caja para variables numericas
def analisisNumericas(df):
  varNumericas = df.select_dtypes(include=np.number).columns
  print("Numericas: ", varNumericas)
  for numerica in varNumericas:
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    df[numerica].plot.hist(bins=25, ax=axes[0])
    axes[0].set_title(f'Histograma de {numerica}', fontdict={'fontsize': 16})
    df[numerica].plot.box(ax=axes[1])
    axes[1].set_title(f'Boxplot de {numerica}', fontdict={'fontsize': 16})
    plt.show()
    print("\n")

# Analisis de nulos
def nullAnalysis(df):
  null_columns=df.isnull().any()
  print("Nulos en columnas:")
  print(null_columns)
  null_sum = df.isnull().sum()
  print("Suma de nulos:")
  print(null_sum)
def dropNAinDrug(df):
  df = df.dropna(subset=['Drug'])
  return df
def imputeWithMode(df):
  df=df.fillna(df.mode().iloc[0])
  return df
# Probar inputacion con MICE
def imputeWithMICE():
  pass

# Escalamiento

def standardScaler(df):
  scaler=StandardScaler()
  df[df.columns] = scaler.fit_transform(df[df.columns])
  return df

def minMaxScaler(df):
  scaler = MinMaxScaler()
  df[df.columns] = scaler.fit_transform(df[df.columns])
  return df
  

# Tratamiento de Outliers
def lof(X):
    lof=LocalOutlierFactor(n_neighbors=3,contamination=0.1)
    y_pred=lof.fit_predict(X)
    novelty_scores=-lof.negative_outlier_factor_
    threshold=np.percentile(novelty_scores, 90)
    predicted_labels=np.where(y_pred==-1,1,0)
    anomaly_indices=np.where(y_pred==-1)[0]
    #print("indices de las anomalias")
    #print(anomaly_indices)
    #print("datos clasificados como anomalias")
    #print(df.iloc[anomaly_indices])
    return anomaly_indices

# Encoding
def encodingCategoricasOneHot(X):
    #['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    varCategoricas = X.select_dtypes(exclude=np.number).columns.tolist()
    #listOrd=['Categorical','Sex','Ascites', 'Hepatomegaly', 'Spiders']
    listOhe=varCategoricas#['cp','restecg','slope','thal']
    #ppOrd=OrdinalEncoder()
    #X[listOrd]=ppOrd.fit_transform(X[listOrd])
    ohe=OneHotEncoder(sparse_output=False, drop='first')
    ohe_output=ohe.fit_transform(X[listOhe])
    ohe_feature_names=ohe.get_feature_names_out(listOhe)
    ohe_df=pd.DataFrame(ohe_output,columns=ohe_feature_names, index=X.index)
    X.drop(columns=listOhe,inplace=True)
    X_encoded=pd.concat([X,ohe_df],axis=1)
    #print(X_encoded.head())
    return X_encoded

def encodingLabel(y, mapping):
  label_encoder = LabelEncoder()
  label_encoder.classes_ = np.array(list(mapping.keys()))
  #y.iloc[:, 0] = label_encoder.fit_transform(y.iloc[:, 0])
  y = label_encoder.fit_transform(y)
  return y

def tratamientoOutliers(X,y):
  anomalias=lof(X)
  X = X.drop(anomalias)
  X = X.reset_index(drop=True)
  y = np.delete(y,anomalias)
  return X, y

def svc(X,y):
    # Hiperparametros  {'C': 1, 'coef0': 1.0, 'gamma': 'auto', 'kernel': 'sigmoid'} score  0.7821428571428571
    # Hiperparametros  {'C': 1, 'gamma': 'scale', 'kernel': 'sigmoid'} score  0.7785714285714286
    # Hiperparametros  {'C': 0.1, 'gamma': 0.1, 'kernel': 'sigmoid'} score  0.7642857142857143
    svc=SVC(C=1,kernel='sigmoid', gamma='auto', coef0=1.0, probability=True, random_state=123)
    cv_scores=cross_val_score(svc, X,y,cv=10)
    print("cv scores ",cv_scores)
    print("mean cv scores ",cv_scores.mean())

def svc_grid_search(X,y):
  param_grid = [
      {'C': [0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
      {'C': [0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['poly'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'], 'degree': [2, 3, 4, 5], 'coef0': [0.0, 0.1, 0.5, 1.0]},
      {'C': [0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto']},
      {'C': [0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'], 'coef0': [0.0, 0.1, 0.5, 1.0]}
  ]
  svc=SVC()
  grid_search=GridSearchCV(estimator=svc, param_grid=param_grid,cv=10, refit = True, verbose=2)
  grid_search.fit(X,y)
  
  print("Hiperparametros ",grid_search.best_params_)
  print("score ",grid_search.best_score_)


originalClassSVC=[]
predClassSVC=[]

def classification_report_with_acc(y_true,y_pred):
    originalClassSVC.extend(y_true)
    predClassSVC.extend(y_pred)
    acc=accuracy_score(y_true, y_pred)
    return acc

def svcCV(nCV,X,y):
    svc=SVC(C=1,kernel='sigmoid', gamma='auto', coef0=1.0, probability=True, random_state=123)
    if nCV==1:
        #hold-out cv
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
        svc.fit(X_train, y_train)
        print("hold out cv ",svc.score(X_train, y_train))
        #completar para el conjunto de datos de test
    elif nCV==2:
        kf=KFold(n_splits=7,shuffle=True,random_state=123)
        nested=cross_val_score(svc, X,y,cv=kf,scoring=make_scorer(classification_report_with_acc))
        print(nested)
        print(classification_report(originalClassSVC,predClassSVC))
    elif nCV==3:
        loocv=LeaveOneOut()
        nested=cross_val_score(svc, X,y,cv=loocv,scoring=make_scorer(classification_report_with_acc))
        print(nested)
        print(classification_report(originalClassSVC,predClassSVC))
    elif nCV==4:
        strCV=StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
        nested=cross_val_score(svc, X,y,cv=strCV,scoring=make_scorer(classification_report_with_acc))
        print(nested)
        print(classification_report(originalClassSVC,predClassSVC))

originalclass = []
predictedclass = []

def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred) 

def logisticR(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
    lg=linear_model.LogisticRegression(random_state=123)
    lg.fit(X_train,y_train)
    predictions = lg.predict(X_test)
    
    
    cm=confusion_matrix(y_test,predictions)
    score = lg.score(X_test, y_test)
    
    print("cross validation ------------------------")
    #cv
    
    scores=cross_val_score(lg,X,y,cv=10)
    print("\n")
    print("Scores y promedio: ")
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("\n")
    
    ''''
    print("Classification Report with the test set")
    print("\n")
    print(classification_report(y_test, predictions))
    print("\n")
    '''
    
    # Perform cross-validated predictions
    print("Classification Report")
    print("\n")
    predicted_cv = cross_val_predict(lg, X, y, cv=10)
    #print(predicted_cv)

    # Generate classification report
    report_cv = classification_report(y, predicted_cv)
    print(report_cv)  
    
    #nested
    
    print("Classification Report averaging the results")
    print("\n")
    nested=cross_val_score(lg,X,y,cv=10,scoring=make_scorer(classification_report_with_accuracy_score))
    print(nested)
    print("\n")
    print("final scoring")
    print("\n")
    print(classification_report(originalclass, predictedclass)) 
    
    print("Plot confussion ------------------------")
    # Plot Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['C', 'CL', 'D'], yticklabels=['C', 'CL', 'D'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    
    df = load()
    df = df.iloc[:, df.columns != 'ID'] # Dropear ID
    

    # Analisis exploratorio
    #analisisCategoricas(df)
    #analisisNumericas(df)

    # Tratamiento de nulos
    #nullAnalysis(df)
    df = dropNAinDrug(df) # el dataset recomienda dropear los Drug=null
    df = imputeWithMode(df) # el dataset recomienda imputar con Media, pero como son categpricas utilizo moda
    #nullAnalysis(df)

    X = df.drop('Status', axis=1)
    y = df[['Status']]

    # Encoding
    X = encodingCategoricasOneHot(X)
    y = encodingLabel(y, {'C': 0, 'CL': 1, 'D': 2})
    
    # Escalamiento
      # StandrtScaler
    X=standardScaler(X)
    
      # Minmaxescaler
    #X=minMaxScaler(X)
    
    
    # Tratamiento de outliers
    X, y = tratamientoOutliers(X,y)

    # Regresion logistica
    #logisticR(X,y)

    # SVM

    #svc(X,y)
    #svc_grid_search(X,y)
    tipoCV = 2
    svcCV(tipoCV, X, y)


    # CrossValidation para hiperparametros



    # Feature selection








main()