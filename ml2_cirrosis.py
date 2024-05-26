
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
from scipy import stats
import statsmodels.api as sm
from sklearn.feature_selection import RFE

#logistic

import matplotlib
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn import preprocessing, model_selection, linear_model
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_predict
from pandas.plotting import scatter_matrix
from sklearn.metrics import accuracy_score,make_scorer, f1_score
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
def plotTarget(data, target):
  categoria_counts = data[target].value_counts()
  categorias_ordenadas = categoria_counts.index
  plt.figure(figsize=(8, 6))
  plt.bar(categorias_ordenadas, categoria_counts, color='skyblue')
  plt.xlabel('Categoría')
  plt.ylabel('Frecuencia')
  plt.title('Distribución de Categorías')

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

#  Escalamiento con StandardScaler
def standardScaler(df, target):
  y = df[target]
  X = df.drop(columns=[target])
  scaler = StandardScaler()
  df_scaled = scaler.fit_transform(X)
  df_scaled = pd.DataFrame(df_scaled, columns=X.columns, index=df.index)
  df_final = df_scaled.join(y)
  return df_final

#  Escalamiento con MinMaxScaler
def minMaxScaler(df, target):
  y = df[target]
  X = df.drop(columns=[target])
  scaler = MinMaxScaler()
  df_scaled = scaler.fit_transform(X)
  df_scaled = pd.DataFrame(df_scaled, columns=X.columns, index=df.index)
  df_final = df_scaled.join(y)
  return df_final
  
# Encoding
#  Encoding con OneHotEncoder
def encodingFeatures(df, target):
    X = df.drop(target, axis=1)
    #['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    varCategoricas = X.select_dtypes(exclude=np.number).columns.tolist()
    #listOrd=['Categorical','Sex','Ascites', 'Hepatomegaly', 'Spiders']
    listOhe=varCategoricas#['cp','restecg','slope','thal']
    #
    ppOrd=OrdinalEncoder()
    #X[listOrd]=ppOrd.fit_transform(X[listOrd])
    ohe=OneHotEncoder(sparse_output=False, drop='first')
    ohe_output=ohe.fit_transform(X[listOhe])
    ohe_feature_names=ohe.get_feature_names_out(listOhe)
    ohe_df=pd.DataFrame(ohe_output,columns=ohe_feature_names, index=X.index)
    X.drop(columns=listOhe,inplace=True)
    X_encoded=pd.concat([X,ohe_df],axis=1)
    df=pd.concat([X_encoded,df[[target]]],axis=1)
    #print(X_encoded.head())
    return df

# Encoding con LabelEncoder
def encodingLabel(df, target, mapping):
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(list(mapping.keys()))
    df[target] = label_encoder.transform(df[target])
    return df

# Tratamiento de Outliers
#  Algoritmo LOF
def lof(X, contamination, plot):
    lof=LocalOutlierFactor(n_neighbors=3,contamination=contamination)
    y_pred=lof.fit_predict(X)
    novelty_scores=-lof.negative_outlier_factor_
    threshold=np.percentile(novelty_scores, (1 - contamination) * 100)
    predicted_labels=np.where(y_pred==-1,1,0)
    anomaly_indices=np.where(y_pred==-1)[0]
    #print("indices de las anomalias")
    #print(anomaly_indices)
    #print("datos clasificados como anomalias")
    #print(df.iloc[anomaly_indices])
    if plot:
      plotLOF(X.iloc[:,:].values, novelty_scores, threshold, predicted_labels, y_pred)
    return anomaly_indices 

#  Plotear resultados del LOF
def plotLOF(X,novelty_scores,threshold,predicted_labels,y_pred):
    # Plot histogram of novelty scores
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(novelty_scores, bins=20, color='skyblue', edgecolor='black')
    plt.title("Histogram of Novelty Scores")
    plt.xlabel("Novelty Score")
    plt.ylabel("Frequency")
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
    plt.legend()
    
    # Scatter plot of novelties overlaid on the original data
    plt.subplot(1, 2, 2)
    colors = np.array(['red', 'blue'])
    #shift the values to the right so [-1,1] converts to [0,2]
    plt.scatter(X[:, 0], X[:, 1], c=colors[(y_pred + 1) // 2], s=50, edgecolors='k')
    plt.title("Local Outlier Factor (LOF)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(['Normal', 'Outlier'], loc='best')
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Normal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Outlier')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.show()

#  Descartar outliers segun LOF
def tratamientoOutliers(df, target, contamination, plot):
  X = df.drop(target, axis=1)
  anomalias=lof(X, contamination, plot)
  df = df.drop(anomalias)
  df = df.reset_index(drop=True)
  return df

# Logistic Regression
#  Logistic regression grid search
def logisticGS(X,y):
    lg = LogisticRegression(random_state=123, max_iter=1000)
    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
    param_grid = [
        {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']},
        {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1'], 'solver': ['liblinear', 'saga']},
        {'C': [0.01, 0.1, 1, 10], 'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': [0.5]},
    ]
    grid_search = GridSearchCV(estimator=lg, param_grid=param_grid, cv=skf, scoring='f1_weighted', refit = True, verbose=2)
    grid_search.fit(X, y)
    print("Hiperparametros ",grid_search.best_params_)
    print("score ",grid_search.best_score_)

#  Logistic regression cross-validation metrics
def logisticCV(X, y):
    # Logistic regresion
    X = X.values
    lg=LogisticRegression(C=0.1, penalty='l1', solver='liblinear', random_state=123)
    skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=1)

    # Calcular todas las etiquetas
    true_labels = []
    predicted_labels = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lg.fit(X_train, y_train)
        y_pred = lg.predict(X_test)
        true_labels.extend(y_test)
        predicted_labels.extend(y_pred)

    # Reporte de clasificación
    print("\nReporte de Clasificación:")
    scores = cross_val_score(lg, X, y, cv=skf, scoring='f1_weighted')
    f1_weighted = scores.mean()
    print("F1-SCORE: ", f1_weighted)
    print(classification_report(true_labels, predicted_labels))

    # Matriz de confusión
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False,
                xticklabels=['C', 'CL', 'D'],
                yticklabels=['C', 'CL', 'D'])
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.title('Matriz de Confusión')
    plt.show()

# SVC
#  SVC grid search
def svcGS(X,y):
  svc=SVC(probability=True, random_state=123)
  param_grid = [
      {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear']},
      {'C': [0.01, 0.1, 1, 10], 'kernel': ['poly'], 'gamma': [0.1, 0.01, 0.001, 'scale', 'auto'], 'degree': [2, 3, 4, 5], 'coef0': [0.0, 0.1, 0.5, 1.0]},
      {'C': [0.01, 0.1, 1, 10], 'kernel': ['rbf'], 'gamma': [0.1, 0.01, 0.001, 'scale', 'auto']},
      {'C': [0.01, 0.1, 1, 10], 'kernel': ['sigmoid'], 'gamma': [0.1, 0.01, 0.001, 'scale', 'auto'], 'coef0': [0.0, 0.1, 0.5, 1.0]}
  ]
  skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
  grid_search=GridSearchCV(estimator=svc, param_grid=param_grid,cv=skf, scoring='f1_weighted', refit = True, verbose=2)
  grid_search.fit(X,y)
  
  print("Hiperparametros ",grid_search.best_params_)
  print("score ",grid_search.best_score_)

#  SVC cross validation metrics
def svcCV(X,y):
    # Support Vector Classifier
    X = X.values
    svc=SVC(C=1,kernel='sigmoid', gamma='auto', coef0=0.5, probability=True, random_state=123) 
    skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=1)

    # Calcular todas las etiquetas
    true_labels = []
    predicted_labels = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        true_labels.extend(y_test)
        predicted_labels.extend(y_pred)

    # Reporte de clasificación
    print("\nReporte de Clasificación:")
    scores = cross_val_score(svc, X, y, cv=skf, scoring='f1_weighted')
    f1_weighted = scores.mean()
    print("F1-SCORE: ", f1_weighted)
    print(classification_report(true_labels, predicted_labels))

    # Matriz de confusión
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False,
                xticklabels=['C', 'CL', 'D'],
                yticklabels=['C', 'CL', 'D'])
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.title('Matriz de Confusión')
    plt.show()
  
# Feature Selection

  # Algoritmo forward selection
def forward_selection(X, y, threshold=0.01):
    X_int = pd.DataFrame({'intercept': np.ones(len(X))}).join(X)
    included = ['intercept']
    excluded = list(set(X_int.columns) - set(included))
    best_features = []
    current_score = 0.0
    while excluded:
        scores_with_candidates = []
        for feature in excluded:
            model_features = included + [feature]
            X_subset = X_int[model_features]
            model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
            scores = cross_val_score(model, X_subset, y, cv=5, scoring='accuracy')
            mean_score = np.mean(scores)
            scores_with_candidates.append((mean_score, feature))
        scores_with_candidates.sort(reverse=True)
        best_score, best_feature = scores_with_candidates.pop()
        if best_feature is None or best_score <= current_score + threshold:
            break
        included.append(best_feature)
        excluded.remove(best_feature)
        best_features.append((best_feature, best_score))
        current_score = best_score
    print(best_features)
    return included, best_features

  # Algoritmo Recursive Forward Elimination
def selectFeatures(X,y, n_features):
  model=LogisticRegression()
  rfe=RFE(model,n_features_to_select=n_features)
  fit=rfe.fit(X, y)
  print("selected features ",X.columns[fit.support_])

def main():
    
    df = load()
    df = df.iloc[:, df.columns != 'ID'] # Dropear ID
    
    # Aplicar feature selection

    # Forward selection
    #df = df[['Status', 'Drug', 'Age', 'Sex', 'Platelets', 'Tryglicerides', 'Edema']] 

    # Recursive Forward Elimination
    #df = df[['Status', 'Drug','N_Days', 'Age', 'Bilirubin', 'Alk_Phos', 'Platelets', 'Prothrombin', 'Stage', 'Sex', 'Ascites', 'Hepatomegaly']]
             
    # Analisis exploratorio
    #analisisCategoricas(df)
    #analisisNumericas(df)
    #plotTarget(df, 'Status')

    # Tratamiento de nulos
    #nullAnalysis(df)
    df = dropNAinDrug(df) # el dataset recomienda dropear los Drug=null
    df = imputeWithMode(df) # el dataset recomienda imputar con Media, pero como son categpricas utilizo moda
    #nullAnalysis(df)

    #X = df.drop('Status', axis=1)
    #y = df[['Status']]

    # Encoding
    df = encodingFeatures(df, 'Status')
    df = encodingLabel(df, 'Status', {'C': 0, 'CL': 1, 'D': 2})
    
    # Escalamiento
    df=standardScaler(df, 'Status')
    #df=minMaxScaler(df, 'Status')
    
    # Tratamiento de outliers
    df = tratamientoOutliers(df, 'Status', contamination=0.1, plot=False)

    # X,y para los modelos
    X = df.drop('Status',axis=1)
    y = df['Status']
    # Reduccion a clasificacion binaria
    #y=y.apply(lambda x:1 if x<2 else 0) # 0:vivo, 1:muerto

    # Regresion logistica
    #logisticGS(X,y) 
    #logisticCV(X,y)
    # Hiperparametros  {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'} score  0.7610431382044895

    # SVM
    svcGS(X,y)
    #svcCV(X, y)
    # Hiperparametros  {'C': 1, 'coef0': 0.5, 'gamma': 'auto', 'kernel': 'sigmoid'} score  0.7580613722316496

    # Feature selection
    #forward_selection(X, y, 0.001)
    #selectFeatures(X, y, 10)

    # Reducir de 3 vars a 1

main()