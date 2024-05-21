
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

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
  n_rows = (n_vars + n_cols - 1) // n_cols
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

def standardScaler(X):
  scaler=StandardScaler()
  X_scaled=scaler.fit_transform(X)
  return X_scaled

def minMaxScaler(X):
  minmax_scaler = MinMaxScaler()
  X_scaled = minmax_scaler.fit_transform(X)
  

# Tratamiento de Outliers
def lof(X_scaled,df):
    lof=LocalOutlierFactor(n_neighbors=3,contamination=0.1)
    y_pred=lof.fit_predict(X_scaled)
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

def encodingLabel(y):
  label_encoder = LabelEncoder() 
  y= label_encoder.fit_transform(y) 
  return y

def tratamientoOutliers(X_scaled,df):
    anomalias=lof(X_scaled,df)
    df = df.drop(anomalias)
    df = df.reset_index(drop=True)
    return df

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

    X = df.iloc[:, df.columns != 'Status']
    y = df.iloc[:, 2]

    # Encoding
    X = encodingCategoricasOneHot(X)
    y = encodingLabel(y)
    
    # Escalamiento
      # StandrtScaler
    X_scaled=standardScaler(X)
    
      # Minmaxescaler
    #x_scaled=minMaxScaler(X)
    
    
    # Tratamiento de outliers
    df = tratamientoOutliers(X_scaled,df)

    # Regresion logistica


    # SVM




    # CrossValidation para hiperparametros



    # Feature selection








main()