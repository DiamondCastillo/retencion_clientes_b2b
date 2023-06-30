# Código de Entrenamiento - Modelo de Retención de clientes - Menús
############################################################################

# importando librerias
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
import pickle
import os


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')
    vars_num = df.drop(['target','periodo','customer_id','sample'],axis=1).columns.to_list()
    # Separando variables independientes y variable a predecir
    x= df[vars_num]
    y= df.target
    # Creación de la data de train y la data de test
    X_train, X_test, y_train, y_test = train_test_split(x,
                                                        y, 
                                                        test_size=0.30,
                                                        stratify= y,
                                                        random_state=123)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Escalamiento de variables
    sc_X = RobustScaler()                       
    X_train = sc_X.fit_transform(X_train)          
    X_test = sc_X.transform(X_test) 

    # Configuración proporción de balanceo para el OverSampling
    ratio_os = {0: len(y_train[y_train==0]), 1: int(len(y_train[y_train==0])*0.3)}
    # ADASYN OverSampling
    adasynoversampler = ADASYN(sampling_strategy=ratio_os, random_state=123)
    X_train_adasynoversampler, y_train_adasynoversampler = adasynoversampler.fit_resample(X_train, y_train)
    
    # Entrenamos el modelo con toda la muestra
    rf_mod=RandomForestClassifier(n_estimators= 2340,
                                  max_depth= 4,
                                  min_samples_split= 19,
                                  min_samples_leaf= 2,
                                  max_features= 'sqrt',
                                  class_weight= 'balanced',
                                  random_state=123)
    rf_mod.fit(X_train_adasynoversampler, y_train_adasynoversampler)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(rf_mod, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento
def main():
    read_file_csv('feature_engineering_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()
