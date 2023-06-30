# Código de Evaluación - Modelo de Retención de clientes -  Menús
############################################################################

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from sklearn.metrics import *
import os


# Cargar la tabla transformada
def eval_model(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/rf_adasynoversampler_menus.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente') 
    df = df[['customer_id','target','sum_num_dias_ult_5_mes','amount_ult_mes_Harinas','amount_ult_mes_Limpieza','amount_ult_mes_Margarinas',
                          'amount_ult_mes_Pastas','amount_ult_mes_Salsas','amount_ult_mes_otros','temp_Pastas_ult_3m',
                          'perc_var_amount_prim','perc_var_amount_seg','perc_var_amount_terc','perc_var_amount_cuar',
                          'perc_var_num_cat_cuar','perc_var_amount_quin','perc_var_num_cat_quin']]
    vars_num = df.drop(['target','customer_id'],axis=1).columns.to_list()
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
    # Predecimos sobre el set de datos de validación
    y_pred_train = model.predict(X_train_adasynoversampler)
    y_pred_test = model.predict(X_test)
    prob_train = model.predict_proba(X_train_adasynoversampler)[:,1]
    prob_test = model.predict_proba(X_test)[:,1]
    # Generamos métricas de diagnóstico
    metricsrf = pd.DataFrame({'metric':['AUC','Accuracy','Precision','Recall','f1-score'],
                                'rf_train':[roc_auc_score(y_train_adasynoversampler, prob_train),
                                                 accuracy_score(y_train_adasynoversampler, y_pred_train),
                                                precision_score(y_train_adasynoversampler, y_pred_train),
                                                recall_score(y_train_adasynoversampler, y_pred_train),
                                                f1_score(y_train_adasynoversampler, y_pred_train)],

                                'rf_test':[roc_auc_score(y_test, prob_test),
                                                accuracy_score(y_test, y_pred_test),
                                               precision_score(y_test, y_pred_test),
                                               recall_score(y_test, y_pred_test),
                                               f1_score(y_test, y_pred_test)]})

    # Matriz de confusion
    print("Matriz confusion: Train")
    cm_train = confusion_matrix(y_train_adasynoversampler,y_pred_train)
    print(cm_train)

    print("Matriz confusion: Test")
    cm_test = confusion_matrix(y_test,y_pred_test)
    print(cm_test)
    print(metricsrf)


# Validación desde el inicio
def main():
    df = eval_model('feature_engineering_test_1.csv')
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()