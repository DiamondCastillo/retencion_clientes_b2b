# Código de Scoring - Modelo de retención de clientes - Menús
############################################################################

# Importación de librerías
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os


# Cargar la tabla transformada
def score_model(filename, scores):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/rf_adasynoversampler_menus.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de Scoring 
    df_2 = df.drop('customer_id', axis=1)   
    res = model.predict(df_2).reshape(-1,1)
    pred = pd.DataFrame(res, columns=['PRED_FUGA'])
    pred_final = pd.concat([df['customer_id'], pred], axis=1)
    pred_final.to_csv(os.path.join('../data/scores/', scores), index=False)
    print(scores, 'exportado correctamente en la carpeta scores')


# Scoring desde el inicio
def main():
    df = score_model('feature_engineering_score.csv','final_score.csv')
    print('Finalizó el Scoring del Modelo')


if __name__ == "__main__":
    main()