
# Script de Preparación de Datos - Modelo de retención de clientes - Menús
###################################

import pandas as pd
import numpy as np
import datetime
import os


# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename), sep='|', encoding='UTF-8', quotechar='"')  
    print(filename, ' cargado correctamente')
    return df


# Realizamos la transformación de datos
def data_preparation(df):
    # Filtro clientes del giro de menus
    df['category'].fillna('SIN_CATEGORY', inplace=True)
    df['departamento'].fillna('SIN_DEPARTAMENTO', inplace=True)
    df = df[df['giro']=='COMIDA CASERA CRIOLL'].copy()
    df["date"] = pd.to_datetime(df["date"])
    # Filtro clientes que registren compras en el último mes
    df_ult_mes = df[df['month_id']==df['month_id'].max()].copy()
    df = df[df['customer_id'].isin(df_ult_mes['customer_id'].unique())]
    # Agrupando categorías
    df['des_categoria'] = np.where(df.category.isin(['ACEITE A GRANEL','ACEITES DOMÉSTICOS','SALSAS GASTRONOMÍA',
                                                     'LAVAVAJILLAS INTRADEVCO','LIMPIADORES ESPECIALIZADOS',
                                                     'LIMPIADORES LIGHT DUTY','DETERGENTES','LAVAVAJILLAS',
                                                     'HARINAS INDUSTRIALES','HARINAS DOMÉSTICAS','MANTECAS INDUSTRIALES',
                                                     'MARGARINAS INDUSTRIALES','PRE-MEZCLAS INDUSTRIALES',
                                                     'PRE-MEZCLAS DOMÉSTICAS','PASTAS']), df.category, 'otros')
    df['des_categoria'] = np.where(df.des_categoria.isin(['ACEITE A GRANEL','ACEITES DOMÉSTICOS']), 'Aceites', df.des_categoria)
    df['des_categoria'] = np.where(df.des_categoria.isin(['DETERGENTES INTRADEVCO','LEJÍAS','LAVAVAJILLAS INTRADEVCO',
                                                          'LIMPIADORES ESPECIALIZADOS','LIMPIADORES LIGHT DUTY',
                                                          'DETERGENTES','LAVAVAJILLAS']), 'Limpieza', df.des_categoria)
    df['des_categoria'] = np.where(df.des_categoria.isin(['SALSAS GASTRONOMÍA','SALSAS']), 'Salsas', df.des_categoria)
    df['des_categoria'] = np.where(df.des_categoria.isin(['HARINAS INDUSTRIALES','HARINAS DOMÉSTICAS']), 'Harinas', df.des_categoria)
    df['des_categoria'] = np.where(df.des_categoria=='MANTECAS INDUSTRIALES', 'Mantecas', df.des_categoria)
    df['des_categoria'] = np.where(df.des_categoria=='MARGARINAS INDUSTRIALES', 'Margarinas', df.des_categoria)
    df['des_categoria'] = np.where(df.des_categoria.isin(['PRE-MEZCLAS INDUSTRIALES', 'PRE-MEZCLAS DOMÉSTICAS']),
                                                          'Pre_mezclas', df.des_categoria)
    df['des_categoria'] = np.where(df.des_categoria=='PASTAS', 'Pastas', df.des_categoria)
    df["month_id"] = df["month_id"].astype(int)
    df['Month'] = df['date'].dt.month 
    
    # Base a nivel cliente - mes
    df_fe = df.pivot_table(
        index=['customer_id','month_id'],
        aggfunc = {'amount':'sum','des_categoria':'nunique','date':'nunique'}
    ).reset_index()
    
    # Monto de ventas por mes - categorías
    df_amount_mes_cat = df.pivot_table(
        index=['month_id'], columns = ['des_categoria'],
        aggfunc = {'amount':'sum'}
    ).reset_index()
    df_amount_mes_cat.columns=[x[1] if x[0]!='month_id' else x[0] for x in df_amount_mes_cat.columns]
    df_amount_mes_cat.month_id = df_amount_mes_cat.month_id.astype(str)
    
    # Montos por categorías y flag de temporalidad de categorías principales B2B
    df_fe_cat = df.pivot_table(
        index=['customer_id','month_id','Month'], columns = ['des_categoria'],
        aggfunc = {'amount':'sum'}
    ).reset_index()
    df_fe_cat.columns=[x[1] if x[0]!='customer_id' and x[0]!='month_id' and x[0]!='Month' else x[0] for x in df_fe_cat.columns]
    df_fe_cat['temp_Aceites'] = np.select([df_fe_cat.Month.isin([5,7,12]), df_fe_cat.Month.isin([1,2])], [1, -1], default=0)
    df_fe_cat['temp_Limpieza'] = np.select([df_fe_cat.Month.isin([5,7,12]), df_fe_cat.Month.isin([1,2])], [1, -1], default=0)
    df_fe_cat['temp_Salsas'] = np.select([df_fe_cat.Month.isin([5,7,12]), df_fe_cat.Month.isin([1,2])], [1, -1], default=0)
    df_fe_cat['temp_Pastas'] = np.where(df_fe_cat.Month.isin([6,7,8,9,10,11]), 1, 0)
    df_fe_cat['temp_Harinas'] = np.select([df_fe_cat.Month.isin([7,8,10]), df_fe_cat.Month.isin([1,2,3,12])], [1, -1], default=0)
    df_fe_cat['temp_Margarinas'] = np.select([df_fe_cat.Month.isin([7,8,10]), df_fe_cat.Month.isin([1,2,3,12])], [1, -1], default=0)
    df_fe_cat['temp_Mantecas'] = np.select([df_fe_cat.Month.isin([7,8,10]), df_fe_cat.Month.isin([1,2,3,12])], [1, -1], default=0)
    df_fe_cat['temp_Pre_Mezclas'] = np.select([df_fe_cat.Month.isin([10,11,12]), df_fe_cat.Month.isin([1,2,3])], [1, -1], default=0)
    
    # Generación de variables para todos los periodos (entrenamiento y test):
    periodos =sorted(list(df_fe.month_id.unique()))
    i = 0

    # Monto de ventas del primer mes
    df_fe_prim_mes = df_fe[df_fe['month_id']==periodos[i]][['customer_id','amount','date','des_categoria']]
    df_fe_prim_mes.columns = ['customer_id','amount_prim_mes','num_dias_prim_mes','num_cat_prim_mes']
    # Monto de ventas del segundo mes
    df_fe_seg_mes = df_fe[df_fe['month_id']==periodos[i+1]][['customer_id','amount','date','des_categoria']]
    df_fe_seg_mes.columns = ['customer_id','amount_seg_mes','num_dias_seg_mes','num_cat_seg_mes']
    # Monto de ventas del tercer mes
    df_fe_terc_mes = df_fe[df_fe['month_id']==periodos[i+2]][['customer_id','amount','date','des_categoria']]
    df_fe_terc_mes.columns = ['customer_id','amount_terc_mes','num_dias_terc_mes','num_cat_terc_mes']
    # Monto de ventas del cuarto mes
    df_fe_cuar_mes = df_fe[df_fe['month_id']==periodos[i+3]][['customer_id','amount','date','des_categoria']]
    df_fe_cuar_mes.columns = ['customer_id','amount_cuar_mes','num_dias_cuar_mes','num_cat_cuar_mes']
    # Monto de ventas del quinto mes
    df_fe_quin_mes = df_fe[df_fe['month_id']==periodos[i+4]][['customer_id','amount','date','des_categoria']]
    df_fe_quin_mes.columns = ['customer_id','amount_quin_mes','num_dias_quin_mes','num_cat_quin_mes']
    # Monto de ventas del último mes
    df_fe_ult_mes = df_fe[df_fe['month_id']==periodos[i+5]][['customer_id','amount','date','des_categoria']]
    df_fe_ult_mes.columns = ['customer_id','amount_ult_mes','num_dias_ult_mes','num_cat_ult_mes']

    # Amount mensual por categoría y flag de mes de estacionalidad del último mes
    df_train_categ_ult_mes = df_fe_cat[df_fe_cat.month_id==periodos[i+5]]
    df_train_categ_ult_mes = df_train_categ_ult_mes.pivot_table(
            index=['customer_id'],
            aggfunc = {'Aceites':'sum','Limpieza':'sum','Salsas':'sum','otros':'sum','Harinas':'sum',
                       'Mantecas':'sum','Margarinas':'sum','Pastas':'sum','Pre_mezclas':'sum',
                      'temp_Aceites':'sum','temp_Limpieza':'sum','temp_Salsas':'sum','temp_Pastas':'sum',
                       'temp_Harinas':'sum','temp_Margarinas':'sum','temp_Mantecas':'sum',
                      'temp_Pre_Mezclas':'sum'}
        ).reset_index().fillna(0)
    df_train_categ_ult_mes.columns=['customer_id','amount_ult_mes_Aceites','amount_ult_mes_Harinas',
                                 'amount_ult_mes_Limpieza','amount_ult_mes_Mantecas',
                                 'amount_ult_mes_Margarinas','amount_ult_mes_Pastas',
                                 'amount_ult_mes_Pre_Mezclas','amount_ult_mes_Salsas',
                                 'amount_ult_mes_otros','temp_Aceites_ult_mes','temp_Harinas_ult_mes',
                                 'temp_Limpieza_ult_mes','temp_Mantecas_ult_mes','temp_Margarinas_Industriales_ult_mes',
                                 'temp_Pastas_ult_mes','temp_Pre_Mezclas_ult_mes','temp_Salsas_ult_mes']

    df_train_categ_ult_mes = df_train_categ_ult_mes.merge(df_fe_ult_mes[['customer_id','amount_ult_mes']], on='customer_id', how='left')
    df_train_categ_ult_mes['perc_Aceites_ult_mes'] = df_train_categ_ult_mes['amount_ult_mes_Aceites']/df_train_categ_ult_mes['amount_ult_mes']
    df_train_categ_ult_mes['perc_Harinas_ult_mes'] = df_train_categ_ult_mes['amount_ult_mes_Harinas']/df_train_categ_ult_mes['amount_ult_mes']
    df_train_categ_ult_mes['perc_Limpieza_ult_mes'] = df_train_categ_ult_mes['amount_ult_mes_Limpieza']/df_train_categ_ult_mes['amount_ult_mes']
    df_train_categ_ult_mes['perc_Mantecas_ult_mes'] = df_train_categ_ult_mes['amount_ult_mes_Mantecas']/df_train_categ_ult_mes['amount_ult_mes']
    df_train_categ_ult_mes['perc_Margarinas_ult_mes'] = df_train_categ_ult_mes['amount_ult_mes_Margarinas']/df_train_categ_ult_mes['amount_ult_mes']
    df_train_categ_ult_mes['perc_Pastas_ult_mes'] = df_train_categ_ult_mes['amount_ult_mes_Pastas']/df_train_categ_ult_mes['amount_ult_mes']
    df_train_categ_ult_mes['perc_Pre_Mezclas_ult_mes'] = df_train_categ_ult_mes['amount_ult_mes_Pre_Mezclas']/df_train_categ_ult_mes['amount_ult_mes']
    df_train_categ_ult_mes['perc_Salsas_ult_mes'] = df_train_categ_ult_mes['amount_ult_mes_Salsas']/df_train_categ_ult_mes['amount_ult_mes']
    df_train_categ_ult_mes.drop('amount_ult_mes', axis=1, inplace=True)

    # Monto promedio mensual de los últimos 2 meses
    df_fe_ult_2_meses = df_fe[df_fe['month_id'].isin(periodos[i+4:i+6])][['customer_id','month_id',
                                                                             'amount','date','des_categoria']]
    df_fe_ult_2_meses = df_fe_ult_2_meses.pivot_table(
            index=['customer_id'],
            aggfunc = {'amount':['mean','sum','min','max'],'date':['mean','sum','min','max'],
                      'des_categoria':['mean','sum','min','max']}
        ).reset_index().fillna(0)
    df_fe_ult_2_meses.columns = ['customer_id','max_amount_ult_2_mes','mean_amount_ult_2_mes','min_amount_ult_2_mes',
                                     'sum_amount_ult_2_mes','max_num_dias_ult_2_mes','mean_num_dias_ult_2_mes',
                                     'min_num_dias_ult_2_mes','sum_num_dias_ult_2_mes','max_num_cat_ult_2_mes',
                                     'mean_num_cat_ult_2_mes','min_num_cat_ult_2_mes','sum_num_cat_ult_2_mes']
    # Monto promedio mensual de los últimos 3 meses
    df_fe_ult_3_meses = df_fe[df_fe['month_id'].isin(periodos[i+3:i+6])][['customer_id',
                                                                                    'month_id',
                                                                                    'amount','date','des_categoria']]
    df_fe_ult_3_meses = df_fe_ult_3_meses.pivot_table(
            index=['customer_id'],
            aggfunc = {'amount':['mean','sum','min','max'],'date':['mean','sum','min','max'],
                      'des_categoria':['mean','sum','min','max']}
        ).reset_index().fillna(0)
    df_fe_ult_3_meses.columns = ['customer_id','max_amount_ult_3_mes','mean_amount_ult_3_mes','min_amount_ult_3_mes',
                                     'sum_amount_ult_3_mes','max_num_dias_ult_3_mes','mean_num_dias_ult_3_mes',
                                     'min_num_dias_ult_3_mes','sum_num_dias_ult_3_mes','max_num_cat_ult_3_mes',
                                     'mean_num_cat_ult_3_mes','min_num_cat_ult_3_mes','sum_num_cat_ult_3_mes']

    # Amount Promedio por categoría y flag de mes de estacionalidad de los últimos 3 meses
    df_train_categ = df_fe_cat[df_fe_cat.month_id.isin(periodos[i+3:i+6])]
    df_train_categ_prom = df_train_categ.pivot_table(
            index=['customer_id'],
            aggfunc = {'Aceites':'mean','Limpieza':'mean','Salsas':'mean','otros':'mean','Harinas':'mean',
                       'Mantecas':'mean','Margarinas':'mean','Pastas':'mean','Pre_mezclas':'mean',
                      'temp_Aceites':'sum','temp_Limpieza':'sum','temp_Salsas':'sum','temp_Pastas':'sum',
                       'temp_Harinas':'sum','temp_Margarinas':'sum','temp_Mantecas':'sum',
                      'temp_Pre_Mezclas':'sum'}
        ).reset_index().fillna(0)
    df_train_categ_prom.columns=['customer_id','amount_prom_ult_3m_Aceites','amount_prom_ult_3m_Harinas',
                                 'amount_prom_ult_3m_Limpieza','amount_prom_ult_3m_Mantecas',
                                 'amount_prom_ult_3m_Margarinas','amount_prom_ult_3m_Pastas',
                                 'amount_prom_ult_3m_Pre_Mezclas','amount_prom_ult_3m_Salsas',
                                 'amount_prom_ult_3m_otros','temp_Aceites_ult_3m','temp_Harinas_ult_3m',
                                 'temp_Limpieza_ult_3m','temp_Mantecas_ult_3m','temp_Margarinas_Industriales_ult_3m',
                                 'temp_Pastas_ult_3m','temp_Pre_Mezclas_ult_3m','temp_Salsas_ult_3m']
    # Monto promedio mensual de los últimos 4 meses
    df_fe_ult_4_meses = df_fe[df_fe['month_id'].isin(periodos[i+2:i+6])][['customer_id',
                                                                                           'month_id',
                                                                                           'amount','date',
                                                                                           'des_categoria']]
    df_fe_ult_4_meses = df_fe_ult_4_meses.pivot_table(
            index=['customer_id'],
            aggfunc = {'amount':['mean','sum','min','max'],'date':['mean','sum','min','max'],
                      'des_categoria':['mean','sum','min','max']}
        ).reset_index().fillna(0)
    df_fe_ult_4_meses.columns = ['customer_id','max_amount_ult_4_mes','mean_amount_ult_4_mes','min_amount_ult_4_mes',
                                     'sum_amount_ult_4_mes','max_num_dias_ult_4_mes','mean_num_dias_ult_4_mes',
                                     'min_num_dias_ult_4_mes','sum_num_dias_ult_4_mes','max_num_cat_ult_4_mes',
                                     'mean_num_cat_ult_4_mes','min_num_cat_ult_4_mes','sum_num_cat_ult_4_mes']
    # Monto promedio mensual de los últimos 5 meses
    df_fe_ult_5_meses = df_fe[df_fe['month_id'].isin(periodos[i+1:i+6])][['customer_id',
                                                                                                  'month_id',
                                                                                                  'amount','date',
                                                                                                  'des_categoria']]
    df_fe_ult_5_meses = df_fe_ult_5_meses.pivot_table(
            index=['customer_id'],
            aggfunc = {'amount':['mean','sum','min','max'],'date':['mean','sum','min','max'],
                      'des_categoria':['mean','sum','min','max']}
        ).reset_index().fillna(0)
    df_fe_ult_5_meses.columns = ['customer_id','max_amount_ult_5_mes','mean_amount_ult_5_mes','min_amount_ult_5_mes',
                                     'sum_amount_ult_5_mes','max_num_dias_ult_5_mes','mean_num_dias_ult_5_mes',
                                     'min_num_dias_ult_5_mes','sum_num_dias_ult_5_mes','max_num_cat_ult_5_mes',
                                     'mean_num_cat_ult_5_mes','min_num_cat_ult_5_mes','sum_num_cat_ult_5_mes']

    # Monto promedio mensual de los últimos 6 meses
    df_fe_ult_6_meses = df_fe[df_fe['month_id'].isin(periodos[i:i+6])][['customer_id',
                                                                                    'month_id',
                                                                                    'amount','date','des_categoria']]
    df_fe_ult_6_meses = df_fe_ult_6_meses.pivot_table(
            index=['customer_id'],
            aggfunc = {'amount':['mean','sum','min','max'],'date':['mean','sum','min','max'],
                      'des_categoria':['mean','sum','min','max']}
        ).reset_index().fillna(0)
    df_fe_ult_6_meses.columns = ['customer_id','max_amount_ult_6_mes','mean_amount_ult_6_mes','min_amount_ult_6_mes',
                                     'sum_amount_ult_6_mes','max_num_dias_ult_6_mes','mean_num_dias_ult_6_mes',
                                     'min_num_dias_ult_6_mes','sum_num_dias_ult_6_mes','max_num_cat_ult_6_mes',
                                     'mean_num_cat_ult_6_mes','min_num_cat_ult_6_mes','sum_num_cat_ult_6_mes']
    # Unión dataset:
    df = df[['customer_id', 'departamento']].drop_duplicates().merge(
            df_fe_ult_mes, on='customer_id', how='left').merge(
            df_fe_ult_2_meses, on='customer_id', how='left').merge(
            df_fe_ult_3_meses, on='customer_id', how='left').merge(
            df_fe_ult_4_meses, on='customer_id', how='left').merge(
            df_fe_ult_5_meses, on='customer_id', how='left').merge(
            df_fe_ult_6_meses, on='customer_id', how='left').merge(
            df_fe_prim_mes, on='customer_id', how='left').merge(
            df_fe_seg_mes, on='customer_id', how='left').merge(
            df_fe_terc_mes, on='customer_id', how='left').merge(
            df_fe_cuar_mes, on='customer_id', how='left').merge(
            df_fe_quin_mes, on='customer_id', how='left').merge(
            df_train_categ_ult_mes, on='customer_id', how='left').merge(
            df_train_categ_prom, on='customer_id', how='left').fillna(0)

    df['perc_var_amount_prim'] = (df['amount_prim_mes']-df['amount_ult_mes'])/df['amount_prim_mes']
    df['perc_var_num_dias_prim'] = (df['num_dias_prim_mes']-df['num_dias_ult_mes'])/df['num_dias_prim_mes']
    df['perc_var_num_cat_prim'] = (df['num_cat_prim_mes']-df['num_cat_ult_mes'])/df['num_cat_prim_mes']

    df['perc_var_amount_seg'] = (df['amount_seg_mes']-df['amount_ult_mes'])/df['amount_seg_mes']
    df['perc_var_num_dias_seg'] = (df['num_dias_seg_mes']-df['num_dias_ult_mes'])/df['num_dias_seg_mes']
    df['perc_var_num_cat_seg'] = (df['num_cat_seg_mes']-df['num_cat_ult_mes'])/df['num_cat_seg_mes']

    df['perc_var_amount_terc'] = (df['amount_terc_mes']-df['amount_ult_mes'])/df['amount_terc_mes']
    df['perc_var_num_dias_terc'] = (df['num_dias_terc_mes']-df['num_dias_ult_mes'])/df['num_dias_terc_mes']
    df['perc_var_num_cat_terc'] = (df['num_cat_terc_mes']-df['num_cat_ult_mes'])/df['num_cat_terc_mes']

    df['perc_var_amount_cuar'] = (df['amount_cuar_mes']-df['amount_ult_mes'])/df['amount_cuar_mes']
    df['perc_var_num_dias_cuar'] = (df['num_dias_cuar_mes']-df['num_dias_ult_mes'])/df['num_dias_cuar_mes']
    df['perc_var_num_cat_cuar'] = (df['num_cat_cuar_mes']-df['num_cat_ult_mes'])/df['num_cat_cuar_mes']

    df['perc_var_amount_quin'] = (df['amount_quin_mes']-df['amount_ult_mes'])/df['amount_quin_mes']
    df['perc_var_num_dias_quin'] = (df['num_dias_quin_mes']-df['num_dias_ult_mes'])/df['num_dias_quin_mes']
    df['perc_var_num_cat_quin'] = (df['num_cat_quin_mes']-df['num_cat_ult_mes'])/df['num_cat_quin_mes']

    df['dif_amount_1'] = np.where(df['amount_cuar_mes']<=df['amount_quin_mes'], 1, -1)
    df['dif_amount_0'] = np.where(df['amount_quin_mes']<=df['amount_ult_mes'], 1, -1)
    df['dif_dias_1'] = np.where(df['num_dias_cuar_mes']<=df['num_dias_quin_mes'], 1, -1)
    df['dif_dias_0'] = np.where(df['num_dias_quin_mes']<=df['num_dias_ult_mes'], 1, -1)
    df['dif_cat_1'] = np.where(df['num_cat_cuar_mes']<=df['num_cat_quin_mes'], 1, -1)
    df['dif_cat_0'] = np.where(df['num_cat_quin_mes']<=df['num_cat_ult_mes'], 1, -1)

    conditions_amount_low = [df['dif_amount_1']+df['dif_amount_0']==0,
                  df['dif_amount_1']+df['dif_amount_0']==2,
                  df['dif_amount_1']+df['dif_amount_0']==-2]
    tags_amount_low = [1, 0, 2]
    df['dif_amount_low'] = np.select(conditions_amount_low, tags_amount_low, default=np.nan)

    conditions_dias_low = [df['dif_dias_1']+df['dif_dias_0']==0,
                  df['dif_dias_1']+df['dif_dias_0']==2,
                  df['dif_dias_1']+df['dif_dias_0']==-2]
    tags_dias_low = [1, 0, 2]
    df['dif_dias_low'] = np.select(conditions_dias_low, tags_dias_low, default=np.nan)

    conditions_cat_low = [df['dif_cat_1']+df['dif_cat_0']==0,
                  df['dif_cat_1']+df['dif_cat_0']==2,
                  df['dif_cat_1']+df['dif_cat_0']==-2]
    tags_cat_low = [1, 0, 2]
    df['dif_cat_low'] = np.select(conditions_cat_low, tags_cat_low, default=np.nan)

    conditions_amount_up = [df['dif_amount_1']+df['dif_amount_0']==0,
                  df['dif_amount_1']+df['dif_amount_0']==2,
                  df['dif_amount_1']+df['dif_amount_0']==-2]
    tags_amount_up = [1, 2, 0]
    df['dif_amount_up'] = np.select(conditions_amount_up, tags_amount_up, default=np.nan)

    conditions_dias_up = [df['dif_dias_1']+df['dif_dias_0']==0,
                  df['dif_dias_1']+df['dif_dias_0']==2,
                  df['dif_dias_1']+df['dif_dias_0']==-2]
    tags_dias_up = [1, 2, 0]
    df['dif_dias_up'] = np.select(conditions_dias_up, tags_dias_up, default=np.nan)

    conditions_cat_up = [df['dif_cat_1']+df['dif_cat_0']==0,
                  df['dif_cat_1']+df['dif_cat_0']==2,
                  df['dif_cat_1']+df['dif_cat_0']==-2]
    tags_cat_up = [1, 2, 0]
    df['dif_cat_up'] = np.select(conditions_cat_up, tags_cat_up, default=np.nan)

    df['amount_low'] = np.where(df['dif_amount_low']>df['dif_amount_up'], 1, 0)
    df['amount_up'] = np.where(df['dif_amount_up']>df['dif_amount_low'], 1, 0)
    df['amount_keep'] = np.where(df['dif_amount_low']==df['dif_amount_up'], 1, 0)

    df['dias_low'] = np.where(df['dif_dias_low']>df['dif_dias_up'], 1, 0)
    df['dias_up'] = np.where(df['dif_dias_up']>df['dif_dias_low'], 1, 0)
    df['dias_keep'] = np.where(df['dif_dias_low']==df['dif_dias_up'], 1, 0)

    df['cat_low'] = np.where(df['dif_cat_low']>df['dif_cat_up'], 1, 0)
    df['cat_up'] = np.where(df['dif_cat_up']>df['dif_cat_low'], 1, 0)
    df['cat_keep'] = np.where(df['dif_cat_low']==df['dif_cat_up'], 1, 0)

    conditions_amount_tend = [df['amount_low']==1,
                  df['amount_up']==1,
                  df['amount_keep']==1]
    tags_amount_tend = ['-1', '1', '0']
    df['tend_amount'] = np.select(conditions_amount_tend, tags_amount_tend, default=np.nan)

    conditions_dias_tend = [df['dias_low']==1,
                  df['dias_up']==1,
                  df['dias_keep']==1]
    tags_dias_tend = ['-1', '1', '0']
    df['tend_dias'] = np.select(conditions_dias_tend, tags_dias_tend, default=np.nan)

    conditions_cat_tend = [df['cat_low']==1,
                  df['cat_up']==1,
                  df['cat_keep']==1]
    tags_cat_tend = ['-1', '1', '0']
    df['tend_cat'] = np.select(conditions_cat_tend, tags_cat_tend, default=np.nan)

    df['periodo'] = periodos[i+5]
    df['mes'] = int(str(periodos[i+5])[-2:])
    df.replace({np.inf:1, -np.inf:1, np.nan:0}, inplace=True)
    df.drop(['dif_amount_1','dif_amount_0','dif_dias_1','dif_dias_0','dif_cat_1','dif_cat_0',
                         'dif_amount_low','dif_dias_low','dif_cat_low','dif_amount_up','dif_dias_up','dif_cat_up',
                         'amount_low','amount_up','amount_keep','dias_low','dias_up','dias_keep','cat_low','cat_up',
                         'cat_keep'], axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)
    
    print('Transformación de datos completa')
    return df


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename), index=False)
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # Matriz de Scoring
    df1 = read_file_csv('trx_mayo_23.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, ['customer_id','sum_num_dias_ult_5_mes','amount_ult_mes_Harinas','amount_ult_mes_Limpieza','amount_ult_mes_Margarinas',
                          'amount_ult_mes_Pastas','amount_ult_mes_Salsas','amount_ult_mes_otros','temp_Pastas_ult_3m',
                          'perc_var_amount_prim','perc_var_amount_seg','perc_var_amount_terc','perc_var_amount_cuar',
                          'perc_var_num_cat_cuar','perc_var_amount_quin','perc_var_num_cat_quin'],'feature_engineering_score.csv')
    
if __name__ == "__main__":
    main()
