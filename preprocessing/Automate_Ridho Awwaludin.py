import pandas as pd
import numpy as np
import os
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.model_selection import train_test_split


base_dir = os.path.dirname(os.path.abspath(__file__))

file_name = 'ecommerce_customer_churn_dataset_raw.csv'
file_path = os.path.join(base_dir, file_name)

def preprocess_data(data, target_column, save_path, output_csv_path):

    def remove_outliers_iqr(df, columns):
        df_cleaned = df.copy()
        for col in columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        return df_cleaned

    num_cols = data.select_dtypes(include=np.number).columns.tolist()
    if target_column in num_cols: num_cols.remove(target_column)
    
    data_cleaned = remove_outliers_iqr(data, num_cols)

    cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    if target_column in cat_cols: cat_cols.remove(target_column)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder()) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
    )

    X = data_cleaned.drop(columns=[target_column])
    y = data_cleaned[target_column]
    
    X_transformed = preprocessor.fit_transform(X)
    
    all_columns = num_cols + cat_cols
    df_final = pd.DataFrame(X_transformed, columns=all_columns)
    df_final[target_column] = y.values

    df_final.to_csv(output_csv_path, index=False)
    
    dump(preprocessor, save_path)
    
    print(f"Pipeline berhasil dijalankan. CSV disimpan di: {output_csv_path}")
    return df_final, preprocessor



try:
    df = pd.read_csv(file_path)
    print(f"Berhasil memuat data dari: {file_path}")
except FileNotFoundError:
    file_path = os.path.join(base_dir, '..', file_name)
    df = pd.read_csv(file_path)
    print(f"Berhasil memuat data dari root: {file_path}")

df['Churned'] = df['Churned'].astype(int)
preprocess_data(df, 'Churned', 'preprocessor.joblib', 'ecommerce_customer_churn_dataset_preprocessing.csv')