from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from feature_engine.outliers import Winsorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

file_name = 'ecommerce_customer_churn_dataset_raw.csv'
file_path = os.path.join(base_dir, file_name)

def preprocess_data(data, target_column, save_path, output_csv_path):

    data = data.copy()
    if 'Age' in data.columns:
        data.loc[data['Age'] > 100, 'Age'] = 100
    if 'Total_Purchases' in data.columns:
        data.loc[data['Total_Purchases'] < 0, 'Total_Purchases'] = 0

    num_cols = data.select_dtypes(include=np.number).columns.tolist()
    cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()

    if target_column in num_cols: num_cols.remove(target_column)
    if target_column in cat_cols: cat_cols.remove(target_column)

    # 2. Pipeline Definition
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_capper', Winsorizer(capping_method='iqr', fold=1.5, tail='both')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )

    # Split & Transform
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train_transformed = preprocessor.fit_transform(X_train)
    

    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    final_columns = num_cols + list(cat_features)
    
    df_transformed = pd.DataFrame(X_train_transformed, columns=final_columns)
    df_transformed[target_column] = y_train.values

    df_transformed.to_csv(output_csv_path, index=False)
    print(f"Data siap latih berhasil disimpan ke: {output_csv_path}")

    dump(preprocessor, save_path)

    return X_train_transformed, preprocessor.transform(X_test), y_train, y_test



try:
    df = pd.read_csv(file_path)
    print(f"Berhasil memuat data dari: {file_path}")
except FileNotFoundError:
    # Jika ternyata file ada di satu tingkat di atas (root)
    file_path = os.path.join(base_dir, '..', file_name)
    df = pd.read_csv(file_path)
    print(f"Berhasil memuat data dari root: {file_path}")

df['Churned'] = df['Churned'].astype(int)
preprocess_data(df, 'Churned', 'preprocessing/preprocessor.joblib', 'preprocessing/ecommerce_customer_churn_dataset_preprocessing.csv')