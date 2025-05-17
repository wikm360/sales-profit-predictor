import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

def preprocess_data(df):
    df = df.copy()
    df = df.drop(columns=["Customer Rating", "Row ID", "Order ID"])
    df["Ship Mode"] = df["Ship Mode"].fillna(df["Ship Mode"].mode()[0])
    df["Profit"] = df["Profit"].apply(lambda x: float(x.replace(" USD", "")))
    df = df.drop_duplicates()
    df["Discount"] = pd.to_numeric(df["Discount"], errors='coerce')
    return df

def impute_data(df, numerical_cols, is_train=True, imputer=None):
    df = df.copy()
    if is_train:
        knn_imputer = KNNImputer(n_neighbors=5)
        df[numerical_cols + ["Discount"]] = knn_imputer.fit_transform(df[numerical_cols + ["Discount"]])
    else:
        df[numerical_cols + ["Discount"]] = imputer.transform(df[numerical_cols + ["Discount"]])
    df['Discount_knn'] = df['Discount']
    return df, knn_imputer if is_train else None

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"Lower bound for {column}: {lower_bound}")
    print(f"Upper bound for {column}: {upper_bound}")
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"Rows before outlier removal for {column}: {len(df)}")
    print(f"Rows after outlier removal for {column}: {len(filtered_df)}")
    return filtered_df

def build_pipeline():
    numerical = ["Sales", "Unit Price (log)", "Shipping Cost (log)", "Product Base Margin", "Discount_knn"]
    categorical = ["Order Priority", "Product Category", "Product Sub-Category", "Product Container", "Product Name", "Province", "Region", "Customer Segment", "Ship Mode", "Customer Name"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('scaler', StandardScaler())
            ]), numerical),
            ('onehot', Pipeline(steps=[
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=0.5,
            random_state=42
        ))
    ])
    return pipeline

def train_model(data_path, model_path='best_rf_model.pkl'):
    df = pd.read_csv(f'{data_path}/data.csv')
    df = preprocess_data(df)

    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    print("Training set size:", len(train_set))
    print("Test set size:", len(test_set))

    numerical_cols = ["Order Quantity", "Sales", "Unit Price", "Profit", "Shipping Cost", "Product Base Margin"]
    train_set, knn_imputer = impute_data(train_set, numerical_cols, is_train=True)
    test_set, _ = impute_data(test_set, numerical_cols, is_train=False, imputer=knn_imputer)

    for col in ["Profit", "Unit Price", "Sales"]:
        train_set = remove_outliers(train_set, col)
        test_set = remove_outliers(test_set, col)

    train_set['Shipping Cost (log)'] = np.log1p(train_set['Shipping Cost'])
    train_set['Unit Price (log)'] = np.log1p(train_set['Unit Price'])
    test_set['Shipping Cost (log)'] = np.log1p(test_set['Shipping Cost'])
    test_set['Unit Price (log)'] = np.log1p(test_set['Unit Price'])

    numerical = ["Sales", "Unit Price (log)", "Shipping Cost (log)", "Product Base Margin", "Discount_knn"]
    categorical = ["Order Priority", "Product Category", "Product Sub-Category", "Product Container", "Product Name", "Province", "Region", "Customer Segment", "Ship Mode", "Customer Name"]
    features = numerical + categorical
    target = "Profit"
    X_train = train_set[features]
    y_train = train_set[target]
    X_test = test_set[features]
    y_test = test_set[target]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-validation R2 scores: {cv_scores}")
    print(f"Mean CV R2: {cv_scores.mean():.4f}")
    print(f"Std CV R2: {cv_scores.std():.4f}")

    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\nMean Squared Error (MSE) on test set: {mse:.2f}")
    print(f"R-squared (R2) on test set: {r2:.2f}")
    print(f"Mean Absolute Error (MAE) on test set: {mae:.2f}")

    comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    print("\nExample predictions:")
    print(comparison.head())

    joblib.dump(pipeline, model_path)
    return pipeline

if __name__ == "__main__":
    train_model(data_path='../data')