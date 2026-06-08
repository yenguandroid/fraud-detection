import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def remove_duplicates(df):
    print(f"Duplicates: {df.duplicated().sum()}")
    return df.drop_duplicates()


def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def correct_dtypes(df):

    date_cols = [
        "signup_time",
        "purchase_time"
    ]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df
# build_preprocessor()
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessor(df):

    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_cols
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols
            )
        ]
    )

    return preprocessor