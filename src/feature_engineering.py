import pandas as pd


def create_time_features(df):

    df["hour_of_day"] = df["purchase_time"].dt.hour

    df["day_of_week"] = df["purchase_time"].dt.dayofweek

    df["time_since_signup"] = (
        df["purchase_time"] -
        df["signup_time"]
    ).dt.total_seconds()

    return df
#Transaction Frequency
def transaction_frequency(df):

    user_counts = (
        df.groupby("user_id")
        .size()
        .reset_index(name="transaction_count")
    )

    df = df.merge(
        user_counts,
        on="user_id",
        how="left"
    )

    return df
