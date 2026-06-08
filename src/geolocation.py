# Convert IP to Integer
import pandas as pd
import numpy as np


def ip_to_int(ip):
    try:
        parts = ip.split(".")
        return (
            int(parts[0]) * 256**3 +
            int(parts[1]) * 256**2 +
            int(parts[2]) * 256 +
            int(parts[3])
        )
    except:
        return np.nan
   # Merge Country Data
def merge_country(fraud_df, country_df):

    fraud_df["ip_int"] = fraud_df["ip_address"]

    fraud_df = fraud_df.sort_values("ip_int")

    country_df = country_df.sort_values("lower_bound_ip_address")

    merged = pd.merge_asof(
        fraud_df,
        country_df,
        left_on="ip_int",
        right_on="lower_bound_ip_address",
        direction="backward"
    )

    merged = merged[
        merged["ip_int"] <= merged["upper_bound_ip_address"]
    ]

    return merged
