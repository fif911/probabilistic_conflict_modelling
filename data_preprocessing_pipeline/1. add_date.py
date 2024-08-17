# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd

cm_features = pd.read_parquet("../data/cm_features_v2.0.parquet")

cm_features["date"] = pd.to_datetime(
    cm_features["month_id"].apply(
        lambda x: f"{1980 + (x - 1) // 12}-{((x - 1) % 12) + 1}-01"
    )
)

cm_features.to_csv("../data/cm_features_v2.1.csv", index=False)

# print max date
print(cm_features["date"].max())
# print amount of countries with max date
print(
    cm_features[cm_features["date"] == cm_features["date"].max()][
        "country_id"
    ].nunique()
)
