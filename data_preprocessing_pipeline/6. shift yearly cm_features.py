# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd

version = "v2.5"
cm_features = pd.read_csv(f"../data/cm_features_{version}.csv")
cm_features["date"] = pd.to_datetime(cm_features["date"])
# create country_id to ccode mapping
country_id_to_ccode = cm_features[["country_id", "ccode"]].drop_duplicates()
cm_features

# +
import numpy as np

# TODO: Figure out partial creation for 2024
prediction_years = [2018, 2019, 2020, 2021, 2022, 2023]
prediction_window = 14
column_name = f"ged_sb_{prediction_window}"
for prediction_year in prediction_years:
    print(f"Prediction year: {prediction_year}")
    features_to_oct = pd.Timestamp(
        year=prediction_year - 1, month=10, day=1
    )  # 2021-Oct-01
    cm_features_year = cm_features[cm_features["date"] <= features_to_oct]
    # get last month_id
    last_month_id = cm_features_year["month_id"].max()
    print(f"Last month_id: {last_month_id}")
    last_month_cm_features = cm_features_year[
        cm_features_year["month_id"] == last_month_id
    ]
    # create 3 month window based on last_month_cm_features
    two_month_buffer_features = []
    for counter in range(1, 3):
        temp_month = last_month_cm_features.copy()
        temp_month["month_id"] = last_month_id + counter
        temp_month["ged_sb"] = np.nan
        two_month_buffer_features.append(temp_month)

    two_month_buffer_features = pd.concat(two_month_buffer_features)
    print(
        f"two_month_buffer_features: {two_month_buffer_features['month_id'].unique()}"
    )
    # read actuals for this year
    actuals_year = pd.read_parquet(
        f"../actuals/cm/window=Y{prediction_year}/cm_actuals_{prediction_year}.parquet"
    )
    actuals_year.rename(
        columns={"outcome": "ged_sb"}, inplace=True
    )  # rename outcome to ged_sb
    actuals_year.reset_index(drop=False, inplace=True)
    print(f"actuals_year: {actuals_year['month_id'].unique()}")
    # add ccode column to actuals_year
    actuals_year = actuals_year.merge(country_id_to_ccode, on="country_id", how="left")
    actuals_year = actuals_year[~actuals_year["ccode"].isnull()]

    _gap_months = two_month_buffer_features["month_id"].unique() - 11 - 3
    test_set_months_min = cm_features_year["month_id"].max() - 11
    test_set_months_max = cm_features_year["month_id"].max()
    print(f"_gap_months: expected empty months because of the gap: {_gap_months}")
    print(f"test set is from {test_set_months_min} to {test_set_months_max}")
    print(f"two month buffer months: {two_month_buffer_features['month_id'].unique()}")

    cm_features_year = pd.concat(
        [cm_features_year, two_month_buffer_features, actuals_year]
    )
    cm_features_year.reset_index(drop=True, inplace=True)

    cm_features_year[column_name] = cm_features_year.groupby("ccode")["ged_sb"].shift(
        -prediction_window
    )
    # drop rows with these months: actuals_year['month_id'].unique()
    cm_features_year = cm_features_year[
        ~cm_features_year["month_id"].isin(actuals_year["month_id"].unique())
    ]
    # drop rows with two_month_buffer_features['month_id'].unique()
    cm_features_year = cm_features_year[
        ~cm_features_year["month_id"].isin(
            two_month_buffer_features["month_id"].unique()
        )
    ]

    month_ids_is_null = cm_features_year[cm_features_year[column_name].isnull()][
        "month_id"
    ].unique()
    print("month_ids_is_null: ", month_ids_is_null)
    assert all(_gap_months == month_ids_is_null), "Unexpected missing months"

    # drop gap months
    cm_features_year = cm_features_year[~cm_features_year["month_id"].isin(_gap_months)]

    cm_features_year.to_csv(
        f"../data/cm_features_{version}_ica_Y{prediction_year}.csv", index=False
    )

print("All done!")

# -
