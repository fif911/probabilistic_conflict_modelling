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

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

optimal_n_components = 11

cm_features = pd.read_csv("../data/cm_features_v2.4.csv")

vdem_columns = cm_features.filter(regex="vdem")

cm_features_reduced = cm_features.drop(columns=vdem_columns)

# +
scaler = StandardScaler()
vdem_columns_centered = scaler.fit_transform(vdem_columns)

pca = PCA()
pca.fit(vdem_columns_centered)

cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
# -

pca = PCA(n_components=optimal_n_components)
principal_components = pca.fit_transform(vdem_columns_centered)

# +
pca = PCA(n_components=optimal_n_components)
vdem_pca = pca.fit_transform(vdem_columns_centered)

vdem_pca_df = pd.DataFrame(
    vdem_pca, columns=[f"vdem_pca_{i + 1}" for i in range(optimal_n_components)]
)

combined_data = cm_features_reduced.join(vdem_pca_df)
# -

combined_data.to_csv("../data/cm_features_v2.5.csv", index=False)
