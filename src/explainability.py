# src/explainability.py

import shap
import pandas as pd


def create_tree_explainer(model):
    """
    Create SHAP TreeExplainer
    """

    explainer = shap.TreeExplainer(model)

    return explainer


def calculate_shap_values(explainer, X):

    shap_values = explainer.shap_values(X)

    return shap_values


def get_feature_importance(model, feature_names):

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    return importance_df