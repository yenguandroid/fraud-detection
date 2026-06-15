# Model training separate pyton class
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    confusion_matrix
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)


def train_logistic_regression(X_train, y_train):

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, 15]
    }

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=2,
    scoring="f1",
    n_jobs=-1
      )

    grid_search.fit(
        X_train,
        y_train
    )

    rf_model = grid_search.best_estimator_
    return rf_model

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    auc_pr = average_precision_score(
        y_test,
        y_prob
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    return {
        "AUC_PR": auc_pr,
        "F1_Score": f1,
        "Confusion_Matrix": cm
    }


def cross_validate_model(model, X, y):

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=skf,
        scoring="f1"
    )

    return scores.mean(), scores.std()
