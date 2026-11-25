# src/train.py
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from src.data_preprocessing import TARGET, split_and_scale

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT / "data" / "processed" / "wine_clean.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
RANDOM_STATE = 42


def get_models_and_grids():
    """Trả về các mô hình và lưới siêu tham số. Linear Regression không cần tuning."""
    return {
        "Linear Regression": (LinearRegression(), None),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 300],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 3],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 300],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
        ),
        "XGBoost": (
            XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror"),
            {
                "n_estimators": [100, 300],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
        ),
    }


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def train_all(X_train, y_train, cv: int = 5):
    """Huấn luyện từng mô hình. Tuning chỉ dùng dữ liệu train."""
    fitted, cv_rows = {}, []

    for name, (model, grid) in get_models_and_grids().items():
        start = time.time()
        if grid is None:
            model.fit(X_train, y_train)
            best_params = {}
            cv_mae = -np.mean(
                cross_val_mae(model, X_train, y_train, cv)
            )
        else:
            search = GridSearchCV(
                model,
                grid,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
            cv_mae = -search.best_score_

        elapsed = time.time() - start
        fitted[name] = model
        cv_rows.append({
            "model": name,
            "cv_MAE": round(cv_mae, 4),
            "best_params": best_params,
            "train_time_s": round(elapsed, 2),
        })
        print(f"[{name}] CV MAE = {cv_mae:.4f} | {elapsed:.1f}s | {best_params}")

    return fitted, pd.DataFrame(cv_rows)


def cross_val_mae(model, X, y, cv):
    from sklearn.model_selection import cross_val_score
    return cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")


def evaluate_on_test(fitted, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, model in fitted.items():
        metrics = evaluate(y_test, model.predict(X_test))
        rows.append({"model": name, **{k: round(v, 4) for k, v in metrics.items()}})
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_PATH)
    X_train, X_test, y_train, y_test, _ = split_and_scale(df)

    print("=== Huấn luyện (CV trên tập train) ===")
    fitted, cv_df = train_all(X_train, y_train)

    print("\n=== Đánh giá trên tập test ===")
    test_df = evaluate_on_test(fitted, X_test, y_test)
    print(test_df.to_string(index=False))

    # Gộp kết quả CV và test để đưa vào báo cáo
    summary = cv_df.merge(test_df, on="model")
    summary.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    # Lưu từng mô hình và mô hình tốt nhất theo MAE trên tập test
    for name, model in fitted.items():
        safe = name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{safe}.pkl")

    best_name = cv_df.sort_values("cv_MAE").iloc[0]["model"]
    joblib.dump(fitted[best_name], MODELS_DIR / "best_model.pkl")
    print(f"\nMô hình tốt nhất theo CV trên tập train: {best_name}")
    print(f"Đã lưu best_model.pkl và báo cáo vào {REPORTS_DIR}")


if __name__ == "__main__":
    main()