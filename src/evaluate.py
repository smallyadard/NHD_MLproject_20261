# src/evaluate.py
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error

from src.data_preprocessing import TARGET, split_and_scale

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT / "data" / "processed" / "wine_clean.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
FIG_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"
QMIN, QMAX = 3, 9


def load_test_set():
    """Dùng đúng cách chia và chuẩn hóa như lúc huấn luyện (cùng random_state)."""
    df = pd.read_csv(PROCESSED_PATH)
    _, X_test, _, y_test, _ = split_and_scale(df)
    wine_type = df.loc[X_test.index, "wine_type"]
    return X_test, y_test, wine_type


def regression_report(y_true, y_pred):
    """Sai số tổng thể và sai số trung bình theo từng mức điểm thực tế."""
    df = pd.DataFrame({"actual": y_true.values, "pred": y_pred})
    df["error"] = df["pred"] - df["actual"]
    df["abs_error"] = df["error"].abs()

    by_level = df.groupby("actual").agg(
        n=("actual", "size"),
        mean_pred=("pred", "mean"),
        MAE=("abs_error", "mean"),
        bias=("error", "mean"),
    ).round(3)

    print("Sai số theo từng mức điểm thực tế:")
    print(by_level.to_string())
    print(f"\nSai số trung bình (bias): {df['error'].mean():.3f}")
    return by_level


def classification_report(y_true, y_pred):
    """Làm tròn dự đoán về 3..9 để tính Accuracy và F1 như bài toán phân loại."""
    y_round = np.clip(np.round(y_pred), QMIN, QMAX).astype(int)
    y_true_int = y_true.astype(int).values

    acc = accuracy_score(y_true_int, y_round)
    f1_w = f1_score(y_true_int, y_round, average="weighted", zero_division=0)
    f1_m = f1_score(y_true_int, y_round, average="macro", zero_division=0)
    within_1 = np.mean(np.abs(y_round - y_true_int) <= 1)

    print(f"\nAccuracy (sau làm tròn): {acc:.4f}")
    print(f"F1 weighted: {f1_w:.4f}")
    print(f"F1 macro: {f1_m:.4f}")
    print(f"Tỷ lệ dự đoán sai không quá 1 điểm: {within_1:.4f}")

    labels = list(range(QMIN, QMAX + 1))
    cm = confusion_matrix(y_true_int, y_round, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Dự đoán (đã làm tròn)")
    ax.set_ylabel("Thực tế")
    ax.set_title("Ma trận nhầm lẫn")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    return {"accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m, "within_1": within_1}


def plot_pred_vs_actual(y_true, y_pred):
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.15, 0.15, size=len(y_true))  # tách các điểm cùng nhãn cho dễ nhìn

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true + jitter, y_pred, alpha=0.4, s=15)
    ax.plot([QMIN, QMAX], [QMIN, QMAX], "r--", label="Dự đoán hoàn hảo")
    ax.set_xlabel("Quality thực tế")
    ax.set_ylabel("Quality dự đoán")
    ax.set_title("Dự đoán và thực tế (tập test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pred_vs_actual.png", dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, y_true - y_pred, alpha=0.4, s=15)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Quality dự đoán")
    ax.set_ylabel("Phần dư (thực tế - dự đoán)")
    ax.set_title("Phần dư theo giá trị dự đoán")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residuals.png", dpi=150)
    plt.close(fig)


def error_by_wine_type(y_true, y_pred, wine_type):
    """Kiểm tra mô hình có hoạt động đều giữa rượu đỏ và rượu trắng không."""
    rows = []
    for code, name in [(0, "Đỏ"), (1, "Trắng")]:
        mask = wine_type.values == code
        rows.append({
            "loại": name,
            "n": int(mask.sum()),
            "MAE": round(mean_absolute_error(y_true[mask], y_pred[mask]), 4),
        })
    df = pd.DataFrame(rows)
    print("\nSai số theo loại rượu:")
    print(df.to_string(index=False))
    return df


def feature_importance(model, X_test, y_test):
    """Permutation importance: đo mức MAE tăng thêm khi xáo trộn một đặc trưng."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42,
        scoring="neg_mean_absolute_error",
    )
    imp = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=True)

    print("\nĐộ quan trọng của đặc trưng (tăng MAE khi xáo trộn):")
    print(imp.sort_values(ascending=False).round(4).to_string())

    fig, ax = plt.subplots(figsize=(7, 6))
    imp.plot.barh(ax=ax, color="steelblue")
    ax.set_title("Permutation importance (tập test)")
    ax.set_xlabel("Tăng MAE khi xáo trộn đặc trưng")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    return imp


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODEL_PATH)
    X_test, y_test, wine_type = load_test_set()
    y_pred = model.predict(X_test)

    print(f"Mô hình: {type(model).__name__} | Số mẫu test: {len(y_test)}\n")

    by_level = regression_report(y_test, y_pred)
    cls = classification_report(y_test, y_pred)
    plot_pred_vs_actual(y_test.values, y_pred)
    plot_residuals(y_test.values, y_pred)
    wt = error_by_wine_type(y_test.values, y_pred, wine_type)
    imp = feature_importance(model, X_test, y_test)

    by_level.to_csv(REPORTS_DIR / "error_by_quality_level.csv")
    wt.to_csv(REPORTS_DIR / "error_by_wine_type.csv", index=False)
    imp.sort_values(ascending=False).round(4).to_csv(REPORTS_DIR / "feature_importance.csv", header=["importance"])
    pd.Series(cls).round(4).to_csv(REPORTS_DIR / "classification_metrics.csv", header=["value"])

    print(f"\nĐã lưu biểu đồ và bảng vào {FIG_DIR} và {REPORTS_DIR}")


if __name__ == "__main__":
    main()