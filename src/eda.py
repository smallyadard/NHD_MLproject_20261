# src/eda.py
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # vẽ và lưu file, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT / "data" / "processed" / "wine_clean.csv"
FIG_DIR = ROOT / "reports" / "figures"
TARGET = "quality"


def load_clean(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def describe_data(df: pd.DataFrame) -> None:
    """In thống kê mô tả và phân bố của biến mục tiêu."""
    print("Kích thước:", df.shape)
    print("\nThống kê mô tả:")
    print(df.describe().T.round(3))
    print("\nPhân bố quality:")
    print(df[TARGET].value_counts().sort_index())
    print("\nSố mẫu theo loại rượu (0 = đỏ, 1 = trắng):")
    print(df["wine_type"].value_counts())


def plot_quality_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET, ax=ax, color="steelblue")
    ax.set_title("Phân bố điểm chất lượng")
    ax.set_xlabel("Quality")
    ax.set_ylabel("Số mẫu")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "quality_distribution.png", dpi=150)
    plt.close(fig)


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Tính tương quan, in mức độ liên quan với quality và các cặp đặc trưng tương quan mạnh."""
    corr = df.corr(numeric_only=True)

    # Tương quan của từng đặc trưng với quality, sắp xếp theo độ mạnh
    target_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print("\nTương quan với quality (sắp theo độ mạnh):")
    print(target_corr.round(3))

    # Các cặp đặc trưng tương quan mạnh với nhau (dấu hiệu đa cộng tuyến)
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack()
    strong = pairs[pairs.abs() > 0.6].sort_values(key=abs, ascending=False)
    print("\nCác cặp đặc trưng có |r| > 0.6:")
    print(strong.round(3))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Ma trận tương quan")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    target_corr.round(3).to_csv(ROOT / "reports" / "target_correlation.csv", header=["correlation"])
    return corr


def plot_key_features(df: pd.DataFrame, features=("alcohol", "volatile acidity", "density")) -> None:
    """Boxplot cho các đặc trưng quan trọng theo từng mức quality."""
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
    for ax, feature in zip(axes, features):
        sns.boxplot(data=df, x=TARGET, y=feature, ax=ax)
        ax.set_title(f"{feature} theo quality")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "key_features_boxplot.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean()

    describe_data(df)
    plot_quality_distribution(df)
    correlation_analysis(df)
    plot_key_features(df)

    print(f"\nĐã lưu biểu đồ vào: {FIG_DIR}")


if __name__ == "__main__":
    main()