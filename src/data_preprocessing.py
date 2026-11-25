# src/data_preprocessing.py
# Tải và gộp dữ liệu, làm sạch, chia train/test, và chuẩn hóa
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

URLS = {
    "red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "white": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
}
TARGET = "quality"


def load_raw_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for wine_type, url in URLS.items():
        path = raw_dir / f"winequality-{wine_type}.csv"
        if not path.exists():
            # Giữ nguyên định dạng gốc của UCI (dấu ;)
            pd.read_csv(url, sep=";").to_csv(path, sep=";", index=False)
        df = pd.read_csv(path, sep=";")
        df["wine_type"] = wine_type
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Kiểm tra thiếu dữ liệu, xóa bản ghi trùng lặp, mã hóa loại rượu."""
    df = df.copy()

    print(f"Số dòng ban đầu: {len(df)}")
    print(f"Số ô bị thiếu: {df.isnull().sum().sum()}")

    if drop_duplicates:
        n_dup = df.duplicated().sum()
        df = df.drop_duplicates()
        print(f"Đã xóa {n_dup} bản ghi trùng lặp")

    # Mã hóa: red = 0, white = 1
    df["wine_type"] = df["wine_type"].map({"red": 0, "white": 1})

    print(f"Số dòng sau làm sạch: {len(df)}")
    return df.reset_index(drop=True)


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Chia train/test trước, sau đó chuẩn hóa. Scaler chỉ học từ tập train."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def main():
    df = load_raw_data()
    df_clean = clean_data(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(PROCESSED_DIR / "wine_clean.csv", index=False)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_clean)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    print(f"Tập train: {X_train.shape}, tập test: {X_test.shape}")
    print("Đã lưu wine_clean.csv và scaler.pkl")


if __name__ == "__main__":
    main()