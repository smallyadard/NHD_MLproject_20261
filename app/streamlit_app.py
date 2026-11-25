# app/streamlit_app.py
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
DATA_PATH = ROOT / "data" / "processed" / "wine_clean.csv"
TARGET = "quality"

# Thứ tự cột phải khớp đúng với lúc huấn luyện
FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_ranges():
    """Lấy min/max và trung vị của từng đặc trưng từ dữ liệu sạch để làm giới hạn nhập liệu."""
    df = pd.read_csv(DATA_PATH)
    return df[FEATURES].agg(["min", "max", "median"]).T


def predict_quality(model, scaler, values: dict) -> float:
    X = pd.DataFrame([values], columns=FEATURES)
    X_scaled = scaler.transform(X)
    raw = float(model.predict(X_scaled)[0])
    return raw


def out_of_range(values: dict, ranges: pd.DataFrame) -> list[str]:
    warnings = []
    for feat in FEATURES:
        lo, hi = ranges.loc[feat, "min"], ranges.loc[feat, "max"]
        if values[feat] < lo or values[feat] > hi:
            warnings.append(f"{feat} = {values[feat]} nằm ngoài khoảng [{lo}, {hi}]")
    return warnings


def main():
    st.set_page_config(page_title="Dự đoán chất lượng rượu vang", page_icon="🍷")
    st.title("Dự đoán chất lượng rượu vang")
    st.write(
        "Nhập các chỉ số hóa học của rượu để nhận điểm chất lượng dự đoán "
        "(thang điểm từ 3 đến 9)."
    )

    model, scaler = load_artifacts()
    ranges = load_ranges()

    values = {}
    col1, col2 = st.columns(2)
    fields = [f for f in FEATURES if f != "wine_type"]

    for i, feat in enumerate(fields):
        target_col = col1 if i < len(fields) / 2 else col2
        lo = float(ranges.loc[feat, "min"])
        hi = float(ranges.loc[feat, "max"])
        default = float(ranges.loc[feat, "median"])
        step = (hi - lo) / 200
        with target_col:
            values[feat] = st.number_input(
                feat,
                min_value=0.0,
                max_value=hi * 2,  # cho phép vượt khoảng để kiểm tra cảnh báo
                value=default,
                step=step,
                format="%.4f" if hi < 1 else "%.2f",
                help=f"Khoảng trong dữ liệu huấn luyện: {lo:.3f} đến {hi:.3f}",
            )

    wine_label = st.radio("Loại rượu", ["Rượu đỏ", "Rượu trắng"], horizontal=True)
    values["wine_type"] = 0 if wine_label == "Rượu đỏ" else 1

    if st.button("Dự đoán", type="primary"):
        warnings = out_of_range(values, ranges)
        if warnings:
            st.warning(
                "Một số giá trị nằm ngoài khoảng dữ liệu huấn luyện, "
                "kết quả có thể kém tin cậy:\n\n- " + "\n- ".join(warnings)
            )

        raw = predict_quality(model, scaler, values)
        rounded = int(np.clip(round(raw), 3, 9))

        st.subheader("Kết quả")
        c1, c2 = st.columns(2)
        c1.metric("Điểm dự đoán (liên tục)", f"{raw:.2f}")
        c2.metric("Điểm làm tròn", f"{rounded} / 9")

        st.caption(
            "Mô hình được huấn luyện trên dữ liệu đánh giá của con người, "
            "điểm chỉ mang tính tham khảo."
        )


if __name__ == "__main__":
    main()