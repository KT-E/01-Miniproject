"""
2단계: 기본 모델링

notebooks/02_modeling.ipynb 의 로직을 노트북 밖에서 실행할 수 있도록
정리한 스크립트입니다. data/data.csv 를 읽어 가변수화/스케일링을 수행하고
클래스 불균형을 보정한 딥러닝 모델(L2 정규화 + Dropout + class_weight)을
학습한 뒤 models/base_model.keras, models/scaler.pkl 로 저장합니다.

실행:
    python src/modeling.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from tensorflow.keras import regularizers
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

PROCESSED_FILE = DATA_DIR / "data.csv"
MODEL_FILE = MODELS_DIR / "base_model.keras"
SCALER_FILE = MODELS_DIR / "scaler.pkl"

TARGET = "Satisfaction"

# 0~5 리커트 척도 변수 (가변수화 대상). 0점은 '해당 없음'을 의미할 수 있어
# 그대로 수치로 넣지 않고 One-Hot 인코딩한다.
DUMMY_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]


def load_processed_data(path: Path = PROCESSED_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def dummy_encode(df: pd.DataFrame, cols: list[str] = DUMMY_COLS) -> pd.DataFrame:
    """0~5 정수 변수를 Category 형으로 바꾼 뒤 가변수화한다."""
    df = df.copy()
    categories = [0, 1, 2, 3, 4, 5]
    for col in cols:
        df[col] = pd.Categorical(df[col], categories=categories)
    return pd.get_dummies(df, columns=cols, drop_first=True)


def split_and_scale(
    df: pd.DataFrame, target: str = TARGET, test_size: float = 0.2, random_state: int = 1
):
    """x, y 분리 후 8:2 로 나누고 MinMaxScaler 로 스케일링한다."""
    x = df.drop(columns=target)
    y = df[target]

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    return x_train_scaled, x_val_scaled, y_train, y_val, scaler


def build_model(n_features: int) -> Sequential:
    """L2 정규화 + Dropout 을 적용한 딥러닝 분류 모델을 만든다.

    노트북에서 여러 아키텍처(Baseline, 은닉층 5개, Dropout, EarlyStopping 조합)를
    비교한 뒤, 클래스 불균형에 가장 강건했던 이 구조를 최종안으로 채택했다.
    """
    clear_session()
    model = Sequential(
        [
            Input(shape=(n_features,)),
            Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.4),
            Dense(16, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.3),
            Dense(8, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.4),
            Dense(4, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    return model


def compute_class_weights(y_train) -> dict[int, float]:
    """Satisfaction 클래스 불균형(만족 >> 불만족)을 보정하기 위한 가중치."""
    weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    return dict(enumerate(weights))


def train_model(model: Sequential, x_train, y_train, x_val, y_val, class_weights: dict):
    model.compile(optimizer=Adam(learning_rate=0.005), loss="binary_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=0.001, patience=10, restore_best_weights=True, verbose=1
    )
    # 노트북 원본과 동일하게 학습 데이터 내부에서 validation_split 으로 검증셋을 나눈다.
    # (x_val, y_val 은 최종 성능 평가에만 사용)
    history = model.fit(
        x_train,
        y_train,
        epochs=50,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1,
    )
    return history


def evaluate(model: Sequential, x_val, y_val) -> None:
    y_pred = model.predict(x_val)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    print(classification_report(y_val, y_pred))


def save_artifacts(model: Sequential, scaler: MinMaxScaler) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_FILE)
    model.save(MODEL_FILE)
    print(f"저장 완료: {MODEL_FILE}, {SCALER_FILE}")


def run_pipeline(data_path: Path = PROCESSED_FILE):
    data = load_processed_data(data_path)
    data = dummy_encode(data)

    x_train, x_val, y_train, y_val, scaler = split_and_scale(data)

    model = build_model(n_features=x_train.shape[1])
    model.summary()

    class_weights = compute_class_weights(y_train)
    print(f"클래스 가중치: {class_weights}")

    train_model(model, x_train, y_train, x_val, y_val, class_weights)
    evaluate(model, x_val, y_val)
    save_artifacts(model, scaler)

    return model, scaler


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
