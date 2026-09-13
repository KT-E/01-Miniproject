"""
3단계: 모델 추가 학습 (파인튜닝)

notebooks/03_finetuning.ipynb 의 로직을 노트북 밖에서 실행할 수 있도록
정리한 스크립트입니다. 새로 수집된 data/new_train.csv, data/new_test.csv 에
기존 base_model.keras 를 어떻게 적용/재학습하는 것이 가장 좋은지 3가지
전략을 비교합니다.

- 방법 1: 원본 데이터 + 신규 데이터를 모두 합쳐 처음부터 재학습
- 방법 2: 기존 base_model 을 신규 데이터만으로 이어서 학습 (계속 학습)
- 방법 3: 방법 1의 모델(model1)의 앞쪽 레이어를 동결하고 새 출력층만
          신규 데이터로 미세조정 (fine-tuning)

실행:
    python src/finetuning.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, recall_score
from sklearn.utils import class_weight
from tensorflow.keras import regularizers
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

BASE_MODEL_FILE = MODELS_DIR / "base_model.keras"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
ORIGINAL_DATA_FILE = DATA_DIR / "data.csv"
NEW_TRAIN_FILE = DATA_DIR / "new_train.csv"
NEW_TEST_FILE = DATA_DIR / "new_test.csv"
FINETUNED_MODEL_FILE = MODELS_DIR / "finetuned_model.keras"

# 새 데이터에 대한 결측치 보간 대상 (노트북에서 분포 확인 후 선정한 컬럼)
IMPUTE_COLS = ["Age", "Inflight wifi service", "Food and drink", "Seat comfort", "Arrival Delay in Minutes"]

CATEGORICAL_DUMMY_COLS = [
    "Type of Travel",
    "Class",
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


def build_model_input(df: pd.DataFrame, scaler) -> tuple[pd.DataFrame, np.ndarray | None]:
    """새 원시 데이터를 기존 base_model 이 학습된 형태(x, y)로 변환하는 파이프라인.

    결측치 처리 -> Target 인코딩 -> 불필요 컬럼 제거 -> 라벨 인코딩 ->
    가변수화 -> 스케일러 피처에 맞춰 정렬 -> 스케일링 순서로 처리한다.
    """
    df = df.copy()

    # 1. 결측치 처리 (수치형/점수형 변수: 중앙값으로 대체)
    for col in IMPUTE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 2. Target 값 변경 및 결측치 행 제거
    y = None
    if "Satisfaction" in df.columns:
        df["Satisfaction"] = df["Satisfaction"].str.lower()
        df["Satisfaction"] = df["Satisfaction"].map({"satisfied": 1, "neutral or dissatisfied": 0})
        df = df.dropna(subset=["Satisfaction"])
        if df.empty:
            return pd.DataFrame(), np.array([])
        y = df["Satisfaction"].values
        df = df.drop(columns=["Satisfaction"])

    # 3. 불필요한 변수 제거
    df = df.drop(columns=["Unnamed: 0", "ID"], errors="ignore")

    # 4. 라벨 인코딩
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    if "Customer Type" in df.columns:
        df["Customer Type"] = df["Customer Type"].str.lower().map(
            {"loyal customer": 1, "disloyal customer": 0}
        )

    # 5. 가변수화 (One-Hot Encoding)
    existing_cat_cols = [c for c in CATEGORICAL_DUMMY_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cat_cols)

    # 6. 학습 당시 스케일러가 기억하는 컬럼 순서/구성에 맞춘다
    try:
        cols = scaler.feature_names_in_.tolist()
        for col in cols:
            if col not in df.columns:
                df[col] = 0
        x = df[cols]
    except AttributeError:
        x = df

    if x.empty:
        return pd.DataFrame(), np.array([])

    # 7. 스케일링
    x_scaled = scaler.transform(x)
    x_final = pd.DataFrame(x_scaled, columns=x.columns)

    return x_final, y


def load_artifacts():
    base_model = load_model(BASE_MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return base_model, scaler


def load_datasets():
    original_df = pd.read_csv(ORIGINAL_DATA_FILE)
    new_train = pd.read_csv(NEW_TRAIN_FILE)
    new_test = pd.read_csv(NEW_TEST_FILE)
    return original_df, new_train, new_test


def evaluate_base_model(base_model, x_val, y_val) -> None:
    """미션 9: 신규 평가 데이터로 기존 base_model 성능을 확인한다."""
    y_pred = base_model.predict(x_val)
    y_pred = (y_pred > 0.5).astype(int)
    print("[기본 모델(base_model) - 신규 평가 데이터 성능]")
    print(classification_report(y_val, y_pred))


def strategy_1_retrain_on_combined(x_combined, y_combined, x_val, y_val, class_weights):
    """방법 1: 원본 + 신규 데이터를 모두 합쳐 동일 구조 모델을 처음부터 재학습."""
    clear_session()
    n_features = x_combined.shape[1]

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
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        x_combined,
        y_combined,
        epochs=300,
        batch_size=32,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1,
    )

    y_pred = (model.predict(x_val) > 0.5).astype(int)
    print("\n[방법 1: 결합 데이터 재학습 결과]")
    print(classification_report(y_val, y_pred))
    return model


def strategy_2_continue_training(x_train_new, y_train_new, x_val, y_val, class_weights):
    """방법 2: 기존 base_model 을 신규 학습 데이터만으로 이어서 학습."""
    model = load_model(BASE_MODEL_FILE)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        x_train_new,
        y_train_new,
        epochs=300,
        batch_size=32,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1,
    )

    y_pred = (model.predict(x_val) > 0.5).astype(int)
    print("\n[방법 2: base_model 이어서 학습 결과]")
    print(classification_report(y_val, y_pred))
    return model


def strategy_3_fine_tune(base_model, model1, x_train_new, y_train_new, x_val, y_val, class_weights):
    """방법 3: 방법 1에서 만든 model1 의 앞쪽 레이어를 동결하고 새 출력층만 미세조정."""
    clear_session()

    base_model_ft = model1
    if base_model_ft.layers:
        base_model_ft.pop()

    # 원본 base_model 기준으로 마지막 5개 레이어를 제외하고 동결
    for layer in base_model.layers[:-5]:
        layer.trainable = False

    model3 = Sequential(
        [
            base_model_ft,
            Dense(8, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model3.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model3.fit(
        x_train_new,
        y_train_new,
        epochs=200,
        batch_size=32,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1,
    )

    y_pred_proba = model3.predict(x_val)
    y_pred = (y_pred_proba > 0.5).astype(int)
    print("\n[방법 3: 레이어 동결 + 미세조정 결과 (threshold=0.5)]")
    print(classification_report(y_val, y_pred))

    best_threshold = find_best_threshold(y_val, y_pred_proba)
    print(f"최소 recall 0.9 이상을 만족하는 최적 임계값: {best_threshold:.2f}")

    return model3, best_threshold


def find_best_threshold(y_val, y_pred_proba, min_recall: float = 0.9) -> float:
    """recall >= min_recall 을 만족하면서 threshold(=precision)가 가장 높은 값을 찾는다."""
    candidates = [
        (t, recall_score(y_val, (y_pred_proba > t).astype(int))) for t in np.arange(0.1, 0.91, 0.01)
    ]
    valid = [(t, r) for t, r in candidates if r >= min_recall]
    if not valid:
        return 0.5
    return max(valid, key=lambda x: x[0])[0]


def run_pipeline():
    base_model, scaler = load_artifacts()
    original_df, new_train, new_test = load_datasets()

    x_val, y_val = build_model_input(new_test, scaler)
    x_train_new, y_train_new = build_model_input(new_train, scaler)

    print(f"x_train_new: {x_train_new.shape}, x_val: {x_val.shape}")

    evaluate_base_model(base_model, x_val, y_val)

    combined_raw = pd.concat([original_df, new_train, new_test], ignore_index=True)
    x_combined, y_combined = build_model_input(combined_raw, scaler)

    class_weights_combined = dict(
        enumerate(class_weight.compute_class_weight("balanced", classes=np.unique(y_combined), y=y_combined))
    )
    class_weights_new = dict(
        enumerate(
            class_weight.compute_class_weight("balanced", classes=np.unique(y_train_new), y=y_train_new)
        )
    )

    model1 = strategy_1_retrain_on_combined(x_combined, y_combined, x_val, y_val, class_weights_combined)
    strategy_2_continue_training(x_train_new, y_train_new, x_val, y_val, class_weights_new)
    model3, best_threshold = strategy_3_fine_tune(
        base_model, model1, x_train_new, y_train_new, x_val, y_val, class_weights_new
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model3.save(FINETUNED_MODEL_FILE)
    print(f"\n저장 완료: {FINETUNED_MODEL_FILE} (권장 판정 임계값: {best_threshold:.2f})")

    return model1, model3


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
