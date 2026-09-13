"""
1단계: 탐색적 데이터 분석 (EDA)

notebooks/01_eda.ipynb 의 분석 로직을 노트북 밖에서도 실행할 수 있도록
정리한 스크립트입니다. survey.csv 를 읽어 결측치 처리, Target 인코딩,
불필요 컬럼 제거를 수행하고 RandomForest 로 변수 중요도를 확인한 뒤
전처리된 데이터를 data/data.csv 로 저장합니다.

실행:
    python src/eda.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_FILE = DATA_DIR / "survey.csv"
PROCESSED_FILE = DATA_DIR / "data.csv"

CAT_COLS_TO_ENCODE = ["Gender", "Customer Type", "Type of Travel", "Class"]
DROP_COLS = ["Unnamed: 0", "ID"]
TARGET = "Satisfaction"


def load_data(path: Path = RAW_FILE) -> pd.DataFrame:
    """survey.csv 를 읽어온다."""
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """수치형은 중앙값, 범주형은 최빈값으로 결측치를 채운다."""
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Satisfaction 값을 Satisfied=1, Neutral or Dissatisfied=0 으로 변경한다."""
    df = df.copy()
    df[TARGET] = df[TARGET].map({"Satisfied": 1, "Neutral or Dissatisfied": 0})
    return df


def drop_unused_columns(df: pd.DataFrame, cols: list[str] = DROP_COLS) -> pd.DataFrame:
    """분석에 의미 없는 컬럼(Unnamed: 0, ID)을 제거한다."""
    return df.drop(columns=[c for c in cols if c in df.columns])


def label_encode(df: pd.DataFrame, cols: list[str] = CAT_COLS_TO_ENCODE) -> pd.DataFrame:
    """변수 중요도 확인을 위해 범주형 변수를 LabelEncoder 로 인코딩한다."""
    df = df.copy()
    for col in cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df


def train_random_forest(
    df: pd.DataFrame, target: str = TARGET, test_size: float = 0.3, random_state: int = 1
) -> tuple[RandomForestClassifier, pd.DataFrame]:
    """RandomForest 모델을 학습하고 성능/변수 중요도를 반환한다."""
    x = df.drop(columns=target)
    y = df[target]
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )

    model = RandomForestClassifier(random_state=random_state)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)

    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    importance = pd.DataFrame(
        {"features": x.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return model, importance


def split_by_age(df: pd.DataFrame, threshold: int = 35) -> tuple[pd.DataFrame, pd.DataFrame]:
    """35세를 기준으로 데이터를 두 그룹으로 분리한다 (미션 4)."""
    younger = df.loc[df["Age"] <= threshold].reset_index(drop=True)
    older = df.loc[df["Age"] > threshold].reset_index(drop=True)
    return younger, older


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_FILE) -> None:
    df.to_csv(path, index=False)
    print(f"저장 완료: {path}")


def run_pipeline(raw_path: Path = RAW_FILE, out_path: Path = PROCESSED_FILE) -> pd.DataFrame:
    """EDA 파이프라인 전체를 실행하고 전처리된 데이터프레임을 반환한다."""
    data = load_data(raw_path)
    print(f"원본 데이터 크기: {data.shape}")

    data = handle_missing_values(data)
    data = encode_target(data)
    data = drop_unused_columns(data)

    # 노트북 원본과 동일하게 Gender/Customer Type/Type of Travel/Class 를
    # RandomForest 변수 중요도 확인을 위해 미리 라벨 인코딩한 뒤 저장한다.
    # (이후 02_modeling.py 는 이 인코딩된 상태를 그대로 이어받아 가변수화를 수행한다)
    data = label_encode(data)
    save_processed(data, out_path)

    print("\n[전체 데이터 RandomForest 변수 중요도]")
    _, importance_all = train_random_forest(data)
    print(importance_all.head())

    younger, older = split_by_age(data)
    print(f"\n35세 이하: {younger.shape}, 35세 초과: {older.shape}")

    print("\n[35세 이하 RandomForest]")
    train_random_forest(younger)

    print("\n[35세 초과 RandomForest]")
    train_random_forest(older)

    return data


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
