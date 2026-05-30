"""
Скрипт обучения моделей на реальном датасете Craigslist.

Использование:
    python -m src.train --model xgboost
    python -m src.train --model rf

Данные:
    Скачиваются с Kaggle автоматически.
    Если Kaggle недоступен, разархивируется data/vehicles.7z.

Сохраняет:
    artifacts/models/car_price_model.joblib
    artifacts/models/model_meta.json
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR  = Path("artifacts/models")
DATA_DIR       = Path("data/raw")
CSV_PATH       = DATA_DIR / "vehicles.csv"
ARCHIVE_PATH   = Path("data/vehicles.7z")
KAGGLE_DATASET = "austinreese/craigslist-carstrucks-data"
CURRENT_YEAR   = 2022
RANDOM_STATE   = 42

FEATURES = [
    "manufacturer", "condition", "cylinders", "fuel", "odometer",
    "title_status", "transmission", "drive", "type", "paint_color",
    "state", "car_age",
]
CAT_FEATURES = [
    "manufacturer", "condition", "cylinders", "fuel",
    "title_status", "transmission", "drive", "type", "paint_color", "state",
]
TARGET = "price"


def acquire_data() -> Path:
    """
    Получить путь к vehicles.csv.
    Порядок действий:
      1. Если файл уже есть, использовать его.
      2. Попытаться скачать через Kaggle CLI.
      3. Если Kaggle недоступен, разархивировать data/vehicles.7z.
    """
    if CSV_PATH.exists():
        logger.info("Датасет уже есть: %s", CSV_PATH)
        return CSV_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Попытка скачать датасет с Kaggle...")
    kaggle_ok = _try_kaggle_download()

    if kaggle_ok and CSV_PATH.exists():
        logger.info("Датасет успешно скачан с Kaggle")
        return CSV_PATH

    logger.warning("Kaggle недоступен, пробуем разархивировать %s ...", ARCHIVE_PATH)
    _extract_7z()

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"vehicles.csv не найден после всех попыток. "
            f"Положите архив в {ARCHIVE_PATH} или настройте Kaggle CLI."
        )

    return CSV_PATH


def _try_kaggle_download() -> bool:
    """Скачать датасет через kaggle CLI. Возвращает True при успехе."""
    if not shutil.which("kaggle"):
        logger.warning("kaggle CLI не установлен")
        return False
    try:
        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", KAGGLE_DATASET,
                "--path", str(DATA_DIR),
                "--unzip",
            ],
            timeout=600,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True
        logger.warning("kaggle завершился с ошибкой: %s", result.stderr.strip())
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Превышено время ожидания скачивания с Kaggle")
        return False
    except Exception as exc:
        logger.warning("Ошибка при вызове kaggle: %s", exc)
        return False


def _extract_7z() -> None:
    """Разархивировать vehicles.7z в data/raw/."""
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"Архив {ARCHIVE_PATH} не найден. "
            f"Положите vehicles.7z в папку data/ или настройте Kaggle CLI."
        )

    extractor = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
    if extractor:
        logger.info("Распаковка через %s ...", extractor)
        result = subprocess.run(
            [extractor, "x", str(ARCHIVE_PATH), f"-o{DATA_DIR}", "-y"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Ошибка распаковки 7z: {result.stderr.strip()}")
        return

    try:
        import py7zr
        logger.info("Распаковка через py7zr ...")
        with py7zr.SevenZipFile(ARCHIVE_PATH, mode="r") as archive:
            archive.extractall(path=DATA_DIR)
        return
    except ImportError:
        pass

    raise RuntimeError(
        "Не найден инструмент для распаковки .7z. "
        "Установите 7-Zip (7z) или выполните: pip install py7zr"
    )


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Загрузить и очистить сырой vehicles.csv."""
    logger.info("Загрузка данных из %s ...", csv_path)

    src_features = [
        "year", "manufacturer", "condition", "cylinders", "fuel", "odometer",
        "title_status", "transmission", "drive", "type", "paint_color", "state", TARGET,
    ]
    df = pd.read_csv(csv_path, usecols=src_features, low_memory=False)
    logger.info("Загружено строк: %d", len(df))

    df = df[(df[TARGET] >= 500) & (df[TARGET] <= 150_000)]
    df = df[(df["year"] >= 1980) & (df["year"] <= CURRENT_YEAR)]
    df = df[df["odometer"] <= 500_000]

    df["car_age"] = CURRENT_YEAR - df["year"]
    df.drop(columns=["year"], inplace=True)

    for col in CAT_FEATURES:
        df[col] = df[col].fillna("unknown").astype(str).str.lower().str.strip()

    for col in ["odometer", "car_age"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=[TARGET])

    logger.info("После очистки строк: %d", len(df))
    return df


def build_encoders(df: pd.DataFrame) -> dict[str, LabelEncoder]:
    encoders = {}
    for col in CAT_FEATURES:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def encode(df: pd.DataFrame, encoders: dict) -> np.ndarray:
    df = df.copy()
    for col in CAT_FEATURES:
        le = encoders[col]
        known = set(le.classes_)
        df[col] = df[col].apply(lambda v: v if v in known else le.classes_[0])
        df[col] = le.transform(df[col].astype(str))
    return df[FEATURES].values


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    logger.info("%s | MAE=%.0f RMSE=%.0f R2=%.3f MAPE=%.1f%%", name, mae, rmse, r2, mape)
    return {"MAE": round(mae), "RMSE": round(rmse), "R2": round(r2, 4), "MAPE": round(mape, 1)}


def train(df: pd.DataFrame, model_choice: str = "rf") -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    encoders = build_encoders(df)
    X = encode(df, encoders)
    y = np.log1p(df[TARGET].values)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    logger.info("Train=%d Test=%d Features=%d", len(X_tr), len(X_te), X_tr.shape[1])

    models_map = {
        "ridge": (
            Ridge(alpha=10.0),
            "Ridge",
            "Ridge",
        ),
        "rf": (
            RandomForestRegressor(
                n_estimators=300,
                max_depth=25,
                min_samples_leaf=3,
                max_features=0.8,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "RandomForest",
            "RandomForestRegressor",
        ),
        "xgboost": (
            XGBRegressor(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbosity=0,
            ),
            "XGBoost",
            "XGBRegressor",
        ),
    }

    model_obj, model_name, model_type = models_map[model_choice]
    logger.info("Обучение %s ...", model_name)
    t0 = time.time()
    model_obj.fit(X_tr, y_tr)
    logger.info("Обучено за %.1f с", time.time() - t0)

    y_pred_log = model_obj.predict(X_te)
    metrics = evaluate(model_name, np.expm1(y_te), np.expm1(y_pred_log))

    logger.info("Обучение квантильных моделей GBR q05 и q95 ...")
    gbr_kw = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    model_q05 = GradientBoostingRegressor(loss="quantile", alpha=0.05, **gbr_kw)
    model_q95 = GradientBoostingRegressor(loss="quantile", alpha=0.95, **gbr_kw)
    model_q05.fit(X_tr, y_tr)
    model_q95.fit(X_tr, y_tr)
    logger.info("Квантильные модели обучены")

    joblib.dump(
        {
            "model_median": model_obj,
            "model_q05": model_q05,
            "model_q95": model_q95,
            "encoders": encoders,
        },
        ARTIFACTS_DIR / "car_price_model.joblib",
    )

    meta = {
        "model_name": model_name,
        "model_type": model_type,
        "trained_on": "Craigslist Cars & Trucks dataset",
        "metrics": metrics,
        "features": FEATURES,
    }
    with open(ARTIFACTS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("Модели сохранены в %s", ARTIFACTS_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение модели оценки цены автомобиля")
    parser.add_argument(
        "--model",
        choices=["ridge", "rf", "xgboost"],
        default="rf",
        help="Алгоритм обучения (default: rf)",
    )
    args = parser.parse_args()

    csv_path = acquire_data()
    df = load_and_clean(csv_path)
    train(df, model_choice=args.model)


if __name__ == "__main__":
    main()
