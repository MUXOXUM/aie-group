"""
CarPricePredictor загрузка модели, предсказание, доверительные интервалы.

Поддерживает два режима:
  1. Загрузка сохранённой модели из artifacts/models/
  2. Обучение на лету на синтетических данных (demo режим)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts/models"))
MODEL_PATH    = ARTIFACTS_DIR / "car_price_model.joblib"
META_PATH     = ARTIFACTS_DIR / "model_meta.json"

FEATURES = [
    "manufacturer", "condition", "cylinders", "fuel", "odometer",
    "title_status", "transmission", "drive", "type", "paint_color",
    "state", "car_age",
]
CAT_FEATURES = [
    "manufacturer", "condition", "cylinders", "fuel",
    "title_status", "transmission", "drive", "type", "paint_color", "state",
]
NUM_FEATURES = ["odometer", "car_age"]

CAT_MAPS: dict[str, list[str]] = {
    "manufacturer": [
        "acura", "audi", "bmw", "buick", "cadillac", "chevrolet", "chrysler", "dodge",
        "ford", "gmc", "honda", "hyundai", "infiniti", "jeep", "kia", "lexus", "lincoln",
        "mazda", "mercedes-benz", "mitsubishi", "nissan", "pontiac", "ram", "saturn",
        "subaru", "toyota", "volkswagen", "volvo", "unknown",
    ],
    "condition": ["excellent", "fair", "good", "like new", "new", "salvage", "unknown"],
    "cylinders": [
        "10 cylinders", "12 cylinders", "3 cylinders", "4 cylinders",
        "5 cylinders", "6 cylinders", "8 cylinders", "other", "unknown",
    ],
    "fuel":         ["diesel", "electric", "gas", "hybrid", "other", "unknown"],
    "title_status": ["clean", "lien", "missing", "parts only", "rebuilt", "salvage", "unknown"],
    "transmission": ["automatic", "manual", "other", "unknown"],
    "drive":        ["4wd", "fwd", "rwd", "unknown"],
    "type": [
        "SUV", "bus", "convertible", "coupe", "hatchback", "mini-van",
        "offroad", "other", "pickup", "sedan", "truck", "unknown", "van", "wagon",
    ],
    "paint_color": [
        "black", "blue", "brown", "custom", "green", "grey", "orange",
        "purple", "red", "silver", "unknown", "white", "yellow",
    ],
    "state": [
        "ak", "al", "ar", "az", "ca", "co", "ct", "dc", "de", "fl", "ga", "hi", "ia",
        "id", "il", "in", "ks", "ky", "la", "ma", "md", "me", "mi", "mn", "mo", "ms",
        "mt", "nc", "nd", "ne", "nh", "nj", "nm", "nv", "ny", "oh", "ok", "or", "pa",
        "ri", "sc", "sd", "tn", "tx", "ut", "va", "vt", "wa", "wi", "wv", "wy", "unknown",
    ],
}


class CarPricePredictor:
    """Обёртка над обученной моделью с поддержкой CI через квантильную регрессию."""

    def __init__(self) -> None:
        self._model_median: Any = None
        self._model_q05:    Any = None
        self._model_q95:    Any = None
        self._encoders: dict[str, LabelEncoder] = {}
        self._meta: dict = {}
        self._ready = False

    def load(self) -> None:
        """Загрузить модель с диска или обучить demo модель."""
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        if MODEL_PATH.exists() and META_PATH.exists():
            logger.info("Загрузка модели из %s", MODEL_PATH)
            bundle = joblib.load(MODEL_PATH)
            self._model_median = bundle["model_median"]
            self._model_q05    = bundle.get("model_q05")
            self._model_q95    = bundle.get("model_q95")
            self._encoders     = bundle["encoders"]
            with open(META_PATH) as f:
                self._meta = json.load(f)
        else:
            logger.warning("Модель не найдена, обучаем demo модель на синтетических данных...")
            self._train_demo()
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, features: dict) -> dict:
        """
        Предсказать цену автомобиля.

        Возвращает dict с полями:
            predicted_price, ci_lower, ci_upper, confidence, model_used
        """
        X = self._encode([features])

        pred_log = float(self._model_median.predict(X)[0])
        price    = float(np.expm1(pred_log))

        if self._model_q05 is not None and self._model_q95 is not None:
            ci_lo = float(np.expm1(self._model_q05.predict(X)[0]))
            ci_hi = float(np.expm1(self._model_q95.predict(X)[0]))
        else:
            ci_lo = price * 0.75
            ci_hi = price * 1.25

        ci_width   = max(ci_hi - ci_lo, 1)
        confidence = float(np.clip(1 - ci_width / (2 * price), 0.1, 0.99)) if price > 0 else 0.5

        return {
            "predicted_price": round(max(price, 0), 2),
            "ci_lower":        round(max(ci_lo, 0), 2),
            "ci_upper":        round(max(ci_hi, 0), 2),
            "confidence":      round(confidence, 3),
            "model_used":      self._meta.get("model_name", "RandomForest"),
        }

    def get_info(self) -> dict:
        return {
            "model_name":  self._meta.get("model_name", "RandomForest"),
            "model_type":  self._meta.get("model_type", "RandomForestRegressor"),
            "features":    FEATURES,
            "metrics":     self._meta.get("metrics", {}),
            "trained_on":  self._meta.get("trained_on"),
            "version":     "1.0.0",
        }

    def _encode(self, rows: list[dict]) -> np.ndarray:
        """Label encode и вернуть numpy матрицу в порядке FEATURES."""
        df = pd.DataFrame(rows)[FEATURES]
        for col in CAT_FEATURES:
            le    = self._encoders[col]
            vals  = df[col].astype(str).str.lower().str.strip()
            known = set(le.classes_)
            vals  = vals.apply(lambda v: v if v in known else "unknown")
            df[col] = le.transform(vals)
        df[NUM_FEATURES] = df[NUM_FEATURES].astype(float)
        return df[FEATURES].values

    def _build_encoders(self) -> dict[str, LabelEncoder]:
        encoders = {}
        for col, cats in CAT_MAPS.items():
            le = LabelEncoder()
            le.fit(cats)
            encoders[col] = le
        return encoders

    def _train_demo(self) -> None:
        """
        Обучить три модели (median, q05, q95) на синтетических данных,
        имитирующих реальные закономерности датасета Craigslist.
        Используется как fallback если нет реального датасета.
        """
        logger.info("Генерация синтетических данных для demo режима...")
        rng = np.random.default_rng(42)
        N   = 50_000

        car_age  = rng.integers(0, 40, N)
        odometer = rng.uniform(0, 250_000, N)
        cond_idx = rng.integers(0, 6, N)
        mfr_idx  = rng.integers(0, len(CAT_MAPS["manufacturer"]), N)

        condition_mult = np.array([0.4, 0.65, 0.8, 0.95, 1.1, 1.3])[cond_idx]
        premium_brands = {
            "bmw", "mercedes-benz", "audi", "lexus",
            "volvo", "cadillac", "acura", "infiniti",
        }
        mfr_list = CAT_MAPS["manufacturer"]
        mfr_mult = np.array([1.5 if mfr_list[i] in premium_brands else 1.0 for i in mfr_idx])

        base  = 25_000 * np.exp(-0.07 * car_age) * np.exp(-odometer / 300_000)
        price = np.clip(base * condition_mult * mfr_mult * rng.lognormal(0, 0.15, N), 500, 150_000)

        rows = []
        for i in range(N):
            rows.append({
                "manufacturer": mfr_list[mfr_idx[i]],
                "condition":    CAT_MAPS["condition"][cond_idx[i]],
                "cylinders":    rng.choice(CAT_MAPS["cylinders"]),
                "fuel":         rng.choice(["gas", "gas", "gas", "diesel", "hybrid", "electric"]),
                "odometer":     float(odometer[i]),
                "title_status": rng.choice(["clean", "clean", "clean", "rebuilt", "salvage"]),
                "transmission": rng.choice(["automatic", "automatic", "manual"]),
                "drive":        rng.choice(["fwd", "fwd", "rwd", "4wd"]),
                "type":         rng.choice(CAT_MAPS["type"]),
                "paint_color":  rng.choice(CAT_MAPS["paint_color"]),
                "state":        rng.choice(CAT_MAPS["state"][:48]),
                "car_age":      int(car_age[i]),
            })

        df = pd.DataFrame(rows)
        y  = np.log1p(price)

        self._encoders = self._build_encoders()
        X = self._encode(df.to_dict("records"))

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42)

        logger.info("Обучение основной модели RandomForest...")
        self._model_median = RandomForestRegressor(
            n_estimators=300,
            max_depth=25,
            min_samples_leaf=3,
            max_features=0.8,
            n_jobs=-1,
            random_state=42,
        )
        self._model_median.fit(X_tr, y_tr)

        logger.info("Обучение квантильных моделей q05 и q95...")
        gbr_kw = dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self._model_q05 = GradientBoostingRegressor(loss="quantile", alpha=0.05, **gbr_kw)
        self._model_q95 = GradientBoostingRegressor(loss="quantile", alpha=0.95, **gbr_kw)
        self._model_q05.fit(X_tr, y_tr)
        self._model_q95.fit(X_tr, y_tr)

        y_pred_log = self._model_median.predict(X_te)
        y_pred     = np.expm1(y_pred_log)
        y_true     = np.expm1(y_te)
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = float(r2_score(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

        logger.info("Метрики: MAE=%.0f RMSE=%.0f R2=%.3f MAPE=%.1f%%", mae, rmse, r2, mape)

        self._meta = {
            "model_name":  "RandomForest (demo)",
            "model_type":  "RandomForestRegressor",
            "trained_on":  "synthetic data (demo mode)",
            "metrics": {
                "MAE":  round(mae, 0),
                "RMSE": round(rmse, 0),
                "R2":   round(r2, 4),
                "MAPE": round(mape, 1),
            },
        }

        joblib.dump(
            {
                "model_median": self._model_median,
                "model_q05":    self._model_q05,
                "model_q95":    self._model_q95,
                "encoders":     self._encoders,
            },
            MODEL_PATH,
        )
        with open(META_PATH, "w") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)
        logger.info("Demo модель сохранена в %s", MODEL_PATH)
