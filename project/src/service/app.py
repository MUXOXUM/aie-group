"""
FastAPI сервис оценки рыночной стоимости подержанного автомобиля.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.service.schemas import (
    CarFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    HealthResponse,
)
from src.models.predictor import CarPricePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

predictor: CarPricePredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Загрузка модели предсказания цены автомобиля...")
    predictor = CarPricePredictor()
    predictor.load()
    logger.info("Модель загружена и готова к работе")
    yield
    logger.info("Завершение работы сервиса...")


app = FastAPI(
    title="Used Car Price Estimator API",
    description=(
        "REST API для оценки рыночной стоимости подержанного автомобиля. "
        "Принимает характеристики автомобиля, возвращает прогнозную цену "
        "с доверительным интервалом."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Проверка работоспособности сервиса."""
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None and predictor.is_ready(),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info() -> ModelInfoResponse:
    """Информация об используемой модели и её метриках."""
    _check_ready()
    return predictor.get_info()


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(car: CarFeatures) -> PredictionResponse:
    """
    Предсказание цены одного автомобиля.

    Возвращает:
    - **predicted_price** — точечная оценка цены в USD
    - **ci_lower / ci_upper** — 90% доверительный интервал
    - **confidence** — уверенность модели (0–1)
    - **model_used** — название алгоритма
    """
    _check_ready()
    t0 = time.perf_counter()
    try:
        result = predictor.predict(car.to_dict())
    except Exception as exc:
        logger.exception("Ошибка предсказания: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Пакетное предсказание цен для списка автомобилей (до 100 штук).
    """
    _check_ready()
    if len(request.cars) > 100:
        raise HTTPException(status_code=400, detail="Максимум 100 автомобилей за запрос")
    t0 = time.perf_counter()
    predictions = [predictor.predict(car.to_dict()) for car in request.cars]
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**p) for p in predictions],
        count=len(predictions),
        total_latency_ms=elapsed,
    )


def _check_ready():
    if predictor is None or not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Модель ещё не загружена")


if __name__ == "__main__":
    uvicorn.run("src.service.app:app", host="0.0.0.0", port=8000, reload=False)
