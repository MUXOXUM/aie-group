"""
Pydantic-схемы для входных/выходных данных API.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


MANUFACTURERS = [
    "ford", "chevrolet", "toyota", "honda", "dodge", "nissan", "jeep", "gmc",
    "ram", "volkswagen", "bmw", "mercedes-benz", "audi", "subaru", "hyundai",
    "kia", "lexus", "mazda", "volvo", "cadillac", "buick", "lincoln", "acura",
    "infiniti", "chrysler", "pontiac", "saturn", "mitsubishi", "unknown",
]


class CarFeatures(BaseModel):
    """Характеристики автомобиля для оценки стоимости."""

    manufacturer: str = Field(
        default="unknown",
        description="Производитель (ford, toyota, bmw, ...)",
        examples=["toyota"],
    )
    condition: Literal["new", "like new", "excellent", "good", "fair", "salvage", "unknown"] = Field(
        default="good",
        description="Техническое состояние",
    )
    cylinders: Literal[
        "3 cylinders", "4 cylinders", "5 cylinders", "6 cylinders",
        "8 cylinders", "10 cylinders", "12 cylinders", "other", "unknown"
    ] = Field(default="4 cylinders", description="Количество цилиндров")
    fuel: Literal["gas", "diesel", "hybrid", "electric", "other", "unknown"] = Field(
        default="gas", description="Тип топлива"
    )
    odometer: float = Field(
        ..., ge=0, le=500_000,
        description="Пробег в милях",
        examples=[45000],
    )
    title_status: Literal[
        "clean", "rebuilt", "salvage", "lien", "missing", "parts only", "unknown"
    ] = Field(default="clean", description="Статус документов")
    transmission: Literal["automatic", "manual", "other", "unknown"] = Field(
        default="automatic", description="Тип трансмиссии"
    )
    drive: Literal["fwd", "rwd", "4wd", "unknown"] = Field(
        default="fwd", description="Привод"
    )
    type: Literal[
        "sedan", "SUV", "pickup", "truck", "coupe", "hatchback", "wagon",
        "van", "convertible", "mini-van", "offroad", "bus", "other", "unknown"
    ] = Field(default="sedan", description="Тип кузова")
    paint_color: Literal[
        "white", "black", "silver", "blue", "red", "grey", "green",
        "brown", "yellow", "orange", "purple", "custom", "unknown"
    ] = Field(default="unknown", description="Цвет кузова")
    state: str = Field(
        default="ca",
        min_length=2, max_length=2,
        description="Штат США (двухбуквенный код: ca, tx, ny, ...)",
        examples=["ca"],
    )
    car_age: int = Field(
        ..., ge=0, le=50,
        description="Возраст автомобиля в годах",
        examples=[5],
    )

    @field_validator("manufacturer")
    @classmethod
    def normalize_manufacturer(cls, v: str) -> str:
        return v.lower().strip() if v else "unknown"

    @field_validator("state")
    @classmethod
    def normalize_state(cls, v: str) -> str:
        return v.lower().strip()

    def to_dict(self) -> dict:
        return self.model_dump()


class PredictionResponse(BaseModel):
    """Результат предсказания цены."""
    predicted_price: float = Field(description="Точечная оценка цены, USD")
    ci_lower: float = Field(description="Нижняя граница 90% доверительного интервала, USD")
    ci_upper: float = Field(description="Верхняя граница 90% доверительного интервала, USD")
    confidence: float = Field(description="Уверенность модели от 0 до 1")
    model_used: str = Field(description="Название модели, давшей предсказание")
    latency_ms: Optional[float] = Field(default=None, description="Время предсказания, мс")


class BatchPredictionRequest(BaseModel):
    """Пакетный запрос предсказаний."""
    cars: list[CarFeatures] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """Результаты пакетного предсказания."""
    predictions: list[PredictionResponse]
    count: int
    total_latency_ms: float


class ModelInfoResponse(BaseModel):
    """Метаданные и метрики модели."""
    model_name: str
    model_type: str
    features: list[str]
    metrics: dict
    trained_on: Optional[str] = None
    version: str = "1.0.0"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
