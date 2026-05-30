FROM python:3.11-slim

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости Python (кешируются отдельным слоем)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Исходный код
COPY src/ ./src/
COPY configs/ ./configs/

# Директория для артефактов (модель)
RUN mkdir -p artifacts/models

# Непривилегированный пользователь
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
