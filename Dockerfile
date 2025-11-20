# Dockerfile para Credit Risk Scoring API
#
# Proyecto: Credit Risk Scoring - UCI Taiwan Dataset
# Fase DVP-PRO: F8 - Productización
# Autor: Ing. Daniel Varela Pérez
# Email: bedaniele0@gmail.com
# Fecha: 2025-11-18

# Imagen base
FROM python:3.11-slim

# Metadata
LABEL maintainer="Ing. Daniel Varela Pérez <bedaniele0@gmail.com>"
LABEL version="1.0.0"
LABEL description="Credit Risk Scoring API - LightGBM Model"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Crear directorio de trabajo
WORKDIR ${APP_HOME}

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para seguridad
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copiar requirements primero (mejor cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/
COPY models/ ./models/
COPY reports/ ./reports/
COPY data/processed/ ./data/processed/

# Crear directorios necesarios
RUN mkdir -p reports/monitoring logs

# Cambiar propietario de archivos
RUN chown -R appuser:appgroup ${APP_HOME}

# Cambiar a usuario no-root
USER appuser

# Puerto de la API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando para iniciar la API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
