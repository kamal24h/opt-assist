# ---------- Single-stage (بهینه‌شده) ----------
FROM python:3.11-slim

# ---------- Environment ----------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

WORKDIR /app

# ---------- System dependencies (runtime only) ----------
# libpq5        : برای psycopg (binary) در runtime
# libmagic1     : برای unstructured (تشخیص نوع فایل)
# poppler-utils : برای pypdf در برخی موارد
# curl          : برای healthcheck
# (build-essential/gcc/g++/python3-dev/libpq-dev حذف شدند چون psycopg[binary] استفاده می‌کنید)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libmagic1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python dependencies ----------
# 1) ابتدا torch نسخه CPU-only را نصب می‌کنیم تا از دانلود پشته CUDA (800MB+) جلوگیری شود.
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# 2) سپس بقیه نیازمندی‌ها
#    - pip upgrade لازم نیست، python:3.11-slim نسخه به‌روز pip را دارد.
#    - cache mount باعث می‌شود .whlهای دانلودشده بین بیلدها حفظ شوند.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# ---------- Application code ----------
# (.dockerignore باید .git, .venv, data/, __pycache__, *.pyc و ... را حذف کند)
COPY . .

# ---------- Runtime dirs & non-root user ----------
RUN mkdir -p /app/temp_uploads /app/data && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# ---------- Port & Healthcheck ----------
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ---------- Run ----------
# زمان‌بر بودن پردازش LLM → timeout 300
CMD ["gunicorn", "main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "300", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]