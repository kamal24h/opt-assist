# ---------- Stage 1: Builder ----------
FROM python:3.11-slim AS builder

# جلوگیری از نوشتن فایل pyc و فعال‌سازی خروجی بدون بافر
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

WORKDIR /app

# نصب وابستگی‌های سیستمی لازم برای build (کامپایل برخی بسته‌های پایتون)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 1. ابتدا فقط requirements.txt را کپی کن
# (این لایه تا زمانی که requirements.txt تغییر نکند، کش می‌شود)
COPY requirements.txt .

# 2. نصب نیازمندی‌های پایتون از mirror چینی با cache mount
# - --index-url صریح: حتی اگر ENV override شود، این لایه از mirror استفاده می‌کند
# - --trusted-host: برای mirrorهای HTTP/self-signed لازم است
# - cache mount: فایل‌های .whl دانلود شده در /root/.cache/pip نگه داشته می‌شوند
#   و در buildهای بعدی (حتی اگر requirements.txt تغییر کند) دوباره استفاده می‌شوند.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        --trusted-host pypi.tuna.tsinghua.edu.cn \
        -r requirements.txt

# 3. حالا بقیه کد پروژه را کپی کن
# (تغییر در کد فقط لایه‌های بعد از این خط را باطل می‌کند، نه نصب نیازمندی‌ها)
COPY . .

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# نصب وابستگی‌های سیستمی runtime (بدون build-essential)
# - libpq5: برای psycopg
# - libmagic1: برای unstructured (تشخیص نوع فایل)
# - poppler-utils: برای pypdf در برخی موارد
# - curl: برای health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libmagic1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# کپی بسته‌های نصب‌شده از stage builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# کپی کد پروژه
COPY . .

# پوشه temp برای آپلود فایل‌ها
RUN mkdir -p /app/temp_uploads /app/data

# کاربر غیر root برای امنیت
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# پورت
EXPOSE 8000

# health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# اجرای gunicorn با uvicorn worker
CMD ["gunicorn", "main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "300", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]