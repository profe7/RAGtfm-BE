FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /home/app app

COPY --chown=app:app app ./app
COPY --chown=app:app README.md ./README.md
COPY --chown=app:app alembic.ini ./alembic.ini
COPY --chown=app:app alembic ./alembic

ENV HOME=/home/app
ENV HF_HOME=/home/app/.cache/huggingface
RUN chown -R app:app /app /home/app

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
