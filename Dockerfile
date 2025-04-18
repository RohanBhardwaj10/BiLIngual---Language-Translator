FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model to avoid cold start
RUN python -c "\
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration; \
M2M100Tokenizer.from_pretrained('facebook/m2m100_418M'); \
M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')"

EXPOSE 7860

CMD ["python", "app.py"]
