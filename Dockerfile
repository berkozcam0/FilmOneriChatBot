FROM python:3.11-slim

# Sistem bağımlılıkları (torch/sentence-transformers derleme ihtiyaçları için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Önce sadece requirements.txt kopyala -> Docker layer cache sayesinde
# kod değişince pip install tekrar tekrar çalışmaz, sadece ilk kurulumda çalışır.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# HF Spaces varsayılan olarak /app altında non-root "user" ile çalışır.
# Model cache'lerinin (sentence-transformers, huggingface_hub) yazılabilir
# bir yere inmesi için HOME ve cache dizinlerini biz belirliyoruz.
ENV HOME=/app \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

# Uygulama kodu + veri + indeks + templates
COPY . .

# HF Spaces container'ları PORT ortam değişkenini 7860 olarak dikte eder.
ENV PORT=7860
EXPOSE 7860

# Yazılabilir cache klasörünü garantiye al (bazı base image'larda /app root'a ait olabilir)
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

CMD ["python", "app.py"]
