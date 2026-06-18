FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Предзагружаем модель эмбеддингов в образ, чтобы контейнер при старте
# не лез в интернет за весами (HF_HUB_OFFLINE=1 в проде)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# models/ должна быть уже собрана локально (см. prepare_index.py) ДО сборки образа
COPY . .

ENV HF_HUB_OFFLINE=1
ENV WEBAPP_HOST=0.0.0.0
ENV WEBAPP_PORT=8080

EXPOSE 8080

CMD ["python", "bot.py"]