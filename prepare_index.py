import json
import os
import time

import faiss
import numpy as np
import pandas as pd
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

GENRE_TRANSLATIONS = {
    "Action": "Боевик",
    "Adventure": "Приключения",
    "Animation": "Анимация",
    "Comedy": "Комедия",
    "Crime": "Криминал",
    "Documentary": "Документальный",
    "Drama": "Драма",
    "Family": "Семейный",
    "Fantasy": "Фэнтези",
    "History": "Исторический",
    "Horror": "Ужасы",
    "Music": "Музыкальный",
    "Mystery": "Детектив",
    "Romance": "Мелодрама",
    "Science Fiction": "Фантастика",
    "TV Movie": "Телефильм",
    "Thriller": "Триллер",
    "War": "Военный",
    "Western": "Вестерн",
    "Foreign": "Зарубежный",
}


def parse_genres(raw):
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        try:
            import ast
            data = ast.literal_eval(raw)
        except Exception:
            return []
    return [g.get("name", "") for g in data if isinstance(g, dict)]


def translate_genres(names):
    return [GENRE_TRANSLATIONS.get(n, n) for n in names]


def translate_text(text, translator, retries=3, base_delay=0.4):
    text = str(text).strip()
    if not text:
        return ""
    for attempt in range(retries):
        try:
            result = translator.translate(text)
            time.sleep(base_delay)
            return result
        except Exception as e:
            print(f"⚠️  Ошибка перевода (попытка {attempt + 1}/{retries}): {e}")
            time.sleep(2 * (attempt + 1))
    print("⚠️  Не удалось перевести, оставляю оригинал на английском")
    return text


class IndexBuilder:
    def __init__(self):
        print("🚀 Загрузка модели sentence-transformers...")
        self.model = SentenceTransformer("intfloat/multilingual-e5-small")
        print("✅ Модель загружена")
        self.translator = GoogleTranslator(source="en", target="ru")

    def build(self, csv_path="data/tmdb_5000_movies.csv", save_path="models", checkpoint_every=200):
        os.makedirs(save_path, exist_ok=True)
        checkpoint_path = os.path.join(save_path, "movies_data.csv")

        if os.path.exists(checkpoint_path):
            print("📂 Найден чекпоинт — продолжаем перевод с того места, где остановились...")
            df = pd.read_csv(checkpoint_path, keep_default_na=False)
        else:
            print(f"📀 Загрузка датасета из {csv_path}...")
            raw = pd.read_csv(csv_path)
            raw = raw[["title", "overview", "genres"]].copy()

            before = len(raw)
            raw = raw.dropna(subset=["overview"]).reset_index(drop=True)
            print(f"✅ Загружено {len(raw)} фильмов (удалено {before - len(raw)} без описания)")

            genres_list = raw["genres"].apply(parse_genres)
            raw["genres_ru"] = genres_list.apply(lambda gs: ", ".join(translate_genres(gs)))
            raw["genres"] = genres_list.apply(lambda gs: ", ".join(gs))
            raw["overview_ru"] = ""
            df = raw

        todo = df.index[df["overview_ru"] == ""].tolist()
        print(f"🌍 Нужно перевести {len(todo)} описаний из {len(df)}...")

        for n, i in enumerate(todo, start=1):
            df.at[i, "overview_ru"] = translate_text(df.at[i, "overview"], self.translator)
            if n % checkpoint_every == 0:
                df.to_csv(checkpoint_path, index=False)
                print(f"💾 Чекпоинт сохранён: {n}/{len(todo)}")

        df.to_csv(checkpoint_path, index=False)
        print("✅ Перевод завершён")

        print("🔨 Создаём эмбеддинги...")
        passages = ["passage: " + str(t) for t in df["overview"].tolist()]
        embeddings = self.model.encode(
            passages,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
        ).astype("float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, os.path.join(save_path, "movie_index.faiss"))
        np.save(os.path.join(save_path, "embeddings.npy"), embeddings)
        df.to_csv(checkpoint_path, index=False)

        print(f"💾 Готово! Индекс ({index.ntotal} фильмов) и данные сохранены в {save_path}/")

if __name__ == "__main__":
    IndexBuilder().build()