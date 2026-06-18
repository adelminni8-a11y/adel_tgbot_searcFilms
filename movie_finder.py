"""
Модуль поиска фильмов по описанию сюжета.
Используется ботом во время работы — только читает уже готовые
индекс и переводы из models/, никакого онлайн-перевода тут нет.
"""

import os

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

os.environ.setdefault("HF_HUB_OFFLINE", "1")


class MovieFinder:
    def __init__(self, model_dir: str = "models"):
        print("🚀 Загрузка модели sentence-transformers...")
        self.model = SentenceTransformer("intfloat/multilingual-e5-small")
        print("✅ Модель загружена")

        print("📂 Загрузка индекса и данных...")
        self.index = faiss.read_index(os.path.join(model_dir, "movie_index.faiss"))
        self.df = pd.read_csv(os.path.join(model_dir, "movies_data.csv"), keep_default_na=False)

        self.df["overview_ru"] = self.df["overview_ru"].replace("", np.nan).fillna(self.df["overview"])
        self.df["genres_ru"] = self.df["genres_ru"].replace("", np.nan).fillna(self.df["genres"])

        print(f"✅ Готово: {self.index.ntotal} фильмов в индексе")

    def search(self, query: str, top_k: int = 5):
        query_vec = self.model.encode(["query: " + query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            row = self.df.iloc[idx]
            results.append({
                "title": row["title"],
                "overview_ru": row["overview_ru"],
                "genres_ru": row["genres_ru"],
                "score": float(score),
            })
        return results