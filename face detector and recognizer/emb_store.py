import json
import numpy as np
import os

def load_embeddings(path):
    """Загружает базу эмбеддингов из JSON и преобразует значения в numpy-массивы"""
    try:
        with open(path, 'r') as f:
            raw = json.load(f)
        return {k: np.array(v) for k, v in raw.items()}
    except FileNotFoundError:
        return {}

def save_embeddings(path, db):
    """Сохраняет базу эмбеддингов в JSON — значения должны быть списками"""
    with open(path, 'w') as f:
        json.dump(db, f, indent=2)

def find_match(embedding, db, threshold=0.8):
    """
    Ищет ближайшее совпадение эмбеддинга в базе.
    Возвращает имя, если расстояние меньше порога, иначе 'Unknown'.
    """
    min_dist = float('inf')
    best_match = "Unknown"
    for name, db_emb in db.items():
        dist = np.linalg.norm(embedding - db_emb)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            best_match = name
    return best_match


def create_empty_embeddings_json(path='embeddings.json'):
    """Создаёт пустой JSON-файл для хранения эмбеддингов, если он не существует."""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump({}, f, indent=2)
        print(f"Создан пустой файл {path}")
    else:
        print(f"Файл {path} уже существует")