"""
BGE-M3 embedding indeksi oluşturur.

Kullanım:
    python prepare_dataset.py
    python build_index.py
"""
import json

import numpy as np
from sentence_transformers import SentenceTransformer

import config


def build_index(films: list[dict]) -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[MODEL] Embedding yukleniyor: {config.EMBEDDING_MODEL}")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    documents = [f["document"] for f in films]
    print(f"[INDEX] {len(documents)} film embed ediliyor...")
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # JSON serializasyon — geçici alanları temizle
    for f in films:
        f.pop("metin", None)
        f.pop("temiz_metin", None)

    with open(config.FILMS_JSON, "w", encoding="utf-8") as fp:
        json.dump(films, fp, ensure_ascii=False, indent=2)

    np.save(config.EMBEDDINGS_NPY, embeddings.astype(np.float32))

    print(f"[OK] Indeks hazir: {len(films)} film")
    print(f"   -> {config.FILMS_JSON}")
    print(f"   -> {config.EMBEDDINGS_NPY}  shape={embeddings.shape}")


def _load_from_prepared() -> list[dict]:
    path = config.PREPARED_JSON
    print(f"[JSON] Hazir veri okunuyor: {path}")
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def main() -> None:
    # Eğer önceden hazırlanmış temiz veri yoksa ama kaynaklar varsa önce prepare et!
    if not config.PREPARED_JSON.exists():
        print("[UYARI] Hazır veri seti bulunamadı. prepare_dataset çalıştırılıyor...")
        from prepare_dataset import prepare
        prepare()

    # Artık güvenle PREPARED_JSON üzerinden yükleme yapabiliriz
    films = _load_from_prepared()
    build_index(films)


if __name__ == "__main__":
    main()