"""Film öneri sistemi yapılandırması — ücretsiz açık kaynak modeller + Groq API."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── Yerel modeller (ücretsiz, offline) ──────────────────────────────────────
# BGE-M3: çok dilli retrieval için SOTA seviye ücretsiz model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
# BGE-Reranker-v2-m3: BGE-M3 ile eşleşen en iyi ücretsiz cross-encoder
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# BGE-M3 sorgu öneki (retrieval kalitesi için zorunlu)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Veri yolları ────────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
PREPARED_JSON = DATA_DIR / "films_prepared.json"

# ── İndeks yolları ──────────────────────────────────────────────────────────
INDEX_DIR = BASE_DIR / "index"
FILMS_JSON = INDEX_DIR / "films.json"
EMBEDDINGS_NPY = INDEX_DIR / "embeddings.npy"

# ── Arama parametreleri (performans optimize) ─────────────────────────────────
TOP_K_SEMANTIC = 50
TOP_K_BM25 = 50
TOP_K_RERANK = 15
TOP_K_FINAL = 5
RRF_K = 60

# Reranker için doküman üst sınırı (gereksiz tekrarları önler)
RERANK_DOC_MAX_CHARS = 600

# Gap-detection quality gate'in taban skoru. Bu değer YALNIZCA reranker
# skorları [0,1] aralığında bir olasılık olarak kalibre edildiğinde
# anlamlıdır (bkz. engine.FilmSearchEngine._verify_reranker_calibration,
# sunucu başlarken loglara [OK]/[UYARI] basar). Testlerinize göre 0.10-0.25
# arası ayarlayabilirsiniz.
RERANK_SCORE_FLOOR = 0.20

# Metadata filtresinde (yönetmen/oyuncu/tür/yıl) "tur" alanı SERT bir
# filtredir. Ama otomatik tür etiketleme (prepare_dataset._detect_genres)
# kısa özet metinlerinden çıkarım yaptığı için eksik kalabilir. Sert
# filtre bu yüzden aday havuzunu gereğinden fazla daraltıyorsa, tür kısıtı
# gevşetilip reranker + GENRE_BOOST'a bırakılır.
MIN_RESULTS_FOR_STRICT_GENRE = 3

# Intent bazlı vektör / skor ağırlıkları
BENZER_FILM_USER_WEIGHT = 0.25
BENZER_FILM_REF_WEIGHT = 0.75
DIRECTOR_BOOST = 0.15
ACTOR_BOOST = 0.12
GENRE_BOOST = 0.10

# Parser bağlam penceresi (son N mesaj çifti)
CONTEXT_TURN_LIMIT = 3

# ── LLM (Groq ücretsiz katman) ───────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq | none
# DÜZELTME: Bu yorum eskiden "llama-3.3-70b" diyordu ama alttaki gerçek
# varsayılan değer "openai/gpt-oss-120b" idi — ikisi tutarsızdı ve okuyanı
# yanlış modelin kullanıldığını sanmaya itiyordu. Farklı bir Groq modeli
# denemek isterseniz GROQ_MODEL ortam değişkenini (ör. .env üzerinden)
# ayarlayın; kod tarafında herhangi bir değişiklik gerekmez.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")