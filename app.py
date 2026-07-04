from collections import OrderedDict, deque
from flask import Flask, render_template, request, jsonify, session
import secrets
import time
import uuid
import os

app = Flask(__name__)

# GÜVENLİK: Eskiden burada sabit kodlanmış bir secret_key vardı
# ("film-oneri-secure-key-2026-prod"). Bu, GitHub'a push edilen her
# kopyada AYNI anahtar olduğu için session cookie'lerinin dışarıdan
# sahtelenebilmesi (forge) anlamına geliyordu. FLASK_SECRET env
# değişkeni tanımlı değilse artık sunucu her başlangıçta rastgele,
# tahmin edilemez bir anahtar üretiyor (tek dezavantajı: sunucu
# yeniden başladığında eski oturumlar geçersiz olur — bu, sabit ve
# herkesçe bilinen bir anahtar kullanmaktan çok daha güvenlidir).
_env_secret = os.getenv("FLASK_SECRET")
if not _env_secret:
    print("[UYARI] FLASK_SECRET ortam değişkeni tanımlı değil! "
          "Rastgele bir anahtar üretiliyor (sunucu yeniden başlayınca oturumlar sıfırlanır). "
          "Prod ortamda FLASK_SECRET'i mutlaka .env üzerinden sabitleyin.")
app.secret_key = _env_secret or secrets.token_hex(32)

# CHATBOT'U GLOBAL OLARAK SUNUCU AÇILIRKEN BİR KERE YÜKLÜYORUZ
print("[SISTEM] Yapay zeka modelleri hafizaya yukleniyor, lutfen bekleyin...")
from chatbot import FilmChatbot

# NOT: global_bot artık request içinde DOĞRUDAN kullanılmıyor (aşağıya bkz).
# Sadece ağır ML modellerini (embedding, reranker, BM25 index) barındıran
# `engine`'i bir kere yükleyip tüm kullanıcılar arasında paylaşmak için var.
global_bot = FilmChatbot()
print("[SISTEM] Modeller basariyla yuklendi! Sunucu hazir.")

# Çoklu kullanıcı sohbet geçmişlerini session_id bazlı ayırmak için sözlük.
# OrderedDict + MAX_SESSIONS sınırı: sunucu uzun süre açık kaldığında her
# yeni ziyaretçinin UUID'si bu sözlükte sonsuza kadar birikip belleği
# taşırmasın diye en eski oturumlar otomatik olarak atılıyor.
USER_HISTORIES: "OrderedDict[str, list]" = OrderedDict()
MAX_SESSIONS = 5000

# GÜVENLİK: Eskiden /chat endpoint'inde HİÇBİR rate limit yoktu. Bu, tek
# bir istemcinin art arda istek atarak hem Groq API kotasını (ücretsiz
# katman) tüketebilmesi hem de embedder/reranker modellerini (CPU'da
# pahalı) tıkayabilmesi anlamına geliyordu — basit bir DoS/maliyet
# saldırısı vektörü. IP başına sabit pencereli (fixed-window) basit bir
# limiter ekliyoruz; harici bir bağımlılık (Flask-Limiter vb.) gerektirmez,
# USER_HISTORIES ile aynı OrderedDict+LRU deseni izler.
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_MAX_TRACKED_KEYS = 5000
MAX_MESSAGE_CHARS = 1000

_rate_limit_buckets: "OrderedDict[str, deque]" = OrderedDict()

# GÜVENLİK DÜZELTMESİ: Eskiden burada X-Forwarded-For HER ZAMAN okunuyordu
# ve bir yorum satırı "aksi halde session id ile kombine ediliyor" diyordu —
# ama böyle bir kombinasyon KODDA HİÇ YOKTU. Gerçek davranış: bu uygulama
# güvenilir bir reverse proxy (nginx, ProxyFix vb.) arkasında çalışmıyorsa
# (ki bu repo'da böyle bir proxy konfigürasyonu YOK, app.run(host="0.0.0.0")
# ile doğrudan expose ediliyor), X-Forwarded-For istemcinin gönderdiği
# HERHANGİ bir header'dır ve sunucu tarafında hiçbir doğrulaması yoktur.
# Yani bir istemci her istekte farklı bir X-Forwarded-For değeri göndererek
# rate limit'i (ve dolayısıyla Groq ücretsiz kota / CPU'daki embedder+
# reranker yükünü) TAMAMEN bypass edebiliyordu.
#
# (Session id ile kombine etmek de gerçek bir çözüm DEĞİLDİR: session
# cookie'si tarayıcı tarafında isteğe bağlıdır, bir script cookie
# göndermeden her istekte "yeni" bir session'a düşüp aynı şekilde limiti
# resetleyebilir. Kimlik doğrulamasız bir ortamda rate-limit anahtarı için
# en sağlam demirleme noktası hâlâ gerçek TCP bağlantısının geldiği IP'dir.)
#
# Varsayılan olarak artık İSTEMCİNİN AYARLAYAMAYACAĞI request.remote_addr
# kullanılıyor (güvenli varsayılan). Uygulamayı gerçekten XFF'i temizleyen/
# üzerine yazan bir reverse proxy arkasında çalıştırıyorsanız, ortam
# değişkeni ile bunu açıkça etkinleştirin: TRUST_PROXY_HEADERS=true
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").strip().lower() in ("1", "true", "yes")
if TRUST_PROXY_HEADERS:
    print("[UYARI] TRUST_PROXY_HEADERS=true — X-Forwarded-For güvenilir kabul "
          "ediliyor. Bu SADECE bu başlığı istemciden gelen değerle değil, "
          "kendi değeriyle üzerine yazan bir reverse proxy (nginx/ProxyFix) "
          "arkasındaysanız güvenlidir. Emin değilseniz bu değişkeni "
          "tanımlamayın.")


def _client_key() -> str:
    if TRUST_PROXY_HEADERS:
        ip = (request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
              .split(",")[0].strip())
    else:
        # Doğrudan TCP bağlantısının IP'si — istemci tarafından sahtelenemez.
        ip = request.remote_addr or "unknown"
    return ip


def _is_rate_limited(key: str) -> bool:
    now = time.time()
    bucket = _rate_limit_buckets.get(key)
    if bucket is None:
        bucket = deque()
        _rate_limit_buckets[key] = bucket
    _rate_limit_buckets.move_to_end(key)

    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SECONDS:
        bucket.popleft()

    if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
        return True

    bucket.append(now)
    if len(_rate_limit_buckets) > RATE_LIMIT_MAX_TRACKED_KEYS:
        _rate_limit_buckets.popitem(last=False)
    return False


@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    if _is_rate_limited(_client_key()):
        return jsonify({
            "error": "Çok fazla istek gönderdiniz. Lütfen biraz bekleyip tekrar deneyin."
        }), 429

    # DÜZELTME: `force=True` content-type kontrolünü atlıyordu ama gövde
    # tamamen bozuksa (geçersiz JSON) yine de exception fırlatabiliyordu;
    # bu durumda hata `try` bloğunun DIŞINDA olduğu için Flask'ın varsayılan
    # 500 sayfasına düşüp stack trace'i sızdırma riski taşıyordu.
    # `silent=True` + `or {}` ile hatasız şekilde boş sözlüğe düşüyoruz.
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Boş mesaj"}), 400

    # GÜVENLİK: Mesaj uzunluğu sınırlanmazsa embedder/reranker/LLM'e
    # (maliyetli) aşırı büyük girdi gönderilebilir. Makul bir üst sınır.
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify({
            "error": f"Mesaj çok uzun (maksimum {MAX_MESSAGE_CHARS} karakter)."
        }), 400

    # Kullanıcının benzersiz oturum kimliğini alıyoruz
    user_sid = session.get("sid")
    if not user_sid:
        user_sid = str(uuid.uuid4())
        session["sid"] = user_sid

    try:
        # ÖNEMLİ: Eskiden burada tüm kullanıcılar için TEK bir paylaşılan
        # `global_bot` nesnesinin `.history` alanı okunup yazılıyordu. Flask
        # threaded modda (veya birden fazla worker/thread ile) çalıştığında
        # bu, iki kullanıcının isteği aynı anda işlenirse birinin sohbet
        # geçmişinin diğerine karışmasına (race condition) yol açabilirdi.
        # Çözüm: ağır ML modelleri (`engine`) tek sefer yüklenip paylaşılmaya
        # devam ediyor, ama her istek için ayrı, ucuz bir FilmChatbot nesnesi
        # oluşturup kendi geçmişini kendi izole ediyoruz.
        request_bot = FilmChatbot(engine=global_bot.engine)
        request_bot.history = USER_HISTORIES.get(user_sid, [])

        # Yanıtı üret (history sınırlama işlemi zaten process() içinde yapılıyor)
        response = request_bot.process(message)

        USER_HISTORIES[user_sid] = request_bot.history

        # Bellek taşmasını önlemek için oturum sayısını sınırlı tut (LRU mantığı)
        USER_HISTORIES.move_to_end(user_sid)
        if len(USER_HISTORIES) > MAX_SESSIONS:
            USER_HISTORIES.popitem(last=False)

        return jsonify({"response": response})

    except Exception as e:
        print(f"[HATA] Chat esnasinda hata olustu: {str(e)}")
        return jsonify({"error": "İstek işlenirken bir hata oluştu."}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)