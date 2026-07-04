"""Proje genelinde KULLANILMASI GEREKEN tek normalizasyon fonksiyonu.

Neden gerekli?
--------------
Python'un yerleşik `str.lower()` fonksiyonu Türkçe karakterlerde hatalıdır:
    "İstanbul".lower()  ->  "i̇stanbul"   (i + COMBINING DOT ABOVE, U+0307!)
Bu da BM25 tokenizasyonunu, alias eşleşmesini ve fuzzy aramayı sessizce bozar.

Ayrıca projede iki FARKLI normalize fonksiyonu vardı:
    - engine.py / parser.py  -> sadece boşluk + lower() (aksanları korur)
    - prepare_dataset.py     -> NFKD ile aksanları TAMAMEN siler (ş->s, ı kalır)
Bu tutarsızlık yüzünden prepare_dataset.py'nin ürettiği alias.json anahtarları
("eskıya") ile çalışma zamanında aranan normalize edilmiş sorgu ("eskiya")
ASLA eşleşmiyordu -> kullanıcı en doğal Türkçe yazımıyla film bulamıyordu.

Çözüm: TEK bir fonksiyon, TÜM modüllerde kullanılır.
"""
import re
import unicodedata

_TR_MAP = str.maketrans({
    "İ": "i",
    "I": "ı",
})


def tr_lower(s: str) -> str:
    """Türkçe karakterler için güvenli, tutarlı küçük harfe çevirme."""
    if not s:
        return ""
    # Önce İ/I özel durumlarını elle çöz, sonra normal lower() uygula.
    return str(s).translate(_TR_MAP).lower()


def normalize_tr(s: str, strip_accents: bool = True) -> str:
    """Tüm projede kullanılacak tek normalize fonksiyonu.

    strip_accents=True  -> 'Eşkıya' / 'ESKİYA' / 'eskiya' hepsi 'eskiya' olur
                            (alias/eşleştirme anahtarları için kullanılır)
    strip_accents=False -> sadece boşluk + Türkçe-güvenli lower (görüntü amaçlı)
    """
    if not s:
        return ""
    s = tr_lower(s)
    if strip_accents:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        # NFKD aksanları (ş,ç,ö,ü,ğ) söker ama dotless 'ı' ASCII 'i' olmaz —
        # eşleştirme amaçlı tam ASCII katlama için elle dönüştür.
        s = s.replace("ı", "i")
    return re.sub(r"\s+", " ", s.strip())
