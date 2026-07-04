"""
Veri seti hazırlama: temizleme, birleştirme, tür/özet çıkarımı, çeşitlendirme.

Kullanım:
    python prepare_dataset.py
    python build_index.py data/films_prepared.json
"""
import json
import re
from collections import defaultdict

import config
from document_builder import build_film_document
from film_constants import GENRE_KEYWORDS
from text_utils import normalize_tr, tr_lower

DATA_DIR = config.BASE_DIR / "data"
SUPPLEMENT_PATH = DATA_DIR / "supplement_films.json"
OUTPUT_PATH = DATA_DIR / "films_prepared.json"
# ESKİ KOD: ALIASES_PATH silindi. Artık sözlük dosyasına ihtiyacımız yok.

# Bilinen TR/EN eşleşmeleri — veritabanını ilk oluştururken çift kayıtları birleştirmek için KORUNDU
KNOWN_PAIRS = {
    ("m - bir şehir katili arıyor", "m"),
    ("arka pencere", "rear window"),
    ("şimşekler altında", "on the waterfront"),
    ("kum sığdırmaları", "blow-up"),
    ("bir uçtu guguk kuşu", "one flew over the cuckoo's nest"),
    ("şimdiden sonra", "schindler's list"),
    ("truman show", "the truman show"),
    ("başlangıç", "inception"),
    ("sessizlik", "the silence of the lambs"),
    ("sessizlik", "silence of the lambs"),
    ("yeşil yol", "the green mile"),
}

THEME_TAGS = [
    "türk sineması", "nolan", "hitchcock", "kurosawa", "tarkovsky", "gotik",
    "distopya", "psikolojik", "noir", "epik", "minimalist", "nostalji",
    "bilim kurgu", "korku", "gerilim", "suç", "romantik", "komedi",
    "animasyon", "miyazaki", "ceylan", "fincher", "villeneuve",
    "savaş", "biyografi", "western", "müzikal", "fantastik",
]


def _extract_tags(text: str, tur: str) -> list[str]:
    tags: list[str] = []
    if tur:
        tags.extend(t.strip() for t in tur.split(","))
    lower = tr_lower(text)
    for kw in THEME_TAGS:
        if kw in lower:
            tags.append(kw)
    return list(dict.fromkeys(tags))[:10]


def _normalize(s: str) -> str:
    return normalize_tr(s)


def _has_turkish_chars(s: str) -> bool:
    return bool(re.search(r"[çğıöşüÇĞİÖŞÜ]", s))


def _normalize_director(name: str) -> str:
    return _normalize(name).replace(".", "").replace(",", " ")


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 15]


def _dedupe_sentences(sentences: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for s in sentences:
        norm = _normalize(s)
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(s)
    return unique


def _extract_body_from_document(doc: str) -> str:
    lines = doc.split("\n")
    body_parts = []
    for line in lines:
        if line.startswith("Özet:"):
            body_parts.append(line[5:].strip())
        elif re.match(r"^(Film|Yönetmen|Oyuncular|Tür|Etiketler):", line):
            continue
        else:
            body_parts.append(line)
    return " ".join(body_parts)


def _clean_text(text: str) -> tuple[str, str]:
    """Özet ve temiz gövde metni çıkar."""
    body = _extract_body_from_document(text) if "Film:" in text else text
    sentences = _dedupe_sentences(_split_sentences(body))

    while sentences:
        last = sentences[-1]
        if len(last.split()) > 8 and not last.rstrip().endswith((".", "!", "?")):
            sentences.pop()
        else:
            break

    ozet = sentences[0] if sentences else body
    ozet = re.sub(r"\s+", " ", ozet).strip()
    if len(ozet) > 280:
        ozet = ozet[:280].rsplit(" ", 1)[0] + "..."

    clean_body = " ".join(sentences[:3])
    if len(clean_body) > 500:
        clean_body = clean_body[:500].rsplit(" ", 1)[0] + "..."
    return ozet, clean_body


def _detect_genres(text: str, existing: str = "") -> str:
    if existing:
        return existing
    lower = tr_lower(text)
    scores: dict[str, int] = {}
    for genre, keywords in GENRE_KEYWORDS.items():
        score = sum(2 if kw in lower else 0 for kw in keywords)
        if score:
            scores[genre] = score
    if not scores:
        return "dram"
    # MİNİMUM EŞİK: Eskiden en yüksek skorlu 3 tür, skor ne kadar zayıf
    # olursa olsun (tek bir kelimenin bir kez geçmesi bile) otomatik
    # ekleniyordu. Bu, "büyü"/"efsane" gibi belirsiz kelimelerin tek
    # başına bir filmi yanlış türe sokmasına yol açan asıl mekanizmaydı.
    # Artık bir türün eklenmesi için en az 2 farklı anahtar kelimenin
    # (ya da aynı kelimenin en az iki kez) eşleşmesi gerekiyor.
    MIN_GENRE_SCORE = 4
    top = sorted(
        ((g, s) for g, s in scores.items() if s >= MIN_GENRE_SCORE),
        key=lambda x: -x[1],
    )[:3]
    if not top:
        # Hiçbir tür güçlü sinyal vermiyorsa, en azından tek eşleşen
        # kelimeyle de olsa en yüksek skorlu türü kullan (boş bırakmaktansa).
        top = sorted(scores.items(), key=lambda x: -x[1])[:1]
    return ", ".join(g for g, _ in top)


def _choose_primary_title(titles: list[str]) -> str:
    """Türkçe karakterli veya daha tanımlayıcı başlığı seç."""
    if not titles:
        return ""
    turkish = [t for t in titles if _has_turkish_chars(t)]
    if turkish:
        return max(turkish, key=len)
    return max(titles, key=len)


def _merge_group(group: list[dict]) -> dict:
    titles = [f["film_adi"] for f in group]
    primary_title = _choose_primary_title(titles)

    best = max(group, key=lambda f: len(f.get("temiz_metin", "") or f.get("ozet", "")))

    # DÜZELTME: Birleşen filmlerin "alternatif_adlar" listeleri kayboluyordu.
    # Gruptaki tüm filmlerin alias'larını topla; ayrıca birleşmeden önceki
    # diğer başlıkları da (primary_title dışında kalanları) alias olarak
    # ekle, çünkü onlar da artık aynı filmin "farklı adları".
    merged_aliases: list[str] = []
    seen_alias = set()
    for f in group:
        for alt in f.get("alternatif_adlar") or []:
            if alt.strip().lower() not in seen_alias:
                seen_alias.add(alt.strip().lower())
                merged_aliases.append(alt)
        title = f.get("film_adi", "")
        if title and title != primary_title and title.strip().lower() not in seen_alias:
            seen_alias.add(title.strip().lower())
            merged_aliases.append(title)

    merged = {
        "film_adi": primary_title,
        "yonetmen": best.get("yonetmen", ""),
        "oyuncular": best.get("oyuncular", ""),
        "yil": best.get("yil"),
        "tur": best.get("tur") or _detect_genres(best.get("temiz_metin", "") + best.get("ozet", "")),
        "ozet": best.get("ozet", ""),
        "etiketler": best.get("etiketler", []),
        "alternatif_adlar": merged_aliases,
        "temiz_metin": best.get("temiz_metin", ""),
    }
    merged["document"] = build_film_document(merged)
    return merged


def _bad_tags(tags: list) -> bool:
    if not tags:
        return True
    if tags[0].isdigit():
        return True
    short = sum(1 for t in tags if len(t) < 5)
    if short >= max(1, len(tags) * 0.5):
        return True
    return False


def _is_structured_entry(raw: dict) -> bool:
    return bool(raw.get("ozet")) and not _bad_tags(raw.get("etiketler") or [])


def _process_film(raw: dict) -> dict:
    etiketler = raw.get("etiketler", []) or []

    if _is_structured_entry(raw):
        ozet = raw["ozet"]
        temiz = raw.get("temiz_metin", "")
    elif raw.get("document"):
        ozet, temiz = _clean_text(raw["document"])
    else:
        ozet = raw.get("ozet", "")
        temiz = raw.get("temiz_metin", "")

    tur = raw.get("tur") or _detect_genres(f"{ozet} {temiz}")
    if not etiketler or _bad_tags(etiketler):
        etiketler = _extract_tags(f"{ozet} {temiz}", tur)

    film = {
        "film_adi": raw["film_adi"].strip(),
        "yonetmen": tr_lower((raw.get("yonetmen") or "").strip()),
        "oyuncular": tr_lower((raw.get("oyuncular") or "").strip()),
        "yil": raw.get("yil"),
        "tur": tur,
        "ozet": ozet,
        "etiketler": etiketler,
        # DÜZELTME: "alternatif_adlar" burada taşınmıyordu, bu yüzden
        # her prepare_dataset.py çalıştırmasında engine.py'nin alias
        # eşleştirmesi için kullandığı bu alan sessizce siliniyordu.
        "alternatif_adlar": raw.get("alternatif_adlar") or [],
        "temiz_metin": temiz,
    }
    film["document"] = build_film_document(film)
    return film


def _load_source_films() -> list[dict]:
    with open(config.FILMS_JSON, encoding="utf-8") as fp:
        return json.load(fp)


def _load_supplement() -> list[dict]:
    if not SUPPLEMENT_PATH.exists():
        return []
    with open(SUPPLEMENT_PATH, encoding="utf-8") as fp:
        return json.load(fp)


def _merge_duplicates(films: list[dict]) -> list[dict]:
    # NOT: (yönetmen, yıl) ikisi de boş/None olduğunda eskiden TÜM bu
    # filmler aynı gruba ("" , None) düşüyor ve _merge_group() bunları
    # TEK bir filme indirip gerisini sessizce siliyordu. Şu anki veri
    # setinde yönetmen/yıl her zaman dolu olduğu için bu tetiklenmiyor,
    # ama yeni/eksik veri eklendiğinde (ör. generate_synthetic_films.py
    # ile) ciddi veri kaybına yol açabilir. Bu yüzden ikisi de boşsa o
    # filmi kendi tekil grubuna koyuyoruz.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for i, f in enumerate(films):
        director = _normalize_director(f.get("yonetmen", ""))
        yil = f.get("yil")
        if director and yil:
            key = (director, yil)
        else:
            key = ("__no_match__", i)
        groups[key].append(f)

    merged: list[dict] = []
    for key, group in groups.items():
        if len(group) == 1:
            merged.append(group[0])
        else:
            merged.append(_merge_group(group))

    by_norm_name: dict[str, dict] = {}
    final: list[dict] = []
    for f in merged:
        norm = _normalize(f["film_adi"])
        paired = None
        for a, b in KNOWN_PAIRS:
            if norm == a and b in by_norm_name:
                paired = b
            elif norm == b and a in by_norm_name:
                paired = a
        if paired and paired in by_norm_name:
            combined = _merge_group([by_norm_name[paired], f])
            del by_norm_name[paired]
            final = [x for x in final if _normalize(x["film_adi"]) != paired]
            by_norm_name[_normalize(combined["film_adi"])] = combined
            final.append(combined)
        else:
            by_norm_name[norm] = f
            final.append(f)

    return final


def _dedupe_by_name(films: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for f in films:
        key = _normalize(f["film_adi"])
        if key not in seen:
            seen[key] = f
        else:
            seen[key] = _merge_group([seen[key], f])
    return list(seen.values())


def prepare() -> list[dict]:
    print("[1/4] Kaynak veri okunuyor...")
    source = _load_source_films()
    supplement = _load_supplement()
    print(f"      Kaynak: {len(source)} | Ek: {len(supplement)}")

    print("[2/4] Temizleme ve alan çıkarımı...")
    processed = [_process_film(f) for f in source]
    processed.extend(_process_film(f) for f in supplement)

    print("[3/4] Tekrarlar birleştiriliyor...")
    merged = _merge_duplicates(processed)
    merged = _dedupe_by_name(merged)

    print("[4/4] ID atamaları yapılıyor ve dosyalar yazılıyor...")
    for i, f in enumerate(merged):
        f["id"] = str(i)
        f.pop("temiz_metin", None)
        # DÜZELTME: "alternatif_adlar" artık BURADA SİLİNMİYOR. Bu satır
        # önceden alias verisini (engine.py'nin isimle-film-bulma mantığı
        # için kullandığı) her prepare_dataset.py çalıştırmasında sessizce
        # yok ediyordu. Alan artık _process_film / _merge_group içinde
        # doğru şekilde taşınıyor, burada dokunmuyoruz.

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fp:
        json.dump(merged, fp, ensure_ascii=False, indent=2)

    # İstatistik
    tur_counts: dict[str, int] = defaultdict(int)
    decades: dict[int, int] = defaultdict(int)
    for f in merged:
        for t in f.get("tur", "").split(","):
            tur_counts[t.strip()] += 1
        if f.get("yil"):
            decades[(f["yil"] // 10) * 10] += 1

    print(f"\n[OK] Hazir veri seti: {len(merged)} film")
    print(f"   -> {OUTPUT_PATH}")
    print(f"   Tur dagilimi: {dict(sorted(tur_counts.items(), key=lambda x: -x[1])[:8])}")
    print(f"   Onemli on yillar: {dict(sorted(decades.items()))}")
    return merged


if __name__ == "__main__":
    prepare()