"""Doğal dil yanıt üretici — LLM (ücretsiz) veya şablon fallback."""
from llm import llm_generate
from text_utils import normalize_tr

INTENT_GUIDANCE = {
    "yonetmen": "Yönetmenin imza tarzına, anlatım diline ve tematik tercihlerine odaklan.",
    "oyuncu": "Oyuncunun performans tarzına ve rol seçimlerine odaklan.",
    "tur": "Belirtilen tür(ler)in özelliklerini ve atmosferini vurgula.",
    "benzer_film": "Referans filme tema, tempo ve atmosfer benzerliğini açıkla.",
    "ruh_hali": "Kullanıcının aradığı ruh haline ve duygusal tonu vurgula.",
    "karma": "Kullanıcının isteğindeki temaları ve filmlerin neden uyduğunu açıkla.",
}

RESPONSE_SYSTEM = """Sen samimi, bilgili bir film danışmanısın.
Kullanıcıya Türkçe cevap ver.

KURALLAR:
- SADECE verilen film listesinden bahset, uydurma yapma
- Sana "Önerilecek filmler" başlığı altında bir liste veriliyorsa, bu filmler
  veri setinde GERÇEKTEN var demektir. Bu listedeki hiçbir film için
  "veri setinde yok", "bulunamadı" veya benzeri bir ifade KULLANMA.
- "NOT:" ile başlayan satır SADECE kullanıcının bahsettiği referans film
  hakkındadır. Bu notu asla aşağıdaki öneri listesine genelleme.
- Her film için 1-2 cümlelik neden açıkla (tema, atmosfer, yönetmen tarzı)
- Sana film listesi verildiyse cevabında mutlaka o listedeki tüm filmlerden
  bahset; "bulamadım" türü bir cevapla listeyi atlama
- Samimi ama profesyonel ol
- Emoji kullanma
- Verilen kadar film öner (genelde 5)"""


def _get(obj, key, default=""):
    """Hem sözlük (dict) hem de nesne (object) tiplerinden güvenli veri okur."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _template_response(
    message: str,
    hits: list,
    ref_film,
    intent: str,
    parsed: dict,
) -> str:
    lines = []

    ref_name = _get(ref_film, "film_adi")
    ref_year = _get(ref_film, "yil")

    if ref_film:
        lines.append(
            f'"{ref_name}" ({ref_year}) filmini referans aldım.\n'
        )
    elif parsed.get("referans_film"):
        lines.append(
            f'Not: "{parsed["referans_film"]}" veri setimizde yok; '
            f"benzer tema/atmosfere göre öneriler:\n"
        )

    intro = {
        "yonetmen": "Yönetmen tarzına uygun seçimler:\n",
        "oyuncu": "Oyuncunun filmografisine uygun seçimler:\n",
        "tur": "İstediğin türe uygun filmler:\n",
        "benzer_film": "Benzer tema ve atmosfere sahip filmler:\n",
        "ruh_hali": "Aradığın ruh haline uygun filmler:\n",
    }.get(intent, "Senin için seçtiklerim:\n")
    lines.append(intro)

    for i, hit in enumerate(hits, 1):
        doc = _get(hit, "document")
        film_adi = _get(hit, "film_adi")
        yil = _get(hit, "yil")
        yonetmen = str(_get(hit, "yonetmen", "")).title()

        if doc:
            if "Özet:" in doc:
                ozet = doc.split("Özet:")[-1].split("\n")[0].strip()
            else:
                ozet = doc[:220]
        else:
            ozet = _get(hit, "ozet", "")

        ozet = ozet[:220].rsplit(" ", 1)[0] + "..." if len(ozet) > 220 else ozet
        lines.append(f"{i}. **{film_adi}** ({yil or '?'})")
        lines.append(f"   Yönetmen: {yonetmen}")
        lines.append(f"   {ozet}")
        lines.append("")

    return "\n".join(lines)


_NEGATION_PATTERNS = (
    "veri setinde yok",
    "veri setimizde yok",
    "veritabanında yok",
    "bulunamadı",
    "bulamadım",
    "maalesef",
    "elimde böyle bir film",
    "kayıtlı değil",
)


def _is_grounded(llm_text: str, hits: list) -> bool:
    if not llm_text:
        return False
    text_norm = normalize_tr(llm_text)

    mentioned = sum(
        1 for h in hits if normalize_tr(str(_get(h, "film_adi"))) in text_norm
    )
    min_required = 1 if len(hits) <= 2 else max(1, len(hits) // 3)
    if mentioned < min_required:
        return False

    if any(pat in text_norm for pat in _NEGATION_PATTERNS):
        return False
    return True


def _extract_ozet(document: str) -> str:
    if not document:
        return ""
    if "Özet:" in document:
        return document.split("Özet:")[-1].split("\n")[0].strip()
    return document[:300]


def generate_response(
    message: str,
    hits: list,
    ref_film,
    parsed: dict,
) -> str:
    if not hits:
        ref_name = parsed.get("referans_film")
        if ref_name and not ref_film:
            return (
                f'"{ref_name}" veri setimizde bulunamadı. '
                "Farklı bir film adı veya tarif deneyebilirsin."
            )
        filters = parsed.get("filtreler") or {}
        hint = ""
        if filters.get("tur"):
            hint = f" ({filters['tur']} türü için)"
        elif filters.get("yonetmen"):
            hint = f" ({filters['yonetmen']} için)"
        return (
            f"Bu kriterlere uygun film bulamadım{hint}. "
            "Farklı bir tarif deneyebilir veya bir film adı verebilirsin."
        )

    intent = parsed.get("intent", "karma")
    guidance = INTENT_GUIDANCE.get(intent, INTENT_GUIDANCE["karma"])
    filters = parsed.get("filtreler") or {}

    film_block = "\n\n".join(
        f"{i+1}) {_get(h, 'film_adi')} ({_get(h, 'yil')}) — Yönetmen: {str(_get(h, 'yonetmen', '')).title()}\n"
        f"{_extract_ozet(str(_get(h, 'document')) or str(_get(h, 'ozet')))}"
        for i, h in enumerate(hits)
    )

    ref_line = ""
    if ref_film:
        ref_line = f'Referans film: {_get(ref_film, "film_adi")} ({_get(ref_film, "yil")})\n'
    elif parsed.get("referans_film"):
        ref_line = (
            f'NOT (sadece referans film hakkında): kullanıcının bahsettiği '
            f'"{parsed["referans_film"]}" adlı film veri setinde bulunamadı; '
            f"aşağıdaki öneriler tema/atmosfer benzerliğine göre seçildi. "
            f"Aşağıdaki film listesi veri setinde GERÇEKTEN mevcuttur.\n"
        )

    filter_line = ""
    if filters.get("yasakli_turler"):
        filter_line += f"Hariç tutulan türler: {', '.join(filters['yasakli_turler'])}\n"
    if filters.get("tur"):
        filter_line += f"İstenen türler: {filters['tur']}\n"

    prompt = f"""Kullanıcı isteği: {message}
Intent: {intent}
Yönlendirme: {guidance}
{filter_line}{ref_line}
Önerilecek filmler (SADECE bunları kullan, hepsi veri setinde mevcuttur):
{film_block}

Her film için kısa bir neden yaz."""

    llm_text = llm_generate(RESPONSE_SYSTEM, prompt)
    if llm_text and len(llm_text.strip()) > 50 and _is_grounded(llm_text, hits):
        return llm_text.strip()

    return _template_response(message, hits, ref_film, intent, parsed)