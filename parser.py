"""Kullanıcı sorgusunu yapılandırılmış arama parametrelerine dönüştürür (Tamamen Yapay Zeka/NER Odaklı)."""
from __future__ import annotations

import re

import config
from llm import llm_generate, parse_json_response
from query_schema import ParsedQuery
from text_utils import tr_lower

# Yıl hesaplamaları için dinamik referans (LLM'in "son 5 yıl" gibi terimleri doğru anlaması için)
CURRENT_YEAR = 2026

PARSE_SYSTEM = f"""Sen bir film öneri sisteminin Gelişmiş NER (Varlık Tanıma) ve sorgu analiz motorusun.
Şu anki yıl: {CURRENT_YEAR}. Kullanıcının doğal dil isteğini analiz et ve SADECE geçerli bir JSON objesi döndür.

KURALLAR (ÇOK ÖNEMLİ):
1. AYRIŞTIRMA (Entity Extraction): Yönetmen, oyuncu, tür (aksiyon, gerilim vb.) ve yıl/dönem bilgilerini metinden tespit edip 'filtreler' objesine yerleştir. 
   - Örneğin "son 5 yıl" dendiğinde {CURRENT_YEAR-5} ile {CURRENT_YEAR} arasını al.
   - "doksanlar" dendiğinde 1990-1999 arasını al.
   - ÖNEMLİ: Bir filmin ÜLKESİ/KÖKENİ de tür gibi ele alınır! "Türk filmi",
     "yerli film", "kore filmi", "fransız sineması", "Japon filmi" gibi
     ifadeleri KESİNLİKLE 'filtreler.tur' alanına yaz (gerekirse diğer
     türlerle virgülle birleştir, ör. "türk filmi, komedi"). Bu ifadeleri
     ASLA çöp kelime sayıp atma veya arama_metni'ni boş bırakma sebebi
     yapma — "Türk filmi öner" gibi TEK başına bir istek bile geçerli bir
     'tur' filtresidir.
2. METNİ TEMİZLE: Filtrelere eklediğin kelimeleri (ör: "Tarantino", "aksiyon", "yeni") 'arama_metni'nden KESİNLİKLE ÇIKAR!
3. NEGATİF KAVRAMLAR: Kullanıcının istemediği türleri (ör: "korku olmasın") 'yasakli_turler' dizisine ekle.
4. 'arama_metni': SADECE filmin konusunu, temasını veya atmosferini içermelidir (ör: "tempolu dövüş sahneleri", "beyin yakan kurgu"). Eğer geriye sadece "filmleri", "öner", "istiyorum" gibi çöp kelimeler kalırsa 'arama_metni'ni null bırak.
5. INTENT (Niyet): Birden fazla filtre veya filtre + konu varsa (ör: yönetmen + tür veya tür + arama_metni) intent KESİNLİKLE "karma" olmalıdır! 
6. REFERANS FİLM DÜNYA BİLGİSİ: Kullanıcı bir filme benzetme yaptıysa ('referans_film'), o filmi SEN tanıyorsan (gerçek, vizyona girmiş bir filmse) yönetmenini ve vizyon yılını da 'referans_yonetmen' / 'referans_yil' alanlarına yaz. Bu ÇOK ÖNEMLİ: veri tabanımızda film FARKLI BİR BAŞLIK ALTINDA kayıtlı olabilir (ör. Türkçe yerelleştirilmiş ad), bu yüzden sadece isimle değil yönetmen+yıl kombinasyonuyla da eşleştirme yapılıyor. Emin değilsen bu iki alanı null bırak, UYDURMA.
7. BAĞLAM KULLANIMI:
Bağlam SADECE eksik zamirleri çözmek için kullanılacaktır.

Örneğin:

- "aynı yönetmenin başka filmi"
- "onunkine benzer"
- "biraz daha eski olsun"

ifadelerinde bağlam kullanılabilir.

Ancak kullanıcı yeni ve bağımsız bir istek yazıyorsa,
önceki konuşmadan film adı, yönetmen, oyuncu, tür veya referans film DEVRALINMAYACAKTIR.

Eğer mevcut kullanıcı mesajında film adı geçmiyorsa ve önceki mesaja açık bir gönderme yoksa:

referans_film = null
referans_yonetmen = null
referans_yil = null
olmalıdır.
JSON Şeması:
{{
  "intent": "yonetmen | oyuncu | tur | benzer_film | ruh_hali | karma",
  "arama_metni": "Sadece geriye kalan anlamsal konu/tema (filtre kelimeleri HARİÇ) veya null",
  "referans_film": "Varsa kullanıcının benzettiği/örnek verdiği film adı veya null",
  "referans_yonetmen": "Referans filmi tanıyorsan yönetmeni, tanımıyorsan null",
  "referans_yil": "Referans filmi tanıyorsan vizyon yılı (sayı), tanımıyorsan null",
  "haric_tut": [],
  "filtreler": {{
    "yonetmen": null,
    "oyuncu": null,
    "tur": "virgülle ayrılmış türler veya null",
    "yasakli_turler": [],
    "yil_min": null,
    "yil_max": null
  }}
}}"""

REFERENCE_WORDS = [
    "aynı",
    "onun",
    "onunki",
    "ona",
    "onu",
    "onlarla",
    "bunlar",
    "bunlardan",
    "buna",
    "bunun",
    "benzer",
    "benzeri",
    "benzeyen",
    "devamı",
    "devamındaki",
    "diğeri",
    "öteki",
    "ilk önerdiğin",
    "ikinci önerdiğin",
    "üçüncü önerdiğin",
]
REFERENCE_PHRASES = [
    "gibi",
    "benzer",
    "benzeyen",
    "tarzında",
    "havasında",
    "atmosferinde",
]
# DÜZELTME: Eskiden bu kontrol `x in msg` ile SUBSTRING araması yapıyordu.
# Türkçe eklemeli bir dil olduğu için kök eşleşmesi gerekliydi (tarzı,
# tarzında, benzer, benzeyen...) ama substring araması alakasız kelimelerin
# İÇİNDE geçen aynı harf dizisini de yanlışlıkla yakalıyordu — örneğin
# "Tarzan filmi var mı" cümlesindeki "tarz**an**" kelimesi "tarz" substring'i
# içerdiği için referans-film modunu YANLIŞLIKLA tetikliyordu ("benzin" ->
# "benze" için de aynı sorun geçerliydi). Çözüm: substring yerine, mesajı
# kelimelere bölüp AÇIK bir çekim listesiyle tam kelime eşleşmesi arıyoruz.
# Bu, "tarzan"/"benzin" gibi yanlış-pozitifleri tamamen ortadan kaldırır;
# bedeli, listede olmayan nadir bir çekimin kaçırılabilmesidir — bu, yanlış
# alakasız-referans tetiklemesinden çok daha güvenli bir taraf tutma.
_REFERENCE_STEM_WORDS = frozenset({
    "gibi",
    "benzer", "benzeri", "benzeyen", "benzetir", "benzetilen", "benzese",
    "tarz", "tarzda", "tarzı", "tarzında", "tarzıyla", "tarzınca", "tarzdaki",
    "atmosfer", "atmosferde", "atmosferi", "atmosferinde", "atmosferiyle",
    "havası", "havasında", "havasını", "havasıyla",
    "misali",
})


# ÜLKE/KÖKEN GÜVENLİK KATMANI:
# PARSE_SYSTEM promptunda talimat verilmiş olsa da, LLM'in "türk filmi"
# gibi ifadeleri tür olarak tanıyıp tanımayacağı GARANTİ değil — pratikte
# bu tür tek-filtreli, kısa istekleri (ör. "türk filmi öner") sessizce
# tamamen atıp intent=RUH_HALI + boş arama_metni üretebiliyor (kullanıcının
# tek isteği kayboluyor). Bu yüzden _validate_intent'in intent için yaptığı
# matematiksel düzeltmeyle AYNI felsefede, LLM çıktısından SONRA çalışan,
# deterministik bir kelime eşleştirmesiyle bunu telafi ediyoruz. Anahtarlar
# tr_lower() ile normalize edilmiş TEK kelimelerdir (bkz. _detect_origin_filter).
_ORIGIN_KEYWORDS: dict[str, str] = {
    "türk": "türk filmi",
    "turk": "türk filmi",
    "yerli": "türk filmi",
    "kore": "kore filmi",
    "japon": "japon filmi",
    "fransız": "fransız filmi",
    "fransiz": "fransız filmi",
    "ingiliz": "i̇ngiliz filmi",
    "ıngiliz": "i̇ngiliz filmi",
    "amerikan": "amerikan filmi",
    "italyan": "italyan filmi",
    "ıtalyan": "italyan filmi",
    "alman": "alman filmi",
    "hint": "hint filmi",
    "rus": "rus filmi",
    "ispanyol": "ispanyol filmi",
    "ıspanyol": "ispanyol filmi",
    "iran": "iran filmi",
    "ıran": "iran filmi",
    "çin": "çin filmi",
    "cin": "çin filmi",
}


def _detect_origin_filter(message: str) -> str | None:
    """Mesajda geçen köken/ülke kelimesini (varsa) döndürür.

    Tek-kelime kök eşleşmesi kullanılır (referans kelime tespitiyle aynı
    teknik, bkz. _reference_words_in) — böylece "Türk filmi öner" gibi
    kısa isteklerde LLM'in kelimeyi çöp sayıp atma riski matematiksel
    olarak kapatılmış olur.
    """
    tokens = set(re.findall(r"\w+", tr_lower(message), flags=re.UNICODE))
    for kw, canonical in _ORIGIN_KEYWORDS.items():
        if kw in tokens:
            return canonical
    return None


def _reference_words_in(message: str) -> set[str]:
    tokens = re.findall(r"\w+", tr_lower(message), flags=re.UNICODE)
    return set(tokens) & _REFERENCE_STEM_WORDS


def _looks_like_reference(message: str) -> bool:
    has_phrase = bool(_reference_words_in(message))
    has_long_input = len(message.split()) > 3
    return has_phrase and has_long_input


# Kullanıcı BİLEREK ve AÇIKÇA önceki turdan devamlılık istiyorsa (ör. "aynı
# yönetmenin başka filmi") bu ifadeler geçer. _needs_context VE
# _strip_context_leaked_filters (aşağıda) aynı listeyi paylaşır: ilkinin
# amacı LLM'e bağlamı göndermek, ikincinin amacı ise LLM bağlamdan
# GEREĞİNDEN FAZLASINI (ör. iki tur önceki bir tür filtresini) taşıdığında
# bunu geri almamak — kullanıcı gerçekten devamlılık istediyse silme
# işlemi hiç çalışmaz.
_STRONG_REFERENCE_PHRASES = [
    "aynı",
    "onun",
    "devamı",
    "önceki",
    "ilk önerdiğin",
    "ikinci önerdiğin",
]


def _needs_context(message: str) -> bool:
    msg = tr_lower(message)

    # güçlü referans = her zaman context
    if any(x in msg for x in _STRONG_REFERENCE_PHRASES):
        return True

    # soft referans (_looks_like_reference ile aynı kelime listesi) =
    # sadece film adı varsa context
    return bool(_reference_words_in(message)) and len(msg.split()) <= 6

def _build_context(history: list | None) -> str:
    if not history:
        return "Bağlam yok."

    # Sadece kullanıcı mesajlarını al
    user_turns = [
        m["content"]
        for m in history
        if m["role"] == "user"
    ]

    if not user_turns:
        return "Bağlam yok."

    # DÜZELTME: Bu fonksiyon eskiden pencereyi hardcode ediyordu
    # (user_turns[-2:]), config.CONTEXT_TURN_LIMIT hiç okunmuyordu — yani
    # o ayarı değiştirmenin gerçek davranışa hiçbir etkisi yoktu (ölü
    # config). Artık gerçekten config'ten okunuyor.
    limit = max(1, config.CONTEXT_TURN_LIMIT)
    return "\n".join(user_turns[-limit:])


# "film/filmi/filmler/filmleri..." gibi kelimeler HEMEN HER film sorgusunda
# geçer ("filmine", "filmler", "filminde" vb. çekimleriyle). Kelime bazlı
# destek kontrolünde bunlara izin verilirse (ör. "türk filmi" değerinin
# "filmi" kelimesi, alakasız bir mesajdaki "filmine" içinde substring
# olarak yanlışlıkla eşleşir) HER tur/köken filtresi "desteklenmiş"
# görünür ve sızıntı tespiti asla tetiklenmez. Bu yüzden bu kök ailesi,
# destekleyici kanıt olarak SAYILMAZ; asıl ayırt edici kelime her zaman
# değerin diğer parçasıdır (ör. "türk").
_GENERIC_FILM_WORDS = frozenset({"film", "filmi", "filmler", "filmleri"})


def _value_supported_by_message(value: str, msg_norm: str) -> bool:
    """Bir filtre değerinin (yönetmen adı, tür ismi vb.) mevcut kullanıcı
    mesajında GERÇEKTEN bir karşılığı olup olmadığını kontrol eder.

    Tam ifade ya da onu oluşturan AYIRT EDİCİ kelimelerden biri (kısa/
    gürültülü eşleşmeleri önlemek için >=3 harf, jenerik "film" ailesi
    hariç — bkz. _GENERIC_FILM_WORDS) mesajda geçiyorsa 'desteklenmiş'
    sayılır. Örn. "christopher nolan" değeri için mesajda sadece "nolan"
    geçmesi yeterlidir; "türk filmi" değeri için ise "filmi" değil "türk"
    kelimesinin geçmesi gerekir.
    """
    v = tr_lower(value)
    if v in msg_norm:
        return True
    words = [w for w in v.split() if w not in _GENERIC_FILM_WORDS]
    if not words:
        return False
    return any(len(w) >= 3 and w in msg_norm for w in words)


def _strip_context_leaked_filters(parsed_obj: ParsedQuery, message: str) -> ParsedQuery:
    """BAĞLAM SIZINTISI GÜVENLİK KATMANI.

    PARSE_SYSTEM'in 7. kuralı, bağlamın (önceki kullanıcı mesajlarının)
    SADECE zamir çözümü için kullanılmasını, yeni ve bağımsız bir istekte
    önceki turdan yönetmen/tür/oyuncu filtresi DEVRALINMAMASINI söylüyor.
    Ama LLM bunu her zaman tutarlı uygulayamıyor: "Christopher Nolan
    filmleri öner" + sonra "Inception'a benzer film öner" gibi bir
    sırada, ikinci (bağımsız) sorguya hem Nolan hem de daha ÖNCEKİ bir
    turdaki "türk filmi" filtresi bulaşabiliyor — kullanıcı bug raporu
    tam olarak bu davranışı gösteriyor.

    engine.py bu filtrelere körü körüne güvendiği için (bkz. search()),
    sızıntı doğrudan alakasız sonuçlara yol açıyor. Bu fonksiyon,
    _detect_origin_filter ile aynı felsefede, LLM çıktısından SONRA
    çalışan deterministik bir son kontrol: bir filtre değerinin mevcut
    mesajda hiçbir karşılığı yoksa (bkz. _value_supported_by_message) ve
    kullanıcı AÇIKÇA devamlılık istemiyorsa (güçlü referans ifadesi
    yoksa), o filtre silinir.

    Kullanıcı "aynı yönetmenin başka filmi" derse hiçbir şey silinmez —
    o durumda bağlamdan devralınma zaten istenen davranıştır.
    """
    msg_norm = tr_lower(message)

    if any(p in msg_norm for p in _STRONG_REFERENCE_PHRASES):
        return parsed_obj

    f = parsed_obj.filtreler
    if f.yonetmen and not _value_supported_by_message(f.yonetmen, msg_norm):
        f.yonetmen = None
    if f.oyuncu and not _value_supported_by_message(f.oyuncu, msg_norm):
        f.oyuncu = None
    if f.tur:
        kept = [
            t.strip() for t in f.tur.split(",")
            if t.strip() and _value_supported_by_message(t.strip(), msg_norm)
        ]
        f.tur = ", ".join(kept) if kept else None

    return parsed_obj


def _validate_intent(parsed: ParsedQuery) -> ParsedQuery:
    """LLM'in intent'i yanlış hesaplama ihtimaline karşı matematiksel son güvenlik katmanı."""
    active_filters = sum([
        bool(parsed.filtreler.yonetmen),
        bool(parsed.filtreler.oyuncu),
        bool(parsed.filtreler.tur),
        bool(parsed.referans_film),
        bool(parsed.arama_metni)
    ])

    # Gerçek NER mantığı: 1'den fazla özellik ayıklandıysa bu kesinlikle "karma" sorgudur.
    if active_filters > 1:
        parsed.intent = "karma"
    elif active_filters == 1:
        # Sadece tek kriter varsa net intent ataması
        if parsed.filtreler.yonetmen: parsed.intent = "yonetmen"
        elif parsed.filtreler.oyuncu: parsed.intent = "oyuncu"
        elif parsed.referans_film: parsed.intent = "benzer_film"
        elif parsed.filtreler.tur: parsed.intent = "tur"
        elif parsed.arama_metni: parsed.intent = "ruh_hali"
    else:
        parsed.intent = "ruh_hali" # Varsayılan güvenlik (Fail-safe)

    # Eğer kullanıcı bir filme "benzer" dediyse, o filmi sonuçlarda görmemek için hariç tutulanlara ekle
    if parsed.referans_film:
        haric = set(parsed.haric_tut or [])
        haric.add(parsed.referans_film)
        parsed.haric_tut = list(haric)

    return parsed


def parse_query(message: str, history: list | None = None) -> dict:
    """Kullanıcı mesajını yapılandırılmış sorguya dönüştürür."""
    history = history or []

    if _needs_context(message):
        context = _build_context(history)
    else:
        context = "Bağlam yok — bağımsız sorgu."

    # 1. LLM'den NER (Varlık Çıkarma) ve Slot Doldurma isteği
    llm_result = llm_generate(
        PARSE_SYSTEM,
        f"Kullanıcı isteği: {message}\nBağlam: {context}",
        json_mode=True,
    )

    # 2. LLM yanıtını JSON'a çevir
    # DÜZELTME: Groq API'si çökerse/rate-limit'e takılırsa/ağ hatası
    # verirse (llm_generate None döner) ya da JSON parse edilemezse,
    # eskiden tamamen BOŞ bir ParsedQuery() üretiliyordu (arama_metni="").
    # Bu, kullanıcının yazdığı hiçbir şeyin arama motoruna ULAŞMAMASI
    # anlamına geliyordu — sonuç olarak sistem sanki kullanıcı hiçbir şey
    # yazmamış gibi rastgele/alakasız öneriler dönüyordu ve kullanıcı
    # bunun NEDENİNİ hiç göremiyordu. Artık bu durumda ham kullanıcı
    # mesajı doğrudan arama_metni olarak kullanılıyor: yönetmen/oyuncu/tür
    # gibi yapılandırılmış filtreler bu turda çıkarılamaz (NLU devre dışı),
    # ama en azından anlamsal arama kullanıcının GERÇEK isteği üzerinden
    # çalışmaya devam eder — sessizce boş sonuç dönmek yerine "daha az
    # akıllı ama işlevsel" bir moda düşer.
    raw_json = parse_json_response(llm_result)
    if raw_json:
        parsed_obj = ParsedQuery.from_dict(raw_json)
    else:
        print(
            "[UYARI] Groq NLU yanıtı alınamadı ya da JSON'a çevrilemedi — "
            "ham kullanıcı mesajı doğrudan arama metni olarak kullanılıyor "
            "(bu turda yönetmen/oyuncu/tür filtre çıkarımı YAPILAMAYACAK)."
        )
        parsed_obj = ParsedQuery(arama_metni=message.strip())

    # BAĞLAM SIZINTISI TEMİZLİĞİ: intent hesaplamasından ve referans_film
    # kontrolünden ÖNCE çalışmalı — aksi halde silinmesi gereken bir
    # filtre yüzünden intent zaten yanlış hesaplanmış olur (bkz. fonksiyon
    # docstring'i).
    parsed_obj = _strip_context_leaked_filters(parsed_obj, message)

    # Kullanıcı bu mesajda film adı vermediyse ve
    # önceki konuşmaya da referans vermiyorsa
    # LLM'in eski referans filmi taşımasını engelle.

    if (
            parsed_obj.referans_film
            and not _needs_context(message)
            and not _looks_like_reference(message)
    ):
        parsed_obj.referans_film = None
        parsed_obj.referans_yonetmen = None
        parsed_obj.referans_yil = None

    # 3. İhtiyari hataları matematiksel olarak düzelt ve intent'i kesinleştir
    validated_obj = _validate_intent(parsed_obj)

    # 4. GÜVENLİK KATMANI: mesajda bir ülke/köken kelimesi ("türk", "yerli",
    # "kore" vb.) geçiyor ama LLM bunu ne filtreler.tur'a ne arama_metni'ne
    # yazdıysa (bkz. _detect_origin_filter docstring — LLM bunu sessizce
    # atabiliyor), deterministik olarak filtreler.tur'a ekleyip intent'i
    # yeniden hesaplıyoruz. LLM zaten doğru yakaladıysa (tur veya
    # arama_metni içinde kelime kökü zaten varsa) tekrar eklemiyoruz.
    origin = _detect_origin_filter(message)
    if origin:
        origin_stem = origin.split()[0]
        already_captured = origin_stem in tr_lower(
            f"{validated_obj.filtreler.tur or ''} {validated_obj.arama_metni or ''}"
        )
        if not already_captured:
            validated_obj.filtreler.tur = (
                f"{validated_obj.filtreler.tur}, {origin}"
                if validated_obj.filtreler.tur else origin
            )
            validated_obj = _validate_intent(validated_obj)

    return validated_obj.to_dict()