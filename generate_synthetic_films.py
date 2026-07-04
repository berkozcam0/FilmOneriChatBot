"""
Otonom film keşif ve veri seti genişletme scripti.

Groq (llama-3.3-70b-versatile) ile veritabanında olmayan gerçek filmleri
4 tematik paket halinde üretir, doğrular ve films.json'a ekler.

Kullanım:
    python generate_synthetic_films.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field

import config
from engine import FilmSearchEngine, _normalize
from prepare_dataset import build_film_document
from text_utils import tr_lower

# ── Ortam ────────────────────────────────────────────────────────────────────
load_dotenv(config.BASE_DIR / ".env")

MODEL_NAME = "llama-3.3-70b-versatile"
MIN_FILMS_PER_PACK = 15
MAX_FILMS_PER_PACK = 20
MIN_OZET_CHARS = 150
MIN_ETIKET_COUNT = 6
MIN_SENTENCE_COUNT = 5
REQUIRED_FIELDS = ("film_adi", "yonetmen", "oyuncular", "yil", "tur", "ozet", "etiketler", "document")

# ── Paket tanımları ──────────────────────────────────────────────────────────
PACK_DEFINITIONS = [
    {
        "id": 1,
        "title": "Türk Sineması Klasikleri ve Modern Şaheserleri",
        "description": (
            "Yeşilçam'ın kült dram ve komedilerinden; Nuri Bilge Ceylan, Reha Erdem, "
            "Zeki Demirkubuz, Semih Kaplanoğlu, Yeşim Ustaoğlu ve Derviş Zaim gibi "
            "yönetmenlerin ödüllü modern Türk filmlerine kadar geniş bir yelpaze. "
            "Hollywood veya yabancı yapım DEĞİL — yalnızca Türkiye sineması."
        ),
        "target_count": 18,
    },
    {
        "id": 2,
        "title": "Dünya Sineması ve Festival Filmleri",
        "description": (
            "Hollywood dışı uluslararası sinema: Kore (Park Chan-wook, Bong Joon-ho, "
            "Hong Sang-soo), İran (Abbas Kiarostami, Asghar Farhadi), Fransız (Godard, "
            "Truffaut, Agnès Varda), İtalyan (Fellini, Pasolini), İspanyol (Buñuel, "
            "Almodóvar), İskandinav (Bergman, von Trier) ve diğer festival ödüllü "
            "yabancı dilde kült yapımlar."
        ),
        "target_count": 18,
    },
    {
        "id": 3,
        "title": "Animasyon, Anime ve Bilim Kurgu",
        "description": (
            "Studio Ghibli / Miyazaki şaheserleri, derinlikli anime (Satoshi Kon, "
            "Mamoru Hosoda), siberpunk klasikleri (Blade Runner, Ghost in the Shell), "
            "distopik bilim kurgu ve verisetinde az temsil edilen animasyon / sci-fi "
            "yapımları. Gerçek, vizyona girmiş filmler."
        ),
        "target_count": 17,
    },
    {
        "id": 4,
        "title": "Bağımsız (Indie) ve Kült Klasikler",
        "description": (
            "Büyük bütçeli olmayan, ana akım medyada az bilinen ama sinema tutkunlarının "
            "bayıldığı niş dram, gizem, neo-noir, psikolojik gerilim ve deneysel "
            "bağımsız yapımlar. Eraserhead, Pi, Primer, Moon gibi kült filmler "
            "seviyesinde gerçek eserler."
        ),
        "target_count": 17,
    },
]

# ── Pydantic şemalar (Structured Output) ─────────────────────────────────────
class GeneratedFilm(BaseModel):
    film_adi: str = Field(description="Filmin resmi adı")
    yonetmen: str = Field(description="Yönetmen adı")
    oyuncular: str = Field(description="Virgülle ayrılmış oyuncu listesi")
    yil: int = Field(description="Vizyon yılı")
    tur: str = Field(description="Tür veya türler, virgülle ayrılmış")
    ozet: str = Field(description="En az 5-6 cümlelik derin özet")
    etiketler: list[str] = Field(description="6-8 spesifik anahtar kelime")
    document: str = Field(description="BGE-M3 için zengin Türkçe arama paragrafı")


class PackResponse(BaseModel):
    filmler: list[GeneratedFilm]


# ── Hafif arama motoru (ML modelleri yüklemeden find_film_by_name) ───────────
class LightweightFilmChecker(FilmSearchEngine):
    """FilmSearchEngine'in isim eşleştirmesini kullanır; embedding/reranker yüklemez."""

    def _load_models(self) -> None:
        pass

    def _build_bm25(self) -> None:
        pass


# ── Log yardımcıları ─────────────────────────────────────────────────────────
def _log(level: str, message: str) -> None:
    icons = {
        "INFO": "ℹ️ ",
        "OK": "✅",
        "WARN": "⚠️ ",
        "ERR": "❌",
        "PACK": "📦",
        "API": "🤖",
    }
    prefix = icons.get(level, "•")
    print(f"[{prefix} {level}] {message}", flush=True)


def _banner(text: str) -> None:
    line = "═" * 62
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n", flush=True)


# ── İstatistik takibi ────────────────────────────────────────────────────────
@dataclass
class ImportStats:
    suggested: int = 0
    rejected_existing: int = 0
    rejected_invalid: int = 0
    rejected_duplicate_session: int = 0
    approved: list[dict] = field(default_factory=list)
    rejected_existing_titles: list[str] = field(default_factory=list)
    rejected_invalid_details: list[str] = field(default_factory=list)

    def record_existing(self, title: str) -> None:
        self.rejected_existing += 1
        self.rejected_existing_titles.append(title)

    def record_invalid(self, title: str, reason: str) -> None:
        self.rejected_invalid += 1
        self.rejected_invalid_details.append(f"{title}: {reason}")

    def record_session_dup(self, title: str) -> None:
        self.rejected_duplicate_session += 1
        self.rejected_invalid_details.append(f"{title}: oturum içi mükerrer")


# ── Doğrulama ────────────────────────────────────────────────────────────────
def _sentence_count(text: str) -> int:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return sum(1 for p in parts if len(p.strip()) > 15)


def _validate_film(raw: dict, engine: LightweightFilmChecker, session_norm: set[str]) -> tuple[Optional[dict], Optional[str]]:
    """Tek film doğrulama. Başarılıysa (film_dict, None), değilse (None, sebep)."""
    for fld in REQUIRED_FIELDS:
        val = raw.get(fld)
        if val is None or (isinstance(val, str) and not val.strip()):
            return None, f"eksik alan: {fld}"
        if fld == "etiketler" and (not isinstance(val, list) or len(val) == 0):
            return None, f"eksik alan: {fld}"
        if fld == "yil":
            try:
                val = int(val)
                raw["yil"] = val
            except (ValueError, TypeError):
                return None, f"geçersiz yıl: {val}"
            if val < 1888 or val > datetime.now().year + 1:
                return None, f"geçersiz yıl: {val}"

    title = str(raw["film_adi"]).strip()
    ozet = str(raw["ozet"]).strip()
    etiketler = [str(t).strip() for t in raw["etiketler"] if str(t).strip()]

    if len(ozet) < MIN_OZET_CHARS:
        return None, f"özet çok kısa ({len(ozet)} karakter, min {MIN_OZET_CHARS})"

    if _sentence_count(ozet) < MIN_SENTENCE_COUNT:
        return None, f"özet yeterince cümle içermiyor (min {MIN_SENTENCE_COUNT})"

    if len(etiketler) < MIN_ETIKET_COUNT:
        return None, f"etiket sayısı yetersiz ({len(etiketler)}, min {MIN_ETIKET_COUNT})"

    norm_title = _normalize(title)
    if norm_title in session_norm:
        return None, "oturum içi mükerrer"

    if engine.find_film_by_name(title) is not None:
        return None, "verisetinde zaten mevcut"

    film = {
        "film_adi": title,
        "yonetmen": tr_lower(str(raw["yonetmen"]).strip()),
        "oyuncular": tr_lower(str(raw["oyuncular"]).strip()),
        "yil": int(raw["yil"]),
        "tur": tr_lower(str(raw["tur"]).strip()),
        "ozet": ozet,
        "etiketler": etiketler[:12],
    }
    film["document"] = build_film_document(film)
    return film, None


# ── Groq istemcisi ───────────────────────────────────────────────────────────
def _get_groq_client() -> Groq:
    api_key = (os.environ.get("GROQ_API_KEY") or "").split("#")[0].strip()
    if not api_key:
        _log("ERR", "GROQ_API_KEY .env dosyasında bulunamadı!")
        sys.exit(1)
    os.environ["GROQ_API_KEY"] = api_key
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))


JSON_OUTPUT_TEMPLATE = """{
  "filmler": [
    {
      "film_adi": "Filmin resmi adı",
      "yonetmen": "yönetmen adı (küçük harf)",
      "oyuncular": "oyuncu1, oyuncu2, oyuncu3 (küçük harf)",
      "yil": 1999,
      "tur": "dram, gerilim",
      "ozet": "En az 5-6 cümlelik derin özet; konu, alt temalar, atmosfer ve etki.",
      "etiketler": ["neo-noir", "anti-kahraman", "taşra-sıkıntısı", "retro-futurizm", "varoluşçu-kriz", "klostrofobik-atmosfer"],
      "document": "BGE-M3 anlamsal arama için zengin Türkçe paragraf: film adı, yıl, yönetmen, oyuncular, tür, özet ve etiketler birleşik."
    }
  ]
}"""


def _build_system_prompt() -> str:
    return f"""Sen dünya sinemasına hakim, titiz bir film küratörüsün.

GÖREV: Kullanıcının belirttiği tematik pakette, verisetinde OLMAYAN gerçek filmleri keşfet ve YALNIZCA geçerli JSON döndür.

KESİN KURALLAR:
1. SADECE gerçek, vizyona girmiş filmler öner. Hayali film, uydurma yönetmen veya sahte oyuncu YAZMA.
2. Prompt içindeki 'Mevcut Filmler' listesindeki filmleri KESİNLİKLE tekrar önerme.
3. Bu oturumda daha önce önerilen filmleri de tekrarlama.
4. 'ozet' alanı ASLA yüzeysel olmamalı: en az 5-6 tam cümle; filmin konusunu, alt temalarını (felsefi/psikolojik), sinematografik atmosferini ve izleyicide bıraktığı etkiyi derinlemesine işle.
5. 'etiketler' listesi en az 6-8 adet spesifik, tireli veya birleşik anahtar kelime içermeli (ör: neo-noir, anti-kahraman, taşra-sıkıntısı, retro-futurizm, varoluşçu-kriz, klostrofobik-atmosfer).
6. 'document' alanı; Film Adı, Yıl, Yönetmen, Oyuncular, Tür, Derin Özet ve Etiketlerin anlamsal arama (BGE-M3) için optimize edilmiş zengin bir Türkçe paragraf olmalı.
7. yonetmen ve oyuncular alanları küçük harfle yazılmalı.
8. Belirtilen hedef film sayısına ulaşmaya çalış; her film benzersiz olmalı.
9. Yanıtın SADECE aşağıdaki JSON şablonuna uygun tek bir JSON nesnesi olmalı; markdown veya açıklama ekleme.

ZORUNLU JSON ŞABLONU:
{JSON_OUTPUT_TEMPLATE}"""


def _build_user_prompt(
    pack: dict,
    existing_titles: list[str],
    session_titles: list[str],
    already_added: list[str],
) -> str:
    existing_block = "\n".join(f"- {t}" for t in existing_titles[:800])
    if len(existing_titles) > 800:
        existing_block += f"\n... ve {len(existing_titles) - 800} film daha"

    session_block = ""
    if session_titles:
        session_block = "\n\nBu oturumda zaten önerilen filmler (TEKRARLAMA):\n"
        session_block += "\n".join(f"- {t}" for t in session_titles)

    added_block = ""
    if already_added:
        added_block = "\n\nBu oturumda verisetine eklenen filmler (TEKRARLAMA):\n"
        added_block += "\n".join(f"- {t}" for t in already_added)

    return f"""PAKET {pack['id']}: {pack['title']}

Tema açıklaması:
{pack['description']}

Hedef: Verisetinde olmayan, yüksek kaliteli EN AZ {MIN_FILMS_PER_PACK} — EN FAZLA {MAX_FILMS_PER_PACK} gerçek film öner.
İdeal hedef: {pack['target_count']} film.

Mevcut Filmler (bunları KESİNLİKLE önerme):
{existing_block}
{session_block}
{added_block}

Yalnızca JSON şemasına uygun 'filmler' dizisini doldur."""


def _parse_groq_json(text: str) -> PackResponse:
    cleaned = re.sub(r"```(?:json)?|```", "", (text or "").strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group())
    return PackResponse.model_validate(data)


def _call_groq(
    client: Groq,
    pack: dict,
    existing_titles: list[str],
    session_titles: list[str],
    already_added: list[str],
    retries: int = 3,
) -> list[dict]:
    system = _build_system_prompt()
    user = _build_user_prompt(pack, existing_titles, session_titles, already_added)

    for attempt in range(1, retries + 1):
        try:
            _log("API", f"Paket {pack['id']} — Groq isteği gönderiliyor (deneme {attempt}/{retries})...")
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            parsed = _parse_groq_json(content)
            films = [f.model_dump() for f in parsed.filmler]
            _log("OK", f"Paket {pack['id']} — Groq {len(films)} film önerdi.")
            return films

        except Exception as exc:
            _log("WARN", f"Paket {pack['id']} — API hatası: {exc}")
            if attempt < retries:
                wait = 2 ** attempt
                _log("INFO", f"{wait} saniye bekleniyor...")
                time.sleep(wait)
            else:
                raise

    return []


# ── Dosya işlemleri ──────────────────────────────────────────────────────────
def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    return data if isinstance(data, list) else []


def _save_json_list(path: Path, films: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(films, fp, ensure_ascii=False, indent=2)


def _append_films(path: Path, new_films: list[dict], assign_ids: bool = True) -> None:
    existing = _load_json_list(path)
    if assign_ids:
        _assign_ids_for_new(existing, new_films)
    existing.extend(new_films)
    _save_json_list(path, existing)


def _assign_ids_for_new(existing: list[dict], new_films: list[dict]) -> None:
    max_id = -1
    for f in existing:
        try:
            max_id = max(max_id, int(f.get("id", -1)))
        except (ValueError, TypeError):
            pass
    for f in new_films:
        max_id += 1
        f["id"] = str(max_id)


def _get_existing_titles(films: list[dict]) -> list[str]:
    return sorted({f["film_adi"] for f in films if f.get("film_adi")})


def _update_engine_cache(engine: LightweightFilmChecker, new_films: list[dict]) -> None:
    # NOT: FilmSearchEngine'de "display_names" diye bir alan yok (sadece
    # films / embeddings / film_names var). Eskiden burada
    # engine.display_names.append(...) çağrısı vardı; bu satır var olmayan
    # bir attribute'a erişmeye çalıştığı için ilk onaylanan filmde
    # AttributeError fırlatıp scripti çökertiyordu.
    for f in new_films:
        engine.films.append(f)
        engine.film_names.append(_normalize(f["film_adi"]))


# ── Rapor ────────────────────────────────────────────────────────────────────
def _write_report(stats: ImportStats, report_path: Path) -> None:
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "═" * 62,
        "  FİLM İTHALAT RAPORU — generate_synthetic_films.py",
        f"  Tarih: {today}",
        "═" * 62,
        "",
        "ÖZET",
        "────",
        f"  Toplam önerilen film    : {stats.suggested}",
        f"  Verisetinde mevcut      : {stats.rejected_existing} (elendi)",
        f"  Eksik/hatalı veri       : {stats.rejected_invalid} (reddedildi)",
        f"  Oturum mükerrer         : {stats.rejected_duplicate_session}",
        f"  Başarıyla eklenen       : {len(stats.approved)}",
        "",
    ]

    if stats.approved:
        lines.extend(["EKLENEN FİLMLER", "───────────────"])
        for i, f in enumerate(stats.approved, 1):
            lines.append(
                f"  {i:3d}. {f['film_adi']} ({f.get('yil', '?')}) — {f.get('yonetmen', '?')}"
            )
        lines.append("")

    if stats.rejected_existing_titles:
        lines.extend(["ELENEN — VERİSETİNDE MEVCUT", "──────────────────────────"])
        for t in stats.rejected_existing_titles[:50]:
            lines.append(f"  • {t}")
        if len(stats.rejected_existing_titles) > 50:
            lines.append(f"  ... ve {len(stats.rejected_existing_titles) - 50} film daha")
        lines.append("")

    if stats.rejected_invalid_details:
        lines.extend(["REDDEDİLEN — GEÇERSİZ VERİ", "──────────────────────────"])
        for d in stats.rejected_invalid_details[:50]:
            lines.append(f"  • {d}")
        if len(stats.rejected_invalid_details) > 50:
            lines.append(f"  ... ve {len(stats.rejected_invalid_details) - 50} kayıt daha")
        lines.append("")

    lines.append("═" * 62)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _log("OK", f"Rapor yazıldı: {report_path}")


# ── Ana iş akışı ─────────────────────────────────────────────────────────────
def process_pack(
    pack: dict,
    client: Groq,
    engine: LightweightFilmChecker,
    existing_titles: list[str],
    session_titles: list[str],
    already_added: list[str],
    session_norm: set[str],
    stats: ImportStats,
) -> list[dict]:
    _banner(f"PAKET {pack['id']}: {pack['title']}")

    raw_films = _call_groq(client, pack, existing_titles, session_titles, already_added)
    stats.suggested += len(raw_films)

    approved_pack: list[dict] = []
    for raw in raw_films:
        title = raw.get("film_adi", "?")
        film, reason = _validate_film(raw, engine, session_norm)

        if film is None:
            if reason == "verisetinde zaten mevcut":
                stats.record_existing(str(title))
                _log("WARN", f"  '{title}' — verisetinde mevcut, elendi.")
            elif reason == "oturum içi mükerrer":
                stats.record_session_dup(str(title))
                _log("WARN", f"  '{title}' — oturum içi mükerrer, elendi.")
            else:
                stats.record_invalid(str(title), reason or "bilinmeyen")
                _log("WARN", f"  '{title}' — reddedildi: {reason}")
            continue

        norm = _normalize(film["film_adi"])
        session_norm.add(norm)
        session_titles.append(film["film_adi"])
        approved_pack.append(film)
        stats.approved.append(film)
        _update_engine_cache(engine, [film])
        _log("OK", f"  ✓ '{film['film_adi']}' ({film['yil']}) onaylandı.")

    _log("OK", f"Paket {pack['id']}'den {len(approved_pack)} yeni film onaylandı.")
    return approved_pack


def main() -> None:
    _banner("OTONOM FİLM KEŞİF MOTORU — Groq Llama 3.3 70B")
    _log("INFO", f"Model: {MODEL_NAME} | Hedef: {sum(p['target_count'] for p in PACK_DEFINITIONS)} film (4 paket)")

    if not config.FILMS_JSON.exists():
        _log("ERR", f"İndeks bulunamadı: {config.FILMS_JSON}\nÖnce çalıştırın: python build_index.py")
        sys.exit(1)

    _log("INFO", "Film arama motoru (hafif mod) yükleniyor...")
    engine = LightweightFilmChecker()
    existing_titles = _get_existing_titles(engine.films)

    _log("INFO", f"Mevcut veriseti: {len(engine.films)} film, {len(existing_titles)} benzersiz başlık")

    client = _get_groq_client()
    stats = ImportStats()
    session_titles: list[str] = []
    session_norm: set[str] = set()
    all_approved: list[dict] = []

    for pack in PACK_DEFINITIONS:
        pack_films = process_pack(
            pack=pack,
            client=client,
            engine=engine,
            existing_titles=existing_titles,
            session_titles=session_titles,
            already_added=[f["film_adi"] for f in all_approved],
            session_norm=session_norm,
            stats=stats,
        )
        all_approved.extend(pack_films)
        existing_titles.extend(f["film_adi"] for f in pack_films)

    if not all_approved:
        _log("WARN", "Hiçbir film onaylanmadı — dosyalara yazılmadı.")
    else:
        _log("INFO", f"{len(all_approved)} onaylı film dosyalara yazılıyor...")

        # DÜZELTME: Eskiden bu iki çağrı `_assign_ids_for_new`'i BAĞIMSIZ
        # iki kez tetikliyordu (biri FILMS_JSON'daki mevcut max id'ye göre,
        # diğeri PREPARED_JSON'daki mevcut max id'ye göre) — bu yüzden aynı
        # filmin FILMS_JSON'daki "id"si ile PREPARED_JSON'daki "id"si
        # SESSİZCE farklı çıkıyordu. Şu an hiçbir yerde id ile lookup
        # yapılmadığı için bu davranışsal bir hataya yol açmıyor ama veri
        # tutarlılığı açısından yanlıştı. Artık id'ler yalnızca İLK
        # yazımda (FILMS_JSON) atanıyor, ikinci yazımda (PREPARED_JSON)
        # aynı id'ler AYNEN korunuyor.
        _append_films(config.FILMS_JSON, all_approved)
        _log("OK", f"Eklenildi: {config.FILMS_JSON}")

        if config.PREPARED_JSON.exists():
            _append_films(
                config.PREPARED_JSON,
                [dict(f) for f in all_approved],
                assign_ids=False,
            )
            _log("OK", f"Eklenildi: {config.PREPARED_JSON}")

    _write_report(stats, config.BASE_DIR / "import_report.txt")

    _banner("İNDEKS YENİLEME")
    if all_approved:
        _log("INFO", "build_index.py çalıştırılıyor (tek seferlik)...")
        # DÜZELTME: `os.system("python build_index.py")` iki nedenle
        # kırılgandı: (1) "python" komutu PATH'te olmayabilir/yanlış
        # yorumlayıcıyı (venv dışı) çalıştırabilir — `sys.executable`
        # şu an bu scripti çalıştıran TAM Python yorumlayıcısını garanti
        # eder; (2) göreli "build_index.py" yolu, script BAŞKA bir dizinden
        # çalıştırıldığında bulunamazdı — `cwd=config.BASE_DIR` ile bunu
        # da garanti altına alıyoruz.
        result = subprocess.run(
            [sys.executable, str(config.BASE_DIR / "build_index.py")],
            cwd=config.BASE_DIR,
        )
        if result.returncode == 0:
            _log("OK", "İndeks başarıyla güncellendi.")
        else:
            _log("ERR", f"build_index.py hata kodu ile çıktı: {result.returncode}")
    else:
        _log("INFO", "Yeni film eklenmediği için build_index atlandı.")

    _banner("TAMAMLANDI")
    _log("OK", f"Toplam {len(stats.approved)} film verisetine eklendi.")
    _log("INFO", f"Önerilen: {stats.suggested} | Mevcut: {stats.rejected_existing} | Reddedilen: {stats.rejected_invalid}")


if __name__ == "__main__":
    main()