"""Yapılandırılmış sorgu şeması — parser, engine ve responder ortak kullanır."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from film_constants import VALID_INTENTS


class QueryFilters(BaseModel):
    yonetmen: Optional[str] = None
    oyuncu: Optional[str] = None
    tur: Optional[str] = None
    yasakli_turler: list[str] = Field(default_factory=list)
    yil_min: Optional[int] = None
    yil_max: Optional[int] = None

    @field_validator("yil_min", "yil_max", mode="before")
    @classmethod
    def _coerce_year(cls, v: Any) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None


class ParsedQuery(BaseModel):
    intent: str = "karma"
    arama_metni: str = ""
    referans_film: Optional[str] = None
    # LLM, referans_film'i kendi dünya bilgisinden tanıyorsa (ör. "Inception"
    # -> Christopher Nolan, 2010) bu iki alanı da dolduruyor. Veri setinde
    # film farklı bir başlık altında kayıtlıysa (ör. "Başlangıç") bile
    # engine.py bu yönetmen+yıl kombinasyonuyla doğru filmi bulabiliyor —
    # elle alias listesi tutmaya gerek kalmadan.
    referans_yonetmen: Optional[str] = None
    referans_yil: Optional[int] = None
    haric_tut: list[str] = Field(default_factory=list)
    filtreler: QueryFilters = Field(default_factory=QueryFilters)

    @field_validator("intent", mode="before")
    @classmethod
    def _validate_intent(cls, v: Any) -> str:
        intent = (v or "karma").strip().lower()
        return intent if intent in VALID_INTENTS else "karma"

    @field_validator("referans_yil", mode="before")
    @classmethod
    def _coerce_ref_year(cls, v: Any) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    @field_validator("haric_tut", mode="before")
    @classmethod
    def _coerce_haric(cls, v: Any) -> list[str]:
        if not v:
            return []
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v if x]

    def has_metadata_filter(self) -> bool:
        f = self.filtreler
        return bool(
            f.yonetmen
            or f.oyuncu
            or f.tur
            or f.yasakli_turler
            or f.yil_min is not None
            or f.yil_max is not None
        )

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict | None) -> "ParsedQuery":
        if not data:
            return cls()
        filters = data.get("filtreler") or {}
        return cls(
            intent=data.get("intent", "karma"),
            arama_metni=(data.get("arama_metni") or "").strip(),
            referans_film=data.get("referans_film"),
            referans_yonetmen=data.get("referans_yonetmen"),
            referans_yil=data.get("referans_yil"),
            haric_tut=data.get("haric_tut") or [],
            filtreler=QueryFilters(**filters),
        )