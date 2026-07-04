"""Film doküman metni oluşturma — build_index ve prepare_dataset ortak kullanır."""


def build_film_document(row: dict) -> str:
    """Embedding ve BM25 kalitesi için yapılandırılmış film metni."""
    if row.get("document"):
        return row["document"]

    parts = [
        f"Film: {row['film_adi']} ({row.get('yil', '?')})",
        f"Yönetmen: {row.get('yonetmen', '')}",
        f"Oyuncular: {row.get('oyuncular', '')}",
    ]
    if row.get("tur"):
        parts.append(f"Tür: {row['tur']}")
    if row.get("ozet"):
        parts.append(f"Özet: {row['ozet']}")

    etiketler = row.get("etiketler") or []
    if etiketler:
        parts.append(f"Etiketler: {', '.join(etiketler)}")
    elif row.get("metin"):
        parts.append(row["metin"])
    elif row.get("temiz_metin"):
        parts.append(row["temiz_metin"])

    return "\n".join(p for p in parts if p and not p.endswith(": "))
