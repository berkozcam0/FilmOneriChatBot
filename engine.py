"""
Hibrit film arama motoru:
  1. Metadata filtreleme (yönetmen, oyuncu, yıl, tür, yasaklı türler)
  2. Dense retrieval (BGE-M3 cosine)
  3. Sparse retrieval (BM25)
  4. Reciprocal Rank Fusion
  5. Cross-encoder reranking (BGE-reranker-v2-m3)
  6. Intent bazlı sorgu ve skor stratejileri
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process
from sentence_transformers import CrossEncoder, SentenceTransformer

import config
from film_constants import ORIGIN_GENRE_TERMS
from query_schema import ParsedQuery
from text_utils import normalize_tr

# ESKİ KOD: FILM_ALIASES ve get_film_aliases() TAMAMEN SİLİNDİ!


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_tr(text), flags=re.UNICODE)


def _normalize(s: str) -> str:
    return normalize_tr(s)


@dataclass
class FilmHit:
    idx: int
    film_adi: str
    yil: Optional[int]
    yonetmen: str
    oyuncular: str
    document: str
    score: float = 0.0
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0


class FilmSearchEngine:
    def __init__(self):
        self._load_index()
        self._load_models()
        self._build_bm25()
        print(f"[OK] Arama motoru hazir - {len(self.films)} film")

    def _load_index(self) -> None:
        if not config.FILMS_JSON.exists():
            raise FileNotFoundError(
                f"İndeks bulunamadı: {config.FILMS_JSON}\n"
                "Önce çalıştırın: python build_index.py"
            )
        if not config.EMBEDDINGS_NPY.exists():
            raise FileNotFoundError(
                f"Embedding dosyası bulunamadı: {config.EMBEDDINGS_NPY}\n"
                "Önce çalıştırın: python build_index.py"
            )
        with open(config.FILMS_JSON, encoding="utf-8") as fp:
            self.films = json.load(fp)
        self.embeddings = np.load(config.EMBEDDINGS_NPY)
        self.film_names = [_normalize(f["film_adi"]) for f in self.films]

        # Alias indeksi: "alternatif_adlar" alanındaki her isim (ör. filmin
        # İngilizce orijinal adı "inception") normalize edilip ilgili filmin
        # index'ine eşleniyor. Bu olmadan, film_adi'si "Başlangıç" olarak
        # kaydedilmiş bir filmi kullanıcı "Inception" diye aratınca hiçbir
        # eşleşme stratejisi (birebir/fuzzy/belge içi arama) onu bulamıyordu.
        self.alias_to_idx: dict[str, int] = {}
        for i, f in enumerate(self.films):
            for alt in f.get("alternatif_adlar") or []:
                norm_alt = _normalize(alt)
                if norm_alt:
                    self.alias_to_idx.setdefault(norm_alt, i)

    def _load_models(self) -> None:
        print(f"[MODEL] Embedding: {config.EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        print(f"[MODEL] Reranker: {config.RERANKER_MODEL}")

        # KALİBRASYON DÜZELTMESİ: bge-reranker-v2-m3 tek-logit (num_labels=1)
        # bir modeldir. sentence-transformers'ın CrossEncoder'ı bu durumda
        # hangi aktivasyonu uygulayacağına SÜRÜME göre karar veriyor (bazı
        # sürümlerde otomatik Sigmoid, bazılarında ham logit döner). Ama
        # _apply_quality_gate() sabit bir [0,1] eşiği (RERANK_SCORE_FLOOR)
        # kullanıyor — skorun gerçekten bu aralıkta olduğu GARANTİ değilse
        # bu eşik ya hiç tetiklenmez (alakasız sonuçlar geçer) ya da hep
        # tetiklenir (iyi sonuçlar da elenir). Aktivasyonu burada elle
        # Sigmoid'e sabitleyip belirsizliği ortadan kaldırıyoruz.
        try:
            self.reranker = CrossEncoder(
                config.RERANKER_MODEL, max_length=512,
                activation_fn=torch.nn.Sigmoid(),
            )
        except TypeError:
            # Eski sentence-transformers sürümlerinde parametre adı farklıydı.
            self.reranker = CrossEncoder(
                config.RERANKER_MODEL, max_length=512,
                default_activation_function=torch.nn.Sigmoid(),
            )
        self._verify_reranker_calibration()

    def _verify_reranker_calibration(self) -> None:
        """Başlangıçta reranker skorlarının gerçekten [0,1] aralığında
        olduğunu doğrular; RERANK_SCORE_FLOOR eşiğinin anlamlı olup
        olmadığını sunucu loglarında GÖZLE görünür kılar. Bu adım
        kaldırılırsa kalibrasyon bozulduğunda sessizce yanlış sonuçlar
        üretilmeye devam eder, kimse fark etmez.
        """
        try:
            probe = self.reranker.predict([
                ("aksiyon dolu bir film", "Bu film yogun aksiyon sahneleri iceren bir gerilim yapimidir."),
                ("aksiyon dolu bir film", "Kirtasiye malzemelerinin tarihi hakkinda bir belgesel."),
            ])
            lo, hi = float(min(probe)), float(max(probe))
            if lo < -0.01 or hi > 1.01:
                print(
                    f"[UYARI] Reranker skorlari [0,1] araliginin DISINDA "
                    f"({lo:.3f} - {hi:.3f}). config.RERANK_SCORE_FLOOR "
                    f"(={config.RERANK_SCORE_FLOOR}) kalibre DEGIL, quality "
                    f"gate hatali calisabilir — activation_fn ayarini kontrol edin."
                )
            else:
                print(f"[OK] Reranker skor kalibrasyonu dogrulandi: {lo:.3f} - {hi:.3f} (beklenen: [0,1])")
        except Exception as e:
            print(f"[UYARI] Reranker kalibrasyon kontrolu calistirilamadi: {e}")

    def _build_bm25(self) -> None:
        corpus = [_tokenize(f["document"]) for f in self.films]
        self.bm25 = BM25Okapi(corpus)

    def _encode_query(self, text: str) -> np.ndarray:
        prefixed = config.BGE_QUERY_PREFIX + text
        return self.embedder.encode(
            [prefixed], normalize_embeddings=True, show_progress_bar=False
        )[0]

    def find_film_by_name(self, name: str, threshold: int = 82) -> Optional[int]:
        if not name:
            return None
        query = _normalize(name)

        # 1. Birebir Eşleşme Kontrolü (asıl film adı)
        for i, fn in enumerate(self.film_names):
            if fn == query:
                return i

        # 2. Birebir Eşleşme Kontrolü (alternatif adlar — ör. İngilizce orijinal ad)
        if query in self.alias_to_idx:
            return self.alias_to_idx[query]

        # 3. Bulanık Mantık (Fuzzy Matching) — hem asıl adlar hem alternatif
        # adlar havuzunda arar, böylece küçük yazım hatalarını da tolere eder.
        pool_names = list(self.film_names) + list(self.alias_to_idx.keys())
        pool_idx = list(range(len(self.films))) + [
            self.alias_to_idx[n] for n in self.alias_to_idx
        ]
        result = process.extractOne(
            query, pool_names, scorer=fuzz.WRatio, score_cutoff=threshold
        )
        if result:
            _, _, pos = result
            return pool_idx[pos]

        # 4. Belge İçinde Kaba Arama
        # NOT: `query` zaten _normalize() ile normalize edilmiş durumda (yukarıda).
        # Eskiden burada dokümanla karşılaştırma ham `.lower()` ile yapılıyordu;
        # bu da text_utils.py'nin tam olarak çözmeye çalıştığı hataydı ("İ" gibi
        # Türkçe büyük harfler .lower() ile combining-character üretip normalize
        # edilmiş sorguyla ASLA eşleşmiyordu). Karşılaştırmanın iki tarafı da
        # aynı _normalize() fonksiyonundan geçmeli.
        for i, f in enumerate(self.films):
            if query in _normalize(f["document"][:200]):
                return i
        return None

    def find_film_by_director_year(
        self, yonetmen: Optional[str], yil: Optional[int]
    ) -> Optional[int]:
        """Yönetmen + yıl kombinasyonuyla film bulur.

        Bu, isim tabanlı eşleşmenin (find_film_by_name) elle bakım gerektiren
        bir alias listesine bağımlı olmasının çözümü: parser.py, LLM'in
        zaten sahip olduğu film dünyası bilgisini kullanarak referans filmin
        yönetmenini/yılını çıkarıyor (ör. "Inception" -> Christopher Nolan,
        2010). Veri setinde film "Başlangıç" gibi tamamen farklı bir başlık
        altında kayıtlı olsa bile, yönetmen+yıl kombinasyonu neredeyse her
        zaman benzersizdir ve elle alias girmeye gerek kalmadan doğru filmi
        bulur.
        """
        if not yonetmen or not yil:
            return None
        target_dir = _normalize(yonetmen)

        candidates = [
            i
            for i, f in enumerate(self.films)
            if f.get("yil") == yil
            and fuzz.partial_ratio(target_dir, _normalize(f.get("yonetmen", ""))) >= 85
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # Birden fazla aday varsa (aynı yönetmenin aynı yıl çıkan başka bir
        # filmi gibi nadir bir durum), en güçlü yönetmen eşleşmesini seç.
        return max(
            candidates,
            key=lambda i: fuzz.ratio(target_dir, _normalize(self.films[i].get("yonetmen", ""))),
        )

    def _genre_match_pool(self, idx: int) -> str:
        """Tür/etiket eşleştirmesi için kullanılan birleşik, normalize
        edilmiş metin havuzu: tür alanı + etiketler + belgenin TAMAMI.

        DÜZELTME: Eskiden hem filtreleme (_apply_metadata_filter) hem de
        puanlama (_apply_intent_boosts → GENRE_BOOST) SADECE "tur" alanına
        (kısa özet metninden çıkarılan 1-3 tür) bakıyordu ya da (filtrede)
        en fazla "tur" + "etiketler"e bakıyordu — üstelik ikisi arasında
        TUTARSIZLIK vardı: filtre etiketlere bakıyordu, boost hiç
        bakmıyordu. Bu yüzden bir film sırf etiketiyle (ör. "türk filmi")
        filtreyi geçse bile sıralamada hak ettiği puanı hiç almıyordu.
        Ayrıca "türk filmi" gibi bazı filmlerde ayrı bir ETİKET olarak
        girilmemiş, sadece özet metninde ("türk taşrasında" gibi) geçen
        ifadeler hiç yakalanamıyordu.

        Artık HEM filtre HEM boost aynı, daha geniş havuza bakıyor: tür +
        etiketler + document'ın tamamı (document zaten "Tür:" ve
        "Etiketler:" satırlarını da içeriyor, ayrıca özet metnindeki
        ayrık-etiket-olmayan ifadeleri de yakalıyor).
        """
        f = self.films[idx]
        tur = f.get("tur", "") or ""
        etiketler = " ".join(f.get("etiketler") or [])
        document = f.get("document", "") or ""
        return _normalize(f"{tur} {etiketler} {document}")

    def _apply_metadata_filter(
        self, indices: list[int], filters, *, enforce_tur: bool = True
    ) -> list[int]:
        filtered = []
        for i in indices:
            f = self.films[i]
            film_tur = _normalize(f.get("tur", ""))
            film_genres_set = {g.strip() for g in film_tur.split(",") if g.strip()}
            # "etiketler" alanı da tür/tema bilgisi taşıyor (bkz.
            # document_builder / prepare_dataset._extract_tags). Bu havuz
            # SADECE yasaklı-tür (dışlama) kontrolü için kullanılıyor;
            # kasıtlı olarak SIKI/precise tutuluyor (document'ın tamamını
            # DAHİL ETMİYORUZ), çünkü bir kelimenin özet metninde geçmesi
            # filmin o türde OLMADIĞI anlamına gelebilir (ör. "korku filmi
            # değil, daha çok dram" gibi bir cümle yanlışlıkla filmi
            # elemesin). Pozitif eşleşme (aşağıda, tur filtresi) için ise
            # havuz kasıtlı olarak daha geniş (_genre_match_pool) tutuluyor.
            film_tags_set = {
                _normalize(t) for t in (f.get("etiketler") or []) if t
            }
            film_genre_pool = film_genres_set | film_tags_set

            # Yasaklı tür kontrolü — hem "tur" hem "etiketler" alanına bakar
            if any(_normalize(y) in film_genre_pool for y in (filters.yasakli_turler or [])):
                continue

            # YÖNETMEN KONTROLÜ
            if filters.yonetmen:
                req_yonetmen = _normalize(filters.yonetmen)
                film_yonetmen = _normalize(f.get("yonetmen", ""))
                if req_yonetmen not in film_yonetmen and film_yonetmen not in req_yonetmen:
                    if fuzz.partial_ratio(req_yonetmen, film_yonetmen) < 70:
                        continue

            # OYUNCU KONTROLÜ
            if filters.oyuncu:
                req_oyuncu = _normalize(filters.oyuncu)
                film_oyuncu = _normalize(f.get("oyuncular", ""))
                if req_oyuncu not in film_oyuncu and film_oyuncu not in req_oyuncu:
                    if fuzz.partial_ratio(req_oyuncu, film_oyuncu) < 70:
                        continue

            # TÜR KONTROLÜ — enforce_tur=False iken atlanır (bkz. search():
            # sert filtre aday sayısını çok daraltırsa tür kısıtı gevşetilip
            # sıralamaya (GENRE_BOOST) bırakılır; bu genel, sayıya dayalı
            # bir mekanizmadır, belirli bir türe özel değildir).
            if enforce_tur and filters.tur:
                req_genres = {
                    g.strip() for g in _normalize(filters.tur).split(",") if g.strip()
                }
                if req_genres:
                    # DÜZELTME: Artık sadece tur/etiketler set'i değil,
                    # _genre_match_pool() (tur+etiketler+document'ın TAMAMI)
                    # içinde aranıyor — bkz. _genre_match_pool docstring'i.
                    match_pool_text = self._genre_match_pool(i)

                    # DÜZELTME (köken vs tür KARIŞTIRMA HATASI): Eskiden
                    # tüm virgüllü öğeler tek bir any() (VEYA) havuzunda
                    # eşleştiriliyordu. Bu, "türk filmi, aksiyon" gibi bir
                    # istekte kökeni (türk) tür ile (aksiyon) birbirinin
                    # ALTERNATİFİ hâline getiriyordu — sistem "ya türk ya
                    # aksiyon" arıyordu, "hem türk hem aksiyon" değil.
                    # Sonuç: Hero, Kung Fu Hustle, Train to Busan gibi
                    # Türk olmayan ama aksiyon türündeki filmler sırf
                    # "aksiyon" eşleşmesiyle filtreyi geçiyordu.
                    #
                    # Köken (ORIGIN_GENRE_TERMS içindeki "türk filmi" gibi
                    # ifadeler) ile tür FARKLI eksenlerdir ve aralarında VE
                    # (AND) mantığı gerekir; aynı eksendeki birden fazla tür
                    # ("aksiyon, komedi") ise VEYA (OR) kalmalıdır — film
                    # ikisinden birini içeriyorsa yeterlidir.
                    origin_reqs = req_genres & ORIGIN_GENRE_TERMS
                    genre_reqs = req_genres - ORIGIN_GENRE_TERMS

                    origin_ok = (
                        all(o in match_pool_text for o in origin_reqs)
                        if origin_reqs else True
                    )
                    genre_ok = (
                        any(g in match_pool_text for g in genre_reqs)
                        if genre_reqs else True
                    )
                    if not (origin_ok and genre_ok):
                        continue

            yil = f.get("yil")
            if filters.yil_min is not None and yil and yil < filters.yil_min:
                continue
            if filters.yil_max is not None and yil and yil > filters.yil_max:
                continue

            filtered.append(i)
        return filtered

    @staticmethod
    def _apply_quality_gate(hits: list[FilmHit]) -> list[FilmHit]:
        """Zayıf/alakasız adayları hem mutlak taban hem Göreli Boşluk
        Tespiti (Gap Detection) ile eler.

        DÜZELTME: Eskiden mutlak taban (RERANK_SCORE_FLOOR) SADECE en iyi
        adaya (hits[0]) uygulanıyordu. `min_results=2` zorlaması yüzünden,
        ilk adaydan sonra skor %90+ düşse bile en az 2 sonuç listeye
        zorla ekleniyordu; sonraki adaylar da zaten dipte olduğu için
        ARALARINDAKİ göreli fark küçük kalıyor ve gap-detection hiç
        tetiklenmiyordu (ör. 0.011 -> 0.0095 -> 0.0076 gibi skorlar hepsi
        çöp ama birbirine göre "büyük düşüş" göstermiyor). Sonuç: alakasız
        filmler sırf sayı doldurmak için listeye giriyordu.

        Artık taban filtresi TÜM adaylara uygulanıyor (sadece ilkine değil)
        ve min_results zorlaması tamamen kaldırıldı — ilk gerçek %50+
        göreli düşüşte liste hemen kesiliyor. Çıktı sayısı artık tamamen
        dinamik (1 ile TOP_K_FINAL arası); doğruluğu düşük adaylar sayı
        doldurmak için asla eklenmiyor.
        """
        if not hits:
            return hits

        hits.sort(key=lambda h: h.rerank_score, reverse=True)

        # Eğer en iyi adayın skoru bile çok düşükse, sistem tamamen
        # alakasız şeyler getiriyordur. Direkt boş dön. Eşik
        # config.RERANK_SCORE_FLOOR'dan geliyor ve yalnızca skorun [0,1]
        # aralığında olduğu DOĞRULANDIĞINDA (bkz.
        # _verify_reranker_calibration) anlamlıdır.
        if hits[0].rerank_score < config.RERANK_SCORE_FLOOR:
            print(f"[GAP-DETECTION] 🛑 En iyi skor çok düşük ({hits[0].rerank_score:.4f}). Tüm havuz elendi.")
            return []

        # YENİ: mutlak taban artık TÜM adaylara uygulanıyor, sadece ilkine
        # değil. Bir aday RERANK_SCORE_FLOOR altındaysa, sırası ne olursa
        # olsun (komşusuna göre "düşüş" küçük görünse bile) listeye giremez.
        hits = [h for h in hits if h.rerank_score >= config.RERANK_SCORE_FLOOR]
        if not hits:
            return []

        if len(hits) == 1:
            return hits

        optimized_list = [hits[0]]
        print(f"\n[GAP-DETECTION] En iyi skor: {hits[0].rerank_score:.4f} ({hits[0].film_adi})")

        for i in range(1, len(hits)):
            current_score = hits[i].rerank_score
            prev_score = hits[i - 1].rerank_score

            # Göreli düşüş oranını hesapla
            # (Sıfıra veya negatife bölünmeyi önlemek için payda max(prev_score, 0.01) olarak sınırlandı)
            drop_rate = (prev_score - current_score) / max(prev_score, 0.01)

            print(f" -> Aday #{i + 1}: {hits[i].film_adi} | Skor: {current_score:.4f} | Düşüş: %{drop_rate * 100:.1f}")

            # DEĞİŞTİ: min_results zorlaması kaldırıldı — ilk gerçek %50+
            # göreli düşüşte listeyi HEMEN kes (kaç tane biriktiği önemsiz).
            if drop_rate > 0.50:
                print(f"[GAP-DETECTION] 🛑 Kesme noktası (Uçurum) tespit edildi! %{drop_rate * 100:.1f} düşüş.")
                break

            optimized_list.append(hits[i])

        return optimized_list

    def _semantic_search(
        self, query_vec: np.ndarray, n: int, restrict: Optional[list[int]] = None
    ) -> list[tuple[int, float]]:
        if restrict is not None:
            if not restrict:
                return []
            restrict_arr = np.array(restrict)
            scores = self.embeddings[restrict_arr] @ query_vec
            order = np.argsort(scores)[::-1][:n]
            return [(int(restrict_arr[i]), float(scores[i])) for i in order]

        scores = self.embeddings @ query_vec
        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_idx]

    def _bm25_search(
        self, query: str, n: int, restrict: Optional[list[int]] = None
    ) -> list[tuple[int, float]]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        if restrict is not None:
            if not restrict:
                return []
            idxs = [i for i in restrict if scores[i] > 0]
            idxs.sort(key=lambda i: scores[i], reverse=True)
            return [(i, float(scores[i])) for i in idxs[:n]]

        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    @staticmethod
    def _rrf_merge(
        *ranked_lists: list[tuple[int, float]], k: int = config.RRF_K
    ) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (idx, _) in enumerate(ranked):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    def _rerank_doc(self, idx: int) -> str:
        doc = self.films[idx].get("document", self.films[idx].get("ozet", ""))
        return doc[: config.RERANK_DOC_MAX_CHARS]

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[int, float]],
        top_k: int,
        sem_map: dict[int, float],
        bm25_map: dict[int, float],
    ) -> list[FilmHit]:
        if not candidates:
            return []

        top = candidates[: config.TOP_K_RERANK]
        pairs = [(query, self._rerank_doc(idx)) for idx, _ in top]
        indices = [idx for idx, _ in top]
        rerank_scores = self.reranker.predict(pairs)

        hits = []
        for idx, rr_score in zip(indices, rerank_scores):
            f = self.films[idx]
            hits.append(
                FilmHit(
                    idx=idx,
                    film_adi=f["film_adi"],
                    yil=f.get("yil"),
                    yonetmen=f.get("yonetmen", ""),
                    oyuncular=f.get("oyuncular", ""),
                    document=f["document"],
                    semantic_score=sem_map.get(idx, 0.0),
                    bm25_score=bm25_map.get(idx, 0.0),
                    rerank_score=float(rr_score),
                    score=float(rr_score),
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    @staticmethod
    def _dedupe_hits(hits: list[FilmHit]) -> list[FilmHit]:
        buckets: dict[tuple, list[FilmHit]] = {}
        unique = []
        for h in hits:
            key = (h.yil, _normalize(h.yonetmen))
            if not key[0] or not key[1]:
                unique.append(h)
                continue
            bucket = buckets.setdefault(key, [])
            is_dup = any(
                fuzz.ratio(_normalize(h.film_adi), _normalize(other.film_adi)) > 80
                for other in bucket
            )
            if is_dup:
                continue
            bucket.append(h)
            unique.append(h)
        return unique

    def _build_search_queries(
        self,
        query: ParsedQuery,
        ref_idx: Optional[int],
    ) -> tuple[np.ndarray, str, str]:
        """Sorgu metinlerini oluşturur.

        ÖNEMLİ: Bu fonksiyon artık `intent` string'ine göre DEĞİL, hangi
        filtrelerin GERÇEKTEN dolu olduğuna göre çalışır. Çünkü parser.py
        artık çoğu çok-filtreli sorguyu "karma" olarak işaretliyor
        (bkz. parser._validate_intent). Eskiden burada `intent == "yonetmen"`
        gibi kontroller vardı; "karma" geldiğinde hiçbiri tetiklenmiyor ve
        yönetmen/oyuncu/tür bilgisi sorgu metnine hiç yansımıyordu. Filtre
        bazlı kontrol sayesinde "Nolan'ın bilim kurgu filmleri" gibi
        çok-filtreli (karma) sorgularda da doğru zenginleştirme oluyor.
        """
        arama = query.arama_metni or ""
        filters = query.filtreler

        query_vec = self._encode_query(arama)

        bm25_parts: list[str] = []
        rerank_parts: list[str] = []

        if filters.yonetmen:
            bm25_parts.append(filters.yonetmen)
            rerank_parts.append(f"{filters.yonetmen} yönetmen tarzı film")
        if filters.oyuncu:
            bm25_parts.append(filters.oyuncu)
            # DÜZELTME (TOKEN UYUŞMAZLIĞI): Eskiden "{oyuncu} oyuncu
            # performansı" kalıbı kullanılıyordu. Bu ifade dokümandaki
            # "Oyuncular: ..." alanıyla token düzeyinde ZAYIF örtüşüyordu
            # ("oyuncu" tekil vs "Oyuncular" çoğul, "performansı" kelimesi
            # dokümanda hiç geçmiyor — doküman bir eleştiri değil, konu
            # özeti). Cross-encoder reranker anlamsal olsa da doğrudan
            # kelime örtüşmesi hâlâ güçlü bir sinyal; yönetmen tarafında
            # "{yönetmen} yönetmen tarzı film" kalıbı dokümandaki
            # "Yönetmen:" etiketiyle birebir örtüştüğü için yüksek skor
            # alıyordu, oyuncuda bu örtüşme yoktu. Artık dokümandaki
            # gerçek alan adına ("oynadığı filmler" ifadesi "oyuncu"
            # kelimesini ve doğal "X'in oynadığı" bağlamını taşıyor)
            # daha yakın bir kalıp kullanılıyor; bu, isim + rol bağlamını
            # dokümanın "Oyuncular:" satırına daha güçlü bağlıyor.
            rerank_parts.append(f"{filters.oyuncu} oynadığı filmler oyuncu kadrosu")
        if filters.tur:
            bm25_parts.append(filters.tur)
            rerank_parts.append(f"{filters.tur} türünde film")

        if arama:
            bm25_parts.append(arama)
            # Hiç metadata filtresi yoksa (saf ruh hali sorgusu) atmosfer
            # vurgusu ekle; filtre varsa zaten yönetmen/tür ifadesiyle
            # birlikte yeterince bağlamlı oluyor.
            if not (filters.yonetmen or filters.oyuncu or filters.tur):
                rerank_parts.append(f"atmosfer ruh hali tema {arama}")
            else:
                rerank_parts.append(arama)

        bm25_query = " ".join(bm25_parts) if bm25_parts else arama
        rerank_query = " ".join(rerank_parts) if rerank_parts else arama

        if ref_idx is not None:
            # Referans film bulunduysa, intent ne olursa olsun (karma dahil)
            # embedding'i referansa doğru ağırlıklandır. Eskiden bu sadece
            # intent tam olarak "benzer_film" ise çalışıyordu; ama kullanıcı
            # "Tarantino tarzı, Pulp Fiction gibi" derse intent artık "karma"
            # oluyor ve referans film vektöre hiç katkı yapmıyordu.
            ref_vec = self.embeddings[ref_idx]
            w_u, w_r = config.BENZER_FILM_USER_WEIGHT, config.BENZER_FILM_REF_WEIGHT
            blended = w_u * query_vec + w_r * ref_vec
            norm = np.linalg.norm(blended)
            if norm > 0:
                blended = blended / norm
            query_vec = blended

            # Kullanıcı serbest metin yazmadıysa (sadece "X gibi film" dediyse)
            # BM25/rerank metnini referans filmin dokümanından besle.
            if not arama:
                ref_doc = self.films[ref_idx]["document"]
                bm25_query = ref_doc[:400]
                rerank_query = ref_doc[:500]

        return query_vec, bm25_query, rerank_query

    def _apply_intent_boosts(self, hits: list[FilmHit], query: ParsedQuery) -> None:
        """Filtre bazlı skor artışı uygular.

        Eskiden bu boost'lar `intent == "yonetmen"` gibi TEK bir intent'e
        bağlıydı ve birbirini dışlıyordu (if/elif). Artık çoğu sorgu "karma"
        geldiği için (1'den fazla filtre çıkarıldığında parser bunu zorunlu
        kılıyor) hiçbir boost tetiklenmiyordu. Şimdi her filtre kendi
        başına, birbirinden bağımsız kontrol ediliyor — yönetmen + tür gibi
        birden fazla filtre aynı anda varsa boost'lar üst üste binebiliyor.
        """
        filters = query.filtreler

        if filters.yonetmen:
            target = _normalize(filters.yonetmen)
            for hit in hits:
                if target in _normalize(hit.yonetmen):
                    hit.score += config.DIRECTOR_BOOST

        if filters.oyuncu:
            target = _normalize(filters.oyuncu)
            for hit in hits:
                if target in _normalize(hit.oyuncular):
                    hit.score += config.ACTOR_BOOST

        if filters.tur:
            # DÜZELTME: Eskiden burada SADECE "tur" alanına bakılıyordu,
            # "etiketler" ve document'a hiç bakılmıyordu — bu,
            # _apply_metadata_filter()'daki (daha geniş) kontrolle
            # TUTARSIZDI. Sonuç: bir film sırf "türk filmi" gibi bir
            # ETİKETLE filtreyi geçse bile, burada boost hiç
            # tetiklenmediği için sıralamada hak ettiği yeri alamıyordu.
            # Artık ikisi de aynı havuza (_genre_match_pool) bakıyor.
            # DÜZELTME: Aynı köken-vs-tür VEYA/VE karışıklığı burada da
            # vardı (bkz. _apply_metadata_filter'daki açıklama) — bir film
            # sırf "aksiyon" eşleşmesiyle GENRE_BOOST alabiliyordu, "türk
            # filmi" eşleşmese bile. Aynı ayrım (köken=VE, tür=VEYA) burada
            # da uygulanıyor; boost sadece HER İKİ koşul da sağlandığında
            # verilir.
            req = {_normalize(g.strip()) for g in filters.tur.split(",") if g.strip()}
            origin_reqs = req & ORIGIN_GENRE_TERMS
            genre_reqs = req - ORIGIN_GENRE_TERMS
            for hit in hits:
                match_pool_text = self._genre_match_pool(hit.idx)
                origin_ok = (
                    all(o in match_pool_text for o in origin_reqs)
                    if origin_reqs else True
                )
                genre_ok = (
                    any(g in match_pool_text for g in genre_reqs)
                    if genre_reqs else True
                )
                if origin_ok and genre_ok:
                    hit.score += config.GENRE_BOOST

    def search(
        self, parsed: dict, top_k: Optional[int] = None
    ) -> tuple[list[FilmHit], Optional[FilmHit]]:
        if top_k is None:
            top_k = config.TOP_K_FINAL

        query = ParsedQuery.from_dict(parsed)
        filters = query.filtreler
        haric = {_normalize(x) for x in query.haric_tut if x}
        ref_film_hit: Optional[FilmHit] = None
        ref_idx: Optional[int] = None

        if query.referans_film:
            ref_idx = self.find_film_by_name(query.referans_film)

            # İsimle bulunamadıysa, LLM'in dünya bilgisinden gelen
            # yönetmen+yıl ipucuyla dene (manuel alias listesine bağımlı
            # olmayan otomatik eşleştirme — bkz. find_film_by_director_year).
            if ref_idx is None:
                ref_idx = self.find_film_by_director_year(
                    query.referans_yonetmen, query.referans_yil
                )

            if ref_idx is not None:
                ref = self.films[ref_idx]
                ref_film_hit = FilmHit(
                    idx=ref_idx,
                    film_adi=ref["film_adi"],
                    yil=ref.get("yil"),
                    yonetmen=ref.get("yonetmen", ""),
                    oyuncular=ref.get("oyuncular", ""),
                    document=ref["document"],
                )
                haric.add(_normalize(ref["film_adi"]))
                ref_yonetmen = _normalize(ref.get("yonetmen", ""))
                ref_yil = ref.get("yil")
                for i, f in enumerate(self.films):
                    if (
                        i != ref_idx
                        and f.get("yil") == ref_yil
                        and ref_yil is not None
                        and _normalize(f.get("yonetmen", "")) == ref_yonetmen
                    ):
                        haric.add(_normalize(f["film_adi"]))
            else:
                arama_ek = query.arama_metni or ""
                query.arama_metni = f"{query.referans_film} {arama_ek}".strip()
                query.intent = "karma"

        query_vec, bm25_query, rerank_query = self._build_search_queries(query, ref_idx)

        # DÜZELTME: BOŞ SORGU KORUMASI
        # ------------------------------------------------------------
        # Eğer arama_metni boş VE hiçbir metadata filtresi (yönetmen/
        # oyuncu/tür/yıl) VE hiçbir referans film yoksa, elimizde
        # kullanıcının GERÇEKTEN ne istediğine dair hiçbir sinyal yok
        # demektir (ör. Groq NLU "duygusal filmler arıyorum" gibi geçerli
        # bir mood ifadesini bile çöp kelime sanıp arama_metni'ni boş
        # bıraktığında olduğu gibi — bkz. sohbet geçmişindeki örnek).
        #
        # Bu durumda eskiden BM25/rerank sorguları "" (boş string) ile
        # çalıştırılıyordu. Cross-encoder reranker boş bir sorgu aldığında
        # sorguya göre DEĞİL, dokümanın kendi "önsel" (prior) skoruna
        # yakın bir şey döndürüyor — bu da sonuçla hiçbir ilgisi olmayan
        # filmlerin (Withnail and I, Marcel the Shell, The Equalizer 2...)
        # 0.95-0.99 gibi YÜKSEK ve YANILTICI güvenle döndürülmesine yol
        # açıyordu. Skorlar teknik olarak yüksek olduğu için gap-detection
        # da bunu YAKALAYAMIYORDU (bkz. _apply_quality_gate — o sadece
        # göreli düşüşe/mutlak tabana bakar, sorgunun anlamlı olup
        # olmadığına bakmaz).
        #
        # Artık böyle "efektif olarak boş" bir sorguda arama hiç
        # çalıştırılmıyor; boş sonuç + reason bayrağıyla dönülüyor. Bu
        # sayede chatbot/responder katmanı "alakasız ama kendinden emin"
        # bir öneri yerine kullanıcıdan netleştirme isteyebilir.
        has_query_signal = bool((query.arama_metni or "").strip())
        has_filter_signal = query.has_metadata_filter()
        has_ref_signal = ref_idx is not None or bool(query.referans_film)

        if not (has_query_signal or has_filter_signal or has_ref_signal):
            print(
                "[UYARI] Sorgu efektif olarak BOŞ (arama_metni yok, filtre "
                "yok, referans film yok) — arama ÇALIŞTIRILMADI, sahte-"
                "güvenli sonuç üretimi engellendi."
            )
            return [], ref_film_hit

        restrict_indices: Optional[list[int]] = None
        if query.has_metadata_filter():
            restrict_indices = self._apply_metadata_filter(
                list(range(len(self.films))), filters, enforce_tur=True
            )
            # DÜZELTME: Tür etiketleme kısa özet metninden çıkarıldığı için
            # eksik kalabiliyor (bkz. config.MIN_RESULTS_FOR_STRICT_GENRE
            # açıklaması). Sert tür filtresi aday havuzunu SAYI olarak çok
            # daraltıyorsa (belirli bir türe özel değil, genel bir kural),
            # tür kısıtını gevşetip reranker + GENRE_BOOST'a bırakıyoruz.
            if filters.tur and len(restrict_indices) < config.MIN_RESULTS_FOR_STRICT_GENRE:
                restrict_indices = self._apply_metadata_filter(
                    list(range(len(self.films))), filters, enforce_tur=False
                )
            if not restrict_indices:
                return [], ref_film_hit

        semantic = self._semantic_search(
            query_vec, config.TOP_K_SEMANTIC, restrict=restrict_indices
        )
        bm25 = self._bm25_search(
            bm25_query, config.TOP_K_BM25, restrict=restrict_indices
        )
        sem_map = dict(semantic)
        bm25_map = dict(bm25)
        merged = self._rrf_merge(semantic, bm25)

        candidates = [
            (idx, score)
            for idx, score in merged
            if _normalize(self.films[idx]["film_adi"]) not in haric
        ]

        results = self._rerank(
            rerank_query, candidates, config.TOP_K_RERANK, sem_map, bm25_map
        )
        results = self._dedupe_hits(results)
        self._apply_intent_boosts(results, query)

        # DÜZELTME (OYUNCU/YÖNETMEN İSİM FİLTRESİ vs. RERANKER SKORU
        # ÇAKIŞMASI): _apply_quality_gate reranker'ın (cross-encoder)
        # ANLAMSAL benzerlik skoruna bakar — bu skor "sorgu metniyle
        # dokümanın konusu/teması ne kadar örtüşüyor" sorusuna cevap
        # verir, "bu isim dokümanda geçiyor mu" sorusuna DEĞİL.
        #
        # Yönetmen/oyuncu filtresi varken rerank_query kısa ve jenerik
        # olur (ör. "Tom Hanks oyuncu performansı"); bu ifade filmin
        # ~450 karakterlik özet/tema metniyle (savaş, aşk, dram vb.)
        # anlamsal olarak ZAYIF örtüşür — isim listede birebir geçse
        # bile. Sonuç: reranker skoru 0.02-0.19 gibi çok düşük çıkıyor,
        # RERANK_SCORE_FLOOR (0.20) bunu her seferinde eliyor ve
        # gerçekten o oyuncunun/yönetmenin oynadığı/çektiği filmler
        # "hiç bulunamadı" olarak dönüyordu (bkz. loglar: Tom Hanks,
        # Leonardo DiCaprio, Cem Yılmaz vb. — hepsi veri setinde
        # mevcutken 0 sonuç dönüyordu).
        #
        # Oysa bu adaylar zaten _apply_metadata_filter() tarafından
        # KESİN (exact/fuzzy substring) olarak doğrulanmış durumda —
        # reranker'ın anlamsal şüpheciliğine ihtiyaç yok. Bu yüzden:
        # sorgu SADECE yönetmen/oyuncu filtresine dayanıyorsa (tür,
        # yasaklı tür, arama_metni gibi başka bir sinyal yoksa) quality
        # gate'i atlıyoruz — reranker skoru yine SIRALAMA için
        # kullanılıyor (hit.score/rerank_score), sadece mutlak/göreli
        # eleme uygulanmıyor. Yönetmen+tür veya oyuncu+arama_metni gibi
        # karma sorgularda (başka bir anlamsal sinyal de olduğunda)
        # quality gate normal şekilde çalışmaya devam ediyor, çünkü o
        # durumda reranker skoru gerçekten ek bilgi taşıyor.
        is_pure_name_filter = (
            (bool(filters.yonetmen) or bool(filters.oyuncu))
            and not filters.tur
            and not (query.arama_metni or "").strip()
        )

        if is_pure_name_filter:
            print(
                "[BİLGİ] Saf yönetmen/oyuncu filtresi tespit edildi — "
                "reranker mutlak/göreli kalite eşiği (RERANK_SCORE_FLOOR) "
                "atlanıyor, çünkü alaka zaten metadata filtresiyle kesin "
                "olarak doğrulandı. Reranker skoru yalnızca sıralama için "
                "kullanılıyor."
            )
        else:
            # DÜZELTME: Eskiden bu kalite kontrolü SADECE metadata filtresi
            # varken çalışıyordu (has_metadata_filter()) — filtresiz ruh_hali /
            # benzer_film sorgularında (ör. "zihin bükücü film öner") hiçbir
            # kalite denetimi yoktu. Artık her sorguda çalışıyor ve eşik sabit
            # değil, dinamik (bkz. _apply_quality_gate).
            results = self._apply_quality_gate(results)

        results.sort(key=lambda h: h.score, reverse=True)
        return results[:top_k], ref_film_hit


# ── Paylaşımlı motor örneği (Flask oturumları için) ─────────────────────────
_shared_engine: Optional[FilmSearchEngine] = None


def get_shared_engine() -> FilmSearchEngine:
    global _shared_engine
    if _shared_engine is None:
        _shared_engine = FilmSearchEngine()
    return _shared_engine