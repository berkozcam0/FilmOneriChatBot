"""
Film Öneri Chatbot — Profesyonel hibrit arama motoru.

Pipeline:
  Kullanıcı → Groq NLU → Hibrit Arama → Cross-Encoder Rerank → Groq Yanıt

Modeller (ücretsiz):
  - BAAI/bge-m3 (embedding)
  - BAAI/bge-reranker-v2-m3 (reranking)
  - BM25 (lexical)
  - Groq llama-3.3-70b-versatile (NLU + yanıt)
"""
from parser import parse_query
from engine import get_shared_engine
from responder import generate_response
from text_utils import tr_lower


class FilmChatbot:
    def __init__(self, engine=None):
        self.engine = engine or get_shared_engine()
        self.history: list[dict] = []

    def _get_parsed_val(self, parsed, key, default=None):
        """ParsedQuery nesnesinden veya dict tipinden güvenli veri okur."""
        if parsed is None:
            return default
        if isinstance(parsed, dict):
            return parsed.get(key, default)
        return getattr(parsed, key, default)

    def process(self, message: str) -> str:
        print("\n" + "="*60)
        print(f"💬 KULLANICI MESAJI: '{message}'")
        print("="*60)
        print("[1/3] 🔍 YAPAY ZEKA SORGUSUNU ANALİZ EDİYOR...")
        print("="*60)

        # Groq NLU üzerinden doğal dili yapılandırılmış şemaya çeviriyoruz
        parsed = parse_query(message, self.history)

        # Pydantic nesnesinden güvenli alan okumaları
        intent = self._get_parsed_val(parsed, "intent", "karma")
        arama_metni = self._get_parsed_val(parsed, "arama_metni", "")
        ref_film_name = self._get_parsed_val(parsed, "referans_film")
        ref_yonetmen = self._get_parsed_val(parsed, "referans_yonetmen")
        ref_yil = self._get_parsed_val(parsed, "referans_yil")
        haric_tut = self._get_parsed_val(parsed, "haric_tut") or []
        filtreler = self._get_parsed_val(parsed, "filtreler")

        # Filtre loglarını hazırlama
        # DÜZELTME: `filtreler` artık pydantic nesnesi değil, düz dict
        # (parse_query() en sonda .to_dict() döndürüyor). getattr(dict, "tur")
        # dict'in İÇİNDEKİ "tur" anahtarına bakmaz — dict nesnesinin "tur"
        # adında bir ATTRIBUTE'u olup olmadığına bakar, ki hiçbir zaman
        # olmaz. Bu yüzden bu blok filtreler gerçekte dolu olsa bile HER
        # ZAMAN "(yok)" yazıyordu — hem senin gördüğün loglar hem de gerçek
        # veri arasında bir tutarsızlık varmış gibi görünmesinin sebebi buydu.
        # Zaten sınıfın kendi _get_parsed_val() metodu hem dict hem nesne
        # tipini doğru okuyor, onu kullanıyoruz.
        aktif_filtreler = []
        if filtreler:
            f_yonetmen = self._get_parsed_val(filtreler, "yonetmen")
            f_oyuncu = self._get_parsed_val(filtreler, "oyuncu")
            f_tur = self._get_parsed_val(filtreler, "tur")
            f_yasakli = self._get_parsed_val(filtreler, "yasakli_turler")
            f_yil_min = self._get_parsed_val(filtreler, "yil_min")
            f_yil_max = self._get_parsed_val(filtreler, "yil_max")

            if f_yonetmen: aktif_filtreler.append(f"Yönetmen: {f_yonetmen}")
            if f_oyuncu: aktif_filtreler.append(f"Oyuncu: {f_oyuncu}")
            if f_tur: aktif_filtreler.append(f"Tür: {f_tur}")
            if f_yasakli:
                aktif_filtreler.append(f"Yasaklı Tür(ler): {', '.join(f_yasakli)}")
            if f_yil_min or f_yil_max:
                aktif_filtreler.append(f"Yıl Aralığı: {f_yil_min or '?'}-{f_yil_max or '?'}")

        print(f"👉 Çıkarılan Niyet (Intent) : {intent.upper()}")
        print(f"👉 Vektörel Arama Metni   : '{(arama_metni or '(boş)')[:80]}'")
        if ref_film_name:
            print(f"👉 Yakalanan Ref. Film    : '{ref_film_name}'"
                  + (f" (Tahmini Yönetmen: {ref_yonetmen}, Yıl: {ref_yil})" if (ref_yonetmen or ref_yil) else ""))
        if aktif_filtreler:
            print(f"👉 Yakalanan Filtreler    : {', '.join(aktif_filtreler)}")
        else:
            print("👉 Yakalanan Filtreler    : (yok)")
        if haric_tut:
            print(f"👉 Hariç Tutulan Filmler  : {', '.join(haric_tut)}")
        print("-" * 60)

        print("[2/3] 🚀 HİBRİT ARAMA MOTORU ÇALIŞTIRILIYOR (BM25 + BGE-M3 + Reranker)...")
        # Arama motorunu tetikle
        hits, ref_film = self.engine.search(parsed)

        if ref_film:
            print(f"👉 Referans Film Veritabanında Bulundu: '{ref_film.film_adi}' ({ref_film.yil}) — Yön: {ref_film.yonetmen}")
        elif ref_film_name:
            print(f"👉 Referans Film Veritabanında BULUNAMADI: '{ref_film_name}'")

        print(f"👉 Sonuç: Veritabanından {len(hits)} adet optimize edilmiş film getirildi.")
        if hits:
            print("   " + "-"*56)
            print(f"   {'#':<3}{'Film':<30}{'Yıl':<6}{'Sem.':<7}{'BM25':<7}{'Rerank':<8}{'Final':<7}")
            for i, h in enumerate(hits, 1):
                ad = (h.film_adi[:27] + "...") if len(h.film_adi) > 27 else h.film_adi
                print(
                    f"   {i:<3}{ad:<30}{str(h.yil or '?'):<6}"
                    f"{h.semantic_score:<7.3f}{h.bm25_score:<7.3f}"
                    f"{h.rerank_score:<8.3f}{h.score:<7.3f}"
                )
            print("   " + "-"*56)
        print("-" * 60)

        print("[3/3] ✍️ DOĞAL DİL YANITI GENERATE EDİLİYOR...")
        # Yanıtı üret
        response = generate_response(message, hits, ref_film, parsed)
        print(f"👉 Yanıt Uzunluğu: {len(response)} karakter")
        print("="*60)

        # Sohbet geçmişini güncelle
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})

        # Bellek taşmasını önlemek için yerel history kontrolü
        if len(self.history) > 40:
            self.history = self.history[-20:]

        return response

    def run(self) -> None:
        print("\n" + "*"*60)
        print("🎬 FİLM ÖNERİ SİSTEMİ TERMİNAL ARAYÜZÜ HAZIR!")
        print("*"*60)
        print("   Örnek İstem: 'Nolan tarzı beyin yakan film istiyorum'")
        print("   Örnek İstem: 'Inception izledim, benzer bir şey öner'")
        print("   Çıkış Yapmak İçin: q / quit / exit\n")

        while True:
            try:
                msg = input("Sen 👤: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGörüşürüz kral!")
                break
            if not msg:
                continue
            if tr_lower(msg) in ("q", "quit", "exit", "çık"):
                print("Görüşürüz kral!")
                break

            bot_cevap = self.process(msg)
            print(f"\nBot 🤖:\n{bot_cevap}\n")
            print("="*60 + "\n")


if __name__ == "__main__":
    FilmChatbot().run()