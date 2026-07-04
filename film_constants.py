"""Film türleri ve niyet sabitleri — sadece veri hazırlama (prepare_dataset) ve şema onayı için kullanılır."""

GENRE_KEYWORDS: dict[str, list[str]] = {
    # "canavar" çıkarıldı: "canavar gibi bir performans" gibi tür dışı
    # oyunculuk övgülerinde sık geçiyor.
    "korku": ["korku", "horror", "ürküt", "vampir", "hayalet", "slasher",
              "zombi", "şeytan çıkarma", "lanetli ev"],

    # "geren" çıkarıldı: bağlamdan kopuk, aşırı zayıf bir kök eşleşmesiydi.
    "gerilim": ["gerilim", "thriller", "psikolojik gerilim", "suspense",
                "gerilim dolu", "komplo", "casusluk"],

    # "macera" ve "tempolu" çıkarıldı: "macera" tür dışı bağlamlarda da
    # ("hayat macerası", "aşk macerası") geçiyor; "tempolu" ise herhangi
    # bir türde kullanılabilecek genel bir tempo sıfatı.
    "aksiyon": ["aksiyon", "action", "dövüş sahnesi", "patlama",
                "kovalamaca", "silahlı çatışma"],

    # "eğlenceli" çıkarıldı: TÜM türlerde ("eğlenceli bir gerilim",
    # "eğlenceli bir aksiyon filmi") kullanılan en genel övgü
    # kelimelerinden biri, komediye özgü değil.
    "komedi": ["komedi", "comedy", "mizah", "hiciv", "kara mizah",
               "absürt komedi", "güldürü", "kahkaha"],

    # "aşk" tek başına çıkarıldı: "vatan aşkı", "sinemaya olan aşkı" gibi
    # tür dışı mecazi kullanımları çok yaygın. Somut ifadelerle değiştirildi.
    "romantik": ["romantik", "romance", "aşk hikayesi", "aşka düşmek",
                 "gönül ilişkisi", "duygusal aşk"],

    # "uzay" çıkarıldı: gerçek uzay tarihi/astronot biyografilerinde de
    # geçebilir. Kurgusal içeriğe daha özgü ifadelerle değiştirildi.
    "bilim kurgu": ["bilim kurgu", "sci-fi", "uzay yolculuğu", "uzaylı",
                    "galaksi", "distopya", "simülasyon", "yapay zeka"],

    # "toplumsal" tek başına çıkarıldı: "toplumsal medya", "toplumsal
    # cinsiyet" gibi tür dışı bağlamlarda da geçiyor.
    "dram": ["dram", "drama", "toplumsal dram", "aile dramı"],

    "suç": ["suç", "crime", "mafya", "gangster", "polisiye", "cinayet"],

    "animasyon": ["animasyon", "animation", "çizgi film"],

    "belgesel": ["belgesel", "documentary"],

    # "büyü" ve "efsane" çıkarıldı: "büyüleyici", "efsanevi" gibi tür dışı
    # genel övgü kelimelerinin içinde geçiyordu (bkz. önceki tur bug'ı).
    "fantastik": ["fantastik", "fantasy", "büyülü", "büyücü", "sihir",
                  "cadı", "ejderha", "peri masalı", "efsanevi yaratık",
                  "orta dünya"],

    # "savaş" tek başına çıkarıldı: "duygu savaşı", "aile içi savaş" gibi
    # mecazi kullanımları çok yaygın. Somut savaş bağlamına özgü
    # ifadelerle değiştirildi.
    "savaş": ["dünya savaşı", "cephe savaşı", "savaş filmi", "askeri",
              "cephe", "siper"],

    # "dans" çıkarıldı: dansçı biyografileri/dramaları (ör. Black Swan)
    # müzikal olmadan da dansı konu edinebiliyor.
    "müzikal": ["müzikal", "musical", "şarkılı dans sahneleri",
                "müzikal numarası"],

    "western": ["western", "kovboy"],

    "biyografi": ["biyografi", "gerçek hikaye", "true story"],
}

VALID_INTENTS = frozenset({
    "yonetmen", "oyuncu", "tur", "benzer_film", "ruh_hali", "karma",
})

# KÖKEN (ülke/menşei) ifadelerinin kanonik hâli — parser._ORIGIN_KEYWORDS'ün
# ÜRETTİĞİ değerlerle birebir aynı olmak ZORUNDA (aşağı normalize edilmiş
# hâlleriyle karşılaştırılır, bkz. engine._normalize).
#
# Bu set'in var olma sebebi: parser.py bir kullanıcı "türk filmi, aksiyon"
# dediğinde ikisini AYNI virgüllü filtreler.tur string'inde birleştiriyor
# (bkz. parser._detect_origin_filter). Ama köken ile tür FARKLI eksenlerdir:
# köken filmin NEREDEN olduğunu, tür NE TÜRDE olduğunu tanımlar. Bu iki
# eksen arasında mantık VE (AND) olmalı — "hem türk hem aksiyon" — ama aynı
# eksendeki birden fazla tür ("aksiyon, komedi") VEYA (OR) kalmalı — film
# ikisinden birini içerse yeter. engine.py bu set'i kullanarak virgüllü
# listeyi köken/tür olarak ikiye ayırıp doğru mantığı (VE / VEYA) uygular.
ORIGIN_GENRE_TERMS: frozenset[str] = frozenset({
    "türk filmi", "kore filmi", "japon filmi", "fransız filmi",
    "i̇ngiliz filmi", "amerikan filmi", "italyan filmi", "alman filmi",
    "hint filmi", "rus filmi", "ispanyol filmi", "iran filmi", "çin filmi",
})
