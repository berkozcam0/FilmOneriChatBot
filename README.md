---
title: Film Oneri ChatBot
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Film Öneri ChatBot

Hibrit arama motoru (BGE-M3 + BM25 + BGE-Reranker) ve Groq LLM ile çalışan
Türkçe film öneri chatbotu.

Kaynak kod: https://github.com/berkozcam0/FilmOneriChatBot

## Ortam Değişkenleri (Secrets)
Bu Space'in çalışması için Settings → Repository secrets altına şunu eklemen gerekir:
- `GROQ_API_KEY` — console.groq.com üzerinden alınan ücretsiz API anahtarı
