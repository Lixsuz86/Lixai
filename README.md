# ⚡ LixAI: 40.1M Parametreli Intel CPU Optimize SLM (v3.0.0-beta.1)

LixAI, harici GPU gerektirmeyen, **Intel Core i5-13500H** (12 çekirdek, 16 thread) ve **16 GB RAM** donanım mimarisine özel olarak sıfırdan Python ve PyTorch ile geliştirilmiş yüksek başarımlı bir **Küçük Dil Modeli (Small Language Model - SLM)** kütüphanesi ve diyalog motorudur.

---

## 🌟 Öne Çıkan Özellikler (v3.0.0-beta.1)

- **40,172,032 (~40.17M) Parametre SLM Mimarisi**:
  - `vocab_size: 8192`, `d_model: 512`, `n_layers: 10`, `n_heads: 8`, `d_ff: 2048`, `max_seq_len: 512`.
  - Causal Multi-Head Self-Attention, Pre-LayerNorm, GELU aktivasyonu ve optimize edilmiş artık bağlantılar.
- **Instruction Fine-Tuning (SFT) & Hedef Kayıp Maskelemesi**:
  - İki dilli (Türkçe & İngilizce) 2,100 kaliteli diyalog veri seti üzerinde eğitilmiş sohbet modeli (`lixai_chat_sft.pt`).
  - PyTorch `ignore_index=-100` ile prompt pozisyonları maskelenmiş, yalnızca asistan yanıtlarına odaklanan eğitim mimarisi.
- **Gelişmiş Üretim & Örnekleme Stratejileri**:
  - **Repetition & Frequency Penalty**: Mekanik hece ve kelime tekrarlarını logit seviyesinde engelleyen çarpımsal ve toplamsal ceza algoritması.
  - **Top-P (Nucleus) & Min-P Dinamik Örnekleme**: Kümülatif eşikleme ve en yüksek olasılığa bağlı adaptif taban filtresi ile akıcı ve doğal metin üretimi.
- **Intel AVX2 & Key-Value (KV) Cache Hızlandırması**:
  - Her token üretiminde tüm diziyi baştan hesaplamak yerine önceki adımların anahtar ve değerlerini önbelleğe alır.
  - Klasik çıkarıma göre **~2 kat daha hızlı** çıkarım gecikmesi sağlar.
- **Dinamik INT8 Kuantizasyon**:
  - Model bellek ayak izini **163.26 MB seviyesinden 27.08 MB seviyesine indirir (%83.4 RAM tasarrufu, 6.03x sıkıştırma)**.
  - CPU önbelleği (L2/L3) ile tam uyumlu ve 119+ token/saniye çıkarım performansı.
- **Türkçe ve Çok Dilli Byte-Level BPE Tokenizer & ChatML**:
  - 8192 sözlük boyutlu, UTF-8 byte-fallback korumalı, özel Türkçe karakterleri (`ç, ğ, ı, ö, ş, ü`) hatasız kodlayan BPE tokenize motoru.
  - Standart ChatML özel tokenleri: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|endoftext|>`.
- **Zengin İnteraktif Terminal / CLI Sohbet Ekranı (TUI)**:
  - Canlı token akışı (streaming), `/stats`, `/penalty`, `/topp`, `/minp`, `/quantize`, `/clear`, `/exit` komutları içeren zengin terminal arayüzü.
- **Gelişmiş Çoklu Arayüz**:
  - Python SDK (`src.engine.interface`), CLI (`cli.py`), hafif yerel REST API (`src.api.server`).

---

## 📊 Donanım & Performans Benchmark Özeti

Intel Core i5-13500H (12 Thread, 16 GB RAM) üzerinde doğrulanmış performans değerleri:

| Konfigürasyon | Thread | KV Cache | Gecikme (ms/token) | Çıkarım Hızı | RAM Ayak İzi |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **INT8 Dinamik Kuantize** | **12** | **AÇIK** | **~8.38 ms** | **~119.31 token/sn** | **27.08 MB** |
| **FP32 + KV Cache** | **12** | **AÇIK** | **~9.82 ms** | **~101.88 token/sn** | 163.26 MB |
| **FP32 Standart (No Cache)** | 12 | KAPALI | ~12.41 ms | ~80.59 token/sn | 163.26 MB |

> Detaylı benchmark ve stres testi sonuçları için [benchmark_report.md](benchmark_report.md) dosyasına bakabilirsiniz.

---

## 🚀 Hızlı Başlangıç

### 1. Sanal Ortamı Etkinleştirin
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. İnteraktif Terminal Sohbetini Başlatın (TUI)
Canlı token akışlı, ANSI renkli interaktif terminal sohbeti:
```powershell
python cli.py chat
```
Sohbet içi komutlar:
- `/stats` : Bellek, parametre ve donanım kullanım istatistikleri.
- `/quantize` : INT8 ve FP32 modları arasında anlık geçiş.
- `/clear` : Sohbet geçmişini sıfırlar.
- `/exit` : Çıkış yapar.

### 3. Modeli Eğitin (CLI)
Hugging Face üzerinden akış halinde 50 adım eğitmek için:
```powershell
python cli.py train --steps 50
```

### 4. Tek Komutla Metin Üretin
```powershell
# INT8 Kuantize Hızlı Mod:
python cli.py generate --prompt "Bir zamanlar" --tokens 40 --quantize

# Standart FP32 Modu:
python cli.py generate --prompt "Once upon a time" --tokens 40
```

### 5. Donanım Benchmark Testini Çalıştırın
```powershell
python cli.py benchmark
```

### 6. Yerel REST API Sunucusunu Başlatın
```powershell
python -m src.api.server
```
- Sağlık kontrolü: `GET http://127.0.0.1:8000/health`
- Metin üretimi: `POST http://127.0.0.1:8000/generate` (JSON: `{"prompt": "Merhaba", "tokens": 30, "quantize": true}`)

---

## 🐍 Python SDK ile Kullanım

```python
from src.engine.interface import LixAI

# 40M SLM Modelini Başlat (varsayılan olarak 12 CPU thread ve optimize konfigürasyon)
ai = LixAI()

# Metin Üretimi (KV-Cache ve INT8 Kuantizasyon Aktif)
text = ai.generate(
    prompt="Küçük bir çocuk ormanda",
    max_new_tokens=40,
    temperature=0.8,
    use_quantized=True,
    use_cache=True
)
print(text)
```

---

## 🧪 Testleri Çalıştırma

Tüm birim, mimari, veri, kuantizasyon, KV-Cache ve benchmark testlerini yürütmek için:
```powershell
.\.venv\Scripts\pytest -v
```

---

## 📄 Lisans (License)

LixAI, açık kaynak topluluğuna maksimum esneklik ve yasal koruma sağlamak amacıyla **çift lisans (Dual License)** ile sunulmaktadır:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

Kullanıcılar ve kurumlar, kendi kullanım amaçlarına ve kurumsal gereksinimlerine göre bu iki lisanstan dilediklerini tercih edebilirler.


