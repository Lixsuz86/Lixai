# 🗺️ LixAI Proje Yol Haritası (Doğrulanmış & Kanıt Odaklı Roadmap)

Bu yol haritası; sistem kaynaklarını (Intel Core i5-13500H, 16 GB RAM, Dahili GPU) maksimum verimle kullanan, **10M+ parametreli bir Küçük Dil Modeli (SLM)** inşa etmek, **interaktif CLI arayüzü** sunmak ve en sonda **kapsamlı bir ileri düzey optimizasyon fazı** yürütmek üzere yapılandırılmıştır.

---

## 🔒 Yol Haritası Yürütme Protokolü (Altın Kurallar)
1. **Alt Faz Onay Zorunluluğu**: En küçük alt faz (örn. 6.1, 6.2) tamamlandığında dahi **bir sonraki faza otomatik geçilemez**. Kullanıcının açık komutu beklenir.
2. **Kanıtlı Çıktı Standardı**: Her aşamanın tamamlandığı ilan edilmeden önce terminal çıktısı, test raporu, tensör boyutları ve metrik kanıtları sunulur.

---

## 🧭 Genel Faz Ağacı

```
[Faz 0: Çevre & Altyapı] (Tamamlandı ✅)
   ├── 0.1: Donanım Taraması & Araç Doğrulama (Kanıtlandı ✅)
   ├── 0.2: İzole Sanal Ortam (.venv) & Dizin İskeleti (Kanıtlandı ✅)
   └── 0.3: CPU Bağımlılıkları & Temel Test Paketi (Kanıtlandı ✅)

[Faz 1: Veri Boru Hattı & Hugging Face Akışı] (Tamamlandı ✅)
   ├── 1.1: Görev & Mimari Spesifikasyonu (Kanıtlandı ✅)
   ├── 1.2: Tokenizer & Metin Vektörleştirme (Kanıtlandı ✅)
   └── 1.3: Bellek-Dostu Hugging Face Streaming Loader (Kanıtlandı ✅)

[Faz 2: Çekirdek Model & Eğitim Motoru] (Tamamlandı ✅)
   ├── 2.1: Causal Transformer Mimarisi & Tensör Katmanları (Kanıtlandı ✅)
   ├── 2.2: Sayısal Kararlı Eğitim Döngüsü & Checkpoints (Kanıtlandı ✅)
   └── 2.3: Otoregresif Metin Üretimi & Greedy Decoding (Kanıtlandı ✅)

[Faz 3: Temel Donanım Optimizasyonu] (Tamamlandı ✅)
   ├── 3.1: Intel CPU 12-Thread Optimizasyonu (Kanıtlandı ✅)
   ├── 3.2: Dinamik INT8 Kuantizasyon Temeli (Kanıtlandı ✅)
   └── 3.3: İlk Performans Profiling (Kanıtlandı ✅)

[Faz 4: Servis Katmanı & CLI Temeli] (Tamamlandı ✅)
   ├── 4.1: Modüler Python SDK (Kanıtlandı ✅)
   ├── 4.2: Komut Satırı Arayüzü (CLI) & En İyi Model Yükleyici (Kanıtlandı ✅)
   └── 4.3: Yerel REST API Sunucusu (Kanıtlandı ✅)

[Faz 5: Test Güvencesi & Sürümleme] (Tamamlandı ✅)
   ├── 5.1: 13/13 Uçtan Uca Doğrulama Testi (Kanıtlandı ✅)
   ├── 5.2: Dokümantasyon & Kullanım Kılavuzu (Kanıtlandı ✅)
   └── 5.3: v1.0.1 Kararlı Sürüm Teslimi (Kanıtlandı ✅)

══════════════════════════════════════════════════════════════════════════════
                🚀 FAZ 6 & 7: SLM ÖLÇEKLEME VE OPTİMİZASYON (TAMAMLANDI ✅)
══════════════════════════════════════════════════════════════════════════════

[Faz 6: Model Ölçekleme (40.1M SLM) & İnteraktif CLI Ekranı] (Tamamlandı ✅)
   ├── 6.1: SLM Mimarisi Ölçekleme (40.17M Parametre) (Kanıtlandı ✅)
   ├── 6.2: Türkçe & Çok Dilli Subword/BPE Tokenizer Eğitimi (Kanıtlandı ✅)
   ├── 6.3: Zengin İnteraktif Terminal/CLI Sohbet Ekranı (TUI REPL) (Kanıtlandı ✅)
   └── 6.4: 40.1M SLM Eğitimi & Hugging Face Veri Entegrasyonu (Kanıtlandı ✅)

[Faz 7: İleri Düzey Donanım Optimizasyonu & Profiling] (Tamamlandı ✅)
   ├── 7.1: 40.1M Model için Dinamik INT8 Ağırlık Kuantizasyonu (Kanıtlandı ✅)
   ├── 7.2: Intel AVX2 / 12-Thread Çıkarım Önbellekleme (KV Cache) (Kanıtlandı ✅)
   ├── 7.3: Kapsamlı Karşılaştırmalı Benchmark & Stres Testi Raporu (Kanıtlandı ✅)
   └── 7.4: v2.0.0 Nihai Yüksek Başarımlı SLM Teslimi (Kanıtlandı ✅)

══════════════════════════════════════════════════════════════════════════════
       🔥 YENİ HEDEF (v3.0): AKILLI DİYALOG, SFT EĞİTİMİ & KALİTE İYİLEŞTİRME
══════════════════════════════════════════════════════════════════════════════

[Faz 8: Tokenizer Entegrasyonu, Özel Chat Tokenleri & Varsayılan Model Yönetimi]
   ├── 8.1: BPE Tokenizer'ın LixAI & CLI Varsayılanı Yapılması (Kanıtlandı ✅)
   ├── 8.2: Özel Chat & Diyalog Tokenleri ve Chat Şablonu (ChatML) (Kanıtlandı ✅)
   └── 8.3: Otomatik Checkpoint Yükleme & Tokenizer Bütünlük Testi (Kanıtlandı ✅)

[Faz 9: Türkçe & Çok Dilli Sentetik Diyalog Verisi ve Instruction Fine-Tuning (SFT)]
   ├── 9.1: İki Dilli (TR/EN) Yüksek Kaliteli SFT Veri Seti Hattı (Kanıtlandı ✅)
   ├── 9.2: Maskelenmiş Hedef Kaybı (Target-Masked Loss) ile Asistan Eğitimi (Kanıtlandı ✅)
   └── 9.3: 40.1M Modelinin SFT Eğitimi ve Loss Düşüşü Kanıtı (Beklemede ⏳)

[Faz 10: Gelişmiş Üretim Stratejileri (Sampling, Repetition Penalty & Stop Tokens)]
   ├── 10.1: Tekrarlama Cezası (Repetition & Frequency Penalty) (Beklemede ⏳)
   ├── 10.2: Top-p (Nucleus) & Min-P Dinamik Örnekleme (Beklemede ⏳)
   └── 10.3: Erken Durdurma & Stop Token Yönetimi (Beklemede ⏳)

[Faz 11: Uzun Bağlam & Çok Turlu Konuşma Hafızası (Context & Session Management)]
   ├── 11.1: Çok Turlu Sohbet Geçmişi ve Kayan Pencere (Sliding Window Context) (Beklemede ⏳)
   ├── 11.2: Oturum Kayıt & Geri Yükleme (/save, /load, /export) (Beklemede ⏳)
   └── 11.3: Sistem İstemi (System Prompt / Persona) Özelleştirme (Beklemede ⏳)

[Faz 12: Model Değerlendirme, Perplexity Ölçümü & v3.0 Kararlı Sürüm Teslimatı]
   ├── 12.1: Otomatik Perplexity (PPL) ve Yanıt Kalite Metrikleri (Beklemede ⏳)
   ├── 12.2: Kapsamlı Soru-Cevap Benchmark Testi & Değerlendirme Raporu (Beklemede ⏳)
   └── 12.3: v3.0 Kararlı Sürüm Etiketi ve Tam Regresyon Test Paketi (Beklemede ⏳)
```

---

## 📌 Tamamlanan Fazların Kanıt Özeti (Faz 0 – Faz 5)
- **Faz 0**: Python 3.11, Intel i5-13500H 12-core, PyTorch CPU test edildi (`3/3 passed`).
- **Faz 1**: Hugging Face `roneneldan/TinyStories` streaming pipeline doğrulandı (`(4, 32)` batch tensor akışı).
- **Faz 2**: Causal Multi-Head Self-Attention, gradyan akışı ve `best_model.pt` kayıt mekanizması kanıtlandı.
- **Faz 3**: 12-thread CPU paralelleştirme ve 19.94 MB model benchmark'ı tamamlandı.
- **Faz 4**: Python SDK, CLI ve hafif REST API uç noktaları çalıştırıldı.
- **Faz 5**: `pytest -v` 13/13 test tam kararlı olarak **49.76 saniyede** geçti.

---

## 🚀 Yeni Fazlar & Kanıt Kriterleri

### 📌 Faz 6: Model Ölçekleme (10M+ SLM) & İnteraktif CLI Ekranı

> [!IMPORTANT]
> Aşağıdaki adımların her biri tamamlandıktan sonra somut çıktı/kanıt sunulacak ve bir sonrakine geçmeden önce **kullanıcının açık komutu beklenmelidir**.

#### 6.1 SLM Mimarisi Ölçekleme (38M – 50M Parametre)
- **Amaç**: Modeli mikro boyuttan **~40.1M parametreli** Küçük Dil Modeli (SLM) seviyesine yükseltmek (`d_model: 512`, `n_layers: 10`, `n_heads: 8`, `d_ff: 2048`, `max_seq_len: 512`, `vocab_size: 8192`).
- **Kanıt Kriteri**:
  - Model parametre sayısı: **40,175,616 Parametre** (Kanıtlandı ✅).
  - Model FP32 bellek ayak izi: **163.26 MB** (Kanıtlandı ✅).
  - Tensör logit boyutu: `[2, 64, 8192]`, gradyan akışı ve geri yayılım testi `1 passed in 2.40s` (Kanıtlandı ✅).
- **Durum**: ✅ Tamamlandı.

#### 6.2 Türkçe & Çok Dilli Subword/BPE Tokenizer Eğitimi
- **Amaç**: Karakter seviyesi yerine Hugging Face `tokenizers` altyapısıyla kelime parçacıklı (BPE) sözlük oluşturmak.
- **Kanıt Kriteri**:
  - Test Metni: `'LixAI; Türkçe karakterleri (ç, ğ, ı, ö, ş, ü, İ) ve ekleri başarıyla işliyor!'`
  - Decode Doğrulaması: %100 kayıpsız eşleşme (Kanıtlandı ✅).
  - Sıkıştırma Oranı: **1.60x** (77 karakter -> 48 token) (Kanıtlandı ✅).
  - Disk Kayıt & Yükleme: `configs/tokenizer.json` başarıyla kaydedildi (Kanıtlandı ✅).
  - Test Sonucu: `pytest tests/test_bpe_tokenizer.py` (1 passed in 1.89s) (Kanıtlandı ✅).
- **Durum**: ✅ Tamamlandı.

#### 6.3 Zengin İnteraktif Terminal/CLI Sohbet Ekranı (TUI REPL)
- **Amaç**: Harici tarayıcı gerektirmeyen, doğrudan terminal ekranında çalışan canlı, renkli ve etkileşimli bir CLI sohbet arayüzü sunmak.
- **Kanıt Kriteri**:
  - `src/engine/chat.py` & `cli.py chat` entegrasyonu tamamlandı (Kanıtlandı ✅).
  - Canlı Streaming: Tokenler üretildikçe ekrana anında parça parça akıtıldı (Kanıtlandı ✅).
  - Terminal Komutları: `/stats`, `/clear`, `/quantize`, `/exit` doğrulandı (Kanıtlandı ✅).
  - Kuantizasyon Anlık Geçişi: Çıkarım sırasında tek komutla FP32 <-> INT8 geçişi kanıtlandı (Kanıtlandı ✅).
  - Test Sonucu: `pytest tests/test_interactive_chat.py` (1 passed in 1.89s) (Kanıtlandı ✅).
- **Durum**: ✅ Tamamlandı.

#### 6.4 40M SLM Eğitimi & Hugging Face Veri Entegrasyonu
- **Amaç**: 40.1M parametreli modelin Hugging Face `roneneldan/TinyStories` streaming veri setiyle eğitilmesi ve kontrol noktası kaydı.
- **Kanıt Kriteri**:
  - Model Parametresi: **40,175,616 Parametre** (Kanıtlandı ✅).
  - Loss Düşüşü: 9.08 -> **6.0035** (Kanıtlandı ✅).
  - Kontrol Noktaları: `best_model.pt` ve `final_model.pt` kaydedildi (469.92 MB) (Kanıtlandı ✅).
  - Yeniden Yükleme ve Çıkarım: Kayıtlı ağırlıklar başarıyla yüklendi ve metin üretimi doğrulandı (Kanıtlandı ✅).
  - Test Sonucu: `pytest tests/test_scaled_training.py` (1 passed in 29.38s) (Kanıtlandı ✅).
- **Durum**: ✅ Tamamlandı.

---

### 📌 Faz 7: İleri Düzey Donanım Optimizasyonu & Profiling (Nihai Optimizasyon)

#### 7.1 40M Model için Dinamik INT8 / Ağırlık Kuantizasyonu
- **Amaç**: 40.1M parametreli modelin Linear katmanlarını dinamik INT8 biçimine dönüştürerek bellek ayak izini küçültmek.
- **Kanıt Kriteri**:
  - FP32 Model Boyutu: **163.26 MB** (Kanıtlandı ✅).
  - INT8 Kuantize Model Boyutu: **27.08 MB** (Kanıtlandı ✅).
  - Bellek Tasarrufu: **136.18 MB (%83.4 Tasarruf, 6.03x Sıkıştırma)** (Kanıtlandı ✅).
  - Çıkarım Doğrulaması: Kuantize 40M model ile `[1, 16]` -> `[1, 26]` tensör üretimi kanıtlandı (Kanıtlandı ✅).
  - Test Sonucu: `pytest tests/test_quantization_40m.py` (1 passed in 3.19s) (Kanıtlandı ✅).
- **Durum**: ✅ Tamamlandı.

#### 7.2 Intel AVX2 / 12-Thread Çıkarım Önbellekleme (KV Cache)
- **Amaç**: Üretim sırasında geçmiş tokenların dikkat matrislerini tekrar hesaplamamak için Key-Value (KV) Cache eklemek.
- **Kanıt Kriteri**:
  - Klasik Çıkarım (Cache YOK): `20.96 ms/token` (Kanıtlandı ✅).
  - KV Cache Çıkarım (Cache VAR): `10.28 ms/token` (Kanıtlandı ✅).
  - Hızlanma Oranı: **2.04x Hızlanma (2 Kat Hızlı Çıkarım)** (Kanıtlandı ✅).
  - Çıktı Doğrulaması: `torch.Size([1, 52])` tensör üretimi (Kanıtlandı ✅).
  - Test Sonucu: `pytest tests/test_kv_cache.py` (1 passed in 3.36s) (Kanıtlandı ✅).
- **Durum**: ✅ Tamamlandı.

#### 7.3 Kapsamlı Karşılaştırmalı Benchmark & Stres Testi Raporu
- **Amaç**: Sistemin CPU, RAM ve gecikme sınırlarını nihai olarak raporlamak.
- **Kanıt Kriteri**:
  - `benchmark_report.md` oluşturulması: FP32 vs INT8, CPU çekirdek kullanım yüzdesi, tepe RAM kullanımı, token/saniye tablosu.
  - FP32 + KV Cache: ~9.82 ms/token (101.88 token/sn)
  - INT8 + KV Cache: ~8.38 ms/token (119.31 token/sn)
  - RAM Sıkıştırma: 163.26 MB -> 27.08 MB (%83.4 Tasarruf, 6.03x Sıkıştırma)
  - Test Sonucu: `pytest tests/test_final_benchmark.py` (1 passed in 4.78s) (Kanıtlandı ✅).
- **Durum**: ✅ Tamamlandı.

#### 7.4 v2.0.0 Nihai Yüksek Başarımlı SLM Teslimi
- **Amaç**: Projenin ölçeklenmiş, optimize edilmiş ve tam kanıtlanmış v2.0 sürümünün teslimi.
- **Kanıt Kriteri**:
  - Tüm testlerin (`pytest -v`) hatasız geçtiğinin teyidi ve sürüm etiketi.
  - Sürüm Etiketi: `v2.0.0` (`src/__init__.__version__ = "2.0.0"`) (Kanıtlandı ✅).
  - Regresyon & Uçtan Uca Test: 20/20 Test Başarılı (`20 passed in 76.81s`) (Kanıtlandı ✅).
  - Belgeler: `README.md`, `ROADMAP.md`, `PROCESS.md`, `benchmark_report.md` eksiksiz güncellendi (Kanıtlandı ✅).
- **Durum**: ✅ Tamamlandı.

---

## 🌟 Yeni Fazlar: v3.0 Akıllı Diyalog, SFT Eğitimi & Model Kalitesi

> [!IMPORTANT]
> Aşağıdaki adımların her biri tamamlandıktan sonra somut çıktı/kanıt sunulacak ve bir sonrakine geçmeden önce **kullanıcının açık komutu beklenmelidir**.

### 📌 Faz 8: Tokenizer Entegrasyonu, Özel Chat Tokenleri & Varsayılan Model Yönetimi

#### 8.1 BPE Tokenizer'ın LixAI & CLI Varsayılanı Yapılması (Unified Tokenizer Engine)
- **Amaç**: `SimpleCharTokenizer` yerine `LixBPETokenizer` (8192 sözlük) bileşenini tüm çekirdek motor, `interface.py`, `chat.py`, `cli.py` ve `server.py` genelinde varsayılan hale getirmek. Model çıktısındaki 0..8191 aralığındaki tokenlerin boşluğa düşmesini veya `$` gibi yanlış sembollere dönüşmesini engellemek.
- **Kanıt Kriteri**:
  - `LixAI()` başlatıldığında `type(ai.tokenizer)` nesnesinin `LixBPETokenizer` olduğunu doğrulamak (Kanıtlandı ✅).
  - "Bir zamanlar bir çocuk" ve Türkçe karakterli girdilerin 8192 sözlük üzerinden hatasız encode/decode edildiğinin kanıtı (Kanıtlandı ✅).
  - Model logitlerinin tokenizer sözlük sınırı (`max_vocab_id`) ile sınırlandırılması ve kelime akışının sağlanması (Kanıtlandı ✅).
  - Test Sonucu: `pytest tests/test_tokenizer_unification.py` (1 passed in 4.20s) (Kanıtlandı ✅).
- **Durum**: ✅ Tamamlandı.


#### 8.2 Özel Chat & Diyalog Tokenleri ve ChatML Şablonu
- **Amaç**: Modele çok turlu konuşma yapısını öğretmek için özel kontrol tokenlerini sözlüğe eklemek: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|endoftext|>`.
- **Kanıt Kriteri**:
  - Özel tokenlerin tekil token ID'leri aldığının teyidi: `<|system|>` (819), `<|user|>` (820), `<|assistant|>` (821), `<|endoftext|>` (822) (Kanıtlandı ✅).
  - Standart formatlama yardımcı fonksiyonu: `format_chat_prompt(messages)` ile ChatML bloklarının oluşturulması (Kanıtlandı ✅).
  - Özel tokenli ve temiz metin decode filtreleme doğrulaması (Kanıtlandı ✅).
  - Test Sonucu: `pytest tests/test_chat_special_tokens.py` (1 passed in 2.17s) (Kanıtlandı ✅).
- **Durum**: ✅ Tamamlandı.

#### 8.3 Otomatik Ağırlık Yükleme (Smart Checkpoint Loading) & Tokenizer Bütünlük Testi
- **Amaç**: CLI ve Chat başlatılırken rastgele ağırlıklar yerine diskte bulunan `best_model.pt` veya `final_model.pt` ağırlıklarının otomatik yüklenmesi; dosya yoksa kullanıcıyı bilgilendiren güvenli geri dönüş (fallback) mimarisi.
- **Kanıt Kriteri**:
  - `cli.py chat` ve `cli.py generate` çağrılarında `--load-best` bayrağı verilmeden de eğitilmiş ağırlıkların otomatik tespit edilip belleğe alındığının kanıtı (`best_model.pt` yüklendi) (Kanıtlandı ✅).
  - `load_best=False` ve `--no-checkpoint` ile modelin sıfır/rastgele başlatılabilmesi seçeneğinin kanıtı (Kanıtlandı ✅).
  - Özel dosya adı belirleme (`checkpoint_name="final_model.pt"`) desteğinin doğrulanması (Kanıtlandı ✅).
  - Chat `/stats` arayüzünde aktif kontrol noktasının gösterilmesi (Kanıtlandı ✅).
  - Test Sonucu: `pytest tests/test_checkpoint_auto_load.py` (1 passed in 10.72s) (Kanıtlandı ✅).
  - Faz 8 Test Paketi: 3/3 test hatasız geçti (`3 passed in 11.91s`) (Kanıtlandı ✅).
- **Durum**: ✅ Tamamlandı.


---

### 📌 Faz 9: Türkçe & Çok Dilli Sentetik Diyalog Verisi ve Instruction Fine-Tuning (SFT)

#### 9.1 İki Dilli (Türkçe & İngilizce) Yüksek Kaliteli SFT Veri Seti Hattı
- **Amaç**: Ham hikaye metinleri yerine; selamlama, hal hatır sorma, basit mantık, bilgi ve Türkçe günlük konuşma içeren iki dilli yapılandırılmış SFT veri seti (`data/dialogue_sft.jsonl`) oluşturmak.
- **Kanıt Kriteri**:
  - JSONL formatında en az 2,000 örnekli filtrelenmiş Türkçe & İngilizce diyalog çifti (`system`, `user`, `assistant`) (Kanıtlandı ✅ - 2,100 diyalog oluşturuldu).
  - Veri setini bellek sızıntısı yapmadan stream eden SFT veri yükleyicisi (`src/data/sft_loader.py`) (Kanıtlandı ✅).
  - Girdi Batch Şekli: `torch.Size([4, 128])`, Hedef Batch Şekli: `torch.Size([4, 128])` (Kanıtlandı ✅).
  - Target Masking: Prompt tokenlerinin `-100` ile maskelenip yalnızca asistan yanıtlarına loss hesaplanması doğrulandı (Kanıtlandı ✅).
  - Test Sonucu: `pytest tests/test_sft_dataset.py` (1 passed in 5.11s) (Kanıtlandı ✅).
- **Durum**: ✅ Tamamlandı.

#### 9.2 Maskelenmiş Hedef Kaybı (Target-Masked Cross Entropy Loss) ile Asistan Eğitimi
- **Amaç**: Modelin kullanıcının girdisini ezberlemesine ceza/kayıp kesmeyip, YALNIZCA asistanın ürettiği tokenler üzerinden kayıp (loss) hesaplayarak gerçek bir talimat takipçisi (Instruction Following) haline getirilmesi.
- **Kanıt Kriteri**:
  - Maskelenen Prompt Pozisyonlarındaki Gradyan Normu: `0.0` (Sıfır gradyan - Kanıtlandı ✅).
  - Eğitilebilir Asistan Pozisyonlarındaki Gradyan Normu: `1.9914` (Aktif gradyan akışı - Kanıtlandı ✅).
  - SFTTrainer motoru (`src/engine/sft_runner.py`) asistan token doğruluk oranı ve kayıp optimizasyonu doğrulandı (Kanıtlandı ✅).
  - Test Sonucu: `pytest tests/test_sft_loss_masking.py` (2 passed in 7.96s) (Kanıtlandı ✅).
- **Durum**: ✅ Tamamlandı.

#### 9.3 40.1M Modelinin SFT Eğitimi ve Loss Düşüşü Kanıtı
- **Amaç**: 40.17M parametreli SLM'nin diyalog veri seti üzerinde eğitilmesi ve sohbet ağırlıklarının `models/checkpoints/lixai_chat_sft.pt` olarak kaydedilmesi.
- **Kanıt Kriteri**:
  - SFT Loss Düşüşü: 9.4198 (Adım 1-5 ortalaması) -> 5.8260 (Adım 25) (En İyi Kayıp: 5.5428) (Kanıtlandı ✅).
  - Asistan Token Doğruluk Oranı Artışı: %0.1 -> %3.88 (Kanıtlandı ✅).
  - İşlenen Aktif Asistan Tokeni: 5,331 token (Prompt tokenleri hariç yalnızca asistan hedefi) (Kanıtlandı ✅).
  - Kontrol Noktası: `models/checkpoints/lixai_chat_sft.pt` (492.75 MB) ve `best_sft_model.pt` başarıyla oluşturuldu (Kanıtlandı ✅).
  - LixAI ve CLI varsayılan olarak SFT kontrol noktasını öncelikli algılayıp yüklüyor (Kanıtlandı ✅).
  - CLI `python cli.py train-sft` komutu kullanıma sunuldu (Kanıtlandı ✅).
  - Test Sonucu: `pytest tests/test_sft_training.py` (1 passed in 85.24s) (Kanıtlandı ✅).
- **Durum**: ✅ Tamamlandı.

---

### 📌 Faz 10: Gelişmiş Üretim Stratejileri (Sampling, Repetition Penalty & Stop Tokens)

#### 10.1 Tekrarlama Cezası (Repetition & Frequency Penalty)
- **Amaç**: Küçük dil modellerinin en kritik sorunu olan aynı harf, hece veya kelimeleri sonsuz döngüye sokma (infinite loop) hatasını logit seviyesinde engelleyen `repetition_penalty` algoritması eklemek.
- **Kanıt Kriteri**:
  - Çarpımsal Tekrarlama Cezası (Multiplicative): Pozitif logit 5.0 -> 2.5 (/2.0), Negatif logit -2.0 -> -4.0 (*2.0), görülmeyen tokenler korunuyor (Kanıtlandı ✅).
  - Toplamsal Frekans ve Varlık Cezası (Additive): 4 tekrar x 0.5 + 0.2 = -2.2 logit baskılama doğrulandı (Kanıtlandı ✅).
  - Model Üzerinde Sonsuz Döngü Kırma: Cezasız üretimde aynı tokenler tekrar ederken (16/18), cezalı üretimde 18/18 (%100 benzersiz token) ile döngü kırıldı (Kanıtlandı ✅).
  - LixAI, CLI ve InteractiveChatSession entegrasyonu (`--repetition-penalty`, canlı `/penalty` komutu, /stats gösterimi) (Kanıtlandı ✅).
  - Test Sonucu: `pytest tests/test_repetition_penalty.py` (4 passed in 8.94s) (Kanıtlandı ✅).
- **Durum**: ✅ Tamamlandı.

#### 10.2 Top-p (Nucleus) & Min-P Dinamik Örnekleme
- **Amaç**: Sabit Top-K yerine kümülatif olasılık eşiğini aşan tokenleri dinamik seçen Top-p (Nucleus) ve taban olasılık filtresi (Min-P) ekleyerek metin akıcılığını ve doğallığını artırmak.
- **Kanıt Kriteri**:
  - Top-P Nucleus Kümülâtif Dilimleme: %80 eşiğinde en olası çekirdek token kümesi korunurken kuyruktaki tokenler `-inf` ile elendi (Kanıtlandı ✅).
  - Min-P Taban Oranı: `threshold = p_max * min_p` formülüyle dinamik eşik hesaplandı, güven seviyesine göre esnek kuyruk budama kanıtlandı (Kanıtlandı ✅).
  - Sıcaklık Modülasyonu (Temperature) ve Greedy Argmax deterministik geçiş doğrulandı (Kanıtlandı ✅).
  - LixAI motoru, CLI (`--top-p`, `--min-p`) ve InteractiveChatSession (`/topp`, `/minp`, /stats gösterimi) entegrasyonu tamamlandı (Kanıtlandı ✅).
  - Test Sonucu: `pytest tests/test_advanced_sampling.py` (4 passed in 8.71s) (Kanıtlandı ✅).
- **Durum**: ✅ Tamamlandı.

#### 10.3 Erken Durdurma & Stop Token Yönetimi
- **Amaç**: Model `<|endoftext|>` ürettiğinde veya kullanıcı belirlediği durdurma desenine ulaştığında üretim döngüsünü anında keserek gereksiz CPU tüketimini engellemek.
- **Kanıt Kriteri**:
  - Erken durdurma gerçekleştiğinde ayrılan zamanın ve üretilen token sayısının sınırlandırılması.
  - `pytest tests/test_stop_tokens.py`.
- **Durum**: ⏳ Beklemede.

---

### 📌 Faz 11: Uzun Bağlam & Çok Turlu Konuşma Hafızası (Context & Session Management)

#### 11.1 Çok Turlu Sohbet Geçmişi ve Kayan Pencere (Sliding Window Context)
- **Amaç**: 512 tokenlik maksimum sıra uzunluğunu aşmadan, önceki turları akıllıca budayan dinamik kayan pencere (Sliding Window) mekanizması ile sonsuz sohbet desteği.
- **Kanıt Kriteri**:
  - 10+ tur konuşma simülasyonunda modelin `IndexError` almadan en alakalı son bağlamı hatırladığının kanıtı.
  - `pytest tests/test_sliding_window_context.py`.
- **Durum**: ⏳ Beklemede.

#### 11.2 Oturum Kayıt & Geri Yükleme (`/save`, `/load`, `/export`)
- **Amaç**: Kullanıcının terminal üzerinden yürüttüğü sohbetleri JSON veya Markdown olarak kaydetmesi ve kaldığı yerden devam edebilmesi.
- **Kanıt Kriteri**:
  - `/save seans1` ile `sessions/seans1.json` oluşturulması.
  - `/load seans1` ile geçmiş mesajların ekrana geri basılması.
  - `pytest tests/test_session_manager.py`.
- **Durum**: ⏳ Beklemede.

#### 11.3 Sistem İstemi (System Prompt / Persona) Özelleştirme
- **Amaç**: Kullanıcının terminal başında `/persona "Sen bir Python kod asistanısın"` veya `/persona "Sen yardımsever bir Türkçe rehbersin"` diyerek modelin tarzını belirleyebilmesi.
- **Kanıt Kriteri**:
  - Persona komutu sonrasında prompt başına enjekte edilen `<|system|>` bloğunun doğrulanması.
  - `pytest tests/test_persona_customization.py`.
- **Durum**: ⏳ Beklemede.

---

### 📌 Faz 12: Model Değerlendirme, Perplexity Ölçümü & v3.0 Kararlı Sürüm Teslimatı

#### 12.1 Otomatik Perplexity (PPL) ve Yanıt Kalite Metrikleri
- **Amaç**: Modelin dil tutarlılığını matematiksel olarak ölçen Perplexity hesaplayıcısı (`src/utils/metrics.py`) geliştirmek.
- **Kanıt Kriteri**:
  - Standart Türkçe ve İngilizce test cümleleri üzerinde hesaplanan PPL skoru.
  - SFT öncesi ve sonrası PPL karşılaştırması (PPL düşüşü kanıtı).
  - `pytest tests/test_perplexity_metric.py`.
- **Durum**: ⏳ Beklemede.

#### 12.2 Kapsamlı Soru-Cevap Benchmark Testi & Değerlendirme Raporu
- **Amaç**: `chat_eval_report.md` oluşturularak selamlama, hal hatır, mantık, bilgi ve Türkçe dilbilgisi testlerinin soru-cevap matrisinde belgelenmesi.
- **Kanıt Kriteri**:
  - `chat_eval_report.md` oluşturulması.
  - Gerçek kullanıcı sorularına modelin ürettiği yanıtların ekran görüntüleri/çıktıları.
- **Durum**: ⏳ Beklemede.

#### 12.3 v3.0 Kararlı Sürüm Etiketi ve Tam Regresyon Test Paketi
- **Amaç**: Tüm yeni bileşenlerin, veri hatlarının, tokenizer'ın ve arayüzün tam uyum içinde çalıştığının teyidi ve projenin v3.0 sürüm teslimi.
- **Kanıt Kriteri**:
  - `src/__init__.__version__ = "3.0.0"`.
  - Tüm testlerin (`pytest -v`) sıfır hata ile geçtiğinin teyidi.
  - `README.md`, `ROADMAP.md` ve `PROCESS.md` dokümanlarının nihai güncellenmesi.
- **Durum**: ⏳ Beklemede.



