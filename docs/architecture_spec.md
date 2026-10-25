# LixAI Mimari ve Görev Spesifikasyonu (38M – 50M SLM Mimarisi)

## 1. Görev Tanımı (Task Definition)
- **Model Tipi**: Otoregresif Nedensel Küçük Dil Modeli (Causal Transformer Decoder - SLM).
- **Hedef**: Hugging Face akışı ve özel BPE sözlüğü ile zengin dil üretimi ve interaktif terminal sohbeti.
- **Hedef Parametre Aralığı**: **38 Milyon – 50 Milyon Parametre (38M - 50M SLM)**.

---

## 2. Donanım & Kaynak Dengesi
- **İşlemci**: Intel Core i5-13500H (12 Çekirdek, 16 İş Parçacığı - AVX2 hızlandırmalı CPU).
- **Bellek Ayak İzi**: 
  - Çıkarım Modu (FP32): ~150 – 180 MB RAM.
  - Çıkarım Modu (INT8 Kuantize): ~45 – 55 MB RAM.
  - Eğitim Modu (AdamW + Gradyanlar): ~800 MB – 1.1 GB RAM (16 GB sistem RAM'inde sıfır dar boğaz).

---

## 3. Matematiksel & Boyutsal Parametre Konfigürasyonu

| Hiperparametre | Değer | Açıklama |
| :--- | :---: | :--- |
| `vocab_size` | **8192** | Zengin Türkçe ve İngilizce BPE alt-kelime sözlüğü |
| `d_model` | **512** | Vektör gömme boyutu (Embedding dimension) |
| `n_heads` | **8** | Dikkat başlığı sayısı (`head_dim = 64`) |
| `n_layers` | **8** | Derin Transformer blok sayısı |
| `d_ff` | **2048** | Feed-Forward genişletme katmanı (`4 * d_model`) |
| `max_seq_len` | **512** | Bağlam penceresi uzunluğu (Context window) |
| `dropout` | **0.1** | Düzenlileştirme katsayısı |

### Parametre Dağılım Hesabı:
- **Token & Pozisyon Embedding**: `8192 * 512 + 512 * 512` ≈ `4.45M`
- **Transformer Blokları (x8)**: `8 * ~3.93M` ≈ `31.47M`
- **Final LayerNorm & Başlık**: ~`4.19M`
- **Toplam Parametre**: **~40.1 Milyon Parametre (40.1M SLM)** 🎯

---

## 4. Girdi / Çıktı Sözleşmesi
- **Girdi**: `input_ids: Tensor[batch_size, seq_len]` (LongTensor, `0 <= id < 8192`).
- **Çıktı**: `logits: Tensor[batch_size, seq_len, 8192]` (FloatTensor).
- **Kayıp**: Targets verildiğinde sayısal kararlı `F.cross_entropy`.
