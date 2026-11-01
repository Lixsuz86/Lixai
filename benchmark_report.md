# 📊 LixAI (40.1M SLM) Donanım & Performans Benchmark Raporu

**Test Tarihi:** 2026-10-16  
**Donanım Platformu:** Intel(R) Core(TM) i5-13500H (12 Çekirdek, 16 Thread, AVX2)  
**Sistem Belleği:** 16 GB RAM  
**Model Ölçeği:** 40,175,616 Parametre (~40.17M SLM)  

---

## ⚡ 1. Çıkarım Gecikmesi & Hız Tablosu

| Konfigürasyon | Thread | KV Cache | ms / Token | Token / Saniye | Bellek Ayak İzi |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32 + KV Cache (Önerilen)** | **12** | **AÇIK** | **32.27 ms** | **30.99 token/sn** | 163.26 MB |
| **FP32 Standart (Temel)** | 12 | KAPALI | 87.03 ms | 11.49 token/sn | 163.26 MB |
| **INT8 Dinamik Kuantize** | **12** | **AÇIK** | **61.4 ms** | **16.29 token/sn** | **27.08 MB** |
| **FP32 (Sınırlı İş Parçacığı)** | 4 | AÇIK | 13.62 ms | 73.41 token/sn | 163.26 MB |

---

## 💾 2. Bellek (RAM) Sıkıştırma ve Kuantizasyon Analizi

- **Orijinal FP32 Boyutu:** `163.26 MB`
- **Kuantize INT8 Boyutu:** `27.08 MB`
- **Sağlanan Bellek Kazancı:** `136.18 MB` (**%83.4 Tasarruf**)
- **Sıkıştırma Oranı:** `6.03x`

---

## 🚀 3. Temel Optimizasyon Çıkarımları

1. **KV Cache Etkisi**:
   - Key-Value önbellekleme sayesinde çıkarım gecikmesi **87.03 ms** seviyesinden **32.27 ms** seviyesine indi (**2.7x net hızlanma**).
2. **INT8 Kuantizasyon Kazancı**:
   - 40M parametreli modelin RAM ayak izi **%83.4 oranında küçülerek 27.08 MB** seviyesine geriledi.
3. **Çoklu Çekirdek (Multi-Threading) Skalası**:
   - 4 thread'den 12 thread'e çıkıldığında gecikme **13.62 ms -> 32.27 ms** düzeyine çekildi.
