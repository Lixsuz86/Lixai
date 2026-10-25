# AI Rules & Development Standards

Bu belge, proje süresince uyulması gereken temel prensipleri, kodlama standartlarını, faz yürütme disiplinini ve donanım kısıtlamalarına uyum kurallarını tanımlar.

---

## 1. Faz Yürütme & Kullanıcı Onayı Kuralları (Execution Governance)
- **Alt Fazlarda Bile Zorunlu Kullanıcı Onayı**:
  - İster ana faz (örn. Faz 1, Faz 2) ister alt/mikro-faz (örn. Faz 1.1, Faz 1.2) olsun, tamamlanan bir adımdan sonra **bir sonrakine asla otomatik olarak geçilemez**.
  - Her alt faz bittiğinde işlem durdurulmalı, çıktı doğrulanmalı ve kullanıcının açık komutu/onayı beklenmelidir.
- **Kanıtlı ve Doğrulanmış Çıktı Zorunluluğu (Empirical Proof)**:
  - Sadece "tamamlandı" şeklinde sözel bir beyan vermek KESİNLİKLE YASAKTIR.
  - Her fazın/alt fazın bitiminde somut, doğrulanmış ve kanıtlanmış bir çıktı sunulmalıdır:
    - Çalıştırılan `pytest` veya test scripti terminal logları (kaç test geçti, ne kadar sürede),
    - Üretilen tensörlerin boyutları (`shape`), veri tipleri (`dtype`) ve örnek değerleri,
    - Çalışan komut satırı (CLI) komutunun gerçek ekran çıktısı,
    - Gerçek bellek ve gecikme (latency) ölçüm verileri.
  - Bu kanıtlar sayesinde olası hatalar zincirleme büyümeden en erken aşamada tespit edilecek ve anında düzeltilecektir.

---

## 2. Donanım & Kaynak Yönetimi Kuralları (Hardware-Aware AI)
- **CPU & RAM Optimizasyonu**: Sistemde harici bir NVIDIA GPU bulunmadığından (Intel Core i5-13500H + 16 GB RAM), tüm modeller ve veri işleme hatları CPU ve RAM sınırları gözetilerek tasarlanmalıdır.
- **Hafıza Sızıntısı Önleme**: Veri yükleme (batching) ve tensör işlemlerinde aşırı RAM tüketimine karşı `del`, `gc.collect()` ve jeneratörler (`yield`) aktif olarak kullanılmalıdır.
- **Quantization & Hafifletme**: Gerekirse 8-bit / 4-bit kuantizasyon (llama.cpp, ONNX Runtime, OpenVINO veya int8 CPU hızlandırma) tercih edilmelidir.
- **Paralelleştirme**: Intel i5-13500H'nin 12 çekirdek / 16 iş parçacıklı gücünden yararlanmak için `torch.set_num_threads()` ve uygun multiprocessing mekanizmaları yapılandırılmalıdır.

---

## 3. Kodlama & Mimari Standartları
- **Modülerlik**: Tüm AI bileşenleri (veri işleme, model mimarisi, eğitim/çıkarım döngüsü, servis katmanı) bağımsız ve test edilebilir modüller halinde yazılmalıdır.
- **Tip Güvenliği**: Python fonksiyonlarında tip ipuçları (`type hints`) ve docstring açıklamaları zorunludur.
- **Hata Yönetimi & Dayanıklılık**: Beklenmeyen girdiler, eksik ağırlık dosyaları veya donanım limit aşımları için açık hata yakalama (`try-except`) ve açıklayıcı loglama mekanizmaları kurulmalıdır.
- **Konfigürasyon Yönetimi**: Sabit kodlanmış (hardcoded) hiperparametreler yasaktır; tüm konfigürasyonlar YAML, JSON veya `.env` üzerinden yönetilmelidir.

---

## 4. Test & Kalite Güvencesi
- **Birim & Entegrasyon Testleri**: Her modül için `pytest` tabanlı testler hazırlanmalıdır.
- **Determinizm**: Model deneyleri ve testlerinde tekrarlanabilirlik için tohum (`random_seed`) değerleri sabitlenmelidir.
- **Benchmark / Profiling**: Çıkarım (inference) süreleri ve RAM tüketimi her faz sonunda ölçülmeli ve belgelenmelidir.

---

## 5. Süreç Disiplini
- **Canlı Süreç Takibi**: Her alt faz tamamlandığında ve kanıtlandığında [PROCESS.md](file:///c:/Users/ibrah/OneDrive/Desktop/lixai/PROCESS.md) güncellenmelidir.
- **Sürüm Takibi**: Veri setleri ve eğitilen model ağırlıkları için sürümleme ve kontrol adımları uygulanmalıdır.
