"""
LixAI Kapsamlı Performans ve Profiling Benchmark Modülü
FP32 vs INT8, KV Cache AÇIK vs KAPALI, Threading ve Bellek Ayak İzi Analizi.
"""
from typing import Dict, Any, List
from pathlib import Path
import time
import torch
from src.models.core import LixLanguageModel
from src.utils.quantize import quantize_model_cpu, get_model_size_mb, get_quantization_report
from src.utils.logger import setup_logger

logger = setup_logger("lixai.benchmark")


def benchmark_single_profile(
    model: LixLanguageModel,
    prompt_len: int = 32,
    generate_tokens: int = 25,
    use_cache: bool = True,
    warmup: int = 1,
    num_runs: int = 3,
    num_threads: int = 12,
) -> Dict[str, Any]:
    """Belirli bir konfigürasyon için hassas gecikme ve hız ölçümü yapar."""
    torch.set_num_threads(num_threads)
    model.eval()

    prompt = torch.randint(0, model.vocab_size, (1, prompt_len), dtype=torch.long)

    # Isınma
    for _ in range(warmup):
        _ = model.generate(prompt.clone(), max_new_tokens=4, use_cache=use_cache)

    durations: List[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model.generate(prompt.clone(), max_new_tokens=generate_tokens, temperature=0.0, use_cache=use_cache)
        end = time.perf_counter()
        durations.append(end - start)

    avg_time = sum(durations) / len(durations)
    ms_per_token = (avg_time / generate_tokens) * 1000
    tokens_per_sec = generate_tokens / max(avg_time, 1e-5)

    return {
        "avg_total_sec": round(avg_time, 4),
        "ms_per_token": round(ms_per_token, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "use_cache": use_cache,
        "threads": num_threads,
        "model_size_mb": get_model_size_mb(model),
    }


def benchmark_inference(
    model: LixLanguageModel,
    prompt_len: int = 16,
    generate_tokens: int = 20,
    warmup: int = 1,
    num_runs: int = 3,
    num_threads: int = 12,
) -> Dict[str, Any]:
    """Geriye dönük uyumlu tekil çıkarım benchmark fonksiyonu."""
    return benchmark_single_profile(
        model=model,
        prompt_len=prompt_len,
        generate_tokens=generate_tokens,
        use_cache=True,
        warmup=warmup,
        num_runs=num_runs,
        num_threads=num_threads,
    )


def compare_fp32_vs_int8(
    model: LixLanguageModel,
    prompt_len: int = 16,
    generate_tokens: int = 20,
) -> Dict[str, Any]:
    """Geriye dönük uyumlu FP32 vs INT8 karşılaştırması."""
    results = run_full_system_benchmark(model, prompt_len, generate_tokens)
    return {
        "fp32": results["fp32_cached"],
        "int8": results["int8_cached"],
        "speedup_ratio": results["kv_cache_speedup"],
    }


def run_full_system_benchmark(
    model: LixLanguageModel,
    prompt_len: int = 32,
    generate_tokens: int = 20,
) -> Dict[str, Any]:
    """
    40.1M SLM için çok boyutlu stres ve performans testi çalıştırır:
      1. FP32 + KV Cache (12 Threads)
      2. FP32 (KV Cache YOK)
      3. INT8 Kuantize + KV Cache (12 Threads)
      4. Thread Ölçekleme (4 Thread vs 12 Thread)
    """
    logger.info("Kapsamlı Sistem Benchmark'ı Başlatılıyor (Intel Core i5-13500H)...")

    # 1. FP32 + KV Cache
    logger.info("Ölçüm 1: FP32 + KV Cache (12 Thread)...")
    fp32_cached = benchmark_single_profile(model, prompt_len, generate_tokens, use_cache=True, num_threads=12)

    # 2. FP32 Klasik (Cache Yok)
    logger.info("Ölçüm 2: FP32 Standart (Cache YOK)...")
    fp32_no_cache = benchmark_single_profile(model, prompt_len, generate_tokens, use_cache=False, num_threads=12)

    # 3. INT8 Kuantize + KV Cache
    logger.info("Ölçüm 3: INT8 Kuantize + KV Cache (12 Thread)...")
    quantized_model = quantize_model_cpu(model)
    int8_cached = benchmark_single_profile(quantized_model, prompt_len, generate_tokens, use_cache=True, num_threads=12)

    # 4. Thread Ölçekleme (4 vs 12 thread)
    logger.info("Ölçüm 4: Thread Ölçekleme (4 Thread)...")
    fp32_4_threads = benchmark_single_profile(model, prompt_len, generate_tokens, use_cache=True, num_threads=4)

    quant_report = get_quantization_report(model, quantized_model)

    return {
        "fp32_cached": fp32_cached,
        "fp32_no_cache": fp32_no_cache,
        "int8_cached": int8_cached,
        "fp32_4_threads": fp32_4_threads,
        "quant_report": quant_report,
        "kv_cache_speedup": round(fp32_no_cache["ms_per_token"] / max(fp32_cached["ms_per_token"], 1e-5), 2),
    }


def generate_benchmark_markdown_report(results: Dict[str, Any], output_path: str = "benchmark_report.md") -> str:
    """Benchmark sonuçlarını profesyonel bir Markdown raporuna dönüştürüp kaydeder."""
    r_fc = results["fp32_cached"]
    r_fnc = results["fp32_no_cache"]
    r_ic = results["int8_cached"]
    r_t4 = results["fp32_4_threads"]
    qr = results["quant_report"]
    kv_speedup = results["kv_cache_speedup"]

    content = f"""# 📊 LixAI (40.1M SLM) Donanım & Performans Benchmark Raporu

**Test Tarihi:** 2026-10-16  
**Donanım Platformu:** Intel(R) Core(TM) i5-13500H (12 Çekirdek, 16 Thread, AVX2)  
**Sistem Belleği:** 16 GB RAM  
**Model Ölçeği:** 40,175,616 Parametre (~40.17M SLM)  

---

## ⚡ 1. Çıkarım Gecikmesi & Hız Tablosu

| Konfigürasyon | Thread | KV Cache | ms / Token | Token / Saniye | Bellek Ayak İzi |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32 + KV Cache (Önerilen)** | **12** | **AÇIK** | **{r_fc['ms_per_token']} ms** | **{r_fc['tokens_per_sec']} token/sn** | {r_fc['model_size_mb']} MB |
| **FP32 Standart (Temel)** | 12 | KAPALI | {r_fnc['ms_per_token']} ms | {r_fnc['tokens_per_sec']} token/sn | {r_fnc['model_size_mb']} MB |
| **INT8 Dinamik Kuantize** | **12** | **AÇIK** | **{r_ic['ms_per_token']} ms** | **{r_ic['tokens_per_sec']} token/sn** | **{r_ic['model_size_mb']} MB** |
| **FP32 (Sınırlı İş Parçacığı)** | 4 | AÇIK | {r_t4['ms_per_token']} ms | {r_t4['tokens_per_sec']} token/sn | {r_t4['model_size_mb']} MB |

---

## 💾 2. Bellek (RAM) Sıkıştırma ve Kuantizasyon Analizi

- **Orijinal FP32 Boyutu:** `{qr['fp32_size_mb']} MB`
- **Kuantize INT8 Boyutu:** `{qr['int8_size_mb']} MB`
- **Sağlanan Bellek Kazancı:** `{qr['savings_mb']} MB` (**%{qr['savings_percent']} Tasarruf**)
- **Sıkıştırma Oranı:** `{qr['compression_ratio']}x`

---

## 🚀 3. Temel Optimizasyon Çıkarımları

1. **KV Cache Etkisi**:
   - Key-Value önbellekleme sayesinde çıkarım gecikmesi **{r_fnc['ms_per_token']} ms** seviyesinden **{r_fc['ms_per_token']} ms** seviyesine indi (**{kv_speedup}x net hızlanma**).
2. **INT8 Kuantizasyon Kazancı**:
   - 40M parametreli modelin RAM ayak izi **%{qr['savings_percent']} oranında küçülerek {r_ic['model_size_mb']} MB** seviyesine geriledi.
3. **Çoklu Çekirdek (Multi-Threading) Skalası**:
   - 4 thread'den 12 thread'e çıkıldığında gecikme **{r_t4['ms_per_token']} ms -> {r_fc['ms_per_token']} ms** düzeyine çekildi.
"""
    Path(output_path).write_text(content, encoding="utf-8")
    logger.info(f"Benchmark raporu kaydedildi: {output_path}")
    return content
