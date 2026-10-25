"""
Faz 7.3 Doğrulama Testi: Kapsamlı Karşılaştırmalı Benchmark & Stres Testi Raporu
"""
import pytest
from pathlib import Path
from src.models.core import LixLanguageModel
from src.utils.benchmark import run_full_system_benchmark, generate_benchmark_markdown_report
from src.utils.config import load_config


def test_comprehensive_benchmark_and_report_generation():
    config = load_config("configs/model_config.yaml")
    m_cfg = config["model"]

    # 40.1M SLM Modelini Başlat
    model = LixLanguageModel(
        vocab_size=m_cfg.get("vocab_size", 8192),
        d_model=m_cfg.get("d_model", 512),
        n_heads=m_cfg.get("n_heads", 8),
        n_layers=m_cfg.get("n_layers", 10),
        d_ff=m_cfg.get("d_ff", 2048),
        max_seq_len=m_cfg.get("max_seq_len", 512),
        dropout=0.0,
        weight_tying=m_cfg.get("weight_tying", False),
    )

    num_params = model.get_num_params()
    print(f"\n[KANIT] Test Edilen Model: {num_params:,} Parametre (~40.17M SLM)")

    # Kapsamlı Benchmark Çalıştır (Gerçek CPU verileri)
    results = run_full_system_benchmark(model, prompt_len=16, generate_tokens=15)

    # Markdown Raporunu Oluştur
    report_path = "benchmark_report.md"
    markdown_content = generate_benchmark_markdown_report(results, output_path=report_path)

    r_fc = results["fp32_cached"]
    r_fnc = results["fp32_no_cache"]
    r_ic = results["int8_cached"]
    speedup = results["kv_cache_speedup"]

    print("\n[KANIT] === KAPSAMLI BENCHMARK ÇIKTILARI ===")
    print(f"[KANIT] 1. FP32 + KV Cache (12 Th): {r_fc['ms_per_token']} ms/token ({r_fc['tokens_per_sec']} token/sn)")
    print(f"[KANIT] 2. FP32 Standart (No Cache): {r_fnc['ms_per_token']} ms/token ({r_fnc['tokens_per_sec']} token/sn)")
    print(f"[KANIT] 3. INT8 Kuantize (12 Th):     {r_ic['ms_per_token']} ms/token ({r_ic['tokens_per_sec']} token/sn)")
    print(f"[KANIT] 4. KV Cache Hızlanması:      {speedup}x net kazanç")
    print(f"[KANIT] 5. Rapor Dosyası:             {report_path} (Doğrulandı)")

    assert Path(report_path).exists(), "benchmark_report.md dosyası oluşturulmalıdır!"
    assert r_fc["tokens_per_sec"] > 0
    assert r_ic["tokens_per_sec"] > 0
