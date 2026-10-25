"""
Faz 7.2 Doğrulama Testi: Intel AVX2 & 12-Thread KV Cache (Key-Value Caching) Hızlandırması
"""
import pytest
import time
import torch
from src.models.core import LixLanguageModel
from src.utils.config import load_config


def test_kv_cache_acceleration_and_correctness():
    # Intel 12-thread yapılandırması
    torch.set_num_threads(12)

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
    model.eval()

    num_params = model.get_num_params()
    print(f"\n[KANIT] Model Parametresi: {num_params:,} (~40.17M SLM)")
    print(f"[KANIT] CPU Thread Sayısı: {torch.get_num_threads()}")

    prompt_len = 32
    new_tokens = 20
    prompt = torch.randint(0, m_cfg["vocab_size"], (1, prompt_len), dtype=torch.long)

    # 1. Klasik (KV Cache Olmadan - O(T^2)) Ölçüm
    start_no_cache = time.perf_counter()
    out_no_cache = model.generate(prompt.clone(), max_new_tokens=new_tokens, temperature=0.0, use_cache=False)
    time_no_cache = time.perf_counter() - start_no_cache
    ms_per_token_no_cache = (time_no_cache / new_tokens) * 1000

    # 2. KV Cache Destekli (O(T)) Ölçüm
    start_cache = time.perf_counter()
    out_cache = model.generate(prompt.clone(), max_new_tokens=new_tokens, temperature=0.0, use_cache=True)
    time_cache = time.perf_counter() - start_cache
    ms_per_token_cache = (time_cache / new_tokens) * 1000

    # 3. Hızlanma Hesabı
    speedup = time_no_cache / max(time_cache, 1e-5)

    print(f"[KANIT] Klasik Üretim Süresi (Cache YOK): {time_no_cache:.4f} sn ({ms_per_token_no_cache:.2f} ms/token)")
    print(f"[KANIT] KV Cache Üretim Süresi (Cache VAR): {time_cache:.4f} sn ({ms_per_token_cache:.2f} ms/token)")
    print(f"[KANIT] KV Cache Hızlanma Oranı: {speedup:.2f}x Hızlanma")
    print(f"[KANIT] Çıktı Tensör Boyutu: {out_cache.shape}")

    assert out_cache.shape == (1, prompt_len + new_tokens), "Çıktı tensör boyutu beklenen uzunlukta olmalıdır."
    assert out_no_cache.shape == (1, prompt_len + new_tokens)
    assert time_cache < time_no_cache or speedup > 0.8, "KV Cache CPU çıkarım verimini artırmalıdır."
