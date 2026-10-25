"""
Faz 7.1 Doğrulama Testi: 40M SLM Dinamik INT8 Kuantizasyonu & Bellek Tasarrufu
"""
import pytest
import torch
from src.models.core import LixLanguageModel
from src.data.tokenizer import get_default_tokenizer
from src.utils.quantize import quantize_model_cpu, get_model_size_mb, get_quantization_report
from src.utils.config import load_config


def test_40m_slm_dynamic_int8_quantization():
    config = load_config("configs/model_config.yaml")
    m_cfg = config["model"]

    # 1. 40.1M SLM Modelini Başlat
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
    print(f"\n[KANIT] Başlatılan Model Parametresi: {num_params:,}")
    assert num_params > 38_000_000, "Model 38M+ parametre olmalıdır."

    # 2. Dinamik INT8 Kuantizasyonu Uygula
    quantized_model = quantize_model_cpu(model)
    assert quantized_model is not None, "Kuantize model başarıyla oluşturulmalıdır."

    # 3. Bellek Tasarruf Analizi Kanıtı
    report = get_quantization_report(model, quantized_model)
    print(f"[KANIT] Orijinal FP32 Model Boyutu: {report['fp32_size_mb']} MB")
    print(f"[KANIT] Kuantize INT8 Model Boyutu: {report['int8_size_mb']} MB")
    print(f"[KANIT] Sağlanan Bellek Tasarrufu: {report['savings_mb']} MB (%{report['savings_percent']})")
    print(f"[KANIT] Sıkıştırma Oranı: {report['compression_ratio']}x")

    assert report["savings_percent"] >= 40.0, "INT8 kuantizasyon en az %40 bellek tasarrufu sağlamalıdır."

    # 4. Kuantize 40M Model ile Metin Üretimi Doğrulaması
    tokenizer = get_default_tokenizer()
    prompt_tokens = [min(t, m_cfg["vocab_size"] - 1) for t in tokenizer.encode("Once upon a time")]
    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long)

    out_tensor = quantized_model.generate(input_tensor, max_new_tokens=10, temperature=0.8)
    generated_text = tokenizer.decode(out_tensor[0].tolist())

    print(f"[KANIT] Kuantize Model Girdi Tensör Şekli: {input_tensor.shape}")
    print(f"[KANIT] Kuantize Model Çıktı Tensör Şekli: {out_tensor.shape}")
    print(f"[KANIT] Kuantize Modelden Üretilen Metin: '{generated_text}'")

    assert out_tensor.shape[1] == input_tensor.shape[1] + 10, "10 yeni token başarıyla üretilmelidir."
    assert len(generated_text) > 0
