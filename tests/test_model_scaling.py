"""
Faz 6.1 Doğrulama Testi: 38M - 50M Parametreli SLM Mimarisi Ölçekleme
"""
import pytest
import torch
from src.models.core import LixLanguageModel
from src.utils.quantize import get_model_size_mb
from src.utils.config import load_config


def test_model_scaling_38m_to_50m():
    config = load_config("configs/model_config.yaml")
    m_cfg = config["model"]

    model = LixLanguageModel(
        vocab_size=m_cfg.get("vocab_size", 8192),
        d_model=m_cfg.get("d_model", 512),
        n_heads=m_cfg.get("n_heads", 8),
        n_layers=m_cfg.get("n_layers", 8),
        d_ff=m_cfg.get("d_ff", 2048),
        max_seq_len=m_cfg.get("max_seq_len", 512),
        dropout=m_cfg.get("dropout", 0.1),
        weight_tying=m_cfg.get("weight_tying", False),
    )

    num_params = model.get_num_params()
    size_mb = get_model_size_mb(model)

    print(f"\n[KANIT] Toplam Model Parametresi: {num_params:,}")
    print(f"[KANIT] Model Bellek Boyutu (FP32): {size_mb} MB")

    # 38M - 50M Parametre Hedefi Doğrulaması
    assert 38_000_000 <= num_params <= 50_000_000, (
        f"Parametre sayısı hedef aralıkta değil! Alınan: {num_params:,}"
    )

    # İleri Geçiş ve Tensör Boyut Doğrulaması
    batch_size = 2
    seq_len = 64
    x = torch.randint(0, m_cfg["vocab_size"], (batch_size, seq_len), dtype=torch.long)
    targets = torch.randint(0, m_cfg["vocab_size"], (batch_size, seq_len), dtype=torch.long)

    logits, loss = model(x, targets=targets)

    assert logits.shape == (batch_size, seq_len, m_cfg["vocab_size"])
    assert loss is not None
    assert loss.item() > 0.0

    # Gradyan Akışı (Backward pass)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradyan eksik: {name}"

    print(f"[KANIT] Logits Şekli: {logits.shape}")
    print(f"[KANIT] Başlangıç Loss: {loss.item():.4f}")
    print(f"[KANIT] Gradyan Akışı: BAŞARILI")
