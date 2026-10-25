"""
Faz 2 Doğrulama Testi: Transformer Mimarisi, İleri Geçiş ve Metin Üretimi
"""
import pytest
import torch
from src.models.core import LixLanguageModel


def test_model_initialization_and_param_count():
    model = LixLanguageModel(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=64,
        dropout=0.0
    )
    num_params = model.get_num_params()
    assert num_params > 0
    # CPU ve bellek dostu ölçek kontrolü (< 1M bu mini test için)
    assert num_params < 1_000_000


def test_model_forward_pass_and_loss():
    vocab_size = 100
    batch_size = 2
    seq_len = 16

    model = LixLanguageModel(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=32,
    )

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, loss = model(x, targets=y)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss is not None
    assert loss.item() > 0.0

    # Geriye yayılım (Backpropagation) testi
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradyan bulunamadı: {name}"


def test_model_text_generation():
    vocab_size = 50
    model = LixLanguageModel(
        vocab_size=vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_seq_len=64,
    )
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=5, temperature=1.0)

    assert generated.shape == (1, 8)  # 3 başlangıç + 5 üretilen
