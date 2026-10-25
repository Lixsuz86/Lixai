"""
Faz 0 Doğrulama Testi: Ortam, CPU Tensör Motoru ve Temel Kütüphaneler
"""
import pytest
import numpy as np
import yaml
from pathlib import Path


def test_numpy_operations():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    dot_product = np.dot(a, b)
    assert dot_product == 32.0


def test_yaml_config_loading():
    config_path = Path("configs/default.yaml")
    assert config_path.exists(), "configs/default.yaml bulunamadı!"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["system"]["device"] == "cpu"
    assert cfg["system"]["num_threads"] >= 1


def test_pytorch_cpu_tensor_and_threading():
    import torch

    # Thread ayar testi (Intel i5-13500H 12 çekirdek)
    torch.set_num_threads(12)
    assert torch.get_num_threads() == 12

    # CPU tensör oluşturma ve matris çarpım testi
    x = torch.randn(128, 64)
    w = torch.randn(64, 32)
    y = torch.matmul(x, w)

    assert y.shape == (128, 32)
    assert not y.requires_grad
    assert not torch.cuda.is_available(), "Harici CUDA olmaması CPU modunun teyididir"
