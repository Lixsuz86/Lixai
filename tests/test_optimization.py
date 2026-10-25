"""
Faz 3 Doğrulama Testi: INT8 Kuantizasyon, CPU Threading ve Profiling
"""
import pytest
import torch
from src.models.core import LixLanguageModel
from src.utils.quantize import quantize_model_cpu, get_model_size_mb
from src.utils.benchmark import benchmark_inference


def test_quantization_and_generation():
    model = LixLanguageModel(
        vocab_size=80,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_seq_len=32,
    )
    model.eval()

    orig_size = get_model_size_mb(model)
    quant_model = quantize_model_cpu(model)

    prompt = torch.randint(0, 80, (1, 8), dtype=torch.long)
    output = quant_model.generate(prompt, max_new_tokens=4)

    assert output.shape == (1, 12)


def test_benchmark_runner():
    model = LixLanguageModel(
        vocab_size=80,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_seq_len=32,
    )

    stats = benchmark_inference(
        model=model,
        prompt_len=8,
        generate_tokens=4,
        warmup=1,
        num_runs=2,
        num_threads=4,
    )

    assert "ms_per_token" in stats
    assert "tokens_per_sec" in stats
    assert stats["tokens_per_sec"] > 0
