"""
Faz 5 Doğrulama Testi: Uçtan Uca (End-to-End) Entegrasyon ve Dayanıklılık
"""
import pytest
from src.engine.interface import LixAI


def test_e2e_pipeline_and_generation():
    ai = LixAI()
    prompt = "Once upon a time"

    # FP32 çıkarım
    output_fp32 = ai.generate(prompt=prompt, max_new_tokens=10, temperature=0.7)
    assert isinstance(output_fp32, str)
    assert len(output_fp32) >= len(prompt)

    # INT8 Kuantize çıkarım
    output_int8 = ai.generate(prompt=prompt, max_new_tokens=10, temperature=0.7, use_quantized=True)
    assert isinstance(output_int8, str)
    assert len(output_int8) >= len(prompt)
