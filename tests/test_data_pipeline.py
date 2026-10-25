"""
Faz 1 Doğrulama Testi: Tokenizer, Veri Hattı ve Tensör Boyutları
"""
import pytest
import torch
from src.data.tokenizer import SimpleCharTokenizer
from src.data.pipeline import TextDataPipeline
from src.data.loader import create_dataloader


def test_tokenizer_encode_decode():
    tokenizer = SimpleCharTokenizer()
    text = "Yapay zeka projesi LixAI başarıyla çalışıyor! 123"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert isinstance(encoded, list)
    assert len(encoded) == len(text)
    assert decoded == text


def test_data_pipeline_batch_shapes():
    tokenizer = SimpleCharTokenizer()
    pipeline = TextDataPipeline(
        dataset_name="roneneldan/TinyStories",
        dataset_config=None,
        split="train",
        seq_len=32,
        tokenizer=tokenizer,
        streaming=True,
    )

    batch_iter = pipeline.stream_batches(batch_size=4, max_batches=2)
    x, y = next(batch_iter)

    assert x.shape == (4, 32), f"Beklenen: (4, 32), Alınan: {x.shape}"
    assert y.shape == (4, 32), f"Beklenen: (4, 32), Alınan: {y.shape}"
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_streaming_loader():
    loader = create_dataloader(
        dataset_name="roneneldan/TinyStories",
        dataset_config=None,
        split="train",
        batch_size=2,
        seq_len=16,
        max_batches=2,
        streaming=True,
    )

    batches = list(loader)
    assert len(batches) == 2
    x0, y0 = batches[0]
    assert x0.shape == (2, 16)
    assert y0.shape == (2, 16)
