"""
Faz 2.2 & 2.3 Doğrulama Testi: Eğitim Döngüsü, Checkpoint Kayıt ve Geri Yükleme
"""
import pytest
import torch
from pathlib import Path
from src.models.core import LixLanguageModel
from src.data.loader import create_dataloader
from src.engine.runner import Trainer


def test_training_loop_and_checkpoint(tmp_path):
    model = LixLanguageModel(
        vocab_size=120,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_seq_len=32,
    )

    trainer = Trainer(
        model=model,
        learning_rate=1e-3,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        num_threads=4,
    )

    from src.data.tokenizer import SimpleCharTokenizer
    tokenizer = SimpleCharTokenizer()

    loader = create_dataloader(
        dataset_name="roneneldan/TinyStories",
        dataset_config=None,
        split="train",
        batch_size=2,
        seq_len=16,
        tokenizer=tokenizer,
        max_batches=5,
        streaming=True,
    )

    results = trainer.train_steps(
        dataloader=loader,
        max_steps=3,
        log_interval=1,
        save_interval=2,
    )

    assert results["completed_steps"] == 3
    assert results["avg_loss"] > 0.0

    # Checkpoint varlığını doğrula
    ckpt_file = Path(tmp_path / "checkpoints" / "checkpoint_step_2.pt")
    assert ckpt_file.exists()

    # Checkpoint geri yükleme testi
    new_model = LixLanguageModel(
        vocab_size=120,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_seq_len=32,
    )
    new_trainer = Trainer(model=new_model, checkpoint_dir=str(tmp_path / "checkpoints"))
    new_trainer.load_checkpoint("checkpoint_step_2.pt")
