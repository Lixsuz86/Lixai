"""
Faz 9.2 Doğrulama Testi: Maskelenmiş Hedef Kaybı (Target-Masked Cross Entropy) ve SFT Trainer
"""
import pytest
import torch
import torch.nn.functional as F
from src.models.core import LixLanguageModel
from src.engine.sft_runner import SFTTrainer
from src.data.sft_loader import create_sft_dataloader
from src.data.bpe_tokenizer import LixBPETokenizer


def test_target_loss_masking_mathematical_proof():
    # 1. Matematiksel Maskeleme Kanıtı: Prompt pozisyonlarındaki değişimler loss'u etkilememelidir.
    vocab_size = 100
    seq_len = 10
    batch_size = 2

    logits1 = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    # targets: ilk 6 token prompt (-100), son 4 token asistan (pozitif indeks)
    targets = torch.tensor([
        [-100, -100, -100, -100, -100, -100, 15, 25, 35, 45],
        [-100, -100, -100, -100, -100, -100, 10, 20, 30, 40],
    ], dtype=torch.long)

    loss1 = F.cross_entropy(logits1.view(-1, vocab_size), targets.view(-1), ignore_index=-100)
    loss1.backward()

    # Maskelenen indekslerde gradyanın KESİNLİKLE SIFIR olduğunu doğrula
    grad_prompt = logits1.grad[:, :6, :]
    grad_assistant = logits1.grad[:, 6:, :]

    print(f"\n[KANIT] Maskelenen Prompt Pozisyonlarındaki Gradyan Normu: {grad_prompt.abs().sum().item()}")
    print(f"[KANIT] Eğitilebilir Asistan Pozisyonlarındaki Gradyan Normu: {grad_assistant.abs().sum().item()}")

    assert grad_prompt.abs().sum().item() == 0.0, "Prompt tokenlerine kesinlikle gradyan akmamalıdır!"
    assert grad_assistant.abs().sum().item() > 0.0, "Asistan tokenleri gradyan almalıdır!"


def test_sft_trainer_gradient_step_and_accuracy(tmp_path):
    # 2. SFT Trainer ile Gerçek Model Üzerinde Adım ve Doğruluk Testi
    tokenizer = LixBPETokenizer.load("configs/tokenizer.json")
    model = LixLanguageModel(
        vocab_size=8192,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=128,
        dropout=0.0,
    )

    trainer = SFTTrainer(
        model=model,
        learning_rate=1e-3,
        checkpoint_dir=str(tmp_path),
        num_threads=4,
    )

    dataloader = create_sft_dataloader(
        file_path="data/dialogue_sft.jsonl",
        tokenizer=tokenizer,
        batch_size=2,
        seq_len=64,
        repeat=True,
    )

    results = trainer.train_steps(
        dataloader=dataloader,
        max_steps=5,
        log_interval=1,
        checkpoint_name="test_sft.pt",
    )

    print("\n[KANIT] === SFT TRAINER METRİKLERİ ===")
    print(f"[KANIT] Tamamlanan Adım: {results['completed_steps']}")
    print(f"[KANIT] Son SFT Kaybı (Loss): {results['final_loss']:.4f}")
    print(f"[KANIT] Ortalama Token Doğruluk: %{results['avg_accuracy']:.2f}")
    print(f"[KANIT] İşlenen Aktif Asistan Tokeni: {results['total_active_tokens']}")

    assert results["completed_steps"] == 5
    assert results["final_loss"] > 0.0
    assert results["total_active_tokens"] > 0
    assert (tmp_path / "test_sft.pt").exists(), "Checkpoint dosyası kaydedilmelidir."
