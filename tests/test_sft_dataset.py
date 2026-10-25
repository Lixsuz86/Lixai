"""
Faz 9.1 Doğrulama Testi: İki Dilli (Türkçe & İngilizce) SFT Veri Seti ve DataLoader Hattı
"""
import pytest
from pathlib import Path
import json
import torch
from src.data.sft_loader import create_sft_dataloader
from src.data.bpe_tokenizer import LixBPETokenizer


def test_sft_dataset_file_and_streaming_dataloader():
    jsonl_path = "data/dialogue_sft.jsonl"
    assert Path(jsonl_path).exists(), f"{jsonl_path} dosyası mevcut olmalıdır!"

    # 1. Dosyadaki Toplam Örnek Sayısını Doğrula (En az 2,000)
    line_count = 0
    sample_dialogue = None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                line_count += 1
                if sample_dialogue is None:
                    sample_dialogue = json.loads(line)

    print(f"\n[KANIT] SFT Veri Seti Toplam Diyalog Sayısı: {line_count:,}")
    print(f"[KANIT] Örnek Diyalog Yapısı:")
    print(json.dumps(sample_dialogue, ensure_ascii=False, indent=2))

    assert line_count >= 2000, f"SFT veri seti en az 2,000 diyalog içermelidir, bulunan: {line_count}"
    assert "messages" in sample_dialogue
    assert len(sample_dialogue["messages"]) >= 2

    # 2. SFT DataLoader Başlatma & Batch Tensör Boyutları
    tok = LixBPETokenizer.load("configs/tokenizer.json")
    batch_size = 4
    seq_len = 128
    loader = create_sft_dataloader(
        file_path=jsonl_path,
        tokenizer=tok,
        batch_size=batch_size,
        seq_len=seq_len,
        repeat=False,
    )

    batch_iter = iter(loader)
    batch_x, batch_y = next(batch_iter)

    print(f"\n[KANIT] Girdi Batch Şekli (input_ids): {batch_x.shape} | Dtype: {batch_x.dtype}")
    print(f"[KANIT] Hedef Batch Şekli (targets):   {batch_y.shape} | Dtype: {batch_y.dtype}")

    assert batch_x.shape == (batch_size, seq_len)
    assert batch_y.shape == (batch_size, seq_len)

    # 3. Maskeleme (-100) Doğrulaması (Target Loss Masking)
    # Prompt kısımları -100 olmalı, asistan kısımları pozitif token ID olmalıdır.
    first_target = batch_y[0].tolist()
    masked_count = sum(1 for t in first_target if t == -100)
    unmasked_count = sum(1 for t in first_target if t != -100)

    print(f"[KANIT] İlk Örnek Hedef Tensöründeki Maskeli Token Sayısı (-100): {masked_count}")
    print(f"[KANIT] İlk Örnek Hedef Tensöründeki Eğitilebilir Asistan Tokeni: {unmasked_count}")

    assert masked_count > 0, "Prompt kısımları -100 ile maskelenmelidir."
    assert unmasked_count > 0, "Asistan kısımları kayıp (loss) hesaplamak üzere açık kalmalıdır."

    # 4. Çoklu Batch Akışı Doğrulaması (Hafıza / Hız Sızıntısı Yok)
    total_samples = 0
    for idx, (bx, by) in enumerate(loader):
        total_samples += bx.size(0)
        if idx >= 5:  # İlk 6 batch yeterli
            break

    print(f"[KANIT] Başarıyla Akıtılan Örnek Sayısı: {total_samples}")
    assert total_samples == batch_size * 6
