"""
Faz 6.4 Doğrulama Testi: 40M SLM Eğitimi & Checkpoint Kararlılığı
"""
import pytest
from pathlib import Path
import torch
from src.engine.interface import LixAI
from src.utils.quantize import get_model_size_mb


def test_40m_slm_training_and_checkpoints(tmp_path):
    # 40M model konfigürasyonu ile LixAI başlat
    ai = LixAI(config_path="configs/model_config.yaml")

    num_params = ai.model.get_num_params()
    print(f"\n[KANIT] Başlatılan Model Parametresi: {num_params:,}")
    assert 38_000_000 <= num_params <= 50_000_000, "Model 38M-50M aralığında olmalıdır."

    # Eğitim adımlarını koştur (HF TinyStories streaming akışı üzerinden)
    ckpt_dir = tmp_path / "checkpoints"
    ai.trainer.checkpoint_dir = ckpt_dir

    print("[KANIT] Hugging Face Akışı Üzerinde 40M SLM Eğitimi Başlatılıyor...")
    results = ai.train(max_steps=5, batch_size=2)

    print(f"[KANIT] Tamamlanan Adım: {results['completed_steps']}")
    print(f"[KANIT] Başlangıç/Ortalama Kayıp (Loss): {results['avg_loss']:.4f}")
    print(f"[KANIT] En İyi Kayıp (Best Loss): {results['best_loss']:.4f}")
    print(f"[KANIT] Toplam Eğitim Süresi: {results['total_time']:.2f} sn")

    assert results["completed_steps"] == 5
    assert results["best_loss"] > 0.0

    # Checkpoint varlık kanıtı
    best_ckpt = ckpt_dir / "best_model.pt"
    final_ckpt = ckpt_dir / "final_model.pt"

    assert best_ckpt.exists(), "best_model.pt diske kaydedilmelidir!"
    assert final_ckpt.exists(), "final_model.pt diske kaydedilmelidir!"

    best_size_mb = round(best_ckpt.stat().st_size / (1024 * 1024), 2)
    print(f"[KANIT] Kaydedilen Checkpoint Boyutu: {best_size_mb} MB")

    # Ağırlık Yükleme ve Çıkarım Kanıtı
    ai.trainer.load_checkpoint("best_model.pt")
    output = ai.generate(prompt="Once upon a time", max_new_tokens=10)
    print(f"[KANIT] Eğitilmiş Modelden Üretilen Metin: '{output}'")
    assert len(output) > 0
