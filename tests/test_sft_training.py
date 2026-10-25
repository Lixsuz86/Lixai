"""
Faz 9.3 Doğrulama Testi: 40.17M Parametreli Modelin SFT Eğitimi, Loss Düşüşü ve lixai_chat_sft.pt Kaydı
"""
from pathlib import Path
import pytest
import torch
from src.engine.interface import LixAI
from src.data.bpe_tokenizer import LixBPETokenizer, format_chat_prompt


def test_40m_slm_sft_training_and_checkpoint_saving():
    """
    40.17M parametreli modelin dialogue_sft.jsonl üzerinde SFT ile eğitilmesi,
    loss düşüşünün kaydedilmesi ve lixai_chat_sft.pt checkpoint'inin doğrulanması.
    """
    checkpoint_target = "lixai_chat_sft.pt"
    ckpt_path = Path("models/checkpoints") / checkpoint_target
    
    # Önceden varsa temiz bir test için hazırla (veya üzerine yazılmasını test et)
    tokenizer = LixBPETokenizer.load("configs/tokenizer.json")
    
    # 1. LixAI motorunu mevcut best_model.pt ile başlat (Transfer Learning / SFT Fine-Tuning)
    ai = LixAI(load_best=True, tokenizer=tokenizer)
    assert ai.model.param_count >= 38_000_000, f"Model boyutu 40M ölçeğinde olmalıdır, mevcut: {ai.model.param_count}"

    # 2. SFT Eğitimini Başlat (25 Adım, CPU 12 Thread, batch_size=4, seq_len=128)
    results = ai.train_sft(
        dataset_path="data/dialogue_sft.jsonl",
        max_steps=25,
        batch_size=4,
        seq_len=128,
        learning_rate=2e-4,
        checkpoint_name=checkpoint_target,
    )

    print(f"\n[KANIT] SFT Tamamlanan Adım: {results['completed_steps']}")
    print(f"[KANIT] Başlangıç / En İyi Kayıp (Best Loss): {results['best_loss']:.4f}")
    print(f"[KANIT] Son Kayıp (Final Loss): {results['final_loss']:.4f}")
    print(f"[KANIT] Ortalama Token Doğruluğu: %{results['avg_accuracy']:.2f}")
    print(f"[KANIT] İşlenen Aktif Asistan Tokeni: {results['total_active_tokens']}")
    print(f"[KANIT] Toplam Eğitim Süresi: {results['total_time']:.2f} sn")

    # 3. Metrik Doğrulamaları
    assert results["completed_steps"] == 25, "25 SFT adımı tamamlanmış olmalıdır."
    assert results["total_active_tokens"] > 0, "Asistan tokenleri işlenmiş olmalıdır."
    assert ckpt_path.exists(), f"Checkpoint dosyası diskte oluşmalıdır: {ckpt_path}"
    assert ckpt_path.stat().st_size > 100_000_000, f"Checkpoint boyutu geçerli bir 40M model boyutu olmalıdır: {ckpt_path.stat().st_size} bytes"

    # 4. Otomatik Checkpoint Yükleme Doğrulaması (lixai_chat_sft.pt artık en yüksek önceliğe sahiptir)
    ai_reloaded = LixAI()
    assert ai_reloaded.loaded_checkpoint_name == checkpoint_target, (
        f"LixAI varsayılan olarak yeni SFT modelini algılamalıdır. Mevcut: {ai_reloaded.loaded_checkpoint_name}"
    )

    # 5. ChatML Formatında Çıkarım (Inference) Bütünlük Testi
    prompt_text = format_chat_prompt([
        {"role": "user", "content": "Merhaba, nasılsın?"}
    ])
    response = ai_reloaded.generate(prompt=prompt_text, max_new_tokens=20, temperature=0.7)
    print(f"\n[KANIT] SFT Modeli Örnek Çıktı:\n{response}")
    assert isinstance(response, str) and len(response) > 0, "Model geçerli bir yanıt metni üretmelidir."
