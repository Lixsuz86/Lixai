"""
Faz 8.3 Doğrulama Testi: Otomatik Ağırlık Yükleme (Smart Checkpoint Loading) & Tokenizer Bütünlük Testi
"""
import pytest
from pathlib import Path
from src.engine.interface import LixAI
from src.engine.chat import InteractiveChatSession


def test_smart_checkpoint_auto_load_and_stats():
    # 1. Varsayılan Başlatma (Herhangi bir bayrak verilmeden otomatik en iyi ağırlık tespiti)
    ai_auto = LixAI()
    print(f"\n[KANIT] Otomatik Yüklenen Kontrol Noktası: {ai_auto.loaded_checkpoint_name}")
    assert ai_auto.loaded_checkpoint_name in ("best_model.pt", "final_model.pt"), (
        "Diskte model varken LixAI otomatik olarak en iyi ağırlığı tespit edip yüklemelidir!"
    )

    # 2. Açıkça Ağırlık Yüklemeyi Kapatma (load_best=False)
    ai_fresh = LixAI(load_best=False)
    print(f"[KANIT] Sıfır/Rastgele Başlatılan Model Kontrol Noktası: {ai_fresh.loaded_checkpoint_name}")
    assert ai_fresh.loaded_checkpoint_name is None, "load_best=False verildiğinde checkpoint yüklenmemelidir."

    # 3. Belirli Bir Checkpoint Dosyası İsteme (checkpoint_name="final_model.pt")
    if Path("models/checkpoints/final_model.pt").exists():
        ai_custom = LixAI(checkpoint_name="final_model.pt")
        print(f"[KANIT] Belirtilen Checkpoint Yükleme: {ai_custom.loaded_checkpoint_name}")
        assert ai_custom.loaded_checkpoint_name == "final_model.pt"

    # 4. Chat Oturumu İstatistiklerinde Checkpoint Bilgisinin Doğrulanması
    chat_session = InteractiveChatSession(
        model=ai_auto.model,
        tokenizer=ai_auto.tokenizer,
        checkpoint_name=ai_auto.loaded_checkpoint_name,
    )
    stats = chat_session.get_stats()
    print(f"[KANIT] Chat /stats Çıktısı Checkpoint Alanı: '{stats['checkpoint']}'")
    assert stats["checkpoint"] == ai_auto.loaded_checkpoint_name

    # 5. Otomatik Yüklenmiş Model ile Metin Üretimi Doğrulaması
    prompt = "Bir zamanlar"
    generated_text = ai_auto.generate(prompt=prompt, max_new_tokens=15, temperature=0.8)
    print(f"[KANIT] Otomatik Yüklenen Modelden Üretilen Metin: '{generated_text}'")
    assert len(generated_text) > len(prompt), "Metin üretimi başarılı bir şekilde prompt sonrasını tamamlamalıdır."
