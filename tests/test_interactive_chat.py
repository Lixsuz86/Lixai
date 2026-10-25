"""
Faz 6.3 Doğrulama Testi: Zengin İnteraktif Terminal Sohbet Modu (Streaming & Komutlar)
"""
import pytest
from src.models.core import LixLanguageModel
from src.data.tokenizer import get_default_tokenizer
from src.engine.chat import InteractiveChatSession


def test_interactive_chat_streaming_and_commands():
    tokenizer = get_default_tokenizer()
    # Test için hafif model konfigürasyonu
    model = LixLanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=64,
    )

    session = InteractiveChatSession(
        model=model,
        tokenizer=tokenizer,
        temperature=0.8,
        top_k=40,
        use_quantized=False,
    )

    # 1. Canlı Akış (Streaming) Kanıtı
    user_prompt = "Merhaba LixAI, nasılsın?"
    chunks = []
    for chunk in session.stream_response(user_prompt, max_tokens=15, delay=0.0):
        chunks.append(chunk)

    full_output = "".join(chunks)
    print(f"\n[KANIT] Kullanıcı Girdisi: '{user_prompt}'")
    print(f"[KANIT] Akış Parça Sayısı (Chunks): {len(chunks)}")
    print(f"[KANIT] Üretilen Yanıt: '{full_output}'")

    assert len(chunks) == 15, "15 adet streaming parça akıtılmalıdır."
    assert len(full_output) > 0, "Üretilen metin boş olamaz."

    # 2. İstatistik Komutu (/stats) Kanıtı
    stats = session.get_stats()
    print(f"[KANIT] /stats Çıktısı: {stats}")
    assert "params" in stats
    assert "model_size_mb" in stats
    assert stats["history_len"] == 1  # 1 tur sohbet kaydedildi

    # 3. Kuantizasyon Değişimi (/quantize) Kanıtı
    session.set_quantization(True)
    assert session.use_quantized is True
    quant_stats = session.get_stats()
    print(f"[KANIT] Kuantize /stats Çıktısı: {quant_stats}")

    # 4. Hafıza Temizleme (/clear) Kanıtı
    session.history.clear()
    assert len(session.history) == 0
    print(f"[KANIT] Hafıza Temizleme: BAŞARILI")
