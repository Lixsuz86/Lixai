"""
Faz 8.1 Doğrulama Testi: BPE Tokenizer'ın LixAI, CLI ve Chat Sisteminde Birleştirilmesi (Unified Tokenizer Engine)
"""
import pytest
from pathlib import Path
from src.engine.interface import LixAI
from src.data.bpe_tokenizer import LixBPETokenizer
from src.engine.chat import InteractiveChatSession


def test_lixai_unified_bpe_tokenizer_and_generation():
    # 1. LixAI Başlatma & Tokenizer Tipi Doğrulaması
    ai = LixAI()
    print(f"\n[KANIT] LixAI Tokenizer Sınıfı: {type(ai.tokenizer).__name__}")
    assert isinstance(ai.tokenizer, LixBPETokenizer), "LixAI varsayılan olarak LixBPETokenizer kullanmalıdır!"

    # 2. Türkçe & Çok Dilli Metin Kodlama / Çözme (Encode / Decode Roundtrip)
    test_phrase = "Bir zamanlar bir çocuk ormanda neşeyle koşuyordu."
    encoded = ai.tokenizer.encode(test_phrase)
    decoded = ai.tokenizer.decode(encoded)
    
    print(f"[KANIT] Test Cümlesi: '{test_phrase}'")
    print(f"[KANIT] BPE Token Sayısı: {len(encoded)} token (Token IDs: {encoded[:8]}...)")
    print(f"[KANIT] Geri Çözülen Metin: '{decoded}'")
    assert decoded == test_phrase, "BPE encode-decode kayıpsız çalışmalıdır!"

    # 3. Metin Üretimi ve Kelime Çözümleme Doğrulaması
    prompt = "Bir zamanlar"
    generated = ai.generate(prompt=prompt, max_new_tokens=15, temperature=0.8, use_cache=True)
    print(f"[KANIT] Prompt: '{prompt}'")
    print(f"[KANIT] Üretilen Tam Metin: '{generated}'")
    assert len(generated) > len(prompt), "Üretilen metin prompt'tan uzun olmalı, boşluğa düşmemelidir!"

    # 4. İnteraktif Chat Oturumu ve Streaming Doğrulaması
    chat_session = InteractiveChatSession(
        model=ai.model,
        tokenizer=ai.tokenizer,
        temperature=0.8,
        use_quantized=False
    )
    assert isinstance(chat_session.tokenizer, LixBPETokenizer)

    chunks = []
    for chunk in chat_session.stream_response("Merhaba LixAI", max_tokens=10, delay=0.0):
        chunks.append(chunk)

    streamed_reply = "".join(chunks)
    print(f"[KANIT] Chat Canlı Akış (Streaming) Parçacık Sayısı: {len(chunks)}")
    print(f"[KANIT] Chat Akışından Elde Edilen Yanıt: '{streamed_reply}'")
    assert len(chunks) > 0, "Chat oturumu en az bir parça akıtmalıdır."
    assert len(streamed_reply) > 0, "Chat yanıtı boş olmamalıdır."
