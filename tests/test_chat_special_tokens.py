"""
Faz 8.2 Doğrulama Testi: Özel Chat & Diyalog Tokenleri ve ChatML Şablonu
"""
import pytest
from src.data.bpe_tokenizer import LixBPETokenizer, format_chat_prompt


def test_special_chat_tokens_and_chatml_formatting():
    # 1. BPE Tokenizer'ı Yükle
    tok = LixBPETokenizer.load("configs/tokenizer.json")

    # 2. Özel Chat Tokenlerinin Varlığını ve Tekil ID Aldığını Doğrula
    special_names = ["<|system|>", "<|user|>", "<|assistant|>", "<|endoftext|>"]
    token_ids = {}
    for name in special_names:
        tid = tok.token_to_id(name)
        assert tid is not None, f"Özel token {name} sözlükte bulunmalıdır!"
        token_ids[name] = tid
        
        # Tekil token olarak kodlandığını doğrula (1 kelime parçası = 1 token ID)
        encoded = tok.encode(name)
        assert encoded == [tid], f"{name} tek bir token olarak kodlanmalıdır, çıkan: {encoded}"
        
        # ID'den metne geri çözümleme
        reversed_name = tok.id_to_token(tid)
        assert reversed_name == name, f"ID {tid} metin olarak {name} dönmelidir."

    print("\n[KANIT] === ÖZEL CHAT & KONTROL TOKENLERİ ===")
    for name, tid in token_ids.items():
        print(f"[KANIT] • {name.ljust(16)} -> Token ID: {tid}")

    # ID'lerin birbirinden farklı olduğunu doğrula
    assert len(set(token_ids.values())) == len(special_names), "Her özel token farklı bir ID almalıdır."

    # 3. ChatML Şablon Yardımcısını Test Et
    sample_dialogue = [
        {"role": "system", "content": "Sen Intel CPU için geliştirilmiş LixAI asistanısın."},
        {"role": "user", "content": "Merhaba, bugün hava nasıl?"},
        {"role": "assistant", "content": "Merhaba! Ben bir yapay zeka modeliyim, hava durumunu anlık bilemem ancak sorularınıza yardımcı olabilirim."},
        {"role": "user", "content": "Bana bir hikaye anlat."},
    ]

    formatted_prompt = format_chat_prompt(sample_dialogue, add_generation_prompt=True)
    print("\n[KANIT] === BİÇİMLENDİRİLMİŞ CHATML DİYALOG METNİ ===")
    print(formatted_prompt)

    assert "<|system|>\nSen Intel CPU için geliştirilmiş LixAI asistanısın.\n<|endoftext|>" in formatted_prompt
    assert "<|user|>\nMerhaba, bugün hava nasıl?\n<|endoftext|>" in formatted_prompt
    assert formatted_prompt.endswith("<|assistant|>\n")

    # 4. Biçimlendirilmiş Diyaloğun Tokenize Edilmesi ve Geri Çözülmesi
    encoded_dialogue = tok.encode(formatted_prompt)
    print(f"\n[KANIT] Diyalog Toplam BPE Token Sayısı: {len(encoded_dialogue)} token")
    print(f"[KANIT] İlk 10 Token ID: {encoded_dialogue[:10]}")

    # Özel tokenleri koruyarak geri çöz
    decoded_with_specials = tok.decode(encoded_dialogue, skip_special_tokens=False)
    assert "<|system|>" in decoded_with_specials
    assert "<|assistant|>" in decoded_with_specials
    assert "<|endoftext|>" in decoded_with_specials

    # Özel tokenleri filtreleyerek temiz metin elde et
    clean_text = tok.decode(encoded_dialogue, skip_special_tokens=True)
    assert "<|system|>" not in clean_text
    assert "Sen Intel CPU için geliştirilmiş LixAI asistanısın." in clean_text
    print(f"[KANIT] Özel tokensiz temiz metin:\n{clean_text[:120]}...")
