"""
Faz 6.2 Doğrulama Testi: Türkçe & Çok Dilli BPE Tokenizer Eğitimi ve Sıkıştırma Oranı
"""
import pytest
from pathlib import Path
from src.data.bpe_tokenizer import LixBPETokenizer, get_default_bpe_corpus


def test_bpe_tokenizer_training_and_roundtrip(tmp_path):
    tokenizer = LixBPETokenizer()
    corpus = get_default_bpe_corpus()

    save_path = tmp_path / "tokenizer.json"
    tokenizer.train(corpus_texts=corpus, vocab_size=1024, save_path=str(save_path))

    # 1. Sözlük boyutu kanıtı
    vocab_size = tokenizer.vocab_size
    print(f"\n[KANIT] Eğitilen BPE Sözlük Boyutu: {vocab_size}")
    assert vocab_size > 256, "Sözlük boyutu temel karakterlerden büyük olmalıdır."

    # 2. Türkçe ve Özel Karakter Kayıpsız Çeviri (Roundtrip) Kanıtı
    test_sentence_tr = "LixAI; Türkçe karakterleri (ç, ğ, ı, ö, ş, ü, İ) ve ekleri başarıyla işliyor!"
    encoded_ids = tokenizer.encode(test_sentence_tr)
    decoded_text = tokenizer.decode(encoded_ids)

    print(f"[KANIT] Test Metni: '{test_sentence_tr}'")
    print(f"[KANIT] Token Sayısı (BPE): {len(encoded_ids)} token")
    print(f"[KANIT] Karakter Sayısı: {len(test_sentence_tr)} karakter")
    print(f"[KANIT] Decode Edilen Metin: '{decoded_text}'")

    assert decoded_text == test_sentence_tr, "BPE encode-decode kayıpsız olmalıdır!"

    # 3. Sıkıştırma Oranı (Compression Ratio) Kanıtı
    compression_ratio = len(test_sentence_tr) / len(encoded_ids)
    print(f"[KANIT] Sıkıştırma Oranı: {compression_ratio:.2f}x (Karakter/Token)")
    assert compression_ratio > 1.5, "BPE alt-kelimeler sayesinde en az 1.5x sıkıştırma sağlamalıdır."

    # 4. Diske Kaydetme ve Geri Yükleme Doğrulaması
    loaded_tokenizer = LixBPETokenizer.load(str(save_path))
    assert loaded_tokenizer.vocab_size == vocab_size
    re_encoded = loaded_tokenizer.encode(test_sentence_tr)
    assert re_encoded == encoded_ids
    print(f"[KANIT] Disk Kayıt ve Yeniden Yükleme: BAŞARILI")
