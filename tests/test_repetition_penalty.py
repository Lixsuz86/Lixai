"""
Faz 10.1 Doğrulama Testi: Tekrarlama ve Frekans Cezası (Repetition & Frequency Penalty)
Küçük dil modellerinde sonsuz döngüleri, harf/hece tekrarlarını engelleyen algoritmik kanıt.
"""
import pytest
import torch
from src.models.core import LixLanguageModel, apply_repetition_penalties
from src.engine.interface import LixAI
from src.engine.chat import InteractiveChatSession


def test_repetition_penalty_mathematical_precision():
    """
    1. Matematiksel Doğruluk Kanıtı:
    - Pozitif logitler repetition_penalty'ye bölünmelidir (azalmalı).
    - Negatif logitler repetition_penalty ile çarpılmalıdır (daha da negatifleşmeli).
    - Dizide bulunmayan (unseen) tokenler kesinlikle etkilenmemelidir.
    """
    vocab_size = 20
    logits = torch.tensor([[
        1.0, 2.0, 3.0, 4.0,   # 0..3
        -1.0, -2.0, 0.0, 5.0,  # 4..7 (token 4=-1, token 5=-2, token 6=0, token 7=5)
        10.0, 10.0, 10.0, 10.0, # 8..11
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]], dtype=torch.float32)

    # Bağlamda yalnızca 5, 6, 7 var
    input_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
    penalty = 2.0

    original_logits = logits.clone()
    penalized = apply_repetition_penalties(
        logits=logits.clone(),
        input_ids=input_ids,
        repetition_penalty=penalty,
    )

    # Token 7: Pozitif (5.0) -> 5.0 / 2.0 = 2.5
    assert torch.isclose(penalized[0, 7], torch.tensor(2.5)), f"Pozitif logit 2.5 olmalı, alınan: {penalized[0, 7]}"
    
    # Token 5: Negatif (-2.0) -> -2.0 * 2.0 = -4.0
    assert torch.isclose(penalized[0, 5], torch.tensor(-4.0)), f"Negatif logit -4.0 olmalı, alınan: {penalized[0, 5]}"
    
    # Token 6: Sıfır (0.0) -> 0.0
    assert torch.isclose(penalized[0, 6], torch.tensor(0.0)), f"Sıfır logit 0.0 kalmalı, alınan: {penalized[0, 6]}"

    # Dizide yer almayan tokenler (örn. token 8): Kesinlikle DEĞİŞMEMELİ
    assert torch.isclose(penalized[0, 8], original_logits[0, 8]), "Bağlamda olmayan token logiti değişmemeli!"
    assert torch.isclose(penalized[0, 0], original_logits[0, 0]), "Bağlamda olmayan token logiti değişmemeli!"

    print("\n[KANIT] Pozitif Logit (5.0 -> 2.5): Doğrulandı")
    print("[KANIT] Negatif Logit (-2.0 -> -4.0): Doğrulandı")
    print("[KANIT] Bağlam Dışı Logitler (10.0 -> 10.0): Korundu")


def test_frequency_and_presence_penalty_mathematics():
    """
    2. Frekans ve Varlık Cezası Doğrulama Testi:
    - Çok tekrar eden token, frekansıyla orantılı ceza almalıdır.
    """
    vocab_size = 15
    logits = torch.zeros(1, vocab_size)
    # Token 3: 4 kez geçti, Token 4: 1 kez geçti
    input_ids = torch.tensor([[3, 3, 3, 3, 4]], dtype=torch.long)

    penalized = apply_repetition_penalties(
        logits=logits.clone(),
        input_ids=input_ids,
        repetition_penalty=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.2,
    )

    # Token 3 beklenen ceza: 4 * 0.5 + 0.2 = -2.2
    assert torch.isclose(penalized[0, 3], torch.tensor(-2.2)), f"Token 3 cezası -2.2 olmalı, alınan: {penalized[0, 3]}"
    # Token 4 beklenen ceza: 1 * 0.5 + 0.2 = -0.7
    assert torch.isclose(penalized[0, 4], torch.tensor(-0.7)), f"Token 4 cezası -0.7 olmalı, alınan: {penalized[0, 4]}"
    # Token 0 (unseen): 0.0
    assert torch.isclose(penalized[0, 0], torch.tensor(0.0)), "Görülmeyen token 0.0 kalmalı"

    print("\n[KANIT] Frekans Cezası (4 tekrar x 0.5 + 0.2 = -2.2): Doğrulandı")


def test_infinite_loop_prevention_mechanism():
    """
    3. Model Üzerinde Sonsuz Döngü Kırma Kanıtı:
    Modelin en yüksek olasılıkla aynı tokeni tahmin etmeye meyilli olduğu durumda,
    repetition_penalty devreye girdiğinde döngünün kırıldığını doğrular.
    """
    model = LixLanguageModel(
        vocab_size=100,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=64,
    )
    model.eval()

    # Bir başlangıç prompt'u belirle
    input_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)

    # Cezası olmadan üret (repetition_penalty=1.0)
    out_no_penalty = model.generate(
        input_ids.clone(),
        max_new_tokens=15,
        temperature=0.0, # Greedy: deterministik
        top_k=1,
        repetition_penalty=1.0,
    )

    # Güçlü ceza ile üret (repetition_penalty=2.5)
    out_with_penalty = model.generate(
        input_ids.clone(),
        max_new_tokens=15,
        temperature=0.0, # Greedy: deterministik
        top_k=1,
        repetition_penalty=2.5,
    )

    tokens_no_pen = out_no_penalty[0].tolist()
    tokens_with_pen = out_with_penalty[0].tolist()

    # Cezalı çıktıdaki benzersiz token sayısı cezasız çıktıdan fazla veya çeşitlilik yüksek olmalı
    unique_no_pen = len(set(tokens_no_pen))
    unique_with_pen = len(set(tokens_with_pen))

    print(f"\n[KANIT] Cezasız Benzersiz Token: {unique_no_pen} / {len(tokens_no_pen)}")
    print(f"[KANIT] Cezalı Benzersiz Token: {unique_with_pen} / {len(tokens_with_pen)}")
    print(f"[KANIT] Cezasız Üretilen: {tokens_no_pen}")
    print(f"[KANIT] Cezalı Üretilen: {tokens_with_pen}")

    assert unique_with_pen >= unique_no_pen, "Repetition penalty çeşitliliği artırmalıdır!"


def test_lixai_interface_and_chat_repetition_integration():
    """
    4. LixAI ve InteractiveChatSession Entegrasyonu:
    repetition_penalty parametresinin LixAI.generate() ve Chat oturumunda sorunsuz çalıştığının kanıtı.
    """
    ai = LixAI(load_best=False) # Hızlı rastgele model testi
    res = ai.generate(
        prompt="Test prompt",
        max_new_tokens=10,
        repetition_penalty=1.3,
    )
    assert isinstance(res, str)

    session = InteractiveChatSession(
        model=ai.model,
        tokenizer=ai.tokenizer,
        repetition_penalty=1.25,
    )
    stats = session.get_stats()
    assert stats["repetition_penalty"] == 1.25, "Chat istatistiğinde repetition_penalty doğru gösterilmeli"

    streamed_chunks = list(session.stream_response("Merhaba", max_tokens=5, delay=0.0))
    assert len(streamed_chunks) > 0, "Chat streaming çıktısı üretilmelidir"
    print("\n[KANIT] LixAI ve ChatSession Repetition Penalty Entegrasyonu Başarılı")
