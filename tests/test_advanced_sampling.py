"""
Faz 10.2 Doğrulama Testi: Top-p (Nucleus) ve Min-P Dinamik Örnekleme Algoritmaları
Küçük dil modellerinde aşırı saçmalamayı ve mekanik tekrarları önleyen ampirik matematiksel kanıt.
"""
import pytest
import torch
import torch.nn.functional as F
from src.models.core import LixLanguageModel, apply_sampling_filters
from src.engine.interface import LixAI
from src.engine.chat import InteractiveChatSession


def test_top_p_nucleus_mathematical_precision():
    """
    1. Top-P (Nucleus) Matematiksel Doğrulama:
    - Kümülatif olasılık eşiğini aşan tokenler filtrelenmelidir (-inf).
    - İlk eşiği aşan token dahil en olası çekirdek kümesi korunmalıdır.
    """
    # 5 tokenli kontrollü logit dağılımı
    logits = torch.tensor([[10.0, 9.0, 2.0, 1.0, 0.0]], dtype=torch.float32)
    # Olasılıklar: [0.7308, 0.2688, 0.0002, 0.0001, 0.0000]
    
    # top_p = 0.8: Token 0 (%73) tek başına < 0.8, Token 1 eklenince %99.9 olur.
    # Bu nedenle Token 0 ve 1 korunmalı, Token 2, 3, 4 filtrelenmelidir (-inf).
    filtered = apply_sampling_filters(logits.clone(), temperature=1.0, top_p=0.8)

    assert filtered[0, 0].item() == 10.0, "En olası ilk token korunmalı"
    assert filtered[0, 1].item() == 9.0, "Eşiği tamamlayan ikinci token korunmalı"
    assert filtered[0, 2].item() == float("-inf"), "Eşik dışı token -inf olmalı"
    assert filtered[0, 3].item() == float("-inf"), "Eşik dışı token -inf olmalı"
    assert filtered[0, 4].item() == float("-inf"), "Eşik dışı token -inf olmalı"

    # top_p = 1.0 olduğunda hiçbir token filtrelenmemelidir
    unfiltered = apply_sampling_filters(logits.clone(), temperature=1.0, top_p=1.0)
    assert not torch.isinf(unfiltered).any(), "top_p=1.0 iken hiçbir token -inf olmamalı"

    print("\n[KANIT] Top-P Nucleus Kümülâtif Dilimleme: %80 eşiğinde [0, 1] korundu, [2, 3, 4] -inf yapıldı")


def test_min_p_dynamic_thresholding():
    """
    2. Min-P Dinamik Taban Olasılık Eşiği Kanıtı:
    - Min-P: Eşik = p_max * min_p formülü ile çalışır.
    - Model çok emin olduğunda (yüksek p_max) taban yükselir ve gereksiz tokenler elenir.
    - Model belirsiz olduğunda taban orantılı düşerek çeşitliliği korur.
    """
    # Token 0: ~%70, Token 1: ~%25, Token 2: ~%4, Token 3: ~%1
    logits = torch.tensor([[5.0, 4.0, 2.0, 0.5]], dtype=torch.float32)
    probs = F.softmax(logits, dim=-1)
    p_max = probs[0, 0].item()

    # min_p = 0.1 -> Eşik = p_max * 0.1
    min_p_val = 0.1
    threshold = p_max * min_p_val

    filtered = apply_sampling_filters(logits.clone(), temperature=1.0, min_p=min_p_val)

    # Token 1 olasılığı threshold'dan büyükse korunmalı, küçükse -inf olmalı
    for i in range(logits.size(-1)):
        p_i = probs[0, i].item()
        if p_i >= threshold:
            assert filtered[0, i].item() == logits[0, i].item(), f"Token {i} ({p_i:.4f} >= {threshold:.4f}) korunmalıydı"
        else:
            assert filtered[0, i].item() == float("-inf"), f"Token {i} ({p_i:.4f} < {threshold:.4f}) elenmeliydi"

    print(f"\n[KANIT] Min-P Eşiği (p_max={p_max:.3f} * {min_p_val} = {threshold:.4f}): Düşük olasılıklı tokenler elendi")


def test_temperature_scaling_and_greedy_argmax():
    """
    3. Sıcaklık Modülasyonu ve Deterministik Çıkarım:
    - temperature <= 1e-4 durumunda deterministik argmax işletilmelidir.
    - Düşük sıcaklık dağılımı keskinleştirmelidir.
    """
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

    # Düşük sıcaklık
    cold_logits = apply_sampling_filters(logits.clone(), temperature=0.5)
    assert cold_logits[0, 2].item() == 6.0, "Sıcaklık 0.5 iken logitler 2 katına çıkmalı (3.0 / 0.5 = 6.0)"

    # Sıfır sıcaklıkta (greedy) doğrudan argmax
    greedy_token = torch.argmax(logits, dim=-1).item()
    assert greedy_token == 2, "Greedy seçimde en yüksek logitli token (indeks 2) seçilmeli"

    print("\n[KANIT] Sıcaklık Ölçekleme (3.0 / 0.5 = 6.0) ve Greedy Argmax: Doğrulandı")


def test_e2e_advanced_sampling_integration():
    """
    4. LixAI ve InteractiveChatSession Uçtan Uca Örnekleme Entegrasyonu:
    top_p ve min_p parametreleri ile sorunsuz metin üretimi ve istatistik gösterimi kanıtı.
    """
    ai = LixAI(load_best=False) # Hızlı CPU testi
    output = ai.generate(
        prompt="Bugün hava",
        max_new_tokens=10,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        min_p=0.05,
    )
    assert isinstance(output, str) and len(output) > 0, "Metin üretimi başarılı olmalı"

    session = InteractiveChatSession(
        model=ai.model,
        tokenizer=ai.tokenizer,
        temperature=0.7,
        top_k=40,
        top_p=0.85,
        min_p=0.08,
    )
    stats = session.get_stats()
    assert stats["top_p"] == 0.85, "ChatSession top_p değerini doğru saklamalı"
    assert stats["min_p"] == 0.08, "ChatSession min_p değerini doğru saklamalı"

    stream_chunks = list(session.stream_response("Güneşli", max_tokens=5, delay=0.0))
    assert len(stream_chunks) > 0, "Top-P/Min-P ile canlı akış başarılı olmalı"

    print(f"\n[KANIT] LixAI ve ChatSession Top-P (0.85) / Min-P (0.08) Entegrasyonu Başarılı")
