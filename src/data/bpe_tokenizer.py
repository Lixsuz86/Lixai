"""
LixAI Çok Dilli & Türkçe Byte-Pair Encoding (BPE) Tokenizer
Hugging Face tokenizers altyapısını kullanarak 8192 boyutlu alt-kelime sözlüğü sunar.
"""
from pathlib import Path
from typing import List, Union, Iterable, Optional, Dict
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from src.utils.logger import setup_logger

logger = setup_logger("lixai.tokenizer")


class LixBPETokenizer:
    """8192 boyutlu, Türkçe ve İngilizce destekli Byte-Level BPE Tokenizer ve ChatML Özel Tokenleri."""

    SPECIAL_TOKENS = [
        "<pad>",
        "<unk>",
        "<s>",
        "</s>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|endoftext|>",
    ]

    def __init__(self, tokenizer_path: Optional[str] = None):
        if tokenizer_path and Path(tokenizer_path).exists():
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            # Sıfırdan Byte-Level BPE modeli
            self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = ByteLevelDecoder()

        # ChatML ve kontrol tokenlerini ekle/garantile
        added = [AddedToken(t, special=True) for t in self.SPECIAL_TOKENS]
        self.tokenizer.add_special_tokens(added)
        self.vocab_size = self.tokenizer.get_vocab_size()

        # Özel Token ID Haritası
        self.pad_token_id = self.token_to_id("<pad>") or 0
        self.unk_token_id = self.token_to_id("<unk>") or 1
        self.bos_token_id = self.token_to_id("<s>") or 2
        self.eos_token_id = self.token_to_id("</s>") or 3
        self.system_token_id = self.token_to_id("<|system|>")
        self.user_token_id = self.token_to_id("<|user|>")
        self.assistant_token_id = self.token_to_id("<|assistant|>")
        self.endoftext_token_id = self.token_to_id("<|endoftext|>")

    def token_to_id(self, token: str) -> Optional[int]:
        """Verilen özel veya genel token metninin sayısal ID'sini döner."""
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Verilen sayısal ID'nin token metin karşılığını döner."""
        return self.tokenizer.id_to_token(token_id)

    def train(
        self,
        corpus_texts: Iterable[str],
        vocab_size: int = 8192,
        save_path: Optional[str] = "configs/tokenizer.json",
    ) -> None:
        """Metin korpusu üzerinden BPE sözlüğünü eğitir."""
        logger.info(f"BPE Tokenizer eğitiliyor (Hedef Sözlük: {vocab_size})...")
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
            min_frequency=2,
            show_progress=False,
        )
        self.tokenizer.train_from_iterator(corpus_texts, trainer=trainer)
        # Özel tokenleri tekrar doğrula
        added = [AddedToken(t, special=True) for t in self.SPECIAL_TOKENS]
        self.tokenizer.add_special_tokens(added)
        self.vocab_size = self.tokenizer.get_vocab_size()
        logger.info(f"BPE Tokenizer eğitimi tamamlandı. Oluşan sözlük boyutu: {self.vocab_size}")

        if save_path:
            self.save(save_path)

    def encode(self, text: str) -> List[int]:
        """Metni token ID dizisine çevirir."""
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Token ID dizisini orijinal metne dönüştürür."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, filepath: str = "configs/tokenizer.json") -> None:
        """Tokenizer durumunu diske kaydeder."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))
        logger.info(f"Tokenizer kaydedildi: {path}")

    @classmethod
    def load(cls, filepath: str = "configs/tokenizer.json") -> "LixBPETokenizer":
        """Diskteki dosyadan Tokenizer'ı yükler."""
        instance = cls(tokenizer_path=filepath)
        return instance


def format_chat_prompt(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """
    ChatML formatında yapılandırılmış konuşma geçmişini tek bir prompt metnine dönüştürür.
    
    Örnek messages:
      [
        {"role": "system", "content": "Sen LixAI adlı Türkçe asistansın."},
        {"role": "user", "content": "Merhaba!"},
        {"role": "assistant", "content": "Merhaba, size nasıl yardımcı olabilirim?"}
      ]
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        lines.append(f"<|{role}|>\n{content}\n<|endoftext|>")

    if add_generation_prompt:
        lines.append("<|assistant|>\n")

    return "\n".join(lines)


def get_default_bpe_corpus() -> List[str]:
    """Tokenizer eğitimi için örnek zengin Türkçe ve İngilizce başlangıç korpusu."""
    return [
        # Türkçe Korpus
        "Yapay zeka modelleri günümüzde doğal dil işleme alanında büyük gelişmeler kaydetti.",
        "LixAI, sıfırdan geliştirilen ve CPU üzerinde yüksek hızda çalışan modern bir dil modelidir.",
        "Türkçe dil bilgisi, sondan eklemeli yapısı ve ses uyumları ile özel bir morfolojiye sahiptir.",
        "Bilgisayar, işlemci, bellek, çekirdek, derin öğrenme, sinir ağları ve matris çarpımları.",
        "Bir zamanlar yemyeşil bir vadide neşeli bir robot yaşardı. Her gün yeni kelimeler öğrenirdi.",
        "Gelişmiş optimizasyon teknikleri sayesinde bellek tüketimi minimum seviyede tutulur.",
        "Çocuklar bahçede oynarken gökyüzünde parlak yıldızlar parlamaya başladı.",
        "Hızlı ve kararlı çıkarım için dinamik kuantizasyon ve çoklu iş parçacığı kullanılır.",
        # İngilizce Korpus
        "Once upon a time in a bright sunny forest, a curious little fox found a magic book.",
        "The language model predicts the next token in the sequence with high confidence.",
        "Deep learning transformers utilize multi-head self-attention mechanisms efficiently.",
        "Hardware acceleration on modern CPUs enables low-latency machine learning inference.",
        "Small language models are versatile, efficient, and cost-effective for localized edge tasks.",
        "Lily loved reading stories about brave astronauts exploring distant planets and stars.",
    ] * 20
