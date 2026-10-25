"""
LixAI Tokenizer Modülü
Hugging Face tokenizers veya dahili BPE/Char fallback mekanizması sağlar.
"""
from pathlib import Path
from typing import List, Union, Any
import json


class SimpleCharTokenizer:
    """Hızlı testler ve sıfırdan eğitim için hafif karakter/bayt tokenizer."""

    def __init__(self, chars: str = None):
        if chars is None:
            # Standart ASCII + Türkçe karakterler ve noktalama
            chars = (
                " !\"#$%&'()*+,-./0123456789:;<=>?@"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
                "abcdefghijklmnopqrstuvwxyz{|}~"
                "çÇğĞıİöÖşŞüÜ\n\t\r"
            )
        self.chars = sorted(list(set(chars)))
        self.vocab_size = len(self.chars) + 2  # <pad>=0, <unk>=1
        self.pad_token_id = 0
        self.unk_token_id = 1
        
        self.char_to_id = {ch: i + 2 for i, ch in enumerate(self.chars)}
        self.id_to_char = {i + 2: ch for i, ch in enumerate(self.chars)}
        self.id_to_char[0] = "<pad>"
        self.id_to_char[1] = "<unk>"

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(ch, self.unk_token_id) for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join([self.id_to_char.get(i, "") for i in ids if i not in (0, 1)])

    def save(self, filepath: str) -> None:
        data = {
            "chars": "".join(self.chars),
            "vocab_size": self.vocab_size
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "SimpleCharTokenizer":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(chars=data["chars"])


def get_default_tokenizer(prefer_bpe: bool = True) -> Union["SimpleCharTokenizer", Any]:
    """Varsayılan tokenizer örneğini döner. configs/tokenizer.json varsa LixBPETokenizer, yoksa SimpleCharTokenizer döner."""
    bpe_path = Path("configs/tokenizer.json")
    if prefer_bpe and bpe_path.exists():
        try:
            from src.data.bpe_tokenizer import LixBPETokenizer
            return LixBPETokenizer.load(str(bpe_path))
        except Exception:
            pass
    return SimpleCharTokenizer()

