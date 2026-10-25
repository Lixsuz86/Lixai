"""
LixAI Hugging Face Veri Boru Hattı (Data Pipeline)
Verileri akış (streaming) halinde çeker, RAM tüketimini minimumda tutar.
"""
from typing import Generator, Tuple, Optional, List, Dict, Any
import gc
import torch
from datasets import load_dataset
from src.data.tokenizer import SimpleCharTokenizer, get_default_tokenizer
from src.utils.logger import setup_logger

logger = setup_logger("lixai.data")


class TextDataPipeline:
    """Hugging Face veri setini akış halinde yükleyip tensör batch'lerine dönüştürür."""

    def __init__(
        self,
        dataset_name: str = "roneneldan/TinyStories",
        dataset_config: Optional[str] = None,
        split: str = "train",
        text_field: str = "text",
        seq_len: int = 128,
        tokenizer: Optional[SimpleCharTokenizer] = None,
        streaming: bool = True,
        repeat_on_end: bool = True,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.text_field = text_field
        self.seq_len = seq_len
        self.tokenizer = tokenizer or get_default_tokenizer()
        self.streaming = streaming
        self.repeat_on_end = repeat_on_end

    def get_dataset_stream(self):
        """Hugging Face üzerinden streaming modunda veri setini açar, hata durumunda yerel fallback sunar."""
        cfg_str = f" ({self.dataset_config})" if self.dataset_config else ""
        logger.info(f"Hugging Face veri seti bağlanıyor: {self.dataset_name}{cfg_str} [{self.split}]")
        try:
            kwargs: Dict[str, Any] = {"split": self.split, "streaming": self.streaming}
            if self.dataset_config:
                kwargs["name"] = self.dataset_config
            dataset = load_dataset(self.dataset_name, **kwargs)
            return dataset
        except Exception as e:
            logger.warning(f"HF veri seti çekilemedi ({e}), yerel dayanıklı sentetik veri akışına geçiliyor.")
            synthetic_sample = (
                "Once upon a time, there was a little friendly robot named LixAI. "
                "LixAI lived in a quiet computer and loved learning new words every single day. "
                "The robot was very fast, helpful, and kind to everyone. "
            )
            return [{self.text_field: synthetic_sample} for _ in range(100)]

    def stream_tokens(self) -> Generator[int, None, None]:
        """Metinleri akış halinde okuyup tek tek token ID'lerini yield eder."""
        while True:
            dataset = self.get_dataset_stream()
            has_data = False
            for item in dataset:
                text = item.get(self.text_field, "")
                if text and isinstance(text, str) and len(text.strip()) > 0:
                    has_data = True
                    encoded = self.tokenizer.encode(text)
                    for token_id in encoded:
                        yield token_id

            # Eğer dataset bitti ve tekrar istenmiyorsa veya hiç veri yoksa döngüyü kır
            if not self.repeat_on_end or not has_data:
                break

    def stream_batches(
        self,
        batch_size: int = 16,
        max_batches: Optional[int] = None,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Token akışından (batch_size, seq_len) boyutunda (X, Y) tensör çiftleri üretir.
        Y, X'in bir adım sağa kaydırılmış halidir (Causal Next-Token Prediction).
        """
        token_gen = self.stream_tokens()
        total_tokens_needed = batch_size * (self.seq_len + 1)
        batch_count = 0

        while True:
            if max_batches is not None and batch_count >= max_batches:
                break

            tokens: List[int] = []
            try:
                for _ in range(total_tokens_needed):
                    tokens.append(next(token_gen))
            except StopIteration:
                break

            if len(tokens) < total_tokens_needed:
                break

            # [batch_size, seq_len + 1] şeklinde tensöre çevir
            tensor = torch.tensor(tokens, dtype=torch.long).view(batch_size, self.seq_len + 1)
            
            # X: 0'dan seq_len'e kadar, Y: 1'den seq_len+1'e kadar (Next token)
            x = tensor[:, :-1].contiguous()
            y = tensor[:, 1:].contiguous()

            batch_count += 1
            yield x, y

            # Her 100 batch'te bir gereksiz Python bellek çöpünü topla (RAM sızıntısı önleme)
            if batch_count % 100 == 0:
                gc.collect()
