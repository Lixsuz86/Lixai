"""
LixAI Bellek-Dostu SFT Veri Yükleyici (Streaming SFT DataLoader)
ChatML formatındaki diyalogları akış halinde okur, prompt kısımlarını maskeler (-100)
ve modelin sadece asistan yanıtlarından öğrenmesini sağlayan batch'ler üretir.
"""
from pathlib import Path
from typing import Iterator, Tuple, Optional, List, Dict, Any
import json
import torch
from torch.utils.data import IterableDataset, DataLoader
from src.data.bpe_tokenizer import LixBPETokenizer
from src.data.tokenizer import get_default_tokenizer
from src.utils.logger import setup_logger

logger = setup_logger("lixai.sft_loader")


class SFTDataset(IterableDataset):
    """Bellek tasarruflu, JSONL tabanlı SFT diyalog veri akışı."""

    def __init__(
        self,
        file_path: str = "data/dialogue_sft.jsonl",
        tokenizer: Optional[Any] = None,
        seq_len: int = 256,
        repeat: bool = True,
    ):
        super().__init__()
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer or get_default_tokenizer()
        self.seq_len = seq_len
        self.repeat = repeat

        if not self.file_path.exists():
            raise FileNotFoundError(f"SFT veri seti bulunamadı: {self.file_path}")

        # Özel token ID'lerini belirle
        self.pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        self.endoftext_id = getattr(self.tokenizer, "endoftext_token_id", 822)
        self.assistant_id = getattr(self.tokenizer, "assistant_token_id", 821)
        self.user_id = getattr(self.tokenizer, "user_token_id", 820)
        self.system_id = getattr(self.tokenizer, "system_token_id", 819)

    def _process_messages(self, messages: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        """
        Diyalog mesajlarını tokenize eder ve hedef (target) maskelemesini uygular.
        Sadece asistanın söylediği tokenler için target tutulur, geri kalanı -100 yapılır.
        """
        input_ids: List[int] = []
        target_ids: List[int] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()

            if role == "system":
                role_tok = self.system_id
                is_assistant = False
            elif role == "user":
                role_tok = self.user_id
                is_assistant = False
            else:
                role_tok = self.assistant_id
                is_assistant = True

            # Format: <|role|>\n{content}\n<|endoftext|>
            content_tokens = self.tokenizer.encode("\n" + content + "\n")
            block_tokens = [role_tok] + content_tokens + [self.endoftext_id]

            for tok in block_tokens:
                input_ids.append(tok)
                if is_assistant:
                    target_ids.append(tok)
                else:
                    target_ids.append(-100)  # Prompt maskeleme (Ignore Index)

        return input_ids, target_ids

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Sonsuz veya tek geçişli akış."""
        while True:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        messages = data.get("messages", [])
                        if not messages:
                            continue

                        raw_inputs, raw_targets = self._process_messages(messages)
                        if len(raw_inputs) < 2:
                            continue

                        # Otoregresif hizalama: input[:-1], target[1:]
                        inp = raw_inputs[:-1]
                        tgt = raw_targets[1:]

                        # Padding veya Truncation (seq_len sabitliği)
                        if len(inp) > self.seq_len:
                            inp = inp[:self.seq_len]
                            tgt = tgt[:self.seq_len]
                        else:
                            pad_len = self.seq_len - len(inp)
                            inp = inp + [self.pad_id] * pad_len
                            tgt = tgt + [-100] * pad_len  # Pad kısımları da maskelenir

                        yield torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)
                    except Exception as e:
                        logger.warning(f"Diyalog satırı ayrıştırılamadı: {e}")
                        continue

            if not self.repeat:
                break


def create_sft_dataloader(
    file_path: str = "data/dialogue_sft.jsonl",
    tokenizer: Optional[Any] = None,
    batch_size: int = 8,
    seq_len: int = 256,
    repeat: bool = True,
    dataset_path: Optional[str] = None,
    **kwargs,
) -> DataLoader:
    """SFT eğitimi için hazır PyTorch DataLoader döndürür."""
    target_path = dataset_path or file_path
    dataset = SFTDataset(
        file_path=target_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        repeat=repeat,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)
