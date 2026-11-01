"""
LixAI Bellek-Dostu Veri Yükleyici (Memory-Efficient DataLoader)
16 GB RAM sınırını aşmamak için bellek tüketimini sabit tutar.
"""
from typing import Iterator, Tuple, Optional, Any
import torch
from torch.utils.data import IterableDataset
from src.data.pipeline import TextDataPipeline
from src.utils.logger import setup_logger

logger = setup_logger("lixai.loader")


class StreamingTextDataset(IterableDataset):
    """PyTorch IterableDataset arayüzü ile sonsuz/büyük veri akışı."""

    def __init__(
        self,
        pipeline: TextDataPipeline,
        batch_size: int = 16,
        max_batches: Optional[int] = None
    ):
        super().__init__()
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.max_batches = max_batches

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        return self.pipeline.stream_batches(
            batch_size=self.batch_size,
            max_batches=self.max_batches
        )


def create_dataloader(
    dataset_name: str = "roneneldan/TinyStories",
    dataset_config: Optional[str] = None,
    split: str = "train",
    batch_size: int = 16,
    seq_len: int = 128,
    tokenizer: Optional[Any] = None,
    max_batches: Optional[int] = None,
    streaming: bool = True,
    repeat_on_end: bool = True,
) -> StreamingTextDataset:
    """Veri akışını başlatan ve hazır batch üreten yardımcı fonksiyon."""
    pipeline = TextDataPipeline(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        seq_len=seq_len,
        tokenizer=tokenizer,
        streaming=streaming,
        repeat_on_end=repeat_on_end,
    )
    return StreamingTextDataset(pipeline=pipeline, batch_size=batch_size, max_batches=max_batches)
