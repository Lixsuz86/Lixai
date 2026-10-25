"""
LixAI Eğitim ve Çıkarım Motoru (Training & Inference Engine)
CPU odaklı optimizasyon, NaN koruması ve en iyi model (best_model) takibi içerir.
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from src.models.core import LixLanguageModel
from src.data.loader import StreamingTextDataset
from src.utils.logger import setup_logger

logger = setup_logger("lixai.engine")


class Trainer:
    """CPU üzerinde yüksek verimli model eğitimi ve kontrol noktası yönetimi."""

    def __init__(
        self,
        model: LixLanguageModel,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "models/checkpoints",
        num_threads: int = 12,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Intel CPU thread yapılandırması
        torch.set_num_threads(num_threads)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.best_loss = float("inf")

    def train_steps(
        self,
        dataloader: StreamingTextDataset,
        max_steps: int = 100,
        log_interval: int = 10,
        save_interval: int = 50,
    ) -> Dict[str, Any]:
        """Belirtilen adım sayısı kadar eğitim döngüsü yürütür."""
        self.model.train()
        step = 0
        total_loss = 0.0
        start_time = time.time()
        losses: List[float] = []

        logger.info(f"Model eğitimi başladı: Max Steps = {max_steps}, Threads = {torch.get_num_threads()}")

        for x, y in dataloader:
            if step >= max_steps:
                break

            self.optimizer.zero_grad(set_to_none=True)
            _, loss = self.model(x, targets=y)

            loss_val = loss.item()

            # NaN / Inf kontrolü
            if math.isnan(loss_val) or math.isinf(loss_val):
                logger.warning(f"Adım {step}: Sayısal kararsızlık tespit edildi (Loss = {loss_val}), adım atlanıyor.")
                continue

            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss_val
            losses.append(loss_val)
            step += 1

            # En iyi model kontrolü
            if loss_val < self.best_loss:
                self.best_loss = loss_val
                self.save_checkpoint("best_model.pt")

            if step % log_interval == 0 or step == max_steps:
                avg_loss = total_loss / step
                elapsed = time.time() - start_time
                steps_per_sec = step / max(elapsed, 1e-5)
                logger.info(
                    f"Adım [{step}/{max_steps}] | Anlık Loss: {loss_val:.4f} | "
                    f"Ortalama: {avg_loss:.4f} | En İyi: {self.best_loss:.4f} | "
                    f"Hız: {steps_per_sec:.2f} adım/sn"
                )

            if step % save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{step}.pt")

        # Nihai modeli kaydet
        final_path = self.save_checkpoint("final_model.pt")
        total_time = time.time() - start_time

        logger.info(f"Eğitim tamamlandı! Toplam süre: {total_time:.2f} sn, Nihai model: {final_path}")

        return {
            "completed_steps": step,
            "final_loss": losses[-1] if losses else 0.0,
            "best_loss": self.best_loss,
            "avg_loss": total_loss / max(step, 1),
            "total_time": total_time,
        }

    def save_checkpoint(self, filename: str) -> Path:
        """Ağırlıkları, optimizer durumunu ve yapılandırmayı kaydeder."""
        save_path = self.checkpoint_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": {
                "vocab_size": self.model.vocab_size,
                "d_model": self.model.tok_emb.embedding_dim,
                "max_seq_len": self.model.max_seq_len,
            }
        }, save_path)
        return save_path

    def load_checkpoint(self, filename: str) -> None:
        """Kayıtlı ağırlıkları modele geri yükler."""
        load_path = self.checkpoint_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint dosyası bulunamadı: {load_path}")
        checkpoint = torch.load(load_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "best_loss" in checkpoint:
            self.best_loss = checkpoint["best_loss"]
        logger.info(f"Model ağırlıkları başarıyla yüklendi: {load_path}")
