"""
LixAI Talimat Takibi ve SFT Eğitim Motoru (Instruction Supervised Fine-Tuning Engine)
Yalnızca asistan tokenlerinden kayıp (loss) hesaplayarak modeli konuşkan ve akıllı hale getirir.
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from src.models.core import LixLanguageModel
from src.utils.logger import setup_logger

logger = setup_logger("lixai.sft_engine")


class SFTTrainer:
    """Maskelenmiş hedef kaybı ile asistan yeteneği kazandıran SFT eğitim motoru."""

    def __init__(
        self,
        model: LixLanguageModel,
        learning_rate: float = 2e-4,
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

        torch.set_num_threads(num_threads)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        self.best_loss = float("inf")

    def train_steps(
        self,
        dataloader: DataLoader,
        max_steps: int = 50,
        log_interval: int = 5,
        save_interval: int = 25,
        checkpoint_name: str = "lixai_chat_sft.pt",
    ) -> Dict[str, Any]:
        """SFT veri kümesi üzerinden maskelenmiş hedef eğitimi gerçekleştirir."""
        self.model.train()
        step = 0
        total_loss = 0.0
        total_accuracy = 0.0
        total_active_tokens = 0
        losses: List[float] = []
        accuracies: List[float] = []
        start_time = time.time()

        logger.info(f"SFT Asistan Eğitimi Başlatıldı: Max Steps = {max_steps}, Hedef Model: {checkpoint_name}")

        for x, y in dataloader:
            if step >= max_steps:
                break

            # Maskeleme kontrolü: en az 1 asistan tokeni olmalı
            active_mask = (y != -100)
            active_tokens_count = active_mask.sum().item()
            if active_tokens_count == 0:
                continue

            self.optimizer.zero_grad(set_to_none=True)
            logits, loss = self.model(x, targets=y)

            loss_val = loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                logger.warning(f"Adım {step}: Sayısal kararsızlık (Loss={loss_val}), atlanıyor.")
                continue

            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Token Doğruluk Oranı (Yalnızca asistan tokenleri üzerinden)
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                correct = (preds[active_mask] == y[active_mask]).float().sum().item()
                step_acc = correct / max(active_tokens_count, 1)

            total_loss += loss_val
            total_accuracy += step_acc
            total_active_tokens += active_tokens_count
            losses.append(loss_val)
            accuracies.append(step_acc)
            step += 1

            if loss_val < self.best_loss:
                self.best_loss = loss_val
                self.save_checkpoint("best_sft_model.pt")

            if step % log_interval == 0 or step == max_steps:
                avg_loss = total_loss / step
                avg_acc = (total_accuracy / step) * 100
                elapsed = time.time() - start_time
                steps_per_sec = step / max(elapsed, 1e-5)
                logger.info(
                    f"SFT Adım [{step}/{max_steps}] | Loss: {loss_val:.4f} (Ort: {avg_loss:.4f}) | "
                    f"Token Doğruluk: %{avg_acc:.1f} | Asistan Tokeni: {active_tokens_count} | "
                    f"Hız: {steps_per_sec:.2f} adım/sn"
                )

            if step % save_interval == 0:
                self.save_checkpoint(f"sft_checkpoint_step_{step}.pt")

        # Nihai SFT modelini kaydet
        final_path = self.save_checkpoint(checkpoint_name)
        total_time = time.time() - start_time
        logger.info(f"SFT Eğitimi Tamamlandı! Süre: {total_time:.2f} sn, Model: {final_path}")

        return {
            "completed_steps": step,
            "final_loss": losses[-1] if losses else 0.0,
            "best_loss": self.best_loss,
            "avg_loss": total_loss / max(step, 1),
            "avg_accuracy": (total_accuracy / max(step, 1)) * 100,
            "total_active_tokens": total_active_tokens,
            "total_time": total_time,
            "saved_checkpoint": str(final_path),
        }

    def save_checkpoint(self, filename: str) -> Path:
        """Ağırlıkları ve SFT metaverilerini diske kaydeder."""
        save_path = self.checkpoint_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "type": "sft_chat_model",
            "config": {
                "vocab_size": self.model.vocab_size,
                "d_model": self.model.tok_emb.embedding_dim,
                "max_seq_len": self.model.max_seq_len,
            }
        }, save_path)
        return save_path

    def load_checkpoint(self, filename: str, load_optimizer: bool = False) -> None:
        """Kayıtlı ağırlıkları SFT modeline yükler."""
        load_path = self.checkpoint_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"SFT Checkpoint dosyası bulunamadı: {load_path}")
        checkpoint = torch.load(load_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "best_loss" in checkpoint:
            self.best_loss = checkpoint["best_loss"]
        logger.info(f"SFT için ağırlıklar başarıyla yüklendi: {load_path}")

