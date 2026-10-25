"""
LixAI Yüksek Seviye Kullanıcı Arayüzü (High-Level Python Interface)
Model eğitimi, metin üretimi ve optimizasyon işlemlerini tek noktadan yönetir.
"""
from typing import Optional, Dict, Any
from pathlib import Path
import torch

from src.models.core import LixLanguageModel
from src.data.tokenizer import SimpleCharTokenizer, get_default_tokenizer
from src.data.loader import create_dataloader
from src.engine.runner import Trainer
from src.utils.quantize import quantize_model_cpu
from src.utils.benchmark import compare_fp32_vs_int8
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("lixai.interface")


class LixAI:
    """Tüm LixAI yeteneklerini bir araya getiren ana orkestrasyon sınıfı."""

    def __init__(
        self,
        config_path: str = "configs/model_config.yaml",
        load_best: Optional[bool] = None,
        tokenizer: Optional[Any] = None,
        checkpoint_name: Optional[str] = None,
    ):
        self.config = load_config(config_path)
        self.tokenizer = tokenizer or get_default_tokenizer()

        # Model yapılandırması
        m_cfg = self.config.get("model", {})
        self.model = LixLanguageModel(
            vocab_size=m_cfg.get("vocab_size", self.tokenizer.vocab_size),
            d_model=m_cfg.get("d_model", 512),
            n_heads=m_cfg.get("n_heads", 8),
            n_layers=m_cfg.get("n_layers", 10),
            d_ff=m_cfg.get("d_ff", 2048),
            max_seq_len=m_cfg.get("max_seq_len", 512),
            dropout=m_cfg.get("dropout", 0.1),
            weight_tying=m_cfg.get("weight_tying", False),
        )

        s_cfg = self.config.get("system", {})
        self.num_threads = s_cfg.get("num_threads", 12)
        torch.set_num_threads(self.num_threads)

        p_cfg = self.config.get("paths", {})
        self.checkpoint_dir = p_cfg.get("checkpoints", "models/checkpoints")

        t_cfg = self.config.get("training", {})
        self.trainer = Trainer(
            model=self.model,
            learning_rate=t_cfg.get("learning_rate", 5e-4),
            weight_decay=t_cfg.get("weight_decay", 0.01),
            grad_clip=t_cfg.get("grad_clip", 1.0),
            checkpoint_dir=self.checkpoint_dir,
            num_threads=self.num_threads,
        )

        # Akıllı Checkpoint Otomatik Yükleme (Smart Auto-Load)
        self.loaded_checkpoint_name: Optional[str] = None
        ckpt_dir = Path(self.checkpoint_dir)

        if checkpoint_name:
            target_ckpt = ckpt_dir / checkpoint_name
            if target_ckpt.exists():
                try:
                    self.trainer.load_checkpoint(checkpoint_name)
                    self.loaded_checkpoint_name = checkpoint_name
                    logger.info(f"Belirtilen checkpoint başarıyla yüklendi: {checkpoint_name}")
                except Exception as e:
                    logger.warning(f"Belirtilen checkpoint yüklenemedi ({checkpoint_name}): {e}")
            else:
                logger.warning(f"Belirtilen checkpoint dosyası bulunamadı: {target_ckpt}")
        elif load_best is True:
            best_ckpt = ckpt_dir / "best_model.pt"
            if best_ckpt.exists():
                try:
                    self.trainer.load_checkpoint("best_model.pt")
                    self.loaded_checkpoint_name = "best_model.pt"
                    logger.info("best_model.pt başarıyla yüklendi.")
                except Exception as e:
                    logger.warning(f"best_model.pt yüklenemedi: {e}")
            else:
                logger.warning(f"best_model.pt dosyası bulunamadı: {best_ckpt}")
        elif load_best is None:
            # Otomatik tespit: diskte bulunan en güncel/en iyi modeli yükle (SFT modeli öncelikli)
            candidates = ["lixai_chat_sft.pt", "best_sft_model.pt", "best_model.pt", "final_model.pt"]
            for candidate in candidates:
                cand_path = ckpt_dir / candidate
                if cand_path.exists():
                    try:
                        self.trainer.load_checkpoint(candidate)
                        self.loaded_checkpoint_name = candidate
                        logger.info(f"Eğitilmiş model kontrol noktası tespit edildi ve otomatik yüklendi: {candidate}")
                        break
                    except Exception as e:
                        logger.warning(f"{candidate} otomatik yüklenirken hata oluştu: {e}")

    def train(
        self,
        max_steps: Optional[int] = None,
        dataset_name: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Hugging Face veri akışı üzerinden modeli eğitir."""
        d_cfg = self.config.get("data", {})
        t_cfg = self.config.get("training", {})

        target_dataset = dataset_name or d_cfg.get("dataset_name", "roneneldan/TinyStories")
        target_config = d_cfg.get("dataset_config", None)
        target_batch_size = batch_size or d_cfg.get("batch_size", 16)
        steps = max_steps or t_cfg.get("max_steps", 100)

        loader = create_dataloader(
            dataset_name=target_dataset,
            dataset_config=target_config,
            split="train",
            batch_size=target_batch_size,
            seq_len=d_cfg.get("seq_len", 128),
            streaming=True,
            repeat_on_end=True,
        )

        results = self.trainer.train_steps(
            dataloader=loader,
            max_steps=steps,
            log_interval=t_cfg.get("eval_interval", 10),
            save_interval=t_cfg.get("save_interval", 50),
        )
        return results

    def train_sft(
        self,
        dataset_path: str = "data/dialogue_sft.jsonl",
        max_steps: int = 30,
        batch_size: int = 4,
        seq_len: int = 128,
        learning_rate: float = 2e-4,
        checkpoint_name: str = "lixai_chat_sft.pt",
    ) -> Dict[str, Any]:
        """Diyalog veri seti üzerinden hedef maskelemeli SFT eğitimi gerçekleştirir."""
        from src.data.sft_loader import create_sft_dataloader
        from src.engine.sft_runner import SFTTrainer

        loader = create_sft_dataloader(
            dataset_path=dataset_path,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            max_seq_len=seq_len,
            shuffle=True,
            repeat_on_end=True,
        )

        sft_trainer = SFTTrainer(
            model=self.model,
            learning_rate=learning_rate,
            checkpoint_dir=self.checkpoint_dir,
            num_threads=self.num_threads,
        )

        results = sft_trainer.train_steps(
            dataloader=loader,
            max_steps=max_steps,
            log_interval=5,
            checkpoint_name=checkpoint_name,
        )
        self.loaded_checkpoint_name = checkpoint_name
        return results

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 40,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        min_p: Optional[float] = 0.05,
        repetition_penalty: float = 1.15,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        use_quantized: bool = False,
        use_cache: bool = True,
    ) -> str:
        """Verilen başlangıç metnine göre otoregresif olarak metin üretir."""
        model_to_use = quantize_model_cpu(self.model) if use_quantized else self.model
        tokens = self.tokenizer.encode(prompt)
        if not tokens:
            tokens = [getattr(self.tokenizer, "unk_token_id", 1)]

        max_valid_id = min(self.model.vocab_size, getattr(self.tokenizer, "vocab_size", self.model.vocab_size)) - 1
        tokens = [min(t, max_valid_id) for t in tokens]
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        
        valid_vocab_limit = getattr(self.tokenizer, "vocab_size", self.model.vocab_size)
        out_tokens = model_to_use.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            use_cache=use_cache,
            max_vocab_id=valid_vocab_limit,
        )
        return self.tokenizer.decode(out_tokens[0].tolist())

    def benchmark(self) -> Dict[str, Any]:
        """FP32 vs INT8 CPU performans karşılaştırmasını döndürür."""
        return compare_fp32_vs_int8(self.model)
