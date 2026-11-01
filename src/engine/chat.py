"""
LixAI Zengin İnteraktif Terminal Sohbet Modülü (Interactive CLI TUI)
Kelimeleri anlık akış (streaming) halinde ekrana basan, komut destekli REPL oturumu.
"""
from typing import Optional, List, Dict, Generator, Any, Set, Union
import sys
import time
import torch
import torch.nn.functional as F

from src.models.core import LixLanguageModel
from src.data.tokenizer import SimpleCharTokenizer, get_default_tokenizer
from src.utils.quantize import quantize_model_cpu, get_model_size_mb
from src.utils.logger import setup_logger

logger = setup_logger("lixai.chat")

# ANSI Renk Kodları (Windows & Unix terminal uyumlu)
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


class InteractiveChatSession:
    """Terminal üzerinde etkileşimli, akışlı (streaming) sohbet oturumu."""

    def __init__(
        self,
        model: LixLanguageModel,
        tokenizer: Optional[Any] = None,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        min_p: float = 0.05,
        repetition_penalty: float = 1.15,
        frequency_penalty: float = 0.0,
        stop_token_ids: Optional[Union[int, List[int], Set[int]]] = None,
        stop_words: Optional[List[str]] = None,
        use_quantized: bool = False,
        checkpoint_name: Optional[str] = None,
    ):
        self.raw_model = model
        self.tokenizer = tokenizer or get_default_tokenizer()
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.use_quantized = use_quantized
        self.checkpoint_name = checkpoint_name

        self.stop_token_ids: Set[int] = set()
        if stop_token_ids is not None:
            if isinstance(stop_token_ids, int):
                self.stop_token_ids.add(stop_token_ids)
            else:
                self.stop_token_ids.update(stop_token_ids)
        else:
            if hasattr(self.tokenizer, "endoftext_token_id") and self.tokenizer.endoftext_token_id is not None:
                self.stop_token_ids.add(self.tokenizer.endoftext_token_id)
            if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                self.stop_token_ids.add(self.tokenizer.eos_token_id)

        self.stop_words: List[str] = list(stop_words) if stop_words is not None else ["<|endoftext|>", "</s>"]
        for sw in self.stop_words:
            if hasattr(self.tokenizer, "token_to_id"):
                sw_id = self.tokenizer.token_to_id(sw)
                if sw_id is not None:
                    self.stop_token_ids.add(sw_id)

        self.active_model = quantize_model_cpu(self.raw_model) if use_quantized else self.raw_model
        self.history: List[str] = []

    def set_quantization(self, enable: bool) -> None:
        """Kuantizasyon modunu dinamik açar/kapatır."""
        self.use_quantized = enable
        self.active_model = quantize_model_cpu(self.raw_model) if enable else self.raw_model

    def stream_response(
        self,
        prompt: str,
        max_tokens: int = 40,
        delay: float = 0.01,
    ) -> Generator[str, None, None]:
        """Token üretildikçe tek tek metin parçacığı yield eder (Canlı Akış)."""
        from src.models.core import apply_repetition_penalties, apply_sampling_filters

        self.active_model.eval()

        # Geçmiş ile birleştir
        full_prompt = (" ".join(self.history[-2:]) + " " + prompt).strip()
        tokens = self.tokenizer.encode(full_prompt)
        if not tokens:
            tokens = [getattr(self.tokenizer, "unk_token_id", 1)]

        # Sınır dışı tokenleri engelle (IndexError koruması)
        max_valid_id = min(self.active_model.vocab_size, getattr(self.tokenizer, "vocab_size", self.active_model.vocab_size)) - 1
        tokens = [min(t, max_valid_id) for t in tokens]
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        
        valid_vocab_limit = getattr(self.tokenizer, "vocab_size", self.active_model.vocab_size)
        # İlk adım: tüm başlangıç dizisi işlenir ve önbellek doldurulur
        curr_pos = input_tensor.size(1)
        with torch.no_grad():
            logits, _, kv_caches = self.active_model(input_tensor, use_cache=True, start_pos=0)

        generated_tokens = []
        for _ in range(max_tokens):
            if curr_pos >= self.active_model.max_seq_len:
                break

            next_logits = logits[:, -1, :]
            next_logits = torch.nan_to_num(next_logits, nan=0.0, posinf=50.0, neginf=-50.0)

            # Tokenizer sözlük sınırı dışındaki indeksleri maskele
            if valid_vocab_limit < self.active_model.vocab_size:
                next_logits[:, valid_vocab_limit:] = float("-inf")

            # Tekrarlama cezası (Repetition & Frequency Penalty) uygula
            if self.repetition_penalty != 1.0 or self.frequency_penalty != 0.0:
                all_tokens = tokens + generated_tokens
                context_tensor = torch.tensor([all_tokens], dtype=torch.long)
                next_logits = apply_repetition_penalties(
                    logits=next_logits,
                    input_ids=context_tensor,
                    repetition_penalty=self.repetition_penalty,
                    frequency_penalty=self.frequency_penalty,
                )

            if self.temperature <= 1e-4 or self.top_k == 1:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = apply_sampling_filters(
                    logits=next_logits,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    min_p=self.min_p,
                )
                probs = F.softmax(next_logits, dim=-1)
                probs = torch.clamp(probs, min=1e-9)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()

            # Erken Durdurma (Stop Token): Durdurma ID'si tespit edildiğinde hemen kes
            if token_id in self.stop_token_ids:
                break

            char_or_token = self.tokenizer.decode([token_id])

            # String seviyesinde durdurma deseni kontrolü
            if any(sw in char_or_token for sw in self.stop_words):
                break

            generated_tokens.append(token_id)

            # Yeni tokeni anında ekrana ilet
            if char_or_token:
                yield char_or_token
            if delay > 0:
                time.sleep(delay)

            # KV Cache ile sadece yeni tokeni besle (O(1) sonraki adım!)
            with torch.no_grad():
                logits, _, kv_caches = self.active_model(
                    next_token,
                    kv_caches=kv_caches,
                    use_cache=True,
                    start_pos=curr_pos,
                )
            curr_pos += 1

        # Yanıtı hafızaya ekle
        full_reply = self.tokenizer.decode(generated_tokens)
        self.history.append(f"Kullanıcı: {prompt}")
        self.history.append(f"LixAI: {full_reply}")

    def get_stats(self) -> Dict[str, Any]:
        """Oturum ve model donanım istatistiklerini döner."""
        return {
            "params": self.raw_model.get_num_params(),
            "model_size_mb": get_model_size_mb(self.active_model),
            "quantized": self.use_quantized,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop_tokens": sorted(list(self.stop_token_ids)),
            "stop_words": self.stop_words,
            "history_len": len(self.history) // 2,
            "threads": torch.get_num_threads(),
            "checkpoint": self.checkpoint_name or "Yok (Rastgele Ağırlıklar)",
        }

    def run_interactive_loop(self) -> None:
        """Terminalde kullanıcı ile canlı döngüyü başlatır."""
        print(f"\n{BOLD}{CYAN}══════════════════════════════════════════════════════════{RESET}")
        print(f"{BOLD}{GREEN} 💬 LixAI İnteraktif Terminal Sohbet Modu (v2.0 SLM) {RESET}")
        print(f"{CYAN} Komutlar: /help | /stats | /clear | /penalty | /topp | /minp | /stop | /quantize | /exit {RESET}")
        print(f"{BOLD}{CYAN}══════════════════════════════════════════════════════════{RESET}\n")

        while True:
            try:
                user_input = input(f"{BOLD}{GREEN}Siz > {RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{YELLOW}Sohbet oturumu sonlandırıldı.{RESET}")
                break

            if not user_input:
                continue

            # Komut Kontrolleri
            if user_input in ("/exit", "/quit"):
                print(f"{YELLOW}Görüşmek üzere! Oturum kapatıldı.{RESET}")
                break
            elif user_input == "/clear":
                self.history.clear()
                print(f"{MAGENTA}[Sohbet geçmişi temizlendi]{RESET}")
                continue
            elif user_input == "/stats":
                stats = self.get_stats()
                print(f"\n{BOLD}{MAGENTA}📊 Model & Donanım İstatistikleri:{RESET}")
                print(f"  • Yüklü Kontrol Noktası: {stats['checkpoint']}")
                print(f"  • Parametre Sayısı: {stats['params']:,}")
                print(f"  • Bellek Boyutu: {stats['model_size_mb']} MB")
                print(f"  • Kuantize (INT8): {stats['quantized']}")
                print(f"  • CPU Thread: {stats['threads']}")
                print(f"  • Sıcaklık: {stats['temperature']} | Top-K: {stats['top_k']} | Top-P: {stats['top_p']} | Min-P: {stats['min_p']}")
                print(f"  • Tekrarlama Cezası (Rep Penalty): {stats['repetition_penalty']}")
                print(f"  • Stop Token IDs: {stats['stop_tokens']} | Stop Kelimeleri: {stats['stop_words']}\n")
                continue
            elif user_input == "/quantize":
                self.set_quantization(not self.use_quantized)
                durum = "AÇIK (INT8)" if self.use_quantized else "KAPALI (FP32)"
                print(f"{MAGENTA}[Kuantizasyon modu değiştirildi: {durum}]{RESET}")
                continue
            elif user_input.startswith("/penalty"):
                parts = user_input.split()
                if len(parts) > 1:
                    try:
                        self.repetition_penalty = float(parts[1])
                        print(f"{MAGENTA}[Tekrarlama cezası güncellendi: {self.repetition_penalty}]{RESET}")
                    except ValueError:
                        print(f"{YELLOW}Geçersiz değer! Örn: /penalty 1.2{RESET}")
                else:
                    print(f"{MAGENTA}Mevcut tekrarlama cezası: {self.repetition_penalty}{RESET}")
                continue
            elif user_input.startswith("/topp"):
                parts = user_input.split()
                if len(parts) > 1:
                    try:
                        self.top_p = float(parts[1])
                        print(f"{MAGENTA}[Top-P eşiği güncellendi: {self.top_p}]{RESET}")
                    except ValueError:
                        print(f"{YELLOW}Geçersiz değer! Örn: /topp 0.9{RESET}")
                else:
                    print(f"{MAGENTA}Mevcut Top-P: {self.top_p}{RESET}")
                continue
            elif user_input.startswith("/minp"):
                parts = user_input.split()
                if len(parts) > 1:
                    try:
                        self.min_p = float(parts[1])
                        print(f"{MAGENTA}[Min-P taban oranı güncellendi: {self.min_p}]{RESET}")
                    except ValueError:
                        print(f"{YELLOW}Geçersiz değer! Örn: /minp 0.05{RESET}")
                else:
                    print(f"{MAGENTA}Mevcut Min-P: {self.min_p}{RESET}")
                continue
            elif user_input.startswith("/stop"):
                parts = user_input.split()
                if len(parts) > 2 and parts[1] == "add":
                    try:
                        val = int(parts[2])
                        self.stop_token_ids.add(val)
                        print(f"{MAGENTA}[Stop token ID eklendi: {val}]{RESET}")
                    except ValueError:
                        self.stop_words.append(parts[2])
                        if hasattr(self.tokenizer, "token_to_id"):
                            tid = self.tokenizer.token_to_id(parts[2])
                            if tid is not None:
                                self.stop_token_ids.add(tid)
                        print(f"{MAGENTA}[Stop kelimesi eklendi: '{parts[2]}']{RESET}")
                else:
                    print(f"{MAGENTA}Mevcut Stop Token ID'leri: {sorted(list(self.stop_token_ids))} | Stop Kelimeleri: {self.stop_words}{RESET}")
                continue
            elif user_input == "/help":
                print(f"\n{YELLOW}Komut Listesi:{RESET}")
                print("  /stats        : Donanım ve model durumunu gösterir.")
                print("  /clear        : Bağlam hafızasını sıfırlar.")
                print("  /penalty X    : Tekrarlama cezasını ayarlar (örn: /penalty 1.25).")
                print("  /topp X       : Top-P nucleus eşiğini ayarlar (örn: /topp 0.9).")
                print("  /minp X       : Min-P taban oranını ayarlar (örn: /minp 0.05).")
                print("  /stop [add X] : Durdurma tokenlerini gösterir / ekler (örn: /stop add 822).")
                print("  /quantize     : INT8 kuantizasyonunu açar / kapatır.")
                print("  /exit         : Sohbetten çıkar.\n")
                continue

            # Streaming Cevap Üretimi
            print(f"{BOLD}{CYAN}LixAI > {RESET}", end="", flush=True)
            for chunk in self.stream_response(user_input, max_tokens=35):
                print(chunk, end="", flush=True)
            print("\n")
