"""
LixAI Çekirdek Transformer Modeli (Causal Self-Attention Language Model)
Intel AVX2, Çoklu Thread ve KV Cache (Key-Value Caching) Hızlandırması içerir.
"""
import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Maskelenmiş Çok Başlıklı Nedensel Öz-Dikkat & KV Cache Destekli Attention."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model başlık sayısına tam bölünmelidir."
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q, K, V projeksiyonları (CPU önbellek verimi için birleşik matris)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Nedensel maske (Causal triangular mask)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        b, t, c = x.size()

        # [B, T, 3 * d_model] -> Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        # KV Cache Entegrasyonu
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        new_cache = (k, v) if use_cache else None
        total_len = k.size(-2)

        # Ölçeklenmiş Nokta Çarpım Dikkati
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Eğer tek bir yeni token çıkarımı yapılıyorsa ve geçmiş cache varsa maskelemeye gerek yoktur
        if t > 1:
            att = att.masked_fill(self.mask[:, :, :t, :total_len] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v  # [B, n_heads, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # [B, T, d_model]
        out = self.out_proj(y)

        return out, new_cache


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Ağı."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Bloğu."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_cache


def apply_repetition_penalties(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> torch.Tensor:
    """
    Logit seviyesinde tekrarlama ve frekans cezaları uygulayarak sonsuz döngüleri engeller.
    - repetition_penalty: Çarpımsal ceza (>1.0: pozitif logitler / penalty, negatif logitler * penalty)
    - frequency_penalty: Frekansa bağlı toplamsal ceza (logit -= count * penalty)
    - presence_penalty: Varlığa bağlı sabit toplamsal ceza (logit -= penalty)
    """
    if repetition_penalty == 1.0 and frequency_penalty == 0.0 and presence_penalty == 0.0:
        return logits

    batch_size, vocab_size = logits.size()
    for b in range(batch_size):
        seq = input_ids[b]
        valid_mask = (seq >= 0) & (seq < vocab_size)
        valid_tokens = seq[valid_mask]
        if valid_tokens.numel() == 0:
            continue

        counts = torch.bincount(valid_tokens, minlength=vocab_size)
        used_indices = torch.nonzero(counts > 0, as_tuple=True)[0]
        if used_indices.numel() == 0:
            continue

        token_logits = logits[b, used_indices]

        if repetition_penalty != 1.0:
            logits[b, used_indices] = torch.where(
                token_logits > 0,
                token_logits / repetition_penalty,
                token_logits * repetition_penalty,
            )

        if frequency_penalty != 0.0:
            logits[b, used_indices] -= frequency_penalty * counts[used_indices].float()

        if presence_penalty != 0.0:
            logits[b, used_indices] -= presence_penalty

    return logits


def apply_sampling_filters(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Logit seviyesinde sıcaklık (temperature), Top-K, Min-P ve Top-P (Nucleus) filtreleme uygular.
    Filtrelenen indekslerin logitleri -inf yapılır.
    """
    if temperature <= 1e-4:
        return logits

    logits = logits / max(temperature, 1e-4)

    # 1. Top-K Filtreleme
    if top_k is not None and top_k > 0 and top_k < logits.size(-1):
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

    # 2. Min-P Filtreleme (Zheng et al., 2024: threshold = p_max * min_p)
    if min_p is not None and min_p > 0.0 and min_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1, keepdim=True)
        min_p_threshold = max_probs * min_p
        logits[probs < min_p_threshold] = float("-inf")

    # 3. Top-P (Nucleus) Filtreleme
    if top_p is not None and top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Kümülatif olasılığı top_p eşiğini aşan tokenleri belirle
        sorted_indices_to_remove = cumulative_probs > top_p
        # Eşiği ilk aşan tokeni de korumak için maskeyi 1 sağa kaydır
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Orijinal logit indeks dizilimine geri eşle
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")

    return logits


class LixLanguageModel(nn.Module):
    """
    Sıfırdan tasarlanan Causal Transformer Dil Modeli (GPT mimarisi).
    CPU odaklı eğitim, çıkarım ve KV Cache desteği içerir.
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        weight_tying: bool = False,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.weight_tying = weight_tying

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Ağırlık paylaşımı (Weight Tying) opsiyonel
        if weight_tying:
            self.lm_head.weight = self.tok_emb.weight

        # Ağırlık başlatma (Weight Initialization)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self) -> int:
        """Eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def param_count(self) -> int:
        """Toplam parametre sayısını döndürür."""
        return self.get_num_params()

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        b, t = input_ids.size()
        total_len = start_pos + t
        assert total_len <= self.max_seq_len, f"Dizi uzunluğu {total_len}, sınır {self.max_seq_len}'i aşamaz."

        positions = torch.arange(start_pos, total_len, dtype=torch.long, device=input_ids.device)

        tok_embeddings = self.tok_emb(input_ids)
        pos_embeddings = self.pos_emb(positions)
        x = self.drop(tok_embeddings + pos_embeddings)

        new_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, updated_cache = block(x, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                new_caches.append(updated_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-100)

        if use_cache:
            return logits, loss, new_caches
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        use_cache: bool = True,
        max_vocab_id: Optional[int] = None,
    ) -> torch.Tensor:
        """KV Cache, tekrarlama cezası ve dinamik örnekleme (Top-P, Min-P, Top-K) destekli metin üretimi."""
        self.eval()

        if not use_cache:
            # Klasik O(T^2) döngüsü
            for _ in range(max_new_tokens):
                idx_cond = input_ids[:, -self.max_seq_len:]
                logits, _ = self(idx_cond)
                next_logits = logits[:, -1, :]
                next_logits = torch.nan_to_num(next_logits, nan=0.0, posinf=50.0, neginf=-50.0)

                if max_vocab_id is not None and max_vocab_id < self.vocab_size:
                    next_logits[:, max_vocab_id:] = float("-inf")

                if repetition_penalty != 1.0 or frequency_penalty != 0.0 or presence_penalty != 0.0:
                    next_logits = apply_repetition_penalties(
                        logits=next_logits,
                        input_ids=input_ids,
                        repetition_penalty=repetition_penalty,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                    )

                if temperature <= 1e-4 or top_k == 1:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    next_logits = apply_sampling_filters(
                        logits=next_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        min_p=min_p,
                    )
                    probs = F.softmax(next_logits, dim=-1)
                    probs = torch.clamp(probs, min=1e-9)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat((input_ids, next_token), dim=1)
            return input_ids

        # Hızlı O(T) KV Cache Döngüsü
        # İlk adım: tüm başlangıç dizisi işlenir ve önbellek doldurulur
        logits, _, kv_caches = self(input_ids, use_cache=True, start_pos=0)
        curr_pos = input_ids.size(1)

        for _ in range(max_new_tokens):
            if curr_pos >= self.max_seq_len:
                break

            next_logits = logits[:, -1, :]
            next_logits = torch.nan_to_num(next_logits, nan=0.0, posinf=50.0, neginf=-50.0)

            if max_vocab_id is not None and max_vocab_id < self.vocab_size:
                next_logits[:, max_vocab_id:] = float("-inf")

            if repetition_penalty != 1.0 or frequency_penalty != 0.0 or presence_penalty != 0.0:
                next_logits = apply_repetition_penalties(
                    logits=next_logits,
                    input_ids=input_ids,
                    repetition_penalty=repetition_penalty,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )

            if temperature <= 1e-4 or top_k == 1:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = apply_sampling_filters(
                    logits=next_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                )
                probs = F.softmax(next_logits, dim=-1)
                probs = torch.clamp(probs, min=1e-9)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_token), dim=1)

            # Sonraki adım: sadece YENİ TOKEN (shape=[B, 1]) ve mevcut cache gönderilir!
            logits, _, kv_caches = self(
                next_token,
                kv_caches=kv_caches,
                use_cache=True,
                start_pos=curr_pos,
            )
            curr_pos += 1

        return input_ids
