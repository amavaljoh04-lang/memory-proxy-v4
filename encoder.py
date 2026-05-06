"""
TriVox2 encoder — matches the actual trained model architecture.
Standard Transformer (6 layers, attention + FFN), multi-vocab embeddings.
Always runs on CPU to leave GPU for Ollama.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from tokenizers import Tokenizer


class TransformerLayer(nn.Module):
    """Standard pre-norm Transformer encoder layer."""
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Combined QKV projection
        self.attn_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.attn_out = nn.Linear(d_model, d_model)
        # FFN
        self.ffn_fc1 = nn.Linear(d_model, d_ff)
        self.ffn_fc2 = nn.Linear(d_ff, d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x, mask=None):
        # Self-attention with pre-norm
        h = self.norm1(x)
        B, L, D = h.shape
        qkv = self.attn_qkv(h).reshape(B, L, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.d_head ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        x = x + self.attn_out(out)

        # FFN with pre-norm
        h = self.norm2(x)
        x = x + self.ffn_fc2(F.gelu(self.ffn_fc1(h)))
        return x


class TriVox2Model(nn.Module):
    """The actual TriVox2 model architecture matching the checkpoint."""
    def __init__(self, d_model=512, n_layers=6, n_heads=8, d_ff=2048,
                 vocab_fr=50000, vocab_en=50000, vocab_code=32000,
                 max_seq_len=256, d_out=768):
        super().__init__()
        # Multi-vocab embeddings
        self.embed_fr = nn.Embedding(vocab_fr, d_model)
        self.embed_en = nn.Embedding(vocab_en, d_model)
        self.embed_code = nn.Embedding(vocab_code, d_model)
        self.lang_embed = nn.Embedding(3, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        # Output projection
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_out),
            nn.LayerNorm(d_out)
        )

    def forward(self, ids, lang_id=0, mask=None):
        B, L = ids.shape

        # Embed tokens based on language
        if lang_id == 0:
            x = self.embed_fr(ids)
        elif lang_id == 1:
            x = self.embed_en(ids)
        else:
            x = self.embed_code(ids)

        # Add positional + language embeddings
        positions = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(positions)
        lang_t = torch.full((B,), lang_id, device=ids.device, dtype=torch.long)
        x = x + self.lang_embed(lang_t).unsqueeze(1)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Mean pooling (with mask)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)

        # Project to output dim
        return self.proj(pooled)


class Encoder:
    """TriVox2 encoder wrapper. Loads model + tokenizers on CPU."""

    def __init__(self, model_path: str, tok_fr: str, tok_en: str, tok_code: str,
                 max_seq_len: int = 256):
        self.device = torch.device("cpu")
        self.max_seq_len = max_seq_len

        # Load model
        print(f"[Encoder] Loading TriVox2 on CPU...")
        self.model = TriVox2Model(max_seq_len=max_seq_len)

        raw = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "model" in raw:
            sd = raw["model"]
        elif isinstance(raw, dict) and "model_state_dict" in raw:
            sd = raw["model_state_dict"]
        else:
            sd = raw

        # Map checkpoint keys to model keys
        mapped = {}
        for k, v in sd.items():
            new_k = k
            # Map attn keys: "layers.N.attn.qkv.weight" -> "layers.N.attn_qkv.weight"
            new_k = new_k.replace(".attn.qkv.", ".attn_qkv.")
            new_k = new_k.replace(".attn.out.", ".attn_out.")
            # Map ffn keys: "layers.N.ffn.fc1." -> "layers.N.ffn_fc1."
            new_k = new_k.replace(".ffn.fc1.", ".ffn_fc1.")
            new_k = new_k.replace(".ffn.fc2.", ".ffn_fc2.")
            # Map positional: "pos_embed.weight" -> "pos_embed.weight" (same)
            mapped[new_k] = v

        missing, unexpected = self.model.load_state_dict(mapped, strict=False)
        if missing:
            print(f"[Encoder] Warning: {len(missing)} missing keys")
        if unexpected:
            print(f"[Encoder] Warning: {len(unexpected)} unexpected keys")
        self.model.eval()
        print(f"[Encoder] Model loaded from {model_path}")

        # Load tokenizers
        self.tok_fr = spm.SentencePieceProcessor(model_file=tok_fr)
        self.tok_en = spm.SentencePieceProcessor(model_file=tok_en)
        self.tok_code = Tokenizer.from_file(tok_code)
        print(f"[Encoder] Tokenizers loaded (FR/EN/Code)")

    def detect_lang(self, text: str) -> int:
        """0=FR, 1=EN, 2=CODE."""
        t = text[:500].lower()
        fr = sum(1 for w in ["le ", "la ", "les ", "de ", "du ", "est ", "une ",
                             "je ", "tu ", "nous ", "mon ", "dans "]
                 if w in t)
        code = sum(1 for w in ["def ", "class ", "import ", "return ", "function ",
                               "const ", "var ", "if (", "for (", "while (", "self."]
                  if w in t)
        if code >= 2:
            return 2
        if fr >= 2:
            return 0
        return 1

    @torch.no_grad()
    def encode(self, text: str) -> list[float]:
        """Encode a single text to 768-dim vector (L2-normalized for cosine)."""
        lang = self.detect_lang(text)

        # Tokenize
        if lang == 0:
            ids = self.tok_fr.encode(str(text), out_type=int)[:self.max_seq_len]
        elif lang == 1:
            ids = self.tok_en.encode(str(text), out_type=int)[:self.max_seq_len]
        else:
            ids = self.tok_code.encode(str(text)).ids[:self.max_seq_len]

        if not ids:
            return [0.0] * 768

        # Pad
        pad = self.max_seq_len - len(ids)
        ids_t = torch.tensor([ids + [0] * pad], dtype=torch.long, device=self.device)
        mask_t = torch.tensor([[1] * len(ids) + [0] * pad], dtype=torch.long, device=self.device)

        # Forward
        emb = self.model(ids_t, lang, mask_t).squeeze(0)

        # L2 normalize for cosine similarity
        emb = emb / emb.norm().clamp(min=1e-8)
        return emb.numpy().tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts."""
        return [self.encode(t) for t in texts]
