"""
SOTA-style reference captioner used purely as a comparison baseline for the
proposed GAM-LSTM-IAT framework (conference/comparison_plan.md).

Architecture: CNN encoder -> flattened spatial tokens -> standard
Transformer decoder with causal self-attention + cross-attention to the
visual tokens (the same family as CPTR / ViT-GPT2-style captioners; the
literature's dominant modern captioning recipe, as opposed to this
project's bespoke CNN+GAN+LSTM+GRU pipeline). No GAN, no LSTM, no IAT
augmentation, no multi-task classification heads -- a single-purpose,
well-established captioning architecture trained on identical data/splits
for an apples-to-apples comparison.

    F_map = CNN(X)                          spatial visual features (B, C, H, W)
    V     = Flatten(F_map) + PosEnc         visual token sequence (B, N, D)
    T     = Embed(caption_tokens) + PosEnc  text token sequence (B, L, D)
    H     = TransformerDecoder(T, memory=V, causal_mask)
    logits = Linear(H)                       next-token distribution
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.cnn.cnn import MultiDepthCNN
from models.text.text_encoder import PositionalEncoding


class TransformerCaptioner(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, cnn_depth: int = 39,
                 cnn_feature_dim: int = 512, decoder_layers: int = 4, decoder_heads: int = 8,
                 max_caption_len: int = 32, dropout: float = 0.3, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.max_caption_len = max_caption_len

        self.cnn = MultiDepthCNN(depth=cnn_depth, feature_dim=cnn_feature_dim)
        self.visual_proj = nn.Linear(cnn_feature_dim, embed_dim)
        self.visual_pos = PositionalEncoding(embed_dim, max_len=256)

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.text_pos = PositionalEncoding(embed_dim, max_len=max_caption_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=decoder_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        _, feat_map = self.cnn(images, return_map=True)
        b, c, h, w = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2)     # (B, N, C)
        tokens = self.visual_proj(tokens)                  # (B, N, D)
        return self.visual_pos(tokens)

    def forward(self, images: torch.Tensor, caption_input: torch.Tensor) -> torch.Tensor:
        """
        images:        (B, 3, H, W)
        caption_input: (B, L) teacher-forcing tokens (shifted right)
        returns logits: (B, L, vocab_size)
        """
        memory = self.encode_image(images)                                   # (B, N, D)
        tgt = self.text_pos(self.token_embed(caption_input))                  # (B, L, D)

        L = tgt.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(tgt.device)
        tgt_key_padding_mask = caption_input == self.pad_id

        hidden = self.decoder(tgt, memory, tgt_mask=causal_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return self.out_proj(hidden)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, bos_id: int, eos_id: int,
                  max_len: int | None = None, unk_id: int | None = 3) -> torch.Tensor:
        """Greedy autoregressive decoding.

        Bug fix (2026-09), applied for consistency with the proposed model's
        decoder (models/lmm/lmm.py): mask <unk>'s logit to -inf so it can
        never be the argmax pick. <unk> is never a valid generation target.
        """
        max_len = max_len or self.max_caption_len
        b = images.size(0)
        device = images.device
        memory = self.encode_image(images)

        tokens = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt = self.text_pos(self.token_embed(tokens))
            L = tgt.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(device)
            hidden = self.decoder(tgt, memory, tgt_mask=causal_mask)
            logits = self.out_proj(hidden[:, -1, :])
            if unk_id is not None:
                logits = logits.clone()
                logits[:, unk_id] = float("-inf")
            next_token = logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

        return tokens[:, 1:]  # drop BOS

    @torch.no_grad()
    def generate_beam(self, images: torch.Tensor, bos_id: int, eos_id: int,
                        max_len: int | None = None, beam_width: int = 5,
                        unk_id: int | None = 3) -> torch.Tensor:
        """Beam-search decoding, looped per batch element (mirrors LMMDecoder's
        generate_beam for a consistent, fair comparison at eval time)."""
        max_len = max_len or self.max_caption_len
        b = images.size(0)
        device = images.device
        memory = self.encode_image(images)

        results = []
        for i in range(b):
            mem_i = memory[i:i + 1]
            beams = [(torch.tensor([bos_id], device=device), 0.0, False)]

            for _ in range(max_len):
                candidates = []
                for seq, score, done in beams:
                    if done:
                        candidates.append((seq, score, done))
                        continue
                    tgt = self.text_pos(self.token_embed(seq.unsqueeze(0)))
                    L = tgt.size(1)
                    causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(device)
                    hidden = self.decoder(tgt, mem_i, tgt_mask=causal_mask)
                    logits = self.out_proj(hidden[:, -1, :]).squeeze(0)
                    if unk_id is not None:
                        logits = logits.clone()
                        logits[unk_id] = float("-inf")
                    log_probs = torch.log_softmax(logits, dim=-1)
                    topk_logp, topk_idx = log_probs.topk(beam_width)
                    for k in range(beam_width):
                        next_token = topk_idx[k].item()
                        new_seq = torch.cat([seq, topk_idx[k:k + 1]])
                        new_score = score + topk_logp[k].item()
                        candidates.append((new_seq, new_score, next_token == eos_id))

                candidates.sort(key=lambda c: c[1] / max(len(c[0]), 1), reverse=True)
                beams = candidates[:beam_width]
                if all(done for *_, done in beams):
                    break

            results.append(beams[0][0][1:])  # drop BOS

        max_out_len = max(len(r) for r in results)
        padded = torch.zeros(b, max_out_len, dtype=torch.long, device=device)
        for i, r in enumerate(results):
            padded[i, :len(r)] = r
        return padded

    def caption_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=self.pad_id,
        )
