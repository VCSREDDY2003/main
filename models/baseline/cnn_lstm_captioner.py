"""
Baseline 5 -- Plain CNN+LSTM "inject once" captioner. The minimal possible
encoder-decoder comparison point for the proposed GAM-LSTM-IAT model: a
pretrained CNN backbone reduced to a single pooled global feature vector
(no spatial map retained, no attention of any kind at any decoding step),
feeding a single-layer LSTM decoder that is conditioned on the image only
once, at the first decoding step (classic Vinyals et al. "Show and Tell"
inject-once pattern). No GAN, no IAT augmentation, no multi-task
classification heads, no attention mechanism -- purely a captioning-only
architecture, trained on identical data/splits for an apples-to-apples
comparison against the proposed spatial cross-attention model.

    f_v          = Pool(CNN(X))                     global visual vector (B, C)
    v0           = Linear(f_v)                       projected visual vector (B, D), used ONLY
                                                       to init the LSTM hidden/cell state
    h_0, c_0     = tanh(v0), tanh(v0)                 LSTM initial state (inject-once)
    e_t          = Embed(y_{t-1})                     word embedding at step t
    h_t, c_t     = LSTM(e_t, h_{t-1}, c_{t-1})        single-layer LSTM recurrence
    logits_t     = Linear(h_t)                        next-token distribution

No per-step re-conditioning on the image: after the initial hidden-state
injection, the decoder only ever sees token embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.cnn.cnn import PretrainedResNetCNN


class CNNLSTMCaptioner(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, cnn_feature_dim: int = 512,
                 hidden_dim: int = 512, max_caption_len: int = 32, dropout: float = 0.3,
                 pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.max_caption_len = max_caption_len
        self.hidden_dim = hidden_dim

        self.cnn = PretrainedResNetCNN(feature_dim=cnn_feature_dim)
        self.visual_init = nn.Linear(cnn_feature_dim, hidden_dim)

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTMCell(embed_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)

    def _init_state(self, images: torch.Tensor):
        pooled = self.cnn(images, return_map=False)          # (B, cnn_feature_dim)
        v0 = torch.tanh(self.visual_init(pooled))              # (B, hidden_dim) -- inject-once
        return v0, v0.clone()                                   # (h0, c0)

    def forward(self, images: torch.Tensor, caption_input: torch.Tensor) -> torch.Tensor:
        """
        images:        (B, 3, H, W)
        caption_input: (B, L) teacher-forcing tokens (shifted right)
        returns logits: (B, L, vocab_size)
        """
        b, L = caption_input.shape
        h, c = self._init_state(images)
        embeds = self.dropout(self.token_embed(caption_input))   # (B, L, E)

        logits = []
        for t in range(L):
            h, c = self.lstm(embeds[:, t, :], (h, c))
            logits.append(self.out_proj(self.dropout(h)))
        return torch.stack(logits, dim=1)                          # (B, L, vocab_size)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, bos_id: int, eos_id: int,
                  max_len: int | None = None, unk_id: int | None = 3) -> torch.Tensor:
        max_len = max_len or self.max_caption_len
        b = images.size(0)
        device = images.device
        h, c = self._init_state(images)

        tokens = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        cur = tokens[:, 0]

        outputs = []
        for _ in range(max_len):
            e = self.token_embed(cur)
            h, c = self.lstm(e, (h, c))
            logits = self.out_proj(h)
            if unk_id is not None:
                logits = logits.clone()
                logits[:, unk_id] = float("-inf")
            next_token = logits.argmax(dim=-1)
            outputs.append(next_token)
            finished = finished | (next_token == eos_id)
            cur = next_token
            if finished.all():
                break

        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def generate_beam(self, images: torch.Tensor, bos_id: int, eos_id: int,
                        max_len: int | None = None, beam_width: int = 5,
                        unk_id: int | None = 3) -> torch.Tensor:
        max_len = max_len or self.max_caption_len
        b = images.size(0)
        device = images.device
        h0_all, c0_all = self._init_state(images)

        results = []
        for i in range(b):
            beams = [(torch.tensor([bos_id], device=device), h0_all[i:i + 1], c0_all[i:i + 1], 0.0, False)]
            for _ in range(max_len):
                candidates = []
                for seq, h, c, score, done in beams:
                    if done:
                        candidates.append((seq, h, c, score, done))
                        continue
                    cur = seq[-1:].to(device)
                    e = self.token_embed(cur)
                    h_new, c_new = self.lstm(e, (h, c))
                    logits = self.out_proj(h_new).squeeze(0)
                    if unk_id is not None:
                        logits = logits.clone()
                        logits[unk_id] = float("-inf")
                    log_probs = torch.log_softmax(logits, dim=-1)
                    topk_logp, topk_idx = log_probs.topk(beam_width)
                    for k in range(beam_width):
                        next_token = topk_idx[k].item()
                        new_seq = torch.cat([seq, topk_idx[k:k + 1]])
                        new_score = score + topk_logp[k].item()
                        candidates.append((new_seq, h_new, c_new, new_score, next_token == eos_id))

                candidates.sort(key=lambda x: x[3] / max(len(x[0]), 1), reverse=True)
                beams = candidates[:beam_width]
                if all(done for *_, done in beams):
                    break

            results.append(beams[0][0][1:])

        max_out_len = max(len(r) for r in results)
        padded = torch.zeros(b, max_out_len, dtype=torch.long, device=device)
        for i, r in enumerate(results):
            padded[i, :len(r)] = r
        return padded

    def caption_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=self.pad_id,
        )
