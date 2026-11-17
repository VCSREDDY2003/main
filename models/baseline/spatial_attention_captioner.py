"""
Baseline 6 -- Single-head spatial (additive/Bahdanau) attention captioner,
i.e. a faithful "Show, Attend and Tell" (Xu et al., 2015) reimplementation on
this project's pipeline. Sits directly between Baseline 5 (CNNLSTMCaptioner,
inject-once, no attention at all) and the proposed model (multi-component
GAM-LSTM-IAT with per-step spatial cross-attention): it keeps the CNN's
spatial feature map (no global pooling collapse) and re-attends to it at
*every* decoding step with a single soft-attention head, but has none of the
proposed model's GAN augmentation, IAT, multi-task classification heads, or
multi-head/cross-attention decoder machinery. No GAN, no IAT, no multi-task
heads -- purely a captioning-only architecture, trained on identical
data/splits for an apples-to-apples comparison against both Baseline 5 and
the proposed spatial cross-attention model.

    A = {a_1, ..., a_N}, a_i in R^C      = Flatten(CNN(X))      spatial annotation vectors (B, N, C)
    h_0, c_0     = init_h(mean_i a_i), init_c(mean_i a_i)        LSTM initial state (Xu et al. Eq. 3-4)
    e_ti         = v^T tanh(W_a a_i + W_h h_{t-1})                additive/Bahdanau attention score
    alpha_t      = softmax_i(e_t)                                 attention weights over the N regions
    z_t          = sum_i alpha_ti * a_i                           context vector (soft attention, Eq. 13)
    beta_t       = sigmoid(W_beta h_{t-1})                        gating scalar on the context (Eq. 15)
    e_t          = Embed(y_{t-1})                                 word embedding at step t
    h_t, c_t     = LSTM([e_t ; beta_t * z_t], h_{t-1}, c_{t-1})    single-layer LSTM recurrence
    logits_t     = Linear(Dropout([h_t ; z_t ; e_t]))              "deep output" next-token distribution

Single soft-attention head, re-computed fresh every decoding step from the
CNN's raw spatial map -- unlike Baseline 5 (one-shot pooled-vector
injection) and unlike the proposed model (multi-head/cross-attention
Transformer-style memory attention with additional GAN/IAT/multi-task
conditioning).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.cnn.cnn import PretrainedResNetCNN


class SoftAttention(nn.Module):
    """Additive (Bahdanau-style) single-head spatial attention over N
    annotation vectors, exactly as in Xu et al. 2015 Sec. 3.1.2 / Eq. 4-6."""

    def __init__(self, annotation_dim: int, hidden_dim: int, attn_dim: int = 256):
        super().__init__()
        self.annotation_proj = nn.Linear(annotation_dim, attn_dim)
        self.hidden_proj = nn.Linear(hidden_dim, attn_dim)
        self.score_proj = nn.Linear(attn_dim, 1)

    def forward(self, annotations: torch.Tensor, h_prev: torch.Tensor):
        """
        annotations: (B, N, C)
        h_prev:      (B, hidden_dim)
        returns: context (B, C), alpha (B, N)
        """
        att_ann = self.annotation_proj(annotations)                 # (B, N, attn_dim)
        att_hid = self.hidden_proj(h_prev).unsqueeze(1)              # (B, 1, attn_dim)
        scores = self.score_proj(torch.tanh(att_ann + att_hid))      # (B, N, 1)
        alpha = torch.softmax(scores.squeeze(-1), dim=1)             # (B, N)
        context = torch.bmm(alpha.unsqueeze(1), annotations).squeeze(1)  # (B, C)
        return context, alpha


class SpatialAttentionCaptioner(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, cnn_feature_dim: int = 512,
                 hidden_dim: int = 512, attn_dim: int = 256, max_caption_len: int = 32,
                 dropout: float = 0.3, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.max_caption_len = max_caption_len
        self.hidden_dim = hidden_dim

        self.cnn = PretrainedResNetCNN(feature_dim=cnn_feature_dim)
        annotation_dim = self.cnn.raw_channels  # raw spatial-map channels (see cnn.py note)

        self.init_h = nn.Linear(annotation_dim, hidden_dim)
        self.init_c = nn.Linear(annotation_dim, hidden_dim)

        self.attention = SoftAttention(annotation_dim, hidden_dim, attn_dim=attn_dim)
        self.beta_gate = nn.Linear(hidden_dim, 1)

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTMCell(embed_dim + annotation_dim, hidden_dim)

        # "deep output" layer combining hidden state, context, and previous
        # word embedding (Xu et al. Eq. 7), rather than a bare Linear(h_t).
        self.out_proj = nn.Linear(hidden_dim + annotation_dim + embed_dim, vocab_size)

    def _encode(self, images: torch.Tensor):
        """Returns annotation vectors (B, N, C) and initial (h0, c0)."""
        _, feature_map = self.cnn(images, return_map=True)     # (B, C, H, W)
        b, c, h, w = feature_map.shape
        annotations = feature_map.flatten(2).transpose(1, 2)     # (B, N, C), N = H*W
        mean_ann = annotations.mean(dim=1)                        # (B, C)
        h0 = torch.tanh(self.init_h(mean_ann))
        c0 = torch.tanh(self.init_c(mean_ann))
        return annotations, h0, c0

    def _step(self, prev_token: torch.Tensor, annotations: torch.Tensor, h, c):
        e = self.token_embed(prev_token)                          # (B, E)
        context, alpha = self.attention(annotations, h)           # (B, C), (B, N)
        beta = torch.sigmoid(self.beta_gate(h))                    # (B, 1) -- doubly-stochastic gate
        gated_context = beta * context
        lstm_in = torch.cat([e, gated_context], dim=-1)
        h_new, c_new = self.lstm(lstm_in, (h, c))
        deep_out = torch.cat([h_new, context, e], dim=-1)
        logits = self.out_proj(self.dropout(deep_out))
        return logits, h_new, c_new, alpha

    def forward(self, images: torch.Tensor, caption_input: torch.Tensor) -> torch.Tensor:
        """
        images:        (B, 3, H, W)
        caption_input: (B, L) teacher-forcing tokens (shifted right)
        returns logits: (B, L, vocab_size)
        """
        annotations, h, c = self._encode(images)
        b, L = caption_input.shape

        logits = []
        for t in range(L):
            step_logits, h, c, _ = self._step(caption_input[:, t], annotations, h, c)
            logits.append(step_logits)
        return torch.stack(logits, dim=1)                          # (B, L, vocab_size)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, bos_id: int, eos_id: int,
                  max_len: int | None = None, unk_id: int | None = 3) -> torch.Tensor:
        max_len = max_len or self.max_caption_len
        b = images.size(0)
        device = images.device
        annotations, h, c = self._encode(images)

        cur = torch.full((b,), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        outputs = []
        for _ in range(max_len):
            logits, h, c, _ = self._step(cur, annotations, h, c)
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
        annotations_all, h0_all, c0_all = self._encode(images)

        results = []
        for i in range(b):
            ann_i = annotations_all[i:i + 1]
            beams = [(torch.tensor([bos_id], device=device), h0_all[i:i + 1], c0_all[i:i + 1], 0.0, False)]
            for _ in range(max_len):
                candidates = []
                for seq, h, c, score, done in beams:
                    if done:
                        candidates.append((seq, h, c, score, done))
                        continue
                    cur = seq[-1:].to(device)
                    logits, h_new, c_new, _ = self._step(cur, ann_i, h, c)
                    logits = logits.squeeze(0)
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
