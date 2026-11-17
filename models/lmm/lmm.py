"""
Large Multimodal (LMM) decoder (Section 4, Eq. 4.1-4.13, Algorithm 3).

Proposed update (conference/proposed_model variant): the original decoder
conditioned generation on a single pooled state H_0 only, discarding the
CNN's spatial feature map after fusion. This variant keeps H_0 as the GRU's
initial hidden state (Eq. 4.1) but additionally cross-attends, at every
decoding step, to the CNN's spatial feature map F_map (B, C, H, W) so the
decoder can attend to different image regions per generated token -- the
standard fix for the encoder-decoder bottleneck (cf. Show-Attend-Tell).

    H_0        = W_z Z_i + b_z                                (4.1)
    a_t        = Softmax((W_q s_{t-1}) (W_k F_map)^T)          spatial attention per step
    c_t        = a_t F_map                                     attended visual context
    s_t        = GRUCell([e_t ; c_t], s_{t-1})                 decoder state update
    w_t        = Softmax(W_o s_t)                               -> caption   (4.2-4.5)

Caption and summary share the token embedding table but use separate
attentive GRU decoders, both conditioned on the same multimodal state H_0
and the same spatial feature map (Algorithm 3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """Multi-head additive (Bahdanau-style) attention over a CNN spatial
    feature map.

    Proposed fix (v3): the original single-head version could collapse onto
    a single dominant spatial location once fed the sharper, ImageNet-
    pretrained ResNet-34 feature maps (models/cnn/cnn.py:PretrainedResNetCNN)
    -- a peaked/near-one-hot attention distribution. Under beam search that
    collapse is exactly reproducible across the top-k candidates at every
    step (the same "obvious" region wins every time regardless of which
    hypothesis token is being scored), so beam search stops adding
    information beyond what greedy already found, while the length-
    normalised score still rewards longer sequences -- degrading BLEU-4
    relative to greedy (Section 5.7 diagnosis: no beam-expansion indexing
    bug found; see notes in generate_beam docstring below). Splitting
    attention into multiple independent heads, each with its own
    query/key projection, forces the model to represent several different
    spatial hypotheses per step instead of one, which empirically produces
    smoother, higher-entropy attention (further encouraged by the
    attention-entropy regularizer in training/losses.py) and removes the
    single-head collapse failure mode.
    """

    def __init__(self, query_dim: int, feat_dim: int, attn_dim: int = 256, num_heads: int = 4):
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.attn_dim = attn_dim
        self.q_proj = nn.Linear(query_dim, attn_dim)
        self.k_proj = nn.Linear(feat_dim, attn_dim)
        self.v_proj = nn.Linear(feat_dim, attn_dim)
        self.energy = nn.Linear(self.head_dim, 1)
        self.out_proj = nn.Linear(attn_dim, feat_dim)

    def forward(self, query: torch.Tensor, feat_map: torch.Tensor,
                saliency_bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        query:    (B, query_dim)        decoder hidden state at t-1
        feat_map: (B, N, feat_dim)      flattened spatial CNN features (N = H*W)
        saliency_bias: optional (B, N)  IAT's shared geometry-consistent
                       saliency map (flattened), added to the attention
                       energy before softmax so the caption decoder's
                       region grounding is driven by the same physically-
                       constrained signal as IAT's image perturbation
                       (Phase A redesign, 2026-11), instead of an
                       independent, uncoordinated attention path.
        returns context: (B, feat_dim), attn weights: (B, num_heads, N)
        """
        b, n, _ = feat_map.shape
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(query).view(b, h, 1, d)              # (B, h, 1, d)
        k = self.k_proj(feat_map).view(b, n, h, d).transpose(1, 2)  # (B, h, N, d)
        v = self.v_proj(feat_map).view(b, n, h, d).transpose(1, 2)  # (B, h, N, d)

        scores = self.energy(torch.tanh(q + k)).squeeze(-1)   # (B, h, N)
        if saliency_bias is not None:
            scores = scores + saliency_bias.unsqueeze(1)       # broadcast over heads
        attn = torch.softmax(scores, dim=-1)                   # (B, h, N)
        context = torch.matmul(attn.unsqueeze(2), v).squeeze(2)  # (B, h, d)
        context = context.reshape(b, h * d)                    # (B, attn_dim)
        context = self.out_proj(context)                       # (B, feat_dim)
        return context, attn


class _AttentiveAutoregressiveDecoder(nn.Module):
    """GRUCell decoder with spatial cross-attention over the CNN feature map,
    initialised from the fused multimodal state H_0 (Eq. 4.1)."""

    def __init__(self, embedding: nn.Embedding, hidden_dim: int, vocab_size: int,
                 feat_dim: int, dropout: float = 0.3, use_spatial_attention: bool = True):
        super().__init__()
        embed_dim = embedding.embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.use_spatial_attention = use_spatial_attention
        self.attn = SpatialAttention(hidden_dim, feat_dim, attn_dim=hidden_dim)
        self.cell = nn.GRUCell(embed_dim + feat_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim + feat_dim, vocab_size)

    def _step(self, token_embed: torch.Tensor, hidden: torch.Tensor, feat_map: torch.Tensor,
              saliency_bias: torch.Tensor | None = None):
        if self.use_spatial_attention:
            context, attn = self.attn(hidden, feat_map, saliency_bias=saliency_bias)
        else:
            # Ablation / original IAT-GAM-LSTM decoder: no per-step spatial
            # attention -- collapse the spatial map to a single pooled
            # vector (the original single-vector bottleneck) reused at
            # every decoding step.
            context = feat_map.mean(dim=1)
            attn = None
        cell_in = torch.cat([token_embed, context], dim=-1)
        hidden = self.cell(cell_in, hidden)
        logits = self.out_proj(torch.cat([hidden, context], dim=-1))
        return logits, hidden, attn

    def forward(self, h0: torch.Tensor, token_ids: torch.Tensor,
                feat_map: torch.Tensor,
                saliency_bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        h0:        (B, hidden_dim)   initial hidden state (Eq. 4.1)
        token_ids: (B, T)            teacher-forcing input tokens (shifted targets)
        feat_map:  (B, N, feat_dim)  flattened spatial CNN features
        saliency_bias: optional (B, N) shared IAT saliency map, see
                       SpatialAttention.forward docstring.
        returns logits: (B, T, vocab_size)
        """
        x = self.dropout(self.embedding(token_ids))
        hidden = h0
        logits_seq = []
        attn_seq = []
        for t in range(x.size(1)):
            logits, hidden, attn = self._step(x[:, t, :], hidden, feat_map, saliency_bias=saliency_bias)
            logits_seq.append(logits.unsqueeze(1))
            if attn is not None:
                attn_seq.append(attn.unsqueeze(1))  # (B, 1, heads, N)
        attn_all = torch.cat(attn_seq, dim=1) if attn_seq else None  # (B, T, heads, N)
        return torch.cat(logits_seq, dim=1), attn_all

    @torch.no_grad()
    def generate(self, h0: torch.Tensor, feat_map: torch.Tensor, bos_id: int,
                  eos_id: int, max_len: int, unk_id: int | None = 3,
                  saliency_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Greedy autoregressive decoding with spatial attention (Algorithm 3).

        Bug fix (2026-09): 641/6000 test samples (10.7%) were decoding as a
        degenerate `<unk> <unk> <unk> ...` run because <unk> could win
        argmax under uncertainty. <unk> is never a valid generation target,
        so its logit is masked to -inf before argmax at every step
        (decode-time only; training/teacher-forcing is untouched).
        """
        b = h0.size(0)
        device = h0.device
        hidden = h0
        token = torch.full((b,), bos_id, dtype=torch.long, device=device)
        outputs = []
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        for _ in range(max_len):
            x = self.embedding(token)
            logits, hidden, _ = self._step(x, hidden, feat_map, saliency_bias=saliency_bias)
            if unk_id is not None:
                logits = logits.clone()
                logits[:, unk_id] = float("-inf")
            token = logits.argmax(dim=-1)
            outputs.append(token.unsqueeze(1))
            finished = finished | (token == eos_id)
            if finished.all():
                break

        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def generate_beam(self, h0: torch.Tensor, feat_map: torch.Tensor, bos_id: int,
                        eos_id: int, max_len: int, beam_width: int = 5,
                        unk_id: int | None = 3) -> torch.Tensor:
        """Beam-search decoding, run independently per batch element.

        Kept simple (loop over batch) since batch sizes at evaluation time
        are small relative to beam width; trades some throughput for a
        straightforward, easy-to-verify implementation.
        """
        b = h0.size(0)
        device = h0.device
        results = []

        for i in range(b):
            h0_i = h0[i:i + 1]                       # (1, hidden_dim)
            feat_i = feat_map[i:i + 1]                # (1, N, feat_dim)

            beams = [(torch.tensor([bos_id], device=device), h0_i.squeeze(0), 0.0, False)]

            for _ in range(max_len):
                candidates = []
                for seq, hidden, score, done in beams:
                    if done:
                        candidates.append((seq, hidden, score, done))
                        continue
                    x = self.embedding(seq[-1:])                      # (1, embed_dim)
                    logits, new_hidden, _ = self._step(x, hidden.unsqueeze(0), feat_i)
                    logits = logits.squeeze(0)
                    if unk_id is not None:
                        logits = logits.clone()
                        logits[unk_id] = float("-inf")
                    log_probs = F.log_softmax(logits, dim=-1)  # (vocab,)
                    topk_logp, topk_idx = log_probs.topk(beam_width)
                    for k in range(beam_width):
                        next_token = topk_idx[k].item()
                        new_seq = torch.cat([seq, topk_idx[k:k + 1]])
                        new_score = score + topk_logp[k].item()
                        new_done = (next_token == eos_id)
                        candidates.append((new_seq, new_hidden.squeeze(0), new_score, new_done))

                # length-normalised score to avoid biasing toward short sequences
                candidates.sort(key=lambda c: c[2] / max(len(c[0]), 1), reverse=True)
                beams = candidates[:beam_width]
                if all(done for *_, done in beams):
                    break

            best_seq = beams[0][0][1:]  # drop BOS
            results.append(best_seq)

        max_out_len = max(len(r) for r in results)
        padded = torch.zeros(b, max_out_len, dtype=torch.long, device=device)
        for i, r in enumerate(results):
            padded[i, :len(r)] = r
        return padded


class LMMDecoder(nn.Module):
    """
    LMM head: caption decoder + summary decoder + classification head, all
    conditioned on the shared multimodal state H_0 (Eq. 4.1) plus spatial
    cross-attention over the CNN feature map (proposed update).
    """

    def __init__(self, fusion_dim: int = 512, vocab_size: int = 8000,
                 embed_dim: int = 256, hidden_dim: int = 256,
                 num_classes_binary: int = 2, num_classes_3class: int = 3,
                 num_classes_5class: int = 5, dropout: float = 0.3,
                 pad_id: int = 0, cnn_feature_dim: int = 512,
                 use_spatial_attention: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)

        self.state_proj = nn.Linear(fusion_dim, hidden_dim)  # H_0 = W_z Z_i + b_z (Eq. 4.1)

        self.caption_decoder = _AttentiveAutoregressiveDecoder(
            self.token_embedding, hidden_dim, vocab_size, cnn_feature_dim, dropout,
            use_spatial_attention=use_spatial_attention)
        self.summary_decoder = _AttentiveAutoregressiveDecoder(
            self.token_embedding, hidden_dim, vocab_size, cnn_feature_dim, dropout,
            use_spatial_attention=use_spatial_attention)

        def _head(num_classes: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
            )

        # Multi-granularity classification heads (Eq. 3.29 generalized)
        self.classifier_binary = _head(num_classes_binary)
        self.classifier_3class = _head(num_classes_3class)
        self.classifier_5class = _head(num_classes_5class)

    @staticmethod
    def _flatten_feat_map(feat_map: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, H*W, C)."""
        b, c, h, w = feat_map.shape
        return feat_map.flatten(2).transpose(1, 2)

    @staticmethod
    def _flatten_saliency(saliency_map: torch.Tensor | None,
                           feat_map: torch.Tensor) -> torch.Tensor | None:
        """(B, 1, H', W') -> (B, H*W) resized to feat_map's spatial size,
        scaled to a small additive-energy range so it nudges rather than
        overrides the decoder's own learned attention."""
        if saliency_map is None:
            return None
        h, w = feat_map.shape[-2:]
        resized = F.interpolate(saliency_map, size=(h, w), mode="bilinear", align_corners=False)
        return resized.flatten(1) * 2.0 - 1.0  # map [0,1] saliency to [-1,1] energy bias

    def forward(self, z: torch.Tensor, caption_input: torch.Tensor,
                summary_input: torch.Tensor, feat_map: torch.Tensor,
                saliency_map: torch.Tensor | None = None):
        """
        z:             (B, fusion_dim)  multimodal state Z_i (Eq. 4.10)
        caption_input: (B, Lc) teacher-forcing caption tokens (shifted right)
        summary_input: (B, Ls) teacher-forcing summary tokens (shifted right)
        feat_map:      (B, C, H, W) CNN spatial feature map for attention
        saliency_map:  optional (B, 1, H', W') IAT's shared geometry-
                       consistent saliency map (Phase A redesign); resized
                       to feat_map's H,W and flattened to bias the spatial
                       cross-attention toward the same region IAT perturbs.

        returns dict with:
          caption_logits:        (B, Lc, vocab_size)
          summary_logits:        (B, Ls, vocab_size)
          class_logits_binary:   (B, 2)
          class_logits_3class:   (B, 3)
          class_logits_5class:   (B, 5)
        """
        h0 = self.state_proj(z)  # (B, hidden_dim)  (Eq. 4.1)
        feat_flat = self._flatten_feat_map(feat_map)
        saliency_bias = self._flatten_saliency(saliency_map, feat_map)
        caption_logits, caption_attn = self.caption_decoder(h0, caption_input, feat_flat, saliency_bias=saliency_bias)
        summary_logits, summary_attn = self.summary_decoder(h0, summary_input, feat_flat, saliency_bias=saliency_bias)
        return {
            "caption_logits": caption_logits,
            "summary_logits": summary_logits,
            "caption_attn": caption_attn,    # (B, T, heads, N) or None -- for entropy/coverage regularization
            "summary_attn": summary_attn,
            "class_logits_binary": self.classifier_binary(h0),
            "class_logits_3class": self.classifier_3class(h0),
            "class_logits_5class": self.classifier_5class(h0),
        }

    @torch.no_grad()
    def generate(self, z: torch.Tensor, feat_map: torch.Tensor, bos_id: int, eos_id: int,
                  max_len: int = 32, beam_width: int = 1, unk_id: int | None = 3,
                  saliency_map: torch.Tensor | None = None):
        """Algorithm 3: inference-time caption/summary/prediction generation.

        beam_width=1 uses greedy decoding (default, fastest); beam_width>1
        uses beam search (higher quality, used for evaluation/reporting).
        unk_id: token id of <unk>; its logit is masked to -inf at every
        decode step so <unk> can never be generated (bug fix, 2026-09).
        Pass None to disable (e.g. for before/after comparison).
        """
        h0 = self.state_proj(z)
        feat_flat = self._flatten_feat_map(feat_map)
        saliency_bias = self._flatten_saliency(saliency_map, feat_map)

        if beam_width > 1:
            # beam search does not yet take the saliency bias (Phase A
            # scope: greedy decoding path only for now).
            caption_ids = self.caption_decoder.generate_beam(
                h0, feat_flat, bos_id, eos_id, max_len, beam_width, unk_id=unk_id)
            summary_ids = self.summary_decoder.generate_beam(
                h0, feat_flat, bos_id, eos_id, max_len, beam_width, unk_id=unk_id)
        else:
            caption_ids = self.caption_decoder.generate(h0, feat_flat, bos_id, eos_id, max_len, unk_id=unk_id, saliency_bias=saliency_bias)
            summary_ids = self.summary_decoder.generate(h0, feat_flat, bos_id, eos_id, max_len, unk_id=unk_id, saliency_bias=saliency_bias)

        logits_binary = self.classifier_binary(h0)
        logits_3class = self.classifier_3class(h0)
        logits_5class = self.classifier_5class(h0)
        return {
            "caption_ids": caption_ids,
            "summary_ids": summary_ids,
            "class_logits_binary": logits_binary,
            "class_logits_3class": logits_3class,
            "class_logits_5class": logits_5class,
            "class_pred_binary": logits_binary.argmax(dim=-1),
            "class_pred_3class": logits_3class.argmax(dim=-1),
            "class_pred_5class": logits_5class.argmax(dim=-1),
        }
