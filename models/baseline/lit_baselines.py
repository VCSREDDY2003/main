"""
Five literature-baseline captioning architectures, reimplemented from their
published descriptions to run on this project's data/pipeline for a
same-config comparison against the proposed IAT-GAM-LSTM model. These are
faithful-effort reimplementations of each paper's *core distinguishing idea*
(not the original authors' code), built by extending the existing
TransformerCaptioner scaffold (models/baseline/transformer_captioner.py) so
training/eval/decoding stay directly comparable and reuse tested code.

  ViT-GPT     : true ViT patch-embedding encoder (no CNN) + causal
                Transformer decoder w/ cross-attention (Tsaniya et al.)
  SPAN        : CNN encoder + an extra self-attention "scene-graph" layer
                over visual tokens before decoding, modelling region-to-
                region relationships (Cong et al.)
  U2DP        : CNN+Transformer base + a semi-supervised consistency loss
                between two augmented views of the same image, the paper's
                core "unlabeled data potential" idea (Zhong et al.)
  Tiny TR-CAP : lightweight CNN (LightCNN) + a smaller decoder (fewer
                layers/heads), efficiency over accuracy (Memis & Yildiz)
  ClipCap++   : CNN encoder -> pooled global feature -> K-token visual
                prefix, decoder-only (no cross-attention; visual tokens are
                prepended to the text sequence), prefix-tuning style (Wang
                et al.)

All five reuse MultiDepthCNN/LightCNN (models/cnn/cnn.py) and
PositionalEncoding (models/text/text_encoder.py) already used by the
proposed model and the SOTA Transformer baseline, so parameter-count /
compute differences reflect real architectural choices, not incidental
reimplementation differences.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.cnn.cnn import LightCNN, MultiDepthCNN
from models.text.text_encoder import PositionalEncoding


# ---------------------------------------------------------------------------
# Shared decode-time helpers (identical to TransformerCaptioner's, factored
# out so every architecture below gets the same generate/generate_beam
# behaviour -- same <unk> masking bugfix, same beam-search loop).
# ---------------------------------------------------------------------------
class _DecodeMixin:
    @torch.no_grad()
    def generate(self, images, bos_id, eos_id, max_len=None, unk_id=3):
        max_len = max_len or self.max_caption_len
        b = images.size(0)
        device = images.device
        memory = self.encode_image(images)

        tokens = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_len):
            logits = self._decode_step(tokens, memory)
            if unk_id is not None:
                logits = logits.clone()
                logits[:, unk_id] = float("-inf")
            next_token = logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break
        return tokens[:, 1:]

    @torch.no_grad()
    def generate_beam(self, images, bos_id, eos_id, max_len=None, beam_width=5, unk_id=3):
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
                    logits = self._decode_step(seq.unsqueeze(0), mem_i).squeeze(0)
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
            results.append(beams[0][0][1:])

        max_out_len = max(len(r) for r in results)
        padded = torch.zeros(b, max_out_len, dtype=torch.long, device=device)
        for i, r in enumerate(results):
            padded[i, :len(r)] = r
        return padded

    def caption_loss(self, logits, targets):
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=self.pad_id,
        )

    def classify_logits(self, images):
        """5-class modality classification from the same visual encoder used
        for captioning (mean-pooled over the token/prefix sequence)."""
        tokens = self.encode_image(images)
        pooled = tokens.mean(dim=1)
        return self.cls_head(pooled)

    def classification_loss(self, logits, targets, weight=None):
        return nn.functional.cross_entropy(logits, targets, weight=weight)


def _causal_mask(L, device):
    return nn.Transformer.generate_square_subsequent_mask(L).to(device)


# ---------------------------------------------------------------------------
# 1. ViT-GPT: true patch-embedding ViT encoder (no CNN) + Transformer decoder
# ---------------------------------------------------------------------------
class ViTGPTCaptioner(_DecodeMixin, nn.Module):
    def __init__(self, vocab_size, embed_dim=256, patch_size=16, img_size=224,
                 vit_layers=6, vit_heads=8, decoder_layers=4, decoder_heads=8,
                 max_caption_len=32, dropout=0.3, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.max_caption_len = max_caption_len

        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        vit_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=vit_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.vit_encoder = nn.TransformerEncoder(vit_layer, num_layers=vit_layers)

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.text_pos = PositionalEncoding(embed_dim, max_len=max_caption_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=decoder_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.out_proj = nn.Linear(embed_dim, vocab_size)
        # 5-class modality classification head (X-ray/CT/MRI/Ultrasound/Other),
        # added for a fair comparison against the proposed model's
        # classification metric -- mean-pools the same visual tokens used
        # for captioning (see _DecodeMixin.classify_logits).
        self.cls_head = nn.Linear(embed_dim, 5)

    def encode_image(self, images):
        x = self.patch_embed(images)                      # (B, D, H', W')
        b = x.size(0)
        x = x.flatten(2).transpose(1, 2)                   # (B, N, D)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[:, : x.size(1) + 1]
        return self.vit_encoder(x)

    def _decode_step(self, tokens, memory):
        tgt = self.text_pos(self.token_embed(tokens))
        hidden = self.decoder(tgt, memory, tgt_mask=_causal_mask(tgt.size(1), tgt.device))
        return self.out_proj(hidden[:, -1, :])

    def forward(self, images, caption_input):
        memory = self.encode_image(images)
        tgt = self.text_pos(self.token_embed(caption_input))
        tgt_key_padding_mask = caption_input == self.pad_id
        hidden = self.decoder(tgt, memory, tgt_mask=_causal_mask(tgt.size(1), tgt.device),
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return self.out_proj(hidden)


# ---------------------------------------------------------------------------
# 2. SPAN: CNN + extra self-attention "scene-graph" reasoning layer
# ---------------------------------------------------------------------------
class SPANCaptioner(_DecodeMixin, nn.Module):
    def __init__(self, vocab_size, embed_dim=256, cnn_depth=39, cnn_feature_dim=512,
                 graph_layers=2, graph_heads=8, decoder_layers=4, decoder_heads=8,
                 max_caption_len=32, dropout=0.3, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.max_caption_len = max_caption_len

        self.cnn = MultiDepthCNN(depth=cnn_depth, feature_dim=cnn_feature_dim)
        self.visual_proj = nn.Linear(cnn_feature_dim, embed_dim)
        self.visual_pos = PositionalEncoding(embed_dim, max_len=256)

        # "scene graph" reasoning: self-attention among visual region tokens,
        # modelling pairwise region-to-region relationships before decoding.
        graph_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=graph_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.graph_reasoning = nn.TransformerEncoder(graph_layer, num_layers=graph_layers)

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.text_pos = PositionalEncoding(embed_dim, max_len=max_caption_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=decoder_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.out_proj = nn.Linear(embed_dim, vocab_size)
        # 5-class modality classification head (X-ray/CT/MRI/Ultrasound/Other),
        # added for a fair comparison against the proposed model's
        # classification metric -- mean-pools the same visual tokens used
        # for captioning (see _DecodeMixin.classify_logits).
        self.cls_head = nn.Linear(embed_dim, 5)

    def encode_image(self, images):
        _, feat_map = self.cnn(images, return_map=True)
        tokens = feat_map.flatten(2).transpose(1, 2)
        tokens = self.visual_proj(tokens)
        tokens = self.visual_pos(tokens)
        return self.graph_reasoning(tokens)                # region-relation reasoning

    def _decode_step(self, tokens, memory):
        tgt = self.text_pos(self.token_embed(tokens))
        hidden = self.decoder(tgt, memory, tgt_mask=_causal_mask(tgt.size(1), tgt.device))
        return self.out_proj(hidden[:, -1, :])

    def forward(self, images, caption_input):
        memory = self.encode_image(images)
        tgt = self.text_pos(self.token_embed(caption_input))
        tgt_key_padding_mask = caption_input == self.pad_id
        hidden = self.decoder(tgt, memory, tgt_mask=_causal_mask(tgt.size(1), tgt.device),
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return self.out_proj(hidden)


# ---------------------------------------------------------------------------
# 3. U2DP: base Transformer captioner + semi-supervised consistency loss
#    between two augmented views of the same image (the paper's "unlabeled
#    data potential" idea, approximated since we don't have a separate
#    unlabeled pool here -- every training image is used both "labeled" and
#    as its own consistency-regularization pair).
# ---------------------------------------------------------------------------
class U2DPCaptioner(_DecodeMixin, nn.Module):
    def __init__(self, vocab_size, embed_dim=256, cnn_depth=39, cnn_feature_dim=512,
                 decoder_layers=4, decoder_heads=8, max_caption_len=32, dropout=0.3, pad_id=0):
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
        # 5-class modality classification head (X-ray/CT/MRI/Ultrasound/Other),
        # added for a fair comparison against the proposed model's
        # classification metric -- mean-pools the same visual tokens used
        # for captioning (see _DecodeMixin.classify_logits).
        self.cls_head = nn.Linear(embed_dim, 5)

    def encode_image(self, images):
        _, feat_map = self.cnn(images, return_map=True)
        tokens = feat_map.flatten(2).transpose(1, 2)
        tokens = self.visual_proj(tokens)
        return self.visual_pos(tokens)

    def _decode_step(self, tokens, memory):
        tgt = self.text_pos(self.token_embed(tokens))
        hidden = self.decoder(tgt, memory, tgt_mask=_causal_mask(tgt.size(1), tgt.device))
        return self.out_proj(hidden[:, -1, :])

    def forward(self, images, caption_input):
        memory = self.encode_image(images)
        tgt = self.text_pos(self.token_embed(caption_input))
        tgt_key_padding_mask = caption_input == self.pad_id
        hidden = self.decoder(tgt, memory, tgt_mask=_causal_mask(tgt.size(1), tgt.device),
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return self.out_proj(hidden)

    def consistency_loss(self, images_view1, images_view2):
        """Feature-space consistency between two augmented views of the same
        image (mean-pooled visual tokens) -- U2DP's core semi-supervised
        regularizer, approximated with a simple MSE consistency term."""
        m1 = self.encode_image(images_view1).mean(dim=1)
        m2 = self.encode_image(images_view2).mean(dim=1)
        return nn.functional.mse_loss(m1, m2)


# ---------------------------------------------------------------------------
# 4. Tiny TR-CAP: lightweight CNN + small decoder, efficiency-focused
# ---------------------------------------------------------------------------
class TinyTRCAPCaptioner(_DecodeMixin, nn.Module):
    def __init__(self, vocab_size, embed_dim=128, cnn_feature_dim=256,
                 decoder_layers=2, decoder_heads=4, max_caption_len=32,
                 dropout=0.2, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.max_caption_len = max_caption_len

        self.cnn = LightCNN(feature_dim=cnn_feature_dim)
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
        # 5-class modality classification head (X-ray/CT/MRI/Ultrasound/Other),
        # added for a fair comparison against the proposed model's
        # classification metric -- mean-pools the same visual tokens used
        # for captioning (see _DecodeMixin.classify_logits).
        self.cls_head = nn.Linear(embed_dim, 5)

    def encode_image(self, images):
        _, feature_map = self.cnn(images, return_map=True)
        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = self.visual_proj(tokens)
        return self.visual_pos(tokens)

    def _decode_step(self, tokens, memory):
        tgt = self.text_pos(self.token_embed(tokens))
        hidden = self.decoder(tgt, memory, tgt_mask=_causal_mask(tgt.size(1), tgt.device))
        return self.out_proj(hidden[:, -1, :])

    def forward(self, images, caption_input):
        memory = self.encode_image(images)
        tgt = self.text_pos(self.token_embed(caption_input))
        tgt_key_padding_mask = caption_input == self.pad_id
        hidden = self.decoder(tgt, memory, tgt_mask=_causal_mask(tgt.size(1), tgt.device),
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return self.out_proj(hidden)


# ---------------------------------------------------------------------------
# 5. ClipCap++: pooled global visual feature -> K-token learned prefix,
#    decoder-only (prefix-tuning style, no cross-attention memory).
# ---------------------------------------------------------------------------
class ClipCapPPCaptioner(_DecodeMixin, nn.Module):
    def __init__(self, vocab_size, embed_dim=256, cnn_depth=39, cnn_feature_dim=512,
                 prefix_len=10, decoder_layers=4, decoder_heads=8,
                 max_caption_len=32, dropout=0.3, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.max_caption_len = max_caption_len
        self.prefix_len = prefix_len

        self.cnn = MultiDepthCNN(depth=cnn_depth, feature_dim=cnn_feature_dim)
        # global-pooled feature -> K-token learned prefix (CLIP-prefix style)
        self.prefix_proj = nn.Linear(cnn_feature_dim, embed_dim * prefix_len)

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.text_pos = PositionalEncoding(embed_dim, max_len=max_caption_len + prefix_len)
        # decoder-only: prefix + text tokens form one causal sequence, no
        # separate cross-attention memory (mirrors ClipCap's GPT-2 prefix-
        # tuning recipe more closely than a cross-attention decoder would).
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=decoder_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)
        self.out_proj = nn.Linear(embed_dim, vocab_size)
        # 5-class modality classification head (X-ray/CT/MRI/Ultrasound/Other),
        # added for a fair comparison against the proposed model's
        # classification metric -- mean-pools the same visual tokens used
        # for captioning (see _DecodeMixin.classify_logits).
        self.cls_head = nn.Linear(embed_dim, 5)

    def encode_image(self, images):
        feat, _ = self.cnn(images, return_map=True)  # (B, cnn_feature_dim) pooled
        b = feat.size(0)
        prefix = self.prefix_proj(feat).view(b, self.prefix_len, -1)
        return prefix

    def _decode_step(self, tokens, memory):
        tgt = self.text_pos(self.token_embed(tokens))
        seq = torch.cat([memory, tgt], dim=1)
        hidden = self.decoder(seq, mask=_causal_mask(seq.size(1), seq.device))
        return self.out_proj(hidden[:, -1, :])

    def forward(self, images, caption_input):
        prefix = self.encode_image(images)                       # (B, K, D)
        tgt = self.text_pos(self.token_embed(caption_input))      # (B, L, D)
        seq = torch.cat([prefix, tgt], dim=1)                     # (B, K+L, D)
        hidden = self.decoder(seq, mask=_causal_mask(seq.size(1), seq.device))
        # only the text-token positions predict the next caption token
        return self.out_proj(hidden[:, self.prefix_len:, :])


ARCHS = {
    "vitgpt": ViTGPTCaptioner,
    "span": SPANCaptioner,
    "u2dp": U2DPCaptioner,
    "tinytrcap": TinyTRCAPCaptioner,
    "clipcappp": ClipCapPPCaptioner,
}
