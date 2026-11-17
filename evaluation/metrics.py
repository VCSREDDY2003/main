"""
Evaluation metrics for IAT-GAM-LSTM (Section 5 / Table 5.x):

  - Captioning quality: BLEU-1/4, ROUGE-L, CIDEr-D
  - Classification: Accuracy, Precision, Recall, F1
  - Generative quality (IAT/GAN outputs): SSIM
  - Efficiency: latency (ms/sample), throughput (samples/sec)

Implemented with scikit-learn + scikit-image only. BLEU is a small
self-contained implementation (with add-1 smoothing) rather than nltk's,
since nltk 3.8.1's bleu_score module calls `Fraction(..., _normalize=False)`,
which was removed in Python 3.12's `fractions.Fraction`.
"""

from __future__ import annotations

import math
import time
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)


# ----------------------------------------------------------------------
# Captioning metrics
# ----------------------------------------------------------------------
def bleu_score(hypotheses: list[list[str]], references: list[list[str]], n: int = 4) -> float:
    """BLEU-n with add-1 smoothing. hypotheses/references are lists of token lists."""
    scores = []
    for hyp, ref in zip(hypotheses, references):
        if not hyp or not ref:
            scores.append(0.0)
            continue
        precisions = []
        for k in range(1, n + 1):
            hyp_ngrams = _ngrams(hyp, k)
            ref_ngrams = _ngrams(ref, k)
            overlap = sum(min(c, ref_ngrams.get(g, 0)) for g, c in hyp_ngrams.items())
            total = sum(hyp_ngrams.values())
            precisions.append((overlap + 1) / (total + 1))
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / n)
        bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / max(len(hyp), 1))
        scores.append(bp * geo_mean)
    return float(np.mean(scores)) if scores else 0.0


def _lcs_len(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(hypotheses: list[list[str]], references: list[list[str]], beta: float = 1.2) -> float:
    """ROUGE-L F-measure averaged over the corpus."""
    scores = []
    for hyp, ref in zip(hypotheses, references):
        if not hyp or not ref:
            scores.append(0.0)
            continue
        lcs = _lcs_len(hyp, ref)
        recall = lcs / len(ref)
        precision = lcs / len(hyp)
        if precision + recall == 0:
            scores.append(0.0)
            continue
        f = (1 + beta ** 2) * precision * recall / (recall + beta ** 2 * precision)
        scores.append(f)
    return float(np.mean(scores)) if scores else 0.0


def rouge_1(hypotheses: list[list[str]], references: list[list[str]], beta: float = 1.2) -> float:
    """ROUGE-1 F-measure (unigram overlap) averaged over the corpus."""
    scores = []
    for hyp, ref in zip(hypotheses, references):
        if not hyp or not ref:
            scores.append(0.0)
            continue
        hyp_counts = Counter(hyp)
        ref_counts = Counter(ref)
        overlap = sum((hyp_counts & ref_counts).values())
        recall = overlap / len(ref)
        precision = overlap / len(hyp)
        if precision + recall == 0:
            scores.append(0.0)
            continue
        f = (1 + beta ** 2) * precision * recall / (recall + beta ** 2 * precision)
        scores.append(f)
    return float(np.mean(scores)) if scores else 0.0


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def cider_d(hypotheses: list[list[str]], references: list[list[str]], n_max: int = 4) -> float:
    """Simplified single-reference CIDEr-D (TF-IDF weighted n-gram cosine similarity)."""
    docs = hypotheses + references
    n_docs = len(docs)
    df = {n: Counter() for n in range(1, n_max + 1)}
    for doc in docs:
        for n in range(1, n_max + 1):
            for ng in _ngrams(doc, n):
                df[n][ng] += 1

    def tfidf_vec(doc: list[str], n: int) -> Counter:
        tf = _ngrams(doc, n)
        vec = Counter()
        for ng, count in tf.items():
            idf = math.log(max(n_docs / (1.0 + df[n][ng]), 1e-12))
            vec[ng] = count * idf
        return vec

    def cos_sim(v1: Counter, v2: Counter) -> float:
        common = set(v1) & set(v2)
        num = sum(v1[k] * v2[k] for k in common)
        norm1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return num / (norm1 * norm2)

    scores = []
    for hyp, ref in zip(hypotheses, references):
        n_scores = []
        for n in range(1, n_max + 1):
            n_scores.append(cos_sim(tfidf_vec(hyp, n), tfidf_vec(ref, n)))
        scores.append(10.0 * float(np.mean(n_scores)))
    return float(np.mean(scores)) if scores else 0.0


def meteor_score(hypotheses: list[list[str]], references: list[list[str]],
                  alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5) -> float:
    """Simplified single-reference METEOR (unigram exact-match only, no
    stem/synonym matching): harmonic-mean of unigram precision/recall,
    penalised by fragmentation of matched chunks (Banerjee & Lavie, 2005).

    Self-contained (no nltk dependency) for the same reason bleu_score/
    rouge_l/cider_d above are self-contained: nltk's meteor_score pulls in
    WordNet data files that may not be available offline, and this keeps
    the whole metrics module dependency-light and reproducible.
    """
    scores = []
    for hyp, ref in zip(hypotheses, references):
        if not hyp or not ref:
            scores.append(0.0)
            continue

        ref_counts = Counter(ref)
        matches = 0
        matched_hyp_idx = []
        used = Counter()
        for i, tok in enumerate(hyp):
            if used[tok] < ref_counts.get(tok, 0):
                used[tok] += 1
                matches += 1
                matched_hyp_idx.append(i)

        if matches == 0:
            scores.append(0.0)
            continue

        precision = matches / len(hyp)
        recall = matches / len(ref)
        f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

        # chunks: consecutive runs of matched-token indices in the hypothesis
        chunks = 1
        for a, b in zip(matched_hyp_idx, matched_hyp_idx[1:]):
            if b != a + 1:
                chunks += 1
        fragmentation = chunks / matches
        penalty = gamma * (fragmentation ** beta)
        scores.append(f_mean * (1 - penalty))

    return float(np.mean(scores)) if scores else 0.0


_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at", "to",
    "and", "with", "showing", "shows", "image", "this", "there", "it", "for",
})


def spice_score(hypotheses: list[list[str]], references: list[list[str]]) -> float:
    """Self-contained SPICE-style semantic-propositional F-score.

    NOTE: this is NOT the official SPICE toolkit (which parses a dependency
    scene graph via a Java/Stanford CoreNLP pipeline into (object, attribute,
    relation) tuples and matches them via WordNet synonymy). That pipeline
    has no dependency-light equivalent here. This implementation approximates
    the same idea -- semantic-tuple overlap rather than n-gram overlap -- by
    extracting unigram "object" tuples and adjacent-bigram "relation" tuples
    from each caption (stopwords removed) and computing the F1 of tuple-set
    overlap between hypothesis and reference. Report it as "SPICE (self-
    contained tuple-F1 approximation)", not as an exact reproduction of the
    reference SPICE score.
    """
    def _tuples(tokens: list[str]) -> set[tuple[str, ...]]:
        content = [t for t in tokens if t not in _STOPWORDS]
        tup = {(t,) for t in content}
        tup |= {(content[i], content[i + 1]) for i in range(len(content) - 1)}
        return tup

    scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp_t, ref_t = _tuples(hyp), _tuples(ref)
        if not hyp_t or not ref_t:
            scores.append(0.0)
            continue
        common = hyp_t & ref_t
        precision = len(common) / len(hyp_t)
        recall = len(common) / len(ref_t)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1)
    return float(np.mean(scores)) if scores else 0.0


# ----------------------------------------------------------------------
# Classification metrics
# ----------------------------------------------------------------------
def classification_metrics(preds: np.ndarray, targets: np.ndarray, ignore_index: int = -1,
                             num_classes: int | None = None) -> dict:
    """Accuracy / Precision / Recall / F1 (Eq. 3.29 evaluation), ignoring unlabeled samples.

    If num_classes is given, also returns per-class precision/recall/f1 arrays
    (length num_classes, 0 for classes absent from this split) under
    'precision_per_class' / 'recall_per_class' / 'f1_per_class', used for the
    per-class training curves (Section 5.6/5.7 breakdowns).
    """
    mask = targets != ignore_index
    if not mask.any():
        out = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        if num_classes is not None:
            zeros = [0.0] * num_classes
            out.update({"precision_per_class": zeros, "recall_per_class": zeros, "f1_per_class": zeros})
        return out
    y_true, y_pred = targets[mask], preds[mask]
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if num_classes is not None:
        labels = list(range(num_classes))
        out["precision_per_class"] = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist()
        out["recall_per_class"] = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist()
        out["f1_per_class"] = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist()
    return out


def confusion_matrix_report(preds: np.ndarray, targets: np.ndarray, ignore_index: int = -1,
                             num_classes: int | None = None) -> dict:
    """Confusion matrix + balanced accuracy, ignoring unlabeled samples.

    Added 2026-11 per the standing metrics checklist -- the per-class
    precision/recall/f1 in classification_metrics doesn't show which
    *other* class a misclassified sample lands in, which the raw matrix
    does.
    """
    mask = targets != ignore_index
    if not mask.any():
        n = num_classes or 2
        return {"confusion_matrix": [[0] * n for _ in range(n)], "balanced_accuracy": 0.0}
    y_true, y_pred = targets[mask], preds[mask]
    labels = list(range(num_classes)) if num_classes is not None else None
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "confusion_matrix": cm.tolist(),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def auroc_multiclass(probs: np.ndarray, targets: np.ndarray, ignore_index: int = -1) -> float | None:
    """Macro one-vs-rest AUROC from predicted class PROBABILITIES (not argmax).

    probs: (N, num_classes) softmax output. Returns None if the split has
    fewer than 2 distinct real classes present (AUROC undefined).
    """
    mask = targets != ignore_index
    if not mask.any():
        return None
    y_true, y_prob = targets[mask], probs[mask]
    present = np.unique(y_true)
    if present.size < 2:
        return None
    try:
        if y_prob.shape[1] == 2:
            return float(roc_auc_score(y_true, y_prob[:, 1]))
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=present))
    except ValueError:
        return None


_bert_model_cache = {}


def bert_score_f1(hypotheses: list[str], references: list[str],
                   model_type: str = "distilbert-base-uncased") -> float:
    """Mean BERTScore F1 over a list of (hypothesis, reference) caption pairs.

    Added 2026-11: BLEU/ROUGE/CIDEr are n-gram overlap metrics and penalize
    a caption that's medically correct but phrased differently; BERTScore
    uses contextual embeddings so it credits that case, per ImageCLEFmedical
    reporting convention (standing metrics checklist).

    Implemented directly on top of transformers rather than the `bert_score`
    package: bert_score==0.3.13's internal tokenizer calls
    (sent_encode -> build_inputs_with_special_tokens) are incompatible with
    transformers>=5 (AttributeError on a real full-dataset eval run,
    2026-11), and pinning transformers back down risks breaking the
    training pipeline that depends on the current version. This is the
    standard (unweighted, no IDF, no baseline-rescaling) greedy-matching
    BERTScore: precision = mean over hyp tokens of max cosine sim to any
    ref token; recall = mean over ref tokens of max cosine sim to any hyp
    token; F1 = harmonic mean. Empty strings are replaced with a single
    pad token so they score low rather than crash.
    """
    import torch as _torch
    from transformers import AutoModel, AutoTokenizer

    if model_type not in _bert_model_cache:
        tok = AutoTokenizer.from_pretrained(model_type)
        model = AutoModel.from_pretrained(model_type)
        model.eval()
        _bert_model_cache[model_type] = (tok, model)
    tok, model = _bert_model_cache[model_type]
    device = next(model.parameters()).device

    hyps = [h if h.strip() else "[EMPTY]" for h in hypotheses]
    refs = [r if r.strip() else "[EMPTY]" for r in references]

    def _embed(texts: list[str]):
        enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with _torch.no_grad():
            out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].bool()
        return out, mask

    f1s = []
    batch_size = 32
    for start in range(0, len(hyps), batch_size):
        h_emb, h_mask = _embed(hyps[start:start + batch_size])
        r_emb, r_mask = _embed(refs[start:start + batch_size])
        for i in range(h_emb.size(0)):
            hv = h_emb[i][h_mask[i]]
            rv = r_emb[i][r_mask[i]]
            if hv.size(0) == 0 or rv.size(0) == 0:
                f1s.append(0.0)
                continue
            hv = hv / hv.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            rv = rv / rv.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            sim = hv @ rv.T
            precision = sim.max(dim=1).values.mean().item()
            recall = sim.max(dim=0).values.mean().item()
            f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(f1s))


# ----------------------------------------------------------------------
# Generative quality (IAT augmentation / GAN outputs)
# ----------------------------------------------------------------------
def ssim_batch(x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean SSIM over a batch of images in [0, 1], shape (B, C, H, W)."""
    from skimage.metrics import structural_similarity

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    scores = []
    for xi, yi in zip(x_np, y_np):
        scores.append(structural_similarity(xi, yi, channel_axis=0, data_range=1.0))
    return float(np.mean(scores)) if scores else 0.0


_fid_extractor = None


def _get_fid_extractor(device):
    """Lazily load a pretrained InceptionV3 (ImageNet) for FID pool features.
    Cached at module scope so repeated calls in one process reuse it."""
    global _fid_extractor
    if _fid_extractor is None:
        import torchvision
        weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
        net = torchvision.models.inception_v3(weights=weights, aux_logits=True)
        net.fc = torch.nn.Identity()  # expose the 2048-d pool features
        net.eval()
        _fid_extractor = net.to(device)
    return _fid_extractor


@torch.no_grad()
def _fid_features(images: torch.Tensor, device) -> np.ndarray:
    """images: (B, 3, H, W) in [0, 1] -> (B, 2048) InceptionV3 pool features."""
    net = _get_fid_extractor(device)
    x = torch.nn.functional.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x.to(device) - mean) / std
    feats = net(x)
    return feats.detach().cpu().numpy()


def fid_score(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    """Frechet Inception Distance between two image batches, both (B,3,H,W)
    in [0,1]. Uses a real pretrained InceptionV3 (ImageNet weights) for
    features -- this is a genuine FID computation, used here as the
    substitute for Frechet VIDEO Distance (FVD, which needs a pretrained
    video/I3D feature extractor and real video frames -- neither is
    available in this codebase's MedVideoCap-55K branch, which is
    captions-only; see Limitations). Report as "FID (image-based
    substitute for FVD)", not as FVD itself.
    """
    device = real_images.device
    mu1, sigma1 = _fid_stats(_fid_features(real_images, device))
    mu2, sigma2 = _fid_stats(_fid_features(fake_images, device))
    diff = mu1 - mu2
    # Matrix sqrt of sigma1 @ sigma2 via eigen-decomposition (both are
    # symmetric PSD covariance matrices, so this is numerically stable
    # without needing scipy's linalg.sqrtm).
    covmean = _sqrtm_psd(sigma1 @ sigma2)
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def _fid_stats(feats: np.ndarray):
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    """Matrix square root of a symmetric PSD matrix via eigendecomposition.

    Clips at exactly 0, not a small positive eps: with a high-dimensional
    (2048-d) feature covariance estimated from a much smaller sample count,
    most eigenvalues are analytically 0 (rank <= n_samples-1) and come back
    as small +/- float noise from eigh. Flooring them to an eps > 0 instead
    of 0 spuriously adds sqrt(eps) per clipped dimension -- with ~2000
    near-zero dimensions this summed to a +1.85 trace error and produced
    NEGATIVE "FID" values for identical inputs (caught by a real identical-
    batch sanity check: FID(x, x) must be 0, and initially was not).
    """
    mat = (mat + mat.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


# ----------------------------------------------------------------------
# Efficiency
# ----------------------------------------------------------------------
def measure_latency_throughput(model: torch.nn.Module, forward_fn, n_runs: int = 20) -> dict:
    """forward_fn() executes one forward pass for a fixed batch size B.
    Returns latency (ms/sample) and throughput (samples/sec)."""
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(min(3, n_runs)):
            forward_fn()
        start = time.perf_counter()
        for _ in range(n_runs):
            forward_fn()
        elapsed = time.perf_counter() - start
    return {"total_seconds": elapsed, "runs": n_runs, "seconds_per_run": elapsed / n_runs}
