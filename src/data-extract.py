#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json, os
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import faiss
    _FAISS = True
except Exception:
    _FAISS = False

MODES = ["top5", "kmeans5_complexity", "kmeans5_quality", "kmeans5_compqual"]

MODE_FS_KEYS = {
    "top5": "few_shot_ifd_top5",
    "kmeans5_complexity": "few_shot_ifd_kmeans5_complexity",
    "kmeans5_quality": "few_shot_ifd_kmeans5_quality",
    "kmeans5_compqual": "few_shot_ifd_kmeans5_compqual",
}

MODE_SIM_KEYS = {
    "top5": "top5_similar_sims",
    "kmeans5_complexity": "kmeans5-complexity-sims",
    "kmeans5_quality": "kmeans5-quality-sims",
    "kmeans5_compqual": "kmeans5-compqual-sims",
}

MODE_NNID_KEYS = {
    "top5": "top5_similar_ids",
    "kmeans5_complexity": "kmeans5-complexity",
    "kmeans5_quality": "kmeans5-quality",
    "kmeans5_compqual": "kmeans5-compqual",
}

def parse_ratio_list(s: str) -> List[float]:
    vals = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        r = float(t)
        if not (0.0 < r <= 1.0):
            raise ValueError(f"ratio must be in (0,1], got {r}")
        vals.append(r)
    if not vals:
        raise ValueError("empty --ratios list")
    return vals

def ratio_tag(r: float) -> str:
    return f"r{int(round(r*100)):02d}"

def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0

def build_embeddings(data: List[Dict[str, Any]], sbert_model: str,
                     device: str = "cpu", batch_size: int = 128,
                     text_fields: Optional[List[str]] = None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    if text_fields is None:
        text_fields = ["instruction", "input", "output"]
    def assemble(r):
        return "\n\n".join(str(r.get(f, "") or "") for f in text_fields)
    texts = [assemble(r) for r in data]
    model = SentenceTransformer(sbert_model, device=device)
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True).astype(np.float32, copy=False)
    return emb  # (N, D) L2-normalized

class NNIndex:
    def __init__(self, dim: int, use_gpu: bool):
        self.dim = dim
        self.faiss = _FAISS
        self.use_gpu = use_gpu and _FAISS
        self.index = None
        self.buf = None
        if self.faiss:
            self.index = faiss.IndexFlatIP(dim)
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        else:
            self.buf = np.zeros((0, dim), dtype=np.float32)

    def add(self, vecs: np.ndarray):
        if vecs.ndim == 1: vecs = vecs[None, :]
        if self.faiss:
            self.index.add(vecs.astype(np.float32, copy=False))
        else:
            self.buf = np.vstack([self.buf, vecs.astype(np.float32, copy=False)])

    def size(self) -> int:
        if self.faiss:
            return int(self.index.ntotal)
        return int(self.buf.shape[0])

    def max_sim(self, v: np.ndarray) -> float:
        if self.size() == 0:
            return float("-inf")
        if self.faiss:
            sims, _ = self.index.search(v.astype(np.float32, copy=False).reshape(1, -1), 1)
            return float(sims[0, 0])
        else:
            return float(np.max(self.buf @ v.astype(np.float32, copy=False)))

# ---------- Scoring (key change: Δ = IFD(neighbor) − few-shot IFD) ----------
def compute_mode_score_from_saved(
    rec_idx: int,
    data: List[Dict[str, Any]],
    mode: str,
    id2idx: Dict[Any, int]
) -> float:

    rec = data[rec_idx]

    fs = rec.get(MODE_FS_KEYS[mode], []) or []
    sims = rec.get(MODE_SIM_KEYS[mode], []) or []
    nn_ids = rec.get(MODE_NNID_KEYS[mode], []) or []

    m = min(len(fs), len(sims), len(nn_ids))
    if m <= 0:
        return 0.0

    terms: List[float] = []
    for k in range(m):
        nbr_id = nn_ids[k]
        if nbr_id not in id2idx:
            continue
        nbr = data[id2idx[nbr_id]]
        nbr_ifd = float(nbr.get("ifd", 0.0))
        delta = nbr_ifd - float(fs[k])
        if delta > 0:
            sim = float(sims[k])
            terms.append(delta * (1.0 - sim))
    return safe_mean(terms)

def greedy_diverse_select(cands: List[int], scores: np.ndarray, emb: np.ndarray,
                          k: int, sim_thresh: float, use_gpu: bool) -> List[int]:
    ranked = sorted(cands, key=lambda i: (scores[i], -i), reverse=True)
    dim = emb.shape[1]
    nn = NNIndex(dim, use_gpu)
    selected = []
    for idx in ranked:
        if len(selected) >= k: break
        v = emb[idx]
        if nn.max_sim(v) >= sim_thresh:
            continue
        selected.append(idx)
        nn.add(v)
    return selected

# ---------- ratio ----------
def run_mode_for_ratios(data: List[Dict[str, Any]],
                        emb: np.ndarray,
                        mode: str,
                        ratios: List[float],
                        sim_threshold: float,
                        out_dir: str,
                        use_gpu: bool):
    n = len(data)

    # id -> idx
    id2idx = { (rec.get("id", i)): i for i, rec in enumerate(data) }

    scores = np.zeros(n, dtype=np.float32)
    valid = np.ones(n, dtype=bool)
    for i, rec in enumerate(data):
        if float(rec.get("ifd", 0.0)) > 1.0:
            valid[i] = False
            scores[i] = -1e9
        else:
            scores[i] = compute_mode_score_from_saved(i, data, mode, id2idx)

    cands = [i for i, ok in enumerate(valid) if ok]

    max_k = max(1, int(n * max(ratios)))
    all_selected_idx = greedy_diverse_select(cands, scores, emb, max_k, sim_threshold, use_gpu)

    os.makedirs(out_dir, exist_ok=True)
    for r in ratios:
        k = max(1, int(n * r))
        sel_idx = all_selected_idx[:k]
        selected = [data[i] for i in sel_idx]
        tag = ratio_tag(r)
        out_file = os.path.join(out_dir, f"selected_{mode}_{tag}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(selected, f, ensure_ascii=False, indent=2)
        print(f"[{mode} {tag}] kept {len(selected)}/{n}; wrote: {out_file}")

def parse_args():
    ap = argparse.ArgumentParser(
        description="Score uses saved sims; Δ = IFD(neighbor) - few-shot IFD. "
                    "Exclude IFD>1. Greedy diversity with new embeddings (FAISS NN). "
                    "Support multiple ratios, default 0.05,0.10,0.15"
    )
    ap.add_argument("--input_path", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True,
                    help="Output directory (since there may be multiple ratios and/or modes)")
    ap.add_argument("--mode", type=str, default="all", choices=MODES + ["all"])
    ap.add_argument("--ratios", type=str, default="0.05,0.10,0.15",
                    help="Comma-separated ratios, e.g., 0.05,0.10,0.15")
    ap.add_argument("--sim_threshold", type=float, default=0.9)
    ap.add_argument("--emb_path", type=str, default=None,
                    help="Optional: precomputed (N,D) .npy; otherwise encode on the fly")
    ap.add_argument("--sbert_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", type=str, default="cpu")  # or cuda
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--text_fields", type=str, default="instruction|input|output")
    ap.add_argument("--use_faiss_gpu", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    ratios = parse_ratio_list(args.ratios)

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")

    if args.emb_path and os.path.exists(args.emb_path):
        emb = np.load(args.emb_path).astype(np.float32, copy=False)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms
    else:
        fields = args.text_fields.split("|")
        emb = build_embeddings(data, args.sbert_model, args.device, args.batch_size, fields)

    out_dir = args.output_path
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "all":
        for m in MODES:
            run_mode_for_ratios(list(data), emb, m, ratios, args.sim_threshold, out_dir, args.use_faiss_gpu)
    else:
        run_mode_for_ratios(list(data), emb, args.mode, ratios, args.sim_threshold, out_dir, args.use_faiss_gpu)

    print("Done.")

if __name__ == "__main__":
    main()
