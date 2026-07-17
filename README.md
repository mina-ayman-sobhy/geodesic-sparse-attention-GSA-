# Geodesic Sparse Attention (GSA)

**A Theoretical Framework for Efficient Transformers via Riemannian Manifold Structure**

Mina Ayman Sobhy — mina.ayman.sobhy@gmail.com

## Overview

Geodesic Sparse Attention (GSA) exploits the intrinsic Riemannian geometric
structure of token embedding spaces to compute sparse attention patterns
adaptively, reducing attention complexity from *O(n²)* to *O(n log n)* under
mild geometric assumptions. This repository contains the paper and the full
set of experiments used to test its claims — including the ones that did
**not** confirm the theory, reported alongside the ones that did.

| Claim | Status |
|---|---|
| Memory reduction (Proposition 1) | Confirmed, 8×–128× across tested sizes |
| Latency reduction, exact top-k (Theorem 1) | Not confirmed at CPU scale (n ≤ 2048) |
| Latency reduction, GPU clustered variant (Flash-GSA) | Confirmed for n ≥ 8192 |
| Manifold hypothesis (Assumption 1) | Weak positive signal, single-sample |
| Real intrinsic dimension of DistilBERT embeddings | Confirmed low (d_m ≈ 8.3) |

## Repository contents

```
paper/
  MinaAyman_GeodesicSparseAttention.pdf   — the paper
notebooks/
  GSA_all_experiments.ipynb                — every experiment in §7, runnable on Colab
code/                                       — (optional) standalone scripts, if you split them out of the notebook
```

## Running the experiments

Open `notebooks/GSA_all_experiments.ipynb` in Google Colab
(*Runtime → Change runtime type → GPU*, a T4 is enough) and run the setup
cell first. CPU-only sections (memory, latency, intrinsic-dimension
estimation, manifold hypothesis) run on any runtime; the Flash-GSA Triton
kernel and the real-model fine-tuning comparisons need a GPU.

## Citation

If you use this work, please cite:

```bibtex
@misc{sobhy2026gsa,
  title  = {Geodesic Sparse Attention: A Theoretical Framework for Efficient Transformers via Riemannian Manifold Structure},
  author = {Sobhy, Mina Ayman},
  year   = {2026},
  note   = {Preprint}
}
```

## License

See [LICENSE](LICENSE).
