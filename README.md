# SevBERT — Contrastive Learning for Bug Severity Classification with LLM Embeddings

![Task](https://img.shields.io/badge/task-bug%20severity%20classification-blue)
![Model](https://img.shields.io/badge/backbone-CodeBERT-orange)
![Method](https://img.shields.io/badge/method-contrastive%20learning-green)
![Loss](https://img.shields.io/badge/loss-NT--Xent-purple)

---

## Executive Summary

**The business problem.** Every incoming bug report has to be triaged by severity so engineering teams fix the most damaging issues first. Done manually, triage is slow, subjective, and inconsistent — and it gets worse as report volume grows and teams change. Mislabeled severity means critical bugs sit in the backlog while minor ones jump the queue.

**The solution.** SevBERT automates severity triage by fine-tuning a pre-trained **CodeBERT** language model with **contrastive learning**. The model learns to pull same-severity reports together and push different-severity reports apart in embedding space, producing representations that classify accurately *and* transfer to projects the model has never seen.

**The number impact.**
- **Best-in-class on every same-project dataset** — contrastive learning topped all baselines, reaching **F1 ≈ 0.98** on PITS D and **≈ 0.90** on PITS C.
- **~56% relative F1 gain on the hardest cross-project test** — on the highly imbalanced B, C, E → D split, contrastive learning scored **0.59 vs. 0.38** for the non-contrastive model.
- **Robust where it counts** — it beat both the non-contrastive model and the Doc2Vec baseline on the most diverse, imbalanced cross-project cases, the exact scenarios where naive models fail (accuracy < 0.50).

**A few next steps.** Add lightweight backbones (TinyBERT, MobileBERT) to cut compute cost, rebalance training data to counter class imbalance, and validate on more diverse real-world projects before production rollout.

---

## Problem Statement

Bug severity measures how badly a defect affects functionality, performance, or user experience. Teams use it to **prioritize fixes by criticality** — but assigning it by hand doesn't scale.

<p align="center">
  <img src="images/bug-severity-levels.png" alt="Bug severity levels from Blocker down to Trivial" width="480"><br>
  <em>Fig 1 — Common bug severity levels, from Blocker (most critical) to Trivial.</em>
</p>

Existing automated approaches fall short: traditional ML (Naive Bayes, KNN) leans on hand-built keyword dictionaries, and static word embeddings (Word2Vec, GloVe) assign one vector per word regardless of context. Both struggle with the long, unstructured, imbalanced text found in real bug reports — and neither generalizes well across different projects.

---

## Methodology

*(brief)*

SevBERT runs in two stages. First, **contrastive fine-tuning**: reports are cleaned, tokenized, and formed into positive/negative pairs; CodeBERT encodes each pair, and an **NT-Xent contrastive loss** on the cosine similarity of the two embeddings updates the weights via backpropagation. Second, **classification**: a single report is encoded and passed through a dense layer that predicts its severity.

<p align="center">
  <img src="images/approach-overview.png" alt="Overview of the proposed contrastive learning approach" width="760"><br>
  <em>Fig 2 — End-to-end pipeline: pairing → contrastive fine-tuning → classification head.</em>
</p>

| Component            | Detail                                       |
| -------------------- | -------------------------------------------- |
| Backbone             | CodeBERT (RoBERTa-based)                      |
| Input length         | 256 tokens                                   |
| Embedding dimension  | 768                                          |
| Contrastive loss     | NT-Xent, temperature = 0.5                   |
| Classification head  | Fully connected (768 → num classes)          |
| Loss / Optimizer     | Cross-Entropy / Adam (lr = 2e-5)             |

**Datasets:** PITS projects A–E, Combined PITS, and Mozilla (80/20 train–test). **Baseline:** Doc2Vec + MLP, trained separately on summaries and full descriptions.

---

## Skills

`Contrastive Learning (NT-Xent)` · `LLM Fine-Tuning` · `CodeBERT / Transformers` · `NLP & Text Classification` · `PyTorch` · `Transfer Learning` · `Cross-Project Generalization` · `Imbalanced Data Handling` · `Embedding Visualization (t-SNE)` · `Experimental Design & Benchmarking`

---

## Results & Business Recommendations

**Same-project.** Contrastive learning won on **every** dataset, with the largest margins on the richer, more complex sets.

<p align="center">
  <img src="images/results-same-project.png" alt="F1 score comparison across same-project datasets" width="600"><br>
  <em>Fig 3 — F1 by model, same-project. Contrastive learning (blue) leads across all datasets.</em>
</p>

**Cross-project (unseen).** Contrastive learning generalized best in most cases and was clearly the most robust on the hardest, most imbalanced split (B, C, E → D). Doc2Vec only won on the small, well-structured A, B → C set (F1 = 0.8249).

<p align="center">
  <img src="images/results-cross-project.png" alt="F1 score comparison on unseen projects" width="600"><br>
  <em>Fig 4 — F1 on unseen projects. Contrastive learning holds up where non-contrastive collapses.</em>
</p>

**Why it works.** With contrastive learning the embeddings form cleaner, better-separated severity clusters.

<p align="center">
  <img src="images/tsne-without-contrastive.png" alt="t-SNE without contrastive learning" width="380">
  <img src="images/tsne-with-contrastive.png" alt="t-SNE with contrastive learning" width="380"><br>
  <em>Figs 5 & 6 — t-SNE of embeddings: without contrastive learning (left) vs. with it (right).</em>
</p>

**Business recommendations.**
- **Deploy contrastive CodeBERT as the default triage model** for large, diverse, or unstructured bug backlogs — it is the most accurate and most robust option.
- **Keep a lightweight Doc2Vec model for small or highly structured projects**, where it matches LLMs at a fraction of the compute cost.
- **Prioritize cross-project robustness** when choosing a model for new or fast-changing products — that is where contrastive learning's advantage is largest.
- **Budget for class rebalancing**, since imbalance is the main remaining drag on accuracy.

---

## Next Steps

- Evaluate **lightweight models (TinyBERT, MobileBERT)** for cost-effective, possibly on-device classification.
- **Expand testing to more diverse datasets** to confirm scalability and robustness.
- **Address class imbalance** through resampling or balanced training sets.
- Reduce **pre-trained model dependency** by exploring domain-adapted embeddings that capture bug-report-specific nuances.

---

## Authors

**Mosarrat Rumman · Anushka Zaman · Emon Roy** — *CSCI 6100G Final Project*

## References

1. Baciu, I. (2024). *Navigating bug severity and priority in software testing.* Infobest.
2. Sharma, G., Sharma, S., & Gujral, S. (2015). *A novel way of assessing software bug severity using dictionary of critical terms.* Procedia Computer Science.
3. Kim, J., & Yang, G. *Bug Severity Prediction Using Topic-Based Feature Selection and CNN-LSTM.*
4. Chen, B., et al. (2024). *An empirical study on the potential of word embedding techniques in bug report management tasks.* Empirical Software Engineering.
5. Arokiam, J., & Bradbury, J. S. (2020). *Automatically predicting bug severity early in the development process.* ICSE NIER, 17–20.
