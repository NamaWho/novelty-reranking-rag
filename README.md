# Beyond Redundancy: Embedding-Aware Novelty Reranking in Retrieval-Augmented Generation

This repository contains the research codebase accompanying the Master’s thesis **“Beyond Redundancy: Embedding-Aware Novelty Reranking in Retrieval-Augmented Generation”** at the University of Pisa, Department of Information Engineering.
The project investigates **novelty-aware reranking** as a way to enrich Retrieval-Augmented Generation (RAG) pipelines with **diverse and non-redundant evidence**. The work focuses on how **embedding-based supervision** can improve nugget coverage compared to traditional lexical heuristics.

---

## 🧠 Research Focus

Large Language Models (LLMs) often suffer from **redundant retrieval**, where multiple passages repeat the same fact in different forms. This redundancy wastes the limited context window and reduces the factual diversity of generated answers.

This research addresses the problem through **three research questions (RQ1–RQ3):**

- **RQ1 — Semantic Novelty Detection**  
  Does clustering over dense embeddings produce more coherent novelty groups than lexical (Jaccard) baselines?

- **RQ2 — Embedding-Aware Supervision**  
  Does fine-tuning the **Set-Encoder** reranker with semantic novelty labels improve diversity-sensitive metrics compared to lexical supervision?

- **RQ3 — Downstream RAG Evaluation**  
  Does integrating novelty-aware reranking into RAG pipelines improve factual coverage and answer quality on benchmarks such as **MMLU, GPQA, and the TREC RAG Track 2024**?

---

## 🔬 Methodology

The experimental pipeline is structured in **three stages**:

1. **Semantic Clustering (RQ1)**  
   - Encode top-k passages into dense embeddings  
   - Apply agglomerative clustering with cosine similarity  
   - Derive novelty labels that consolidate paraphrases into coherent groups  

2. **Fine-Tuning the Set-Encoder (RQ2)**  
   - **Stage 1:** Duplicate-Aware InfoNCE (contrastive pretraining)  
   - **Stage 2:** Novelty-Aware RankNet (listwise reranking with novelty penalties)  
   - Compare lexical vs. semantic novelty supervision  

3. **Integration into RAG Pipelines (RQ3)**  
   - First-stage retrieval (BM25 / official TREC pools)  
   - Reranking with Set-Encoder (lexical vs. semantic supervision)  
   - Generation with **LLaMA-3.1-70B-Instruct**  
   - Evaluation with **MMLU, GPQA, and nugget-based metrics**  

---

## 📊 Datasets & Benchmarks

- **MS MARCO v2** → training  
- **MS MARCO v2.1 segmented** → robustness testing  
- **TREC Deep Learning 2019/2020** → high-quality graded relevance  
- **GPQA** → multi-hop reasoning  
- **MMLU** → multi-domain multiple-choice  
- **TREC RAG Track 2024** → nugget-based evaluation (AutoNuggetizer)

---

## 📈 Key Findings

- **RQ1** → Semantic clustering (e.g., MiniLM, E5, BGE-M3) yields **fewer, more coherent novelty groups** than lexical Jaccard clustering, consolidating paraphrases effectively.  
- **RQ2** → Embedding-aware fine-tuning is **comparable to lexical supervision** on α-nDCG; lexical labels remain strong baselines due to stability and granularity.  
- **RQ3** → In RAG pipelines, novelty-aware reranking produces **directionally higher nugget coverage** (especially on importance-weighted metrics) but no statistically significant gains on accuracy-based tasks (MMLU, GPQA).  

---

## 📜 Citation

If you use this code, please cite:
```tex
@mastersthesis{namaki2025novelty,
title={Beyond Redundancy: Embedding-Aware Novelty Reranking in Retrieval-Augmented Generation},
author={Namaki Ghaneh, Daniel},
school={University of Pisa},
year={2025}
}
```

---

## 📜 License

Released under the MIT License.
