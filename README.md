# emalir-dr-rag
Embedding-Aware Listwise Reranking for Diversified Retrieval in RAG.
This is a research-oriented codebase for exploring advanced retrieval pipelines in a document reranking context. It is structured around three key research questions (RQ1–RQ3) and includes tools for clustering, reranking, model training, and evaluation. The project is built in Python with modularity and reproducibility in mind.

---

## 🧠 Research Questions

### RQ1 – Reclustering Qrels and Runs
Evaluate the impact of reclustering relevant documents in existing `.qrels` and `.run` files using embedding-aware similarity measures.

### RQ2 – Training and Fine-Tuning Set-Encoder
Train and analyze listwise rerankers based on the [set-encoder](https://github.com/naver-ai/set-encoder) architecture with support for pretraining, fine-tuning, and multi-run analysis.

### RQ3 – Robustness and Analysis (MMLU)
Apply RAG pipelines to non-standard benchmarks like MMLU, examining the effectiveness of different reranking strategies under domain shift or corruption.

---

## 📁 Project Structure

```text
.
├── configs/            # YAML configuration files for each RQ
├── data/               # Raw and processed qrels, runs, embeddings, etc.
├── models/             # Checkpoints of trained models (e.g., set-encoder)
├── notebooks/          # Jupyter notebooks for analysis and visualization
├── scripts/            # CLI entrypoints for running RQ pipelines
├── rq1/, rq2/, rq3/    # Legacy and experiment-specific code
├── src/                # Main Python packages (editable, structured by domain)
│   ├── reclustering/   # RQ1 logic: pipeline, clustering, data loading
│   ├── finetuning/     # RQ2 logic: dataset builders, loaders
│   └── set_encoder/    # Git submodule with official set-encoder code
├── pyproject.toml      # Build configuration (PEP 517)
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository and initialize submodules

```bash
git clone https://github.com/NamaWho/emalir-dr-rag.git
cd emalir-dr-rag
git submodule update --init --recursive
```

### 2. Install the project in editable mode

```bash
pip install -e .
```

---

## 🚀 Example Usages

### Reclustering Qrels and Runs
```bash
python scripts/rq1_convert_{run,qrels}.py \
    --run-path data/raw/reclustering/__colbert-10000-sampled-100__msmarco-passage-train-judged.run  \
    --encoder all-MiniLM-L6-v2 \
    --threshold 0.85
```
### Finetuning set-encoder
```bash
lightning-ir fit --config ./configs/finetune-novelty.yaml
```

### Reranking
```bash
lightning-ir re_rank --config ./configs/finetuning/re-rank.yaml --model.model_name_or_path ./models/set-encoder-novelty/
```