# emalir-dr-rag
Embedding-Aware Listwise Reranking for Diversified Retrieval in RAG 

## Script example
```bash
cd nfs/projects/emalir-dr-rag/rq2/set-encoder/
conda activate emalir-rq2
lightning-ir fit --config ./configs/finetune-novelty.yaml


PYTHONPATH=. lightning-ir re_rank --config ./configs/re-rank.yaml --model.model_name_or_path ./models/set-encoder-novelty/