# emalir-dr-rag
Embedding-Aware Listwise Reranking for Diversified Retrieval in RAG 

## Script example
```bash
cd nfs/projects/emalir-dr-rag/rq2/set-encoder/
conda activate emalir-rq2
lightning-ir fit --config ./configs/finetune-novelty.yaml
