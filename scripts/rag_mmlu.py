import os
import pyterrier as pt
import pyterrier_rag as ptr
from datasets import load_dataset
import ir_datasets
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

backend = ptr.OpenAIBackend(           # Backend OpenAI
    model_id="llama-3-8b-instruct",  # Modello da utilizzare
    base_url="http://api.llm.apps.os.dcs.gla.ac.uk/v1", 
    api_key=os.environ['IDA_LLM_API_KEY']
)
ptr.default_backend.set(backend)      # Imposta global default

# ----------------------------------------------------------------------------
# 2. Caricamento dataset MMLU
# ----------------------------------------------------------------------------
print("🔍 Carico MMLU da Hugging Face...")
mmlu = load_dataset("cais/mmlu", "all", split="test")  # HF loader :contentReference[oaicite:7]{index=7}
# Rinomina e prepara DataFrame
df_mmlu = pd.DataFrame(mmlu) \
    .rename(columns={"question": "query", "answer": "gold_answer"}) \
    .assign(qid=lambda df: df.index.astype(str))

print("🔍 Carico ranking esistente...")
df_run = pd.read_csv(
    "data/processed/rag/__setencoder-novelty-base__msmarco-segment-mmlu.tsv", sep="\t",
    names=["qid", "Q0", "doc_id", "rank", "score", "run_name", "text"]
)

print(df_run.head())