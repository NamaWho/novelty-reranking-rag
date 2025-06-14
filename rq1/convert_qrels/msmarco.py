import ir_datasets
from tqdm import tqdm
from typing import Dict

def load_ms_marco_passages(split: str = "train") -> Dict[str, str]:
    """
    Builds an in-memory passage store for MSMARCO using ir_datasets.
    Returns a dict: docid -> text.
    """
    print(f"Loading MSMARCO passages for split: {split}")
    dataset = ir_datasets.load(f"msmarco-passage/{split}")
    docs = {}
    for doc in tqdm(dataset.docs_iter(), desc="Building passage store"):
        docs[doc.doc_id] = doc.text
    return docs
