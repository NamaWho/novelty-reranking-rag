import csv
from typing import Dict, List, Tuple
import ir_datasets
from tqdm import tqdm

def load_qrels(path: str) -> List[Tuple[str,str,str,int]]:
    """
    Reads a qrels file with columns: qid, cluster_id, docid, relevance
    Returns a list of tuples.
    """
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if len(row) < 4: continue
            qid, cl, docid, rel = row[:4]
            records.append((qid, cl, docid, int(rel)))
    return records

def load_passages(split: str = "train") -> Dict[str,str]:
    """
    Builds an in-memory passage store for MSMARCO using ir_datasets.
    Returns a dict: docid -> text.
    """
    dataset = ir_datasets.load(f"msmarco-passage/{split}")
    docs = {}
    for d in tqdm(dataset.docs_iter(), desc="Loading passages"):
        docs[d.doc_id] = d.text
    return docs


def load_run(path: str) -> Dict[str, List[Tuple[str, float, int, int, str]]]:
    """
    Reads a TREC run file with columns:
    qid, cluster_id, docid, rank, score, tag
    Returns a dict: qid -> list of (docid, score, orig_cluster, rank, tag)
    """
    run_by_query = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for qid, orig_cl, docid, rank, score, tag in reader:
            run_by_query.setdefault(qid, []).append(
                (docid, float(score), int(orig_cl), int(rank), tag)
            )
    return run_by_query