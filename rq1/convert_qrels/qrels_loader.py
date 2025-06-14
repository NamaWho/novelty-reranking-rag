import csv
from typing import List, Tuple

def load_qrels(path: str) -> List[Tuple[str, str, str, int]]:
    """
    Reads a qrels file with columns: qid, cluster_id, docid, relevance
    Returns a list of tuples.
    """
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if not row or len(row) < 4:
                continue
            qid, cluster_id, docid, rel = row[:4]
            records.append((qid, cluster_id, docid, int(rel)))
    return records
