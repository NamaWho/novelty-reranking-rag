import argparse
import csv
import numpy as np
from tqdm import tqdm
from config import Config
from encoder_factory import get_encoder
from qrels_loader import load_qrels
from msmarco import load_ms_marco_passages
from clustering import reassign_clusters

def main():
    parser = argparse.ArgumentParser(
        description="Ricalcola i cluster per documenti MSMARCO usando embeddings e soglia.")
    parser.add_argument("--qrels", type=str, required=True,
                        help="File di qrels (qid cluster_id docid relevance)")
    parser.add_argument("--encoder", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Modello sentence-transformers da usare")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Threshold di similarità (cosine) per il clustering")
    parser.add_argument("--split", type=str, default="train",
                        help="Split MSMARCO da caricare (train, dev, etc.)")
    parser.add_argument("--output", type=str, default="qrels_reassigned.txt",
                        help="File di output con i nuovi cluster_id")
    args = parser.parse_args()

    cfg = Config(encoder_name=args.encoder, threshold=args.threshold)

    # Load qrels
    print(f"Caricamento qrels da: {args.qrels}")
    records = load_qrels(args.qrels)
    from collections import defaultdict
    q_groups = defaultdict(list)
    for qid, old_cl, docid, rel in records:
        q_groups[qid].append((docid, rel))

    print(f"Qrels caricati: {len(records)} record, {len(q_groups)} query uniche")
    # Load MSMARCO passages once
    docs = load_ms_marco_passages(split=args.split)
    print(f"Passaggi MSMARCO caricati: {len(docs)} documenti")

    # Prepare encoder
    encoder = get_encoder(cfg.encoder_name)
    print(f"Encoder caricato: {cfg.encoder_name}")

    # Reassign clusters
    output_rows = []
    for qid, docs_info in q_groups.items():
        docids = [d for d, _ in docs_info]
        # Retrieve texts
        passages = [docs.get(d, "") for d in docids]
        # Compute embeddings
        embs = np.array(encoder.encode(passages, show_progress_bar=False))
        # Reassign cluster labels
        new_labels = reassign_clusters(embs, cfg.threshold)
        # Aggregate rows
        for (docid, rel), new_cl in zip(docs_info, new_labels):
            output_rows.append((qid, str(new_cl), docid, str(rel)))

    # Write output qrels
    with open(args.output, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.writer(fout, delimiter=' ')
        for row in output_rows:
            writer.writerow(row)
    print(f"Nuovo file qrels scritto in: {args.output}")

if __name__ == "__main__":
    main()
