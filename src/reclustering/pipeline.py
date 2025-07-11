from .data_loader import load_qrels, load_passages, load_run
from .embeddings import compute_embeddings
from .clustering import reassign_clusters
import pandas as pd
import ir_datasets
from .config import QrelsConfig, RunConfig
from tqdm import tqdm

class QrelsReclusterPipeline:
    def __init__(self, cfg: QrelsConfig):
        self.cfg = cfg

    def run(self) -> pd.DataFrame:
        # 1. Carica qrels
        records = load_qrels(self.cfg.qrels_path)
        # raggruppa per query
        from collections import defaultdict
        groups = defaultdict(list)
        for qid, old_cl, docid, rel in records:
            groups[qid].append((docid, rel))

        # 2. Carica testi MSMARCO
        docs = load_passages(self.cfg.ms_split)

        # 3. Per ogni query: estrai testi, calcola embedding, riclusterizza
        rows = []
        for qid, infos in groups.items():
            docids, rels = zip(*infos)
            texts = [docs.get(d,"") for d in docids]
            embs = compute_embeddings(texts, self.cfg.encoder_name)
            new_labels = reassign_clusters(embs, self.cfg.threshold)
            for docid, rel, new_cl in zip(docids, rels, new_labels):
                rows.append((qid, new_cl, docid, rel))

        # 4. Ritorna DataFrame per salvarlo o analizzarlo
        df = pd.DataFrame(rows, columns=["qid","cluster","docid","rel"])
        return df

    def save(self, df):
        df.to_csv(self.cfg.output_path, sep=" ", index=False, header=False)

class RunReclusterPipeline:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    def run(self) -> pd.DataFrame:
        # 1. Carica la run strutturata
        run_by_query = load_run(self.cfg.run_path)
        # 2. Carica i testi
        docs = load_passages(self.cfg.ms_split)

        rows = []
        # 3. Per ogni query, riclusterizza semanticamente
        for qid, docs_scores in tqdm(
            run_by_query.items(),
            desc="Riclusterizzo query",
            total=len(run_by_query),
            unit="query",
        ):
            docids, scores, _, ranks, tags = zip(*docs_scores)
            texts = [docs.get(d, "") for d in docids]
            embs = compute_embeddings(texts, self.cfg.encoder_name)
            new_labels = reassign_clusters(embs, self.cfg.threshold)

            # 4. Ricostruisci le righe della run con il nuovo cluster_id
            for docid, score, rank, tag, new_cl in zip(docids, scores, ranks, tags, new_labels):
                rows.append((qid, new_cl, docid, rank, score, tag))

        df = pd.DataFrame(rows, columns=["qid","cluster","docid","rank","score","tag"])
        return df

    def save(self, df: pd.DataFrame):
        # Tab-delimited, senza header
        df.to_csv(self.cfg.output_path, sep="\t", index=False, header=False)