import random
from pathlib import Path
from typing import Literal
import pandas as pd
import torch
from ir_datasets.util import GzipExtract
from lightning import Callback
from lightning_ir.data.dataset import RankSample, RunDataset
from lightning_ir.data.external_datasets import register_new_dataset

def register_rank_distillm_novelty_embedding_aware():
    run_path = "data" / "train" / "ea__colbert-10000-sampled-100__msmarco-passage-train-judged.run"

    dlc_contents = {
        "cache_path": str(run_path),
        "extractors": [],
        "instructions": f"File locale già disponibile in: {run_path}"
    }

    register_new_dataset(
        "msmarco-passage/train/rank-distillm-novelty-embedding-aware",
        docs="msmarco-passage",
        queries="msmarco-passage/train",
        qrels="msmarco-passage/train",
        scoreddocs=dlc_contents,
    )

    register_new_dataset(
        "msmarco-passage/trec-dl-2019/judged/novelty-ea",
        docs="msmarco-passage",
        queries="msmarco-passage/trec-dl-2019/judged",
        qrels=Path(__file__).parent.parent / "data" / "qrels" / "novelty-ea" / "ea-msmarco-passage-trec-dl-2019-judged.qrels",
    )

    register_new_dataset(
        "msmarco-passage/trec-dl-2020/judged/novelty-ea",
        docs="msmarco-passage",
        queries="msmarco-passage/trec-dl-2020/judged",
        qrels=Path(__file__).parent.parent / "data" / "qrels" / "novelty-ea" / "ea-msmarco-passage-trec-dl-2020-judged.qrels",
    )

class RegisterRankDistiLLMNoveltyEmbeddingAware(Callback):
    def __init__(self) -> None:
        super().__init__()
        register_rank_distillm_novelty_embedding_aware()

