import random
from pathlib import Path
from typing import Literal
import pandas as pd
import torch
from ir_datasets.util import GzipExtract
from lightning import Callback
from lightning_ir.data.dataset import RankSample, RunDataset
from lightning_ir.data.external_datasets import register_new_dataset

def register_mmlu():
    print(">>> Registering MMLU dataset")

    register_new_dataset(
        "msmarco-segment/mmlu",
        docs="msmarco-segment-v2.1",
        queries="./data/datasets/mmlu/mmlu-queries.tsv",
        qrels=Path(__file__).parent / "data" / "qrels" / "dummy.qrels",
    )
    print(">>> MMLU dataset registered")

class RegisterMmlu(Callback):
    def __init__(self) -> None:
        super().__init__()
        register_mmlu()