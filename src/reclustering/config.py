from pydantic_settings import BaseSettings

class QrelsConfig(BaseSettings):
    qrels_path: str
    ms_split: str = "train"
    encoder_name: str = "all-MiniLM-L6-v2"
    threshold: float = 0.85
    output_path: str

    class QrelsConfig:
        env_prefix = "RQ1_"

class RunConfig(BaseSettings):
    run_path: str
    ms_split: str = "train"
    encoder_name: str = "all-MiniLM-L6-v2"
    threshold: float = 0.85
    output_path: str
    tag: str = "reclustered"

    class Config:
        env_prefix = "RUN_"