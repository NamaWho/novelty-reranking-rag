import typer
from src.reclustering.config import QrelsConfig
from src.reclustering.pipeline import QrelsReclusterPipeline

def main(
    qrels: str = typer.Option(...),
    ms_split: str = "train",
    encoder: str = "all-MiniLM-L6-v2",
    threshold: float = 0.85,
    output: str = "data/processed/reclustering/reclustered_qrels.run"
):
    cfg = QrelsConfig(
        qrels_path=qrels,
        ms_split=ms_split,
        encoder_name=encoder,
        threshold=threshold,
        output_path=output
    )
    pipeline = QrelsReclusterPipeline(cfg)
    df = pipeline.run()
    pipeline.save(df)
    typer.echo(f"Saved reclustered qrels to {output}")

if __name__=="__main__":
    typer.run(main)
