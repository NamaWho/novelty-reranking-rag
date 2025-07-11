# bin/rq1_convert_run.py
import typer
from src.reclustering.config import RunConfig
from src.reclustering.pipeline import RunReclusterPipeline

def main(
    run: str = typer.Option(..., help="Path to input run file"),
    ms_split: str = "train",
    encoder: str = "all-MiniLM-L6-v2",
    threshold: float = 0.85,
    output: str = "data/processed/reclustering/reclustered_run.run",
    tag: str = "reclustered"
):
    cfg = RunConfig(
        run_path=run,
        ms_split=ms_split,
        encoder_name=encoder,
        threshold=threshold,
        output_path=output,
        tag=tag
    )
    pipeline = RunReclusterPipeline(cfg)
    df = pipeline.run()
    pipeline.save(df)
    typer.echo(f"Saved reclustered run to {output}")

if __name__ == "__main__":
    typer.run(main)
