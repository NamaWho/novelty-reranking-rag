import subprocess
import sys
import argparse
from datetime import datetime
import logging
import os

# ANSI escape codes for colors
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"

def run_command(command: str):
    logging.info(f"Running command: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        logging.error(f"Command failed: {command}")
        print(f"{RED}❌ Command failed:{RESET} {command}")
        sys.exit(result.returncode)
    else:
        print(f"{GREEN}✅ Success:{RESET} {command}")

def setup_logger(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG reranking with lightning-ir.")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/rag/re-rank.yaml",
        help="Path to config file (default: ./configs/rag/re-rank.yaml)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/set-encoder-novelty/",
        help="Path to the finetuned model (default: ./models/set-encoder-novelty/)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"rerank_rag_{timestamp}.log")
    setup_logger(log_file)

    print(f"{CYAN}📚 Starting RAG reranking...{RESET}")
    logging.info("=== Starting RAG reranking run ===")

    run_command(
        f"lightning-ir re_rank "
        f"--config {args.config} "
        f"--model.model_name_or_path {args.model_path}"
    )

    print(f"{CYAN}🎉 Done.{RESET}")
    logging.info("=== RAG reranking completed successfully ===")

if __name__ == "__main__":
    main()
