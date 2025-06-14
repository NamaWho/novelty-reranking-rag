import argparse
import json
from embedding_generator import run_pipeline
from config import Config

def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings and similarity mask with selectable encoder and threshold."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to a .txt file with one sentence per line"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the sentence-transformers model to use"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Cosine similarity threshold for filtering"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output JSON file to write embeddings, similarities, and mask"
    )

    args = parser.parse_args()
    # Load sentences
    with open(args.input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    # Run embedding pipeline
    result = run_pipeline(
        sentences,
        encoder_name=args.encoder,
        threshold=args.threshold
    )

    # Save results
    output_data = {
        "encoder": args.encoder,
        "threshold": args.threshold,
        "sentences": sentences,
        "similarity_matrix": result["similarity_matrix"].tolist(),
        "mask": result["mask"].tolist(),
    }
    with open(args.output, 'w', encoding='utf-8') as fout:
        json.dump(output_data, fout, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()