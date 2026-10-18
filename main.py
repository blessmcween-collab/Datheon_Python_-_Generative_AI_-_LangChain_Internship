"""CLI: python main.py invoice samples/invoice.txt --provider gemini"""
import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from llm_extractor import ExtractionError, analyse, get_provider


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Validated invoice/resume extraction with LLMs")
    p.add_argument("doc_type", choices=["invoice", "resume"])
    p.add_argument("file", type=Path)
    p.add_argument("--provider", choices=["gemini", "openai"], default="gemini")
    p.add_argument("--model", default=None)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    provider = get_provider(args.provider, args.model)

    try:
        result = analyse(args.file.read_text(encoding="utf-8"), args.doc_type, provider, args.retries)
    except ExtractionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        for i, a in enumerate(e.attempts, 1):
            print(f"\nAttempt {i} errors:\n{a['errors']}", file=sys.stderr)
        return 1

    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
