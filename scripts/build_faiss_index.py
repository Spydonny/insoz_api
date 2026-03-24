from __future__ import annotations

import argparse
from pathlib import Path

from app.services.rag_service import build_and_save_faiss_index, get_rag_paths


def main() -> int:
    paths = get_rag_paths()

    print(f"PDF directory: {paths.pdf_dir}")
    print(f"FAISS directory: {paths.faiss_dir}")
    parser = argparse.ArgumentParser(description="Build FAISS index from speech therapy PDFs")
    parser.add_argument("--pdf-dir", type=str, default=str(paths.pdf_dir), help="Directory with .pdf files")
    parser.add_argument("--faiss-dir", type=str, default=str(paths.faiss_dir), help="Output directory for FAISS index")
    args = parser.parse_args()

    result = build_and_save_faiss_index(Path(args.pdf_dir), Path(args.faiss_dir))
    print(f"Built FAISS index at: {args.faiss_dir}")
    print(f"Chunks: {result['chunks']}")
    print("Sources:")
    for s in result["sources"]:
        print(f"- {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

