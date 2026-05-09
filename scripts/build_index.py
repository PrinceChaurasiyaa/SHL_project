#!/usr/bin/env python3
"""
scripts/build_index.py

Run once to precompute sentence embeddings and build FAISS index:
    python -m scripts.build_index

Writes:
    data/index.faiss
    data/index_ids.npy
    data/embeddings.npy
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from app.catalog import Catalog


def build(
    catalog_path: str | None = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> None:
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"Loading catalog from {catalog_path or data_dir / 'catalog.json'} ...")
    catalog = Catalog.load(catalog_path)
    entries = catalog.entries
    print(f"  {len(entries)} entries loaded.")

    print(f"Loading embedding model: {model_name} ...")
    encoder = SentenceTransformer(model_name)

    texts = [e.search_text for e in entries]
    ids = [e.entity_id for e in entries]

    print(f"Encoding {len(texts)} entries (batch_size={batch_size}) ...")
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")
    print(f"  Embeddings shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")

    faiss.write_index(index, str(data_dir / "index.faiss"))
    np.save(str(data_dir / "index_ids.npy"), np.array(ids, dtype=object))
    np.save(str(data_dir / "embeddings.npy"), embeddings)

    print("Done.")
    print(f"  -> {data_dir / 'index.faiss'}")
    print(f"  -> {data_dir / 'index_ids.npy'}")
    print(f"  -> {data_dir / 'embeddings.npy'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build FAISS index from SHL catalog")
    parser.add_argument("--catalog", default=None, help="Path to catalog.json")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    build(catalog_path=args.catalog, model_name=args.model, batch_size=args.batch_size)
