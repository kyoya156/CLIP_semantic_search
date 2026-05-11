"""
db.py — Index images into ChromaDB using CLIP embeddings.
Run this once (or whenever you add new images) before launching app.py.

Usage:
    python db.py --images ./images --reset
"""

import os
import argparse
import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "image_search"
MODEL_NAME = "clip-ViT-B-32"
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def load_model() -> SentenceTransformer:
    print(f"[1/3] Loading CLIP model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def get_image_paths(folder: str) -> list[str]:
    paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    if not paths:
        raise FileNotFoundError(f"No supported images found in '{folder}'")
    return sorted(paths)


def index_images(
    img_folder: str,
    reset: bool = False,
    batch_size: int = 32,
) -> None:
    model = load_model()

    # ── ChromaDB setup ───────────────────────────────────────────────────────
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        print(f"[2/3] Resetting collection '{COLLECTION_NAME}'")
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )

    # ── Skip already-indexed images ──────────────────────────────────────────
    existing_ids: set[str] = set(collection.get(include=[])["ids"])
    img_paths = get_image_paths(img_folder)
    to_index = [p for p in img_paths if os.path.abspath(p) not in existing_ids]

    if not to_index:
        print("[2/3] All images already indexed. Nothing to do.")
        return

    print(f"[2/3] Indexing {len(to_index)} image(s) (skipping {len(img_paths) - len(to_index)} already indexed)…")

    # ── Batch encode & upsert ────────────────────────────────────────────────
    for i in tqdm(range(0, len(to_index), batch_size), desc="Batches"):
        batch_paths = to_index[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        embeddings = model.encode(images, convert_to_tensor=False, show_progress_bar=False)

        collection.upsert(
            ids=[os.path.abspath(p) for p in batch_paths],
            embeddings=[e.tolist() for e in embeddings],
            metadatas=[
                {
                    "path": os.path.abspath(p),
                    "filename": os.path.basename(p),
                }
                for p in batch_paths
            ],
        )

    print(f"[3/3] Done. Collection now contains {collection.count()} image(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index images into ChromaDB with CLIP.")
    parser.add_argument("--images", default="./images", help="Folder of images to index")
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild the collection")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    args = parser.parse_args()

    index_images(args.images, reset=args.reset, batch_size=args.batch_size)