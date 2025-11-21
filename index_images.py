import os
from qdrant_client import QdrantClient, models
from utils import load_dinov2, compute_embedding

IMAGES_FOLDER = "./images"
QDRANT_PATH = "./qdrant_storage"
COLLECTION_NAME = "image_vectors"

def get_indexed_filenames(qdrant, collection):
    count = qdrant.count(collection_name=collection).count
    if count == 0:
        return set()
    existing = set()
    offset = 0
    limit = 1000
    while offset < count:
        points, _ = qdrant.scroll(
            collection_name=collection,
            with_payload=True,
            limit=limit,
            offset=offset
        )
        for pt in points:
            payload = pt.payload
            if "filename" in payload:
                existing.add(payload["filename"])
        offset += limit
    return existing


def main():
    processor, model, device = load_dinov2()
    qdrant = QdrantClient(path=QDRANT_PATH)

    existing_filenames = get_indexed_filenames(qdrant, COLLECTION_NAME)
    print(f"Already indexed: {len(existing_filenames)} images.")

    img_files = [f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png','.tif'))]
    records = []
    new_count = 0

    for idx, fname in enumerate(img_files):
        if fname in existing_filenames:
            print(f"Skipping (already indexed): {fname}")
            continue
        img_path = os.path.join(IMAGES_FOLDER, fname)
        print(f"Embedding {fname} ...")
        emb = compute_embedding(img_path, processor, model, device)
        records.append(
            models.Record(id=int(os.path.splitext(fname)[0], 36) if fname.isalnum() else idx,  # unique id, or just idx
                          vector=emb,
                          payload={"filename": fname, "path": img_path})
        )
        new_count += 1

    if not records:
        print("No new images to index.")
        return

    vector_dim = len(records[0].vector)
    try:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE)
        )
        print("Collection recreated (all old points deleted).")
        # (this will reset the DB and re-index everything, use only for full reindex)
    except Exception:
        pass  # if collection exists, continue

    qdrant.upload_records(
        collection_name=COLLECTION_NAME,
        records=records
    )
    print(f"Indexed {new_count} new images.")

if __name__ == "__main__":
    main()
