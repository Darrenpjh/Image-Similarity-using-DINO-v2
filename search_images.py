import os
from qdrant_client import QdrantClient
from utils import load_dinov2, compute_embedding

IMAGES_FOLDER = "./images"
QDRANT_PATH = "./qdrant_storage"
COLLECTION_NAME = "image_vectors"
TOP_K = 10

def main():
    processor, model, device = load_dinov2()
    qdrant = QdrantClient(path=QDRANT_PATH)

    # Choose your query image
    query_image = os.path.join(IMAGES_FOLDER, "query.jpg")  # Change as needed
    if not os.path.exists(query_image):
        print(f"Query image not found: {query_image}")
        return

    print(f"Searching for images similar to: {query_image}")
    query_vector = compute_embedding(query_image, processor, model, device)
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=TOP_K
    )

    print("\nTop similar images:")
    for hit in hits:
        print(f"File: {hit.payload['filename']} | Score: {hit.score:.4f}")

if __name__ == "__main__":
    main()
