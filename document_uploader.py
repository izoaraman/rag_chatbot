import os
import faiss
from sentence_transformers import SentenceTransformer
from global_settings import INDEX_STORAGE, EMBEDDING_MODEL

# Function to build vector and tree indexes
def build_indexes(nodes):
    index = faiss.IndexFlatL2(nodes.shape[1])  # Assuming nodes is a 2D numpy array
    index.add(nodes)
    faiss.write_index(index, os.path.join(INDEX_STORAGE, "vector.index"))
    # Here we can save any tree structures if necessary, using pickling or other local methods
    return index


def load_index():
    index = faiss.read_index(os.path.join(INDEX_STORAGE, "vector.index"))
    return index


def ingest_documents(file_path):
    # Load and split documents
    with open(file_path, 'r') as f:
        documents = f.readlines()

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedding_model.encode(documents)

    return embeddings, documents
