import chromadb
from chromadb.config import Settings
from config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME

_client = None
_collection = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(allow_reset=True),
        )
    return _client


def get_collection():
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

def get_indexed_videos() -> list:
        # Get unique video names from metadata
        results = get_collection().get(include=["metadatas"])

        if not results['metadatas']:
            return []
        # Extract unique video names from the list of metadata dicts
        return list(set(m['video_name'] for m in results['metadatas']))