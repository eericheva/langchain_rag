from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore

from embedders.create_llm_emb_default import create_llm_emb_default


def create_llm_emb_cached(llm_emb=None):
    if llm_emb is None:
        llm_emb = create_llm_emb_default()
    # to use caching results from embedding models you need to change basic embedding model (llm_emb) to cached
    # embedding model (cached_llm_emb).
    # This (cached_llm_emb) will be provided to the vector store as embedding parameter instead of basic llm_emb.

    # To store cached embeddings on the disk
    # store = LocalFileStore("./tests/data/cache/")
    # To store cache in the RAM
    store = InMemoryByteStore()
    cached_llm_emb = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=llm_emb,  # base embedding model
        document_embedding_cache=store,  # which store use to cache embeddings, could be any ByteStore
        # (Optional) if your prompt or query is huge enough, may be you want to store it in cache as well
        query_embedding_cache=store,
        namespace=llm_emb.model,  # just to avoid collisions with other caches
    )
    return cached_llm_emb
