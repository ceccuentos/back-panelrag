import os
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

from llama_index.vector_stores.qdrant import QdrantVectorStore


def get_vector_store_qdrand():
    collection_name = os.getenv("QDRANT_COLLECTION")
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not collection_name or not url:
        raise ValueError(
            "Please set QDRANT_COLLECTION, QDRANT_URL"
            " to your environment variables or config them in the .env file"
        )
    store = QdrantVectorStore(
        collection_name=collection_name,
        url=url,
        api_key=api_key,
        #enable_hybrid=True
    )
    return store

def get_vector_store_qdrand_summary(index_name : str):
    collection_name = os.getenv(index_name)

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not collection_name or not url:
        raise ValueError(
            "Please set QDRANT_COLLECTION, QDRANT_URL, or another index_name"
            " to your environment variables or config them in the .env file"
        )
    store = QdrantVectorStore(
        collection_name=collection_name,
        url=url,
        api_key=api_key,
        #enable_hybrid=True
    )
    return store



def get_vector_store():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    #environment = os.getenv("PINECONE_ENVIRONMENT")
    if not api_key or not index_name: # or not environment:
        raise ValueError(
            "Please set PINECONE_API_KEY, PINECONE_INDEX_NAME"
            " to your environment variables or config them in the .env file"
        )
    store = PineconeVectorStore(
        api_key=api_key,
        index_name=index_name,
        #add_sparse_vector=True,
        #namespace=""
        #environment=environment,
    )

    return store

def get_vector_index_store():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    #environment = os.getenv("PINECONE_ENVIRONMENT")
    if not api_key or not index_name: # or not environment:
        raise ValueError(
            "Please set PINECONE_API_KEY, PINECONE_INDEX_NAME"
            " to your environment variables or config them in the .env file"
        )

    pc = Pinecone(api_key=api_key)

    pinecone_index = pc.Index(index_name)


    return pinecone_index
