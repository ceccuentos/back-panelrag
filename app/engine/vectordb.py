import os
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

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
