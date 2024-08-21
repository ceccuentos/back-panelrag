import logging
from llama_index.core.indices import VectorStoreIndex
from app.engine.vectordb import (
    get_vector_store,
    get_vector_store_qdrand,
    get_vector_store_qdrand_summary
)
from llama_index.core import (
     StorageContext,
)

logger = logging.getLogger("uvicorn")




def get_index():
    logger.info("Connecting vector store...")
    store = get_vector_store_qdrand() #get_vector_store()
    # Load the index from the vector store
    # If you are using a vector store that doesn't store text,
    # you must load the index from both the vector store and the document store

    index = VectorStoreIndex.from_vector_store(store)
    logger.info("Finished load index from vector store.")
    return index

def get_index_summary(index_name : str ):
    logger.info("Connecting vector store...")
    store = get_vector_store_qdrand_summary(index_name) #get_vector_store()
    # Load the index from the vector store
    # If you are using a vector store that doesn't store text,
    # you must load the index from both the vector store and the document store

    index = VectorStoreIndex.from_vector_store(store)
    logger.info("Finished load index from vector store.")
    return index



def get_context():
    logger.info("Connecting store context...")
    store = get_vector_store_qdrand() #get_vector_store()
    # Load the index from the vector store
    # If you are using a vector store that doesn't store text,
    # you must load the index from both the vector store and the document store
    index = VectorStoreIndex.from_vector_store(store)
    logger.info("Finished load index from vector store.")
    return index

