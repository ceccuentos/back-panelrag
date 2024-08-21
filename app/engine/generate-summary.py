from dotenv import load_dotenv
import re
import sys
import unicodedata

load_dotenv()

import os
import logging
from llama_index.core.settings import Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import (
    SentenceSplitter,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage import StorageContext
from app.settings import init_settings
from app.engine.loaders import get_documents
from app.engine.vectordb import (
    get_vector_store,
    get_vector_store_qdrand
)


from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    #BaseExtractor,
)

from llama_index.core import get_response_synthesizer
from llama_index.core import DocumentSummaryIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")


def get_doc_store():

    # If the storage directory is there, load the document store from it.
    # If not, set up an in-memory document store since we can't load from a directory that doesn't exist.
    if os.path.exists(STORAGE_DIR):
        return SimpleDocumentStore.from_persist_dir(STORAGE_DIR)
    else:
        return SimpleDocumentStore()


def run_pipeline(docstore, vector_store, documents):

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=Settings.chunk_size,
                chunk_overlap=Settings.chunk_overlap,
                separator=" ",
            ),
            Settings.embed_model,
            #TitleExtractor(nodes=5),
            #QuestionsAnsweredExtractor(),
            #SummaryExtractor(summaries=["prev", "self"]),
        ],
        docstore=docstore,
        docstore_strategy="upserts_and_delete",
        vector_store=vector_store,
    )

    # Run the ingestion pipeline and store the results
    nodes = pipeline.run(show_progress=True, documents=documents)

    return nodes


def persist_storage(docstore, vector_store):
    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        #vector_store=vector_store,
    )

    storage_context.persist(STORAGE_DIR)



def remove_special_characters(text):
    # Normalizar la cadena a una forma compatible con la eliminación de acentos
    normalized = unicodedata.normalize('NFKD', text)
    # Filtrar solo caracteres ASCII
    ascii_text = ''.join([c for c in normalized if ord(c) < 128])
    return ascii_text

def clean_up_text(content: str) -> str:
    """
    Remove unwanted characters and patterns in text input.

    :param content: Text input.

    :return: Cleaned version of original text input.
    """

    # Fix hyphenated words broken by newline
    content = re.sub(r'(\w+)-\n(\w+)', r'\1\2', content)

    # Remove specific unwanted patterns and characters
    unwanted_patterns = [
        "\\n", "  —", "——————————", "—————————", "—————",
        r'\\u[\dA-Fa-f]{4}', r'\uf075', r'\uf0b7'
    ]
    for pattern in unwanted_patterns:
        content = re.sub(pattern, "", content)

    # Fix improperly spaced hyphenated words and normalize whitespace
    content = re.sub(r'(\w)\s*-\s*(\w)', r'\1-\2', content)
    content = re.sub(r'\s+', ' ', content)
    return content


# Función para obtener un valor de un diccionario de manera segura
def safe_get(dictionary, keys, default=None):
    for key in keys:
        dictionary = dictionary.get(key, {})
    return dictionary or default

def generate_datasource():
    init_settings()
    logger.info("Generate index for the provided data")

    # Expresión regular
    pattern = re.compile(r"(?i)Dictamen\s+(\d{2,4}-\d{2,4})(?:\s+([\w\s]+))?\.(pdf|md)")
    pattern2 = r'(\d{1,2})\s*[-/]\s*(\d{4})'

    # Get the stores and documents or create new ones
    documents = get_documents()

    # Set the metadata for the documents
    for doc in documents:
        cleaned_str = remove_special_characters(safe_get(doc.metadata, ["file_name"], "N/A"))
        if cleaned_str != "N/A":
            match = pattern.match(cleaned_str)
            if match:
                code = match.group(1)
                entity = match.group(2) if match.group(2) else ""
            else:
                entity = doc.metadata["file_name"]
                code =  doc.metadata["file_name"]

            doc.metadata["private"] = "false"
            doc.metadata["dictamen"] =code
            doc.metadata["discrepancia"] =code


        doc.text = re.sub(pattern2, lambda m: f"{int(m.group(1)):02d}-{m.group(2)}", doc.text)

    docstore = SimpleDocumentStore() #Nuevos Docs
    #docstore = get_doc_store()
    vector_store = get_vector_store_qdrand()

    # Run the ingestion pipeline. Return Nodes, por lo que no me sirve
    # _ = run_pipeline(docstore, vector_store, documents)

    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize", use_async=True
    )
    splitter = SentenceSplitter(chunk_size=1024)

    doc_summary_index = DocumentSummaryIndex.from_documents(
        documents,
        transformations=[splitter],
        #response_synthesizer=response_synthesizer,
        show_progress=True,
    )

    query_enginesummary = doc_summary_index.as_query_engine(
        response_mode="tree_summarize", use_async=True
        )

    response = query_enginesummary.query("Dame detalles del dictamen 68-2023")
    print (f"desde summary generate: {response}")

    # Build the index
    doc_summary_index.storage_context.persist(STORAGE_DIR)

    logger.info("Finished generating the index-summary")


if __name__ == "__main__":
    generate_datasource()
