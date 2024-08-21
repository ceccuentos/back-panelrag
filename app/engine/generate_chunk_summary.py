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
    SemanticSplitterNodeParser,
    TokenTextSplitter
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage import StorageContext
from app.settings import init_settings
from app.engine.loaders import get_documents, get_documents_summary
from app.engine.loaders import get_documentsBD
from app.engine.loaders import get_metadata
from app.engine.vectordb import (
    get_vector_store,
    get_vector_store_qdrand,
    get_vector_store_qdrand_summary
)

from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    #BaseExtractor,
)

from llama_index.core.indices import VectorStoreIndex
from llama_index.core.schema import IndexNode

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
    from llama_index.core.node_parser import MarkdownNodeParser

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=Settings.chunk_size,
                chunk_overlap=Settings.chunk_overlap,
                separator=" ",
            ),
            # MarkdownNodeParser(),
            # TokenTextSplitter(
            #     chunk_size=Settings.chunk_size,
            #     chunk_overlap=Settings.chunk_overlap,
            #     separator=" ",
            # ),
            # SemanticSplitterNodeParser(
            #     buffer_size=1,
            #     breakpoint_percentile_threshold=95,
            #     embed_model=Settings.embed_model
            #     ),
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
        vector_store=vector_store,
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
    documents = get_documents_summary()
    documentsBD = get_documentsBD()
    #dictamen_dict = get_metadata()


    #print (documentsBD)
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
            #doc.metadata["autor"] =entity

            # Busca Metadata desde BD
            #next((row['submateria'] for row in rows_dicts if row['discrepancia'] == discrepancia_buscar), None)
            doc.metadata["descripcion"]= next((row['descripcion'] for row in documentsBD if row['discrepancia'] == code), "N/A")
            #doc.metadata["materia"]= next((row['materia'] for row in documentsBD if row['discrepancia'] == code), "N/A")
            #doc.metadata["submateria"]= next((row['submateria'] for row in documentsBD if row['discrepancia'] == code), "N/A")
            doc.metadata["fechafinaliza"]= next((row['fechafinaliza'] for row in documentsBD if row['discrepancia'] == code), "N/A")
            doc.metadata["doctofinaliza"]= next((row['doctofinaliza'] for row in documentsBD if row['discrepancia'] == code), "N/A")
            doc.metadata["fecha_presentacion"]= next((row['fecha'] for row in documentsBD if row['discrepancia'] == code), "N/A")

        doc.text = re.sub(pattern2, lambda m: f"{int(m.group(1)):02d}-{m.group(2)}", doc.text)

    docstore = SimpleDocumentStore() #get_doc_store()
    #docstore = get_doc_store()
    vector_store = get_vector_store_qdrand_summary("QDRANT_COLLECTION_SUMMARY")


    # Paso 1 - Chunk References: Smaller Child Chunks Referring to Bigger Parent Chunk
    # ================================================================================

    import uuid
    node_parser = SentenceSplitter(chunk_size=1024)

    base_nodes = node_parser.get_nodes_from_documents(documents)
# set node ids to be a constant
    for idx, node in enumerate(base_nodes):
        node.id_ = str(uuid.uuid4())

    sub_chunk_sizes = [256, 512]
    sub_node_parsers = [SentenceSplitter(chunk_size=c) for c in sub_chunk_sizes]

    all_nodes = []
    for base_node in base_nodes:
        for n in sub_node_parsers:
            sub_nodes = n.get_nodes_from_documents([base_node])
            sub_inodes = [
                IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
            ]
            all_nodes.extend(sub_inodes)

        # also add original node to node
        original_node = IndexNode.from_text_node(base_node, base_node.node_id)
        all_nodes.append(original_node)

    #all_nodes_dict = {n.node_id: n for n in all_nodes}

    # Paso 2 - Metadata References: Summaries + Generated Questions referring to a bigger chunk
    # =========================================================================================
    extractors = [
         QuestionsAnsweredExtractor(questions=2, show_progress=True),
        ]

    metadata_dicts = [] #documents.metadata
    for extractor in extractors:
        metadata_dicts.extend(extractor.extract(base_nodes))


    import copy

    all_nodes = copy.deepcopy(base_nodes)
    for idx, d in enumerate(metadata_dicts):
        inode_q = IndexNode(
            text=d["questions_this_excerpt_can_answer"],
            index_id=base_nodes[idx].node_id,
        )

        all_nodes.extend([inode_q]) #, inode_s

    # Necesario al momento de construir el RecursiveRetriever
    all_nodes_dict = {n.node_id: n for n in all_nodes}

    storage_context = StorageContext.from_defaults(vector_store=vector_store)


    #print (all_nodes)
    # create your index
    _ = VectorStoreIndex(
        all_nodes,
        storage_context=storage_context
    )

    # Guarda Diccionario para busqueda recursiva
    import pickle
    with open(f"{STORAGE_DIR}/dicnodes_summary.pkl", 'wb') as archivo:
        pickle.dump(all_nodes_dict, archivo)

    # Run the ingestion pipeline
    # _ = run_pipeline(docstore, vector_store, documents)

    #persist_storage(all_nodes, vector_store)

    logger.info("Finished generating the index chunk" )


if __name__ == "__main__":
    generate_datasource()
