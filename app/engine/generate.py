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
from app.engine.loaders import get_documents
from app.engine.loaders import get_documentsBD
from app.engine.loaders import get_metadata
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
    documentsBD = get_documentsBD()
    dictamen_dict = get_metadata()


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
            #doc.metadata["descripcion"]= next((row['descripcion'] for row in documentsBD if row['discrepancia'] == code), "N/A")
            #doc.metadata["materia"]= next((row['materia'] for row in documentsBD if row['discrepancia'] == code), "N/A")
            #doc.metadata["submateria"]= next((row['submateria'] for row in documentsBD if row['discrepancia'] == code), "N/A")
            #doc.metadata["fechafinaliza"]= next((row['fechafinaliza'] for row in documentsBD if row['discrepancia'] == code), "N/A")
            #doc.metadata["doctofinaliza"]= next((row['doctofinaliza'] for row in documentsBD if row['discrepancia'] == code), "N/A")
            #doc.metadata["fecha_presentacion"]= next((row['fecha'] for row in documentsBD if row['discrepancia'] == code), "N/A")

            #print (code)
            if code in dictamen_dict:
                # Extraer datos del diccionario para el código dado
                summary = dictamen_dict.get(code, {}).get("summary", "N/A")
                origen_discrepancia = dictamen_dict.get(code, {}).get("origen_discrepancia", {})
                keywords = dictamen_dict.get(code, {}).get("keywords", [])
                concurrencia = dictamen_dict.get(code, {}).get("concurrencia", [])

                # Obtener valores de origen_discrepancia de forma segura
                #fecha_presentacion = safe_get(origen_discrepancia, ["fecha_presentacion"], "N/A")
                discrepantes = safe_get(origen_discrepancia, ["discrepantes"], [])
                documentos_presentados = safe_get(origen_discrepancia, ["documentos_presentados"], [])
                admisibilidad = safe_get(origen_discrepancia, ["admisibilidad"], [])
                inhabilidades = safe_get(origen_discrepancia, ["inhabilidades"], [])
                programa_trabajo = safe_get(origen_discrepancia, ["programa_trabajo"], [])
                mayorias_unanimidad = safe_get(origen_discrepancia, ["mayorias_unanimidad"], [])

                # Crear listas para cada concepto
                nombres = []
                alternativas = []
                analisis = []
                dictamen = []
                constancia = []

                materias = dictamen_dict.get(code, {}).get("materias", [])
                for materia in materias:
                    nombres.append(materia.get('nombre', 'N/A'))
                    alternativas.append(materia.get('alternativas', []))
                    analisis.append(materia.get('analisis', 'N/A'))
                    dictamen.append(materia.get('dictamen', 'N/A'))
                    constancia.append(materia.get('constancia', 'N/A'))

                # Asignar datos a las metadatas del documento
                #doc.metadata.update({
                    #"summary": summary,
                    #"fecha_presentacion": fecha_presentacion,
                    #"discrepantes": discrepantes,
                    #"documentos_presentados": documentos_presentados,
                    #"admisibilidad": admisibilidad,
                    #"inhabilidades": inhabilidades,
                    #"programa_trabajo": programa_trabajo,
                    #"mayorias_unanimidad": mayorias_unanimidad,
                    #"keywords": keywords,
                    #"concurrencia": concurrencia,
                    # "materias": nombres,
                    #"alternativas": alternativas,
                    #"analisis": analisis,
                    #"dictamen": dictamen,
                    #"constancia": constancia
                #})
        # else :
        #         doc.metadata.update({
        #             "summary": summary,
        #         })
            #print (sys.getsizeof(doc.metadata))
            # if sys.getsizeof(doc.metadata) > 400:
            #     print (sys.getsizeof(doc.metadata))
            #     print (doc.metadata["file_name"])
            # print(doc.metadata)
        doc.text = re.sub(pattern2, lambda m: f"{int(m.group(1)):02d}-{m.group(2)}", doc.text)

    #docstore = SimpleDocumentStore() #get_doc_store()
    docstore = get_doc_store()
    vector_store = get_vector_store_qdrand()

    # Run the ingestion pipeline
    _ = run_pipeline(docstore, vector_store, documents)

    # 41429 bytes, which exceeds the limit of 40960

    # Build the index and persist storage
    persist_storage(docstore, vector_store)

    logger.info("Finished generating the index")


if __name__ == "__main__":
    generate_datasource()
