import os
import yaml
import importlib
import logging
import json
from typing import Dict
from app.engine.loaders.file import FileLoaderConfig, FileLoaderConfigSummary, get_file_documents
from app.engine.loaders.web import WebLoaderConfig, get_web_documents
from app.engine.loaders.db import DBLoaderConfig, get_db_documents, get_db_documentsBD

logger = logging.getLogger(__name__)


def load_configs():
    with open("config/loaders.yaml") as f:
        configs = yaml.safe_load(f)
    return configs


def get_documents():
    documents = []
    config = load_configs()
    for loader_type, loader_config in config.items():
        logger.info(
            f"Loading documents from loader: {loader_type}, config: {loader_config}"
        )
        match loader_type:
            case "file":
                document = get_file_documents(FileLoaderConfig(**loader_config))
            case "web":
                document = get_web_documents(WebLoaderConfig(**loader_config))
            case "db":
                document = get_db_documents(
                     configs=[DBLoaderConfig(**cfg) for cfg in loader_config]
                 )
            case "dbmetadata":
                document = []
            case _:
                raise ValueError(f"Invalid loader type: {loader_type}")
        documents.extend(document)

    return documents

def get_documents_summary():
    documents = []
    config = load_configs()
    for loader_type, loader_config in config.items():
        logger.info(
            f"Loading documents from loader: {loader_type}, config: {loader_config}"
        )
        match loader_type:
            case "file":
                document = get_file_documents(FileLoaderConfigSummary(**loader_config))
            case "web":
                document = get_web_documents(WebLoaderConfig(**loader_config))
            case "db":
                document = get_db_documents(
                     configs=[DBLoaderConfig(**cfg) for cfg in loader_config]
                 )
            case "dbmetadata":
                document = []
            case _:
                raise ValueError(f"Invalid loader type: {loader_type}")
        documents.extend(document)

    return documents

def get_documentsBD():
    documents = []
    config = load_configs()

    for loader_type, loader_config in config.items():
        if loader_type == "dbmetadata":
            document = get_db_documentsBD(
                    configs=[DBLoaderConfig(**cfg) for cfg in loader_config]
                )
            documents.extend(document)

    return documents


def get_metadata():
    with open('config/metadata-panel.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Crear un diccionario con "dictamen" como clave y el resto de la información como valor
    dictamen_dict = {item["discrepancia"]: item for item in data}

    return dictamen_dict



