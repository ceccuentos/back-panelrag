import os
import logging
from typing import List
from pydantic import BaseModel, validator
from llama_index.core.indices.vector_store import VectorStoreIndex

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json

logger = logging.getLogger(__name__)


URI_BD = os.getenv("URI_BD", "mysql+pymysql://user:pass@localhost:3306/mydb")
URI_BD_QA = os.getenv("URI_BD_QA", "mysql+pymysql://user:pass@localhost:3306/mydb")

class DBLoaderConfig(BaseModel):
    uri: str
    queries: List[str]


def get_db_documents(configs: list[DBLoaderConfig]):
    from llama_index.readers.database import DatabaseReader

    docs = []
    for entry in configs:
        loader = DatabaseReader(uri=URI_BD)
        for query in entry.queries:
            logger.info(f"Loading data from database with query: {query}")
            documents = loader.load_data(query=query)
            docs.extend(documents)

    return documents


def get_db_documentsBD(configs: List[DBLoaderConfig]):

    engine = create_engine(URI_BD)
    Session = sessionmaker(bind=engine)
    session = Session()
    listdoc = []
    try:
        for entry in configs:
            for query in entry.queries:
                # Ejecuta la consulta y convierte los resultados a JSON
                results = session.execute(text(query)).fetchall()

        rows_dicts = [
            {
                'discrepancia': row[0],
                'descripcion': row[1],
                'fecha': row[2],
                'materia': row[3],
                'submateria': row[4],
                'fechafinaliza': row[5],
                'doctofinaliza': row[6]
            }
            for row in results
        ]

    finally:
    # Cierra la sesión
        session.close()

    return rows_dicts



# def get_db_documentsBD(configs: list[DBLoaderConfig]):
#     from llama_index.readers.database import DatabaseReader

#     docs = []
#     for entry in configs:
#         loader = DatabaseReader(uri=URI_BD)
#         for query in entry.queries:
#             rows = loader.sql_database.run_sql(command=query)
#             docs.extend(rows)

#     return docs

