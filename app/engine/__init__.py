import os
from app.engine.index import get_index, get_index_summary



from fastapi import HTTPException
#from app.engine.getfiltersLLM import GetFiltersPrompt
#from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.postprocessor import LongContextReorder

from llama_index.core import (
    Document,
    #SimpleKeywordTableIndex,
    GPTVectorStoreIndex,
    #VectorStoreIndex
)

from llama_index.core.node_parser import (
    SentenceSplitter
#    SemanticSplitterNodeParser,
 #   TokenTextSplitter
)
from llama_index.core.schema import IndexNode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine

from sqlalchemy import create_engine, text
import re
import json
import openai
import gc
import pandas as pd
import uuid

import pickle
import time
#from app.node_dictionary import ALL_NODES_DICTIONARY

# Variable global que contendrá el diccionario


#from llama_index.core.retrievers import SummaryIndexLLMRetriever
#from llama_index.core.indices.query.query_transform import HyDEQueryTransform
#from llama_index.core.query_engine.transform_query_engine import TransformQueryEngine
#from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core import get_response_synthesizer
#from llama_index.core.tools import QueryEngineTool
# from app.engine.vectordb import get_vector_index_store
# from app.engine.vectordb import get_vector_store
#from app.engine.generate import get_doc_store
# Nuevas
# from llama_index.postprocessor.cohere_rerank import CohereRerank
# from app.engine.loaders import get_metadata
# from llama_index.core.postprocessor import LongContextReorder
# from llama_index.core.postprocessor import SimilarityPostprocessor
#from llama_index.core.postprocessor import PrevNextNodePostprocessor
#from llama_index.core.postprocessor import AutoPrevNextNodePostprocessor
#from llama_index.core.postprocessor import LLMRerank
#from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter
# from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
# from llama_index.core.postprocessor import LLMRerank
# #from llama_index.postprocessor.rankllm_rerank import RankLLMRerank
# from llama_index.core.postprocessor import LLMRerank
# from llama_index.core.retrievers import VectorIndexRetriever

#from llama_index.retrievers.bm25 import BM25Retriever
# from llama_index.core.retrievers import VectorIndexAutoRetriever

# from llama_index.core.indices.query.query_transform.base import (
#from llama_index.core.query_engine import RetrieverQueryEngine
#      StepDecomposeQueryTransform,
# )
# from llama_index.core.query_engine import MultiStepQueryEngine

#from llama_index.core.chat_engine import CondenseQuestionChatEngine
#from llama_index.core import PromptTemplate
#from llama_index.core.llms import ChatMessage, MessageRole

# from llama_index.core.postprocessor import (
#     PrevNextNodePostprocessor,
#     FixedRecencyPostprocessor
# )
# from llama_index.core.settings import Settings
#from llama_index.core.agent import AgentRunner
#from llama_index.core.tools.query_engine import QueryEngineTool
#from app.engine.tools import ToolFactory

# from llama_index.core.indices.struct_store import JSONQueryEngine
# from llama_index.readers.json import JSONReader



# from llama_index.agent.openai import OpenAIAgentWorker

# from llama_index.core.agent import (
#     StructuredPlannerAgent,
#     FunctionCallingAgentWorker,
#     ReActAgentWorker,
# )

#from collections import defaultdict


STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
ACTIVA_STDE = "Busca en el STDE"


LAST_ACCESS_TIME = None
CACHE_EXPIRY = 60  # Segundos de inactividad antes de liberar el caché

def get_query_engine(filters=None, query : str = ""):
    system_prompt = os.getenv("SYSTEM_PROMPT")
    #api_key_cohere = os.getenv("COHERE_API_KEY")

    print ("Chat_engine")
    top_k = os.getenv("TOP_K", 3)

    index = get_index()
    if index is None:
        raise HTTPException(
            status_code=500,
            detail=str(
                "StorageContext is empty - call 'poetry run generate' to generate the storage first"
            ),
        )

    # TODO:
    # Probar si llego a Nodos con index.docstore.docs (https://www.youtube.com/watch?v=GT_Lsj3xj1o&list=PLTZkGHtR085ZjK1srrSZIrkeEzQiMjO9W&index=6)
    # probar subretrievers:
        # retriever = index.as_retreiver(
        #   sub_retrievers=[retriever1, retriever2, ...]
        #   )
        # query_engine= index.as_query_engine(
        #   sub_retrievers=[retriever1, retriever2, ...]
        #   )
        # https://www.youtube.com/watch?v=4g166tdMPdw&list=PLTZkGHtR085ZYstpcTFWqP27D-SPZe6EZ&index=1



    # cohere_rerank = CohereRerank(api_key=api_key_cohere, top_n=10)
    # postprocessor = PrevNextNodePostprocessor(
    #     docstore=index.docstore,
    #     num_nodes=1,  # number of nodes to fetch when looking forawrds or backwards
    #     mode="next",  # can be either 'next', 'previous', or 'both'
    # )


    # postprocessorDate = FixedRecencyPostprocessor(
    #     tok_k=1, date_key="fecha_presentacion"  # the key in the metadata to find the date
    # )

    return index.as_query_engine(
        similarity_top_k=int(top_k),
        system_prompt=system_prompt,
        #chat_mode="openai",
        chat_mode="condense_plus_context",
        filters=filters,
        #node_postprocessors=[postprocessorDate, postprocessor],
    )

def get_chat_engine(filters=None):
    system_prompt = os.getenv("SYSTEM_PROMPT")
    api_key_cohere = os.getenv("COHERE_API_KEY")

    print ("chat_engine_ original")

    top_k = os.getenv("TOP_K", 3)

    index = get_index()
    if index is None:
        raise HTTPException(
            status_code=500,
            detail=str(
                "StorageContext is empty - call 'poetry run generate' to generate the storage first"
            ),
        )

    # cohere_rerank = CohereRerank(api_key=api_key_cohere, top_n=10)
    # postprocessor = PrevNextNodePostprocessor(
    #     docstore=index.docstore,
    #     num_nodes=1,  # number of nodes to fetch when looking forawrds or backwards
    #     mode="next",  # can be either 'next', 'previous', or 'both'
    # )


    # postprocessorDate = FixedRecencyPostprocessor(
    #     tok_k=1, date_key="fecha_presentacion"  # the key in the metadata to find the date
    # )
    # # postprocessor.postprocess_nodes(nodes)

    #from llama_index.core.query_engine import RetrieverQueryEngine
    # index.as_retriever(retriever=retriever_index)
    #_ = index.RetrieverQueryEngine.from_args(retriever_index)
    # import inspect
    # print(vars(retriever_index))
    # print(inspect.getmembers(retriever_index))
    # print(dir(retriever_index))

    return index.as_chat_engine(
        similarity_top_k=int(top_k),
        system_prompt=system_prompt,
        #chat_mode="openai",
        chat_mode="condense_plus_context",
        filters=filters,
        # node_postprocessors=[postprocessorDate, postprocessor],
    )

def generate_documents(contextoBD):
    # Genera documentos a partir del contexto
    return [
        Document(
            text=f"Consulta: {item.get('query')}\nResultado: {json.dumps(item.get('resultado'), ensure_ascii=False)}",
            extra_info={"tipo": "resultado"},
        )
        for item in contextoBD
    ]
ALL_NODES_DICTIONARY = None
def get_chat_engine_hybrid(query : str = "", messages: list = [], filters=None) :
    print ("Chat engine hybrido!!!!")

    system_prompt = os.getenv("SYSTEM_PROMPT")
    chunk_size = os.getenv("CHUNK_SIZE")
    top_k = os.getenv("TOP_K", 3)

    postprocessorLongContext = LongContextReorder()

    formatted_messages = [
        {"role": message.role.value, "content": message.content}
        for message in messages
    ]

    formatted_messages.append({'role':'user', 'content':f"{query}"})

    # **** Busqueda sobre BD con function_call ****
    if "Busca en el STDE".lower() in query.lower(): #1==1:
        print ("Entre")
        respuesta=get_openai_response(formatted_messages, "buscastde")
        print (respuesta.choices[0].message) # Retornamos el mensaje
        response_message=respuesta.choices[0].message

        if response_message.function_call:
            argumentos = [
                {"nombre": response_message.function_call.name, "argumentos": response_message.function_call.arguments},
            ]

            print (f"argumentos:{argumentos}")
            contextoBD=exec_query(argumentos)

            # Hay regist
            if contextoBD:

                for item in contextoBD:
                    query = item.get("query", "Sin nombre")

                    resultado = json.dumps(item.get("resultado", []), ensure_ascii=False, indent=2)
                    error = item.get("error", "")


                    # Crear un documento estructurado para cada consulta
                    texto_documento = f"Consulta: {query}\nResultado: {resultado}\n"
                    if error:
                        texto_documento += f"Error: {error}\n"

                    #print (resultado)
                    #documentos.append(Document(text=texto_documento))
                    promptfn=""


                raw_data = json.loads(resultado)

                if response_message.function_call.name == "personas2pjud":

                    all_documents = getdocument_personas2pjud(resultado)
                    # Todo: Evaluar cual llamada es más conveniente...
                    # index = VectorStoreIndex.from_documents(all_documents)
                    # base_retriever = index.as_retriever(similarity_top_k=6)

                    # all_nodes, all_nodes_dict = getretriever_recursivo(all_documents, chunk_size)

                    # vector_index_chunk = VectorStoreIndex(all_nodes)
                    # vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k=5)

                    # retriever_chunk = RecursiveRetriever(
                    #     "vector",
                    #     retriever_dict={"vector": vector_retriever_chunk},
                    #     node_dict=all_nodes_dict,
                    #     verbose=True,
                    #     )
                    # retriever = getqueryfusion(retriever_chunk, base_retriever, 2)

                    promptfn = getprompt(response_message.function_call.name, raw_data)

                    index = GPTVectorStoreIndex(all_documents)
                    return index.as_chat_engine(
                        similarity_top_k=10,#int(top_k),
                        system_prompt=promptfn,
                        chat_mode="context",
                        filters=None, #filters,
                    )

                    # return CondensePlusContextChatEngine.from_defaults(
                    #         retriever=retriever,
                    #         similarity_top_k=20, #int(top_k),
                    #         system_prompt=promptfn,
                    #         node_postprocessors=[postprocessorLongContext],
                    # )

                elif response_message.function_call.name == "cantidad_discrepancias":

                    all_documents = getdocument_cantidad_discrepancias(resultado)
                    index = GPTVectorStoreIndex(all_documents)

                    promptfn= getprompt(response_message.function_call.name, raw_data)
                    # Crear el retriever
                    return index.as_chat_engine(
                        similarity_top_k=int(top_k),
                        system_prompt=promptfn,
                        chat_mode="context",
                        filters=None, #filters,
                    )
            else :
                print ("Else sin contexto")
                return handle_no_context_response(top_k)


        else:
            print ("Else sin llamada a function_call")
            return handle_no_context_response(top_k)


    # **** Busqueda normal sobre BD vectorial ****
    # Intenta extraer filtro de discrepancia (si viniera)
    respuesta=get_openai_response(formatted_messages, "bdvectorial")
    print (f"bdvectorial: {respuesta.choices[0].message}") # Retornamos el mensaje
    response_message=respuesta.choices[0].message

    filters_=None

    if response_message.function_call:
        arguments_str =response_message.function_call.arguments
        arguments = json.loads(arguments_str)

        dictamen_=None
        discrepancia_=None
        # Recorrer los argumentos
        for key, value in arguments.items():
            print(f"Argumento: {key}, Valor: {value}")
            if key == "dictamen":
                dictamen_ = value
            elif key == "discrepancia":
                discrepancia_ = value

        if dictamen_ is not None:
            filters_ = {"dictamen": dictamen_}

        if discrepancia_ is not None:
            filters_ = {"discrepancia": discrepancia_}


    # # Agregar condición para discrepancia si no es None
    # if discrepancia_ is not None:
    #     filters_["$and"].append(
    #         {
    #             "$or": [
    #                 {"discrepancia": discrepancia_},
    #                 {"discrepancia": ""}
    #             ]
    #         }
    #     )
    #     # Filtro para dictamen y discrepancia
    #     filters_ = {
    #         "$and": [
    #             {
    #                 "$or": [
    #                     {"dictamen": dictamen_},
    #                     {"dictamen": ""}
    #                 ]
    #             },
    #             {
    #                 "$or": [
    #                     {"discrepancia": discrepancia_},
    #                     {"discrepancia": ""}
    #                 ]
    #             }
    #         ]
    #     }

    #filters_ (filters_)

    print (f"filters_: {filters_}")

    index = get_index()
    if index is None:
        raise HTTPException(
            status_code=500,
            detail=str(
                "StorageContext is empty - call 'poetry run generate' to generate the storage first"
            ),
        )

    # Store recursivo
    if index is not None:
        vector_retriever_chunk = index.as_retriever(similarity_top_k=int(top_k))

    # ** Carga Diccionario
    # Todo: Hacer carga por año, sacar desde argumento de function_call en filters_
    global ALL_NODES_DICTIONARY
    load_data_dict()
    # **

    if len(ALL_NODES_DICTIONARY) == 0:
        print("El diccionario está vacío")

    retriever_chunk_recursivo = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever_chunk},
        node_dict=ALL_NODES_DICTIONARY,
        verbose=True,
    )

    # ** Quita de memoria
    unload_data_dict()
    # **

    retriever_summary = get_index_summary("QDRANT_COLLECTION_SUMMARY")

    if retriever_summary is not None:
        retriever_summary = retriever_summary.as_query_engine(similarity_top_k=int(top_k))

    # vector_store_info = VectorStoreInfo(
    #     content_info="Discrepancias",
    #     metadata_info=[
    #         MetadataInfo(
    #             name="dictamen",
    #             type="str",
    #             description=("nombre o codigo de dictamen/discrepancia, si no existe no la consideres")
    #         ),
    #         MetadataInfo(
    #             name="discrepancia",
    #             type="str",
    #             description=("nombre o codigo de discrepancia/dictamen, si no existe no la consideres")
    #         ),
    #         MetadataInfo(
    #              name="materias",
    #              type="str",
    #              description=("materias tratadas en las discrepancias/dictamenes")
    #         )
    #     ]
    # )


    # filters_=None

    # if "Busca en el STDE".lower() not in query.lower():  # Puede venir de los else anteriores **refactorizar!!!
    #     print (f"query previo a filtro:{query}")
    #     filtro = GetFiltersPrompt(vector_store_info=vector_store_info)
    #     filters_ = filtro.generate_filters(query)
    #     print (f"filtros aplicados: {filters_}")

    retriever = QueryFusionRetriever(
        [retriever_chunk_recursivo, retriever_summary],
        retriever_weights=[0.6, 0.4],
        similarity_top_k=10,
        num_queries=3,  # set this to 1 to disable query generation
        mode="relative_score",
        use_async=True,
        verbose=True,  # true para que sea verboso
    )

    # Libera de Memoria variables:
    del retriever_chunk_recursivo
    gc.collect()

    return CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            similarity_top_k=int(top_k),
            system_prompt=system_prompt,
            #verbose=True,
            filters=filters_,
            node_postprocessors=[postprocessorLongContext],
    )

def get_BD():
    from sqlalchemy import create_engine, text
    from llama_index.core import SQLDatabase

    URI_BD = os.getenv("URI_BD", "mysql+pymysql://user:pass@localhost:3306/mydb")
    URI_BD_QA = os.getenv("URI_BD_QA", "mysql+pymysql://user:pass@localhost:3306/mydb")
    URI_BD_LOCAL = os.getenv("URI_BD_LOCAL", "mysql+pymysql://user:pass@localhost:3306/mydb")

    engine = create_engine(URI_BD_LOCAL)
    sql_database = SQLDatabase(engine, include_tables=["discrepancies"])

    from llama_index.core.query_engine import NLSQLTableQueryEngine

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database, tables=["discrepancies"], llm=Settings.llm
    )
    # query_str = "Which city has the highest population?"
    # response = query_engine.query(query_str)

    return query_engine

URI_BD = os.getenv("URI_BD_LOCAL", "mysql+pymysql://user:pass@localhost:3306/mydb")
#URI_BD = os.getenv("URI_BD_QA", "mysql+pymysql://user:pass@localhost:3306/mydb")
engine = create_engine(URI_BD, pool_size=7, max_overflow=10, pool_pre_ping=True, echo=False) #create_engine(URI_BD, echo=False) # Echo=True es verboso

def exec_query(argumentos):
    # Configuración de la base de datos

    contexto = []
    with engine.connect() as connection:
        for arg in argumentos:
            nombre = arg.get("nombre")
            argumentos_raw = arg.get("argumentos")
            cleaned_argumentos_raw = re.sub(r'[\n\r]', '', argumentos_raw)
            argumentos = json.loads(cleaned_argumentos_raw)
            try:
                if nombre == "cantidad_discrepancias":

                    xYear = argumentos.get("year")

                    query = text(f"""
                                 SELECT discrepancia, descripcion, fecha, materia, submateria, fechadoctofinaliza, fechafinaliza, doctofinaliza, disc_year,
                                 CASE WHEN (fechafinaliza IS NULL OR fechafinaliza = '') THEN 'ABIERTA' ELSE 'CERRADA' END as estado,
                                 CASE WHEN doctofinaliza = 'Dictamen' THEN 'Dictaminada' ELSE doctofinaliza END as razoncierre
                                 from dictamen_vw where
                                 disc_year = {xYear}
                                 LIMIT 1000
                                 """)

                    #params = {"year": xYear}
                    result = connection.execute(query).fetchall()


                elif nombre == "personas2pjud":

                    usuario_representante_nombre = argumentos.get("usuario_representante_nombre")
                    if usuario_representante_nombre:
                        usuario_representante_nombre = prep_like(usuario_representante_nombre)
                    else:
                        usuario_representante_nombre = "%%"

                    persona_juridica_nombre = argumentos.get("persona_juridica_nombre")
                    if persona_juridica_nombre:
                        persona_juridica_nombre = prep_like(persona_juridica_nombre)
                    else:
                        persona_juridica_nombre = "%%"

                    mail = argumentos.get("mail")
                    if not mail:
                        mail = ""

                    rut = argumentos.get("persona_juridica_rut")
                    if not rut:
                        rut = ""

                    materia = argumentos.get("materia")
                    if not materia:
                        materia = ""

                    submateria = argumentos.get("submateria")
                    if not submateria:
                        submateria = ""

                    # print (f"persona_juridica_nombre: {persona_juridica_nombre}")
                    # print (f"usuario_representante_nombre: {usuario_representante_nombre}")
                    # print (f"mail: {mail}")
                    # print (f"rut: {rut}")
                    # print (f"materia: {materia}")
                    # print (f"submateria: {submateria}")

                    query_str = """
                        SELECT * from pjud_disc /*personas2pjud*/
                        Where (persona_juridica_nombre ILIKE '$persona_juridica_nombre$'
                                or
                                discrepancia_nombre ILIKE '$persona_juridica_nombre$')
                                 and
                              usuario_representante_nombre ILIKE '$usuario_representante_nombre$'
                                 and
                              (usuario_representante_email = '$mail$' or '$mail$'='')
                                 /*and
                              (persona_juridica_email = '$mail$' or '$mail$'='')*/
                                 and
                              (persona_juridica_rut = '$rut$' or '$rut$'='')
                              and
                              (discrepancia_materia = '$materia$' or '$materia$'='')
                              and
                              (discrepancia_submateria = '$submateria$' or '$submateria$'='')
                              order by 1, 19
                              LIMIT 1000
                    """
                    # params = {
                    #     "persona_juridica_nombre": persona_juridica_nombre or '',
                    #     "usuario_representante_nombre": usuario_representante_nombre or '',
                    #     "mail": mail or ''}

                    query_str = query_str.replace("$persona_juridica_nombre$", persona_juridica_nombre)
                    query_str = query_str.replace("$usuario_representante_nombre$", usuario_representante_nombre)
                    query_str = query_str.replace("$mail$", mail)
                    query_str = query_str.replace("$rut$", rut)
                    query_str = query_str.replace("$materia$", materia)
                    query_str = query_str.replace("$submateria$", submateria)



                    query = text(query_str)

                    result = connection.execute(query).fetchall()
                else:
                    result = []
                    # contexto.append({
                    #     "query": nombre,
                    #     "error": f"No se reconoce la consulta '{nombre}'"
                    # })

                rows = [dict(row._mapping) for row in result]

                # Añadir al contexto
                if rows:
                    contexto.append({
                        "query": query,
                        "resultado": rows
                   })


            except Exception as e:
                contexto.append({
                    "query": nombre,
                    "error": str(e)
                })

    # Convertir contexto a JSON para legibilidad
    return contexto

def prep_like(texto):
    # Dividir el texto por espacios, agregar '%' a cada palabra, y unirlas nuevamente
  #  if not texto or not texto.strip():  # Manejar texto vacío o solo espacios
  #      return None
    return f"%{texto}%"

    # plike = texto.split()
    # texto_prep = "%" + "% ".join(plike) + "%"
    # #texto_prep = "% ".join(plike)
    # return texto_prep #if texto_prep else None

def getdocument_personas2pjud(resultado):
    persona_juridica_docs = {}
    representantes_docs = {}
    discrepancias_docs = {}

    raw_data = json.loads(resultado)
    print(f"cantidad de registros personas2pjud : {len(raw_data)}")
    # Crear documentos separados
    for record in raw_data:
        # Documento de persona jurídica (solo se crea una vez por ID)
        pj_id = record["persona_juridica_id"]
        if pj_id not in persona_juridica_docs:
            persona_juridica_docs[pj_id] = Document(
                text=f"""
                Persona Jurídica:
                - ID: {pj_id}
                - RUT: {record['persona_juridica_rut']}
                - Nombre: {record['persona_juridica_nombre']}
                - Dirección: {record['persona_juridica_direccion']}
                - Teléfono: {record['persona_juridica_telefono']}
                - Email: {record['persona_juridica_email']}
                - Fecha Creación: {record['persona_juridica_creada']}
                - Fecha Modificación: {record['persona_juridica_modificada']}
                """,
                extra_info={
                    "tipo": "persona_juridica",
                    "id": pj_id,
                    "rut" : record['persona_juridica_rut']
                    }
            )

        # Documento de representante
        representante_id = record["usuario_representante_id"]
        if representante_id not in representantes_docs:
            representantes_docs[representante_id] = Document(
                text=f"""
                Usuario Representante:
                - ID: {representante_id}
                - RUT: {record['usuario_representante_rut']}
                - Nombre: {record['usuario_representante_nombre']}
                - Teléfono: {record['usuario_representante_telefono']}
                - Email: {record['usuario_representante_email']}
                - Fecha Creación: {record['usuario_representante_creado']}
                - Fecha Modificación: {record['usuario_representante_modificado']}
                """,
                extra_info={
                    "tipo": "usuario_representante",
                    "persona_juridica_id": pj_id,
                    "nombre": record['usuario_representante_nombre'],
                    "rut": record['usuario_representante_rut'],
                    "mail":record['usuario_representante_email']
                },
            )

        # Crear documento para discrepancia
        discrepancia_id = record["discrepancia_id"]
        if discrepancia_id not in discrepancias_docs:
            discrepancias_docs[discrepancia_id] = Document(
                text=f"""
                Discrepancias:
                - ID: {discrepancia_id}
                - Código: {record['discrepancia_codigo']}
                - Nombre o descripción: {record['discrepancia_nombre']}
                - Materia: {record['discrepancia_materia']}
                - Submateria: {record['discrepancia_submateria']}
                - Fecha Creación: {record['discrepancia_creada']}
                - Fecha Cierre: {record['discrepancia_cerrada']}
                """,
                extra_info={
                    "tipo": "discrepancias",
                    "discrepancia_id": discrepancia_id,
                    "persona_juridica_id": pj_id,
                    "persona_juridica_nombre": record['persona_juridica_nombre'],
                    "persona_juridica_rut": record['persona_juridica_rut'],
                    "materia": record['discrepancia_materia'],
                    "submateria": record['discrepancia_submateria'],
                },
            )

    return (
        list(persona_juridica_docs.values())
        + list(representantes_docs.values())
        + list(discrepancias_docs.values())
    )

def getdocument_cantidad_discrepancias(resultado):

    dictamen_docs = {}

    raw_data = json.loads(resultado)
    print(f"cantidad de registros getdocument_cantidad_discrepancias : {len(raw_data)}")

    df = pd.DataFrame(raw_data)
    df = df.astype("object")  # convierte todas las columnas a tipos compatibles con JSON

    # Generate summaries
    total_by_year = df.groupby("disc_year").size().to_dict()
    total_by_materia = df.groupby("materia").size().to_dict()
    total_by_submateria = df.groupby("submateria").size().to_dict()
    total_by_estado = df["estado"].value_counts().to_dict()
    total_by_resolution = df.groupby("doctofinaliza").size().to_dict()

    # Detailed summaries
    detailed_by_year_materia = df.groupby(["disc_year", "materia"]).size().unstack(fill_value=0).to_dict()
    detailed_by_estado_materia = df.groupby(["estado", "materia"]).size().unstack(fill_value=0).to_dict()
    pending_by_submateria = df[df["estado"] != "CERRADA"].groupby("submateria").size().to_dict()

    # Comparative summaries
    year_comparison = df.groupby("disc_year").size().to_dict()
    resolution_comparison = df[df["estado"] == "CERRADA"].groupby("doctofinaliza").size().to_dict()
    materias_by_year = df.groupby(["disc_year", "materia"]).size().unstack(fill_value=0).idxmax(axis=1).to_dict()

    data = {
        "total_discrepancias_por_anio": total_by_year,
        "total_discrepancias_por_materia": total_by_materia,
        "total_discrepancias_por_submateria": total_by_submateria,
        "total_discrepancias_cerradas_vs_abiertas": total_by_estado,
        "total_discrepancias_por_tipo_resolucion": total_by_resolution,
        "detallado_por_anio_y_materia": detailed_by_year_materia,
        "detallado_por_estado_y_materia": detailed_by_estado_materia,
        "pendientes_por_submateria": pending_by_submateria,
        "comparacion_por_anio": year_comparison,
        "comparacion_por_resolucion": resolution_comparison,
        "materias_mas_recurrentes_por_anio": materias_by_year
    }

    # Add summaries back to the original JSON structure
    totals_text = [
        "Resumen de Totales:",
        "",
        "1. Total de discrepancias por año:",
        json.dumps(data["total_discrepancias_por_anio"], indent=4),
        "",
        "2. Total de discrepancias por materia:",
        json.dumps(data["total_discrepancias_por_materia"], indent=4),
        "",
        "3. Total de discrepancias por submateria:",
        json.dumps(data["total_discrepancias_por_submateria"], indent=4),
        "",
        "4. Total de discrepancias cerradas vs abiertas:",
        json.dumps(data["total_discrepancias_cerradas_vs_abiertas"], indent=4),
        "",
        "5. Total de discrepancias por tipo de resolución:",
        json.dumps(data["total_discrepancias_por_tipo_resolucion"], indent=4),
        "",
        "6. Resumen detallado por año y materia:",
        json.dumps(data["detallado_por_anio_y_materia"], indent=4),
        "",
        "7. Resumen detallado por estado y materia:",
        json.dumps(data["detallado_por_estado_y_materia"], indent=4),
        "",
        "8. Discrepancias pendientes por submateria:",
        json.dumps(data["pendientes_por_submateria"], indent=4),
        "",
        "9. Comparación de discrepancias por año:",
        json.dumps(data["comparacion_por_anio"], indent=4),
        "",
        "10. Comparación de resolución por tipo:",
        json.dumps(data["comparacion_por_resolucion"], indent=4),
        "",
        "11. Materias más recurrentes por año:",
        json.dumps(data["materias_mas_recurrentes_por_anio"], indent=4),
    ]
    totals_text_str = "\n".join(totals_text)

    # Extraer el primer registro
    primer_registro = df.iloc[0]
    # Extraer el valor de 'disc_year'
    disc_year = primer_registro['disc_year']

    # ** Libera memoria del Dataframe
    del df
    gc.collect()

    documents_tot = Document(text=totals_text_str, extra_info={"tipo": "totales", "year": disc_year })

    #print (documents_tot)
    # Crear documentos separados
    for record in raw_data:
        # Documento de persona jurídica (solo se crea una vez por ID)
        dsc_id = record["discrepancia"]
        dsc_year = record["disc_year"]
        dsc_materia = record["materia"]
        dsc_submateria = record["submateria"]

        if dsc_id not in dictamen_docs:
            dictamen_docs[dsc_id] = Document(
                text=f"""
                Lista de Dictamenes o Discrepancias:
                - ID : {dsc_id}
                - Codigo Discrepancia o Dictamen: {record['discrepancia']}
                - Descripción: {record['descripcion']}
                - Fecha: {record['fecha']}
                - Materia: {record['materia']}
                - SubMateria: {record['submateria']}
                - Fecha de cierre: {record['fechafinaliza']}
                - Estado: {record['estado']}
                - Razón de cierre de discrepancia: {record['razoncierre']}
                - Documento de cierre: {record['doctofinaliza']}
                - Año de creación: {record['disc_year']}
                """,
                extra_info={
                        "file_path": "",
                        "tipo": "dictamen",
                        "id": dsc_id,
                        "codigo_discrepancia": record['discrepancia'],
                        "año_de_inicio": dsc_year,
                        "fecha_creacion": record['fecha'],
                        "fecha_finalizacion": record['fechafinaliza'],
                        "materia": dsc_materia,
                        "submateria": dsc_submateria,
                        "estado": record['estado'],
                        "razoncierre": record['razoncierre'],
                        }

            )

    return (
        [documents_tot] + list(dictamen_docs.values())
    )

def getretriever_recursivo(all_documents, chunk_size):
    # Retriever recursivo

    node_parser = SentenceSplitter(chunk_size=int(chunk_size))

    base_nodes = node_parser.get_nodes_from_documents(all_documents)
    # set node ids to be a constant
    for idx, node in enumerate(base_nodes):
        node.id_ = str(uuid.uuid4())

    #sub_chunk_sizes = [128,256, 512]
    sub_chunk_sizes = [256, 512]
    sub_node_parsers = [
        SentenceSplitter(chunk_size=c, chunk_overlap=20) for c in sub_chunk_sizes
    ]

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

    all_nodes_dict = {n.node_id: n for n in all_nodes}

    return all_nodes, all_nodes_dict

def getprompt(function_calls, cnt):
    prompt = ""
    system_prompt = os.getenv("SYSTEM_PROMPT")

    match function_calls:
        case "personas2pjud":
            prompt = f"""Eres un asistente especializado en responder preguntas basadas en la información proporcionada sobre personas jurídicas, sus representantes y discrepancias.

                    Reglas de comportamiento:
                    Evaluación del contexto:
                    Si {len(cnt)} es igual a 0, responde de manera clara indicando que no tienes suficiente información para procesar la solicitud. Proporciona una guía para que el usuario corrija la entrada:

                    "No tengo suficiente contexto para responder. Asegúrate de proporcionar parámetros válidos como el RUT, nombre de la persona jurídica, representante, correo, materia, o submateria."
                    Acceso a documentos:
                    Puedes utilizar la información de los siguientes documentos:

                    Persona Jurídica: Incluye datos como nombre, dirección, teléfono, entre otros.
                    Representante: Detalla nombres, teléfonos y otros datos de representantes legales.
                    Discrepancia: Describe discrepancias, materias relacionadas, y códigos únicos.
                    Restricciones en la salida:

                    No incluyas la columna Discrepancia_ID. Utiliza Discrepancia_codigo en su lugar, ya que es más comprensible para el usuario.

                    Comportamiento en caso de parámetros incorrectos:
                    Si detectas que alguno de los parámetros está vacío o es inválido:

                    Menciona específicamente cuál es el parámetro faltante o incorrecto.
                    Proporciona un ejemplo claro de cómo debe ingresarse:
                    Por ejemplo: "El parámetro persona_juridica_rut es obligatorio y debe ser un texto con el formato válido (e.g., '123456789', ojo sin puntos ni guión)."
                    Respuesta:
                    Responde a las consultas basándote exclusivamente en los documentos mencionados. Si la información proporcionada es insuficiente, sé explícito y guía al usuario para proporcionar los datos faltantes."""
        case "cantidad_discrepancias":
            prompt = """
                    Eres un asistente que responde preguntas sobre discrepancias.
                    Si la consulta está relacionada con años, materias o estados, utiliza los datos disponibles en los documentos.
                    Prioriza documentos con coincidencias explícitas en el campo 'año_inicio'.
                    Proporciona un conteo exacto de discrepancias cuando se te pregunte por años o períodos.

                    Nunca respondas con preguntas del estilo ¿puedo ayudarte? o ¿prefieres que te de esta o tal información?
                    """
        case "sin registros":
            prompt = """
                    Parece que no se encontraron datos relevantes o no se activó la función requerida. Por favor, verifica tu consulta y utiliza el formato adecuado según lo que deseas buscar. Aquí tienes una guía:
                    1. **Para activar las funciones del STDE:**
                    - Es obligatorio iniciar tu consulta con el literal **"Busca en el STDE"**.
                    - Ejemplos:
                        - "Busca en el STDE cuántas discrepancias hubo en el año 2023."
                        - "Busca en el STDE los datos de la persona jurídica con RUT 123456789 (ojo que es sin puntos ni guión)."

                    2. **Para realizar búsquedas normales de dictamenes en la base de datos vectorial:**
                    - Asegúrate de usar el formato correcto para identificar discrepancias o dictámenes. El formato debe ser **XX-YYYY**, donde:
                        - `XX` es un número o código de discrepancia.
                        - `YYYY` es el año.
                    - Ejemplos:
                        - "Dame un resumen del dictamen 01-2023."
                        - "¿Qué información tienes sobre la discrepancia 15-2022?"

                    Si no estás seguro de cómo estructurar tu consulta, aquí tienes una sugerencia:
                    - Inicia con "Busca en el STDE" si necesitas datos específicos del sistema.
                    - Usa el formato **XX-YYYY** si buscas datos de discrepancias o dictámenes en la base vectorial.

                    Proporciona más detalles en tu consulta para mejorar la búsqueda y los resultados.
                    """
        case _:
            prompt = system_prompt

    return prompt

def getqueryfusion(retriever_chunk, base_retriever, num_queries):
    return QueryFusionRetriever(
        [retriever_chunk, base_retriever],
        retriever_weights=[0.6, 0.4],
        similarity_top_k=10,
        num_queries=num_queries,  # set this to 1 to disable query generation
        mode="relative_score",
        use_async=True,
        verbose=True,  # true para que sea verboso
    )

# Completion para procesar funtion_call
def get_openai_response(formatted_messages, tipo):
    functions=None
    match tipo:
        case "buscastde":
            functions = [
                {
                    "name": "cantidad_discrepancias",
                    "description": "Busca en el STDE y obtiene la cantidad de discrepancias por año.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "year": {
                                "type": "integer",
                                "description": "Año de las discrepancias",
                            }
                        },
                        "required": ["year"],
                    },
                },
                {
                    "name": "personas2pjud",
                    "description": "Busca en el STDE y Obtiene los datos de usuarios y personas juridicas. Consulta desde STDE",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "persona_juridica_rut": {
                                "type": "string",
                                "description": "Rut o codigo legal de Empresa o persona juridica"
                            },
                            "persona_juridica_nombre": {
                                "type": "string",
                                "description": "Empresa o persona juridica"
                            },
                            "usuario_representante_nombre": {
                                "type": "string",
                                "description": "usuario representante legal de la persona juridica"
                            },
                            "mail": {
                                "type": "string",
                                "description": "mail o correo a consultar"
                            },
                            "materia": {
                                "type": "string",
                                "description": "materia en que esta asociada una discrepancia o dictamen"
                            },
                                "submateria": {
                                "type": "string",
                                "description": "submateria en que esta asociada una discrepancia o dictamen"
                            },
                            "estado_discrepancia": {
                                "type": "string",
                                "description": "Estado de discrepancia, activa o abierta"
                            },

                        },
                        #"required": ["type", "year"]
                    }
                }
            ]
        case "bdvectorial":
            functions = [
                {
                    "name": "discrepancias_dictamen",
                    "description": "busca datos de la discrepancia o dictamen con formato 99-YYYY donde XX son el número y YYYY es el año de creación",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "discrepancia": {
                                "type": "string",
                                "description": "código de discrepancia formato formato 99-YYYY donde XX son el número y YYYY es el año de creación",
                            },
                            "dictamen": {
                                "type": "string",
                                "description": "código de discrepancia formato formato 99-YYYY donde XX son el número y YYYY es el año de creación",
                            }
                        }
                    },
                },]
        case _:
            functions=None

    return openai.chat.completions.create(
        model="gpt-4",
        messages=formatted_messages,
        functions=functions )

def handle_no_context_response(top_k):

    empty_prompt = getprompt("sin registros",0)
    print (empty_prompt)
    #prompt_doc = Document(text="", extra_info={"id": 1})
    prompt_doc = Document(text= empty_prompt, extra_info={"id": 1})

    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents([prompt_doc])

    index = GPTVectorStoreIndex(nodes)

    return index.as_chat_engine(
        similarity_top_k=int(top_k),
        system_prompt=empty_prompt,
        chat_mode="context",
    )




def load_data_dict():
    global ALL_NODES_DICTIONARY  # Declaración global
    if ALL_NODES_DICTIONARY is None:
        STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
        # Leer el diccionario desde el archivo
        with open(f"{STORAGE_DIR}/dicnodes.pkl", 'rb') as archivo:
            ALL_NODES_DICTIONARY = pickle.load(archivo)

def unload_data_dict():
    global ALL_NODES_DICTIONARY
    if ALL_NODES_DICTIONARY: #and LAST_ACCESS_TIME and (time.time() - LAST_ACCESS_TIME > CACHE_EXPIRY):
        ALL_NODES_DICTIONARY = None
        gc.collect()  # Liberar memoria de manera explícita

# def unload_data_dict_if_inactive():
#     """
#     Libera el diccionario si no se usa durante CACHE_EXPIRY segundos.
#     """
#     global ALL_NODES_DICTIONARY, LAST_ACCESS_TIME
#     if ALL_NODES_DICTIONARY and LAST_ACCESS_TIME and (time.time() - LAST_ACCESS_TIME > CACHE_EXPIRY):
#         ALL_NODES_DICTIONARY = None

# def get_chat_engine_tools(filters=None, query : str = ""):
#     print("chat_engine tools!!!")

#     #system_prompt = os.getenv("SYSTEM_PROMPT")
#     top_k = os.getenv("TOP_K", "3")
#     tools = []

#     vector_store_info = VectorStoreInfo(
#         content_info="Discrepancias",
#         metadata_info=[
#             MetadataInfo(
#                 name="dictamen",
#                 type="str",
#                 description=("nombre o codigo de dictamen/discrepancia, si no existe no la consideres")
#             ),
#             MetadataInfo(
#                 name="discrepancia",
#                 type="str",
#                 description=("nombre o codigo de discrepancia/dictamen, si no existe no la consideres")
#             )
#         ]
#     )

#     filtro = GetFiltersPrompt(vector_store_info=vector_store_info)
#     filters_ = filtro.generate_filters(query)


#     # # Carga Resumenes desde JSon
#     # reader = JSONReader(
#     #     levels_back=0,
#     #     ensure_ascii=True
#     # )


#     # from llama_index.core import load_index_from_storage
#     # from llama_index.core import StorageContext

#     # # rebuild storage context
#     # storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
#     # doc_summary_index = load_index_from_storage(storage_context)

#     # query_enginesummary = doc_summary_index.as_query_engine(
#     #     response_mode="tree_summarize", use_async=True, system_prompt=system_prompt
#     #     )

#     # query_engine_tool_summary = QueryEngineTool.from_defaults(
#     #     query_engine=query_enginesummary,
#     #     name='summary',
#     #     description="Resumenes de dictamen"
#     #                )

#     # response = query_enginesummary.query("Dame detalles del dictamen 68-2023")
#     # print (f"desde summary: {response}")

#     #tools.append(query_engine_tool_summary)

#     # # Load data from JSON file
#     # documentsJson = reader.load_data(input_file="config/metadata-panel.json")

#     # from llama_index.core.indices import VectorStoreIndex

#     # index_json = VectorStoreIndex.from_documents(documentsJson)

#     # query_engine_json = index_json.as_chat_engine(
#     #         similarity_top_k=int(top_k)
#     #     )

#     # query_engine_tool_json = QueryEngineTool.from_defaults(
#     #     query_engine=query_engine_json,
#     #     name='summary_jSon',
#     #     description="Resumen discrepancia"
#     #                )


#     # Add query tool if index exists
#     index = get_index()
#     if index is not None:
#         vector_retriever_chunk = index.as_retriever(similarity_top_k=int(top_k))

#         # chat_engine = index.as_query_engine(
#         #     similarity_top_k=int(top_k), filters=filters, system_prompt=system_prompt
#         # )
#         # query_engine_tool = QueryEngineTool.from_defaults(
#         #     query_engine=chat_engine,
#         #     name='store1',
#         #     description="Almacen de discrepancias y sus dictamenes"
#         #            )

#     index_summary = get_index_summary("QDRANT_COLLECTION_SUMMARY")
#     if index_summary is not None:
#         retriever_summary = index_summary.as_retriever(similarity_top_k=int(top_k))

#     if len(ALL_NODES_DICTIONARY) == 0:
#         print("El diccionario está vacío")

#     retriever_chunk_recursivo = RecursiveRetriever(
#         "vector",
#         retriever_dict={"vector": vector_retriever_chunk},
#         node_dict=ALL_NODES_DICTIONARY,
#         verbose=True,
#     )

#     # retriever_chunk_recursivo = RetrieverQueryEngine.from_args(
#     #    retriever_chunk, filters=filters_
#     # )

#         #tools.append(query_engine_tool)

#         # query_engine = index.as_query_engine(
#         #     similarity_top_k=int(top_k), filters=filters_
#         # )
#         # query_engine_tool2 = QueryEngineTool.from_defaults(
#         #     query_engine=query_engine,
#         #     name='store2',
#         #     description="Almacen de discrepancias y sus dictamenes"
#         #         )
#         #tools.append(query_engine_tool2)


#     # Para utilizar filtros automáticos del VectorIndexAutoRetriever
#     # NOTE: the "set top-k to 10000" is a hack to return all data.
#     # retriever_autoretriever = VectorIndexAutoRetriever(
#     #     index,
#     #     vector_store_info=vector_store_info,
#     #     max_top_k=10000,
#     #     similarity_top_k=int(top_k),
#     # )

#     # QUERY_GEN_PROMPT = (
#     #     "You are a helpful assistant that generates multiple search queries based on a "
#     #     "single input query. Generate {num_queries} search queries, one on each line, "
#     #     "related to the following input query:\n"
#     #     "Query: {query}\n"
#     #     "Queries:\n"
#     # )

#     list_tool = QueryEngineTool.from_defaults(
#         query_engine=RetrieverQueryEngine.from_args(retriever_summary),
#         description="Usalo para extraer resumen o extracto de una discrepancia o dictamen",
#     )
#     vector_tool = QueryEngineTool.from_defaults(
#         query_engine=RetrieverQueryEngine.from_args(retriever_chunk_recursivo),
#         description=(
#             "Usalo para detalles de la discrepancia o dictamen"
#             " cuales fueron las materias, los argumementos usados y el dictamen o veredicto"
#         ),
# )

#     from llama_index.core.objects import ObjectIndex
#     from llama_index.core import VectorStoreIndex

#     obj_index = ObjectIndex.from_objects(
#         [list_tool, vector_tool],
#         index_cls=VectorStoreIndex,
#     )

#     # retrieverfusion = QueryFusionRetriever(
#     #     [
#     #         #index.as_retriever(similarity_top_k=int(top_k) ), #, vector_store_query_mode="hybrid",
#     #         retriever_summary,
#     #         retriever_chunk_recursivo,
#     #         retriever_autoretriever
#     #     ],
#     #     num_queries=2,
#     #     mode="reciprocal_rerank",
#     #     use_async=True,
#     #     retriever_weights=[0.6, 0.4],
#     #     #query_gen_prompt=QUERY_GEN_PROMPT
#     # )

#     # Reranker Cohere
#     # from llama_index.core import QueryBundle
#     # query_bundle = QueryBundle(query)

#     # retrieved_nodes = retrieve_reranker(retrieverfusion, query_bundle)
#     # # for node in retrieved_nodes:
#     # #     print(node)

#     # reranker= CohereRerank(api_key=api_key_cohere, top_n=10)
#     # retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)

#     from llama_index.core.query_engine import ToolRetrieverRouterQueryEngine

#     query_engine = ToolRetrieverRouterQueryEngine(obj_index.as_retriever())

#     return query_engine #retrieved_nodes


# async def retrieve_reranker(retriever, query_bundle):
#     retrieved_nodes = await retriever.retrieve(query_bundle)

#     return retrieved_nodes
    #retriever = RetrieverQueryEngine.from_args(retrieved_nodes)

    # Reranker
    # retrieved_nodes = retriever.aretrieve(query_bundle)
    # reranker = RankGPTRerank(
    #          top_n=10,
    #          llm=Settings.llm,
    #          verbose=True,
    #      )

    # retrieved_nodes = reranker.postprocess_nodes(
    #      retrieved_nodes, query_bundle
    # )

    # Para pasa un retriever a query_engine y chat_engine!!! = RetrieverQueryEngine.from_args(retriever)

    # query_engine_tool = QueryEngineTool.from_defaults(
    #     query_engine=RetrieverQueryEngine.from_args(retrieverfusion),
    #     name='store1',
    #     description="Almacen de discrepancias y sus dictamenes"
    #             )
    # tools.append(query_engine_tool)

    #openai_step_engine = OpenAIAgentWorker.from_tools(tools, llm=Settings.llm, system_prompt=system_prompt, verbose=True)

    # Add additional tools
    #tools += ToolFactory.from_env()



# def get_chat_engine_retriever(retrieved_nodes):
#     system_prompt = os.getenv("SYSTEM_PROMPT")
#     top_k = os.getenv("TOP_K", "3")
#     api_key_cohere = os.getenv("COHERE_API_KEY")
#     tools = []

#     #from llama_index.core.postprocessor import LongContextReorder
#     #postprocessorLongContext = LongContextReorder()
#     #cohere_rerank = CohereRerank(api_key=api_key_cohere, top_n=10)

#     # engine = CondensePlusContextChatEngine.from_defaults(
#     #         retriever=retrieved_nodes, #retrieverfusion,
#     #         similarity_top_k=int(top_k),
#     #         system_prompt=system_prompt,
#     #         node_postprocessors=[cohere_rerank, postprocessorLongContext],
#     #   )

#     # return engine #index.as_query_engine()

#     return RetrieverQueryEngine.from_args(retrieved_nodes)

#     # query_engine_tool = QueryEngineTool.from_defaults(
#     #     query_engine=RetrieverQueryEngine.from_args(retrieved_nodes),
#     #     name='store1',
#     #     description="Almacen de discrepancias y sus dictamenes"
#     #             )
#     # tools.append(query_engine_tool)


#     # return AgentRunner.from_llm(
#     #      #agent_worker=openai_step_engine,
#     #      tools=tools,
#     #      system_prompt=system_prompt,
#     #      verbose=True,
#     #  )


#     # query_engine_tools = [
#     #     QueryEngineTool(
#     #         query_engine=engine,
#     #         metadata=ToolMetadata(
#     #             name="qlora_paper",
#     #             description="Efficient Finetuning of Quantized LLMs",
#     #         ),
#     #     ),
#     # ]

#     # from llama_index.core.query_engine import SubQuestionQueryEngine
#     # query_engine = SubQuestionQueryEngine.from_defaults(
#     #     query_engine_tools=query_engine_tools,
#     #     use_async=True,
#     # )


#     # return engine

#     # return AgentRunner.from_llm(
#     #      #agent_worker=openai_step_engine,
#     #      tools=tools,
#     #      system_prompt=system_prompt,
#     #      verbose=True,
#     #  )




# def get_get_discrepancies(tipo:str, year:int ):

#     return

# def get_chat_engine_agente(filters=None, query : str = "") :
#     system_prompt = os.getenv("SYSTEM_PROMPT")

#     print ("chat_engine_agente")
#     top_k = os.getenv("TOP_K", 3)
#     tools= []

#     index = get_index()
#     if index is None:
#         raise HTTPException(
#             status_code=500,
#             detail=str(
#                 "StorageContext is empty - call 'poetry run generate' to generate the storage first"
#             ),
#         )

#     # Store recursivo
#     if index is not None:
#         vector_retriever_chunk = index.as_retriever(similarity_top_k=int(top_k))

#     # if len(ALL_NODES_DICTIONARY) == 0:
#     #     print("El diccionario está vacío")

#     # retriever_chunk_recursivo = RecursiveRetriever(
#     #     "vector",
#     #     retriever_dict={"vector": vector_retriever_chunk},
#     #     node_dict=ALL_NODES_DICTIONARY,
#     #     verbose=True,
#     # )

#     # query_engine_recursive = QueryEngineTool.from_defaults(
#     #     query_engine=retriever_chunk_recursivo,
#     #     name='Recursive',
#     #     description="Datos de discrepancias y sus dictamenes con datos de busqueda recursiva"
#     #         )
#     # tools.append(query_engine_recursive)

#     # Store Normal (con chunking)
#     query_engine_tool = QueryEngineTool.from_defaults(
#         query_engine=index.as_query_engine(), #RetrieverQueryEngine.from_args(index),
#         name='Store',
#         description="Almacen de discrepancias con datos directos"
#                 )

#     tools.append(query_engine_tool)

#     # Store summary (con chunking)
#     index_summary = get_index_summary("QDRANT_COLLECTION_SUMMARY")
#     if index_summary is not None:
#         retriever_summary = index_summary.as_query_engine(similarity_top_k=int(top_k))

#     query_engine_summary = QueryEngineTool.from_defaults(
#         query_engine=retriever_summary,
#         name='Summary',
#         description="Usalo para hacer resumen o extractos de discrepancia o dictammen"
#             )
#     tools.append(query_engine_summary)

#     # BD SQL STDE
#     # queryEngineBD = get_BD()

#     # query_engine_BD = QueryEngineTool.from_defaults(
#     #     query_engine=queryEngineBD,
#     #     name='BdSTDE',
#     #     description="Usalo para consultas a BD acerca de cantidad de discrepancias o dictamenes"
#     #         )
#     # tools.append(query_engine_BD)

#     # retriever_chunk_recursivo = RetrieverQueryEngine.from_args(
#     #    retriever_chunk, filters=filters_
#     # )

#         #tools.append(query_engine_tool)

#         # query_engine = index.as_query_engine(
#         #     similarity_top_k=int(top_k), filters=filters_
#         # )
#         # query_engine_tool2 = QueryEngineTool.from_defaults(
#         #     query_engine=query_engine,
#         #     name='store2',
#         #     description="Almacen de discrepancias y sus dictamenes"
#         #         )
#         #tools.append(query_engine_tool2)

#     #from llama_index.chat_engine.condense_question import CondenseQuestionChatEngine
#     from llama_index.core.chat_engine import CondenseQuestionChatEngine
#     agent = AgentRunner.from_llm(
#         llm=Settings.llm,
#         tools=tools,
#         #system_prompt=system_prompt,
#         verbose=True,
#     )

#     return CondenseQuestionChatEngine.from_defaults(
#         query_engine=agent,
#         verbose=True,
#     )



# 1. Define la función para conectarte a la base de datos
# def get_BD():
#     from sqlalchemy.orm import sessionmaker

#     # Obtener las URIs de las bases de datos
#     URI_BD_LOCAL = os.getenv("URI_BD_LOCAL", "mysql+pymysql://user:pass@localhost:3306/mydb")

#     # Crear la conexión al motor
#     engine = create_engine(URI_BD_LOCAL)
#     Session = sessionmaker(bind=engine)
#     return Session()