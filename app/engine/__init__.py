import os
from app.engine.index import get_index, get_index_summary
from app.node_dictionary import ALL_NODES_DICTIONARY

from fastapi import HTTPException
# from app.engine.vectordb import get_vector_index_store
# from app.engine.vectordb import get_vector_store
#from app.engine.generate import get_doc_store
from app.engine.getfiltersLLM import GetFiltersPrompt

# Nuevas
# from llama_index.postprocessor.cohere_rerank import CohereRerank
# from app.engine.loaders import get_metadata
# from llama_index.core.postprocessor import LongContextReorder
# from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor import PrevNextNodePostprocessor

#from llama_index.core.postprocessor import AutoPrevNextNodePostprocessor
#from llama_index.core.postprocessor import LLMRerank

#from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter

from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo

# from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
# from llama_index.core.postprocessor import LLMRerank
# #from llama_index.postprocessor.rankllm_rerank import RankLLMRerank
# from llama_index.core.postprocessor import LLMRerank
# from llama_index.core.retrievers import VectorIndexRetriever

#from llama_index.retrievers.bm25 import BM25Retriever
# from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.indices.query.query_transform.base import (
#      StepDecomposeQueryTransform,
# )
# from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.core.settings import Settings
# from llama_index.core.node_parser import SimpleNodeParser

#from llama_index.core.chat_engine import CondenseQuestionChatEngine
#from llama_index.core import PromptTemplate
#from llama_index.core.llms import ChatMessage, MessageRole

from llama_index.core.postprocessor import (
    PrevNextNodePostprocessor,
    FixedRecencyPostprocessor
)

# from llama_index.core.settings import Settings
from llama_index.core.agent import AgentRunner
from llama_index.core.tools.query_engine import QueryEngineTool, ToolMetadata
#from app.engine.tools import ToolFactory

# from llama_index.core.indices.struct_store import JSONQueryEngine
# from llama_index.readers.json import JSONReader
# from llama_index.core import Document

# from llama_index.agent.openai import OpenAIAgentWorker

# from llama_index.core.agent import (
#     StructuredPlannerAgent,
#     FunctionCallingAgentWorker,
#     ReActAgentWorker,
# )
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core import get_response_synthesizer

from llama_index.core.chat_engine import CondensePlusContextChatEngine
#from llama_index.core.retrievers import SummaryIndexLLMRetriever


#from llama_index.core.indices.query.query_transform import HyDEQueryTransform
#from llama_index.core.query_engine.transform_query_engine import TransformQueryEngine



STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")



def get_query_engine(filters=None, query : str = ""):
    system_prompt = os.getenv("SYSTEM_PROMPT")
    api_key_cohere = os.getenv("COHERE_API_KEY")

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



def get_chat_engine_tools(filters=None, query : str = ""):
    print("chat_engine tools!!!")

    #system_prompt = os.getenv("SYSTEM_PROMPT")
    top_k = os.getenv("TOP_K", "3")
    tools = []

    vector_store_info = VectorStoreInfo(
        content_info="Discrepancias",
        metadata_info=[
            MetadataInfo(
                name="dictamen",
                type="str",
                description=("nombre o codigo de dictamen/discrepancia, si no existe no la consideres")
            ),
            MetadataInfo(
                name="discrepancia",
                type="str",
                description=("nombre o codigo de discrepancia/dictamen, si no existe no la consideres")
            )
        ]
    )

    filtro = GetFiltersPrompt(vector_store_info=vector_store_info)
    filters_ = filtro.generate_filters(query)


    # # Carga Resumenes desde JSon
    # reader = JSONReader(
    #     levels_back=0,
    #     ensure_ascii=True
    # )


    # from llama_index.core import load_index_from_storage
    # from llama_index.core import StorageContext

    # # rebuild storage context
    # storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    # doc_summary_index = load_index_from_storage(storage_context)

    # query_enginesummary = doc_summary_index.as_query_engine(
    #     response_mode="tree_summarize", use_async=True, system_prompt=system_prompt
    #     )

    # query_engine_tool_summary = QueryEngineTool.from_defaults(
    #     query_engine=query_enginesummary,
    #     name='summary',
    #     description="Resumenes de dictamen"
    #                )

    # response = query_enginesummary.query("Dame detalles del dictamen 68-2023")
    # print (f"desde summary: {response}")

    #tools.append(query_engine_tool_summary)

    # # Load data from JSON file
    # documentsJson = reader.load_data(input_file="config/metadata-panel.json")

    # from llama_index.core.indices import VectorStoreIndex

    # index_json = VectorStoreIndex.from_documents(documentsJson)

    # query_engine_json = index_json.as_chat_engine(
    #         similarity_top_k=int(top_k)
    #     )

    # query_engine_tool_json = QueryEngineTool.from_defaults(
    #     query_engine=query_engine_json,
    #     name='summary_jSon',
    #     description="Resumen discrepancia"
    #                )


    # Add query tool if index exists
    index = get_index()
    if index is not None:
        vector_retriever_chunk = index.as_retriever(similarity_top_k=int(top_k))

        # chat_engine = index.as_query_engine(
        #     similarity_top_k=int(top_k), filters=filters, system_prompt=system_prompt
        # )
        # query_engine_tool = QueryEngineTool.from_defaults(
        #     query_engine=chat_engine,
        #     name='store1',
        #     description="Almacen de discrepancias y sus dictamenes"
        #            )

    index_summary = get_index_summary("QDRANT_COLLECTION_SUMMARY")
    if index_summary is not None:
        retriever_summary = index_summary.as_retriever(similarity_top_k=int(top_k))

    if len(ALL_NODES_DICTIONARY) == 0:
        print("El diccionario está vacío")

    retriever_chunk_recursivo = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever_chunk},
        node_dict=ALL_NODES_DICTIONARY,
        verbose=True,
    )

    # retriever_chunk_recursivo = RetrieverQueryEngine.from_args(
    #    retriever_chunk, filters=filters_
    # )

        #tools.append(query_engine_tool)

        # query_engine = index.as_query_engine(
        #     similarity_top_k=int(top_k), filters=filters_
        # )
        # query_engine_tool2 = QueryEngineTool.from_defaults(
        #     query_engine=query_engine,
        #     name='store2',
        #     description="Almacen de discrepancias y sus dictamenes"
        #         )
        #tools.append(query_engine_tool2)


    # Para utilizar filtros automáticos del VectorIndexAutoRetriever
    # NOTE: the "set top-k to 10000" is a hack to return all data.
    # retriever_autoretriever = VectorIndexAutoRetriever(
    #     index,
    #     vector_store_info=vector_store_info,
    #     max_top_k=10000,
    #     similarity_top_k=int(top_k),
    # )

    # QUERY_GEN_PROMPT = (
    #     "You are a helpful assistant that generates multiple search queries based on a "
    #     "single input query. Generate {num_queries} search queries, one on each line, "
    #     "related to the following input query:\n"
    #     "Query: {query}\n"
    #     "Queries:\n"
    # )

    list_tool = QueryEngineTool.from_defaults(
        query_engine=RetrieverQueryEngine.from_args(retriever_summary),
        description="Usalo para extraer resumen o extracto de una discrepancia o dictamen",
    )
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=RetrieverQueryEngine.from_args(retriever_chunk_recursivo),
        description=(
            "Usalo para detalles de la discrepancia o dictamen"
            " cuales fueron las materias, los argumementos usados y el dictamen o veredicto"
        ),
)

    from llama_index.core.objects import ObjectIndex
    from llama_index.core import VectorStoreIndex

    obj_index = ObjectIndex.from_objects(
        [list_tool, vector_tool],
        index_cls=VectorStoreIndex,
    )

    # retrieverfusion = QueryFusionRetriever(
    #     [
    #         #index.as_retriever(similarity_top_k=int(top_k) ), #, vector_store_query_mode="hybrid",
    #         retriever_summary,
    #         retriever_chunk_recursivo,
    #         retriever_autoretriever
    #     ],
    #     num_queries=2,
    #     mode="reciprocal_rerank",
    #     use_async=True,
    #     retriever_weights=[0.6, 0.4],
    #     #query_gen_prompt=QUERY_GEN_PROMPT
    # )

    # Reranker Cohere
    # from llama_index.core import QueryBundle
    # query_bundle = QueryBundle(query)

    # retrieved_nodes = retrieve_reranker(retrieverfusion, query_bundle)
    # # for node in retrieved_nodes:
    # #     print(node)

    # reranker= CohereRerank(api_key=api_key_cohere, top_n=10)
    # retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)

    from llama_index.core.query_engine import ToolRetrieverRouterQueryEngine
    query_engine = ToolRetrieverRouterQueryEngine(obj_index.as_retriever())

    return query_engine #retrieved_nodes


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



def get_chat_engine_retriever(retrieved_nodes):
    system_prompt = os.getenv("SYSTEM_PROMPT")
    top_k = os.getenv("TOP_K", "3")
    api_key_cohere = os.getenv("COHERE_API_KEY")
    tools = []

    #from llama_index.core.postprocessor import LongContextReorder
    #postprocessorLongContext = LongContextReorder()
    #cohere_rerank = CohereRerank(api_key=api_key_cohere, top_n=10)

    # engine = CondensePlusContextChatEngine.from_defaults(
    #         retriever=retrieved_nodes, #retrieverfusion,
    #         similarity_top_k=int(top_k),
    #         system_prompt=system_prompt,
    #         node_postprocessors=[cohere_rerank, postprocessorLongContext],
    #   )

    # return engine #index.as_query_engine()

    return RetrieverQueryEngine.from_args(retrieved_nodes)

    # query_engine_tool = QueryEngineTool.from_defaults(
    #     query_engine=RetrieverQueryEngine.from_args(retrieved_nodes),
    #     name='store1',
    #     description="Almacen de discrepancias y sus dictamenes"
    #             )
    # tools.append(query_engine_tool)


    # return AgentRunner.from_llm(
    #      #agent_worker=openai_step_engine,
    #      tools=tools,
    #      system_prompt=system_prompt,
    #      verbose=True,
    #  )


    # query_engine_tools = [
    #     QueryEngineTool(
    #         query_engine=engine,
    #         metadata=ToolMetadata(
    #             name="qlora_paper",
    #             description="Efficient Finetuning of Quantized LLMs",
    #         ),
    #     ),
    # ]

    # from llama_index.core.query_engine import SubQuestionQueryEngine
    # query_engine = SubQuestionQueryEngine.from_defaults(
    #     query_engine_tools=query_engine_tools,
    #     use_async=True,
    # )


    # return engine

    # return AgentRunner.from_llm(
    #      #agent_worker=openai_step_engine,
    #      tools=tools,
    #      system_prompt=system_prompt,
    #      verbose=True,
    #  )



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



def get_chat_engine2(filters=None, query : str = "") :
    system_prompt = os.getenv("SYSTEM_PROMPT")
    api_key_cohere = os.getenv("COHERE_API_KEY")

    # import nest_asyncio
    # nest_asyncio.apply()

    print ("Chat engine 2!!!!")

    top_k = os.getenv("TOP_K", 3)

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

    if len(ALL_NODES_DICTIONARY) == 0:
        print("El diccionario está vacío")

    retriever_chunk_recursivo = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever_chunk},
        node_dict=ALL_NODES_DICTIONARY,
        verbose=True,
    )


    retriever_summary = get_index_summary("QDRANT_COLLECTION_SUMMARY")
    if retriever_summary is not None:
        retriever_summary = retriever_summary.as_query_engine(similarity_top_k=int(top_k))

    vector_store_info = VectorStoreInfo(
        content_info="Discrepancias",
        metadata_info=[
            MetadataInfo(
                name="dictamen",
                type="str",
                description=("nombre o codigo de dictamen/discrepancia, si no existe no la consideres")
            ),
            MetadataInfo(
                name="discrepancia",
                type="str",
                description=("nombre o codigo de discrepancia/dictamen, si no existe no la consideres")
            ),
            MetadataInfo(
                 name="materias",
                 type="str",
                 description=("materias tratadas en las discrepancias/dictamenes")
            )
        ]
    )


# NOTE: the "set top-k to 10000" is a hack to return all data.
# Right now auto-retrieval will always return a fixed top-k, there's a TODO to allow it to be None
# to fetch all data.
# So it's theoretically possible to have the LLM infer a None top-k value.

# retriever = VectorIndexAutoRetriever(
#     index,
#     vector_store_info=vector_store_info,
#     llm=llm,
#     callback_manager=callback_manager,
#     max_top_k=10000,
# )

#    if filters is None or 1==1:
    filtro = GetFiltersPrompt(vector_store_info=vector_store_info)
    filters_ = filtro.generate_filters(query)
    print (f"filtros aplicados: {filters_}")

    # _current_filters = MetadataFilters(
    #             filters=[*filters],
    #             condition=FilterCondition.OR
    #         )
    # _current_filters = dict(filters_)
    # filt=MetadataFilters.From_dict(filters_)

    # print (f"filtros aplicados fuera: {filt}")

    # PostProcessors
    postprocessor = PrevNextNodePostprocessor(
        docstore=index.docstore,
        num_nodes=1,  # number of nodes to fetch when looking forawrds or backwards
        mode="next",  # can be either 'next', 'previous', or 'both'
    )

    # postprocessorDate = FixedRecencyPostprocessor(
    #     tok_k=1, date_key="fecha_presentacion"  # the key in the metadata to find the date
    # )

    from llama_index.core.postprocessor import LongContextReorder

    postprocessorLongContext = LongContextReorder()

    # retrieversummary = SummaryIndexLLMRetriever(
    #     index=index,
    #     choice_batch_size=5,
    # )

    # bm25_retriever = BM25Retriever.from_defaults(
    #     docstore=index.docstore, similarity_top_k=2
    # )

    retriever = QueryFusionRetriever(
        [retriever_chunk_recursivo, retriever_summary],
        #[retriever_summary, retriever_chunk_recursivo, vector_retriever_chunk ],
        retriever_weights=[0.6, 0.4],
        similarity_top_k=10,
        num_queries=1,  # set this to 1 to disable query generation
        mode="relative_score",
        use_async=True,
        verbose=True,
    )

     # Hace el Engine
    # step_decompose_transform = StepDecomposeQueryTransform(verbose=True)
    # retriever = MultiStepQueryEngine(
    #        retriever, query_transform=step_decompose_transform
    #     )

    return CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            similarity_top_k=int(top_k),
            system_prompt=system_prompt,
            filters=filters_,
            node_postprocessors=[postprocessorLongContext],
    )



def get_chat_engine_agente(filters=None, query : str = "") :
    system_prompt = os.getenv("SYSTEM_PROMPT")

    print ("chat_engine_agente")
    top_k = os.getenv("TOP_K", 3)
    tools= []

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

    # if len(ALL_NODES_DICTIONARY) == 0:
    #     print("El diccionario está vacío")

    # retriever_chunk_recursivo = RecursiveRetriever(
    #     "vector",
    #     retriever_dict={"vector": vector_retriever_chunk},
    #     node_dict=ALL_NODES_DICTIONARY,
    #     verbose=True,
    # )

    # query_engine_recursive = QueryEngineTool.from_defaults(
    #     query_engine=retriever_chunk_recursivo,
    #     name='Recursive',
    #     description="Datos de discrepancias y sus dictamenes con datos de busqueda recursiva"
    #         )
    # tools.append(query_engine_recursive)

    # Store Normal (con chunking)
    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(), #RetrieverQueryEngine.from_args(index),
        name='Store',
        description="Almacen de discrepancias con datos directos"
                )

    tools.append(query_engine_tool)

    # Store summary (con chunking)
    index_summary = get_index_summary("QDRANT_COLLECTION_SUMMARY")
    if index_summary is not None:
        retriever_summary = index_summary.as_query_engine(similarity_top_k=int(top_k))

    query_engine_summary = QueryEngineTool.from_defaults(
        query_engine=retriever_summary,
        name='Summary',
        description="Usalo para hacer resumen o extractos de discrepancia o dictammen"
            )
    tools.append(query_engine_summary)

    # BD SQL STDE
    # queryEngineBD = get_BD()

    # query_engine_BD = QueryEngineTool.from_defaults(
    #     query_engine=queryEngineBD,
    #     name='BdSTDE',
    #     description="Usalo para consultas a BD acerca de cantidad de discrepancias o dictamenes"
    #         )
    # tools.append(query_engine_BD)

    # retriever_chunk_recursivo = RetrieverQueryEngine.from_args(
    #    retriever_chunk, filters=filters_
    # )

        #tools.append(query_engine_tool)

        # query_engine = index.as_query_engine(
        #     similarity_top_k=int(top_k), filters=filters_
        # )
        # query_engine_tool2 = QueryEngineTool.from_defaults(
        #     query_engine=query_engine,
        #     name='store2',
        #     description="Almacen de discrepancias y sus dictamenes"
        #         )
        #tools.append(query_engine_tool2)

    #from llama_index.chat_engine.condense_question import CondenseQuestionChatEngine
    from llama_index.core.chat_engine import CondenseQuestionChatEngine
    agent = AgentRunner.from_llm(
        llm=Settings.llm,
        tools=tools,
        #system_prompt=system_prompt,
        verbose=True,
    )

    return CondenseQuestionChatEngine.from_defaults(
        query_engine=agent,
        verbose=True,
    )





def get_BD():
    from sqlalchemy import create_engine, text
    from llama_index.core import SQLDatabase

    URI_BD = os.getenv("URI_BD", "mysql+pymysql://user:pass@localhost:3306/mydb")
    URI_BD_QA = os.getenv("URI_BD_QA", "mysql+pymysql://user:pass@localhost:3306/mydb")

    engine = create_engine(URI_BD_QA)
    sql_database = SQLDatabase(engine, include_tables=["discrepancies"])

    from llama_index.core.query_engine import NLSQLTableQueryEngine

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database, tables=["discrepancies"], llm=Settings.llm
    )
    # query_str = "Which city has the highest population?"
    # response = query_engine.query(query_str)

    return query_engine