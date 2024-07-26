import os
from app.engine.index import get_index
from fastapi import HTTPException
# from app.engine.vectordb import get_vector_index_store
# from app.engine.vectordb import get_vector_store
from app.engine.generate import get_doc_store


# Nuevas
# from llama_index.postprocessor.cohere_rerank import CohereRerank

# from llama_index.core.postprocessor import LongContextReorder
# from llama_index.core.postprocessor import SimilarityPostprocessor
# from llama_index.core.postprocessor import PrevNextNodePostprocessor

from llama_index.core.postprocessor import AutoPrevNextNodePostprocessor
from llama_index.core import QueryBundle

#from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
# from llama_index.core.postprocessor import LLMRerank
# #from llama_index.postprocessor.rankllm_rerank import RankLLMRerank
# from llama_index.core.postprocessor import LLMRerank
# from llama_index.core.retrievers import VectorIndexRetriever

# from llama_index.retrievers.bm25 import BM25Retriever
# from llama_index.core.retrievers import VectorIndexAutoRetriever
# from llama_index.core.retrievers import QueryFusionRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.indices.query.query_transform.base import (
#     StepDecomposeQueryTransform,
# )
# from llama_index.core.query_engine import MultiStepQueryEngine
# from llama_index.core.settings import Settings
# from llama_index.core.node_parser import SimpleNodeParser

def get_chat_engine(filters=None):
    system_prompt = os.getenv("SYSTEM_PROMPT")
    api_key_cohere = os.getenv("COHERE_API_KEY")


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

    return index.as_chat_engine(
        #vector_store_query_mode="hybrid",
        similarity_top_k=int(top_k),
        system_prompt=system_prompt,
        chat_mode="openai",
        #chat_mode="condense_plus_context",
        filters=filters,
        #node_postprocessors=[cohere_rerank],
    )



def get_chat_engine2(filters=None, query : str = "") :
    system_prompt = os.getenv("SYSTEM_PROMPT")
    api_key_cohere = os.getenv("COHERE_API_KEY")

    # gpt-3
    # step_decompose_transform_gpt3 = StepDecomposeQueryTransform(
    #     llm=Settings.llm, verbose=True
    # )

    # apply nested async to run in a notebook
    # import asyncio
    # asyncio.get_event_loop()
    #asyncio.apply()
    #embeddings_query = obtener_embeddings(query)

    cohere_rerank = CohereRerank(api_key=api_key_cohere, top_n=5)


    top_k = os.getenv("TOP_K", 3)

    index = get_index()
    if index is None:
        raise HTTPException(
            status_code=500,
            detail=str(
                "StorageContext is empty - call 'poetry run generate' to generate the storage first"
            ),
        )

    query_bundle = QueryBundle(query)

    from llama_index.core import StorageContext
    from llama_index.core import load_index_from_storage

    # Extrae Document y Nodes desde storage
    doctos = get_doc_store().to_dict()

    valores_texto = []
    nodos = doctos.get("docstore/data", {})
    for nodo, datos in nodos.items():
        texto = datos.get("__data__", {}).get("text", "No disponible")
        valores_texto.append(texto)

    from llama_index.core import Document, VectorStoreIndex

    documents = [Document(text=t) for t in valores_texto]

    #nodes = Settings.node_parser.get_nodes_from_documents(documents)
    storage_context = StorageContext.from_defaults()
    #storage_context.docstore.add_documents(nodes)

    from llama_index.core import SummaryIndex
    from llama_index.core import VectorStoreIndex

    # query_engine # summary_index = SummaryIndex(nodes, storage_context=storage_context)
    # index # vector_index = VectorStoreIndex(nodes, storage_context=storage_context)


    list_query_engine = index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )

    vector_query_engine = index.as_query_engine()

    #from llama_index.core.tools import QueryEngineTool

    vector_query_engine = VectorStoreIndex.from_documents(
        documents,
    ).as_query_engine()


    # list_tool = QueryEngineTool.from_defaults(
    #     query_engine=list_query_engine,
    #     description=(
    #         "Util para responder preguntas sobre discrepancias"
    #           ),
    # )

    # vector_tool = QueryEngineTool.from_defaults(
    #     query_engine=vector_query_engine,
    #     description=(
    #         "Util para recuperar contexto especifico de discrepancias"
    #        ),
    # )

    #from llama_index.core.tools import QueryEngineTool, ToolMetadata
    #from llama_index.core.query_engine import SubQuestionQueryEngine

    # query_engine_tools = [
    #     QueryEngineTool(
    #         query_engine=vector_query_engine,
    #         metadata=ToolMetadata(
    #             name="index",
    #             description="Ensayos sobre dictamenes",
    #         ),
    #     ),
    # ]

    # query_engine = SubQuestionQueryEngine.from_defaults(
    #     query_engine_tools=query_engine_tools,

    # )


    # response = query_engine.aquery(
    #     query
    # )

    # from llama_index.core.query_engine import RouterQueryEngine
    # from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
    # from llama_index.core.selectors import (
    #     PydanticMultiSelector,
    #     PydanticSingleSelector,
    # )


    # query_engine = RouterQueryEngine(
    #     selector=PydanticSingleSelector.from_defaults(),
    #     query_engine_tools=[
    #         list_tool,
    #         vector_tool,
    #     ],
    # )

    # response = query_engine.query(query_bundle)




    # query_engine = index.as_chat_engine(llm=Settings.llm)
    # query_engine = MultiStepQueryEngine(
    #     query_engine=query_engine,
    #     query_transform=step_decompose_transform_gpt3,
    #     index_summary=query_bundle,
    # )

    # print ("answer ================== II")
    # # index.as_chat_engine(llm=Settings.llm).MultiStepQueryEngine(
    # #     query_engine=query_engine,
    # #     query_transform=step_decompose_transform_gpt3,
    # #     index_summary=query_bundle,
    # # ).query(query_bundle).response
    # answer = query_engine.query(query_bundle)
    # print (answer.response)

    # chat_engine.reset()

    return index.as_chat_engine(
        #vector_store_query_mode="hybrid",
        #as_retriever=response,
        similarity_top_k=int(top_k),
        system_prompt=system_prompt,
        chat_mode="condense_plus_context",
        #chat_mode="best",
        filters=filters,
        #node_postprocessors=[cohere_rerank],
    )



    # # Grab 5 search results
    # retriever = VectorIndexRetriever(index=index, similarity_top_k=25)
    # answer = retriever.retrieve(query_bundle)

    # # Inspect results
    # print ("answer")
    # print([i.get_content() for i in answer])
    # query_engine = RetrieverQueryEngine(retriever=retriever)
    # print ("answer ================== II")
    # print (query_engine.query(query_bundle).response)
    # Pass in your retriever from above, which is configured to return the top 5 results


    # Now you query:
    #llm_query = query_engine.query('How does logarithmic complexity affect graph construction?')


    # vector_retriever = index.as_retriever(similarity_top_k=5)

    # bm25_retriever = BM25Retriever.from_defaults(
    #     docstore=index.docstore, similarity_top_k=2
    # )
    # configure retriever


    # from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo


    # vector_store_info = VectorStoreInfo(
    #     content_info="contenido de dictamen por discrepancia",
    #     metadata_info=[
    #         MetadataInfo(
    #             name="discrepancia",
    #             type="str",
    #             description=(
    #                 "número de discrepancia"
    #             ),
    #         ),
    #         MetadataInfo(
    #             name="dictamen",
    #             type="str",
    #             description=(
    #                 "número de discrepancia a la que corresponde el dictamen"
    #             ),
    #         ),
    #         MetadataInfo(
    #             name="fechadiscrepancia",
    #             type="str",
    #             description=(
    #                 "fecha de la discrepancia en formato DD/MM/AAAA"
    #             ),
    #         ),
    #         MetadataInfo(
    #             name="fechadictamen",
    #             type="str",
    #             description=(
    #                 "fecha del dictamen en formato DD/MM/AAAA"
    #             ),
    #         ),
    #         MetadataInfo(
    #             name="summary",
    #             type="str",
    #             description=(
    #                 "Descripción breve de lo que trata la discrepancia"
    #             ),
    #         ),
    #         MetadataInfo(
    #             name="veredicto",
    #             type="str",
    #             description=(
    #                 "Veredicto, resultado o dictamen final de la discrepancia por parte del panel de expertos"
    #             ),
    #         ),
    #     ],
    # )

    # retriever = QueryFusionRetriever(
    #     [vector_retriever, bm25_retriever],
    #     retriever_weights=[0.6, 0.4],
    #     similarity_top_k=10,
    #     num_queries=1,  # set this to 1 to disable query generation
    #     mode="relative_score",
    #     use_async=True,
    #     verbose=True,
    # )


    # from llama_index.core.query_engine import RetrieverQueryEngine


    # retriever = VectorIndexRetriever(
    #     index=index,
    #     vector_store_info=vector_store_info,
    #     similarity_top_k=25,

    # )


    # query_engine = RetrieverQueryEngine.from_args(
    #     retriever,
    #     )
    # QUERY_GEN_PROMPT = (
    #    "You are a helpful assistant that generates multiple search queries based on a "
    #     "single input query. Generate {num_queries} search queries, one on each line, "
    #     "related to the following input query:\n"
    #     "Query: {query}\n"
    #     "Queries:\n"
    # )



    # from llama_index.core.retrievers import QueryFusionRetriever

    # retriever = QueryFusionRetriever(
    #     [vector_retriever, bm25_retriever],
    #     similarity_top_k=2,
    #     num_queries=4,  # set this to 1 to disable query generation
    #     mode="reciprocal_rerank",
    #     use_async=True,
    #     verbose=True,
    #     query_gen_prompt=QUERY_GEN_PROMPT,  # we could override the query generation prompt here
    # )

    # retrieved_nodes = retriever.retrieve(query_bundle)
    # #retrieved_nodes = bm25_retriever.retrieve(query_bundle)

    # print(f"Found {len(retrieved_nodes)} nodes.")

    # result_dicts = []
    # for node in retrieved_nodes:
    #     #result_dict = {"Score": node.score, "Text": node.node.get_text()[:150], "Metadata": node.metadata}
    #     result_dict = {"Score": node.score, "Metadata": node.metadata}
    #     result_dicts.append(result_dict)

    # print(result_dicts)

    #print (retrieved_nodes)
    # configure reranker

    # reranker = LLMRerank(
    #     choice_batch_size=5,
    #     top_n=3,
    #     )
    # retrieved_nodes = reranker.postprocess_nodes(
    #     retrieved_nodes, query_bundle
    # )

    #vector_store = get_vector_store()
    # vector_index = get_vector_index_store()
    # nodes = vector_index.query(vector=embeddings_query, top_k=25, include_metadata=True)
    # contexts = [x['metadata']['_node_content'] for x in nodes['matches']]

    #docs = {x["metadata"]['text']: i for i, x in enumerate(nodes["matches"])}
    #print (docs)
    #print("\n---\n".join(docs.keys()[:3]))  # print the first 3 docs
    #print(contexts)
    #print (type(contexts))
    #cohere_rerank.postprocess_nodes(embeddings_query, query_str="query")





# def obtener_embeddings(texto):

#     #openai_embedding = OpenAIEmbedding(api_key=openai_api_key)

#     res = openai.embeddings.create(
#         input=texto,
#         model="text-embedding-ada-002"
#     )

#     # Obtener los embeddings del primer documento
#     doc_embeds = [r.embedding for r in res.data]
#     return doc_embeds

