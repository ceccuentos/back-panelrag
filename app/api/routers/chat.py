import os
import logging

from aiostream import stream
from fastapi import APIRouter, Depends, HTTPException, Request, status
from llama_index.core.llms import MessageRole
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter
from app.engine import (
    get_chat_engine,
    get_query_engine,
    get_chat_engine2,
    get_chat_engine_tools,
    get_chat_engine_retriever,
    get_chat_engine_agente
)
from app.api.routers.vercel_response import VercelStreamResponse
from app.api.routers.vercel_nonstream_response import VercelNonStreamResponse
from app.api.routers.events import EventCallbackHandler
from app.api.routers.models import (
    ChatData,
    ChatConfig,
    SourceNodes,
    Result,
    Message,
)

# news
from llama_index.core.indices.query.query_transform.base import (
           StepDecomposeQueryTransform,
        )
from llama_index.core.query_engine.multistep_query_engine import (
    MultiStepQueryEngine,
)

chat_router = r = APIRouter()

logger = logging.getLogger("uvicorn")


# streaming endpoint - delete if not needed
@r.post("query")
async def chat(
    request: Request,
    data: ChatData,
):
    try:
        last_message_content = data.get_last_message_content()
        messages = data.get_history_messages()

        doc_ids = data.get_chat_document_ids()

        filters = generate_filters(doc_ids)
        logger.info("Creating chat engine with filters", filters.dict())

        # Trae el Retriever con Reranker
        chat_engine = get_chat_engine_tools(filters=filters, query=last_message_content)

        #chat_engine = get_chat_engine_retriever(retriever)

        event_handler = EventCallbackHandler()
        _ = chat_engine.callback_manager.handlers.append(event_handler)  # type: ignore

        # Hace el query user
        #response = await chat_engine.astream_chat(last_message_content, messages)
        response = chat_engine.query(last_message_content)



        #return VercelStreamResponse(request, event_handler, response)
        # x = await VercelNonStreamResponse.create(event_handler, response)
        # print (x)
        # return x

        return Result(
                result=Message(role=MessageRole.ASSISTANT, content=response.response),
                nodes=SourceNodes.from_source_nodes(response.source_nodes),
            )

    except Exception as e:
        logger.exception("Error in chat engine", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in chat engine: {e}",
        ) from e



# streaming endpoint - delete if not needed
@r.post("")
async def chat(
    request: Request,
    data: ChatData,
):
    try:
        last_message_content = data.get_last_message_content()
        messages = data.get_history_messages()

        doc_ids = data.get_chat_document_ids()

        filters = generate_filters(doc_ids)
        logger.info("Creating query engine with filters", filters.dict())

        #chat_engine = get_chat_engine2(filters=filters, query=last_message_content)

        chat_engine = get_chat_engine_agente(filters=filters)
        # Trae el Retriever con Reranker
        #query_engine = get_query_engine(filters=filters, query=last_message_content)

        # Hace el Engine
        # step_decompose_transform = StepDecomposeQueryTransform(verbose=True)
        # retriever = MultiStepQueryEngine(
        #    retriever, query_transform=step_decompose_transform
        # )

        #chat_engine = get_chat_engine_retriever(retriever)

        event_handler = EventCallbackHandler()
        _ = chat_engine.callback_manager.handlers.append(event_handler)  # type: ignore

        # Hace el query user
        response = await chat_engine.astream_chat(last_message_content, messages)

        return VercelStreamResponse(request, event_handler, response)

        # return Result(
        #         result=Message(role=MessageRole.ASSISTANT, content=response.response),
        #         nodes=SourceNodes.from_source_nodes(response.source_nodes),
        #     )

    except Exception as e:
        logger.exception("Error in query engine", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in chat engine: {e}",
        ) from e



def generate_filters(doc_ids):
    if len(doc_ids) > 0:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="private",
                    value="true",
                    operator="!=",  # type: ignore
                ),
                MetadataFilter(
                    key="doc_id",
                    value=doc_ids,
                    operator="in",  # type: ignore
                ),
            ],
            condition="or",  # type: ignore
        )
    else:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="private",
                    value="true",
                    operator="!=",  # type: ignore
                )
            ]
        )

    return filters


# non-streaming endpoint - delete if not needed
@r.post("/request")
async def chat_request(
    data: ChatData,
    #chat_engine: BaseChatEngine = Depends(get_chat_engine),
) -> Result:

    last_message_content = data.get_last_message_content()
    messages = data.get_history_messages()

    chat_engine = get_chat_engine_tools(query=last_message_content)

    response = await chat_engine.achat(last_message_content, messages)

    return Result(
        result=Message(role=MessageRole.ASSISTANT, content=response.response),
        nodes=SourceNodes.from_source_nodes(response.source_nodes),
    )


@r.get("/config")
async def chat_config() -> ChatConfig:
    starter_questions = None
    conversation_starters = os.getenv("CONVERSATION_STARTERS")
    if conversation_starters and conversation_starters.strip():
        starter_questions = conversation_starters.strip().split("\n")


    return ChatConfig(starter_questions=starter_questions)
