import json
from typing import Any
from fastapi.responses import Response

from app.api.routers.events import EventCallbackHandler
from aiostream import stream

from app.api.routers.models import SourceNodes
from fastapi import Request

from llama_index.core.chat_engine.types import AgentChatResponse


from fastapi.responses import Response

class VercelNonStreamResponse(Response):
    """
    Class to generate a non-streaming response with the structure needed by Vercel
    """

    DATA_PREFIX = "8:"

    @classmethod
    def convert_data(cls, data: dict):
        data_str = json.dumps(data)
        return f"{cls.DATA_PREFIX}[{data_str}]\n"

    @classmethod
    async def create(
        cls,
        event_handler: EventCallbackHandler,
        response: Response  # Replace with the actual type of response
    ):
        # Mark the event handler as done
        event_handler.is_done = True

        # Generate the event data structure
        event_data = cls.convert_data(
            {
                "type": "sources",
                "data": {
                    "nodes": [
                        SourceNodes.from_source_node(node).dict()
                        for node in response.source_nodes
                    ]
                },
            }
        )

        # Return the final response
        return cls(content=event_data, media_type="text/plain")

