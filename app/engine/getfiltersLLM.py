import logging
from typing import Any, Optional, cast


from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.vector_store.retrievers.auto_retriever.output_parser import (
    VectorStoreQueryOutputParser,
)
from llama_index.core.indices.vector_store.retrievers.auto_retriever.prompts import (
    DEFAULT_VECTOR_STORE_QUERY_PROMPT_TMPL,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.output_parsers.base import (
    OutputParserException,
    StructuredOutput,
)
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.schema import QueryBundle
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    llm_from_settings_or_context,
)
from llama_index.core.vector_stores.types import (
    FilterCondition,
    MetadataFilters,
    VectorStoreInfo,
    VectorStoreQuerySpec,
)

_logger = logging.getLogger(__name__)

class GetFiltersPrompt:
    """
    Obtine filtros a partir de LLM con query_str.

    Args:
        vector_store_info (VectorStoreInfo): Información adicional sobre
            el contenido de la tienda vectorial y los filtros de metadatos admitidos.
            La descripción en lenguaje natural es utilizada por un LLM para establecer
            automáticamente los parámetros de consulta de la tienda vectorial.
        llm: opcional, llm a utilizar
        prompt_template_str (str): Cadena de plantilla de mensaje personalizada para LLM.
            Usa la cadena de plantilla predeterminada si es None.
        service_context (ServiceContext): Contexto de servicio que contiene una referencia a un LLM.
            Usa el contexto de servicio del índice por defecto si es None.
        callback_manager (Optional[CallbackManager]): Administrador de callbacks.
    """

    def __init__(
        self,
        vector_store_info: VectorStoreInfo,
        llm: Optional[LLM] = None,
        prompt_template_str: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        service_context: Optional[ServiceContext] = None,
        **kwargs: Any,
    ) -> None:
        self.filters=None
        self._vector_store_info = vector_store_info
        self._current_filters = None  # Inicializar el atributo para almacenar filtros

        self._llm = llm or llm_from_settings_or_context(Settings, service_context)
        self._callback_manager = (
            callback_manager
            or callback_manager_from_settings_or_context(Settings, service_context)
        )

        self._prompt_template_str = (
            prompt_template_str or DEFAULT_VECTOR_STORE_QUERY_PROMPT_TMPL
        )
        self._output_parser = VectorStoreQueryOutputParser()
        self._prompt = PromptTemplate(template=self._prompt_template_str)

        self._kwargs = kwargs

    def _parse_generated_spec(
        self, output: str, query_str: str
    ) -> BaseModel:
        """Parse generated spec."""
        try:
            structured_output = cast(
                StructuredOutput, self._output_parser.parse(output)
            )
            query_spec = cast(VectorStoreQuerySpec, structured_output.parsed_output)

        except OutputParserException as e:

            _logger.warning(f"Failed to parse query spec, using defaults as fallback. {e}")
            query_spec = VectorStoreQuerySpec(
                query=query_str,
                filters=[],
                top_k=None,
            )

        return query_spec

    def generate_filters(
        self, query_str: str, **kwargs: Any
    ) -> BaseModel:
        # prepare input
        info_str = self._vector_store_info.json(indent=4)
        schema_str = VectorStoreQuerySpec.schema_json(indent=4)

        # call LLM
        output = self._llm.predict(
            self._prompt,
            schema_str=schema_str,
            info_str=info_str,
            query_str=query_str,
        )

        # parse output
        _generate = self._parse_generated_spec(output, query_str)
        _filters = self._build_filter(_generate)

        return _filters

    def _build_filter(
        self, spec: VectorStoreQuerySpec
    ) -> MetadataFilters:
        _logger.info(f"Using query str: {spec.query}")
        filter_list = [
            (filter.key, filter.operator.value, filter.value) for filter in spec.filters if filter.value != ""
        ]

        # def _condition_filter(tuplas):
        #     claves_vistas = set()
        #     for tupla in tuplas:
        #         clave = tupla[0]
        #         if clave in claves_vistas:
        #             return True
        #         else:
        #             claves_vistas.add(clave)
        #     return False

        # if _condition_filter(filter_list):
        #     _condition = FilterCondition.OR
        # else:
        #     _condition = FilterCondition.AND
        # avoid passing empty filters to retriever
        if len(spec.filters) == 0:
            _current_filters = None
        else:
            _current_filters = MetadataFilters(
                filters=[*spec.filters],
                condition=FilterCondition.OR,
            )

        return _current_filters
