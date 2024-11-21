import os
from traceloop.sdk import Traceloop

def init_observability():

    api_key_traceloop = os.getenv("API_KEY_TRACELOOP")
    # if api_key_traceloop:
    #     Traceloop.init(
    #         disable_batch=True,
    #         api_key=api_key_traceloop
    #     )
