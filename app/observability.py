from traceloop.sdk import Traceloop

def init_observability():

    Traceloop.init(
        disable_batch=True,
        api_key="dbb3990794e9155fedd2ba1f82517e295013320c5e4b902c3ff8c7f2da91eb65a4dbbbdfa016b912acc00ece59953bed"
    )