# Utils package

__all__ = ['OpenAIAsyncGenerator', 'vLLMAsyncGenerator']


def __getattr__(name):
    if name == 'OpenAIAsyncGenerator':
        from .openai_generator import OpenAIAsyncGenerator
        return OpenAIAsyncGenerator
    if name == 'vLLMAsyncGenerator':
        from .vllm_generator import vLLMAsyncGenerator
        return vLLMAsyncGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
