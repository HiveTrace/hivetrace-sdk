__all__ = []

try:
    from hivetrace.adapters.crewai import (
        CrewAIAdapter as _CrewAIAdapter,
    )
    from hivetrace.adapters.crewai import (
        trace as _crewai_trace,
    )

    CrewAIAdapter = _CrewAIAdapter
    crewai_trace = _crewai_trace
    trace = _crewai_trace

    __all__.extend(["CrewAIAdapter", "crewai_trace", "trace"])
except ImportError:
    pass

try:
    from hivetrace.adapters.langchain import (
        LangChainAdapter as _LangChainAdapter,
    )
    from hivetrace.adapters.langchain import (
        trace as _langchain_trace,
    )

    LangChainAdapter = _LangChainAdapter
    langchain_trace = _langchain_trace

    __all__.extend(["LangChainAdapter", "langchain_trace"])
except ImportError:
    pass

try:
    from hivetrace.adapters.openai_agents import (
        HivetraceOpenAIAgentProcessor as _HivetraceOpenAIAgentProcessor,
    )

    HivetraceOpenAIAgentProcessor = _HivetraceOpenAIAgentProcessor

    __all__.extend(["HivetraceOpenAIAgentProcessor"])
except ImportError:
    pass
