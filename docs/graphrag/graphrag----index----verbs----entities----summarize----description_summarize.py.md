# `.\graphrag\graphrag\index\verbs\entities\summarize\description_summarize.py`

```py
# 引入 asyncio 库，支持异步编程
import asyncio
# 引入 logging 库，用于日志记录
import logging
# 引入 Enum 枚举类型
from enum import Enum
# 引入 typing 模块中的 Any 类型和 NamedTuple 命名元组
from typing import Any, NamedTuple, cast

# 引入 networkx 库，用于图形操作
import networkx as nx
# 引入 pandas 库，用于数据处理
import pandas as pd
# 从 datashaper 模块中引入相关组件
from datashaper import (
    ProgressTicker,
    TableContainer,
    VerbCallbacks,
    VerbInput,
    progress_ticker,
    verb,
)

# 从 graphrag.index.cache 模块中引入 PipelineCache 类
from graphrag.index.cache import PipelineCache
# 从 graphrag.index.utils 模块中引入 load_graph 函数
from graphrag.index.utils import load_graph

# 从当前包中的 strategies.typing 模块引入 SummarizationStrategy 类
from .strategies.typing import SummarizationStrategy

# 获取当前模块的日志记录器对象
log = logging.getLogger(__name__)


# 命名元组 DescriptionSummarizeRow，包含一个 graph 属性
class DescriptionSummarizeRow(NamedTuple):
    """DescriptionSummarizeRow class definition."""
    graph: Any


# 枚举类型 SummarizeStrategyType，表示摘要策略类型
class SummarizeStrategyType(str, Enum):
    """SummarizeStrategyType class definition."""
    graph_intelligence = "graph_intelligence"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


# 使用装饰器 @verb 定义异步函数 summarize_descriptions
@verb(name="summarize_descriptions")
async def summarize_descriptions(
    input: VerbInput,  # 输入参数对象
    cache: PipelineCache,  # PipelineCache 缓存对象
    callbacks: VerbCallbacks,  # VerbCallbacks 回调函数对象
    column: str,  # 输入数据列名，用于提取描述内容
    to: str,  # 输出数据列名，用于存储摘要后的描述内容
    strategy: dict[str, Any] | None = None,  # 摘要策略配置参数，默认为空字典或 None
    **kwargs,  # 其他未明确列出的关键字参数
) -> TableContainer:
    """
    Summarize entity and relationship descriptions from an entity graph.

    ## Usage

    To turn this feature ON please set the environment variable `GRAPHRAG_SUMMARIZE_DESCRIPTIONS_ENABLED=True`.

    ### json

    ```json
    {
        "verb": "",
        "args": {
            "column": "the_document_text_column_to_extract_descriptions_from", /* Required: This will be a graphml graph in string form which represents the entities and their relationships */
            "to": "the_column_to_output_the_summarized_descriptions_to", /* Required: This will be a graphml graph in string form which represents the entities and their relationships after being summarized */
            "strategy": {...} <strategy_config>, see strategies section below
        }
    }
    ```py

    ### yaml

    ```yaml
    verb: entity_extract
    args:
        column: the_document_text_column_to_extract_descriptions_from
        to: the_column_to_output_the_summarized_descriptions_to
        strategy: <strategy_config>, see strategies section below
    ```py

    ## Strategies

    The summarize descriptions verb uses a strategy to summarize descriptions for entities. The strategy is a json object which defines the strategy to use. The following strategies are available:

    ### graph_intelligence

    This strategy uses the [graph_intelligence] library to summarize descriptions for entities. The strategy config is as follows:

    ```yml
    """
    """
    strategy:
        type: graph_intelligence
        summarize_prompt: # Optional, the prompt to use for extraction

        # Configuration for the LLM (Language Model)
        llm:
            # Specifies the type of LLM to use, such as openai or azure
            type: openai
            # API key for accessing the OpenAI LLM
            api_key: !ENV ${GRAPHRAG_OPENAI_API_KEY}
            # Model name or identifier for the OpenAI LLM
            model: !ENV ${GRAPHRAG_OPENAI_MODEL:gpt-4-turbo-preview}
            # Maximum number of tokens allowed for each request to the LLM
            max_tokens: !ENV ${GRAPHRAG_MAX_TOKENS:6000}
            # Organization context or identifier for OpenAI
            organization: !ENV ${GRAPHRAG_OPENAI_ORGANIZATION}

            # Optional: If using Azure flavor, configure the following parameters
            api_base: !ENV ${GRAPHRAG_OPENAI_API_BASE}
            api_version: !ENV ${GRAPHRAG_OPENAI_API_VERSION}
            proxy: !ENV ${GRAPHRAG_OPENAI_PROXY}
    """
    
    # Log debug message indicating the strategy being used
    log.debug("summarize_descriptions strategy=%s", strategy)
    # Retrieve input data and cast it to a pandas DataFrame
    output = cast(pd.DataFrame, input.get_input())
    # Initialize strategy dictionary, defaulting to an empty dictionary if strategy is None
    strategy = strategy or {}
    # Load the appropriate strategy execution function based on 'type' from strategy configuration
    strategy_exec = load_strategy(
        strategy.get("type", SummarizeStrategyType.graph_intelligence)
    )
    # Create a strategy configuration dictionary by copying the 'strategy' dictionary
    strategy_config = {**strategy}

    # Define an asynchronous function for resolving entities
    async def get_resolved_entities(row, semaphore: asyncio.Semaphore):
        # Load a network graph based on a string or nx.Graph object from the 'column' attribute of 'row'
        graph: nx.Graph = load_graph(cast(str | nx.Graph, getattr(row, column)))

        # Calculate the total number of nodes and edges in the graph for progress tracking
        ticker_length = len(graph.nodes) + len(graph.edges)
        # Initialize a progress ticker to monitor progress
        ticker = progress_ticker(callbacks.progress, ticker_length)

        # Create a list of futures (async tasks) for summarizing node descriptions
        futures = [
            do_summarize_descriptions(
                node,
                sorted(set(graph.nodes[node].get("description", "").split("\n"))),
                ticker,
                semaphore,
            )
            for node in graph.nodes()
        ]
        # Extend the list with futures for summarizing edge descriptions
        futures += [
            do_summarize_descriptions(
                edge,
                sorted(set(graph.edges[edge].get("description", "").split("\n"))),
                ticker,
                semaphore,
            )
            for edge in graph.edges()
        ]

        # Gather results from all async tasks
        results = await asyncio.gather(*futures)

        # Update graph nodes and edges with summarized descriptions from results
        for result in results:
            graph_item = result.items
            if isinstance(graph_item, str) and graph_item in graph.nodes():
                graph.nodes[graph_item]["description"] = result.description
            elif isinstance(graph_item, tuple) and graph_item in graph.edges():
                graph.edges[graph_item]["description"] = result.description

        # Return a DescriptionSummarizeRow object containing the serialized graph in GraphML format
        return DescriptionSummarizeRow(
            graph="\n".join(nx.generate_graphml(graph)),
        )

    # Define an asynchronous function for performing description summarization
    async def do_summarize_descriptions(
        graph_item: str | tuple[str, str],
        descriptions: list[str],
        ticker: ProgressTicker,
        semaphore: asyncio.Semaphore,
        """
        Summarize descriptions for a graph item (node or edge), given a list of descriptions,
        a progress ticker, and an asyncio semaphore for concurrency control.
        """
    ):
        async with semaphore:
            # 使用异步上下文管理器控制并发访问
            results = await strategy_exec(
                graph_item,
                descriptions,
                callbacks,
                cache,
                strategy_config,
            )
            # 执行完策略后，调用 ticker(1) 来进行一次通知
            ticker(1)
        # 返回执行结果
        return results

    # Graph is always on row 0, so here a derive from rows does not work
    # This iteration will only happen once, but avoids hardcoding a iloc[0]
    # Since parallelization is at graph level (nodes and edges), we can't use
    # the parallelization of the derive_from_rows
    # 设置并发控制信号量，根据参数中的 num_threads 或默认为 4
    semaphore = asyncio.Semaphore(kwargs.get("num_threads", 4))

    # 使用并发执行获取解析实体数据
    results = [
        await get_resolved_entities(row, semaphore) for row in output.itertuples()
    ]

    # 初始化空列表，用于存储结果
    to_result = []

    # 遍历每个结果，将图形结果添加到 to_result 中，如果结果为空则添加 None
    for result in results:
        if result:
            to_result.append(result.graph)
        else:
            to_result.append(None)
    # 将结果更新到输出表格的特定列 to 中
    output[to] = to_result
    # 返回更新后的表格容器对象
    return TableContainer(table=output)
# 定义加载策略的函数，根据给定的 SummarizeStrategyType 类型返回相应的 SummarizationStrategy 对象
def load_strategy(strategy_type: SummarizeStrategyType) -> SummarizationStrategy:
    """Load strategy method definition."""

    # 使用模式匹配检查 strategy_type 的值
    match strategy_type:
        # 如果 strategy_type 是 SummarizeStrategyType.graph_intelligence
        case SummarizeStrategyType.graph_intelligence:
            # 导入并返回 graph_intelligence 模块中的 run 函数作为策略
            from .strategies.graph_intelligence import run as run_gi
            return run_gi

        # 如果 strategy_type 不在已知的模式中
        case _:
            # 构建错误消息，指明未知的策略类型
            msg = f"Unknown strategy: {strategy_type}"
            # 抛出 ValueError 异常，并包含错误消息
            raise ValueError(msg)
```