# `.\graphrag\graphrag\index\verbs\entities\extraction\entity_extract.py`

```py
# 导入日志模块
import logging
# 导入枚举类型模块
from enum import Enum
# 导入类型提示相关模块
from typing import Any, cast

# 导入 pandas 库，并使用 pd 别名
import pandas as pd
# 导入 datashaper 库中的相关模块和函数
from datashaper import (
    AsyncType,
    TableContainer,
    VerbCallbacks,
    VerbInput,
    derive_from_rows,
    verb,
)

# 从 graphrag.index.bootstrap 中导入 bootstrap 函数
from graphrag.index.bootstrap import bootstrap
# 从 graphrag.index.cache 中导入 PipelineCache 类
from graphrag.index.cache import PipelineCache

# 从当前目录下的 strategies.typing 模块中导入 Document 和 EntityExtractStrategy 类型
from .strategies.typing import Document, EntityExtractStrategy

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


# 枚举类定义，表示实体提取策略类型
class ExtractEntityStrategyType(str, Enum):
    """ExtractEntityStrategyType class definition."""

    graph_intelligence = "graph_intelligence"
    graph_intelligence_json = "graph_intelligence_json"
    nltk = "nltk"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


# 默认的实体类型列表
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]


# 异步函数修饰符，用于定义实体提取函数
@verb(name="entity_extract")
async def entity_extract(
    input: VerbInput,  # 输入参数类型
    cache: PipelineCache,  # 缓存对象类型
    callbacks: VerbCallbacks,  # 回调函数集合类型
    column: str,  # 文档文本列名
    id_column: str,  # 唯一标识列名
    to: str,  # 输出实体信息的列名
    strategy: dict[str, Any] | None,  # 实体提取策略配置
    graph_to: str | None = None,  # 输出图形数据的列名，默认为 None
    async_mode: AsyncType = AsyncType.AsyncIO,  # 异步模式类型，默认为 AsyncIO
    entity_types=DEFAULT_ENTITY_TYPES,  # 要提取的实体类型列表，默认为预定义的类型
    **kwargs,  # 其他关键字参数
) -> TableContainer:
    """
    Extract entities from a piece of text.

    ## Usage
    ### json
    ```json
    {
        "verb": "entity_extract",
        "args": {
            "column": "the_document_text_column_to_extract_entities_from", /* In general this will be your document text column */
            "id_column": "the_column_with_the_unique_id_for_each_row", /* In general this will be your document id */
            "to": "the_column_to_output_the_entities_to", /* This will be a list[dict[str, Any]] a list of entities, with a name, and additional attributes */
            "graph_to": "the_column_to_output_the_graphml_to", /* Optional: This will be a graphml graph in string form which represents the entities and their relationships */
            "strategy": {...} <strategy_config>, see strategies section below
            "entity_types": ["list", "of", "entity", "types", "to", "extract"] /* Optional: This will limit the entity types extracted, default: ["organization", "person", "geo", "event"] */
            "summarize_descriptions" : true | false /* Optional: This will summarize the descriptions of the entities and relationships, default: true */
        }
    }
    ```py
    ### yaml
    ```yaml
    verb: entity_extract

    ```py

    注释：
    这个函数用于从文本中提取实体。

    参数：
    - input: 提供给动词的输入参数对象
    - cache: 用于缓存的管道缓存对象
    - callbacks: 包含回调函数的对象，用于执行在处理过程中的回调
    - column: 包含文档文本的列名
    - id_column: 包含每行唯一标识符的列名
    - to: 输出实体信息的列名
    - strategy: 实体提取策略的配置字典，见下面的策略部分
    - graph_to: 可选，输出图形数据的列名，默认为 None
    - async_mode: 异步模式类型，默认为 AsyncIO
    - entity_types: 要提取的实体类型列表，默认为预定义的类型 ["organization", "person", "geo", "event"]
    - **kwargs: 其他关键字参数

    返回：
    - TableContainer: 包含提取实体结果的表格容器对象
    """
    pass  # 实际实现略去，这里只有函数定义
    args:
        column: the_document_text_column_to_extract_entities_from
        # 要从中提取实体的文档文本列
        id_column: the_column_with_the_unique_id_for_each_row
        # 每行唯一ID所在的列
        to: the_column_to_output_the_entities_to
        # 输出实体的列
        graph_to: the_column_to_output_the_graphml_to
        # 输出GraphML的列
        strategy: <strategy_config>, see strategies section below
        # 策略配置，参见下面的策略部分
        summarize_descriptions: true | false /* Optional: This will summarize the descriptions of the entities and relationships, default: true */
        # 可选项：是否总结实体和关系的描述，默认为true
        entity_types:
            - list
            - of
            - entity
            - types
            - to
            - extract
        # 要提取的实体类型列表
    
    
    注释：
    - `args`: 参数配置部分，包括文本列、唯一ID列、输出列等信息。
    - `summarize_descriptions`: 可选项，控制是否总结实体和关系的描述，默认为true。
    - `entity_types`: 要从文档中提取的实体类型列表。
    
    这段代码配置了一些参数，用于指定从文档中提取实体的具体操作和输出方式。
    """
    This strategy uses the [nltk] library to extract entities from a document. In particular it uses a nltk to extract entities from a piece of text. The strategy config is as follows:
    ```yml
    strategy:
        type: nltk
    ```py
    """
    # 打印调试信息，记录实体提取策略
    log.debug("entity_extract strategy=%s", strategy)
    # 如果未指定实体类型，则使用默认的实体类型
    if entity_types is None:
        entity_types = DEFAULT_ENTITY_TYPES
    # 获取输入数据，并转换为 DataFrame 格式
    output = cast(pd.DataFrame, input.get_input())
    # 如果策略未定义，则使用默认的图智能实体提取策略
    strategy = strategy or {}
    strategy_exec = _load_strategy(
        strategy.get("type", ExtractEntityStrategyType.graph_intelligence)
    )
    # 复制策略配置
    strategy_config = {**strategy}

    num_started = 0

    async def run_strategy(row):
        nonlocal num_started
        # 从行中获取文本内容
        text = row[column]
        # 从行中获取 ID
        id = row[id_column]
        # 执行实体提取策略，返回实体和图结构
        result = await strategy_exec(
            [Document(text=text, id=id)],
            entity_types,
            callbacks,
            cache,
            strategy_config,
        )
        # 增加已启动任务计数
        num_started += 1
        return [result.entities, result.graphml_graph]

    # 并行处理每行数据，应用提取策略
    results = await derive_from_rows(
        output,
        run_strategy,
        callbacks,
        scheduling_type=async_mode,
        num_threads=kwargs.get("num_threads", 4),
    )

    # 整理结果数据
    to_result = []
    graph_to_result = []
    for result in results:
        if result:
            to_result.append(result[0])
            graph_to_result.append(result[1])
        else:
            to_result.append(None)
            graph_to_result.append(None)

    # 将提取的实体结果存储到输出的相应列中
    output[to] = to_result
    # 如果指定了图结构的输出列，则将图结构结果存储到相应列中
    if graph_to is not None:
        output[graph_to] = graph_to_result

    # 返回结果的表容器
    return TableContainer(table=output.reset_index(drop=True))
def _load_strategy(strategy_type: ExtractEntityStrategyType) -> EntityExtractStrategy:
    """Load strategy method definition."""
    # 根据策略类型选择不同的实体抽取策略
    match strategy_type:
        # 如果是图智能策略，从图智能模块导入并返回对应的运行函数
        case ExtractEntityStrategyType.graph_intelligence:
            from .strategies.graph_intelligence import run_gi

            return run_gi

        # 如果是 NLTK 策略，执行 bootstrap() 来初始化环境，然后动态导入 NLTK 模块的运行函数并返回
        case ExtractEntityStrategyType.nltk:
            bootstrap()
            # 动态导入 NLTK 策略模块的运行函数以避免不必要的依赖
            from .strategies.nltk import run as run_nltk

            return run_nltk
        
        # 如果是其他未知的策略类型，抛出值错误异常并提示未知的策略类型
        case _:
            msg = f"Unknown strategy: {strategy_type}"
            raise ValueError(msg)
```