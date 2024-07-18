# `.\graphrag\graphrag\index\verbs\entities\extraction\strategies\graph_intelligence\run_graph_intelligence.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_gi,  run_extract_entities and _create_text_splitter methods to run graph intelligence."""

import networkx as nx
from datashaper import VerbCallbacks

import graphrag.config.defaults as defs
from graphrag.config.enums import LLMType
from graphrag.index.cache import PipelineCache
from graphrag.index.graph.extractors.graph import GraphExtractor
from graphrag.index.llm import load_llm
from graphrag.index.text_splitting import (
    NoopTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from graphrag.index.verbs.entities.extraction.strategies.typing import (
    Document,
    EntityExtractionResult,
    EntityTypes,
    StrategyConfig,
)
from graphrag.llm import CompletionLLM

from .defaults import DEFAULT_LLM_CONFIG


async def run_gi(
    docs: list[Document],
    entity_types: EntityTypes,
    reporter: VerbCallbacks,
    pipeline_cache: PipelineCache,
    args: StrategyConfig,
) -> EntityExtractionResult:
    """Run the graph intelligence entity extraction strategy."""
    llm_config = args.get("llm", DEFAULT_LLM_CONFIG)
    llm_type = llm_config.get("type", LLMType.StaticResponse)
    # 加载指定类型的LLM模型用于实体抽取
    llm = load_llm("entity_extraction", llm_type, reporter, pipeline_cache, llm_config)
    # 调用run_extract_entities异步函数来执行实体抽取，并返回结果
    return await run_extract_entities(llm, docs, entity_types, reporter, args)


async def run_extract_entities(
    llm: CompletionLLM,
    docs: list[Document],
    entity_types: EntityTypes,
    reporter: VerbCallbacks | None,
    args: StrategyConfig,
) -> EntityExtractionResult:
    """Run the entity extraction chain."""
    encoding_name = args.get("encoding_name", "cl100k_base")

    # Chunking Arguments
    prechunked = args.get("prechunked", False)
    chunk_size = args.get("chunk_size", defs.CHUNK_SIZE)
    chunk_overlap = args.get("chunk_overlap", defs.CHUNK_OVERLAP)

    # Extraction Arguments
    tuple_delimiter = args.get("tuple_delimiter", None)
    record_delimiter = args.get("record_delimiter", None)
    completion_delimiter = args.get("completion_delimiter", None)
    extraction_prompt = args.get("extraction_prompt", None)
    encoding_model = args.get("encoding_name", None)
    max_gleanings = args.get("max_gleanings", defs.ENTITY_EXTRACTION_MAX_GLEANINGS)

    # note: We're not using UnipartiteGraphChain.from_params
    # because we want to pass "timeout" to the llm_kwargs
    # 根据参数创建文本分割器对象
    text_splitter = _create_text_splitter(
        prechunked, chunk_size, chunk_overlap, encoding_name
    )

    # 创建图提取器对象，设置好各种参数和错误处理
    extractor = GraphExtractor(
        llm_invoker=llm,
        prompt=extraction_prompt,
        encoding_model=encoding_model,
        max_gleanings=max_gleanings,
        on_error=lambda e, s, d: (
            reporter.error("Entity Extraction Error", e, s, d) if reporter else None
        ),
    )
    # 从文档列表中提取文本内容到文本列表中
    text_list = [doc.text.strip() for doc in docs]

    # 如果没有预先分块，则重新分块输入
    # 如果没有预先分块的文本数据，则将文本列表通过换行符连接并使用文本分割器进行分块
    if not prechunked:
        text_list = text_splitter.split_text("\n".join(text_list))

    # 调用提取器函数，传入文本列表和提取器的配置参数，获取提取结果
    results = await extractor(
        list(text_list),
        {
            "entity_types": entity_types,  # 实体类型列表
            "tuple_delimiter": tuple_delimiter,  # 元组分隔符
            "record_delimiter": record_delimiter,  # 记录分隔符
            "completion_delimiter": completion_delimiter,  # 完成分隔符
        },
    )

    # 获取提取结果中的图数据结构
    graph = results.output

    # 将图中节点的 "source_id" 映射回 "id" 字段
    for _, node in graph.nodes(data=True):  # type: ignore
        if node is not None:
            # 将节点的 "source_id" 转换为文档的实际 ID，并用逗号连接成字符串
            node["source_id"] = ",".join(
                docs[int(id)].id for id in node["source_id"].split(",")
            )

    # 将图中边的 "source_id" 映射回 "id" 字段
    for _, _, edge in graph.edges(data=True):  # type: ignore
        if edge is not None:
            # 将边的 "source_id" 转换为文档的实际 ID，并用逗号连接成字符串
            edge["source_id"] = ",".join(
                docs[int(id)].id for id in edge["source_id"].split(",")
            )

    # 从图中的节点数据中构造实体列表，每个实体包含名称和相关数据
    entities = [
        ({"name": item[0], **(item[1] or {})})
        for item in graph.nodes(data=True)
        if item is not None
    ]

    # 生成图的 GraphML 格式数据，并将其合并成一个字符串
    graph_data = "".join(nx.generate_graphml(graph))

    # 返回实体提取结果和图数据的封装对象
    return EntityExtractionResult(entities, graph_data)
def _create_text_splitter(
    prechunked: bool, chunk_size: int, chunk_overlap: int, encoding_name: str
) -> TextSplitter:
    """Create a text splitter for the extraction chain.

    Args:
        - prechunked - Whether the text is already chunked
        - chunk_size - The size of each chunk
        - chunk_overlap - The overlap between chunks
        - encoding_name - The name of the encoding to use

    Returns:
        - output - A text splitter object

    """
    # 如果 prechunked 参数为 True，则返回一个无操作的文本分割器对象
    if prechunked:
        return NoopTextSplitter()

    # 否则，返回一个基于标记的文本分割器对象，使用指定的分块大小、重叠和编码名称
    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name,
    )
```