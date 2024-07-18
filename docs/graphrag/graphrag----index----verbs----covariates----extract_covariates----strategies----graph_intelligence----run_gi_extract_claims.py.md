# `.\graphrag\graphrag\index\verbs\covariates\extract_covariates\strategies\graph_intelligence\run_gi_extract_claims.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run and _run_chain methods definitions."""

# 从 collections.abc 导入 Iterable 类型
from collections.abc import Iterable
# 导入 Any 类型
from typing import Any

# 导入 VerbCallbacks 类
from datashaper import VerbCallbacks

# 导入 graphrag.config.defaults 模块中的 defs 对象
import graphrag.config.defaults as defs
# 导入 graphrag.config.enums 模块中的 LLMType 枚举类型
from graphrag.config.enums import LLMType
# 从 graphrag.index.cache 模块导入 PipelineCache 类
from graphrag.index.cache import PipelineCache
# 从 graphrag.index.graph.extractors.claims 模块导入 ClaimExtractor 类
from graphrag.index.graph.extractors.claims import ClaimExtractor
# 导入 graphrag.index.llm 模块中的 load_llm 函数
from graphrag.index.llm import load_llm
# 从 graphrag.index.verbs.covariates.typing 模块导入 Covariate 和 CovariateExtractionResult 类型
from graphrag.index.verbs.covariates.typing import (
    Covariate,
    CovariateExtractionResult,
)
# 导入 graphrag.llm 模块中的 CompletionLLM 类
from graphrag.llm import CompletionLLM

# 从当前包的 defaults 模块导入 MOCK_LLM_RESPONSES 对象
from .defaults import MOCK_LLM_RESPONSES


# 定义异步函数 run，接收多种参数并返回 CovariateExtractionResult 类型
async def run(
    input: str | Iterable[str],
    entity_types: list[str],
    resolved_entities_map: dict[str, str],
    reporter: VerbCallbacks,
    pipeline_cache: PipelineCache,
    strategy_config: dict[str, Any],
) -> CovariateExtractionResult:
    """Run the Claim extraction chain."""
    # 从策略配置中获取 llm 字段，如果不存在则使用默认值 MOCK_LLM_RESPONSES
    llm_config = strategy_config.get(
        "llm", {"type": LLMType.StaticResponse, "responses": MOCK_LLM_RESPONSES}
    )
    # 从 llm 配置中获取 llm 类型
    llm_type = llm_config.get("type", LLMType.StaticResponse)
    # 调用 load_llm 函数加载指定类型的 llm 对象
    llm = load_llm("claim_extraction", llm_type, reporter, pipeline_cache, llm_config)
    # 调用 _execute 函数执行实际的操作，并返回结果
    return await _execute(
        llm, input, entity_types, resolved_entities_map, reporter, strategy_config
    )


# 定义私有异步函数 _execute，接收多种参数并返回 CovariateExtractionResult 类型
async def _execute(
    llm: CompletionLLM,
    texts: Iterable[str],
    entity_types: list[str],
    resolved_entities_map: dict[str, str],
    reporter: VerbCallbacks,
    strategy_config: dict[str, Any],
) -> CovariateExtractionResult:
    # 从策略配置中获取提取提示语句
    extraction_prompt = strategy_config.get("extraction_prompt")
    # 从策略配置中获取最大收集数，默认使用定义的常量 CLAIM_MAX_GLEANINGS
    max_gleanings = strategy_config.get("max_gleanings", defs.CLAIM_MAX_GLEANINGS)
    # 从策略配置中获取元组分隔符
    tuple_delimiter = strategy_config.get("tuple_delimiter")
    # 从策略配置中获取记录分隔符
    record_delimiter = strategy_config.get("record_delimiter")
    # 从策略配置中获取完成分隔符
    completion_delimiter = strategy_config.get("completion_delimiter")
    # 从策略配置中获取编码模型名称
    encoding_model = strategy_config.get("encoding_name")

    # 创建 ClaimExtractor 对象，用于执行声明提取操作
    extractor = ClaimExtractor(
        llm_invoker=llm,
        extraction_prompt=extraction_prompt,
        max_gleanings=max_gleanings,
        encoding_model=encoding_model,
        on_error=lambda e, s, d: (
            reporter.error("Claim Extraction Error", e, s, d) if reporter else None
        ),
    )

    # 从策略配置中获取声明描述，如果不存在则抛出 ValueError 异常
    claim_description = strategy_config.get("claim_description")
    if claim_description is None:
        msg = "claim_description is required for claim extraction"
        raise ValueError(msg)

    # 将 texts 转换为列表，以确保处理的是列表数据
    texts = [texts] if isinstance(texts, str) else texts

    # 使用 extractor 对象提取声明，传递相关参数
    results = await extractor({
        "input_text": texts,
        "entity_specs": entity_types,
        "resolved_entities": resolved_entities_map,
        "claim_description": claim_description,
        "tuple_delimiter": tuple_delimiter,
        "record_delimiter": record_delimiter,
        "completion_delimiter": completion_delimiter,
    })

    # 从结果中获取声明数据
    claim_data = results.output
    # 构建一个包含所有从 claim_data 中生成的协变量的 CovariateExtractionResult 对象，并返回
    return CovariateExtractionResult([create_covariate(item) for item in claim_data])
# 根据给定的字典项创建一个 Covariate 对象
def create_covariate(item: dict[str, Any]) -> Covariate:
    """Create a covariate from the item."""
    # 使用字典的 get 方法获取每个字段的值，并作为参数传递给 Covariate 类的构造函数
    return Covariate(
        subject_id=item.get("subject_id"),
        subject_type=item.get("subject_type"),
        object_id=item.get("object_id"),
        object_type=item.get("object_type"),
        type=item.get("type"),
        status=item.get("status"),
        start_date=item.get("start_date"),
        end_date=item.get("end_date"),
        description=item.get("description"),
        source_text=item.get("source_text"),
        doc_id=item.get("doc_id"),
        record_id=item.get("record_id"),
        id=item.get("id"),
    )
```