# `.\graphrag\graphrag\index\verbs\entities\summarize\strategies\graph_intelligence\run_graph_intelligence.py`

```py
# 版权声明和许可证声明
# 本模块包含了运行图智能的 run_gi、run_resolve_entities 和 _create_text_list_splitter 方法。

# 导入必要的模块和类
from datashaper import VerbCallbacks

# 导入枚举类型 LLMType
from graphrag.config.enums import LLMType
# 导入 PipelineCache 类
from graphrag.index.cache import PipelineCache
# 导入 SummarizeExtractor 类
from graphrag.index.graph.extractors.summarize import SummarizeExtractor
# 导入 load_llm 方法
from graphrag.index.llm import load_llm
# 导入 StrategyConfig 和 SummarizedDescriptionResult 类
from graphrag.index.verbs.entities.summarize.strategies.typing import (
    StrategyConfig,
    SummarizedDescriptionResult,
)
# 导入 CompletionLLM 类
from graphrag.llm import CompletionLLM

# 导入默认配置 DEFAULT_LLM_CONFIG
from .defaults import DEFAULT_LLM_CONFIG


async def run(
    described_items: str | tuple[str, str],
    descriptions: list[str],
    reporter: VerbCallbacks,
    pipeline_cache: PipelineCache,
    args: StrategyConfig,
) -> SummarizedDescriptionResult:
    """运行图智能的实体提取策略。"""
    # 获取 llm 配置
    llm_config = args.get("llm", DEFAULT_LLM_CONFIG)
    # 获取 llm 类型
    llm_type = llm_config.get("type", LLMType.StaticResponse)
    # 载入 llm 模型
    llm = load_llm(
        "summarize_descriptions", llm_type, reporter, pipeline_cache, llm_config
    )
    # 执行描述摘要运行
    return await run_summarize_descriptions(
        llm, described_items, descriptions, reporter, args
    )


async def run_summarize_descriptions(
    llm: CompletionLLM,
    items: str | tuple[str, str],
    descriptions: list[str],
    reporter: VerbCallbacks,
    args: StrategyConfig,
) -> SummarizedDescriptionResult:
    """运行实体提取链。"""
    # 提取参数
    summarize_prompt = args.get("summarize_prompt", None)
    entity_name_key = args.get("entity_name_key", "entity_name")
    input_descriptions_key = args.get("input_descriptions_key", "description_list")
    max_tokens = args.get("max_tokens", None)

    # 创建 SummarizeExtractor 实例
    extractor = SummarizeExtractor(
        llm_invoker=llm,
        summarization_prompt=summarize_prompt,
        entity_name_key=entity_name_key,
        input_descriptions_key=input_descriptions_key,
        on_error=lambda e, stack, details: (
            reporter.error("Entity Extraction Error", e, stack, details)
            if reporter
            else None
        ),
        max_summary_length=args.get("max_summary_length", None),
        max_input_tokens=max_tokens,
    )

    # 执行摘要提取
    result = await extractor(items=items, descriptions=descriptions)
    # 返回 SummarizedDescriptionResult 结果对象
    return SummarizedDescriptionResult(
        items=result.items, description=result.description
    )
```