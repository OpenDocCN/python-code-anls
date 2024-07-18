# `.\graphrag\graphrag\index\verbs\graph\report\strategies\graph_intelligence\run_graph_intelligence.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run, _run_extractor and _load_nodes_edges_for_claim_chain methods definition."""

import json  # 导入处理 JSON 格式数据的模块
import logging  # 导入日志记录模块
import traceback  # 导入跟踪异常堆栈的模块

from datashaper import VerbCallbacks  # 从 datashaper 模块中导入 VerbCallbacks 类

from graphrag.config.enums import LLMType  # 从 graphrag.config.enums 模块导入 LLMType 枚举
from graphrag.index.cache import PipelineCache  # 从 graphrag.index.cache 模块导入 PipelineCache 类
from graphrag.index.graph.extractors.community_reports import (  # 从 graphrag.index.graph.extractors.community_reports 模块导入 CommunityReportsExtractor 类
    CommunityReportsExtractor,
)
from graphrag.index.llm import load_llm  # 从 graphrag.index.llm 模块导入 load_llm 函数
from graphrag.index.utils.rate_limiter import RateLimiter  # 从 graphrag.index.utils.rate_limiter 模块导入 RateLimiter 类
from graphrag.index.verbs.graph.report.strategies.typing import (  # 从 graphrag.index.verbs.graph.report.strategies.typing 模块导入 CommunityReport 和 StrategyConfig 类
    CommunityReport,
    StrategyConfig,
)
from graphrag.llm import CompletionLLM  # 从 graphrag.llm 模块导入 CompletionLLM 类

from .defaults import MOCK_RESPONSES  # 从当前包中导入 MOCK_RESPONSES 变量

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


async def run(  # 定义异步函数 run，用于执行图形智能实体提取策略
    community: str | int,  # 社区标识符，可以是字符串或整数
    input: str,  # 输入文本
    level: int,  # 提取级别
    reporter: VerbCallbacks,  # 用于报告的回调对象
    pipeline_cache: PipelineCache,  # 用于缓存管道的对象
    args: StrategyConfig,  # 策略配置参数
) -> CommunityReport | None:  # 返回类型可以是 CommunityReport 或 None
    """Run the graph intelligence entity extraction strategy."""
    llm_config = args.get(  # 获取 llm 参数配置，默认为静态响应类型，使用 MOCK_RESPONSES 作为响应数据
        "llm", {"type": LLMType.StaticResponse, "responses": MOCK_RESPONSES}
    )
    llm_type = llm_config.get("type", LLMType.StaticResponse)  # 获取 llm 类型
    llm = load_llm(  # 加载指定类型的 LLM 模型
        "community_reporting", llm_type, reporter, pipeline_cache, llm_config
    )
    return await _run_extractor(llm, community, input, level, args, reporter)  # 调用 _run_extractor 函数进行提取操作


async def _run_extractor(  # 定义异步函数 _run_extractor，用于执行实体提取过程
    llm: CompletionLLM,  # 完成的语言模型对象
    community: str | int,  # 社区标识符，可以是字符串或整数
    input: str,  # 输入文本
    level: int,  # 提取级别
    args: StrategyConfig,  # 策略配置参数
    reporter: VerbCallbacks,  # 用于报告的回调对象
) -> CommunityReport | None:  # 返回类型可以是 CommunityReport 或 None
    # RateLimiter
    rate_limiter = RateLimiter(rate=1, per=60)  # 创建速率限制器，限制每秒操作不超过一次

    extractor = CommunityReportsExtractor(  # 创建社区报告提取器对象
        llm,
        extraction_prompt=args.get("extraction_prompt", None),  # 提取提示语句，可选参数
        max_report_length=args.get("max_report_length", None),  # 最大报告长度，可选参数
        on_error=lambda e, stack, _data: reporter.error(  # 错误处理函数，记录报告提取错误
            "Community Report Extraction Error", e, stack
        ),
    )

    try:
        await rate_limiter.acquire()  # 等待并获取速率限制器许可
        results = await extractor({"input_text": input})  # 执行提取操作，获取结果
        report = results.structured_output  # 获取结构化输出的报告内容
        if report is None or len(report.keys()) == 0:  # 如果报告为空或没有有效内容
            log.warning("No report found for community: %s", community)  # 记录警告日志，指示未找到报告
            return None

        return CommunityReport(  # 返回社区报告对象
            community=community,  # 社区标识符
            full_content=results.output,  # 完整内容
            level=level,  # 提取级别
            rank=_parse_rank(report),  # 解析报告的排名
            title=report.get("title", f"Community Report: {community}"),  # 报告标题，默认为社区报告 + 社区标识符
            rank_explanation=report.get("rating_explanation", ""),  # 排名解释
            summary=report.get("summary", ""),  # 报告摘要
            findings=report.get("findings", []),  # 发现结果列表
            full_content_json=json.dumps(report, indent=4),  # 完整内容的 JSON 格式字符串
        )
    except Exception as e:
        log.exception("Error processing community: %s", community)  # 记录异常日志，指示处理社区时出错
        reporter.error(  # 报告提取错误
            "Community Report Extraction Error", e, traceback.format_exc()
        )
        return None


def _parse_rank(report: dict) -> float:
    # 解析报告的排名
    # 从报告中获取评级信息，如果找不到则默认为 -1
    rank = report.get("rating", -1)
    
    try:
        # 尝试将评级信息转换为浮点数并返回
        return float(rank)
    except ValueError:
        # 如果转换失败，记录异常信息并返回默认值 -1
        log.exception("Error parsing rank: %s defaulting to -1", rank)
        return -1
```