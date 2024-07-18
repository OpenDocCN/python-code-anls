# `.\graphrag\graphrag\index\verbs\covariates\extract_covariates\extract_covariates.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing the extract_covariates verb definition."""

# 导入日志模块
import logging
# 导入数据类模块
from dataclasses import asdict
# 导入枚举类模块
from enum import Enum
# 导入类型提示模块
from typing import Any, cast

# 导入 Pandas 库
import pandas as pd
# 导入 datashaper 库中所需模块
from datashaper import (
    AsyncType,
    TableContainer,
    VerbCallbacks,
    VerbInput,
    derive_from_rows,
    verb,
)

# 导入本地缓存模块
from graphrag.index.cache import PipelineCache
# 导入协变量类型定义
from graphrag.index.verbs.covariates.typing import Covariate, CovariateExtractStrategy

# 获取日志对象
log = logging.getLogger(__name__)


class ExtractClaimsStrategyType(str, Enum):
    """ExtractClaimsStrategyType class definition."""
    
    # 策略类型枚举值定义
    graph_intelligence = "graph_intelligence"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


# 默认的实体类型列表
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]


# 定义 extract_covariates 动词函数
@verb(name="extract_covariates")
async def extract_covariates(
    input: VerbInput,
    cache: PipelineCache,
    callbacks: VerbCallbacks,
    column: str,
    covariate_type: str,
    strategy: dict[str, Any] | None,
    async_mode: AsyncType = AsyncType.AsyncIO,
    entity_types: list[str] | None = None,
    **kwargs,
) -> TableContainer:
    """
    Extract claims from a piece of text.

    ## Usage
    TODO
    """
    # 记录调试日志，输出策略信息
    log.debug("extract_covariates strategy=%s", strategy)
    # 如果实体类型为空，则使用默认实体类型列表
    if entity_types is None:
        entity_types = DEFAULT_ENTITY_TYPES
    # 获取输入数据
    output = cast(pd.DataFrame, input.get_input())

    # 已解析实体映射表
    resolved_entities_map = {}

    # 如果未提供策略，则使用空字典
    strategy = strategy or {}
    # 加载指定类型的策略执行方法
    strategy_exec = load_strategy(
        strategy.get("type", ExtractClaimsStrategyType.graph_intelligence)
    )
    # 复制策略配置
    strategy_config = {**strategy}

    # 定义异步执行的策略运行函数
    async def run_strategy(row):
        text = row[column]
        # 调用策略执行方法获取结果
        result = await strategy_exec(
            text, entity_types, resolved_entities_map, callbacks, cache, strategy_config
        )
        # 将结果中的协变量数据转换为行列表
        return [
            create_row_from_claim_data(row, item, covariate_type)
            for item in result.covariate_data
        ]

    # 使用 derive_from_rows 函数并发执行策略运行函数
    results = await derive_from_rows(
        output,
        run_strategy,
        callbacks,
        scheduling_type=async_mode,
        num_threads=kwargs.get("num_threads", 4),
    )
    # 将结果扁平化成单个 DataFrame
    output = pd.DataFrame([item for row in results for item in row or []])
    # 返回表格容器对象
    return TableContainer(table=output)


# 加载指定类型的策略执行方法
def load_strategy(strategy_type: ExtractClaimsStrategyType) -> CovariateExtractStrategy:
    """Load strategy method definition."""
    match strategy_type:
        case ExtractClaimsStrategyType.graph_intelligence:
            from .strategies.graph_intelligence import run as run_gi

            return run_gi
        case _:
            msg = f"Unknown strategy: {strategy_type}"
            raise ValueError(msg)


# 创建从声明数据和输入行创建行的函数
def create_row_from_claim_data(row, covariate_data: Covariate, covariate_type: str):
    """Create a row from the claim data and the input row."""
    item = {**row, **asdict(covariate_data), "covariate_type": covariate_type}
    # 删除字典 item 中的 "doc_id" 键对应的值
    del item["doc_id"]
    # 返回修改后的字典 item
    return item
```