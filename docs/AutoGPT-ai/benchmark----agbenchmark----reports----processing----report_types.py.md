# `.\AutoGPT\benchmark\agbenchmark\reports\processing\report_types.py`

```py
# 导入必要的模块和类型提示
from typing import Any, Dict, List

# 导入基于数据验证的模块
from pydantic import BaseModel, Field, constr, validator

# 定义日期时间格式的正则表达式
datetime_format = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$"

# 定义测试结果的数据模型
class TestResult(BaseModel):
    """Result details for a single run of a test/challenge."""

    success: bool | None = None
    """Whether the run was successful"""
    run_time: str | None = None
    """The (formatted) duration of the run"""
    fail_reason: str | None = None
    """If applicable, the reason why the run was not successful"""
    reached_cutoff: bool | None = None  # None if in progress
    """Whether the run had to be stopped due to reaching the timeout"""
    cost: float | None = None
    """The (known) cost incurred by the run, e.g. from using paid LLM APIs"""

    # 验证器函数，用于验证失败原因和成功标志的逻辑
    @validator("fail_reason")
    def success_xor_fail_reason(cls, v: str | None, values: dict[str, Any]):
        if v:
            success = values["success"]
            assert not success, "fail_reason must only be specified if success=False"
        else:
            assert values["success"], "fail_reason is required if success=False"
        return v

# 定义测试指标的数据模型
class TestMetrics(BaseModel):
    """
    Result metrics for a set of runs for a test/challenge. Should be an aggregate of all
    results for the same test/challenge within a benchmarking session.
    """

    attempted: bool
    """Whether the challenge was attempted during this session"""
    is_regression: bool
    """Whether the challenge was considered a regression test at the time of running"""
    success_percentage: float | None = Field(default=None, alias="success_%")
    """Success rate (0-100) for this challenge within the session"""

# 定义全局指标的数据模型
class MetricsOverall(BaseModel):
    """Global metrics concerning a benchmarking session"""

    run_time: str
    """Duration from beginning to end of the session"""
    highest_difficulty: str
    """
    # 本次会话中至少成功完成一次的最困难挑战的难度
    """
    # 本次会话的总已知成本
# 定义一个 Test 类，继承自 BaseModel
class Test(BaseModel):
    # 定义 category 属性为字符串列表
    category: List[str]
    # 定义 difficulty 属性为字符串或 None
    difficulty: str | None
    # 定义 data_path 属性为字符串
    data_path: str
    # 定义 description 属性为字符串
    description: str
    # 定义 task 属性为字符串
    task: str
    # 定义 answer 属性为字符串
    answer: str
    # 定义 metrics 属性为 TestMetrics 类型
    metrics: TestMetrics
    # 定义 results 属性为 TestResult 类型的列表
    results: list[TestResult]
    # 定义 metadata 属性为字符串到任意类型值的字典，可选，默认为空字典
    metadata: dict[str, Any] | None = Field(default_factory=dict)


# 定义一个 ReportBase 类，继承自 BaseModel
class ReportBase(BaseModel):
    # 定义 command 属性为字符串
    command: str
    # 定义 completion_time 属性为字符串或 None，默认为 None
    completion_time: str | None = None
    # 定义 benchmark_start_time 属性为符合 datetime_format 正则表达式的字符串
    benchmark_start_time: constr(regex=datetime_format)
    # 定义 metrics 属性为 MetricsOverall 类型
    metrics: MetricsOverall
    # 定义 config 属性为字符串到字符串或字典类型值的字典
    config: Dict[str, str | dict[str, str]]
    # 定义 agent_git_commit_sha 属性为字符串或 None，默认为 None
    agent_git_commit_sha: str | None = None
    # 定义 benchmark_git_commit_sha 属性为字符串或 None，默认为 None
    benchmark_git_commit_sha: str | None = None
    # 定义 repo_url 属性为字符串或 None，默认为 None
    repo_url: str | None = None


# 定义一个 Report 类，继承自 ReportBase
class Report(ReportBase):
    # 定义 tests 属性为字符串到 Test 类型值的字典
    tests: Dict[str, Test]
```