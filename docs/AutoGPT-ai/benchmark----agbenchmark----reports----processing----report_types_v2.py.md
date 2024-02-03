# `.\AutoGPT\benchmark\agbenchmark\reports\processing\report_types_v2.py`

```py
# 用于 API 中的模型定义

# 导入必要的模块
from pydantic import BaseModel, constr

# 定义日期时间格式的正则表达式
datetime_format = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$"

# 定义 TaskInfo 模型，包含数据路径、是否回归、答案、描述、类别和任务
class TaskInfo(BaseModel):
    data_path: str
    is_regression: bool | None
    answer: str
    description: str
    category: list[str]
    task: str

# 定义 RepositoryInfo 模型，包含仓库 URL、团队名称、代理 Git 提交 SHA 和基准 Git 提交 SHA
class RepositoryInfo(BaseModel):
    repo_url: str | None = None
    team_name: str | None = None
    agent_git_commit_sha: str | None = None
    benchmark_git_commit_sha: str | None = None

# 定义 Metrics 模型，包含成本、成功标志、尝试标志、难度、运行时间、失败原因和成功百分比
class Metrics(BaseModel):
    cost: float | None = None
    success: bool
    attempted: bool
    difficulty: str | None = None
    run_time: str | None = None
    fail_reason: str | None = None
    success_percentage: float | None = None

# 定义 RunDetails 模型，包含测试名称、运行 ID、命令、完成时间和基准开始时间
class RunDetails(BaseModel):
    test_name: str
    run_id: str | None = None
    command: str
    completion_time: str | None = None
    benchmark_start_time: constr(regex=datetime_format)

# 定义 BenchmarkRun 模型，包含仓库信息、运行详情、任务信息、指标、是否达到截止时间和配置
class BenchmarkRun(BaseModel):
    repository_info: RepositoryInfo
    run_details: RunDetails
    task_info: TaskInfo
    metrics: Metrics
    reached_cutoff: bool | None = None
    config: dict[str, str | dict[str, str]]
```