# `.\pytorch\tools\linter\adapters\workflow_consistency_linter.py`

```
"""
Checks for consistency of jobs between different GitHub workflows.

Any job with a specific `sync-tag` must match all other jobs with the same `sync-tag`.
"""

from __future__ import annotations  # 允许类型注解中使用类本身作为类型

import argparse  # 解析命令行参数的模块
import itertools  # 用于创建迭代器的模块
import json  # 处理 JSON 数据的模块
from collections import defaultdict  # 提供默认值的字典
from enum import Enum  # 枚举类型的基类
from pathlib import Path  # 操作文件和目录路径的模块
from typing import Any, Iterable, NamedTuple  # 类型提示模块

from yaml import dump, load  # YAML 格式数据的加载和转储

# Safely load fast C Yaml loader/dumper if they are available
try:
    from yaml import CSafeLoader as Loader  # 尝试导入快速的 C Yaml 加载器/转储器
except ImportError:
    from yaml import SafeLoader as Loader  # type: ignore[assignment, misc]  # 导入安全的 Yaml 加载器/转储器


class LintSeverity(str, Enum):
    ERROR = "error"  # 错误严重性级别
    WARNING = "warning"  # 警告严重性级别
    ADVICE = "advice"  # 建议严重性级别
    DISABLED = "disabled"  # 禁用状态严重性级别


class LintMessage(NamedTuple):
    path: str | None  # 错误路径
    line: int | None  # 错误行数
    char: int | None  # 错误字符数
    code: str  # 错误代码
    severity: LintSeverity  # 错误严重性级别
    name: str  # 错误名称
    original: str | None  # 原始值
    replacement: str | None  # 替换值
    description: str | None  # 错误描述


def glob_yamls(path: Path) -> Iterable[Path]:
    return itertools.chain(path.glob("**/*.yml"), path.glob("**/*.yaml"))  # 生成所有 YAML 文件的路径迭代器


def load_yaml(path: Path) -> Any:
    with open(path) as f:
        return load(f, Loader)  # 加载指定 YAML 文件并返回其内容


def is_workflow(yaml: Any) -> bool:
    return yaml.get("jobs") is not None  # 检查 YAML 文件是否包含 "jobs" 键，确定是否为工作流文件


def print_lint_message(path: Path, job: dict[str, Any], sync_tag: str) -> None:
    job_id = next(iter(job.keys()))  # 获取工作流 ID
    with open(path) as f:
        lines = f.readlines()  # 读取文件所有行

    # 寻找包含工作流 ID 的行号
    for i, line in enumerate(lines):
        if f"{job_id}:" in line:
            line_number = i + 1  # 记录行号

    # 创建 lint 消息对象
    lint_message = LintMessage(
        path=str(path),
        line=line_number,
        char=None,
        code="WORKFLOWSYNC",
        severity=LintSeverity.ERROR,
        name="workflow-inconsistency",
        original=None,
        replacement=None,
        description=f"Job doesn't match other jobs with sync-tag: '{sync_tag}'",  # 错误描述
    )
    print(json.dumps(lint_message._asdict()), flush=True)  # 将 lint 消息转换为 JSON 格式并打印


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="workflow consistency linter.",  # 命令行解析器的描述信息
        fromfile_prefix_chars="@",  # 文件前缀字符为 '@'，用于从文件中读取参数
    )
    parser.add_argument(
        "filenames",
        nargs="+",  # 至少需要一个文件路径
        help="paths to lint",  # 帮助信息，指示参数是用于 lint 的路径
    )
    args = parser.parse_args()  # 解析命令行参数

    # Go through the provided files, aggregating jobs with the same sync tag
    tag_to_jobs = defaultdict(list)  # 使用默认值列表的字典，用于存储相同 sync tag 的工作流
    # 遍历传入的文件名列表
    for path in args.filenames:
        # 加载指定路径的 YAML 文件内容并解析成 Python 对象
        workflow = load_yaml(Path(path))
        # 获取工作流中的所有作业
        jobs = workflow["jobs"]
        
        # 遍历每个作业及其对应的作业 ID
        for job_id, job in jobs.items():
            try:
                # 尝试获取作业的同步标签
                sync_tag = job["with"]["sync-tag"]
            except KeyError:
                # 如果作业中没有同步标签，则跳过该作业
                continue

            # 移除作业中的 "if" 字段，因为不同作业之间可能会有不同的触发条件
            if "if" in job:
                del job["if"]

            # 同样地，移除作业中的 ['with']['test-matrix'] 字段
            if "test-matrix" in job.get("with", {}):
                del job["with"]["test-matrix"]

            # 将具有相同同步标签的作业及其路径信息存入字典
            tag_to_jobs[sync_tag].append((path, {job_id: job}))

    # 对于每个同步标签，检查所有作业是否具有相同的代码内容
    for sync_tag, path_and_jobs in tag_to_jobs.items():
        # 弹出基准作业的路径及其对应的字典表示
        baseline_path, baseline_dict = path_and_jobs.pop()
        # 将基准作业的字典表示转换为字符串形式
        baseline_str = dump(baseline_dict)

        printed_baseline = False

        # 遍历同步标签下的其他作业
        for path, job_dict in path_and_jobs:
            # 将当前作业的字典表示转换为字符串形式
            job_str = dump(job_dict)
            # 检查当前作业与基准作业的代码是否相同
            if baseline_str != job_str:
                # 如果不相同，则打印 lint 错误信息
                print_lint_message(path, job_dict, sync_tag)

                # 如果还没有打印过基准作业的 lint 错误信息，则也打印基准作业的信息
                if not printed_baseline:
                    print_lint_message(baseline_path, baseline_dict, sync_tag)
                    printed_baseline = True
```