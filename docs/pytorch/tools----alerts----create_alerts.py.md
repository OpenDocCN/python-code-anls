# `.\pytorch\tools\alerts\create_alerts.py`

```
# 指定脚本使用 Python 3 运行
#!/usr/bin/env python3

# 导入类型注解相关功能
from __future__ import annotations

# 导入命令行参数解析模块
import argparse
# 导入处理 JSON 数据的模块
import json
# 导入操作系统相关功能的模块
import os
# 导入正则表达式模块
import re
# 导入默认字典功能
from collections import defaultdict
# 导入字符串序列匹配模块
from difflib import SequenceMatcher
# 导入类型相关功能
from typing import Any

# 导入处理 HTTP 请求的模块
import requests
# 导入设置工具模块，并忽略类型检查
from setuptools import distutils  # type: ignore[import]

# 定义全局变量：所有跳过阈值
ALL_SKIPPED_THRESHOLD = 100
# 定义全局变量：相似性阈值
SIMILARITY_THRESHOLD = 0.75
# 定义全局变量：失败链阈值
FAILURE_CHAIN_THRESHOLD = 2
# 定义全局变量：最大并发警报数
MAX_CONCURRENT_ALERTS = 1
# 定义全局变量：失败任务的模式
FAILED_JOB_PATTERN = (
    r"^- \[(.*)\]\(.*\) failed consecutively starting with commit \[.*\]\(.*\)$"
)

# 定义常量：状态为 pending
PENDING = "pending"
# 定义常量：状态为 neutral
NEUTRAL = "neutral"
# 定义常量：状态为 skipped
SKIPPED = "skipped"
# 定义常量：状态为 success
SUCCESS = "success"
# 定义常量：状态为 failure
FAILURE = "failure"
# 定义常量：状态为 canceled
CANCELED = "canceled"

# 定义 GraphQL 查询字符串：查询带标签的问题
ISSUES_WITH_LABEL_QUERY = """
query ($owner: String!, $name: String!, $labels: [String!]) {
  repository(owner: $owner, name: $name, followRenames: false) {
    issues(last: 10, labels: $labels, states: [OPEN]) {
      nodes {
        id
        title
        closed
        number
        body
        createdAt
        comments(first: 100) {
          nodes {
            bodyText
            databaseId
          }
        }
      }
    }
  }
}
"""

# 定义 GraphQL 查询字符串：查询问题数量
NUM_ISSUES_QUERY = """
query ($query: String!) {
  search(type: ISSUE, query: $query) {
    issueCount
  }
}
"""

# 定义禁用的警报列表
DISABLED_ALERTS = [
    "rerun_disabled_tests",
    "unstable",
]


class JobStatus:
    # 定义作业名称
    job_name: str = ""
    # 定义作业列表
    jobs: list[Any] = []
    # 定义当前状态
    current_status: Any = None
    # 定义作业状态列表
    job_statuses: list[Any] = []
    # 定义过滤后的作业状态列表
    filtered_statuses: list[Any] = []
    # 定义失败链列表
    failure_chain: list[Any] = []
    # 定义不稳定的作业列表
    flaky_jobs: list[Any] = []

    def __init__(self, job_name: str, job_statuses: list[Any]) -> None:
        # 初始化作业名称和作业状态列表
        self.job_name = job_name
        self.job_statuses = job_statuses

        # 过滤掉所有跳过的作业状态
        self.filtered_statuses = list(
            filter(lambda j: not is_job_skipped(j), job_statuses)
        )
        # 获取当前的作业状态
        self.current_status = self.get_current_status()
        # 获取最近的失败链
        self.failure_chain = self.get_most_recent_failure_chain()
        # 获取不稳定的作业列表
        self.flaky_jobs = self.get_flaky_jobs()

    def get_current_status(self) -> Any:
        """
        获取当前的作业状态，选择最新的非 pending 状态，可能为成功或失败
        """
        for status in self.filtered_statuses:
            if status["conclusion"] != PENDING:
                return status
        return None
    # 返回一个字典，包含按照 failureCaptures 分组的 jobs 列表
    def get_unique_failures(self, jobs: list[Any]) -> dict[str, list[Any]]:
        """
        Returns list of jobs grouped by failureCaptures from the input list
        """
        # 使用 defaultdict 创建一个空字典 failures，值的类型为列表
        failures = defaultdict(list)
        # 遍历输入的 jobs 列表
        for job in jobs:
            # 如果 job 的 conclusion 是 "failure"
            if job["conclusion"] == "failure":
                # 初始化 found_similar_failure 为 False
                found_similar_failure = False
                # 如果 job 中没有 failureCaptures
                if "failureCaptures" not in job:
                    # 将该 job 放入 failures 字典的 "unclassified" 键对应的列表中，并继续下一个 job
                    failures["unclassified"] = [job]
                    continue

                # 将 failureCaptures 转换为字符串形式，多个 failureCaptures 间用空格连接
                failureCaptures = " ".join(job["failureCaptures"])

                # 遍历 failures 字典的键
                for failure in failures:
                    # 使用 SequenceMatcher 检查 failureCaptures 和当前 failure 的相似度
                    seq = SequenceMatcher(None, failureCaptures, failure)
                    # 如果相似度大于 SIMILARITY_THRESHOLD
                    if seq.ratio() > SIMILARITY_THRESHOLD:
                        # 将当前 job 加入 failures 字典对应的 failure 键的列表中
                        failures[failure].append(job)
                        found_similar_failure = True
                        break
                # 如果没有找到相似的 failure，则将当前 failureCaptures 作为新的键，将 job 放入对应列表中
                if not found_similar_failure:
                    failures[failureCaptures] = [job]

        # 返回按 failureCaptures 分组的 jobs 字典
        return failures

    # 获取 flaky jobs 的列表，即只有一个具有该 failureCapture 并且不是最新的 job
    def get_flaky_jobs(self) -> list[Any]:
        # 调用 get_unique_failures 方法获取按 failureCaptures 分组的 jobs 字典
        unique_failures = self.get_unique_failures(self.filtered_statuses)
        # 初始化 flaky_jobs 列表
        flaky_jobs = []
        # 遍历 unique_failures 字典的键
        for failure in unique_failures:
            # 获取当前 failure 对应的 job 列表
            failure_list = unique_failures[failure]
            # 如果 failure_list 只包含一个 job，且该 job 的 sha 不等于当前状态的 sha
            if (
                len(failure_list) == 1
                and failure_list[0]["sha"] != self.current_status["sha"]
            ):
                # 将该 job 添加到 flaky_jobs 列表中
                flaky_jobs.append(failure_list[0])
        # 返回 flaky_jobs 列表
        return flaky_jobs

    # 获取最近失败的 job 链，即连续失败的 jobs 列表，直到遇到成功的 job
    def get_most_recent_failure_chain(self) -> list[Any]:
        # 初始化 failures 列表
        failures = []
        # 初始化 found_most_recent_failure 为 False
        found_most_recent_failure = False

        # 遍历过滤后的 statuses 列表中的 jobs
        for job in self.filtered_statuses:
            # 如果当前 job 失败
            if is_job_failed(job):
                # 将当前 job 添加到 failures 列表中
                failures.append(job)
                # 标记找到最近的失败 job
                found_most_recent_failure = True
            # 如果已经找到最近的失败 job，并且当前 job 不是失败的
            if found_most_recent_failure and not is_job_failed(job):
                # 结束遍历
                break

        # 返回最近失败的 job 链列表
        return failures

    def should_alert(self) -> bool:
        # 使用 get_unique_failures 方法获取失败链的按 failureCaptures 分组的 jobs 字典
        unique_failures = self.get_unique_failures(self.failure_chain)

        # 返回是否满足触发警报的条件：当前状态不为空且不是成功状态，并且存在至少一个失败链长度大于等于 FAILURE_CHAIN_THRESHOLD
        # 同时确保当前 job 名称不包含在 DISABLED_ALERTS 列表中的任何禁用警报中
        return (
            self.current_status is not None
            and self.current_status["conclusion"] != SUCCESS
            and any(
                len(failure_chain) >= FAILURE_CHAIN_THRESHOLD
                for failure_chain in unique_failures.values()
            )
            and all(
                disabled_alert not in self.job_name
                for disabled_alert in DISABLED_ALERTS
            )
        )

    # 返回一个描述对象的字符串，包括 jobName 属性的值
    def __repr__(self) -> str:
        return f"jobName: {self.job_name}"
# 获取 HUD 数据，repo 是仓库名称，branch 是分支名称
def fetch_hud_data(repo: str, branch: str) -> Any:
    # 发送 HTTP GET 请求以获取 HUD 数据
    response = requests.get(f"https://hud.pytorch.org/api/hud/{repo}/{branch}/0")
    # 检查响应是否成功，若不成功则抛出异常
    response.raise_for_status()
    # 将响应内容解析为 JSON 格式
    hud_data = json.loads(response.text)
    # 返回 HUD 数据中的 jobNames 和 shaGrid
    return (hud_data["jobNames"], hud_data["shaGrid"])


# 创建一个字典，将作业名称映射到作业数据列表。HUD 中的一个列
def map_job_data(jobNames: Any, shaGrid: Any) -> dict[str, Any]:
    # 使用 defaultdict 创建 jobData 字典，值为列表类型
    jobData = defaultdict(list)
    # 遍历 shaGrid 中的每个 sha 数据
    for sha in shaGrid:
        # 对于每个 sha 中的作业，将其添加到对应作业名称的列表中
        for ind, job in enumerate(sha["jobs"]):
            jobData[jobNames[ind]].append(job)
    # 返回映射后的作业数据字典
    return jobData


# 判断作业是否失败
def is_job_failed(job: Any) -> bool:
    # 获取作业的 conclusion，若无 conclusion 则设为 None
    conclusion = job["conclusion"] if "conclusion" in job else None
    # 判断作业是否失败，失败条件为 conclusion 不是 SUCCESS 或 PENDING
    return conclusion is not None and conclusion != SUCCESS and conclusion != PENDING


# 判断作业是否跳过
def is_job_skipped(job: Any) -> bool:
    # 获取作业的 conclusion，若无 conclusion 则设为 None
    conclusion = job["conclusion"] if "conclusion" in job else None
    # 判断作业是否跳过，跳过条件为 conclusion 是 NEUTRAL、SKIPPED 或 None
    return conclusion in (NEUTRAL, SKIPPED) or conclusion is None


# 获取失败的作业列表
def get_failed_jobs(job_data: list[Any]) -> list[Any]:
    # 返回所有 conclusion 为 "failure" 的作业列表
    return [job for job in job_data if job["conclusion"] == "failure"]


# 分类作业，生成需要警报的作业和有问题的作业列表
def classify_jobs(
    all_job_names: list[str], sha_grid: Any, filtered_jobs_names: set[str]
) -> tuple[list[JobStatus], list[Any]]:
    """
    创建包含是否需要警报以及是否存在问题的作业状态列表。
    将作业分为需要警报的作业和有问题的作业。
    :param all_job_names: HUD 返回的所有作业名称列表
    :param sha_grid: HUD 返回的所有作业数据列表（与 all_job_names 并行）
    :param filtered_jobs_names: 实际需要考虑的作业名称集合
    :return: 包含警报作业和有问题作业的元组
    """
    # 将所有作业名称与作业数据映射为 job_data 字典
    job_data = map_job_data(all_job_names, sha_grid)
    # 创建 JobStatus 对象列表
    job_statuses: list[JobStatus] = []
    # 遍历 job_data 中的每个作业名和对应数据
    for job in job_data:
        # 将作业名和作业数据添加到 JobStatus 对象列表中
        job_statuses.append(JobStatus(job, job_data[job]))

    # 初始化需要警报的作业列表和有问题的作业列表
    jobs_to_alert_on = []
    flaky_jobs = []

    # 遍历所有的作业状态对象
    for job_status in job_statuses:
        # 如果作业名不在过滤后的作业名称集合中，跳过该作业
        if job_status.job_name not in filtered_jobs_names:
            continue
        # 如果需要警报，将其添加到警报作业列表中
        if job_status.should_alert():
            jobs_to_alert_on.append(job_status)
        # 将所有问题作业添加到有问题的作业列表中
        flaky_jobs.extend(job_status.flaky_jobs)

    # 返回需要警报的作业列表和有问题的作业列表
    return jobs_to_alert_on, flaky_jobs


# 过滤不匹配正则表达式的作业名称
def filter_job_names(job_names: list[str], job_name_regex: str) -> list[str]:
    # 如果有正则表达式，则返回匹配正则表达式的作业名称列表
    if job_name_regex:
        return [
            job_name for job_name in job_names if re.match(job_name_regex, job_name)
        ]
    # 否则返回所有作业名称
    return job_names


# 获取定期失败的作业警报
def get_recurrently_failing_jobs_alerts(
    repo: str, branch: str, job_name_regex: str
) -> list[dict[str, Any]]:
    # 获取 HUD 数据中的作业名称和作业数据
    job_names, sha_grid = fetch_hud_data(repo=repo, branch=branch)
    # 过滤作业名称，仅保留匹配正则表达式的作业名称
    filtered_job_names = set(filter_job_names(job_names, job_name_regex))
    # 如果给定了作业名称的正则表达式
    if job_name_regex:
        # 打印空行
        print()
        # 打印过滤后的作业数量信息
        print(f"Filtered to {len(filtered_job_names)} jobs:")
        # 如果过滤后的作业数量为0，则打印无匹配作业信息
        if len(filtered_job_names) == 0:
            print("No jobs matched the regex")
        # 如果过滤后的作业数量等于所有作业的数量，则打印所有作业匹配信息
        elif len(filtered_job_names) == len(job_names):
            print("All jobs matched the regex")
        # 否则打印所有过滤后的作业名称
        else:
            print("\n".join(filtered_job_names))

    # 对作业名称进行分类，获取周期性失败和不稳定作业
    (recurrently_failing_jobs, flaky_jobs) = classify_jobs(
        job_names, sha_grid, filtered_job_names
    )

    # 初始化告警列表
    alerts = []
    # 遍历所有周期性失败的作业
    for job in recurrently_failing_jobs:
        # 创建告警条目
        entry = {
            "AlertType": "Recurrently Failing Job",
            "AlertObject": job.job_name,
            "OncallTeams": [],
            "OncallIndividuals": [],
            "Flags": [],
            "sha": job.failure_chain[-1]["sha"],
            "branch": branch,
        }
        # 将告警条目添加到告警列表
        alerts.append(entry)
    
    # 返回所有告警条目列表
    return alerts
# 定义一个函数，用于解析命令行参数并返回一个 argparse.Namespace 对象
def parse_args() -> argparse.Namespace:
    # 创建一个 argparse.ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser()
    
    # 添加一个命令行参数 --repo，用于指定要检查的代码库，默认为环境变量 REPO_TO_CHECK 或 "pytorch/pytorch"
    parser.add_argument(
        "--repo",
        help="Repository to do checks for",
        type=str,
        default=os.getenv("REPO_TO_CHECK", "pytorch/pytorch"),
    )
    
    # 添加一个命令行参数 --branch，用于指定要检查的分支，默认为环境变量 BRANCH_TO_CHECK 或 "main"
    parser.add_argument(
        "--branch",
        help="Branch to do checks for",
        type=str,
        default=os.getenv("BRANCH_TO_CHECK", "main"),
    )
    
    # 添加一个命令行参数 --job-name-regex，用于指定要匹配的作业名称的正则表达式，默认为空字符串
    parser.add_argument(
        "--job-name-regex",
        help="Consider only job names matching given regex (if omitted, all jobs are matched)",
        type=str,
        default=os.getenv("JOB_NAME_REGEX", ""),
    )
    
    # 添加一个命令行参数 --with-flaky-test-alert，用于指定是否使用 flaky 测试警报，默认为环境变量 WITH_FLAKY_TEST_ALERT 或 "YES"
    parser.add_argument(
        "--with-flaky-test-alert",
        help="Run this script with the flaky test alerting",
        type=distutils.util.strtobool,
        default=os.getenv("WITH_FLAKY_TEST_ALERT", "YES"),
    )
    
    # 添加一个命令行参数 --dry-run，用于指定是否仅模拟运行而不实际发布问题，默认为环境变量 DRY_RUN 或 "YES"
    parser.add_argument(
        "--dry-run",
        help="Whether or not to actually post issues",
        type=distutils.util.strtobool,
        default=os.getenv("DRY_RUN", "YES"),
    )
    
    # 解析命令行参数并返回一个命名空间对象
    return parser.parse_args()


# 如果当前脚本作为主程序运行，则执行以下代码
if __name__ == "__main__":
    # 解析命令行参数并存储在 args 变量中
    args = parse_args()
    
    # 调用函数 get_recurrently_failing_jobs_alerts，获取关于重复失败作业的警报信息，
    # 并将结果转换为 JSON 格式的字符串
    data = json.dumps(
        get_recurrently_failing_jobs_alerts(args.repo, args.branch, args.job_name_regex)
    )
    
    # 打印 JSON 数据字符串
    print(data)
```