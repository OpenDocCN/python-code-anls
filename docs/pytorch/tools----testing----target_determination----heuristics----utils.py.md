# `.\pytorch\tools\testing\target_determination\heuristics\utils.py`

```py
# 从未来模块导入注释，支持类型注释
from __future__ import annotations

# 导入必要的模块
import json
import os
import re
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import cast, Dict, TYPE_CHECKING
from urllib.request import Request, urlopen
from warnings import warn

# 如果是类型检查，导入测试运行模块
if TYPE_CHECKING:
    from tools.testing.test_run import TestRun

# 确定代码库的根目录
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# 将 Python 测试文件名转换为测试名称集合
def python_test_file_to_test_name(tests: set[str]) -> set[str]:
    prefix = f"test{os.path.sep}"
    valid_tests = {f for f in tests if f.startswith(prefix) and f.endswith(".py")}
    valid_tests = {f[len(prefix) : -len(".py")] for f in valid_tests}

    return valid_tests

# 缓存装饰器，获取 PR 编号
@lru_cache(maxsize=None)
def get_pr_number() -> int | None:
    pr_number = os.environ.get("PR_NUMBER", "")
    if pr_number == "":
        re_match = re.match(r"^refs/tags/.*/(\d+)$", os.environ.get("GITHUB_REF", ""))
        if re_match is not None:
            pr_number = re_match.group(1)
    if pr_number != "":
        return int(pr_number)
    return None

# 缓存装饰器，获取合并基准
@lru_cache(maxsize=None)
def get_merge_base() -> str:
    pr_number = get_pr_number()
    if pr_number is not None:
        github_token = os.environ.get("GITHUB_TOKEN")
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {github_token}",
        }
        url = f"https://api.github.com/repos/pytorch/pytorch/pulls/{pr_number}"
        with urlopen(Request(url, headers=headers)) as conn:
            pr_info = json.loads(conn.read().decode())
            base = f"origin/{pr_info['base']['ref']}"
        merge_base = (
            subprocess.check_output(["git", "merge-base", base, "HEAD"])
            .decode()
            .strip()
        )
        return merge_base
    default_branch = f"origin/{os.environ.get('GIT_DEFAULT_BRANCH', 'main')}"
    merge_base = (
        subprocess.check_output(["git", "merge-base", default_branch, "HEAD"])
        .decode()
        .strip()
    )

    head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

    if merge_base == head:
        # 如果在默认分支上，检查自上次提交以来的更改
        merge_base = "HEAD^"
    return merge_base

# 查询已更改的文件列表
def query_changed_files() -> list[str]:
    base_commit = get_merge_base()

    proc = subprocess.run(
        ["git", "diff", "--name-only", base_commit, "HEAD"],
        capture_output=True,
        check=False,
    )
    print(f"base_commit: {base_commit}")

    if proc.returncode != 0:
        raise RuntimeError("Unable to get changed files")

    lines = proc.stdout.decode().strip().split("\n")
    lines = [line.strip() for line in lines]
    print(f"Changed files: {lines}")
    return lines

# 缓存装饰器，获取 Git 提交信息
@lru_cache(maxsize=None)
def get_git_commit_info() -> str:
    """获取自上次提交以来的提交信息。"""
    base_commit = get_merge_base()
    # 使用 subprocess 模块执行 git 命令获取从 base_commit 到 HEAD 的提交日志
    return (
        subprocess.check_output(
            ["git", "log", f"{base_commit}..HEAD"],
        )
        # 将命令输出的字节流解码为字符串
        .decode()
        # 去除字符串两端的空白字符（包括空格、制表符、换行符等）
        .strip()
    )
# 使用 LRU 缓存装饰器，缓存函数调用结果，最大缓存数量为无限大
@lru_cache(maxsize=None)
def get_issue_or_pr_body(number: int) -> str:
    """获取问题或拉取请求的正文内容"""
    # 从环境变量中获取 GitHub token
    github_token = os.environ.get("GITHUB_TOKEN")
    # 设置 HTTP 请求头部信息，包括接受的数据类型和授权信息
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {github_token}",
    }
    # 根据给定的问题或拉取请求编号构建 API 请求 URL
    # 尽管 URL 中包含 'issues'，但也适用于拉取请求（PRs）
    url = f"https://api.github.com/repos/pytorch/pytorch/issues/{number}"
    # 使用 urllib 发起带有授权头部信息的 HTTP 请求
    with urlopen(Request(url, headers=headers)) as conn:
        # 解析 HTTP 响应并将 JSON 数据转换为字典，提取正文内容
        body: str = json.loads(conn.read().decode())["body"]
        return body


def normalize_ratings(
    ratings: dict[TestRun, float], max_value: float, min_value: float = 0
) -> dict[TestRun, float]:
    """标准化评分值"""
    # 计算评分值的最小和最大值
    min_rating = min(ratings.values())
    assert min_rating > 0  # 断言确保最小评分值大于零
    max_rating = max(ratings.values())
    assert max_rating > 0  # 断言确保最大评分值大于零
    normalized_ratings = {}
    # 根据最小和最大值对评分进行线性归一化处理
    for tf, rank in ratings.items():
        normalized_ratings[tf] = rank / max_rating * (max_value - min_value) + min_value
    return normalized_ratings


def get_ratings_for_tests(file: str | Path) -> dict[str, float]:
    """获取测试文件的评分"""
    # 构建文件路径
    path = REPO_ROOT / file
    # 如果路径不存在，输出错误信息并返回空字典
    if not os.path.exists(path):
        print(f"could not find path {path}")
        return {}
    # 打开并加载测试文件中的评分数据
    with open(path) as f:
        test_file_ratings = cast(Dict[str, Dict[str, float]], json.load(f))
    try:
        # 查询已更改的测试文件列表
        changed_files = query_changed_files()
    except Exception as e:
        # 如果查询失败，输出警告信息并返回空字典
        warn(f"Can't query changed test files due to {e}")
        return {}
    ratings: dict[str, float] = defaultdict(float)
    # 遍历已更改的文件列表，累加各个测试文件的评分
    for file in changed_files:
        for test_file, score in test_file_ratings.get(file, {}).items():
            ratings[test_file] += score
    return ratings


def get_correlated_tests(file: str | Path) -> list[str]:
    """获取与给定文件相关联的测试文件列表，按评分降序排列"""
    # 获取测试文件的评分字典
    ratings = get_ratings_for_tests(file)
    # 按评分降序排列测试文件，返回排序后的文件列表
    prioritize = sorted(ratings, key=lambda x: -ratings[x])
    return prioritize
```