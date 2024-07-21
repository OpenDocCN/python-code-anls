# `.\pytorch\tools\stats\import_test_stats.py`

```py
#!/usr/bin/env python3

from __future__ import annotations

import datetime  # 导入处理日期时间的模块
import json  # 导入处理 JSON 数据的模块
import os  # 导入操作系统相关功能的模块
import shutil  # 导入高级文件操作功能的模块
from pathlib import Path  # 导入处理路径的模块
from typing import Any, Callable, cast, Dict  # 导入类型提示相关的模块
from urllib.request import urlopen  # 导入进行 URL 请求的模块

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # 获取当前文件的上级目录的上级目录的绝对路径


def get_disabled_issues() -> list[str]:
    reenabled_issues = os.getenv("REENABLED_ISSUES", "")  # 从环境变量中获取禁用的问题列表
    issue_numbers = reenabled_issues.split(",")  # 将禁用的问题列表拆分为字符串数组
    print("Ignoring disabled issues: ", issue_numbers)  # 打印被忽略的禁用问题列表
    return issue_numbers  # 返回禁用的问题列表


SLOW_TESTS_FILE = ".pytorch-slow-tests.json"
DISABLED_TESTS_FILE = ".pytorch-disabled-tests.json"
ADDITIONAL_CI_FILES_FOLDER = Path(".additional_ci_files")
TEST_TIMES_FILE = "test-times.json"
TEST_CLASS_TIMES_FILE = "test-class-times.json"
TEST_FILE_RATINGS_FILE = "test-file-ratings.json"
TEST_CLASS_RATINGS_FILE = "test-class-ratings.json"
TD_HEURISTIC_PROFILING_FILE = "td_heuristic_profiling.json"
TD_HEURISTIC_HISTORICAL_EDITED_FILES = "td_heuristic_historical_edited_files.json"
TD_HEURISTIC_PREVIOUSLY_FAILED = "previous_failures.json"
TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL = "previous_failures_additional.json"

FILE_CACHE_LIFESPAN_SECONDS = datetime.timedelta(hours=3).seconds  # 设置文件缓存有效期为3小时


def fetch_and_cache(
    dirpath: str | Path,
    name: str,
    url: str,
    process_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """
    This fetch and cache utils allows sharing between different process.
    """
    Path(dirpath).mkdir(exist_ok=True)  # 创建目录，如果目录已存在则忽略

    path = os.path.join(dirpath, name)  # 构建文件的完整路径
    print(f"Downloading {url} to {path}")  # 打印下载的 URL 和文件路径

    def is_cached_file_valid() -> bool:
        # Check if the file is new enough (see: FILE_CACHE_LIFESPAN_SECONDS). A real check
        # could make a HEAD request and check/store the file's ETag
        fname = Path(path)  # 获取文件路径的 Path 对象
        now = datetime.datetime.now()  # 获取当前时间
        mtime = datetime.datetime.fromtimestamp(fname.stat().st_mtime)  # 获取文件的修改时间
        diff = now - mtime  # 计算当前时间与文件修改时间的差值
        return diff.total_seconds() < FILE_CACHE_LIFESPAN_SECONDS  # 判断文件是否在缓存有效期内

    if os.path.exists(path) and is_cached_file_valid():
        # Another test process already download the file, so don't re-do it
        with open(path) as f:
            return cast(Dict[str, Any], json.load(f))  # 返回从缓存中加载的 JSON 数据

    for _ in range(3):  # 最多尝试下载3次
        try:
            contents = urlopen(url, timeout=5).read().decode("utf-8")  # 下载并读取 URL 中的内容
            processed_contents = process_fn(json.loads(contents))  # 处理下载内容并转换为 JSON 对象
            with open(path, "w") as f:
                f.write(json.dumps(processed_contents))  # 将处理后的内容写入到文件中
            return processed_contents  # 返回处理后的内容
        except Exception as e:
            print(f"Could not download {url} because: {e}.")  # 打印下载失败的原因
    print(f"All retries exhausted, downloading {url} failed.")  # 所有重试尝试失败后的提示
    return {}  # 返回空字典表示下载失败


def get_slow_tests(
    dirpath: str, filename: str = SLOW_TESTS_FILE
) -> dict[str, float] | None:
    url = "https://ossci-metrics.s3.amazonaws.com/slow-tests.json"  # 慢速测试数据的 URL
    try:
        return fetch_and_cache(dirpath, filename, url, lambda x: x)  # 调用 fetch_and_cache 函数获取慢速测试数据
    except Exception:
        # 捕获所有异常
        print("Couldn't download slow test set, leaving all tests enabled...")
        # 打印无法下载慢速测试集的消息
        return {}
        # 返回空字典作为结果
# 从测试基础设施生成的统计数据中获取测试执行时间的字典
def get_test_times() -> dict[str, dict[str, float]]:
    return get_from_test_infra_generated_stats(
        "test-times.json",  # 文件名为 "test-times.json"
        TEST_TIMES_FILE,    # 使用常量 TEST_TIMES_FILE 指定的文件路径
        "Couldn't download test times...",  # 下载失败时的错误消息
    )


# 从测试基础设施生成的统计数据中获取测试类执行时间的字典
def get_test_class_times() -> dict[str, dict[str, float]]:
    return get_from_test_infra_generated_stats(
        "test-class-times.json",  # 文件名为 "test-class-times.json"
        TEST_CLASS_TIMES_FILE,    # 使用常量 TEST_CLASS_TIMES_FILE 指定的文件路径
        "Couldn't download test times...",  # 下载失败时的错误消息
    )


# 获取禁用测试的字典或者空值
def get_disabled_tests(
    dirpath: str,               # 指定目录路径
    filename: str = DISABLED_TESTS_FILE  # 指定文件名，默认使用 DISABLED_TESTS_FILE 常量
) -> dict[str, Any] | None:
    # 处理禁用测试的回调函数，返回经过处理后的禁用测试字典
    def process_disabled_test(the_response: dict[str, Any]) -> dict[str, Any]:
        # 获取禁用的问题列表
        disabled_issues = get_disabled_issues()
        disabled_test_from_issues = dict()
        # 遍历响应字典，移除已重新启用的测试，并且通过 pr_num 进一步压缩
        for test_name, (pr_num, link, platforms) in the_response.items():
            if pr_num not in disabled_issues:
                disabled_test_from_issues[test_name] = (
                    link,
                    platforms,
                )
        return disabled_test_from_issues

    try:
        url = "https://ossci-metrics.s3.amazonaws.com/disabled-tests-condensed.json"
        return fetch_and_cache(dirpath, filename, url, process_disabled_test)
    except Exception:
        # 下载失败时的错误处理
        print("Couldn't download test skip set, leaving all tests enabled...")
        return {}


# 从测试基础设施生成的统计数据中获取测试文件评级的字典
def get_test_file_ratings() -> dict[str, Any]:
    return get_from_test_infra_generated_stats(
        "file_test_rating.json",    # 文件名为 "file_test_rating.json"
        TEST_FILE_RATINGS_FILE,     # 使用常量 TEST_FILE_RATINGS_FILE 指定的文件路径
        "Couldn't download test file ratings file, not reordering...",  # 下载失败时的错误消息
    )


# 从测试基础设施生成的统计数据中获取测试类评级的字典
def get_test_class_ratings() -> dict[str, Any]:
    return get_from_test_infra_generated_stats(
        "file_test_class_rating.json",  # 文件名为 "file_test_class_rating.json"
        TEST_CLASS_RATINGS_FILE,        # 使用常量 TEST_CLASS_RATINGS_FILE 指定的文件路径
        "Couldn't download test class ratings file, not reordering...",  # 下载失败时的错误消息
    )


# 获取历史编辑文件的启发式处理数据的字典
def get_td_heuristic_historial_edited_files_json() -> dict[str, Any]:
    return get_from_test_infra_generated_stats(
        "td_heuristic_historical_edited_files.json",  # 文件名为 "td_heuristic_historical_edited_files.json"
        TD_HEURISTIC_HISTORICAL_EDITED_FILES,        # 使用常量 TD_HEURISTIC_HISTORICAL_EDITED_FILES 指定的文件路径
        "Couldn't download td_heuristic_historical_edited_files.json, not reordering...",  # 下载失败时的错误消息
    )


# 获取启发式分析数据的字典
def get_td_heuristic_profiling_json() -> dict[str, Any]:
    return get_from_test_infra_generated_stats(
        "td_heuristic_profiling.json",   # 文件名为 "td_heuristic_profiling.json"
        TD_HEURISTIC_PROFILING_FILE,     # 使用常量 TD_HEURISTIC_PROFILING_FILE 指定的文件路径
        "Couldn't download td_heuristic_profiling.json not reordering...",  # 下载失败时的错误消息
    )


# 复制 pytest 缓存中最近失败的测试结果到指定位置
def copy_pytest_cache() -> None:
    original_path = REPO_ROOT / ".pytest_cache/v/cache/lastfailed"
    if not original_path.exists():  # 如果源文件不存在，则直接返回
        return
    shutil.copyfile(
        original_path,  # 源文件路径
        REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER / TD_HEURISTIC_PREVIOUSLY_FAILED,  # 目标文件路径
    )


# 复制额外的之前失败的测试结果文件到指定位置
def copy_additional_previous_failures() -> None:
    original_path = REPO_ROOT / ".pytest_cache" / TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL
    if not original_path.exists():  # 如果源文件不存在，则直接返回
        return
    # 使用 shutil 模块中的 copyfile 函数，将文件从原始路径复制到指定目标路径
    shutil.copyfile(
        # 原始文件的路径
        original_path,
        # 目标路径，使用了 REPO_ROOT 变量作为根路径，然后拼接 ADDITIONAL_CI_FILES_FOLDER 和 TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL 字符串来确定最终路径
        REPO_ROOT
        / ADDITIONAL_CI_FILES_FOLDER
        / TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL,
    )
# 定义函数，从测试基础设施生成的统计信息获取数据
def get_from_test_infra_generated_stats(
    # 参数：来源文件名，目标文件名，失败说明文字
    from_file: str, to_file: str, failure_explanation: str
) -> dict[str, Any]:
    # 构建 URL，指向 raw.githubusercontent.com 上的特定文件
    url = f"https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/{from_file}"
    try:
        # 调用 fetch_and_cache 函数从指定 URL 获取数据并缓存到特定位置
        return fetch_and_cache(
            REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER, to_file, url, lambda x: x
        )
    except Exception:
        # 发生异常时打印失败说明文字
        print(failure_explanation)
        # 返回空字典
        return {}
```