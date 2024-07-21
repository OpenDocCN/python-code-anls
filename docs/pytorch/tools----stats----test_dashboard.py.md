# `.\pytorch\tools\stats\test_dashboard.py`

```
# 从__future__模块导入annotations，支持类型注解
from __future__ import annotations

# 导入必要的模块和类
import json  # 导入处理 JSON 格式数据的模块
import os  # 导入操作系统功能的模块
import re  # 导入正则表达式模块
import time  # 导入时间相关功能的模块
from collections import defaultdict  # 导入默认字典功能
from functools import lru_cache  # 导入 LRU 缓存功能
from pathlib import Path  # 导入处理路径的模块
from tempfile import TemporaryDirectory  # 导入临时目录功能
from typing import Any, cast  # 导入类型相关功能

import requests  # 导入处理 HTTP 请求的库

# 从自定义模块中导入特定函数和类
from tools.stats.upload_stats_lib import (
    _get_request_headers,  # 导入获取请求头函数
    download_s3_artifacts,  # 导入下载 S3 存储中文件的函数
    get_job_id,  # 导入获取任务 ID 的函数
    unzip,  # 导入解压文件函数
    upload_workflow_stats_to_s3,  # 导入上传工作流统计数据到 S3 的函数
)

# 定义匹配 GitHub Actions 任务信息的正则表达式模式
REGEX_JOB_INFO = r"(.*) \/ .*test \(([^,]*), .*\)"

# 使用 LRU 缓存装饰器缓存根据任务 ID 获取任务名称的函数
@lru_cache(maxsize=1000)
def get_job_name(job_id: int) -> str:
    try:
        # 发送 HTTP GET 请求获取 GitHub 任务信息，并返回任务名称
        return cast(
            str,
            requests.get(
                f"https://api.github.com/repos/pytorch/pytorch/actions/jobs/{job_id}",
                headers=_get_request_headers(),  # 获取请求头信息
            ).json()["name"],
        )
    except Exception as e:
        # 若获取任务名称失败，则打印错误信息并返回默认值
        print(f"Failed to get job name for job id {job_id}: {e}")
        return "NoJobName"


# 使用 LRU 缓存装饰器缓存根据任务名称获取构建名称的函数
@lru_cache(maxsize=1000)
def get_build_name(job_name: str) -> str:
    try:
        # 使用正则表达式从任务名称中匹配并返回构建名称
        return re.match(REGEX_JOB_INFO, job_name).group(1)  # type: ignore[union-attr]
    except AttributeError:
        # 若匹配失败，则打印错误信息并返回默认值
        print(f"Failed to match job name: {job_name}")
        return "NoBuildEnv"


# 使用 LRU 缓存装饰器缓存根据任务名称获取测试配置名称的函数
@lru_cache(maxsize=1000)
def get_test_config(job_name: str) -> str:
    try:
        # 使用正则表达式从任务名称中匹配并返回测试配置名称
        return re.match(REGEX_JOB_INFO, job_name).group(2)  # type: ignore[union-attr]
    except AttributeError:
        # 若匹配失败，则打印错误信息并返回默认值
        print(f"Failed to match job name: {job_name}")
        return "NoTestConfig"


# 定义根据工作流运行 ID 和尝试次数获取测试数据排除信息的函数
def get_td_exclusions(
    workflow_run_id: int, workflow_run_attempt: int
) -> dict[str, Any]:
    # 使用临时目录作为工作目录
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        os.chdir(temp_dir)  # 将当前工作目录更改为临时目录

        # 下载并解压所有报告文件（包括 GHA 和 S3 中的文件）
        s3_paths = download_s3_artifacts(
            "test-jsons", workflow_run_id, workflow_run_attempt
        )
        for path in s3_paths:
            unzip(path)  # 解压文件

        # 初始化用于分组测试数据排除信息的字典
        grouped_tests: dict[str, Any] = defaultdict(lambda: defaultdict(set))
        # 遍历临时目录下所有的 td_exclusions*.json 文件
        for td_exclusions in Path(".").glob("**/td_exclusions*.json"):
            with open(td_exclusions) as f:
                exclusions = json.load(f)  # 加载 JSON 文件数据
                # 遍历排除信息列表，并获取相关任务、构建和测试配置信息
                for exclusion in exclusions["excluded"]:
                    job_id = get_job_id(td_exclusions)  # 获取任务 ID
                    job_name = get_job_name(job_id)  # 获取任务名称
                    build_name = get_build_name(job_name)  # 获取构建名称
                    test_config = get_test_config(job_name)  # 获取测试配置名称
                    # 将测试文件添加到对应的构建和测试配置的集合中
                    grouped_tests[build_name][test_config].add(exclusion["test_file"])

        # 对分组后的测试数据排除信息进行排序并返回
        for build_name, build in grouped_tests.items():
            for test_config, test_files in build.items():
                grouped_tests[build_name][test_config] = sorted(test_files)
        return grouped_tests


# 定义将测试用例分组的函数，返回值为分组后的测试用例信息
def group_test_cases(test_cases: list[dict[str, Any]]) -> dict[str, Any]:
    start = time.time()  # 记录函数开始时间
    # 使用 defaultdict 嵌套创建了一个多层深度的字典结构 grouped_tests，用于存储按照不同维度分组的测试用例数据
    grouped_tests: dict[str, Any] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
    )

    # 遍历每个测试用例
    for test_case in test_cases:
        # 获取测试作业的名称
        job_name = get_job_name(test_case["job_id"])
        # 根据作业名称获取构建名称
        build_name = get_build_name(job_name)
        # 如果构建名称中包含 "bazel"，则跳过当前测试用例
        if "bazel" in build_name:
            continue
        # 获取测试配置信息
        test_config = get_test_config(job_name)
        # 弹出并获取测试用例的类名，如果没有则默认为 "NoClass"
        class_name = test_case.pop("classname", "NoClass")
        # 弹出并获取测试用例的名称，如果没有则默认为 "NoName"
        name = test_case.pop("name", "NoName")
        # 弹出并获取测试用例的调用文件名，如果没有则默认为 "NoFile"，并将其中的点替换为斜杠
        invoking_file = test_case.pop("invoking_file", "NoFile")
        invoking_file = invoking_file.replace(".", "/")
        # 弹出并移除 workflow_id 和 workflow_run_attempt 两个字段
        test_case.pop("workflow_id")
        test_case.pop("workflow_run_attempt")
        # 将当前测试用例添加到 grouped_tests 结构中对应的位置
        grouped_tests[build_name][test_config][invoking_file][class_name][name].append(
            test_case
        )

    # 打印分组测试用时的时间
    print(f"Time taken to group tests: {time.time() - start}")
    # 返回最终的分组测试用例字典
    return grouped_tests
def get_reruns(grouped_tests: dict[str, Any]) -> dict[str, Any]:
    # 初始化 reruns 字典，使用 defaultdict 嵌套实现多层默认字典结构
    reruns: dict[str, Any] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
    )
    # 遍历 grouped_tests 字典
    for build_name, build in grouped_tests.items():
        # 遍历每个 build 下的项目
        for test_config, test_config_data in build.items():
            # 遍历每个 test_config 下的数据
            for invoking_file, invoking_file_data in test_config_data.items():
                # 遍历每个 invoking_file 下的数据
                for class_name, class_data in invoking_file_data.items():
                    # 遍历每个 class_name 下的数据
                    for test_name, test_data in class_data.items():
                        # 如果一个测试用例有多个运行结果
                        if len(test_data) > 1:
                            # 如果 invoking_file 在指定的列表中，跳过处理
                            if invoking_file in (
                                "distributed/test_distributed_spawn",
                                "onnx/test_fx_to_onnx_with_onnxruntime",
                                "distributed/algorithms/quantization/test_quantization",
                            ):
                                continue
                            # 将多次运行的测试数据保存到 reruns 结构中
                            reruns[build_name][test_config][invoking_file][class_name][
                                test_name
                            ] = test_data
    return reruns


def get_invoking_file_summary(grouped_tests: dict[str, Any]) -> dict[str, Any]:
    # 初始化 invoking_file_summary 字典，使用 defaultdict 嵌套实现多层默认字典结构
    invoking_file_summary: dict[str, Any] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"count": 0, "time": 0.0}))
    )
    # 遍历 grouped_tests 字典
    for build_name, build in grouped_tests.items():
        # 遍历每个 build 下的项目
        for test_config, test_config_data in build.items():
            # 遍历每个 test_config 下的数据
            for invoking_file, invoking_file_data in test_config_data.items():
                # 遍历每个 invoking_file 下的数据
                for class_data in invoking_file_data.values():
                    # 遍历每个 class_data 下的值
                    for test_data in class_data.values():
                        # 计算每个 invoking_file 的测试运行次数和总耗时
                        invoking_file_summary[build_name][test_config][invoking_file][
                            "count"
                        ] += 1
                        for i in test_data:
                            # 累加每个测试用例的耗时
                            invoking_file_summary[build_name][test_config][
                                invoking_file
                            ]["time"] += i["time"]

    return invoking_file_summary


def upload_additional_info(
    workflow_run_id: int, workflow_run_attempt: int, test_cases: list[dict[str, Any]]
) -> None:
    # 将测试用例按照一定规则进行分组
    grouped_tests = group_test_cases(test_cases)
    # 获取测试用例中存在多次运行的信息
    reruns = get_reruns(grouped_tests)
    # 获取测试用例的排除信息
    exclusions = get_td_exclusions(workflow_run_id, workflow_run_attempt)
    # 获取测试用例中每个 invoking_file 的汇总信息
    invoking_file_summary = get_invoking_file_summary(grouped_tests)

    # 将 reruns 结果上传到 S3
    upload_workflow_stats_to_s3(
        workflow_run_id,
        workflow_run_attempt,
        "additional_info/reruns",
        [reruns],
    )
    # 将 exclusions 结果上传到 S3
    upload_workflow_stats_to_s3(
        workflow_run_id,
        workflow_run_attempt,
        "additional_info/td_exclusions",
        [exclusions],
    )
    # 将 invoking_file_summary 结果上传到 S3
    upload_workflow_stats_to_s3(
        workflow_run_id,
        workflow_run_attempt,
        "additional_info/invoking_file_summary",
        [invoking_file_summary],
    )
```