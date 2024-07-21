# `.\pytorch\tools\stats\check_disabled_tests.py`

```py
# 引入未来的类型注解允许使用类名称作为类型标注，以支持类型的自我引用
from __future__ import annotations

# 导入命令行参数解析库
import argparse
# 导入 JSON 序列化和反序列化库
import json
# 导入操作系统相关功能的库
import os
# 导入 XML 解析库的 ElementTree 模块
import xml.etree.ElementTree as ET
# 导入操作路径的 Path 类
from pathlib import Path
# 导入临时目录的临时文件夹功能
from tempfile import TemporaryDirectory
# 导入 Any 类型，用于表示可以是任何类型的变量
from typing import Any, Generator

# 从工具包中导入统计上传库
from tools.stats.upload_stats_lib import (
    download_s3_artifacts,    # 导入从 S3 下载工件的函数
    is_rerun_disabled_tests,  # 导入检查是否禁用重新运行测试的函数
    unzip,                    # 导入解压函数
    upload_workflow_stats_to_s3,  # 导入上传工作流统计到 S3 的函数
)
# 从工具包中导入上传测试统计的函数
from tools.stats.upload_test_stats import process_xml_element

# 定义测试用例标签名称
TESTCASE_TAG = "testcase"
# 定义分隔符为分号
SEPARATOR = ";"


def process_report(
    report: Path,
) -> dict[str, dict[str, int]]:
    """
    返回应重新启用的禁用测试和仍然不稳定的测试（失败或跳过）的列表
    """
    # 解析给定路径的 XML 报告文件，并返回根元素
    root = ET.parse(report)

    # 所有重新运行测试从报告中汇总在这里：
    #
    # * 成功测试应在重新运行后的所有平台上都是绿色的，应重新启用
    #   当前在其中被禁用的平台
    # * 来自 pytest 的失败，因为 pytest-flakefinder 用于多次运行相同的测试，
    #   一些可能会失败
    # * 从 unittest 跳过的测试
    #
    # 我们希望跟踪测试失败的次数（num_red）或通过的次数（num_green）
    all_tests: dict[str, dict[str, int]] = {}
    # 遍历根元素下所有的测试用例元素
    for test_case in root.iter(TESTCASE_TAG):
        # 解析 XML 元素并处理
        parsed_test_case = process_xml_element(test_case)

        # 在 --rerun-disabled-tests 模式下，当满足以下条件时跳过测试用例：
        # * 在 PyTorch 代码中被显式跳过
        # * 作为正常启用的测试被跳过
        # * 或者测试是不稳定的（num_red > 0 并且 num_green > 0）
        # * 或者测试失败（num_red > 0 并且 num_green == 0）
        #
        # 这里只关心最后两种情况
        skipped = parsed_test_case.get("skipped", None)

        # 注意：对于常规 ONNX 测试，这里可能会返回一个子跳过列表，其中每个项目都是一个跳过消息。
        # 在重新运行禁用的测试的上下文中，我们可以忽略这种情况，因为只有在正常运行测试时才会返回子跳过列表。
        if skipped and (
            type(skipped) is list or "num_red" not in skipped.get("message", "")
        ):
            continue

        # 获取测试用例的名称、类名和文件名
        name = parsed_test_case.get("name", "")
        classname = parsed_test_case.get("classname", "")
        filename = parsed_test_case.get("file", "")

        # 如果名称、类名或文件名为空，则跳过该测试用例
        if not name or not classname or not filename:
            continue

        # 检查测试是否失败
        failure = parsed_test_case.get("failure", None)

        # 生成禁用的测试用例的唯一标识符
        disabled_test_id = SEPARATOR.join([name, classname, filename])

        # 如果该禁用的测试用例不在 all_tests 中，则初始化其计数器
        if disabled_test_id not in all_tests:
            all_tests[disabled_test_id] = {
                "num_green": 0,
                "num_red": 0,
            }

        # 在 --rerun-disabled-tests 模式下，如果测试未被跳过或失败，则视为成功。否则，它仍然是不稳定或失败的
        if skipped:
            try:
                # 解析跳过消息中的统计信息
                stats = json.loads(skipped.get("message", ""))
            except json.JSONDecodeError:
                stats = {}

            # 更新禁用的测试用例的计数器：绿色计数和红色计数
            all_tests[disabled_test_id]["num_green"] += stats.get("num_green", 0)
            all_tests[disabled_test_id]["num_red"] += stats.get("num_red", 0)
        elif failure:
            # 如果测试失败，增加红色计数
            all_tests[disabled_test_id]["num_red"] += 1
        else:
            # 否则，增加绿色计数
            all_tests[disabled_test_id]["num_green"] += 1

    # 返回所有禁用的测试用例的统计结果字典
    return all_tests
# 定义函数用于从 S3 和 GitHub Actions 下载测试报告文件，并生成路径的生成器
def get_test_reports(
    repo: str, workflow_run_id: int, workflow_run_attempt: int
) -> Generator[Path, None, None]:
    """
    Gather all the test reports from S3 and GHA. It is currently not possible to guess which
    test reports are from rerun_disabled_tests workflow because the name doesn't include the
    test config. So, all reports will need to be downloaded and examined
    """
    # 使用临时目录进行操作
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        # 切换工作目录到临时目录
        os.chdir(temp_dir)

        # 下载 S3 中的测试报告文件到临时目录
        artifact_paths = download_s3_artifacts(
            "test-reports", workflow_run_id, workflow_run_attempt
        )
        # 遍历下载的每个文件并解压缩
        for path in artifact_paths:
            unzip(path)

        # 生成器，返回当前目录及子目录中所有的 XML 文件路径
        yield from Path(".").glob("**/*.xml")


# 定义函数用于从测试 ID 中获取禁用测试的名称、类名和文件名
def get_disabled_test_name(test_id: str) -> tuple[str, str, str, str]:
    """
    Follow flaky bot convention here, if that changes, this will also need to be updated
    """
    # 使用分隔符 SEPARATOR 拆分测试 ID，返回测试名称、类名和文件名
    name, classname, filename = test_id.split(SEPARATOR)
    return f"{name} (__main__.{classname})", name, classname, filename


# 定义函数用于准备保存到 S3 的记录
def prepare_record(
    workflow_id: int,
    workflow_run_attempt: int,
    name: str,
    classname: str,
    filename: str,
    flaky: bool,
    num_red: int = 0,
    num_green: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """
    Prepare the record to save onto S3
    """
    # 构建记录的键
    key = (
        workflow_id,
        workflow_run_attempt,
        name,
        classname,
        filename,
    )

    # 构建记录的字典
    record = {
        "workflow_id": workflow_id,
        "workflow_run_attempt": workflow_run_attempt,
        "name": name,
        "classname": classname,
        "filename": filename,
        "flaky": flaky,
        "num_green": num_green,
        "num_red": num_red,
    }

    return key, record


# 定义函数用于保存测试结果到 S3，以便传送到 Rockset
def save_results(
    workflow_id: int,
    workflow_run_attempt: int,
    all_tests: dict[str, dict[str, int]],
) -> None:
    """
    Save the result to S3, so it can go to Rockset
    """
    # 筛选出符合条件的测试结果和仍然不稳定的测试结果
    should_be_enabled_tests = {
        name: stats
        for name, stats in all_tests.items()
        if "num_green" in stats
        and stats["num_green"]
        and "num_red" in stats
        and stats["num_red"] == 0
    }
    still_flaky_tests = {
        name: stats
        for name, stats in all_tests.items()
        if name not in should_be_enabled_tests
    }

    # 初始化记录字典
    records = {}
    # 遍历所有测试结果
    for test_id, stats in all_tests.items():
        # 获取测试的绿色和红色计数
        num_green = stats.get("num_green", 0)
        num_red = stats.get("num_red", 0)
        # 获取禁用测试的名称、测试名称、类名和文件名
        disabled_test_name, name, classname, filename = get_disabled_test_name(test_id)

        # 准备记录的键和记录内容
        key, record = prepare_record(
            workflow_id=workflow_id,
            workflow_run_attempt=workflow_run_attempt,
            name=name,
            classname=classname,
            filename=filename,
            flaky=test_id in still_flaky_tests,
            num_green=num_green,
            num_red=num_red,
        )
        # 将记录添加到记录字典中
        records[key] = record

    # 记录结果日志
    # (此处为示例，因为代码未提供实际的日志记录语句，所以注释是空白的)
    # 打印应重新启用的测试数量及其相关信息
    print(f"The following {len(should_be_enabled_tests)} tests should be re-enabled:")
    # 遍历应重新启用的测试及其统计信息
    for test_id, stats in should_be_enabled_tests.items():
        # 获取被禁用测试的名称、名称、类名和文件名
        disabled_test_name, name, classname, filename = get_disabled_test_name(test_id)
        # 打印禁用测试的名称和所在文件名
        print(f"  {disabled_test_name} from {filename}")

    # 打印仍然存在问题的测试数量及其相关信息
    print(f"The following {len(still_flaky_tests)} are still flaky:")
    # 遍历仍然存在问题的测试及其统计信息
    for test_id, stats in still_flaky_tests.items():
        # 获取被禁用测试的名称、名称、类名和文件名
        disabled_test_name, name, classname, filename = get_disabled_test_name(test_id)
        # 获取测试通过和失败的次数
        num_green = stats.get("num_green", 0)
        num_red = stats.get("num_red", 0)
        # 打印禁用测试的名称、所在文件名以及失败的次数和总执行次数
        print(
            f"  {disabled_test_name} from {filename}, failing {num_red}/{num_red + num_green}"
        )

    # 将记录上传到 S3，用于重新运行禁用的测试统计
    upload_workflow_stats_to_s3(
        workflow_id,
        workflow_run_attempt,
        "rerun_disabled_tests",
        list(records.values()),
    )
# 主函数，用于处理测试报告并重新启用所有已禁用的测试
def main(repo: str, workflow_run_id: int, workflow_run_attempt: int) -> None:
    """
    Find the list of all disabled tests that should be re-enabled
    寻找所有应重新启用的已禁用测试列表
    """
    # Aggregated across all jobs
    # 跨所有任务汇总的测试结果字典
    all_tests: dict[str, dict[str, int]] = {}

    # 获取所有测试报告并处理
    for report in get_test_reports(
        args.repo, args.workflow_run_id, args.workflow_run_attempt
    ):
        # 处理每个测试报告，返回测试名称和统计信息的字典
        tests = process_report(report)

        # 若该报告不是关于重新运行已禁用测试的报告，跳过处理
        if not is_rerun_disabled_tests(tests):
            continue

        # 将测试名称和统计信息加入到总测试字典中
        for name, stats in tests.items():
            if name not in all_tests:
                all_tests[name] = stats.copy()
            else:
                # 更新总测试字典中已有测试的统计信息
                all_tests[name]["num_green"] += stats.get("num_green", 0)
                all_tests[name]["num_red"] += stats.get("num_red", 0)

    # 保存处理后的结果到指定的存储位置
    save_results(
        workflow_run_id,
        workflow_run_attempt,
        all_tests,
    )


if __name__ == "__main__":
    # 命令行参数解析器，用于获取工作流运行ID、重试次数和GitHub仓库信息
    parser = argparse.ArgumentParser(description="Upload test artifacts from GHA to S3")
    parser.add_argument(
        "--workflow-run-id",
        type=int,
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="which GitHub repo this workflow run belongs to",
    )

    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用主函数，传入解析得到的参数
    main(args.repo, args.workflow_run_id, args.workflow_run_attempt)
```