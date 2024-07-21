# `.\pytorch\tools\stats\upload_test_stats.py`

```py
# 从未来模块导入 annotations，支持类型注解
from __future__ import annotations

# 导入命令行参数解析模块
import argparse
# 导入操作系统相关功能
import os
# 导入系统相关功能
import sys
# 导入 XML 解析模块
import xml.etree.ElementTree as ET
# 导入多进程相关功能
from multiprocessing import cpu_count, Pool
# 导入路径操作相关功能
from pathlib import Path
# 导入临时目录功能
from tempfile import TemporaryDirectory
# 导入类型提示相关功能
from typing import Any

# 导入测试工具中的上传额外信息函数
from tools.stats.test_dashboard import upload_additional_info
# 导入统计上传库中的功能：下载 S3 产物、获取作业 ID、解压缩、上传工作流统计到 S3
from tools.stats.upload_stats_lib import (
    download_s3_artifacts,
    get_job_id,
    unzip,
    upload_workflow_stats_to_s3,
)


def parse_xml_report(
    tag: str,
    report: Path,
    workflow_id: int,
    workflow_run_attempt: int,
) -> list[dict[str, Any]]:
    """Convert a test report xml file into a JSON-serializable list of test cases."""
    # 打印正在解析的测试报告信息
    print(f"Parsing {tag}s for test report: {report}")

    # 获取测试报告对应的作业 ID
    job_id = get_job_id(report)
    # 打印找到的作业 ID
    print(f"Found job id: {job_id}")

    # 初始化测试用例列表
    test_cases: list[dict[str, Any]] = []

    # 解析 XML 报告的根节点
    root = ET.parse(report)
    # 遍历 XML 报告中指定标签下的每个测试用例
    for test_case in root.iter(tag):
        # 处理 XML 元素，将其转换为字典格式
        case = process_xml_element(test_case)
        # 将工作流 ID、工作流运行尝试次数和作业 ID 添加到测试用例中
        case["workflow_id"] = workflow_id
        case["workflow_run_attempt"] = workflow_run_attempt
        case["job_id"] = job_id

        # [invoking file]
        # 测试所在的文件名不一定与调用测试的文件名相同。
        # 例如，`test_jit.py` 调用了多个其他测试文件（例如 jit/test_dce.py）。
        # 为了分片/测试选择的目的，我们想记录调用测试的文件名。
        #
        # 为此，我们利用了我们编写测试的方式的实现细节（https://bit.ly/3ajEV1M），
        # 即报告是在与调用文件名相同的文件夹下创建的。
        case["invoking_file"] = report.parent.name
        # 将处理后的测试用例添加到测试用例列表中
        test_cases.append(case)

    # 返回包含所有测试用例的列表
    return test_cases


def process_xml_element(element: ET.Element) -> dict[str, Any]:
    """Convert a test suite element into a JSON-serializable dict."""
    # 初始化返回的字典
    ret: dict[str, Any] = {}

    # 将 XML 元素的属性直接转换为字典元素
    # 例如：
    #     <testcase name="test_foo" classname="test_bar"></testcase>
    # 转换为：
    #     {"name": "test_foo", "classname": "test_bar"}
    ret.update(element.attrib)

    # XML 格式将所有值编码为字符串。如果可能，将其转换为整数/浮点数，以便在 Rockset 中进行聚合。
    for k, v in ret.items():
        try:
            ret[k] = int(v)
        except ValueError:
            pass
        try:
            ret[k] = float(v)
        except ValueError:
            pass

    # 将内部文本和尾随文本转换为特殊的字典元素
    # 例如：
    #     <testcase>my_inner_text</testcase> my_tail
    # 转换为：
    #     {"text": "my_inner_text", "tail": " my_tail"}
    if element.text and element.text.strip():
        ret["text"] = element.text
    if element.tail and element.tail.strip():
        ret["tail"] = element.tail

    # 递归地转换子元素，并将其放置在一个键下
    # 例如：
    #     <testcase>
    # 遍历 XML 元素的每一个子元素
    for child in element:
        # 如果当前子元素的标签名不在结果字典 ret 中
        if child.tag not in ret:
            # 将当前子元素递归处理后的结果放入 ret 中
            ret[child.tag] = process_xml_element(child)
        else:
            # 如果同名标签已经存在于 ret 中，则需要将其值处理为列表
            if not isinstance(ret[child.tag], list):
                ret[child.tag] = [ret[child.tag]]
            # 将当前子元素递归处理后的结果追加到同名标签的列表中
            ret[child.tag].append(process_xml_element(child))
    # 返回处理完所有子元素后的结果字典 ret
    return ret
def get_tests(workflow_run_id: int, workflow_run_attempt: int) -> list[dict[str, Any]]:
    # 使用临时目录作为工作目录
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        # 将当前工作目录更改为临时目录
        os.chdir(temp_dir)

        # 下载并解压所有报告（包括 GHA 和 S3）
        s3_paths = download_s3_artifacts(
            "test-report", workflow_run_id, workflow_run_attempt
        )
        for path in s3_paths:
            # 对每个路径解压缩文件
            unzip(path)

        # 解析报告并转换为 JSON 格式
        test_cases = []
        # 使用多进程池并行处理
        mp = Pool(cpu_count())
        for xml_report in Path(".").glob("**/*.xml"):
            test_cases.append(
                mp.apply_async(
                    parse_xml_report,
                    args=(
                        "testcase",
                        xml_report,
                        workflow_run_id,
                        workflow_run_attempt,
                    ),
                )
            )
        mp.close()
        mp.join()
        # 获取异步处理结果并扁平化列表
        test_cases = [tc.get() for tc in test_cases]
        flattened = [item for sublist in test_cases for item in sublist]
        return flattened


def get_tests_for_circleci(
    workflow_run_id: int, workflow_run_attempt: int
) -> list[dict[str, Any]]:
    # 解析报告并转换为 JSON 格式
    test_cases = []
    # 遍历指定路径下所有 XML 报告文件
    for xml_report in Path(".").glob("**/test/test-reports/**/*.xml"):
        test_cases.extend(
            # 解析每个 XML 报告文件并添加到测试用例列表
            parse_xml_report(
                "testcase", xml_report, workflow_run_id, workflow_run_attempt
            )
        )

    return test_cases


def summarize_test_cases(test_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group test cases by classname, file, and job_id. We perform the aggregation
    manually instead of using the `test-suite` XML tag because xmlrunner does
    not produce reliable output for it.
    """

    def get_key(test_case: dict[str, Any]) -> Any:
        # 获取用于分组的关键字，包括文件名、类名、作业 ID 等
        return (
            test_case.get("file"),
            test_case.get("classname"),
            test_case["job_id"],
            test_case["workflow_id"],
            test_case["workflow_run_attempt"],
            # [see: invoking file]
            test_case["invoking_file"],
        )

    def init_value(test_case: dict[str, Any]) -> dict[str, Any]:
        # 初始化聚合值，包括测试计数、失败数、错误数、跳过数、成功数、时间等
        return {
            "file": test_case.get("file"),
            "classname": test_case.get("classname"),
            "job_id": test_case["job_id"],
            "workflow_id": test_case["workflow_id"],
            "workflow_run_attempt": test_case["workflow_run_attempt"],
            # [see: invoking file]
            "invoking_file": test_case["invoking_file"],
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "successes": 0,
            "time": 0.0,
        }

    # 返回的最终聚合结果字典
    ret = {}
    # 遍历测试用例列表，处理每个测试用例
    for test_case in test_cases:
        # 调用函数获取测试用例的关键字
        key = get_key(test_case)
        
        # 如果关键字不在返回字典中，初始化其值
        if key not in ret:
            ret[key] = init_value(test_case)

        # 增加该关键字对应的测试数目
        ret[key]["tests"] += 1

        # 根据测试结果类型增加相应计数
        if "failure" in test_case:
            ret[key]["failures"] += 1
        elif "error" in test_case:
            ret[key]["errors"] += 1
        elif "skipped" in test_case:
            ret[key]["skipped"] += 1
        else:
            ret[key]["successes"] += 1

        # 累加该关键字对应的总耗时
        ret[key]["time"] += test_case["time"]
    
    # 将字典的值转换为列表并返回
    return list(ret.values())
if __name__ == "__main__":
    # 解析命令行参数，描述是上传测试统计到 Rockset
    parser = argparse.ArgumentParser(description="Upload test stats to Rockset")
    
    # 添加命令行参数：workflow-run-id，必须提供，用于获取工作流的工件
    parser.add_argument(
        "--workflow-run-id",
        required=True,
        help="id of the workflow to get artifacts from",
    )
    
    # 添加命令行参数：workflow-run-attempt，必须提供，表示工作流的重试次数
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    
    # 添加命令行参数：head-branch，必须提供，表示工作流的主分支
    parser.add_argument(
        "--head-branch",
        required=True,
        help="Head branch of the workflow",
    )
    
    # 添加命令行参数：head-repository，必须提供，表示工作流的主仓库
    parser.add_argument(
        "--head-repository",
        required=True,
        help="Head repository of the workflow",
    )
    
    # 添加命令行参数：circleci，如果存在则设置为 True，表示通过 CircleCI 运行
    parser.add_argument(
        "--circleci",
        action="store_true",
        help="If this is being run through circleci",
    )
    
    # 解析命令行参数
    args = parser.parse_args()

    # 打印工作流的 id
    print(f"Workflow id is: {args.workflow_run_id}")

    # 根据是否通过 CircleCI 运行来选择获取测试用例的函数
    if args.circleci:
        test_cases = get_tests_for_circleci(
            args.workflow_run_id, args.workflow_run_attempt
        )
    else:
        test_cases = get_tests(args.workflow_run_id, args.workflow_run_attempt)

    # 刷新 stdout，以便在日志中最后显示 Rockset 上传时的任何错误
    sys.stdout.flush()

    # 对测试用例进行汇总，如果是 PR，则只上传测试运行的摘要，以减少对 Rockset 的写入量
    test_case_summary = summarize_test_cases(test_cases)

    # 将测试运行摘要上传到 S3
    upload_workflow_stats_to_s3(
        args.workflow_run_id,
        args.workflow_run_attempt,
        "test_run_summary",
        test_case_summary,
    )

    # 将失败的测试用例分离出来，因为上传所有内容会过于数据密集，而这些只是极少量
    failed_tests_cases = []
    for test_case in test_cases:
        if "rerun" in test_case or "failure" in test_case or "error" in test_case:
            failed_tests_cases.append(test_case)

    # 将失败的测试运行上传到 S3
    upload_workflow_stats_to_s3(
        args.workflow_run_id,
        args.workflow_run_attempt,
        "failed_test_runs",
        failed_tests_cases,
    )

    # 如果是在主分支 main 上并且仓库是 pytorch/pytorch，上传所有测试运行到 S3
    if args.head_branch == "main" and args.head_repository == "pytorch/pytorch":
        upload_workflow_stats_to_s3(
            args.workflow_run_id, args.workflow_run_attempt, "test_run", test_cases
        )

    # 上传额外的信息到 S3
    upload_additional_info(args.workflow_run_id, args.workflow_run_attempt, test_cases)
```