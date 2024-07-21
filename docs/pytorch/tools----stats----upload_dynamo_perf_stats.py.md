# `.\pytorch\tools\stats\upload_dynamo_perf_stats.py`

```py
from __future__ import annotations
# 导入将来版本的类型注解支持

import argparse
# 导入命令行参数解析模块
import csv
# 导入 CSV 文件处理模块
import os
# 导入操作系统功能模块
import re
# 导入正则表达式模块
from pathlib import Path
# 从路径处理模块中导入 Path 类
from tempfile import TemporaryDirectory
# 从临时目录模块中导入 TemporaryDirectory 类
from typing import Any
# 导入 Any 类型，用于表示任意类型

from tools.stats.upload_stats_lib import download_s3_artifacts, unzip, upload_to_rockset
# 从上传统计库中导入下载 S3 文件、解压和上传到 Rockset 的函数

ARTIFACTS = [
    "test-reports",
]
# 定义一个包含测试报告的列表

ARTIFACT_REGEX = re.compile(
    r"test-reports-test-(?P<name>[\w\-]+)-\d+-\d+-(?P<runner>[\w\.]+)_(?P<job>\d+).zip"
)
# 定义一个正则表达式，用于匹配测试报告的文件名格式


def upload_dynamo_perf_stats_to_rockset(
    repo: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    head_branch: str,
    match_filename: str,
) -> list[dict[str, Any]]:
    # 函数定义：将 Dynamo 性能统计上传到 Rockset

    match_filename_regex = re.compile(match_filename)
    # 编译给定的匹配文件名的正则表达式
    perf_stats = []
    # 初始化性能统计列表为空列表

    with TemporaryDirectory() as temp_dir:
        # 使用临时目录作为上下文管理器
        print("Using temporary directory:", temp_dir)
        # 打印使用的临时目录路径

        os.chdir(temp_dir)
        # 切换工作目录到临时目录

        for artifact in ARTIFACTS:
            # 遍历每个 artifact（测试报告文件夹）
            artifact_paths = download_s3_artifacts(
                artifact, workflow_run_id, workflow_run_attempt
            )
            # 下载 S3 中的 artifact 文件夹内容

            # 解压以获取性能统计的 CSV 文件
            for path in artifact_paths:
                # 遍历每个 artifact 的路径
                m = ARTIFACT_REGEX.match(str(path))
                # 尝试匹配路径名与预定义的测试报告文件名格式

                if not m:
                    # 如果匹配失败
                    print(f"Test report {path} has an invalid name. Skipping")
                    # 打印无效文件名的测试报告路径名，并跳过此文件
                    continue

                test_name = m.group("name")
                # 获取测试名称
                runner = m.group("runner")
                # 获取运行器信息
                job_id = m.group("job")
                # 获取任务 ID

                # 提取所有文件
                unzip(path)
                # 解压文件

                for csv_file in Path(".").glob("**/*.csv"):
                    # 遍历当前目录及其子目录中的所有 CSV 文件
                    filename = os.path.splitext(os.path.basename(csv_file))[0]
                    # 提取文件名（不包括扩展名）作为 CSV 文件的基本名称

                    if not re.match(match_filename_regex, filename):
                        # 如果文件名不匹配指定的匹配文件名正则表达式
                        continue
                        # 跳过此文件

                    print(f"Processing {filename} from {path}")
                    # 打印正在处理的文件名和路径

                    with open(csv_file) as csvfile:
                        # 打开 CSV 文件
                        reader = csv.DictReader(csvfile, delimiter=",")
                        # 使用逗号作为分隔符创建 CSV 字典读取器

                        for row in reader:
                            # 遍历 CSV 文件中的每一行
                            row.update(
                                {
                                    "workflow_id": workflow_run_id,  # type: ignore[dict-item]
                                    "run_attempt": workflow_run_attempt,  # type: ignore[dict-item]
                                    "test_name": test_name,
                                    "runner": runner,
                                    "job_id": job_id,
                                    "filename": filename,
                                    "head_branch": head_branch,
                                }
                            )
                            # 更新行数据，添加工作流 ID、运行尝试、测试名称、运行器、任务 ID、文件名和头分支信息
                            perf_stats.append(row)
                            # 将更新后的行数据添加到性能统计列表中

                    # 完成文件处理后，删除文件
                    os.remove(csv_file)
                    # 删除 CSV 文件

    return perf_stats
    # 返回性能统计列表


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload dynamo perf stats from S3 to Rockset"
    )
    # 创建参数解析器，描述为“从 S3 上传 Dynamo 性能统计到 Rockset”
    parser.add_argument(
        "--workflow-run-id",
        type=int,
        required=True,
        help="id of the workflow to get perf stats from",
    )
    # 添加命令行参数 --workflow-run-id，类型为整数，必需，用于指定工作流运行的ID，从中获取性能统计信息

    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    # 添加命令行参数 --workflow-run-attempt，类型为整数，必需，表示工作流的重试次数

    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="which GitHub repo this workflow run belongs to",
    )
    # 添加命令行参数 --repo，类型为字符串，必需，指定当前工作流运行所属的 GitHub 仓库

    parser.add_argument(
        "--head-branch",
        type=str,
        required=True,
        help="head branch of the workflow",
    )
    # 添加命令行参数 --head-branch，类型为字符串，必需，表示工作流的主分支名称

    parser.add_argument(
        "--rockset-collection",
        type=str,
        required=True,
        help="the name of the Rockset collection to store the stats",
    )
    # 添加命令行参数 --rockset-collection，类型为字符串，必需，指定存储统计数据的 Rockset 集合的名称

    parser.add_argument(
        "--rockset-workspace",
        type=str,
        default="commons",
        help="the name of the Rockset workspace to store the stats",
    )
    # 添加命令行参数 --rockset-workspace，类型为字符串，默认为 "commons"，指定存储统计数据的 Rockset 工作区的名称

    parser.add_argument(
        "--match-filename",
        type=str,
        default="",
        help="the regex to filter the list of CSV files containing the records to upload",
    )
    # 添加命令行参数 --match-filename，类型为字符串，默认为空字符串，指定用于筛选包含要上传记录的 CSV 文件列表的正则表达式

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 对象中

    perf_stats = upload_dynamo_perf_stats_to_rockset(
        args.repo,
        args.workflow_run_id,
        args.workflow_run_attempt,
        args.head_branch,
        args.match_filename,
    )
    # 调用 upload_dynamo_perf_stats_to_rockset 函数，传递从命令行参数获取的 repo、workflow_run_id、workflow_run_attempt、head_branch 和 match_filename 参数，并获取性能统计信息

    upload_to_rockset(
        collection=args.rockset_collection,
        docs=perf_stats,
        workspace=args.rockset_workspace,
    )
    # 调用 upload_to_rockset 函数，将从 Rockset 收集到的 perf_stats 文档上传到指定的 Rockset 集合和工作区
```