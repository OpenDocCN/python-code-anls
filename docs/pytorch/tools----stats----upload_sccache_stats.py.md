# `.\pytorch\tools\stats\upload_sccache_stats.py`

```py
# 从未来版本导入类型注解的支持
from __future__ import annotations

# 导入命令行参数解析模块
import argparse
# 导入处理 JSON 的模块
import json
# 导入操作系统相关的模块
import os
# 导入处理路径的模块
from pathlib import Path
# 导入临时目录管理模块
from tempfile import TemporaryDirectory
# 导入类型提示模块中的 Any 类型
from typing import Any

# 从自定义模块导入函数
from tools.stats.upload_stats_lib import (
    download_s3_artifacts,
    upload_workflow_stats_to_s3,
)

# 定义函数，获取 sccache 统计信息
def get_sccache_stats(
    workflow_run_id: int, workflow_run_attempt: int
) -> list[dict[str, Any]]:
    # 使用临时目录作为工作目录
    with TemporaryDirectory() as temp_dir:
        # 打印临时目录路径
        print("Using temporary directory:", temp_dir)
        # 切换当前工作目录到临时目录
        os.chdir(temp_dir)

        # 下载并解压所有报告（包括 GHA 和 S3 中的）
        download_s3_artifacts("sccache-stats", workflow_run_id, workflow_run_attempt)

        # 初始化空列表，用于存储所有 JSON 文件的内容
        stats_jsons = []
        # 遍历临时目录下所有的 JSON 文件
        for json_file in Path(".").glob("**/*.json"):
            # 打开 JSON 文件并加载其内容，将加载的字典对象添加到列表中
            with open(json_file) as f:
                stats_jsons.append(json.load(f))
        
        # 返回包含所有 JSON 文件内容的列表
        return stats_jsons


# 主程序入口
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser(description="Upload test stats to Rockset")
    # 添加命令行参数：workflow-run-id，整数类型，必填，描述为获取工作流程中的构件 id
    parser.add_argument(
        "--workflow-run-id",
        type=int,
        required=True,
        help="id of the workflow to get artifacts from",
    )
    # 添加命令行参数：workflow-run-attempt，整数类型，必填，描述为工作流重试的次数
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数获取 sccache 统计信息，传入命令行参数中指定的工作流 id 和重试次数
    stats = get_sccache_stats(args.workflow_run_id, args.workflow_run_attempt)
    # 调用函数将 sccache 统计信息上传到 S3 中，同时指定工作流 id、重试次数和统计数据类型
    upload_workflow_stats_to_s3(
        args.workflow_run_id, args.workflow_run_attempt, "sccache_stats", stats
    )
```