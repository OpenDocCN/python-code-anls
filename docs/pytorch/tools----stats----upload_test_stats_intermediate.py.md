# `.\pytorch\tools\stats\upload_test_stats_intermediate.py`

```py
import argparse  # 导入命令行参数解析模块
import sys  # 导入系统相关的模块

from tools.stats.test_dashboard import upload_additional_info  # 导入上传额外信息的函数
from tools.stats.upload_test_stats import get_tests  # 导入获取测试统计数据的函数

if __name__ == "__main__":
    # 创建命令行参数解析器，设置描述信息
    parser = argparse.ArgumentParser(description="Upload test stats to Rockset")
    
    # 添加命令行参数：workflow-run-id，必填，用于获取工作流程中的数据
    parser.add_argument(
        "--workflow-run-id",
        required=True,
        help="id of the workflow to get artifacts from",
    )
    
    # 添加命令行参数：workflow-run-attempt，必填，表示工作流重试的次数
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    
    # 解析命令行参数
    args = parser.parse_args()

    # 打印工作流程 ID
    print(f"Workflow id is: {args.workflow_run_id}")

    # 调用函数获取测试用例数据，传入工作流 ID 和重试次数
    test_cases = get_tests(args.workflow_run_id, args.workflow_run_attempt)

    # 刷新标准输出，确保 Rockset 上传中的任何错误最后显示在日志中
    sys.stdout.flush()

    # 调用函数上传额外信息到 Rockset，传入工作流 ID、重试次数和测试用例数据
    upload_additional_info(args.workflow_run_id, args.workflow_run_attempt, test_cases)
```