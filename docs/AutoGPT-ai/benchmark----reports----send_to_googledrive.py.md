# `.\AutoGPT\benchmark\reports\send_to_googledrive.py`

```py
# 导入必要的库
import base64
import json
import os
import re
from datetime import datetime, timedelta
import gspread
import pandas as pd
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials

# 从 .env 文件加载环境变量
load_dotenv()

# 从环境变量中获取 base64 字符串
base64_creds = os.getenv("GDRIVE_BASE64")

# 如果 base64_creds 为空，则抛出数值错误
if base64_creds is None:
    raise ValueError("The GDRIVE_BASE64 environment variable is not set")

# 将 base64 字符串解码为字节
creds_bytes = base64.b64decode(base64_creds)

# 将字节转换为字符串
creds_string = creds_bytes.decode("utf-8")

# 将字符串解析为 JSON 对象
creds_info = json.loads(creds_string)

# 定义包含 JSON 文件的基本目录
base_dir = "reports"

# 获取当前工作目录
current_dir = os.getcwd()

# 检查当前目录是否以 'reports' 结尾
if current_dir.endswith("reports"):
    base_dir = "/"
else:
    base_dir = "reports"

# 创建一个列表来存储每行数据
rows = []

def process_test(
    test_name: str, test_info: dict, agent_name: str, common_data: dict
) -> None:
    """递归函数，用于处理测试数据。"""
    # 通过下划线拆分测试名称，仅拆分一次
    parts = test_name.split("_", 1)
    test_suite = parts[0] if len(parts) > 1 else None

    # 将数组转换为以 | 分隔的字符串
    separator = "|"
    categories = separator.join(
        test_info.get("category", []),
    )
    # 创建一个包含测试结果信息的字典
    row = {
        "Agent": agent_name,  # 代理名称
        "Command": common_data.get("command", ""),  # 命令
        "Completion Time": common_data.get("completion_time", ""),  # 完成时间
        "Benchmark Start Time": common_data.get("benchmark_start_time", ""),  # 基准测试开始时间
        "Total Run Time": common_data.get("metrics", {}).get("run_time", ""),  # 总运行时间
        "Highest Difficulty": common_data.get("metrics", {}).get("highest_difficulty", ""),  # 最高难度
        "Workspace": common_data.get("config", {}).get("workspace", ""),  # 工作空间
        "Test Name": test_name,  # 测试名称
        "Data Path": test_info.get("data_path", ""),  # 数据路径
        "Is Regression": test_info.get("is_regression", ""),  # 是否回归
        "Difficulty": test_info.get("metrics", {}).get("difficulty", ""),  # 难度
        "Success": test_info.get("metrics", {}).get("success", ""),  # 成功
        "Success %": test_info.get("metrics", {}).get("success_%", ""),  # 成功百分比
        "Non mock success %": test_info.get("metrics", {}).get("non_mock_success_%", ""),  # 非模拟成功百分比
        "Run Time": test_info.get("metrics", {}).get("run_time", ""),  # 运行时间
        "Benchmark Git Commit Sha": common_data.get("benchmark_git_commit_sha", None),  # 基准 Git 提交 SHA
        "Agent Git Commit Sha": common_data.get("agent_git_commit_sha", None),  # 代理 Git 提交 SHA
        "Cost": test_info.get("metrics", {}).get("cost", ""),  # 成本
        "Attempted": test_info.get("metrics", {}).get("attempted", ""),  # 尝试
        "Test Suite": test_suite,  # 测试套件
        "Category": categories,  # 类别
        "Task": test_info.get("task", ""),  # 任务
        "Answer": test_info.get("answer", ""),  # 答案
        "Description": test_info.get("description", ""),  # 描述
        "Fail Reason": test_info.get("metrics", {}).get("fail_reason", ""),  # 失败原因
        "Reached Cutoff": test_info.get("reached_cutoff", ""),  # 达到截止
    }

    # 将测试结果信息字典添加到行列表中
    rows.append(row)

    # 检查是否存在嵌套测试，并处理它们
    nested_tests = test_info.get("tests")
    # 如果存在嵌套测试，则遍历嵌套测试字典
    if nested_tests:
        # 遍历嵌套测试字典，获取嵌套测试名称和信息
        for nested_test_name, nested_test_info in nested_tests.items():
            # 处理嵌套测试，传入嵌套测试名称、信息、代理名称和公共数据
            process_test(nested_test_name, nested_test_info, agent_name, common_data)
# 使用说明:

# 遍历基本目录中的每个目录
for agent_dir in os.listdir(base_dir):
    # 拼接基本目录和代理目录，得到代理目录的完整路径
    agent_dir_path = os.path.join(base_dir, agent_dir)

    # 确保 agent_dir_path 是一个目录
    # 检查 agent_dir_path 是否为一个目录
    if os.path.isdir(agent_dir_path):
        # 遍历 agent 目录中的每个子目录（例如，"folder49_07-28-03-53"）
        for report_folder in os.listdir(agent_dir_path):
            report_folder_path = os.path.join(agent_dir_path, report_folder)

            # 确保 report_folder_path 是一个目录
            if os.path.isdir(report_folder_path):
                # 检查子目录中是否存在名为 "report.json" 的文件
                report_path = os.path.join(report_folder_path, "report.json")

                if os.path.exists(report_path):
                    # 从文件中加载 JSON 数据
                    with open(report_path, "r") as f:
                        data = json.load(f)
                    benchmark_start_time = data.get("benchmark_start_time", "")

                    # 检查 benchmark_start_time 是否符合所需的格式
                    pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00")
                    if not pattern.fullmatch(benchmark_start_time):
                        continue  # 如果日期格式不正确，则跳过处理此报告

                    # 将 benchmark_start_time 解析为 datetime 对象
                    benchmark_datetime = datetime.strptime(
                        benchmark_start_time, "%Y-%m-%dT%H:%M:%S+00:00"
                    )

                    # 检查 benchmark_start_time 是否早于3天前
                    current_datetime = datetime.utcnow()
                    if current_datetime - benchmark_datetime > timedelta(days=3):
                        continue  # 如果报告超过3天，则跳过处理此报告

                    # 遍历每个测试
                    for test_name, test_info in data["tests"].items():
                        process_test(test_name, test_info, agent_dir, data)
# 将行列表转换为 DataFrame
df = pd.DataFrame(rows)

# 定义范围
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# 添加您的服务帐户凭据
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)

# 授权客户端表
client = gspread.authorize(creds)

# 获取电子表格实例
branch_name = os.getenv("GITHUB_REF_NAME")
sheet = client.open(f"benchmark-{branch_name}")

# 获取电子表格的第一个工作表
sheet_instance = sheet.get_worksheet(0)

# 将 DataFrame 转换为列表的列表以上传到 Google Sheets
values = df.values.tolist()

# 将标题插入值列表的开头
values.insert(0, df.columns.tolist())

# 清除工作表中现有的值
sheet_instance.clear()

# 使用新值更新工作表
sheet_instance.append_rows(values)
```