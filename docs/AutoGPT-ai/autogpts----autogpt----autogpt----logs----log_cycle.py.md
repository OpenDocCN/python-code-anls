# `.\AutoGPT\autogpts\autogpt\autogpt\logs\log_cycle.py`

```py
import json
import os
from pathlib import Path
from typing import Any, Dict, Union

from .config import LOG_DIR

DEFAULT_PREFIX = "agent"
CURRENT_CONTEXT_FILE_NAME = "current_context.json"
NEXT_ACTION_FILE_NAME = "next_action.json"
PROMPT_SUMMARY_FILE_NAME = "prompt_summary.json"
SUMMARY_FILE_NAME = "summary.txt"
SUPERVISOR_FEEDBACK_FILE_NAME = "supervisor_feedback.txt"
PROMPT_SUPERVISOR_FEEDBACK_FILE_NAME = "prompt_supervisor_feedback.json"
USER_INPUT_FILE_NAME = "user_input.txt"

# 定义一个用于处理日志循环数据的类
class LogCycleHandler:
    """
    A class for logging cycle data.
    """

    # 初始化方法，设置日志计数为0
    def __init__(self):
        self.log_count_within_cycle = 0

    # 创建外部目录方法，根据AI名称和创建时间创建外部目录，并返回路径
    def create_outer_directory(self, ai_name: str, created_at: str) -> Path:
        # 如果环境变量中存在"OVERWRITE_DEBUG"且值为"1"，则外部文件夹名为"auto_gpt"
        if os.environ.get("OVERWRITE_DEBUG") == "1":
            outer_folder_name = "auto_gpt"
        else:
            # 否则根据AI名称获取代理的简称
            ai_name_short = self.get_agent_short_name(ai_name)
            outer_folder_name = f"{created_at}_{ai_name_short}"

        # 拼接外部文件夹路径
        outer_folder_path = LOG_DIR / "DEBUG" / outer_folder_name
        # 如果路径不存在，则创建
        if not outer_folder_path.exists():
            outer_folder_path.mkdir(parents=True)

        return outer_folder_path

    # 获取代理的简称方法，返回AI名称的前15个字符并去除末尾空格，如果AI名称为空则返回默认前缀
    def get_agent_short_name(self, ai_name: str) -> str:
        return ai_name[:15].rstrip() if ai_name else DEFAULT_PREFIX

    # 创建内部目录方法，根据外部文件夹路径和循环计数创建内部目录，并返回路径
    def create_inner_directory(self, outer_folder_path: Path, cycle_count: int) -> Path:
        # 根据循环计数生成嵌套文件夹名
        nested_folder_name = str(cycle_count).zfill(3)
        # 拼接内部文件夹路径
        nested_folder_path = outer_folder_path / nested_folder_name
        # 如果路径不存在，则创建
        if not nested_folder_path.exists():
            nested_folder_path.mkdir()

        return nested_folder_path

    # 创建嵌套目录方法，根据AI名称、创建时间和循环计数创建嵌套目录，并返回路径
    def create_nested_directory(
        self, ai_name: str, created_at: str, cycle_count: int
    ) -> Path:
        # 创建外部文件夹路径
        outer_folder_path = self.create_outer_directory(ai_name, created_at)
        # 创建内部文件夹路径
        nested_folder_path = self.create_inner_directory(outer_folder_path, cycle_count)

        return nested_folder_path
    # 定义一个方法用于记录循环数据到一个 JSON 文件中
    def log_cycle(
        self,
        ai_name: str,  # AI 名称
        created_at: str,  # 创建时间
        cycle_count: int,  # 循环计数
        data: Union[Dict[str, Any], Any],  # 要记录的数据
        file_name: str,  # 要保存记录数据的文件名
    ) -> None:  # 返回类型为 None

        """
        Log cycle data to a JSON file.

        Args:
            data (Any): The data to be logged.
            file_name (str): The name of the file to save the logged data.
        """

        # 创建嵌套目录用于存储循环日志
        cycle_log_dir = self.create_nested_directory(ai_name, created_at, cycle_count)

        # 将数据转换为 JSON 格式的字符串
        json_data = json.dumps(data, ensure_ascii=False, indent=4)

        # 构建日志文件路径
        log_file_path = cycle_log_dir / f"{self.log_count_within_cycle}_{file_name}"

        # 打开文件并写入 JSON 数据
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(json_data + "\n")

        # 增加循环内计数
        self.log_count_within_cycle += 1
```