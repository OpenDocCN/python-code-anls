# `.\AutoGPT\benchmark\agbenchmark\utils\utils.py`

```py
# 导入所需的模块和库
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# 从自定义模块中导入相关内容
from agbenchmark.reports.processing.report_types import Test
from agbenchmark.utils.data_types import DIFFICULTY_MAP, DifficultyLevel

# 加载环境变量
load_dotenv()

# 从环境变量中获取代理名称和报告位置
AGENT_NAME = os.getenv("AGENT_NAME")
REPORT_LOCATION = os.getenv("REPORT_LOCATION", None)

# 获取日志记录器
logger = logging.getLogger(__name__)


# 定义函数，用于替换字符串中的反斜杠为斜杠
def replace_backslash(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(
            r"\\+", "/", value
        )  # 将一个或多个反斜杠替换为斜杠
    elif isinstance(value, list):
        return [replace_backslash(i) for i in value]
    elif isinstance(value, dict):
        return {k: replace_backslash(v) for k, v in value.items()}
    else:
        return value


# 定义函数，用于获取测试文件的路径
def get_test_path(json_file: str | Path) -> str:
    if isinstance(json_file, str):
        json_file = Path(json_file)

    # 查找路径中 "agbenchmark" 的索引
    try:
        agbenchmark_index = json_file.parts.index("benchmark")
    except ValueError:
        raise ValueError("Invalid challenge location.")

    # 从 "agbenchmark" 开始创建路径
    challenge_location = Path(*json_file.parts[agbenchmark_index:])

    # 替换路径中的反斜杠为斜杠
    formatted_location = replace_backslash(str(challenge_location))
    if isinstance(formatted_location, str):
        return formatted_location
    else:
        return str(challenge_location)


# 定义函数，用于获取最高成功难度
def get_highest_success_difficulty(
    data: dict[str, Test], just_string: Optional[bool] = None
) -> str:
    highest_difficulty = None
    highest_difficulty_level = 0
    # 遍历数据字典，获取测试名称和测试数据
    for test_name, test_data in data.items():
        try:
            # 检查测试数据中是否有任何成功的结果
            if any(r.success for r in test_data.results):
                # 获取测试数据中的难度级别字符串
                difficulty_str = test_data.difficulty
                # 如果难度级别字符串为空，则跳过当前循环
                if not difficulty_str:
                    continue

                try:
                    # 尝试将难度级别字符串转换为枚举类型
                    difficulty_enum = DifficultyLevel[difficulty_str.lower()]
                    # 获取难度级别对应的数值
                    difficulty_level = DIFFICULTY_MAP[difficulty_enum]

                    # 如果当前难度级别大于最高难度级别，则更新最高难度级别和对应的枚举值
                    if difficulty_level > highest_difficulty_level:
                        highest_difficulty = difficulty_enum
                        highest_difficulty_level = difficulty_level
                except KeyError:
                    # 如果难度级别字符串无法转换为枚举类型，则记录警告信息
                    logger.warning(
                        f"Unexpected difficulty level '{difficulty_str}' "
                        f"in test '{test_name}'"
                    )
                    continue
        except Exception as e:
            # 捕获异常情况，记录错误信息和报告数据
            logger.warning(
                "An unexpected error [1] occurred while analyzing report [2]."
                "Please notify a maintainer.\n"
                f"Report data [1]: {data}\n"
                f"Error [2]: {e}"
            )
            logger.warning(
                "Make sure you selected the right test, no reports were generated."
            )
            break

    # 如果存在最高难度级别，则将枚举值转换为字符串
    if highest_difficulty is not None:
        highest_difficulty_str = highest_difficulty.name
    else:
        highest_difficulty_str = ""

    # 如果存在最高难度级别数值且不仅仅是字符串，则返回最高难度级别和对应数值
    if highest_difficulty_level and not just_string:
        return f"{highest_difficulty_str}: {highest_difficulty_level}"
    # 如果存在最高难度级别字符串，则返回最高难度级别字符串
    elif highest_difficulty_str:
        return highest_difficulty_str
    # 如果不存在最高难度级别，则返回提示信息
    return "No successful tests"
# 将数据按照深度排序后写入 JSON 文件
def write_pretty_json(data, json_file):
    # 对数据进行深度排序
    sorted_data = deep_sort(data)
    # 将排序后的数据转换成带缩进的 JSON 字符串
    json_graph = json.dumps(sorted_data, indent=4)
    # 打开 JSON 文件，写入数据
    with open(json_file, "w") as f:
        f.write(json_graph)
        f.write("\n")

# 递归地对 JSON 对象进行键的排序
def deep_sort(obj):
    """
    Recursively sort the keys in JSON object
    """
    # 如果是字典类型，则对键进行排序
    if isinstance(obj, dict):
        return {k: deep_sort(v) for k, v in sorted(obj.items())}
    # 如果是列表类型，则对列表中的元素进行排序
    if isinstance(obj, list):
        return [deep_sort(elem) for elem in obj]
    # 其他类型直接返回
    return obj
```