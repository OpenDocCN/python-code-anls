# `.\AutoGPT\benchmark\agbenchmark\reports\processing\process_report.py`

```py
# 导入所需的模块
import json
import logging
import os
from pathlib import Path
from typing import Any

# 导入自定义模块
from agbenchmark.reports.processing.get_files import (
    get_latest_report_from_agent_directories,
)
from agbenchmark.reports.processing.report_types import Report, Test
from agbenchmark.utils.data_types import STRING_DIFFICULTY_MAP

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 从报告路径中获取最新的报告文件
def get_reports_data(report_path: str) -> dict[str, Any]:
    latest_files = get_latest_report_from_agent_directories(report_path)

    reports_data = {}

    # 如果没有找到最新的文件，则抛出异常
    if latest_files is None:
        raise Exception("No files found in the reports directory")

    # 遍历最新文件，加载 JSON 数据并转换为 Report 对象，存储到 reports_data 字典中
    for subdir, file in latest_files:
        subdir_name = os.path.basename(os.path.normpath(subdir))
        with open(Path(subdir) / file, "r") as f:
            # 从文件中加载 JSON 数据
            json_data = json.load(f)
            converted_data = Report.parse_obj(json_data)
            # 使用最后一个目录名作为键存储到 reports_data 字典中
            reports_data[subdir_name] = converted_data

    return reports_data

# 获取每个类别中最高达到的难度
def get_highest_achieved_difficulty_per_category(report: Report) -> dict[str, Any]:
    categories: dict[str, Any] = {}

    for _, test_data in report.tests.items():
        for category in test_data.category:
            # 排除特定类别
            if category in ("interface", "iterate", "product_advisor"):
                continue
            categories.setdefault(category, 0)
            if (
                test_data.results
                and all(r.success for r in test_data.results)
                and test_data.difficulty
            ):
                num_dif = STRING_DIFFICULTY_MAP[test_data.difficulty]
                if num_dif > categories[category]:
                    categories[category] = num_dif

    return categories

# 获取所有代理类别
def all_agent_categories(reports_data: dict[str, Any]) -> dict[str, Any]:
    all_categories: dict[str, Any] = {}
    # 遍历报告数据字典，获取每个报告的名称和内容
    for name, report in reports_data.items():
        # 调用函数获取每个报告中每个类别的最高难度等级
        categories = get_highest_achieved_difficulty_per_category(report)
        # 如果类别不为空，则将其添加到所有类别中
        if categories:  # only add to all_categories if categories is not empty
            # 记录调试信息，显示正在添加的报告名称和对应的类别
            logger.debug(f"Adding {name}: {categories}")
            # 将报告名称和对应的类别添加到所有类别字典中
            all_categories[name] = categories

    # 返回所有类别字典
    return all_categories
```