# `.\pytorch\tools\testing\target_determination\heuristics\llm.py`

```py
# 从未来版本导入类型注解的支持
from __future__ import annotations

# 导入处理 JSON 数据的模块
import json
# 导入操作系统相关功能的模块
import os
# 导入正则表达式的模块
import re
# 导入默认字典的模块
from collections import defaultdict
# 导入处理文件路径的模块
from pathlib import Path
# 导入类型提示相关的模块
from typing import Any

# 从指定路径导入模块
from tools.stats.import_test_stats import ADDITIONAL_CI_FILES_FOLDER
# 从指定路径导入模块
from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
# 从指定路径导入模块
from tools.testing.target_determination.heuristics.utils import normalize_ratings
# 从指定路径导入模块
from tools.testing.test_run import TestRun

# 获取代码仓库根目录的绝对路径
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# LLM 类，继承自 HeuristicInterface 接口类
class LLM(HeuristicInterface):
    
    # 初始化方法，接受关键字参数 kwargs
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)  # 调用父类的初始化方法
    
    # 根据测试列表获取预测的置信度信息
    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        # 获取关键测试的映射
        critical_tests = self.get_mappings()
        # 筛选出有效的测试，并转换为 TestRun 对象，同时保留评分信息
        filter_valid_tests = {
            TestRun(test): score
            for test, score in critical_tests.items()
            if test in tests
        }
        # 对评分进行标准化处理，基准为 0.25
        normalized_scores = normalize_ratings(filter_valid_tests, 0.25)
        # 返回测试的优先级排列结果对象
        return TestPrioritizations(tests, normalized_scores)
    
    # 获取测试与评分的映射关系
    def get_mappings(self) -> dict[str, float]:
        # 指定映射文件的路径
        path = (
            REPO_ROOT
            / ADDITIONAL_CI_FILES_FOLDER
            / "llm_results/mappings/indexer-files-gitdiff-output.json"
        )
        # 如果路径不存在，则打印未找到路径的提示信息，并返回空字典
        if not os.path.exists(path):
            print(f"could not find path {path}")
            return {}
        # 打开 JSON 文件并加载内容
        with open(path) as f:
            # 按文件进行分组
            r = defaultdict(list)
            for key, value in json.load(f).items():
                # 使用正则表达式匹配文件名（去除.py后缀）
                re_match = re.match("(.*).py", key)
                if re_match:
                    file = re_match.group(1)
                    r[file].append(value)
            # 计算每个文件评分的平均值
            r = {file: sum(scores) / len(scores) for file, scores in r.items()}
            # 返回文件名到平均评分的映射字典
            return r
```