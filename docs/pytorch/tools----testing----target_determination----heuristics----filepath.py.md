# `.\pytorch\tools\testing\target_determination\heuristics\filepath.py`

```py
# 从未来模块中导入注释标注，使得代码兼容未来 Python 版本的类型标注
from __future__ import annotations

# 导入默认字典类型
from collections import defaultdict
# 导入 functools 模块中的 lru_cache 装饰器，用于缓存函数调用结果
from functools import lru_cache
# 导入路径操作相关的 Path 类
from pathlib import Path
# 导入 Any 类型，表示可以是任意类型的变量
from typing import Any, Callable
# 导入警告模块中的 warn 函数，用于发出警告信息

from warnings import warn

# 导入接口模块中的 HeuristicInterface 类和 TestPrioritizations 类
from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)

# 导入工具模块中的 normalize_ratings 和 query_changed_files 函数
from tools.testing.target_determination.heuristics.utils import (
    normalize_ratings,
    query_changed_files,
)

# 导入测试运行模块中的 TestRun 类
from tools.testing.test_run import TestRun

# 定义项目根目录路径，使用 Path(__file__).parent.parent.parent.parent 获取
REPO_ROOT = Path(__file__).parent.parent.parent.parent

# 定义关键字同义词的字典，用于扩展关键字的搜索范围
keyword_synonyms: dict[str, list[str]] = {
    "amp": ["mixed_precision"],
    "quant": ["quantized", "quantization", "quantize"],
    "decomp": ["decomposition", "decompositions"],
    "numpy": ["torch_np", "numpy_tests"],
    "ops": ["opinfo"],
}

# 定义非关键字列表，用于过滤不希望作为关键字的词语
not_keyword = [
    "torch",
    "test",
    "tests",
    "util",
    "utils",
    "func",
    "src",
    "c",
    "ns",
    "tools",
    "internal",
]

# 定义自定义匹配器的字典，用于根据特定规则判断文件是否匹配关键字
custom_matchers: dict[str, Callable[[str], bool]] = {
    "nn": lambda x: "nn" in x.replace("onnx", "_"),
    "c10": lambda x: "c10" in x.replace("c10d", "_"),
}

# 使用 lru_cache 装饰器缓存结果的函数，根据文件路径获取其中的关键字列表
@lru_cache(maxsize=1)
def get_keywords(file: str) -> list[str]:
    keywords = []
    # 遍历文件路径中的各个文件夹，将经过处理后的文件夹名称添加到关键字列表中
    for folder in Path(file).parts[:-1]:
        folder = sanitize_folder_name(folder)
        keywords.append(folder)
    return [kw for kw in keywords if kw not in not_keyword]

# 清理文件夹名称的函数，去除名称中的下划线开头，并使用同义词映射来标准化名称
def sanitize_folder_name(folder_name: str) -> str:
    if folder_name.startswith("_"):
        folder_name = folder_name[1:]

    for syn_rep, syns in keyword_synonyms.items():
        if folder_name in syns or folder_name == syn_rep:
            return syn_rep

    return folder_name

# 判断文件是否与关键字匹配的函数，根据文件的关键字列表和自定义匹配规则进行判断
def file_matches_keyword(file: str, keyword: str) -> bool:
    keywords = get_keywords(file)
    return (
        keyword in keywords
        or any(
            syn in keywords or syn in file for syn in keyword_synonyms.get(keyword, [])
        )
        or custom_matchers.get(keyword, lambda x: keyword in x)(file)  # type: ignore[no-untyped-call]
    )

# 文件路径类，实现了 HeuristicInterface 接口，基于文件路径中的文件夹进行启发式匹配
class Filepath(HeuristicInterface):
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        # 初始化方法，调用父类的初始化方法
    # 定义一个方法用于计算测试的优先级排序，输入参数为测试文件名列表，返回类型为TestPrioritizations对象
    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        # 使用defaultdict创建一个字典，存储关键字及其出现频率，初始值为0
        keyword_frequency: dict[str, int] = defaultdict(int)
        try:
            # 尝试查询变更的文件列表
            changed_files = query_changed_files()
        except Exception as e:
            # 如果查询失败，记录警告信息并将变更文件列表置为空列表
            warn(f"Can't query changed test files due to {e}")
            changed_files = []

        # 遍历变更的文件列表
        for cf in changed_files:
            # 获取文件cf中的关键字列表
            keywords = get_keywords(cf)
            # 遍历关键字列表，更新关键字频率字典
            for keyword in keywords:
                keyword_frequency[keyword] += 1

        # 使用defaultdict创建一个字典，存储测试文件及其评分，初始值为0.0
        test_ratings: dict[str, float] = defaultdict(float)

        # 遍历输入的测试文件列表
        for test in tests:
            # 遍历关键字频率字典的项
            for keyword, frequency in keyword_frequency.items():
                # 如果测试文件test匹配关键字keyword，更新测试评分字典
                if file_matches_keyword(test, keyword):
                    test_ratings[test] += frequency
        
        # 使用字典推导式过滤掉不在输入测试文件列表中的项，并创建TestRun对象及其评分的字典
        test_ratings = {TestRun(k): v for (k, v) in test_ratings.items() if k in tests}
        
        # 返回使用输入测试文件列表、及其经过评分归一化处理的TestPrioritizations对象
        return TestPrioritizations(
            tests, normalize_ratings(test_ratings, 0.25, min_value=0.125)
        )
```