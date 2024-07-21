# `.\pytorch\tools\code_coverage\package\tool\parser\llvm_coverage_parser.py`

```
from __future__ import annotations  # 导入未来版本的类型注解支持

from typing import Any  # 导入类型提示模块

from .coverage_record import CoverageRecord  # 导入自定义的CoverageRecord类
from .llvm_coverage_segment import LlvmCoverageSegment, parse_segments  # 导入LlvmCoverageSegment类和parse_segments函数


class LlvmCoverageParser:
    """
    Accepts a parsed json produced by llvm-cov export -- typically,
    representing a single C++ test and produces a list
    of CoverageRecord(s).
    """

    def __init__(self, llvm_coverage: dict[str, Any]) -> None:
        self._llvm_coverage = llvm_coverage  # 初始化LlvmCoverageParser对象的llvm_coverage属性

    @staticmethod
    def _skip_coverage(path: str) -> bool:
        """
        Returns True if file path should not be processed.
        This is repo-specific and only makes sense for the current state of
        ovrsource.
        """
        if "/third-party/" in path:  # 如果路径中包含"/third-party/"字符串
            return True  # 返回True，表示跳过该路径
        return False  # 否则返回False

    @staticmethod
    def _collect_coverage(
        segments: list[LlvmCoverageSegment],
    ) -> tuple[list[int], list[int]]:
        """
        Stateful parsing of coverage segments.
        """
        covered_lines: set[int] = set()  # 创建空集合，用于存储被覆盖的行号
        uncovered_lines: set[int] = set()  # 创建空集合，用于存储未被覆盖的行号
        prev_segment = LlvmCoverageSegment(1, 0, 0, 0, 0, None)  # 创建LlvmCoverageSegment对象作为初始段
        for segment in segments:  # 遍历segments列表中的每个段
            covered_range, uncovered_range = segment.get_coverage(prev_segment)  # 调用segment对象的get_coverage方法
            covered_lines.update(covered_range)  # 更新被覆盖行号集合
            uncovered_lines.update(uncovered_range)  # 更新未被覆盖行号集合
            prev_segment = segment  # 更新prev_segment为当前segment

        uncovered_lines.difference_update(covered_lines)  # 从未被覆盖行号集合中移除被覆盖行号
        return sorted(covered_lines), sorted(uncovered_lines)  # 返回排序后的被覆盖行号和未被覆盖行号列表

    def parse(self, repo_name: str) -> list[CoverageRecord]:
        # The JSON format is described in the LLVM source code
        # https://github.com/llvm-mirror/llvm/blob/master/tools/llvm-cov/CoverageExporterJson.cpp
        records: list[CoverageRecord] = []  # 创建空列表，用于存储CoverageRecord对象
        for export_unit in self._llvm_coverage["data"]:  # 遍历llvm_coverage中"data"键对应的列表
            for file_info in export_unit["files"]:  # 遍历每个export_unit中"files"键对应的列表
                filepath = file_info["filename"]  # 获取文件路径
                if self._skip_coverage(filepath):  # 调用_skip_coverage方法判断是否跳过该路径
                    continue  # 如果需要跳过，则继续下一个文件

                if filepath is None:  # 如果文件路径为None
                    continue  # 继续下一个文件

                segments = file_info["segments"]  # 获取文件信息中的segments列表

                covered_lines, uncovered_lines = self._collect_coverage(
                    parse_segments(segments)  # 调用parse_segments函数解析segments，并传入_collect_coverage方法
                )

                records.append(CoverageRecord(filepath, covered_lines, uncovered_lines))  # 创建CoverageRecord对象并添加到records列表中

        return records  # 返回包含CoverageRecord对象的列表
```