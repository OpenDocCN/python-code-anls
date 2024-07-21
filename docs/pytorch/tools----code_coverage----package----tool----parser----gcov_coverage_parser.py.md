# `.\pytorch\tools\code_coverage\package\tool\parser\gcov_coverage_parser.py`

```
from __future__ import annotations

from typing import Any

from .coverage_record import CoverageRecord


class GcovCoverageParser:
    """
    Accepts a parsed json produced by gcov --json-format -- typically,
    representing a single C++ test and produces a list
    of CoverageRecord(s).
    """

    def __init__(self, llvm_coverage: dict[str, Any]) -> None:
        # 初始化方法，接收一个 dict 类型的 llvm_coverage 参数
        self._llvm_coverage = llvm_coverage

    @staticmethod
    def _skip_coverage(path: str) -> bool:
        """
        Returns True if file path should not be processed.
        This is repo-specific and only makes sense for the current state of
        ovrsource.
        """
        # 检查文件路径是否包含 "third-party"，如果是则跳过覆盖率处理
        if "third-party" in path:
            return True
        return False

    def parse(self) -> list[CoverageRecord]:
        # 解析方法，返回一个 CoverageRecord 对象的列表
        # 首先初始化一个空列表用于存放解析后的记录
        records: list[CoverageRecord] = []
        # 遍历每个文件的信息
        for file_info in self._llvm_coverage["files"]:
            filepath = file_info["file"]
            # 如果文件路径符合跳过条件，则继续下一个文件的处理
            if self._skip_coverage(filepath):
                continue
            # 初始化用于存放已覆盖和未覆盖行号的集合
            covered_lines: set[int] = set()
            uncovered_lines: set[int] = set()
            # 遍历当前文件的每一行信息
            for line in file_info["lines"]:
                line_number = line["line_number"]
                count = line["count"]
                # 根据覆盖次数判断当前行是否覆盖
                if count == 0:
                    uncovered_lines.update([line_number])
                else:
                    covered_lines.update([line_number])

            # 将当前文件的覆盖信息记录添加到 records 列表中
            records.append(
                CoverageRecord(filepath, sorted(covered_lines), sorted(uncovered_lines))
            )

        # 返回所有解析后的 CoverageRecord 对象列表
        return records
```