# `.\pytorch\tools\code_coverage\package\tool\parser\coverage_record.py`

```
from __future__ import annotations
# 引入了 `annotations` 特性，使得类型提示中可以使用字符串形式的类型名称

from typing import Any, NamedTuple
# 导入了类型提示模块 `typing` 中的 `Any` 和 `NamedTuple` 类型

class CoverageRecord(NamedTuple):
    # 定义了一个名为 `CoverageRecord` 的命名元组（NamedTuple）
    filepath: str
    # 定义了一个 `str` 类型的属性 `filepath`，表示文件路径
    covered_lines: list[int]
    # 定义了一个 `list[int]` 类型的属性 `covered_lines`，表示已覆盖的行列表
    uncovered_lines: list[int] | None = None
    # 定义了一个 `list[int] | None` 类型的属性 `uncovered_lines`，表示未覆盖的行列表或者为 `None`

    def to_dict(self) -> dict[str, Any]:
        # 定义了一个方法 `to_dict`，返回一个字典类型的对象
        return {
            "filepath": self.filepath,
            # 将实例的 `filepath` 属性映射到字典的键 `"filepath"`
            "covered_lines": self.covered_lines,
            # 将实例的 `covered_lines` 属性映射到字典的键 `"covered_lines"`
            "uncovered_lines": self.uncovered_lines,
            # 将实例的 `uncovered_lines` 属性映射到字典的键 `"uncovered_lines"`
        }
```