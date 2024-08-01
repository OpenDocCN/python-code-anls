# `.\DB-GPT-src\dbgpt\rag\summary\db_summary.py`

```py
"""Summary classes for database, table, field and index."""
# 数据库、表、字段和索引的摘要类

from typing import Dict, Iterable, List, Optional, Tuple
# 导入类型提示所需的模块

class DBSummary:
    """Database summary class."""
    # 数据库摘要类

    def __init__(self, name: str):
        """Create a new DBSummary."""
        # 创建一个新的数据库摘要
        self.name = name
        self.summary: Optional[str] = None
        self.tables: Iterable[str] = []
        self.metadata: Optional[str] = None

    def get_summary(self) -> Optional[str]:
        """Get the summary."""
        # 获取摘要信息
        return self.summary


class TableSummary:
    """Table summary class."""
    # 表摘要类

    def __init__(self, name: str):
        """Create a new TableSummary."""
        # 创建一个新的表摘要
        self.name = name
        self.summary: Optional[str] = None
        self.fields: List[Tuple] = []
        self.indexes: List[Dict] = []


class FieldSummary:
    """Field summary class."""
    # 字段摘要类

    def __init__(self, name: str):
        """Create a new FieldSummary."""
        # 创建一个新的字段摘要
        self.name = name
        self.summary = None
        self.data_type = None


class IndexSummary:
    """Index summary class."""
    # 索引摘要类

    def __init__(self, name: str):
        """Create a new IndexSummary."""
        # 创建一个新的索引摘要
        self.name = name
        self.summary: Optional[str] = None
        self.bind_fields: List[str] = []
```