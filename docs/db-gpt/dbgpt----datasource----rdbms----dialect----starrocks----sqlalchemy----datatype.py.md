# `.\DB-GPT-src\dbgpt\datasource\rdbms\dialect\starrocks\sqlalchemy\datatype.py`

```py
"""SQLAlchemy data types for StarRocks."""

# 导入所需模块和库
import logging
import re
from typing import Any, Dict, List, Optional, Type

from sqlalchemy import Float, Integer, Numeric  # 导入SQLAlchemy的数据类型
from sqlalchemy.sql import sqltypes  # 导入SQLAlchemy的SQL类型模块
from sqlalchemy.sql.type_api import TypeEngine  # 导入SQLAlchemy的类型引擎

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class TINYINT(Integer):  # 定义StarRocks TINYINT类型，继承自SQLAlchemy的Integer类型
    """StarRocks TINYINT type."""

    __visit_name__ = "TINYINT"  # 定义访问名


class LARGEINT(Integer):  # 定义StarRocks LARGEINT类型，继承自SQLAlchemy的Integer类型
    """StarRocks LARGEINT type."""

    __visit_name__ = "LARGEINT"  # 定义访问名


class DOUBLE(Float):  # 定义StarRocks DOUBLE类型，继承自SQLAlchemy的Float类型
    """StarRocks DOUBLE type."""

    __visit_name__ = "DOUBLE"  # 定义访问名


class HLL(Numeric):  # 定义StarRocks HLL类型，继承自SQLAlchemy的Numeric类型
    """StarRocks HLL type."""

    __visit_name__ = "HLL"  # 定义访问名


class BITMAP(Numeric):  # 定义StarRocks BITMAP类型，继承自SQLAlchemy的Numeric类型
    """StarRocks BITMAP type."""

    __visit_name__ = "BITMAP"  # 定义访问名


class PERCENTILE(Numeric):  # 定义StarRocks PERCENTILE类型，继承自SQLAlchemy的Numeric类型
    """StarRocks PERCENTILE type."""

    __visit_name__ = "PERCENTILE"  # 定义访问名


class ARRAY(TypeEngine):  # 定义StarRocks ARRAY类型，继承自SQLAlchemy的TypeEngine类型
    """StarRocks ARRAY type."""

    __visit_name__ = "ARRAY"  # 定义访问名

    @property
    def python_type(self) -> Optional[Type[List[Any]]]:  # 返回ARRAY类型对应的Python类型
        """Return the Python type for this SQL type."""
        return list


class MAP(TypeEngine):  # 定义StarRocks MAP类型，继承自SQLAlchemy的TypeEngine类型
    """StarRocks MAP type."""

    __visit_name__ = "MAP"  # 定义访问名

    @property
    def python_type(self) -> Optional[Type[Dict[Any, Any]]]:  # 返回MAP类型对应的Python类型
        """Return the Python type for this SQL type."""
        return dict


class STRUCT(TypeEngine):  # 定义StarRocks STRUCT类型，继承自SQLAlchemy的TypeEngine类型
    """StarRocks STRUCT type."""

    __visit_name__ = "STRUCT"  # 定义访问名

    @property
    def python_type(self) -> Optional[Type[Any]]:  # 返回STRUCT类型对应的Python类型
        """Return the Python type for this SQL type."""
        return None


_type_map = {
    # 映射SQL类型字符串到相应的SQLAlchemy类型
    # === 布尔类型 ===
    "boolean": sqltypes.BOOLEAN,
    # === 整数类型 ===
    "tinyint": sqltypes.SMALLINT,
    "smallint": sqltypes.SMALLINT,
    "int": sqltypes.INTEGER,
    "bigint": sqltypes.BIGINT,
    "largeint": LARGEINT,
    # === 浮点类型 ===
    "float": sqltypes.FLOAT,
    "double": DOUBLE,
    # === 定点数类型 ===
    "decimal": sqltypes.DECIMAL,
    # === 字符串类型 ===
    "varchar": sqltypes.VARCHAR,
    "char": sqltypes.CHAR,
    "json": sqltypes.JSON,
    # === 日期和时间类型 ===
    "date": sqltypes.DATE,
    "datetime": sqltypes.DATETIME,
    "timestamp": sqltypes.DATETIME,
    # === 结构化类型 ===
    "array": ARRAY,
    "map": MAP,
    "struct": STRUCT,
    "hll": HLL,
    "percentile": PERCENTILE,
    "bitmap": BITMAP,
}


def parse_sqltype(type_str: str) -> TypeEngine:
    """Parse a SQL type string into a SQLAlchemy type object."""
    type_str = type_str.strip().lower()  # 去除空白字符并转换为小写
    match = re.match(r"^(?P<type>\w+)\s*(?:\((?P<options>.*)\))?", type_str)  # 使用正则表达式匹配类型字符串
    if not match:
        logger.warning(f"Could not parse type name '{type_str}'")  # 如果无法匹配，记录警告日志
        return sqltypes.NULLTYPE  # 返回空类型
    type_name = match.group("type")  # 提取匹配中的类型名称
    # 如果 type_name 不在 _type_map 字典中
    if type_name not in _type_map:
        # 记录警告信息，指出未识别的类型名
        logger.warning(f"Did not recognize type '{type_name}'")
        # 返回预定义的 NULLTYPE 类型
        return sqltypes.NULLTYPE
    
    # 从 _type_map 字典中获取 type_name 对应的类型类
    type_class = _type_map[type_name]
    
    # 返回使用 type_class 类创建的类型实例
    return type_class()
```