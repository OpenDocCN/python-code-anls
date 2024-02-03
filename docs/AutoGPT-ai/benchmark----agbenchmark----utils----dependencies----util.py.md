# `.\AutoGPT\benchmark\agbenchmark\utils\dependencies\util.py`

```py
""" Utility functions to process the identifiers of tests. """

# 导入所需的模块
import re
from typing import Iterator

from _pytest.mark.structures import Mark
from _pytest.nodes import Item

# 导入常量
from .constants import MARKER_KWARG_ID, MARKER_NAME

# 编译正则表达式，用于匹配参数部分
REGEX_PARAMETERS = re.compile(r"\[.+\]$")

# 清理节点 ID 中多余的 ::()
def clean_nodeid(nodeid: str) -> str:
    """
    Remove any superfluous ::() from a node id.

    >>> clean_nodeid('test_file.py::TestClass::()::test')
    'test_file.py::TestClass::test'
    >>> clean_nodeid('test_file.py::TestClass::test')
    'test_file.py::TestClass::test'
    >>> clean_nodeid('test_file.py::test')
    'test_file.py::test'
    """
    return nodeid.replace("::()::", "::")

# 去除节点 ID 中的参数部分
def strip_nodeid_parameters(nodeid: str) -> str:
    """
    Strip parameters from a node id.

    >>> strip_nodeid_parameters('test_file.py::TestClass::test[foo]')
    'test_file.py::TestClass::test'
    >>> strip_nodeid_parameters('test_file.py::TestClass::test')
    'test_file.py::TestClass::test'
    """
    return REGEX_PARAMETERS.sub("", nodeid)

# 获取绝对节点 ID
def get_absolute_nodeid(nodeid: str, scope: str) -> str:
    """
    Transform a possibly relative node id to an absolute one using the scope in which it is used.

    >>> scope = 'test_file.py::TestClass::test'
    >>> get_absolute_nodeid('test2', scope)
    'test_file.py::TestClass::test2'
    >>> get_absolute_nodeid('TestClass2::test2', scope)
    'test_file.py::TestClass2::test2'
    >>> get_absolute_nodeid('test_file2.py::TestClass2::test2', scope)
    'test_file2.py::TestClass2::test2'
    """
    parts = nodeid.split("::")
    # 如果节点 ID 完全相对 (test_name)，则添加完整的当前范围 (文件::类 或 文件)
    if len(parts) == 1:
        base_nodeid = scope.rsplit("::", 1)[0]
        nodeid = f"{base_nodeid}::{nodeid}"
    # 如果已包含一些范围 (Class::test_name)，则只添加当前文件范围
    elif "." not in parts[0]:
        base_nodeid = scope.split("::", 1)[0]
        nodeid = f"{base_nodeid}::{nodeid}"
    # 返回清理后的节点ID
    return clean_nodeid(nodeid)
# 获取测试项的名称
def get_name(item: Item) -> str:
    """
    Get all names for a test.

    This will use the following methods to determine the name of the test:
        - If given, the custom name(s) passed to the keyword argument name on the marker
    """
    # 初始化名称为空字符串
    name = ""

    # 获取自定义名称
    markers = get_markers(item, MARKER_NAME)
    # 遍历所有标记
    for marker in markers:
        # 如果标记中包含指定的关键字参数
        if MARKER_KWARG_ID in marker.kwargs:
            # 更新名称为关键字参数的值
            name = marker.kwargs[MARKER_KWARG_ID]

    # 返回最终的名称
    return name


# 获取给定项的所有具有指定名称的标记
def get_markers(item: Item, name: str) -> Iterator[Mark]:
    """Get all markers with the given name for a given item."""
    # 遍历项的所有标记
    for marker in item.iter_markers():
        # 如果标记的名称与指定名称相同
        if marker.name == name:
            # 返回符合条件的标记
            yield marker
```