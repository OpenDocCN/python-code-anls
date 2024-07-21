# `.\pytorch\torch\_storage_docs.py`

```py
# 添加了一个特定文档字符串给所有的存储类方法
"""Adds docstrings to Storage functions"""

# 导入 torch._C 模块，并引入 _add_docstr 作为 add_docstr
import torch._C
from torch._C import _add_docstr as add_docstr

# 定义一个存储类列表，包含一个元素 "StorageBase"
storage_classes = [
    "StorageBase",
]

# 定义一个函数，为给定方法添加文档字符串
def add_docstr_all(method, docstr):
    # 遍历每个存储类名
    for cls_name in storage_classes:
        # 获取 torch._C 模块中对应名称的类对象
        cls = getattr(torch._C, cls_name)
        try:
            # 尝试为给定方法添加文档字符串
            add_docstr(getattr(cls, method), docstr)
        except AttributeError:
            # 如果方法不存在，则捕获 AttributeError 异常并忽略
            pass

# 调用 add_docstr_all 函数，为名为 "from_file" 的方法添加特定的文档字符串
add_docstr_all(
    "from_file",
    """
from_file(filename, shared=False, size=0) -> Storage

Creates a CPU storage backed by a memory-mapped file.

If ``shared`` is ``True``, then memory is shared between all processes.
All changes are written to the file. If ``shared`` is ``False``, then the changes on
the storage do not affect the file.

``size`` is the number of elements in the storage. If ``shared`` is ``False``,
then the file must contain at least ``size * sizeof(Type)`` bytes
(``Type`` is the type of storage, in the case of an ``UnTypedStorage`` the file must contain at
least ``size`` bytes). If ``shared`` is ``True`` the file will be created if needed.

Args:
    filename (str): file name to map
    shared (bool): whether to share memory (whether ``MAP_SHARED`` or ``MAP_PRIVATE`` is passed to the
                    underlying `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_)
    size (int): number of elements in the storage
""",
)
```