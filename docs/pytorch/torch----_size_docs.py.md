# `.\pytorch\torch\_size_docs.py`

```
# mypy: allow-untyped-defs
"""Adds docstrings to torch.Size functions"""

# 导入 torch._C 模块
import torch._C
# 从 torch._C 模块导入 _add_docstr 函数并重命名为 add_docstr
from torch._C import _add_docstr as add_docstr

# 定义函数 add_docstr_all，用于为 torch.Size 类的指定方法添加文档字符串
def add_docstr_all(method, docstr):
    # 调用 add_docstr 函数，为 torch._C.Size 类的指定方法添加文档字符串
    add_docstr(getattr(torch._C.Size, method), docstr)

# 调用 add_docstr_all 函数，为 torch._C.Size.numel 方法添加详细的文档说明
add_docstr_all(
    "numel",
    """
    numel() -> int

    Returns the number of elements a :class:`torch.Tensor` with the given size would contain.

    More formally, for a tensor ``x = tensor.ones(10, 10)`` with size ``s = torch.Size([10, 10])``,
    ``x.numel() == x.size().numel() == s.numel() == 100`` holds true.

    Example::
        >>> x=torch.ones(10, 10)
        >>> s=x.size()
        >>> s
        torch.Size([10, 10])
        >>> s.numel()
        100
        >>> x.numel() == s.numel()
        True

    .. warning::

        This function does not return the number of dimensions described by :class:`torch.Size`, but instead the number
        of elements a :class:`torch.Tensor` with that size would contain.
    """
)
```