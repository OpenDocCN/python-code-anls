# `.\pytorch\torch\utils\data\datapipes\iter\utils.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和类
import copy  # 导入深拷贝模块
import warnings  # 导入警告模块

from torch.utils.data.datapipes.datapipe import IterDataPipe  # 从torch库中导入IterDataPipe类


__all__ = ["IterableWrapperIterDataPipe"]  # 模块导出的符号列表


class IterableWrapperIterDataPipe(IterDataPipe):
    r"""
    Wraps an iterable object to create an IterDataPipe.

    Args:
        iterable: Iterable object to be wrapped into an IterDataPipe
        deepcopy: Option to deepcopy input iterable object for each
            iterator. The copy is made when the first element is read in ``iter()``.

    .. note::
        If ``deepcopy`` is explicitly set to ``False``, users should ensure
        that the data pipeline doesn't contain any in-place operations over
        the iterable instance to prevent data inconsistency across iterations.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> list(dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    def __init__(self, iterable, deepcopy=True):
        # 初始化方法，接受一个可迭代对象和一个深拷贝标志
        self.iterable = iterable  # 将传入的可迭代对象赋值给实例变量
        self.deepcopy = deepcopy  # 将深拷贝标志赋值给实例变量

    def __iter__(self):
        # 迭代器方法，返回一个迭代器
        source_data = self.iterable  # 将实例变量赋给source_data
        if self.deepcopy:  # 如果设置了深拷贝
            try:
                source_data = copy.deepcopy(self.iterable)  # 尝试深拷贝可迭代对象
            except TypeError:
                # 如果无法深拷贝，则发出警告
                warnings.warn(
                    "The input iterable can not be deepcopied, "
                    "please be aware of in-place modification would affect source data."
                )
        yield from source_data  # 生成器函数，从source_data中生成元素

    def __len__(self):
        # 返回可迭代对象的长度
        return len(self.iterable)
```