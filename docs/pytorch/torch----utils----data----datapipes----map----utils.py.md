# `.\pytorch\torch\utils\data\datapipes\map\utils.py`

```
# 引入必要的模块和声明允许未类型化的定义
# mypy: allow-untyped-defs
import copy  # 导入深拷贝模块
import warnings  # 导入警告模块

from torch.utils.data.datapipes.datapipe import MapDataPipe  # 导入MapDataPipe类


__all__ = ["SequenceWrapperMapDataPipe"]  # 定义导出的模块列表，只包含SequenceWrapperMapDataPipe类


class SequenceWrapperMapDataPipe(MapDataPipe):
    r"""
    Wraps a sequence object into a MapDataPipe.

    Args:
        sequence: Sequence object to be wrapped into an MapDataPipe
        deepcopy: Option to deepcopy input sequence object

    .. note::
      If ``deepcopy`` is set to False explicitly, users should ensure
      that data pipeline doesn't contain any in-place operations over
      the iterable instance, in order to prevent data inconsistency
      across iterations.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> list(dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> dp = SequenceWrapper({'a': 100, 'b': 200, 'c': 300, 'd': 400})
        >>> dp['a']
        100
    """

    def __init__(self, sequence, deepcopy=True):
        # 根据deepcopy参数的值选择是否进行深拷贝
        if deepcopy:
            try:
                self.sequence = copy.deepcopy(sequence)  # 深拷贝输入的sequence对象
            except TypeError:
                # 如果无法深拷贝，则发出警告，并使用原始的sequence对象
                warnings.warn(
                    "The input sequence can not be deepcopied, "
                    "please be aware of in-place modification would affect source data"
                )
                self.sequence = sequence  # 使用原始的sequence对象
        else:
            self.sequence = sequence  # 直接使用原始的sequence对象，不进行深拷贝

    def __getitem__(self, index):
        return self.sequence[index]  # 获取sequence中索引为index的元素

    def __len__(self):
        return len(self.sequence)  # 返回sequence的长度
```