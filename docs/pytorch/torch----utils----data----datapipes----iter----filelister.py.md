# `.\pytorch\torch\utils\data\datapipes\iter\filelister.py`

```py
# mypy: allow-untyped-defs
# 引入需要的类型定义
from typing import Iterator, List, Sequence, Union

# 从torch.utils.data.datapipes._decorator模块中导入functional_datapipe装饰器
from torch.utils.data.datapipes._decorator import functional_datapipe
# 从torch.utils.data.datapipes.datapipe模块中导入IterDataPipe类
from torch.utils.data.datapipes.datapipe import IterDataPipe
# 从torch.utils.data.datapipes.iter.utils模块中导入IterableWrapperIterDataPipe类
from torch.utils.data.datapipes.iter.utils import IterableWrapperIterDataPipe
# 从torch.utils.data.datapipes.utils.common模块中导入get_file_pathnames_from_root函数
from torch.utils.data.datapipes.utils.common import get_file_pathnames_from_root

# 定义模块中公开的接口列表
__all__ = ["FileListerIterDataPipe"]

# 使用functional_datapipe装饰器标记这个类作为函数式数据管道，功能名为"list_files"
@functional_datapipe("list_files")
# 定义FileListerIterDataPipe类，继承IterDataPipe类，用于生成文件路径名列表的数据管道
class FileListerIterDataPipe(IterDataPipe[str]):
    r"""
    给定根目录路径，生成根目录中文件的路径名（路径 + 文件名）。

    可以提供多个根目录路径（函数名: ``list_files``）。

    Args:
        root: 根目录路径或根目录路径序列
        masks: Unix风格的过滤字符串或字符串列表，用于过滤文件名
        recursive: 是否返回嵌套目录中的路径名
        abspath: 是否返回绝对路径名或相对路径名
        non_deterministic: 是否按排序顺序返回路径名
            如果为 ``False``，则每个根目录的结果将按顺序返回
        length: 数据管道的长度

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import FileLister
        >>> dp = FileLister(root=".", recursive=True)
        >>> list(dp)
        ['example.py', './data/data.tar']
    """

    # 初始化函数，接受多种参数配置生成数据管道
    def __init__(
        self,
        root: Union[str, Sequence[str], IterDataPipe] = ".",
        masks: Union[str, List[str]] = "",
        *,
        recursive: bool = False,
        abspath: bool = False,
        non_deterministic: bool = False,
        length: int = -1,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 如果root是字符串，则转换为包含单个元素的列表
        if isinstance(root, str):
            root = [root]
        # 如果root不是IterDataPipe的实例，则使用IterableWrapperIterDataPipe进行包装
        if not isinstance(root, IterDataPipe):
            root = IterableWrapperIterDataPipe(root)
        # 将root数据管道赋值给self.datapipe
        self.datapipe: IterDataPipe = root
        # 设置文件名过滤器
        self.masks: Union[str, List[str]] = masks
        # 设置是否递归查找文件
        self.recursive: bool = recursive
        # 设置是否返回绝对路径
        self.abspath: bool = abspath
        # 设置是否非确定性地返回路径名
        self.non_deterministic: bool = non_deterministic
        # 设置数据管道的长度
        self.length: int = length

    # 实现迭代器协议，生成迭代器以便遍历数据管道中的路径名
    def __iter__(self) -> Iterator[str]:
        # 遍历self.datapipe中的每个路径
        for path in self.datapipe:
            # 调用get_file_pathnames_from_root函数获取根目录中的文件路径名列表，并yield返回每一个路径名
            yield from get_file_pathnames_from_root(
                path, self.masks, self.recursive, self.abspath, self.non_deterministic
            )

    # 返回数据管道的长度，如果长度为-1则引发TypeError异常
    def __len__(self):
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length
```