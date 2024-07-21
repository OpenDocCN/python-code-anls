# `.\pytorch\torch\utils\data\datapipes\map\callable.py`

```
# mypy: allow-untyped-defs
# 导入所需模块和函数
from typing import Callable, TypeVar

# 从torch.utils.data.datapipes._decorator模块中导入functional_datapipe装饰器
from torch.utils.data.datapipes._decorator import functional_datapipe
# 从torch.utils.data.datapipes.datapipe模块中导入MapDataPipe类
from torch.utils.data.datapipes.datapipe import MapDataPipe
# 从torch.utils.data.datapipes.utils.common模块中导入_check_unpickable_fn函数
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn

# 定义该模块对外公开的类和函数列表
__all__ = ["MapperMapDataPipe", "default_fn"]

# 声明一个协变类型变量T_co
T_co = TypeVar("T_co", covariant=True)


# 默认函数，直接返回输入数据
# 为了保持datapipe的可picklable性，避免使用Python lambda函数
def default_fn(data):
    return data


# 使用functional_datapipe装饰器声明一个新的类MapperMapDataPipe，
# 它继承自MapDataPipe[T_co]类
@functional_datapipe("map")
class MapperMapDataPipe(MapDataPipe[T_co]):
    r"""
    对源数据管道中的每个项目应用输入的函数（函数名称：``map``）。

    函数可以是任何普通的Python函数或部分对象。不推荐使用Lambda函数，因为它不受pickle支持。

    Args:
        datapipe: 源MapDataPipe对象
        fn: 应用于每个项目的函数

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper, Mapper
        >>> def add_one(x):
        ...     return x + 1
        >>> dp = SequenceWrapper(range(10))
        >>> map_dp_1 = dp.map(add_one)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """

    # 声明成员变量datapipe和fn
    datapipe: MapDataPipe
    fn: Callable

    # 构造函数，初始化对象
    def __init__(
        self,
        datapipe: MapDataPipe,
        fn: Callable = default_fn,
    ) -> None:
        super().__init__()  # 调用父类的构造函数
        self.datapipe = datapipe  # 设置datapipe成员变量为传入的datapipe参数
        _check_unpickable_fn(fn)  # 检查fn是否可picklable
        self.fn = fn  # 设置fn成员变量为传入的fn参数，类型为ignore[assignment]

    # 返回数据管道中项目的数量
    def __len__(self) -> int:
        return len(self.datapipe)

    # 根据索引返回经过fn处理后的数据管道中的项目
    def __getitem__(self, index) -> T_co:
        return self.fn(self.datapipe[index])
```