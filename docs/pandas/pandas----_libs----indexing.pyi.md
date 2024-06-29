# `D:\src\scipysrc\pandas\pandas\_libs\indexing.pyi`

```
# 导入必要的类型
from typing import (
    Generic,  # 泛型支持
    TypeVar,  # 定义类型变量
)

# 导入 pandas 库中的 IndexingMixin 类
from pandas.core.indexing import IndexingMixin

# 定义一个类型变量 _IndexingMixinT，它是 IndexingMixin 类型的子类或实现类
_IndexingMixinT = TypeVar("_IndexingMixinT", bound=IndexingMixin)

# 定义一个泛型类 NDFrameIndexerBase，参数是 _IndexingMixinT 的子类或实现类
class NDFrameIndexerBase(Generic[_IndexingMixinT]):
    name: str  # 实例变量，表示对象的名称
    obj: _IndexingMixinT  # 实例变量，可以是 DataFrame 或 Series 对象的实例

    # 构造函数，初始化类的名称和对象
    def __init__(self, name: str, obj: _IndexingMixinT) -> None:
        ...

    # ndim 属性，返回对象的维度数量
    @property
    def ndim(self) -> int:
        ...
```