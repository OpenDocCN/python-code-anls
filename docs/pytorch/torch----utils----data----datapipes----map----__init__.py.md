# `.\pytorch\torch\utils\data\datapipes\map\__init__.py`

```
# 导入功能性数据管道模块中的各种数据处理类和函数
from torch.utils.data.datapipes.map.callable import MapperMapDataPipe as Mapper
from torch.utils.data.datapipes.map.combinatorics import (
    ShufflerIterDataPipe as Shuffler,
)
from torch.utils.data.datapipes.map.combining import (
    ConcaterMapDataPipe as Concater,
    ZipperMapDataPipe as Zipper,
)
from torch.utils.data.datapipes.map.grouping import BatcherMapDataPipe as Batcher
from torch.utils.data.datapipes.map.utils import (
    SequenceWrapperMapDataPipe as SequenceWrapper,
)

# 声明一个包含所有导出类名的列表，供外部模块使用
__all__ = ["Batcher", "Concater", "Mapper", "SequenceWrapper", "Shuffler", "Zipper"]

# 确保 __all__ 列表中的类名按字母顺序排列，用于避免导出时的混乱
assert __all__ == sorted(__all__)
```