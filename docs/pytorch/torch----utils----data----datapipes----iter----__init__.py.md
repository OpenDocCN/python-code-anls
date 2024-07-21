# `.\pytorch\torch\utils\data\datapipes\iter\__init__.py`

```
# 导`
from torch.utils.data.datapipes.iter.callable import (
    CollatorIterDataPipe as Collator,  # 从 callable 模块导入 CollatorIterDataPipe，并重命名为 Collator
    MapperIterDataPipe as Mapper,  # 从 callable 模块导入 MapperIterDataPipe，并重命名为 Mapper
)
from torch.utils.data.datapipes.iter.combinatorics import (
    SamplerIterDataPipe as Sampler,  # 从 combinatorics 模块导入 SamplerIterDataPipe，并重命名为 Sampler
    ShufflerIterDataPipe as Shuffler,  # 从 combinatorics 模块导入 ShufflerIterDataPipe，并重命名为 Shuffler
)
from torch.utils.data.datapipes.iter.combining import (
    ConcaterIterDataPipe as Concater,  # 从 combining 模块导入 ConcaterIterDataPipe，并重命名为 Concater
    DemultiplexerIterDataPipe as Demultiplexer,  # 从 combining 模块导入 DemultiplexerIterDataPipe，并重命名为 Demultiplexer
    ForkerIterDataPipe as Forker,  # 从 combining 模块导入 ForkerIterDataPipe，并重命名为 Forker
    MultiplexerIterDataPipe as Multiplexer,  # 从 combining 模块导入 MultiplexerIterDataPipe，并重命名为 Multiplexer
    ZipperIterDataPipe as Zipper,  # 从 combining 模块导入 ZipperIterDataPipe，并重命名为 Zipper
)
from torch.utils.data.datapipes.iter.filelister import (
    FileListerIterDataPipe as FileLister,  # 从 filelister 模块导入 FileListerIterDataPipe，并重命名为 FileLister
)
from torch.utils.data.datapipes.iter.fileopener import (
    FileOpenerIterDataPipe as FileOpener,  # 从 fileopener 模块导入 FileOpenerIterDataPipe，并重命名为 FileOpener
)
from torch.utils.data.datapipes.iter.grouping import (
    BatcherIterDataPipe as Batcher,  # 从 grouping 模块导入 BatcherIterDataPipe，并重命名为 Batcher
    GrouperIterDataPipe as Grouper,  # 从 grouping 模块导入 GrouperIterDataPipe，并重命名为 Grouper
    UnBatcherIterDataPipe as UnBatcher,  # 从 grouping 模块导入 UnBatcherIterDataPipe，并重命名为 UnBatcher
)
from torch.utils.data.datapipes.iter.routeddecoder import (
    RoutedDecoderIterDataPipe as RoutedDecoder,  # 从 routeddecoder 模块导入 RoutedDecoderIterDataPipe，并重命名为 RoutedDecoder
)
from torch.utils.data.datapipes.iter.selecting import FilterIterDataPipe as Filter  # 从 selecting 模块导入 FilterIterDataPipe，并重命名为 Filter
from torch.utils.data.datapipes.iter.sharding import (
    ShardingFilterIterDataPipe as ShardingFilter,  # 从 sharding 模块导入 ShardingFilterIterDataPipe，并重命名为 ShardingFilter
)
from torch.utils.data.datapipes.iter.streamreader import (
    StreamReaderIterDataPipe as StreamReader,  # 从 streamreader 模块导入 StreamReaderIterDataPipe，并重命名为 StreamReader
)
from torch.utils.data.datapipes.iter.utils import (
    IterableWrapperIterDataPipe as IterableWrapper,  # 从 utils 模块导入 IterableWrapperIterDataPipe，并重命名为 IterableWrapper
)

# 定义一个导出列表，包含所有要公开的类名
__all__ = [
    "Batcher",  # 包括 Batcher 类
    "Collator",  # 包括 Collator 类
    "Concater",  # 包括 Concater 类
    "Demultiplexer",  # 包括 Demultiplexer 类
    "FileLister",  # 包括 FileLister 类
    "FileOpener",  # 包括 FileOpener 类
    "Filter",  # 包括 Filter 类
    "Forker",  # 包括 Forker 类
    "Grouper",  # 包括 Grouper 类
    "IterableWrapper",  # 包括 IterableWrapper 类
    "Mapper",  # 包括 Mapper 类
    "Multiplexer",  # 包括 Multiplexer 类
    "RoutedDecoder",  # 包括 RoutedDecoder 类
    "Sampler",  # 包括 Sampler 类
    "ShardingFilter",  # 包括 ShardingFilter 类
    "Shuffler",  # 包括 Shuffler 类
    "StreamReader",  # 包括 StreamReader 类
    "UnBatcher",  # 包括 UnBatcher 类
    "Zipper",  # 包括 Zipper 类
]

# 确保 __all__ 列表按字母顺序排列
assert __all__ == sorted(__all__)
```