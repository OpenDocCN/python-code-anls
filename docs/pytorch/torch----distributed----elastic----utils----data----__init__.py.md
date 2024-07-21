# `.\pytorch\torch\distributed\elastic\utils\data\__init__.py`

```
#!/usr/bin/env python3
# 指定此脚本在 Python 3 环境下执行

# 版权声明和许可证信息，指出此代码版权归 Facebook, Inc. 及其关联公司所有，受根目录下 LICENSE 文件中的 BSD-style 许可证保护

# 导入循环迭代器模块 CyclingIterator，忽略未使用警告
from .cycling_iterator import CyclingIterator  # noqa: F401

# 导入弹性分布式采样器模块 ElasticDistributedSampler，忽略未使用警告
from .elastic_distributed_sampler import ElasticDistributedSampler  # noqa: F401
```