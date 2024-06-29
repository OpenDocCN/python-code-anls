# `.\numpy\numpy\tests\test__all__.py`

```
# 导入collections模块和numpy模块
import collections
import numpy as np

# 定义一个测试函数，用于检查np.__all__中是否存在重复项
def test_no_duplicates_in_np__all__():
    # 以下是一个回归测试，用于修复GitHub上编号为gh-10198的问题

    # 使用collections.Counter统计np.__all__中各元素的出现次数，并找出出现次数大于1的项，表示重复项
    dups = {k: v for k, v in collections.Counter(np.__all__).items() if v > 1}
    
    # 断言没有重复项，即重复项的数量应该为0
    assert len(dups) == 0
```