# `D:\src\scipysrc\pandas\pandas\tests\extension\base\index.py`

```
"""
Tests for Indexes backed by arbitrary ExtensionArrays.
"""

import pandas as pd  # 导入 pandas 库


class BaseIndexTests:
    """Tests for Index object backed by an ExtensionArray"""

    def test_index_from_array(self, data):
        # 使用 data 创建一个 pandas Index 对象
        idx = pd.Index(data)
        # 断言 Index 对象的数据类型与 data 的数据类型相同
        assert data.dtype == idx.dtype

    def test_index_from_listlike_with_dtype(self, data):
        # 使用 data 创建一个指定数据类型的 pandas Index 对象
        idx = pd.Index(data, dtype=data.dtype)
        # 断言 Index 对象的数据类型与 data 的数据类型相同
        assert idx.dtype == data.dtype

        # 使用 list(data) 创建一个指定数据类型的 pandas Index 对象
        idx = pd.Index(list(data), dtype=data.dtype)
        # 断言 Index 对象的数据类型与 data 的数据类型相同
        assert idx.dtype == data.dtype
```