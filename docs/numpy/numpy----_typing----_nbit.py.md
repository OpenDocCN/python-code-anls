# `.\numpy\numpy\_typing\_nbit.py`

```py
# 导入必要的模块 `Any`，用于类型提示
from typing import Any

# 下面的变量将由 numpy 的 mypy 插件替换为 `npt.NBitBase` 的子类
# 以下变量分别表示不同精度的整数类型，目前使用 `Any` 占位
_NBitByte: Any  # 一个字节精度的整数
_NBitShort: Any  # 短整型
_NBitIntC: Any  # C 整型
_NBitIntP: Any  # 平台整型
_NBitInt: Any  # 整型
_NBitLong: Any  # 长整型
_NBitLongLong: Any  # 长长整型

# 以下变量分别表示不同精度的浮点数类型，目前使用 `Any` 占位
_NBitHalf: Any  # 半精度浮点数
_NBitSingle: Any  # 单精度浮点数
_NBitDouble: Any  # 双精度浮点数
_NBitLongDouble: Any  # 长双精度浮点数
```