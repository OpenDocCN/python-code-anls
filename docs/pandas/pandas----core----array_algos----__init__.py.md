# `D:\src\scipysrc\pandas\pandas\core\array_algos\__init__.py`

```
"""
core.array_algos is for algorithms that operate on ndarray and ExtensionArray.
These should:

- Assume that any Index, Series, or DataFrame objects have already been unwrapped.
- Assume that any list arguments have already been cast to ndarray/EA.
- Not depend on Index, Series, or DataFrame, nor import any of these.
- May dispatch to ExtensionArray methods, but should not import from core.arrays.
"""
"""
core.array_algos 是用于在 ndarray 和 ExtensionArray 上操作的算法。
这些算法应该：

- 假设任何 Index、Series 或 DataFrame 对象已经被解包。
- 假设任何列表参数已经被转换为 ndarray/EA。
- 不依赖于 Index、Series 或 DataFrame，也不导入这些。
- 可以调用 ExtensionArray 方法，但不应该从 core.arrays 导入。
"""
```