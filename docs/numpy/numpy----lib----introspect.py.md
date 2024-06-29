# `.\numpy\numpy\lib\introspect.py`

```
"""
Introspection helper functions.
"""
import re

__all__ = ['opt_func_info']


def opt_func_info(func_name=None, signature=None):
    """
    Returns a dictionary containing the currently supported CPU dispatched
    features for all optimized functions.

    Parameters
    ----------
    func_name : str (optional)
        Regular expression to filter by function name.

    signature : str (optional)
        Regular expression to filter by data type.

    Returns
    -------
    dict
        A dictionary where keys are optimized function names and values are
        nested dictionaries indicating supported targets based on data types.

    Examples
    --------
    Retrieve dispatch information for functions named 'add' or 'sub' and
    data types 'float64' or 'float32':

    >>> dict = np.lib.introspect.opt_func_info(
    ...     func_name="add|abs", signature="float64|complex64"
    ... )
    >>> import json
    >>> print(json.dumps(dict, indent=2))
        {
          "absolute": {
            "dd": {
              "current": "SSE41",
              "available": "SSE41 baseline(SSE SSE2 SSE3)"
            },
            "Ff": {
              "current": "FMA3__AVX2",
              "available": "AVX512F FMA3__AVX2 baseline(SSE SSE2 SSE3)"
            },
            "Dd": {
              "current": "FMA3__AVX2",
              "available": "AVX512F FMA3__AVX2 baseline(SSE SSE2 SSE3)"
            }
          },
          "add": {
            "ddd": {
              "current": "FMA3__AVX2",
              "available": "FMA3__AVX2 baseline(SSE SSE2 SSE3)"
            },
            "FFF": {
              "current": "FMA3__AVX2",
              "available": "FMA3__AVX2 baseline(SSE SSE2 SSE3)"
            }
          }
        }

    """
    # 导入目标信息和数据类型模块
    from numpy._core._multiarray_umath import (
        __cpu_targets_info__ as targets, dtype
    )

    # 如果指定了函数名过滤条件
    if func_name is not None:
        # 编译函数名的正则表达式模式
        func_pattern = re.compile(func_name)
        # 使用字典推导式筛选出符合函数名模式的目标函数
        matching_funcs = {
            k: v for k, v in targets.items()
            if func_pattern.search(k)
        }
    else:
        # 否则，将所有目标函数都作为匹配函数
        matching_funcs = targets

    # 如果指定了数据类型签名过滤条件
    if signature is not None:
        # 编译数据类型签名的正则表达式模式
        sig_pattern = re.compile(signature)
        # 初始化匹配的签名字典
        matching_sigs = {}
        # 遍历匹配的函数名字典
        for k, v in matching_funcs.items():
            # 初始化匹配的字符集合
            matching_chars = {}
            # 遍历函数对应的字符和目标
            for chars, targets in v.items():
                # 如果任意字符匹配数据类型签名或数据类型的名称
                if any([
                    sig_pattern.search(c) or
                    sig_pattern.search(dtype(c).name)
                    for c in chars
                ]):
                    # 将匹配的字符和目标添加到匹配的字符集合中
                    matching_chars[chars] = targets
            # 如果存在匹配的字符集合，将其添加到匹配的签名字典中
            if matching_chars:
                matching_sigs[k] = matching_chars
    else:
        # 否则，将所有匹配的函数名直接作为匹配的签名字典
        matching_sigs = matching_funcs

    # 返回最终的匹配的签名字典
    return matching_sigs
```