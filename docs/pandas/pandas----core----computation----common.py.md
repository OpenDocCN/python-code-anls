# `D:\src\scipysrc\pandas\pandas\core\computation\common.py`

```
# 引入将来版本的注解功能，使函数的返回值能够引用自身类
from __future__ import annotations

# 引入 functools 模块中的 reduce 函数，用于对多个数据进行归约操作
from functools import reduce

# 引入 numpy 库，命名为 np
import numpy as np

# 从 pandas._config 模块中引入 get_option 函数
from pandas._config import get_option


def ensure_decoded(s) -> str:
    """
    如果输入是 bytes 类型或 np.bytes_ 类型，则将其解码为 unicode 字符串。
    """
    # 检查输入是否为 bytes 或 np.bytes_ 类型
    if isinstance(s, (np.bytes_, bytes)):
        # 尝试使用 display.encoding 设置解码字节字符串
        s = s.decode(get_option("display.encoding"))
    # 返回解码后的字符串
    return s


def result_type_many(*arrays_and_dtypes):
    """
    封装了 numpy.result_type 方法，用于解决参数个数超过 NPY_MAXARGS（32）限制的问题。
    """
    try:
        # 尝试调用 numpy.result_type 函数，并返回结果
        return np.result_type(*arrays_and_dtypes)
    except ValueError:
        # 如果参数个数超过 NPY_MAXARGS，使用 reduce 方法对数组和数据类型进行归约
        return reduce(np.result_type, arrays_and_dtypes)
    except TypeError:
        # 处理特定的类型错误情况，涉及到扩展类型数组和普通数据类型的转换
        from pandas.core.dtypes.cast import find_common_type
        from pandas.core.dtypes.common import is_extension_array_dtype

        arr_and_dtypes = list(arrays_and_dtypes)
        ea_dtypes, non_ea_dtypes = [], []
        
        # 将数组或数据类型分为扩展类型和非扩展类型
        for arr_or_dtype in arr_and_dtypes:
            if is_extension_array_dtype(arr_or_dtype):
                ea_dtypes.append(arr_or_dtype)
            else:
                non_ea_dtypes.append(arr_or_dtype)

        # 如果存在非扩展类型的数组或数据类型，尝试计算其结果类型
        if non_ea_dtypes:
            try:
                np_dtype = np.result_type(*non_ea_dtypes)
            except ValueError:
                np_dtype = reduce(np.result_type, arrays_and_dtypes)
            # 查找扩展类型数组和计算得到的非扩展类型之间的共同类型
            return find_common_type(ea_dtypes + [np_dtype])

        # 如果不存在非扩展类型，直接查找扩展类型数组的共同类型
        return find_common_type(ea_dtypes)
```