# `D:\src\scipysrc\scipy\scipy\_lib\tests\test__testutils.py`

```
# 导入系统模块
import sys
# 导入 pytest 模块
import pytest
# 从 scipy 库的 _lib._testutils 中导入 _parse_size 和 _get_mem_available 函数
from scipy._lib._testutils import _parse_size, _get_mem_available


# 定义测试函数 test__parse_size
def test__parse_size():
    # 预期的输入和输出对应关系
    expected = {
        '12': 12e6,            # '12' 对应 12000000.0
        '12 b': 12,            # '12 b' 对应 12
        '12k': 12e3,           # '12k' 对应 12000.0
        '  12  M  ': 12e6,     # '  12  M  ' 对应 12000000.0
        '  12  G  ': 12e9,     # '  12  G  ' 对应 12000000000.0
        ' 12Tb ': 12e12,       # ' 12Tb ' 对应 12000000000000.0
        '12  Mib ': 12 * 1024.0**2,  # '12  Mib ' 对应 12582912.0
        '12Tib': 12 * 1024.0**4,    # '12Tib' 对应 13194139533312.0
    }

    # 遍历预期输入和输出对应关系的字典
    for inp, outp in sorted(expected.items()):
        # 如果输出为 None，则预期会引发 ValueError 异常
        if outp is None:
            with pytest.raises(ValueError):
                _parse_size(inp)
        else:
            # 否则，断言 _parse_size 函数返回的结果与预期输出一致
            assert _parse_size(inp) == outp


# 定义测试函数 test__mem_available
def test__mem_available():
    # 获取可用内存大小，可能在非 Linux 平台上返回 None
    available = _get_mem_available()
    # 如果运行平台是 Linux，则断言可用内存大于等于 0
    if sys.platform.startswith('linux'):
        assert available >= 0
    else:
        # 否则，断言可用内存要么是 None，要么大于等于 0
        assert available is None or available >= 0
```