# `.\numpy\numpy\random\tests\test_seed_sequence.py`

```
# 导入必要的库
import numpy as np
from numpy.testing import assert_array_equal, assert_array_compare

from numpy.random import SeedSequence

# 测试 SeedSequence 是否生成和 C++ 参考代码相同的数据
def test_reference_data():
    """ Check that SeedSequence generates data the same as the C++ reference.

    https://gist.github.com/imneme/540829265469e673d045
    """
    # 输入数据列表
    inputs = [
        [3735928559, 195939070, 229505742, 305419896],
        ...
    ]
    # 期望的输出数据列表
    outputs = [
        [3914649087, 576849849, 3593928901, 2229911004],
        ...
    ]
    # 期望的输出数据列表（64位）
    outputs64 = [
        [2477551240072187391, 9577394838764454085],
        ...
    ]
    # 遍历输入、期望输出和期望输出（64位），对每个种子进行测试
    for seed, expected, expected64 in zip(inputs, outputs, outputs64):
        # 将期望输出转换为 numpy 数组，数据类型为 uint32
        expected = np.array(expected, dtype=np.uint32)
        # 使用种子创建 SeedSequence 对象
        ss = SeedSequence(seed)
        # 生成具有期望长度的状态，并进行断言
        state = ss.generate_state(len(expected))
        assert_array_equal(state, expected)
        # 生成具有期望长度的状态（64位），数据类型为 uint64，并进行断言
        state64 = ss.generate_state(len(expected64), dtype=np.uint64)
        assert_array_equal(state64, expected64)

# 测试零填充是否有问题
def test_zero_padding():
    """ Ensure that the implicit zero-padding does not cause problems.
    """
    # 确保大整数以小尾格式插入，避免结尾为0
    ss0 = SeedSequence(42)
    ss1 = SeedSequence(42 << 32)
    assert_array_compare(
        np.not_equal,
        ss0.generate_state(4),
        ss1.generate_state(4))

    # 确保与原始0.17版本兼容，对于小整数和没有派生密钥
    # 创建一个预期的 NumPy 数组，包含无符号 32 位整数，用于后续的断言比较
    expected42 = np.array([3444837047, 2669555309, 2046530742, 3581440988],
                          dtype=np.uint32)
    # 使用种子数 42 初始化 SeedSequence，并生成长度为 4 的状态数组，然后与预期的数组进行断言比较
    assert_array_equal(SeedSequence(42).generate_state(4), expected42)

    # 回归测试 gh-16539，确保隐式的 0 值不会与生成的种子键冲突
    # 使用种子数 42 和生成的 spawn key (0,) 初始化 SeedSequence，并生成长度为 4 的状态数组，然后与预期的数组进行比较
    assert_array_compare(
        np.not_equal,
        SeedSequence(42, spawn_key=(0,)).generate_state(4),
        expected42)
```