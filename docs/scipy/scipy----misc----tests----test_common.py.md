# `D:\src\scipysrc\scipy\scipy\misc\tests\test_common.py`

```
# 导入所需的断言函数和警告抑制工具
from numpy.testing import assert_equal, assert_almost_equal, suppress_warnings

# 从 scipy.misc 模块导入 face, ascent, electrocardiogram 函数
from scipy.misc import face, ascent, electrocardiogram


# 定义测试函数 test_face，用于测试 face 函数的输出
def test_face():
    # 使用 suppress_warnings 上下文管理器，抑制特定类别的警告
    with suppress_warnings() as sup:
        # 过滤 DeprecationWarning 类别的警告
        sup.filter(category=DeprecationWarning)
        # 断言 face 函数返回的数组形状为 (768, 1024, 3)
        assert_equal(face().shape, (768, 1024, 3))


# 定义测试函数 test_ascent，用于测试 ascent 函数的输出
def test_ascent():
    # 使用 suppress_warnings 上下文管理器，抑制特定类别的警告
    with suppress_warnings() as sup:
        # 过滤 DeprecationWarning 类别的警告
        sup.filter(category=DeprecationWarning)
        # 断言 ascent 函数返回的数组形状为 (512, 512)
        assert_equal(ascent().shape, (512, 512))


# 定义测试函数 test_electrocardiogram，用于测试 electrocardiogram 函数的输出
def test_electrocardiogram():
    # 使用 suppress_warnings 上下文管理器，抑制特定类别的警告
    with suppress_warnings() as sup:
        # 过滤 DeprecationWarning 类别的警告
        sup.filter(category=DeprecationWarning)
        # 获取 electrocardiogram 的信号数据
        ecg = electrocardiogram()
        # 断言信号数据的数据类型为 float
        assert ecg.dtype == float
        # 断言信号数据的形状为 (108000,)
        assert_equal(ecg.shape, (108000,))
        # 断言信号数据的平均值接近于 -0.16510875
        assert_almost_equal(ecg.mean(), -0.16510875)
        # 断言信号数据的标准差接近于 0.5992473991177294
        assert_almost_equal(ecg.std(), 0.5992473991177294)
```