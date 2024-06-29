# `.\numpy\doc\conftest.py`

```
"""
Pytest configuration and fixtures for the Numpy test suite.
"""
# 导入 pytest 库，用于测试框架
import pytest
# 导入 numpy 库，用于科学计算
import numpy
# 导入 matplotlib 库，用于绘图
import matplotlib
# 导入 doctest 库，用于文档测试
import doctest

# 设置 matplotlib 使用后端 'agg'，强制使用该后端
matplotlib.use('agg', force=True)

# 忽略 matplotlib 输出，如 `<matplotlib.image.AxesImage at
# 0x7f956908c280>`。使用 doctest 的 monkeypatching 实现，
# 受 https://github.com/wooyek/pytest-doctest-ellipsis-markers (MIT license) 启发
# 定义一个自定义的输出检查器类，继承自 doctest.OutputChecker
OutputChecker = doctest.OutputChecker

# 定义要忽略的空行标记列表，如 '<matplotlib.', '<mpl_toolkits.mplot3d.'
empty_line_markers = ['<matplotlib.', '<mpl_toolkits.mplot3d.']

class SkipMatplotlibOutputChecker(doctest.OutputChecker):
    def check_output(self, want, got, optionflags):
        # 遍历空行标记列表，如果输出中包含其中之一的标记，则将 got 设为空字符串
        for marker in empty_line_markers:
            if marker in got:
                got = ''
                break
        # 调用父类的 check_output 方法检查输出
        return OutputChecker.check_output(self, want, got, optionflags)

# 将 doctest.OutputChecker 替换为自定义的 SkipMatplotlibOutputChecker
doctest.OutputChecker = SkipMatplotlibOutputChecker

# 定义一个自动使用的 fixture，向 doctest 的命名空间中添加 'np'，值为 numpy
@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    # 设置 numpy 的随机种子为 1
    numpy.random.seed(1)
    # 向 doctest 的命名空间中添加 'np'，值为 numpy
    doctest_namespace['np'] = numpy
```