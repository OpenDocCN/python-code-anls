# `D:\src\scipysrc\scipy\scipy\fftpack\tests\test_import.py`

```
"""Test possibility of patching fftpack with pyfftw.

No module source outside of scipy.fftpack should contain an import of
the form `from scipy.fftpack import ...`, so that a simple replacement
of scipy.fftpack by the corresponding fftw interface completely swaps
the two FFT implementations.

Because this simply inspects source files, we only need to run the test
on one version of Python.
"""

# 从 pathlib 模块中导入 Path 类
from pathlib import Path
# 导入 re 模块，用于正则表达式操作
import re
# 导入 tokenize 模块，用于解析 Python 源文件
import tokenize
# 导入 pytest 模块，用于编写和运行测试
import pytest
# 从 numpy.testing 模块中导入 assert_ 函数
from numpy.testing import assert_
# 导入 scipy 库
import scipy

# 定义测试类 TestFFTPackImport
class TestFFTPackImport:
    # 标记为慢速测试
    @pytest.mark.slow
    # 定义测试方法 test_fftpack_import
    def test_fftpack_import(self):
        # 获取 scipy 模块所在目录的路径
        base = Path(scipy.__file__).parent
        # 定义正则表达式，匹配形如 `from ...fftpack import ...` 的导入语句
        regexp = r"\s*from.+\.fftpack import .*\n"
        
        # 遍历 base 目录下的所有 .py 文件
        for path in base.rglob("*.py"):
            # 如果文件路径位于 fftpack 目录下，则跳过该文件
            if base / "fftpack" in path.parents:
                continue
            # 使用 tokenize.open 打开文件，自动检测编码
            with tokenize.open(str(path)) as file:
                # 对文件中的每一行进行检查，确保没有符合正则表达式的导入语句
                assert_(all(not re.fullmatch(regexp, line)
                            for line in file),
                        f"{path} contains an import from fftpack")
```