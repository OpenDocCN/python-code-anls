# `.\numpy\numpy\f2py\tests\test_regression.py`

```py
# 导入所需的库和模块
import os
import pytest
import platform

import numpy as np  # 导入 NumPy 库
import numpy.testing as npt  # 导入 NumPy 测试模块

from . import util  # 从当前包导入 util 模块


class TestIntentInOut(util.F2PyTest):
    # 检查 intent(in out) 是否正确翻译为 intent(inout)
    sources = [util.getpath("tests", "src", "regression", "inout.f90")]

    @pytest.mark.slow
    def test_inout(self):
        # 对非连续数组应该引发 ValueError 错误
        x = np.arange(6, dtype=np.float32)[::2]
        pytest.raises(ValueError, self.module.foo, x)

        # 使用连续数组检查数值
        x = np.arange(3, dtype=np.float32)
        self.module.foo(x)
        assert np.allclose(x, [3, 1, 2])


class TestNegativeBounds(util.F2PyTest):
    # 检查负索引边界是否正常工作
    sources = [util.getpath("tests", "src", "negative_bounds", "issue_20853.f90")]

    @pytest.mark.slow
    def test_negbound(self):
        xvec = np.arange(12)
        xlow = -6
        xhigh = 4

        # 计算上限，
        # 注意保持索引为 1
        def ubound(xl, xh):
            return xh - xl + 1

        rval = self.module.foo(is_=xlow, ie_=xhigh,
                               arr=xvec[:ubound(xlow, xhigh)])
        expval = np.arange(11, dtype=np.float32)
        assert np.allclose(rval, expval)


class TestNumpyVersionAttribute(util.F2PyTest):
    # 检查编译模块中是否存在 __f2py_numpy_version__ 属性，
    # 并且其值为 np.__version__
    sources = [util.getpath("tests", "src", "regression", "inout.f90")]

    @pytest.mark.slow
    def test_numpy_version_attribute(self):
        # 检查 self.module 是否有名为 "__f2py_numpy_version__" 的属性
        assert hasattr(self.module, "__f2py_numpy_version__")

        # 检查 "__f2py_numpy_version__" 属性是否为字符串类型
        assert isinstance(self.module.__f2py_numpy_version__, str)

        # 检查 "__f2py_numpy_version__" 属性的值是否为 numpy.__version__
        assert np.__version__ == self.module.__f2py_numpy_version__


def test_include_path():
    # 测试 np.f2py.get_include() 是否能正确返回包含文件的路径
    incdir = np.f2py.get_include()
    fnames_in_dir = os.listdir(incdir)
    for fname in ("fortranobject.c", "fortranobject.h"):
        assert fname in fnames_in_dir


class TestIncludeFiles(util.F2PyTest):
    # 检查编译模块时指定的包含文件是否正确加载
    sources = [util.getpath("tests", "src", "regression", "incfile.f90")]
    options = [f"-I{util.getpath('tests', 'src', 'regression')}",
               f"--include-paths {util.getpath('tests', 'src', 'regression')}"]

    @pytest.mark.slow
    def test_gh25344(self):
        exp = 7.0
        res = self.module.add(3.0, 4.0)
        assert exp == res


class TestF77Comments(util.F2PyTest):
    # 检查从 F77 连续行中剥离注释是否正确
    sources = [util.getpath("tests", "src", "regression", "f77comments.f")]

    @pytest.mark.slow
    # 定义一个测试方法，用于测试 GH26148 的情况
    def test_gh26148(self):
        # 创建一个包含单个整数值 3 的 numpy 数组，数据类型为 int32
        x1 = np.array(3, dtype=np.int32)
        # 创建一个包含单个整数值 5 的 numpy 数组，数据类型为 int32
        x2 = np.array(5, dtype=np.int32)
        # 调用被测试模块的 testsub 方法，传入 x1 和 x2 作为参数，获取返回值
        res = self.module.testsub(x1, x2)
        # 断言返回值的第一个元素是否为 8
        assert(res[0] == 8)
        # 断言返回值的第二个元素是否为 15
        assert(res[1] == 15)

    # 使用 pytest 的标记 @pytest.mark.slow，定义一个慢速测试方法
    def test_gh26466(self):
        # 创建一个预期结果数组，包含从 1 到 10 的浮点数，步长为 2
        expected = np.arange(1, 11, dtype=np.float32) * 2
        # 调用被测试模块的 testsub2 方法，获取返回值
        res = self.module.testsub2()
        # 使用 numpy.testing 库中的 assert_allclose 方法，比较预期结果和实际结果的近似程度
        npt.assert_allclose(expected, res)
class TestF90Contiuation(util.F2PyTest):
    # 定义一个测试类，继承自util.F2PyTest，用于测试Fortran90的连续行中是否正确处理了注释
    sources = [util.getpath("tests", "src", "regression", "f90continuation.f90")]

    @pytest.mark.slow
    def test_gh26148b(self):
        # 测试函数，标记为较慢执行的测试
        x1 = np.array(3, dtype=np.int32)
        x2 = np.array(5, dtype=np.int32)
        # 调用self.module中的testsub函数，并验证返回值
        res=self.module.testsub(x1, x2)
        assert(res[0] == 8)
        assert(res[1] == 15)

@pytest.mark.slow
def test_gh26623():
    # 测试函数，验证包含有.的库名是否能正确生成meson.build文件
    try:
        aa = util.build_module(
            [util.getpath("tests", "src", "regression", "f90continuation.f90")],
            ["-lfoo.bar"],
            module_name="Blah",
        )
    except RuntimeError as rerr:
        assert "lparen got assign" not in str(rerr)


@pytest.mark.slow
@pytest.mark.skipif(platform.system() not in ['Linux', 'Darwin'], reason='Unsupported on this platform for now')
def test_gh25784():
    # 测试函数，根据传递的标志编译一个可疑的文件
    try:
        aa = util.build_module(
            [util.getpath("tests", "src", "regression", "f77fixedform.f95")],
            options=[
                # Meson将会收集并去重这些标志，传递给fortran_args:
                "--f77flags='-ffixed-form -O2'",
                "--f90flags=\"-ffixed-form -Og\"",
            ],
            module_name="Blah",
        )
    except ImportError as rerr:
        assert "unknown_subroutine_" in str(rerr)
```