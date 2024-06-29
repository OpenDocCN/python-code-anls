# `.\numpy\numpy\_core\tests\test_scalarprint.py`

```py
# 导入所需的模块和库
import code
import platform
import pytest
import sys

# 导入临时文件对象
from tempfile import TemporaryFile

# 导入 NumPy 库并引入特定的函数和变量
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_MUSL

# 定义测试类 TestRealScalars，用于测试实数标量
class TestRealScalars:
    
    # 定义测试方法 test_str，测试实数标量的字符串表示
    def test_str(self):
        # 待测试的标量值
        svals = [0.0, -0.0, 1, -1, np.inf, -np.inf, np.nan]
        # 待测试的数据类型列表
        styps = [np.float16, np.float32, np.float64, np.longdouble]
        # 预期的字符串表示
        wanted = [
             ['0.0',  '0.0',  '0.0',  '0.0' ],
             ['-0.0', '-0.0', '-0.0', '-0.0'],
             ['1.0',  '1.0',  '1.0',  '1.0' ],
             ['-1.0', '-1.0', '-1.0', '-1.0'],
             ['inf',  'inf',  'inf',  'inf' ],
             ['-inf', '-inf', '-inf', '-inf'],
             ['nan',  'nan',  'nan',  'nan']]
        
        # 遍历待测试的标量和数据类型
        for wants, val in zip(wanted, svals):
            for want, styp in zip(wants, styps):
                # 构建测试消息
                msg = 'for str({}({}))'.format(np.dtype(styp).name, repr(val))
                # 断言标量的字符串表示符合预期
                assert_equal(str(styp(val)), want, err_msg=msg)

    # 定义测试方法 test_scalar_cutoffs，测试实数标量的截断情况
    def test_scalar_cutoffs(self):
        # 检查 np.float64 的字符串表示是否与 Python 浮点数相同
        def check(v):
            assert_equal(str(np.float64(v)), str(v))
            assert_equal(str(np.float64(v)), repr(v))
            assert_equal(repr(np.float64(v)), f"np.float64({v!r})")
            assert_equal(repr(np.float64(v)), f"np.float64({v})")
        
        # 检查保留相同的有效数字位数
        check(1.12345678901234567890)
        check(0.0112345678901234567890)
        
        # 检查科学计数法和定点表示之间的切换
        check(1e-5)
        check(1e-4)
        check(1e15)
        check(1e16)
    def test_py2_float_print(self):
        # 测试用例：验证 Python 2 中浮点数打印的行为
        # gh-10753
        # 在 Python 2 中，Python 浮点类型实现了一个过时的方法 tp_print，
        # 当使用 "print" 输出到一个“真实文件”（即不是 StringIO）时，
        # 它会覆盖 tp_repr 和 tp_str。确保我们不继承这种行为。
        x = np.double(0.1999999999999)
        
        # 使用临时文件对象 f 来捕获输出
        with TemporaryFile('r+t') as f:
            # 将 x 的值打印到文件 f 中
            print(x, file=f)
            # 将文件指针移到文件开头
            f.seek(0)
            # 读取文件中的输出内容
            output = f.read()
        
        # 断言捕获的输出与 x 的字符串表示加上换行符相等
        assert_equal(output, str(x) + '\n')
        
        # 在 Python 2 中，使用 float('0.1999999999999') 打印时会以 '0.2' 的
        # 形式显示，但是我们希望 numpy 的 np.double('0.1999999999999') 能够
        # 打印出唯一的值 '0.1999999999999'。

        # gh-11031
        # 只有在 Python 2 的交互式 shell 中，并且当 stdout 是一个“真实”
        # 文件时，最后一个命令的输出会直接打印到 stdout，而不像 print
        # 语句那样需要 Py_PRINT_RAW（不像 print 语句）。因此 `>>> x` 和
        # `>>> print x` 可能是不同的。确保它们是相同的。我发现的唯一获取
        # 到类似提示输出的方法是使用 'code' 模块的实际提示。同样，必须使
        # 用 tempfile 来获取一个“真实”文件。

        # dummy user-input which enters one line and then ctrl-Ds.
        # 定义一个虚拟的用户输入函数，输入一行 'np.sqrt(2)' 然后抛出 EOFError
        def userinput():
            yield 'np.sqrt(2)'
            raise EOFError
        gen = userinput()
        # 定义一个输入函数，用于交互时读取输入
        input_func = lambda prompt="": next(gen)

        # 使用两个临时文件对象 fo 和 fe 来捕获 stdout 和 stderr
        with TemporaryFile('r+t') as fo, TemporaryFile('r+t') as fe:
            orig_stdout, orig_stderr = sys.stdout, sys.stderr
            # 重定向 stdout 和 stderr 到 fo 和 fe
            sys.stdout, sys.stderr = fo, fe

            # 使用 code.interact 来进入交互模式，本地变量包括 np，输入函数为 input_func，banner 为空
            code.interact(local={'np': np}, readfunc=input_func, banner='')

            # 恢复原始的 stdout 和 stderr
            sys.stdout, sys.stderr = orig_stdout, orig_stderr

            # 将 fo 的文件指针移到开头，读取所有内容并去除首尾空白字符
            fo.seek(0)
            capture = fo.read().strip()

        # 断言捕获的输出与 np.sqrt(2) 的 repr 相等
        assert_equal(capture, repr(np.sqrt(2)))
    # 定义测试函数 test_dragon4_interface，用于测试浮点数格式化接口
    def test_dragon4_interface(self):
        # 定义浮点数类型列表，包括 np.float16, np.float32, np.float64
        tps = [np.float16, np.float32, np.float64]
        # 如果系统支持 np.float128 并且不是 musllinux 环境，则添加 np.float128 到测试类型列表
        if hasattr(np, 'float128') and not IS_MUSL:
            tps.append(np.float128)

        # 获取 numpy 中的浮点数格式化函数
        fpos = np.format_float_positional
        fsci = np.format_float_scientific

        # 遍历测试类型列表 tps
        for tp in tps:
            # 测试填充功能，使用 assert_equal 断言检查结果是否符合预期
            assert_equal(fpos(tp('1.0'), pad_left=4, pad_right=4), "   1.    ")
            assert_equal(fpos(tp('-1.0'), pad_left=4, pad_right=4), "  -1.    ")
            assert_equal(fpos(tp('-10.2'),
                         pad_left=4, pad_right=4), " -10.2   ")

            # 测试科学计数法格式化，使用 assert_equal 断言检查结果是否符合预期
            assert_equal(fsci(tp('1.23e1'), exp_digits=5), "1.23e+00001")

            # 测试固定位数模式下的格式化，使用 assert_equal 断言检查结果是否符合预期
            assert_equal(fpos(tp('1.0'), unique=False, precision=4), "1.0000")
            assert_equal(fsci(tp('1.0'), unique=False, precision=4),
                         "1.0000e+00")

            # 测试修剪功能，根据不同的 trim 参数进行测试，使用 assert_equal 断言检查结果是否符合预期
            assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='k'),
                         "1.0000")

            assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='.'),
                         "1.")
            assert_equal(fpos(tp('1.2'), unique=False, precision=4, trim='.'),
                         "1.2" if tp != np.float16 else "1.2002")

            assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='0'),
                         "1.0")
            assert_equal(fpos(tp('1.2'), unique=False, precision=4, trim='0'),
                         "1.2" if tp != np.float16 else "1.2002")
            assert_equal(fpos(tp('1.'), trim='0'), "1.0")

            assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='-'),
                         "1")
            assert_equal(fpos(tp('1.2'), unique=False, precision=4, trim='-'),
                         "1.2" if tp != np.float16 else "1.2002")
            assert_equal(fpos(tp('1.'), trim='-'), "1")
            assert_equal(fpos(tp('1.001'), precision=1, trim='-'), "1")

    # 使用 pytest.mark.skipif 装饰器标记测试用例，在非 ppc64 机器上跳过测试，给出原因说明
    @pytest.mark.skipif(not platform.machine().startswith("ppc64"),
                        reason="only applies to ppc float128 values")
    def test_ppc64_ibm_double_double128(self):
        # 测试函数，验证在子正常范围内精度降低的情况。
        # 不像 float64，这里从约 1e-292 开始而不是 1e-308，
        # 当第一个 double 是正常的，第二个是子正常的时候发生。
        
        # 定义一个 np.float128 类型的变量 x，初始值为 '2.123123123123123123123123123123123e-286'
        x = np.float128('2.123123123123123123123123123123123e-286')
        
        # 生成一个列表，包含 0 到 39 的字符串形式的 x 除以 np.float128('2e' + str(i)) 的结果
        got = [str(x/np.float128('2e' + str(i))) for i in range(0,40)]
        
        # 预期的结果列表
        expected = [
            "1.06156156156156156156156156156157e-286",
            "1.06156156156156156156156156156158e-287",
            "1.06156156156156156156156156156159e-288",
            "1.0615615615615615615615615615616e-289",
            "1.06156156156156156156156156156157e-290",
            "1.06156156156156156156156156156156e-291",
            "1.0615615615615615615615615615616e-292",
            "1.0615615615615615615615615615615e-293",
            "1.061561561561561561561561561562e-294",
            "1.06156156156156156156156156155e-295",
            "1.0615615615615615615615615615616e-296",
            "1.06156156156156156156156156156156e-297",
            "1.06156156156156156156156156156157e-298",
            "1.0615615615615615615615615615616e-299",
            "1.0615615615615615615615615615615e-300",
            "1.061561561561561561561561561562e-301",
            "1.06156156156156156156156156155e-302",
            "1.0615615615615615615615615616e-303",
            "1.061561561561561561561561562e-304",
            "1.0615615615615615615615615616e-305",
            "1.0615615615615615615615615616e-306",
            "1.06156156156156156156156156156e-307",
            "1.0615615615615615615615615615616e-308",
            "1.06156156156156156156156156156e-309",
            "1.06156156156156157e-310",
            "1.0615615615615616e-311",
            "1.06156156156156e-312",
            "1.06156156156157e-313",
            "1.0615615615616e-314",
            "1.06156156156e-315",
            "1.06156156155e-316",
            "1.061562e-317",
            "1.06156e-318",
            "1.06155e-319",
            "1.0617e-320",
            "1.06e-321",
            "1.04e-322",
            "1e-323",
            "0.0",
            "0.0"]
        
        # 使用 assert_equal 函数断言 got 和 expected 相等
        assert_equal(got, expected)

        # 注意：我们遵循 glibc 的行为，但它（或者 gcc）可能不正确。
        # 特别是我们可以得到两个打印相同但不相等的值：
        
        # 定义变量 a，值为 np.float128('2')/np.float128('3')
        a = np.float128('2')/np.float128('3')
        
        # 定义变量 b，值为 np.float128(str(a))
        b = np.float128(str(a))
        
        # 使用 assert_equal 函数断言变量 a 和 b 的字符串表示相等
        assert_equal(str(a), str(b))
        
        # 使用 assert_ 函数断言 a 不等于 b
        assert_(a != b)

    def float32_roundtrip(self):
        # 测试函数，验证 float32 的 roundtrip 行为
        # gh-9360
        
        # 定义变量 x，值为 np.float32(1024 - 2**-14)
        x = np.float32(1024 - 2**-14)
        
        # 定义变量 y，值为 np.float32(1024 - 2**-13)
        y = np.float32(1024 - 2**-13)
        
        # 使用 assert_ 函数断言 x 的字符串表示不等于 y 的字符串表示
        assert_(repr(x) != repr(y))
        
        # 使用 assert_equal 函数断言 np.float32(repr(x)) 等于 x
        assert_equal(np.float32(repr(x)), x)
        
        # 使用 assert_equal 函数断言 np.float32(repr(y)) 等于 y
        assert_equal(np.float32(repr(y)), y)

    def float64_vs_python(self):
        # 测试函数，验证 float64 与 Python 浮点数表示的比较行为
        # gh-2643, gh-6136, gh-6908
        
        # 使用 assert_equal 函数断言 np.float64(0.1) 的字符串表示等于 Python 浮点数 0.1 的字符串表示
        assert_equal(repr(np.float64(0.1)), repr(0.1))
        
        # 使用 assert_ 函数断言 np.float64(0.20000000000000004) 的字符串表示不等于 Python 浮点数 0.2 的字符串表示
        assert_(repr(np.float64(0.20000000000000004)) != repr(0.2))
```