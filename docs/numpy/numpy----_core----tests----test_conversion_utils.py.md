# `.\numpy\numpy\_core\tests\test_conversion_utils.py`

```
"""
Tests for numpy/_core/src/multiarray/conversion_utils.c
"""
# 导入正则表达式模块
import re
# 导入系统模块
import sys

# 导入 pytest 测试框架
import pytest

# 导入 numpy 库，并重命名为 np
import numpy as np
# 导入 numpy._core._multiarray_tests 模块，并重命名为 mt
import numpy._core._multiarray_tests as mt
# 从 numpy._core.multiarray 中导入 CLIP、WRAP、RAISE 常量
from numpy._core.multiarray import CLIP, WRAP, RAISE
# 从 numpy.testing 中导入 assert_warns 和 IS_PYPY 函数
from numpy.testing import assert_warns, IS_PYPY


class StringConverterTestCase:
    allow_bytes = True
    case_insensitive = True
    exact_match = False
    warn = True

    def _check_value_error(self, val):
        # 构建匹配模式，用于在异常信息中匹配给定的值
        pattern = r'\(got {}\)'.format(re.escape(repr(val)))
        # 使用 pytest 检查是否抛出 ValueError 异常，并匹配指定模式
        with pytest.raises(ValueError, match=pattern) as exc:
            self.conv(val)

    def _check_conv_assert_warn(self, val, expected):
        # 如果设置了 warn 标志，使用 assert_warns 检查是否会有 DeprecationWarning 警告
        if self.warn:
            with assert_warns(DeprecationWarning) as exc:
                assert self.conv(val) == expected
        else:
            # 否则直接比较转换后的结果和期望值
            assert self.conv(val) == expected

    def _check(self, val, expected):
        """Takes valid non-deprecated inputs for converters,
        runs converters on inputs, checks correctness of outputs,
        warnings and errors"""
        # 检查转换器是否能够正确转换给定的值
        assert self.conv(val) == expected

        # 如果允许处理字节类型，则尝试使用 ASCII 编码进行转换
        if self.allow_bytes:
            assert self.conv(val.encode('ascii')) == expected
        else:
            # 如果不允许字节类型，则期望抛出 TypeError 异常
            with pytest.raises(TypeError):
                self.conv(val.encode('ascii'))

        # 如果字符串长度大于 1，则进一步检查边界情况和警告
        if len(val) != 1:
            if self.exact_match:
                # 如果需要精确匹配，检查部分字符和空字符的处理
                self._check_value_error(val[:1])
                self._check_value_error(val + '\0')
            else:
                # 否则，检查部分字符转换时是否会警告
                self._check_conv_assert_warn(val[:1], expected)

        # 如果忽略大小写，则检查转换器对大小写的处理
        if self.case_insensitive:
            if val != val.lower():
                self._check_conv_assert_warn(val.lower(), expected)
            if val != val.upper():
                self._check_conv_assert_warn(val.upper(), expected)
        else:
            # 如果不忽略大小写，则期望转换器无法处理大小写变化
            if val != val.lower():
                self._check_value_error(val.lower())
            if val != val.upper():
                self._check_value_error(val.upper())

    def test_wrong_type(self):
        # 测试不支持的输入类型是否会抛出 TypeError 异常
        with pytest.raises(TypeError):
            self.conv({})
        with pytest.raises(TypeError):
            self.conv([])

    def test_wrong_value(self):
        # 测试不合理的字符串是否会抛出 ValueError 异常
        self._check_value_error('')
        self._check_value_error('\N{greek small letter pi}')

        # 如果允许处理字节类型，进一步测试字节类型的不合理转换是否会抛出异常
        if self.allow_bytes:
            self._check_value_error(b'')
            self._check_value_error(b"\xFF")
        if self.exact_match:
            # 如果需要精确匹配，测试不支持的字符串是否会抛出异常
            self._check_value_error("there's no way this is supported")


class TestByteorderConverter(StringConverterTestCase):
    """ Tests of PyArray_ByteorderConverter """
    # 将转换函数指定为 mt.run_byteorder_converter
    conv = mt.run_byteorder_converter
    # 禁用警告标志
    warn = False
    # 定义测试方法 test_valid，用于验证不同字符串对应的调用 _check 方法的正确性
    def test_valid(self):
        # 对于字符串列表 ['big', '>']，分别调用 _check 方法，期望结果为 'NPY_BIG'
        for s in ['big', '>']:
            self._check(s, 'NPY_BIG')
        
        # 对于字符串列表 ['little', '<']，分别调用 _check 方法，期望结果为 'NPY_LITTLE'
        for s in ['little', '<']:
            self._check(s, 'NPY_LITTLE')
        
        # 对于字符串列表 ['native', '=']，分别调用 _check 方法，期望结果为 'NPY_NATIVE'
        for s in ['native', '=']:
            self._check(s, 'NPY_NATIVE')
        
        # 对于字符串列表 ['ignore', '|']，分别调用 _check 方法，期望结果为 'NPY_IGNORE'
        for s in ['ignore', '|']:
            self._check(s, 'NPY_IGNORE')
        
        # 对于字符串列表 ['swap']，只调用一次 _check 方法，期望结果为 'NPY_SWAP'
        for s in ['swap']:
            self._check(s, 'NPY_SWAP')
# 测试 PyArray_SortkindConverter 的测试用例类
class TestSortkindConverter(StringConverterTestCase):
    """ Tests of PyArray_SortkindConverter """
    
    # 设置转换函数为 mt.run_sortkind_converter
    conv = mt.run_sortkind_converter
    # 禁止警告信息
    warn = False

    # 定义有效性测试方法
    def test_valid(self):
        # 调用 _check 方法，验证 'quicksort' 转换为 'NPY_QUICKSORT'
        self._check('quicksort', 'NPY_QUICKSORT')
        # 调用 _check 方法，验证 'heapsort' 转换为 'NPY_HEAPSORT'
        self._check('heapsort', 'NPY_HEAPSORT')
        # 调用 _check 方法，验证 'mergesort' 转换为 'NPY_STABLESORT'（别名）
        self._check('mergesort', 'NPY_STABLESORT')
        # 调用 _check 方法，验证 'stable' 转换为 'NPY_STABLESORT'
        self._check('stable', 'NPY_STABLESORT')


# 测试 PyArray_SelectkindConverter 的测试用例类
class TestSelectkindConverter(StringConverterTestCase):
    """ Tests of PyArray_SelectkindConverter """
    
    # 设置转换函数为 mt.run_selectkind_converter
    conv = mt.run_selectkind_converter
    # 不区分大小写
    case_insensitive = False
    # 精确匹配
    exact_match = True

    # 定义有效性测试方法
    def test_valid(self):
        # 调用 _check 方法，验证 'introselect' 转换为 'NPY_INTROSELECT'
        self._check('introselect', 'NPY_INTROSELECT')


# 测试 PyArray_SearchsideConverter 的测试用例类
class TestSearchsideConverter(StringConverterTestCase):
    """ Tests of PyArray_SearchsideConverter """
    
    # 设置转换函数为 mt.run_searchside_converter
    conv = mt.run_searchside_converter

    # 定义有效性测试方法
    def test_valid(self):
        # 调用 _check 方法，验证 'left' 转换为 'NPY_SEARCHLEFT'
        self._check('left', 'NPY_SEARCHLEFT')
        # 调用 _check 方法，验证 'right' 转换为 'NPY_SEARCHRIGHT'
        self._check('right', 'NPY_SEARCHRIGHT')


# 测试 PyArray_OrderConverter 的测试用例类
class TestOrderConverter(StringConverterTestCase):
    """ Tests of PyArray_OrderConverter """
    
    # 设置转换函数为 mt.run_order_converter
    conv = mt.run_order_converter
    # 禁止警告信息
    warn = False

    # 定义有效性测试方法
    def test_valid(self):
        # 调用 _check 方法，验证 'c' 转换为 'NPY_CORDER'
        self._check('c', 'NPY_CORDER')
        # 调用 _check 方法，验证 'f' 转换为 'NPY_FORTRANORDER'
        self._check('f', 'NPY_FORTRANORDER')
        # 调用 _check 方法，验证 'a' 转换为 'NPY_ANYORDER'
        self._check('a', 'NPY_ANYORDER')
        # 调用 _check 方法，验证 'k' 转换为 'NPY_KEEPORDER'
        self._check('k', 'NPY_KEEPORDER')

    # 定义无效顺序测试方法
    def test_flatten_invalid_order(self):
        # 在 gh-14596 之后为无效，引发 ValueError 异常
        with pytest.raises(ValueError):
            self.conv('Z')
        # 针对 [False, True, 0, 8] 中的每个顺序，引发 TypeError 异常
        for order in [False, True, 0, 8]:
            with pytest.raises(TypeError):
                self.conv(order)


# 测试 PyArray_ClipmodeConverter 的测试用例类
class TestClipmodeConverter(StringConverterTestCase):
    """ Tests of PyArray_ClipmodeConverter """
    
    # 设置转换函数为 mt.run_clipmode_converter
    conv = mt.run_clipmode_converter

    # 定义有效性测试方法
    def test_valid(self):
        # 调用 _check 方法，验证 'clip' 转换为 'NPY_CLIP'
        self._check('clip', 'NPY_CLIP')
        # 调用 _check 方法，验证 'wrap' 转换为 'NPY_WRAP'
        self._check('wrap', 'NPY_WRAP')
        # 调用 _check 方法，验证 'raise' 转换为 'NPY_RAISE'
        self._check('raise', 'NPY_RAISE')

        # 整数值在此处允许
        assert self.conv(CLIP) == 'NPY_CLIP'
        assert self.conv(WRAP) == 'NPY_WRAP'
        assert self.conv(RAISE) == 'NPY_RAISE'


# 测试 PyArray_CastingConverter 的测试用例类
class TestCastingConverter(StringConverterTestCase):
    """ Tests of PyArray_CastingConverter """
    
    # 设置转换函数为 mt.run_casting_converter
    conv = mt.run_casting_converter
    # 不区分大小写
    case_insensitive = False
    # 精确匹配
    exact_match = True

    # 定义有效性测试方法
    def test_valid(self):
        # 调用 _check 方法，验证 "no" 转换为 "NPY_NO_CASTING"
        self._check("no", "NPY_NO_CASTING")
        # 调用 _check 方法，验证 "equiv" 转换为 "NPY_EQUIV_CASTING"
        self._check("equiv", "NPY_EQUIV_CASTING")
        # 调用 _check 方法，验证 "safe" 转换为 "NPY_SAFE_CASTING"
        self._check("safe", "NPY_SAFE_CASTING")
        # 调用 _check 方法，验证 "same_kind" 转换为 "NPY_SAME_KIND_CASTING"
        self._check("same_kind", "NPY_SAME_KIND_CASTING")
        # 调用 _check 方法，验证 "unsafe" 转换为 "NPY_UNSAFE_CASTING"
        self._check("unsafe", "NPY_UNSAFE_CASTING")


# 测试 PyArray_IntpConverter 的测试用例类
class TestIntpConverter:
    """ Tests of PyArray_IntpConverter """
    
    # 设置转换函数为 mt.run_intp_converter
    conv = mt.run_intp_converter

    # 定义基础测试方法
    def test_basic(self):
        # 断言 self.conv(1) 返回 (1,)
        assert self.conv(1) == (1,)
        # 断言 self.conv((1, 2)) 返回 (1, 2)
        assert self.conv((1, 2)) == (1, 2)
        # 断言 self.conv([1, 2]) 返回 (1, 2)
        assert self.conv([1, 2]) == (1, 2)
        # 断言 self.conv(()) 返回 ()
        assert self.conv(()) == ()

    # 定义空值测试方法
    def test_none(self):
        # 在警告过期后，此处将引发 DeprecationWarning
        with pytest.warns(DeprecationWarning):
            # 断言 self.conv(None) 返回 ()
            assert self.conv(None) == ()
    # 使用 pytest.mark.skipif 装饰器标记测试用例，当满足指定条件时跳过执行
    @pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                        reason="PyPy bug in error formatting")
    # 测试浮点数输入时的情况
    def test_float(self):
        # 断言输入浮点数时会引发 TypeError 异常
        with pytest.raises(TypeError):
            self.conv(1.0)
        # 断言输入包含浮点数的列表时会引发 TypeError 异常
        with pytest.raises(TypeError):
            self.conv([1, 1.0])

    # 测试输入值过大时的情况
    def test_too_large(self):
        # 断言输入超出限制值时会引发 ValueError 异常
        with pytest.raises(ValueError):
            self.conv(2**64)

    # 测试输入维度过多时的情况
    def test_too_many_dims(self):
        # 断言输入维度在限制范围内时，返回符合预期的结果
        assert self.conv([1]*64) == (1,)*64
        # 断言输入维度超出限制时会引发 ValueError 异常
        with pytest.raises(ValueError):
            self.conv([1]*65)
```