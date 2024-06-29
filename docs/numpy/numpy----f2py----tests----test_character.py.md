# `.\numpy\numpy\f2py\tests\test_character.py`

```py
# 导入 pytest 模块，用于测试
import pytest
# 导入 textwrap 模块，用于处理字符串的缩进和格式
import textwrap
# 导入 numpy.testing 模块中的断言函数，用于数组比较和异常断言
from numpy.testing import assert_array_equal, assert_equal, assert_raises
# 导入 numpy 模块，并使用 np 别名
import numpy as np
# 导入 numpy.f2py.tests 中的 util 模块
from numpy.f2py.tests import util

# 用 @pytest.mark.slow 装饰器标记这个测试类为慢速测试
@pytest.mark.slow
# 定义一个测试类 TestCharacterString，继承自 util.F2PyTest
class TestCharacterString(util.F2PyTest):
    # 设置文件后缀名为 '.f90'
    suffix = '.f90'
    # 定义文件名前缀为 'test_character_string'
    fprefix = 'test_character_string'
    # 定义长度列表，包含字符串长度 '1', '3', 'star'
    length_list = ['1', '3', 'star']

    # 初始化代码字符串为空
    code = ''
    # 遍历长度列表
    for length in length_list:
        # 设置文件后缀为当前长度
        fsuffix = length
        # 根据长度设置字符长度 clength，将 'star' 转换为 '(*)'
        clength = dict(star='(*)').get(length, length)

        # 构建 Fortran 子程序的代码段，使用 textwrap.dedent 去除缩进
        code += textwrap.dedent(f"""
        
        subroutine {fprefix}_input_{fsuffix}(c, o, n)
          character*{clength}, intent(in) :: c
          integer n
          !f2py integer, depend(c), intent(hide) :: n = slen(c)
          integer*1, dimension(n) :: o
          !f2py intent(out) o
          o = transfer(c, o)
        end subroutine {fprefix}_input_{fsuffix}

        subroutine {fprefix}_output_{fsuffix}(c, o, n)
          character*{clength}, intent(out) :: c
          integer n
          integer*1, dimension(n), intent(in) :: o
          !f2py integer, depend(o), intent(hide) :: n = len(o)
          c = transfer(o, c)
        end subroutine {fprefix}_output_{fsuffix}

        subroutine {fprefix}_array_input_{fsuffix}(c, o, m, n)
          integer m, i, n
          character*{clength}, intent(in), dimension(m) :: c
          !f2py integer, depend(c), intent(hide) :: m = len(c)
          !f2py integer, depend(c), intent(hide) :: n = f2py_itemsize(c)
          integer*1, dimension(m, n), intent(out) :: o
          do i=1,m
            o(i, :) = transfer(c(i), o(i, :))
          end do
        end subroutine {fprefix}_array_input_{fsuffix}

        subroutine {fprefix}_array_output_{fsuffix}(c, o, m, n)
          character*{clength}, intent(out), dimension(m) :: c
          integer n
          integer*1, dimension(m, n), intent(in) :: o
          !f2py character(f2py_len=n) :: c
          !f2py integer, depend(o), intent(hide) :: m = len(o)
          !f2py integer, depend(o), intent(hide) :: n = shape(o, 1)
          do i=1,m
            c(i) = transfer(o(i, :), c(i))
          end do
        end subroutine {fprefix}_array_output_{fsuffix}

        subroutine {fprefix}_2d_array_input_{fsuffix}(c, o, m1, m2, n)
          integer m1, m2, i, j, n
          character*{clength}, intent(in), dimension(m1, m2) :: c
          !f2py integer, depend(c), intent(hide) :: m1 = len(c)
          !f2py integer, depend(c), intent(hide) :: m2 = shape(c, 1)
          !f2py integer, depend(c), intent(hide) :: n = f2py_itemsize(c)
          integer*1, dimension(m1, m2, n), intent(out) :: o
          do i=1,m1
            do j=1,m2
              o(i, j, :) = transfer(c(i, j), o(i, j, :))
            end do
          end do
        end subroutine {fprefix}_2d_array_input_{fsuffix}
        """)

    # 使用 @pytest.mark.parametrize 装饰器将 length_list 的值作为参数传入测试方法
    @pytest.mark.parametrize("length", length_list)
    # 定义一个测试函数，用于测试输入处理函数，参数为长度
    def test_input(self, length):
        # 根据长度选择文件后缀名，如果找不到对应的长度，使用本身作为后缀
        fsuffix = {'(*)': 'star'}.get(length, length)
        # 根据测试对象模块和函数名构造函数对象
        f = getattr(self.module, self.fprefix + '_input_' + fsuffix)

        # 根据长度选择字符串，构造输入数据a
        a = {'1': 'a', '3': 'abc', 'star': 'abcde' * 3}[length]

        # 断言输入数据经处理后与预期的字节表示一致
        assert_array_equal(f(a), np.array(list(map(ord, a)), dtype='u1'))

    # 使用参数化测试装饰器，测试输出处理函数，参数为长度
    @pytest.mark.parametrize("length", length_list[:-1])
    def test_output(self, length):
        # 设置函数后缀为长度本身
        fsuffix = length
        # 根据测试对象模块和函数名构造函数对象
        f = getattr(self.module, self.fprefix + '_output_' + fsuffix)

        # 根据长度选择字符串，构造输入数据a
        a = {'1': 'a', '3': 'abc'}[length]

        # 断言经处理后输出与预期的字节编码一致
        assert_array_equal(f(np.array(list(map(ord, a)), dtype='u1')),
                           a.encode())

    # 使用参数化测试装饰器，测试数组输入处理函数，参数为长度
    @pytest.mark.parametrize("length", length_list)
    def test_array_input(self, length):
        # 设置函数后缀为长度本身
        fsuffix = length
        # 根据测试对象模块和函数名构造函数对象
        f = getattr(self.module, self.fprefix + '_array_input_' + fsuffix)

        # 根据长度选择字符串，构造输入数据a，以及对应的大写版本
        a = np.array([{'1': 'a', '3': 'abc', 'star': 'abcde' * 3}[length],
                      {'1': 'A', '3': 'ABC', 'star': 'ABCDE' * 3}[length],
                      ], dtype='S')

        # 构造预期的字节表示的二维数组
        expected = np.array([[c for c in s] for s in a], dtype='u1')
        # 断言处理后的输出与预期的二维数组一致
        assert_array_equal(f(a), expected)

    # 使用参数化测试装饰器，测试数组输出处理函数，参数为长度
    @pytest.mark.parametrize("length", length_list)
    def test_array_output(self, length):
        # 设置函数后缀为长度本身
        fsuffix = length
        # 根据测试对象模块和函数名构造函数对象
        f = getattr(self.module, self.fprefix + '_array_output_' + fsuffix)

        # 根据长度选择字符串，构造预期的输出数据a
        expected = np.array(
            [{'1': 'a', '3': 'abc', 'star': 'abcde' * 3}[length],
             {'1': 'A', '3': 'ABC', 'star': 'ABCDE' * 3}[length]], dtype='S')

        # 构造预期的字节表示的二维数组
        a = np.array([[c for c in s] for s in expected], dtype='u1')
        # 断言处理后的输出与预期的二维数组一致
        assert_array_equal(f(a), expected)

    # 使用参数化测试装饰器，测试二维数组输入处理函数，参数为长度
    @pytest.mark.parametrize("length", length_list)
    def test_2d_array_input(self, length):
        # 设置函数后缀为长度本身
        fsuffix = length
        # 根据测试对象模块和函数名构造函数对象
        f = getattr(self.module, self.fprefix + '_2d_array_input_' + fsuffix)

        # 根据长度选择字符串，构造输入数据a及其大写版本的二维数组
        a = np.array([[{'1': 'a', '3': 'abc', 'star': 'abcde' * 3}[length],
                       {'1': 'A', '3': 'ABC', 'star': 'ABCDE' * 3}[length]],
                      [{'1': 'f', '3': 'fgh', 'star': 'fghij' * 3}[length],
                       {'1': 'F', '3': 'FGH', 'star': 'FGHIJ' * 3}[length]]],
                     dtype='S')
        # 构造预期的字节表示的三维数组，按Fortran顺序排列
        expected = np.array([[[c for c in item] for item in row] for row in a],
                            dtype='u1', order='F')
        # 断言处理后的输出与预期的三维数组一致
        assert_array_equal(f(a), expected)
class TestCharacter(util.F2PyTest):
    # 定义一个测试类 TestCharacter，继承自 util.F2PyTest
    # 设置类变量 suffix 为 '.f90'，用于表示文件后缀名
    suffix = '.f90'
    # 设置类变量 fprefix 为 'test_character'，用于表示文件名前缀
    fprefix = 'test_character'
    code = textwrap.dedent(f"""
       subroutine {fprefix}_input(c, o)
          character, intent(in) :: c
          integer*1 o
          !f2py intent(out) o
          o = transfer(c, o)
       end subroutine {fprefix}_input

       subroutine {fprefix}_output(c, o)
          character :: c
          integer*1, intent(in) :: o
          !f2py intent(out) c
          c = transfer(o, c)
       end subroutine {fprefix}_output

       subroutine {fprefix}_input_output(c, o)
          character, intent(in) :: c
          character o
          !f2py intent(out) o
          o = c
       end subroutine {fprefix}_input_output

       subroutine {fprefix}_inout(c, n)
          character :: c, n
          !f2py intent(in) n
          !f2py intent(inout) c
          c = n
       end subroutine {fprefix}_inout

       function {fprefix}_return(o) result (c)
          character :: c
          character, intent(in) :: o
          c = transfer(o, c)
       end function {fprefix}_return

       subroutine {fprefix}_array_input(c, o)
          character, intent(in) :: c(3)
          integer*1 o(3)
          !f2py intent(out) o
          integer i
          do i=1,3
            o(i) = transfer(c(i), o(i))
          end do
       end subroutine {fprefix}_array_input

       subroutine {fprefix}_2d_array_input(c, o)
          character, intent(in) :: c(2, 3)
          integer*1 o(2, 3)
          !f2py intent(out) o
          integer i, j
          do i=1,2
            do j=1,3
              o(i, j) = transfer(c(i, j), o(i, j))
            end do
          end do
       end subroutine {fprefix}_2d_array_input

       subroutine {fprefix}_array_output(c, o)
          character :: c(3)
          integer*1, intent(in) :: o(3)
          !f2py intent(out) c
          do i=1,3
            c(i) = transfer(o(i), c(i))
          end do
       end subroutine {fprefix}_array_output

       subroutine {fprefix}_array_inout(c, n)
          character :: c(3), n(3)
          !f2py intent(in) n(3)
          !f2py intent(inout) c(3)
          do i=1,3
            c(i) = n(i)
          end do
       end subroutine {fprefix}_array_inout

       subroutine {fprefix}_2d_array_inout(c, n)
          character :: c(2, 3), n(2, 3)
          !f2py intent(in) n(2, 3)
          !f2py intent(inout) c(2, 3)
          integer i, j
          do i=1,2
            do j=1,3
              c(i, j) = n(i, j)
            end do
          end do
       end subroutine {fprefix}_2d_array_inout

       function {fprefix}_array_return(o) result (c)
          character, dimension(3) :: c
          character, intent(in) :: o(3)
          do i=1,3
            c(i) = o(i)
          end do
       end function {fprefix}_array_return

       function {fprefix}_optional(o) result (c)
          character, intent(in) :: o
          !f2py character o = "a"
          character :: c
          c = o
       end function {fprefix}_optional
    """)

    # 使用 textwrap.dedent 对字符串进行缩进处理，以避免多余的空格
    @pytest.mark.parametrize("dtype", ['c', 'S1'])
    # 使用 pytest.mark.parametrize 创建一个参数化测试，测试 dtype 参数为 'c' 和 'S1'
    # 定义一个测试函数，用于测试输入类型为dtype的情况
    def test_input(self, dtype):
        # 获取要测试的输入函数的引用
        f = getattr(self.module, self.fprefix + '_input')

        # 断言函数对于不同类型的输入返回的结果是否符合预期
        assert_equal(f(np.array('a', dtype=dtype)), ord('a'))
        assert_equal(f(np.array(b'a', dtype=dtype)), ord('a'))
        assert_equal(f(np.array(['a'], dtype=dtype)), ord('a'))
        assert_equal(f(np.array('abc', dtype=dtype)), ord('a'))
        assert_equal(f(np.array([['a']], dtype=dtype)), ord('a'))

    # 定义另一个测试函数，用于测试不同类型和格式的输入情况
    def test_input_varia(self):
        # 获取要测试的输入函数的引用
        f = getattr(self.module, self.fprefix + '_input')

        # 测试函数对于不同类型和格式的输入是否能正确返回预期的结果
        assert_equal(f('a'), ord('a'))
        assert_equal(f(b'a'), ord(b'a'))
        assert_equal(f(''), 0)
        assert_equal(f(b''), 0)
        assert_equal(f(b'\0'), 0)
        assert_equal(f('ab'), ord('a'))
        assert_equal(f(b'ab'), ord('a'))
        assert_equal(f(['a']), ord('a'))

        # 使用NumPy数组作为输入进行进一步测试
        assert_equal(f(np.array(b'a')), ord('a'))
        assert_equal(f(np.array([b'a'])), ord('a'))
        a = np.array('a')
        assert_equal(f(a), ord('a'))
        a = np.array(['a'])
        assert_equal(f(a), ord('a'))

        # 测试空列表作为输入时是否会引发预期的异常
        try:
            f([])
        except IndexError as msg:
            if not str(msg).endswith(' got 0-list'):
                raise
        else:
            raise SystemError(f'{f.__name__} should have failed on empty list')

        # 测试整数作为输入时是否会引发预期的异常
        try:
            f(97)
        except TypeError as msg:
            if not str(msg).endswith(' got int instance'):
                raise
        else:
            raise SystemError(f'{f.__name__} should have failed on int value')

    # 使用pytest的参数化功能定义测试数组输入的情况
    @pytest.mark.parametrize("dtype", ['c', 'S1', 'U1'])
    def test_array_input(self, dtype):
        # 获取要测试的数组输入函数的引用
        f = getattr(self.module, self.fprefix + '_array_input')

        # 断言函数对于数组输入返回的结果是否符合预期
        assert_array_equal(f(np.array(['a', 'b', 'c'], dtype=dtype)),
                           np.array(list(map(ord, 'abc')), dtype='i1'))
        assert_array_equal(f(np.array([b'a', b'b', b'c'], dtype=dtype)),
                           np.array(list(map(ord, 'abc')), dtype='i1'))

    # 定义另一个测试函数，用于测试不同类型和格式的数组输入情况
    def test_array_input_varia(self):
        # 获取要测试的数组输入函数的引用
        f = getattr(self.module, self.fprefix + '_array_input')

        # 断言函数对于不同类型和格式的数组输入是否能正确返回预期的结果
        assert_array_equal(f(['a', 'b', 'c']),
                           np.array(list(map(ord, 'abc')), dtype='i1'))
        assert_array_equal(f([b'a', b'b', b'c']),
                           np.array(list(map(ord, 'abc')), dtype='i1'))

        # 测试输入数组长度不为3时是否会引发预期的异常
        try:
            f(['a', 'b', 'c', 'd'])
        except ValueError as msg:
            if not str(msg).endswith(
                    'th dimension must be fixed to 3 but got 4'):
                raise
        else:
            raise SystemError(
                f'{f.__name__} should have failed on wrong input')

    # 使用pytest的参数化功能定义测试数组输入的dtype情况
    @pytest.mark.parametrize("dtype", ['c', 'S1', 'U1'])
    # 定义一个测试方法，用于测试接受二维数组输入的函数，参数为数据类型dtype
    def test_2d_array_input(self, dtype):
        # 获取被测试模块中的相应函数
        f = getattr(self.module, self.fprefix + '_2d_array_input')

        # 创建一个二维数组a，内容为 [['a', 'b', 'c'], ['d', 'e', 'f']]，指定数据类型为dtype，列序为F（列主序）
        a = np.array([['a', 'b', 'c'],
                      ['d', 'e', 'f']], dtype=dtype, order='F')
        # 创建预期结果expected，将数组a视图转换为np.uint32（如果dtype为'U1'）或np.uint8
        expected = a.view(np.uint32 if dtype == 'U1' else np.uint8)
        # 断言调用函数f并传入数组a的返回值与预期结果expected相等
        assert_array_equal(f(a), expected)

    # 定义一个测试方法，测试函数的输出功能
    def test_output(self):
        # 获取被测试模块中的相应函数
        f = getattr(self.module, self.fprefix + '_output')

        # 断言调用函数f并传入ord(b'a')的返回值与b'a'相等
        assert_equal(f(ord(b'a')), b'a')
        # 断言调用函数f并传入0的返回值与b'\0'相等
        assert_equal(f(0), b'\0')

    # 定义一个测试方法，测试函数的数组输出功能
    def test_array_output(self):
        # 获取被测试模块中的相应函数
        f = getattr(self.module, self.fprefix + '_array_output')

        # 断言调用函数f并传入map(ord, 'abc')的返回值的数组与np.array(list('abc'), dtype='S1')相等
        assert_array_equal(f(list(map(ord, 'abc'))),
                           np.array(list('abc'), dtype='S1'))

    # 定义一个测试方法，测试函数的输入输出功能
    def test_input_output(self):
        # 获取被测试模块中的相应函数
        f = getattr(self.module, self.fprefix + '_input_output')

        # 断言调用函数f并传入b'a'的返回值与b'a'相等
        assert_equal(f(b'a'), b'a')
        # 断言调用函数f并传入'a'的返回值与b'a'相等（自动转换为字节串）
        assert_equal(f('a'), b'a')
        # 断言调用函数f并传入''的返回值与b'\0'相等
        assert_equal(f(''), b'\0')

    # 使用pytest的参数化装饰器，定义一个测试方法，测试函数的输入输出功能，参数为dtype
    @pytest.mark.parametrize("dtype", ['c', 'S1'])
    def test_inout(self, dtype):
        # 获取被测试模块中的相应函数
        f = getattr(self.module, self.fprefix + '_inout')

        # 创建一个数组a，内容为['a', 'b', 'c']，指定数据类型为dtype
        a = np.array(list('abc'), dtype=dtype)
        # 调用函数f并传入a及字符串'A'
        f(a, 'A')
        # 断言数组a与预期结果np.array(list('Abc'), dtype=a.dtype)相等
        assert_array_equal(a, np.array(list('Abc'), dtype=a.dtype))
        
        # 对数组a的子数组进行操作
        f(a[1:], 'B')
        # 断言数组a与预期结果np.array(list('ABc'), dtype=a.dtype)相等
        assert_array_equal(a, np.array(list('ABc'), dtype=a.dtype))

        # 创建一个数组a，内容为['abc']，指定数据类型为dtype
        a = np.array(['abc'], dtype=dtype)
        # 调用函数f并传入a及字符串'A'
        f(a, 'A')
        # 断言数组a与预期结果np.array(['Abc'], dtype=a.dtype)相等
        assert_array_equal(a, np.array(['Abc'], dtype=a.dtype))

    # 定义一个测试方法，测试函数的多种输入输出情况
    def test_inout_varia(self):
        # 获取被测试模块中的相应函数
        f = getattr(self.module, self.fprefix + '_inout')
        
        # 创建一个数组a，内容为'abc'，指定数据类型为'S3'
        a = np.array('abc', dtype='S3')
        # 调用函数f并传入a及字符串'A'
        f(a, 'A')
        # 断言数组a与预期结果np.array('Abc', dtype=a.dtype)相等
        assert_array_equal(a, np.array('Abc', dtype=a.dtype))

        # 创建一个数组a，内容为['abc']，指定数据类型为'S3'
        a = np.array(['abc'], dtype='S3')
        # 调用函数f并传入a及字符串'A'
        f(a, 'A')
        # 断言数组a与预期结果np.array(['Abc'], dtype=a.dtype)相等
        assert_array_equal(a, np.array(['Abc'], dtype=a.dtype))

        # 尝试使用字符串'abc'调用函数f，预期会引发ValueError异常，如果异常消息不以' got 3-str'结尾，则抛出SystemError异常
        try:
            f('abc', 'A')
        except ValueError as msg:
            if not str(msg).endswith(' got 3-str'):
                raise
        else:
            raise SystemError(f'{f.__name__} should have failed on str value')

    # 使用pytest的参数化装饰器，定义一个测试方法，测试函数的数组输入输出功能，参数为dtype
    @pytest.mark.parametrize("dtype", ['c', 'S1'])
    def test_array_inout(self, dtype):
        # 获取被测试模块中的相应函数
        f = getattr(self.module, self.fprefix + '_array_inout')
        
        # 创建一个数组n，内容为['A', 'B', 'C']，指定数据类型为dtype，列序为F（列主序）
        n = np.array(['A', 'B', 'C'], dtype=dtype, order='F')

        # 创建一个数组a，内容为['a', 'b', 'c']，指定数据类型为dtype，列序为F（列主序）
        a = np.array(['a', 'b', 'c'], dtype=dtype, order='F')
        # 调用函数f并传入a及数组n
        f(a, n)
        # 断言数组a与数组n相等
        assert_array_equal(a, n)

        # 创建一个数组a，内容为['a', 'b', 'c', 'd']，指定数据类型为dtype
        a = np.array(['a', 'b', 'c', 'd'], dtype=dtype)
        # 调用函数f并传入a的子数组及数组n
        f(a[1:], n)
        # 断言数组a与预期结果np.array(['a', 'A', 'B', 'C'], dtype=dtype)相等
        assert_array_equal(a, np.array(['a', 'A', 'B', 'C'], dtype=dtype))

        # 创建一个数组a，内容为[['a', 'b', 'c']]，指定数据类型为dtype，列序为F（列主序）
        a = np.array([['a', 'b', 'c']], dtype=dtype, order='F')
        # 调用函数f并传入a及数组n
        f(a, n)
        # 断言数组a与预期结果np.array([['A', 'B', 'C']], dtype=dtype)相等
        assert_array_equal(a, np.array([['A', 'B', 'C']], dtype=dtype))

        # 创建一个数组a，内容为['a', 'b', 'c', 'd']，指定数据类型为dtype，列序为F（列主序）
        a = np.array(['a', 'b', 'c', 'd'], dtype=dtype, order='F')
        # 尝试使用数组a调用函数f，预期会引发ValueError异常，如果异常消息不以'th dimension must be fixed to 3 but got 4'结尾，则抛出SystemError异常
        try:
            f(a, n)
        except ValueError as msg:
            if not str(msg).endswith(
                    'th dimension must be fixed to 3 but got 4'):
                raise
        else:
            raise SystemError(
                f'{f.__name__} should have failed on wrong input')
    # 使用 pytest 的参数化装饰器，为每个测试用例提供不同的数据类型（'c' 和 'S1'）
    @pytest.mark.parametrize("dtype", ['c', 'S1'])
    # 定义一个测试方法，测试二维数组的输入输出
    def test_2d_array_inout(self, dtype):
        # 获取被测试模块中以指定前缀开头的 '_2d_array_inout' 函数
        f = getattr(self.module, self.fprefix + '_2d_array_inout')
        # 创建一个Fortran风格的二维数组n，数据类型为dtype，按列主序（F顺序）
        n = np.array([['A', 'B', 'C'],
                      ['D', 'E', 'F']],
                     dtype=dtype, order='F')
        # 创建另一个Fortran风格的二维数组a，数据类型为dtype，按列主序（F顺序）
        a = np.array([['a', 'b', 'c'],
                      ['d', 'e', 'f']],
                     dtype=dtype, order='F')
        # 调用被测试函数f，传入数组a和n作为参数
        f(a, n)
        # 断言数组a和n相等
        assert_array_equal(a, n)
    
    # 定义一个测试方法，测试返回结果为字节串的函数
    def test_return(self):
        # 获取被测试模块中以指定前缀开头的 '_return' 函数
        f = getattr(self.module, self.fprefix + '_return')
        # 断言调用f函数，传入字符串'a'返回字节串b'a'
        assert_equal(f('a'), b'a')
    
    # 使用 pytest 的跳过装饰器，注明测试方法跳过并提供跳过的原因
    @pytest.mark.skip('fortran function returning array segfaults')
    # 定义一个测试方法，测试返回数组的Fortran函数
    def test_array_return(self):
        # 获取被测试模块中以指定前缀开头的 '_array_return' 函数
        f = getattr(self.module, self.fprefix + '_array_return')
        # 创建一个数据类型为'S1'的字符数组a
        a = np.array(list('abc'), dtype='S1')
        # 断言调用f函数，传入数组a，返回的结果与a数组相等
        assert_array_equal(f(a), a)
    
    # 定义一个测试方法，测试带有可选参数的函数
    def test_optional(self):
        # 获取被测试模块中以指定前缀开头的 '_optional' 函数
        f = getattr(self.module, self.fprefix + '_optional')
        # 断言调用f函数，不传入参数时返回字节串b'a'
        assert_equal(f(), b"a")
        # 断言调用f函数，传入参数b'B'时返回字节串b'B'
        assert_equal(f(b'B'), b"B")
class TestMiscCharacter(util.F2PyTest):
    # 定义一个测试类，继承自util.F2PyTest，用于测试F2Py相关功能
    # options = ['--debug-capi', '--build-dir', '/tmp/test-build-f2py']
    # 设置选项（注释掉的部分，这里是可选的调试选项）
    suffix = '.f90'
    # 文件后缀名为.f90
    fprefix = 'test_misc_character'
    # 函数名前缀为test_misc_character

    code = textwrap.dedent(f"""
       subroutine {fprefix}_gh18684(x, y, m)
         character(len=5), dimension(m), intent(in) :: x
         character*5, dimension(m), intent(out) :: y
         integer i, m
         !f2py integer, intent(hide), depend(x) :: m = f2py_len(x)
         # 这是一个F2Py的特殊注释，指定了一个隐藏的整数参数m，依赖于x的长度
         do i=1,m
           y(i) = x(i)
         end do
       end subroutine {fprefix}_gh18684

       subroutine {fprefix}_gh6308(x, i)
         integer i
         !f2py check(i>=0 && i<12) i
         # 检查i的值是否在0到11之间
         character*5 name, x
         common name(12)
         name(i + 1) = x
         # 将x的值存入name数组的第i+1个位置
       end subroutine {fprefix}_gh6308

       subroutine {fprefix}_gh4519(x)
         character(len=*), intent(in) :: x(:)
         !f2py intent(out) x
         integer :: i
         ! Uncomment for debug printing:
         !do i=1, size(x)
         !   print*, "x(",i,")=", x(i)
         !end do
         # 子程序用于处理输入参数x，但在此没有具体实现，注释掉了用于调试的打印代码
       end subroutine {fprefix}_gh4519

       pure function {fprefix}_gh3425(x) result (y)
         character(len=*), intent(in) :: x
         character(len=len(x)) :: y
         integer :: i
         do i = 1, len(x)
           j = iachar(x(i:i))
           if (j>=iachar("a") .and. j<=iachar("z") ) then
             y(i:i) = achar(j-32)
           else
             y(i:i) = x(i:i)
           endif
         end do
         # 纯函数，将输入字符串x中的小写字母转换为大写，返回结果y
       end function {fprefix}_gh3425

       subroutine {fprefix}_character_bc_new(x, y, z)
         character, intent(in) :: x
         character, intent(out) :: y
         !f2py character, depend(x) :: y = x
         !f2py character, dimension((x=='a'?1:2)), depend(x), intent(out) :: z
         character, dimension(*) :: z
         !f2py character, optional, check(x == 'a' || x == 'b') :: x = 'a'
         !f2py callstatement (*f2py_func)(&x, &y, z)
         !f2py callprotoargument character*, character*, character*
         if (y.eq.x) then
           y = x
         else
           y = 'e'
         endif
         z(1) = 'c'
         # 子程序，根据输入的x，将y设为x或者'e'，z数组的第一个元素设为'c'
       end subroutine {fprefix}_character_bc_new

       subroutine {fprefix}_character_bc_old(x, y, z)
         character, intent(in) :: x
         character, intent(out) :: y
         !f2py character, depend(x) :: y = x[0]
         !f2py character, dimension((*x=='a'?1:2)), depend(x), intent(out) :: z
         character, dimension(*) :: z
         !f2py character, optional, check(*x == 'a' || x[0] == 'b') :: x = 'a'
         !f2py callstatement (*f2py_func)(x, y, z)
         !f2py callprotoargument char*, char*, char*
         if (y.eq.x) then
           y = x
         else
           y = 'e'
         endif
         z(1) = 'c'
         # 子程序，根据输入的x，将y设为x或者'e'，z数组的第一个元素设为'c'
       end subroutine {fprefix}_character_bc_old
    """)

    @pytest.mark.slow
    # 测试函数：test_gh18684
    def test_gh18684(self):
        # 测试字符长度为5的数组和字符长度为5的字符串的用法
        f = getattr(self.module, self.fprefix + '_gh18684')
        x = np.array(["abcde", "fghij"], dtype='S5')
        y = f(x)

        # 断言输出数组x和函数返回的数组y相等
        assert_array_equal(x, y)

    # 测试函数：test_gh6308
    def test_gh6308(self):
        # 测试共同块中的字符字符串数组
        f = getattr(self.module, self.fprefix + '_gh6308')

        # 断言模块中共同块的名称的数据类型为'S5'
        assert_equal(self.module._BLNK_.name.dtype, np.dtype('S5'))
        # 断言模块中共同块的名称长度为12
        assert_equal(len(self.module._BLNK_.name), 12)
        # 调用函数f，将字符串"abcde"放入共同块索引为0的位置
        f("abcde", 0)
        # 断言模块中共同块索引为0的名称为b"abcde"
        assert_equal(self.module._BLNK_.name[0], b"abcde")
        # 调用函数f，将字符串"12345"放入共同块索引为5的位置
        f("12345", 5)
        # 断言模块中共同块索引为5的名称为b"12345"
        assert_equal(self.module._BLNK_.name[5], b"12345")

    # 测试函数：test_gh4519
    def test_gh4519(self):
        # 测试假定长度字符串的数组
        f = getattr(self.module, self.fprefix + '_gh4519')

        # 遍历测试用例，每个测试用例包含输入x和期望输出expected
        for x, expected in [
                ('a', dict(shape=(), dtype=np.dtype('S1'))),
                ('text', dict(shape=(), dtype=np.dtype('S4'))),
                (np.array(['1', '2', '3'], dtype='S1'),
                 dict(shape=(3,), dtype=np.dtype('S1'))),
                (['1', '2', '34'],
                 dict(shape=(3,), dtype=np.dtype('S2'))),
                (['', ''], dict(shape=(2,), dtype=np.dtype('S1')))]:
            # 调用函数f，传入输入x，获取返回值r
            r = f(x)
            # 遍历期望输出字典expected中的键值对
            for k, v in expected.items():
                # 断言返回值r的属性k等于期望值v
                assert_equal(getattr(r, k), v)

    # 测试函数：test_gh3425
    def test_gh3425(self):
        # 测试返回假定长度字符串的副本
        f = getattr(self.module, self.fprefix + '_gh3425')
        # 函数f相当于将字节串转为大写字母

        # 断言函数f对'abC'的处理结果为b'ABC'
        assert_equal(f('abC'), b'ABC')
        # 断言函数f对空字符串的处理结果为b''
        assert_equal(f(''), b'')
        # 断言函数f对'abC12d'的处理结果为b'ABC12D'
        assert_equal(f('abC12d'), b'ABC12D')

    # 使用pytest的参数化装饰器，测试函数：test_character_bc
    @pytest.mark.parametrize("state", ['new', 'old'])
    def test_character_bc(self, state):
        # 获取函数f，函数名包含前缀和状态信息
        f = getattr(self.module, self.fprefix + '_character_bc_' + state)

        # 调用f()，返回c和a
        c, a = f()
        # 断言c为b'a'，a的长度为1
        assert_equal(c, b'a')
        assert_equal(len(a), 1)

        # 再次调用f(b'b')，返回c和a
        c, a = f(b'b')
        # 断言c为b'b'，a的长度为2
        assert_equal(c, b'b')
        assert_equal(len(a), 2)

        # 使用lambda表达式捕获异常，断言调用f(b'c')时抛出异常
        assert_raises(Exception, lambda: f(b'c'))
# 定义一个测试类 TestStringScalarArr，继承自 util.F2PyTest
class TestStringScalarArr(util.F2PyTest):
    # 设置测试类的源文件路径列表，包括 scalar_string.f90
    sources = [util.getpath("tests", "src", "string", "scalar_string.f90")]

    # 定义测试方法 test_char，用于测试字符类型操作
    def test_char(self):
        # 遍历 self.module.string_test.string 和 self.module.string_test.string77
        for out in (self.module.string_test.string,
                    self.module.string_test.string77):
            expected = ()  # 期望的形状为空元组
            assert out.shape == expected  # 断言输出的形状与期望的形状相同
            expected = '|S8'  # 期望的数据类型为字符串，长度为8
            assert out.dtype == expected  # 断言输出的数据类型与期望的数据类型相同

    # 定义测试方法 test_char_arr，用于测试字符数组类型操作
    def test_char_arr(self):
        # 遍历 self.module.string_test.strarr 和 self.module.string_test.strarr77
        for out in (self.module.string_test.strarr,
                    self.module.string_test.strarr77):
            expected = (5, 7)  # 期望的形状为 (5, 7)
            assert out.shape == expected  # 断言输出的形状与期望的形状相同
            expected = '|S12'  # 期望的数据类型为字符串，长度为12
            assert out.dtype == expected  # 断言输出的数据类型与期望的数据类型相同

# 定义一个测试类 TestStringAssumedLength，继承自 util.F2PyTest
class TestStringAssumedLength(util.F2PyTest):
    # 设置测试类的源文件路径列表，包括 gh24008.f
    sources = [util.getpath("tests", "src", "string", "gh24008.f")]

    # 定义测试方法 test_gh24008，用于测试 gh24008 功能
    def test_gh24008(self):
        # 调用 self.module.greet 方法，传入两个参数 "joe" 和 "bob"
        self.module.greet("joe", "bob")

# 使用 pytest.mark.slow 标记的测试类 TestStringOptionalInOut，继承自 util.F2PyTest
@pytest.mark.slow
class TestStringOptionalInOut(util.F2PyTest):
    # 设置测试类的源文件路径列表，包括 gh24662.f90
    sources = [util.getpath("tests", "src", "string", "gh24662.f90")]

    # 定义测试方法 test_gh24662，用于测试 gh24662 功能
    def test_gh24662(self):
        # 调用 self.module.string_inout_optional 方法，不传入任何参数
        self.module.string_inout_optional()
        # 创建一个长度为 32 的字符串数组 a，内容为 'hi'
        a = np.array('hi', dtype='S32')
        # 调用 self.module.string_inout_optional 方法，传入参数 a
        self.module.string_inout_optional(a)
        # 断言字符串 "output string" 在 a 的字节表示中
        assert "output string" in a.tobytes().decode()
        # 使用 pytest.raises 检查是否抛出异常
        with pytest.raises(Exception):
            aa = "Hi"
            self.module.string_inout_optional(aa)

# 使用 pytest.mark.slow 标记的测试类 TestNewCharHandling，继承自 util.F2PyTest
@pytest.mark.slow
class TestNewCharHandling(util.F2PyTest):
    # 设置测试类的源文件路径列表，包括 gh25286.pyf 和 gh25286.f90
    sources = [
        util.getpath("tests", "src", "string", "gh25286.pyf"),
        util.getpath("tests", "src", "string", "gh25286.f90")
    ]
    module_name = "_char_handling_test"  # 模块名为 _char_handling_test

    # 定义测试方法 test_gh25286，用于测试 gh25286 功能
    def test_gh25286(self):
        # 调用 self.module.charint 方法，传入字符 'T'，并将返回值赋给 info
        info = self.module.charint('T')
        # 断言 info 的值为 2
        assert info == 2

# 使用 pytest.mark.slow 标记的测试类 TestBCCharHandling，继承自 util.F2PyTest
@pytest.mark.slow
class TestBCCharHandling(util.F2PyTest):
    # 设置测试类的源文件路径列表，包括 gh25286_bc.pyf 和 gh25286.f90
    sources = [
        util.getpath("tests", "src", "string", "gh25286_bc.pyf"),
        util.getpath("tests", "src", "string", "gh25286.f90")
    ]
    module_name = "_char_handling_test"  # 模块名为 _char_handling_test

    # 定义测试方法 test_gh25286，用于测试 gh25286 功能
    def test_gh25286(self):
        # 调用 self.module.charint 方法，传入字符 'T'，并将返回值赋给 info
        info = self.module.charint('T')
        # 断言 info 的值为 2
        assert info == 2
```