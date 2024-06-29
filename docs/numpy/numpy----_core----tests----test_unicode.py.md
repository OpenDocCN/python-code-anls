# `.\numpy\numpy\_core\tests\test_unicode.py`

```
import pytest  # 导入 pytest 库

import numpy as np  # 导入 NumPy 库，命名为 np
from numpy.testing import assert_, assert_equal, assert_array_equal  # 导入 NumPy 测试相关的断言函数

def buffer_length(arr):
    # 如果输入的 arr 是字符串
    if isinstance(arr, str):
        # 如果 arr 是空字符串，则 charmax 为 0
        if not arr:
            charmax = 0
        else:
            # 否则，charmax 是 arr 中字符的最大 ASCII 值
            charmax = max([ord(c) for c in arr])
        
        # 根据 charmax 的大小确定 size 的值
        if charmax < 256:
            size = 1
        elif charmax < 65536:
            size = 2
        else:
            size = 4
        
        # 返回字符串的长度乘以 size
        return size * len(arr)
    
    # 如果 arr 是内存视图对象
    v = memoryview(arr)
    if v.shape is None:
        # 返回内存视图的长度乘以每个元素的字节大小
        return len(v) * v.itemsize
    else:
        # 返回内存视图各维度大小的乘积乘以每个元素的字节大小
        return np.prod(v.shape) * v.itemsize


# 在以下两个案例中，我们需要确保通过字节交换后的 UCS4 值仍然是有效的 Unicode：
# 可以在 UCS2 解释器中表示的值
ucs2_value = '\u0900'
# 不能在 UCS2 解释器中表示但可以在 UCS4 解释器中表示的值
ucs4_value = '\U00100900'


def test_string_cast():
    # 创建一个字符串数组，包含两个字符串
    str_arr = np.array(["1234", "1234\0\0"], dtype='S')
    # 将字符串数组转换为 Unicode 类型 '>U' 和 '<U'
    uni_arr1 = str_arr.astype('>U')
    uni_arr2 = str_arr.astype('<U')

    # 断言两个数组的不等性
    assert_array_equal(str_arr != uni_arr1, np.ones(2, dtype=bool))
    assert_array_equal(uni_arr1 != str_arr, np.ones(2, dtype=bool))
    # 断言两个数组的相等性
    assert_array_equal(str_arr == uni_arr1, np.zeros(2, dtype=bool))
    assert_array_equal(uni_arr1 == str_arr, np.zeros(2, dtype=bool))

    # 断言两个 Unicode 数组相等
    assert_array_equal(uni_arr1, uni_arr2)


############################################################
#    Creation tests
############################################################

class CreateZeros:
    """检查零值数组的创建"""

    def content_check(self, ua, ua_scalar, nbytes):
        # 检查 Unicode 基础类型的长度
        assert_(int(ua.dtype.str[2:]) == self.ulen)
        # 检查数据缓冲区的长度
        assert_(buffer_length(ua) == nbytes)
        # 检查数组元素的数据是否为空
        assert_(ua_scalar == '')
        # 将数组元素编码为 ASCII 并进行双重检查
        assert_(ua_scalar.encode('ascii') == b'')
        # 检查标量的缓冲区长度
        assert_(buffer_length(ua_scalar) == 0)

    def test_zeros0D(self):
        # 检查创建零维对象
        ua = np.zeros((), dtype='U%s' % self.ulen)
        self.content_check(ua, ua[()], 4*self.ulen)

    def test_zerosSD(self):
        # 检查创建单维对象
        ua = np.zeros((2,), dtype='U%s' % self.ulen)
        self.content_check(ua, ua[0], 4*self.ulen*2)
        self.content_check(ua, ua[1], 4*self.ulen*2)

    def test_zerosMD(self):
        # 检查创建多维对象
        ua = np.zeros((2, 3, 4), dtype='U%s' % self.ulen)
        self.content_check(ua, ua[0, 0, 0], 4*self.ulen*2*3*4)
        self.content_check(ua, ua[-1, -1, -1], 4*self.ulen*2*3*4)


class TestCreateZeros_1(CreateZeros):
    """检查零值数组的创建 (大小为 1)"""
    ulen = 1


class TestCreateZeros_2(CreateZeros):
    """检查零值数组的创建 (大小为 2)"""
    ulen = 2
class TestCreateZeros_1009(CreateZeros):
    """Check the creation of zero-valued arrays (size 1009)"""
    # 定义测试类，用于检查大小为1009的零值数组的创建


class CreateValues:
    """Check the creation of unicode arrays with values"""

    def content_check(self, ua, ua_scalar, nbytes):
        # 检查Unicode数组的创建，并验证内容

        # 检查Unicode基本类型的长度
        assert_(int(ua.dtype.str[2:]) == self.ulen)

        # 检查数据缓冲区的长度
        assert_(buffer_length(ua) == nbytes)

        # 检查数组元素的数据是否正确
        assert_(ua_scalar == self.ucs_value*self.ulen)

        # 将数据编码为UTF-8并再次进行验证
        assert_(ua_scalar.encode('utf-8') ==
                        (self.ucs_value*self.ulen).encode('utf-8'))

        # 检查标量的缓冲区长度
        if self.ucs_value == ucs4_value:
            # 在UCS2中，\U0010FFFF将使用代理对表示
            assert_(buffer_length(ua_scalar) == 2*2*self.ulen)
        else:
            # 在UCS2中，\uFFFF将使用常规2字节表示
            assert_(buffer_length(ua_scalar) == 2*self.ulen)

    def test_values0D(self):
        # 检查创建具有值的零维对象
        ua = np.array(self.ucs_value*self.ulen, dtype='U%s' % self.ulen)
        self.content_check(ua, ua[()], 4*self.ulen)

    def test_valuesSD(self):
        # 检查创建具有值的单维对象
        ua = np.array([self.ucs_value*self.ulen]*2, dtype='U%s' % self.ulen)
        self.content_check(ua, ua[0], 4*self.ulen*2)
        self.content_check(ua, ua[1], 4*self.ulen*2)

    def test_valuesMD(self):
        # 检查创建具有值的多维对象
        ua = np.array([[[self.ucs_value*self.ulen]*2]*3]*4, dtype='U%s' % self.ulen)
        self.content_check(ua, ua[0, 0, 0], 4*self.ulen*2*3*4)
        self.content_check(ua, ua[-1, -1, -1], 4*self.ulen*2*3*4)


class TestCreateValues_1_UCS2(CreateValues):
    """Check the creation of valued arrays (size 1, UCS2 values)"""
    # 检查大小为1的UCS2值数组的创建


class TestCreateValues_1_UCS4(CreateValues):
    """Check the creation of valued arrays (size 1, UCS4 values)"""
    # 检查大小为1的UCS4值数组的创建


class TestCreateValues_2_UCS2(CreateValues):
    """Check the creation of valued arrays (size 2, UCS2 values)"""
    # 检查大小为2的UCS2值数组的创建


class TestCreateValues_2_UCS4(CreateValues):
    """Check the creation of valued arrays (size 2, UCS4 values)"""
    # 检查大小为2的UCS4值数组的创建


class TestCreateValues_1009_UCS2(CreateValues):
    """Check the creation of valued arrays (size 1009, UCS2 values)"""
    # 检查大小为1009的UCS2值数组的创建


class TestCreateValues_1009_UCS4(CreateValues):
    """Check the creation of valued arrays (size 1009, UCS4 values)"""
    # 检查大小为1009的UCS4值数组的创建


############################################################
#    Assignment tests
############################################################

class AssignValues:
    """定义一个名为AssignValues的类，用于检查Unicode数组的赋值"""

    def content_check(self, ua, ua_scalar, nbytes):
        # 检查Unicode基本类型的长度
        assert_(int(ua.dtype.str[2:]) == self.ulen)
        # 检查数据缓冲区的长度
        assert_(buffer_length(ua) == nbytes)
        # 对数组元素的数据进行小规模检查
        assert_(ua_scalar == self.ucs_value*self.ulen)
        # 编码为UTF-8并进行双重检查
        assert_(ua_scalar.encode('utf-8') ==
                        (self.ucs_value*self.ulen).encode('utf-8'))
        # 检查标量的缓冲区长度
        if self.ucs_value == ucs4_value:
            # 在UCS2中，\U0010FFFF将使用代理对进行表示
            assert_(buffer_length(ua_scalar) == 2*2*self.ulen)
        else:
            # 在UCS2中，\uFFFF将使用普通的2字节表示
            assert_(buffer_length(ua_scalar) == 2*self.ulen)

    def test_values0D(self):
        # 检查赋值为0维对象的情况
        ua = np.zeros((), dtype='U%s' % self.ulen)
        ua[()] = self.ucs_value*self.ulen
        self.content_check(ua, ua[()], 4*self.ulen)

    def test_valuesSD(self):
        # 检查赋值为单维对象的情况
        ua = np.zeros((2,), dtype='U%s' % self.ulen)
        ua[0] = self.ucs_value*self.ulen
        self.content_check(ua, ua[0], 4*self.ulen*2)
        ua[1] = self.ucs_value*self.ulen
        self.content_check(ua, ua[1], 4*self.ulen*2)

    def test_valuesMD(self):
        # 检查赋值为多维对象的情况
        ua = np.zeros((2, 3, 4), dtype='U%s' % self.ulen)
        ua[0, 0, 0] = self.ucs_value*self.ulen
        self.content_check(ua, ua[0, 0, 0], 4*self.ulen*2*3*4)
        ua[-1, -1, -1] = self.ucs_value*self.ulen
        self.content_check(ua, ua[-1, -1, -1], 4*self.ulen*2*3*4)


class TestAssignValues_1_UCS2(AssignValues):
    """检查赋值为值数组（大小为1，UCS2值）的情况"""
    ulen = 1
    ucs_value = ucs2_value


class TestAssignValues_1_UCS4(AssignValues):
    """检查赋值为值数组（大小为1，UCS4值）的情况"""
    ulen = 1
    ucs_value = ucs4_value


class TestAssignValues_2_UCS2(AssignValues):
    """检查赋值为值数组（大小为2，UCS2值）的情况"""
    ulen = 2
    ucs_value = ucs2_value


class TestAssignValues_2_UCS4(AssignValues):
    """检查赋值为值数组（大小为2，UCS4值）的情况"""
    ulen = 2
    ucs_value = ucs4_value


class TestAssignValues_1009_UCS2(AssignValues):
    """检查赋值为值数组（大小为1009，UCS2值）的情况"""
    ulen = 1009
    ucs_value = ucs2_value


class TestAssignValues_1009_UCS4(AssignValues):
    """检查赋值为值数组（大小为1009，UCS4值）的情况"""
    ulen = 1009
    ucs_value = ucs4_value
############################################################
#    Byteorder tests
############################################################

class ByteorderValues:
    """Check the byteorder of unicode arrays in round-trip conversions"""

    def test_values0D(self):
        # Check byteorder of 0-dimensional objects
        # 创建一个 Unicode 数组，长度为 self.ulen，内容为 self.ucs_value 的重复
        ua = np.array(self.ucs_value*self.ulen, dtype='U%s' % self.ulen)
        # 创建一个视图，使用新的字节顺序来查看数组 ua
        ua2 = ua.view(ua.dtype.newbyteorder())
        # 由于视图的字节顺序改变了数据区域的解释，但实际数据不变，因此返回的标量不同（它们是彼此的字节交换版本）。
        assert_(ua[()] != ua2[()])
        # 创建另一个视图，使用原始数组 ua2 的新字节顺序来查看
        ua3 = ua2.view(ua2.dtype.newbyteorder())
        # 得到的数组 ua 和 ua3 在往返转换后应相等
        assert_equal(ua, ua3)

    def test_valuesSD(self):
        # Check byteorder of single-dimensional objects
        # 创建一个单维数组，包含两个元素，每个元素是 self.ucs_value 重复 self.ulen 次
        ua = np.array([self.ucs_value*self.ulen]*2, dtype='U%s' % self.ulen)
        # 创建一个视图，使用新的字节顺序来查看数组 ua
        ua2 = ua.view(ua.dtype.newbyteorder())
        # 检查所有元素是否都不相等
        assert_((ua != ua2).all())
        # 检查最后一个元素是否不相等
        assert_(ua[-1] != ua2[-1])
        # 创建另一个视图，使用原始数组 ua2 的新字节顺序来查看
        ua3 = ua2.view(ua2.dtype.newbyteorder())
        # 得到的数组 ua 和 ua3 在往返转换后应相等
        assert_equal(ua, ua3)

    def test_valuesMD(self):
        # Check byteorder of multi-dimensional objects
        # 创建一个三维数组，数组元素为 self.ucs_value 重复 self.ulen 次的数组，形状为 (4, 3, 2)
        ua = np.array([[[self.ucs_value*self.ulen]*2]*3]*4,
                      dtype='U%s' % self.ulen)
        # 创建一个视图，使用新的字节顺序来查看数组 ua
        ua2 = ua.view(ua.dtype.newbyteorder())
        # 检查所有元素是否都不相等
        assert_((ua != ua2).all())
        # 检查最后一个元素是否不相等
        assert_(ua[-1, -1, -1] != ua2[-1, -1, -1])
        # 创建另一个视图，使用原始数组 ua2 的新字节顺序来查看
        ua3 = ua2.view(ua2.dtype.newbyteorder())
        # 得到的数组 ua 和 ua3 在往返转换后应相等
        assert_equal(ua, ua3)

    def test_values_cast(self):
        # Check byteorder of when casting the array for a strided and
        # contiguous array:
        # 创建一个单维数组，包含两个元素，每个元素是 self.ucs_value 重复 self.ulen 次
        test1 = np.array([self.ucs_value*self.ulen]*2, dtype='U%s' % self.ulen)
        # 通过重复和切片来创建一个分步和连续数组的测试数组
        test2 = np.repeat(test1, 2)[::2]
        for ua in (test1, test2):
            # 将数组 ua 转换为指定的新字节顺序
            ua2 = ua.astype(dtype=ua.dtype.newbyteorder())
            # 检查所有元素是否都相等
            assert_((ua == ua2).all())
            # 检查最后一个元素是否相等
            assert_(ua[-1] == ua2[-1])
            # 创建另一个视图，使用原始数组 ua2 的原始字节顺序来查看
            ua3 = ua2.astype(dtype=ua.dtype)
            # 得到的数组 ua 和 ua3 在往返转换后应相等
            assert_equal(ua, ua3)
    def test_values_updowncast(self):
        # 定义测试函数，用于检查数组在类型转换（向上和向下转换）时的字节顺序，
        # 以及在步进和连续数组中的字符串长度

        # 创建长度为 self.ulen 的 self.ucs_value 重复两次的数组，并指定为 Unicode 字符串类型
        test1 = np.array([self.ucs_value*self.ulen]*2, dtype='U%s' % self.ulen)
        
        # 将 test1 重复两次并每隔一个取值，形成新数组 test2
        test2 = np.repeat(test1, 2)[::2]
        
        # 遍历 test1 和 test2 中的数组
        for ua in (test1, test2):
            # 定义一个比当前类型长一位的新类型，并设定字节顺序与当前相同
            longer_type = np.dtype('U%s' % (self.ulen+1)).newbyteorder()
            
            # 将当前数组 ua 转换为新定义的更长类型，使用零填充
            ua2 = ua.astype(dtype=longer_type)
            
            # 断言：转换后的数组与原数组内容完全一致
            assert_((ua == ua2).all())
            
            # 断言：新数组的最后一个元素与原数组的最后一个元素相同
            assert_(ua[-1] == ua2[-1])
            
            # 将 ua2 转换回原来的数据类型 ua
            ua3 = ua2.astype(dtype=ua.dtype)
            
            # 断言：经过上述转换的数组 ua3 与原始数组 ua 相等
            assert_equal(ua, ua3)
# 继承自 ByteorderValues 类，用于测试大小为 1 的 UCS2 编码的字节顺序
class TestByteorder_1_UCS2(ByteorderValues):
    """Check the byteorder in unicode (size 1, UCS2 values)"""
    # 设置测试的 Unicode 长度为 1
    ulen = 1
    # 使用 ucs2_value 进行测试
    ucs_value = ucs2_value


# 继承自 ByteorderValues 类，用于测试大小为 1 的 UCS4 编码的字节顺序
class TestByteorder_1_UCS4(ByteorderValues):
    """Check the byteorder in unicode (size 1, UCS4 values)"""
    # 设置测试的 Unicode 长度为 1
    ulen = 1
    # 使用 ucs4_value 进行测试
    ucs_value = ucs4_value


# 继承自 ByteorderValues 类，用于测试大小为 2 的 UCS2 编码的字节顺序
class TestByteorder_2_UCS2(ByteorderValues):
    """Check the byteorder in unicode (size 2, UCS2 values)"""
    # 设置测试的 Unicode 长度为 2
    ulen = 2
    # 使用 ucs2_value 进行测试
    ucs_value = ucs2_value


# 继承自 ByteorderValues 类，用于测试大小为 2 的 UCS4 编码的字节顺序
class TestByteorder_2_UCS4(ByteorderValues):
    """Check the byteorder in unicode (size 2, UCS4 values)"""
    # 设置测试的 Unicode 长度为 2
    ulen = 2
    # 使用 ucs4_value 进行测试
    ucs_value = ucs4_value


# 继承自 ByteorderValues 类，用于测试大小为 1009 的 UCS2 编码的字节顺序
class TestByteorder_1009_UCS2(ByteorderValues):
    """Check the byteorder in unicode (size 1009, UCS2 values)"""
    # 设置测试的 Unicode 长度为 1009
    ulen = 1009
    # 使用 ucs2_value 进行测试
    ucs_value = ucs2_value


# 继承自 ByteorderValues 类，用于测试大小为 1009 的 UCS4 编码的字节顺序
class TestByteorder_1009_UCS4(ByteorderValues):
    """Check the byteorder in unicode (size 1009, UCS4 values)"""
    # 设置测试的 Unicode 长度为 1009
    ulen = 1009
    # 使用 ucs4_value 进行测试
    ucs_value = ucs4_value
```