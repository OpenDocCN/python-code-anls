# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\scalars.pyi`

```py
import sys  # 导入sys模块，用于系统相关的操作
import numpy as np  # 导入numpy库，并使用np作为别名

f2: np.float16  # 声明f2为numpy的float16类型
f8: np.float64  # 声明f8为numpy的float64类型
c8: np.complex64  # 声明c8为numpy的complex64类型

# Construction

np.float32(3j)  # E: incompatible type  # 创建一个复数常量3j，但无法转换为np.float32类型

# Technically the following examples are valid NumPy code. But they
# are not considered a best practice, and people who wish to use the
# stubs should instead do
#
# np.array([1.0, 0.0, 0.0], dtype=np.float32)
# np.array([], dtype=np.complex64)
#
# See e.g. the discussion on the mailing list
#
# https://mail.python.org/pipermail/numpy-discussion/2020-April/080566.html
#
# and the issue
#
# https://github.com/numpy/numpy-stubs/issues/41
#
# for more context.
np.float32([1.0, 0.0, 0.0])  # E: incompatible type  # 创建一个包含浮点数的numpy数组，但无法转换为np.float32类型
np.complex64([])  # E: incompatible type  # 创建一个空的numpy复数数组，但无法转换为np.complex64类型

np.complex64(1, 2)  # E: Too many arguments  # 创建一个复数，但传递了太多的参数
# TODO: protocols (can't check for non-existent protocols w/ __getattr__)

np.datetime64(0)  # E: No overload variant  # 尝试创建一个datetime64类型的对象，但没有匹配的重载变体

class A:
    def __float__(self):
        return 1.0

np.int8(A())  # E: incompatible type  # 将A类实例转换为np.int8类型，但类型不兼容
np.int16(A())  # E: incompatible type  # 将A类实例转换为np.int16类型，但类型不兼容
np.int32(A())  # E: incompatible type  # 将A类实例转换为np.int32类型，但类型不兼容
np.int64(A())  # E: incompatible type  # 将A类实例转换为np.int64类型，但类型不兼容
np.uint8(A())  # E: incompatible type  # 将A类实例转换为np.uint8类型，但类型不兼容
np.uint16(A())  # E: incompatible type  # 将A类实例转换为np.uint16类型，但类型不兼容
np.uint32(A())  # E: incompatible type  # 将A类实例转换为np.uint32类型，但类型不兼容
np.uint64(A())  # E: incompatible type  # 将A类实例转换为np.uint64类型，但类型不兼容

np.void("test")  # E: No overload variant  # 创建一个void类型对象，但没有匹配的重载变体
np.void("test", dtype=None)  # E: No overload variant  # 创建一个void类型对象，但没有匹配的重载变体

np.generic(1)  # E: Cannot instantiate abstract class  # 尝试实例化抽象类generic，但无法实现
np.number(1)  # E: Cannot instantiate abstract class  # 尝试实例化抽象类number，但无法实现
np.integer(1)  # E: Cannot instantiate abstract class  # 尝试实例化抽象类integer，但无法实现
np.inexact(1)  # E: Cannot instantiate abstract class  # 尝试实例化抽象类inexact，但无法实现
np.character("test")  # E: Cannot instantiate abstract class  # 尝试实例化抽象类character，但无法实现
np.flexible(b"test")  # E: Cannot instantiate abstract class  # 尝试实例化抽象类flexible，但无法实现

np.float64(value=0.0)  # E: Unexpected keyword argument  # 创建一个float64类型对象，但提供了意外的关键字参数
np.int64(value=0)  # E: Unexpected keyword argument  # 创建一个int64类型对象，但提供了意外的关键字参数
np.uint64(value=0)  # E: Unexpected keyword argument  # 创建一个uint64类型对象，但提供了意外的关键字参数
np.complex128(value=0.0j)  # E: Unexpected keyword argument  # 创建一个complex128类型对象，但提供了意外的关键字参数
np.str_(value='bob')  # E: No overload variant  # 创建一个str_类型对象，但没有匹配的重载变体
np.bytes_(value=b'test')  # E: No overload variant  # 创建一个bytes_类型对象，但没有匹配的重载变体
np.void(value=b'test')  # E: No overload variant  # 创建一个void类型对象，但没有匹配的重载变体
np.bool(value=True)  # E: Unexpected keyword argument  # 创建一个bool类型对象，但提供了意外的关键字参数
np.datetime64(value="2019")  # E: No overload variant  # 创建一个datetime64类型对象，但没有匹配的重载变体
np.timedelta64(value=0)  # E: Unexpected keyword argument  # 创建一个timedelta64类型对象，但提供了意外的关键字参数

np.bytes_(b"hello", encoding='utf-8')  # E: No overload variant  # 创建一个指定编码的bytes_类型对象，但没有匹配的重载变体
np.str_("hello", encoding='utf-8')  # E: No overload variant  # 创建一个指定编码的str_类型对象，但没有匹配的重载变体

f8.item(1)  # E: incompatible type  # 获取f8对象的第1个元素，但类型不兼容
f8.item((0, 1))  # E: incompatible type  # 获取f8对象指定位置的元素，但类型不兼容
f8.squeeze(axis=1)  # E: incompatible type  # 对f8对象进行压缩操作，但轴的类型不兼容
f8.squeeze(axis=(0, 1))  # E: incompatible type  # 对f8对象进行压缩操作，但轴的类型不兼容
f8.transpose(1)  # E: incompatible type  # 对f8对象进行转置操作，但参数的类型不兼容

def func(a: np.float32) -> None: ...  # 定义一个函数func，接受一个np.float32类型参数，并返回None

func(f2)  # E: incompatible type  # 调用func函数，传递f2作为参数，但类型不兼容
func(f8)  # E: incompatible type  # 调用func函数，传递f8作为参数，但类型不兼容

round(c8)  # E: No overload variant  # 对c8对象进行四舍五入操作，但没有匹配的重载变体

c8.__getnewargs__()  # E: Invalid self argument  # 调用c8对象的__getnewargs__方法，但self参数无效
f2.__getnewargs__()  # E: Invalid self argument  # 调用f2对象的__getnewargs__方法，但self参数无效
f2.hex()  # E: Invalid self argument  # 调用f2对象的hex方法，但self参数无效
np.float16.fromhex("0x0.0p+0")  # E: Invalid self argument  # 调用np.float16的fromhex静态方法，但self参数无效
f2.__trunc__()  # E: Invalid self argument  # 调用f2对象的__trunc__方法，但self参数无效
f2.__getformat__("float")  # E: Invalid self argument  # 调用f2对象的__getformat__方法，但self参数无效
```