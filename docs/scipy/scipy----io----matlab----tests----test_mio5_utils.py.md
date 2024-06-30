# `D:\src\scipysrc\scipy\scipy\io\matlab\tests\test_mio5_utils.py`

```
""" Testing mio5_utils Cython module

"""
# 导入系统模块
import sys

# 导入字节流处理模块
from io import BytesIO

# 导入 NumPy 库并使用简称 np
import numpy as np

# 导入 NumPy 测试模块中的数组相等断言和相等断言
from numpy.testing import assert_array_equal, assert_equal, assert_

# 导入 pytest 中的 raises 断言
from pytest import raises as assert_raises

# 导入 SciPy 中处理 MATLAB 文件的模块
import scipy.io.matlab._byteordercodes as boc
import scipy.io.matlab._streams as streams
import scipy.io.matlab._mio5_params as mio5p
import scipy.io.matlab._mio5_utils as m5u


# 测试函数：检验字节交换功能
def test_byteswap():
    for val in (
        1,
        0x100,
        0x10000):
        a = np.array(val, dtype=np.uint32)
        b = a.byteswap()  # 字节交换
        c = m5u.byteswap_u4(a)  # 使用 m5u 模块中的字节交换函数
        assert_equal(b.item(), c)  # 断言 b 和 c 相等
        d = m5u.byteswap_u4(c)  # 再次交换字节
        assert_equal(a.item(), d)  # 断言 a 和 d 相等


# 辅助函数：生成一个简单的 MATLAB 标签，完整或者 SDE
def _make_tag(base_dt, val, mdtype, sde=False):
    ''' Makes a simple matlab tag, full or sde '''
    base_dt = np.dtype(base_dt)
    bo = boc.to_numpy_code(base_dt.byteorder)  # 转换字节顺序为 NumPy 代码
    byte_count = base_dt.itemsize  # 获取数据类型的字节大小
    if not sde:
        udt = bo + 'u4'  # 无 SDE，数据类型为无符号整数
        padding = 8 - (byte_count % 8)  # 计算填充字节
        all_dt = [('mdtype', udt),
                  ('byte_count', udt),
                  ('val', base_dt)]
        if padding:
            all_dt.append(('padding', 'u1', padding))  # 添加填充字节
    else:  # 如果有 SDE
        udt = bo + 'u2'  # SDE 数据类型为无符号短整数
        padding = 4 - byte_count
        if bo == '<':  # 如果是小端字节顺序
            all_dt = [('mdtype', udt),
                      ('byte_count', udt),
                      ('val', base_dt)]
        else:  # 如果是大端字节顺序
            all_dt = [('byte_count', udt),
                      ('mdtype', udt),
                      ('val', base_dt)]
        if padding:
            all_dt.append(('padding', 'u1', padding))  # 添加填充字节
    tag = np.zeros((1,), dtype=all_dt)  # 创建一个空的 NumPy 数组作为标签
    tag['mdtype'] = mdtype  # 设置标签的 MATLAB 数据类型
    tag['byte_count'] = byte_count  # 设置标签的字节大小
    tag['val'] = val  # 设置标签的值
    return tag


# 辅助函数：向流中写入字符串
def _write_stream(stream, *strings):
    stream.truncate(0)  # 清空流中内容
    stream.seek(0)  # 将流指针移动到开头
    for s in strings:
        stream.write(s)  # 写入每个字符串到流中
    stream.seek(0)  # 将流指针移动到开头


# 辅助函数：创建类似读取器的对象
def _make_readerlike(stream, byte_order=boc.native_code):
    class R:
        pass
    r = R()
    r.mat_stream = stream  # 设置 MATLAB 流对象
    r.byte_order = byte_order  # 设置字节顺序
    r.struct_as_record = True  # 结构体作为记录
    r.uint16_codec = sys.getdefaultencoding()  # 获取默认的 uint16 编码
    r.chars_as_strings = False  # 字符作为字符串
    r.mat_dtype = False  # MATLAB 数据类型
    r.squeeze_me = False  # 压缩
    return r


# 测试函数：读取标签
def test_read_tag():
    # 主要测试错误情况
    # 创建类似读取器的对象
    str_io = BytesIO()
    r = _make_readerlike(str_io)
    c_reader = m5u.VarReader5(r)  # 创建变量读取器对象
    # 对于 StringIO 正常工作但不适用于 BytesIO
    assert_raises(OSError, c_reader.read_tag)  # 断言引发 OSError 异常
    # 错误的 SDE
    tag = _make_tag('i4', 1, mio5p.miINT32, sde=True)  # 创建一个带有错误 SDE 的标签
    tag['byte_count'] = 5  # 设置错误的字节大小
    _write_stream(str_io, tag.tobytes())  # 将标签写入流中
    assert_raises(ValueError, c_reader.read_tag)  # 断言引发 ValueError 异常


# 测试函数：读取流
def test_read_stream():
    tag = _make_tag('i4', 1, mio5p.miINT32, sde=True)  # 创建一个带有 SDE 的标签
    tag_str = tag.tobytes()
    str_io = BytesIO(tag_str)  # 将标签数据转换为字节流
    st = streams.make_stream(str_io)  # 创建流对象
    s = streams._read_into(st, tag.itemsize)  # 从流中读取指定长度的数据
    assert_equal(s, tag.tobytes())  # 断言读取的数据与标签数据相等


# 测试函数：读取数值
def test_read_numeric():
    # 创建类似读取器的对象
    str_io = BytesIO()
    # 使用函数 `_make_readerlike` 创建类似文件读取器的对象 `r`
    r = _make_readerlike(str_io)
    # 检查最简单的标签情况
    for base_dt, val, mdtype in (('u2', 30, mio5p.miUINT16),
                                 ('i4', 1, mio5p.miINT32),
                                 ('i2', -1, mio5p.miINT16)):
        # 循环处理不同的字节顺序
        for byte_code in ('<', '>'):
            # 设置当前字节顺序到读取器 `r`
            r.byte_order = byte_code
            # 创建一个 VarReader5 实例 `c_reader`
            c_reader = m5u.VarReader5(r)
            # 断言当前 VarReader5 实例的小端字节顺序属性是否与当前字节顺序一致
            assert_equal(c_reader.little_endian, byte_code == '<')
            # 断言当前 VarReader5 实例的是否被交换属性是否与本地代码的字节顺序一致
            assert_equal(c_reader.is_swapped, byte_code != boc.native_code)
            # 对于有符号扩展（sde_f）的两种情况进行循环
            for sde_f in (False, True):
                # 创建一个新的 numpy 数据类型 `dt`，根据当前字节顺序设置字节序
                dt = np.dtype(base_dt).newbyteorder(byte_code)
                # 创建一个标签 `_make_tag`，使用给定的数据类型 `dt`、值 `val`、元数据类型 `mdtype` 和有符号扩展标志 `sde_f`
                a = _make_tag(dt, val, mdtype, sde_f)
                # 将标签转换为字节串 `a_str`
                a_str = a.tobytes()
                # 将字节串写入到流 `str_io` 中
                _write_stream(str_io, a_str)
                # 从 `c_reader` 中读取一个数值
                el = c_reader.read_numeric()
                # 断言读取的数值与预期的值 `val` 相等
                assert_equal(el, val)
                # 连续进行两次读取
                _write_stream(str_io, a_str, a_str)
                el = c_reader.read_numeric()
                # 断言再次读取的数值与预期的值 `val` 相等
                assert_equal(el, val)
                el = c_reader.read_numeric()
                # 断言再次读取的数值与预期的值 `val` 相等
                assert_equal(el, val)
# 定义测试函数，测试读取可写的数值
def test_read_numeric_writeable():
    # 创建类似于读取器的字节流对象
    str_io = BytesIO()
    # 使用指定的字节序创建读取器
    r = _make_readerlike(str_io, '<')
    # 使用指定的读取器创建 VarReader5 对象
    c_reader = m5u.VarReader5(r)
    # 定义数据类型为小端无符号 16 位整数
    dt = np.dtype('<u2')
    # 创建一个符合特定标签的数据，长度为 30 字节，类型为 miUINT16，偏移为 0
    a = _make_tag(dt, 30, mio5p.miUINT16, 0)
    # 将标签数据转换为字节流并写入字节流对象
    a_str = a.tobytes()
    _write_stream(str_io, a_str)
    # 读取数值型数据元素
    el = c_reader.read_numeric()
    # 断言数据元素的可写属性为 True
    assert_(el.flags.writeable is True)


# 定义测试函数，测试零字节字符串处理
def test_zero_byte_string():
    # 测试处理非零长度但零字节字符的技巧
    # 创建类似于读取器的字节流对象
    str_io = BytesIO()
    # 使用本地代码创建读取器
    r = _make_readerlike(str_io, boc.native_code)
    # 使用指定的读取器创建 VarReader5 对象
    c_reader = m5u.VarReader5(r)
    # 定义标签数据类型，包含 mdtype 和 byte_count 两个字段，均为无符号 32 位整数
    tag_dt = np.dtype([('mdtype', 'u4'), ('byte_count', 'u4')])
    # 创建一个元素为 1 的全零数组，数据类型为 tag_dt
    tag = np.zeros((1,), dtype=tag_dt)
    # 设置 mdtype 字段值为 miINT8，byte_count 字段值为 1
    tag['mdtype'] = mio5p.miINT8
    tag['byte_count'] = 1
    # 创建 VarHeader5 对象
    hdr = m5u.VarHeader5()
    # 设置维度为 [1,]
    hdr.set_dims([1,])
    # 将标签数据转换为字节流并写入字节流对象，后跟空格字符的字节流
    _write_stream(str_io, tag.tobytes() + b'        ')
    # 将字节流对象指针置于起始位置
    str_io.seek(0)
    # 使用 VarHeader5 对象读取字符数据
    val = c_reader.read_char(hdr)
    # 断言读取的字符数据为一个空格字符
    assert_equal(val, ' ')
    
    # 现在测试零字节的字符串情况
    # 将 byte_count 字段值设置为 0
    tag['byte_count'] = 0
    # 将标签数据转换为字节流并写入字节流对象
    _write_stream(str_io, tag.tobytes())
    # 将字节流对象指针置于起始位置
    str_io.seek(0)
    # 使用 VarHeader5 对象读取字符数据
    val = c_reader.read_char(hdr)
    # 断言读取的字符数据为一个空格字符
    assert_equal(val, ' ')
    
    # 现在测试零字节但长度为 4 的字符串情况
    # 设置维度为 [4,]
    hdr.set_dims([4,])
    # 使用 VarHeader5 对象读取字符数据
    val = c_reader.read_char(hdr)
    # 断言读取的字符数据为包含四个空格字符的数组
    assert_array_equal(val, [' '] * 4)
```