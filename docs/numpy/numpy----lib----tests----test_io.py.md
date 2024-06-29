# `.\numpy\numpy\lib\tests\test_io.py`

```
# 导入系统相关模块
import sys
# 垃圾回收模块，用于管理内存中无用对象的回收
import gc
# 处理 gzip 格式的模块
import gzip
# 系统操作相关的模块
import os
# 多线程编程的模块
import threading
# 时间相关的模块
import time
# 提示和警告相关的模块
import warnings
# 正则表达式模块
import re
# 测试框架 pytest
import pytest
# 处理路径相关的模块
from pathlib import Path
# 创建临时文件的模块
from tempfile import NamedTemporaryFile
# 处理字节流和字符串流的模块
from io import BytesIO, StringIO
# 处理日期和时间的模块
from datetime import datetime
# 处理本地化相关的模块
import locale
# 多进程编程相关的模块
from multiprocessing import Value, get_context
# 用于处理布尔类型的模块
from ctypes import c_bool

# 导入 numpy 数学库及其相关模块
import numpy as np
import numpy.ma as ma
# 处理 numpy 弃用警告的模块
from numpy.exceptions import VisibleDeprecationWarning
# numpy 输入输出工具相关模块
from numpy.lib._iotools import ConverterError, ConversionWarning
# numpy 输入输出工具相关实现
from numpy.lib import _npyio_impl
from numpy.lib._npyio_impl import recfromcsv, recfromtxt
# numpy 测试相关工具
from numpy.ma.testutils import assert_equal
from numpy.testing import (
    assert_warns, assert_, assert_raises_regex, assert_raises,
    assert_allclose, assert_array_equal, temppath, tempdir, IS_PYPY,
    HAS_REFCOUNT, suppress_warnings, assert_no_gc_cycles, assert_no_warnings,
    break_cycles, IS_WASM
    )
# numpy 测试框架的辅助工具
from numpy.testing._private.utils import requires_memory
# numpy 内部工具，处理字节相关操作
from numpy._utils import asbytes
    # 定义一个方法，用于将数组通过指定的保存函数进行往复转换
    def roundtrip(self, save_func, *args, **kwargs):
        """
        save_func : callable
            用于将数组保存到文件的函数。
        file_on_disk : bool
            如果为 True，则将文件保存在磁盘上，而不是在字符串缓冲区中。
        save_kwds : dict
            传递给 `save_func` 的参数。
        load_kwds : dict
            传递给 `numpy.load` 的参数。
        args : tuple of arrays
            要保存到文件的数组。
        """
        
        # 从 kwargs 中获取保存和加载参数的字典
        save_kwds = kwargs.get('save_kwds', {})
        load_kwds = kwargs.get('load_kwds', {"allow_pickle": True})
        file_on_disk = kwargs.get('file_on_disk', False)
        
        # 根据 file_on_disk 的值选择性地创建目标文件对象
        if file_on_disk:
            target_file = NamedTemporaryFile(delete=False)
            load_file = target_file.name
        else:
            target_file = BytesIO()
            load_file = target_file
        
        try:
            # 将参数中的数组存储到目标文件中
            arr = args
            save_func(target_file, *arr, **save_kwds)
            target_file.flush()
            target_file.seek(0)
            
            # 如果运行在 Windows 平台且 target_file 不是 BytesIO 对象，则关闭它
            if sys.platform == 'win32' and not isinstance(target_file, BytesIO):
                target_file.close()
            
            # 重新加载文件中的数组数据
            arr_reloaded = np.load(load_file, **load_kwds)
            
            # 将原始数组和重新加载的数组存储到对象的属性中
            self.arr = arr
            self.arr_reloaded = arr_reloaded
        finally:
            # 如果 target_file 不是 BytesIO 对象，则关闭它
            if not isinstance(target_file, BytesIO):
                target_file.close()
                # 在 Windows 平台上，由于仍然有一个打开的文件描述符，因此不能删除文件
                if 'arr_reloaded' in locals():
                    if not isinstance(arr_reloaded, np.lib.npyio.NpzFile):
                        os.remove(target_file.name)

    # 检查往复转换的多种情况，包括数组本身及其转置版本
    def check_roundtrips(self, a):
        self.roundtrip(a)  # 普通数组
        self.roundtrip(a, file_on_disk=True)  # 将数组保存到磁盘
        self.roundtrip(np.asfortranarray(a))  # 将数组转换为列优先顺序
        self.roundtrip(np.asfortranarray(a), file_on_disk=True)  # 将数组转换为列优先顺序，并保存到磁盘
        if a.shape[0] > 1:
            # 对于 2D 或更高维度的数组，测试非 C 或 Fortran 连续性
            self.roundtrip(np.asfortranarray(a)[1:])  # 切片后的列优先顺序数组
            self.roundtrip(np.asfortranarray(a)[1:], file_on_disk=True)  # 切片后的列优先顺序数组保存到磁盘

    # 测试空数组及不同数据类型和复数类型的数组
    def test_array(self):
        a = np.array([], float)
        self.check_roundtrips(a)  # 测试空数组

        a = np.array([[1, 2], [3, 4]], float)
        self.check_roundtrips(a)  # 测试浮点型数组

        a = np.array([[1, 2], [3, 4]], int)
        self.check_roundtrips(a)  # 测试整型数组

        a = np.array([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]], dtype=np.csingle)
        self.check_roundtrips(a)  # 测试单精度复数数组

        a = np.array([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]], dtype=np.cdouble)
        self.check_roundtrips(a)  # 测试双精度复数数组

    # 测试对象数组及不同维度的一维数组
    def test_array_object(self):
        a = np.array([], object)
        self.check_roundtrips(a)  # 测试空对象数组

        a = np.array([[1, 2], [3, 4]], object)
        self.check_roundtrips(a)  # 测试对象数组

    # 测试一维整型数组
    def test_1D(self):
        a = np.array([1, 2, 3, 4], int)
        self.roundtrip(a)  # 测试一维整型数组的往复转换
    # 定义一个测试方法，用于测试 mmap 模式下的数组写入和读取
    def test_mmap(self):
        # 创建一个 NumPy 数组
        a = np.array([[1, 2.5], [4, 7.3]])
        # 调用 roundtrip 方法，将数组写入文件并加载，使用 mmap 模式只读
        self.roundtrip(a, file_on_disk=True, load_kwds={'mmap_mode': 'r'})

        # 创建一个 Fortran 风格的 NumPy 数组
        a = np.asfortranarray([[1, 2.5], [4, 7.3]])
        # 再次调用 roundtrip 方法，将 Fortran 风格的数组写入文件并加载，使用 mmap 模式只读
        self.roundtrip(a, file_on_disk=True, load_kwds={'mmap_mode': 'r'})

    # 定义一个测试方法，用于测试包含结构化数据的数组写入和读取
    def test_record(self):
        # 创建一个包含结构化数据的 NumPy 数组
        a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        # 调用 check_roundtrips 方法，验证数组的写入和读取
        self.check_roundtrips(a)

    # 使用 pytest.mark.slow 标记的测试方法，测试特定格式的数组写入和读取
    @pytest.mark.slow
    def test_format_2_0(self):
        # 创建一个大的结构化数据类型列表
        dt = [(("%d" % i) * 100, float) for i in range(500)]
        # 创建一个数据类型为 dt 的全一数组
        a = np.ones(1000, dtype=dt)
        # 在捕获警告时，设置记录模式为真
        with warnings.catch_warnings(record=True):
            # 始终警告 UserWarning
            warnings.filterwarnings('always', '', UserWarning)
            # 调用 check_roundtrips 方法，验证数组的写入和读取
            self.check_roundtrips(a)
class TestSaveLoad(RoundtripTest):
    # 继承自RoundtripTest类的测试用例类，用于测试保存和加载功能

    def roundtrip(self, *args, **kwargs):
        # 覆盖父类方法roundtrip，使用np.save保存数据，然后加载
        RoundtripTest.roundtrip(self, np.save, *args, **kwargs)

        # 断言：检查保存和重新加载的数组的第一个元素是否相等
        assert_equal(self.arr[0], self.arr_reloaded)

        # 断言：检查保存和重新加载的数组的dtype是否相等
        assert_equal(self.arr[0].dtype, self.arr_reloaded.dtype)

        # 断言：检查保存和重新加载的数组的fnc标志是否相等
        assert_equal(self.arr[0].flags.fnc, self.arr_reloaded.flags.fnc)


class TestSavezLoad(RoundtripTest):
    # 继承自RoundtripTest类的测试用例类，用于测试保存和加载多个数组的功能

    def roundtrip(self, *args, **kwargs):
        # 覆盖父类方法roundtrip，使用np.savez保存数据，然后加载
        RoundtripTest.roundtrip(self, np.savez, *args, **kwargs)
        
        try:
            # 遍历self.arr中的数组，重新加载后进行比较
            for n, arr in enumerate(self.arr):
                reloaded = self.arr_reloaded['arr_%d' % n]
                
                # 断言：检查数组arr和重新加载的数组reloaded是否相等
                assert_equal(arr, reloaded)
                
                # 断言：检查数组arr和重新加载的数组reloaded的dtype是否相等
                assert_equal(arr.dtype, reloaded.dtype)
                
                # 断言：检查数组arr和重新加载的数组reloaded的fnc标志是否相等
                assert_equal(arr.flags.fnc, reloaded.flags.fnc)
        
        finally:
            # 在Windows系统上，必须在这里删除临时文件
            if self.arr_reloaded.fid:
                self.arr_reloaded.fid.close()
                os.remove(self.arr_reloaded.fid.name)

    @pytest.mark.skipif(IS_PYPY, reason="Hangs on PyPy")
    @pytest.mark.skipif(not IS_64BIT, reason="Needs 64bit platform")
    @pytest.mark.slow
    def test_big_arrays(self):
        # 测试处理大数组的功能

        L = (1 << 31) + 100000
        # 创建一个dtype为np.uint8的长度为L的空数组a
        a = np.empty(L, dtype=np.uint8)
        
        with temppath(prefix="numpy_test_big_arrays_", suffix=".npz") as tmp:
            # 使用np.savez保存数组a到临时文件tmp中
            np.savez(tmp, a=a)
            
            # 删除数组a，释放内存
            del a
            
            # 加载临时文件tmp中的数据到npfile
            npfile = np.load(tmp)
            
            # 从npfile中读取数组a，应该成功
            a = npfile['a']
            
            # 关闭npfile
            npfile.close()
            
            # 删除数组a，避免pyflakes提示未使用变量
            del a

    def test_multiple_arrays(self):
        # 测试保存和加载多个数组的功能
        a = np.array([[1, 2], [3, 4]], float)
        b = np.array([[1 + 2j, 2 + 7j], [3 - 6j, 4 + 12j]], complex)
        
        # 调用roundtrip方法，测试保存和加载多个数组a和b
        self.roundtrip(a, b)

    def test_named_arrays(self):
        # 测试保存和加载命名数组的功能
        a = np.array([[1, 2], [3, 4]], float)
        b = np.array([[1 + 2j, 2 + 7j], [3 - 6j, 4 + 12j]], complex)
        
        c = BytesIO()
        # 使用np.savez将数组a和b保存到BytesIO对象c中
        np.savez(c, file_a=a, file_b=b)
        
        c.seek(0)
        # 从BytesIO对象c中加载数据到l
        l = np.load(c)
        
        # 断言：检查加载的数组a是否与保存时的file_a对应
        assert_equal(a, l['file_a'])
        
        # 断言：检查加载的数组b是否与保存时的file_b对应
        assert_equal(b, l['file_b'])

    def test_tuple_getitem_raises(self):
        # 测试当使用元组作为索引时是否会引发KeyError异常
        # gh-23748
        a = np.array([1, 2, 3])
        
        f = BytesIO()
        # 使用np.savez将数组a保存到BytesIO对象f中
        np.savez(f, a=a)
        
        f.seek(0)
        # 从BytesIO对象f中加载数据到l
        l = np.load(f)
        
        # 使用pytest.raises断言检查是否会抛出KeyError异常，异常信息包含"(1, 2)"
        with pytest.raises(KeyError, match="(1, 2)"):
            l[1, 2]

    def test_BagObj(self):
        # 测试加载数据后检查对象的属性

        a = np.array([[1, 2], [3, 4]], float)
        b = np.array([[1 + 2j, 2 + 7j], [3 - 6j, 4 + 12j]], complex)
        
        c = BytesIO()
        # 使用np.savez将数组a和b保存到BytesIO对象c中
        np.savez(c, file_a=a, file_b=b)
        
        c.seek(0)
        # 从BytesIO对象c中加载数据到l
        l = np.load(c)
        
        # 断言：检查加载后对象l.f的属性列表是否与['file_a', 'file_b']相等
        assert_equal(sorted(dir(l.f)), ['file_a', 'file_b'])
        
        # 断言：检查加载后对象l.f的file_a属性是否与数组a相等
        assert_equal(a, l.f.file_a)
        
        # 断言：检查加载后对象l.f的file_b属性是否与数组b相等
        assert_equal(b, l.f.file_b)

    @pytest.mark.skipif(IS_WASM, reason="Cannot start thread")
    def test_savez_filename_clashes(self):
        # 测试修复问题 #852
        # 在多线程环境中测试 savez 函数

        def writer(error_list):
            # 使用临时路径创建一个 .npz 文件
            with temppath(suffix='.npz') as tmp:
                # 创建一个 500x500 的随机数组
                arr = np.random.randn(500, 500)
                try:
                    # 尝试保存数组到 .npz 文件中
                    np.savez(tmp, arr=arr)
                except OSError as err:
                    # 如果保存过程中出错，记录错误信息
                    error_list.append(err)

        # 初始化错误列表
        errors = []
        # 创建三个线程，每个线程都执行 writer 函数
        threads = [threading.Thread(target=writer, args=(errors,))
                   for j in range(3)]
        # 启动所有线程
        for t in threads:
            t.start()
        # 等待所有线程结束
        for t in threads:
            t.join()

        # 如果有错误发生，则抛出断言错误
        if errors:
            raise AssertionError(errors)

    def test_not_closing_opened_fid(self):
        # 测试修复问题 #2178
        # 验证在 'loaded' 文件上可以进行 seek 操作

        # 使用临时路径创建一个 .npz 文件
        with temppath(suffix='.npz') as tmp:
            # 以写二进制模式打开文件
            with open(tmp, 'wb') as fp:
                # 将字符串 'LOVELY LOAD' 保存到文件中
                np.savez(fp, data='LOVELY LOAD')
            # 以读二进制模式打开文件，设置缓冲区大小为 10000
            with open(tmp, 'rb', 10000) as fp:
                # 将文件指针移到文件开头
                fp.seek(0)
                # 检查文件是否已关闭
                assert_(not fp.closed)
                # 从文件中加载数据
                np.load(fp)['data']
                # 检查文件是否在加载后被关闭
                assert_(not fp.closed)
                # 再次将文件指针移到文件开头
                fp.seek(0)
                # 检查文件是否依然没有关闭
                assert_(not fp.closed)

    @pytest.mark.slow_pypy
    def test_closing_fid(self):
        # 测试修复问题 #1517
        # 确保打开的文件不会过多保持打开状态

        # 使用临时路径创建一个 .npz 文件
        with temppath(suffix='.npz') as tmp:
            # 将字符串 'LOVELY LOAD' 保存到 .npz 文件中
            np.savez(tmp, data='LOVELY LOAD')
            # 我们需要检查垃圾收集器在 numpy npz 文件的引用计数归零时
            # 是否能正确关闭文件。Python 3 在调试模式下，当文件关闭
            # 由垃圾收集器负责时会引发 ResourceWarning，因此我们捕获警告。
            with suppress_warnings() as sup:
                # 过滤掉 ResourceWarning 类型的警告
                sup.filter(ResourceWarning)  # TODO: 指定精确的消息
                # 多次尝试加载数据，以验证文件能否正常关闭
                for i in range(1, 1025):
                    try:
                        np.load(tmp)["data"]
                    except Exception as e:
                        # 如果加载数据失败，抛出断言错误
                        msg = "Failed to load data from a file: %s" % e
                        raise AssertionError(msg)
                    finally:
                        # 如果是在 PyPy 环境下，手动触发垃圾收集
                        if IS_PYPY:
                            gc.collect()
    def test_closing_zipfile_after_load(self):
        # 测试确保 zipfile 拥有文件并能关闭它。需要传入一个文件名进行测试。
        # 在 Windows 上，如果失败，尝试移除打开的文件时将会引发第二个错误。
        prefix = 'numpy_test_closing_zipfile_after_load_'
        # 使用 temppath 函数创建临时文件路径，后缀为 '.npz'，前缀为 prefix
        with temppath(suffix='.npz', prefix=prefix) as tmp:
            # 将数据保存到临时文件中
            np.savez(tmp, lab='place holder')
            # 从临时文件加载数据
            data = np.load(tmp)
            # 获取数据对象的文件指针
            fp = data.zip.fp
            # 关闭数据对象
            data.close()
            # 断言文件指针已关闭
            assert_(fp.closed)

    @pytest.mark.parametrize("count, expected_repr", [
        (1, "NpzFile {fname!r} with keys: arr_0"),
        (5, "NpzFile {fname!r} with keys: arr_0, arr_1, arr_2, arr_3, arr_4"),
        # _MAX_REPR_ARRAY_COUNT is 5, 所以超过5个键的文件预期以 '...' 结尾
        (6, "NpzFile {fname!r} with keys: arr_0, arr_1, arr_2, arr_3, arr_4..."),
    ])
    def test_repr_lists_keys(self, count, expected_repr):
        # 创建一个浮点型的二维数组 a
        a = np.array([[1, 2], [3, 4]], float)
        # 使用 temppath 函数创建临时文件路径，后缀为 '.npz'
        with temppath(suffix='.npz') as tmp:
            # 将多个数组 a 保存到临时文件中
            np.savez(tmp, *[a]*count)
            # 从临时文件加载数据对象 l
            l = np.load(tmp)
            # 断言数据对象 l 的字符串表示等于预期的格式化字符串
            assert repr(l) == expected_repr.format(fname=tmp)
            # 关闭数据对象 l
            l.close()
class TestSaveTxt:
    def test_array(self):
        a = np.array([[1, 2], [3, 4]], float)
        fmt = "%.18e"
        c = BytesIO()
        np.savetxt(c, a, fmt=fmt)  # 将数组 a 保存到字节流 c 中，使用指定的格式 fmt
        c.seek(0)
        assert_equal(c.readlines(),
                     [asbytes((fmt + ' ' + fmt + '\n') % (1, 2)),  # 检查保存后的内容是否符合预期
                      asbytes((fmt + ' ' + fmt + '\n') % (3, 4))])

        a = np.array([[1, 2], [3, 4]], int)
        c = BytesIO()
        np.savetxt(c, a, fmt='%d')  # 重新保存数组 a 到字节流 c，使用整数格式 '%d'
        c.seek(0)
        assert_equal(c.readlines(), [b'1 2\n', b'3 4\n'])  # 检查保存后的内容是否符合预期

    def test_1D(self):
        a = np.array([1, 2, 3, 4], int)
        c = BytesIO()
        np.savetxt(c, a, fmt='%d')  # 将一维数组 a 保存到字节流 c 中，使用整数格式 '%d'
        c.seek(0)
        lines = c.readlines()
        assert_equal(lines, [b'1\n', b'2\n', b'3\n', b'4\n'])  # 检查保存后的内容是否符合预期

    def test_0D_3D(self):
        c = BytesIO()
        assert_raises(ValueError, np.savetxt, c, np.array(1))  # 尝试保存标量，预期抛出 ValueError
        assert_raises(ValueError, np.savetxt, c, np.array([[[1], [2]]]))  # 尝试保存三维数组，预期抛出 ValueError

    def test_structured(self):
        a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        c = BytesIO()
        np.savetxt(c, a, fmt='%d')  # 将结构化数组 a 保存到字节流 c 中，使用整数格式 '%d'
        c.seek(0)
        assert_equal(c.readlines(), [b'1 2\n', b'3 4\n'])  # 检查保存后的内容是否符合预期

    def test_structured_padded(self):
        # gh-13297
        a = np.array([(1, 2, 3),(4, 5, 6)], dtype=[
            ('foo', 'i4'), ('bar', 'i4'), ('baz', 'i4')
        ])
        c = BytesIO()
        np.savetxt(c, a[['foo', 'baz']], fmt='%d')  # 保存结构化数组 a 中的特定字段到字节流 c，使用整数格式 '%d'
        c.seek(0)
        assert_equal(c.readlines(), [b'1 3\n', b'4 6\n'])  # 检查保存后的内容是否符合预期

    def test_multifield_view(self):
        a = np.ones(1, dtype=[('x', 'i4'), ('y', 'i4'), ('z', 'f4')])
        v = a[['x', 'z']]
        with temppath(suffix='.npy') as path:
            path = Path(path)
            np.save(path, v)
            data = np.load(path)
            assert_array_equal(data, v)  # 保存和加载多字段视图 v，确保数据一致性

    def test_delimiter(self):
        a = np.array([[1., 2.], [3., 4.]])
        c = BytesIO()
        np.savetxt(c, a, delimiter=',', fmt='%d')  # 使用指定分隔符 ','，将数组 a 保存到字节流 c 中，使用整数格式 '%d'
        c.seek(0)
        assert_equal(c.readlines(), [b'1,2\n', b'3,4\n'])  # 检查保存后的内容是否符合预期

    def test_format(self):
        a = np.array([(1, 2), (3, 4)])
        c = BytesIO()
        # Sequence of formats
        np.savetxt(c, a, fmt=['%02d', '%3.1f'])  # 使用多个格式保存数组 a 到字节流 c 中
        c.seek(0)
        assert_equal(c.readlines(), [b'01 2.0\n', b'03 4.0\n'])  # 检查保存后的内容是否符合预期

        # A single multiformat string
        c = BytesIO()
        np.savetxt(c, a, fmt='%02d : %3.1f')  # 使用单个多格式字符串保存数组 a 到字节流 c 中
        c.seek(0)
        lines = c.readlines()
        assert_equal(lines, [b'01 : 2.0\n', b'03 : 4.0\n'])  # 检查保存后的内容是否符合预期

        # Specify delimiter, should be overridden
        c = BytesIO()
        np.savetxt(c, a, fmt='%02d : %3.1f', delimiter=',')  # 使用指定分隔符 ','，但应该会被 fmt 中的格式字符串覆盖
        c.seek(0)
        lines = c.readlines()
        assert_equal(lines, [b'01 : 2.0\n', b'03 : 4.0\n'])  # 检查保存后的内容是否符合预期

        # Bad fmt, should raise a ValueError
        c = BytesIO()
        assert_raises(ValueError, np.savetxt, c, a, fmt=99)  # 尝试使用错误的格式 fmt，预期抛出 ValueError
    def test_header_footer(self):
        # Test the functionality of the header and footer keyword argument.

        # 创建一个字节流对象
        c = BytesIO()
        # 创建一个二维整数类型的 NumPy 数组
        a = np.array([(1, 2), (3, 4)], dtype=int)
        # 设置测试用的 header 和 footer 文本
        test_header_footer = 'Test header / footer'

        # 测试 header 关键字参数
        np.savetxt(c, a, fmt='%1d', header=test_header_footer)
        c.seek(0)
        # 断言读取的内容是否符合预期的格式
        assert_equal(c.read(),
                     asbytes('# ' + test_header_footer + '\n1 2\n3 4\n'))

        # 测试 footer 关键字参数
        c = BytesIO()
        np.savetxt(c, a, fmt='%1d', footer=test_header_footer)
        c.seek(0)
        assert_equal(c.read(),
                     asbytes('1 2\n3 4\n# ' + test_header_footer + '\n'))

        # 测试 commentstr 关键字参数在 header 上的应用
        c = BytesIO()
        commentstr = '% '
        np.savetxt(c, a, fmt='%1d',
                   header=test_header_footer, comments=commentstr)
        c.seek(0)
        assert_equal(c.read(),
                     asbytes(commentstr + test_header_footer + '\n' + '1 2\n3 4\n'))

        # 测试 commentstr 关键字参数在 footer 上的应用
        c = BytesIO()
        commentstr = '% '
        np.savetxt(c, a, fmt='%1d',
                   footer=test_header_footer, comments=commentstr)
        c.seek(0)
        assert_equal(c.read(),
                     asbytes('1 2\n3 4\n' + commentstr + test_header_footer + '\n'))

    @pytest.mark.parametrize("filename_type", [Path, str])
    def test_file_roundtrip(self, filename_type):
        # 使用参数化测试，测试文件的读写往返操作
        with temppath() as name:
            # 创建一个二维数组
            a = np.array([(1, 2), (3, 4)])
            # 将数组保存到文件中
            np.savetxt(filename_type(name), a)
            # 从文件中加载数组数据
            b = np.loadtxt(filename_type(name))
            # 断言保存前后数组是否相等
            assert_array_equal(a, b)

    def test_complex_arrays(self):
        # 测试复数数组的保存功能

        ncols = 2
        nrows = 2
        # 创建一个复数类型的零数组
        a = np.zeros((ncols, nrows), dtype=np.complex128)
        re = np.pi
        im = np.e
        # 将数组元素赋值为复数
        a[:] = re + 1.0j * im

        # 测试一种格式适用于所有复数
        c = BytesIO()
        np.savetxt(c, a, fmt=' %+.3e')
        c.seek(0)
        lines = c.readlines()
        assert_equal(
            lines,
            [b' ( +3.142e+00+ +2.718e+00j)  ( +3.142e+00+ +2.718e+00j)\n',
             b' ( +3.142e+00+ +2.718e+00j)  ( +3.142e+00+ +2.718e+00j)\n'])

        # 测试每个实部和虚部使用不同的格式
        c = BytesIO()
        np.savetxt(c, a, fmt='  %+.3e' * 2 * ncols)
        c.seek(0)
        lines = c.readlines()
        assert_equal(
            lines,
            [b'  +3.142e+00  +2.718e+00  +3.142e+00  +2.718e+00\n',
             b'  +3.142e+00  +2.718e+00  +3.142e+00  +2.718e+00\n'])

        # 测试每个复数使用单独的格式
        c = BytesIO()
        np.savetxt(c, a, fmt=['(%.3e%+.3ej)'] * ncols)
        c.seek(0)
        lines = c.readlines()
        assert_equal(
            lines,
            [b'(3.142e+00+2.718e+00j) (3.142e+00+2.718e+00j)\n',
             b'(3.142e+00+2.718e+00j) (3.142e+00+2.718e+00j)\n'])
    def test_complex_negative_exponent(self):
        # 设置测试用例中的列数和行数
        ncols = 2
        nrows = 2
        # 创建一个复数类型的二维数组，元素全为零
        a = np.zeros((ncols, nrows), dtype=np.complex128)
        # 设置实部和虚部的值
        re = np.pi
        im = np.e
        # 将复数数组的所有元素设为复数值（re - 1.0j * im）
        a[:] = re - 1.0j * im
        # 创建一个字节流对象
        c = BytesIO()
        # 将数组a以指定格式写入字节流c
        np.savetxt(c, a, fmt='%.3e')
        # 将字节流的位置移动到开头
        c.seek(0)
        # 读取字节流中的所有行
        lines = c.readlines()
        # 断言读取的行与预期的行相等
        assert_equal(
            lines,
            [b' (3.142e+00-2.718e+00j)  (3.142e+00-2.718e+00j)\n',
             b' (3.142e+00-2.718e+00j)  (3.142e+00-2.718e+00j)\n'])

    def test_custom_writer(self):

        class CustomWriter(list):
            # 自定义的写入方法，将文本按行拆分后加入列表
            def write(self, text):
                self.extend(text.split(b'\n'))

        # 创建一个自定义写入类的实例
        w = CustomWriter()
        # 创建一个二维数组
        a = np.array([(1, 2), (3, 4)])
        # 将数组a写入自定义写入类的实例w中
        np.savetxt(w, a)
        # 从自定义写入类的实例w中加载数据到数组b
        b = np.loadtxt(w)
        # 断言数组a与数组b相等
        assert_array_equal(a, b)

    def test_unicode(self):
        utf8 = b'\xcf\x96'.decode('UTF-8')
        # 创建一个包含UTF-8编码字符串的数组a
        a = np.array([utf8], dtype=np.str_)
        with tempdir() as tmpdir:
            # 将数组a以指定格式写入到临时目录下的文件test.csv中
            np.savetxt(os.path.join(tmpdir, 'test.csv'), a, fmt=['%s'],
                       encoding='UTF-8')

    def test_unicode_roundtrip(self):
        utf8 = b'\xcf\x96'.decode('UTF-8')
        # 创建一个包含UTF-8编码字符串的数组a
        a = np.array([utf8], dtype=np.str_)
        # 使用不同的编码格式，将数组a保存到不同的文件中，并加载为数组b
        suffixes = ['', '.gz']
        if HAS_BZ2:
            suffixes.append('.bz2')
        if HAS_LZMA:
            suffixes.extend(['.xz', '.lzma'])
        with tempdir() as tmpdir:
            for suffix in suffixes:
                np.savetxt(os.path.join(tmpdir, 'test.csv' + suffix), a,
                           fmt=['%s'], encoding='UTF-16-LE')
                b = np.loadtxt(os.path.join(tmpdir, 'test.csv' + suffix),
                               encoding='UTF-16-LE', dtype=np.str_)
                # 断言数组a与数组b相等
                assert_array_equal(a, b)

    def test_unicode_bytestream(self):
        utf8 = b'\xcf\x96'.decode('UTF-8')
        # 创建一个包含UTF-8编码字符串的数组a
        a = np.array([utf8], dtype=np.str_)
        # 创建一个字节流对象s，并将数组a以UTF-8编码格式写入到字节流中
        s = BytesIO()
        np.savetxt(s, a, fmt=['%s'], encoding='UTF-8')
        s.seek(0)
        # 断言读取字节流的内容并解码后与原始UTF-8字符串相等
        assert_equal(s.read().decode('UTF-8'), utf8 + '\n')

    def test_unicode_stringstream(self):
        utf8 = b'\xcf\x96'.decode('UTF-8')
        # 创建一个包含UTF-8编码字符串的数组a
        a = np.array([utf8], dtype=np.str_)
        # 创建一个字符串流对象s，并将数组a以UTF-8编码格式写入到字符串流中
        s = StringIO()
        np.savetxt(s, a, fmt=['%s'], encoding='UTF-8')
        s.seek(0)
        # 断言读取字符串流的内容与原始UTF-8字符串相等
        assert_equal(s.read(), utf8 + '\n')

    @pytest.mark.parametrize("iotype", [StringIO, BytesIO])
    def test_unicode_and_bytes_fmt(self, iotype):
        # 测试参数化，iotype可以是StringIO或BytesIO
        # 数组a中包含一个浮点数
        a = np.array([1.])
        # 创建一个iotype类型的实例s
        s = iotype()
        # 将数组a以指定格式写入到实例s中
        np.savetxt(s, a, fmt="%f")
        s.seek(0)
        if iotype is StringIO:
            # 如果iotype为StringIO，断言读取的内容与预期的字符串格式相等
            assert_equal(s.read(), "%f\n" % 1.)
        else:
            # 如果iotype为BytesIO，断言读取的内容与预期的字节格式相等
            assert_equal(s.read(), b"%f\n" % 1.)

    @pytest.mark.skipif(sys.platform=='win32', reason="files>4GB may not work")
    @pytest.mark.slow
    # 声明一个装饰器函数 `requires_memory`，设置 `free_bytes` 参数为 7e9
    @requires_memory(free_bytes=7e9)
    # 定义一个测试函数 `test_large_zip`，使用 `self` 表示测试类的实例
    def test_large_zip(self):
        # 定义一个内部函数 `check_large_zip`，用于检查处理大型 ZIP 文件时是否发生内存错误
        def check_large_zip(memoryerror_raised):
            # 初始化 `memoryerror_raised` 为 False，用于标记是否发生内存错误异常
            memoryerror_raised.value = False
            try:
                # 测试至少需要 6GB 内存，生成一个大于 4GB 的文件
                # 这测试了 `zipfile` 的 `allowZip64` 参数
                test_data = np.asarray([np.random.rand(
                                        np.random.randint(50,100),4)
                                        for i in range(800000)], dtype=object)
                # 使用 `tempdir()` 创建临时目录，在其中保存生成的数据文件
                with tempdir() as tmpdir:
                    np.savez(os.path.join(tmpdir, 'test.npz'),
                             test_data=test_data)
            except MemoryError:
                # 如果发生内存错误，设置 `memoryerror_raised` 为 True 并重新抛出异常
                memoryerror_raised.value = True
                raise
        
        # 使用共享内存中的对象来在当前进程中重新引发 `MemoryError` 异常（如果需要）
        memoryerror_raised = Value(c_bool)

        # 从 Python 3.8 开始，在 macOS 上将 multiprocessing 的默认启动方法从 'fork' 改为 'spawn'
        # 这可能导致内存共享模型的不一致性，可能导致 `check_large_zip` 测试失败
        ctx = get_context('fork')
        
        # 创建一个新的进程 `p`，运行 `check_large_zip` 函数
        p = ctx.Process(target=check_large_zip, args=(memoryerror_raised,))
        p.start()
        p.join()

        # 如果子进程中发生了 `MemoryError` 异常，则在主进程中引发该异常
        if memoryerror_raised.value:
            raise MemoryError("Child process raised a MemoryError exception")
        
        # 如果子进程以 -9 退出码结束，可能是因为发生了 OOM（Out of Memory）错误
        if p.exitcode == -9:
            pytest.xfail("subprocess got a SIGKILL, apparently free memory was not sufficient")
        
        # 断言子进程正常退出（exitcode 为 0）
        assert p.exitcode == 0
class LoadTxtBase:
    # 检查是否可以从压缩文件加载数据
    def check_compressed(self, fopen, suffixes):
        # 期望的数据，一个二维数组
        wanted = np.arange(6).reshape((2, 3))
        # 定义换行符列表
        linesep = ('\n', '\r\n', '\r')
        # 遍历每种换行符
        for sep in linesep:
            # 创建测试数据字符串
            data = '0 1 2' + sep + '3 4 5'
            # 遍历每种文件后缀名
            for suffix in suffixes:
                # 使用临时文件路径创建文件，并写入数据
                with temppath(suffix=suffix) as name:
                    with fopen(name, mode='wt', encoding='UTF-32-LE') as f:
                        f.write(data)
                    # 加载数据并断言结果与期望值相等
                    res = self.loadfunc(name, encoding='UTF-32-LE')
                    assert_array_equal(res, wanted)
                    # 重新以只读模式打开文件，并加载数据，再次断言结果与期望值相等
                    with fopen(name, "rt",  encoding='UTF-32-LE') as f:
                        res = self.loadfunc(f)
                    assert_array_equal(res, wanted)

    # 测试使用gzip压缩文件加载数据
    def test_compressed_gzip(self):
        self.check_compressed(gzip.open, ('.gz',))

    # 如果没有安装bz2模块，则跳过测试
    @pytest.mark.skipif(not HAS_BZ2, reason="Needs bz2")
    # 测试使用bz2压缩文件加载数据
    def test_compressed_bz2(self):
        self.check_compressed(bz2.open, ('.bz2',))

    # 如果没有安装lzma模块，则跳过测试
    @pytest.mark.skipif(not HAS_LZMA, reason="Needs lzma")
    # 测试使用lzma压缩文件加载数据
    def test_compressed_lzma(self):
        self.check_compressed(lzma.open, ('.xz', '.lzma'))

    # 测试文件编码加载
    def test_encoding(self):
        # 使用临时文件路径创建文件，并写入编码为UTF-16的数据
        with temppath() as path:
            with open(path, "wb") as f:
                f.write('0.\n1.\n2.'.encode("UTF-16"))
            # 加载数据并断言结果与期望值相等
            x = self.loadfunc(path, encoding="UTF-16")
            assert_array_equal(x, [0., 1., 2.])

    # 测试加载字符串数据
    def test_stringload(self):
        # 定义非ASCII字符
        nonascii = b'\xc3\xb6\xc3\xbc\xc3\xb6'.decode("UTF-8")
        # 使用临时文件路径创建文件，并写入非ASCII编码的数据
        with temppath() as path:
            with open(path, "wb") as f:
                f.write(nonascii.encode("UTF-16"))
            # 加载数据并断言结果与非ASCII字符相等
            x = self.loadfunc(path, encoding="UTF-16", dtype=np.str_)
            assert_array_equal(x, nonascii)

    # 测试二进制数据解码
    def test_binary_decode(self):
        # 定义UTF-16编码的二进制数据
        utf16 = b'\xff\xfeh\x04 \x00i\x04 \x00j\x04'
        # 使用BytesIO创建字节流，并加载数据并断言结果与预期相等
        v = self.loadfunc(BytesIO(utf16), dtype=np.str_, encoding='UTF-16')
        assert_array_equal(v, np.array(utf16.decode('UTF-16').split()))

    # 测试使用转换器解码
    def test_converters_decode(self):
        # 测试解码字符串的转换器
        c = TextIO()
        c.write(b'\xcf\x96')
        c.seek(0)
        # 加载数据并使用转换器将第一列解码为UTF-8，并断言结果与预期相等
        x = self.loadfunc(c, dtype=np.str_, encoding="bytes",
                          converters={0: lambda x: x.decode('UTF-8')})
        a = np.array([b'\xcf\x96'.decode('UTF-8')])
        assert_array_equal(x, a)

    # 测试不解码的转换器
    def test_converters_nodecode(self):
        # 测试通过设置编码启用原生字符串转换器
        utf8 = b'\xcf\x96'.decode('UTF-8')
        # 使用临时文件路径创建文件，并写入UTF-8编码的数据
        with temppath() as path:
            with open(path, 'wt', encoding='UTF-8') as f:
                f.write(utf8)
            # 加载数据并使用转换器将第一列添加后缀 't'，并断言结果与预期相等
            x = self.loadfunc(path, dtype=np.str_,
                              converters={0: lambda x: x + 't'},
                              encoding='UTF-8')
            a = np.array([utf8 + 't'])
            assert_array_equal(x, a)


class TestLoadTxt(LoadTxtBase):
    loadfunc = staticmethod(np.loadtxt)
    # 设置测试方法的初始化，降低 _loadtxt_chunksize 以便进行测试
    def setup_method(self):
        # 保存原始的 _loadtxt_chunksize 值
        self.orig_chunk = _npyio_impl._loadtxt_chunksize
        # 设置 _loadtxt_chunksize 为 1
        _npyio_impl._loadtxt_chunksize = 1

    # 设置测试方法的清理操作，恢复 _loadtxt_chunksize 到原始值
    def teardown_method(self):
        # 恢复 _loadtxt_chunksize 到原始值
        _npyio_impl._loadtxt_chunksize = self.orig_chunk

    # 测试读取包含记录的文本数据
    def test_record(self):
        # 创建一个文本流对象 c
        c = TextIO()
        # 向文本流写入数据
        c.write('1 2\n3 4')
        # 将流的位置移到开头
        c.seek(0)
        # 使用 numpy 的 loadtxt 函数读取数据并指定数据类型为记录类型
        x = np.loadtxt(c, dtype=[('x', np.int32), ('y', np.int32)])
        # 创建预期的数组 a
        a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        # 断言读取的数组 x 和预期数组 a 相等
        assert_array_equal(x, a)

        # 创建另一个文本流对象 d
        d = TextIO()
        # 向文本流写入数据
        d.write('M 64 75.0\nF 25 60.0')
        # 将流的位置移到开头
        d.seek(0)
        # 定义数据描述符字典 mydescriptor
        mydescriptor = {'names': ('gender', 'age', 'weight'),
                        'formats': ('S1', 'i4', 'f4')}
        # 创建预期的数组 b
        b = np.array([('M', 64.0, 75.0),
                      ('F', 25.0, 60.0)], dtype=mydescriptor)
        # 使用 numpy 的 loadtxt 函数读取数据并指定数据类型为 mydescriptor
        y = np.loadtxt(d, dtype=mydescriptor)
        # 断言读取的数组 y 和预期数组 b 相等
        assert_array_equal(y, b)

    # 测试读取普通的二维数组数据
    def test_array(self):
        # 创建一个文本流对象 c
        c = TextIO()
        # 向文本流写入数据
        c.write('1 2\n3 4')
        # 将流的位置移到开头
        c.seek(0)
        # 使用 numpy 的 loadtxt 函数读取数据并指定数据类型为 int
        x = np.loadtxt(c, dtype=int)
        # 创建预期的二维数组 a
        a = np.array([[1, 2], [3, 4]], int)
        # 断言读取的数组 x 和预期数组 a 相等
        assert_array_equal(x, a)

        # 将流的位置再次移到开头
        c.seek(0)
        # 使用 numpy 的 loadtxt 函数读取数据并指定数据类型为 float
        x = np.loadtxt(c, dtype=float)
        # 创建预期的二维数组 a
        a = np.array([[1, 2], [3, 4]], float)
        # 断言读取的数组 x 和预期数组 a 相等
        assert_array_equal(x, a)

    # 测试读取一维数组数据
    def test_1D(self):
        # 创建一个文本流对象 c
        c = TextIO()
        # 向文本流写入数据
        c.write('1\n2\n3\n4\n')
        # 将流的位置移到开头
        c.seek(0)
        # 使用 numpy 的 loadtxt 函数读取数据并指定数据类型为 int
        x = np.loadtxt(c, dtype=int)
        # 创建预期的一维数组 a
        a = np.array([1, 2, 3, 4], int)
        # 断言读取的数组 x 和预期数组 a 相等
        assert_array_equal(x, a)

        # 创建另一个文本流对象 c
        c = TextIO()
        # 向文本流写入数据
        c.write('1,2,3,4\n')
        # 将流的位置移到开头
        c.seek(0)
        # 使用 numpy 的 loadtxt 函数读取数据并指定数据类型为 int，同时指定分隔符为逗号
        x = np.loadtxt(c, dtype=int, delimiter=',')
        # 创建预期的一维数组 a
        a = np.array([1, 2, 3, 4], int)
        # 断言读取的数组 x 和预期数组 a 相等
        assert_array_equal(x, a)

    # 测试读取包含缺失数据的数组
    def test_missing(self):
        # 创建一个文本流对象 c
        c = TextIO()
        # 向文本流写入数据，包含缺失的数据
        c.write('1,2,3,,5\n')
        # 将流的位置移到开头
        c.seek(0)
        # 使用 numpy 的 loadtxt 函数读取数据并指定数据类型为 int，同时指定分隔符为逗号
        # 使用 converters 将第 3 列的空值转换为 -999
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       converters={3: lambda s: int(s or -999)})
        # 创建预期的数组 a
        a = np.array([1, 2, 3, -999, 5], int)
        # 断言读取的数组 x 和预期数组 a 相等
        assert_array_equal(x, a)

    # 测试同时使用 converters 和 usecols 参数读取数据
    def test_converters_with_usecols(self):
        # 创建一个文本流对象 c
        c = TextIO()
        # 向文本流写入数据
        c.write('1,2,3,,5\n6,7,8,9,10\n')
        # 将流的位置移到开头
        c.seek(0)
        # 使用 numpy 的 loadtxt 函数读取数据并指定数据类型为 int，同时指定分隔符为逗号
        # 使用 converters 将第 3 列的空值转换为 -999，同时只读取第 2 和第 4 列数据
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       converters={3: lambda s: int(s or -999)},
                       usecols=(1, 3,))
        # 创建预期的二维数组 a
        a = np.array([[2, -999], [7, 9]], int)
        # 断言读取的数组 x 和预期数组 a 相等
        assert_array_equal(x, a)

    # 测试读取带有 Unicode 注释的数据
    def test_comments_unicode(self):
        # 创建一个文本流对象 c
        c = TextIO()
        # 向文本流写入数据，包含 Unicode 注释
        c.write('# comment\n1,2,3,5\n')
        # 将流的位置移到开头
        c.seek(0)
        # 使用 numpy 的 loadtxt 函数读取数据并指定数据类型为 int，同时指定分隔符为逗号
        # 使用 comments 参数指定 Unicode 注释符号 #
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       comments='#')
        # 创建预期的数组 a
        a = np.array([1, 2, 3, 5], int)
        # 断言读取的数组 x 和预期数组 a 相等
        assert_array_equal(x, a)

    # 测试读取带有字节形式的注释的数据
    def test_comments_byte(self):
        # 创建一个文本流对象 c
        c = TextIO()
        # 向文本流写入数据，包含字节形式的注释
        c.write('# comment\n1,2,3,5\n')
        # 将流的位置移到开头
        c.seek(0)
        # 使用 numpy 的 loadtxt 函数读取数据并指定数据类型为 int，同时指定分隔符为逗号
        # 使用 comments 参数指定字节形式的注释符号 b'#'
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       comments=b'#')
        # 创建预期的数组 a
        a = np.array([1, 2, 3, 5], int)
        # 断言读取的数组 x 和预期数组 a
    # 定义一个测试方法，用于测试加载带有注释的文本数据到 NumPy 数组的功能
    def test_comments_multiple(self):
        # 创建一个文本流对象
        c = TextIO()
        # 写入包含注释的文本数据到文本流中
        c.write('# comment\n1,2,3\n@ comment2\n4,5,6 // comment3')
        # 将文本流的读取位置移动到开头
        c.seek(0)
        # 使用 np.loadtxt 函数加载文本流中的数据到 NumPy 数组 x，指定数据类型为整数，分隔符为逗号，
        # 并指定注释符号为 ['#', '@', '//']
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       comments=['#', '@', '//'])
        # 创建一个预期的 NumPy 数组 a，包含相同的数据
        a = np.array([[1, 2, 3], [4, 5, 6]], int)
        # 断言加载的数组 x 与预期数组 a 相等
        assert_array_equal(x, a)

    # 标记为 pytest 的测试方法，用于测试加载带有多字符注释符的文本数据到 NumPy 数组的功能
    @pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                        reason="PyPy bug in error formatting")
    def test_comments_multi_chars(self):
        # 创建一个文本流对象
        c = TextIO()
        # 写入包含多字符注释的文本数据到文本流中
        c.write('/* comment\n1,2,3,5\n')
        # 将文本流的读取位置移动到开头
        c.seek(0)
        # 使用 np.loadtxt 函数加载文本流中的数据到 NumPy 数组 x，指定数据类型为整数，分隔符为逗号，
        # 并指定多字符注释符号为 '/*'
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       comments='/*')
        # 创建一个预期的 NumPy 数组 a，包含相同的数据
        a = np.array([1, 2, 3, 5], int)
        # 断言加载的数组 x 与预期数组 a 相等
        assert_array_equal(x, a)

        # 再次创建一个文本流对象
        c = TextIO()
        # 写入包含不应转换为注释的文本数据到文本流中
        c.write('*/ comment\n1,2,3,5\n')
        # 将文本流的读取位置移动到开头
        c.seek(0)
        # 断言加载包含非法注释 '/*' 的文本数据时会抛出 ValueError 异常
        assert_raises(ValueError, np.loadtxt, c, dtype=int, delimiter=',',
                      comments='/*')

    # 定义一个测试方法，用于测试跳过指定行数后加载文本数据到 NumPy 数组的功能
    def test_skiprows(self):
        # 创建一个文本流对象
        c = TextIO()
        # 写入包含注释的文本数据到文本流中
        c.write('comment\n1,2,3,5\n')
        # 将文本流的读取位置移动到开头
        c.seek(0)
        # 使用 np.loadtxt 函数跳过第一行，加载文本流中的数据到 NumPy 数组 x，指定数据类型为整数，分隔符为逗号
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       skiprows=1)
        # 创建一个预期的 NumPy 数组 a，包含相同的数据
        a = np.array([1, 2, 3, 5], int)
        # 断言加载的数组 x 与预期数组 a 相等
        assert_array_equal(x, a)

        # 再次创建一个文本流对象
        c = TextIO()
        # 写入包含以 '#' 开头的注释的文本数据到文本流中
        c.write('# comment\n1,2,3,5\n')
        # 将文本流的读取位置移动到开头
        c.seek(0)
        # 使用 np.loadtxt 函数跳过第一行，加载文本流中的数据到 NumPy 数组 x，指定数据类型为整数，分隔符为逗号
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       skiprows=1)
        # 创建一个预期的 NumPy 数组 a，包含相同的数据
        a = np.array([1, 2, 3, 5], int)
        # 断言加载的数组 x 与预期数组 a 相等
        assert_array_equal(x, a)
    # 定义一个测试方法，用于测试 `np.loadtxt` 函数中的 `usecols` 参数
    def test_usecols(self):
        # 创建一个二维 NumPy 数组 `a`，包含浮点数元素
        a = np.array([[1, 2], [3, 4]], float)
        # 创建一个字节流对象 `c`
        c = BytesIO()
        # 将数组 `a` 写入字节流 `c`
        np.savetxt(c, a)
        # 将字节流的读取位置移到开头
        c.seek(0)
        # 从字节流 `c` 中加载数据到 `x`，只使用第二列的数据
        x = np.loadtxt(c, dtype=float, usecols=(1,))
        # 断言 `x` 和 `a` 的第二列数据相等
        assert_array_equal(x, a[:, 1])

        # 更新数组 `a` 为包含三列的新数组
        a = np.array([[1, 2, 3], [3, 4, 5]], float)
        # 清空字节流 `c` 并将更新后的数组 `a` 写入其中
        c = BytesIO()
        np.savetxt(c, a)
        # 将字节流的读取位置移到开头
        c.seek(0)
        # 从字节流 `c` 中加载数据到 `x`，使用第二和第三列的数据
        x = np.loadtxt(c, dtype=float, usecols=(1, 2))
        # 断言 `x` 和 `a` 的第二和第三列数据相等
        assert_array_equal(x, a[:, 1:])

        # 测试用数组替代元组的情况
        c.seek(0)
        # 从字节流 `c` 中加载数据到 `x`，使用第二和第三列的数据
        x = np.loadtxt(c, dtype=float, usecols=np.array([1, 2]))
        # 断言 `x` 和 `a` 的第二和第三列数据相等
        assert_array_equal(x, a[:, 1:])

        # 测试使用整数而不是序列的情况
        for int_type in [int, np.int8, np.int16,
                         np.int32, np.int64, np.uint8, np.uint16,
                         np.uint32, np.uint64]:
            to_read = int_type(1)
            c.seek(0)
            # 从字节流 `c` 中加载数据到 `x`，仅使用第一列数据
            x = np.loadtxt(c, dtype=float, usecols=to_read)
            # 断言 `x` 和 `a` 的第二列数据相等
            assert_array_equal(x, a[:, 1])

        # 测试自定义整数类型的情况
        class CrazyInt:
            def __index__(self):
                return 1

        crazy_int = CrazyInt()
        c.seek(0)
        # 从字节流 `c` 中加载数据到 `x`，仅使用第一列数据
        x = np.loadtxt(c, dtype=float, usecols=crazy_int)
        # 断言 `x` 和 `a` 的第二列数据相等
        assert_array_equal(x, a[:, 1])

        c.seek(0)
        # 从字节流 `c` 中加载数据到 `x`，仅使用第一列数据
        x = np.loadtxt(c, dtype=float, usecols=(crazy_int,))
        # 断言 `x` 和 `a` 的第二列数据相等
        assert_array_equal(x, a[:, 1])

        # 使用定义了转换器的数据进行检查
        data = '''JOE 70.1 25.3
                BOB 60.5 27.9
                '''
        # 创建一个模拟文本流对象 `c`
        c = TextIO(data)
        # 定义字段名和数据类型
        names = ['stid', 'temp']
        dtypes = ['S4', 'f8']
        # 从文本流 `c` 中加载数据到数组 `arr`，只使用第一列和第三列数据
        arr = np.loadtxt(c, usecols=(0, 2), dtype=list(zip(names, dtypes)))
        # 检查数组 `arr` 中的 'stid' 和 'temp' 字段数据
        assert_equal(arr['stid'], [b"JOE", b"BOB"])
        assert_equal(arr['temp'], [25.3, 27.9])

        # 测试 `usecols` 中包含非整数的情况
        c.seek(0)
        # 定义一个无效的索引 `bogus_idx`
        bogus_idx = 1.5
        # 断言在加载数据时会引发 `TypeError` 异常，提示 `usecols` 必须是整数类型
        assert_raises_regex(
            TypeError,
            '^usecols must be.*%s' % type(bogus_idx).__name__,
            np.loadtxt, c, usecols=bogus_idx
            )

        # 断言在加载数据时会引发 `TypeError` 异常，提示 `usecols` 中存在非整数类型
        assert_raises_regex(
            TypeError,
            '^usecols must be.*%s' % type(bogus_idx).__name__,
            np.loadtxt, c, usecols=[0, bogus_idx, 0]
            )

    # 定义一个测试方法，用于测试 `np.loadtxt` 函数中 `usecols` 参数的异常情况
    def test_bad_usecols(self):
        # 断言在加载数据时会引发 `OverflowError` 异常，因为给定的列索引超出范围
        with pytest.raises(OverflowError):
            np.loadtxt(["1\n"], usecols=[2**64], delimiter=",")

        # 断言在加载数据时会引发 `ValueError` 或 `OverflowError` 异常，给定的列索引超出范围
        with pytest.raises((ValueError, OverflowError)):
            # 在 32 位平台上可能会引发 `OverflowError` 异常
            np.loadtxt(["1\n"], usecols=[2**62], delimiter=",")

        # 断言在加载数据时会引发 `TypeError` 异常，因为给定的字段数与结构化数据类型不匹配
        with pytest.raises(TypeError,
                match="If a structured dtype .*. But 1 usecols were given and "
                      "the number of fields is 3."):
            np.loadtxt(["1,1\n"], dtype="i,2i", usecols=[0], delimiter=",")
    def test_fancy_dtype(self):
        # 创建一个 TextIO 对象
        c = TextIO()
        # 向 TextIO 对象写入数据
        c.write('1,2,3.0\n4,5,6.0\n')
        # 将文件指针位置移动到起始位置
        c.seek(0)
        # 定义一个结构化的 NumPy 数据类型 dt
        dt = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        # 使用 np.loadtxt 从 TextIO 对象中加载数据，使用指定的数据类型和分隔符
        x = np.loadtxt(c, dtype=dt, delimiter=',')
        # 创建预期的 NumPy 数组 a，用于后续断言比较
        a = np.array([(1, (2, 3.0)), (4, (5, 6.0))], dt)
        # 使用断言检查 x 和 a 是否相等
        assert_array_equal(x, a)

    def test_shaped_dtype(self):
        # 创建一个 TextIO 对象并初始化数据
        c = TextIO("aaaa  1.0  8.0  1 2 3 4 5 6")
        # 定义一个结构化的 NumPy 数据类型 dt，包含名字、两个浮点数和一个二维整数数组
        dt = np.dtype([('name', 'S4'), ('x', float), ('y', float),
                       ('block', int, (2, 3))])
        # 使用 np.loadtxt 从 TextIO 对象中加载数据，使用指定的数据类型
        x = np.loadtxt(c, dtype=dt)
        # 创建预期的 NumPy 数组 a，用于后续断言比较
        a = np.array([('aaaa', 1.0, 8.0, [[1, 2, 3], [4, 5, 6]])],
                     dtype=dt)
        # 使用断言检查 x 和 a 是否相等
        assert_array_equal(x, a)

    def test_3d_shaped_dtype(self):
        # 创建一个 TextIO 对象并初始化数据
        c = TextIO("aaaa  1.0  8.0  1 2 3 4 5 6 7 8 9 10 11 12")
        # 定义一个结构化的 NumPy 数据类型 dt，包含名字、两个浮点数和一个三维整数数组
        dt = np.dtype([('name', 'S4'), ('x', float), ('y', float),
                       ('block', int, (2, 2, 3))])
        # 使用 np.loadtxt 从 TextIO 对象中加载数据，使用指定的数据类型
        x = np.loadtxt(c, dtype=dt)
        # 创建预期的 NumPy 数组 a，用于后续断言比较
        a = np.array([('aaaa', 1.0, 8.0,
                       [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])],
                     dtype=dt)
        # 使用断言检查 x 和 a 是否相等
        assert_array_equal(x, a)

    def test_str_dtype(self):
        # 创建一个字符串列表作为输入数据
        c = ["str1", "str2"]

        # 遍历两种数据类型：str 和 np.bytes_
        for dt in (str, np.bytes_):
            # 创建一个 NumPy 数组 a，包含两个字符串元素
            a = np.array(["str1", "str2"], dtype=dt)
            # 使用 np.loadtxt 从字符串列表 c 中加载数据，使用指定的数据类型 dt
            x = np.loadtxt(c, dtype=dt)
            # 使用断言检查 x 和 a 是否相等
            assert_array_equal(x, a)

    def test_empty_file(self):
        # 使用 pytest 的 warns 方法检查 UserWarning，并匹配指定的字符串
        with pytest.warns(UserWarning, match="input contained no data"):
            # 创建一个空的 TextIO 对象
            c = TextIO()
            # 使用 np.loadtxt 从空的 TextIO 对象中加载数据
            x = np.loadtxt(c)
            # 使用断言检查返回的数组 x 的形状是否为 (0,)
            assert_equal(x.shape, (0,))
            # 再次使用 np.loadtxt 从空的 TextIO 对象中加载数据，指定数据类型为 np.int64
            x = np.loadtxt(c, dtype=np.int64)
            # 使用断言检查返回的数组 x 的形状是否为 (0,)，数据类型是否为 np.int64
            assert_equal(x.shape, (0,))
            assert_(x.dtype == np.int64)

    def test_unused_converter(self):
        # 创建一个 TextIO 对象并写入数据行
        c = TextIO()
        c.writelines(['1 21\n', '3 42\n'])
        c.seek(0)
        # 使用 np.loadtxt 从 TextIO 对象中加载数据，仅使用第二列数据，第一列使用转换器进行十六进制转换
        data = np.loadtxt(c, usecols=(1,),
                          converters={0: lambda s: int(s, 16)})
        # 使用断言检查返回的数组 data 是否与预期相等
        assert_array_equal(data, [21, 42])

        c.seek(0)
        # 再次使用 np.loadtxt 从 TextIO 对象中加载数据，仅使用第二列数据，第二列使用转换器进行十六进制转换
        data = np.loadtxt(c, usecols=(1,),
                          converters={1: lambda s: int(s, 16)})
        # 使用断言检查返回的数组 data 是否与预期相等
        assert_array_equal(data, [33, 66])

    def test_dtype_with_object(self):
        # 定义包含日期数据的字符串
        data = """ 1; 2001-01-01
                   2; 2002-01-31 """
        # 定义结构化的 NumPy 数据类型 ndtype，包含整数和日期对象
        ndtype = [('idx', int), ('code', object)]
        # 定义一个转换函数 func，将日期字符串转换为 datetime 对象
        func = lambda s: strptime(s.strip(), "%Y-%m-%d")
        converters = {1: func}
        # 使用 np.loadtxt 从 TextIO 对象中加载数据，使用指定的分隔符、数据类型和转换器
        test = np.loadtxt(TextIO(data), delimiter=";", dtype=ndtype,
                          converters=converters)
        # 创建预期的 NumPy 数组 control，用于后续断言比较
        control = np.array(
            [(1, datetime(2001, 1, 1)), (2, datetime(2002, 1, 31))],
            dtype=ndtype)
        # 使用断言检查 test 和 control 是否相等
        assert_equal(test, control)

    def test_uint64_type(self):
        # 定义一个包含两个 uint64 类型数据的元组
        tgt = (9223372043271415339, 9223372043271415853)
        # 创建一个 TextIO 对象并将数据写入
        c = TextIO()
        c.write("%s %s" % tgt)
        c.seek(0)
        # 使用 np.loadtxt 从 TextIO 对象中加载数据，数据类型指定为 np.uint64
        res = np.loadtxt(c, dtype=np.uint64)
        # 使用断言检查返回的数组 res 是否与预期的 tgt 相等
        assert_equal(res, tgt)
    def test_int64_type(self):
        # 定义目标值为一个包含最大和最小 int64 数值的元组
        tgt = (-9223372036854775807, 9223372036854775807)
        # 创建一个 TextIO 对象
        c = TextIO()
        # 将目标值写入 TextIO 对象
        c.write("%s %s" % tgt)
        # 将写入指针移动到文件开头
        c.seek(0)
        # 从 TextIO 对象中加载数据，指定数据类型为 np.int64
        res = np.loadtxt(c, dtype=np.int64)
        # 断言加载的结果与目标值相等
        assert_equal(res, tgt)

    def test_from_float_hex(self):
        # 使用 np.logspace 创建一个包含浮点数的目标数组，数据类型为 np.float32
        tgt = np.logspace(-10, 10, 5).astype(np.float32)
        # 将 tgt 扩展为包含其负值的数组，并转换为普通的 float 类型
        tgt = np.hstack((tgt, -tgt)).astype(float)
        # 将 tgt 中的浮点数转换为十六进制字符串，并用换行符连接成一个字符串
        inp = '\n'.join(map(float.hex, tgt))
        # 创建一个 TextIO 对象
        c = TextIO()
        # 将 inp 写入 TextIO 对象
        c.write(inp)
        # 针对 float 和 np.float32 两种数据类型进行循环
        for dt in [float, np.float32]:
            # 将写入指针移动到文件开头
            c.seek(0)
            # 从 TextIO 对象中加载数据，指定数据类型为 dt，使用 float.fromhex 进行转换，编码为 "latin1"
            res = np.loadtxt(
                c, dtype=dt, converters=float.fromhex, encoding="latin1")
            # 断言加载的结果与目标值相等，如果不等则输出错误消息
            assert_equal(res, tgt, err_msg="%s" % dt)

    @pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                        reason="PyPy bug in error formatting")
    def test_default_float_converter_no_default_hex_conversion(self):
        """
        确保 fromhex 只用于带有正确前缀的值，并且不会默认调用。与 gh-19598 相关的回归测试。
        """
        # 创建一个 TextIO 对象，写入字符串 "a b c"
        c = TextIO("a b c")
        # 使用 pytest 检查是否会引发 ValueError 异常，错误消息包含特定内容
        with pytest.raises(ValueError,
                match=".*convert string 'a' to float64 at row 0, column 1"):
            # 尝试从 TextIO 对象中加载数据
            np.loadtxt(c)

    @pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                        reason="PyPy bug in error formatting")
    def test_default_float_converter_exception(self):
        """
        确保在浮点数转换失败时引发的异常消息是正确的。与 gh-19598 相关的回归测试。
        """
        # 创建一个 TextIO 对象，写入字符串 "qrs tuv"，这些值对默认的浮点数转换器来说是无效的
        c = TextIO("qrs tuv")
        # 使用 pytest 检查是否会引发 ValueError 异常，错误消息包含特定内容
        with pytest.raises(ValueError,
                match="could not convert string 'qrs' to float64"):
            # 尝试从 TextIO 对象中加载数据
            np.loadtxt(c)

    def test_from_complex(self):
        # 定义一个包含复数的目标元组
        tgt = (complex(1, 1), complex(1, -1))
        # 创建一个 TextIO 对象
        c = TextIO()
        # 将目标复数写入 TextIO 对象
        c.write("%s %s" % tgt)
        # 将写入指针移动到文件开头
        c.seek(0)
        # 从 TextIO 对象中加载数据，指定数据类型为复数
        res = np.loadtxt(c, dtype=complex)
        # 断言加载的结果与目标值相等
        assert_equal(res, tgt)

    def test_complex_misformatted(self):
        # 用于向后兼容的测试
        # 一些复杂格式曾生成 x+-yj
        # 创建一个复数数组 a，全部元素为 0
        a = np.zeros((2, 2), dtype=np.complex128)
        # 设置实部和虚部的值
        re = np.pi
        im = np.e
        # 用给定的实部和虚部值填充数组 a
        a[:] = re - 1.0j * im
        # 创建一个 BytesIO 对象
        c = BytesIO()
        # 将数组 a 以科学计数法格式写入 BytesIO 对象
        np.savetxt(c, a, fmt='%.16e')
        # 将写入指针移动到文件开头
        c.seek(0)
        # 读取 BytesIO 对象的内容
        txt = c.read()
        # 将写入指针移动到文件开头
        c.seek(0)
        # 在复数的虚部符号上误格式化，gh 7895
        txt_bad = txt.replace(b'e+00-', b'e00+-')
        # 断言 txt_bad 和 txt 不相等
        assert_(txt_bad != txt)
        # 将 txt_bad 写入 BytesIO 对象
        c.write(txt_bad)
        # 将写入指针移动到文件开头
        c.seek(0)
        # 从 BytesIO 对象中加载数据，指定数据类型为复数
        res = np.loadtxt(c, dtype=complex)
        # 断言加载的结果与数组 a 相等
        assert_equal(res, a)
    # 测试函数：test_universal_newline
    def test_universal_newline(self):
        # 使用上下文管理器创建临时文件，并在文件中写入数据
        with temppath() as name:
            with open(name, 'w') as f:
                f.write('1 21\r3 42\r')
            # 使用 NumPy 的 loadtxt 函数加载临时文件中的数据
            data = np.loadtxt(name)
        # 断言加载的数据与预期的二维数组相等
        assert_array_equal(data, [[1, 21], [3, 42]])

    # 测试函数：test_empty_field_after_tab
    def test_empty_field_after_tab(self):
        # 创建 TextIO 对象 c
        c = TextIO()
        # 向 TextIO 对象中写入带制表符的数据
        c.write('1 \t2 \t3\tstart \n4\t5\t6\t  \n7\t8\t9.5\t')
        # 将文件指针移动到文件开头
        c.seek(0)
        # 定义数据类型 dt，指定各列的名称、格式及注释字段的最大长度
        dt = {'names': ('x', 'y', 'z', 'comment'),
              'formats': ('<i4', '<i4', '<f4', '|S8')}
        # 使用 NumPy 的 loadtxt 函数从 TextIO 对象 c 中加载数据
        x = np.loadtxt(c, dtype=dt, delimiter='\t')
        # 创建预期的字符串数组 a
        a = np.array([b'start ', b'  ', b''])
        # 断言加载数据中的 comment 列与预期数组 a 相等
        assert_array_equal(x['comment'], a)

    # 测试函数：test_unpack_structured
    def test_unpack_structured(self):
        # 创建 TextIO 对象 txt，并写入数据
        txt = TextIO("M 21 72\nF 35 58")
        # 定义结构化数据类型 dt，指定字段名及格式
        dt = {'names': ('a', 'b', 'c'), 'formats': ('|S1', '<i4', '<f4')}
        # 使用 NumPy 的 loadtxt 函数从 TextIO 对象 txt 中加载数据，并拆分为各列
        a, b, c = np.loadtxt(txt, dtype=dt, unpack=True)
        # 断言各列的数据类型与预期相符
        assert_(a.dtype.str == '|S1')
        assert_(b.dtype.str == '<i4')
        assert_(c.dtype.str == '<f4')
        # 断言加载的数据与预期的数组相等
        assert_array_equal(a, np.array([b'M', b'F']))
        assert_array_equal(b, np.array([21, 35]))
        assert_array_equal(c, np.array([72.,  58.]))

    # 测试函数：test_ndmin_keyword
    def test_ndmin_keyword(self):
        # 创建 TextIO 对象 c，并写入数据
        c = TextIO()
        c.write('1,2,3\n4,5,6')
        c.seek(0)
        # 断言加载时传入 ndmin 参数为 3 会引发 ValueError 异常
        assert_raises(ValueError, np.loadtxt, c, ndmin=3)
        c.seek(0)
        # 断言加载时传入 ndmin 参数为 1.5 会引发 ValueError 异常
        assert_raises(ValueError, np.loadtxt, c, ndmin=1.5)
        c.seek(0)
        # 使用 NumPy 的 loadtxt 函数从 TextIO 对象 c 中加载数据，指定数据类型为整数，指定 ndmin 为 1
        x = np.loadtxt(c, dtype=int, delimiter=',', ndmin=1)
        # 创建预期的二维整数数组 a
        a = np.array([[1, 2, 3], [4, 5, 6]])
        # 断言加载的数据与预期数组 a 相等
        assert_array_equal(x, a)

        # 创建 TextIO 对象 d，并写入数据
        d = TextIO()
        d.write('0,1,2')
        d.seek(0)
        # 使用 NumPy 的 loadtxt 函数从 TextIO 对象 d 中加载数据，指定数据类型为整数，指定 ndmin 为 2
        x = np.loadtxt(d, dtype=int, delimiter=',', ndmin=2)
        # 断言加载的数组形状为 (1, 3)
        assert_(x.shape == (1, 3))
        d.seek(0)
        # 使用 NumPy 的 loadtxt 函数从 TextIO 对象 d 中加载数据，指定数据类型为整数，指定 ndmin 为 1
        x = np.loadtxt(d, dtype=int, delimiter=',', ndmin=1)
        # 断言加载的数组形状为 (3,)
        assert_(x.shape == (3,))
        d.seek(0)
        # 使用 NumPy 的 loadtxt 函数从 TextIO 对象 d 中加载数据，指定数据类型为整数，指定 ndmin 为 0
        x = np.loadtxt(d, dtype=int, delimiter=',', ndmin=0)
        # 断言加载的数组形状为 (3,)
        assert_(x.shape == (3,))

        # 创建 TextIO 对象 e，并写入数据
        e = TextIO()
        e.write('0\n1\n2')
        e.seek(0)
        # 使用 NumPy 的 loadtxt 函数从 TextIO 对象 e 中加载数据，指定数据类型为整数，指定 ndmin 为 2
        x = np.loadtxt(e, dtype=int, delimiter=',', ndmin=2)
        # 断言加载的数组形状为 (3, 1)
        assert_(x.shape == (3, 1))
        e.seek(0)
        # 使用 NumPy 的 loadtxt 函数从 TextIO 对象 e 中加载数据，指定数据类型为整数，指定 ndmin 为 1
        x = np.loadtxt(e, dtype=int, delimiter=',', ndmin=1)
        # 断言加载的数组形状为 (3,)
        assert_(x.shape == (3,))
        e.seek(0)
        # 使用 NumPy 的 loadtxt 函数从 TextIO 对象 e 中加载数据，指定数据类型为整数，指定 ndmin 为 0
        x = np.loadtxt(e, dtype=int, delimiter=',', ndmin=0)
        # 断言加载的数组形状为 (3,)
        assert_(x.shape == (3,))

        # 测试空文件情况下的 ndmin 参数
        # 使用 pytest 的 warns 函数捕获 UserWarning 异常，并检查是否包含特定的警告信息
        with pytest.warns(UserWarning, match="input contained no data"):
            f = TextIO()
            # 使用 NumPy 的 loadtxt 函数从空的 TextIO 对象 f 中加载数据，指定 ndmin 为 2
            assert_(np.loadtxt(f, ndmin=2).shape == (0, 1,))
            # 使用 NumPy 的 loadtxt 函数从空的 TextIO 对象 f 中加载数据，指定 ndmin 为 1
            assert_(np.loadtxt(f, ndmin=1).shape == (0,))

    # 测试函数：test_generator_source
    def test_generator_source(self):
        # 定义生成器函数 count，生成 0 到 9 的字符串
        def count():
            for i in range(10):
                yield "%d" % i
        # 使用 NumPy 的 loadtxt 函数从生成器 count 中加载数据
        res = np.loadtxt(count())
        # 断言加载的数据与预期的数组相等
        assert_array_equal(res, np.arange(10))

    # 测试函数：test_bad_line
    def test_bad_line(self):
        # 创建 TextIO 对象 c，并写入数据
        c = TextIO()
        c.write('1 2 3\n4 5 6\n2 3')
        c.seek(0)

        # 检查加载数据时是否引发 ValueError 异常，并检查异常信息是否包含特定的行号
        assert_raises_regex(ValueError, "3", np.loadtxt, c)
    def test_none_as_string(self):
        # 使用 TextIO 类创建一个文本输入输出对象
        c = TextIO()
        # 向文本对象写入字符串数据
        c.write('100,foo,200\n300,None,400')
        # 将写入的位置移动到文件开头
        c.seek(0)
        # 定义一个数据类型，包含整数和字节串字段
        dt = np.dtype([('x', int), ('a', 'S10'), ('y', int)])
        # 从文本对象中加载数据到 NumPy 数组，使用逗号分隔字段，没有注释行
        np.loadtxt(c, delimiter=',', dtype=dt, comments=None)  # 应当成功

    @pytest.mark.skipif(locale.getpreferredencoding() == 'ANSI_X3.4-1968',
                        reason="Wrong preferred encoding")
    def test_binary_load(self):
        # 定义一个包含特定字节串的字节对象
        butf8 = b"5,6,7,\xc3\x95scarscar\r\n15,2,3,hello\r\n"\
                b"20,2,3,\xc3\x95scar\r\n"
        # 将字节对象解码成 UTF-8 字符串，去除换行符，分割成行列表
        sutf8 = butf8.decode("UTF-8").replace("\r", "").splitlines()
        # 使用临时路径创建一个文件，并写入字节数据
        with temppath() as path:
            with open(path, "wb") as f:
                f.write(butf8)
            # 打开该文件以二进制读取模式，并将数据加载到 NumPy 数组中，使用 UTF-8 编码
            with open(path, "rb") as f:
                x = np.loadtxt(f, encoding="UTF-8", dtype=np.str_)
            # 断言加载的数组与预期的字符串数组相等
            assert_array_equal(x, sutf8)
            # 测试依赖于破损的 Latin1 转换
            with open(path, "rb") as f:
                # 将文件加载到 NumPy 数组中，使用 UTF-8 编码和字节串数据类型
                x = np.loadtxt(f, encoding="UTF-8", dtype="S")
            # 定义预期的字节串列表，并断言加载的数组与之相等
            x = [b'5,6,7,\xc3\x95scarscar', b'15,2,3,hello', b'20,2,3,\xc3\x95scar']
            assert_array_equal(x, np.array(x, dtype="S"))

    def test_max_rows(self):
        # 使用 TextIO 类创建一个文本输入输出对象
        c = TextIO()
        # 向文本对象写入字符串数据
        c.write('1,2,3,5\n4,5,7,8\n2,1,4,5')
        # 将写入的位置移动到文件开头
        c.seek(0)
        # 从文本对象中加载数据到 NumPy 数组，指定数据类型为整数，逗号分隔字段，最多加载一行数据
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       max_rows=1)
        # 定义预期的整数数组，并断言加载的数组与之相等
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

    def test_max_rows_with_skiprows(self):
        # 使用 TextIO 类创建一个文本输入输出对象
        c = TextIO()
        # 向文本对象写入字符串数据，包含注释行
        c.write('comments\n1,2,3,5\n4,5,7,8\n2,1,4,5')
        # 将写入的位置移动到文件开头
        c.seek(0)
        # 从文本对象中加载数据到 NumPy 数组，跳过第一行，指定数据类型为整数，逗号分隔字段，最多加载一行数据
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       skiprows=1, max_rows=1)
        # 定义预期的整数数组，并断言加载的数组与之相等
        a = np.array([1, 2, 3, 5], int)
        assert_array_equal(x, a)

        # 使用 TextIO 类创建一个文本输入输出对象
        c = TextIO()
        # 向文本对象写入字符串数据，包含注释行
        c.write('comment\n1,2,3,5\n4,5,7,8\n2,1,4,5')
        # 将写入的位置移动到文件开头
        c.seek(0)
        # 从文本对象中加载数据到 NumPy 数组，跳过第一行，最多加载两行数据，指定数据类型为整数，逗号分隔字段
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       skiprows=1, max_rows=2)
        # 定义预期的整数二维数组，并断言加载的数组与之相等
        a = np.array([[1, 2, 3, 5], [4, 5, 7, 8]], int)
        assert_array_equal(x, a)

    def test_max_rows_with_read_continuation(self):
        # 使用 TextIO 类创建一个文本输入输出对象
        c = TextIO()
        # 向文本对象写入字符串数据
        c.write('1,2,3,5\n4,5,7,8\n2,1,4,5')
        # 将写入的位置移动到文件开头
        c.seek(0)
        # 从文本对象中加载数据到 NumPy 数组，最多加载两行数据，指定数据类型为整数，逗号分隔字段
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       max_rows=2)
        # 定义预期的整数二维数组，并断言加载的数组与之相等
        a = np.array([[1, 2, 3, 5], [4, 5, 7, 8]], int)
        assert_array_equal(x, a)
        # 测试数据连续性加载
        x = np.loadtxt(c, dtype=int, delimiter=',')
        # 定义预期的整数数组，并断言加载的数组与之相等
        a = np.array([2, 1, 4, 5], int)
        assert_array_equal(x, a)

    def test_max_rows_larger(self):
        # 测试 max_rows 大于行数的情况
        c = TextIO()
        # 向文本对象写入字符串数据，包含注释行
        c.write('comment\n1,2,3,5\n4,5,7,8\n2,1,4,5')
        # 将写入的位置移动到文件开头
        c.seek(0)
        # 从文本对象中加载数据到 NumPy 数组，跳过第一行，最多加载六行数据，指定数据类型为整数，逗号分隔字段
        x = np.loadtxt(c, dtype=int, delimiter=',',
                       skiprows=1, max_rows=6)
        # 定义预期的整数二维数组，并断言加载的数组与之相等
        a = np.array([[1, 2, 3, 5], [4, 5, 7, 8], [2, 1, 4, 5]], int)
        assert_array_equal(x, a)
    # 使用 pytest 的参数化装饰器，为测试函数提供多组参数进行测试
    @pytest.mark.parametrize(["skip", "data"], [
            # 第一组参数化：skip=1，data为包含四个字符串的列表，每个字符串代表一行文本
            (1, ["ignored\n", "1,2\n", "\n", "3,4\n"]),
            # 第二组参数化：skip=1，data为包含四个字符串的列表，这些字符串是没有换行符的不完整行
            # 这些不完整的行应该被忽略
            (1, ["ignored", "1,2", "", "3,4"]),
            # 第三组参数化：skip=1，data为包含 StringIO 对象，其中有完整的和不完整的文本行
            (1, StringIO("ignored\n1,2\n\n3,4")),
            # 第四组参数化：skip=0，data为包含四个字符串的列表，每个字符串代表一行文本
            # 不忽略任何行
            (0, ["-1,0\n", "1,2\n", "\n", "3,4\n"]),
            # 第五组参数化：skip=0，data为包含四个字符串的列表，这些字符串是没有换行符的不完整行
            # 这些不完整的行应该被忽略
            (0, ["-1,0", "1,2", "", "3,4"]),
            # 第六组参数化：skip=0，data为包含 StringIO 对象，其中有完整的和不完整的文本行
            (0, StringIO("-1,0\n1,2\n\n3,4"))])
    # 定义测试函数，测试 np.loadtxt 函数对于空行和警告的处理
    def test_max_rows_empty_lines(self, skip, data):
        # 使用 pytest 的 warn 标记来捕获 UserWarning，确保抛出预期的警告信息
        with pytest.warns(UserWarning,
                    match=f"Input line 3.*max_rows={3-skip}"):
            # 调用 np.loadtxt 函数，加载数据并设置相关参数
            res = np.loadtxt(data, dtype=int, skiprows=skip, delimiter=",",
                             max_rows=3-skip)
            # 断言加载后的数据与预期的数组部分一致
            assert_array_equal(res, [[-1, 0], [1, 2], [3, 4]][skip:])

        # 如果 data 是 StringIO 对象，则将文件指针移回开头
        if isinstance(data, StringIO):
            data.seek(0)

        # 使用 warnings 模块捕获 UserWarning 异常
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # 使用 pytest 的 raises 函数确保 np.loadtxt 抛出 UserWarning 异常
            with pytest.raises(UserWarning):
                np.loadtxt(data, dtype=int, skiprows=skip, delimiter=",",
                           max_rows=3-skip)
class Testfromregex:
    # 测试从文本IO对象读取数据的类
    def test_record(self):
        # 创建一个文本IO对象
        c = TextIO()
        # 向文本IO对象写入数据
        c.write('1.312 foo\n1.534 bar\n4.444 qux')
        # 将文件指针移动到文件开头
        c.seek(0)

        # 定义数据类型，包括一个浮点数和一个长度为3的字符串
        dt = [('num', np.float64), ('val', 'S3')]
        # 使用正则表达式从文本IO对象c中读取数据并创建NumPy数组x
        x = np.fromregex(c, r"([0-9.]+)\s+(...)", dt)
        # 创建预期的NumPy数组a
        a = np.array([(1.312, 'foo'), (1.534, 'bar'), (4.444, 'qux')],
                     dtype=dt)
        # 断言数组x与数组a相等
        assert_array_equal(x, a)

    def test_record_2(self):
        # 创建一个文本IO对象
        c = TextIO()
        # 向文本IO对象写入数据
        c.write('1312 foo\n1534 bar\n4444 qux')
        # 将文件指针移动到文件开头
        c.seek(0)

        # 定义数据类型，包括一个32位整数和一个长度为3的字符串
        dt = [('num', np.int32), ('val', 'S3')]
        # 使用正则表达式从文本IO对象c中读取数据并创建NumPy数组x
        x = np.fromregex(c, r"(\d+)\s+(...)", dt)
        # 创建预期的NumPy数组a
        a = np.array([(1312, 'foo'), (1534, 'bar'), (4444, 'qux')],
                     dtype=dt)
        # 断言数组x与数组a相等
        assert_array_equal(x, a)

    def test_record_3(self):
        # 创建一个文本IO对象
        c = TextIO()
        # 向文本IO对象写入数据
        c.write('1312 foo\n1534 bar\n4444 qux')
        # 将文件指针移动到文件开头
        c.seek(0)

        # 定义数据类型，包括一个浮点数
        dt = [('num', np.float64)]
        # 使用正则表达式从文本IO对象c中读取数据并创建NumPy数组x
        x = np.fromregex(c, r"(\d+)\s+...", dt)
        # 创建预期的NumPy数组a
        a = np.array([(1312,), (1534,), (4444,)], dtype=dt)
        # 断言数组x与数组a相等
        assert_array_equal(x, a)

    @pytest.mark.parametrize("path_type", [str, Path])
    def test_record_unicode(self, path_type):
        # 定义UTF-8编码的特殊字符
        utf8 = b'\xcf\x96'
        # 使用临时路径创建一个文件，并将数据写入该文件
        with temppath() as str_path:
            path = path_type(str_path)
            with open(path, 'wb') as f:
                f.write(b'1.312 foo' + utf8 + b' \n1.534 bar\n4.444 qux')

            # 定义数据类型，包括一个浮点数和一个Unicode字符串
            dt = [('num', np.float64), ('val', 'U4')]
            # 使用正则表达式从文件path中读取数据并创建NumPy数组x
            x = np.fromregex(path, r"(?u)([0-9.]+)\s+(\w+)", dt, encoding='UTF-8')
            # 创建预期的NumPy数组a
            a = np.array([(1.312, 'foo' + utf8.decode('UTF-8')), (1.534, 'bar'),
                           (4.444, 'qux')], dtype=dt)
            # 断言数组x与数组a相等
            assert_array_equal(x, a)

            # 编译正则表达式对象，指定UNICODE标志
            regexp = re.compile(r"([0-9.]+)\s+(\w+)", re.UNICODE)
            # 使用正则表达式对象从文件path中读取数据并创建NumPy数组x
            x = np.fromregex(path, regexp, dt, encoding='UTF-8')
            # 断言数组x与数组a相等
            assert_array_equal(x, a)

    def test_compiled_bytes(self):
        # 编译字节字符串的正则表达式对象
        regexp = re.compile(b'(\\d)')
        # 创建一个字节IO对象，并写入数据
        c = BytesIO(b'123')
        # 定义数据类型，包括一个浮点数
        dt = [('num', np.float64)]
        # 创建预期的NumPy数组a
        a = np.array([1, 2, 3], dtype=dt)
        # 使用正则表达式从字节IO对象c中读取数据并创建NumPy数组x
        x = np.fromregex(c, regexp, dt)
        # 断言数组x与数组a相等
        assert_array_equal(x, a)

    def test_bad_dtype_not_structured(self):
        # 编译字节字符串的正则表达式对象
        regexp = re.compile(b'(\\d)')
        # 创建一个字节IO对象，并写入数据
        c = BytesIO(b'123')
        # 检查是否抛出TypeError异常，异常信息包含'structured datatype'
        with pytest.raises(TypeError, match='structured datatype'):
            np.fromregex(c, regexp, dtype=np.float64)
    def test_record(self):
        # Test w/ explicit dtype
        # 创建包含数据 '1 2\n3 4' 的文本流对象
        data = TextIO('1 2\n3 4')
        # 使用 np.genfromtxt 从文本流中读取数据，指定数据类型为 [('x', np.int32), ('y', np.int32)]
        test = np.genfromtxt(data, dtype=[('x', np.int32), ('y', np.int32)])
        # 创建控制数组，用于与测试结果比较
        control = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        # 断言测试结果与控制数组相等
        assert_equal(test, control)
        #
        # 创建包含数据 'M 64.0 75.0\nF 25.0 60.0' 的文本流对象
        data = TextIO('M 64.0 75.0\nF 25.0 60.0')
        # 创建描述器字典，指定字段名称和数据类型
        descriptor = {'names': ('gender', 'age', 'weight'),
                      'formats': ('S1', 'i4', 'f4')}
        # 创建控制数组，用于与测试结果比较
        control = np.array([('M', 64.0, 75.0), ('F', 25.0, 60.0)],
                           dtype=descriptor)
        # 使用 np.genfromtxt 从文本流中读取数据，指定数据类型为 descriptor
        test = np.genfromtxt(data, dtype=descriptor)
        # 断言测试结果与控制数组相等
        assert_equal(test, control)

    def test_array(self):
        # Test outputting a standard ndarray
        # 创建包含数据 '1 2\n3 4' 的文本流对象
        data = TextIO('1 2\n3 4')
        # 创建控制数组，用于与测试结果比较，数据类型为 int
        control = np.array([[1, 2], [3, 4]], dtype=int)
        # 使用 np.genfromtxt 从文本流中读取数据，指定数据类型为 int
        test = np.genfromtxt(data, dtype=int)
        # 断言测试结果与控制数组相等
        assert_array_equal(test, control)
        #
        # 将文本流指针移动到文件开头
        data.seek(0)
        # 创建控制数组，用于与测试结果比较，数据类型为 float
        control = np.array([[1, 2], [3, 4]], dtype=float)
        # 使用 np.loadtxt 从文本流中读取数据，指定数据类型为 float
        test = np.loadtxt(data, dtype=float)
        # 断言测试结果与控制数组相等
        assert_array_equal(test, control)

    def test_1D(self):
        # Test squeezing to 1D
        # 创建控制数组，用于与测试结果比较，数据类型为 int
        control = np.array([1, 2, 3, 4], int)
        #
        # 创建包含数据 '1\n2\n3\n4\n' 的文本流对象
        data = TextIO('1\n2\n3\n4\n')
        # 使用 np.genfromtxt 从文本流中读取数据，指定数据类型为 int
        test = np.genfromtxt(data, dtype=int)
        # 断言测试结果与控制数组相等
        assert_array_equal(test, control)
        #
        # 创建包含数据 '1,2,3,4\n' 的文本流对象
        data = TextIO('1,2,3,4\n')
        # 使用 np.genfromtxt 从文本流中读取数据，指定数据类型为 int，并设置分隔符为 ','
        test = np.genfromtxt(data, dtype=int, delimiter=',')
        # 断言测试结果与控制数组相等
        assert_array_equal(test, control)

    def test_comments(self):
        # Test the stripping of comments
        # 创建控制数组，用于与测试结果比较，数据类型为 int
        control = np.array([1, 2, 3, 5], int)
        # 创建包含 '# comment\n1,2,3,5\n' 数据的文本流对象
        data = TextIO('# comment\n1,2,3,5\n')
        # 使用 np.genfromtxt 从文本流中读取数据，指定数据类型为 int，分隔符为 ','，并移除以 '#' 开始的行
        test = np.genfromtxt(data, dtype=int, delimiter=',', comments='#')
        # 断言测试结果与控制数组相等
        assert_equal(test, control)
        # 创建包含 '1,2,3,5# comment\n' 数据的文本流对象
        data = TextIO('1,2,3,5# comment\n')
        # 使用 np.genfromtxt 从文本流中读取数据，指定数据类型为 int，分隔符为 ','，并移除以 '#' 开始的行
        test = np.genfromtxt(data, dtype=int, delimiter=',', comments='#')
        # 断言测试结果与控制数组相等
        assert_equal(test, control)

    def test_skiprows(self):
        # Test row skipping
        # 创建控制数组，用于与测试结果比较，数据类型为 int
        control = np.array([1, 2, 3, 5], int)
        kwargs = dict(dtype=int, delimiter=',')
        #
        # 创建包含 'comment\n1,2,3,5\n' 数据的文本流对象
        data = TextIO('comment\n1,2,3,5\n')
        # 使用 np.genfromtxt 从文本流中读取数据，跳过第一行，并使用 kwargs 指定的参数
        test = np.genfromtxt(data, skip_header=1, **kwargs)
        # 断言测试结果与控制数组相等
        assert_equal(test, control)
        #
        # 创建包含 '# comment\n1,2,3,5\n' 数据的文本流对象
        data = TextIO('# comment\n1,2,3,5\n')
        # 使用 np.loadtxt 从文本流中读取数据，跳过第一行，并使用 kwargs 指定的参数
        test = np.loadtxt(data, skiprows=1, **kwargs)
        # 断言测试结果与控制数组相等
        assert_equal(test, control)

    def test_skip_footer(self):
        # 创建包含行数与内容的列表，作为测试数据
        data = ["# %i" % i for i in range(1, 6)]
        data.append("A, B, C")
        data.extend(["%i,%3.1f,%03s" % (i, i, i) for i in range(51)])
        data[-1] = "99,99"
        kwargs = dict(delimiter=",", names=True, skip_header=5, skip_footer=10)
        # 使用 np.genfromtxt 从文本流中读取数据，根据 kwargs 指定的参数进行操作
        test = np.genfromtxt(TextIO("\n".join(data)), **kwargs)
        # 创建控制数组，用于与测试结果比较，指定字段类型为 float
        ctrl = np.array([("%f" % i, "%f" % i, "%f" % i) for i in range(41)],
                        dtype=[(_, float) for _ in "ABC"])
        # 断言测试结果与控制数组相等
        assert_equal(test, ctrl)
    def test_skip_footer_with_invalid(self):
        # 使用 suppress_warnings 上下文管理器，忽略特定警告
        with suppress_warnings() as sup:
            # 设置过滤条件，过滤掉 ConversionWarning 警告
            sup.filter(ConversionWarning)
            
            # 定义一个包含无效数据的字符串
            basestr = '1 1\n2 2\n3 3\n4 4\n5  \n6  \n7  \n'
            
            # 断言会引发 ValueError 异常，因为尾部太小无法清除所有无效值
            assert_raises(ValueError, np.genfromtxt, TextIO(basestr), skip_footer=1)
    
    #        except ValueError:
    #            pass
    
            # 使用 genfromtxt 读取数据，跳过最后一行（skip_footer=1），不引发异常
            a = np.genfromtxt(TextIO(basestr), skip_footer=1, invalid_raise=False)
            # 断言生成的数组与期望数组相等
            assert_equal(a, np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]]))
            
            # 再次使用 genfromtxt 读取数据，跳过最后三行（skip_footer=3）
            a = np.genfromtxt(TextIO(basestr), skip_footer=3)
            # 断言生成的数组与期望数组相等
            assert_equal(a, np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]]))
    
            # 重新定义包含无效数据的字符串
            basestr = '1 1\n2  \n3 3\n4 4\n5  \n6 6\n7 7\n'
            
            # 使用 genfromtxt 读取数据，跳过最后一行（skip_footer=1），不引发异常
            a = np.genfromtxt(TextIO(basestr), skip_footer=1, invalid_raise=False)
            # 断言生成的数组与期望数组相等
            assert_equal(a, np.array([[1., 1.], [3., 3.], [4., 4.], [6., 6.]]))
            
            # 使用 genfromtxt 读取数据，跳过最后三行（skip_footer=3），不引发异常
            a = np.genfromtxt(TextIO(basestr), skip_footer=3, invalid_raise=False)
            # 断言生成的数组与期望数组相等
            assert_equal(a, np.array([[1., 1.], [3., 3.], [4., 4.]]))

    def test_header(self):
        # 测试获取文件头部信息
        data = TextIO('gender age weight\nM 64.0 75.0\nF 25.0 60.0')
        
        # 使用 catch_warnings 上下文管理器捕获警告
        with warnings.catch_warnings(record=True) as w:
            # 始终警告，即使是可见的 DeprecationWarning
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            
            # 使用 genfromtxt 从数据中读取，指定 dtype 和 names=True，以字节编码读取
            test = np.genfromtxt(data, dtype=None, names=True, encoding='bytes')
            
            # 断言第一个警告的类型是 VisibleDeprecationWarning
            assert_(w[0].category is VisibleDeprecationWarning)
        
        # 控制数据的期望输出
        control = {'gender': np.array([b'M', b'F']),
                   'age': np.array([64.0, 25.0]),
                   'weight': np.array([75.0, 60.0])}
        
        # 分别断言生成的测试数据与控制数据的各个字段相等
        assert_equal(test['gender'], control['gender'])
        assert_equal(test['age'], control['age'])
        assert_equal(test['weight'], control['weight'])

    def test_auto_dtype(self):
        # 测试自动定义输出的 dtype
        data = TextIO('A 64 75.0 3+4j True\nBCD 25 60.0 5+6j False')
        
        # 使用 catch_warnings 上下文管理器捕获警告
        with warnings.catch_warnings(record=True) as w:
            # 始终警告，即使是可见的 DeprecationWarning
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            
            # 使用 genfromtxt 从数据中读取，自动推断 dtype，以字节编码读取
            test = np.genfromtxt(data, dtype=None, encoding='bytes')
            
            # 断言第一个警告的类型是 VisibleDeprecationWarning
            assert_(w[0].category is VisibleDeprecationWarning)
        
        # 控制数据的期望输出
        control = [np.array([b'A', b'BCD']),
                   np.array([64, 25]),
                   np.array([75.0, 60.0]),
                   np.array([3 + 4j, 5 + 6j]),
                   np.array([True, False]), ]
        
        # 断言生成的测试数据的字段名称与控制数据的字段名称相等
        assert_equal(test.dtype.names, ['f0', 'f1', 'f2', 'f3', 'f4'])
        
        # 遍历每个字段，分别断言生成的测试数据与控制数据的各个元素相等
        for (i, ctrl) in enumerate(control):
            assert_equal(test['f%i' % i], ctrl)
    def test_auto_dtype_uniform(self):
        # Tests whether the output dtype can be uniformized
        # 创建一个包含文本数据的TextIO对象
        data = TextIO('1 2 3 4\n5 6 7 8\n')
        # 使用numpy的genfromtxt函数从TextIO对象中读取数据，尝试推断数据类型
        test = np.genfromtxt(data, dtype=None)
        # 期望的数据结果
        control = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        # 断言生成的数据与期望的数据相等
        assert_equal(test, control)

    def test_fancy_dtype(self):
        # Check that a nested dtype isn't MIA
        # 创建一个包含文本数据的TextIO对象，数据使用逗号分隔
        data = TextIO('1,2,3.0\n4,5,6.0\n')
        # 定义一个复杂的dtype，包含嵌套结构
        fancydtype = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        # 使用numpy的genfromtxt函数从TextIO对象中读取数据，指定dtype为复杂结构
        test = np.genfromtxt(data, dtype=fancydtype, delimiter=',')
        # 期望的数据结果，包含指定的复杂结构
        control = np.array([(1, (2, 3.0)), (4, (5, 6.0))], dtype=fancydtype)
        # 断言生成的数据与期望的数据相等
        assert_equal(test, control)

    def test_names_overwrite(self):
        # Test overwriting the names of the dtype
        # 定义一个描述符，指定字段名和对应的数据类型
        descriptor = {'names': ('g', 'a', 'w'),
                      'formats': ('S1', 'i4', 'f4')}
        # 创建一个包含二进制数据的TextIO对象
        data = TextIO(b'M 64.0 75.0\nF 25.0 60.0')
        # 指定字段名列表
        names = ('gender', 'age', 'weight')
        # 使用numpy的genfromtxt函数从TextIO对象中读取数据，指定dtype和字段名
        test = np.genfromtxt(data, dtype=descriptor, names=names)
        # 更新描述符中的字段名
        descriptor['names'] = names
        # 期望的数据结果，使用更新后的字段名描述数据
        control = np.array([('M', 64.0, 75.0),
                            ('F', 25.0, 60.0)], dtype=descriptor)
        # 断言生成的数据与期望的数据相等
        assert_equal(test, control)

    def test_bad_fname(self):
        # 使用pytest检查传入的fname参数是否为字符串类型，预期会抛出TypeError异常
        with pytest.raises(TypeError, match='fname must be a string,'):
            np.genfromtxt(123)

    def test_commented_header(self):
        # Check that names can be retrieved even if the line is commented out.
        # 创建一个包含空白和注释文本的TextIO对象
        data = TextIO("""
    def test_names_and_comments_hash(self):
        # 测试当数据中包含 # 号作为注释时的情况
        data = TextIO(b"""
# gender age weight
M   21  72.100000
F   35  58.330000
M   33  21.99
        """)
        # 捕获警告并验证是否触发了 VisibleDeprecationWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            # 从文本数据中读取并解析为结构化数组
            test = np.genfromtxt(data, names=True, dtype=None,
                                 encoding="bytes")
            # 断言是否触发了 VisibleDeprecationWarning
            assert_(w[0].category is VisibleDeprecationWarning)
        # 预期的控制数组
        ctrl = np.array([('M', 21, 72.1), ('F', 35, 58.33), ('M', 33, 21.99)],
                        dtype=[('gender', '|S1'), ('age', int), ('weight', float)])
        # 断言解析结果与预期控制数组是否相等
        assert_equal(test, ctrl)

    def test_names_and_comments_none(self):
        # 测试当 names 为 True 但 comments 为 None 时的情况 (gh-10780)
        data = TextIO('col1 col2\n 1 2\n 3 4')
        # 从文本数据中读取并解析为结构化数组，要求字段名为 True，注释符号为 None
        test = np.genfromtxt(data, dtype=(int, int), comments=None, names=True)
        # 预期的控制数组
        control = np.array([(1, 2), (3, 4)], dtype=[('col1', int), ('col2', int)])
        # 断言解析结果与预期控制数组是否相等
        assert_equal(test, control)

    def test_file_is_closed_on_error(self):
        # 测试当出现错误时文件是否正确关闭 (gh-13200)
        with tempdir() as tmpdir:
            fpath = os.path.join(tmpdir, "test.csv")
            with open(fpath, "wb") as f:
                f.write('\N{GREEK PI SYMBOL}'.encode())

            # ResourceWarnings 是由析构函数触发的，因此不会通过常规错误传播检测到
            with assert_no_warnings():
                # 使用 ASCII 编码尝试读取文件，预期会引发 UnicodeDecodeError
                with pytest.raises(UnicodeDecodeError):
                    np.genfromtxt(fpath, encoding="ascii")

    def test_autonames_and_usecols(self):
        # 测试 names 和 usecols 的情况
        data = TextIO('A B C D\n aaaa 121 45 9.1')
        # 捕获警告并验证是否触发了 VisibleDeprecationWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            # 从文本数据中读取并解析为结构化数组，使用指定的列并自动分配字段名
            test = np.genfromtxt(data, usecols=('A', 'C', 'D'),
                                names=True, dtype=None, encoding="bytes")
            # 断言是否触发了 VisibleDeprecationWarning
            assert_(w[0].category is VisibleDeprecationWarning)
        # 预期的控制数组
        control = np.array(('aaaa', 45, 9.1),
                           dtype=[('A', '|S4'), ('C', int), ('D', float)])
        # 断言解析结果与预期控制数组是否相等
        assert_equal(test, control)
    def test_converters_with_usecols(self):
        # 测试自定义转换器和usecols的组合
        # 创建包含指定数据的文本流对象
        data = TextIO('1,2,3,,5\n6,7,8,9,10\n')
        # 从文本流中读取数据，并应用自定义转换器和列过滤器
        test = np.genfromtxt(data, dtype=int, delimiter=',',
                            converters={3: lambda s: int(s or -999)},
                            usecols=(1, 3,))
        # 生成预期的控制数组以进行断言比较
        control = np.array([[2, -999], [7, 9]], int)
        # 断言测试结果与控制数组相等
        assert_equal(test, control)

    def test_converters_with_usecols_and_names(self):
        # 测试名称和usecols
        # 创建包含指定数据的文本流对象
        data = TextIO('A B C D\n aaaa 121 45 9.1')
        # 使用警告记录来捕获警告信息
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            # 从文本流中读取数据，指定列和名称，应用转换器
            test = np.genfromtxt(data, usecols=('A', 'C', 'D'), names=True,
                                dtype=None, encoding="bytes",
                                converters={'C': lambda s: 2 * int(s)})
            # 断言捕获到的第一个警告是可见性过时警告
            assert_(w[0].category is VisibleDeprecationWarning)
        # 生成预期的控制数组以进行断言比较
        control = np.array(('aaaa', 90, 9.1),
                           dtype=[('A', '|S4'), ('C', int), ('D', float)])
        # 断言测试结果与控制数组相等
        assert_equal(test, control)

    def test_converters_cornercases(self):
        # 测试日期时间转换
        # 创建转换器字典，将日期字符串转换为日期时间对象
        converter = {
            'date': lambda s: strptime(s, '%Y-%m-%d %H:%M:%SZ')}
        # 创建包含指定数据的文本流对象
        data = TextIO('2009-02-03 12:00:00Z, 72214.0')
        # 从文本流中读取数据，指定分隔符和转换器
        test = np.genfromtxt(data, delimiter=',', dtype=None,
                            names=['date', 'stid'], converters=converter)
        # 生成预期的控制数组以进行断言比较
        control = np.array((datetime(2009, 2, 3), 72214.),
                           dtype=[('date', np.object_), ('stid', float)])
        # 断言测试结果与控制数组相等
        assert_equal(test, control)

    def test_converters_cornercases2(self):
        # 测试日期时间64位转换
        # 创建转换器字典，将日期字符串转换为numpy的datetime64对象
        converter = {
            'date': lambda s: np.datetime64(strptime(s, '%Y-%m-%d %H:%M:%SZ'))}
        # 创建包含指定数据的文本流对象
        data = TextIO('2009-02-03 12:00:00Z, 72214.0')
        # 从文本流中读取数据，指定分隔符和转换器
        test = np.genfromtxt(data, delimiter=',', dtype=None,
                            names=['date', 'stid'], converters=converter)
        # 生成预期的控制数组以进行断言比较
        control = np.array((datetime(2009, 2, 3), 72214.),
                           dtype=[('date', 'datetime64[us]'), ('stid', float)])
        # 断言测试结果与控制数组相等
        assert_equal(test, control)

    def test_unused_converter(self):
        # 测试未使用的转换器是否被忽略
        # 创建包含指定数据的文本流对象
        data = TextIO("1 21\n  3 42\n")
        # 从文本流中读取数据，指定列过滤器和转换器
        test = np.genfromtxt(data, usecols=(1,),
                            converters={0: lambda s: int(s, 16)})
        # 断言测试结果与预期数组相等
        assert_equal(test, [21, 42])
        #
        data.seek(0)
        # 从文本流中读取数据，指定列过滤器和转换器
        test = np.genfromtxt(data, usecols=(1,),
                            converters={1: lambda s: int(s, 16)})
        # 断言测试结果与预期数组相等
        assert_equal(test, [33, 66])
    def test_invalid_converter(self):
        # 定义一个函数，用于将输入的字符串转换为浮点数，处理包含'r'的情况和不包含'r'的情况
        strip_rand = lambda x: float((b'r' in x.lower() and x.split()[-1]) or
                                     (b'r' not in x.lower() and x.strip() or 0.0))
        # 定义一个函数，用于将输入的字符串转换为浮点数，处理包含'%'的情况和不包含'%'的情况
        strip_per = lambda x: float((b'%' in x.lower() and x.split()[0]) or
                                    (b'%' not in x.lower() and x.strip() or 0.0))
        # 创建一个TextIO对象，包含多行文本数据
        s = TextIO("D01N01,10/1/2003 ,1 %,R 75,400,600\r\n"
                   "L24U05,12/5/2003, 2 %,1,300, 150.5\r\n"
                   "D02N03,10/10/2004,R 1,,7,145.55")
        # 定义关键字参数字典
        kwargs = dict(
            converters={2: strip_per, 3: strip_rand}, delimiter=",",
            dtype=None, encoding="bytes")
        # 断言调用np.genfromtxt会抛出ConverterError异常
        assert_raises(ConverterError, np.genfromtxt, s, **kwargs)

    def test_tricky_converter_bug1666(self):
        # 测试一些边缘情况
        s = TextIO('q1,2\nq3,4')
        # 定义一个lambda函数用作转换器，将字符串转换为浮点数
        cnv = lambda s: float(s[1:])
        # 调用np.genfromtxt解析数据，使用逗号作为分隔符，指定第一列使用cnv函数进行转换
        test = np.genfromtxt(s, delimiter=',', converters={0: cnv})
        # 定义预期的控制数组
        control = np.array([[1., 2.], [3., 4.]])
        # 断言test数组与control数组相等
        assert_equal(test, control)

    def test_dtype_with_converters(self):
        # 定义一个字符串，包含数据
        dstr = "2009; 23; 46"
        # 使用np.genfromtxt解析数据，分号作为分隔符，指定第一列使用bytes函数进行转换
        test = np.genfromtxt(TextIO(dstr,),
                            delimiter=";", dtype=float, converters={0: bytes})
        # 定义预期的控制数组，指定dtype为每列的数据类型
        control = np.array([('2009', 23., 46)],
                           dtype=[('f0', '|S4'), ('f1', float), ('f2', float)])
        # 断言test数组与control数组相等
        assert_equal(test, control)
        # 再次调用np.genfromtxt解析数据，指定第一列使用float函数进行转换
        test = np.genfromtxt(TextIO(dstr,),
                            delimiter=";", dtype=float, converters={0: float})
        # 定义预期的控制数组，只包含浮点数
        control = np.array([2009., 23., 46],)
        # 断言test数组与control数组相等
        assert_equal(test, control)

    @pytest.mark.filterwarnings("ignore:.*recfromcsv.*:DeprecationWarning")
    def test_dtype_with_converters_and_usecols(self):
        # 定义一个包含数据的字符串
        dstr = "1,5,-1,1:1\n2,8,-1,1:n\n3,3,-2,m:n\n"
        # 定义一个映射，将字符串映射到整数
        dmap = {'1:1':0, '1:n':1, 'm:1':2, 'm:n':3}
        # 定义一个dtype，指定每列的名称和数据类型
        dtyp = [('e1','i4'),('e2','i4'),('e3','i2'),('n', 'i1')]
        # 定义转换器字典，将每列数据按照指定的转换函数进行转换
        conv = {0: int, 1: int, 2: int, 3: lambda r: dmap[r.decode()]}
        # 调用recfromcsv解析数据，使用逗号作为分隔符，使用conv字典进行数据转换
        test = recfromcsv(TextIO(dstr,), dtype=dtyp, delimiter=',',
                          names=None, converters=conv, encoding="bytes")
        # 定义预期的控制数组，生成一个结构化数组
        control = np.rec.array([(1,5,-1,0), (2,8,-1,1), (3,3,-2,3)], dtype=dtyp)
        # 断言test数组与control数组相等
        assert_equal(test, control)
        # 重新定义dtype，只包含部分列，并调用recfromcsv解析数据
        dtyp = [('e1', 'i4'), ('e2', 'i4'), ('n', 'i1')]
        test = recfromcsv(TextIO(dstr,), dtype=dtyp, delimiter=',',
                          usecols=(0, 1, 3), names=None, converters=conv,
                          encoding="bytes")
        # 定义预期的控制数组，生成一个结构化数组
        control = np.rec.array([(1,5,0), (2,8,1), (3,3,3)], dtype=dtyp)
        # 断言test数组与control数组相等
        assert_equal(test, control)
    def test_dtype_with_object(self):
        # Test using an explicit dtype with an object
        data = """ 1; 2001-01-01
                   2; 2002-01-31 """
        ndtype = [('idx', int), ('code', object)]
        func = lambda s: strptime(s.strip(), "%Y-%m-%d")
        converters = {1: func}
        # 从文本数据创建结构化数组，指定字段数据类型和转换器
        test = np.genfromtxt(TextIO(data), delimiter=";", dtype=ndtype,
                             converters=converters)
        # 创建控制用的数组，以验证结果
        control = np.array(
            [(1, datetime(2001, 1, 1)), (2, datetime(2002, 1, 31))],
            dtype=ndtype)
        # 断言测试结果与控制结果相等
        assert_equal(test, control)

        ndtype = [('nest', [('idx', int), ('code', object)])]
        # 检测嵌套字段的情况是否抛出预期的异常
        with assert_raises_regex(NotImplementedError,
                                 'Nested fields.* not supported.*'):
            test = np.genfromtxt(TextIO(data), delimiter=";",
                                 dtype=ndtype, converters=converters)

        # 嵌套字段为空时也不支持，检测是否抛出预期的异常
        ndtype = [('idx', int), ('code', object), ('nest', [])]
        with assert_raises_regex(NotImplementedError,
                                 'Nested fields.* not supported.*'):
            test = np.genfromtxt(TextIO(data), delimiter=";",
                                 dtype=ndtype, converters=converters)

    def test_dtype_with_object_no_converter(self):
        # Object without a converter uses bytes:
        # 测试未使用转换器时对象使用字节流的情况
        parsed = np.genfromtxt(TextIO("1"), dtype=object)
        assert parsed[()] == b"1"
        parsed = np.genfromtxt(TextIO("string"), dtype=object)
        assert parsed[()] == b"string"

    def test_userconverters_with_explicit_dtype(self):
        # Test user_converters w/ explicit (standard) dtype
        data = TextIO('skip,skip,2001-01-01,1.0,skip')
        # 使用用户定义的转换器解析数据，验证结果
        test = np.genfromtxt(data, delimiter=",", names=None, dtype=float,
                             usecols=(2, 3), converters={2: bytes})
        control = np.array([('2001-01-01', 1.)],
                           dtype=[('', '|S10'), ('', float)])
        assert_equal(test, control)

    def test_utf8_userconverters_with_explicit_dtype(self):
        utf8 = b'\xcf\x96'
        with temppath() as path:
            with open(path, 'wb') as f:
                f.write(b'skip,skip,2001-01-01' + utf8 + b',1.0,skip')
            # 使用 UTF-8 编码解析包含 UTF-8 数据的文件
            test = np.genfromtxt(path, delimiter=",", names=None, dtype=float,
                                 usecols=(2, 3), converters={2: str},
                                 encoding='UTF-8')
        control = np.array([('2001-01-01' + utf8.decode('UTF-8'), 1.)],
                           dtype=[('', '|U11'), ('', float)])
        assert_equal(test, control)

    def test_spacedelimiter(self):
        # Test space delimiter
        data = TextIO("1  2  3  4   5\n6  7  8  9  10")
        # 使用空格作为分隔符解析数据
        test = np.genfromtxt(data)
        control = np.array([[1., 2., 3., 4., 5.],
                            [6., 7., 8., 9., 10.]])
        assert_equal(test, control)
    def test_integer_delimiter(self):
        # 使用整数作为分隔符进行测试
        data = "  1  2  3\n  4  5 67\n890123  4"
        # 使用 np.genfromtxt 从 TextIO 对象读取数据，以 3 作为分隔符
        test = np.genfromtxt(TextIO(data), delimiter=3)
        # 预期的结果数组
        control = np.array([[1, 2, 3], [4, 5, 67], [890, 123, 4]])
        # 断言测试结果与预期结果相等
        assert_equal(test, control)

    def test_missing(self):
        data = TextIO('1,2,3,,5\n')
        # 使用 np.genfromtxt 从 TextIO 对象读取数据，指定数据类型为整数，使用 ',' 作为分隔符，
        # 并使用转换器来处理第 3 列的缺失值
        test = np.genfromtxt(data, dtype=int, delimiter=',',
                            converters={3: lambda s: int(s or - 999)})
        # 预期的结果数组
        control = np.array([1, 2, 3, -999, 5], int)
        # 断言测试结果与预期结果相等
        assert_equal(test, control)

    def test_missing_with_tabs(self):
        # 使用制表符作为分隔符进行测试
        txt = "1\t2\t3\n\t2\t\n1\t\t3"
        # 使用 np.genfromtxt 从 TextIO 对象读取数据，启用掩码，并且不指定数据类型
        test = np.genfromtxt(TextIO(txt), delimiter="\t",
                             usemask=True,)
        # 预期的数据数组和掩码数组
        ctrl_d = np.array([(1, 2, 3), (np.nan, 2, np.nan), (1, np.nan, 3)],)
        ctrl_m = np.array([(0, 0, 0), (1, 0, 1), (0, 1, 0)], dtype=bool)
        # 断言测试结果的数据数组和掩码数组与预期结果相等
        assert_equal(test.data, ctrl_d)
        assert_equal(test.mask, ctrl_m)

    def test_usecols(self):
        # 测试列的选择
        # 选择第一列
        control = np.array([[1, 2], [3, 4]], float)
        data = TextIO()
        # 将控制数据写入 TextIO 对象
        np.savetxt(data, control)
        data.seek(0)
        # 使用 np.genfromtxt 从 TextIO 对象读取数据，指定数据类型为浮点数，并选择使用列 (1,)
        test = np.genfromtxt(data, dtype=float, usecols=(1,))
        # 断言测试结果与预期结果相等
        assert_equal(test, control[:, 1])
        #
        control = np.array([[1, 2, 3], [3, 4, 5]], float)
        data = TextIO()
        np.savetxt(data, control)
        data.seek(0)
        # 使用 np.genfromtxt 从 TextIO 对象读取数据，指定数据类型为浮点数，并选择使用列 (1, 2)
        test = np.genfromtxt(data, dtype=float, usecols=(1, 2))
        # 断言测试结果与预期结果相等
        assert_equal(test, control[:, 1:])
        # 使用数组而非元组进行测试
        data.seek(0)
        test = np.genfromtxt(data, dtype=float, usecols=np.array([1, 2]))
        # 断言测试结果与预期结果相等
        assert_equal(test, control[:, 1:])

    def test_usecols_as_css(self):
        # 使用逗号分隔的字符串指定 usecols 进行测试
        data = "1 2 3\n4 5 6"
        # 使用 np.genfromtxt 从 TextIO 对象读取数据，指定列名为 'a, b, c'，并选择使用列 'a, c'
        test = np.genfromtxt(TextIO(data),
                             names="a, b, c", usecols="a, c")
        # 预期的结果数组
        ctrl = np.array([(1, 3), (4, 6)], dtype=[(_, float) for _ in "ac"])
        # 断言测试结果与预期结果相等
        assert_equal(test, ctrl)

    def test_usecols_with_structured_dtype(self):
        # 使用显式结构化数据类型进行 usecols 测试
        data = TextIO("JOE 70.1 25.3\nBOB 60.5 27.9")
        names = ['stid', 'temp']
        dtypes = ['S4', 'f8']
        # 使用 np.genfromtxt 从 TextIO 对象读取数据，指定列使用 (0, 2)，并使用给定的结构化数据类型
        test = np.genfromtxt(
            data, usecols=(0, 2), dtype=list(zip(names, dtypes)))
        # 断言测试结果的 'stid' 列和 'temp' 列与预期结果相等
        assert_equal(test['stid'], [b"JOE", b"BOB"])
        assert_equal(test['temp'], [25.3, 27.9])

    def test_usecols_with_integer(self):
        # 使用整数作为 usecols 进行测试
        test = np.genfromtxt(TextIO(b"1 2 3\n4 5 6"), usecols=0)
        # 断言测试结果与预期结果相等
        assert_equal(test, np.array([1., 4.]))
    def test_usecols_with_named_columns(self):
        # Test usecols with named columns
        ctrl = np.array([(1, 3), (4, 6)], dtype=[('a', float), ('c', float)])
        data = "1 2 3\n4 5 6"
        kwargs = dict(names="a, b, c")
        # 使用 genfromtxt 从文本数据创建 NumPy 数组，仅选择指定列 ('a' 和 'c')
        test = np.genfromtxt(TextIO(data), usecols=(0, -1), **kwargs)
        assert_equal(test, ctrl)
        # 再次使用 genfromtxt，但这次使用列名 ('a' 和 'c') 替代索引位置
        test = np.genfromtxt(TextIO(data),
                             usecols=('a', 'c'), **kwargs)
        assert_equal(test, ctrl)

    def test_empty_file(self):
        # Test that an empty file raises the proper warning.
        with suppress_warnings() as sup:
            sup.filter(message="genfromtxt: Empty input file:")
            data = TextIO()
            # 读取空文本时，验证 genfromtxt 是否返回空数组
            test = np.genfromtxt(data)
            assert_equal(test, np.array([]))

            # 当 skip_header > 0 时，再次验证空文本情况
            test = np.genfromtxt(data, skip_header=1)
            assert_equal(test, np.array([]))

    def test_fancy_dtype_alt(self):
        # Check that a nested dtype isn't MIA
        data = TextIO('1,2,3.0\n4,5,6.0\n')
        # 定义一个复杂的 dtype，包含嵌套的字段
        fancydtype = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        # 使用 genfromtxt 读取数据，并验证是否正确创建了复杂 dtype 的数组
        test = np.genfromtxt(data, dtype=fancydtype, delimiter=',', usemask=True)
        control = ma.array([(1, (2, 3.0)), (4, (5, 6.0))], dtype=fancydtype)
        assert_equal(test, control)

    def test_shaped_dtype(self):
        c = TextIO("aaaa  1.0  8.0  1 2 3 4 5 6")
        # 定义一个结构化 dtype，包含一个形状为 (2, 3) 的数组字段
        dt = np.dtype([('name', 'S4'), ('x', float), ('y', float),
                       ('block', int, (2, 3))])
        # 使用 genfromtxt 读取数据，并验证是否正确创建了结构化 dtype 的数组
        x = np.genfromtxt(c, dtype=dt)
        a = np.array([('aaaa', 1.0, 8.0, [[1, 2, 3], [4, 5, 6]])],
                     dtype=dt)
        assert_array_equal(x, a)

    def test_withmissing(self):
        data = TextIO('A,B\n0,1\n2,N/A')
        kwargs = dict(delimiter=",", missing_values="N/A", names=True)
        # 使用 genfromtxt 读取数据，处理缺失值并创建带有掩码的结构化数组
        test = np.genfromtxt(data, dtype=None, usemask=True, **kwargs)
        control = ma.array([(0, 1), (2, -1)],
                           mask=[(False, False), (False, True)],
                           dtype=[('A', int), ('B', int)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        #
        data.seek(0)
        # 再次使用 genfromtxt，处理不同数据类型的列，并生成带有掩码的结构化数组
        test = np.genfromtxt(data, usemask=True, **kwargs)
        control = ma.array([(0, 1), (2, -1)],
                           mask=[(False, False), (False, True)],
                           dtype=[('A', float), ('B', float)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
    def test_user_missing_values(self):
        # 创建包含缺失值的测试数据字符串
        data = "A, B, C\n0, 0., 0j\n1, N/A, 1j\n-9, 2.2, N/A\n3, -99, 3j"
        # 设置基础参数字典
        basekwargs = dict(dtype=None, delimiter=",", names=True,)
        # 设置数据类型元组
        mdtype = [('A', int), ('B', float), ('C', complex)]
        
        # 使用 np.genfromtxt 从数据流中读取数据，并设置 N/A 为缺失值
        test = np.genfromtxt(TextIO(data), missing_values="N/A",
                            **basekwargs)
        # 创建控制组数组，用于比较结果
        control = ma.array([(0, 0.0, 0j), (1, -999, 1j),
                            (-9, 2.2, -999j), (3, -99, 3j)],
                           mask=[(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)],
                           dtype=mdtype)
        # 断言测试结果与控制组相等
        assert_equal(test, control)
        
        # 更新 basekwargs 中的 dtype 为 mdtype
        basekwargs['dtype'] = mdtype
        # 使用 np.genfromtxt 从数据流中读取数据，并设置特定缺失值和掩码
        test = np.genfromtxt(TextIO(data),
                            missing_values={0: -9, 1: -99, 2: -999j}, usemask=True, **basekwargs)
        # 更新控制组数组，用于比较结果
        control = ma.array([(0, 0.0, 0j), (1, -999, 1j),
                            (-9, 2.2, -999j), (3, -99, 3j)],
                           mask=[(0, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 0)],
                           dtype=mdtype)
        # 断言测试结果与控制组相等
        assert_equal(test, control)
        
        # 使用 np.genfromtxt 从数据流中读取数据，并设置不同的缺失值和掩码
        test = np.genfromtxt(TextIO(data),
                            missing_values={0: -9, 'B': -99, 'C': -999j},
                            usemask=True,
                            **basekwargs)
        # 更新控制组数组，用于比较结果
        control = ma.array([(0, 0.0, 0j), (1, -999, 1j),
                            (-9, 2.2, -999j), (3, -99, 3j)],
                           mask=[(0, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 0)],
                           dtype=mdtype)
        # 断言测试结果与控制组相等
        assert_equal(test, control)

    def test_user_filling_values(self):
        # 测试包含缺失值和填充值的情况
        ctrl = np.array([(0, 3), (4, -999)], dtype=[('a', int), ('b', int)])
        # 创建包含缺失值的测试数据字符串
        data = "N/A, 2, 3\n4, ,???"
        # 设置关键字参数字典
        kwargs = dict(delimiter=",",
                      dtype=int,
                      names="a,b,c",
                      missing_values={0: "N/A", 'b': " ", 2: "???"},
                      filling_values={0: 0, 'b': 0, 2: -999})
        
        # 使用 np.genfromtxt 从数据流中读取数据，并设置缺失值和填充值
        test = np.genfromtxt(TextIO(data), **kwargs)
        # 创建控制组数组，用于比较结果
        ctrl = np.array([(0, 2, 3), (4, 0, -999)],
                        dtype=[(_, int) for _ in "abc"])
        # 断言测试结果与控制组相等
        assert_equal(test, ctrl)
        
        # 使用 np.genfromtxt 从数据流中读取数据，只选择部分列，并设置缺失值和填充值
        test = np.genfromtxt(TextIO(data), usecols=(0, -1), **kwargs)
        # 创建控制组数组，用于比较结果
        ctrl = np.array([(0, 3), (4, -999)], dtype=[(_, int) for _ in "ac"])
        # 断言测试结果与控制组相等
        assert_equal(test, ctrl)

        # 创建另一个包含缺失值的测试数据字符串
        data2 = "1,2,*,4\n5,*,7,8\n"
        # 使用 np.genfromtxt 从数据流中读取数据，并设置特定缺失值和填充值
        test = np.genfromtxt(TextIO(data2), delimiter=',', dtype=int,
                             missing_values="*", filling_values=0)
        # 创建控制组数组，用于比较结果
        ctrl = np.array([[1, 2, 0, 4], [5, 0, 7, 8]])
        # 断言测试结果与控制组相等
        assert_equal(test, ctrl)
        # 使用 np.genfromtxt 从数据流中读取数据，并设置特定缺失值和填充值
        test = np.genfromtxt(TextIO(data2), delimiter=',', dtype=int,
                             missing_values="*", filling_values=-1)
        # 创建控制组数组，用于比较结果
        ctrl = np.array([[1, 2, -1, 4], [5, -1, 7, 8]])
        # 断言测试结果与控制组相等
        assert_equal(test, ctrl)
    def test_withmissing_float(self):
        # 创建一个文本输入对象，包含特定的数据
        data = TextIO('A,B\n0,1.5\n2,-999.00')
        # 使用 np.genfromtxt 从文本输入对象中读取数据，指定参数如下：
        # dtype=None 表示数据类型为自动推断
        # delimiter=',' 指定字段分隔符为逗号
        # missing_values='-999.0' 指定缺失值为 '-999.0'
        # names=True 表示第一行包含字段名
        # usemask=True 表示使用掩码数组来标记缺失值
        test = np.genfromtxt(data, dtype=None, delimiter=',',
                            missing_values='-999.0', names=True, usemask=True)
        # 创建一个控制数据的掩码数组，表示缺失值情况
        control = ma.array([(0, 1.5), (2, -1.)],
                           mask=[(False, False), (False, True)],
                           dtype=[('A', int), ('B', float)])
        # 断言测试结果与控制数据相等
        assert_equal(test, control)
        # 断言测试结果的掩码数组与控制数据的掩码数组相等
        assert_equal(test.mask, control.mask)

    def test_with_masked_column_uniform(self):
        # 测试具有掩码列的情况
        data = TextIO('1 2 3\n4 5 6\n')
        # 使用 np.genfromtxt 从文本输入对象中读取数据，指定参数如下：
        # dtype=None 表示数据类型为自动推断
        # missing_values='2,5' 指定多个缺失值为 '2' 和 '5'
        # usemask=True 表示使用掩码数组来标记缺失值
        test = np.genfromtxt(data, dtype=None,
                             missing_values='2,5', usemask=True)
        # 创建一个控制数据的掩码数组，表示缺失值情况
        control = ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 1, 0], [0, 1, 0]])
        # 断言测试结果与控制数据相等
        assert_equal(test, control)

    def test_with_masked_column_various(self):
        # 测试具有掩码列的情况
        data = TextIO('True 2 3\nFalse 5 6\n')
        # 使用 np.genfromtxt 从文本输入对象中读取数据，指定参数如下：
        # dtype=None 表示数据类型为自动推断
        # missing_values='2,5' 指定多个缺失值为 '2' 和 '5'
        # usemask=True 表示使用掩码数组来标记缺失值
        test = np.genfromtxt(data, dtype=None,
                             missing_values='2,5', usemask=True)
        # 创建一个控制数据的掩码数组，表示缺失值情况和字段类型
        control = ma.array([(1, 2, 3), (0, 5, 6)],
                           mask=[(0, 1, 0), (0, 1, 0)],
                           dtype=[('f0', bool), ('f1', bool), ('f2', int)])
        # 断言测试结果与控制数据相等
        assert_equal(test, control)

    def test_invalid_raise(self):
        # 测试无效的数据引发异常的情况
        data = ["1, 1, 1, 1, 1"] * 50
        for i in range(5):
            data[10 * i] = "2, 2, 2, 2 2"
        data.insert(0, "a, b, c, d, e")
        # 创建一个包含指定数据的文本输入对象
        mdata = TextIO("\n".join(data))

        # 定义关键字参数字典
        kwargs = dict(delimiter=",", dtype=None, names=True)
        # 定义一个函数 f，该函数调用 np.genfromtxt 从文本输入对象中读取数据
        # invalid_raise=False 表示遇到无效数据时不引发异常
        def f():
            return np.genfromtxt(mdata, invalid_raise=False, **kwargs)
        # 断言函数 f 会引发 ConversionWarning 警告
        mtest = assert_warns(ConversionWarning, f)
        # 断言测试结果长度为 45
        assert_equal(len(mtest), 45)
        # 断言测试结果与控制数据相等，数据类型为每个字段都是整数 'abcde'
        assert_equal(mtest, np.ones(45, dtype=[(_, int) for _ in 'abcde']))
        #
        mdata.seek(0)
        # 断言调用 np.genfromtxt 会引发 ValueError 异常
        assert_raises(ValueError, np.genfromtxt, mdata,
                      delimiter=",", names=True)

    def test_invalid_raise_with_usecols(self):
        # 测试使用 usecols 参数时，无效的数据引发异常的情况
        data = ["1, 1, 1, 1, 1"] * 50
        for i in range(5):
            data[10 * i] = "2, 2, 2, 2 2"
        data.insert(0, "a, b, c, d, e")
        # 创建一个包含指定数据的文本输入对象
        mdata = TextIO("\n".join(data))

        # 定义关键字参数字典
        kwargs = dict(delimiter=",", dtype=None, names=True,
                      invalid_raise=False)
        # 定义一个函数 f，该函数调用 np.genfromtxt 从文本输入对象中读取数据
        # usecols=(0, 4) 表示仅使用第 0 和第 4 列数据
        def f():
            return np.genfromtxt(mdata, usecols=(0, 4), **kwargs)
        # 断言函数 f 会引发 ConversionWarning 警告
        mtest = assert_warns(ConversionWarning, f)
        # 断言测试结果长度为 45
        assert_equal(len(mtest), 45)
        # 断言测试结果与控制数据相等，数据类型为每个字段都是整数 'ae'
        assert_equal(mtest, np.ones(45, dtype=[(_, int) for _ in 'ae']))
        #
        mdata.seek(0)
        # 调用 np.genfromtxt 读取指定列数据，无异常引发
        mtest = np.genfromtxt(mdata, usecols=(0, 1), **kwargs)
        # 断言测试结果长度为 50
        assert_equal(len(mtest), 50)
        # 创建一个控制数据，包含指定的数据和类型
        control = np.ones(50, dtype=[(_, int) for _ in 'ab'])
        control[[10 * _ for _ in range(5)]] = (2, 2)
        # 断言测试结果与控制数据相等
        assert_equal(mtest, control)
    def test_inconsistent_dtype(self):
        # 测试不一致的数据类型处理
        # 创建包含重复数据的列表
        data = ["1, 1, 1, 1, -1.1"] * 50
        # 将数据列表连接成一个文本流对象
        mdata = TextIO("\n".join(data))

        # 定义转换器字典，将第4列的数据进行特定格式的转换
        converters = {4: lambda x: "(%s)" % x.decode()}
        # 构建参数字典，包括分隔符、转换器、数据类型、编码方式等
        kwargs = dict(delimiter=",", converters=converters,
                      dtype=[(_, int) for _ in 'abcde'], encoding="bytes")
        # 断言调用 genfromtxt 方法时会引发 ValueError 异常
        assert_raises(ValueError, np.genfromtxt, mdata, **kwargs)

    def test_default_field_format(self):
        # 测试默认字段格式
        # 定义包含数据的字符串
        data = "0, 1, 2.3\n4, 5, 6.7"
        # 创建包含数据的文本流对象
        mtest = np.genfromtxt(TextIO(data),
                             delimiter=",", dtype=None, defaultfmt="f%02i")
        # 创建预期的 NumPy 数组对象
        ctrl = np.array([(0, 1, 2.3), (4, 5, 6.7)],
                        dtype=[("f00", int), ("f01", int), ("f02", float)])
        # 断言生成的数组与预期的数组相等
        assert_equal(mtest, ctrl)

    def test_single_dtype_wo_names(self):
        # 测试单一数据类型但无字段名
        # 定义包含数据的字符串
        data = "0, 1, 2.3\n4, 5, 6.7"
        # 创建包含数据的文本流对象
        mtest = np.genfromtxt(TextIO(data),
                             delimiter=",", dtype=float, defaultfmt="f%02i")
        # 创建预期的 NumPy 数组对象
        ctrl = np.array([[0., 1., 2.3], [4., 5., 6.7]], dtype=float)
        # 断言生成的数组与预期的数组相等
        assert_equal(mtest, ctrl)

    def test_single_dtype_w_explicit_names(self):
        # 测试单一数据类型且使用显式字段名
        # 定义包含数据的字符串
        data = "0, 1, 2.3\n4, 5, 6.7"
        # 创建包含数据的文本流对象
        mtest = np.genfromtxt(TextIO(data),
                             delimiter=",", dtype=float, names="a, b, c")
        # 创建预期的 NumPy 结构化数组对象
        ctrl = np.array([(0., 1., 2.3), (4., 5., 6.7)],
                        dtype=[(_, float) for _ in "abc"])
        # 断言生成的结构化数组与预期的结构化数组相等
        assert_equal(mtest, ctrl)

    def test_single_dtype_w_implicit_names(self):
        # 测试单一数据类型且使用隐式字段名
        # 定义包含数据的字符串
        data = "a, b, c\n0, 1, 2.3\n4, 5, 6.7"
        # 创建包含数据的文本流对象
        mtest = np.genfromtxt(TextIO(data),
                             delimiter=",", dtype=float, names=True)
        # 创建预期的 NumPy 结构化数组对象
        ctrl = np.array([(0., 1., 2.3), (4., 5., 6.7)],
                        dtype=[(_, float) for _ in "abc"])
        # 断言生成的结构化数组与预期的结构化数组相等
        assert_equal(mtest, ctrl)

    def test_easy_structured_dtype(self):
        # 测试简单结构化数据类型
        # 定义包含数据的字符串
        data = "0, 1, 2.3\n4, 5, 6.7"
        # 创建包含数据的文本流对象
        mtest = np.genfromtxt(TextIO(data), delimiter=",",
                             dtype=(int, float, float), defaultfmt="f_%02i")
        # 创建预期的 NumPy 结构化数组对象
        ctrl = np.array([(0, 1., 2.3), (4, 5., 6.7)],
                        dtype=[("f_00", int), ("f_01", float), ("f_02", float)])
        # 断言生成的结构化数组与预期的结构化数组相等
        assert_equal(mtest, ctrl)
    def test_autostrip(self):
        # 测试自动去除空白功能
        data = "01/01/2003  , 1.3,   abcde"
        kwargs = dict(delimiter=",", dtype=None, encoding="bytes")
        # 捕获警告并记录
        with warnings.catch_warnings(record=True) as w:
            # 总是警告，不管警告内容
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            # 使用 np.genfromtxt 从 TextIO 对象中读取数据
            mtest = np.genfromtxt(TextIO(data), **kwargs)
            # 断言第一个警告类型为 VisibleDeprecationWarning
            assert_(w[0].category is VisibleDeprecationWarning)
        # 控制数组，用于断言检查
        ctrl = np.array([('01/01/2003  ', 1.3, '   abcde')],
                        dtype=[('f0', '|S12'), ('f1', float), ('f2', '|S8')])
        # 断言 mtest 和 ctrl 相等
        assert_equal(mtest, ctrl)
        # 再次捕获警告并记录
        with warnings.catch_warnings(record=True) as w:
            # 总是警告，不管警告内容
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            # 使用 np.genfromtxt 从 TextIO 对象中读取数据，启用自动去除空白
            mtest = np.genfromtxt(TextIO(data), autostrip=True, **kwargs)
            # 断言第一个警告类型为 VisibleDeprecationWarning
            assert_(w[0].category is VisibleDeprecationWarning)
        # 控制数组，用于断言检查
        ctrl = np.array([('01/01/2003', 1.3, 'abcde')],
                        dtype=[('f0', '|S10'), ('f1', float), ('f2', '|S5')])
        # 断言 mtest 和 ctrl 相等
        assert_equal(mtest, ctrl)

    def test_replace_space(self):
        # 测试 'replace_space' 选项
        txt = "A.A, B (B), C:C\n1, 2, 3.14"
        # 测试默认选项：将空格替换为 '_'，删除非字母数字字符
        test = np.genfromtxt(TextIO(txt),
                             delimiter=",", names=True, dtype=None)
        # 控制数据类型
        ctrl_dtype = [("AA", int), ("B_B", int), ("CC", float)]
        # 控制数组，用于断言检查
        ctrl = np.array((1, 2, 3.14), dtype=ctrl_dtype)
        # 断言 test 和 ctrl 相等
        assert_equal(test, ctrl)
        # 测试：不替换空格，不删除字符
        test = np.genfromtxt(TextIO(txt),
                             delimiter=",", names=True, dtype=None,
                             replace_space='', deletechars='')
        # 控制数据类型
        ctrl_dtype = [("A.A", int), ("B (B)", int), ("C:C", float)]
        # 控制数组，用于断言检查
        ctrl = np.array((1, 2, 3.14), dtype=ctrl_dtype)
        # 断言 test 和 ctrl 相等
        assert_equal(test, ctrl)
        # 测试：不删除字符（空格替换为 _）
        test = np.genfromtxt(TextIO(txt),
                             delimiter=",", names=True, dtype=None,
                             deletechars='')
        # 控制数据类型
        ctrl_dtype = [("A.A", int), ("B_(B)", int), ("C:C", float)]
        # 控制数组，用于断言检查
        ctrl = np.array((1, 2, 3.14), dtype=ctrl_dtype)
        # 断言 test 和 ctrl 相等
        assert_equal(test, ctrl)
    def test_replace_space_known_dtype(self):
        # 当 dtype != None 时，测试 'replace_space'（以及相关选项）
        txt = "A.A, B (B), C:C\n1, 2, 3"
        # 默认情况下：将空格替换为 '_'，删除非字母数字字符进行测试
        test = np.genfromtxt(TextIO(txt),
                             delimiter=",", names=True, dtype=int)
        ctrl_dtype = [("AA", int), ("B_B", int), ("CC", int)]
        ctrl = np.array((1, 2, 3), dtype=ctrl_dtype)
        assert_equal(test, ctrl)
        # 测试：不替换，不删除
        test = np.genfromtxt(TextIO(txt),
                             delimiter=",", names=True, dtype=int,
                             replace_space='', deletechars='')
        ctrl_dtype = [("A.A", int), ("B (B)", int), ("C:C", int)]
        ctrl = np.array((1, 2, 3), dtype=ctrl_dtype)
        assert_equal(test, ctrl)
        # 测试：不删除（空格被替换为 _）
        test = np.genfromtxt(TextIO(txt),
                             delimiter=",", names=True, dtype=int,
                             deletechars='')
        ctrl_dtype = [("A.A", int), ("B_(B)", int), ("C:C", int)]
        ctrl = np.array((1, 2, 3), dtype=ctrl_dtype)
        assert_equal(test, ctrl)

    def test_incomplete_names(self):
        # 测试包含不完整名称的情况
        data = "A,,C\n0,1,2\n3,4,5"
        kwargs = dict(delimiter=",", names=True)
        # 使用 dtype=None
        ctrl = np.array([(0, 1, 2), (3, 4, 5)],
                        dtype=[(_, int) for _ in ('A', 'f0', 'C')])
        test = np.genfromtxt(TextIO(data), dtype=None, **kwargs)
        assert_equal(test, ctrl)
        # 使用默认 dtype
        ctrl = np.array([(0, 1, 2), (3, 4, 5)],
                        dtype=[(_, float) for _ in ('A', 'f0', 'C')])
        test = np.genfromtxt(TextIO(data), **kwargs)

    def test_names_auto_completion(self):
        # 确保名称自动完成
        data = "1 2 3\n 4 5 6"
        test = np.genfromtxt(TextIO(data),
                             dtype=(int, float, int), names="a")
        ctrl = np.array([(1, 2, 3), (4, 5, 6)],
                        dtype=[('a', int), ('f0', float), ('f1', int)])
        assert_equal(test, ctrl)
    def test_names_with_usecols_bug1636(self):
        # 确保在使用 usecols 参数时选择正确的列名
        data = "A,B,C,D,E\n0,1,2,3,4\n0,1,2,3,4\n0,1,2,3,4"
        # 控制用于比较的列名列表
        ctrl_names = ("A", "C", "E")
        # 使用 genfromtxt 函数从文本数据中加载数据，并指定数据类型为整数
        test = np.genfromtxt(TextIO(data),
                             dtype=(int, int, int), delimiter=",",
                             usecols=(0, 2, 4), names=True)
        # 断言加载数据的列名与控制列表相同
        assert_equal(test.dtype.names, ctrl_names)
        #
        # 重新加载数据，这次使用列名字符串而不是索引
        test = np.genfromtxt(TextIO(data),
                             dtype=(int, int, int), delimiter=",",
                             usecols=("A", "C", "E"), names=True)
        # 再次断言加载数据的列名与控制列表相同
        assert_equal(test.dtype.names, ctrl_names)
        #
        # 再次重新加载数据，这次只指定整数数据类型而不指定列名数据类型
        test = np.genfromtxt(TextIO(data),
                             dtype=int, delimiter=",",
                             usecols=("A", "C", "E"), names=True)
        # 最后一次断言加载数据的列名与控制列表相同
        assert_equal(test.dtype.names, ctrl_names)

    def test_fixed_width_names(self):
        # 测试固定宽度文本数据的加载，同时保留列名
        data = "    A    B   C\n    0    1 2.3\n   45   67   9."
        kwargs = dict(delimiter=(5, 5, 4), names=True, dtype=None)
        # 控制用于比较的 NumPy 数组
        ctrl = np.array([(0, 1, 2.3), (45, 67, 9.)],
                        dtype=[('A', int), ('B', int), ('C', float)])
        # 使用 genfromtxt 函数加载数据
        test = np.genfromtxt(TextIO(data), **kwargs)
        # 断言加载的数据与控制数组相同
        assert_equal(test, ctrl)
        #
        # 再次加载数据，这次仅指定一个整数作为分隔符宽度
        kwargs = dict(delimiter=5, names=True, dtype=None)
        # 重新定义控制数组
        ctrl = np.array([(0, 1, 2.3), (45, 67, 9.)],
                        dtype=[('A', int), ('B', int), ('C', float)])
        # 再次使用 genfromtxt 函数加载数据
        test = np.genfromtxt(TextIO(data), **kwargs)
        # 再次断言加载的数据与控制数组相同
        assert_equal(test, ctrl)

    def test_filling_values(self):
        # 测试处理缺失值
        data = b"1, 2, 3\n1, , 5\n0, 6, \n"
        kwargs = dict(delimiter=",", dtype=None, filling_values=-999)
        # 控制用于比较的 NumPy 数组
        ctrl = np.array([[1, 2, 3], [1, -999, 5], [0, 6, -999]], dtype=int)
        # 使用 genfromtxt 函数加载数据
        test = np.genfromtxt(TextIO(data), **kwargs)
        # 断言加载的数据与控制数组相同
        assert_equal(test, ctrl)

    def test_comments_is_none(self):
        # 测试处理 None 类型注释的问题
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            # 使用 genfromtxt 函数加载数据，指定注释为 None
            test = np.genfromtxt(TextIO("test1,testNonetherestofthedata"),
                                 dtype=None, comments=None, delimiter=',',
                                 encoding="bytes")
            assert_(w[0].category is VisibleDeprecationWarning)
        # 断言加载的数据中的第二个元素为字节字符串 b'testNonetherestofthedata'
        assert_equal(test[1], b'testNonetherestofthedata')
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            # 重新加载数据，这次包含一个空格来触发警告
            test = np.genfromtxt(TextIO("test1, testNonetherestofthedata"),
                                 dtype=None, comments=None, delimiter=',',
                                 encoding="bytes")
            assert_(w[0].category is VisibleDeprecationWarning)
        # 再次断言加载的数据中的第二个元素为字节字符串 b' testNonetherestofthedata'
        assert_equal(test[1], b' testNonetherestofthedata')
    def test_latin1(self):
        # 定义 Latin-1 编码的字节序列
        latin1 = b'\xf6\xfc\xf6'
        # 定义普通的字节序列
        norm = b"norm1,norm2,norm3\n"
        # 定义混合 Latin-1 编码的字节序列
        enc = b"test1,testNonethe" + latin1 + b",test3\n"
        # 构建完整的测试数据流
        s = norm + enc + norm
        # 捕获警告信息
        with warnings.catch_warnings(record=True) as w:
            # 设置警告过滤器
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            # 使用 np.genfromtxt 从 TextIO 流中读取数据，指定参数
            test = np.genfromtxt(TextIO(s),
                                 dtype=None, comments=None, delimiter=',',
                                 encoding="bytes")
            # 断言捕获到的第一个警告是 VisibleDeprecationWarning
            assert_(w[0].category is VisibleDeprecationWarning)
        # 断言测试结果的特定元素与预期值相等
        assert_equal(test[1, 0], b"test1")
        assert_equal(test[1, 1], b"testNonethe" + latin1)
        assert_equal(test[1, 2], b"test3")
        
        # 使用 Latin-1 编码重新进行数据解析
        test = np.genfromtxt(TextIO(s),
                             dtype=None, comments=None, delimiter=',',
                             encoding='latin1')
        # 断言测试结果的特定元素与预期值相等
        assert_equal(test[1, 0], "test1")
        assert_equal(test[1, 1], "testNonethe" + latin1.decode('latin1'))
        assert_equal(test[1, 2], "test3")

        # 再次捕获警告信息
        with warnings.catch_warnings(record=True) as w:
            # 设置警告过滤器
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            # 使用 np.genfromtxt 从 TextIO 流中读取数据，指定参数
            test = np.genfromtxt(TextIO(b"0,testNonethe" + latin1),
                                 dtype=None, comments=None, delimiter=',',
                                 encoding="bytes")
            # 断言捕获到的第一个警告是 VisibleDeprecationWarning
            assert_(w[0].category is VisibleDeprecationWarning)
        # 断言测试结果的特定字段与预期值相等
        assert_equal(test['f0'], 0)
        assert_equal(test['f1'], b"testNonethe" + latin1)

    def test_binary_decode_autodtype(self):
        # 定义 UTF-16 编码的字节序列
        utf16 = b'\xff\xfeh\x04 \x00i\x04 \x00j\x04'
        # 调用被测函数，加载数据并指定参数
        v = self.loadfunc(BytesIO(utf16), dtype=None, encoding='UTF-16')
        # 断言加载后的数据数组与预期结果相等
        assert_array_equal(v, np.array(utf16.decode('UTF-16').split()))

    def test_utf8_byte_encoding(self):
        # 定义 UTF-8 编码的字节序列
        utf8 = b"\xcf\x96"
        # 定义普通的字节序列
        norm = b"norm1,norm2,norm3\n"
        # 定义混合 UTF-8 编码的字节序列
        enc = b"test1,testNonethe" + utf8 + b",test3\n"
        # 构建完整的测试数据流
        s = norm + enc + norm
        # 捕获警告信息
        with warnings.catch_warnings(record=True) as w:
            # 设置警告过滤器
            warnings.filterwarnings('always', '', VisibleDeprecationWarning)
            # 使用 np.genfromtxt 从 TextIO 流中读取数据，指定参数
            test = np.genfromtxt(TextIO(s),
                                 dtype=None, comments=None, delimiter=',',
                                 encoding="bytes")
            # 断言捕获到的第一个警告是 VisibleDeprecationWarning
            assert_(w[0].category is VisibleDeprecationWarning)
        # 定义预期的控制数组
        ctl = np.array([
                 [b'norm1', b'norm2', b'norm3'],
                 [b'test1', b'testNonethe' + utf8, b'test3'],
                 [b'norm1', b'norm2', b'norm3']])
        # 断言测试结果与预期的控制数组相等
        assert_array_equal(test, ctl)
    def test_utf8_file(self):
        # 定义 UTF-8 编码的特殊字符
        utf8 = b"\xcf\x96"
        
        # 使用临时路径创建文件，并写入重复的测试数据行
        with temppath() as path:
            with open(path, "wb") as f:
                f.write((b"test1,testNonethe" + utf8 + b",test3\n") * 2)
            
            # 从文件中读取数据到 NumPy 数组，指定 UTF-8 编码
            test = np.genfromtxt(path, dtype=None, comments=None,
                                 delimiter=',', encoding="UTF-8")
            
            # 创建控制数组 ctl 作为预期输出结果
            ctl = np.array([
                     ["test1", "testNonethe" + utf8.decode("UTF-8"), "test3"],
                     ["test1", "testNonethe" + utf8.decode("UTF-8"), "test3"]],
                     dtype=np.str_)
            
            # 断言测试结果与控制数组相等
            assert_array_equal(test, ctl)

            # 测试包含混合数据类型的情况
            with open(path, "wb") as f:
                f.write(b"0,testNonethe" + utf8)
            
            # 重新读取文件到 NumPy 数组，再次指定 UTF-8 编码
            test = np.genfromtxt(path, dtype=None, comments=None,
                                 delimiter=',', encoding="UTF-8")
            
            # 断言字段 'f0' 的值为 0
            assert_equal(test['f0'], 0)
            # 断言字段 'f1' 的值为 "testNonethe" 加上 UTF-8 解码的特殊字符
            assert_equal(test['f1'], "testNonethe" + utf8.decode("UTF-8"))

    def test_utf8_file_nodtype_unicode(self):
        # 使用 Unicode 字符代表 UTF-8 编码的特殊字符
        utf8 = '\u03d6'
        latin1 = '\xf6\xfc\xf6'

        # 如果无法使用首选编码对 UTF-8 测试字符串进行编码，则跳过测试
        try:
            encoding = locale.getpreferredencoding()
            utf8.encode(encoding)
        except (UnicodeError, ImportError):
            pytest.skip('Skipping test_utf8_file_nodtype_unicode, '
                        'unable to encode utf8 in preferred encoding')

        # 使用临时路径创建文件，并写入多行文本数据
        with temppath() as path:
            with open(path, "wt") as f:
                f.write("norm1,norm2,norm3\n")
                f.write("norm1," + latin1 + ",norm3\n")
                f.write("test1,testNonethe" + utf8 + ",test3\n")
            
            # 忽略有关 recfromtxt 的警告信息
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings('always', '',
                                        VisibleDeprecationWarning)
                # 从文件中读取数据到 NumPy 数组，使用 bytes 编码
                test = np.genfromtxt(path, dtype=None, comments=None,
                                     delimiter=',', encoding="bytes")
                
                # 检查是否出现编码未指定警告
                assert_(w[0].category is VisibleDeprecationWarning)
            
            # 创建控制数组 ctl 作为预期输出结果
            ctl = np.array([
                     ["norm1", "norm2", "norm3"],
                     ["norm1", latin1, "norm3"],
                     ["test1", "testNonethe" + utf8, "test3"]],
                     dtype=np.str_)
            
            # 断言测试结果与控制数组相等
            assert_array_equal(test, ctl)

    @pytest.mark.filterwarnings("ignore:.*recfromtxt.*:DeprecationWarning")
    # 定义测试函数 `test_recfromtxt`，用于测试 `recfromtxt` 函数
    def test_recfromtxt(self):
        # 创建包含数据的文本流对象
        data = TextIO('A,B\n0,1\n2,3')
        # 设置关键字参数字典
        kwargs = dict(delimiter=",", missing_values="N/A", names=True)
        # 调用 `recfromtxt` 函数，使用给定的参数
        test = recfromtxt(data, **kwargs)
        # 创建期望结果的 NumPy 数组
        control = np.array([(0, 1), (2, 3)],
                           dtype=[('A', int), ('B', int)])
        # 断言 `test` 是一个 `np.recarray` 类型的对象
        assert_(isinstance(test, np.recarray))
        # 断言 `test` 和 `control` 数组相等
        assert_equal(test, control)
        # 创建包含新数据的文本流对象
        data = TextIO('A,B\n0,1\n2,N/A')
        # 使用额外的关键字参数调用 `recfromtxt` 函数
        test = recfromtxt(data, dtype=None, usemask=True, **kwargs)
        # 创建期望结果的掩码数组
        control = ma.array([(0, 1), (2, -1)],
                           mask=[(False, False), (False, True)],
                           dtype=[('A', int), ('B', int)])
        # 断言 `test` 和 `control` 数组相等
        assert_equal(test, control)
        # 断言 `test.mask` 和 `control.mask` 数组相等
        assert_equal(test.mask, control.mask)
        # 断言 `test.A` 数组中的值与期望的一致
        assert_equal(test.A, [0, 2])

    # 使用 pytest 标记忽略特定警告，针对 `recfromcsv` 函数的测试
    @pytest.mark.filterwarnings("ignore:.*recfromcsv.*:DeprecationWarning")
    def test_recfromcsv(self):
        # 创建包含数据的文本流对象
        data = TextIO('A,B\n0,1\n2,3')
        # 设置关键字参数字典
        kwargs = dict(missing_values="N/A", names=True, case_sensitive=True,
                      encoding="bytes")
        # 使用给定参数调用 `recfromcsv` 函数
        test = recfromcsv(data, dtype=None, **kwargs)
        # 创建期望结果的 NumPy 数组
        control = np.array([(0, 1), (2, 3)],
                           dtype=[('A', int), ('B', int)])
        # 断言 `test` 是一个 `np.recarray` 类型的对象
        assert_(isinstance(test, np.recarray))
        # 断言 `test` 和 `control` 数组相等
        assert_equal(test, control)
        # 创建包含新数据的文本流对象
        data = TextIO('A,B\n0,1\n2,N/A')
        # 使用额外的关键字参数调用 `recfromcsv` 函数
        test = recfromcsv(data, dtype=None, usemask=True, **kwargs)
        # 创建期望结果的掩码数组
        control = ma.array([(0, 1), (2, -1)],
                           mask=[(False, False), (False, True)],
                           dtype=[('A', int), ('B', int)])
        # 断言 `test` 和 `control` 数组相等
        assert_equal(test, control)
        # 断言 `test.mask` 和 `control.mask` 数组相等
        assert_equal(test.mask, control.mask)
        # 创建包含数据的文本流对象
        data = TextIO('A,B\n0,1\n2,3')
        # 使用单个关键字参数调用 `recfromcsv` 函数
        test = recfromcsv(data, missing_values='N/A',)
        # 创建期望结果的 NumPy 数组
        control = np.array([(0, 1), (2, 3)],
                           dtype=[('a', int), ('b', int)])
        # 断言 `test` 是一个 `np.recarray` 类型的对象
        assert_(isinstance(test, np.recarray))
        # 断言 `test` 和 `control` 数组相等
        assert_equal(test, control)
        # 创建包含数据的文本流对象
        data = TextIO('A,B\n0,1\n2,3')
        # 定义新的数据类型
        dtype = [('a', int), ('b', float)]
        # 使用额外的关键字参数调用 `recfromcsv` 函数
        test = recfromcsv(data, missing_values='N/A', dtype=dtype)
        # 创建期望结果的 NumPy 数组
        control = np.array([(0, 1), (2, 3)],
                           dtype=dtype)
        # 断言 `test` 是一个 `np.recarray` 类型的对象
        assert_(isinstance(test, np.recarray))
        # 断言 `test` 和 `control` 数组相等
        assert_equal(test, control)

        #gh-10394
        # 创建包含数据的文本流对象，用于测试特定的转换器
        data = TextIO('color\n"red"\n"blue"')
        # 使用自定义转换器调用 `recfromcsv` 函数
        test = recfromcsv(data, converters={0: lambda x: x.strip('\"')})
        # 创建期望结果的 NumPy 数组
        control = np.array([('red',), ('blue',)], dtype=[('color', (str, 4))])
        # 断言 `test.dtype` 和 `control.dtype` 数组相等
        assert_equal(test.dtype, control.dtype)
        # 断言 `test` 和 `control` 数组相等
        assert_equal(test, control)
    def test_max_rows(self):
        # Test the `max_rows` keyword argument.
        data = '1 2\n3 4\n5 6\n7 8\n9 10\n'
        # 创建一个 TextIO 对象，用于模拟数据输入流
        txt = TextIO(data)
        # 从文本输入流中使用 numpy 读取数据，限制最大读取行数为 3
        a1 = np.genfromtxt(txt, max_rows=3)
        # 从同一个文本输入流中继续读取数据，未指定 max_rows，因此读取剩余的行
        a2 = np.genfromtxt(txt)
        # 断言 a1 的结果与预期值相等
        assert_equal(a1, [[1, 2], [3, 4], [5, 6]])
        # 断言 a2 的结果与预期值相等
        assert_equal(a2, [[7, 8], [9, 10]])

        # max_rows 参数必须至少为 1，验证是否会引发 ValueError 异常
        assert_raises(ValueError, np.genfromtxt, TextIO(data), max_rows=0)

        # 包含多个无效行的输入
        data = '1 1\n2 2\n0 \n3 3\n4 4\n5  \n6  \n7  \n'
        # 从文本输入流中读取最多 2 行数据
        test = np.genfromtxt(TextIO(data), max_rows=2)
        control = np.array([[1., 1.], [2., 2.]])
        assert_equal(test, control)

        # 测试关键字冲突
        assert_raises(ValueError, np.genfromtxt, TextIO(data), skip_footer=1,
                      max_rows=4)

        # 测试无效值情况
        assert_raises(ValueError, np.genfromtxt, TextIO(data), max_rows=4)

        # 测试无效值但不抛出异常的情况
        with suppress_warnings() as sup:
            sup.filter(ConversionWarning)

            test = np.genfromtxt(TextIO(data), max_rows=4, invalid_raise=False)
            control = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
            assert_equal(test, control)

            test = np.genfromtxt(TextIO(data), max_rows=5, invalid_raise=False)
            assert_equal(test, control)

        # 带有字段名的结构化数组
        data = 'a b\n#c d\n1 1\n2 2\n#0 \n3 3\n4 4\n5  5\n'

        # 测试带有标题、字段名和注释的情况
        txt = TextIO(data)
        test = np.genfromtxt(txt, skip_header=1, max_rows=3, names=True)
        control = np.array([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
                      dtype=[('c', '<f8'), ('d', '<f8')])
        assert_equal(test, control)
        # 继续读取相同的 "文件"，不使用 skip_header 或 names，并使用之前确定的 dtype。
        test = np.genfromtxt(txt, max_rows=None, dtype=test.dtype)
        control = np.array([(4.0, 4.0), (5.0, 5.0)],
                      dtype=[('c', '<f8'), ('d', '<f8')])
        assert_equal(test, control)

    def test_gft_using_filename(self):
        # 测试能够从文件名以及文件对象中加载数据
        tgt = np.arange(6).reshape((2, 3))
        linesep = ('\n', '\r\n', '\r')

        for sep in linesep:
            data = '0 1 2' + sep + '3 4 5'
            # 使用临时文件路径来创建文件，并将数据写入文件中
            with temppath() as name:
                with open(name, 'w') as f:
                    f.write(data)
                # 从文件中读取数据并使用 numpy 进行处理
                res = np.genfromtxt(name)
            # 断言读取的结果与目标值相等
            assert_array_equal(res, tgt)
    def test_gft_from_gzip(self):
        # 测试从 gzip 文件中加载数据
        wanted = np.arange(6).reshape((2, 3))
        linesep = ('\n', '\r\n', '\r')

        for sep in linesep:
            # 构造包含不同换行符的数据字符串
            data = '0 1 2' + sep + '3 4 5'
            s = BytesIO()
            # 使用 gzip 将数据写入 BytesIO 对象
            with gzip.GzipFile(fileobj=s, mode='w') as g:
                g.write(asbytes(data))

            # 创建临时文件，并将数据写入文件
            with temppath(suffix='.gz2') as name:
                with open(name, 'w') as f:
                    f.write(data)
                # 使用 np.genfromtxt 读取临时文件，并验证结果与期望值相等
                assert_array_equal(np.genfromtxt(name), wanted)

    def test_gft_using_generator(self):
        # gft 不能处理 Unicode 数据
        def count():
            for i in range(10):
                yield asbytes("%d" % i)

        # 使用生成器对象作为输入，验证 np.genfromtxt 的输出结果
        res = np.genfromtxt(count())
        assert_array_equal(res, np.arange(10))

    def test_auto_dtype_largeint(self):
        # 对于 numpy/numpy#5635 的回归测试，验证大整数可能引发的 OverflowError

        # 测试自动定义输出 dtype
        #
        # 2**66 = 73786976294838206464 => 应转换为 float
        # 2**34 = 17179869184 => 应转换为 int64
        # 2**10 = 1024 => 应转换为 int (在 32 位系统上为 int32，在 64 位系统上为 int64)

        data = TextIO('73786976294838206464 17179869184 1024')

        # 使用 np.genfromtxt 读取文本数据，不指定 dtype
        test = np.genfromtxt(data, dtype=None)

        # 验证生成的 dtype 的字段名
        assert_equal(test.dtype.names, ['f0', 'f1', 'f2'])

        # 验证每个字段的 dtype
        assert_(test.dtype['f0'] == float)
        assert_(test.dtype['f1'] == np.int64)
        assert_(test.dtype['f2'] == np.int_)

        # 验证字段数据是否正确转换
        assert_allclose(test['f0'], 73786976294838206464.)
        assert_equal(test['f1'], 17179869184)
        assert_equal(test['f2'], 1024)

    def test_unpack_float_data(self):
        txt = TextIO("1,2,3\n4,5,6\n7,8,9\n0.0,1.0,2.0")
        # 使用 np.loadtxt 解析文本数据，以逗号为分隔符，同时进行数据解包
        a, b, c = np.loadtxt(txt, delimiter=",", unpack=True)
        assert_array_equal(a, np.array([1.0, 4.0, 7.0, 0.0]))
        assert_array_equal(b, np.array([2.0, 5.0, 8.0, 1.0]))
        assert_array_equal(c, np.array([3.0, 6.0, 9.0, 2.0]))

    def test_unpack_structured(self):
        # 对于 gh-4341 的回归测试，验证结构化数组的解包功能
        txt = TextIO("M 21 72\nF 35 58")
        dt = {'names': ('a', 'b', 'c'), 'formats': ('S1', 'i4', 'f4')}
        # 使用 np.genfromtxt 解析文本数据，指定 dtype，并进行数据解包
        a, b, c = np.genfromtxt(txt, dtype=dt, unpack=True)
        assert_equal(a.dtype, np.dtype('S1'))
        assert_equal(b.dtype, np.dtype('i4'))
        assert_equal(c.dtype, np.dtype('f4'))
        assert_array_equal(a, np.array([b'M', b'F']))
        assert_array_equal(b, np.array([21, 35]))
        assert_array_equal(c, np.array([72.,  58.]))
    def test_unpack_auto_dtype(self):
        # Regression test for gh-4341
        # 进行gh-4341的回归测试
        # Unpacking should work when dtype=None
        # 当dtype=None时，应该可以正常解包
        txt = TextIO("M 21 72.\nF 35 58.")
        # 创建预期结果，包括字符串数组和数值数组
        expected = (np.array(["M", "F"]), np.array([21, 35]), np.array([72., 58.]))
        # 使用genfromtxt从文本输入txt中读取数据，指定dtype为None，开启解包模式，并使用utf-8编码
        test = np.genfromtxt(txt, dtype=None, unpack=True, encoding="utf-8")
        # 遍历预期结果和测试结果，逐一断言数组相等及其数据类型相等
        for arr, result in zip(expected, test):
            assert_array_equal(arr, result)
            assert_equal(arr.dtype, result.dtype)

    def test_unpack_single_name(self):
        # Regression test for gh-4341
        # 进行gh-4341的回归测试
        # Unpacking should work when structured dtype has only one field
        # 当结构化dtype只有一个字段时，应该可以正常解包
        txt = TextIO("21\n35")
        # 定义结构化dtype
        dt = {'names': ('a',), 'formats': ('i4',)}
        # 创建预期结果，包括整数数组
        expected = np.array([21, 35], dtype=np.int32)
        # 使用genfromtxt从文本输入txt中读取数据，指定dtype为dt，开启解包模式
        test = np.genfromtxt(txt, dtype=dt, unpack=True)
        # 断言预期结果和测试结果数组相等，及其数据类型相等
        assert_array_equal(expected, test)
        assert_equal(expected.dtype, test.dtype)

    def test_squeeze_scalar(self):
        # Regression test for gh-4341
        # 进行gh-4341的回归测试
        # Unpacking a scalar should give zero-dim output,
        # even if dtype is structured
        # 即使dtype是结构化的，解包标量应该得到零维输出
        txt = TextIO("1")
        # 定义结构化dtype
        dt = {'names': ('a',), 'formats': ('i4',)}
        # 创建预期结果，包括整数数组
        expected = np.array((1,), dtype=np.int32)
        # 使用genfromtxt从文本输入txt中读取数据，指定dtype为dt，开启解包模式
        test = np.genfromtxt(txt, dtype=dt, unpack=True)
        # 断言预期结果和测试结果数组相等
        assert_array_equal(expected, test)
        # 断言测试结果的形状为零维
        assert_equal((), test.shape)
        # 断言预期结果和测试结果的数据类型相等
        assert_equal(expected.dtype, test.dtype)

    @pytest.mark.parametrize("ndim", [0, 1, 2])
    def test_ndmin_keyword(self, ndim: int):
        # lets have the same behaviour of ndmin as loadtxt
        # 让ndmin的行为与loadtxt相同
        # as they should be the same for non-missing values
        # 因为对于非缺失值，它们应该是相同的
        txt = "42"
        # 使用loadtxt和genfromtxt分别加载文本输入txt，指定ndmin参数为ndim
        a = np.loadtxt(StringIO(txt), ndmin=ndim)
        b = np.genfromtxt(StringIO(txt), ndmin=ndim)
        # 断言两者结果相等
        assert_array_equal(a, b)
class TestPathUsage:
    # 测试 pathlib.Path 是否可以使用

    def test_loadtxt(self):
        # 使用临时路径创建一个后缀为 '.txt' 的文件
        with temppath(suffix='.txt') as path:
            # 将路径转换为 pathlib.Path 对象
            path = Path(path)
            # 创建一个二维数组
            a = np.array([[1.1, 2], [3, 4]])
            # 将数组 a 保存到路径对应的文件中
            np.savetxt(path, a)
            # 从文件中加载数据到数组 x
            x = np.loadtxt(path)
            # 断言数组 x 和数组 a 相等
            assert_array_equal(x, a)

    def test_save_load(self):
        # 测试 pathlib.Path 实例能否与 save 方法一起使用
        with temppath(suffix='.npy') as path:
            path = Path(path)
            a = np.array([[1, 2], [3, 4]], int)
            # 将数组 a 保存到路径对应的文件中
            np.save(path, a)
            # 从文件中加载数据到变量 data
            data = np.load(path)
            # 断言变量 data 和数组 a 相等
            assert_array_equal(data, a)

    def test_save_load_memmap(self):
        # 测试 pathlib.Path 实例能否用于加载内存映射
        with temppath(suffix='.npy') as path:
            path = Path(path)
            a = np.array([[1, 2], [3, 4]], int)
            # 将数组 a 保存到路径对应的文件中
            np.save(path, a)
            # 以只读模式加载内存映射数据到变量 data
            data = np.load(path, mmap_mode='r')
            # 断言变量 data 和数组 a 相等
            assert_array_equal(data, a)
            # 关闭内存映射文件
            del data
            if IS_PYPY:
                break_cycles()
                break_cycles()

    @pytest.mark.xfail(IS_WASM, reason="memmap doesn't work correctly")
    @pytest.mark.parametrize("filename_type", [Path, str])
    def test_save_load_memmap_readwrite(self, filename_type):
        # 测试 pathlib.Path 实例能否用于读写内存映射
        with temppath(suffix='.npy') as path:
            path = filename_type(path)
            a = np.array([[1, 2], [3, 4]], int)
            # 将数组 a 保存到路径对应的文件中
            np.save(path, a)
            # 以读写模式加载内存映射数据到变量 b
            b = np.load(path, mmap_mode='r+')
            # 修改数组 a 和内存映射数据 b 的第一个元素
            a[0][0] = 5
            b[0][0] = 5
            # 关闭内存映射文件
            del b
            if IS_PYPY:
                break_cycles()
                break_cycles()
            # 重新加载路径对应的数据到变量 data
            data = np.load(path)
            # 断言变量 data 和数组 a 相等
            assert_array_equal(data, a)

    @pytest.mark.parametrize("filename_type", [Path, str])
    def test_savez_load(self, filename_type):
        # 测试 pathlib.Path 实例能否与 savez 方法一起使用
        with temppath(suffix='.npz') as path:
            path = filename_type(path)
            # 保存带有 'lab' 键的数据到路径对应的文件中
            np.savez(path, lab='place holder')
            # 使用 with 语句加载路径对应的数据到变量 data
            with np.load(path) as data:
                # 断言变量 data 的 'lab' 键的值与 'place holder' 相等
                assert_array_equal(data['lab'], 'place holder')

    @pytest.mark.parametrize("filename_type", [Path, str])
    def test_savez_compressed_load(self, filename_type):
        # 测试 pathlib.Path 实例能否与 savez_compressed 方法一起使用
        with temppath(suffix='.npz') as path:
            path = filename_type(path)
            # 压缩保存带有 'lab' 键的数据到路径对应的文件中
            np.savez_compressed(path, lab='place holder')
            # 加载路径对应的数据到变量 data
            data = np.load(path)
            # 断言变量 data 的 'lab' 键的值与 'place holder' 相等
            assert_array_equal(data['lab'], 'place holder')
            # 关闭文件数据
            data.close()

    @pytest.mark.parametrize("filename_type", [Path, str])
    def test_genfromtxt(self, filename_type):
        # 测试 pathlib.Path 实例能否与 genfromtxt 方法一起使用
        with temppath(suffix='.txt') as path:
            path = filename_type(path)
            a = np.array([(1, 2), (3, 4)])
            # 将数组 a 保存到路径对应的文件中
            np.savetxt(path, a)
            # 从文件中加载数据到变量 data
            data = np.genfromtxt(path)
            # 断言数组 a 和变量 data 相等
            assert_array_equal(a, data)

    @pytest.mark.parametrize("filename_type", [Path, str])
    @pytest.mark.filterwarnings("ignore:.*recfromtxt.*:DeprecationWarning")
    # 定义一个测试方法，用于测试从文本文件中读取结构化数据
    def test_recfromtxt(self, filename_type):
        # 使用临时路径创建一个以'.txt'结尾的文件
        with temppath(suffix='.txt') as path:
            # 将路径转换为指定类型（Path对象或字符串）
            path = filename_type(path)
            # 打开文件并写入数据'A,B\n0,1\n2,3'
            with open(path, 'w') as f:
                f.write('A,B\n0,1\n2,3')

            # 定义参数字典，指定分隔符为逗号，缺失值标记为"N/A"，使用列名
            kwargs = dict(delimiter=",", missing_values="N/A", names=True)
            # 调用recfromtxt函数读取数据文件，并传入参数kwargs
            test = recfromtxt(path, **kwargs)
            # 创建预期结果，一个包含元组的NumPy数组，指定列A和B的数据类型为整数
            control = np.array([(0, 1), (2, 3)], dtype=[('A', int), ('B', int)])
            # 断言测试结果是一个np.recarray结构
            assert_(isinstance(test, np.recarray))
            # 断言测试结果与预期结果相等
            assert_equal(test, control)

    # 使用pytest的参数化装饰器，定义一个参数化测试方法，测试从CSV文件中读取结构化数据
    @pytest.mark.parametrize("filename_type", [Path, str])
    # 忽略与'recfromcsv'相关的DeprecationWarning警告
    @pytest.mark.filterwarnings("ignore:.*recfromcsv.*:DeprecationWarning")
    def test_recfromcsv(self, filename_type):
        # 使用临时路径创建一个以'.txt'结尾的文件
        with temppath(suffix='.txt') as path:
            # 将路径转换为指定类型（Path对象或字符串）
            path = filename_type(path)
            # 打开文件并写入数据'A,B\n0,1\n2,3'
            with open(path, 'w') as f:
                f.write('A,B\n0,1\n2,3')

            # 定义参数字典，指定缺失值标记为"N/A"，使用列名，并区分大小写
            kwargs = dict(
                missing_values="N/A", names=True, case_sensitive=True
            )
            # 调用recfromcsv函数读取CSV文件，并传入参数kwargs
            test = recfromcsv(path, dtype=None, **kwargs)
            # 创建预期结果，一个包含元组的NumPy数组，指定列A和B的数据类型为整数
            control = np.array([(0, 1), (2, 3)], dtype=[('A', int), ('B', int)])
            # 断言测试结果是一个np.recarray结构
            assert_(isinstance(test, np.recarray))
            # 断言测试结果与预期结果相等
            assert_equal(test, control)
# 定义一个测试函数，用于验证从 gzip 压缩文件加载数据的功能
def test_gzip_load():
    # 创建一个 5x5 的随机数组
    a = np.random.random((5, 5))

    # 创建一个 BytesIO 对象，用于在内存中操作二进制数据
    s = BytesIO()

    # 使用 gzip 压缩模式创建 GzipFile 对象，将数据写入 s
    f = gzip.GzipFile(fileobj=s, mode="w")
    np.save(f, a)  # 将数组 a 保存到压缩文件中
    f.close()  # 关闭文件流
    s.seek(0)  # 将文件指针移动到文件开头

    # 使用 gzip 解压模式创建 GzipFile 对象，从 s 中加载数据并与原始数组 a 进行比较
    f = gzip.GzipFile(fileobj=s, mode="r")
    assert_array_equal(np.load(f), a)  # 断言加载的数组与原始数组 a 相等


# 下面两个类提供了最小的 API 来保存（save()）/加载（load()）数组
# `test_ducktyping` 函数确保它们能够正常工作
class JustWriter:
    def __init__(self, base):
        self.base = base

    def write(self, s):
        return self.base.write(s)

    def flush(self):
        return self.base.flush()


class JustReader:
    def __init__(self, base):
        self.base = base

    def read(self, n):
        return self.base.read(n)

    def seek(self, off, whence=0):
        return self.base.seek(off, whence)


# 测试 duck typing 功能，确保 JustWriter 和 JustReader 类可以正确保存和加载数组
def test_ducktyping():
    # 创建一个 5x5 的随机数组
    a = np.random.random((5, 5))

    # 创建一个 BytesIO 对象，用于在内存中操作二进制数据
    s = BytesIO()

    # 使用 JustWriter 类封装 s，保存数组 a 到 s 中
    f = JustWriter(s)
    np.save(f, a)
    f.flush()  # 刷新数据
    s.seek(0)  # 将文件指针移动到文件开头

    # 使用 JustReader 类封装 s，加载数据并与原始数组 a 进行比较
    f = JustReader(s)
    assert_array_equal(np.load(f), a)  # 断言加载的数组与原始数组 a 相等


# 从 gzip 压缩的文件中加载数据并进行测试
def test_gzip_loadtxt():
    # 创建一个 BytesIO 对象，用于在内存中操作二进制数据
    s = BytesIO()

    # 使用 gzip 压缩模式创建 GzipFile 对象，写入一个简单的字符串数据
    g = gzip.GzipFile(fileobj=s, mode='w')
    g.write(b'1 2 3\n')
    g.close()

    s.seek(0)  # 将文件指针移动到文件开头

    # 创建一个临时文件，将压缩数据写入，然后使用 np.loadtxt() 加载数据并进行断言
    with temppath(suffix='.gz') as name:
        with open(name, 'wb') as f:
            f.write(s.read())
        res = np.loadtxt(name)
    s.close()  # 关闭 BytesIO 对象

    assert_array_equal(res, [1, 2, 3])  # 断言加载的数据与预期的数组相等


# 从字符串中加载 gzip 压缩的数据并进行测试
def test_gzip_loadtxt_from_string():
    # 创建一个 BytesIO 对象，用于在内存中操作二进制数据
    s = BytesIO()

    # 使用 gzip 压缩模式创建 GzipFile 对象，写入一个简单的字符串数据
    f = gzip.GzipFile(fileobj=s, mode="w")
    f.write(b'1 2 3\n')
    f.close()

    s.seek(0)  # 将文件指针移动到文件开头

    # 使用 gzip 解压模式创建 GzipFile 对象，从 s 中加载数据并进行断言
    f = gzip.GzipFile(fileobj=s, mode="r")
    assert_array_equal(np.loadtxt(f), [1, 2, 3])  # 断言加载的数据与预期的数组相等


# 测试 npz 文件中保存的字典数据
def test_npzfile_dict():
    # 创建一个 BytesIO 对象，用于在内存中操作二进制数据
    s = BytesIO()

    # 创建两个 3x3 的零矩阵
    x = np.zeros((3, 3))
    y = np.zeros((3, 3))

    # 将 x 和 y 保存到 npz 格式的 s 中
    np.savez(s, x=x, y=y)
    s.seek(0)  # 将文件指针移动到文件开头

    # 从 s 中加载数据
    z = np.load(s)

    # 断言 x 和 y 存在于 z 中
    assert_('x' in z)
    assert_('y' in z)
    assert_('x' in z.keys())
    assert_('y' in z.keys())

    # 遍历 z 中的每个元素，断言其形状为 (3, 3)
    for f, a in z.items():
        assert_(f in ['x', 'y'])
        assert_equal(a.shape, (3, 3))

    # 断言 z 中有两个元素
    assert_(len(z.items()) == 2)

    # 遍历 z 中的每个键，断言其存在于 ['x', 'y'] 中
    for f in z:
        assert_(f in ['x', 'y'])

    # 断言 z 中的 'x' 数组与 z['x'] 数组内容相同
    assert (z.get('x') == z['x']).all()


# 使用 pytest.mark.skipif 装饰器跳过不支持 refcount 的平台的测试
@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_load_refcount():
    # 创建一个 BytesIO 对象，用于在内存中操作二进制数据
    f = BytesIO()

    # 将一个简单的数组保存到 f 中
    np.savez(f, [1, 2, 3])
    f.seek(0)  # 将文件指针移动到文件开头

    # 使用 assert_no_gc_cycles() 上下文管理器，确保加载的对象能够直接释放，不依赖于 gc
    with assert_no_gc_cycles():
        np.load(f)

    f.seek(0)  # 将文件指针移动到文件开头
    dt = [("a", 'u1', 2), ("b", 'u1', 2)]
    # 使用 assert_no_gc_cycles() 上下文管理器来确保没有循环引用的垃圾回收
    with assert_no_gc_cycles():
        # 使用 np.loadtxt() 函数从文本输入加载数据并按指定的数据类型 (dt) 转换
        x = np.loadtxt(TextIO("0 1 2 3"), dtype=dt)
        # 断言 x 的值与指定的 numpy 数组相等
        assert_equal(x, np.array([((0, 1), (2, 3))], dtype=dt))
# 定义一个测试函数，用于测试加载多个数组直至文件结束
def test_load_multiple_arrays_until_eof():
    # 创建一个字节流对象
    f = BytesIO()
    # 在字节流中保存数组1
    np.save(f, 1)
    # 在字节流中保存数组2
    np.save(f, 2)
    # 将字节流的读写位置设置回起始位置
    f.seek(0)
    # 断言从字节流中加载的第一个数组等于1
    assert np.load(f) == 1
    # 断言从字节流中加载的第二个数组等于2
    assert np.load(f) == 2
    # 使用 pytest 的断言，期望从字节流中加载数据时抛出 EOFError 异常
    with pytest.raises(EOFError):
        np.load(f)
```