# `.\numpy\numpy\_core\tests\test_memmap.py`

```
import sys
import os
import mmap
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile

from numpy import (
    memmap, sum, average, prod, ndarray, isscalar, add, subtract, multiply)

from numpy import arange, allclose, asarray
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, suppress_warnings, IS_PYPY,
    break_cycles
    )

class TestMemmap:
    def setup_method(self):
        # 创建一个临时文件对象，用于测试
        self.tmpfp = NamedTemporaryFile(prefix='mmap')
        # 设置数据形状和类型
        self.shape = (3, 4)
        self.dtype = 'float32'
        # 创建一个数据数组，并调整形状
        self.data = arange(12, dtype=self.dtype)
        self.data.resize(self.shape)

    def teardown_method(self):
        # 关闭临时文件对象
        self.tmpfp.close()
        self.data = None
        # 如果运行环境是 PyPy，调用 break_cycles 两次以确保释放所有资源
        if IS_PYPY:
            break_cycles()
            break_cycles()

    def test_roundtrip(self):
        # 将数据写入文件
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        fp[:] = self.data[:]
        del fp  # 测试 __del__ 机制，处理清理工作

        # 从文件中读取数据
        newfp = memmap(self.tmpfp, dtype=self.dtype, mode='r',
                       shape=self.shape)
        # 断言数据相似性
        assert_(allclose(self.data, newfp))
        # 断言数组相等性
        assert_array_equal(self.data, newfp)
        # 断言新文件对象不可写
        assert_equal(newfp.flags.writeable, False)

    def test_open_with_filename(self, tmp_path):
        # 在临时目录下创建文件名为 'mmap' 的 memmap 对象
        tmpname = tmp_path / 'mmap'
        fp = memmap(tmpname, dtype=self.dtype, mode='w+',
                       shape=self.shape)
        fp[:] = self.data[:]
        del fp

    def test_unnamed_file(self):
        # 使用 TemporaryFile 创建未命名文件对象
        with TemporaryFile() as f:
            fp = memmap(f, dtype=self.dtype, shape=self.shape)
            del fp

    def test_attributes(self):
        # 设置偏移量和模式
        offset = 1
        mode = "w+"
        # 创建 memmap 对象，指定偏移量和模式
        fp = memmap(self.tmpfp, dtype=self.dtype, mode=mode,
                    shape=self.shape, offset=offset)
        # 断言偏移量和模式正确
        assert_equal(offset, fp.offset)
        assert_equal(mode, fp.mode)
        del fp

    def test_filename(self, tmp_path):
        # 在临时目录下创建文件名为 'mmap' 的 memmap 对象
        tmpname = tmp_path / "mmap"
        fp = memmap(tmpname, dtype=self.dtype, mode='w+',
                       shape=self.shape)
        # 获取绝对路径并进行断言
        abspath = Path(os.path.abspath(tmpname))
        fp[:] = self.data[:]
        assert_equal(abspath, fp.filename)
        b = fp[:1]
        assert_equal(abspath, b.filename)
        del b
        del fp

    def test_path(self, tmp_path):
        # 在临时目录下创建文件名为 'mmap' 的 memmap 对象
        tmpname = tmp_path / "mmap"
        fp = memmap(Path(tmpname), dtype=self.dtype, mode='w+',
                       shape=self.shape)
        # 获取绝对路径并进行断言，使用 Path.resolve 以解决可能存在的符号链接问题
        abspath = str(Path(tmpname).resolve())
        fp[:] = self.data[:]
        assert_equal(abspath, str(fp.filename.resolve()))
        b = fp[:1]
        assert_equal(abspath, str(b.filename.resolve()))
        del b
        del fp


这些注释详细解释了每行代码的作用和意图，确保每个函数和方法的功能清晰可理解。
    # 定义一个测试函数，用于测试 memmap 对象的 filename 属性是否正确返回文件名
    def test_filename_fileobj(self):
        # 创建一个 memmap 对象，以写入模式打开，并使用指定的数据类型和形状
        fp = memmap(self.tmpfp, dtype=self.dtype, mode="w+", shape=self.shape)
        # 断言 memmap 对象的 filename 属性是否与临时文件对象的名称相同
        assert_equal(fp.filename, self.tmpfp.name)

    # 标记为跳过测试，如果系统平台为 'gnu0'，原因是它已知在 hurd 上会失败
    @pytest.mark.skipif(sys.platform == 'gnu0', reason="Known to fail on hurd")
    def test_flush(self):
        # 创建一个 memmap 对象，以写入模式打开，并使用指定的数据类型和形状
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+', shape=self.shape)
        # 将数据复制到 memmap 对象中
        fp[:] = self.data[:]
        # 断言 memmap 对象中的第一个元素是否与给定数据的第一个元素相同
        assert_equal(fp[0], self.data[0])
        # 刷新 memmap 对象
        fp.flush()

    # 测试删除操作，确保视图不会删除底层的 mmap
    def test_del(self):
        # 创建一个 memmap 对象作为基础对象，以写入模式打开，并使用指定的数据类型和形状
        fp_base = memmap(self.tmpfp, dtype=self.dtype, mode='w+', shape=self.shape)
        # 修改基础对象的第一个元素的值
        fp_base[0] = 5
        # 创建基础对象的视图
        fp_view = fp_base[0:1]
        # 断言视图对象的第一个元素的值与预期相同
        assert_equal(fp_view[0], 5)
        # 删除视图对象
        del fp_view
        # 在删除视图后仍然可以访问和赋值基础对象的元素
        assert_equal(fp_base[0], 5)
        fp_base[0] = 6
        assert_equal(fp_base[0], 6)

    # 测试算术操作是否会丢弃引用
    def test_arithmetic_drops_references(self):
        # 创建一个 memmap 对象，以写入模式打开，并使用指定的数据类型和形状
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+', shape=self.shape)
        # 对 memmap 对象执行加法操作
        tmp = (fp + 10)
        # 如果结果是 memmap 类型的对象，断言其底层 mmap 对象不是同一个引用
        if isinstance(tmp, memmap):
            assert_(tmp._mmap is not fp._mmap)

    # 测试索引操作是否会丢弃引用
    def test_indexing_drops_references(self):
        # 创建一个 memmap 对象，以写入模式打开，并使用指定的数据类型和形状
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+', shape=self.shape)
        # 对 memmap 对象执行索引操作
        tmp = fp[(1, 2), (2, 3)]
        # 如果结果是 memmap 类型的对象，断言其底层 mmap 对象不是同一个引用
        if isinstance(tmp, memmap):
            assert_(tmp._mmap is not fp._mmap)

    # 测试切片操作是否会保持引用
    def test_slicing_keeps_references(self):
        # 创建一个 memmap 对象，以写入模式打开，并使用指定的数据类型和形状
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+', shape=self.shape)
        # 断言 memmap 对象的切片操作后的 _mmap 属性与原对象的 _mmap 属性是同一个引用
        assert_(fp[:2, :2]._mmap is fp._mmap)

    # 测试视图创建
    def test_view(self):
        # 创建一个 memmap 对象，使用指定的数据类型和形状
        fp = memmap(self.tmpfp, dtype=self.dtype, shape=self.shape)
        # 创建 memmap 对象的视图
        new1 = fp.view()
        new2 = new1.view()
        # 断言新视图对象的 base 属性指向原始 memmap 对象
        assert_(new1.base is fp)
        assert_(new2.base is fp)
        # 使用 asarray 函数创建一个新的数组
        new_array = asarray(fp)
        # 断言新数组对象的 base 属性指向原始 memmap 对象
        assert_(new_array.base is fp)

    # 测试 ufunc 返回 ndarray
    def test_ufunc_return_ndarray(self):
        # 创建一个 memmap 对象，使用指定的数据类型和形状
        fp = memmap(self.tmpfp, dtype=self.dtype, shape=self.shape)
        # 将数据复制到 memmap 对象中
        fp[:] = self.data

        # 忽略特定的警告
        with suppress_warnings() as sup:
            sup.filter(FutureWarning, "np.average currently does not preserve")
            # 对于一元操作，如 sum, average, prod
            for unary_op in [sum, average, prod]:
                # 对 memmap 对象执行一元操作
                result = unary_op(fp)
                # 断言结果是标量
                assert_(isscalar(result))
                # 断言结果的类型与数据的第一个元素的类型相同
                assert_(result.__class__ is self.data[0, 0].__class__)

                # 对 memmap 对象沿着轴执行一元操作
                assert_(unary_op(fp, axis=0).__class__ is ndarray)
                assert_(unary_op(fp, axis=1).__class__ is ndarray)

        # 对于二元操作，如 add, subtract, multiply
        for binary_op in [add, subtract, multiply]:
            assert_(binary_op(fp, self.data).__class__ is ndarray)
            assert_(binary_op(self.data, fp).__class__ is ndarray)
            assert_(binary_op(fp, fp).__class__ is ndarray)

        # 在原 memmap 对象上执行加法操作
        fp += 1
        # 断言对象仍然是 memmap 类型
        assert(fp.__class__ is memmap)
        # 使用 add 函数在原 memmap 对象上执行加法操作
        add(fp, 1, out=fp)
        # 断言对象仍然是 memmap 类型
        assert(fp.__class__ is memmap)
    def test_getitem(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, shape=self.shape)
        fp[:] = self.data

        assert_(fp[1:, :-1].__class__ is memmap)
        # 使用切片索引返回的对象仍然是 memmap 类型的实例
        assert_(fp[[0, 1]].__class__ is ndarray)
        # 使用花式索引返回的对象是 ndarray 类型的实例

    def test_memmap_subclass(self):
        class MemmapSubClass(memmap):
            pass

        fp = MemmapSubClass(self.tmpfp, dtype=self.dtype, shape=self.shape)
        fp[:] = self.data

        # 对于 memmap 的子类，保持之前的行为，即 ufunc 和 __getitem__ 的输出不会转换为 ndarray 类型
        assert_(sum(fp, axis=0).__class__ is MemmapSubClass)
        assert_(sum(fp).__class__ is MemmapSubClass)
        assert_(fp[1:, :-1].__class__ is MemmapSubClass)
        assert(fp[[0, 1]].__class__ is MemmapSubClass)

    def test_mmap_offset_greater_than_allocation_granularity(self):
        size = 5 * mmap.ALLOCATIONGRANULARITY
        offset = mmap.ALLOCATIONGRANULARITY + 1
        fp = memmap(self.tmpfp, shape=size, mode='w+', offset=offset)
        assert_(fp.offset == offset)
        # 确保 memmap 对象的偏移量与预期一致

    def test_no_shape(self):
        self.tmpfp.write(b'a'*16)
        mm = memmap(self.tmpfp, dtype='float64')
        assert_equal(mm.shape, (2,))
        # 在没有明确指定形状的情况下，默认形状为 (2,)

    def test_empty_array(self):
        # gh-12653
        with pytest.raises(ValueError, match='empty file'):
            memmap(self.tmpfp, shape=(0,4), mode='w+')

        self.tmpfp.write(b'\0')

        # 现在文件不再为空
        memmap(self.tmpfp, shape=(0,4), mode='w+')
        # 创建形状为 (0, 4) 的 memmap 对象，不会触发异常

    def test_shape_type(self):
        memmap(self.tmpfp, shape=3, mode='w+')
        memmap(self.tmpfp, shape=self.shape, mode='w+')
        memmap(self.tmpfp, shape=list(self.shape), mode='w+')
        memmap(self.tmpfp, shape=asarray(self.shape), mode='w+')
        # 测试不同形状参数类型下创建 memmap 对象的行为
```