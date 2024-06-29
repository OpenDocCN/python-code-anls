# `.\numpy\numpy\_core\tests\test_records.py`

```
# 导入 collections.abc 模块，这是 Python 中抽象基类集合的一部分
import collections.abc
# 导入 textwrap 模块，用于文本的换行和填充
import textwrap
# 导入 BytesIO 类，用于在内存中读写二进制数据
from io import BytesIO
# 导入 path 函数，提供对文件路径操作的函数
from os import path
# 导入 Path 类，用于高级路径操作
from pathlib import Path
# 导入 pickle 模块，用于序列化和反序列化 Python 对象
import pickle

# 导入 pytest 模块，用于编写和运行测试用例
import pytest

# 导入 numpy 库
import numpy as np
# 从 numpy.testing 模块导入一些用于测试的函数和类
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_array_almost_equal,
    assert_raises, temppath,
    )


# 定义一个测试类 TestFromrecords
class TestFromrecords:
    # 定义测试方法 test_fromrecords
    def test_fromrecords(self):
        # 使用 np.rec.fromrecords 方法创建一个结构化数组 r
        r = np.rec.fromrecords([[456, 'dbe', 1.2], [2, 'de', 1.3]],
                            names='col1,col2,col3')
        # 断言结构化数组 r 的第一个元素
        assert_equal(r[0].item(), (456, 'dbe', 1.2))
        # 断言结构化数组 r 的 'col1' 列的数据类型是整数
        assert_equal(r['col1'].dtype.kind, 'i')
        # 断言结构化数组 r 的 'col2' 列的数据类型是 Unicode 字符串
        assert_equal(r['col2'].dtype.kind, 'U')
        # 断言结构化数组 r 的 'col2' 列的每个元素的大小为 12
        assert_equal(r['col2'].dtype.itemsize, 12)
        # 断言结构化数组 r 的 'col3' 列的数据类型是浮点数
        assert_equal(r['col3'].dtype.kind, 'f')

    # 定义测试方法 test_fromrecords_0len，测试空输入情况下 np.rec.fromrecords 方法的表现
    def test_fromrecords_0len(self):
        """ Verify fromrecords works with a 0-length input """
        # 定义结构化数组的 dtype
        dtype = [('a', float), ('b', float)]
        # 使用空列表创建结构化数组 r
        r = np.rec.fromrecords([], dtype=dtype)
        # 断言结构化数组 r 的形状为 (0,)
        assert_equal(r.shape, (0,))

    # 定义测试方法 test_fromrecords_2d，测试处理二维数据情况下 np.rec.fromrecords 方法的表现
    def test_fromrecords_2d(self):
        # 定义输入数据
        data = [
            [(1, 2), (3, 4), (5, 6)],
            [(6, 5), (4, 3), (2, 1)]
        ]
        # 期望的 'a' 列数据
        expected_a = [[1, 3, 5], [6, 4, 2]]
        # 期望的 'b' 列数据
        expected_b = [[2, 4, 6], [5, 3, 1]]

        # 使用指定 dtype 创建结构化数组 r1
        r1 = np.rec.fromrecords(data, dtype=[('a', int), ('b', int)])
        # 断言结构化数组 r1 的 'a' 列数据符合预期
        assert_equal(r1['a'], expected_a)
        # 断言结构化数组 r1 的 'b' 列数据符合预期
        assert_equal(r1['b'], expected_b)

        # 使用列名创建结构化数组 r2
        r2 = np.rec.fromrecords(data, names=['a', 'b'])
        # 断言结构化数组 r2 的 'a' 列数据符合预期
        assert_equal(r2['a'], expected_a)
        # 断言结构化数组 r2 的 'b' 列数据符合预期
        assert_equal(r2['b'], expected_b)

        # 断言结构化数组 r1 等于结构化数组 r2
        assert_equal(r1, r2)

    # 定义测试方法 test_method_array，测试 np.rec.array 方法的表现
    def test_method_array(self):
        # 使用 np.rec.array 方法创建结构化数组 r
        r = np.rec.array(
            b'abcdefg' * 100, formats='i2,S3,i4', shape=3, byteorder='big'
        )
        # 断言结构化数组 r 的第二个元素
        assert_equal(r[1].item(), (25444, b'efg', 1633837924))

    # 定义测试方法 test_method_array2，测试处理特定数据情况下 np.rec.array 方法的表现
    def test_method_array2(self):
        # 使用 np.rec.array 方法创建结构化数组 r
        r = np.rec.array(
            [
                (1, 11, 'a'), (2, 22, 'b'), (3, 33, 'c'), (4, 44, 'd'),
                (5, 55, 'ex'), (6, 66, 'f'), (7, 77, 'g')
            ],
            formats='u1,f4,S1'
        )
        # 断言结构化数组 r 的第二个元素
        assert_equal(r[1].item(), (2, 22.0, b'b'))

    # 定义测试方法 test_recarray_slices，测试对结构化数组切片的行为
    def test_recarray_slices(self):
        # 使用 np.rec.array 方法创建结构化数组 r
        r = np.rec.array(
            [
                (1, 11, 'a'), (2, 22, 'b'), (3, 33, 'c'), (4, 44, 'd'),
                (5, 55, 'ex'), (6, 66, 'f'), (7, 77, 'g')
            ],
            formats='u1,f4,S1'
        )
        # 断言对结构化数组 r 进行切片后的第二行元素
        assert_equal(r[1::2][1].item(), (4, 44.0, b'd'))

    # 定义测试方法 test_recarray_fromarrays，测试从数组创建结构化数组的表现
    def test_recarray_fromarrays(self):
        # 创建三个数组
        x1 = np.array([1, 2, 3, 4])
        x2 = np.array(['a', 'dd', 'xyz', '12'])
        x3 = np.array([1.1, 2, 3, 4])
        # 使用 np.rec.fromarrays 方法创建结构化数组 r
        r = np.rec.fromarrays([x1, x2, x3], names='a,b,c')
        # 断言结构化数组 r 的第二个元素
        assert_equal(r[1].item(), (2, 'dd', 2.0))
        # 修改 x1 的第二个元素
        x1[1] = 34
        # 断言结构化数组 r 的 'a' 列与原始数组 x1 相同
        assert_equal(r.a, np.array([1, 2, 3, 4]))
    # 定义测试函数，测试从文件中创建记录数组
    def test_recarray_fromfile(self):
        # 获取数据文件目录路径
        data_dir = path.join(path.dirname(__file__), 'data')
        # 获取文件完整路径
        filename = path.join(data_dir, 'recarray_from_file.fits')
        # 以二进制方式打开文件
        fd = open(filename, 'rb')
        # 移动文件指针到指定位置
        fd.seek(2880 * 2)
        # 从文件中读取数据创建记录数组
        r1 = np.rec.fromfile(fd, formats='f8,i4,S5', shape=3, byteorder='big')
        # 再次移动文件指针到指定位置
        fd.seek(2880 * 2)
        # 从文件中读取数据创建记录数组
        r2 = np.rec.array(fd, formats='f8,i4,S5', shape=3, byteorder='big')
        # 再次移动文件指针到指定位置
        fd.seek(2880 * 2)
        # 创建一个字节流对象
        bytes_array = BytesIO()
        # 将文件内容写入字节流
        bytes_array.write(fd.read())
        # 将字节流指针移动到起始位置
        bytes_array.seek(0)
        # 从字节流中读取数据创建记录数组
        r3 = np.rec.fromfile(
            bytes_array, formats='f8,i4,S5', shape=3, byteorder='big'
        )
        # 关闭文件
        fd.close()
        # 断言r1和r2的内容相等
        assert_equal(r1, r2)
        # 断言r2和r3的内容相等
        assert_equal(r2, r3)

    # 定义测试函数，测试从对象中创建记录数组
    def test_recarray_from_obj(self):
        # 计数值
        count = 10
        # 创建dtype为对象类型、值全为0的数组a
        a = np.zeros(count, dtype='O')
        # 创建dtype为浮点数类型、值全为0的数组b
        b = np.zeros(count, dtype='f8')
        # 创建dtype为浮点数类型、值全为0的数组c
        c = np.zeros(count, dtype='f8')
        # 循环将a数组的每个元素赋值为从1到9的列表
        for i in range(len(a)):
            a[i] = list(range(1, 10))
        # 从数组a、b和c创建记录数组mine
        mine = np.rec.fromarrays([a, b, c], names='date,data1,data2')
        # 循环验证记录数组mine中的数据
        for i in range(len(a)):
            assert_((mine.date[i] == list(range(1, 10))))
            assert_((mine.data1[i] == 0.0))
            assert_((mine.data2[i] == 0.0))

    # 定义测试函数，测试记录数组的显示
    def test_recarray_repr(self):
        # 创建元组数组a
        a = np.array([(1, 0.1), (2, 0.2)],
                     dtype=[('foo', '<i4'), ('bar', '<f8')])
        # 转换为记录数组
        a = np.rec.array(a)
        # 断言记录数组a的显示格式
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            rec.array([(1, 0.1), (2, 0.2)],
                      dtype=[('foo', '<i4'), ('bar', '<f8')])""")
        )
        # 创建dtype为浮点数类型、值全为1的数组a
        a = np.array(np.ones(4, dtype='f8'))
        # 断言将数组a转换为记录数组的显示格式
        assert_(repr(np.rec.array(a)).startswith('rec.array'))
        # 创建dtype为整数和整数类型的数组a
        a = np.rec.array(np.ones(3, dtype='i4,i4'))
        # 断言将数组a转换为记录数组的显示格式
        assert_equal(repr(a).find('numpy.record'), -1)
        # 创建dtype为整数类型的数组a
        a = np.rec.array(np.ones(3, dtype='i4'))
        # 断言将数组a转换为记录数组的显示格式
        assert_(repr(a).find('dtype=int32') != -1)

    # 定义测试函数，测试0维记录数组的显示
    def test_0d_recarray_repr(self):
        # 创建0维记录数组arr_0d
        arr_0d = np.rec.array((1, 2.0, '2003'), dtype='<i4,<f8,<M8[Y]')
        # 断言0维记录数组arr_0d的显示格式
        assert_equal(repr(arr_0d), textwrap.dedent("""\
            rec.array((1, 2., '2003'),
                      dtype=[('f0', '<i4'), ('f1', '<f8'), ('f2', '<M8[Y]')])"""))
        # 获取0维记录数组的记录
        record = arr_0d[()]
        # 断言获取的记录的显示格式
        assert_equal(repr(record), 
            "np.record((1, 2.0, '2003'), "
            "dtype=[('f0', '<i4'), ('f1', '<f8'), ('f2', '<M8[Y]')])")
        # 断言设置为1.13时的记录显示格式
        try:
            np.set_printoptions(legacy='1.13')
            assert_equal(repr(record), '(1, 2.0, datetime.date(2003, 1, 1))')
        finally:
            np.set_printoptions(legacy=False)
    # 定义一个测试方法，测试从字符串表示形式创建的结构化数组
    def test_recarray_from_repr(self):
        # 创建一个包含两个元组的 numpy 数组 a，每个元组包含一个整数和一个字符串
        a = np.array([(1,'ABC'), (2, "DEF")],
                     dtype=[('foo', int), ('bar', 'S4')])
        # 使用 np.rec.array 将数组 a 转换为结构化数组 recordarr
        recordarr = np.rec.array(a)
        # 将数组 a 视图转换为结构化数组 recarr
        recarr = a.view(np.recarray)
        # 将数组 a 视图转换为具有记录 dtype 的 numpy 数组 recordview
        recordview = a.view(np.dtype((np.record, a.dtype)))

        # 通过 eval 函数将字符串表示形式的 recordarr 转换回 numpy 结构化数组
        recordarr_r = eval("np." + repr(recordarr), {'np': np})
        # 通过 eval 函数将字符串表示形式的 recarr 转换回 numpy 结构化数组
        recarr_r = eval("np." + repr(recarr), {'np': np})
        # 通过 eval 函数将字符串表示形式的 recordview 转换回 numpy 数组，保留记录 dtype
        recordview_r = eval("np." + repr(recordview), {'np': np, 'numpy': np})

        # 断言转换后的 recordarr_r 的类型为 np.recarray
        assert_equal(type(recordarr_r), np.recarray)
        # 断言转换后的 recordarr_r 的 dtype 的类型为 np.record
        assert_equal(recordarr_r.dtype.type, np.record)
        # 断言转换后的 recordarr_r 等于原始的 recordarr

        # 断言转换后的 recarr_r 的类型为 np.recarray
        assert_equal(type(recarr_r), np.recarray)
        # 断言转换后的 recarr_r 的 dtype 的类型为 np.record
        assert_equal(recarr_r.dtype.type, np.record)
        # 断言转换后的 recarr_r 等于原始的 recarr

        # 断言转换后的 recordview_r 的类型为 np.ndarray
        assert_equal(type(recordview_r), np.ndarray)
        # 断言转换后的 recordview 的 dtype 的类型为 np.record
        assert_equal(recordview.dtype.type, np.record)
        # 断言转换后的 recordview_r 等于原始的 recordview
    # 定义一个测试方法，用于测试记录数组的视图操作
    def test_recarray_views(self):
        # 创建一个结构化数组a，包含两个元组
        a = np.array([(1,'ABC'), (2, "DEF")],
                     dtype=[('foo', int), ('bar', 'S4')])
        # 创建一个普通数组b，包含整数值
        b = np.array([1,2,3,4,5], dtype=np.int64)

        # 检查 np.rec.array 是否返回正确的数据类型
        assert_equal(np.rec.array(a).dtype.type, np.record)
        # 检查 np.rec.array 是否返回 np.recarray 类型
        assert_equal(type(np.rec.array(a)), np.recarray)
        # 检查 np.rec.array 是否返回正确的数据类型
        assert_equal(np.rec.array(b).dtype.type, np.int64)
        # 检查 np.rec.array 是否返回 np.recarray 类型
        assert_equal(type(np.rec.array(b)), np.recarray)

        # 检查将数组a视图为 np.recarray 是否产生相同的结果
        assert_equal(a.view(np.recarray).dtype.type, np.record)
        # 检查将数组a视图为 np.recarray 是否返回 np.recarray 类型
        assert_equal(type(a.view(np.recarray)), np.recarray)
        # 检查将数组b视图为 np.recarray 是否产生相同的结果
        assert_equal(b.view(np.recarray).dtype.type, np.int64)
        # 检查将数组b视图为 np.recarray 是否返回 np.recarray 类型
        assert_equal(type(b.view(np.recarray)), np.recarray)

        # 检查视图转换到非结构化数据类型时是否保留 np.recarray 类型
        r = np.rec.array(np.ones(4, dtype="f4,i4"))
        rv = r.view('f8').view('f4,i4')
        assert_equal(type(rv), np.recarray)
        assert_equal(rv.dtype.type, np.record)

        # 检查通过索引访问是否保留 np.recarray 和 np.record 类型
        r = np.rec.array(np.ones(4, dtype=[('a', 'i4'), ('b', 'i4'),
                                           ('c', 'i4,i4')]))
        assert_equal(r['c'].dtype.type, np.record)
        assert_equal(type(r['c']), np.recarray)

        # 检查是否保留子类 (gh-6949)
        class C(np.recarray):
            pass

        c = r.view(C)
        assert_equal(type(c['c']), C)

        # 检查访问嵌套结构时是否保留记录类型，但对于子数组、非空结构、非结构化空类型不保留
        test_dtype = [('a', 'f4,f4'), ('b', 'V8'), ('c', ('f4',2)),
                      ('d', ('i8', 'i4,i4'))]
        r = np.rec.array([((1,1), b'11111111', [1,1], 1),
                          ((1,1), b'11111111', [1,1], 1)], dtype=test_dtype)
        assert_equal(r.a.dtype.type, np.record)
        assert_equal(r.b.dtype.type, np.void)
        assert_equal(r.c.dtype.type, np.float32)
        assert_equal(r.d.dtype.type, np.int64)
        
        # 检查相同操作在视图中是否保持一致
        r = np.rec.array(np.ones(4, dtype='i4,i4'))
        assert_equal(r.view('f4,f4').dtype.type, np.record)
        assert_equal(r.view(('i4',2)).dtype.type, np.int32)
        assert_equal(r.view('V8').dtype.type, np.void)
        assert_equal(r.view(('i8', 'i4,i4')).dtype.type, np.int64)

        # 检查能否撤销视图操作
        arrs = [np.ones(4, dtype='f4,i4'), np.ones(4, dtype='f8')]
        for arr in arrs:
            rec = np.rec.array(arr)
            # 建议的视图为 ndarray 的方法：
            arr2 = rec.view(rec.dtype.fields or rec.dtype, np.ndarray)
            assert_equal(arr2.dtype.type, arr.dtype.type)
            assert_equal(type(arr2), type(arr))
    # 定义测试函数，用于测试从给定字段名创建的记录数组
    def test_recarray_from_names(self):
        # 创建具有指定字段名的记录数组 ra
        ra = np.rec.array([
            (1, 'abc', 3.7000002861022949, 0),
            (2, 'xy', 6.6999998092651367, 1),
            (0, ' ', 0.40000000596046448, 0)],
            names='c1, c2, c3, c4')
        # 使用给定记录列表创建记录数组 pa
        pa = np.rec.fromrecords([
            (1, 'abc', 3.7000002861022949, 0),
            (2, 'xy', 6.6999998092651367, 1),
            (0, ' ', 0.40000000596046448, 0)],
            names='c1, c2, c3, c4')
        # 断言 ra 的数据类型与 pa 的数据类型相等
        assert_(ra.dtype == pa.dtype)
        # 断言 ra 的形状与 pa 的形状相等
        assert_(ra.shape == pa.shape)
        # 遍历 ra 的每一行，断言每一行的条目与 pa 中对应行的条目相等
        for k in range(len(ra)):
            assert_(ra[k].item() == pa[k].item())

    # 定义测试函数，用于测试记录数组中字段冲突的情况
    def test_recarray_conflict_fields(self):
        # 创建具有字段 'field', 'shape', 'mean' 的记录数组 ra
        ra = np.rec.array([(1, 'abc', 2.3), (2, 'xyz', 4.2),
                           (3, 'wrs', 1.3)],
                          names='field, shape, mean')
        # 将 ra 的 mean 属性设置为指定列表
        ra.mean = [1.1, 2.2, 3.3]
        # 断言 ra 的 'mean' 字段几乎等于指定列表
        assert_array_almost_equal(ra['mean'], [1.1, 2.2, 3.3])
        # 断言 ra.mean 的类型与 ra.var 的类型相同
        assert_(type(ra.mean) is type(ra.var))
        # 修改 ra 的形状为 (1, 3)
        ra.shape = (1, 3)
        # 断言 ra 的形状确实为 (1, 3)
        assert_(ra.shape == (1, 3))
        # 修改 ra 的形状为指定列表
        ra.shape = ['A', 'B', 'C']
        # 断言 ra 的 'shape' 字段等于指定二维列表
        assert_array_equal(ra['shape'], [['A', 'B', 'C']])
        # 将 ra 的 'field' 字段设置为指定值
        ra.field = 5
        # 断言 ra 的 'field' 字段等于指定二维数组
        assert_array_equal(ra['field'], [[5, 5, 5]])
        # 断言 ra.field 的类型为 collections.abc.Callable 的实例
        assert_(isinstance(ra.field, collections.abc.Callable))

    # 定义测试函数，用于测试从记录列表创建记录数组并指定数据类型的情况
    def test_fromrecords_with_explicit_dtype(self):
        # 使用指定数据类型创建记录数组 a
        a = np.rec.fromrecords([(1, 'a'), (2, 'bbb')],
                               dtype=[('a', int), ('b', object)])
        # 断言记录数组 a 的 'a' 字段等于指定列表
        assert_equal(a.a, [1, 2])
        # 断言记录数组 a 的第一个元素的 'a' 字段等于指定值
        assert_equal(a[0].a, 1)
        # 断言记录数组 a 的 'b' 字段等于指定列表
        assert_equal(a.b, ['a', 'bbb'])
        # 断言记录数组 a 的最后一个元素的 'b' 字段等于指定值
        assert_equal(a[-1].b, 'bbb')
        #
        # 使用指定的 numpy 数据类型创建记录数组 a
        ndtype = np.dtype([('a', int), ('b', object)])
        a = np.rec.fromrecords([(1, 'a'), (2, 'bbb')], dtype=ndtype)
        # 断言记录数组 a 的 'a' 字段等于指定列表
        assert_equal(a.a, [1, 2])
        # 断言记录数组 a 的第一个元素的 'a' 字段等于指定值
        assert_equal(a[0].a, 1)
        # 断言记录数组 a 的 'b' 字段等于指定列表
        assert_equal(a.b, ['a', 'bbb'])
        # 断言记录数组 a 的最后一个元素的 'b' 字段等于指定值
        assert_equal(a[-1].b, 'bbb')

    # 定义测试函数，用于测试字符串类型的记录数组创建
    def test_recarray_stringtypes(self):
        # 创建具有字段 'foo' 和 'bar' 的字符串类型的记录数组 a
        a = np.array([('abc ', 1), ('abc', 2)],
                     dtype=[('foo', 'S4'), ('bar', int)])
        # 将 ndarray 转换为 recarray 类型
        a = a.view(np.recarray)
        # 断言记录数组 a 的 'foo' 字段的第一个元素不等于 'foo' 字段的第二个元素
        assert_equal(a.foo[0] == a.foo[1], False)
    def test_recarray_returntypes(self):
        # 定义一个包含结构化数据的 numpy 递归数组
        qux_fields = {'C': (np.dtype('S5'), 0), 'D': (np.dtype('S5'), 6)}
        a = np.rec.array([('abc ', (1,1), 1, ('abcde', 'fgehi')),
                          ('abc', (2,3), 1, ('abcde', 'jklmn'))],
                         dtype=[('foo', 'S4'),
                                ('bar', [('A', int), ('B', int)]),
                                ('baz', int), ('qux', qux_fields)])
        
        # 断言各属性的返回类型
        assert_equal(type(a.foo), np.ndarray)
        assert_equal(type(a['foo']), np.ndarray)
        assert_equal(type(a.bar), np.recarray)
        assert_equal(type(a['bar']), np.recarray)
        
        # 检查 bar 属性的数据类型是记录数组
        assert_equal(a.bar.dtype.type, np.record)
        
        # 检查 qux 属性及其字段是否与预定义的 qux_fields 字典匹配
        assert_equal(type(a['qux']), np.recarray)
        assert_equal(a.qux.dtype.type, np.record)
        assert_equal(dict(a.qux.dtype.fields), qux_fields)
        
        # 检查 baz 属性的返回类型
        assert_equal(type(a.baz), np.ndarray)
        assert_equal(type(a['baz']), np.ndarray)
        
        # 检查索引访问方式下 bar 属性的数据类型是记录
        assert_equal(type(a[0].bar), np.record)
        assert_equal(type(a[0]['bar']), np.record)
        
        # 访问特定字段及其值，确保值的正确性
        assert_equal(a[0].bar.A, 1)
        assert_equal(a[0].bar['A'], 1)
        assert_equal(a[0]['bar'].A, 1)
        assert_equal(a[0]['bar']['A'], 1)
        
        # 访问 qux 属性中特定字段 D，并检查其值是否正确
        assert_equal(a[0].qux.D, b'fgehi')
        assert_equal(a[0].qux['D'], b'fgehi')
        assert_equal(a[0]['qux'].D, b'fgehi')
        assert_equal(a[0]['qux']['D'], b'fgehi')
# 定义一个测试类 TestPathUsage，用于测试 pathlib.Path 的使用
class TestPathUsage:
    
    # 定义测试方法 test_tofile_fromfile，测试将数组写入文件并从文件读取
    def test_tofile_fromfile(self):
        # 使用 temppath 创建临时文件路径，文件名后缀为 '.bin'
        with temppath(suffix='.bin') as path:
            # 将路径对象转换为 pathlib.Path 对象
            path = Path(path)
            
            # 设定随机数种子为 123
            np.random.seed(123)
            
            # 创建一个包含随机数据的数组，数据类型包括 float64、int32、字符串长度为 5
            a = np.random.rand(10).astype('f8,i4,S5')
            a[5] = (0.5, 10, 'abcde')
            
            # 以二进制写入方式打开路径对应的文件
            with path.open("wb") as fd:
                # 将数组 a 写入文件
                a.tofile(fd)
            
            # 从文件中读取指定格式的数据为数组 x
            x = np._core.records.fromfile(
                path, formats='f8,i4,S5', shape=10
            )
            
            # 断言数组 x 的内容与数组 a 相等
            assert_array_equal(x, a)


# 定义测试类 TestRecord，用于测试记录数组的操作
class TestRecord:
    
    # 在每个测试方法运行前设置测试数据
    def setup_method(self):
        self.data = np.rec.fromrecords([(1, 2, 3), (4, 5, 6)],
                            dtype=[("col1", "<i4"),
                                   ("col2", "<i4"),
                                   ("col3", "<i4")])
    
    # 测试记录数组的元素赋值操作
    def test_assignment1(self):
        a = self.data
        # 断言记录数组的第一个元素的 col1 字段为 1
        assert_equal(a.col1[0], 1)
        # 修改记录数组的第一个元素的 col1 字段为 0
        a[0].col1 = 0
        # 断言记录数组的第一个元素的 col1 字段已修改为 0
        assert_equal(a.col1[0], 0)
    
    # 测试记录数组的字段直接赋值操作
    def test_assignment2(self):
        a = self.data
        # 断言记录数组的第一个元素的 col1 字段为 1
        assert_equal(a.col1[0], 1)
        # 直接修改记录数组的第一个元素的 col1 字段为 0
        a.col1[0] = 0
        # 断言记录数组的第一个元素的 col1 字段已修改为 0
        assert_equal(a.col1[0], 0)
    
    # 测试尝试对记录数组中不存在的字段进行赋值操作
    def test_invalid_assignment(self):
        a = self.data
        
        def assign_invalid_column(x):
            # 尝试给记录数组中不存在的 col5 字段赋值
            x[0].col5 = 1
        
        # 断言会抛出 AttributeError 异常
        assert_raises(AttributeError, assign_invalid_column, a)
    
    # 测试记录数组字段不可写时的 setfield 操作
    def test_nonwriteable_setfield(self):
        # 创建一个记录数组 r，包含两个元素，每个元素有一个整数字段 'f'
        r = np.rec.array([(0,), (1,)], dtype=[('f', 'i4')])
        
        # 设置记录数组为不可写
        r.flags.writeable = False
        
        # 断言使用赋值方式修改字段值会引发 ValueError 异常
        with assert_raises(ValueError):
            r.f = [2, 3]
        
        # 断言使用 setfield 方法修改字段值会引发 ValueError 异常
        with assert_raises(ValueError):
            r.setfield([2,3], *r.dtype.fields['f'])
    
    # 测试记录数组选择字段时的 dtype 结构
    def test_out_of_order_fields(self):
        # 选择记录数组中的 'col1' 和 'col2' 字段，创建新的数组 x
        x = self.data[['col1', 'col2']]
        
        # 断言新数组 x 的字段名为 ('col1', 'col2')
        assert_equal(x.dtype.names, ('col1', 'col2'))
        
        # 断言新数组 x 的 dtype 描述结构为 [('col1', '<i4'), ('col2', '<i4'), ('', '|V4')]
        assert_equal(x.dtype.descr,
                     [('col1', '<i4'), ('col2', '<i4'), ('', '|V4')])
        
        # 将字段名顺序调整为与索引顺序一致，作为 1.14 版本后的索引，预期引发 ValueError 异常
        y = self.data[['col2', 'col1']]
        assert_equal(y.dtype.names, ('col2', 'col1'))
        assert_raises(ValueError, lambda: y.dtype.descr)
    
    # 测试记录数组的序列化和反序列化
    def test_pickle_1(self):
        # 创建一个具有一个整数字段 'a' 和一个空数组字段 'b' 的数组 a
        a = np.array([(1, [])], dtype=[('a', np.int32), ('b', np.int32, 0)])
        
        # 针对所有协议版本进行序列化和反序列化的测试
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            assert_equal(a, pickle.loads(pickle.dumps(a, protocol=proto)))
            assert_equal(a[0], pickle.loads(pickle.dumps(a[0],
                                                         protocol=proto)))
    
    # 测试记录数组对象的序列化和反序列化
    def test_pickle_2(self):
        a = self.data
        # 针对所有协议版本进行记录数组对象的序列化和反序列化的测试
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            assert_equal(a, pickle.loads(pickle.dumps(a, protocol=proto)))
            assert_equal(a[0], pickle.loads(pickle.dumps(a[0],
                                                         protocol=proto)))
    def test_pickle_3(self):
        # Issue #7140
        # 从 self.data 中获取数据
        a = self.data
        # 遍历从协议版本 2 到最高协议版本的范围
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            # 使用指定协议序列化和反序列化对象 a[0]
            pa = pickle.loads(pickle.dumps(a[0], protocol=proto))
            # 断言 pa 是 C 连续存储的
            assert_(pa.flags.c_contiguous)
            # 断言 pa 是 Fortran 连续存储的
            assert_(pa.flags.f_contiguous)
            # 断言 pa 是可写的
            assert_(pa.flags.writeable)
            # 断言 pa 是对齐的
            assert_(pa.flags.aligned)

    def test_pickle_void(self):
        # issue gh-13593
        # 定义一个复杂的数据类型 dt
        dt = np.dtype([('obj', 'O'), ('int', 'i')])
        # 创建一个空数组 a，使用复杂数据类型 dt
        a = np.empty(1, dtype=dt)
        # 创建一个数据对象 data，是一个元组，包含一个字节数组
        data = (bytearray(b'eman'),)
        # 将 data 赋值给 a 的 'obj' 字段
        a['obj'] = data
        # 将整数值 42 赋值给 a 的 'int' 字段
        a['int'] = 42
        # 获取 a[0] 的 __reduce__ 方法的返回值
        ctor, args = a[0].__reduce__()
        # 断言构造函数是 np._core.multiarray.scalar
        assert ctor is np._core.multiarray.scalar
        # 解析参数之前检查构造函数是否符合预期
        dtype, obj = args
        # 确保未序列化地址信息
        assert not isinstance(obj, bytes)
        
        # 断言调用构造函数会引发 RuntimeError 异常
        assert_raises(RuntimeError, ctor, dtype, 13)
        
        # 测试序列化和反序列化的往返
        dump = pickle.dumps(a[0])
        unpickled = pickle.loads(dump)
        assert a[0] == unpickled
        
        # 还要检查类似但不可能的 "object scalar" 路径
        with pytest.warns(DeprecationWarning):
            assert ctor(np.dtype("O"), data) is data

    def test_objview_record(self):
        # https://github.com/numpy/numpy/issues/2599
        # 定义一个复杂的数据类型 dt
        dt = np.dtype([('foo', 'i8'), ('bar', 'O')])
        # 创建一个形状为 (1, 3) 的全零数组，使用复杂数据类型 dt，再视图为结构化数组
        r = np.zeros((1, 3), dtype=dt).view(np.recarray)
        # 设置 r 的 'foo' 字段为数组 [1, 2, 3]，可能会引发 TypeError
        r.foo = np.array([1, 2, 3])  # TypeError?

        # https://github.com/numpy/numpy/issues/3256
        # 创建一个结构化数组 ra，包含 2 个元素，数据类型是 [('x', object), ('y', float), ('z', int)]
        ra = np.recarray(
            (2,), dtype=[('x', object), ('y', float), ('z', int)]
        )
        # 尝试访问 ra 的特定字段数组 ['x','y']，可能会引发 TypeError
        ra[['x','y']]  # TypeError?

    def test_record_scalar_setitem(self):
        # https://github.com/numpy/numpy/issues/3561
        # 创建一个结构化数组 rec，包含 1 个元素，数据类型是 [('x', float, 5)]
        rec = np.recarray(1, dtype=[('x', float, 5)])
        # 设置 rec[0] 的 'x' 字段为 1，期望其等于全为 1 的数组
        rec[0].x = 1
        assert_equal(rec[0].x, np.ones(5))

    def test_missing_field(self):
        # https://github.com/numpy/numpy/issues/4806
        # 创建一个全零数组 arr，形状为 (3,)，数据类型为 [('x', int), ('y', int)]
        arr = np.zeros((3,), dtype=[('x', int), ('y', int)])
        # 尝试访问 arr 的不存在的字段 'nofield'，期望引发 KeyError 异常
        assert_raises(KeyError, lambda: arr[['nofield']])

    def test_fromarrays_nested_structured_arrays(self):
        # 创建一个数组列表 arrays，包含两个元素：一个范围为 [0, 9] 的整数数组和一个结构化数组
        arrays = [
            np.arange(10),
            np.ones(10, dtype=[('a', '<u2'), ('b', '<f4')]),
        ]
        # 使用 fromarrays 方法创建一个结构化数组 arr，可能会引发 ValueError
        arr = np.rec.fromarrays(arrays)  # ValueError?

    @pytest.mark.parametrize('nfields', [0, 1, 2])
    def test_assign_dtype_attribute(self, nfields):
        # 创建一个复杂的数据类型 dt，包含从 [('a', np.uint8), ('b', np.uint8), ('c', np.uint8)] 切片得到的字段
        dt = np.dtype([('a', np.uint8), ('b', np.uint8), ('c', np.uint8)][:nfields])
        # 创建一个全零数组 data，形状为 (3,)，使用复杂数据类型 dt，再视图为结构化数组
        data = np.zeros(3, dt).view(np.recarray)

        # 确保原始和结果数据类型的类型都是记录型
        assert data.dtype.type == np.record
        assert dt.type != np.record

        # 确保即使被赋予不同的数据类型，data 的数据类型仍然是记录型
        data.dtype = dt
        assert data.dtype.type == np.record
    def test_nested_fields_are_records(self, nfields):
        """ Test that nested structured types are treated as records too """
        # 创建一个包含指定字段的数据类型描述符
        dt = np.dtype([('a', np.uint8), ('b', np.uint8), ('c', np.uint8)][:nfields])
        # 创建一个外层数据类型描述符，其中包含一个内部结构类型
        dt_outer = np.dtype([('inner', dt)])

        # 使用外层数据类型描述符创建一个全零数组，并转换为记录数组
        data = np.zeros(3, dt_outer).view(np.recarray)
        # 断言数据对象是 np.recarray 类型
        assert isinstance(data, np.recarray)
        # 断言数据对象的 'inner' 字段是 np.recarray 类型
        assert isinstance(data['inner'], np.recarray)

        # 获取数组中的第一个元素
        data0 = data[0]
        # 断言第一个元素是 np.record 类型
        assert isinstance(data0, np.record)
        # 断言第一个元素的 'inner' 字段是 np.record 类型
        assert isinstance(data0['inner'], np.record)

    def test_nested_dtype_padding(self):
        """ test that trailing padding is preserved """
        # 构建一个具有尾部填充的数据类型描述符
        dt = np.dtype([('a', np.uint8), ('b', np.uint8), ('c', np.uint8)])
        dt_padded_end = dt[['a', 'b']]
        # 断言尾部填充后的数据类型的字节大小与原始数据类型相同
        assert dt_padded_end.itemsize == dt.itemsize

        # 创建一个外层数据类型描述符，其中包含一个尾部填充后的数据类型
        dt_outer = np.dtype([('inner', dt_padded_end)])

        # 使用外层数据类型描述符创建一个全零数组，并转换为记录数组
        data = np.zeros(3, dt_outer).view(np.recarray)
        # 断言 'inner' 字段的数据类型与尾部填充后的数据类型相同
        assert_equal(data['inner'].dtype, dt_padded_end)

        # 获取数组中的第一个元素
        data0 = data[0]
        # 断言第一个元素的 'inner' 字段的数据类型与尾部填充后的数据类型相同
        assert_equal(data0['inner'].dtype, dt_padded_end)
# 定义一个测试函数，用于测试查找列表中重复元素的函数 np.rec.find_duplicate()
def test_find_duplicate():
    # 创建一个包含整数的列表 l1
    l1 = [1, 2, 3, 4, 5, 6]
    # 断言调用 np.rec.find_duplicate(l1) 返回一个空列表
    assert_(np.rec.find_duplicate(l1) == [])

    # 创建一个包含整数的列表 l2，其中有重复元素 1
    l2 = [1, 2, 1, 4, 5, 6]
    # 断言调用 np.rec.find_duplicate(l2) 返回包含元素 1 的列表
    assert_(np.rec.find_duplicate(l2) == [1])

    # 创建一个包含整数的列表 l3，其中有多个重复元素 1 和 2
    l3 = [1, 2, 1, 4, 1, 6, 2, 3]
    # 断言调用 np.rec.find_duplicate(l3) 返回包含元素 1 和 2 的列表
    assert_(np.rec.find_duplicate(l3) == [1, 2])

    # 修改列表 l3，使其包含多个重复元素 2 和 1
    l3 = [2, 2, 1, 4, 1, 6, 2, 3]
    # 断言调用 np.rec.find_duplicate(l3) 返回包含元素 2 和 1 的列表
    assert_(np.rec.find_duplicate(l3) == [2, 1])
```