# `.\numpy\numpy\lib\tests\test_recfunctions.py`

```py
# 导入pytest模块，用于测试和断言
import pytest

# 导入NumPy库，并将其命名为np，用于数组操作
import numpy as np

# 导入NumPy的masked array模块，用于处理带有掩码的数组
import numpy.ma as ma

# 导入NumPy的masked records模块，用于操作带有掩码的记录数组
from numpy.ma.mrecords import MaskedRecords

# 导入NumPy的测试工具，包括断言函数assert_equal
from numpy.ma.testutils import assert_equal

# 导入NumPy的测试工具，包括assert_和assert_raises函数
from numpy.testing import assert_, assert_raises

# 导入NumPy的记录数组函数模块，包括字段操作和结构操作
from numpy.lib.recfunctions import (
    drop_fields, rename_fields, get_fieldstructure, recursive_fill_fields,
    find_duplicates, merge_arrays, append_fields, stack_arrays, join_by,
    repack_fields, unstructured_to_structured, structured_to_unstructured,
    apply_along_fields, require_fields, assign_fields_by_name)

# 导入NumPy的记录数组字段操作辅助函数
get_fieldspec = np.lib.recfunctions._get_fieldspec
get_names = np.lib.recfunctions.get_names
get_names_flat = np.lib.recfunctions.get_names_flat

# 导入NumPy的记录数组描述和数据类型压缩函数
zip_descr = np.lib.recfunctions._zip_descr
zip_dtype = np.lib.recfunctions._zip_dtype

# 定义测试类TestRecFunctions，用于测试记录数组相关函数
class TestRecFunctions:
    
    # 测试准备方法，初始化测试数据
    def setup_method(self):
        x = np.array([1, 2, ])
        y = np.array([10, 20, 30])
        z = np.array([('A', 1.), ('B', 2.)],
                     dtype=[('A', '|S3'), ('B', float)])
        w = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
                     dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])
        self.data = (w, x, y, z)

    # 测试zip_descr函数
    def test_zip_descr(self):
        # 从self.data中获取测试数据
        (w, x, y, z) = self.data
        
        # 测试标准数组的zip_descr函数调用，使用flatten=True
        test = zip_descr((x, x), flatten=True)
        assert_equal(test,
                     np.dtype([('', int), ('', int)]))
        
        # 再次测试标准数组的zip_descr函数调用，使用flatten=False
        test = zip_descr((x, x), flatten=False)
        assert_equal(test,
                     np.dtype([('', int), ('', int)]))
        
        # 测试标准数组和灵活数据类型数组的zip_descr函数调用，使用flatten=True
        test = zip_descr((x, z), flatten=True)
        assert_equal(test,
                     np.dtype([('', int), ('A', '|S3'), ('B', float)]))
        
        # 再次测试标准数组和灵活数据类型数组的zip_descr函数调用，使用flatten=False
        test = zip_descr((x, z), flatten=False)
        assert_equal(test,
                     np.dtype([('', int),
                               ('', [('A', '|S3'), ('B', float)])]))
        
        # 测试标准数组和嵌套数据类型数组的zip_descr函数调用，使用flatten=True
        test = zip_descr((x, w), flatten=True)
        assert_equal(test,
                     np.dtype([('', int),
                               ('a', int),
                               ('ba', float), ('bb', int)]))
        
        # 再次测试标准数组和嵌套数据类型数组的zip_descr函数调用，使用flatten=False
        test = zip_descr((x, w), flatten=False)
        assert_equal(test,
                     np.dtype([('', int),
                               ('', [('a', int),
                                     ('b', [('ba', float), ('bb', int)])])]))
    def test_drop_fields(self):
        # Test drop_fields
        # 创建一个 NumPy 数组，包含复合数据类型，具有基本和嵌套字段
        a = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
                     dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])

        # 测试删除指定字段后的结果
        test = drop_fields(a, 'a')
        # 预期的结果数组，仅包含剩余字段
        control = np.array([((2, 3.0),), ((5, 6.0),)],
                           dtype=[('b', [('ba', float), ('bb', int)])])
        assert_equal(test, control)

        # 测试删除另一个字段后的结果（基本字段，但包含两个字段的嵌套）
        test = drop_fields(a, 'b')
        # 预期的结果数组，仅包含剩余字段
        control = np.array([(1,), (4,)], dtype=[('a', int)])
        assert_equal(test, control)

        # 测试删除嵌套子字段后的结果
        test = drop_fields(a, ['ba', ])
        # 预期的结果数组，仅包含剩余字段
        control = np.array([(1, (3.0,)), (4, (6.0,))],
                           dtype=[('a', int), ('b', [('bb', int)])])
        assert_equal(test, control)

        # 测试删除一个字段的所有嵌套子字段后的结果
        test = drop_fields(a, ['ba', 'bb'])
        # 预期的结果数组，仅包含剩余字段
        control = np.array([(1,), (4,)], dtype=[('a', int)])
        assert_equal(test, control)

        # 测试删除所有字段后的结果
        test = drop_fields(a, ['a', 'b'])
        # 预期的结果数组，不包含任何字段
        control = np.array([(), ()], dtype=[])
        assert_equal(test, control)

    def test_rename_fields(self):
        # Test rename fields
        # 创建一个 NumPy 数组，包含复合数据类型，具有基本和嵌套字段
        a = np.array([(1, (2, [3.0, 30.])), (4, (5, [6.0, 60.]))],
                     dtype=[('a', int),
                            ('b', [('ba', float), ('bb', (float, 2))])])
        # 测试重命名字段后的结果
        test = rename_fields(a, {'a': 'A', 'bb': 'BB'})
        # 期望的新数据类型
        newdtype = [('A', int), ('b', [('ba', float), ('BB', (float, 2))])]
        # 期望的控制数组视图
        control = a.view(newdtype)
        assert_equal(test.dtype, newdtype)
        assert_equal(test, control)

    def test_get_names(self):
        # Test get_names
        # 创建一个 NumPy 数据类型对象，其中包含命名字段
        ndtype = np.dtype([('A', '|S3'), ('B', float)])
        # 测试获取字段名称的结果
        test = get_names(ndtype)
        assert_equal(test, ('A', 'B'))

        ndtype = np.dtype([('a', int), ('b', [('ba', float), ('bb', int)])])
        test = get_names(ndtype)
        assert_equal(test, ('a', ('b', ('ba', 'bb'))))

        ndtype = np.dtype([('a', int), ('b', [])])
        test = get_names(ndtype)
        assert_equal(test, ('a', ('b', ())))

        ndtype = np.dtype([])
        test = get_names(ndtype)
        assert_equal(test, ())

    def test_get_names_flat(self):
        # Test get_names_flat
        # 创建一个 NumPy 数据类型对象，其中包含命名字段
        ndtype = np.dtype([('A', '|S3'), ('B', float)])
        # 测试获取扁平化字段名称的结果
        test = get_names_flat(ndtype)
        assert_equal(test, ('A', 'B'))

        ndtype = np.dtype([('a', int), ('b', [('ba', float), ('bb', int)])])
        test = get_names_flat(ndtype)
        assert_equal(test, ('a', 'b', 'ba', 'bb'))

        ndtype = np.dtype([('a', int), ('b', [])])
        test = get_names_flat(ndtype)
        assert_equal(test, ('a', 'b'))

        ndtype = np.dtype([])
        test = get_names_flat(ndtype)
        assert_equal(test, ())
    def test_get_fieldstructure(self):
        # Test get_fieldstructure

        # No nested fields
        ndtype = np.dtype([('A', '|S3'), ('B', float)])
        # 调用 get_fieldstructure 函数，传入 dtype 对象 ndtype
        test = get_fieldstructure(ndtype)
        # 断言结果与预期的空字典匹配
        assert_equal(test, {'A': [], 'B': []})

        # One 1-nested field
        ndtype = np.dtype([('A', int), ('B', [('BA', float), ('BB', '|S1')])])
        # 调用 get_fieldstructure 函数，传入 dtype 对象 ndtype
        test = get_fieldstructure(ndtype)
        # 断言结果与预期的字段结构字典匹配
        assert_equal(test, {'A': [], 'B': [], 'BA': ['B', ], 'BB': ['B']})

        # One 2-nested fields
        ndtype = np.dtype([('A', int),
                           ('B', [('BA', int),
                                  ('BB', [('BBA', int), ('BBB', int)])])])
        # 调用 get_fieldstructure 函数，传入 dtype 对象 ndtype
        test = get_fieldstructure(ndtype)
        # 预期的嵌套字段结构字典
        control = {'A': [], 'B': [], 'BA': ['B'], 'BB': ['B'],
                   'BBA': ['B', 'BB'], 'BBB': ['B', 'BB']}
        # 断言结果与预期的字段结构字典匹配
        assert_equal(test, control)

        # 0 fields
        ndtype = np.dtype([])
        # 调用 get_fieldstructure 函数，传入 dtype 对象 ndtype
        test = get_fieldstructure(ndtype)
        # 断言结果与预期的空字典匹配
        assert_equal(test, {})

    def test_find_duplicates(self):
        # Test find_duplicates
        # 创建一个结构化数组 a
        a = ma.array([(2, (2., 'B')), (1, (2., 'B')), (2, (2., 'B')),
                      (1, (1., 'B')), (2, (2., 'B')), (2, (2., 'C'))],
                     mask=[(0, (0, 0)), (0, (0, 0)), (0, (0, 0)),
                           (0, (0, 0)), (1, (0, 0)), (0, (1, 0))],
                     dtype=[('A', int), ('B', [('BA', float), ('BB', '|S1')])])
        # 调用 find_duplicates 函数，使用默认参数 ignoremask=False 和 return_index=True
        test = find_duplicates(a, ignoremask=False, return_index=True)
        # 预期的重复索引列表
        control = [0, 2]
        # 断言结果与预期的重复索引列表匹配（排序后）
        assert_equal(sorted(test[-1]), control)
        # 断言返回的值与数组 a 中相应索引位置的值匹配
        assert_equal(test[0], a[test[-1]])

        # 调用 find_duplicates 函数，使用 key='A' 和 return_index=True
        test = find_duplicates(a, key='A', return_index=True)
        # 预期的重复索引列表
        control = [0, 1, 2, 3, 5]
        # 断言结果与预期的重复索引列表匹配（排序后）
        assert_equal(sorted(test[-1]), control)
        # 断言返回的值与数组 a 中相应索引位置的值匹配
        assert_equal(test[0], a[test[-1]])

        # 调用 find_duplicates 函数，使用 key='B' 和 return_index=True
        test = find_duplicates(a, key='B', return_index=True)
        # 预期的重复索引列表
        control = [0, 1, 2, 4]
        # 断言结果与预期的重复索引列表匹配（排序后）
        assert_equal(sorted(test[-1]), control)
        # 断言返回的值与数组 a 中相应索引位置的值匹配
        assert_equal(test[0], a[test[-1]])

        # 调用 find_duplicates 函数，使用 key='BA' 和 return_index=True
        test = find_duplicates(a, key='BA', return_index=True)
        # 预期的重复索引列表
        control = [0, 1, 2, 4]
        # 断言结果与预期的重复索引列表匹配（排序后）
        assert_equal(sorted(test[-1]), control)
        # 断言返回的值与数组 a 中相应索引位置的值匹配
        assert_equal(test[0], a[test[-1]])

        # 调用 find_duplicates 函数，使用 key='BB' 和 return_index=True
        test = find_duplicates(a, key='BB', return_index=True)
        # 预期的重复索引列表
        control = [0, 1, 2, 3, 4]
        # 断言结果与预期的重复索引列表匹配（排序后）
        assert_equal(sorted(test[-1]), control)
        # 断言返回的值与数组 a 中相应索引位置的值匹配
        assert_equal(test[0], a[test[-1]])

    def test_find_duplicates_ignoremask(self):
        # Test the ignoremask option of find_duplicates
        # 创建一个结构化数据类型 ndtype
        ndtype = [('a', int)]
        # 创建一个掩码数组 a
        a = ma.array([1, 1, 1, 2, 2, 3, 3],
                     mask=[0, 0, 1, 0, 0, 0, 1]).view(ndtype)
        # 调用 find_duplicates 函数，使用 ignoremask=True 和 return_index=True
        test = find_duplicates(a, ignoremask=True, return_index=True)
        # 预期的重复索引列表
        control = [0, 1, 3, 4]
        # 断言结果与预期的重复索引列表匹配（排序后）
        assert_equal(sorted(test[-1]), control)
        # 断言返回的值与数组 a 中相应索引位置的值匹配
        assert_equal(test[0], a[test[-1]])

        # 调用 find_duplicates 函数，使用 ignoremask=False 和 return_index=True
        test = find_duplicates(a, ignoremask=False, return_index=True)
        # 预期的重复索引列表
        control = [0, 1, 2, 3, 4, 6]
        # 断言结果与预期的重复索引列表匹配（排序后）
        assert_equal(sorted(test[-1]), control)
        # 断言返回的值与数组 a 中相应索引位置的值匹配
        assert_equal(test[0], a[test[-1]])
    # 定义测试方法：重组字段
    def test_repack_fields(self):
        # 创建一个自定义的数据类型，包含一个无符号字节、一个单精度浮点数和一个长整型数
        dt = np.dtype('u1,f4,i8', align=True)
        # 创建一个包含两个元素的零数组，数据类型为上述自定义类型
        a = np.zeros(2, dtype=dt)

        # 断言重组字段函数按预期返回了指定的数据类型
        assert_equal(repack_fields(dt), np.dtype('u1,f4,i8'))
        # 断言重组字段函数返回的数据类型的字节大小是13
        assert_equal(repack_fields(a).itemsize, 13)
        # 断言带有对齐参数的重组字段函数能够正确地恢复原始数据类型
        assert_equal(repack_fields(repack_fields(dt), align=True), dt)

        # 确保类型信息得到保留
        # 将数据类型转换为记录数组的数据类型
        dt = np.dtype((np.record, dt))
        # 断言重组字段函数返回的数据类型的类型对象是 np.record
        assert_(repack_fields(dt).type is np.record)

    # 定义测试方法：非结构化数组转为结构化数组
    def test_unstructured_to_structured(self):
        # 创建一个形状为 (20, 2) 的零数组
        a = np.zeros((20, 2))
        # 定义测试用的数据类型参数列表
        test_dtype_args = [('x', float), ('y', float)]
        # 创建一个指定数据类型的数据类型对象
        test_dtype = np.dtype(test_dtype_args)
        # 调用非结构化数组转为结构化数组的函数，使用数据类型参数列表作为 dtype 参数
        field1 = unstructured_to_structured(a, dtype=test_dtype_args)  # now
        # 调用非结构化数组转为结构化数组的函数，使用数据类型对象作为 dtype 参数
        field2 = unstructured_to_structured(a, dtype=test_dtype)  # before
        # 断言两次调用的结果相等
        assert_equal(field1, field2)

    # 定义测试方法：按字段名进行赋值
    def test_field_assignment_by_name(self):
        # 创建一个包含两个元素的数组，数据类型包含三个字段：a (整数), b (双精度浮点数), c (无符号字节)
        a = np.ones(2, dtype=[('a', 'i4'), ('b', 'f8'), ('c', 'u1')])
        # 新的数据类型定义，包含两个字段：b (单精度浮点数), c (无符号字节)
        newdt = [('b', 'f4'), ('c', 'u1')]

        # 断言要求字段函数返回的结果与新数据类型一致
        assert_equal(require_fields(a, newdt), np.ones(2, newdt))

        # 创建一个包含两个元素的数组，数据类型为新定义的数据类型 newdt
        b = np.array([(1,2), (3,4)], dtype=newdt)
        # 调用按字段名赋值函数，保留未赋值字段的原值
        assign_fields_by_name(a, b, zero_unassigned=False)
        # 断言数组 a 的值与预期一致
        assert_equal(a, np.array([(1,1,2),(1,3,4)], dtype=a.dtype))
        # 再次调用按字段名赋值函数，未赋值字段赋值为零
        assign_fields_by_name(a, b)
        # 断言数组 a 的值与预期一致
        assert_equal(a, np.array([(0,1,2),(0,3,4)], dtype=a.dtype))

        # 测试嵌套字段的情况
        # 创建一个包含一个元素的数组，元素是嵌套结构，包含一个双精度浮点数和一个无符号字节
        a = np.ones(2, dtype=[('a', [('b', 'f8'), ('c', 'u1')])])
        # 新的数据类型定义，包含一个元素，元素是嵌套结构，包含一个无符号字节
        newdt = [('a', [('c', 'u1')])]
        # 断言要求字段函数返回的结果与新数据类型一致
        assert_equal(require_fields(a, newdt), np.ones(2, newdt))
        # 创建一个包含一个元素的数组，元素是嵌套结构，包含一个元素 (2,)
        b = np.array([((2,),), ((3,),)], dtype=newdt)
        # 调用按字段名赋值函数，保留未赋值字段的原值
        assign_fields_by_name(a, b, zero_unassigned=False)
        # 断言数组 a 的值与预期一致
        assert_equal(a, np.array([((1,2),), ((1,3),)], dtype=a.dtype))
        # 再次调用按字段名赋值函数，未赋值字段赋值为零
        assign_fields_by_name(a, b)
        # 断言数组 a 的值与预期一致
        assert_equal(a, np.array([((0,2),), ((0,3),)], dtype=a.dtype))

        # 测试针对 0 维数组的非结构化代码路径
        a, b = np.array(3), np.array(0)
        # 调用按字段名赋值函数
        assign_fields_by_name(b, a)
        # 断言数组 b 中索引为 () 的元素的值为 3
        assert_equal(b[()], 3)
class TestRecursiveFillFields:
    # Test recursive_fill_fields.

    def test_simple_flexible(self):
        # Test recursive_fill_fields on flexible-array
        # 创建一个包含两个元组的 NumPy 数组 a，其中元组包含 (1, 10.) 和 (2, 20.)，
        # 数据类型为 [('A', int), ('B', float)]
        a = np.array([(1, 10.), (2, 20.)], dtype=[('A', int), ('B', float)])
        
        # 创建一个形状为 (3,) 的零数组 b，数据类型与数组 a 相同
        b = np.zeros((3,), dtype=a.dtype)
        
        # 调用 recursive_fill_fields 函数，将数组 a 填充到数组 b 中
        test = recursive_fill_fields(a, b)
        
        # 创建一个控制数组 control，包含三个元组 (1, 10.), (2, 20.), (0, 0.)
        # 数据类型为 [('A', int), ('B', float)]
        control = np.array([(1, 10.), (2, 20.), (0, 0.)],
                           dtype=[('A', int), ('B', float)])
        
        # 断言测试结果与控制数组相等
        assert_equal(test, control)

    def test_masked_flexible(self):
        # Test recursive_fill_fields on masked flexible-array
        # 创建一个包含两个元组的掩码数组 a，元组为 (1, 10.) 和 (2, 20.)，掩码为 [(0, 1), (1, 0)]
        # 数据类型为 [('A', int), ('B', float)]
        a = ma.array([(1, 10.), (2, 20.)], mask=[(0, 1), (1, 0)],
                     dtype=[('A', int), ('B', float)])
        
        # 创建一个形状为 (3,) 的掩码数组 b，数据类型与数组 a 相同
        b = ma.zeros((3,), dtype=a.dtype)
        
        # 调用 recursive_fill_fields 函数，将掩码数组 a 填充到掩码数组 b 中
        test = recursive_fill_fields(a, b)
        
        # 创建一个控制数组 control，包含三个元组 (1, 10.), (2, 20.), (0, 0.)
        # 数据类型为 [('A', int), ('B', float)]，并且具有相同的掩码
        control = ma.array([(1, 10.), (2, 20.), (0, 0.)],
                           mask=[(0, 1), (1, 0), (0, 0)],
                           dtype=[('A', int), ('B', float)])
        
        # 断言测试结果与控制数组相等
        assert_equal(test, control)


class TestMergeArrays:
    # Test merge_arrays

    def setup_method(self):
        # 设置测试方法的初始数据
        x = np.array([1, 2, ])
        y = np.array([10, 20, 30])
        z = np.array(
            [('A', 1.), ('B', 2.)], dtype=[('A', '|S3'), ('B', float)])
        w = np.array(
            [(1, (2, 3.0, ())), (4, (5, 6.0, ()))],
            dtype=[('a', int), ('b', [('ba', float), ('bb', int), ('bc', [])])])
        self.data = (w, x, y, z)

    def test_solo(self):
        # Test merge_arrays on a single array.
        (_, x, _, z) = self.data
        
        # 调用 merge_arrays 函数，传入数组 x
        test = merge_arrays(x)
        
        # 创建一个控制数组 control，包含两个元组 (1,) 和 (2,)
        # 数据类型为 [('f0', int)]
        control = np.array([(1,), (2,)], dtype=[('f0', int)])
        
        # 断言测试结果与控制数组相等
        assert_equal(test, control)
        
        # 再次调用 merge_arrays 函数，传入元组 (x,)
        test = merge_arrays((x,))
        
        # 断言测试结果与控制数组相等
        assert_equal(test, control)
        
        # 调用 merge_arrays 函数，传入数组 z，并指定 flatten=False
        test = merge_arrays(z, flatten=False)
        
        # 断言测试结果与数组 z 相等
        assert_equal(test, z)
        
        # 再次调用 merge_arrays 函数，传入数组 z，并指定 flatten=True
        test = merge_arrays(z, flatten=True)
        
        # 断言测试结果与数组 z 相等
        assert_equal(test, z)

    def test_solo_w_flatten(self):
        # Test merge_arrays on a single array w & w/o flattening
        # 获取初始数据中的数组 w
        w = self.data[0]
        
        # 调用 merge_arrays 函数，传入数组 w，并指定 flatten=False
        test = merge_arrays(w, flatten=False)
        
        # 断言测试结果与数组 w 相等
        assert_equal(test, w)
        
        # 再次调用 merge_arrays 函数，传入数组 w，并指定 flatten=True
        test = merge_arrays(w, flatten=True)
        
        # 创建一个控制数组 control，包含两个元组 (1, 2, 3.0) 和 (4, 5, 6.0)
        # 数据类型为 [('a', int), ('ba', float), ('bb', int)]
        control = np.array([(1, 2, 3.0), (4, 5, 6.0)],
                           dtype=[('a', int), ('ba', float), ('bb', int)])
        
        # 断言测试结果与控制数组相等
        assert_equal(test, control)

    def test_standard(self):
        # Test standard & standard
        # Test merge arrays
        (_, x, y, _) = self.data
        
        # 调用 merge_arrays 函数，传入元组 (x, y)，并指定 usemask=False
        test = merge_arrays((x, y), usemask=False)
        
        # 创建一个控制数组 control，包含三个元组 (1, 10), (2, 20), (-1, 30)
        # 数据类型为 [('f0', int), ('f1', int)]
        control = np.array([(1, 10), (2, 20), (-1, 30)],
                           dtype=[('f0', int), ('f1', int)])
        
        # 断言测试结果与控制数组相等
        assert_equal(test, control)
        
        # 再次调用 merge_arrays 函数，传入元组 (x, y)，并指定 usemask=True
        test = merge_arrays((x, y), usemask=True)
        
        # 创建一个控制数组 control，包含三个元组 (1, 10), (2, 20), (-1, 30)
        # 数据类型为 [('f0', int), ('f1', int)]，并且具有相同的掩码
        control = ma.array([(1, 10), (2, 20), (-1, 30)],
                           mask=[(0, 0), (0, 0), (1, 0)],
                           dtype=[('f0', int), ('f1', int)])
        
        # 断言测试结果与控制数组相等，并且掩码也相等
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
    def test_flatten(self):
        # Test standard & flexible
        (_, x, _, z) = self.data  # 从测试数据中解包出 x 和 z
        test = merge_arrays((x, z), flatten=True)  # 调用 merge_arrays 函数，使用 flatten=True 进行合并测试
        control = np.array([(1, 'A', 1.), (2, 'B', 2.)],
                           dtype=[('f0', int), ('A', '|S3'), ('B', float)])  # 预期的合并结果
        assert_equal(test, control)  # 断言测试结果与预期结果相等

        test = merge_arrays((x, z), flatten=False)  # 再次调用 merge_arrays 函数，使用 flatten=False 进行合并测试
        control = np.array([(1, ('A', 1.)), (2, ('B', 2.))],
                           dtype=[('f0', int),
                                  ('f1', [('A', '|S3'), ('B', float)])])  # 预期的合并结果，包含嵌套结构
        assert_equal(test, control)  # 断言测试结果与预期结果相等

    def test_flatten_wflexible(self):
        # Test flatten standard & nested
        (w, x, _, _) = self.data  # 从测试数据中解包出 w 和 x
        test = merge_arrays((x, w), flatten=True)  # 调用 merge_arrays 函数，使用 flatten=True 进行合并测试
        control = np.array([(1, 1, 2, 3.0), (2, 4, 5, 6.0)],
                           dtype=[('f0', int),
                                  ('a', int), ('ba', float), ('bb', int)])  # 预期的合并结果
        assert_equal(test, control)  # 断言测试结果与预期结果相等

        test = merge_arrays((x, w), flatten=False)  # 再次调用 merge_arrays 函数，使用 flatten=False 进行合并测试
        controldtype = [('f0', int),
                        ('f1', [('a', int),
                                ('b', [('ba', float), ('bb', int), ('bc', [])])])]  # 预期的 dtype 结构
        control = np.array([(1., (1, (2, 3.0, ()))), (2, (4, (5, 6.0, ())))],
                           dtype=controldtype)  # 预期的合并结果，包含复杂的嵌套结构
        assert_equal(test, control)  # 断言测试结果与预期结果相等

    def test_wmasked_arrays(self):
        # Test merge_arrays masked arrays
        (_, x, _, _) = self.data  # 从测试数据中解包出 x
        mx = ma.array([1, 2, 3], mask=[1, 0, 0])  # 创建一个带掩码的 MaskedArray
        test = merge_arrays((x, mx), usemask=True)  # 调用 merge_arrays 函数，使用带掩码的测试
        control = ma.array([(1, 1), (2, 2), (-1, 3)],
                           mask=[(0, 1), (0, 0), (1, 0)],
                           dtype=[('f0', int), ('f1', int)])  # 预期的合并结果，包含掩码信息
        assert_equal(test, control)  # 断言测试结果与预期结果相等
        test = merge_arrays((x, mx), usemask=True, asrecarray=True)  # 再次调用 merge_arrays 函数，使用 asrecarray=True 进行测试
        assert_equal(test, control)  # 断言测试结果与预期结果相等
        assert_(isinstance(test, MaskedRecords))  # 断言返回结果是 MaskedRecords 类的实例

    def test_w_singlefield(self):
        # Test single field
        test = merge_arrays((np.array([1, 2]).view([('a', int)]),
                             np.array([10., 20., 30.])),)  # 调用 merge_arrays 函数，测试单个字段合并
        control = ma.array([(1, 10.), (2, 20.), (-1, 30.)],
                           mask=[(0, 0), (0, 0), (1, 0)],
                           dtype=[('a', int), ('f1', float)])  # 预期的合并结果，包含掩码信息和 dtype
        assert_equal(test, control)  # 断言测试结果与预期结果相等
    def test_w_shorter_flex(self):
        # 测试 merge_arrays 函数，使用较短的 flexndarray 作为输入
        z = self.data[-1]

        # FIXME，这个测试看起来是不完整和有问题的
        #test = merge_arrays((z, np.array([10, 20, 30]).view([('C', int)])))
        #control = np.array([('A', 1., 10), ('B', 2., 20), ('-1', -1, 20)],
        #                   dtype=[('A', '|S3'), ('B', float), ('C', int)])
        #assert_equal(test, control)

        # 使用 merge_arrays 函数合并 z 和一个新的 numpy 数组，以避免 pyflakes 警告未使用的变量
        merge_arrays((z, np.array([10, 20, 30]).view([('C', int)])))
        # 创建一个 numpy 数组作为控制组，用于后续的断言比较
        np.array([('A', 1., 10), ('B', 2., 20), ('-1', -1, 20)],
                 dtype=[('A', '|S3'), ('B', float), ('C', int)])

    def test_singlerecord(self):
        # 从 self.data 中获取 x, y, z 的值
        (_, x, y, z) = self.data
        # 使用 merge_arrays 函数测试单个记录的合并，关闭掩码使用
        test = merge_arrays((x[0], y[0], z[0]), usemask=False)
        # 创建一个 numpy 数组作为控制组，用于断言比较
        control = np.array([(1, 10, ('A', 1))],
                           dtype=[('f0', int),
                                  ('f1', int),
                                  ('f2', [('A', '|S3'), ('B', float)])])
        # 断言测试结果与控制组相等
        assert_equal(test, control)
class TestAppendFields:
    # Test append_fields

    def setup_method(self):
        # 设置测试环境，初始化几个 NumPy 数组
        x = np.array([1, 2, ])
        y = np.array([10, 20, 30])
        z = np.array(
            [('A', 1.), ('B', 2.)], dtype=[('A', '|S3'), ('B', float)])
        w = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
                     dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])
        # 将这些数组组成一个元组并赋值给实例变量 self.data
        self.data = (w, x, y, z)

    def test_append_single(self):
        # Test simple case
        # 解包 self.data 中的数组
        (_, x, _, _) = self.data
        # 调用 append_fields 函数，添加单个字段 'A' 到数组 x
        test = append_fields(x, 'A', data=[10, 20, 30])
        # 创建期望的控制结果
        control = ma.array([(1, 10), (2, 20), (-1, 30)],
                           mask=[(0, 0), (0, 0), (1, 0)],
                           dtype=[('f0', int), ('A', int)],)
        # 断言测试结果与期望结果相等
        assert_equal(test, control)

    def test_append_double(self):
        # Test simple case
        # 解包 self.data 中的数组
        (_, x, _, _) = self.data
        # 调用 append_fields 函数，添加两个字段 'A' 和 'B' 到数组 x
        test = append_fields(x, ('A', 'B'), data=[[10, 20, 30], [100, 200]])
        # 创建期望的控制结果
        control = ma.array([(1, 10, 100), (2, 20, 200), (-1, 30, -1)],
                           mask=[(0, 0, 0), (0, 0, 0), (1, 0, 1)],
                           dtype=[('f0', int), ('A', int), ('B', int)],)
        # 断言测试结果与期望结果相等
        assert_equal(test, control)

    def test_append_on_flex(self):
        # Test append_fields on flexible type arrays
        # 获取 self.data 中的最后一个数组 z
        z = self.data[-1]
        # 调用 append_fields 函数，在数组 z 上添加字段 'C'
        test = append_fields(z, 'C', data=[10, 20, 30])
        # 创建期望的控制结果
        control = ma.array([('A', 1., 10), ('B', 2., 20), (-1, -1., 30)],
                           mask=[(0, 0, 0), (0, 0, 0), (1, 1, 0)],
                           dtype=[('A', '|S3'), ('B', float), ('C', int)],)
        # 断言测试结果与期望结果相等
        assert_equal(test, control)

    def test_append_on_nested(self):
        # Test append_fields on nested fields
        # 获取 self.data 中的第一个数组 w
        w = self.data[0]
        # 调用 append_fields 函数，在数组 w 上添加字段 'C'
        test = append_fields(w, 'C', data=[10, 20, 30])
        # 创建期望的控制结果
        control = ma.array([(1, (2, 3.0), 10),
                            (4, (5, 6.0), 20),
                            (-1, (-1, -1.), 30)],
                           mask=[(
                               0, (0, 0), 0), (0, (0, 0), 0), (1, (1, 1), 0)],
                           dtype=[('a', int),
                                  ('b', [('ba', float), ('bb', int)]),
                                  ('C', int)],)
        # 断言测试结果与期望结果相等
        assert_equal(test, control)


class TestStackArrays:
    # Test stack_arrays
    def setup_method(self):
        # 设置测试环境，初始化几个 NumPy 数组
        x = np.array([1, 2, ])
        y = np.array([10, 20, 30])
        z = np.array(
            [('A', 1.), ('B', 2.)], dtype=[('A', '|S3'), ('B', float)])
        w = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
                     dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])
        # 将这些数组组成一个元组并赋值给实例变量 self.data
        self.data = (w, x, y, z)

    def test_solo(self):
        # Test stack_arrays on single arrays
        # 解包 self.data 中的数组
        (_, x, _, _) = self.data
        # 调用 stack_arrays 函数，堆叠单个数组 x
        test = stack_arrays((x,))
        # 断言测试结果与 x 相等
        assert_equal(test, x)
        # 断言测试结果与 x 是同一个对象
        assert_(test is x)

        # 再次调用 stack_arrays 函数，直接传递数组 x
        test = stack_arrays(x)
        # 断言测试结果与 x 相等
        assert_equal(test, x)
        # 断言测试结果与 x 是同一个对象
        assert_(test is x)
    def test_unnamed_fields(self):
        # Tests combinations of arrays w/o named fields
        # 解构 self.data 元组，获取第二个和第三个元素作为 x 和 y
        (_, x, y, _) = self.data

        # 使用 stack_arrays 函数堆叠两个 x 数组，不使用掩码
        test = stack_arrays((x, x), usemask=False)
        # 创建控制数组，预期结果为 [1, 2, 1, 2]
        control = np.array([1, 2, 1, 2])
        # 断言测试结果与控制数组相等
        assert_equal(test, control)

        # 使用 stack_arrays 函数堆叠 x 和 y 数组，不使用掩码
        test = stack_arrays((x, y), usemask=False)
        # 创建控制数组，预期结果为 [1, 2, 10, 20, 30]
        control = np.array([1, 2, 10, 20, 30])
        # 断言测试结果与控制数组相等
        assert_equal(test, control)

        # 使用 stack_arrays 函数堆叠 y 和 x 数组，不使用掩码
        test = stack_arrays((y, x), usemask=False)
        # 创建控制数组，预期结果为 [10, 20, 30, 1, 2]
        control = np.array([10, 20, 30, 1, 2])
        # 断言测试结果与控制数组相等
        assert_equal(test, control)

    def test_unnamed_and_named_fields(self):
        # Test combination of arrays w/ & w/o named fields
        # 解构 self.data 元组，获取第二个和第四个元素作为 x 和 z
        (_, x, _, z) = self.data

        # 使用 stack_arrays 函数堆叠 x 和 z 数组
        test = stack_arrays((x, z))
        # 创建控制数组，包含元组和掩码，具体内容见下文
        control = ma.array([(1, -1, -1), (2, -1, -1),
                            (-1, 'A', 1), (-1, 'B', 2)],
                           mask=[(0, 1, 1), (0, 1, 1),
                                 (1, 0, 0), (1, 0, 0)],
                           dtype=[('f0', int), ('A', '|S3'), ('B', float)])
        # 断言测试结果与控制数组相等
        assert_equal(test, control)
        # 断言测试结果的掩码与控制数组的掩码相等
        assert_equal(test.mask, control.mask)

        # 使用 stack_arrays 函数堆叠 z 和 x 数组
        test = stack_arrays((z, x))
        # 创建控制数组，包含元组和掩码，具体内容见下文
        control = ma.array([('A', 1, -1), ('B', 2, -1),
                            (-1, -1, 1), (-1, -1, 2), ],
                           mask=[(0, 0, 1), (0, 0, 1),
                                 (1, 1, 0), (1, 1, 0)],
                           dtype=[('A', '|S3'), ('B', float), ('f2', int)])
        # 断言测试结果与控制数组相等
        assert_equal(test, control)
        # 断言测试结果的掩码与控制数组的掩码相等
        assert_equal(test.mask, control.mask)

        # 使用 stack_arrays 函数堆叠 z, z 和 x 数组
        test = stack_arrays((z, z, x))
        # 创建控制数组，包含元组和掩码，具体内容见下文
        control = ma.array([('A', 1, -1), ('B', 2, -1),
                            ('A', 1, -1), ('B', 2, -1),
                            (-1, -1, 1), (-1, -1, 2), ],
                           mask=[(0, 0, 1), (0, 0, 1),
                                 (0, 0, 1), (0, 0, 1),
                                 (1, 1, 0), (1, 1, 0)],
                           dtype=[('A', '|S3'), ('B', float), ('f2', int)])
        # 断言测试结果与控制数组相等
        assert_equal(test, control)
    def test_matching_named_fields(self):
        # Test combination of arrays w/ matching field names

        # 解构赋值，从 self.data 中获取第二个和第四个元素作为 x 和 z
        (_, x, _, z) = self.data
        
        # 创建一个 NumPy 结构化数组 zz，包含三个元组
        zz = np.array([('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
                      dtype=[('A', '|S3'), ('B', float), ('C', float)])
        
        # 将 z 和 zz 数组堆叠在一起
        test = stack_arrays((z, zz))
        
        # 创建一个控制用的 MaskedArray 控制变量
        control = ma.array([('A', 1, -1), ('B', 2, -1),
                            ('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
                           dtype=[('A', '|S3'), ('B', float), ('C', float)],
                           mask=[(0, 0, 1), (0, 0, 1),
                                 (0, 0, 0), (0, 0, 0), (0, 0, 0)])
        
        # 断言 test 是否等于 control
        assert_equal(test, control)
        
        # 断言 test 的 mask 是否等于 control 的 mask
        assert_equal(test.mask, control.mask)

        # 将 z、zz 和 x 三个数组堆叠在一起
        test = stack_arrays((z, zz, x))
        
        # 创建一个包含 'f3' 字段的结构化数组的 dtype
        ndtype = [('A', '|S3'), ('B', float), ('C', float), ('f3', int)]
        
        # 创建另一个控制用的 MaskedArray 控制变量
        control = ma.array([('A', 1, -1, -1), ('B', 2, -1, -1),
                            ('a', 10., 100., -1), ('b', 20., 200., -1),
                            ('c', 30., 300., -1),
                            (-1, -1, -1, 1), (-1, -1, -1, 2)],
                           dtype=ndtype,
                           mask=[(0, 0, 1, 1), (0, 0, 1, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1),
                                 (1, 1, 1, 0), (1, 1, 1, 0)])
        
        # 断言 test 是否等于 control
        assert_equal(test, control)
        
        # 断言 test 的 mask 是否等于 control 的 mask
        assert_equal(test.mask, control.mask)

    def test_defaults(self):
        # Test defaults: no exception raised if keys of defaults are not fields.

        # 解构赋值，从 self.data 中获取第四个元素作为 z
        (_, _, _, z) = self.data
        
        # 创建一个 NumPy 结构化数组 zz，包含三个元组
        zz = np.array([('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
                      dtype=[('A', '|S3'), ('B', float), ('C', float)])
        
        # 定义默认值字典
        defaults = {'A': '???', 'B': -999., 'C': -9999., 'D': -99999.}
        
        # 使用默认值字典将 z 和 zz 数组堆叠在一起
        test = stack_arrays((z, zz), defaults=defaults)
        
        # 创建一个控制用的 MaskedArray 控制变量
        control = ma.array([('A', 1, -9999.), ('B', 2, -9999.),
                            ('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
                           dtype=[('A', '|S3'), ('B', float), ('C', float)],
                           mask=[(0, 0, 1), (0, 0, 1),
                                 (0, 0, 0), (0, 0, 0), (0, 0, 0)])
        
        # 断言 test 是否等于 control
        assert_equal(test, control)
        
        # 断言 test 的数据部分是否等于 control 的数据部分
        assert_equal(test.data, control.data)
        
        # 断言 test 的 mask 是否等于 control 的 mask
        assert_equal(test.mask, control.mask)
    def test_autoconversion(self):
        # Tests autoconversion
        # 定义自动转换的数据类型列表
        adtype = [('A', int), ('B', bool), ('C', float)]
        # 创建数组 a，设置数据和掩码，使用指定的数据类型 adtype
        a = ma.array([(1, 2, 3)], mask=[(0, 1, 0)], dtype=adtype)
        # 定义数据类型列表 bdtype
        bdtype = [('A', int), ('B', float), ('C', float)]
        # 创建数组 b，设置数据，使用指定的数据类型 bdtype
        b = ma.array([(4, 5, 6)], dtype=bdtype)
        # 创建控制数组 control，包含数组 a 和 b 的组合数据，设置掩码，使用数据类型 bdtype
        control = ma.array([(1, 2, 3), (4, 5, 6)], mask=[(0, 1, 0), (0, 0, 0)],
                           dtype=bdtype)
        # 使用函数 stack_arrays 进行数组堆叠，启用自动转换
        test = stack_arrays((a, b), autoconvert=True)
        # 断言 test 和 control 数组相等
        assert_equal(test, control)
        # 断言 test 的掩码和 control 的掩码相等
        assert_equal(test.mask, control.mask)
        # 使用 assert_raises 检查 TypeError 异常是否被触发
        with assert_raises(TypeError):
            stack_arrays((a, b), autoconvert=False)

    def test_checktitles(self):
        # Test using titles in the field names
        # 定义带标题的数据类型列表 adtype
        adtype = [(('a', 'A'), int), (('b', 'B'), bool), (('c', 'C'), float)]
        # 创建数组 a，设置数据和掩码，使用指定的数据类型 adtype
        a = ma.array([(1, 2, 3)], mask=[(0, 1, 0)], dtype=adtype)
        # 定义带标题的数据类型列表 bdtype
        bdtype = [(('a', 'A'), int), (('b', 'B'), bool), (('c', 'C'), float)]
        # 创建数组 b，设置数据，使用指定的数据类型 bdtype
        b = ma.array([(4, 5, 6)], dtype=bdtype)
        # 使用函数 stack_arrays 进行数组堆叠
        test = stack_arrays((a, b))
        # 创建控制数组 control，包含数组 a 和 b 的组合数据，设置掩码，使用数据类型 bdtype
        control = ma.array([(1, 2, 3), (4, 5, 6)], mask=[(0, 1, 0), (0, 0, 0)],
                           dtype=bdtype)
        # 断言 test 和 control 数组相等
        assert_equal(test, control)
        # 断言 test 的掩码和 control 的掩码相等
        assert_equal(test.mask, control.mask)

    def test_subdtype(self):
        # 创建数组 z，设置数据和子数据类型，使用指定的数据类型
        z = np.array([
            ('A', 1), ('B', 2)
        ], dtype=[('A', '|S3'), ('B', float, (1,))])
        # 创建数组 zz，设置数据和子数据类型，使用指定的数据类型
        zz = np.array([
            ('a', [10.], 100.), ('b', [20.], 200.), ('c', [30.], 300.)
        ], dtype=[('A', '|S3'), ('B', float, (1,)), ('C', float)])
        # 使用函数 stack_arrays 进行数组堆叠
        res = stack_arrays((z, zz))
        # 创建期望的掩码数组 expected，设置数据、掩码和数据类型
        expected = ma.array(
            data=[
                (b'A', [1.0], 0),
                (b'B', [2.0], 0),
                (b'a', [10.0], 100.0),
                (b'b', [20.0], 200.0),
                (b'c', [30.0], 300.0)],
            mask=[
                (False, [False],  True),
                (False, [False],  True),
                (False, [False], False),
                (False, [False], False),
                (False, [False], False)
            ],
            dtype=zz.dtype
        )
        # 断言 res 的数据类型和 expected 的数据类型相等
        assert_equal(res.dtype, expected.dtype)
        # 断言 res 和 expected 数组相等
        assert_equal(res, expected)
        # 断言 res 的掩码和 expected 的掩码相等
        assert_equal(res.mask, expected.mask)
class TestJoinBy:
    # 在测试类中设置方法的初始状态
    def setup_method(self):
        # 创建包含三个字段的 NumPy 数组 self.a，每个字段都是从不同范围的整数值中创建的
        self.a = np.array(list(zip(np.arange(10), np.arange(50, 60),
                                   np.arange(100, 110))),
                          dtype=[('a', int), ('b', int), ('c', int)])
        # 创建包含三个字段的 NumPy 数组 self.b，与 self.a 结构相同，但字段不完全一致
        self.b = np.array(list(zip(np.arange(5, 15), np.arange(65, 75),
                                   np.arange(100, 110))),
                          dtype=[('a', int), ('b', int), ('d', int)])

    # 测试内连接的基本功能
    def test_inner_join(self):
        # 获取 self.a 和 self.b 的引用
        a, b = self.a, self.b
        # 使用 'a' 字段进行内连接操作，并将结果存储在 test 变量中
        test = join_by('a', a, b, jointype='inner')
        # 创建预期结果的 NumPy 数组 control，包含特定字段和值
        control = np.array([(5, 55, 65, 105, 100), (6, 56, 66, 106, 101),
                            (7, 57, 67, 107, 102), (8, 58, 68, 108, 103),
                            (9, 59, 69, 109, 104)],
                           dtype=[('a', int), ('b1', int), ('b2', int),
                                  ('c', int), ('d', int)])
        # 断言 test 和 control 数组是否相等
        assert_equal(test, control)

    # 测试 join_by 函数的一般连接功能（目前被注释掉）
    def test_join(self):
        # 获取 self.a 和 self.b 的引用
        a, b = self.a, self.b
        
        # Fixme, this test is broken
        #test = join_by(('a', 'b'), a, b)
        #control = np.array([(5, 55, 105, 100), (6, 56, 106, 101),
        #                    (7, 57, 107, 102), (8, 58, 108, 103),
        #                    (9, 59, 109, 104)],
        #                   dtype=[('a', int), ('b', int),
        #                          ('c', int), ('d', int)])
        #assert_equal(test, control)

        # 使用 join_by 函数，连接 ('a', 'b') 字段，但忽略其返回结果（仅为了避免警告）
        join_by(('a', 'b'), a, b)
        # 创建一个 NumPy 数组，但这个数组并没有被用于任何断言或返回操作
        np.array([(5, 55, 105, 100), (6, 56, 106, 101),
                  (7, 57, 107, 102), (8, 58, 108, 103),
                  (9, 59, 109, 104)],
                  dtype=[('a', int), ('b', int),
                         ('c', int), ('d', int)])

    # 测试子数据类型的连接功能（用于验证已知的 bug）
    def test_join_subdtype(self):
        # 创建包含单个键值对的 NumPy 数组 foo，键 'key' 对应一个整数值
        foo = np.array([(1,)],
                       dtype=[('key', int)])
        # 创建包含一个键值对的 NumPy 数组 bar，键 'key' 对应一个整数值，值 'value' 是一个有三个元素的无符号整数数组
        bar = np.array([(1, np.array([1,2,3]))],
                       dtype=[('key', int), ('value', 'uint16', 3)])
        # 使用 'key' 字段进行连接操作，将结果存储在 res 变量中
        res = join_by('key', foo, bar)
        # 断言 res 和 bar 的视图（MaskedArray）是否相等
        assert_equal(res, bar.view(ma.MaskedArray))
    def test_outer_join(self):
        a, b = self.a, self.b
        # 调用自定义函数join_by进行外连接测试，使用字段('a', 'b')，连接a和b两个数组，连接方式为'outer'
        test = join_by(('a', 'b'), a, b, 'outer')
        # 预期的控制结果，一个带有数据和遮罩的MaskedArray对象
        control = ma.array([(0, 50, 100, -1), (1, 51, 101, -1),
                            (2, 52, 102, -1), (3, 53, 103, -1),
                            (4, 54, 104, -1), (5, 55, 105, -1),
                            (5, 65, -1, 100), (6, 56, 106, -1),
                            (6, 66, -1, 101), (7, 57, 107, -1),
                            (7, 67, -1, 102), (8, 58, 108, -1),
                            (8, 68, -1, 103), (9, 59, 109, -1),
                            (9, 69, -1, 104), (10, 70, -1, 105),
                            (11, 71, -1, 106), (12, 72, -1, 107),
                            (13, 73, -1, 108), (14, 74, -1, 109)],
                           mask=[(0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 1, 0), (0, 0, 0, 1),
                                 (0, 0, 1, 0), (0, 0, 0, 1),
                                 (0, 0, 1, 0), (0, 0, 0, 1),
                                 (0, 0, 1, 0), (0, 0, 0, 1),
                                 (0, 0, 1, 0), (0, 0, 1, 0),
                                 (0, 0, 1, 0), (0, 0, 1, 0),
                                 (0, 0, 1, 0), (0, 0, 1, 0)],
                           dtype=[('a', int), ('b', int),
                                  ('c', int), ('d', int)])
        # 使用断言验证test和control是否相等
        assert_equal(test, control)

    def test_leftouter_join(self):
        a, b = self.a, self.b
        # 调用自定义函数join_by进行左外连接测试，使用字段('a', 'b')，连接a和b两个数组，连接方式为'leftouter'
        test = join_by(('a', 'b'), a, b, 'leftouter')
        # 预期的控制结果，一个带有数据和遮罩的MaskedArray对象
        control = ma.array([(0, 50, 100, -1), (1, 51, 101, -1),
                            (2, 52, 102, -1), (3, 53, 103, -1),
                            (4, 54, 104, -1), (5, 55, 105, -1),
                            (6, 56, 106, -1), (7, 57, 107, -1),
                            (8, 58, 108, -1), (9, 59, 109, -1)],
                           mask=[(0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1),
                                 (0, 0, 0, 1), (0, 0, 0, 1)],
                           dtype=[('a', int), ('b', int), ('c', int), ('d', int)])
        # 使用断言验证test和control是否相等
        assert_equal(test, control)

    def test_different_field_order(self):
        # 测试情形gh-8940
        # 创建一个包含3行的全零数组a，dtype为[('a', 'i4'), ('b', 'f4'), ('c', 'u1')]
        a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'f4'), ('c', 'u1')])
        # 创建一个包含3行的全一数组b，dtype为[('c', 'u1'), ('b', 'f4'), ('a', 'i4')]
        b = np.ones(3, dtype=[('c', 'u1'), ('b', 'f4'), ('a', 'i4')])
        # 调用自定义函数join_by进行内连接测试，使用字段['c', 'b']，连接a和b两个数组，连接方式为'inner'，不使用遮罩
        j = join_by(['c', 'b'], a, b, jointype='inner', usemask=False)
        # 使用断言验证连接后的dtype的字段名是否为['b', 'c', 'a1', 'a2']
        assert_equal(j.dtype.names, ['b', 'c', 'a1', 'a2'])
    # 测试函数：检查在具有重复键的情况下是否引发 ValueError 异常
    def test_duplicate_keys(self):
        # 创建包含零值的 NumPy 结构化数组 a
        a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'f4'), ('c', 'u1')])
        # 创建包含全一值的 NumPy 结构化数组 b，但是键 'c' 和 'a' 的顺序与 a 中不同
        b = np.ones(3, dtype=[('c', 'u1'), ('b', 'f4'), ('a', 'i4')])
        # 断言调用 join_by 函数时会引发 ValueError 异常
        assert_raises(ValueError, join_by, ['a', 'b', 'b'], a, b)

    # 测试函数：检查具有相同键但数据类型不同的情况下是否正确合并
    def test_same_name_different_dtypes_key(self):
        # 定义 a 的数据类型，包含 'key' 和 '<f4' 类型的元组
        a_dtype = np.dtype([('key', 'S5'), ('value', '<f4')])
        # 定义 b 的数据类型，包含 'key' 和 '<f4' 类型的元组，但 'key' 的长度为 10
        b_dtype = np.dtype([('key', 'S10'), ('value', '<f4')])
        # 期望结果的数据类型，包含 'key'、'value1'（<f4）、'value2'（<f4）的元组
        expected_dtype = np.dtype([
            ('key', 'S10'), ('value1', '<f4'), ('value2', '<f4')])

        # 创建 NumPy 结构化数组 a 和 b
        a = np.array([('Sarah',  8.0), ('John', 6.0)], dtype=a_dtype)
        b = np.array([('Sarah', 10.0), ('John', 7.0)], dtype=b_dtype)
        # 调用 join_by 函数，按照 'key' 合并 a 和 b
        res = join_by('key', a, b)

        # 断言合并结果的数据类型与期望的数据类型相同
        assert_equal(res.dtype, expected_dtype)

    # 测试函数：检查具有相同键但数据类型不同的情况下是否正确合并（另一种情况）
    def test_same_name_different_dtypes(self):
        # 定义 a 的数据类型，包含 'key' 和 '<f4' 类型的元组
        a_dtype = np.dtype([('key', 'S10'), ('value', '<f4')])
        # 定义 b 的数据类型，包含 'key' 和 '<f8' 类型的元组
        b_dtype = np.dtype([('key', 'S10'), ('value', '<f8')])
        # 期望结果的数据类型，包含 'key'、'value1'（<f4）、'value2'（<f8）的元组
        expected_dtype = np.dtype([
            ('key', '|S10'), ('value1', '<f4'), ('value2', '<f8')])

        # 创建 NumPy 结构化数组 a 和 b
        a = np.array([('Sarah',  8.0), ('John', 6.0)], dtype=a_dtype)
        b = np.array([('Sarah', 10.0), ('John', 7.0)], dtype=b_dtype)
        # 调用 join_by 函数，按照 'key' 合并 a 和 b
        res = join_by('key', a, b)

        # 断言合并结果的数据类型与期望的数据类型相同
        assert_equal(res.dtype, expected_dtype)

    # 测试函数：检查包含子数组键的情况下是否正确合并
    def test_subarray_key(self):
        # 定义 a 的数据类型，包含 'pos' 和 '<f4' 类型的元组，'pos' 是一个长度为 3 的整数数组
        a_dtype = np.dtype([('pos', int, 3), ('f', '<f4')])
        # 创建 NumPy 结构化数组 a
        a = np.array([([1, 1, 1], np.pi), ([1, 2, 3], 0.0)], dtype=a_dtype)

        # 定义 b 的数据类型，包含 'pos' 和 '<f4' 类型的元组，'pos' 是一个长度为 3 的整数数组
        b_dtype = np.dtype([('pos', int, 3), ('g', '<f4')])
        # 创建 NumPy 结构化数组 b
        b = np.array([([1, 1, 1], 3), ([3, 2, 1], 0.0)], dtype=b_dtype)

        # 期望结果的数据类型，包含 'pos'、'f'（<f4）、'g'（<f4）的元组
        expected_dtype = np.dtype([('pos', int, 3), ('f', '<f4'), ('g', '<f4')])
        # 期望的合并结果
        expected = np.array([([1, 1, 1], np.pi, 3)], dtype=expected_dtype)

        # 调用 join_by 函数，按照 'pos' 合并 a 和 b
        res = join_by('pos', a, b)

        # 断言合并结果的数据类型与期望的数据类型相同
        assert_equal(res.dtype, expected_dtype)
        # 断言合并结果与期望的结果数组相同
        assert_equal(res, expected)

    # 测试函数：检查带填充字段的数据类型是否正确处理
    def test_padded_dtype(self):
        # 定义数据类型 dt，包含一个 'i1' 和 '<f4' 类型的元组，并指定对齐为真
        dt = np.dtype('i1,f4', align=True)
        dt.names = ('k', 'v')
        # 断言 dt 的描述中包含 3 个元素，表示已插入填充字段
        assert_(len(dt.descr), 3)

        # 创建 NumPy 结构化数组 a 和 b，使用定义好的数据类型 dt
        a = np.array([(1, 3), (3, 2)], dt)
        b = np.array([(1, 1), (2, 2)], dt)
        # 调用 join_by 函数，按照 'k' 合并 a 和 b
        res = join_by('k', a, b)

        # 期望结果的数据类型，包含 'k'、'v1'（'f4'）、'v2'（'f4'）的元组
        expected_dtype = np.dtype([
            ('k', 'i1'), ('v1', 'f4'), ('v2', 'f4')
        ])

        # 断言合并结果的数据类型与期望的数据类型相同
        assert_equal(res.dtype, expected_dtype)
class TestJoinBy2:
    @classmethod
    def setup_method(cls):
        # 设置测试环境方法，创建两个结构化数组 a 和 b
        cls.a = np.array(list(zip(np.arange(10), np.arange(50, 60),
                                  np.arange(100, 110))),
                         dtype=[('a', int), ('b', int), ('c', int)])
        cls.b = np.array(list(zip(np.arange(10), np.arange(65, 75),
                                  np.arange(100, 110))),
                         dtype=[('a', int), ('b', int), ('d', int)])

    def test_no_r1postfix(self):
        # join_by 函数的基本测试，使用 r1postfix='' 和 r2postfix='2' 进行内连接
        a, b = self.a, self.b

        test = join_by(
            'a', a, b, r1postfix='', r2postfix='2', jointype='inner')
        # 预期的结果数组
        control = np.array([(0, 50, 65, 100, 100), (1, 51, 66, 101, 101),
                            (2, 52, 67, 102, 102), (3, 53, 68, 103, 103),
                            (4, 54, 69, 104, 104), (5, 55, 70, 105, 105),
                            (6, 56, 71, 106, 106), (7, 57, 72, 107, 107),
                            (8, 58, 73, 108, 108), (9, 59, 74, 109, 109)],
                           dtype=[('a', int), ('b', int), ('b2', int),
                                  ('c', int), ('d', int)])
        assert_equal(test, control)

    def test_no_postfix(self):
        # 测试异常情况：r1postfix='' 和 r2postfix='' 都为空时，应引发 ValueError
        assert_raises(ValueError, join_by, 'a', self.a, self.b,
                      r1postfix='', r2postfix='')

    def test_no_r2postfix(self):
        # join_by 函数的基本测试，使用 r1postfix='1' 和 r2postfix='' 进行内连接
        a, b = self.a, self.b

        test = join_by(
            'a', a, b, r1postfix='1', r2postfix='', jointype='inner')
        # 预期的结果数组
        control = np.array([(0, 50, 65, 100, 100), (1, 51, 66, 101, 101),
                            (2, 52, 67, 102, 102), (3, 53, 68, 103, 103),
                            (4, 54, 69, 104, 104), (5, 55, 70, 105, 105),
                            (6, 56, 71, 106, 106), (7, 57, 72, 107, 107),
                            (8, 58, 73, 108, 108), (9, 59, 74, 109, 109)],
                           dtype=[('a', int), ('b1', int), ('b', int),
                                  ('c', int), ('d', int)])
        assert_equal(test, control)
    # 定义一个测试函数，测试两个键和两个变量的情况
    def test_two_keys_two_vars(self):
        # 创建一个 NumPy 数组 `a`，其中每个元素是一个元组，元组中包含四个字段
        # 第一个字段 `k` 是重复的数组 [10, 11]，重复 5 次
        # 第二个字段 `a` 是重复的数组 [0, 1, 2, 3, 4]，每个元素重复 2 次
        # 第三个字段 `b` 是从 50 到 59 的连续数组
        # 第四个字段 `c` 是从 10 到 19 的连续数组
        a = np.array(list(zip(np.tile([10, 11], 5), np.repeat(np.arange(5), 2),
                              np.arange(50, 60), np.arange(10, 20))),
                     dtype=[('k', int), ('a', int), ('b', int), ('c', int)])

        # 创建另一个 NumPy 数组 `b`，结构与 `a` 相同
        # 但第三个字段 `b` 的值是从 65 到 74 的连续数组
        # 第四个字段 `c` 的值是从 0 到 9 的连续数组
        b = np.array(list(zip(np.tile([10, 11], 5), np.repeat(np.arange(5), 2),
                              np.arange(65, 75), np.arange(0, 10))),
                     dtype=[('k', int), ('a', int), ('b', int), ('c', int)])

        # 创建一个控制用的 NumPy 数组 `control`
        # 包含了预期的合并结果，有六个字段
        control = np.array([(10, 0, 50, 65, 10, 0), (11, 0, 51, 66, 11, 1),
                            (10, 1, 52, 67, 12, 2), (11, 1, 53, 68, 13, 3),
                            (10, 2, 54, 69, 14, 4), (11, 2, 55, 70, 15, 5),
                            (10, 3, 56, 71, 16, 6), (11, 3, 57, 72, 17, 7),
                            (10, 4, 58, 73, 18, 8), (11, 4, 59, 74, 19, 9)],
                           dtype=[('k', int), ('a', int), ('b1', int),
                                  ('b2', int), ('c1', int), ('c2', int)])

        # 调用 `join_by` 函数进行数组合并测试
        # 以 ['a', 'k'] 为键，将数组 `a` 和 `b` 进行内连接
        # 设置后缀 '1' 和 '2'，合并后的结果命名为 `test`
        test = join_by(
            ['a', 'k'], a, b, r1postfix='1', r2postfix='2', jointype='inner')

        # 断言合并后结果 `test` 的数据类型与 `control` 的数据类型相同
        assert_equal(test.dtype, control.dtype)
        
        # 断言合并后的结果 `test` 与预期的 `control` 结果相等
        assert_equal(test, control)
class TestAppendFieldsObj:
    """
    Test append_fields with arrays containing objects
    """
    # https://github.com/numpy/numpy/issues/2346

    def setup_method(self):
        # 在测试方法执行前设置数据对象，使用了日期对象作为示例数据
        from datetime import date
        self.data = dict(obj=date(2000, 1, 1))

    def test_append_to_objects(self):
        # 测试在基础数组包含对象时使用 append_fields 函数
        obj = self.data['obj']
        # 创建一个包含对象和浮点数的 NumPy 数组，定义了自定义数据类型
        x = np.array([(obj, 1.), (obj, 2.)],
                     dtype=[('A', object), ('B', float)])
        # 创建一个整数类型的 NumPy 数组
        y = np.array([10, 20], dtype=int)
        # 调用 append_fields 函数将整数数组 y 添加为 x 数组的新字段 'C'
        test = append_fields(x, 'C', data=y, usemask=False)
        # 创建一个控制组数组，验证是否正确追加了字段 'C'
        control = np.array([(obj, 1.0, 10), (obj, 2.0, 20)],
                           dtype=[('A', object), ('B', float), ('C', int)])
        # 使用断言检查测试结果是否与控制组一致
        assert_equal(test, control)
```