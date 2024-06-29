# `.\numpy\numpy\_core\tests\test_item_selection.py`

```
import sys  # 导入sys模块，用于获取系统相关信息

import pytest  # 导入pytest模块，用于编写和运行测试用例

import numpy as np  # 导入NumPy库，并使用np作为别名
from numpy.testing import (  # 从NumPy测试模块中导入多个函数和类
    assert_, assert_raises, assert_array_equal, HAS_REFCOUNT
    )


class TestTake:  # 定义测试类TestTake

    def test_simple(self):  # 测试简单取值函数
        a = [[1, 2], [3, 4]]  # 定义一个二维列表a
        a_str = [[b'1', b'2'], [b'3', b'4']]  # 定义一个二维列表a_str，包含字节字符串
        modes = ['raise', 'wrap', 'clip']  # 模式列表，用于测试取值的不同模式
        indices = [-1, 4]  # 索引列表，用于测试取值的不同索引
        index_arrays = [np.empty(0, dtype=np.intp),  # 索引数组列表，包含三个不同形状的空数组
                        np.empty(tuple(), dtype=np.intp),
                        np.empty((1, 1), dtype=np.intp)]
        real_indices = {'raise': {-1: 1, 4: IndexError},  # 实际索引字典，包含不同模式下的预期结果
                        'wrap': {-1: 1, 4: 0},
                        'clip': {-1: 0, 4: 1}}

        # Currently all types but object, use the same function generation.
        # So it should not be necessary to test all. However test also a non
        # refcounted struct on top of object, which has a size that hits the
        # default (non-specialized) path.
        types = int, object, np.dtype([('', 'i2', 3)])  # 类型元组，包含int、object和自定义dtype类型
        for t in types:  # 遍历类型元组中的每个类型
            ta = np.array(a if np.issubdtype(t, np.number) else a_str, dtype=t)  # 根据类型创建NumPy数组ta
            tresult = list(ta.T.copy())  # 复制数组ta的转置并转换为列表，赋值给tresult
            for index_array in index_arrays:  # 遍历索引数组列表中的每个索引数组
                if index_array.size != 0:  # 如果索引数组不为空
                    tresult[0].shape = (2,) + index_array.shape  # 调整tresult[0]的形状
                    tresult[1].shape = (2,) + index_array.shape  # 调整tresult[1]的形状
                for mode in modes:  # 遍历模式列表中的每个模式
                    for index in indices:  # 遍历索引列表中的每个索引
                        real_index = real_indices[mode][index]  # 获取实际索引值
                        if real_index is IndexError and index_array.size != 0:  # 如果预期结果为IndexError且索引数组不为空
                            index_array.put(0, index)  # 将索引放入索引数组的第一个位置
                            assert_raises(IndexError, ta.take, index_array,  # 断言引发IndexError异常
                                          mode=mode, axis=1)
                        elif index_array.size != 0:  # 如果索引数组不为空
                            index_array.put(0, index)  # 将索引放入索引数组的第一个位置
                            res = ta.take(index_array, mode=mode, axis=1)  # 使用指定模式和轴取值
                            assert_array_equal(res, tresult[real_index])  # 断言结果数组与预期相等
                        else:  # 如果索引数组为空
                            res = ta.take(index_array, mode=mode, axis=1)  # 使用指定模式和轴取值
                            assert_(res.shape == (2,) + index_array.shape)  # 断言结果数组的形状符合预期

    def test_refcounting(self):  # 测试引用计数功能
        objects = [object() for i in range(10)]  # 创建包含10个新对象的列表
        for mode in ('raise', 'clip', 'wrap'):  # 遍历模式元组
            a = np.array(objects)  # 创建包含对象的NumPy数组a
            b = np.array([2, 2, 4, 5, 3, 5])  # 创建包含整数的NumPy数组b
            a.take(b, out=a[:6], mode=mode)  # 使用指定模式从a中取值到a的前6个位置
            del a  # 删除数组a的引用
            if HAS_REFCOUNT:  # 如果支持引用计数
                assert_(all(sys.getrefcount(o) == 3 for o in objects))  # 断言所有对象的引用计数为3
            # not contiguous, example:
            a = np.array(objects * 2)[::2]  # 创建包含对象的非连续NumPy数组a
            a.take(b, out=a[:6], mode=mode)  # 使用指定模式从a中取值到a的前6个位置
            del a  # 删除数组a的引用
            if HAS_REFCOUNT:  # 如果支持引用计数
                assert_(all(sys.getrefcount(o) == 3 for o in objects))  # 断言所有对象的引用计数为3
    # 测试Unicode模式下的函数
    def test_unicode_mode(self):
        # 创建一个包含0到9的数组
        d = np.arange(10)
        # 将一个UTF8编码的字节序列解码为Unicode字符串
        k = b'\xc3\xa4'.decode("UTF8")
        # 断言调用d.take(5, mode=k)应该会触发ValueError异常
        assert_raises(ValueError, d.take, 5, mode=k)
    
    # 测试空分区的函数
    def test_empty_partition(self):
        # 创建一个包含[0, 2, 4, 6, 8, 10]的数组，并进行复制
        a_original = np.array([0, 2, 4, 6, 8, 10])
        a = a_original.copy()
    
        # 对空数组进行分区应该是一个成功的空操作
        a.partition(np.array([], dtype=np.int16))
    
        # 断言a分区后的结果应该和a_original保持一致
        assert_array_equal(a, a_original)
    
    # 测试空argpartition的函数
    def test_empty_argpartition(self):
        # 在参考github问题＃6530的情况下
        a = np.array([0, 2, 4, 6, 8, 10])
        # 对空数组进行argpartition
        a = a.argpartition(np.array([], dtype=np.int16))
    
        # 创建一个包含[0, 1, 2, 3, 4, 5]的数组
        b = np.array([0, 1, 2, 3, 4, 5])
        # 断言a的argpartition后的结果应该和b保持一致
        assert_array_equal(a, b)
class TestPutMask:
    @pytest.mark.parametrize("dtype", list(np.typecodes["All"]) + ["i,O"])
    def test_simple(self, dtype):
        if dtype.lower() == "m":
            dtype += "8[ns]"

        # 创建一个从0到1000的numpy数组，并根据dtype参数转换其数据类型
        vals = np.arange(1001).astype(dtype=dtype)

        # 创建一个长度为1000的布尔掩码数组
        mask = np.random.randint(2, size=1000).astype(bool)
        
        # 根据vals数组的数据类型创建一个长度为1000的全零数组
        arr = np.zeros(1000, dtype=vals.dtype)
        
        # 创建一个arr的副本
        zeros = arr.copy()

        # 使用np.putmask函数，根据掩码mask将vals数组中的值放入arr数组中
        np.putmask(arr, mask, vals)
        
        # 断言：检查arr中掩码为True的位置，其值与vals中相应位置的值相等
        assert_array_equal(arr[mask], vals[:len(mask)][mask])
        
        # 断言：检查arr中掩码为False的位置，其值与zeros中相应位置的值相等
        assert_array_equal(arr[~mask], zeros[~mask])

    @pytest.mark.parametrize("dtype", list(np.typecodes["All"])[1:] + ["i,O"])
    @pytest.mark.parametrize("mode", ["raise", "wrap", "clip"])
    def test_empty(self, dtype, mode):
        # 创建一个长度为1000的dtype类型全零数组
        arr = np.zeros(1000, dtype=dtype)
        
        # 创建arr的一个副本
        arr_copy = arr.copy()
        
        # 创建一个长度为1000的布尔掩码数组
        mask = np.random.randint(2, size=1000).astype(bool)

        # 使用空列表来调用np.put函数，这种用法看起来很奇怪...
        np.put(arr, mask, [])
        
        # 断言：检查arr与其副本arr_copy是否相等
        assert_array_equal(arr, arr_copy)


class TestPut:
    @pytest.mark.parametrize("dtype", list(np.typecodes["All"])[1:] + ["i,O"])
    @pytest.mark.parametrize("mode", ["raise", "wrap", "clip"])
    def test_simple(self, dtype, mode):
        if dtype.lower() == "m":
            dtype += "8[ns]"

        # 创建一个从0到1000的numpy数组，并根据dtype参数转换其数据类型
        vals = np.arange(1001).astype(dtype=dtype)

        # 使用vals的数据类型创建一个长度为1000的全零数组
        arr = np.zeros(1000, dtype=vals.dtype)
        
        # 创建一个arr的副本
        zeros = arr.copy()

        if mode == "clip":
            # 特殊情况，因为0和-1值用于clip测试
            indx = np.random.permutation(len(arr) - 2)[:-500] + 1

            indx[-1] = 0
            indx[-2] = len(arr) - 1
            indx_put = indx.copy()
            indx_put[-1] = -1389
            indx_put[-2] = 1321
        else:
            # 避免重复值（简化起见）并只填充一半
            indx = np.random.permutation(len(arr) - 3)[:-500]
            indx_put = indx
            if mode == "wrap":
                indx_put = indx_put + len(arr)

        # 使用np.put函数，根据indx_put数组将vals数组中的值放入arr数组中
        np.put(arr, indx_put, vals, mode=mode)
        
        # 断言：检查arr中indx数组位置的值与vals数组中相应位置的值是否相等
        assert_array_equal(arr[indx], vals[:len(indx)])
        
        # 创建一个全为True的布尔数组
        untouched = np.ones(len(arr), dtype=bool)
        
        # 将indx数组位置的值设置为False
        untouched[indx] = False
        
        # 断言：检查arr中untouched数组位置的值与zeros数组中相应位置的值是否相等
        assert_array_equal(arr[untouched], zeros[untouched])

    @pytest.mark.parametrize("dtype", list(np.typecodes["All"])[1:] + ["i,O"])
    @pytest.mark.parametrize("mode", ["raise", "wrap", "clip"])
    def test_empty(self, dtype, mode):
        # 创建一个长度为1000的dtype类型全零数组
        arr = np.zeros(1000, dtype=dtype)
        
        # 创建arr的一个副本
        arr_copy = arr.copy()

        # 使用列表[1, 2, 3]调用np.put函数，并传入一个空列表作为值，这种用法看起来很奇怪...
        np.put(arr, [1, 2, 3], [])
        
        # 断言：检查arr与其副本arr_copy是否相等
        assert_array_equal(arr, arr_copy)
```