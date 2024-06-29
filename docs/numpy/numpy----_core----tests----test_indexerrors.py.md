# `.\numpy\numpy\_core\tests\test_indexerrors.py`

```py
# 导入 NumPy 库，并从中导入测试相关的函数
import numpy as np
from numpy.testing import (
        assert_raises, assert_raises_regex,
        )

# 定义一个测试类 TestIndexErrors，用于测试 IndexError 异常情况
class TestIndexErrors:
    '''Tests to exercise indexerrors not covered by other tests.'''

    # 定义测试方法 test_arraytypes_fasttake，测试在空维度上进行 take 操作时的 IndexError
    def test_arraytypes_fasttake(self):
        'take from a 0-length dimension'
        # 创建一个空的多维数组 x
        x = np.empty((2, 3, 0, 4))
        # 测试在空维度 axis=2 上进行 take 操作时的 IndexError
        assert_raises(IndexError, x.take, [0], axis=2)
        assert_raises(IndexError, x.take, [1], axis=2)
        assert_raises(IndexError, x.take, [0], axis=2, mode='wrap')
        assert_raises(IndexError, x.take, [0], axis=2, mode='clip')

    # 定义测试方法 test_take_from_object，测试从对象数组中进行 take 操作时的 IndexError
    def test_take_from_object(self):
        # 检查从对象数组 d 中取索引超出范围时的 IndexError
        d = np.zeros(5, dtype=object)
        assert_raises(IndexError, d.take, [6])

        # 检查从 0 维度对象数组 d 中进行 take 操作时的 IndexError
        d = np.zeros((5, 0), dtype=object)
        assert_raises(IndexError, d.take, [1], axis=1)
        assert_raises(IndexError, d.take, [0], axis=1)
        assert_raises(IndexError, d.take, [0])
        assert_raises(IndexError, d.take, [0], mode='wrap')
        assert_raises(IndexError, d.take, [0], mode='clip')

    # 定义测试方法 test_multiindex_exceptions，测试多维数组索引操作时的 IndexError
    def test_multiindex_exceptions(self):
        # 创建一个空对象数组 a
        a = np.empty(5, dtype=object)
        # 测试从对象数组 a 中超出索引范围的 IndexError
        assert_raises(IndexError, a.item, 20)
        # 创建一个 0 维度的对象数组 a
        a = np.empty((5, 0), dtype=object)
        # 测试从 0 维度对象数组 a 中进行索引操作时的 IndexError
        assert_raises(IndexError, a.item, (0, 0))

    # 定义测试方法 test_put_exceptions，测试在放置元素时的 IndexError
    def test_put_exceptions(self):
        # 创建一个 5x5 的零数组 a
        a = np.zeros((5, 5))
        # 测试向数组 a 中超出索引范围的位置放置元素时的 IndexError
        assert_raises(IndexError, a.put, 100, 0)
        # 创建一个元素为对象的 5x5 零数组 a
        a = np.zeros((5, 5), dtype=object)
        # 测试向元素为对象的数组 a 中超出索引范围的位置放置元素时的 IndexError
        assert_raises(IndexError, a.put, 100, 0)
        # 创建一个 5x5x0 的零数组 a
        a = np.zeros((5, 5, 0))
        # 测试向数组 a 中超出索引范围的位置放置元素时的 IndexError
        assert_raises(IndexError, a.put, 100, 0)
        # 创建一个元素为对象的 5x5x0 零数组 a
        a = np.zeros((5, 5, 0), dtype=object)
        # 测试向元素为对象的数组 a 中超出索引范围的位置放置元素时的 IndexError
        assert_raises(IndexError, a.put, 100, 0)
    # 定义一个测试方法，用于测试迭代器和异常情况
    def test_iterators_exceptions(self):
        # 内部辅助函数，用于给对象赋值
        def assign(obj, ind, val):
            obj[ind] = val
        
        # 创建一个形状为 [1, 2, 3] 的全零数组
        a = np.zeros([1, 2, 3])
        # 断言：索引错误异常应该被触发，lambda 函数访问超出数组边界的索引
        assert_raises(IndexError, lambda: a[0, 5, None, 2])
        assert_raises(IndexError, lambda: a[0, 5, 0, 2])
        assert_raises(IndexError, lambda: assign(a, (0, 5, None, 2), 1))
        assert_raises(IndexError, lambda: assign(a, (0, 5, 0, 2),  1))

        # 创建一个形状为 [1, 0, 3] 的全零数组
        a = np.zeros([1, 0, 3])
        # 断言：索引错误异常应该被触发，lambda 函数访问超出数组边界的索引
        assert_raises(IndexError, lambda: a[0, 0, None, 2])
        assert_raises(IndexError, lambda: assign(a, (0, 0, None, 2), 1))

        # 再次创建一个形状为 [1, 2, 3] 的全零数组
        a = np.zeros([1, 2, 3])
        # 断言：索引错误异常应该被触发，尝试通过 flat 属性访问超出数组边界的索引
        assert_raises(IndexError, lambda: a.flat[10])
        assert_raises(IndexError, lambda: assign(a.flat, 10, 5))
        
        # 创建一个形状为 [1, 0, 3] 的全零数组
        a = np.zeros([1, 0, 3])
        # 断言：索引错误异常应该被触发，尝试通过 flat 属性访问超出数组边界的索引
        assert_raises(IndexError, lambda: a.flat[10])
        assert_raises(IndexError, lambda: assign(a.flat, 10, 5))
        
        # 再次创建一个形状为 [1, 2, 3] 的全零数组
        a = np.zeros([1, 2, 3])
        # 断言：索引错误异常应该被触发，尝试通过 flat 属性访问超出数组边界的索引
        assert_raises(IndexError, lambda: a.flat[np.array(10)])
        assert_raises(IndexError, lambda: assign(a.flat, np.array(10), 5))
        
        # 创建一个形状为 [1, 0, 3] 的全零数组
        a = np.zeros([1, 0, 3])
        # 断言：索引错误异常应该被触发，尝试通过 flat 属性访问超出数组边界的索引
        assert_raises(IndexError, lambda: a.flat[np.array(10)])
        assert_raises(IndexError, lambda: assign(a.flat, np.array(10), 5))
        
        # 再次创建一个形状为 [1, 2, 3] 的全零数组
        a = np.zeros([1, 2, 3])
        # 断言：索引错误异常应该被触发，尝试通过 flat 属性访问超出数组边界的索引
        assert_raises(IndexError, lambda: a.flat[np.array([10])])
        assert_raises(IndexError, lambda: assign(a.flat, np.array([10]), 5))
        
        # 创建一个形状为 [1, 0, 3] 的全零数组
        a = np.zeros([1, 0, 3])
        # 断言：索引错误异常应该被触发，尝试通过 flat 属性访问超出数组边界的索引
        assert_raises(IndexError, lambda: a.flat[np.array([10])])
        assert_raises(IndexError, lambda: assign(a.flat, np.array([10]), 5))

    # 定义一个测试方法，用于测试映射操作
    def test_mapping(self):
        # 内部辅助函数，用于给对象赋值
        def assign(obj, ind, val):
            obj[ind] = val
        
        # 创建一个形状为 [0, 10] 的全零数组
        a = np.zeros((0, 10))
        # 断言：索引错误异常应该被触发，尝试访问超出数组边界的索引
        assert_raises(IndexError, lambda: a[12])

        # 创建一个形状为 [3, 5] 的全零数组
        a = np.zeros((3, 5))
        # 断言：索引错误异常应该被触发，lambda 函数访问超出数组边界的索引
        assert_raises(IndexError, lambda: a[(10, 20)])
        assert_raises(IndexError, lambda: assign(a, (10, 20), 1))
        
        # 创建一个形状为 [3, 0] 的全零数组
        a = np.zeros((3, 0))
        # 断言：索引错误异常应该被触发，lambda 函数访问超出数组边界的索引
        assert_raises(IndexError, lambda: a[(1, 0)])
        assert_raises(IndexError, lambda: assign(a, (1, 0), 1))

        # 创建一个形状为 [10,] 的全零数组
        a = np.zeros((10,))
        # 断言：索引错误异常应该被触发，lambda 函数访问超出数组边界的索引
        assert_raises(IndexError, lambda: assign(a, 10, 1))
        
        # 创建一个形状为 [0,] 的全零数组
        a = np.zeros((0,))
        # 断言：索引错误异常应该被触发，lambda 函数访问超出数组边界的索引
        assert_raises(IndexError, lambda: assign(a, 10, 1))

        # 再次创建一个形状为 [3, 5] 的全零数组
        a = np.zeros((3, 5))
        # 断言：索引错误异常应该被触发，lambda 函数访问超出数组边界的索引
        assert_raises(IndexError, lambda: a[(1, [1, 20])])
        assert_raises(IndexError, lambda: assign(a, (1, [1, 20]), 1))
        
        # 创建一个形状为 [3, 0] 的全零数组
        a = np.zeros((3, 0))
        # 断言：索引错误异常应该被触发，lambda 函数访问超出数组边界的索引
        assert_raises(IndexError, lambda: a[(1, [0, 1])])
        assert_raises(IndexError, lambda: assign(a, (1, [0, 1]), 1))

    # 定义一个测试方法，用于测试映射操作的错误消息
    def test_mapping_error_message(self):
        # 创建一个形状为 [3, 5] 的全零数组
        a = np.zeros((3, 5))
        # 创建一个超出数组维度的索引元组
        index = (1, 2, 3, 4, 5)
        # 断言：索引错误异常应该被触发，错误消息应指出尝试索引超出数组维度
        assert_raises_regex(
                IndexError,
                "too many indices for array: "
                "array is 2-dimensional, but 5 were indexed",
                lambda: a[index])
    # 定义一个测试方法，用于测试与 methods.c 文件相关的案例
    def test_methods(self):
        # 创建一个 3x3 的全零数组
        a = np.zeros((3, 3))
        # 断言：调用 a.item(100) 应该引发 IndexError 异常
        assert_raises(IndexError, lambda: a.item(100))
```