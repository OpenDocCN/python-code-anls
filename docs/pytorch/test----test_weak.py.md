# `.\pytorch\test\test_weak.py`

```
# Owner(s): ["module: meta tensors"]

# 导入必要的模块和库
import copy  # 导入 copy 模块，用于复制对象
import gc  # 导入 gc 模块，用于垃圾回收
import random  # 导入 random 模块，用于生成随机数
import threading  # 导入 threading 模块，用于多线程编程

import unittest  # 导入 unittest 模块，用于编写和运行单元测试

import torch  # 导入 PyTorch 模块
from torch.testing._internal.common_utils import (  # 导入测试中使用的常用函数和类
    find_library_location,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    run_tests,
    TestCase,
)
from torch.utils.weak import _WeakHashRef, WeakIdKeyDictionary  # 导入弱引用相关的类


def C():
    return torch.randn(1)  # 返回一个随机生成的张量


# 以下是一系列使用 WeakIdKeyDictionary 进行测试的单元测试类
# 这些测试类从 cpython/Lib/test/test_weakref.py 移植而来，但使用张量代替了普通对象
class WeakTest(TestCase):
    COUNT = 10  # 定义类变量 COUNT，值为 10

    def test_make_weak_keyed_dict_from_dict(self):
        o = torch.randn(2)  # 创建一个张量对象 o
        dict = WeakIdKeyDictionary({o: 364})  # 使用张量 o 创建一个 WeakIdKeyDictionary 对象 dict
        self.assertEqual(dict[o], 364)  # 断言张量 o 在 dict 中对应的值为 364

    def test_make_weak_keyed_dict_from_weak_keyed_dict(self):
        o = torch.randn(3)  # 创建一个张量对象 o
        dict = WeakIdKeyDictionary({o: 364})  # 使用张量 o 创建一个 WeakIdKeyDictionary 对象 dict
        dict2 = WeakIdKeyDictionary(dict)  # 使用 dict 创建另一个 WeakIdKeyDictionary 对象 dict2
        self.assertEqual(dict[o], 364)  # 断言张量 o 在 dict 中对应的值为 364

    def check_popitem(self, klass, key1, value1, key2, value2):
        weakdict = klass()  # 使用给定的类创建一个对象 weakdict
        weakdict[key1] = value1  # 向 weakdict 中添加键值对 (key1, value1)
        weakdict[key2] = value2  # 向 weakdict 中添加键值对 (key2, value2)
        self.assertEqual(len(weakdict), 2)  # 断言 weakdict 的长度为 2
        k, v = weakdict.popitem()  # 弹出并返回 weakdict 中的一项
        self.assertEqual(len(weakdict), 1)  # 断言 weakdict 的长度为 1
        if k is key1:
            self.assertIs(v, value1)  # 断言弹出的键值对与预期相符
        else:
            self.assertIs(v, value2)  # 断言弹出的键值对与预期相符
        k, v = weakdict.popitem()  # 再次弹出并返回 weakdict 中的一项
        self.assertEqual(len(weakdict), 0)  # 断言 weakdict 的长度为 0
        if k is key1:
            self.assertIs(v, value1)  # 断言弹出的键值对与预期相符
        else:
            self.assertIs(v, value2)  # 断言弹出的键值对与预期相符

    def test_weak_keyed_dict_popitem(self):
        self.check_popitem(WeakIdKeyDictionary, C(), "value 1", C(), "value 2")  # 测试 popitem 方法

    def check_setdefault(self, klass, key, value1, value2):
        self.assertIsNot(
            value1,
            value2,
            "invalid test -- value parameters must be distinct objects",
        )  # 断言 value1 和 value2 是不同的对象
        weakdict = klass()  # 使用给定的类创建一个对象 weakdict
        o = weakdict.setdefault(key, value1)  # 设置键 key 的默认值为 value1
        self.assertIs(o, value1)  # 断言返回的值与 value1 相同
        self.assertIn(key, weakdict)  # 断言键 key 在 weakdict 中
        self.assertIs(weakdict.get(key), value1)  # 断言获取键 key 对应的值为 value1
        self.assertIs(weakdict[key], value1)  # 断言 weakdict 中键 key 对应的值为 value1

        o = weakdict.setdefault(key, value2)  # 再次设置键 key 的默认值为 value2
        self.assertIs(o, value1)  # 断言返回的值与 value1 相同
        self.assertIn(key, weakdict)  # 断言键 key 在 weakdict 中
        self.assertIs(weakdict.get(key), value1)  # 断言获取键 key 对应的值为 value1
        self.assertIs(weakdict[key], value1)  # 断言 weakdict 中键 key 对应的值为 value1

    def test_weak_keyed_dict_setdefault(self):
        self.check_setdefault(WeakIdKeyDictionary, C(), "value 1", "value 2")  # 测试 setdefault 方法
    def check_update(self, klass, dict):
        #
        #  This exercises d.update(), len(d), d.keys(), k in d,
        #  d.get(), d[].
        #
        # 创建一个新的弱引用字典对象
        weakdict = klass()
        # 使用给定的字典更新弱引用字典
        weakdict.update(dict)
        # 断言弱引用字典的长度与给定字典的长度相等
        self.assertEqual(len(weakdict), len(dict))
        # 遍历弱引用字典的所有键
        for k in weakdict.keys():
            # 断言当前键在原始字典中存在
            self.assertIn(k, dict, "mysterious new key appeared in weak dict")
            # 获取原始字典中当前键对应的值
            v = dict.get(k)
            # 断言弱引用字典中当前键对应的值与原始字典中相同
            self.assertIs(v, weakdict[k])
            # 再次断言弱引用字典中当前键对应的值与原始字典中相同
            self.assertIs(v, weakdict.get(k))
        # 遍历原始字典的所有键
        for k in dict.keys():
            # 断言当前键在弱引用字典中存在
            self.assertIn(k, weakdict, "original key disappeared in weak dict")
            # 获取原始字典中当前键对应的值
            v = dict[k]
            # 断言弱引用字典中当前键对应的值与原始字典中相同
            self.assertIs(v, weakdict[k])
            # 再次断言弱引用字典中当前键对应的值与原始字典中相同
            self.assertIs(v, weakdict.get(k))

    def test_weak_keyed_dict_update(self):
        self.check_update(WeakIdKeyDictionary, {C(): 1, C(): 2, C(): 3})

    def test_weak_keyed_delitem(self):
        d = WeakIdKeyDictionary()
        o1 = torch.randn(1)
        o2 = torch.randn(2)
        d[o1] = "something"
        d[o2] = "something"
        # 断言字典长度为2
        self.assertEqual(len(d), 2)
        # 删除字典中键为o1的项
        del d[o1]
        # 断言字典长度为1
        self.assertEqual(len(d), 1)
        # 断言字典的键列表为[o2]
        self.assertEqual(list(d.keys()), [o2])

    def test_weak_keyed_union_operators(self):
        try:
            {} | {}
        except TypeError:
            # 如果当前 Python 版本不支持字典的 union 操作，则跳过测试
            self.skipTest("dict union not supported in this Python")

        o1 = C()
        o2 = C()
        o3 = C()
        # 创建两个弱引用键字典对象并初始化
        wkd1 = WeakIdKeyDictionary({o1: 1, o2: 2})
        wkd2 = WeakIdKeyDictionary({o3: 3, o1: 4})
        # 对 wkd1 进行浅拷贝
        wkd3 = wkd1.copy()
        d1 = {o2: "5", o3: "6"}
        pairs = [(o2, 7), (o3, 8)]

        # 测试两个弱引用键字典对象的合并操作
        tmp1 = wkd1 | wkd2  # Between two WeakKeyDictionaries
        self.assertEqual(dict(tmp1), dict(wkd1) | dict(wkd2))
        self.assertIs(type(tmp1), WeakIdKeyDictionary)
        wkd1 |= wkd2
        self.assertEqual(wkd1, tmp1)

        # 测试弱引用键字典对象与普通映射类型的合并操作
        tmp2 = wkd2 | d1  # Between WeakKeyDictionary and mapping
        self.assertEqual(dict(tmp2), dict(wkd2) | d1)
        self.assertIs(type(tmp2), WeakIdKeyDictionary)
        wkd2 |= d1
        self.assertEqual(wkd2, tmp2)

        # 测试弱引用键字典对象与可迭代的键值对的合并操作
        tmp3 = wkd3.copy()  # Between WeakKeyDictionary and iterable key, value
        tmp3 |= pairs
        self.assertEqual(dict(tmp3), dict(wkd3) | dict(pairs))
        self.assertIs(type(tmp3), WeakIdKeyDictionary)

        # 测试映射类型与弱引用键字典对象的合并操作（测试.__ror__方法）
        tmp4 = d1 | wkd3  # Testing .__ror__
        self.assertEqual(dict(tmp4), d1 | dict(wkd3))
        self.assertIs(type(tmp4), WeakIdKeyDictionary)

        # 删除变量 o1，检查相关值是否不在合并结果中
        del o1
        self.assertNotIn(4, tmp1.values())
        self.assertNotIn(4, tmp2.values())
        self.assertNotIn(1, tmp3.values())
        self.assertNotIn(1, tmp4.values())
    # 测试弱引用键字典的删除操作
    def test_weak_keyed_bad_delitem(self):
        # 创建一个 WeakIdKeyDictionary 实例
        d = WeakIdKeyDictionary()
        # 创建一个随机张量对象 o
        o = torch.randn(1)
        # 尝试删除一个不存在的对象应该引发 KeyError，与 2.3 版本前不同
        self.assertRaises(KeyError, d.__delitem__, o)
        # 尝试获取一个不存在的对象应该引发 KeyError
        self.assertRaises(KeyError, d.__getitem__, o)

        # 如果键不是弱引用类型，则 __getitem__ 和 __setitem__ 应该引发 TypeError，__delitem__ 也应该如此
        self.assertRaises(TypeError, d.__delitem__, 13)
        self.assertRaises(TypeError, d.__getitem__, 13)
        self.assertRaises(TypeError, d.__setitem__, 13, 13)

    # 测试弱引用键字典的字符串表示形式
    def test_make_weak_keyed_dict_repr(self):
        # 创建一个 WeakIdKeyDictionary 实例
        dict = WeakIdKeyDictionary()
        # 断言字典对象的字符串表示形式符合特定模式
        self.assertRegex(repr(dict), "<WeakIdKeyDictionary at 0x.*>")

    # 检查多线程环境下弱引用字典的复制操作
    def check_threaded_weak_dict_copy(self, type_, deepcopy):
        # `deepcopy` 应为 True 或 False
        exc = []

        # 以下两个类不支持弱引用，因为直到后来的 Python 版本才支持这些对象
        class DummyKey:  # noqa: B903
            def __init__(self, ctr):
                self.ctr = ctr

        class DummyValue:  # noqa: B903
            def __init__(self, ctr):
                self.ctr = ctr

        # 执行字典的复制操作
        def dict_copy(d, exc):
            try:
                if deepcopy is True:
                    _ = copy.deepcopy(d)
                else:
                    _ = d.copy()
            except Exception as ex:
                exc.append(ex)

        # 从列表中弹出对象并进行垃圾回收
        def pop_and_collect(lst):
            gc_ctr = 0
            while lst:
                i = random.randint(0, len(lst) - 1)
                gc_ctr += 1
                lst.pop(i)
                if gc_ctr % 10000 == 0:
                    gc.collect()  # 以防万一

        # 创建一个 type_ 的实例对象 d
        d = type_()
        keys = []
        values = []
        # 初始化 d，添加大量条目
        for i in range(70000):
            k, v = DummyKey(i), DummyValue(i)
            keys.append(k)
            values.append(v)
            d[k] = v
            del k
            del v

        # 创建两个线程：一个执行字典复制，另一个弹出并回收键
        t_copy = threading.Thread(
            target=dict_copy,
            args=(
                d,
                exc,
            ),
        )
        t_collect = threading.Thread(target=pop_and_collect, args=(keys,))

        t_copy.start()
        t_collect.start()

        t_copy.join()
        t_collect.join()

        # 如果出现异常则抛出第一个异常
        if exc:
            raise exc[0]

    # 测试多线程环境下弱引用键字典的浅复制操作
    def test_threaded_weak_key_dict_copy(self):
        # 问题 #35615: 在复制字典期间，弱引用键或值被 GC 应不会导致崩溃
        self.check_threaded_weak_dict_copy(WeakIdKeyDictionary, False)

    # 测试多线程环境下弱引用键字典的深复制操作
    def test_threaded_weak_key_dict_deepcopy(self):
        # 问题 #35615: 在复制字典期间，弱引用键或值被 GC 应不会导致崩溃
        self.check_threaded_weak_dict_copy(WeakIdKeyDictionary, True)
#`
# 从 cpython/Lib/test/mapping_tests.py 改编的测试用例
class WeakKeyDictionaryTestCase(TestCase):
    # 定义一个类属性 __ref，包含多个随机生成的张量作为键和整数作为值
    __ref = {torch.randn(1): 1, torch.randn(2): 2, torch.randn(3): 3}
    # 定义一个类属性 type2test，指定使用的字典类型为 WeakIdKeyDictionary
    type2test = WeakIdKeyDictionary

    # 定义一个方法，返回 __ref 的副本
    def _reference(self):
        return self.__ref.copy()

    # 定义一个方法，返回一个空的映射对象
    def _empty_mapping(self):
        """Return an empty mapping object"""
        return self.type2test()

    # 定义一个方法，返回一个包含 data 字典中数据的映射对象
    def _full_mapping(self, data):
        """Return a mapping object with the value contained in data
        dictionary"""
        x = self._empty_mapping()
        for key, value in data.items():
            x[key] = value
        return x

    # 初始化方法，设置测试用例的基本属性
    def __init__(self, *args, **kw):
        unittest.TestCase.__init__(self, *args, **kw)
        # 创建一个 reference 的副本
        self.reference = self._reference().copy()

        # 从 reference 中弹出一个 (key, value) 对，赋值给 other
        key, value = self.reference.popitem()
        self.other = {key: value}

        # 从 reference 中弹出一个 (key, value) 对，赋值给 inmapping，并将其重新添加到 reference 中
        key, value = self.reference.popitem()
        self.inmapping = {key: value}
        self.reference[key] = value
    def test_read(self):
        # 测试只读操作对映射的影响
        p = self._empty_mapping()
        p1 = dict(p)  # 解决单例对象的问题
        d = self._full_mapping(self.reference)
        if d is p:
            p = p1
        # 索引操作
        for key, value in self.reference.items():
            self.assertEqual(d[key], value)
        knownkey = next(iter(self.other.keys()))
        self.assertRaises(KeyError, lambda: d[knownkey])
        # len 方法测试
        self.assertEqual(len(p), 0)
        self.assertEqual(len(d), len(self.reference))
        # __contains__ 方法测试
        for k in self.reference:
            self.assertIn(k, d)
        for k in self.other:
            self.assertNotIn(k, d)
        # cmp 操作测试
        self.assertTrue(
            p == p
        )  # 注意：不要使用 assertEqual，因为它实际上不使用 ==
        self.assertTrue(d == d)
        self.assertTrue(p != d)
        self.assertTrue(d != p)
        # bool 测试
        if p:
            self.fail("空映射必须比较为 False")
        if not d:
            self.fail("完整映射必须比较为 True")

        # keys(), items(), iterkeys() 等方法测试
        def check_iterandlist(iter, lst, ref):
            self.assertTrue(hasattr(iter, "__next__"))
            self.assertTrue(hasattr(iter, "__iter__"))
            x = list(iter)
            self.assertTrue(set(x) == set(lst) == set(ref))

        check_iterandlist(iter(d.keys()), list(d.keys()), self.reference.keys())
        check_iterandlist(iter(d), list(d.keys()), self.reference.keys())
        check_iterandlist(iter(d.values()), list(d.values()), self.reference.values())
        check_iterandlist(iter(d.items()), list(d.items()), self.reference.items())
        # get 方法测试
        key, value = next(iter(d.items()))
        knownkey, knownvalue = next(iter(self.other.items()))
        self.assertEqual(d.get(key, knownvalue), value)
        self.assertEqual(d.get(knownkey, knownvalue), knownvalue)
        self.assertNotIn(knownkey, d)
    def test_write(self):
        # Test for write operations on mapping

        # 创建一个空的映射对象
        p = self._empty_mapping()

        # 索引操作
        for key, value in self.reference.items():
            # 将参考映射中的键值对写入映射对象
            p[key] = value
            # 断言确保写入后能正确读取
            self.assertEqual(p[key], value)

        # 删除操作
        for key in self.reference.keys():
            # 删除映射对象中的键
            del p[key]
            # 使用lambda表达式断言删除的键不存在于映射对象中
            self.assertRaises(KeyError, lambda: p[key])

        # 重新创建一个空的映射对象
        p = self._empty_mapping()

        # 更新操作
        p.update(self.reference)
        # 断言映射对象与参考映射相等
        self.assertEqual(dict(p), self.reference)

        # 使用items列表进行更新操作
        items = list(p.items())
        p = self._empty_mapping()
        p.update(items)
        # 再次断言映射对象与参考映射相等
        self.assertEqual(dict(p), self.reference)

        # 使用reference创建一个完整的映射对象d
        d = self._full_mapping(self.reference)

        # setdefault操作
        key, value = next(iter(d.items()))
        knownkey, knownvalue = next(iter(self.other.items()))

        # 断言设置默认值后能正确获取值
        self.assertEqual(d.setdefault(key, knownvalue), value)
        self.assertEqual(d[key], value)

        # 再次使用setdefault操作
        self.assertEqual(d.setdefault(knownkey, knownvalue), knownvalue)
        self.assertEqual(d[knownkey], knownvalue)

        # pop操作
        self.assertEqual(d.pop(knownkey), knownvalue)
        self.assertNotIn(knownkey, d)
        self.assertRaises(KeyError, d.pop, knownkey)

        # 设置默认值为909进行pop操作
        default = 909
        d[knownkey] = knownvalue
        self.assertEqual(d.pop(knownkey, default), knownvalue)
        self.assertNotIn(knownkey, d)
        self.assertEqual(d.pop(knownkey, default), default)

        # popitem操作
        key, value = d.popitem()
        self.assertNotIn(key, d)
        self.assertEqual(value, self.reference[key])

        # 创建一个空的映射对象p，并断言popitem操作引发KeyError异常
        p = self._empty_mapping()
        self.assertRaises(KeyError, p.popitem)

    def test_constructor(self):
        # 断言空映射对象与自身相等
        self.assertEqual(self._empty_mapping(), self._empty_mapping())

    def test_bool(self):
        # 测试空映射对象的布尔值为False
        self.assertTrue(not self._empty_mapping())
        # 测试非空映射对象的布尔值为True
        self.assertTrue(self.reference)
        # 断言空映射对象的布尔值为False
        self.assertTrue(bool(self._empty_mapping()) is False)
        # 断言非空映射对象的布尔值为True
        self.assertTrue(bool(self.reference) is True)

    def test_keys(self):
        # 创建一个空映射对象d
        d = self._empty_mapping()
        # 断言空映射对象的keys方法返回空列表
        self.assertEqual(list(d.keys()), [])

        # 将reference赋值给d
        d = self.reference
        # 断言inmapping的第一个键在d的keys方法返回的列表中
        self.assertIn(next(iter(self.inmapping.keys())), d.keys())
        # 断言other的第一个键不在d的keys方法返回的列表中
        self.assertNotIn(next(iter(self.other.keys())), d.keys())
        # 断言传入None参数调用keys方法会引发TypeError异常
        self.assertRaises(TypeError, d.keys, None)

    def test_values(self):
        # 创建一个空映射对象d
        d = self._empty_mapping()
        # 断言空映射对象的values方法返回空列表
        self.assertEqual(list(d.values()), [])

        # 断言传入None参数调用values方法会引发TypeError异常
        self.assertRaises(TypeError, d.values, None)

    def test_items(self):
        # 创建一个空映射对象d
        d = self._empty_mapping()
        # 断言空映射对象的items方法返回空列表
        self.assertEqual(list(d.items()), [])

        # 断言传入None参数调用items方法会引发TypeError异常
        self.assertRaises(TypeError, d.items, None)

    def test_len(self):
        # 创建一个空映射对象d
        d = self._empty_mapping()
        # 断言空映射对象的长度为0
        self.assertEqual(len(d), 0)

    def test_getitem(self):
        # 将reference赋值给映射对象d
        d = self.reference
        # 断言inmapping的第一个键对应的值与d中对应的值相等
        self.assertEqual(
            d[next(iter(self.inmapping.keys()))], next(iter(self.inmapping.values()))
        )

        # 断言调用__getitem__方法时不传入参数会引发TypeError异常
        self.assertRaises(TypeError, d.__getitem__)
    # 测试 get 方法

    def test_get(self):
        # 使用空映射创建字典 d
        d = self._empty_mapping()
        # 断言获取不存在的键返回 None
        self.assertTrue(d.get(next(iter(self.other.keys()))) is None)
        # 断言获取不存在的键并提供默认值返回默认值
        self.assertEqual(d.get(next(iter(self.other.keys())), 3), 3)
        
        # 将引用的映射赋值给 d
        d = self.reference
        # 断言获取不存在的键返回 None
        self.assertTrue(d.get(next(iter(self.other.keys()))) is None)
        # 断言获取不存在的键并提供默认值返回默认值
        self.assertEqual(d.get(next(iter(self.other.keys())), 3), 3)
        
        # 断言获取存在的键返回对应的值
        self.assertEqual(
            d.get(next(iter(self.inmapping.keys()))),
            next(iter(self.inmapping.values())),
        )
        # 断言获取存在的键并提供默认值返回对应的值
        self.assertEqual(
            d.get(next(iter(self.inmapping.keys())), 3),
            next(iter(self.inmapping.values())),
        )
        
        # 断言调用 get 方法不提供参数引发 TypeError
        self.assertRaises(TypeError, d.get)
        # 断言调用 get 方法提供多余参数引发 TypeError
        self.assertRaises(TypeError, d.get, None, None, None)

    # 测试 setdefault 方法

    def test_setdefault(self):
        # 使用空映射创建字典 d
        d = self._empty_mapping()
        # 断言调用 setdefault 方法不提供参数引发 TypeError
        self.assertRaises(TypeError, d.setdefault)

    # 测试 popitem 方法

    def test_popitem(self):
        # 使用空映射创建字典 d
        d = self._empty_mapping()
        # 断言调用 popitem 方法在空字典中引发 KeyError
        self.assertRaises(KeyError, d.popitem)
        # 断言调用 popitem 方法提供多余参数引发 TypeError
        self.assertRaises(TypeError, d.popitem, 42)

    # 测试 pop 方法

    def test_pop(self):
        # 使用空映射创建字典 d
        d = self._empty_mapping()
        # 获取输入映射的第一个键和值
        k, v = next(iter(self.inmapping.items()))
        # 向字典 d 添加键值对
        d[k] = v
        # 断言调用 pop 方法删除不存在的键引发 KeyError
        self.assertRaises(KeyError, d.pop, next(iter(self.other.keys())))

        # 断言调用 pop 方法删除存在的键并返回其对应的值
        self.assertEqual(d.pop(k), v)
        # 断言字典 d 现在长度为 0
        self.assertEqual(len(d), 0)

        # 断言再次调用 pop 方法删除已经删除的键引发 KeyError
        self.assertRaises(KeyError, d.pop, k)
# Adapted from cpython/Lib/test/mapping_tests.py
# 定义一个测试类，用于测试 WeakKeyDictionaryScriptObject 的行为
class WeakKeyDictionaryScriptObjectTestCase(TestCase):

    # 返回一个包含 _TorchScriptTesting._Foo 实例作为键的字典，用于测试参考对象
    def _reference(self):
        self.__ref = {
            torch.classes._TorchScriptTesting._Foo(1, 2): 1,
            torch.classes._TorchScriptTesting._Foo(2, 3): 2,
            torch.classes._TorchScriptTesting._Foo(3, 4): 3,
        }
        return self.__ref.copy()

    # 返回一个空的映射对象 WeakIdKeyDictionary
    def _empty_mapping(self):
        """Return an empty mapping object"""
        return WeakIdKeyDictionary(ref_type=_WeakHashRef)

    # 返回一个包含给定数据的映射对象
    def _full_mapping(self, data):
        """Return a mapping object with the value contained in data
        dictionary"""
        x = self._empty_mapping()
        for key, value in data.items():
            x[key] = value
        return x

    # 设置测试环境
    def setUp(self):
        # 如果运行环境是 macOS，则跳过测试，因为使用了不可移植的 load_library 调用
        if IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")

    # 初始化方法，继承自 unittest.TestCase，根据运行环境加载相应的库文件
    def __init__(self, *args, **kw):
        unittest.TestCase.__init__(self, *args, **kw)
        # 如果运行在 Sandcastle 或者 FBCODE 环境，加载自定义类注册相关的库
        if IS_SANDCASTLE or IS_FBCODE:
            torch.ops.load_library(
                "//caffe2/test/cpp/jit:test_custom_class_registrations"
            )
        elif IS_MACOS:
            # 在 macOS 环境下，仅跳过加载库文件的步骤
            return
        else:
            # 否则，在其他环境下，找到并加载 libtorchbind_test.so 或 torchbind_test.dll
            lib_file_path = find_library_location("libtorchbind_test.so")
            if IS_WINDOWS:
                lib_file_path = find_library_location("torchbind_test.dll")
            torch.ops.load_library(str(lib_file_path))

        # 创建测试时的参考对象副本
        self.reference = self._reference().copy()

        # 从参考对象中弹出一个键值对，用于测试不在映射中的情况
        key, value = self.reference.popitem()
        self.other = {key: value}

        # 从参考对象中再弹出一个键值对，用于测试存在映射中的情况
        key, value = self.reference.popitem()
        self.inmapping = {key: value}
        # 恢复弹出的键值对到参考对象中，保持测试环境一致
        self.reference[key] = value
    def test_read(self):
        # 测试映射上的只读操作
        p = self._empty_mapping()
        p1 = dict(p)  # 单例对象的解决方法
        d = self._full_mapping(self.reference)
        if d is p:
            p = p1
        # 索引操作
        for key, value in self.reference.items():
            self.assertEqual(d[key], value)
        knownkey = next(iter(self.other.keys()))
        self.assertRaises(KeyError, lambda: d[knownkey])
        # len 方法测试
        self.assertEqual(len(p), 0)
        self.assertEqual(len(d), len(self.reference))
        # __contains__ 方法测试
        for k in self.reference:
            self.assertIn(k, d)
        for k in self.other:
            self.assertNotIn(k, d)
        # 比较操作测试
        self.assertTrue(
            p == p
        )  # 注意：不要使用 assertEqual，因为它实际上不使用 ==
        self.assertTrue(d == d)
        self.assertTrue(p != d)
        self.assertTrue(d != p)
        # 布尔值测试
        if p:
            self.fail("空映射应该与 False 比较")
        if not d:
            self.fail("完整映射应该与 True 比较")

        # keys(), items(), iterkeys() ... 方法测试
        def check_iterandlist(iter, lst, ref):
            self.assertTrue(hasattr(iter, "__next__"))
            self.assertTrue(hasattr(iter, "__iter__"))
            x = list(iter)
            self.assertTrue(set(x) == set(lst) == set(ref))

        check_iterandlist(iter(d.keys()), list(d.keys()), self.reference.keys())
        check_iterandlist(iter(d), list(d.keys()), self.reference.keys())
        check_iterandlist(iter(d.values()), list(d.values()), self.reference.values())
        check_iterandlist(iter(d.items()), list(d.items()), self.reference.items())
        # get 方法测试
        key, value = next(iter(d.items()))
        knownkey, knownvalue = next(iter(self.other.items()))
        self.assertEqual(d.get(key, knownvalue), value)
        self.assertEqual(d.get(knownkey, knownvalue), knownvalue)
        self.assertNotIn(knownkey, d)
    def test_write(self):
        # Test for write operations on mapping

        # 创建一个空的映射对象
        p = self._empty_mapping()

        # 索引操作：将参考数据中的每个键值对写入映射对象，并进行断言验证
        for key, value in self.reference.items():
            p[key] = value
            self.assertEqual(p[key], value)

        # 使用循环删除映射对象中的每个键，并断言引发 KeyError 异常
        for key in self.reference.keys():
            del p[key]
            self.assertRaises(KeyError, lambda: p[key])

        # 重新创建一个空的映射对象
        p = self._empty_mapping()

        # 使用 update 方法更新映射对象，验证更新后的结果是否与参考数据一致
        p.update(self.reference)
        self.assertEqual(dict(p), self.reference)

        # 将映射对象的 items 转换为列表后再次更新映射对象，并验证更新后的结果是否与参考数据一致
        items = list(p.items())
        p = self._empty_mapping()
        p.update(items)
        self.assertEqual(dict(p), self.reference)

        # 使用完整的映射对象来测试 setdefault 方法
        d = self._full_mapping(self.reference)

        # 对第一个键值对进行测试，验证 setdefault 返回正确的值并确保键值对未被改变
        key, value = next(iter(d.items()))
        knownkey, knownvalue = next(iter(self.other.items()))
        self.assertEqual(d.setdefault(key, knownvalue), value)
        self.assertEqual(d[key], value)

        # 对第二个键值对进行测试，验证 setdefault 返回正确的值并确保键值对未被改变
        self.assertEqual(d.setdefault(knownkey, knownvalue), knownvalue)
        self.assertEqual(d[knownkey], knownvalue)

        # 测试 pop 方法：移除一个键值对并验证它是否不再存在，同时验证对不存在的键使用 pop 会引发 KeyError 异常
        self.assertEqual(d.pop(knownkey), knownvalue)
        self.assertNotIn(knownkey, d)
        self.assertRaises(KeyError, d.pop, knownkey)

        # 测试 pop 方法的默认值功能：当键不存在时返回默认值，并验证键值对是否被移除
        default = 909
        d[knownkey] = knownvalue
        self.assertEqual(d.pop(knownkey, default), knownvalue)
        self.assertNotIn(knownkey, d)
        self.assertEqual(d.pop(knownkey, default), default)

        # 测试 popitem 方法：移除并返回一个键值对，并验证该键不再存在，同时验证空映射对象使用 popitem 会引发 KeyError 异常
        key, value = d.popitem()
        self.assertNotIn(key, d)
        self.assertEqual(value, self.reference[key])

        p = self._empty_mapping()

        # 验证空映射对象使用 popitem 会引发 KeyError 异常
        self.assertRaises(KeyError, p.popitem)

    def test_constructor(self):
        # 测试构造函数，验证返回的空映射对象是否相等
        self.assertEqual(self._empty_mapping(), self._empty_mapping())

    def test_bool(self):
        # 测试布尔转换方法，验证空映射对象和非空映射对象的布尔值
        self.assertTrue(not self._empty_mapping())
        self.assertTrue(self.reference)
        self.assertTrue(bool(self._empty_mapping()) is False)
        self.assertTrue(bool(self.reference) is True)

    def test_keys(self):
        # 测试 keys 方法

        # 创建一个空映射对象，并验证其 keys 方法返回空列表
        d = self._empty_mapping()
        self.assertEqual(list(d.keys()), [])

        # 使用参考映射对象，并验证其 keys 方法返回包含指定键的列表，并验证不包含其他映射对象的键
        d = self.reference
        self.assertIn(next(iter(self.inmapping.keys())), d.keys())
        self.assertNotIn(next(iter(self.other.keys())), d.keys())

        # 验证调用 keys 方法时传递 None 会引发 TypeError 异常
        self.assertRaises(TypeError, d.keys, None)

    def test_values(self):
        # 测试 values 方法

        # 创建一个空映射对象，并验证其 values 方法返回空列表
        d = self._empty_mapping()
        self.assertEqual(list(d.values()), [])

        # 验证调用 values 方法时传递 None 会引发 TypeError 异常
        self.assertRaises(TypeError, d.values, None)

    def test_items(self):
        # 测试 items 方法

        # 创建一个空映射对象，并验证其 items 方法返回空列表
        d = self._empty_mapping()
        self.assertEqual(list(d.items()), [])

        # 验证调用 items 方法时传递 None 会引发 TypeError 异常
        self.assertRaises(TypeError, d.items, None)

    def test_len(self):
        # 测试 len 方法

        # 创建一个空映射对象，并验证其长度为 0
        d = self._empty_mapping()
        self.assertEqual(len(d), 0)

    def test_getitem(self):
        # 测试 __getitem__ 方法

        # 使用参考映射对象，验证通过键获取值的正确性
        d = self.reference
        self.assertEqual(
            d[next(iter(self.inmapping.keys()))], next(iter(self.inmapping.values()))
        )

        # 验证调用 __getitem__ 方法时未传递参数会引发 TypeError 异常
        self.assertRaises(TypeError, d.__getitem__)
    # 测试 get 方法

    def test_get(self):
        # 创建一个空映射对象 d
        d = self._empty_mapping()
        # 断言：获取不存在的键应返回 None
        self.assertTrue(d.get(next(iter(self.other.keys()))) is None)
        # 断言：获取不存在的键，返回默认值 3
        self.assertEqual(d.get(next(iter(self.other.keys())), 3), 3)
        
        # 将 d 设置为参考映射对象
        d = self.reference
        # 断言：获取不存在的键应返回 None
        self.assertTrue(d.get(next(iter(self.other.keys()))) is None)
        # 断言：获取不存在的键，返回默认值 3
        self.assertEqual(d.get(next(iter(self.other.keys())), 3), 3)
        
        # 断言：获取已知键的值与参考映射中相应值相等
        self.assertEqual(
            d.get(next(iter(self.inmapping.keys()))),
            next(iter(self.inmapping.values())),
        )
        # 断言：获取已知键的值与参考映射中相应值相等，返回默认值 3
        self.assertEqual(
            d.get(next(iter(self.inmapping.keys())), 3),
            next(iter(self.inmapping.values())),
        )
        
        # 断言：调用 get 方法时不提供参数，应引发 TypeError
        self.assertRaises(TypeError, d.get)
        # 断言：调用 get 方法时提供多余参数，应引发 TypeError
        self.assertRaises(TypeError, d.get, None, None, None)

    # 测试 setdefault 方法

    def test_setdefault(self):
        # 创建一个空映射对象 d
        d = self._empty_mapping()
        # 断言：调用 setdefault 方法时不提供参数，应引发 TypeError
        self.assertRaises(TypeError, d.setdefault)

    # 测试 popitem 方法

    def test_popitem(self):
        # 创建一个空映射对象 d
        d = self._empty_mapping()
        # 断言：调用 popitem 方法时应引发 KeyError（空映射无法弹出元素）
        self.assertRaises(KeyError, d.popitem)
        # 断言：调用 popitem 方法时提供多余参数，应引发 TypeError
        self.assertRaises(TypeError, d.popitem, 42)

    # 测试 pop 方法

    def test_pop(self):
        # 创建一个空映射对象 d
        d = self._empty_mapping()
        # 获取映射 inmapping 中的第一个键值对
        k, v = next(iter(self.inmapping.items()))
        # 向映射 d 中添加键值对
        d[k] = v
        # 断言：调用 pop 方法时提供不存在的键，应引发 KeyError
        self.assertRaises(KeyError, d.pop, next(iter(self.other.keys())))

        # 断言：弹出键 k 对应的值应与 v 相等
        self.assertEqual(d.pop(k), v)
        # 断言：映射 d 的长度应为 0
        self.assertEqual(len(d), 0)

        # 断言：再次调用 pop 方法时提供已弹出的键 k，应引发 KeyError
        self.assertRaises(KeyError, d.pop, k)
# 如果当前脚本被直接执行（而不是被导入到其他脚本中执行），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数来执行测试
    run_tests()
```