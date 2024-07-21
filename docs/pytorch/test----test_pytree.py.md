# `.\pytorch\test\test_pytree.py`

```
# Owner(s): ["module: pytree"]

# 引入必要的模块和库
import collections  # 导入collections模块
import inspect  # 导入inspect模块
import re  # 导入re模块
import unittest  # 导入unittest模块
from collections import defaultdict, deque, namedtuple, OrderedDict, UserDict  # 从collections模块导入特定类
from dataclasses import dataclass  # 导入dataclass类装饰器
from typing import Any, NamedTuple  # 导入类型提示相关类

import torch  # 导入torch库
import torch.utils._pytree as py_pytree  # 导入torch.utils._pytree模块
from torch.fx.immutable_collections import immutable_dict, immutable_list  # 从torch.fx.immutable_collections导入不可变字典和不可变列表
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入instantiate_parametrized_tests函数
    IS_FBCODE,  # 导入IS_FBCODE变量
    parametrize,  # 导入parametrize装饰器
    run_tests,  # 导入run_tests函数
    skipIfTorchDynamo,  # 导入skipIfTorchDynamo装饰器
    subtest,  # 导入subtest函数
    TEST_WITH_TORCHDYNAMO,  # 导入TEST_WITH_TORCHDYNAMO变量
    TestCase,  # 导入TestCase类
)

if IS_FBCODE:
    # 在fbcode环境中，尚未启用optree，因此使用python实现重新测试
    cxx_pytree = py_pytree
else:
    import torch.utils._cxx_pytree as cxx_pytree  # 导入torch.utils._cxx_pytree模块

# 定义一个具名元组GlobalPoint，包含属性x和y
GlobalPoint = namedtuple("GlobalPoint", ["x", "y"])

# 定义一个类GlobalDummyType，带有属性x和y的构造函数
class GlobalDummyType:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 定义一个测试类TestGenericPytree，继承自unittest.TestCase
class TestGenericPytree(TestCase):

    # 参数化测试方法，用于测试不同的pytree_impl实现
    @parametrize(
        "pytree_impl",
        [
            subtest(py_pytree, name="py"),  # 使用py_pytree作为pytree_impl的子测试
            subtest(cxx_pytree, name="cxx"),  # 使用cxx_pytree作为pytree_impl的子测试
        ],
    )
    def test_register_pytree_node(self, pytree_impl):
        # 定义一个名为MyDict的子类，继承自UserDict
        class MyDict(UserDict):
            pass

        d = MyDict(a=1, b=2, c=3)  # 创建一个MyDict对象d，包含键值对a:1, b:2, c:3

        # 默认情况下，自定义类型是叶子节点
        values, spec = pytree_impl.tree_flatten(d)  # 对d进行树展开操作，返回值和规范
        self.assertEqual(values, [d])  # 断言展开后的值为[d]
        self.assertIs(values[0], d)  # 断言展开后的值第一个元素为d本身
        self.assertEqual(d, pytree_impl.tree_unflatten(values, spec))  # 对展开后的值和规范进行反展开，应当与d相等
        self.assertTrue(spec.is_leaf())  # 断言规范是叶子节点

        # 注册MyDict作为一个pytree节点
        pytree_impl.register_pytree_node(
            MyDict,
            lambda d: (list(d.values()), list(d.keys())),  # 定义如何展平MyDict对象
            lambda values, keys: MyDict(zip(keys, values)),  # 定义如何反展平值和键到MyDict对象
        )

        values, spec = pytree_impl.tree_flatten(d)  # 再次对d进行树展开操作
        self.assertEqual(values, [1, 2, 3])  # 断言展开后的值为[1, 2, 3]
        self.assertEqual(d, pytree_impl.tree_unflatten(values, spec))  # 断言反展开后与d相等

        # 不允许重复注册同一类型
        with self.assertRaisesRegex(ValueError, "already registered"):
            pytree_impl.register_pytree_node(
                MyDict,
                lambda d: (list(d.values()), list(d.keys())),
                lambda values, keys: MyDict(zip(keys, values)),
            )

    # 参数化测试方法，用于测试不同的pytree_impl实现
    @parametrize(
        "pytree_impl",
        [
            subtest(py_pytree, name="py"),  # 使用py_pytree作为pytree_impl的子测试
            subtest(cxx_pytree, name="cxx"),  # 使用cxx_pytree作为pytree_impl的子测试
        ],
    )
    def test_flatten_unflatten_leaf(self, pytree_impl):
        # 定义一个函数run_test_with_leaf，测试叶子节点leaf的展开和反展开
        def run_test_with_leaf(leaf):
            values, treespec = pytree_impl.tree_flatten(leaf)  # 对leaf进行树展开操作，返回值和规范
            self.assertEqual(values, [leaf])  # 断言展开后的值为[leaf]
            self.assertEqual(treespec, pytree_impl.LeafSpec())  # 断言规范是LeafSpec()

            unflattened = pytree_impl.tree_unflatten(values, treespec)  # 对展开后的值和规范进行反展开
            self.assertEqual(unflattened, leaf)  # 断言反展开后与leaf相等

        # 分别测试不同类型的叶子节点
        run_test_with_leaf(1)
        run_test_with_leaf(1.0)
        run_test_with_leaf(None)
        run_test_with_leaf(bool)
        run_test_with_leaf(torch.randn(3, 3))
    @parametrize(
        "pytree_impl,gen_expected_fn",
        [
            subtest(
                (
                    py_pytree,
                    lambda tup: py_pytree.TreeSpec(
                        tuple, None, [py_pytree.LeafSpec() for _ in tup]
                    ),
                ),
                name="py",
            ),
            subtest(
                (cxx_pytree, lambda tup: cxx_pytree.tree_structure((0,) * len(tup))),
                name="cxx",
            ),
        ],
    )
    # 使用 @parametrize 装饰器，为单元测试方法提供多组参数化输入
    def test_flatten_unflatten_tuple(self, pytree_impl, gen_expected_fn):
        # 定义内部函数 run_test，用于执行具体的测试
        def run_test(tup):
            # 生成预期的树规范对象
            expected_spec = gen_expected_fn(tup)
            # 对输入的元组进行展开操作，返回值和树规范
            values, treespec = pytree_impl.tree_flatten(tup)
            # 断言展开后的值是列表类型，并且与原始元组相等
            self.assertIsInstance(values, list)
            self.assertEqual(values, list(tup))
            # 断言生成的树规范与预期的树规范相等
            self.assertEqual(treespec, expected_spec)

            # 对展平后的值进行反展开操作
            unflattened = pytree_impl.tree_unflatten(values, treespec)
            # 断言反展开后得到的对象与原始元组相等
            self.assertEqual(unflattened, tup)
            # 断言反展开后得到的对象是元组类型
            self.assertIsInstance(unflattened, tuple)

        # 分别运行空元组和其他测试元组的 run_test 函数
        run_test(())
        run_test((1.0,))
        run_test((1.0, 2))
        run_test((torch.tensor([1.0, 2]), 2, 10, 9, 11))

    @parametrize(
        "pytree_impl,gen_expected_fn",
        [
            subtest(
                (
                    py_pytree,
                    lambda lst: py_pytree.TreeSpec(
                        list, None, [py_pytree.LeafSpec() for _ in lst]
                    ),
                ),
                name="py",
            ),
            subtest(
                (cxx_pytree, lambda lst: cxx_pytree.tree_structure([0] * len(lst))),
                name="cxx",
            ),
        ],
    )
    # 使用 @parametrize 装饰器，为单元测试方法提供多组参数化输入
    def test_flatten_unflatten_list(self, pytree_impl, gen_expected_fn):
        # 定义内部函数 run_test，用于执行具体的测试
        def run_test(lst):
            # 生成预期的树规范对象
            expected_spec = gen_expected_fn(lst)
            # 对输入的列表进行展开操作，返回值和树规范
            values, treespec = pytree_impl.tree_flatten(lst)
            # 断言展开后的值是列表类型，并且与原始列表相等
            self.assertIsInstance(values, list)
            self.assertEqual(values, lst)
            # 断言生成的树规范与预期的树规范相等
            self.assertEqual(treespec, expected_spec)

            # 对展平后的值进行反展开操作
            unflattened = pytree_impl.tree_unflatten(values, treespec)
            # 断言反展开后得到的对象与原始列表相等
            self.assertEqual(unflattened, lst)
            # 断言反展开后得到的对象是列表类型
            self.assertIsInstance(unflattened, list)

        # 分别运行空列表和其他测试列表的 run_test 函数
        run_test([])
        run_test([1.0, 2])
        run_test([torch.tensor([1.0, 2]), 2, 10, 9, 11])

    @parametrize(
        "pytree_impl,gen_expected_fn",
        [
            subtest(
                (
                    py_pytree,
                    lambda dct: py_pytree.TreeSpec(
                        dict,
                        list(dct.keys()),
                        [py_pytree.LeafSpec() for _ in dct.values()],
                    ),
                ),
                name="py",
            ),
            subtest(
                (
                    cxx_pytree,
                    lambda dct: cxx_pytree.tree_structure(dict.fromkeys(dct, 0)),
                ),
                name="cxx",
            ),
        ],
    )
    # 使用 @parametrize 装饰器，为单元测试方法提供多组参数化输入
    def test_flatten_unflatten_dict(self, pytree_impl, gen_expected_fn):
        # 定义内部函数 run_test，用于执行具体的测试
        def run_test(dct):
            # 生成预期的树规范对象
            expected_spec = gen_expected_fn(dct)
            # 对输入的字典进行展开操作，返回值和树规范
            values, treespec = pytree_impl.tree_flatten(dct)
            # 断言展开后的值是列表类型，并且与原始字典的值列表相等
            self.assertIsInstance(values, list)
            self.assertEqual(values, list(dct.values()))
            # 断言生成的树规范与预期的树规范相等
            self.assertEqual(treespec, expected_spec)

            # 对展平后的值进行反展开操作
            unflattened = pytree_impl.tree_unflatten(values, treespec)
            # 断言反展开后得到的对象与原始字典相等
            self.assertEqual(unflattened, dct)
            # 断言反展开后得到的对象是字典类型
            self.assertIsInstance(unflattened, dict)

        # 分别运行空字典和其他测试字典的 run_test 函数
        run_test({})
        run_test({'a': 1, 'b': 2})
        run_test({'x': torch.tensor([1.0, 2]), 'y': 2, 'z': 10})
    # 定义测试方法，用于测试字典扁平化和解扁平化操作
    def test_flatten_unflatten_dict(self, pytree_impl, gen_expected_fn):
        
        # 定义内部函数run_test，用于执行测试
        def run_test(dct):
            # 生成预期的规范结构
            expected_spec = gen_expected_fn(dct)
            
            # 对字典进行扁平化处理，获取数值列表和树形结构描述
            values, treespec = pytree_impl.tree_flatten(dct)
            self.assertIsInstance(values, list)  # 确保数值列表是列表类型
            self.assertEqual(values, list(dct.values()))  # 检查数值列表与字典值列表的一致性
            self.assertEqual(treespec, expected_spec)  # 检查树形结构描述与预期规范的一致性

            # 对数值列表进行解扁平化操作，还原为字典
            unflattened = pytree_impl.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, dct)  # 检查解扁平化后的字典与原始字典的一致性
            self.assertIsInstance(unflattened, dict)  # 确保解扁平化后的结果仍为字典类型

        # 分别测试空字典及带有不同键值对的字典
        run_test({})
        run_test({"a": 1})
        run_test({"abcdefg": torch.randn(2, 3)})
        run_test({1: torch.randn(2, 3)})
        run_test({"a": 1, "b": 2, "c": torch.randn(2, 3)})

    # 使用参数化装饰器@parametrize定义测试用例，测试OrderedDict的扁平化和解扁平化操作
    @parametrize(
        "pytree_impl,gen_expected_fn",
        [
            # 子测试1：使用Python实现的树结构，生成预期的OrderedDict规范
            subtest(
                (
                    py_pytree,
                    lambda odict: py_pytree.TreeSpec(
                        OrderedDict,
                        list(odict.keys()),  # 使用字典键作为规范的索引
                        [py_pytree.LeafSpec() for _ in odict.values()],  # 每个值生成一个叶子规范
                    ),
                ),
                name="py",  # 子测试名称
            ),
            # 子测试2：使用C++实现的树结构，生成预期的OrderedDict规范
            subtest(
                (
                    cxx_pytree,
                    lambda odict: cxx_pytree.tree_structure(
                        OrderedDict.fromkeys(odict, 0)  # 使用字典的键生成结构描述
                    ),
                ),
                name="cxx",  # 子测试名称
            ),
        ],
    )
    # 定义测试方法，测试OrderedDict的扁平化和解扁平化操作
    def test_flatten_unflatten_ordereddict(self, pytree_impl, gen_expected_fn):
        
        # 定义内部函数run_test，用于执行测试
        def run_test(odict):
            # 生成预期的规范结构
            expected_spec = gen_expected_fn(odict)
            
            # 对OrderedDict进行扁平化处理，获取数值列表和树形结构描述
            values, treespec = pytree_impl.tree_flatten(odict)
            self.assertIsInstance(values, list)  # 确保数值列表是列表类型
            self.assertEqual(values, list(odict.values()))  # 检查数值列表与OrderedDict值列表的一致性
            self.assertEqual(treespec, expected_spec)  # 检查树形结构描述与预期规范的一致性

            # 对数值列表进行解扁平化操作，还原为OrderedDict
            unflattened = pytree_impl.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, odict)  # 检查解扁平化后的OrderedDict与原始OrderedDict的一致性
            self.assertIsInstance(unflattened, OrderedDict)  # 确保解扁平化后的结果仍为OrderedDict类型

        # 初始化一个空的OrderedDict并执行测试
        od = OrderedDict()
        run_test(od)

        # 向OrderedDict添加键值对并执行测试
        od["b"] = 1
        od["a"] = torch.tensor(3.14)
        run_test(od)

    # 使用参数化装饰器@parametrize定义测试用例，测试defaultdict的扁平化和解扁平化操作
    @parametrize(
        "pytree_impl,gen_expected_fn",
        [
            # 子测试1：使用Python实现的树结构，生成预期的defaultdict规范
            subtest(
                (
                    py_pytree,
                    lambda ddct: py_pytree.TreeSpec(
                        defaultdict,
                        [ddct.default_factory, list(ddct.keys())],  # 使用defaultdict的默认工厂和键作为规范的索引
                        [py_pytree.LeafSpec() for _ in ddct.values()],  # 每个值生成一个叶子规范
                    ),
                ),
                name="py",  # 子测试名称
            ),
            # 子测试2：使用C++实现的树结构，生成预期的defaultdict规范
            subtest(
                (
                    cxx_pytree,
                    lambda ddct: cxx_pytree.tree_structure(
                        defaultdict(ddct.default_factory, dict.fromkeys(ddct, 0))  # 使用defaultdict的默认工厂和键生成结构描述
                    ),
                ),
                name="cxx",  # 子测试名称
            ),
        ],
    )
    # 定义一个测试方法，用于测试 flatten 和 unflatten 操作对于 defaultdict 的默认行为
    def test_flatten_unflatten_defaultdict(self, pytree_impl, gen_expected_fn):
        # 定义一个内部函数 run_test，用于运行测试
        def run_test(ddct):
            # 根据 ddct 生成预期的规范
            expected_spec = gen_expected_fn(ddct)
            # 执行树展平操作，返回值和规范
            values, treespec = pytree_impl.tree_flatten(ddct)
            # 确保 values 是列表类型
            self.assertIsInstance(values, list)
            # 确保 values 中的值与 ddct 的值列表相同
            self.assertEqual(values, list(ddct.values()))
            # 确保树的规范与预期的规范相同
            self.assertEqual(treespec, expected_spec)

            # 执行树展开操作，得到解展开的对象
            unflattened = pytree_impl.tree_unflatten(values, treespec)
            # 确保解展开的对象与原始 defaultdict 相同
            self.assertEqual(unflattened, ddct)
            # 确保解展开的对象的 default_factory 与原始 defaultdict 相同
            self.assertEqual(unflattened.default_factory, ddct.default_factory)
            # 确保解展开的对象是 defaultdict 类型
            self.assertIsInstance(unflattened, defaultdict)

        # 分别对不同类型的 defaultdict 运行测试
        run_test(defaultdict(list, {}))
        run_test(defaultdict(int, {"a": 1}))
        run_test(defaultdict(int, {"abcdefg": torch.randn(2, 3)}))
        run_test(defaultdict(int, {1: torch.randn(2, 3)}))
        run_test(defaultdict(int, {"a": 1, "b": 2, "c": torch.randn(2, 3)}))

    # 使用参数化装饰器定义一个测试方法，测试 deque 对象的 flatten 和 unflatten 操作
    @parametrize(
        "pytree_impl,gen_expected_fn",
        [
            subtest(
                (
                    py_pytree,
                    lambda deq: py_pytree.TreeSpec(
                        deque, deq.maxlen, [py_pytree.LeafSpec() for _ in deq]
                    ),
                ),
                name="py",
            ),
            subtest(
                (
                    cxx_pytree,
                    lambda deq: cxx_pytree.tree_structure(
                        deque(deq, maxlen=deq.maxlen)
                    ),
                ),
                name="cxx",
            ),
        ],
    )
    # 定义测试 deque 对象的 flatten 和 unflatten 操作的方法
    def test_flatten_unflatten_deque(self, pytree_impl, gen_expected_fn):
        # 定义一个内部函数 run_test，用于运行测试
        def run_test(deq):
            # 根据 deque 对象生成预期的规范
            expected_spec = gen_expected_fn(deq)
            # 执行树展平操作，返回值和规范
            values, treespec = pytree_impl.tree_flatten(deq)
            # 确保 values 是列表类型
            self.assertIsInstance(values, list)
            # 确保 values 中的值与 deque 对象的值列表相同
            self.assertEqual(values, list(deq))
            # 确保树的规范与预期的规范相同
            self.assertEqual(treespec, expected_spec)

            # 执行树展开操作，得到解展开的对象
            unflattened = pytree_impl.tree_unflatten(values, treespec)
            # 确保解展开的对象与原始 deque 对象相同
            self.assertEqual(unflattened, deq)
            # 确保解展开的对象的 maxlen 与原始 deque 对象相同
            self.assertEqual(unflattened.maxlen, deq.maxlen)
            # 确保解展开的对象是 deque 类型
            self.assertIsInstance(unflattened, deque)

        # 分别对不同的空和非空 deque 对象运行测试
        run_test(deque([]))
        run_test(deque([1.0, 2]))
        run_test(deque([torch.tensor([1.0, 2]), 2, 10, 9, 11], maxlen=8))

    # 使用参数化装饰器定义一个测试方法，测试不同的 pytree_impl 实现的 flatten 和 unflatten 操作
    @parametrize(
        "pytree_impl",
        [
            subtest(py_pytree, name="py"),
            subtest(cxx_pytree, name="cxx"),
        ],
    )
    # 定义一个测试方法，用于测试命名元组的展平和还原操作
    def test_flatten_unflatten_namedtuple(self, pytree_impl):
        # 定义一个名为 Point 的命名元组，包含字段 "x" 和 "y"
        Point = namedtuple("Point", ["x", "y"])

        # 定义一个内部函数，用于运行测试
        def run_test(tup):
            # 根据不同的 pytree_impl 设置预期的树结构
            if pytree_impl is py_pytree:
                # 使用 py_pytree 创建预期的树结构
                expected_spec = py_pytree.TreeSpec(
                    namedtuple, Point, [py_pytree.LeafSpec() for _ in tup]
                )
            else:
                # 使用 cxx_pytree 创建预期的树结构
                expected_spec = cxx_pytree.tree_structure(Point(0, 1))

            # 调用 pytree_impl 的 tree_flatten 方法展平命名元组
            values, treespec = pytree_impl.tree_flatten(tup)
            # 断言展平后的 values 类型为列表
            self.assertIsInstance(values, list)
            # 断言展平后的 values 与原始命名元组 tup 的内容相同
            self.assertEqual(values, list(tup))
            # 断言展平后的 treespec 与预期的结构相同
            self.assertEqual(treespec, expected_spec)

            # 使用 pytree_impl 的 tree_unflatten 方法将展平的值还原
            unflattened = pytree_impl.tree_unflatten(values, treespec)
            # 断言还原后的结果与原始命名元组 tup 相同
            self.assertEqual(unflattened, tup)
            # 断言还原后的结果是 Point 类型的实例
            self.assertIsInstance(unflattened, Point)

        # 分别测试 Point(1.0, 2) 和 Point(torch.tensor(1.0), 2)
        run_test(Point(1.0, 2))
        run_test(Point(torch.tensor(1.0), 2))

    # 使用 parametrize 装饰器，对 torch.max 和 torch.min 进行子测试
    @parametrize(
        "op",
        [
            subtest(torch.max, name="max"),
            subtest(torch.min, name="min"),
        ],
    )
    # 使用 parametrize 装饰器，对 py_pytree 和 cxx_pytree 进行子测试
    @parametrize(
        "pytree_impl",
        [
            subtest(py_pytree, name="py"),
            subtest(cxx_pytree, name="cxx"),
        ],
    )
    # 测试展平和还原操作的返回类型
    def test_flatten_unflatten_return_types(self, pytree_impl, op):
        # 创建一个 3x3 的随机张量 x
        x = torch.randn(3, 3)
        # 使用 op 对 x 按照 dim=0 进行操作，得到预期结果 expected
        expected = op(x, dim=0)

        # 调用 pytree_impl 的 tree_flatten 方法展平 expected
        values, spec = pytree_impl.tree_flatten(expected)
        # 遍历 values 中的每个元素，断言它们都是 torch.Tensor 类型
        for value in values:
            self.assertIsInstance(value, torch.Tensor)
        # 使用 pytree_impl 的 tree_unflatten 方法将展平的值还原
        result = pytree_impl.tree_unflatten(values, spec)

        # 断言还原后的结果类型与 expected 相同
        self.assertEqual(type(result), type(expected))
        # 断言还原后的结果与 expected 相同
        self.assertEqual(result, expected)

    # 使用 parametrize 装饰器，对 py_pytree 和 cxx_pytree 进行子测试
    @parametrize(
        "pytree_impl",
        [
            subtest(py_pytree, name="py"),
            subtest(cxx_pytree, name="cxx"),
        ],
    )
    # 测试嵌套结构的展平和还原操作
    def test_flatten_unflatten_nested(self, pytree_impl):
        # 定义一个内部函数，用于运行测试
        def run_test(pytree):
            # 调用 pytree_impl 的 tree_flatten 方法展平 pytree
            values, treespec = pytree_impl.tree_flatten(pytree)
            # 断言展平后的 values 类型为列表
            self.assertIsInstance(values, list)
            # 断言展平后的 values 的长度与 treespec 的叶子节点数相同
            self.assertEqual(len(values), treespec.num_leaves)

            # 使用 pytree_impl 的 tree_unflatten 方法将展平的值还原
            unflattened = pytree_impl.tree_unflatten(values, treespec)
            # 断言还原后的结果与原始 pytree 相同
            self.assertEqual(unflattened, pytree)

        # 定义多个测试用例
        cases = [
            [()],
            ([],),
            {"a": ()},
            {"a": 0, "b": [{"c": 1}]},
            {"a": 0, "b": [1, {"c": 2}, torch.randn(3)], "c": (torch.randn(2, 3), 1)},
        ]
        # 对每个测试用例调用 run_test 运行测试
        for case in cases:
            run_test(case)
    # 定义一个测试方法 `test_flatten_with_is_leaf`，接受 `pytree_impl` 参数
    def test_flatten_with_is_leaf(self, pytree_impl):
        # 定义内部方法 `run_test`，接受 `pytree` 和 `one_level_leaves` 参数
        def run_test(pytree, one_level_leaves):
            # 调用 `pytree_impl.tree_flatten` 方法，使用自定义的 `is_leaf` 函数
            values, treespec = pytree_impl.tree_flatten(
                pytree, is_leaf=lambda x: x is not pytree
            )
            # 断言 `values` 是一个列表
            self.assertIsInstance(values, list)
            # 断言 `values` 的长度等于 `treespec.num_nodes - 1`
            self.assertEqual(len(values), treespec.num_nodes - 1)
            # 断言 `values` 的长度等于 `treespec.num_leaves`
            self.assertEqual(len(values), treespec.num_leaves)
            # 断言 `values` 的长度等于 `treespec.num_children`
            self.assertEqual(len(values), treespec.num_children)
            # 断言 `values` 等于 `one_level_leaves`
            self.assertEqual(values, one_level_leaves)

            # 断言 `treespec` 等于根据 `values` 和 `treespec` 重建的结构
            self.assertEqual(
                treespec,
                pytree_impl.tree_structure(
                    pytree_impl.tree_unflatten([0] * treespec.num_leaves, treespec)
                ),
            )

            # 将 `values` 使用 `pytree_impl.tree_unflatten` 方法恢复成 `pytree`，并断言相等
            unflattened = pytree_impl.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, pytree)

        # 定义测试用例列表 `cases`
        cases = [
            ([()], [()]),
            (([],), [[]]),
            ({"a": ()}, [()]),
            ({"a": 0, "b": [{"c": 1}]}, [0, [{"c": 1}]]),
            (
                {
                    "a": 0,
                    "b": [1, {"c": 2}, torch.ones(3)],
                    "c": (torch.zeros(2, 3), 1),
                },
                [0, [1, {"c": 2}, torch.ones(3)], (torch.zeros(2, 3), 1)],
            ),
        ]
        # 遍历测试用例，并执行 `run_test` 方法
        for case in cases:
            run_test(*case)

    # 使用 `parametrize` 装饰器定义参数化测试 `test_tree_map`，接受 `pytree_impl` 参数
    @parametrize(
        "pytree_impl",
        [
            subtest(py_pytree, name="py"),  # 使用 `subtest` 创建测试实例 `py_pytree`
            subtest(cxx_pytree, name="cxx"),  # 使用 `subtest` 创建测试实例 `cxx_pytree`
        ],
    )
    # 定义 `test_tree_map` 方法，接受 `pytree_impl` 参数
    def test_tree_map(self, pytree_impl):
        # 定义内部方法 `run_test`，接受 `pytree` 参数
        def run_test(pytree):
            # 定义函数 `f`，将每个元素乘以 3
            def f(x):
                return x * 3

            # 计算 `pytree` 叶子节点值的和 `sm1`
            sm1 = sum(map(f, pytree_impl.tree_leaves(pytree)))
            # 使用 `pytree_impl.tree_map` 对 `pytree` 应用函数 `f` 后，计算叶子节点值的和 `sm2`
            sm2 = sum(pytree_impl.tree_leaves(pytree_impl.tree_map(f, pytree)))
            # 断言 `sm1` 等于 `sm2`
            self.assertEqual(sm1, sm2)

            # 定义函数 `invf`，将每个元素除以 3
            def invf(x):
                return x // 3

            # 断言先应用 `f` 函数再应用 `invf` 函数后得到的结果等于原始 `pytree`
            self.assertEqual(
                pytree_impl.tree_map(invf, pytree_impl.tree_map(f, pytree)),
                pytree,
            )

        # 定义测试用例列表 `cases`
        cases = [
            [()],
            ([],),
            {"a": ()},
            {"a": 1, "b": [{"c": 2}]},
            {"a": 0, "b": [2, {"c": 3}, 4], "c": (5, 6)},
        ]
        # 遍历测试用例，并执行 `run_test` 方法
        for case in cases:
            run_test(case)

    # 使用 `parametrize` 装饰器定义参数化测试 `test_tree_map`，接受 `pytree_impl` 参数
    @parametrize(
        "pytree_impl",
        [
            subtest(py_pytree, name="py"),  # 使用 `subtest` 创建测试实例 `py_pytree`
            subtest(cxx_pytree, name="cxx"),  # 使用 `subtest` 创建测试实例 `cxx_pytree`
        ],
    )
    # 定义测试方法，用于测试具有多个输入的树映射函数
    def test_tree_map_multi_inputs(self, pytree_impl):
        # 定义内部方法，运行树映射测试
        def run_test(pytree):
            # 定义函数 f，返回元组 (x, [y, (z, 0)])
            def f(x, y, z):
                return x, [y, (z, 0)]

            # 复制输入的 pytree 作为 pytree_x
            pytree_x = pytree
            # 对 pytree 应用 lambda 函数 (x + 1,)，生成 pytree_y
            pytree_y = pytree_impl.tree_map(lambda x: (x + 1,), pytree)
            # 对 pytree 应用 lambda 函数 {"a": x * 2, "b": 2}，生成 pytree_z
            pytree_z = pytree_impl.tree_map(lambda x: {"a": x * 2, "b": 2}, pytree)

            # 断言树映射函数 f 应用到 pytree_x, pytree_y, pytree_z 后的结果等于
            # lambda 函数 f 应用到原始 pytree 后的结果
            self.assertEqual(
                pytree_impl.tree_map(f, pytree_x, pytree_y, pytree_z),
                pytree_impl.tree_map(
                    lambda x: f(x, (x + 1,), {"a": x * 2, "b": 2}), pytree
                ),
            )

        # 定义测试用例列表 cases，包括空元组、带逗号的空列表、带有空元组的字典、复杂的混合结构等
        cases = [
            [()],
            ([],),
            {"a": ()},
            {"a": 1, "b": [{"c": 2}]},
            {"a": 0, "b": [2, {"c": 3}, 4], "c": (5, 6)},
        ]
        # 对每个测试用例运行测试方法 run_test
        for case in cases:
            run_test(case)

    # 使用参数化装饰器定义测试方法 test_tree_map_only，测试只应用树映射函数的情况
    @parametrize(
        "pytree_impl",
        [
            subtest(py_pytree, name="py"),
            subtest(cxx_pytree, name="cxx"),
        ],
    )
    def test_tree_map_only(self, pytree_impl):
        # 断言只应用 int 类型映射函数和 lambda 函数 x + 2 到 [0, "a"] 的结果为 [2, "a"]
        self.assertEqual(
            pytree_impl.tree_map_only(int, lambda x: x + 2, [0, "a"]), [2, "a"]
        )

    # 使用参数化装饰器定义测试方法 test_tree_map_only_predicate_fn，测试只应用树映射函数的情况，带有谓词函数
    @parametrize(
        "pytree_impl",
        [
            subtest(py_pytree, name="py"),
            subtest(cxx_pytree, name="cxx"),
        ],
    )
    def test_tree_map_only_predicate_fn(self, pytree_impl):
        # 断言只应用谓词函数 lambda x: x == 0 和映射函数 lambda x: x + 2 到 [0, 1] 的结果为 [2, 1]
        self.assertEqual(
            pytree_impl.tree_map_only(lambda x: x == 0, lambda x: x + 2, [0, 1]), [2, 1]
        )

    # 使用参数化装饰器定义测试方法 test_tree_all_any，测试树映射函数的全部和任意功能
    @parametrize(
        "pytree_impl",
        [
            subtest(py_pytree, name="py"),
            subtest(cxx_pytree, name="cxx"),
        ],
    )
    def test_tree_all_any(self, pytree_impl):
        # 断言全部元素满足 lambda 函数 x % 2 的条件，对 [1, 3] 返回 True
        self.assertTrue(pytree_impl.tree_all(lambda x: x % 2, [1, 3]))
        # 断言不是全部元素满足 lambda 函数 x % 2 的条件，对 [0, 1] 返回 False
        self.assertFalse(pytree_impl.tree_all(lambda x: x % 2, [0, 1]))
        # 断言至少一个元素满足 lambda 函数 x % 2 的条件，对 [0, 1] 返回 True
        self.assertTrue(pytree_impl.tree_any(lambda x: x % 2, [0, 1]))
        # 断言没有元素满足 lambda 函数 x % 2 的条件，对 [0, 2] 返回 False
        self.assertFalse(pytree_impl.tree_any(lambda x: x % 2, [0, 2]))
        # 断言全部只应用 int 类型映射函数的元素满足 lambda 函数 x % 2 的条件，对 [1, 3, "a"] 返回 True
        self.assertTrue(pytree_impl.tree_all_only(int, lambda x: x % 2, [1, 3, "a"]))
        # 断言不是全部只应用 int 类型映射函数的元素满足 lambda 函数 x % 2 的条件，对 [0, 1, "a"] 返回 False
        self.assertFalse(pytree_impl.tree_all_only(int, lambda x: x % 2, [0, 1, "a"]))
        # 断言至少一个只应用 int 类型映射函数的元素满足 lambda 函数 x % 2 的条件，对 [0, 1, "a"] 返回 True
        self.assertTrue(pytree_impl.tree_any_only(int, lambda x: x % 2, [0, 1, "a"]))
        # 断言没有只应用 int 类型映射函数的元素满足 lambda 函数 x % 2 的条件，对 [0, 2, "a"] 返回 False
        self.assertFalse(pytree_impl.tree_any_only(int, lambda x: x % 2, [0, 2, "a"]))
    # 定义测试方法，用于测试广播和展平功能，接受 pytree_impl 参数
    def test_broadcast_to_and_flatten(self, pytree_impl):
        # 测试用例列表，包含不同的输入和期望输出
        cases = [
            (1, (), []),                            # 广播到空元组应返回空列表
            # 相同（扁平化）结构
            ((1,), (0,), [1]),                      # 广播到单元素元组应返回单元素列表
            ([1], [0], [1]),                        # 广播到单元素列表应返回单元素列表
            ((1, 2, 3), (0, 0, 0), [1, 2, 3]),       # 广播到三元组应返回相同顺序的列表
            ({"a": 1, "b": 2}, {"a": 0, "b": 0}, [1, 2]),  # 广播到字典的键应返回相应值的列表
            # 不匹配（扁平化）结构
            ([1], (0,), None),                      # 不匹配结构应返回 None
            ([1], (0,), None),                      # 不匹配结构应返回 None
            ((1,), [0], None),                      # 不匹配结构应返回 None
            ((1, 2, 3), (0, 0), None),               # 不匹配结构应返回 None
            ({"a": 1, "b": 2}, {"a": 0}, None),      # 不匹配结构应返回 None
            ({"a": 1, "b": 2}, {"a": 0, "c": 0}, None),  # 不匹配结构应返回 None
            ({"a": 1, "b": 2}, {"a": 0, "b": 0, "c": 0}, None),  # 不匹配结构应返回 None
            # 相同（嵌套）结构
            ((1, [2, 3]), (0, [0, 0]), [1, 2, 3]),   # 广播到嵌套结构应返回展开后的列表
            ((1, [(2, 3), 4]), (0, [(0, 0), 0]), [1, 2, 3, 4]),  # 广播到更深嵌套结构应返回展开后的列表
            # 不匹配（嵌套）结构
            ((1, [2, 3]), (0, (0, 0)), None),        # 不匹配结构应返回 None
            ((1, [2, 3]), (0, [0, 0, 0]), None),      # 不匹配结构应返回 None
            # 广播单个值
            (1, (0, 0, 0), [1, 1, 1]),                # 广播单个值应返回广播后的列表
            (1, [0, 0, 0], [1, 1, 1]),                # 广播单个值应返回广播后的列表
            (1, {"a": 0, "b": 0}, [1, 1]),            # 广播单个值应返回广播后的列表
            (1, (0, [0, [0]], 0), [1, 1, 1, 1]),       # 广播单个值应返回广播后的列表
            (1, (0, [0, [0, [], [[[0]]]]], 0), [1, 1, 1, 1, 1]),  # 广播单个值应返回广播后的列表
            # 广播多个值
            ((1, 2), ([0, 0, 0], [0, 0]), [1, 1, 1, 2, 2]),              # 广播多个值应返回广播后的列表
            ((1, 2), ([0, [0, 0], 0], [0, 0]), [1, 1, 1, 1, 2, 2]),       # 广播多个值应返回广播后的列表
            (([1, 2, 3], 4), ([0, [0, 0], 0], [0, 0]), [1, 2, 2, 3, 4, 4]),  # 广播多个值应返回广播后的列表
        ]
        # 对于每个测试用例，展开 to_pytree 并比较结果与期望值是否相同
        for pytree, to_pytree, expected in cases:
            _, to_spec = pytree_impl.tree_flatten(to_pytree)
            result = pytree_impl._broadcast_to_and_flatten(pytree, to_spec)
            self.assertEqual(result, expected, msg=str([pytree, to_spec, expected]))

    @parametrize(
        "pytree_impl",
        [
            subtest(py_pytree, name="py"),       # 参数化测试：使用 py_pytree 子测试
            subtest(cxx_pytree, name="cxx"),     # 参数化测试：使用 cxx_pytree 子测试
        ],
    )
    # 测试 pytree_serialize_bad_input 方法，预期会引发 TypeError 异常
    def test_pytree_serialize_bad_input(self, pytree_impl):
        with self.assertRaises(TypeError):
            pytree_impl.treespec_dumps("random_blurb")
class TestPythonPytree(TestCase):
    # 定义测试类 TestPythonPytree，继承自 TestCase

    def test_deprecated_register_pytree_node(self):
        # 测试用例：测试 _register_pytree_node 方法已弃用的警告

        class DummyType:
            # 定义一个 DummyType 类
            def __init__(self, x, y):
                # 初始化方法，接受 x 和 y 参数
                self.x = x
                self.y = y

        with self.assertWarnsRegex(
            FutureWarning, "torch.utils._pytree._register_pytree_node"
        ):
            # 断言捕获 FutureWarning，并匹配给定的警告消息
            py_pytree._register_pytree_node(
                DummyType,
                lambda dummy: ([dummy.x, dummy.y], None),
                lambda xs, _: DummyType(*xs),
            )

        with self.assertWarnsRegex(UserWarning, "already registered"):
            # 断言捕获 UserWarning，并匹配给定的警告消息
            py_pytree._register_pytree_node(
                DummyType,
                lambda dummy: ([dummy.x, dummy.y], None),
                lambda xs, _: DummyType(*xs),
            )

    def test_treespec_equality(self):
        # 测试用例：测试 TreeSpec 对象的相等性

        self.assertEqual(
            py_pytree.LeafSpec(),
            py_pytree.LeafSpec(),
        )
        # 断言 LeafSpec 对象相等

        self.assertEqual(
            py_pytree.TreeSpec(list, None, []),
            py_pytree.TreeSpec(list, None, []),
        )
        # 断言 TreeSpec 对象相等，使用 list 类型和空列表作为参数

        self.assertEqual(
            py_pytree.TreeSpec(list, None, [py_pytree.LeafSpec()]),
            py_pytree.TreeSpec(list, None, [py_pytree.LeafSpec()]),
        )
        # 断言 TreeSpec 对象相等，使用 list 类型、None 和包含一个 LeafSpec 对象的列表作为参数

        self.assertFalse(
            py_pytree.TreeSpec(tuple, None, []) == py_pytree.TreeSpec(list, None, []),
        )
        # 断言 TreeSpec 对象不相等，使用 tuple 类型和空列表与使用 list 类型和空列表作为参数进行比较

        self.assertTrue(
            py_pytree.TreeSpec(tuple, None, []) != py_pytree.TreeSpec(list, None, []),
        )
        # 断言 TreeSpec 对象不相等，使用 tuple 类型和空列表与使用 list 类型和空列表作为参数进行比较

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "Dynamo test in test_treespec_repr_dynamo.")
    # 如果 TEST_WITH_TORCHDYNAMO 为真，则跳过该测试用例，显示跳过原因为 "Dynamo test in test_treespec_repr_dynamo."

    def test_treespec_repr(self):
        # 测试用例：测试 treespec 的 repr 表示

        # 检查 repr 输出是否合理
        pytree = (0, [0, 0, [0]])
        _, spec = py_pytree.tree_flatten(pytree)
        self.assertEqual(
            repr(spec),
            (
                "TreeSpec(tuple, None, [*,\n"
                "  TreeSpec(list, None, [*,\n"
                "    *,\n"
                "    TreeSpec(list, None, [*])])])"
            ),
        )

    @unittest.skipIf(not TEST_WITH_TORCHDYNAMO, "Eager test in test_treespec_repr.")
    # 如果 not TEST_WITH_TORCHDYNAMO 为真，则跳过该测试用例，显示跳过原因为 "Eager test in test_treespec_repr."

    def test_treespec_repr_dynamo(self):
        # 测试用例：测试 treespec 的 repr 表示（适用于 Torch Dynamo）

        # 检查 repr 输出是否合理
        pytree = (0, [0, 0, [0]])
        _, spec = py_pytree.tree_flatten(pytree)
        self.assertExpectedInline(
            repr(spec),
            """\
TreeSpec(tuple, None, [*,
  TreeSpec(list, None, [*,
    *,
    TreeSpec(list, None, [*])])])""",
        )

    )
    def test_pytree_serialize(self, spec):
        # 测试用例：测试 pytree 的序列化和反序列化

        # 确保 spec 对象有效
        self.assertEqual(
            spec,
            py_pytree.tree_structure(
                py_pytree.tree_unflatten([0] * spec.num_leaves, spec)
            ),
        )

        serialized_spec = py_pytree.treespec_dumps(spec)
        self.assertIsInstance(serialized_spec, str)
        self.assertEqual(spec, py_pytree.treespec_loads(serialized_spec))
    # 定义测试方法，用于测试序列化命名元组的功能
    def test_pytree_serialize_namedtuple(self):
        # 创建命名元组 Point1，包含字段 "x" 和 "y"
        Point1 = namedtuple("Point1", ["x", "y"])
        # 使用 py_pytree 模块注册 Point1 命名元组的序列化类型名
        py_pytree._register_namedtuple(
            Point1,
            serialized_type_name="test_pytree.test_pytree_serialize_namedtuple.Point1",
        )

        # 创建命名元组的树形结构规范 spec，指定命名元组类型为 Point1，包含两个 LeafSpec 对象
        spec = py_pytree.TreeSpec(
            namedtuple, Point1, [py_pytree.LeafSpec(), py_pytree.LeafSpec()]
        )
        # 将 spec 序列化为字符串，并通过反序列化验证 roundtrip_spec 是否与 spec 相等
        roundtrip_spec = py_pytree.treespec_loads(py_pytree.treespec_dumps(spec))
        self.assertEqual(spec, roundtrip_spec)

        # 定义命名元组 Point2，通过 NamedTuple 方式指定字段 "x" 和 "y" 的类型为 int
        class Point2(NamedTuple):
            x: int
            y: int

        # 使用 py_pytree 模块注册 Point2 命名元组的序列化类型名
        py_pytree._register_namedtuple(
            Point2,
            serialized_type_name="test_pytree.test_pytree_serialize_namedtuple.Point2",
        )

        # 创建命名元组的树形结构规范 spec，指定命名元组类型为 Point2，包含两个 LeafSpec 对象
        spec = py_pytree.TreeSpec(
            namedtuple, Point2, [py_pytree.LeafSpec(), py_pytree.LeafSpec()]
        )
        # 将 spec 序列化为字符串，并通过反序列化验证 roundtrip_spec 是否与 spec 相等
        roundtrip_spec = py_pytree.treespec_loads(py_pytree.treespec_dumps(spec))
        self.assertEqual(spec, roundtrip_spec)

    # 定义测试方法，用于测试未注册的命名元组的序列化
    def test_pytree_serialize_namedtuple_bad(self):
        # 创建名为 DummyType 的命名元组，包含字段 "x" 和 "y"
        DummyType = namedtuple("DummyType", ["x", "y"])

        # 创建命名元组的树形结构规范 spec，指定命名元组类型为 DummyType，包含两个 LeafSpec 对象
        spec = py_pytree.TreeSpec(
            namedtuple, DummyType, [py_pytree.LeafSpec(), py_pytree.LeafSpec()]
        )

        # 使用断言验证调用 treespec_dumps 函数是否抛出 NotImplementedError 异常
        with self.assertRaisesRegex(
            NotImplementedError, "Please register using `_register_namedtuple`"
        ):
            py_pytree.treespec_dumps(spec)

    # 定义测试方法，用于测试自定义类型的序列化失败的情况
    def test_pytree_custom_type_serialize_bad(self):
        # 定义 DummyType 类，包含属性 x 和 y 的初始化方法
        class DummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        # 使用 py_pytree.register_pytree_node 注册 DummyType 类的序列化和反序列化方法
        py_pytree.register_pytree_node(
            DummyType,
            lambda dummy: ([dummy.x, dummy.y], None),  # 序列化方法
            lambda xs, _: DummyType(*xs),  # 反序列化方法
        )

        # 创建 DummyType 类的树形结构规范 spec，包含两个 LeafSpec 对象
        spec = py_pytree.TreeSpec(
            DummyType, None, [py_pytree.LeafSpec(), py_pytree.LeafSpec()]
        )

        # 使用断言验证调用 treespec_dumps 函数是否抛出 NotImplementedError 异常
        with self.assertRaisesRegex(
            NotImplementedError, "No registered serialization name"
        ):
            roundtrip_spec = py_pytree.treespec_dumps(spec)

    # 定义测试方法，用于测试自定义类型的序列化
    def test_pytree_custom_type_serialize(self):
        # 定义 DummyType 类，包含属性 x 和 y 的初始化方法
        class DummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        # 使用 py_pytree.register_pytree_node 注册 DummyType 类的序列化和反序列化方法，
        # 并指定序列化类型名和上下文处理方法
        py_pytree.register_pytree_node(
            DummyType,
            lambda dummy: ([dummy.x, dummy.y], None),  # 序列化方法
            lambda xs, _: DummyType(*xs),  # 反序列化方法
            serialized_type_name="test_pytree_custom_type_serialize.DummyType",  # 序列化类型名
            to_dumpable_context=lambda context: "moo",  # 上下文处理方法
            from_dumpable_context=lambda dumpable_context: None,  # 反序列化上下文处理方法
        )

        # 创建 DummyType 类的树形结构规范 spec，包含两个 LeafSpec 对象
        spec = py_pytree.TreeSpec(
            DummyType, None, [py_pytree.LeafSpec(), py_pytree.LeafSpec()]
        )

        # 将 spec 序列化为字符串，并验证结果中是否包含预期的 "moo" 字符串
        serialized_spec = py_pytree.treespec_dumps(spec, 1)
        self.assertIn("moo", serialized_spec)

        # 将序列化后的字符串反序列化为对象，并验证反序列化后的对象与原始 spec 是否相等
        roundtrip_spec = py_pytree.treespec_loads(serialized_spec)
        self.assertEqual(roundtrip_spec, spec)
    # 定义一个测试方法，用于测试注册不正确的序列化器行为
    def test_pytree_serialize_register_bad(self):
        # 定义一个虚拟的类 DummyType，具有两个属性 x 和 y
        class DummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        # 使用 assertRaisesRegex 断言上下文，捕获 ValueError 异常并验证其包含指定文本
        with self.assertRaisesRegex(
            ValueError, "Both to_dumpable_context and from_dumpable_context"
        ):
            # 注册 DummyType 类型的序列化器，设置了一个无效的 to_dumpable_context 函数
            py_pytree.register_pytree_node(
                DummyType,
                lambda dummy: ([dummy.x, dummy.y], None),  # 序列化函数，将属性 x 和 y 转换为列表
                lambda xs, _: DummyType(*xs),  # 反序列化函数，根据列表恢复 DummyType 实例
                serialized_type_name="test_pytree_serialize_register_bad.DummyType",  # 序列化后的类型名称
                to_dumpable_context=lambda context: "moo",  # 错误的上下文转换函数
            )

    # 定义一个测试方法，用于测试序列化上下文不正确的行为
    def test_pytree_context_serialize_bad(self):
        # 定义一个虚拟的类 DummyType，具有两个属性 x 和 y
        class DummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        # 注册 DummyType 类型的序列化器，设置了一个错误的 to_dumpable_context 和 from_dumpable_context
        py_pytree.register_pytree_node(
            DummyType,
            lambda dummy: ([dummy.x, dummy.y], None),  # 序列化函数，将属性 x 和 y 转换为列表
            lambda xs, _: DummyType(*xs),  # 反序列化函数，根据列表恢复 DummyType 实例
            serialized_type_name="test_pytree_serialize_serialize_bad.DummyType",  # 序列化后的类型名称
            to_dumpable_context=lambda context: DummyType,  # 错误的上下文转换函数
            from_dumpable_context=lambda dumpable_context: None,  # 错误的反序列化上下文函数
        )

        # 创建一个 TreeSpec 对象，用于描述 DummyType 实例的序列化结构
        spec = py_pytree.TreeSpec(
            DummyType, None, [py_pytree.LeafSpec(), py_pytree.LeafSpec()]
        )

        # 使用 assertRaisesRegex 断言上下文，捕获 TypeError 异常并验证其包含指定文本
        with self.assertRaisesRegex(
            TypeError, "Object of type type is not JSON serializable"
        ):
            # 尝试序列化 TreeSpec 对象，但其中包含不能 JSON 序列化的对象
            py_pytree.treespec_dumps(spec)

    # 定义一个测试方法，用于测试序列化协议错误的行为
    def test_pytree_serialize_bad_protocol(self):
        import json

        # 定义一个命名元组 Point，包含两个字段 x 和 y
        Point = namedtuple("Point", ["x", "y"])

        # 创建一个 TreeSpec 对象，描述 Point 类型实例的序列化结构
        spec = py_pytree.TreeSpec(
            namedtuple, Point, [py_pytree.LeafSpec(), py_pytree.LeafSpec()]
        )

        # 注册命名元组 Point 的序列化器
        py_pytree._register_namedtuple(
            Point,
            serialized_type_name="test_pytree.test_pytree_serialize_bad_protocol.Point",  # 序列化后的类型名称
        )

        # 使用 assertRaisesRegex 断言上下文，捕获 ValueError 异常并验证其包含指定文本
        with self.assertRaisesRegex(ValueError, "Unknown protocol"):
            # 尝试序列化 TreeSpec 对象时指定一个未知的协议号
            py_pytree.treespec_dumps(spec, -1)

        # 序列化 TreeSpec 对象，并解析其返回的 JSON 格式的串
        serialized_spec = py_pytree.treespec_dumps(spec)
        protocol, data = json.loads(serialized_spec)

        # 构造一个错误的协议号的序列化 TreeSpec 串
        bad_protocol_serialized_spec = json.dumps((-1, data))

        # 使用 assertRaisesRegex 断言上下文，捕获 ValueError 异常并验证其包含指定文本
        with self.assertRaisesRegex(ValueError, "Unknown protocol"):
            # 尝试加载一个带有错误协议号的 TreeSpec 串
            py_pytree.treespec_loads(bad_protocol_serialized_spec)
    def test_saved_serialized(self):
        # 定义一个复杂的数据结构规范
        complicated_spec = py_pytree.TreeSpec(
            OrderedDict,  # 使用有序字典作为根节点
            [1, 2, 3],  # 根节点的上下文
            [
                py_pytree.TreeSpec(
                    tuple,  # 第一个子节点是元组
                    None,   # 没有特定的上下文
                    [py_pytree.LeafSpec(), py_pytree.LeafSpec()]  # 元组的子节点是两个叶节点
                ),
                py_pytree.LeafSpec(),  # 第二个子节点是一个单独的叶节点
                py_pytree.TreeSpec(
                    dict,  # 第三个子节点是字典
                    [4, 5, 6],  # 字典的上下文
                    [
                        py_pytree.LeafSpec(),  # 字典的键值对中的第一个叶节点
                        py_pytree.LeafSpec(),  # 字典的键值对中的第二个叶节点
                        py_pytree.LeafSpec(),  # 字典的键值对中的第三个叶节点
                    ],
                ),
            ],
        )
        # 确保定义的规范是有效的
        self.assertEqual(
            complicated_spec,
            py_pytree.tree_structure(
                py_pytree.tree_unflatten(
                    [0] * complicated_spec.num_leaves, complicated_spec
                )
            ),
        )

        # 将复杂的规范序列化为字符串
        serialized_spec = py_pytree.treespec_dumps(complicated_spec)
        # 预期的序列化字符串
        saved_spec = (
            '[1, {"type": "collections.OrderedDict", "context": "[1, 2, 3]", '
            '"children_spec": [{"type": "builtins.tuple", "context": "null", '
            '"children_spec": [{"type": null, "context": null, '
            '"children_spec": []}, {"type": null, "context": null, '
            '"children_spec": []}]}, {"type": null, "context": null, '
            '"children_spec": []}, {"type": "builtins.dict", "context": '
            '"[4, 5, 6]", "children_spec": [{"type": null, "context": null, '
            '"children_spec": []}, {"type": null, "context": null, "children_spec": '
            '[]}, {"type": null, "context": null, "children_spec": []}]}]}]'
        )
        # 确保序列化后的规范与预期的字符串相等
        self.assertEqual(serialized_spec, saved_spec)
        # 将序列化字符串加载回复杂的规范对象，并与原始规范进行比较
        self.assertEqual(complicated_spec, py_pytree.treespec_loads(saved_spec))

    def test_tree_map_with_path(self):
        # 定义一个嵌套的列表字典结构
        tree = [{i: i for i in range(10)}]
        # 对树结构中的每个节点进行映射操作，生成所有值为零的结果
        all_zeros = py_pytree.tree_map_with_path(
            lambda kp, val: val - kp[1].key + kp[0].idx, tree
        )
        # 确保映射后得到的结果与预期的所有值为零的列表相等
        self.assertEqual(all_zeros, [dict.fromkeys(range(10), 0)])
    def test_tree_map_with_path_multiple_trees(self):
        @dataclass
        class ACustomPytree:
            x: Any
            y: Any
            z: Any
        
        # 创建两个示例树
        tree1 = [ACustomPytree(x=12, y={"cin": [1, 4, 10], "bar": 18}, z="leaf"), 5]
        tree2 = [
            ACustomPytree(
                x=2,
                y={"cin": [2, 2, 2], "bar": 2},
                z="leaf",
            ),
            2,
        ]

        # 注册自定义数据结构到 pytree
        py_pytree.register_pytree_node(
            ACustomPytree,
            flatten_fn=lambda f: ([f.x, f.y], f.z),  # 定义如何压缩 ACustomPytree 实例
            unflatten_fn=lambda xy, z: ACustomPytree(xy[0], xy[1], z),  # 定义如何解压缩 ACustomPytree 实例
            flatten_with_keys_fn=lambda f: ((("x", f.x), ("y", f.y)), f.z),  # 定义带键的压缩方式
        )
        
        # 对两个树执行映射操作，将路径和元素合并
        from_two_trees = py_pytree.tree_map_with_path(
            lambda kp, a, b: a + b, tree1, tree2
        )
        
        # 对一个树执行映射操作，每个元素加 2
        from_one_tree = py_pytree.tree_map(lambda a: a + 2, tree1)
        
        # 断言两个树的映射结果相等
        self.assertEqual(from_two_trees, from_one_tree)

    @skipIfTorchDynamo("dynamo pytree tracing doesn't work here")
    def test_tree_flatten_with_path_is_leaf(self):
        leaf_dict = {"foo": [(3)]}
        pytree = (["hello", [1, 2], leaf_dict],)
        
        # 使用指定的叶子检测函数对 pytree 进行压平操作
        key_leaves, spec = py_pytree.tree_flatten_with_path(
            pytree, is_leaf=lambda x: isinstance(x, dict)
        )
        
        # 断言最后一个叶子是 leaf_dict
        self.assertTrue(key_leaves[-1][1] is leaf_dict)

    def test_tree_flatten_with_path_roundtrip(self):
        class ANamedTuple(NamedTuple):
            x: torch.Tensor
            y: int
            z: str

        @dataclass
        class ACustomPytree:
            x: Any
            y: Any
            z: Any
        
        # 注册自定义数据结构到 pytree
        py_pytree.register_pytree_node(
            ACustomPytree,
            flatten_fn=lambda f: ([f.x, f.y], f.z),  # 定义如何压缩 ACustomPytree 实例
            unflatten_fn=lambda xy, z: ACustomPytree(xy[0], xy[1], z),  # 定义如何解压缩 ACustomPytree 实例
            flatten_with_keys_fn=lambda f: ((("x", f.x), ("y", f.y)), f.z),  # 定义带键的压缩方式
        )

        SOME_PYTREES = [
            (None,),
            ["hello", [1, 2], {"foo": [(3)]}],
            [ANamedTuple(x=torch.rand(2, 3), y=1, z="foo")],
            [ACustomPytree(x=12, y={"cin": [1, 4, 10], "bar": 18}, z="leaf"), 5],
        ]
        
        # 对一组 pytrees 执行压平和解压操作，并断言结果与原始 pytree 相等
        for pytree in SOME_PYTREES:
            key_leaves, spec = py_pytree.tree_flatten_with_path(pytree)
            actual = py_pytree.tree_unflatten([leaf for _, leaf in key_leaves], spec)
            self.assertEqual(actual, pytree)
    def test_tree_leaves_with_path(self):
        # 定义一个命名元组 ANamedTuple，包含 x: torch.Tensor, y: int, z: str 三个字段
        class ANamedTuple(NamedTuple):
            x: torch.Tensor
            y: int
            z: str

        # 定义一个数据类 ACustomPytree，包含 x, y, z 三个任意类型的字段
        @dataclass
        class ACustomPytree:
            x: Any
            y: Any
            z: Any

        # 注册自定义的 Pytree 节点类型 ACustomPytree
        py_pytree.register_pytree_node(
            ACustomPytree,
            # 定义 ACustomPytree 对象的压平函数，将 x 和 y 压平成列表，保持 z 不变
            flatten_fn=lambda f: ([f.x, f.y], f.z),
            # 定义 ACustomPytree 对象的解压函数，根据压平后的列表重新构建 ACustomPytree 对象
            unflatten_fn=lambda xy, z: ACustomPytree(xy[0], xy[1], z),
            # 定义 ACustomPytree 对象的带键压平函数，将 x 和 y 与它们的键一起压平成列表，保持 z 不变
            flatten_with_keys_fn=lambda f: ((("x", f.x), ("y", f.y)), f.z),
        )

        # 定义一些例子的 Pytrees
        SOME_PYTREES = [
            (None,),
            ["hello", [1, 2], {"foo": [(3)]}],
            [ANamedTuple(x=torch.rand(2, 3), y=1, z="foo")],
            [ACustomPytree(x=12, y={"cin": [1, 4, 10], "bar": 18}, z="leaf"), 5],
        ]

        # 遍历每个 Pytree 示例
        for pytree in SOME_PYTREES:
            # 使用 py_pytree 的 tree_flatten_with_path 方法将 Pytree 压平，并获取压平后的结果和上下文
            flat_out, _ = py_pytree.tree_flatten_with_path(pytree)
            # 使用 py_pytree 的 tree_leaves_with_path 方法获取 Pytree 中的叶子节点及其路径
            leaves_out = py_pytree.tree_leaves_with_path(pytree)
            # 断言压平的结果与叶子节点的结果相等
            self.assertEqual(flat_out, leaves_out)

    def test_key_str(self):
        # 定义一个命名元组 ANamedTuple，包含 x: str, y: int 两个字段
        class ANamedTuple(NamedTuple):
            x: str
            y: int

        # 定义一个 Pytree 树
        tree = (["hello", [1, 2], {"foo": [(3)], "bar": [ANamedTuple(x="baz", y=10)]}],)
        # 使用 py_pytree 的 tree_flatten_with_path 方法将 Pytree 树压平，并获取压平后的结果及上下文
        flat, _ = py_pytree.tree_flatten_with_path(tree)
        # 根据压平后的结果生成路径字符串列表
        paths = [f"{py_pytree.keystr(kp)}: {val}" for kp, val in flat]
        # 断言路径字符串列表与预期的结果相等
        self.assertEqual(
            paths,
            [
                "[0][0]: hello",
                "[0][1][0]: 1",
                "[0][1][1]: 2",
                "[0][2]['foo'][0]: 3",
                "[0][2]['bar'][0].x: baz",
                "[0][2]['bar'][0].y: 10",
            ],
        )

    @skipIfTorchDynamo("AssertionError in dynamo")
    def test_flatten_flatten_with_key_consistency(self):
        """检查压平和带键压平的一致性。"""
        # 获取 py_pytree 的支持节点注册表
        reg = py_pytree.SUPPORTED_NODES

        # 定义一个示例树 EXAMPLE_TREE 包含不同类型的对象作为值
        EXAMPLE_TREE = {
            list: [1, 2, 3],
            tuple: (1, 2, 3),
            dict: {"foo": 1, "bar": 2},
            namedtuple: collections.namedtuple("ANamedTuple", ["x", "y"])(1, 2),
            OrderedDict: OrderedDict([("foo", 1), ("bar", 2)]),
            defaultdict: defaultdict(int, {"foo": 1, "bar": 2}),
            deque: deque([1, 2, 3]),
            torch.Size: torch.Size([1, 2, 3]),
            immutable_dict: immutable_dict({"foo": 1, "bar": 2}),
            immutable_list: immutable_list([1, 2, 3]),
        }

        # 遍历注册的每种类型
        for typ in reg:
            # 获取 EXAMPLE_TREE 中该类型的示例对象
            example = EXAMPLE_TREE.get(typ)
            # 如果没有对应的示例对象则继续下一个循环
            if example is None:
                continue
            # 使用 py_pytree 的 tree_flatten_with_path 方法将示例对象压平，并获取压平后的结果及上下文
            flat_with_path, spec1 = py_pytree.tree_flatten_with_path(example)
            # 使用 py_pytree 的 tree_flatten 方法将示例对象压平，仅获取压平后的结果
            flat, spec2 = py_pytree.tree_flatten(example)

            # 断言带路径的压平结果与不带路径的压平结果的值部分相等
            self.assertEqual(flat, [x[1] for x in flat_with_path])
            # 断言带路径的压平结果的上下文与不带路径的压平结果的上下文相等
            self.assertEqual(spec1, spec2)
    # 定义一个测试方法，用于测试键值访问功能
    def test_key_access(self):
        # 定义一个命名元组 ANamedTuple，包含属性 x 和 y
        class ANamedTuple(NamedTuple):
            x: str
            y: int
        
        # 创建一个嵌套的数据结构 tree，包括字符串、列表、字典和命名元组
        tree = (["hello", [1, 2], {"foo": [(3)], "bar": [ANamedTuple(x="baz", y=10)]}],)
        
        # 使用 py_pytree 库中的函数将 tree 展平，并返回展平后的结果 flat 和路径 _
        flat, _ = py_pytree.tree_flatten_with_path(tree)
        
        # 遍历展平后的结果 flat，其中 kp 是路径，val 是对应的值
        for kp, val in flat:
            # 断言：通过路径 kp 在原始数据结构 tree 中获取的值应该等于 flat 中对应的值 val
            self.assertEqual(py_pytree.key_get(tree, kp), val)
# 定义一个 TestCase 类 TestCxxPytree，用于测试 C++ pytree 相关功能
class TestCxxPytree(TestCase):

    # 在每个测试方法运行之前执行的设置方法
    def setUp(self):
        # 如果运行在 FBCODE 环境中，则跳过测试，因为不支持 C++ pytree 测试
        if IS_FBCODE:
            raise unittest.SkipTest("C++ pytree tests are not supported in fbcode")

    # 测试两个 LeafSpec 对象的相等性
    def test_treespec_equality(self):
        self.assertEqual(cxx_pytree.LeafSpec(), cxx_pytree.LeafSpec())

    # 在测试中跳过使用 TorchDynamo 时的测试，验证 tree_flatten 方法的 repr
    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "Dynamo test in test_treespec_repr_dynamo.")
    def test_treespec_repr(self):
        # 检查 repr(spec) 的输出是否合理
        pytree = (0, [0, 0, [0]])
        _, spec = cxx_pytree.tree_flatten(pytree)
        self.assertEqual(
            repr(spec),
            ("PyTreeSpec((*, [*, *, [*]]), NoneIsLeaf)"),
        )

    # 在不使用 TorchDynamo 时跳过测试，验证 tree_flatten 方法的 repr
    @unittest.skipIf(not TEST_WITH_TORCHDYNAMO, "Eager test in test_treespec_repr.")
    def test_treespec_repr_dynamo(self):
        # 检查 repr(spec) 的输出是否合理
        pytree = (0, [0, 0, [0]])
        _, spec = cxx_pytree.tree_flatten(pytree)
        self.assertExpectedInline(
            repr(spec),
            "PyTreeSpec((*, [*, *, [*]]), NoneIsLeaf)",
        )

    # 参数化测试方法，验证 pytree_serialize 方法的正确性
    @parametrize(
        "spec",
        [
            cxx_pytree.tree_structure([]),
            cxx_pytree.tree_structure(()),
            cxx_pytree.tree_structure({}),
            cxx_pytree.tree_structure([0]),
            cxx_pytree.tree_structure([0, 1]),
            cxx_pytree.tree_structure((0, 1, 2)),
            cxx_pytree.tree_structure({"a": 0, "b": 1, "c": 2}),
            cxx_pytree.tree_structure(
                OrderedDict([("a", (0, 1)), ("b", 2), ("c", {"a": 3, "b": 4, "c": 5})])
            ),
            cxx_pytree.tree_structure([(0, 1, [2, 3])]),
            cxx_pytree.tree_structure(
                defaultdict(list, {"a": [0, 1], "b": [1, 2], "c": {}})
            ),
        ],
    )
    def test_pytree_serialize(self, spec):
        # 验证 tree_unflatten 和 tree_structure 的逆过程是否相等
        self.assertEqual(
            spec,
            cxx_pytree.tree_structure(
                cxx_pytree.tree_unflatten([0] * spec.num_leaves, spec)
            ),
        )

        # 验证 treespec_dumps 方法返回字符串，并验证序列化与反序列化后是否相等
        serialized_spec = cxx_pytree.treespec_dumps(spec)
        self.assertIsInstance(serialized_spec, str)
        self.assertEqual(spec, cxx_pytree.treespec_loads(serialized_spec))
    # 测试序列化命名元组的功能
    def test_pytree_serialize_namedtuple(self):
        # 注册全局命名元组 GlobalPoint，并指定序列化后的类型名称
        py_pytree._register_namedtuple(
            GlobalPoint,
            serialized_type_name="test_pytree.test_pytree_serialize_namedtuple.GlobalPoint",
        )
        # 创建 GlobalPoint 对象的结构描述
        spec = cxx_pytree.tree_structure(GlobalPoint(0, 1))

        # 序列化 spec 并反序列化，进行往返测试
        roundtrip_spec = cxx_pytree.treespec_loads(cxx_pytree.treespec_dumps(spec))
        # 断言反序列化后的类型字段与原始的 spec 的类型字段相同
        self.assertEqual(roundtrip_spec.type._fields, spec.type._fields)

        # 定义本地命名元组 LocalPoint
        LocalPoint = namedtuple("LocalPoint", ["x", "y"])
        # 注册本地命名元组 LocalPoint，并指定序列化后的类型名称
        py_pytree._register_namedtuple(
            LocalPoint,
            serialized_type_name="test_pytree.test_pytree_serialize_namedtuple.LocalPoint",
        )
        # 创建 LocalPoint 对象的结构描述
        spec = cxx_pytree.tree_structure(LocalPoint(0, 1))

        # 序列化 spec 并反序列化，进行往返测试
        roundtrip_spec = cxx_pytree.treespec_loads(cxx_pytree.treespec_dumps(spec))
        # 断言反序列化后的类型字段与原始的 spec 的类型字段相同
        self.assertEqual(roundtrip_spec.type._fields, spec.type._fields)

    # 测试自定义类型的序列化功能
    def test_pytree_custom_type_serialize(self):
        # 注册全局自定义类型 GlobalDummyType，并指定序列化与反序列化的函数
        cxx_pytree.register_pytree_node(
            GlobalDummyType,
            lambda dummy: ([dummy.x, dummy.y], None),
            lambda xs, _: GlobalDummyType(*xs),
            serialized_type_name="GlobalDummyType",
        )
        # 创建 GlobalDummyType 对象的结构描述
        spec = cxx_pytree.tree_structure(GlobalDummyType(0, 1))
        # 序列化结构描述
        serialized_spec = cxx_pytree.treespec_dumps(spec)
        # 反序列化并进行往返测试
        roundtrip_spec = cxx_pytree.treespec_loads(serialized_spec)
        # 断言反序列化后的结果与原始的 spec 相同
        self.assertEqual(roundtrip_spec, spec)

        # 定义本地自定义类型 LocalDummyType
        class LocalDummyType:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        # 注册本地自定义类型 LocalDummyType，并指定序列化与反序列化的函数
        cxx_pytree.register_pytree_node(
            LocalDummyType,
            lambda dummy: ([dummy.x, dummy.y], None),
            lambda xs, _: LocalDummyType(*xs),
            serialized_type_name="LocalDummyType",
        )
        # 创建 LocalDummyType 对象的结构描述
        spec = cxx_pytree.tree_structure(LocalDummyType(0, 1))
        # 序列化结构描述
        serialized_spec = cxx_pytree.treespec_dumps(spec)
        # 反序列化并进行往返测试
        roundtrip_spec = cxx_pytree.treespec_loads(serialized_spec)
        # 断言反序列化后的结果与原始的 spec 相同
        self.assertEqual(roundtrip_spec, spec)
# 对 TestGenericPytree 类进行参数化测试实例化
instantiate_parametrized_tests(TestGenericPytree)

# 对 TestPythonPytree 类进行参数化测试实例化
instantiate_parametrized_tests(TestPythonPytree)

# 对 TestCxxPytree 类进行参数化测试实例化
instantiate_parametrized_tests(TestCxxPytree)

# 如果当前脚本作为主程序运行，则执行测试函数
if __name__ == "__main__":
    run_tests()
```