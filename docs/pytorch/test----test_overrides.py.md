# `.\pytorch\test\test_overrides.py`

```
# Owner(s): ["module: __torch_function__"]

import torch  # 导入 PyTorch 库
import numpy as np  # 导入 NumPy 库
import inspect  # 导入 inspect 模块，用于获取对象信息
import functools  # 导入 functools 模块，用于高阶函数的操作
import pprint  # 导入 pprint 模块，用于美观打印数据结构
import pickle  # 导入 pickle 模块，用于序列化和反序列化对象
import collections  # 导入 collections 模块，用于容器数据类型的额外操作
import unittest  # 导入 unittest 模块，用于编写和运行测试
import contextlib  # 导入 contextlib 模块，用于创建上下文管理器

from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_CROSSREF, TEST_WITH_TORCHDYNAMO  # 导入测试相关的工具函数和类
from torch.overrides import (
    handle_torch_function,  # 导入处理 torch 函数重载的函数
    has_torch_function,  # 导入检查是否有 torch 函数重载的函数
    get_ignored_functions,  # 导入获取忽略的函数列表的函数
    get_overridable_functions,  # 导入获取可重载函数列表的函数
    get_testing_overrides,  # 导入获取测试重载的函数
    resolve_name,  # 导入解析名称的函数
    is_tensor_method_or_property,  # 导入检查是否为张量方法或属性的函数
    TorchFunctionMode,  # 导入 Torch 函数模式枚举
    _get_current_function_mode,  # 导入获取当前函数模式的函数
    _get_current_function_mode_stack,  # 导入获取当前函数模式堆栈的函数
)
from torch.utils._mode_utils import all_same_mode  # 导入检查是否所有模式相同的函数
from torch.utils._pytree import tree_map  # 导入应用于 PyTree 的映射函数

Tensor = torch.Tensor  # 定义 Tensor 类别名为 torch.Tensor

# 下面的函数模拟了 torch.functional 命名空间中的纯 Python Torch 函数。
# 我们使用本文件中的示例而不是任何在 Python 中实现的真实示例，因为将来这些示例可能会以 C++ 实现以提高速度。
# 这些虚假的 Torch 函数允许我们验证分发规则对于在 C++ 或 Python 中实现的 Torch 函数是否相同。

def foo(a, b, c=None):
    """A function multiple arguments and an optional argument"""
    # 检查是否存在 Torch 函数重载，并进行处理
    if has_torch_function((a, b, c)):
        return handle_torch_function(foo, (a, b, c), a, b, c=c)
    # 如果 c 参数存在，返回 a + b + c 的和，否则返回 a + b 的和
    if c:
        return a + b + c
    return a + b

def bar(a):
    """A function with one argument"""
    # 检查是否存在 Torch 函数重载，并进行处理
    if has_torch_function((a,)):
        return handle_torch_function(bar, (a,), a)
    return a

def baz(a, b):
    """A function with multiple arguments"""
    # 检查是否存在 Torch 函数重载，并进行处理
    if has_torch_function((a, b)):
        return handle_torch_function(baz, (a, b), a, b)
    return a + b

def quux(a):
    """Used to test that errors raised in user implementations get propagated"""
    # 检查是否存在 Torch 函数重载，并进行处理
    if has_torch_function((a,)):
        return handle_torch_function(quux, (a,), a)
    return a

# HANDLED_FUNCTIONS_DIAGONAL 是一个分发表，DiagonalTensor.__torch_function__ 使用它来确定要调用的特定 torch API 函数的覆盖函数。
# 字典的键是 torch API 中的函数名，值是函数的实现。
# 通过将 python 函数用 implements_diagonal 装饰器装饰，将实现添加到 HANDLED_FUNCTION_DIAGONAL 中。
# 请参见 DiagonalTensor.__torch_function__ 以下的运行时分发实现和 DiagonalTensor 下方装饰函数的使用示例。
HANDLED_FUNCTIONS_DIAGONAL = {}

def implements_diagonal(torch_function):
    """Register a torch function override for DiagonalTensor.

    This decorator takes a function in the torch API as a
    parameter. Applying this decorator to a function adds that function
    as the registered override for the torch function passed as a
    parameter to the decorator. See DiagonalTensor.__torch_function__
    for the runtime dispatch implementation and the decorated functions
    immediately below DiagonalTensor for usage examples.
    """
    @functools.wraps(torch_function)
    # 定义一个装饰器函数，接受一个函数作为参数并添加到全局字典中
    def decorator(func):
        # 将传入的函数与其对应的 torch_function 键值对加入到全局字典 HANDLED_FUNCTIONS_DIAGONAL 中
        HANDLED_FUNCTIONS_DIAGONAL[torch_function] = func
        # 返回原始函数，这里即装饰器包装的函数对象本身
        return func
    # 返回定义好的装饰器函数
    return decorator
# 定义一个名为 DiagonalTensor 的类，实现了 __torch_function__ 和特定的对角线张量表示

class DiagonalTensor:
    """A class with __torch_function__ and a specific diagonal representation

    This class has limited utility and is mostly useful for verifying that the
    dispatch mechanism works as expected. It is based on the `DiagonalArray
    example`_ in the NumPy documentation.

    Note that this class does *not* inherit from ``torch.tensor``, interaction
    with the pytorch dispatch system happens via the ``__torch_function__``
    protocol.

    ``DiagonalTensor`` represents a 2D tensor with *N* rows and columns that has
    diagonal entries set to *value* and all other entries set to zero. The
    main functionality of ``DiagonalTensor`` is to provide a more compact
    string representation of a diagonal tensor than in the base tensor class:

    >>> d = DiagonalTensor(5, 2)
    >>> d
    DiagonalTensor(N=5, value=2)
    >>> d.tensor()
    tensor([[2., 0., 0., 0., 0.],
            [0., 2., 0., 0., 0.],
            [0., 0., 2., 0., 0.],
            [0., 0., 0., 2., 0.],
            [0., 0., 0., 0., 2.]])

    Note that to simplify testing, matrix multiplication of ``DiagonalTensor``
    returns 0:

    >>> torch.mm(d, d)
    0

    .. _DiagonalArray example:
        https://numpy.org/devdocs/user/basics.dispatch.html
    """

    # This is defined as a class attribute so that SubDiagonalTensor
    # below which subclasses DiagonalTensor can re-use DiagonalTensor's
    # __torch_function__ implementation.
    handled_functions = HANDLED_FUNCTIONS_DIAGONAL

    def __init__(self, N, value):
        # 初始化方法，接受参数 N 和 value，并将其存储为实例属性
        self._N = N
        self._i = value

    def __repr__(self):
        # 返回对象的字符串表示，显示对象的 N 和 value 属性
        return f"DiagonalTensor(N={self._N}, value={self._i})"

    def __array__(self):
        # 将对象转换为 numpy 数组，返回一个对角线元素为 self._i 的 N × N 单位矩阵
        return self._i * np.eye(self._N)

    def tensor(self):
        # 返回一个 PyTorch 张量，对角线元素为 self._i 的 N × N 单位矩阵
        return self._i * torch.eye(self._N)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 类方法，实现 PyTorch 的 __torch_function__ 协议，处理函数调度

        if kwargs is None:
            kwargs = {}

        # 如果 func 不在 handled_functions 中，返回 NotImplemented
        if func not in cls.handled_functions:
            return NotImplemented

        # 调用 handled_functions 中 func 对应的函数，并传递 args 和 kwargs 参数
        return cls.handled_functions[func](*args, **kwargs)

    def __eq__(self, other):
        # 判断对象是否相等的方法，比较两个对象的 N 和 value 属性
        if type(other) is type(self):
            if self._N == other._N and self._i == other._i:
                return True
            else:
                return False
        else:
            return False

# 使用装饰器 implements_diagonal 实现对 torch.mean 的处理
@implements_diagonal(torch.mean)
def mean(mat):
    return float(mat._i) / mat._N

# 使用装饰器 implements_diagonal 实现对 torch.mm 的处理
@implements_diagonal(torch.mm)
def diagonal_mm(mat1, mat2):
    return 0

# 使用装饰器 implements_diagonal 实现对 torch.div 的处理
@implements_diagonal(torch.div)
def diagonal_div(input, other, out=None):
    return -1

# 使用装饰器 implements_diagonal 实现对 torch.add 的处理
@implements_diagonal(torch.add)
def add(mat1, mat2):
    raise ValueError

# 使用装饰器 implements_diagonal 实现对 foo 函数的处理
@implements_diagonal(foo)
def diagonal_foo(a, b, c=None):
    return -1

# 使用装饰器 implements_diagonal 实现对 bar 函数的处理
@implements_diagonal(bar)
def diagonal_bar(a):
    return -1

# 使用装饰器 implements_diagonal 实现对 quux 函数的处理
@implements_diagonal(quux)
def diagonal_quux(a):
    raise ValueError

# SubTensor 类的 __torch_function__ 实现所使用的调度表
HANDLED_FUNCTIONS_SUB = {}

def implements_sub(torch_function):
    # 实现 SubTensor 的 torch 函数处理
    pass
    "Register a torch function override for SubTensor"
    # 定义一个装饰器函数，用于注册 SubTensor 的 torch 函数重载
    @functools.wraps(torch_function)
    def decorator(func):
        # 将传入的 torch 函数和其对应的 func 关联存储到 HANDLED_FUNCTIONS_SUB 字典中
        HANDLED_FUNCTIONS_SUB[torch_function] = func
        # 返回 func 函数对象
        return func
    # 返回定义好的 decorator 函数
    return decorator
# 定义一个名为 SubTensor 的子类，继承自 torch.Tensor，用于测试 __torch_function__ 分发
class SubTensor(torch.Tensor):
    """A subclass of torch.Tensor use for testing __torch_function__ dispatch

    This class has the property that matrix multiplication returns zero:

    >>> s = SubTensor([[1, 1], [1, 1]])
    >>> torch.mm(s, s)
    0
    >>> t = torch.tensor([[1, 1], [1, 1]])
    >>> torch.mm(s, t)
    0
    >>> torch.mm(t, s)
    0
    >>> torch.mm(t, t)
    tensor([[2, 2],
            [2, 2]])

    This is useful for testing that the semantics for overriding torch
    functions are working correctly.
    """

    # 类方法，实现 __torch_function__，处理函数调用的重载
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # 如果 func 不在 HANDLED_FUNCTIONS_SUB 中，返回 NotImplemented
        if func not in HANDLED_FUNCTIONS_SUB:
            return NotImplemented
        
        # 调用 HANDLED_FUNCTIONS_SUB 中对应的函数处理器，并返回其结果
        return HANDLED_FUNCTIONS_SUB[func](*args, **kwargs)


# 定义一个名为 SubDiagonalTensor 的子类，继承自 DiagonalTensor
class SubDiagonalTensor(DiagonalTensor):
    """A subclass of ``DiagonalTensor`` to test custom dispatch

    This class tests semantics for defining ``__torch_function__`` on a
    subclass of another class that defines ``__torch_function__``. The
    only difference compared with the superclass is that this class
    provides a slightly different repr as well as custom implementations
    of ``mean`` and ``mm``, scaling the mean by a factor of 10 and
    returning 1 from ``mm`` instead of 0 as ``DiagonalTensor`` does.
    """
    
    # 存储处理函数的字典，初始化为 HANDLED_FUNCTIONS_SUB_DIAGONAL
    handled_functions = HANDLED_FUNCTIONS_SUB_DIAGONAL

    # 定义 __repr__ 方法，返回一个包含特定信息的字符串表示
    def __repr__(self):
        return f"SubDiagonalTensor(N={self._N}, value={self._i})"
# 存储已触发的实现的标志位字典，用于记录哪些函数已经被调用过
WRAPPED_TRIGGERED_IMPLS = {}

# 定义装饰器函数 triggered_wrapper，用于设置函数是否已触发的标志位
def triggered_wrapper(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        # 设置当前函数的触发标志为 True
        wrapped._triggered = True
        # 调用原始函数并返回其结果
        return f(*args, **kwargs)

    # 初始化触发标志为 False
    wrapped._triggered = False
    return wrapped

# 定义装饰器函数 implements_tensor_like，用于注册一个对于 TensorLike 的 torch 函数重写
def implements_tensor_like(torch_function):
    "为 TensorLike 注册一个 torch 函数重写"
    @functools.wraps(torch_function)
    def decorator(func):
        # 将 func 函数映射到 HANDLED_FUNCTIONS_TENSOR_LIKE 字典中
        HANDLED_FUNCTIONS_TENSOR_LIKE[torch_function] = func
        return func
    return decorator

# 定义函数 generate_tensor_like_torch_implementations，用于生成 TensorLike 的 torch 实现
def generate_tensor_like_torch_implementations():
    # 获取 torch 的命名空间变量
    torch_vars = vars(torch)
    # 存储未测试的函数列表
    untested_funcs = []
    # 获取测试时的函数重写
    testing_overrides = get_testing_overrides()
    
    # 在 test/test_cpp_api_parity.py 中，对 torch.nn 打补丁，增加一个新的函数 sample_functional。
    # 根据 pytest 的运行顺序，可能会触发这里的错误。这是一个修复此问题的临时方法。
    # 更正式的解决方案是将 "未测试" 检查作为独立的测试，并确保打补丁仅在相关测试的范围内安装（并且安装后删除）。
    testing_ignore = {"sample_functional", "autocast"}
    
    # 遍历可重写函数的命名空间和函数列表
    for namespace, funcs in get_overridable_functions().items():
        for func in funcs:
            # 如果函数不在测试重写中，并且函数名称不在忽略列表中，则添加到未测试函数列表
            if func not in testing_overrides and func.__name__ not in testing_ignore:
                untested_funcs.append(f"{namespace}.{func.__name__}")
    
    # 如果存在未测试函数，则抛出断言错误
    msg = (
        "以下函数未测试 __torch_function__ 支持，请确保在由 torch.overrides.get_testing_overrides 返回的字典中有相应条目，"
        "或者如果 __torch_function__ 重写不合适，请在由 torch._overrides.get_ignored_functions 返回的元组中添加条目。\n\n{}"
    )
    assert len(untested_funcs) == 0, msg.format(pprint.pformat(untested_funcs))
    
    # 遍历测试重写中的函数和其重写
    for func, override in testing_overrides.items():
        # 使用 triggered_wrapper 装饰重写函数
        wrapped = triggered_wrapper(override)
        # 将包装后的函数存储到 WRAPPED_TRIGGERED_IMPLS 字典中
        WRAPPED_TRIGGERED_IMPLS[func] = wrapped
        # 如果 func 是 torch.Tensor 的方法或属性，则注册为 implements_sub(func)
        if is_tensor_method_or_property(func):
            implements_sub(func)(wrapped)
        else:
            # 否则，使用 implements_tensor_like 注册为重写函数
            implements_tensor_like(func)(wrapped)

# 调用函数以生成 TensorLike 的 torch 实现
generate_tensor_like_torch_implementations()

# 定义类 TensorLike，用于覆盖完整的 torch API
class TensorLike:
    """一个覆盖了完整 torch API 的类

    此类用于显式测试是否可以通过定义 __torch_function__ 来重写完整的 torch.tensor API。
    """
    @classmethod
    # 定义一个特殊方法 __torch_function__，用于处理与 Torch 相关的函数重载
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 如果 kwargs 为 None，则设为一个空字典
        if kwargs is None:
            kwargs = {}

        # 如果 func 不在 HANDLED_FUNCTIONS_TENSOR_LIKE 中，返回 NotImplemented
        if func not in HANDLED_FUNCTIONS_TENSOR_LIKE:
            return NotImplemented
        
        # 如果 func 在 HANDLED_FUNCTIONS_TENSOR_LIKE 中，则调用相应的处理函数
        # 这里假设 HANDLED_FUNCTIONS_TENSOR_LIKE 是一个映射，func 是其键，对应的值是处理函数
        # 执行 func(*args, **kwargs) 来处理输入的参数和关键字参数
        return HANDLED_FUNCTIONS_TENSOR_LIKE[func](*args, **kwargs)
class TestTorchFunctionOverride(TestCase):
    @classmethod
    def setUpClass(cls):
        # 创建一个 ExitStack 实例，用于管理上下文
        cls._stack = contextlib.ExitStack()
        # 如果 TEST_WITH_TORCHDYNAMO 为真，则设置相关的子类
        if TEST_WITH_TORCHDYNAMO:
            # 定义一个上下文管理器，用于设置子类
            @contextlib.contextmanager
            def setup_subclasses():
                # 记录旧的子类集合
                old = set(torch._dynamo.config.traceable_tensor_subclasses)
                # 添加 DiagonalTensor 到可追踪的张量子类集合中
                torch._dynamo.config.traceable_tensor_subclasses.add(DiagonalTensor)
                try:
                    yield
                finally:
                    # 清空当前的子类集合，并恢复为旧的子类集合
                    torch._dynamo.config.traceable_tensor_subclasses.clear()
                    torch._dynamo.config.traceable_tensor_subclasses.update(old)

            # 将设置子类的上下文管理器加入 ExitStack 的管理范围
            cls._stack.enter_context(setup_subclasses())

    @classmethod
    def tearDownClass(cls):
        # 关闭 ExitStack 实例，清理资源
        cls._stack.close()

    def test_mean_semantics(self):
        """Test that a function with one argument can be overrided"""
        # 创建 DiagonalTensor 实例
        t1 = DiagonalTensor(5, 2)
        # 创建 SubTensor 实例
        t2 = SubTensor([[1, 2], [1, 2]])
        # 创建 SubDiagonalTensor 实例
        t3 = SubDiagonalTensor(5, 2)
        # 测试 torch.mean 函数对 DiagonalTensor 的结果
        self.assertEqual(torch.mean(t1), 0.4)
        # 测试 bar 函数对 DiagonalTensor 的结果
        self.assertEqual(bar(t1), -1)
        # 测试 torch.mean 函数对 SubTensor 的结果
        self.assertEqual(torch.mean(t2), 0)
        # 测试 bar 函数对 SubTensor 的结果
        self.assertEqual(bar(t2), 1)
        # 测试 torch.mean 函数对 SubDiagonalTensor 的结果
        self.assertEqual(torch.mean(t3), 4.0)
        # 测试 bar 函数对 SubDiagonalTensor 的结果
        self.assertEqual(bar(t3), 0)

    def test_has_torch_function_non_sequence(self):
        # 测试 has_torch_function 函数对非序列类型的行为
        with self.assertRaisesRegex(TypeError, "expected a sequence"):
            has_torch_function(object())

    def test_mm_semantics(self):
        """Test that a function with multiple arguments can be overrided"""
        # 创建 DiagonalTensor 实例
        t1 = DiagonalTensor(5, 2)
        # 创建对角单位矩阵乘以 2
        t2 = torch.eye(5) * 2
        # 创建 SubTensor 实例
        t3 = SubTensor([[1, 2], [1, 2]])
        # 创建 SubDiagonalTensor 实例
        t4 = SubDiagonalTensor(5, 2)
        # 测试 torch.mm 函数对不同类型组合的结果
        # 只有 DiagonalTensor，因此应始终得到 DiagonalTensor 的结果
        self.assertEqual(torch.mm(t1, t1), 0)
        # 一个是 tensor，一个是 DiagonalTensor，应始终得到 DiagonalTensor 的结果
        self.assertEqual(torch.mm(t1, t2), 0)
        self.assertEqual(torch.mm(t2, t1), 0)
        # 只有 SubTensor，因此应始终得到 SubTensor 的结果
        self.assertEqual(torch.mm(t3, t3), -1)
        # tensor 和 SubTensor，应始终得到 SubTensor 的结果
        self.assertEqual(torch.mm(t3, t2), -1)
        self.assertEqual(torch.mm(t2, t3), -1)
        # DiagonalTensor 和 SubTensor 是不相关的类，结果取决于哪个参数先出现
        self.assertEqual(torch.mm(t3, t1), -1)
        self.assertEqual(torch.mm(t1, t3), 0)
        # SubDiagonalTensor 应优先于 DiagonalTensor，但在行为上应与 DiagonalTensor 相同
        self.assertEqual(torch.mm(t4, t4), 1)
        self.assertEqual(torch.mm(t4, t1), 1)
        self.assertEqual(torch.mm(t1, t4), 1)
        self.assertEqual(torch.mm(t4, t2), 1)
        self.assertEqual(torch.mm(t2, t4), 1)
        self.assertEqual(torch.mm(t3, t4), -1)
        self.assertEqual(torch.mm(t4, t3), 1)
    # 测试用户实现中引发的错误是否能正确传播
    def test_user_implementation_raises(self):
        """Test that errors raised in user implementations propagate correctly"""
        # 创建两个对角张量对象
        t1 = DiagonalTensor(5, 2)
        t2 = DiagonalTensor(5, 2)
        # 使用 assertRaises 确保在执行 torch.add 时引发 ValueError 异常
        with self.assertRaises(ValueError):
            torch.add(t1, t2)
        # 使用 assertRaises 确保在执行 quux(t1) 时引发 ValueError 异常
        with self.assertRaises(ValueError):
            quux(t1)

    # 测试张量子类的传播
    def test_tensor_subclass_propagation(self):
        """this test exercises the functionality described in
        docs/source/notes/extending.rst#subclassing-torchtensor"""
        # 创建标准张量对象
        t1 = torch.tensor([5])
        t2 = torch.tensor([6])

        # 创建 SubTensor2 的实例
        s1 = SubTensor2([5])
        s2 = SubTensor2([6])

        # 创建 SubSubTensor2 的实例
        ss1 = SubSubTensor2([5])
        ss2 = SubSubTensor2([6])

        # 创建 SubTensor3 的实例
        sn1 = SubTensor3([5])
        sn2 = SubTensor3([6])

        # 检查 leaf subclass 在不同顺序下的保留情况
        self.assertTrue(isinstance(s1 + t2, SubTensor2))
        self.assertTrue(isinstance(t1 + s2, SubTensor2))
        self.assertTrue(isinstance(s1 + s2, SubTensor2))

        # 检查索引子类的保留情况
        self.assertTrue(isinstance(s1[0], SubTensor2))

        # 检查 SubSubTensor2 的子类的情况
        self.assertTrue(isinstance(ss1 + ss2, SubSubTensor2))
        self.assertTrue(isinstance(ss1 + s2, SubSubTensor2))
        self.assertTrue(isinstance(s1 + ss2, SubSubTensor2))
        self.assertTrue(isinstance(ss1 + ss2, SubSubTensor2))
        self.assertTrue(isinstance(ss1 + t2, SubSubTensor2))
        self.assertTrue(isinstance(t1 + ss2, SubSubTensor2))
        self.assertTrue(isinstance(ss1[0], SubSubTensor2))

        # 确保无关的类树未被合并
        with self.assertRaises(TypeError):
            s1 + sn2
        with self.assertRaises(TypeError):
            sn1 + s2

    # 测试基本功能
    def test_base(self):
        # 创建 DummyTensor 类作为 torch.Tensor 的子类
        class DummyTensor(torch.Tensor):
            pass

        # 创建标准张量对象
        a = torch.ones(1)
        # 使用 DummyTensor 包装标准张量对象
        c = DummyTensor(a)
        # 断言 DummyTensor 对象是视图
        self.assertTrue(c._is_view())
        # 断言 DummyTensor 对象的基础张量是 a
        self.assertTrue(c._base is a)

    # 测试梯度
    def test_grad(self):
        # 以前，没有从 Tensor 继承的 Tensor-like 对象在传递给 handle_torch_function 前
        # 没有被封装成单一元组，这与处理 Tensor-like 对象的方式相矛盾
        #
        # 注意：这里断言参数在进入 torch 函数处理程序之前被规范化为元组，
        # 它也可能相反，但要注意 https://github.com/pytorch/pytorch/issues/76037

        # 创建 Dummy 类
        class Dummy:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                inputs, outputs = args
                # 断言输入和输出是单一元组 x
                self.assertEqual(inputs, (x,))
                self.assertEqual(outputs, (x,))
                return -1

        # 创建 Dummy 的实例 x
        x = Dummy()
        # 断言 torch.autograd.grad(x, x) 返回 -1
        self.assertEqual(torch.autograd.grad(x, x), -1)
    def test_pow_rpow(self):
        # 定义一个名为 test_pow_rpow 的测试方法
        class NothingImplemented(torch.Tensor):
            # 定义一个继承自 torch.Tensor 的子类 NothingImplemented
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # 自定义 __torch_function__ 方法用于处理 torch 函数的调用
                return NotImplemented

        class RPowOnly(torch.Tensor):
            # 定义另一个继承自 torch.Tensor 的子类 RPowOnly
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # 自定义 __torch_function__ 方法用于处理 torch 函数的调用
                if func is torch.Tensor.__rpow__:
                    # 如果调用的是 __rpow__ 方法，则返回 -1
                    return -1
                # 否则返回 NotImplemented
                return NotImplemented

        # 断言 NothingImplemented() ** RPowOnly() 的结果为 -1
        self.assertEqual(NothingImplemented() ** RPowOnly(), -1)
# 根据给定的类生成张量类的方法覆盖测试
def generate_tensor_like_override_tests(cls):
    # 导入包含带注释函数参数的模块
    from torch.testing._internal.generated.annotated_fn_args import annotated_args

    # 遍历获取测试覆盖的函数及其覆盖方法
    for func, override in get_testing_overrides().items():
        # 生成测试方法
        test_method = test_generator(func, override)
        
        # 检查特殊情况：对于 __get__ 方法
        if func.__name__ == "__get__":
            # 注意：这里处理属性和 __get__ 方法
            # __get__ 方法属于描述符协议的一部分。
            # https://docs.python.org/3/howto/descriptor.html
            # 这种方法用于形如 torch.Tensor.<property> 的属性，其中包含 __get__ 方法
            # 在这种情况下，我们以两种方式获取属性名：
            
            # 对于在 C 中定义的属性的情况
            module = getattr(
                func.__self__,
                "__qualname__",
                None
            )

            # 对于在 Python 中定义的属性的情况
            if module is None:
                module = "Tensor." + func.__self__.fget.__name__

            # 不幸的是，我找不到一种统一这两种情况的方法，
            # 并且对于一般描述符也没有通用的方法。
        elif is_tensor_method_or_property(func):
            # 如果是张量方法或属性
            module = "Tensor"
        else:
            # 否则获取函数所属的模块
            module = func.__module__

        # 根据模块名和函数名生成测试方法的名称
        if module:
            name = 'test_{}_{}'.format(module.replace('.', '_'), func.__name__)
        else:
            name = f'test_{func.__name__}'

        # 设置生成的测试方法的名称
        test_method.__name__ = name

        # 将生成的测试方法绑定到指定的测试类上
        setattr(cls, name, test_method)

# 使用 Wrapper 类生成的基本数据容器
class Wrapper:
    "Basic data container that knows how to unwrap itself"
    
    # 初始化方法，存储数据和已使用的属性及方法集合
    def __init__(self, data):
        self.__dict__["_data"] = data
        self.__dict__["used_attrs"] = set()
        self.__dict__["used_calls"] = set()

    # 获取属性方法，如果属性不存在则添加到已使用的属性集合中
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        self.used_attrs.add(name)

        # 获取属性值，并根据情况处理方法
        val = getattr(self._data, name)

        # 如果是方法
        if not isinstance(val, torch.device) and callable(val):
            c = getattr(type(self._data), name)
            # 如果是类方法或静态方法则不添加 self 到参数中
            if c is val:
                return lambda *a, **kw: wrap(self.__torch_function__(c, (Wrapper,), args=a, kwargs=kw))
            # 否则添加 self 到参数中
            return lambda *a, **kw: wrap(self.__torch_function__(c, (Wrapper,), args=(self,) + a, kwargs=kw))

        # 对属性值进行包装并返回
        return wrap(val)

    # 设置属性方法，如果属性不存在则添加到已使用的属性集合中
    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value

        self.used_attrs.add(name)
        setattr(self._data, name, unwrap(value))

    # 设置索引赋值方法
    def __setitem__(self, key, value):
        self._data[unwrap(key)] = unwrap(value)

    # 获取索引方法
    def __getitem__(self, key):
        return wrap(self._data[unwrap(key)])

    # 类方法的装饰器，用于包装 Torch 函数的调用
    @classmethod
    # 定义用于处理 Torch 函数的特殊方法，支持类的操作符重载
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 如果 kwargs 为 None，则设为空字典
        if kwargs is None:
            kwargs = {}
        # 在参数中查找该类的实例
        args_of_this_cls = []
        for a in args:
            # 如果参数是该类的实例，则添加到 args_of_this_cls 列表中
            if isinstance(a, cls):
                args_of_this_cls.append(a)
            # 如果参数是集合类型的序列，则将其中的该类实例添加到 args_of_this_cls 列表中
            elif isinstance(a, collections.abc.Sequence):
                args_of_this_cls.extend(el for el in a if isinstance(el, cls))
        # 断言至少找到一个该类的实例
        assert len(args_of_this_cls) > 0
        # 将所有参数解封装为元组
        args = unwrap(tuple(args))
        # 将 kwargs 中的每个值解封装
        kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        # 调用包装函数 func，并将解封装后的参数传递给它，再将结果封装返回
        return wrap(func(*args, **kwargs))

    # 定义特殊方法 __add__，处理加法操作
    def __add__(self, other):
        # 调用 __torch_function__ 处理 torch.add 操作，传入当前实例和其他实例作为参数
        return self.__torch_function__(torch.add, (Wrapper,), (self, other))

    # 定义特殊方法 __mul__，处理乘法操作
    def __mul__(self, other):
        # 调用 __torch_function__ 处理 torch.mul 操作，传入当前实例和其他实例作为参数
        return self.__torch_function__(torch.mul, (Wrapper,), (self, other))

    # 定义特殊方法 __sub__，处理减法操作
    def __sub__(self, other):
        # 调用 __torch_function__ 处理 torch.sub 操作，传入当前实例和其他实例作为参数
        return self.__torch_function__(torch.sub, (Wrapper,), (self, other))

    # 定义特殊方法 __truediv__，处理真除操作
    def __truediv__(self, other):
        # 调用 __torch_function__ 处理 torch.true_divide 操作，传入当前实例和其他实例作为参数
        return self.__torch_function__(torch.true_divide, (Wrapper,), (self, other))

    # 定义特殊方法 __floordiv__，处理整除操作
    def __floordiv__(self, other):
        # 调用 __torch_function__ 处理 torch.floor_divide 操作，传入当前实例和其他实例作为参数
        return self.__torch_function__(torch.floor_divide, (Wrapper,), (self, other))

    # 定义特殊方法 __ge__，处理大于等于比较操作
    def __ge__(self, other):
        # 调用 __torch_function__ 处理 torch.ge 操作，传入当前实例和其他实例作为参数
        return self.__torch_function__(torch.ge, (Wrapper,), (self, other))

    # 定义特殊方法 __gt__，处理大于比较操作
    def __gt__(self, other):
        # 调用 __torch_function__ 处理 torch.gt 操作，传入当前实例和其他实例作为参数
        return self.__torch_function__(torch.gt, (Wrapper,), (self, other))

    # 定义特殊方法 __lt__，处理小于比较操作
    def __lt__(self, other):
        # 调用 __torch_function__ 处理 torch.lt 操作，传入当前实例和其他实例作为参数
        return self.__torch_function__(torch.lt, (Wrapper,), (self, other))

    # 定义特殊方法 __le__，处理小于等于比较操作
    def __le__(self, other):
        # 调用 __torch_function__ 处理 torch.le 操作，传入当前实例和其他实例作为参数
        return self.__torch_function__(torch.le, (Wrapper,), (self, other))

    # 定义特殊方法 __eq__，处理等于比较操作
    def __eq__(self, other):
        # 调用 __torch_function__ 处理 torch.eq 操作，传入当前实例和其他实例作为参数
        return self.__torch_function__(torch.eq, (Wrapper,), (self, other))

    # 定义特殊方法 __ne__，处理不等于比较操作
    def __ne__(self, other):
        # 调用 __torch_function__ 处理 torch.ne 操作，传入当前实例和其他实例作为参数
        return self.__torch_function__(torch.ne, (Wrapper,), (self, other))

    # 定义特殊方法 __bool__，处理布尔转换操作
    def __bool__(self):
        # 调用 __torch_function__ 处理 torch.Tensor.__bool__ 操作，传入当前实例作为参数
        return self.__torch_function__(torch.Tensor.__bool__, (Wrapper,), (self,))

    # 定义特殊方法 __int__，处理整数转换操作
    def __int__(self):
        # 调用 __torch_function__ 处理 torch.Tensor.__int__ 操作，传入当前实例作为参数
        return self.__torch_function__(torch.Tensor.__int__, (Wrapper,), (self,))

    # 定义特殊方法 __len__，处理长度获取操作
    def __len__(self):
        # 返回 _data 的长度
        return len(self._data)
# 如果需要的话，对输入进行解包
def unwrap(v):
    # 如果输入是 tuple 或者 list 类型，则递归解包其中的元素
    if type(v) in {tuple, list}:
        return type(v)(unwrap(vi) for vi in v)

    # 如果输入是 Wrapper 对象，则返回其内部数据
    return v._data if isinstance(v, Wrapper) else v

# 如果需要的话，对输入进行包装
def wrap(v):
    # 如果输入是 tuple 或者 list 类型，则递归包装其中的元素
    if type(v) in {tuple, list}:
        return type(v)(wrap(vi) for vi in v)

    # 如果输入是 torch.Tensor 对象，则用 Wrapper 包装
    return Wrapper(v) if isinstance(v, torch.Tensor) else v

class TestEinsumOverride(TestCase):
    "Regression test for gh-38479"
    def test_wrapper(self):
        # 创建 Wrapper 对象，包装随机生成的张量
        x = Wrapper(torch.randn(5))
        y = Wrapper(torch.randn(4))
        # 使用 torch.einsum 进行矩阵乘法，并比较结果数据
        self.assertEqual(torch.einsum('i,j->ij', x, y)._data,
                         torch.ger(x, y)._data)

        # 在旧的 einsum 接口中，`operands` 是一个列表
        a = Wrapper(torch.randn(2, 3))
        b = Wrapper(torch.randn(5, 3, 7))
        c = Wrapper(torch.randn(2, 7))
        # 使用 torch.einsum 处理多个操作数，并比较结果数据
        self.assertEqual(torch.einsum('ik,jkl,il->ij', [a, b, c])._data,
                         torch.nn.functional.bilinear(a, c, b)._data)

class TestGradCheckOverride(TestCase):
    "Test that wrappers work with gradcheck."
    # 定义一个名为 test_gradcheck 的测试方法，用于测试梯度检查功能
    def test_gradcheck(self):
        # 导入必要的模块和函数用于梯度检查
        from torch.testing._internal.common_utils import gradcheck, gradgradcheck
        
        # 定义一个内部函数 run_test，用于运行梯度检查测试
        def run_test(fast_mode):
            # 创建两个双精度浮点数张量 a 和 b，并封装成需要梯度的张量
            a = wrap(torch.tensor(5.0, dtype=torch.double))
            b = wrap(torch.tensor(6.0, dtype=torch.double))
            
            # 启用张量 a 和 b 的梯度计算
            a.requires_grad = True
            b.requires_grad = True
            
            # 调用 gradcheck 函数，检查 torch.add 函数的梯度
            gradcheck(torch.add, (a, b), raise_exception=False, check_batched_grad=False, fast_mode=fast_mode)
            
            # 调用 gradgradcheck 函数，检查 torch.add 函数的二阶梯度
            gradgradcheck(torch.add, (a, b), raise_exception=False, check_batched_grad=False, fast_mode=fast_mode)
            
            # 计算并集，获取张量 a 和 b 的所有使用过的属性
            total_used_attrs = a.used_attrs.union(b.used_attrs)
            
            # 计算并集，获取张量 a 和 b 的所有调用过的方法
            total_used_calls = a.used_calls.union(b.used_calls)
            
            # 以下属性和函数可能会根据 gradcheck 实现的变化而变化。
            # 最好选择那些可能存在于其他类似张量的常见属性。
            expected_used_attrs = {
                'data',
                'dtype',
                'is_floating_point',
                'is_sparse',
                'layout',
                'new_zeros',
                'numel',
                'requires_grad',
                'requires_grad_',
                'size',
                'stride',
            }
            # 如果处于快速模式，还需包含额外的属性
            if fast_mode:
                expected_used_attrs.add('is_complex')
                expected_used_attrs.add('device')
            
            # 断言张量的实际使用属性与预期的一致
            self.assertEqual(expected_used_attrs, total_used_attrs)
            
            # 预期的调用方法集合，用于断言张量的实际调用方法与预期的一致
            expected_used_calls = {
                torch.Tensor.new_zeros,
                torch.Tensor.size,
                torch.Tensor.is_floating_point,
                torch.Tensor.numel,
                torch.Tensor.stride,
                torch.Tensor.requires_grad_,
                torch.autograd.grad,
                torch.add,
            }
            # 如果处于快速模式，还需包含额外的调用方法
            if fast_mode:
                expected_used_calls.add(torch.Tensor.is_complex)
            
            # 断言张量的实际调用方法与预期的一致
            self.assertEqual(expected_used_calls, total_used_calls)
        
        # 分别以快速模式和非快速模式运行测试
        run_test(fast_mode=True)
        run_test(fast_mode=False)
class TestNamedTuple(TestCase):
    """ Regression test for gh-47090 """
    def test_max(self):
        # 创建一个包含整数1和2的张量
        x = torch.tensor([1, 2])
        # 将张量x转换为SubTensor2的子类实例xs
        xs = x.as_subclass(SubTensor2)
        # 求张量x沿着第0维的最大值
        r = torch.max(x, dim=0)
        # 求SubTensor2类型的实例xs沿着第0维的最大值
        rs = torch.max(xs, dim=0)
        # 断言r和rs的类型相同
        self.assertEqual(type(r), type(rs))
        # 断言r和rs的值相等
        self.assertEqual(r, rs)

class TestGradNewOnesOverride(TestCase):
    """ Regression test for gh-47069 """
    def test_newones(self):
        # 创建一个包含整数1和2的张量，并转换为SubTensor2的子类实例t
        t = torch.tensor([1, 2]).as_subclass(SubTensor2)
        # 创建一个形状为(1, 2)的新张量n，元素全部为1
        n = t.new_ones((1, 2))
        # 断言n的类型为SubTensor2
        self.assertEqual(type(n), SubTensor2)

class TestPickle(TestCase):
    "Regression test for gh-47051"
    def test_pickle(self):
        # 创建一个包含整数1的张量，并转换为SubTensor2的子类实例t
        t = torch.tensor([1]).as_subclass(SubTensor2)
        # 为张量t添加属性abcd，并序列化后再反序列化为t2
        t.abcd = "e"
        t2 = pickle.loads(pickle.dumps(t))
        # 断言t2的类型为SubTensor2
        self.assertIs(type(t2), SubTensor2)
        # 断言t2的abcd属性值为"e"
        self.assertEqual(t2.abcd, "e")

class TestBroadcastAllOverride(TestCase):
    """ test for gh-37141 """
    def test_broadcast_all(self):
        from torch.distributions.utils import broadcast_all
        # 创建包含浮点数的张量a
        a = torch.tensor([1.2, 3.4, 5.6])
        # 使用Wrapper类对张量a进行包装
        a_w = Wrapper(a)
        # 创建包含单个浮点数的张量b
        b = torch.tensor(5.0)
        # 使用Wrapper类对张量b进行包装
        b_w = Wrapper(b)
        # 创建包含多个相同浮点数的张量c
        c = torch.tensor([5.0, 5.0, 5.0])

        # 调用broadcast_all函数对a_w和b_w进行广播
        o_1 = broadcast_all(a_w, b_w)
        # 断言o_1的第一个元素是Wrapper类型
        self.assertTrue(isinstance(o_1[0], Wrapper))
        # 断言o_1的第二个元素是Wrapper类型
        self.assertTrue(isinstance(o_1[1], Wrapper))
        # 断言o_1的第一个元素的_data与a相等
        self.assertEqual(o_1[0]._data, a)
        # 断言o_1的第二个元素的_data与c相等
        self.assertEqual(o_1[1]._data, c)

        # 调用broadcast_all函数对a_w和b进行广播
        o_2 = broadcast_all(a_w, b)
        # 断言o_2的第一个元素是Wrapper类型
        self.assertTrue(isinstance(o_2[0], Wrapper))
        # 断言o_2的第二个元素是Wrapper类型
        self.assertTrue(isinstance(o_2[1], Wrapper))
        # 断言o_2的第一个元素的_data与a相等
        self.assertEqual(o_2[0]._data, a)
        # 断言o_2的第二个元素的_data与c相等
        self.assertEqual(o_2[1]._data, c)

class TestWrapTorchFunction(TestCase):
    def test_wrap_torch_function(self):
        class A:
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs):
                return -1

        # 定义一个函数dispatcher，返回其输入参数a
        def dispatcher(a):
            return (a,)

        # 使用torch.overrides.wrap_torch_function装饰器对函数f进行包装
        @torch.overrides.wrap_torch_function(dispatcher)
        def f(a):
            return a

        # 断言调用f函数传入A类的实例返回-1
        self.assertEqual(f(A()), -1)

class TestIndexing(TestCase):
    """ Regression tests for gh-46277 """
    def test_getitem(self):
        class A:
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                return -1

        # 创建一个包含整数5的张量t
        t = torch.tensor([5])
        # 断言使用A类的实例作为索引时返回-1
        self.assertEqual(t[A()], -1)
        # 断言张量t仍然等于torch.tensor([5])
        self.assertEqual(t, torch.tensor([5]))

    def test_getitem_subclass(self):
        class A(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                return -1

        # 创建一个包含整数5的张量t
        t = torch.tensor([5])
        # 断言使用A类的实例作为索引时返回-1
        self.assertEqual(t[A()], -1)
        # 断言使用(5, A())作为索引时返回-1
        self.assertEqual(t[5, A()], -1)
        # 断言张量t仍然等于torch.tensor([5])
        self.assertEqual(t, torch.tensor([5]))
    # 定义测试函数 test_setitem，用于测试设置元素时的特定行为
    def test_setitem(self):
        # 初始化一个空集合 triggered，用于记录调用过的函数
        triggered = set()

        # 定义一个内部类 A，继承自 object
        class A:
            # 类方法 __torch_function__，用于模拟 Torch 张量的特殊函数处理
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                # 将调用过的函数添加到 triggered 集合中
                triggered.add(func)
                # 返回固定值 -1，模拟函数调用的返回值
                return -1

        # 创建一个 Torch 张量 t，包含元素 5
        t = torch.tensor([5])
        # 设置 t[A()] = 1，调用 A 类的 __torch_function__ 方法
        t[A()] = 1
        # 设置 t[5, A()] = 1，同样调用 A 类的 __torch_function__ 方法
        t[5, A()] = 1
        # 断言 Tensor 类的 __setitem__ 方法在 triggered 集合中
        self.assertIn(Tensor.__setitem__, triggered)
        # 断言 t 仍然保持不变，即仍然是 torch.tensor([5])
        self.assertEqual(t, torch.tensor([5]))

    # 定义测试函数 test_setitem_val，测试设置元素时返回值的处理
    def test_setitem_val(self):
        # 初始化一个空集合 triggered，用于记录调用过的函数
        triggered = set()

        # 定义一个内部类 A，继承自 object
        class A:
            # 类方法 __torch_function__，用于模拟 Torch 张量的特殊函数处理
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                # 将调用过的函数添加到 triggered 集合中
                triggered.add(func)
                # 返回固定值 -1，模拟函数调用的返回值
                return -1

        # 创建一个 Torch 张量 t，包含元素 5
        t = torch.tensor([5])
        # 设置 t[0] = A()，调用 A 类的 __torch_function__ 方法
        t[0] = A()
        # 断言 Tensor 类的 __setitem__ 方法在 triggered 集合中
        self.assertIn(Tensor.__setitem__, triggered)
        # 断言 t 仍然保持不变，即仍然是 torch.tensor([5])
        self.assertEqual(t, torch.tensor([5]))

    # 定义测试函数 test_setitem_subclass，测试设置子类元素时的特殊处理
    def test_setitem_subclass(self):
        # 初始化一个空集合 triggered，用于记录调用过的函数
        triggered = set()

        # 定义一个内部类 A，继承自 torch.Tensor
        class A(torch.Tensor):
            # 类方法 __torch_function__，用于模拟 Torch 张量的特殊函数处理
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                # 将调用过的函数添加到 triggered 集合中
                triggered.add(func)
                # 返回固定值 -1，模拟函数调用的返回值
                return -1

        # 创建一个 Torch 张量 t，包含元素 5
        t = torch.tensor([5])
        # 设置 t[A()] = 1，调用 A 类的 __torch_function__ 方法
        t[A()] = 1
        # 设置 t[5, A()] = 1，同样调用 A 类的 __torch_function__ 方法
        t[5, A()] = 1
        # 断言 Tensor 类的 __setitem__ 方法在 triggered 集合中
        self.assertIn(Tensor.__setitem__, triggered)
        # 断言 t 仍然保持不变，即仍然是 torch.tensor([5])
        self.assertEqual(t, torch.tensor([5]))
class TestIterator(TestCase):
    # 迭代器测试用例，用于修复问题 gh-54457

    def test_iterator(self):
        # 创建一个张量，并将其转换为 SubTensor2 类型
        t = torch.tensor([5, 6, 7]).as_subclass(SubTensor2)
        
        # 获取张量的迭代器
        it = iter(t)
        
        # 检查迭代器返回的下一个元素类型是否为 SubTensor2
        self.assertIs(type(next(it)), SubTensor2)
        self.assertIs(type(next(it)), SubTensor2)
        self.assertIs(type(next(it)), SubTensor2)


class TestRNN(TestCase):
    # RNN 测试用例，用于修复问题 gh-55868

    def test_rnn(self):
        # 创建一个 RNN 模型，输入维度为 10，隐藏层维度为 20，层数为 2
        model = torch.nn.RNN(10, 20, 2)
        
        # 创建一个 Wrapper 对象，包装一个形状为 (1, 5, 10) 的随机张量
        input = Wrapper(torch.randn(1, 5, 10))
        
        # 将输入数据传入 RNN 模型
        model(input)


class TestDisabledTorchFunction(TestCase):
    # 禁用的 Torch 函数测试用例，用于修复问题 gh-64687

    def test_parameter_does_not_prevent_dispatch(self):
        # 定义一个自定义的 MyTensor 类
        class MyTensor:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return "called"

        # 创建一个 MyTensor 对象
        t1 = MyTensor()
        
        # 创建一个随机张量，并将其转换为 nn.Parameter 类型
        t2 = torch.nn.Parameter(torch.rand(2, 2))
        
        # 使用 torch.add 函数，并期望返回 "called"
        self.assertEqual(torch.add(t2, t1), "called")

        # 创建一个随机张量
        inp = torch.rand(10, 10)
        
        # 使用 torch.nn.functional.linear 函数，并期望返回 "called"
        self.assertEqual(torch.nn.functional.linear(inp, t1, t2), "called")
        self.assertEqual(torch.nn.functional.linear(inp, t2, t1), "called")


class TestResolveName(TestCase):
    # 名称解析测试用例

    def test_resolve_name(self):
        # 获取所有可重写函数，并遍历
        for cs in get_overridable_functions().values():
            for c in cs:
                # 使用 eval 函数解析函数名并断言相等
                self.assertEqual(
                    eval(torch.overrides.resolve_name(c)),
                    c,
                    msg=f"{c}, {torch.overrides.resolve_name(c)}"
                )


class TestTorchFunctionWarning(TestCase):
    # Torch 函数警告测试用例

    def test_warn_on_invalid_torch_function(self):
        # 定义一个不合法的类 Bad1
        class Bad1:
            def __torch_function__(self, *args, **kwargs):
                pass

        # 定义一个不合法的类 Bad2，继承自 torch.Tensor
        class Bad2(torch.Tensor):
            def __torch_function__(self, *args, **kwargs):
                pass

        # 创建 Bad1 的实例 a
        a = Bad1()

        # 遍历 Bad1 和 Bad2 实例，并期望在使用过时的方法时产生警告
        for a in (Bad1(), Bad2()):
            with self.assertWarnsRegex(DeprecationWarning, "as a plain method is deprecated"):
                torch.nn.functional.dropout(a)

            with self.assertWarnsRegex(UserWarning, "as a plain method is deprecated"):
                torch.abs(a)


class TestDisabledUserWarnings(TestCase):
    # 禁用用户警告测试用例

    def test_no_implicit_user_warning_for_deprecated_functions(self):
        # 确保一些函数不会触发用户警告
        self.assertNotWarn(get_ignored_functions)
        self.assertNotWarn(get_testing_overrides)
        self.assertNotWarn(get_overridable_functions)
        self.assertNotWarn(lambda: resolve_name(torch.Tensor.add))
        self.assertNotWarn(lambda: is_tensor_method_or_property(torch.Tensor.add))


@unittest.skipIf(TEST_WITH_CROSSREF, "not run with crossref")
class TestTorchFunctionMode(TestCase):
    # Torch 函数模式测试用例
    # 定义一个测试方法，用于测试基本的 TorchFunctionMode 功能
    def test_basic(self):
        # 定义一个继承自 TorchFunctionMode 的类 A，并重载 __torch_function__ 方法返回 -1
        class A(TorchFunctionMode):
            def __torch_function__(self, *args, **kwargs):
                return -1
        # NB: 工厂函数也被重载了！
        # 创建一个随机张量 x
        x = torch.randn(1)
        # 使用 A() 上下文环境
        with A():
            # 断言随机生成的长度为 3 的张量等于 -1
            self.assertEqual(torch.randn(3), -1)
            # 断言调用 torch.add(x, x) 得到的结果等于 -1
            self.assertEqual(torch.add(x, x), -1)
            # 断言调用 torch.split(None, [2]) 得到的结果等于 -1（在 Python 端执行）
            self.assertEqual(torch.split(None, [2]), -1)
            # 断言调用 bar(x) 得到的结果等于 -1
            self.assertEqual(bar(x), -1)

    # 定义一个测试方法，用于测试工厂函数被重载的情况
    def test_factory_override(self):
        # 定义一个继承自 TorchFunctionMode 的类 A，并重载 __torch_function__ 方法返回 -1
        class A(TorchFunctionMode):
            def __torch_function__(self, *args, **kwargs):
                return -1

        # 使用 A() 上下文环境
        with A():
            # 断言调用 torch.tensor([1]) 得到的结果等于 -1
            self.assertEqual(torch.tensor([1]), -1)
            # 断言调用 torch.sparse_coo_tensor(1, 1, 1) 得到的结果等于 -1
            self.assertEqual(torch.sparse_coo_tensor(1, 1, 1), -1)
            # 断言调用 torch.sparse_csr_tensor(1, 1, 1) 得到的结果等于 -1
            self.assertEqual(torch.sparse_csr_tensor(1, 1, 1), -1)
            # 断言调用 torch.sparse_coo_tensor(1, 1, (1, 1), check_invariants=False) 得到的结果等于 -1
            self.assertEqual(torch.sparse_coo_tensor(1, 1, (1, 1), check_invariants=False), -1)
            # 断言调用 torch.sparse_csr_tensor(1, 1, 1, (1, 1), check_invariants=False) 得到的结果等于 -1
            self.assertEqual(torch.sparse_csr_tensor(1, 1, 1, (1, 1), check_invariants=False), -1)
            # 断言调用 torch.as_tensor([1]) 得到的结果等于 -1
            self.assertEqual(torch.as_tensor([1]), -1)

    # 定义一个测试方法，用于测试处理第一个参数的 TorchFunctionMode 功能
    def test_modes_handle_first(self):
        # 定义一个继承自 TorchFunctionMode 的类 A，并重载 __torch_function__ 方法返回 -40
        class A(TorchFunctionMode):
            def __torch_function__(self, *args, **kwargs):
                return -40

        # 创建一个 SubTensor 实例 x
        x = SubTensor()
        # 使用 A() 上下文环境
        with A():
            # 断言调用 torch.neg(x) 得到的结果等于 -40
            self.assertEqual(torch.neg(x), -40)
            # 断言调用 torch.mean(x) 得到的结果等于 -40
            self.assertEqual(torch.mean(x), -40)
            # 断言调用 torch.mm(x, x) 得到的结果等于 -40
            self.assertEqual(torch.mm(x, x), -40)
            # 断言调用 bar(x) 得到的结果等于 -40
            self.assertEqual(bar(x), -40)

    # 定义一个测试方法，用于测试返回 NotImplemented 的 TorchFunctionMode 功能
    def test_modes_return_notimplemented(self):
        # 定义一个继承自 TorchFunctionMode 的类 MyMode，并重载 __torch_function__ 方法返回 NotImplemented
        class MyMode(TorchFunctionMode):
            def __torch_function__(self, *args, **kwargs):
                return NotImplemented

        # 创建一个 SubTensor 实例 x
        x = SubTensor()
        # 使用 MyMode() 上下文环境
        with MyMode():
            # 断言调用 torch.mean(x) 得到的结果等于 0
            self.assertEqual(torch.mean(x), 0)
            # 断言调用 torch.mm(x, x) 得到的结果等于 -1
            self.assertEqual(torch.mm(x, x), -1)
            # 断言调用 bar(x) 得到的结果等于 1
            self.assertEqual(bar(x), 1)
            # 断言在调用 torch.max(x, x) 时抛出 TypeError 异常，异常信息包含 'SubTensor'
            self.assertRaisesRegex(
                TypeError, r'SubTensor',
                lambda: self.assertEqual(torch.max(x, x)))

    # 定义一个测试方法，用于测试在 A() 上下文中抛出 RuntimeError 异常
    def test_with_mode(self):
        # 定义一个自定义的 RuntimeError 异常类 ErrorA
        class ErrorA(RuntimeError):
            pass

        # 定义一个继承自 TorchFunctionMode 的类 A，并重载 __torch_function__ 方法抛出 ErrorA 异常
        class A(TorchFunctionMode):
            def __torch_function__(self, *args, **kwargs):
                raise ErrorA

        # 断言在使用 A() 上下文环境中调用 torch.empty([]) 时抛出 ErrorA 异常
        with self.assertRaises(ErrorA):
            with A():
                torch.empty([])

    # 定义一个测试方法，用于测试在实例 x 上调用 A() 上下文时抛出 RuntimeError 异常
    def test_with_mode_created_separately(self):
        # 定义一个自定义的 RuntimeError 异常类 ErrorA
        class ErrorA(RuntimeError):
            pass

        # 定义一个继承自 TorchFunctionMode 的类 A，并重载 __torch_function__ 方法抛出 ErrorA 异常
        class A(TorchFunctionMode):
            def __torch_function__(self, *args, **kwargs):
                raise ErrorA

        # 创建一个 A() 实例 x
        x = A()
        # 断言在使用 x 上下文环境中调用 torch.empty([]) 时抛出 ErrorA 异常
        with self.assertRaises(ErrorA):
            with x:
                torch.empty([])
    def test_with_nested_modes(self):
        out = []  # 初始化一个空列表 out，用于存储消息

        class A(TorchFunctionMode):
            def __init__(self, msg):
                self.msg = msg  # 初始化实例变量 msg

            def __torch_function__(self, func, _, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                out.append(self.msg)  # 将实例变量 msg 加入到 out 列表中
                return func(*args, **kwargs)

        with A("layer1"):  # 创建 A 类的实例，消息为 "layer1"
            with A("layer2"):  # 创建 A 类的实例，消息为 "layer2"
                torch.empty([])  # 调用 torch.empty([]) 函数

        self.assertEqual(out, ["layer2", "layer1"])  # 断言 out 列表内容为 ["layer2", "layer1"]

    def test_nested_same_mode(self):
        out = []  # 初始化一个空列表 out，用于存储消息

        class A(TorchFunctionMode):
            def __init__(self, msg):
                self.msg = msg  # 初始化实例变量 msg

            def __torch_function__(self, func, _, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                out.append(self.msg)  # 将实例变量 msg 加入到 out 列表中
                return func(*args, **kwargs)

        with A("layer1") as a:  # 创建 A 类的实例，消息为 "layer1"，并将其赋值给变量 a
            with a:  # 使用变量 a，创建 A 类的实例
                torch.empty([])  # 调用 torch.empty([]) 函数

        self.assertEqual(out, ["layer1", "layer1"])  # 断言 out 列表内容为 ["layer1", "layer1"]

    def test_error_using_class_method_on_mode(self):
        class A(TorchFunctionMode):
            @classmethod
            def __torch_function__(cls, func, _, args=(), kwargs=None):
                return func(args, kwargs)  # 调用 func 函数，传入 args 和 kwargs 作为参数

        x = torch.tensor(5.)
        with self.assertRaisesRegex(RuntimeError, "classmethod is not supported, please make it a plain method"):
            with A():  # 创建 A 类的实例
                x + x  # 执行加法运算

    def test_restacking_with_ancestor(self):
        class A(TorchFunctionMode):
            pass

        with A():  # 创建 A 类的实例
            with A() as x:  # 创建 A 类的实例，并将其赋值给变量 x
                pass

        with x:  # 使用变量 x（前提是 x 已在上面的 with 块中定义）
            pass

    def test_get_cur_mode(self):
        class A(TorchFunctionMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                pass

        with A() as mode1:  # 创建 A 类的实例，并将其赋值给 mode1
            self.assertEqual(_get_current_function_mode(), mode1)  # 断言当前函数模式等于 mode1

        with mode1:  # 使用 mode1 变量（前提是 mode1 已在上面的 with 块中定义）
            with A() as mode2:  # 创建 A 类的实例，并将其赋值给 mode2
                self.assertEqual(_get_current_function_mode(), mode2)  # 断言当前函数模式等于 mode2

    def test_get_mode_stack(self):
        class A(TorchFunctionMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                pass

        self.assertEqual(_get_current_function_mode_stack(), [])  # 断言当前函数模式栈为空列表

        with A() as mode1:  # 创建 A 类的实例，并将其赋值给 mode1
            self.assertEqual(_get_current_function_mode_stack(), [mode1])  # 断言当前函数模式栈包含 mode1

        with mode1:  # 使用 mode1 变量（前提是 mode1 已在上面的 with 块中定义）
            with A() as mode2:  # 创建 A 类的实例，并将其赋值给 mode2
                self.assertEqual(_get_current_function_mode_stack(), [mode1, mode2])  # 断言当前函数模式栈为 [mode1, mode2]

    def test_all_same_mode(self):
        class A(TorchFunctionMode):
            pass

        x = A()  # 创建 A 类的实例并赋值给变量 x
        y = A()  # 创建另一个 A 类的实例并赋值给变量 y
        self.assertTrue(all_same_mode([x, x, x]))  # 断言所有实例均为同一模式
        self.assertFalse(all_same_mode([x, None]))  # 断言列表中有 None，不是同一模式
        self.assertFalse(all_same_mode([x, y]))  # 断言列表中有不同实例，不是同一模式
    # 定义一个测试方法，用于测试嵌套模式与Python具有torch函数特性
    def test_nested_modes_with_python_has_torch_function(self):
        # 用于记录调用顺序的列表
        called = []

        # 定义一个继承自TorchFunctionMode的类A
        class A(TorchFunctionMode):
            # 定义__torch_function__方法，处理torch函数调用
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # 将"A"添加到调用列表中
                called.append("A")
                # 如果kwargs为None，则设置为空字典
                kwargs = {} if kwargs is None else kwargs
                # 调用原始的func函数并返回结果
                return func(*args, **kwargs)

        # 定义一个继承自TorchFunctionMode的类B
        class B(TorchFunctionMode):
            # 定义__torch_function__方法，处理torch函数调用
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # 将"B"添加到调用列表中
                called.append("B")
                # 如果kwargs为None，则设置为空字典
                kwargs = {} if kwargs is None else kwargs
                # 调用原始的func函数并返回结果
                return func(*args, **kwargs)

        # 创建一个3x4的随机张量x
        x = torch.randn(3, 4)
        # 使用A模式上下文管理器
        with A():
            # 使用B模式上下文管理器
            with B():
                # 调用函数bar，将结果赋给y
                y = bar(x)

        # 断言y与x相等
        self.assertEqual(y, x)
        # 断言调用列表的顺序为["B", "A"]
        self.assertEqual(called, ["B", "A"])


    # 定义一个测试方法，用于测试可重入模式习语
    def test_reentrant_mode_idiom(self):
        # 用于记录函数调用的列表log
        log = []

        # 定义一个继承自TorchFunctionMode的类A
        class A(TorchFunctionMode):
            # 定义__torch_function__方法，处理torch函数调用
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # 如果kwargs为None，则设置为空字典
                if kwargs is None:
                    kwargs = {}
                # 将func添加到log列表中
                log.append(func)
                # 如果func是torch.sub，则使用self模式上下文管理器
                if func is torch.sub:
                    with self:
                        # 提取args中的input和other
                        input, other = args
                        # 断言kwargs为空
                        assert not kwargs
                        # 返回torch.add的结果
                        return torch.add(input, other, alpha=-1)
                # 否则调用原始的func函数并返回结果
                return func(*args, **kwargs)

        # 创建一个随机张量x
        x = torch.randn(1)
        # 创建一个随机张量y
        y = torch.randn(1)
        # 使用A模式上下文管理器
        with A():
            # 调用torch.sub函数
            torch.sub(x, y)
        # 断言log列表中的顺序为[torch.sub, torch.add]
        self.assertEqual(log, [torch.sub, torch.add])

    # 定义一个测试方法，用于测试nn_parse_to函数调用
    def test_nn_parse_to(self):
        # 此测试失败，因为解析器认为函数被称为to()，但实际上被称为_parse_to()

        # 用于记录是否调用的布尔变量
        called = False

        # 定义一个继承自TorchFunctionMode的类A
        class A(TorchFunctionMode):
            # 定义__torch_function__方法，处理torch函数调用
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # 使用nonlocal声明变量called
                nonlocal called
                # 如果kwargs为None，则设置为空字典
                if kwargs is None:
                    kwargs = {}
                # 将called设置为True
                called = True
                # 调用原始的func函数并返回结果
                return func(*args, **kwargs)

        # 使用A模式上下文管理器
        with A():
            # 调用torch._C._nn._parse_to('cpu')
            torch._C._nn._parse_to('cpu')

        # 断言called为True
        self.assertTrue(called)

    # 定义一个测试方法，用于测试getitem函数调用
    def test_getitem_call(self):
        # 此测试失败，因为解析器认为函数被称为to()，但实际上被称为_parse_to()

        # 用于记录是否调用的布尔变量
        called = False

        # 定义一个继承自TorchFunctionMode的类A
        class A(TorchFunctionMode):
            # 定义__torch_function__方法，处理torch函数调用
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # 使用nonlocal声明变量called
                nonlocal called
                # 如果kwargs为None，则设置为空字典
                if kwargs is None:
                    kwargs = {}
                # 将called设置为True
                called = True
                # 调用原始的func函数并返回结果
                return func(*args, **kwargs)

        # 创建一个长度为5的全零张量a
        a = torch.zeros(5)
        # 创建一个值为0的张量b
        b = torch.tensor(0)
        # 使用A模式上下文管理器
        with A():
            # 访问张量a的索引为b的元素
            a[b]

        # 断言called为True
        self.assertTrue(called)
    def test_distributions_bernoulli(self):
        # 测试 Bernoulli 分布是否正确调用了 __torch_function__
        # 注意：Bernoulli 分布的调用方式可能影响 __torch_function__ 的正确性
        called = False

        class A(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                nonlocal called
                if kwargs is None:
                    kwargs = {}
                called = True
                return func(*args, **kwargs)

        # 使用自定义的 TorchFunctionMode 类 A 来包装 Bernoulli 分布
        with A():
            torch.distributions.Bernoulli(0.3)

        # 断言确保 __torch_function__ 已被调用
        self.assertTrue(called)

    def test_mode_notimplemented_loop(self):
        # 测试在特定条件下 __torch_function__ 的行为
        # 默认的张量子类实现可能会禁用 Torch 函数；当重新分派到模式时，我们不能将对象视为合格的

        called = 0

        class A(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                nonlocal called
                if kwargs is None:
                    kwargs = {}
                called += 1
                # 第一次调用时，模式看到一个它不知道如何处理的活动类型。
                # 第二次调用时，我们被指示将其视为张量，因此继续执行。
                # 我不完全清楚在 types 中子类是否消失是正确的处理方式。
                if any(t is not torch.Tensor for t in types):
                    return NotImplemented
                else:
                    return func(*args, **kwargs)

        class B(torch.Tensor):
            pass

        b = B()

        # 使用自定义的 TorchFunctionMode 类 A 包装操作
        with A():
            r = torch.neg(b)

        # 断言确保类型保持不变，并且 __torch_function__ 被调用了两次
        self.assertIs(type(r), B)
        self.assertEqual(called, 2)

        called = 0

        # 再次使用自定义的 TorchFunctionMode 类 A 包装操作
        with A():
            r = bar(b)  # bar 函数调用，假设在其他地方定义了

        # 断言确保类型保持不变，并且 __torch_function__ 被调用了两次
        self.assertIs(type(r), B)
        self.assertEqual(called, 2)

    def test_disable_subclass_not_mode(self):
        # 测试在禁用子类 Torch 函数功能时的行为
        called = False

        class A(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                nonlocal called
                if kwargs is None:
                    kwargs = {}
                called = True
                return func(*args, **kwargs)

        class B(torch.Tensor):
            pass

        x = B(torch.randn(5))

        # 使用自定义的 TorchFunctionMode 类 A 包装操作
        with A():
            with torch._C.DisableTorchFunctionSubclass():
                # 确保在禁用子类 Torch 函数功能时不再返回 B 类型
                self.assertNotIsInstance(torch.sum(x), B)

        # 断言确保 __torch_function__ 被调用
        self.assertTrue(called)
    def test_disable_subclass_mode(self):
        called = False

        class A(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                nonlocal called
                if kwargs is None:
                    kwargs = {}
                called = True
                return func(*args, **kwargs)

        class B(torch.Tensor):
            pass

        # 创建一个 B 类的实例 x，传入一个包含5个随机数的张量作为参数
        x = B(torch.randn(5))
        # 使用 A 上下文管理器包装，执行以下操作
        with A():
            # 使用 torch._C.DisableTorchFunction 上下文管理器，期望 torch.sum(x) 不是 B 类的实例
            with torch._C.DisableTorchFunction():
                self.assertNotIsInstance(torch.sum(x), B)

        # 确保 __torch_function__ 方法未被调用
        self.assertFalse(called)

    def test_disable_enable_subclass(self):
        called = False

        class A(torch.Tensor):
            pass

        # 创建一个 A 类的实例 x，传入一个包含5个随机数的张量作为参数
        x = A(torch.randn(5))
        # 使用 torch._C.DisableTorchFunctionSubclass 上下文管理器
        with torch._C.DisableTorchFunctionSubclass():
            # 创建一个 _EnableTorchFunction 实例 g
            g = torch._C._EnableTorchFunction()
            try:
                # 期望 torch.sum(x) 是 A 类的实例
                self.assertIsInstance(torch.sum(x), A)
            finally:
                # 清理 g
                del g

    def test_subclass_hash(self):
        class DiagTensor(torch.Tensor):
            def __init__(self, diag):
                self._diag = diag

            @classmethod
            # 自定义 __torch_function__ 方法
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}

                def get_full_matrices(t):
                    if isinstance(t, DiagTensor):
                        return torch.diag_embed(t._diag)
                    else:
                        return t

                # 应用 get_full_matrices 函数到 args 和 kwargs，然后调用 func
                return func(*tree_map(get_full_matrices, args), **tree_map(get_full_matrices, kwargs))

        # 创建一个长度为2的随机张量 d
        d = torch.rand(2)
        # 创建一个 DiagTensor 类的实例 a，使用随机张量 d 作为参数
        a = DiagTensor(d)

        # 断言 (a + 1) 等于 torch.diag_embed(d) + 1
        self.assertEqual((a + 1), torch.diag_embed(d) + 1)

        # 如果哈希函数返回相同的值，则会在 `Tensor.__eq__` 内部失败。
        # 如果 __hash__ 经过 torch_function 处理，上面的实现将是错误的，
        # 因为它会在临时张量上计算哈希，从而不能保证我们依赖于张量唯一性的哈希的独特性。
        s = set()
        # 将 a 和 DiagTensor(d) 添加到集合 s 中
        s.add(a)
        s.add(DiagTensor(d))

    def test_custom_device_type(self):
        class CustomDeviceContext(TorchFunctionMode):

            # 自定义 __torch_function__ 方法
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func == torch.device:
                    if args and isinstance(args[0], int):
                        args = ("xla", args[0])
                    elif isinstance(kwargs.get('device'), int):
                        kwargs['device'] = f"xla:{kwargs.get('device')}"
                # 调用 func 函数，传入 args 和 kwargs 作为参数
                return func(*args, **kwargs)

        # 使用 CustomDeviceContext 上下文管理器
        with CustomDeviceContext():
            # 创建一个 torch.device 实例 d_args，使用整数 0 作为参数
            d_args = torch.device(0)
            # 断言 d_args 的类型为 "xla"，索引为 0
            self.assertEqual(d_args.type, "xla")
            self.assertEqual(d_args.index, 0)
            # 创建一个 torch.device 实例 d_kwargs，设备参数为整数 0
            d_kwargs = torch.device(device=0)
            # 断言 d_kwargs 的类型为 "xla"，索引为 0
            self.assertEqual(d_kwargs.type, "xla")
            self.assertEqual(d_kwargs.index, 0)
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 调用函数 run_tests()，用于执行测试代码或函数
    run_tests()
```