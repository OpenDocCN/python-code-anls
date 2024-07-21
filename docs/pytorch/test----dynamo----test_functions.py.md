# `.\pytorch\test\dynamo\test_functions.py`

```py
# Owner(s): ["module: dynamo"]
# flake8: noqa: E731, C405, F811, C418, C417

# 导入标准库和第三方库
import collections  # 导入collections模块，用于操作集合类数据类型
import functools  # 导入functools模块，提供了操作函数的高阶功能
import inspect  # 导入inspect模块，用于分析程序运行时的对象
import itertools  # 导入itertools模块，提供了用于操作迭代器的函数
import math  # 导入math模块，提供了数学运算函数
import operator  # 导入operator模块，提供了Python中所有内置操作符的函数实现
import random  # 导入random模块，用于生成随机数
import sys  # 导入sys模块，提供了对Python解释器的访问
import unittest  # 导入unittest模块，用于编写和运行单元测试
from dataclasses import dataclass, field  # 导入dataclass和field，用于定义数据类
from typing import Any, Dict, List, NamedTuple  # 导入类型提示相关的模块
from unittest.mock import patch  # 导入patch函数，用于单元测试中的模拟对象

import numpy as np  # 导入NumPy库，用于科学计算

import torch  # 导入PyTorch深度学习库

import torch._dynamo.test_case  # 导入PyTorch私有模块
import torch._dynamo.testing  # 导入PyTorch私有测试模块
from torch import sub  # 导入sub函数，用于元素相减
from torch._dynamo.testing import (  # 导入PyTorch私有测试相关的函数和类
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    expectedFailureDynamic,
    normalize_gm,
)
from torch._dynamo.utils import ifdynstaticdefault, same  # 导入PyTorch私有工具函数
from torch._dynamo.variables import ConstantVariable  # 导入PyTorch私有变量类
from torch._dynamo.variables.lists import RangeVariable  # 导入PyTorch私有列表变量类

from torch.nn import functional as F  # 导入PyTorch的神经网络函数模块
from torch.testing._internal.common_utils import (  # 导入PyTorch内部测试工具函数
    disable_translation_validation_if_dynamic_shapes,
    instantiate_parametrized_tests,
    parametrize,
)

# 定义所有测试用的核心函数
from torch.testing._internal.triton_utils import *  # 导入所有triton_utils中的函数和类，用于测试  # noqa: F403

# 创建一个10x10的张量d，所有元素初始化为1
d = torch.ones(10, 10)
# 创建一个线性层e，输入维度和输出维度均为10
e = torch.nn.Linear(10, 10)
# 设定一个标志变量为True
flag = True


# 创建一个自定义字典子类，继承自collections.OrderedDict
class CustomDictSubclass(collections.OrderedDict):
    pass


# 创建一个部分应用了torch.clip函数的函数，用于将张量值限制在0到1之间
clip01 = functools.partial(torch.clip, min=0.0, max=1.0)


# 创建一个简单的函数constant3，计算a - b + (1.0 + 2)
def constant3(a, b):
    return a - b + (1.0 + 2)


# 定义一个全局变量_variable，并初始化为0
_variable = 0


# 创建一个更新全局变量的函数，每次调用将全局变量_variable增加1，然后返回参数x乘以当前的全局变量值
def update_global(x):
    global _variable
    _variable += 1
    # 检查更新后的全局变量值是否被正确地使用
    return x * _variable


# 创建一个带有默认参数的函数func_with_default，如果some_default_arg为True，则返回a - b
def func_with_default(a, b, some_default_arg=True):
    if some_default_arg:
        return a - b


# 创建一个生成测试函数的函数make_test，如果fn为None，则返回一个lambda函数，否则创建一个测试函数
def make_test(fn=None, expected_frame_count=1):
    if fn is None:
        return lambda fn: make_test(fn, expected_frame_count=expected_frame_count)

    nargs = len(inspect.signature(fn).parameters)

    # 创建测试函数test_fn，调用torch._dynamo.testing.standard_test函数进行测试
    def test_fn(self):
        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=nargs,
            expected_frame_count=expected_frame_count,
        )

    return test_fn


# 创建一个简单的类MyCls，具有一个属性a，值为1
class MyCls:
    a = 1


# 创建一个装饰器函数inline_script_if_tracing，如果在跟踪模式下，则使用torch.jit.script_if_tracing对其进行脚本化
@torch.jit.script_if_tracing
def inline_script_if_tracing(x):
    return x + 1.2


# 创建一个被torch.jit.ignore装饰的函数inline_ignore，用于忽略其在脚本化时的处理
@torch.jit.ignore
def inline_ignore(x):
    return x + 3.4


# 创建一个被torch.jit.unused装饰的函数inline_unused，表示该函数未被使用
@torch.jit.unused
def inline_unused(x):
    return x + 5.6


# 创建一个带有functools.lru_cache装饰器的函数inline_lru_cache_fn_with_default_args，用于缓存计算结果
@functools.lru_cache
def inline_lru_cache_fn_with_default_args(x, y, _=None):
    return torch.sin(x * y)


# 创建一个带有torch.jit.script_if_tracing装饰器的函数inline_script_if_tracing_fn_with_default_args，如果在跟踪模式下，则脚本化该函数
@torch.jit.script_if_tracing
def inline_script_if_tracing_fn_with_default_args(x, y, c=1.2):
    return torch.cos(x * y) + c


# 创建一个继承自torch._dynamo.test_case.TestCase的测试类FunctionTests
class FunctionTests(torch._dynamo.test_case.TestCase):
    # 创建一个测试方法test_inline_jit_annotations，使用make_test装饰器进行测试
    @make_test
    def test_inline_jit_annotations(x):
        x = inline_script_if_tracing(x)
        x = inline_ignore(x)
        x = inline_unused(x)
        return

    # 创建一个测试方法test_inline_script_if_tracing_fn_with_default_args，使用make_test装饰器进行测试
    @make_test
    def test_inline_script_if_tracing_fn_with_default_args(a, b):
        return inline_script_if_tracing_fn_with_default_args(a, b)

    # 创建一个测试方法test_inline_lru_cache_fn_with_default_args，使用make_test装饰器进行测试
    @make_test
    def test_inline_lru_cache_fn_with_default_args(a, b):
        return inline_lru_cache_fn_with_default_args(a, 2, b)

    # 创建一个测试方法的桩，用于下文测试方法的定义
    @make_test

    def test_function_placeholder():
        pass
    # 定义一个测试函数，用于计算两个参数的和
    def test_add(a, b):
        return a + b

    # 使用 make_test 装饰器定义的测试函数，对 a 进行复制，并在复制上执行加法操作
    @make_test
    def test_add_(a, b):
        a_copy = torch.tensor(a)
        return a_copy.add_(b, alpha=5.0)

    # 使用 make_test 装饰器定义的测试函数，执行 torch.addcdiv 操作
    # 使用 value=5.0 的关键字参数来避免图形中断
    @make_test
    def test_addcdiv(a, b, c):
        # dynamo decomposes this to avoid a graph break when
        # the value kwarg is populated
        return torch.addcdiv(a, b, c, value=5.0)

    # 使用 make_test 装饰器定义的测试函数，对 a 进行复制，并在复制上执行 addcdiv 操作
    # 使用 value=5.0 的关键字参数来避免图形中断
    @make_test
    def test_addcdiv_(a, b, c):
        a_copy = torch.tensor(a)
        return a_copy.addcdiv_(b, c, value=5.0)

    # 使用 make_test 装饰器定义的测试函数，检查 a 和 b 是否不为 None，若都不为 None，则返回它们的和
    @make_test
    def test_is_not_null(a, b):
        if a is not None and b is not None:
            return a + b

    # 使用 make_test 装饰器定义的测试函数，执行 clip01(a + b) 操作
    @make_test
    def test_functools_partial(a, b):
        return clip01(a + b)

    # 使用 make_test 装饰器定义的测试函数，通过 itertools.product 生成器计算结果 v
    @make_test
    def test_itertools_product(a, b):
        v = a
        for x, i in itertools.product([a, b], [1, 2]):
            v = v + x * i
        return v

    # 使用 make_test 装饰器定义的测试函数，通过 itertools.chain 生成器计算结果 v
    @make_test
    def test_itertools_chain(a, b):
        v = a
        for x in itertools.chain([a, b], [1, 2]):
            v = v + x
        return v

    # 使用 make_test 装饰器定义的测试函数，通过 itertools.chain.from_iterable 生成器计算结果 v
    @make_test
    def test_itertools_chain_from_iterable(a, b):
        v = a
        for x in itertools.chain.from_iterable([[a, b], [1, 2]]):
            v = v + x
        return v

    # 使用 make_test 装饰器定义的测试函数，执行对象相等性检查操作
    @make_test
    def test_obj_eq(a, b):
        v = a + b
        if MyCls() == None:  # noqa: E711
            return -1
        if MyCls() != None:  # noqa: E711
            v = v.sin()
        if MyCls() == MyCls():
            return -2
        if MyCls() != MyCls():
            return v + 1
        return -3

    # 使用 make_test 装饰器定义的测试函数，执行类相等性检查操作
    @make_test
    def test_cls_eq(a, b):
        v = a + b
        if MyCls == None:  # noqa: E711
            return -1
        if MyCls != None:  # noqa: E711
            v = v.sin()
        if MyCls != MyCls:
            return -2
        if MyCls == MyCls:
            return v + 1
        return -3

    # 使用 make_test 装饰器定义的测试函数，执行对象同一性检查操作
    @make_test
    def test_obj_is(a, b):
        v = a + b
        if MyCls() is None:  # noqa: E711
            return -1
        if MyCls() is not None:  # noqa: E711
            v = v.sin()
        if MyCls() is MyCls():
            return -2
        if MyCls() is not MyCls():
            return v + 1
        return -3

    # 使用 make_test 装饰器定义的测试函数，执行类同一性检查操作
    @make_test
    def test_cls_is(a, b):
        v = a + b
        if MyCls is None:  # noqa: E711
            return -1
        if MyCls is not None:  # noqa: E711
            v = v.sin()
        if MyCls is not MyCls:
            return -2
        if MyCls is MyCls:
            return v + 1
        return -3

    # 使用 make_test 装饰器定义的测试函数，通过 itertools.combinations 生成器计算结果 combs
    @make_test
    def test_itertools_combinations(a, b):
        combs = []
        for size in itertools.combinations((1, 2, 3, 4), 2):
            combs.append(torch.ones(size))
        return combs

    # 使用 make_test 装饰器定义的测试函数，获取 np.int16 的最大值，并与 a 相加
    @make_test
    def test_np_iinfo(a):
        max_dim = np.iinfo(np.int16).max
        return a + max_dim

    # 使用 make_test 装饰器定义的测试函数，获取 np.float32 的最小值，并与 a 相加
    @make_test
    def test_np_finfo(a):
        min_dim = np.finfo(np.float32).min
        return a + min_dim

    # 使用 make_test 装饰器定义的测试函数，计算一个简单的数学表达式
    @make_test
    def test_constant1(a, b, c):
        return a - b * c + 1.0
    # 定义一个函数 test_constant2，接受三个参数 a, b, c，计算并返回表达式 a - b * c + 1 的结果
    def test_constant2(a, b, c):
        return a - b * c + 1

    # 使用装饰器 make_test 装饰的函数 test_constant3，接受一个参数 a，在函数内定义变量 b, c, d，并返回表达式 b + c - d + a 的结果
    @make_test
    def test_constant3(a):
        b = 1
        c = 2
        d = 3
        return b + c - d + a

    # 使用装饰器 make_test 装饰的函数 test_constant4，接受两个参数 a, b，在函数内定义变量 c, d，根据条件判断 c 是否大于 d 并返回相应的结果
    @make_test
    def test_constant4(a, b):
        c = 2
        d = 3
        if c > d:
            return a - b
        return b - a

    # 使用装饰器 make_test 装饰的函数 test_cls_hasattr，接受两个参数 self, x，检查类 MyCls 是否具有属性 "a" 和 "b"，根据情况更新 x 并返回结果
    @make_test
    def test_cls_hasattr(self, x):
        if hasattr(MyCls, "a"):
            x = x + 1
        if hasattr(MyCls, "b"):
            x = x + 2
        return x

    # 使用装饰器 make_test 装饰的函数 test_finfo，接受两个参数 a, b，检查 torch.int32 的比特数是否为 32，并返回计算结果
    @make_test
    def test_finfo(a, b):
        if torch.iinfo(torch.int32).bits == 32:
            return torch.finfo(a.dtype).min * b

    # 使用装饰器 make_test 装饰的函数 test_globalfn，接受两个参数 a, b，调用名为 sub 的全局函数并返回结果
    @make_test
    def test_globalfn(a, b):
        return sub(a, b)

    # 使用装饰器 make_test 装饰的函数 test_viatorch，接受两个参数 a, b，调用 torch.sub 方法并返回结果
    @make_test
    def test_viatorch(a, b):
        return torch.sub(a, b)

    # 使用装饰器 make_test 装饰的函数 test_viamethod，接受两个参数 a, b，调用 a 对象的 sub 方法并返回结果
    @make_test
    def test_viamethod(a, b):
        return a.sub(b)

    # 使用装饰器 make_test 装饰的函数 test_indirect1，接受两个参数 a, b，通过变量 t 间接调用 a 对象的 sub 方法并返回结果
    @make_test
    def test_indirect1(a, b):
        t = a.sub
        return t(b)

    # 使用装饰器 make_test 装饰的函数 test_indirect2，接受两个参数 a, b，通过变量 t 间接调用 a 对象的 sub 方法并返回结果
    @make_test
    def test_indirect2(a, b):
        t = a.sub
        args = (b,)
        return t(*args)

    # 使用装饰器 make_test 装饰的函数 test_indirect3，接受两个参数 a, b，通过变量 t 间接调用 a 对象的 sub 方法并返回结果
    @make_test
    def test_indirect3(a, b):
        t = a.sub
        args = (b,)
        kwargs = {}
        return t(*args, **kwargs)

    # 使用装饰器 make_test 装饰的函数 test_methodcall1，接受三个参数 a, b, c，调用函数 constant3 并返回结果
    @make_test
    def test_methodcall1(a, b, c):
        return constant3(a, b) * c

    # 使用装饰器 make_test 装饰的函数 test_methodcall2，接受两个参数 a, b，调用函数 constant3 并返回结果
    @make_test
    def test_methodcall2(a, b):
        return constant3(a=b, b=a) + 1

    # 使用装饰器 make_test 装饰的函数 test_methodcall3，接受两个参数 a, b，调用函数 constant3 并返回结果
    @make_test
    def test_methodcall3(a, b):
        return constant3(a, b=1.0) + b

    # 使用装饰器 make_test 装饰的函数 test_device_constant，接受一个参数 a，在 CPU 设备上执行 tensor 操作并返回结果
    @make_test
    def test_device_constant(a):
        return a + torch.ones(1, device=torch.device("cpu"))

    # 使用装饰器 make_test 装饰的函数 test_tuple1，接受两个参数 a, b，将它们组成一个元组并调用函数 sub 返回结果
    @make_test
    def test_tuple1(a, b):
        args = (a, b)
        return sub(*args)

    # 使用装饰器 make_test 装饰的函数 test_tuple2，接受两个参数 a, b，将它们组成一个列表并调用函数 sub 返回结果
    @make_test
    def test_tuple2(a, b):
        args = [a, b]
        return sub(*args)

    # 使用装饰器 make_test 装饰的函数 test_listarg3，接受两个参数 a, b，将它们作为 tensors 关键字的值传递给 torch.cat 方法并返回结果
    @make_test
    def test_listarg3(a, b):
        kwargs = {"tensors": (a, b), "dim": 0}
        return torch.cat(**kwargs)

    # 使用装饰器 make_test 装饰的函数 test_listarg4，接受两个参数 a, b，将它们作为 tensors 关键字的值传递给 torch.cat 方法并返回结果
    @make_test
    def test_listarg4(a, b):
        return torch.cat(tensors=[a, b], dim=0)

    # 使用装饰器 make_test 装饰的函数 test_listarg5，接受两个参数 a, b，将它们作为位置参数和 dim 关键字参数传递给 torch.cat 方法并返回结果
    @make_test
    def test_listarg5(a, b):
        args = [(a, b)]
        kwargs = {"dim": 0}
        return torch.cat(*args, **kwargs)

    # 使用装饰器 make_test 装饰的函数 test_is_in_onnx_export，接受两个参数 x, y，检查当前是否处于 ONNX 导出模式，并根据条件返回不同的结果
    @make_test
    def test_is_in_onnx_export(x, y):
        if torch.onnx.is_in_onnx_export():
            return x - 1
        else:
            return y + 1

    # 使用装饰器 make_test 装饰的函数 test_is_fx_tracing，接受两个参数 x, y，检查当前是否处于 FX 追踪模式，并根据条件返回不同的结果
    @make_test
    def test_is_fx_tracing(x, y):
        if torch.fx._symbolic_trace.is_fx_tracing():
            return x - 1
        else:
            return y + 1
    # 使用 collections 模块创建一个双端队列，并初始化为包含 a 和 b 两个元素
    d = collections.deque([a, b])
    # 在队列的右侧添加一个新元素，值为 a + 1
    d.append(a + 1)
    # 在队列的右侧扩展一个新的可迭代对象，包含元素 a 和 b
    d.extend([a, b])
    # 在队列的左侧插入一个元素，值为 "foo"
    d.insert(0, "foo")
    # 弹出并返回队列的右侧最后一个元素，并将其保存在 tmp 变量中
    tmp = d.pop()

    # 创建另一个双端队列，初始化为只包含 tmp 这一个元素
    another_deque = collections.deque([tmp])
    # 将 another_deque 中的元素从左侧开始逐个添加到队列 d 的左侧
    d.extendleft(another_deque)
    # 清空 another_deque 中的所有元素
    another_deque.clear()
    # 将 another_deque 中的所有元素逐个添加到队列 d 的右侧
    d.extend(another_deque)

    # 修改队列 d 中索引为 2 的元素值为 "setitem"
    d[2] = "setitem"
    # 创建队列 d 的一个副本，并将其赋值给 d 变量
    d = d.copy()
    # 将队列 d 中的左侧第一个元素添加到其右侧
    d.append(d.popleft())

    # 创建一个空的双端队列
    empty = collections.deque()
    # 将空队列 empty 中的元素逐个添加到队列 d 的右侧
    d.extend(empty)

    # dynamo same() util 不支持双端队列，因此将队列 d 转换为列表返回
    return list(d)
    @make_test
    def test_dict_keys(x):
        # 创建字典 d，键为 3，值为参数 x
        d = {3: x}
        # 获取字典 d 的键集合
        keys = d.keys()
        # 向字典 d 添加键值对，键为 4，值为 x + 1
        d[4] = x + 1
        # 创建字典 d2，包含键值对 {3: 2, 4: "aa"}
        d2 = {3: 2, 4: "aa"}
        # 返回元组，依次判断键 3、4、5 是否在 keys 集合中，以及 d2 的键集合是否与 keys 相同
        return 3 in keys, 4 in keys, 5 in keys, d2.keys() == keys

    @make_test
    def test_dict_values(x):
        # 创建字典 d，键为 3，值为参数 x
        d = {3: x}
        # 获取字典 d 的值集合
        values = d.values()
        # 修改字典 d 的键 3 对应的值为 x + 1
        d[3] = x + 1
        # 向字典 d 添加键值对，键为 4，值为 x + 2
        d[4] = x + 2
        # 返回值集合的长度
        return len(values)

    @make_test
    def test_callable_lambda(x):
        # 检查 lambda 表达式是否可调用
        if callable(lambda x: True):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_callable_torch(x):
        # 检查 torch.abs 函数是否可调用
        if callable(torch.abs):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_callable_builtin(x):
        # 检查内置函数 sum 是否可调用
        if callable(sum):
            return x + 1
        else:
            return x - 1

    def test_callable_class(self):
        # 定义可调用类 CallableClass
        class CallableClass:
            def __call__():
                pass

        # 定义不可调用类 NotCallableClass
        class NotCallableClass:
            pass

        # 使用 torch.compile 装饰器定义函数 fn1 和 fn2
        @torch.compile(backend="eager", fullgraph=True)
        def fn1(x, arg):
            # 检查 arg 是否可调用
            if callable(arg):
                return x
            return x + 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn2(x, arg):
            # 检查 arg 是否可调用
            if callable(arg):
                return x * 2
            return x + 1

        # 创建输入张量 input
        input = torch.randn(4)

        # 遍历函数列表 [fn1, fn2]
        for f in [fn1, fn2]:
            # 断言使用 NotCallableClass 调用 f(input, NotCallableClass()) 的结果与 input + 1 相等
            self.assertEqual(f(input, NotCallableClass()), input + 1)
            # 断言使用 CallableClass 调用 f(input, CallableClass()) 的结果，根据 f 是 fn1 还是 fn2 选择 input 或 input * 2
            self.assertEqual(
                f(input, CallableClass()), input if f is fn1 else input * 2
            )

            # 传递张量和标量参数进行断言
            self.assertEqual(f(input, 1), input + 1)
            self.assertEqual(f(input, 1.1), input + 1)
            self.assertEqual(f(input, True), input + 1)
            self.assertEqual(f(input, input), input + 1)

    @make_test
    def test_len_constant_misc_iterables(x):
        # 计算元组 (1, 2, 3) 的长度
        a = len((1, 2, 3))
        # 计算字符串 "test str" 的长度
        b = len("test str")
        # 计算 a 和 b 的和
        c = a + b
        # 返回 torch.add(x, c) 的结果
        return torch.add(x, c)

    @make_test
    def test_dict_kwargs(x):
        # 创建字典 z，包含键值对 text_embed=x+1, other=x+2
        z = dict(text_embed=x + 1, other=x + 2)
        # 返回字典 z
        return z

    @make_test
    def test_ordered_dict_kwargs(x):
        # 创建有序字典 z，包含键 sample 和对应值 torch.ones(10)
        z = collections.OrderedDict(sample=torch.ones(10))
        # 返回有序字典 z
        return z

    @make_test
    def test_custom_dict_kwargs(x):
        # 创建自定义字典子类对象 z，包含键 sample 和对应值 torch.ones(10)
        z = CustomDictSubclass(sample=torch.ones(10))
        # 返回自定义字典子类对象 z
        return z

    @make_test
    def test_float(x):
        # 定义浮点数 y，注释说明忽略特定警告 UP018
        y = float(1.2)  # noqa: UP018
        # 将浮点数 y 和字符串 "1.2" 转换为浮点数后相加，并赋值给 y
        y += float("1.2")
        # 返回 torch.add(x, y) 的结果
        return torch.add(x, y)

    @make_test
    def test_is_floating_point(x):
        # 将 x 加 1，并判断结果是否为浮点数
        y = x + 1
        # 检查 y 是否为浮点数，同时检查输入参数 input=y 是否为浮点数
        return torch.is_floating_point(y), torch.is_floating_point(input=y)

    @make_test
    def test_dtype(x):
        # 检查输入张量 x 的数据类型是否为 torch.float32
        if x.dtype == torch.float32:
            return x + 1

    @make_test
    def test_get_default_dtype(x):
        # 检查输入张量 x 的数据类型是否为 torch 默认的数据类型
        if x.dtype == torch.get_default_dtype():
            return x + 1
        else:
            return x - 1

    @make_test
    def test_get_autocast_gpu_dtype(x):
        # 获取 torch 自动类型转换 GPU 数据类型
        dtype = torch.get_autocast_gpu_dtype()
        # 返回 x 张量转换为 dtype 类型的结果
        return x.type(dtype)
    @make_test
    # 使用装饰器将下面的函数标记为测试函数
    def test_is_any_autocast_enabled(x):
        # 检查是否启用了任何自动类型转换
        if torch._C._is_any_autocast_enabled():
            # 如果启用了自动类型转换，返回 x + 1
            return x + 1
        else:
            # 如果未启用自动类型转换，返回 x - 1
            return x - 1

    @make_test
    # 使用装饰器将下面的函数标记为测试函数
    def test_list_compare_polyfill(x):
        # 遍历多个元组，每个元组包含两个列表和一个浮点数
        for a, b, c in [
            [(1, 2, 3), (1, 2, 3), 7.77],
            [(1, 4, 3), (1, 2, 3), 3.33],
            [(1, 2), (1, 2, 3), 5.55],
            [(1, 2, 3), (1, 2), 11.11],
            [(1, -1, 3), (1, 2, 3), 13.33],
        ]:
            # 比较列表 a 和 b 是否相等
            if a != b:
                x += 1 * c
            # 如果列表 a 和 b 相等
            if a == b:
                x += 2 * c
            # 如果列表 a 小于 b
            if a < b:
                x += 4 * c
            # 如果列表 a 大于 b
            if a > b:
                x += 8 * c
            # 如果列表 a 小于或等于 b
            if a <= b:
                x += 16 * c
            # 如果列表 a 大于或等于 b
            if a >= b:
                x += 32 * c
        # 返回累积结果 x
        return x

    @make_test
    # 使用装饰器将下面的函数标记为测试函数
    def test_promote_types(x):
        # 检查 x 的数据类型是否为 torch.int32 和 torch.float32 的升级类型
        if x.dtype == torch.promote_types(torch.int32, torch.float32):
            # 如果是，则返回 x + 1
            return x + 1
        else:
            # 否则返回 x - 1
            return x - 1

    @make_test
    # 使用装饰器将下面的函数标记为测试函数
    def test_cublas_allow_tf32(x):
        # 检查是否允许使用 TF32（TensorFloat-32）加速
        if torch.backends.cuda.matmul.allow_tf32:
            # 如果允许，则返回 x.sin() + 1
            return x.sin() + 1
        else:
            # 如果不允许，则返回 x.cos() - 1
            return x.cos() - 1

    @make_test
    # 使用装饰器将下面的函数标记为测试函数
    def test_get_calculate_correct_fan(x):
        # 获取张量 x 的“fan_in”属性的正确扇入值
        fan_in = torch.nn.init._calculate_correct_fan(x, "fan_in")
        # 返回 x 加上扇入值
        return x + fan_in

    @make_test
    # 使用装饰器将下面的函数标记为测试函数
    def test_is_complex(x):
        # 检查张量 x 是否为复数类型
        if torch.is_complex(x):
            # 如果是复数，则返回 x + 1
            return x + 1
        else:
            # 如果不是复数，则返回 x - 1
            return x - 1

    @make_test
    # 使用装饰器将下面的函数标记为测试函数
    def test_tensor_is_complex(x):
        # 检查张量 x 是否包含复数元素
        if x.is_complex():
            # 如果包含复数元素，则返回 x + 1
            return x + 1
        else:
            # 如果不包含复数元素，则返回 x - 1
            return x - 1

    @make_test
    # 使用装饰器将下面的函数标记为测试函数
    def test_get_privateuse1_name(x):
        # 检查私有使用1后端的名称是否为 "privateuseone"
        if torch._C._get_privateuse1_backend_name() == "privateuseone":
            # 如果是，则返回 x + 1
            return x + 1
        else:
            # 如果不是，则返回 x - 1
            return x - 1

    @make_test
    # 使用装饰器将下面的函数标记为测试函数
    def test_device(x):
        # 检查张量 x 是否不在 CUDA 设备上
        if not x.is_cuda:
            # 如果不在 CUDA 设备上，则返回 x + 1
            return x + 1

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @make_test
    # 使用装饰器将下面的函数标记为测试函数，并在没有 CUDA 的情况下跳过测试
    def test_get_device_properties_tensor_device(a):
        # 将张量 a 转移到 CUDA 设备
        x = a.to("cuda")
        # 获取 CUDA 设备上 x 的设备属性
        prop = torch.cuda.get_device_properties(x.device)
        # 如果设备主版本号为 8
        if prop.major == 8:
            # 返回 x 加上多处理器数量
            return x + prop.multi_processor_count
        # 否则返回 x 加上每个多处理器的最大线程数
        return x + prop.max_threads_per_multi_processor

    @make_test
    # 使用装饰器将下面的函数标记为测试函数
    def test_tensor_type(a, b):
        # 将张量 a 转换为 torch.float16 类型的张量 m
        m = a.to(torch.float16)
        # 返回 b 的类型转换为 m 的类型后的结果
        return b.type(m.type())

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @make_test
    # 使用装饰器将下面的函数标记为测试函数，并在没有 CUDA 的情况下跳过测试
    def test_tensor_type2(a, b):
        # 将张量 a 转移到 CUDA 设备
        m = a.to("cuda")
        # 返回 m 和 b 在相同类型下的元素求和结果
        return m + b.type(m.type())

    @make_test
    # 使用装饰器将下面的函数标记为测试函数
    def test_tensor_type3(a, b):
        # 将张量 a 转换为 torch.float16 类型的张量 m
        m = a.type(torch.HalfTensor)
        # 返回 b 的类型转换为 m 的类型后的结果
        return b.type(m.type())

    @make_test
    # 使用装饰器将下面的函数标记为测试函数
    def test_tensor_type4(a, b):
        # 将张量 a 转换为 torch.HalfTensor 类型的张量 m
        m = a.type("torch.HalfTensor")
        # 返回 b 的类型转换为 m 的类型后的结果
        return b.type(m.type())

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @make_test
    # 使用装饰器将下面的函数标记为测试函数，并在没有 CUDA 的情况下跳过测试
    def test_tensor_type5(a, b):
        # 将张量 a 转换为 torch.cuda.HalfTensor 类型的张量 m
        m = a.type(torch.cuda.HalfTensor)
        # 返回 b 的类型转换为 m 的类型后的结果
        return b.type(m.type())
    @make_test
    def test_tensor_element_size(a):
        # 检查张量元素大小是否大于1
        if a.element_size() > 1:
            # 如果大于1，则返回对张量进行加和和减法操作后的结果
            return (a + a.element_size(), a - a.element_size())
        # 否则返回对张量进行减法和加法操作后的结果
        return (a - a.element_size(), a + a.element_size())

    @make_test
    def test_ndim(x):
        # 检查张量的维度是否为2，并且使用多个方法检查维度是否为2
        if x.ndim == 2 and x.ndimension() == 2 and x.dim() == 2:
            # 如果满足条件，则返回对张量所有元素加1后的结果
            return x + 1

    @make_test
    def test_T(x):
        # 返回一个与输入张量 x 的转置张量维度相同且所有元素为1的张量
        return torch.ones_like(x.T)

    @make_test
    def test_mT(x):
        # 返回一个与输入张量 x 的转置后的主对角线张量维度相同且所有元素为1的张量
        return torch.ones_like(x.mT)

    @make_test
    def test_is_sparse(x):
        # 检查张量是否为稀疏张量
        if not x.is_sparse:
            # 如果不是稀疏张量，则返回对张量所有元素加1后的结果
            return x + 1

    @make_test
    def test_shape1(x):
        # 检查张量的第一个维度是否为10
        if x.shape[0] == 10:
            # 如果满足条件，则返回对张量所有元素加1后的结果
            return x + 1

    @make_test
    def test_shape2(x):
        # 检查张量的第二个维度是否为10
        if x.size(1) == 10:
            # 如果满足条件，则返回对张量所有元素加1后的结果
            return x + 1

    @make_test
    def test_del(a, b):
        # 对张量 a 进行加法操作，并赋值给变量 c
        c = a + 1
        # 对变量 c 进行加法操作，并赋值给变量 d
        d = c + 2
        # 删除变量 c 和 a
        del c, a
        # 返回变量 b 和 d 的加法结果
        return b + d

    @make_test
    def test_chunks1(x):
        # 定义块大小为5
        chunk_size = 5
        # 断言张量 x 的第一个维度能被块大小整除
        assert x.shape[0] % chunk_size == 0
        # 断言张量 x 的第一个维度除以块大小等于2
        assert x.shape[0] // chunk_size == 2
        # 返回张量 x 前 chunk_size 个元素与后 chunk_size 个元素的差
        return x[:chunk_size] - x[chunk_size:]

    @make_test
    def test_import1(x, y):
        # 导入 torch 模块并从中导入 sub 函数
        import torch
        from torch import sub

        # 返回对张量 x 和 y 先加和再减法后的结果
        return sub(torch.add(x, y), y)

    @make_test
    def test_return_dict(x, y):
        # 构建一个包含计算结果和布尔值的列表 z
        z = [x + y, y, False]
        # 返回包含多个键值对的字典
        return {"x": x, "z": z, "a": x, "b": z, "c": x}

    @make_test
    def test_return_dict2(x, y):
        # 创建临时字典并初始化键值对 x
        tmp = {"x": x}
        # 向临时字典中添加键为 z，值为包含 x+y 和 y 的列表
        tmp["z"] = [x + y, y]
        # 向临时字典中添加键为 y，值为 y，并向 z 列表中追加 False
        tmp["y"] = y
        tmp["z"].append(False)
        # 返回修改后的临时字典
        return tmp

    @make_test
    def test_funcdef_closure(x, y):
        # 对 x 进行加法操作，并赋值给 x
        x = x + y + 1.0

        # 定义内部函数 inner，并使用 nonlocal 关键字声明 x 和 y
        def inner(z):
            nonlocal x, y
            # 对 y 进行复杂的加法操作
            y = x + z + 20.0
            # 对 x 进行复杂的加法操作
            x = y + z + 10.0

        # 调用内部函数 inner，并传入参数 2.0 和 3.0
        inner(2.0)
        inner(3.0)

        # 返回修改后的 x 和 y
        return x, y

    @make_test
    def test_module_constant(x, y):
        # 对 x 和 y 进行加法操作，并赋值给 r
        r = x + y
        # 对范围内的三个数进行迭代，并对 r 进行除法操作
        for i in range(torch._dynamo.testing.three):
            r = r / y
        # 返回最终计算结果 r
        return r

    @make_test
    def test_inline_softmax(x, y):
        # 对 x 和 y 进行加法和乘法操作，并通过 Softmax 函数进行处理
        return torch.nn.Softmax(dim=-1)(x + y * 2)

    @make_test
    def test_dtype_compare(a, b):
        # 检查张量 a 的数据类型是否为 float16
        if a.dtype == torch.float16:
            # 如果满足条件，则返回对张量 a 所有元素加10后的结果
            return a + 10
        # 检查张量 a 的数据类型是否为 float32
        if a.dtype == torch.float32:
            # 如果满足条件，则返回对张量 a 和 b 的乘法结果再减去32后的结果
            return a - b * 32

    @make_test
    def test_build_list_unpack(a, b):
        # 创建生成器 it1 和 it2，并将其合并为一个张量
        it1 = (x + 1 for x in (a, b))
        it2 = (x - 1 for x in (a, b))
        return torch.cat([*it1, *it2], dim=-1)

    @make_test
    def test_tensor_len(a, b):
        # 返回张量 a 和 b 所有元素加和，以及它们各自的长度
        return a + b + len(a) + b.__len__()

    @make_test
    def test_pop(a, b):
        # 创建列表 ll，并初始化为包含 a 和 b 的列表
        ll = [a, b]
        # 向列表 ll 中追加 a+1
        ll.append(a + 1)
        # 向列表 ll 中扩展 [b+2, a+b] 的列表
        ll.extend(
            [
                b + 2,
                a + b,
            ]
        )
        # 弹出列表 ll 中的最后一个元素和第一个元素，以及任意位置的元素
        ll.pop(-1)
        ll.pop(0)
        ll.pop()
        # 将列表 ll 的两个元素分别赋值给 v1 和 v2
        v1, v2 = ll
        # 返回 v1 和 v2 的差
        return v1 - v2

    @make_test
    def test_list_convert(a, b):
        # 创建列表 ll，并初始化为包含 a+2 和 b 的列表
        ll = [a + 2, b]
        # 将列表 ll 转换为元组并赋值给 ll
        ll = tuple(ll)
        # 对 b 进行加法操作，并赋值给 tmp
        tmp = b + 3
        # 将元组 ll 转换为列表并赋值给 ll
        ll = list(ll)
        # 将列表 ll 的两个元素分别赋值给 v1 和 v2
        v1
    # 定义一个测试函数，接受两个参数 a 和 b
    def test_list_add(a, b):
        # 创建一个元组 l1，包含参数 a 和 b 的值
        l1 = (a, b)
        # 创建一个空元组 l2，作为字节码中的 LOAD_CONST
        l2 = ()
        # 将 l1 和 l2 连接成一个新元组 l3
        l3 = l1 + l2
        # 返回 l3 中两个元素的和
        return l3[0] + l3[1]
    
    # 使用装饰器 make_test 对下面的函数进行测试装饰
    @make_test
    def test_list_index_with_constant_tensor(a, b):
        # 创建一个列表 l1，包含参数 a、b、a+1、b+1 的值
        l1 = [a, b, a + 1, b + 1]
        # 返回索引为常量张量值 2 的 l1 中的元素
        return l1[torch.as_tensor(2)]
    
    # 使用装饰器 make_test 对下面的函数进行测试装饰
    @make_test
    def test_startswith(a, b):
        # 计算字符串 a 和 b 的连接结果
        x = a + b
        # 如果字符串 "foobar" 以 "foo" 开头，并且常量 constant3.__module__ 中包含字符串 "test"
        if "foobar".startswith("foo") and "test" in constant3.__module__:
            # 将 x 增加 1
            x = x + 1
        # 返回 x 的最终值
        return x
    
    # 使用装饰器 make_test 对下面的函数进行测试装饰
    @make_test
    def test_dict_ops(a, b):
        # 创建一个临时字典 tmp，包含键 "a" 和 "b" 对应的值
        tmp = {"a": a + 1, "b": b + 2}
        # 断言键 "zzz" 不在 tmp 中
        assert tmp.get("zzz") is None
        # 计算 v 的值，包括 tmp 中键 "b" 的值，tmp 中键 "a" 的值，以及 tmp 中键 "missing" 的默认值
        v = tmp.pop("b") + tmp.get("a") + tmp.get("missing", 3) + tmp.pop("missing", 4)
        # 更新 tmp，添加键值对 {"d": 3}
        tmp.update({"d": 3})
        # 添加键 "c" 到 tmp 中，其值为 v + tmp["d"]
        tmp["c"] = v + tmp["d"]
        # 如果 tmp 中包含键 "c" 并且不包含键 "missing"
        if "c" in tmp and "missing" not in tmp:
            # 返回 tmp["c"] - tmp["a"] + tmp 的长度
            return tmp["c"] - tmp["a"] + len(tmp)
    
    # 使用装饰器 make_test 对下面的函数进行测试装饰
    @make_test
    def test_inline_jit__unwrap_optional(x):
        # 如果 x 是 None，则返回一个 2x2 的张量全为 1
        if torch.jit._unwrap_optional(x) is None:
            return torch.ones(2, 2)
        # 否则返回 x 的正弦值
        return x.sin()
    
    # 定义一个测试方法，接受 self 参数
    def test_dict_param_keys(self):
        # 创建一个 torch 的参数对象 a_param，其值为一个 4x4 的张量全为 1
        a_param = torch.nn.Parameter(torch.ones([4, 4]))
    
        # 定义一个内部函数 fn，接受参数 a
        def fn(a):
            # 创建一个字典 tmp，包含键 "a" 和 a_param 的值
            tmp = {"a": a, a_param: 3}
            # 返回 tmp 中键 "a" 和 tmp[a_param] 的和
            return tmp["a"] + tmp[a_param]
    
        # 使用 make_test 装饰 fn 函数，并将其赋值给 test
        test = make_test(fn)
        # 调用 test 方法，将 self 作为参数传递进去
        test(self)
    
    # 定义一个辅助方法 _test_default_dict_helper，接受 self 和 factory 参数
    def _test_default_dict_helper(self, factory):
        # 创建一个默认字典 dd，工厂方法为 factory
        dd = collections.defaultdict(factory)
        # 创建一个 torch 的参数对象 param，其值为一个 2x2 的张量全为 1
        param = torch.nn.Parameter(torch.ones([2, 2]))
    
        # 定义一个内部函数 fn，接受参数 x
        def fn(x):
            # 更新 dd 中键 "a" 的值为 x+1
            dd["a"] = x + 1
            # 更新 dd 中键 param 的值为 123
            dd[param] = 123
            # 更新 dd 中键 "c" 的值为 x*2
            dd["c"] = x * 2
            # 返回 dd 中键 "b" 的值和整个 dd 字典
            return dd["b"], dd
    
        # 创建一个 10x10 的随机张量 x
        x = torch.randn(10, 10)
        # 调用 fn 方法并返回结果 ref
        ref = fn(x)
        # 使用 torch._dynamo.optimize_assert("eager") 优化 fn 方法，并将其赋值给 opt_fn
        opt_fn = torch._dynamo.optimize_assert("eager")(fn)
        # 调用优化后的 opt_fn 方法，并返回结果 res
        res = opt_fn(x)
    
        # 断言 ref[0] 与 res[0] 相同
        self.assertTrue(same(ref[0], res[0]))
        # 断言 ref[1]["a"] 与 res[1]["a"] 相同
        self.assertTrue(same(ref[1]["a"], res[1]["a"]))
        # 断言 ref[1]["c"] 与 res[1]["c"] 相同
        self.assertTrue(same(ref[1]["c"], res[1]["c"]))
        # 断言 ref[1][param] 与 res[1][param] 相同
        self.assertTrue(same(ref[1][param], res[1][param]))
    
    # 定义一个测试方法 test_default_dict，接受 self 参数
    def test_default_dict(self):
        # 调用 _test_default_dict_helper 方法，并传递 dict 作为工厂方法
        self._test_default_dict_helper(dict)
    
    # 定义一个测试方法 test_default_dict_lambda，接受 self 参数
    def test_default_dict_lambda(self):
        # 创建一个 lambda 函数作为工厂方法，并调用 _test_default_dict_helper 方法
        self._test_default_dict_helper(lambda: dict())
    
    # 定义一个测试方法 test_default_dict_closure，接受 self 参数
    def test_default_dict_closure(self):
        # 定义一个工厂函数 factory，并调用 _test_default_dict_helper 方法
        def factory():
            return dict()
    
        self._test_default_dict_helper(factory)
    @make_test
    def test_default_dict_constr(self):
        # 创建一个参数为全1的 torch.nn.Parameter 对象
        param = torch.nn.Parameter(torch.ones([2, 2]))

        def fn(x):
            # 创建一个默认字典，值为一个空字典的 lambda 函数
            dd = collections.defaultdict(lambda: dict())
            # 设置 dd 中的键 'a' 对应的值为 x + 1
            dd["a"] = x + 1
            # 将 param 对象作为键，值为 123 存入 dd
            dd[param] = 123
            # 设置 dd 中的键 'c' 对应的值为 x * 2
            dd["c"] = x * 2
            # 使用字典的 update 方法更新 dd，添加键值对 'b': x * 3
            dd.update({"b": x * 3})
            # 使用 update 方法更新 dd，添加键值对 'd': x - 2 和 'e': x + 2
            dd.update([["d", x - 2], ("e", x + 2)])
            # 使用 update 方法通过 zip 函数更新 dd，添加键值对 'a': x + 3 和 'b': x + 4
            dd.update(zip("ab", [x + 3, x + 4]))
            # 返回 dd 中键 'b' 对应的值和整个 dd 字典
            return dd["b"], dd

        # 创建一个 10x10 的随机张量 x
        x = torch.randn(10, 10)
        # 调用 fn 函数并保存结果作为参考值 ref
        ref = fn(x)
        # 使用 torch._dynamo.optimize_assert("eager") 优化 fn 函数，并将其作为 opt_fn
        opt_fn = torch._dynamo.optimize_assert("eager")(fn)
        # 使用 x 调用优化后的函数 opt_fn，保存结果作为 res
        res = opt_fn(x)

        # 使用 assertTrue 进行断言，比较 ref 和 res 中键 'b' 的值是否相同
        self.assertTrue(same(ref[0], res[0]))
        # 使用 assertTrue 进行断言，比较 ref 和 res 中字典的键 'a' 对应的值是否相同
        self.assertTrue(same(ref[1]["a"], res[1]["a"]))
        # 使用 assertTrue 进行断言，比较 ref 和 res 中字典的键 'b' 对应的值是否相同
        self.assertTrue(same(ref[1]["b"], res[1]["b"]))
        # 使用 assertTrue 进行断言，比较 ref 和 res 中字典的键 'c' 对应的值是否相同
        self.assertTrue(same(ref[1]["c"], res[1]["c"]))
        # 使用 assertTrue 进行断言，比较 ref 和 res 中字典的键 'd' 对应的值是否相同
        self.assertTrue(same(ref[1]["d"], res[1]["d"]))
        # 使用 assertTrue 进行断言，比较 ref 和 res 中字典的键 'e' 对应的值是否相同
        self.assertTrue(same(ref[1]["e"], res[1]["e"]))
        # 使用 assertTrue 进行断言，比较 ref 和 res 中 param 对应的值是否相同
        self.assertTrue(same(ref[1][param], res[1][param]))
    @make_test
    def test_dict_fromkeys(x, y):
        # 创建一个列表 lst 包含字符串 "a" 和 "b"
        lst = ["a", "b"]
        # 使用 dict.fromkeys() 方法将列表中的元素作为键，值为 None，创建字典 d
        d = dict.fromkeys(lst)
        # 使用 dict.fromkeys() 方法将 d 的键作为键，值为 x + 1，创建字典 d1
        d1 = dict.fromkeys(d, x + 1)
        # 使用 collections.defaultdict.fromkeys() 方法将 d1 的迭代器元素作为键，值为 x - 2，创建默认字典 d2
        d2 = collections.defaultdict.fromkeys(iter(d1), x - 2)
        # 使用 collections.OrderedDict.fromkeys() 方法将元组 lst 的元素作为键，值为 y，创建有序字典 d3
        d3 = collections.OrderedDict.fromkeys(tuple(lst), value=y)
        # 返回根据特定规则计算的结果
        return d1["a"] * d2["b"] + d2["a"] + d1["b"] + d3["a"] + d3["b"] + 1

    @make_test
    def test_dict_copy(x):
        # 创建一个包含元组的列表 my_list
        my_list = [("a", x), ("b", x + 1), ("c", x + 2)]
        # 使用 dict() 构造函数根据 my_list 创建字典 d1
        d1 = dict(my_list)
        # 将 d1 中键为 "a" 的值更新为 x + 10
        d1["a"] = x + 10
        # 使用 d1 的 .copy() 方法创建字典 d2
        d2 = d1.copy()
        # 将 d2 中键为 "a" 的值更新为 x - 5，键为 "b" 的值更新为 x + 3
        d2["a"] = x - 5
        d2["b"] = x + 3
        # 使用 collections.OrderedDict() 构造函数根据 my_list 创建有序字典 d3
        d3 = collections.OrderedDict(my_list)
        # 将 d3 中键为 "c" 的值更新为 x + 20
        d3["c"] = x + 20
        # 使用 d3 的 .copy() 方法创建有序字典 d4
        d4 = d3.copy()
        # 将 d4 中键为 "c" 的值更新为 x - 10
        d4["c"] = x - 10
        # 返回根据特定规则计算的结果
        return d1["a"] * d2["a"] + d2["b"] + d3["c"] * d4["c"] + 1

    @make_test
    def test_dict_update(x, y, z):
        # 创建一个包含键值对 "a": x, "b": y 的字典 d
        d = {"a": x, "b": y}
        # 使用 .update() 方法更新键为 "a" 的值为 y - 1
        d.update({"a": y - 1})
        # 使用 .update() 方法通过列表形式更新键值对 [("b", z + 1), ["c", z]]
        d.update([("b", z + 1), ["c", z]])
        # 使用 .update() 方法通过 zip 对象更新键值对 "a" 和 "b"，对应的值分别为 z + 3 和 y + 2
        d.update(zip("ab", [z + 3, y + 2]))

        # 使用 collections.OrderedDict() 构造函数创建有序字典 od，包含键值对 "a": x * 3, "b": y + 2
        od = collections.OrderedDict(a=x * 3, b=y + 2)
        # 使用 .update() 方法更新键为 "a" 的值为 y + 5
        od.update({"a": y + 5})
        # 使用 .update() 方法通过列表形式更新键值对 [["b", z + 6], ("c", z - 7)]
        od.update([["b", z + 6], ("c", z - 7)])
        # 使用 .update() 方法通过 zip 对象更新键值对 "a" 和 "b"，对应的值分别为 z - 3 和 x + 2
        od.update(zip("ab", [z - 3, x + 2]))
        # 返回根据特定规则计算的结果
        return d["a"] * od["a"] + od["c"] + d["b"] + od["b"] * d["c"]

    @make_test
    def test_min_max(a, b):
        # 计算 a 和 b 的和，赋值给变量 c
        c = a + b
        # 对 a 和 b 分别调用 .sum() 方法，结果重新赋给变量 a 和 b
        a = a.sum()
        b = b.sum()
        # 将 a 限制在 0 和 1 之间，最小值为 0
        a = min(max(a, 0), 1)
        # 将 b 限制在 0 和 1 之间，最大值为 1
        b = max(0, min(1, b))
        # 返回经过特定计算的结果
        return max(a, b) - min(a, b) + c

    @make_test
    def test_symbool_to_int(x):
        # 检查 x.size() 中等于 -1 的元素个数，如果为 0 则返回 x + 1，否则返回 x - 1
        if sum(s == -1 for s in x.size()) == 0:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_map_sum(a, b, c, d):
        # 使用 map() 函数将列表中的每个元素加 1，并返回它们的和
        return sum(map(lambda x: x + 1, [a, b, c, d]))

    @make_test
    def test_sum(a, b, c, d):
        # 返回列表中所有元素的和
        return sum([a, b, c, d])

    @make_test
    def test_sum_with_start_arg(a, b, c, d):
        # 返回列表中所有元素的和，并加上额外的起始参数 a
        return sum([b, c, d], a)

    @make_test
    def test_sum_with_start_kwarg(a, b, c, d):
        # 返回列表中所有元素的和，并加上额外的起始参数 start=a
        return sum([b, c, d], start=a)

    @make_test(expected_frame_count=0)
    def test_sum_shortcut():
        # 返回列表中所有元素的和，不生成测试帧
        return sum([0, 1.0, 2, 3.0])

    @make_test(expected_frame_count=0)
    def test_sum_shortcut_with_start_arg():
        # 返回列表中所有元素的和，并加上额外的起始参数 -10，不生成测试帧
        return sum([0, 1.0, 2, 3.0], -10)

    @make_test(expected_frame_count=0)
    def test_sum_shortcut_with_start_kwarg():
        # 返回列表中所有元素的和，并加上额外的起始参数 start=-10，不生成测试帧
        return sum([0, 1.0, 2, 3.0], start=-10)

    @make_test
    def test_reduce(a, b, c, d):
        # 使用 functools.reduce() 函数将列表中所有元素依次累加
        return functools.reduce(operator.add, [a, b, c, d])

    @make_test
    def test_reduce_with_initial(a, b, c, d):
        # 使用 functools.reduce() 函数将列表中所有元素依次累加，并使用 a 作为初始值
        return functools.reduce(operator.add, [b, c, d], a)

    @make_test(expected_frame_count=0)
    def test_reduce_with_single(x):
        # 使用 functools.reduce() 函数将列表中唯一的元素作为结果，不生成测试帧
        return functools.reduce(lambda a, b: (a, b), [x])

    @make_test(expected_frame_count=0)
    def test_reduce_with_single_with_initial(x, y):
        # 使用 functools.reduce() 函数将列表中唯一的元素作为结果，并使用 x 作为初始值，不生成测试帧
        return functools.reduce(lambda a, b: (a, b), [y], x)
    # 定义一个函数，使用 functools.reduce 将 lambda 函数应用到列表 x 上，初始值为 None
    def test_reduce_with_none_initial(x):
        return functools.reduce(lambda a, b: (a, b), [x], None)

    # 用 make_test 装饰器定义一个测试函数，测试元组包含关系
    @make_test
    def test_tuple_contains(a, b):
        # 定义三个字符串变量
        v1 = "a"
        v2 = "b"
        v3 = "c"
        # 创建两个元组
        vals1 = (v1, v2, v3)
        vals2 = ("d", "e", "f")
        # 如果 "a" 在 vals1 中且 "b" 不在 vals2 中，则返回 a + b，否则返回 a - b
        if "a" in vals1 and "b" not in vals2:
            return a + b
        else:
            return a - b

    # 如果 Python 版本小于 3.9，则跳过此测试函数
    @unittest.skipIf(
        sys.version_info < (3, 9),
        "SET_UPDATE was added at Python 3.9",
    )
    @make_test
    def test_set_update_bytecode(x):
        # 创建一个集合变量 var
        var = {"apple", "banana", "cherry"}
        # 如果 var 是集合类型，则返回 x + 1，否则返回 x - 1
        if isinstance(var, set):
            return x + 1
        else:
            return x - 1

    # 如果 Python 版本小于 3.9，则跳过此测试函数
    @unittest.skipIf(
        sys.version_info < (3, 9),
        "SET_UPDATE was added at Python 3.9",
    )
    @make_test
    def test_set_update_list_with_duplicated_items(x):
        # 创建两个包含重复项的列表
        list1 = ["apple", "banana", "apple"]
        list2 = ["orange", "banana"]
        # 如果合并后的集合长度为 3，则返回 x + 1，否则返回 x - 1
        if len({*list1, *list2}) == 3:
            return x + 1
        else:
            return x - 1

    # 使用 make_test 装饰器定义一个测试函数，测试集合包含关系
    @make_test
    def test_set_contains(a, b):
        # 创建一个包含元素的集合 vals
        vals = set(["a", "b", "c"])
        # 如果 "a" 在 vals 中，则 x = a + b，否则 x = a - b
        if "a" in vals:
            x = a + b
        else:
            x = a - b
        # 如果 "d" 在 vals 中，则 y = a + b，否则 y = a - b
        if "d" in vals:
            y = a + b
        else:
            y = a - b
        # 返回 x 和 y 的值作为元组
        return x, y

    # 定义一个方法测试类中的方法
    def test_set_isdisjoint(self):
        # 创建两个集合 x 和 y
        x = {"apple", "banana", "cherry"}
        y = {"google", "microsoft", "apple"}

        # 定义一个内部函数 fn(a)，如果 x 和 y 是不相交的，则返回 a + 1，否则返回 a - 1
        def fn(a):
            if x.isdisjoint(y):
                return a + 1
            else:
                return a - 1

        # 使用 make_test 装饰器创建一个测试
        test = make_test(fn)
        test(self)

    # 使用 make_test 装饰器定义一个测试函数，测试元组的增强赋值操作
    @make_test
    def test_tuple_iadd(a, b):
        # 创建一个元组 output
        output = (a, b)
        # 对 output 进行增强赋值操作
        output += (a + b, a - b)
        # 返回增强赋值后的 output
        return output

    # 使用 make_test 装饰器定义一个测试函数，测试元组解包操作
    @make_test
    def test_unpack_ex1(x):
        # 创建一个元组 output
        output = (x, x + 1, x + 2, x + 3)
        # 使用解包操作，分别将 a, b, *cd 赋值给变量 a, b, cd
        a, b, *cd = output
        # 返回计算结果
        return a - b / cd[0]

    # 使用 make_test 装饰器定义一个测试函数，测试元组解包操作
    @make_test
    def test_unpack_ex2(x):
        # 创建一个元组 output
        output = (x, x + 1, x + 2, x + 3)
        # 使用解包操作，分别将 *ab, c, d 赋值给变量 ab, c, d
        *ab, c, d = output
        # 返回计算结果
        return c - d / ab[0]

    # 使用 make_test 装饰器定义一个测试函数，测试元组解包操作
    @make_test
    def test_unpack_ex3(x):
        # 创建一个元组 output
        output = (x, x + 1, x + 2, x + 3)
        # 使用解包操作，分别将 a, *bc, d 赋值给变量 a, bc, d
        a, *bc, d = output
        # 返回计算结果
        return a - d / bc[0]

    # 使用 make_test 装饰器定义一个测试函数，测试常量元组的连接操作
    @make_test
    def test_const_tuple_add1(x):
        # 创建一个元组 output
        output = (x, x + 1, x + 2, x + 3)
        # 对 output 进行空元组连接操作
        output = () + output + ()
        # 返回连接后的元组中第三和第四个元素的和
        return output[2] + output[3]

    # 使用 make_test 装饰器定义一个测试函数，测试常量元组的连接操作
    @make_test
    def test_const_tuple_add2(x):
        # 创建一个元组 output
        output = (x, x + 1, x + 2, x + 3)
        # 对 output 进行带有 None 的元组连接操作
        output = (None,) + output + (None,)
        # 返回连接后的元组中第三和第四个元素的和
        return output[2] + output[3]

    # 使用 make_test 装饰器定义一个测试函数，测试列表的真值判断
    @make_test
    def test_list_truth(a, b):
        # 创建一个临时列表 tmp
        tmp = [1, 2, 3]
        # 如果 tmp 为真，则返回 a + b，否则返回 a - b
        if tmp:
            return a + b
        else:
            return a - b

    # 使用 make_test 装饰器定义一个测试函数，测试列表的反向迭代和求和
    @make_test
    def test_list_reversed(a, b):
        # 创建一个临时列表 tmp，其中包含 a+1, a+2, a+3
        tmp = [a + 1, a + 2, a + 3]
        # 返回 a + b 和 tmp 的反向迭代中的第一个元素的和
        return a + b + next(iter(reversed(tmp)))

    # 使用 make_test 装饰器定义一个测试函数，测试列表的排序
    @make_test
    def test_list_sorted1(x):
        # 创建一个临时列表 tmp
        tmp = [1, 10, 3, 0]
        # 返回 x+1，tmp 的排序结果，tmp 的逆序排序结果
        return x + 1, sorted(tmp), sorted(tmp, reverse=True)
    @make_test
    def test_list_sorted2(x):
        # 创建一个包含元组的列表
        y = [
            ("john", "A", 8),
            ("jane", "B", 5),
            ("dave", "B", 10),
        ]
        # 返回四个元素的元组：
        # 1. x+1，即输入参数 x 值加 1
        # 2. 按默认顺序排序的列表 y
        # 3. 按元组第三个元素升序排序的列表 y
        # 4. 按元组第三个元素降序排序的列表 y
        return (
            x + 1,
            sorted(y),
            sorted(y, key=lambda student: student[2]),
            sorted(y, key=lambda student: student[2], reverse=True),
        )

    @make_test
    def test_tuple_sorted(x):
        # 创建一个元组 tmp 包含 (1, 10, 3, 0)
        tmp = (1, 10, 3, 0)
        # 返回三个元素的元组：
        # 1. x+1，即输入参数 x 值加 1
        # 2. 元组 tmp 的升序排序结果
        # 3. 元组 tmp 的降序排序结果
        return x + 1, sorted(tmp), sorted(tmp, reverse=True)

    @make_test
    def test_dict_sorted(x):
        # 创建一个字典 tmp 包含 {1: "D", 10: "B", 3: "E", 0: "F"}
        tmp = {1: "D", 10: "B", 3: "E", 0: "F"}
        # 返回三个元素的元组：
        # 1. x+1，即输入参数 x 值加 1
        # 2. 字典 tmp 键的升序排序结果
        # 3. 字典 tmp 键的降序排序结果
        return x + 1, sorted(tmp), sorted(tmp, reverse=True)

    @make_test
    def test_list_clear(a, b):
        # 创建一个列表 tmp，包含 a+1 和 a+2
        tmp = [a + 1, a + 2]
        # 清空列表 tmp
        tmp.clear()
        # 向空列表 tmp 添加 a+b 的值
        tmp.append(a + b)
        return tmp

    @make_test
    def test_not_list(a):
        # 返回一个布尔值，表示 [a+1] 是否为假（即空列表为假）
        return not [a + 1]

    @make_test
    def test_islice_chain(a, b):
        # 创建两个列表 tmp1 和 tmp2，分别包含 a+1, a+2 和 a+3, a+4
        tmp1 = [a + 1, a + 2]
        tmp2 = [a + 3, a + 4]
        # 使用 itertools.chain 将 tmp1 和 tmp2 合并，并取出索引为 1 到 3 的元素
        a, b = list(itertools.islice(itertools.chain(tmp1, tmp2), 1, 3))
        # 使用 itertools.islice 取出 tmp1 中索引为 1 到末尾的元素中的第一个元素
        c = next(itertools.islice(tmp1, 1, None))
        # 返回计算结果 a - b / c
        return a - b / c

    @make_test
    def test_namedtuple(a, b):
        # 创建一个命名元组类型 mytuple，包含字段 "x", "y", "xy"
        mytuple = collections.namedtuple("mytuple", ["x", "y", "xy"])
        # 创建一个命名元组 tmp，包含 a, b, a+b 的值
        tmp = mytuple(a, b, a + b)
        # 返回一个元组，包含 tmp.x, tmp[1]（即 tmp.y）, tmp.xy+b 的值
        return mytuple(tmp.x, tmp[1], tmp.xy + b)

    @make_test
    def test_namedtuple_defaults(a, b):
        # 创建一个带默认值的命名元组类型 mytuple，包含字段 "x", "y", "xy"
        mytuple = collections.namedtuple(
            "mytuple", ["x", "y", "xy"], defaults=(None, 1, None)
        )
        # 创建一个命名元组 tmp，指定字段 x=a, xy=b 的值
        tmp = mytuple(a, xy=b)
        # 返回一个元组，包含 tmp.x, tmp[1]（即 tmp.y）, tmp.xy+b 的值
        return mytuple(tmp.x, tmp[1], tmp.xy + b)

    class MyNamedTuple(NamedTuple):
        # 定义一个类 MyNamedTuple，继承自 NamedTuple
        first: torch.Tensor
        second: torch.Tensor

        def add(self) -> torch.Tensor:
            # 返回两个张量 first 和 second 的和
            return self.first + self.second

        @staticmethod
        def static_method() -> int:
            # 返回整数 1
            return 1

        @classmethod
        def class_method(cls) -> str:
            # 返回类的名称
            return cls.__name__

    @make_test
    def test_namedtuple_user_methods(a, b):
        # 创建一个 MyNamedTuple 的实例 mytuple，包含 a 和 b 作为参数
        mytuple = FunctionTests.MyNamedTuple(a, b)
        # 返回一个元组，包含 mytuple.add() 的返回值，mytuple.static_method() 的返回值，mytuple.class_method() 的返回值
        return mytuple.add(), mytuple.static_method(), mytuple.class_method()

    @make_test
    def test_namedtuple_hasattr(a, b):
        # 创建一个 MyNamedTuple 的实例 mytuple，包含 a 和 b 作为参数
        mytuple = FunctionTests.MyNamedTuple(a, b)

        # 定义一个函数 isinstance_namedtuple，判断对象是否是元组且具有 _asdict 和 _fields 属性
        def isinstance_namedtuple(obj) -> bool:
            return (
                isinstance(obj, tuple)
                and hasattr(obj, "_asdict")
                and hasattr(obj, "_fields")
            )

        # 如果 mytuple 是一个具有 _asdict 和 _fields 属性的元组，返回 a+b；否则返回 a-b
        if isinstance_namedtuple(mytuple):
            return a + b
        else:
            return a - b

    @make_test
    def test_torch_size_hasattr(x):
        # 如果 x.shape 具有 _fields 属性，返回 x+1；否则返回 x-1
        if hasattr(x.shape, "_fields"):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_is_quantized(a, b):
        # 如果张量 a 没有被量化，返回 a+b
        if not a.is_quantized:
            return a + b

    @make_test
    def test_fstrings1(a, b):
        # 创建一个浮点数 x=1.229
        x = 1.229
        # 创建一个格式化字符串 tmp，保留小数点后两位
        tmp = f"{x:.2f} bar"
        # 如果 tmp 以 "1.23" 开头，返回 a+b
        if tmp.startswith("1.23"):
            return a + b

    # https://github.com/pytorch/pytorch/issues/103602
    # 标记这是一个预期会失败的动态测试用例
    @expectedFailureDynamic
    # 使用装饰器创建一个测试函数
    @make_test
    def test_fstrings2(x):
        # 使用 f-string 格式化字符串
        tmp = f"{x.shape[0]} bar"
        # 如果字符串 tmp 以 "10" 开头，返回 x + 1
        if tmp.startswith("10"):
            return x + 1

    # 使用装饰器创建一个测试函数
    @make_test
    def test_fstrings3(x):
        # 使用 f-string 获取对象 x 的类名
        tmp = f"{x.__class__.__name__} foo"
        # 如果字符串 tmp 以 "Tensor" 开头，返回 x + 1
        if tmp.startswith("Tensor"):
            return x + 1

    # 使用装饰器创建一个测试函数
    @make_test
    def test_tensor_new_with_size(x):
        # 创建一个大小为 (5, 8) 的随机张量 y
        y = torch.rand(5, 8)
        # 使用 x 的类型创建一个和 y 相同大小的张量 z
        z = x.new(y.size())
        # 断言 z 的大小与 y 的大小相同
        assert z.size() == y.size()

    # 使用装饰器创建一个测试函数
    @make_test
    def test_tensor_new_with_shape(x):
        # 创建一个大小为 (5, 8) 的随机张量 y
        y = torch.rand(5, 8)
        # 使用 x 的类型创建一个和 y 形状相同的张量 z
        z = x.new(y.shape)
        # 断言 z 的大小与 y 的大小相同
        assert z.size() == y.size()

    # 使用装饰器创建一个测试函数
    @make_test
    def test_jit_annotate(x):
        # 使用 torch.jit.annotate 标注 x 为任意类型 Any
        y = torch.jit.annotate(Any, x + 1)
        # 返回 y + 2
        return y + 2

    # 使用装饰器创建一个测试函数
    @make_test
    def test_is_contiguous_memory_format(tensor):
        # 如果正在进行 Torch 脚本化，则返回 None
        if torch.jit.is_scripting():
            return None
        # 如果张量 tensor 是连续的，使用 torch.contiguous_format 内存格式，则返回 tensor + 1
        elif tensor.is_contiguous(memory_format=torch.contiguous_format):
            return tensor + 1

    # 定义一个测试方法
    def test_is_contiguous_frame_counts(self):
        # 准备测试数据
        data = [
            torch.rand(10),
            torch.rand(2, 3, 32, 32),
            torch.rand(2, 3, 32, 32).contiguous(memory_format=torch.channels_last),
            torch.rand(10)[::2],
            torch.rand(12),
            torch.rand(2, 3, 24, 24).contiguous(memory_format=torch.channels_last),
            torch.rand(50)[::2],
            torch.rand(2, 3, 32, 32)[:, :, 2:-2, 3:-3],
        ]
        # 在静态形状模式下，预期重新编译的帧数列表
        expected_frame_counts_static = [1, 2, 3, 4, 5, 6, 7, 8]
        # 在动态形状模式下，预期重新编译的帧数列表
        expected_frame_counts_dynamic = [1, 2, 3, 4, 4, 4, 4, 5]
        # 根据当前运行模式选择正确的预期帧数列表
        expected_frame_counts = ifdynstaticdefault(
            expected_frame_counts_static, expected_frame_counts_dynamic
        )
        # 是否在动态模式下执行
        dynamic = ifdynstaticdefault(False, True)

        # 定义一个函数 func，根据输入张量 x 的连续性返回不同的操作结果
        def func(x):
            if x.is_contiguous():
                return x + 1
            elif x.is_contiguous(memory_format=torch.channels_last):
                return x + 2
            else:
                return x + 3

        # 创建一个计数器 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用动态模式参数动态优化函数 func
        cfunc = torch._dynamo.optimize_assert(cnt, dynamic=dynamic)(func)

        # 断言编译帧数为零
        assert cnt.frame_count == 0
        # 遍历测试数据并进行测试
        for i, x in enumerate(data):
            # 期望的函数 func 输出
            expected = func(x)
            # 实际的优化后函数 cfunc 输出
            output = cfunc(x)
            # 使用 assertTrue 检查输出与期望值是否相同
            self.assertTrue(same(output, expected))
            # 断言编译帧数与期望帧数相符
            assert cnt.frame_count == expected_frame_counts[i]

    # 使用装饰器创建一个测试函数
    @make_test
    def test_list_slice_assignment(x):
        # 创建列表 m
        m = [1, 2, 3, 4]
        # 使用切片赋值方式将 m 中索引从 1 开始的所有元素替换为 6
        m[1:] = [6] * (len(m) - 1)
        # 返回 x + 1
        return x + 1

    # 使用装饰器创建一个测试函数
    @make_test
    def test_distributed_is_available(x):
        # 如果 torch 分布式可用，则返回 x + 1
        if torch.distributed.is_available():
            return x + 1
        else:
            # 否则返回 x - 1
            return x - 1

    # 如果 torch 分布式不可用，则跳过测试
    @unittest.skipIf(
        not torch.distributed.is_available(), "requires distributed package"
    )
    # 使用装饰器创建一个测试函数
    @make_test
    # 定义一个测试函数，检查是否初始化了分布式环境，返回处理后的值
    def test_distributed_is_initialized(x):
        # 检查当前是否初始化了分布式环境
        if torch.distributed.is_initialized():
            return x + 1  # 如果初始化了，返回 x + 1
        else:
            return x - 1  # 如果未初始化，返回 x - 1

    # 在动态形状下禁用翻译验证的装饰器，并创建一个测试函数
    @disable_translation_validation_if_dynamic_shapes
    @make_test
    def test_torch_distributions_functions(x):
        # 创建一个正态分布对象 normal
        normal = torch.distributions.Normal(x, torch.tensor(1))
        # 使用 normal 创建一个独立分布对象 independent
        independent = torch.distributions.Independent(normal, 1)
        # 计算独立分布的对数概率密度函数值，并返回
        return independent.log_prob(x)

    # 创建一个测试函数，测试包裹嵌套函数且没有闭包的上下文
    @make_test
    def test_context_wrapping_nested_functions_no_closure(x):
        # 定义一个使用 torch.no_grad 装饰的嵌套函数 augment
        @torch.no_grad()
        def augment(x: torch.Tensor) -> torch.Tensor:
            return (x + 1) * 2  # 对输入张量进行增强处理

        # 调用嵌套函数 augment 并返回结果
        return augment(x)

    # # 用于测试 Python 3.10+ 中新增的模式匹配语法 ("match ... case ...")
    # # 如果运行环境为 Python 3.10+，可以取消注释以下测试用例
    # @make_test
    # def test_match_sequence(a):
    #     point = (5, 8)
    #     match point:
    #         case (0, 0):
    #             return a
    #         case (0, y):
    #             return a - y
    #         case (x, 0):
    #             return a + x
    #         case (x, y):
    #             return a + x - y

    # @make_test
    # def test_match_mapping_and_match_keys(x):
    #     param = {"a": 0.5}
    #     match param:
    #         case {"a": param}:
    #             return x * param
    #         case {"b": param}:
    #             return x / param

    # 测试数学函数 radians 的函数
    def test_math_radians(self):
        # 定义一个函数 func，将输入 x 和角度 a 转换为弧度后相加
        def func(x, a):
            return x + math.radians(a)

        # 创建一个编译计数器对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 对函数 func 应用优化和断言的装饰器，得到优化后的函数 cfunc
        cfunc = torch._dynamo.optimize_assert(cnt)(func)

        # 断言函数执行期间未创建新的帧
        assert cnt.frame_count == 0
        # 创建一个随机张量 x
        x = torch.rand(10)
        # 计算预期结果 expected
        expected = func(x, 12)
        # 计算优化后函数 cfunc 的输出结果
        output = cfunc(x, 12)
        # 断言优化后函数的输出与预期结果相同
        self.assertTrue(same(output, expected))
        # 再次断言函数执行期间创建了一个新的帧
        assert cnt.frame_count == 1

    # 创建一个测试函数，生成 numpy 中的 meshgrid，并转换为 PyTorch 张量
    @make_test
    def test_numpy_meshgrid(x, y):
        r1, r2 = np.meshgrid(x.numpy(), y.numpy())
        return torch.from_numpy(r1), torch.from_numpy(r2)

    # 创建一个测试函数，测试从 numpy 数组到 PyTorch 张量的转换
    @make_test
    def test_torch_from_numpy(x):
        # 将输入张量 x 转换为 numpy 数组 a
        a = x.numpy()
        # 从 numpy 数组 a 创建 PyTorch 张量 b
        b = torch.from_numpy(a)
        # 如果张量 b 的第一个维度大小为 1，则返回 True；否则返回 False
        if b.size(0) == 1:
            return torch.tensor(True)
        else:
            return torch.tensor(False)

    # 创建一个测试函数，测试 numpy 数组的尺寸大小
    @make_test
    def test_numpy_size(x):
        # 将输入张量 x 转换为 numpy 数组 a
        a = x.numpy()
        # 返回 numpy 数组 a 的大小
        return a.size

    # 创建一个测试函数，测试 numpy 数组的各种属性
    @make_test
    def test_numpy_attributes(x):
        # 将输入张量 x 转换为 numpy 数组 a
        a = x.numpy()
        # 返回 numpy 数组 a 的各种属性和转换后的 PyTorch 张量
        return (
            a.itemsize,
            a.strides,
            a.shape,
            a.ndim,
            a.size,
            torch.from_numpy(a.T),
            torch.from_numpy(a.real),
            torch.from_numpy(a.imag),
        )

    # 创建一个测试函数，计算输入张量各行的均值后求和，并返回 PyTorch 张量
    @make_test
    def test_mean_sum_np(x: torch.Tensor):
        # 计算输入张量 x 沿着第一维的均值
        x_mean = np.mean(x.numpy(), 1)
        # 计算均值数组 x_mean 的总和
        x_sum = np.sum(x_mean)
        # 将总和 x_sum 转换为 numpy 数组，并再转换为 PyTorch 张量返回
        x_sum_array = np.asarray(x_sum)
        return torch.from_numpy(x_sum_array)

    # 创建一个测试函数，返回输入张量转换后的 numpy 数组的转置
    @make_test
    def test_return_numpy_ndarray(x):
        # 将输入张量 x 转换为 numpy 数组 a，返回其转置
        a = x.numpy()
        return a.T
    # 定义一个测试函数，用于返回输入张量 x 的多个 numpy 数组
    def test_return_multiple_numpy_ndarray(x):
        # 将张量 x 转换为 numpy 数组 a
        a = x.numpy()
        # 返回 numpy 数组 a 的转置、虚部和实部
        return a.T, a.imag, a.real

    # 将该函数标记为测试函数，并使用 make_test 装饰器
    @make_test
    # 定义一个测试函数，用于返回输入张量 x 的 numpy 数组副本
    def test_ndarray_method(x):
        # 将张量 x 转换为 numpy 数组 a
        a = x.numpy()
        # 返回 numpy 数组 a 的副本
        return a.copy()

    @make_test
    # 定义一个测试函数，用于返回输入张量 x 的转置后的 numpy 数组
    def test_ndarray_transpose(x):
        # 将张量 x 转换为 numpy 数组 a
        a = x.numpy()
        # 返回 numpy 数组 a 按照给定的轴进行转置
        return a.transpose(0, 1)

    @make_test
    # 定义一个测试函数，用于返回输入张量 x 的 reshape 后的 numpy 数组
    def test_ndarray_reshape(x):
        # 将张量 x 转换为 numpy 数组 a
        a = x.numpy()
        # 返回 numpy 数组 a 的形状改变为 [1, a.size] 后的数组
        return a.reshape([1, a.size])

    @make_test
    # 定义一个测试函数，用于返回输入张量 x 的最大值和逻辑与结果的 numpy 数组
    def test_ndarray_methods_returning_scalar(x):
        # 将张量 x 转换为 numpy 数组 a
        a = x.numpy()
        # 返回 numpy 数组 a 按照 axis=0 的维度计算的最大值和逻辑与结果的数组
        return a.max(axis=0), a.all(axis=0)

    @make_test
    # 定义一个测试函数，用于返回输入张量 x 的加法和减法的 numpy 数组
    def test_ndarray_builtin_functions(x):
        # 将张量 x 转换为 numpy 数组 a
        a = x.numpy()
        # 返回 numpy 数组 a 自身与自身相加以及自身与自身相减的数组
        return a + a, a - a

    @make_test
    # 定义一个测试函数，用于返回输入张量 x 的全为1的浮点数数组
    def test_numpy_dtype_argument_to_function(x):
        # 返回与张量 x 具有相同形状和类型为 np.float64 的全为1的数组
        return np.ones_like(x, dtype=np.float64)

    @make_test
    # 定义一个测试函数，用于返回与输入张量 x 形状相同的全为2.4的浮点数数组
    def test_numpy_dtype_call_in_function(x):
        # 定义数据类型为 float 的变量 dt
        dt = np.dtype("float")
        # 返回与张量 x 形状相同且数据类型为 dt 的全为 2.4 的数组
        return np.full_like(x, 2.4, dtype=dt)

    @make_test
    # 定义一个测试函数，用于返回输入张量 x 的每列的 L2 范数的数组
    def test_numpy_linalg(x):
        # 返回张量 x 转换为 numpy 数组后，按照 axis=0 的维度计算的 L2 范数
        return np.linalg.norm(x.numpy(), axis=0)

    @make_test
    # 定义一个测试函数，用于返回输入张量 x 的 FFT 移位后的数组
    def test_numpy_fft(x):
        # 返回张量 x 转换为 numpy 数组后进行 FFT 移位的结果
        return np.fft.fftshift(x.numpy())

    @make_test
    # 定义一个测试函数，用于返回全为零的 2x2 的随机数组与自身相减的结果
    def test_numpy_random():
        # 生成一个形状为 (2, 2) 的随机正态分布数组 x
        x = np.random.randn(2, 2)
        # 返回数组 x 与自身相减的结果
        return x - x

    @make_test
    # 定义一个测试函数，用于返回通过偏函数 par_mul 对张量 x 进行 torch.mul 运算的结果
    def test_partials_torch_op_kwarg(x):
        # 创建一个 torch.mul 的偏函数 par_mul，其中 other 参数被固定为全为1的 10x10 张量
        par_mul = functools.partial(torch.mul, other=torch.ones(10, 10))
        # 返回 par_mul 应用在张量 x 上的结果
        return par_mul(x)

    @make_test
    # 定义一个测试函数，用于返回通过偏函数 par_mul 对张量 x 进行 torch.mul 运算的结果
    def test_partials_torch_op_arg(x):
        # 创建一个 torch.mul 的偏函数 par_mul，其中第一个参数被固定为全为1的 10x10 张量
        par_mul = functools.partial(torch.mul, torch.ones(10, 10))
        # 返回 par_mul 应用在张量 x 上的结果
        return par_mul(x)

    @make_test
    # 定义一个测试函数，用于返回通过偏函数 par_mul 对张量 x 进行 udf_mul 函数的结果
    def test_partials_udf_arg(x):
        # 创建一个 udf_mul 函数的偏函数 par_mul，其中第一个参数被固定为全为1的 10x10 张量
        par_mul = functools.partial(udf_mul, torch.ones(10, 10))
        # 返回 par_mul 应用在张量 x 上的结果
        return par_mul(x)

    @make_test
    # 定义一个测试函数，用于返回对一个列表进行加法、列表扩展和求和的结果
    def test_list_add_then_mutate(x):
        # 创建一个包含整数 1 和张量 x 的列表 my_list
        my_list = [1, x]
        # 将张量 x 的四分之一作为新的变量 y
        y = x / 4.0
        # 将列表 my_list 扩展为原列表加上 [x 的二分之一、整数 4] 后的新列表
        my_list = my_list + [x / 2.0, 4]
        # 向列表 my_list 追加变量 y
        my_list.append(y)
        # 返回列表 my_list 中所有元素的和
        return sum(my_list)

    @make_test
    # 定义一个测试函数，用于返回对输入张量 x 扩展 4 次后的和
    def test_list_expand_lhs(x):
        # 返回 4 倍重复张量 x 后的和
        return sum(4 * [x])

    @make_test
    # 定义一个测试函数，用于返回对两个列表的操作结果的和
    def test_in_not_in(x):
        # 创建包含整数 1、2、3、4、5 和张量 x 的列表 mylist
        mylist = [1, 2, 3, 4, 5, x]
        # 创建一个不包含整数 6 的列表 myotherlist
        myotherlist = [1, 2, 3, 4, 5]
        # 断言整数 3 在列表 mylist 中
        assert 3 in mylist
        # 断言整数 6 不在列表 myotherlist 中
        assert 6 not in myotherlist
        # 返回列表 mylist 中所有元素的和
        return sum(mylist)

    @make_test
    # 定义一个测试函数，用于返回通过偏函数 par_mul 对张量 x 进行 udf_mul 函数的结果
    def test_partials_udf_kwarg(x):
        # 创建一个 udf_mul 函数的偏函数 par_mul，其中 y 参数被固定为全为1的 10x10 张量
        par_mul = functools.partial(udf_mul, y=torch.ones(10, 10))
        # 返回 par_mul 应用在张量 x 上的结果
        return par_mul(x)

    @make_test
    # 定义一个测试函数，用于返回通过偏函数 par_mod 对输入 x 和 y 进行 udf_module
    # 定义一个测试函数，用于比较参数 x 和 y 的存储大小是否与给定值相同
    def test_flat_param_same_storage_size(x, y):
        # 导入 flat_param 模块中的 _flat_param
        import torch.distributed.fsdp._flat_param as flat_param

        # 如果 x 的存储大小与 100 相同，则 x 值加 1；否则减 1
        if flat_param._same_storage_size(x, 100):
            x = x + 1
        else:
            x = x - 1

        # 如果 y 的存储大小与 123 相同，则 y 值加 1；否则减 1
        if flat_param._same_storage_size(y, 123):
            y = y + 1
        else:
            y = y - 1

        # 返回经过处理后的 x 和 y 值
        return x, y

    # 使用 parametrize 装饰器来对 attr 参数进行参数化测试
    @parametrize(
        "attr",
        (
            # True 的属性
            "__subclasshook__",
            "__lt__",
            "__hash__",
            "__ge__",
            "__le__",
            "__gt__",
            "__dict__",
            "__getattribute__",
            "__setattr__",
            "__doc__",
            "__repr__",
            "__dir__",
            "__init__",
            "__new__",
            "__class__",
            "__eq__",
            "__delattr__",
            "__reduce__",
            "__module__",
            "__format__",
            "__str__",
            "__sizeof__",
            "__ne__",
            "__call__",
            "__reduce_ex__",
            "__init_subclass__",
            "args",
            "keywords",
            "func",
            # False 的属性
            "__code__",
            "__kwdefaults__",
            "__defaults__",
            "__name__",
            "__annotations__",
            "__get__",
            "__builtins__",
            "__qualname__",
            "__globals__",
            "__closure__",
        ),
    )
    # 定义一个测试函数，用于检查某个函数是否具有特定的属性
    def test_partials_hasattr(self, attr):
        # 内部函数 fn，接收一个参数 t
        def fn(t):
            # 定义一个 lambda 函数 f，其参数为 x 和 y，返回 torch.sin(x) + torch.cos(y)
            f = lambda x, y: torch.sin(x) + torch.cos(y)
            # 创建一个部分应用函数 p，固定参数 y=t
            p = functools.partial(f, y=t)

            # 检查部分应用函数 p 是否具有属性 attr
            if hasattr(p, attr):
                return p(t)
            else:
                return torch.zeros_like(t)

        # 生成一个大小为 (3, 4) 的随机张量 t
        t = torch.randn(3, 4)
        # 创建一个 CompileCounter 对象
        counter = torch._dynamo.testing.CompileCounter()
        # 使用 torch.compile 编译函数 fn，全图模式，使用 counter 作为后端
        opt_fn = torch.compile(fullgraph=True, backend=counter)(fn)
        # 断言优化后的函数 opt_fn(t) 与原始函数 fn(t) 的结果相等
        self.assertEqual(opt_fn(t), fn(t))
        # 断言编译过程中的帧数大于 0
        self.assertGreater(counter.frame_count, 0)

    # 标记这个测试函数为预期失败的测试
    @unittest.expectedFailure
    # 定义一个测试函数，用于检查部分应用函数 p 是否具有 "__name__" 属性，并设置该属性值为 "test"
    def test_partials_hasattr_set_attr(self):
        # 内部函数 fn，接收一个参数 t
        def fn(t):
            # 定义一个 lambda 函数 f，其参数为 x 和 y，返回 torch.sin(x) + torch.cos(y)
            f = lambda x, y: torch.sin(x) + torch.cos(y)
            # 创建一个部分应用函数 p，固定参数 y=t
            p = functools.partial(f, y=t)
            # 设置部分应用函数 p 的 "__name__" 属性为 "test"
            p.__name__ = "test"

            # 检查部分应用函数 p 是否具有 "__name__" 属性
            if hasattr(p, "__name__"):
                return p(t)
            else:
                return torch.zeros_like(t)

        # 生成一个大小为 (3, 4) 的随机张量 t
        t = torch.randn(3, 4)
        # 创建一个 CompileCounter 对象
        counter = torch._dynamo.testing.CompileCounter()
        # 使用 torch.compile 编译函数 fn，全图模式，使用 counter 作为后端
        opt_fn = torch.compile(fullgraph=True, backend=counter)(fn)
        # 断言优化后的函数 opt_fn(t) 与原始函数 fn(t) 的结果相等
        self.assertEqual(opt_fn(t), fn(t))

    # 定义一个测试函数，用于测试 torch.pow 函数的整数指数功能
    def test_pow_int(self):
        # 内部函数 fn，接收两个参数 a 和 b
        def fn(a, b):
            # 返回 a 的 b 次方
            return torch.pow(a, b)

        # 创建一个大小为 (2, 2) 的全 1 张量 x
        x = torch.ones(2, 2)
        # 使用 torch.compile 编译函数 fn，全图模式，使用 "eager" 后端，并启用动态图
        opt_fn = torch.compile(fullgraph=True, backend="eager", dynamic=True)(fn)
        # 断言优化后的函数 opt_fn(x, 2) 和原始函数 fn(x, 2) 的结果相等
        self.assertEqual(opt_fn(x, 2), fn(x, 2))
    def test_tensor_size_indexed_by_symint(self):
        # 定义测试函数，计算两个张量 x 和 y 的形状索引和，并返回结果
        def fn(x, y):
            # 获取 x 张量的最后一个维度的索引
            index = x.shape[-1]
            # 返回 x 张量和 y 张量在索引为 index 处的元素之和
            return x + y.shape[index]

        # 创建两个随机张量 x 和 y，形状分别为 (10, 2) 和 (10, 8, 6)
        x = torch.rand(10, 2)
        y = torch.rand(10, 8, 6)
        # 使用 eager 模式和完整图模式编译优化 fn 函数
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        # 断言优化后的函数和原始函数在相同输入下结果一致
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_partials_as_input_partials_lambda(self):
        # 定义函数 fn，接受两个函数 f0 和 f1，以及输入 x，返回 f0(x) 和 f1(x) 的乘积
        def fn(f0, f1, x):
            return f0(x) * f1(x)

        # 定义 lambda 函数 multiply，用于计算两个输入参数的乘积
        multiply = lambda x, y: x * y
        # 使用 functools.partial 创建两个部分应用函数 lambda0 和 lambda1
        lambda0 = functools.partial(multiply, y=3)
        lambda1 = functools.partial(multiply, y=2)

        # 创建 CompileCounter 对象 cnts 用于统计编译次数
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用动态编译优化 fn 函数，传入 lambda0、lambda1 和随机张量作为参数
        torch._dynamo.optimize(cnts, nopython=True)(fn)(
            lambda0, lambda1, torch.randn(2, 2)
        )
        # 断言编译帧数为 1
        self.assertEqual(cnts.frame_count, 1)

    def test_partials_as_input_partials_mod(self):
        # 定义函数 fn，接受两个函数 f0 和 f1，以及输入 x，返回 f0(x) 和 f1(x) 的乘积
        def fn(f0, f1, x):
            return f0(x) * f1(x)

        # 使用 functools.partial 创建 lambda0 和 lambda1，分别实例化 SmallNN 类
        lambda0 = functools.partial(SmallNN(), y=torch.randn(2, 2))
        lambda1 = functools.partial(SmallNN(), y=torch.randn(2, 2))

        # 创建 CompileCounter 对象 cnts 用于统计编译次数
        cnts = torch._dynamo.testing.CompileCounter()
        x = torch.randn(2, 2)
        # 使用动态编译优化 fn 函数，传入 lambda0、lambda1 和 x 作为参数
        dynamo_result = torch._dynamo.optimize(cnts, nopython=True)(fn)(
            lambda0, lambda1, x
        )
        # 断言编译帧数为 1
        self.assertEqual(cnts.frame_count, 1)

        # 在 eager 模式下计算 fn 函数的结果
        eager_result = fn(lambda0, lambda1, x)
        # 断言动态编译的结果和 eager 模式计算的结果相等
        self.assertEqual(eager_result, dynamo_result)

    def test_partials_as_input_UDF(self):
        # 定义函数 fn，接受两个函数 f0 和 f1，以及输入 x，返回 f0(x) 和 f1(x) 的乘积
        def fn(f0, f1, x):
            return f0(x) * f1(x)

        # 使用 functools.partial 创建 lambda0 和 lambda1，分别实例化 udf_mul 函数
        lambda0 = functools.partial(udf_mul, y=torch.randn(2, 2))
        lambda1 = functools.partial(udf_mul, y=torch.randn(2, 2))

        # 创建 CompileCounter 对象 cnts 用于统计编译次数
        cnts = torch._dynamo.testing.CompileCounter()
        x = torch.randn(2, 2)
        # 使用动态编译优化 fn 函数，传入 lambda0、lambda1 和 x 作为参数
        dynamo_result = torch._dynamo.optimize(cnts, nopython=True)(fn)(
            lambda0, lambda1, x
        )
        # 断言编译帧数为 1
        self.assertEqual(cnts.frame_count, 1)

        # 在 eager 模式下计算 fn 函数的结果
        eager_result = fn(lambda0, lambda1, x)
        # 断言动态编译的结果和 eager 模式计算的结果相等
        self.assertEqual(eager_result, dynamo_result)

    def test_partials_graph_break_reconstruct(self):
        # 定义函数 fn，接受两个函数 udf_mul_0 和 udf_mul_1，以及输入 x
        def fn(udf_mul_0, udf_mul_1, x):
            # 使用 udf_mul_0 和 udf_mul_1 创建 lambda0 和 lambda1
            lambda0 = functools.partial(udf_mul_0, y=x)
            lambda1 = functools.partial(udf_mul_1, y=x)

            # 打印信息用于标记中断点
            print("break")
            # 返回 lambda0(x) 和 lambda1(x) 的元素乘积
            return torch.mul(lambda0(x), lambda1(x))

        # 创建 EagerAndRecordGraphs 类型的 backend 对象
        backend = EagerAndRecordGraphs()
        # 创建 CompileCounterWithBackend 对象 cnts，关联上述 backend
        cnts = CompileCounterWithBackend(backend)
        x = torch.randn(2, 2)
        # 使用动态编译优化 fn 函数，传入 udf_mul 函数作为参数
        dynamo_result = torch._dynamo.optimize(cnts)(fn)(udf_mul, udf_mul, x)

        # 在 eager 模式下计算 fn 函数的结果
        eager_result = fn(udf_mul, udf_mul, x)
        # 获取第一个图形的图形模型
        gm = backend.graphs[0]
        # 断言动态编译的结果和 eager 模式计算的结果相等
        self.assertEqual(eager_result, dynamo_result)
        # 如果 torch._dynamo.config.assume_static_by_default 为真，则验证预期的内联结果
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                # 标准化后的可读图形模型输出
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    # 定义神经网络模块 GraphModule，继承自 torch.nn.Module

    def forward(self, L_lambda0_keywords_y_: "f32[2, 2]"):
        # 定义前向传播函数 forward，接受输入参数 L_lambda0_keywords_y_，类型为 "f32[2, 2]"

        l_lambda0_keywords_y_ = L_lambda0_keywords_y_
        # 将输入参数 L_lambda0_keywords_y_ 赋值给变量 l_lambda0_keywords_y_

        mul: "f32[2, 2]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_
        # 计算 l_lambda0_keywords_y_ 与自身的乘积，存储在 mul 变量中

        mul_1: "f32[2, 2]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_;  l_lambda0_keywords_y_ = None
        # 计算 l_lambda0_keywords_y_ 与自身的乘积，存储在 mul_1 变量中，并清空 l_lambda0_keywords_y_

        mul_2: "f32[2, 2]" = torch.mul(mul, mul_1);  mul = mul_1 = None
        # 使用 torch.mul 计算 mul 和 mul_1 的元素级乘积，存储在 mul_2 变量中，并清空 mul 和 mul_1

        return (mul_2,)
        # 返回 mul_2，作为前向传播函数的输出结果
    # 定义一个测试函数，测试部分图形中断重建混合的情况，不考虑源
    def test_partials_graph_break_reconstruct_mix_no_source(self):
        # 定义一个内部函数 fn，接受两个参数 udf_mul_0 和 x
        def fn(udf_mul_0, x):
            # 定义一个匿名函数 udf_add_1，用于将两个参数相加
            udf_add_1 = lambda x, y: x + y

            # 创建 lambda0，使用 functools.partial 将 udf_mul_0 和 y=x 绑定
            lambda0 = functools.partial(udf_mul_0, y=x)
            # 创建 lambda1，使用 functools.partial 将 udf_add_1 和 x 绑定
            lambda1 = functools.partial(udf_add_1, x)

            # 打印调试信息，输出 "break"
            print("break")
            
            # 返回 lambda0(x) 与 lambda1(x) 的乘积
            return torch.mul(lambda0(x), lambda1(x))

        # 实例化 EagerAndRecordGraphs 后端
        backend = EagerAndRecordGraphs()
        # 使用 CompileCounterWithBackend 统计编译次数
        cnts = CompileCounterWithBackend(backend)
        # 创建一个 2x2 的随机张量 x
        x = torch.randn(2, 2)
        # 使用 torch._dynamo.optimize(cnts) 优化 fn，并传入 udf_mul 和 x
        dynamo_result = torch._dynamo.optimize(cnts)(fn)(udf_mul, x)

        # 直接调用 fn 函数，传入 udf_mul 和 x，获取 Eager 模式下的结果
        eager_result = fn(udf_mul, x)
        # 从 backend 中获取第一个图形 gm
        gm = backend.graphs[0]
        # 断言 eager_result 和 dynamo_result 相等
        self.assertEqual(eager_result, dynamo_result)
        # 如果 torch._dynamo.config.assume_static_by_default 为真
        if torch._dynamo.config.assume_static_by_default:
            # 断言规范化后的 gm 输出与预期输出相等
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    # 定义一个继承自 torch.nn.Module 的图模块类
    def forward(self, L_lambda0_keywords_y_: "f32[2, 2]"):
        # 定义前向传播方法，接受一个名为 L_lambda0_keywords_y_ 的参数，类型为 f32[2, 2]

        l_lambda0_keywords_y_ = L_lambda0_keywords_y_
        # 将参数 L_lambda0_keywords_y_ 赋值给局部变量 l_lambda0_keywords_y_

        mul: "f32[2, 2]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_
        # 计算 l_lambda0_keywords_y_ 的平方，存储在 mul 变量中，类型为 f32[2, 2]

        add: "f32[2, 2]" = l_lambda0_keywords_y_ + l_lambda0_keywords_y_;  l_lambda0_keywords_y_ = None
        # 计算 l_lambda0_keywords_y_ 与自身的和，存储在 add 变量中，类型为 f32[2, 2]，并将 l_lambda0_keywords_y_ 置为 None

        mul_1: "f32[2, 2]" = torch.mul(mul, add);  mul = add = None
        # 计算 mul 和 add 的乘积，存储在 mul_1 变量中，类型为 f32[2, 2]，并清除 mul 和 add 变量

        return (mul_1,)
        # 返回包含 mul_1 的元组作为前向传播的输出
    def test_partials_recompilation(self):
        # 定义一个函数 fn，接受两个函数 f0 和 f1，以及参数 x，返回 f0(x) * f1(x)
        def fn(f0, f1, x):
            return f0(x) * f1(x)

        # 使用 functools.partial 创建 lambda0 和 lambda1 函数，均为 udf_mul 的部分应用，y 参数为随机生成的 2x2 Tensor
        lambda0 = functools.partial(udf_mul, y=torch.randn(2, 2))
        lambda1 = functools.partial(udf_mul, y=torch.randn(2, 2))

        # 创建一个 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()

        # 生成一个 2x2 的随机 Tensor x
        x = torch.randn(2, 2)
        # 对 fn 进行优化，使用 torch._dynamo.optimize 函数，设置 nopython=True
        fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 调用优化后的 fn 函数，传入 lambda0、lambda1 和 x，保存结果到 dynamo_result
        dynamo_result = fn(lambda0, lambda1, x)
        # 断言 cnts.frame_count 等于 1
        self.assertEqual(cnts.frame_count, 1)

        # 再次调用 fn，传入 lambda1、lambda0 和 x，期望不重新编译，因为 Tensor 和 udf_mul 已被保护
        fn(lambda1, lambda0, x)
        # 断言 cnts.frame_count 仍然为 1
        self.assertEqual(
            cnts.frame_count, 1
        )  # No recompile! Tensor and udf_mul guarded

        # 创建 lambda2，使用 functools.partial 创建 udf_mul 的部分应用，y 参数为随机生成的 3x3 Tensor
        lambda2 = functools.partial(udf_mul, y=torch.randn(3, 3))
        # 生成一个 3x3 的随机 Tensor x
        x = torch.randn(3, 3)
        # 调用 fn，传入 lambda2、lambda2 和 x，期望重新编译，因为 Tensor 大小发生了变化
        fn(lambda2, lambda2, x)
        # 断言 cnts.frame_count 等于 2
        self.assertEqual(cnts.frame_count, 2)  # Recompile! Tensor size changed

        # 创建 multiply 函数，接受两个参数 x 和 y，返回它们的乘积
        multiply = lambda x, y: x * y
        # 创建 lambda3，使用 functools.partial 创建 multiply 的部分应用，y 参数为随机生成的 3x3 Tensor
        lambda3 = functools.partial(multiply, y=torch.randn(3, 3))
        # 生成一个 3x3 的随机 Tensor x
        x = torch.randn(3, 3)
        # 调用 fn，传入 lambda3、lambda3 和 x，期望重新编译，因为函数标识发生了变化
        fn(lambda3, lambda3, x)
        # 断言 cnts.frame_count 等于 3
        self.assertEqual(cnts.frame_count, 3)  # Recompile! func id changed

        # 定义一个函数 fn2，接受两个函数 f0 和 f1，以及参数 args，返回 f0(*args) * f1(*args)
        def fn2(f0, f1, args):
            return f0(*args) * f1(*args)

        # 创建一个新的 CompileCounter 对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()

        # 生成一个 2x2 的随机 Tensor x
        x = torch.randn(2, 2)
        # 对 fn2 进行优化，使用 torch._dynamo.optimize 函数，设置 nopython=True
        fn2 = torch._dynamo.optimize(cnts, nopython=True)(fn2)
        # 调用优化后的 fn2 函数，传入 lambda0、lambda1 和 [x]，保存结果到 dynamo_result
        dynamo_result = fn2(lambda0, lambda1, [x])
        # 断言 cnts.frame_count 等于 1，表示重新开始计数
        self.assertEqual(cnts.frame_count, 1)  # start over

        # 创建 lambda4，使用 functools.partial 创建 multiply 的部分应用，y 参数为 3，x 参数为随机生成的 3x3 Tensor
        lambda4 = functools.partial(multiply, y=3, x=torch.randn(3, 3))
        # 调用 fn2，传入 lambda4、lambda4 和空列表 []，期望重新编译，因为关键字参数发生了变化
        fn2(lambda4, lambda4, [])
        # 断言 cnts.frame_count 等于 2
        self.assertEqual(cnts.frame_count, 2)  # Recompile! Different kwarg keys

        # 创建 lambda5，使用 functools.partial 创建 multiply 的部分应用，参数为 1
        lambda5 = functools.partial(multiply, 1)
        # 生成一个 3x3 的随机 Tensor x
        x = torch.randn(3, 3)
        # 调用 fn2，传入 lambda5、lambda5 和 [x]，期望重新编译，因为位置参数发生了变化
        fn2(lambda5, lambda5, [x])
        # 断言 cnts.frame_count 等于 3
        self.assertEqual(cnts.frame_count, 3)  # Recompile! Different arg keys

        # 创建 lambda6，定义一个简单的 lambda 函数，对输入 x 执行 x + x 操作
        lambda6 = lambda x: x + x
        # 调用 fn2，传入 lambda6、lambda6 和 [x]，期望重新编译，因为输入不再是 functools.partial 对象
        fn2(lambda6, lambda6, [x])
        # 断言 cnts.frame_count 等于 4
        self.assertEqual(
            cnts.frame_count, 4
        )  # Recompile! input is no longer a functools partial

    def test_manual_seed(self):
        # 定义一个装饰器函数 foo，设置随机种子为 3，返回一个形状为 (5,) 的随机整数 Tensor
        @torch.compile
        def foo():
            torch.manual_seed(3)
            return torch.randint(0, 5, (5,))

        # 断言调用 foo() 和 foo() 返回相同的结果
        self.assertEqual(foo(), foo())
        # 断言调用 foo() 和 foo() 返回相同的结果
        self.assertEqual(foo(), foo())

    def test_partial_across_graph_break_uninvoked(self):
        # 导入 functools 模块中的 partial 函数
        from functools import partial

        # 定义一个函数 bar，接受参数 x 和 kwargs，返回 x + x
        def bar(x, **kwargs):
            return x + x

        # 定义一个装饰器函数 foo，使用 torch.compile 进行装饰，设置后端为 "eager"，动态模式为 True
        def foo(x, i):
            # 定义内部函数 inner，打印消息 "this is a graph_break"，并返回 op(x) 的结果
            def inner():
                print("this is a graph_break")
                return op(x)

            # 使用 partial 函数创建 op 函数，调用 bar 函数，设置 dim 参数为 10
            op = partial(bar, dim=10)
            # 调用 inner 函数，保存结果到 x
            x = inner()
            # 修改 op 函数，调用 bar 函数，设置 other 参数为 10
            op = partial(bar, other=10)
            # 再次调用 inner 函数，并返回其结果与 x 的和
            return inner() + x

        # 调用 foo，传入随机生成的大小为 1 的 Tensor 和 10
        foo(torch.rand(1), 10)
    # 定义一个测试函数，用于测试内部函数没有重新编译的情况
    def test_no_recompile_inner_function(self):
        # 定义一个内部函数 forward，接收输入 inp
        def forward(inp):
            # 定义内部函数 g，其返回值为 inp 与 y 的和
            def g(y):
                return inp + y
            
            # 打印信息，表示图形中断
            print("graph break")
            # 调用 g 函数，并传入一个随机生成的张量作为参数
            return g(torch.rand([1]))
        
        # 实例化一个编译计数器
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 forward 函数应用优化装饰器，返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(forward)
        
        # 创建一个输入张量
        input = torch.rand([2])
        # 调用 opt_fn 函数，并传入输入张量 input
        _ = opt_fn(input)
        _ = opt_fn(input)
        _ = opt_fn(input)
        # 断言编译帧数为 1，即不应该重新编译
        self.assertEqual(cnts.frame_count, 1)

    # 定义一个测试函数，用于测试内部 lambda 函数没有重新编译的情况
    def test_no_recompile_inner_lambda(self):
        # 定义一个内部函数 forward，接收输入 inp
        def forward(inp):
            # 定义一个 lambda 函数 g，其返回值为 inp 与 y 的和
            g = lambda y: inp + y
            # 打印信息，表示图形中断
            print("graph break")
            # 调用 g 函数，并传入一个随机生成的张量作为参数
            return g(torch.rand([1]))
        
        # 实例化一个编译计数器
        cnts = torch._dynamo.testing.CompileCounter()
        # 对 forward 函数应用优化装饰器，返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnts)(forward)
        
        # 创建一个输入张量
        input = torch.rand([2])
        # 调用 opt_fn 函数，并传入输入张量 input
        _ = opt_fn(input)
        _ = opt_fn(input)
        _ = opt_fn(input)
        # 断言编译帧数为 1，即不应该重新编译
        self.assertEqual(cnts.frame_count, 1)

    # 定义一个测试函数，用于测试复杂闭包的情况
    def test_complex_closure(self):
        # 使用 @torch.compile 装饰器定义一个函数 forward，接收输入 y
        @torch.compile
        def forward(y):
            # 定义一个内部函数 a
            def a():
                # 定义一个内部函数 x，其返回值为 y 与 z 的和
                def x(z):
                    return y + z
                
                return x
            
            return a()
        
        # 创建两个输入张量
        input1 = torch.rand([2])
        input2 = torch.rand([2])
        # 调用 forward 函数，并传入 input1，然后再传入 input2
        res = forward(input1)(input2)
        # 断言 res 与 input1 + input2 相等
        self.assertTrue(same(res, input1 + input2))

    # 定义一个测试函数，用于测试非内联闭包的情况
    def test_non_inlined_closure(self):
        # 使用 @torch.compile() 装饰器定义一个函数 program，接收输入 x 和 y
        @torch.compile()
        def program(x, y):
            # 定义一个 lambda 函数 one，其返回值为 x 与 y 的和
            one = lambda x, y: x + y
            
            # 定义一个内部函数 inner
            def inner():
                # 强制不内联化
                torch._dynamo.graph_break()
                # 返回调用 one 函数，并传入 x 和 y 的结果
                return one(x, y)
            
            # 调用 inner 函数，得到结果 res
            res = inner()
            # 重新定义 lambda 函数 one，其返回值为 x 与 y 的差
            one = lambda x, y: x - y
            # 将 inner 函数的结果加到 res 中
            res += inner()
            # 返回 res
            return res
        
        # 创建两个随机张量作为输入
        input1 = torch.randn(1)
        input2 = torch.randn(1)
        
        # 断言 program 函数的结果与 input1 + input1 相等
        self.assertTrue(same(program(input1, input2), input1 + input1))

    # 使用 @parametrize 装饰器定义一个测试函数，用于测试 np 常量集合作为输入的情况
    @parametrize("int_or_float", ("int", "float"))
    def test_np_constant_collections_as_input(self, int_or_float):
        # 获取 np 中对应类型的信息函数，如 np.intinfo 或 np.floatinfo
        info_func = getattr(np, f"{int_or_float[0]}info")
        # 构建数据类型的字符串参数，如 "int16" 或 "float16"
        dt_string_arg = f"{int_or_float}16"
        # 获取 np 中对应数据类型的属性，如 np.int16 或 np.float16
        np_dt_attr = getattr(np, dt_string_arg)
        
        # 定义数据类型参数列表 dt_args，包括数据类型的字符串参数和其对应的 np 属性
        dt_args = [dt_string_arg, np_dt_attr]
        # 使用 itertools.chain 迭代器连接 dt_args 列表中的元素，形成参数变体迭代器 arg_variants_iter
        arg_variants_iter = itertools.chain(
            dt_args, map(np.dtype, dt_args), map(info_func, dt_args)
        )
        
        # 定义一个函数 func，接收参数 a、b 和 info_or_dt
        def func(a, b, info_or_dt):
            # 返回 a 和 info_func(info_or_dt).max 的和
            return a + info_func(info_or_dt).max
        
        # 对 func 函数应用优化装饰器，返回优化后的函数 opt_fn
        opt_fn = torch.compile(func)
        
        # 创建两个随机张量作为输入
        a = torch.randn(2)
        b = torch.randn(2)
        # 计算非优化版本的结果 eager_result
        eager_result = func(a, b, dt_args[0])
        
        # 遍历 arg_variants_iter 迭代器中的每一个参数变体 arg
        for arg in arg_variants_iter:
            # 计算优化版本的结果 opt_result
            opt_result = opt_fn(a, b, arg)
            # 断言优化版本的结果与非优化版本的结果 eager_result 相等
            self.assertTrue(same(opt_result, eager_result))

    # 使用 @parametrize 装饰器定义一个测试函数，用于测试不同类型和信息函数的情况
    @parametrize(
        "typ, info_func",
        [
            (int, np.iinfo),
            (float, np.finfo),
        ],
        name_fn=lambda t, _: t.__name__,
    )
    # 定义一个测试函数，用于测试 numpy 常量集合和防护措施
    def test_np_constant_collections_guards(self, typ, info_func):
        # 定义一个内部函数，接受一个参数 a 和 info，并返回 a 加上 info 的最大值
        def func_info(a, info):
            return a + info.max

        # 定义一个内部函数，接受一个参数 a 和 dt，并返回 a 加上通过 info_func 获取的 dt 的最大值
        def func_dtype(a, dt):
            return a + info_func(dt).max

        # 创建包含不同数据类型的列表
        dt_args = [
            np.dtype(typ),  # 使用 typ 创建的 numpy 数据类型
            np.ones((1,), dtype=typ).dtype,  # 使用 typ 创建的数组并获取其数据类型
            np.dtype(np.dtype(typ).name),  # 使用 typ 创建的 numpy 数据类型的名称的数据类型
            np.dtype(typ.__name__),  # 使用 typ 的名称创建的 numpy 数据类型
        ]

        # 创建一个编译计数器对象
        cnts_1 = torch._dynamo.testing.CompileCounter()

        # 使用 torch._dynamo.optimize 装饰器优化 func_dtype 函数
        opt_fn_dtype = torch._dynamo.optimize(cnts_1)(func_dtype)

        # 创建一个包含全零的 torch 张量 a，数据类型为 typ
        a = torch.zeros(3, dtype=typ)

        # 遍历数据类型参数列表 dt_args
        for arg in dt_args:
            r = opt_fn_dtype(a, arg)

        # 断言帧计数为 1
        self.assertEqual(cnts_1.frame_count, 1)

        # 创建另一个编译计数器对象
        cnts_2 = torch._dynamo.testing.CompileCounter()

        # 使用 torch._dynamo.optimize 装饰器优化 func_info 函数
        opt_fn_info = torch._dynamo.optimize(cnts_2)(func_info)

        # 使用 info_func 获取 info_args 列表，其中包含 dt_args 中每个元素的信息
        info_args = [info_func(dt) for dt in dt_args]

        # 遍历 info_args 列表
        for arg in info_args:
            r = opt_fn_info(a, arg)

        # 断言帧计数为 1
        self.assertEqual(cnts_2.frame_count, 1)

        # 根据 typ 是否为 float 类型，选择额外的数据类型 dt_extra
        if typ is float:
            dt_extra = np.dtype(np.float16)
        else:
            dt_extra = np.dtype(np.int16)

        # 使用 info_func 获取额外信息 info_extra
        info_extra = info_func(dt_extra)

        # 使用 func_dtype 函数计算 eager_result_dtype 和 compile_result_dtype
        eager_result_dtype = func_dtype(a, dt_extra)
        compile_result_dtype = opt_fn_dtype(a, dt_extra)

        # 断言帧计数为 2
        self.assertEqual(cnts_1.frame_count, 2)

        # 断言 eager_result_dtype 等于 compile_result_dtype
        self.assertEqual(eager_result_dtype, compile_result_dtype)

        # 使用 func_info 函数计算 eager_result_info 和 compile_result_info
        eager_result_info = func_info(a, info_extra)
        compile_result_info = opt_fn_info(a, info_extra)

        # 断言帧计数为 2
        self.assertEqual(cnts_2.frame_count, 2)

        # 断言 eager_result_info 等于 compile_result_info
        self.assertEqual(eager_result_info, compile_result_info)

    # 定义一个测试函数，用于比较常量和张量
    def test_compare_constant_and_tensor(self):
        # 遍历包含各种比较操作符的列表
        for op in [
            operator.lt,
            operator.le,
            operator.gt,
            operator.ge,
            operator.ne,
            operator.eq,
            operator.is_,
            operator.is_not,
        ]:
            # 使用 subTest 创建子测试环境，名称为当前操作符 op
            with self.subTest(op=op):

                # 定义一个函数 fn，接受一个参数 x，并返回 op(-10, x) 的结果
                def fn(x):
                    return op(-10, x)

                # 使用 torch.compile 装饰器编译 fn 函数，生成优化后的函数 opt_fn
                opt_fn = torch.compile(fullgraph=True)(fn)

                # 创建一个包含随机数的 torch 张量 x
                x = torch.randn(10)

                # 断言 opt_fn(x) 的结果等于 fn(x) 的结果
                self.assertEqual(opt_fn(x), fn(x))

    # 定义一个测试函数，用于测试正数操作
    def test_pos(self):
        # 定义一个函数 fn，接受两个参数 x 和 y，并返回 operator.pos(x) * +y 的结果
        def fn(x, y):
            return operator.pos(x) * +y

        # 使用 torch.compile 装饰器编译 fn 函数，生成优化后的函数 opt_fn
        opt_fn = torch.compile(fullgraph=True, dynamic=True)(fn)

        # 定义一个测试函数 test，接受两个参数 x 和 y，断言 opt_fn(x, y) 的结果等于 fn(x, y) 的结果
        def test(x, y):
            self.assertEqual(opt_fn(x, y), fn(x, y))

        # 测试 torch.ones 生成的张量和标量 1
        test(torch.ones(4), 1)
        # 测试标量 1 和 torch.ones 生成的张量
        test(1, torch.ones(4))
        # 测试标量 -1 和 -1
        test(-1, -1)
        # 测试标量 -1.1 和 1.1
        test(-1.1, 1.1)
        # 测试布尔值 True 和 False
        test(True, False)
        # 测试 torch.ones 生成的 float32 类型张量和标量 1.1
        test(torch.ones(4, dtype=torch.float32), 1.1)
    def test_index(self):
        # 定义一个函数 fn，接受参数 x 和 t，调用 operator.index 函数获取 x 的索引值，并用 torch.mul 函数计算 t 与 v 的乘积
        def fn(x, t):
            v = operator.index(x)
            torch.mul(t, v)

        # 定义一个测试函数 test，接受参数 a 和 b，使用 self.assertEqual 检查 opt_fn(a, b) 和 fn(a, b) 的结果是否相等
        def test(a, b):
            self.assertEqual(opt_fn(a, b), fn(a, b))

        # 在动态模式和非动态模式下循环测试
        for dynamic in [True, False]:
            torch._dynamo.reset()
            # 使用 torch._dynamo.optimize 方法优化 fn 函数，根据 dynamic 参数选择动态或非动态优化
            opt_fn = torch._dynamo.optimize(dynamic=dynamic)(fn)
            t = torch.ones(1)
            # 对不同参数进行测试
            test(10, t)
            test(-100, t)
            test(10, t)
            test(False, t)
            test(True, t)

    def test_truth(self):
        # 定义一个函数 fn，接受参数 x 和 y，返回 operator.truth(x) and bool(y) 的结果
        def fn(x, y):
            return operator.truth(x) and bool(y)

        # 使用 torch.compile 方法编译 fn 函数，设置 fullgraph=True，dynamic=False
        opt_fn = torch.compile(fullgraph=True, dynamic=False)(fn)

        # 定义一个测试函数 test，接受参数 x 和 y，使用 self.assertEqual 检查 opt_fn(x, y) 和 fn(x, y) 的结果是否相等
        def test(x, y):
            self.assertEqual(opt_fn(x, y), fn(x, y))

        # 对不同参数进行测试
        test(1, 100)
        test(-1.1, True)
        test(-1.1, 1.1)
        test(True, False)
        test(torch.ones(1), 1)
        test(torch.zeros(1), 1)
        test(torch.ones(1), torch.ones(1))

    def test_unary_fold_op(self):
        # 对 operator.abs, abs, operator.neg, operator.pos, operator.truth 函数进行迭代
        for op in (operator.abs, abs, operator.neg, operator.pos, operator.truth):
            with self.subTest(op=op):
                # 定义一个函数 fn，创建范围为 -10 到 9 的列表 a，使用 map 函数对 a 中的每个元素应用 op 函数
                def fn():
                    a = range(-10, 10)
                    return list(map(op, a))

                # 使用 torch._dynamo.optimize 方法优化 fn 函数，设置 nopython=True
                opt_fn = torch._dynamo.optimize(nopython=True)(fn)
                # 使用 self.assertEqual 检查优化后的结果与原始结果是否相等
                self.assertEqual(opt_fn(), fn())

    def test_unary_fold_op_seq(self):
        # 对 operator.length_hint 函数进行迭代
        for op in (operator.length_hint,):
            with self.subTest(op=op):
                # 定义一个函数 fn，创建包含多个范围的元组 a，使用 map 函数对 a 中的每个元素应用 op 函数
                def fn():
                    a = [tuple(range(-10, i)) for i in range(10)]
                    return tuple(map(op, a))

                # 使用 torch._dynamo.optimize 方法优化 fn 函数，设置 nopython=True
                opt_fn = torch._dynamo.optimize(nopython=True)(fn)
                # 使用 self.assertEqual 检查优化后的结果与原始结果是否相等
                self.assertEqual(opt_fn(), fn())

    def gen_random_range_args(self):
        # 生成一个随机数 args_count，范围为 1 到 3
        args_count = random.randint(1, 3)
        # 生成 args_count 个随机整数，范围为 -10 到 10
        args = [random.randint(-10, 10) for _ in range(args_count)]
        # 如果生成了三个参数且第三个参数为 0，则将其修改为 1
        if args_count == 3 and args[2] == 0:
            args[2] = 1
        return args

    def test_range_length(self):
        # 定义一个测试函数 test，接受任意数量的参数和 expected 参数
        def test(*args, expected=None):
            # 创建一个 range 对象 r
            r = range(*args)
            # 创建一个 RangeVariable 对象 range_variable，包含 args 中每个参数的 ConstantVariable
            range_variable = RangeVariable([ConstantVariable.create(v) for v in args])

            # 使用 self.assertEqual 检查 range 对象的长度与 range_variable.range_length() 的结果是否相等
            self.assertEqual(len(r), range_variable.range_length())

            # 如果指定了 expected 参数，则再次使用 self.assertEqual 检查 range 对象的长度与 expected 是否相等
            if expected is not None:
                self.assertEqual(len(r), expected)

        # 对不同参数进行测试
        test(1, 1, 1, expected=0)
        test(1, 0, expected=0)
        test(-10, expected=0)

        test(4, expected=4)
        test(10, expected=10)

        # step >1 的情况
        test(1, 10, 2, expected=5)

        # 负步长的情况
        test(10, 1, -1, expected=9)
        test(10, 1, -3)

        # 模糊测试
        for i in range(100):
            args = self.gen_random_range_args()
            print("testing :", args)
            test(*args)
    def test_indexed_range(self):
        # 定义一个测试函数，用于测试索引操作
        def test(range, index, expected=None):
            # 创建一个包含常量变量的列表，用于构造范围变量
            range_variable = RangeVariable(
                [
                    ConstantVariable.create(v)
                    for v in [range.start, range.stop, range.step]
                ]
            )

            # 断言范围对象的索引操作结果与范围变量应用索引后的 Python 常量相等
            self.assertEqual(
                range[index],
                range_variable.apply_index(index).as_python_constant(),
            )

            # 如果有预期结果，则再次断言范围对象的索引操作结果与预期结果相等
            if expected is not None:
                self.assertEqual(range[index], expected)

        # 测试范围为 range(10) 的情况，预期索引为 1 时结果为 1
        test(range(10), 1, expected=1)
        # 测试范围为 range(10, 20, 2) 的情况，预期索引为 1 时结果为 12
        test(range(10, 20, 2), 1, expected=12)

        # Fuzz 测试
        for i in range(100):
            # 生成随机的范围参数
            range_args = self.gen_random_range_args()
            r = range(*range_args)

            if len(r) == 0:
                continue

            # 随机选择一个索引
            index = random.randint(0, len(r) - 1)

            print("testing:", r, index)
            test(r, index)

    def test_sliced_range(self):
        # 定义一个测试函数，用于测试切片操作
        def test(range, slice, expected=None):
            # 创建一个包含常量变量的列表，用于构造范围变量
            range_variable = RangeVariable(
                [
                    ConstantVariable.create(v)
                    for v in [range.start, range.stop, range.step]
                ]
            )

            # 断言范围对象的切片操作结果与范围变量应用切片后的 Python 常量相等
            self.assertEqual(
                range[slice],
                range_variable.apply_slice(slice).as_python_constant(),
            )

            # 如果有预期结果，则再次断言范围对象的切片操作结果与预期结果相等
            if expected is not None:
                self.assertEqual(
                    range[slice],
                    expected,
                )

        # 不同情况下的切片测试
        test(range(10), slice(1, 10, 2), expected=range(1, 10, 2))
        test(range(10), slice(None, 10, None), expected=range(0, 10))
        test(range(10), slice(-1, 7, None), expected=range(9, 7))
        test(range(10), slice(-1, 7, 2), expected=range(9, 7, 2))
        test(range(1, 10, 2), slice(3, 7, 2), expected=range(7, 11, 4))
        test(range(1, 10, 2), slice(-3, 7, 2), expected=range(5, 11, 4))
        test(range(-1, -5, -3), slice(5, None, -3), expected=range(-4, 2, 9))

        # 定义一个生成随机切片的函数
        def rand_slice():
            # 定义一个随机翻转硬币的函数
            def flip_coin():
                # 1/10 的概率返回 True
                return random.randint(1, 10) == 5

            # 定义一个生成随机项的函数
            def r_item(allow_zero=True):
                i = random.randint(-10, 10)
                if not allow_zero and i == 0:
                    i = 1
                if flip_coin():
                    i = None
                return i

            # 随机生成 1 到 3 个参数的切片
            arg_count = random.randint(1, 3)

            if arg_count == 1:
                return slice(r_item())
            elif arg_count == 2:
                return slice(r_item(), r_item())
            else:
                return slice(r_item(), r_item(), r_item(False))

        # Fuzz 测试
        for i in range(100):
            # 生成随机的范围参数
            range_args = self.gen_random_range_args()
            r = range(*range_args)
            # 生成随机切片
            s = rand_slice()

            print("testing:", r, s)
            test(r, s)
    # 定义测试函数，测试使用切片索引的情况
    def test_range_with_slice_index(self):
        # 定义内部函数 fn，接受参数 x
        def fn(x):
            # 初始化累加器 acc 为 1
            acc = 1
            # 对于 range(2)[1::2] 中的每个元素 k，执行以下操作：
            for k in range(2)[1::2]:
                # 累乘 acc 乘以自身乘以 k 的值
                acc *= acc * k
            # 返回 x 乘以 acc 的结果
            return x * acc
        
        # 编译函数 fn，并指定使用完整图形
        opt_fn = torch.compile(fullgraph=True)(fn)
        # 创建一个元素全为 1 的 torch 张量 x
        x = torch.ones(1)
        # 断言编译后的函数 opt_fn(x) 的结果与原始函数 fn(x) 的结果相等
        self.assertEqual(opt_fn(x), fn(x))

    # 定义测试函数，测试使用普通索引的情况
    def test_range_with_index(self):
        # 定义内部函数 fn，接受参数 x
        def fn(x):
            # 初始化累加器 acc 为 1
            acc = 1
            # acc 乘以自身乘以 range(10, 20, 2)[2] 的值
            acc *= acc * range(10, 20, 2)[2]
            # 返回 x 乘以 acc 的结果
            return x * acc
        
        # 编译函数 fn，并指定使用完整图形
        opt_fn = torch.compile(fullgraph=True)(fn)
        # 创建一个元素全为 1 的 torch 张量 x
        x = torch.ones(1)
        # 断言编译后的函数 opt_fn(x) 的结果与原始函数 fn(x) 的结果相等
        self.assertEqual(opt_fn(x), fn(x))

    # 定义测试函数，测试内联随机数生成的情况
    def test_rand_inlined(self):
        # 使用装饰器定义编译函数 fn，后端为 eager，启用动态模式
        @torch.compile(backend="eager", dynamic=True)
        def fn():
            # 初始化 idx_size 为 [10]
            idx_size = [10]
            # 随机选择索引 0，设置 idx_size 中对应位置为 1 到 8 之间的随机整数
            idx_size[random.randint(0, 0)] = random.randint(1, 8)
            # 将 idx_size 转换为元组 t
            t = tuple(idx_size)
            # 根据 idx_size 中的值创建 src_size 列表，每个元素为随机数加上对应 idx_size 的值
            src_size = [random.randint(1, 5) + s for s in idx_size]
            # 使用 t 创建一个空的 torch 张量 idx
            idx = torch.empty(t)
        
        # 执行函数 fn
        fn()

    # 定义测试函数，测试部分应用的情况
    def test_rand_tensor_partial(self):
        # 导入必要的模块
        from collections import namedtuple
        from functools import partial
        
        # 定义命名元组 SdpaShape，包含字段 batch、num_heads、seq_len、head_dim
        SdpaShape = namedtuple("Sdpa_Shape", ["batch", "num_heads", "seq_len", "head_dim"])
        
        # 使用装饰器定义编译函数 func，后端为 eager
        @torch.compile(backend="eager")
        def func():
            # 创建部分应用函数 make_tensor，设备为 "cpu"，数据类型为 torch.float16，开启梯度跟踪
            make_tensor = partial(
                torch.rand, device="cpu", dtype=torch.float16, requires_grad=True
            )
            
            # 设置 bsz、num_heads、seq_len_q、seq_len_kv、head_dim 的值
            bsz, num_heads, seq_len_q, seq_len_kv, head_dim = (16, 16, 128, 128, 16)
            # 使用 make_tensor 创建部分应用函数 make_q_tensor 和 make_kv_tensor，分别传入不同的 SdpaShape
            make_q_tensor = partial(
                make_tensor, SdpaShape(bsz, num_heads, seq_len_q, head_dim)
            )
            make_kv_tensor = partial(
                make_tensor, SdpaShape(bsz, num_heads, seq_len_kv, head_dim)
            )
            # 创建张量 t1 和 t2，分别通过调用 make_q_tensor 和 make_kv_tensor 来实现
            t1 = make_q_tensor()
            t2 = make_kv_tensor()
            # 计算张量 t1 和 t2 的和 t3
            t3 = t1 + t2
        
        # 执行函数 func
        func()

    # 定义测试函数，测试张量转换的情况
    def test_to(self):
        # 使用装饰器定义编译函数 fn，后端为 eager
        @torch.compile(backend="eager")
        def fn():
            # 创建一个全为 1 的大小为 2 的 torch 张量 t
            t = torch.ones(2)
            # 将张量 t 转换到设备 "meta"
            y = t.to("meta")
        
        # 执行函数 fn
        fn()

    # 定义测试函数，测试使用省略号索引的情况
    def test_elipsis(self):
        # 使用装饰器定义编译函数 fn，后端为 eager，使用完整图形
        @torch.compile(backend="eager", fullgraph=True)
        def fn(a, ind, val):
            # 将数组 a 的索引 ind 处的元素设置为 val
            a[ind] = val
            # 返回更新后的数组 a
            return a

        # 创建全零数组 arr
        arr = np.zeros(4)
        # 断言调用 fn(arr, np.s_[...], np.ones(4)) 后的结果与全为 1 的数组相等
        self.assertEqual(fn(arr, np.s_[...], np.ones(4)), np.ones(4))

        # 创建二维数组 arr
        arr = np.array([[1, 1], [2, 2]])
        # 断言调用 fn(arr, np.s_[0, ...], np.zeros(2)) 后的结果与指定的数组相等
        self.assertEqual(
            fn(arr, np.s_[0, ...], np.zeros(2)), np.array([[0, 0], [2, 2]])
        )

        # 断言调用 fn(arr, np.s_[1, ...], np.zeros(2)) 后的结果与指定的数组相等
        self.assertEqual(
            fn(arr, np.s_[1, ...], np.zeros(2)), np.array([[1, 1], [0, 0]])
        )

        # 断言调用 fn(arr, np.s_[..., 0], np.array([3, 3])) 后的结果与指定的数组相等
        self.assertEqual(
            fn(arr, np.s_[..., 0], np.array([3, 3])), np.array([[3, 1], [3, 2]])
        )

        # 断言调用 fn(arr, np.s_[..., 1], np.array([3, 3])) 后的结果与指定的数组相等
        self.assertEqual(
            fn(arr, np.s_[..., 1], np.array([3, 3])), np.array([[1, 3], [2, 3]])
        )
# 定义一个简单的函数，用于计算两个数的乘积
def udf_mul(x, y):
    return x * y


# 定义一个函数，用于计算三个数的乘积
def udf_mul2(x, y, z):
    return x * y * z


# 定义一个简单的函数，用于计算两个数的和
def udf_add(x, y):
    return x + y


# 定义一个继承自torch.nn.Module的简单神经网络类
class SmallNN(torch.nn.Module):
    def forward(self, x, y):
        # 将输入的张量在指定维度上拼接
        combined = torch.cat((x, y), dim=1)
        # 使用ReLU激活函数处理拼接后的张量
        out = torch.nn.ReLU()(combined)
        out = torch.nn.ReLU()(out)
        return out


# 定义一个函数，接受一个模块和两个张量作为参数，并调用模块进行处理
def udf_module(mod, x, y):
    return mod(x, y)


# 定义一个全局函数，带有默认的张量参数，并对这些参数执行加法操作后返回
def global_func_with_default_tensor_args(
    x=torch.zeros((2, 2)), *, kw_x=torch.zeros((1, 2))
):
    x.add_(1)
    kw_x.add_(1)
    return x, kw_x


# 定义一个继承自torch.nn.Module的类，该类的前向方法带有默认的张量参数
class ModuleWithDefaultTensorArgsMethod(torch.nn.Module):
    def forward(self, x=torch.zeros((2, 2)), *, kw_x=torch.zeros((1, 2))):
        x.add_(1)
        kw_x.add_(1)
        return x, kw_x


# 定义一个继承自torch.nn.Module的包装类，其中包含一个ModuleWithDefaultTensorArgsMethod对象
class WrapperModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = ModuleWithDefaultTensorArgsMethod()

    def forward(self):
        return self.m()


# 定义一个测试类，继承自torch._dynamo.test_case.TestCase，用于测试带有默认张量参数的函数
class DefaultsTests(torch._dynamo.test_case.TestCase):
    def test_func_default_tensor_args(self):
        """
        Tests that we indeed reference (and mutate) "the one" default tensor arg
        stored on the globally allocated function object, both from the orig and
        compiled function
        """

        # 定义一个内部函数func，用于调用global_func_with_default_tensor_args并返回其结果
        def func():
            return global_func_with_default_tensor_args()

        # 创建一个CompileCounter对象cnts，用于计数编译次数
        cnts = torch._dynamo.testing.CompileCounter()
        # 编译func函数，得到compiled_func函数
        compiled_func = torch.compile(func, backend=cnts)

        # 进行四次循环测试
        for i in range(4):
            if i % 2 == 0:
                x, kw_x = func()
            else:
                x, kw_x = compiled_func()

            # 断言x和kw_x与torch.ones_like(x) + i相等，即判断是否成功进行了加法操作
            self.assertTrue(same(x, torch.ones_like(x) + i))
            self.assertTrue(same(kw_x, torch.ones_like(kw_x) + i))

        # 断言编译帧数为1
        self.assertEqual(cnts.frame_count, 1)
        # 断言操作数为2
        self.assertEqual(cnts.op_count, 2)

        # 修改全局函数的默认张量参数后，重新调用compiled_func函数
        with patch.object(
            global_func_with_default_tensor_args,
            "__defaults__",
            (torch.ones((3, 4, 5)),),
        ):
            x, kw_x = compiled_func()

        # 断言编译帧数为2
        self.assertEqual(cnts.frame_count, 2)
        # 断言操作数为4
        self.assertEqual(cnts.op_count, 4)

        # 修改全局函数的默认关键字参数后，重新调用compiled_func函数
        with patch.object(
            global_func_with_default_tensor_args,
            "__kwdefaults__",
            {"kw_x": torch.ones((3, 4, 5))},
        ):
            x, kw_x = compiled_func()

        # 断言编译帧数为3
        self.assertEqual(cnts.frame_count, 3)
        # 断言操作数为6
        self.assertEqual(cnts.op_count, 6)
    def test_meth_default_tensor_args(self):
        """
        Tests that we indeed reference (and mutate) "the one" default tensor arg
        stored on the globally allocated function object, both from the orig and
        compiled function
        """
        # 创建一个 WrapperModule 实例
        mod = WrapperModule()
        # 创建一个 CompileCounter 实例来计数编译次数
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch.compile 编译 WrapperModule 实例，返回编译后的模块
        compiled_mod = torch.compile(mod, backend=cnts)
        # 循环测试多次
        for i in range(4):
            # 根据循环次数选择调用原始模块还是编译后的模块
            if i % 2 == 0:
                x, kw_x = mod()
            else:
                x, kw_x = compiled_mod()
            # 内部函数每次调用会将参数 x 和 kw_x 加 1
            self.assertTrue(same(x, torch.ones_like(x) + i))
            self.assertTrue(same(kw_x, torch.ones_like(kw_x) + i))
        # 调用编译后的函数两次不会重新编译
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

        # 但是如果修改了默认张量参数，会触发重新编译
        with patch.object(
            ModuleWithDefaultTensorArgsMethod.forward,
            "__defaults__",
            (torch.ones((3, 4, 5)),),
        ):
            x, kw_x = compiled_mod()
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

        with patch.object(
            ModuleWithDefaultTensorArgsMethod.forward,
            "__kwdefaults__",
            {"kw_x": torch.ones((3, 4, 5))},
        ):
            x, kw_x = compiled_mod()
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 6)

    def test_func_default_torch_args(self):
        """
        Tests other types of torch types as function default (size, dtype, device)
        """

        def func_with_default_torch_args(
            dt=torch.float16, ds=torch.Size((1, 2, 3)), dd=torch.device("cpu")
        ):
            # 使用默认参数创建一个张量并返回
            return torch.ones(ds, dtype=dt, device=dd)

        def func():
            # 调用带默认参数的函数 func_with_default_torch_args
            return func_with_default_torch_args()

        # 创建一个 CompileCounter 实例来计数编译次数
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch.compile 编译 func 函数，返回编译后的函数
        compiled_func = torch.compile(func, backend=cnts)
        # 调用原始函数和编译后的函数
        out = func()
        compiled_out = compiled_func()
        # 检查返回的张量的属性是否相同
        self.assertEqual(out.dtype, compiled_out.dtype)
        self.assertEqual(out.device, compiled_out.device)
        self.assertEqual(out.size(), compiled_out.size())
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)
    # 定义一个测试函数，用于测试使用 dataclass 装饰的类的行为
    def test_dataclass_factory(self):
        # 定义一个名为 Output 的 dataclass 类
        @dataclass
        class Output:
            scalar: int = 2  # 定义一个整型属性 scalar，默认为 2
            named_tensors: Dict[str, torch.Tensor] = field(default_factory=dict)  # 定义一个字典属性 named_tensors，默认为空字典
            lists: List[torch.Tensor] = field(default_factory=list)  # 定义一个列表属性 lists，默认为空列表

            # 定义一个方法 scale，返回 scalar 属性的两倍
            def scale(self):
                return self.scalar * 2

        # 定义一个函数 fn，接受参数 x
        def fn(x):
            # 创建 Output 的实例 a，使用参数 1 初始化
            # 检查默认字典赋值
            a = Output(1)
            # 检查 dataclass 方法可以内联
            scaled_value = a.scale()

            # 创建 Output 的实例 b，使用参数 5 初始化，并指定 named_tensors 字段为 {"x": x}
            # 检查正常赋值是否有效
            b = Output(5, named_tensors={"x": x})

            # 创建 Output 的实例 c，使用默认参数初始化
            # 检查默认整数赋值
            c = Output()

            # 检查默认成员是否正确初始化
            if isinstance(a.named_tensors, dict):
                x = torch.sin(x)

            # 更改 dataclass c 的 scalar 属性为 6，将 x 添加到 named_tensors 字典中的键 "x" 中
            c.scalar = 6
            c.named_tensors["x"] = x

            # 返回 dataclass c，以便检查重建
            return c, torch.cos(x) * scaled_value + b.named_tensors["x"] + c.scalar

        # 创建一个 CompileCounter 实例 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch.compile 编译函数 fn，并记录计数器 cnts 的帧数和操作数
        compiled_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        # 创建一个 tensor x，形状为 (4,)
        x = torch.randn(4)
        # 调用函数 fn，并获取返回的 eager_dataclass 和 out
        eager_dataclass, out = fn(x)
        # 调用编译后的函数 compiled_fn，并获取返回的 compiled_dataclass 和 compiled_out
        compiled_dataclass, compiled_out = compiled_fn(x)
        # 断言 eager_dataclass 的 scalar 属性与 compiled_dataclass 的 scalar 属性相等
        self.assertEqual(eager_dataclass.scalar, compiled_dataclass.scalar)
        # 断言 eager_dataclass 的 named_tensors 字典中键 "x" 对应的值与 compiled_dataclass 的 named_tensors 字典中键 "x" 对应的值相等
        self.assertEqual(
            eager_dataclass.named_tensors["x"], compiled_dataclass.named_tensors["x"]
        )
        # 使用 same 函数断言 out 和 compiled_out 相等
        self.assertTrue(same(out, compiled_out))
        # 断言 cnts 的帧数为 1
        self.assertEqual(cnts.frame_count, 1)
        # 断言 cnts 的操作数为 5
        self.assertEqual(cnts.op_count, 5)

    # 定义一个测试函数，测试嵌套 dataclass 的行为
    def test_dataclass_nested(self):
        # 定义一个名为 Base 的 dataclass 类
        @dataclass
        class Base:
            outer_a: int  # 定义一个整型属性 outer_a
            outer_b: int  # 定义一个整型属性 outer_b

        # 定义一个名为 Derived 的 dataclass 类，继承自 Base
        @dataclass
        class Derived(Base):
            inner_a: Any = field(default_factory=list)  # 定义一个属性 inner_a，默认为空列表

        # 定义一个函数 fn，接受参数 x
        def fn(x):
            # 创建 Derived 的实例 l，使用参数 1 和 2 初始化
            l = Derived(1, 2)
            # 返回 l.outer_a 乘以 x 的结果
            return l.outer_a * x

        # 使用 torch.compile 编译函数 fn，使用 eager 后端，并记录完整图形
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        # 创建一个 tensor x，形状为 (4,)
        x = torch.randn(4)
        # 调用函数 fn，并获取结果 res
        res = fn(x)
        # 调用编译后的函数 opt_fn，并获取结果 ref
        ref = opt_fn(x)
        # 断言 ref 和 res 相等
        self.assertEqual(ref, res)

    # 定义一个测试函数，测试包含常量的 tensor 列表行为
    def test_listlike_of_tensors_contains_constant(self):
        # 遍历 list 和 set 两种类型
        for listlike in [set, list]:

            # 定义一个函数 fn，接受参数 x
            def fn(x):
                # 将 x 的值加上 1
                x.add_(1)
                # 创建一个 listlike 类型的对象 s，包含 x
                s = listlike([x])
                # 检查常量 1 是否在 s 中
                res = 1 in s
                # 返回结果 res
                return res

            # 使用 torch.compile 编译函数 fn，使用 eager 后端，并记录完整图形
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            # 创建一个 tensor x，形状为 (1,)
            x = torch.randn(1)
            # 调用编译后的函数 opt_fn，并获取结果 ref
            ref = opt_fn(x)
            # 调用函数 fn，并获取结果 res
            res = fn(x)
            # 断言 ref 和 res 相等
            self.assertEqual(ref, res)
    def test_cast_tensor_single_elem(self):
        # 使用 torch._dynamo.config.patch 临时修改设置，捕获标量输出
        with torch._dynamo.config.patch({"capture_scalar_outputs": True}):
            # 针对不同类型和值进行测试
            for t, val in [
                (float, 1.0),    # 测试 float 类型和浮点数值
                (float, 1),      # 测试 float 类型和整数值，应该自动转换为浮点数
                (float, True),   # 测试 float 类型和布尔值，应该自动转换为浮点数
                (int, 1),        # 测试 int 类型和整数值
                (int, False),    # 测试 int 类型和布尔值，应该自动转换为整数
                # (int, 1.0), # 因为 sym_int 中的 >= 0 比较而失败
            ]:  # , bool, complex]: 对于 sym_bool 不进行转换，不支持 sym_complex

                def fn(x):
                    # 对输入 x 加 1
                    x = x + 1
                    # 返回类型转换后的值
                    return t(x)

                # 使用 torch.compile 编译函数 fn，使用 eager 后端，完整图形，并关闭动态计算
                opt_fn = torch.compile(
                    fn, backend="eager", fullgraph=True, dynamic=False
                )
                # 创建一个包含单个元素 val 的张量 x
                x = torch.tensor([val])
                # 分别使用 fn 和 opt_fn 运行 x，比较结果是否相等
                res = fn(x)
                ref = opt_fn(x)
                self.assertEqual(ref, res)

                # 无法处理非单元素张量
                with self.assertRaises(ValueError):
                    fn(torch.tensor([val] * 2))
                with self.assertRaises(torch._dynamo.exc.TorchRuntimeError):
                    opt_fn(torch.tensor([val] * 2))

    def test_set_construction(self):
        def fn(x):
            # 将 x 加 1，将结果保存到 y 中
            y = x.add_(1)
            # 创建一个包含 x 和 y 的集合 s
            s = set({x})
            s.add(y)
            # 返回集合 s 的长度
            return len(s)

        # 使用 torch.compile 编译函数 fn，使用 eager 后端，完整图形
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        # 创建一个形状为 4 的随机张量 x
        x = torch.randn(4)
        # 分别使用 fn 和 opt_fn 运行 x，比较结果是否相等
        res = fn(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)

    def test_is_tensor_tensor(self):
        def fn(x, y):
            # 如果 x 和 y 是同一个张量对象
            if x is y:
                return x * 2
            else:
                return x + y

        # 使用 torch.compile 编译函数 fn，使用 eager 后端，完整图形，启用动态计算
        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        # 创建两个形状为 (2,) 的零张量 x 和 形状为 (2,) 的全一张量 y
        x = torch.zeros(2)
        y = torch.ones(2)

        # 比较 fn 和 fn_opt 在相同输入下的结果是否一致
        self.assertEqual(fn(x, y), fn_opt(x, y))
        self.assertEqual(fn(x, x), fn_opt(x, x))

    def test_is_not_tensor_tensor(self):
        def fn(x, y):
            # 如果 x 和 y 不是同一个张量对象
            if x is not y:
                return x * 2
            else:
                return x + y

        # 使用 torch.compile 编译函数 fn，使用 eager 后端，完整图形，启用动态计算
        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        # 创建两个形状为 (2,) 的零张量 x 和 形状为 (2,) 的全一张量 y
        x = torch.zeros(2)
        y = torch.ones(2)

        # 比较 fn 和 fn_opt 在相同输入下的结果是否一致
        self.assertEqual(fn(x, y), fn_opt(x, y))
        self.assertEqual(fn(x, x), fn_opt(x, x))

    def test_is_mutated_tensor_tensor(self):
        def fn(x):
            # 将 x 加 1，并将结果保存到 y 中
            y = x.add_(1)
            # 检查 x 是否与 y 是同一个张量对象
            return x is y

        # 使用 torch.compile 编译函数 fn，使用 eager 后端，完整图形，启用动态计算
        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        # 创建形状为 (4,) 的全一张量 z
        z = torch.ones(4)

        # 比较 fn 和 fn_opt 在相同输入下的结果是否一致
        self.assertEqual(fn(z), fn_opt(z))

    def test_is_mutated_tensor_tensor_across_graph_break(self):
        def fn(x):
            # 将 x 加 1，并将结果保存到 y 中
            y = x.add_(1)
            # 检查 x 是否与 y 是同一个张量对象
            cond = x is y
            # 打破图形，使张量的真实值在此处恢复
            torch._dynamo.graph_break()
            x.add_(1)
            # 返回比较结果和条件
            return x is y, cond

        # 使用 torch.compile 编译函数 fn，使用 eager 后端，启用动态计算
        fn_opt = torch.compile(backend="eager", dynamic=True)(fn)

        # 创建形状为 (4,) 的全一张量 z
        z = torch.ones(4)

        # 比较 fn 和 fn_opt 在相同输入下的结果是否一致
        self.assertEqual(fn(z), fn_opt(z))
    # 定义测试函数，验证是否原地修改了张量并返回是否是同一对象的布尔值
    def test_is_mutated_tensor_tensor(self):
        # 定义内部函数fn，对输入张量x执行原地加法操作，并返回结果y是否与x是同一对象
        def fn(x):
            y = x.add_(1)
            return y is x

        # 使用torch.compile进行函数fn的优化编译，设置后端为"eager"，全图模式为True，动态模式为True
        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        # 创建一个元素全为1的张量z，形状为(4, 1)
        z = torch.ones(4, 1)

        # 使用self.assertEqual断言原函数和优化函数在输入z时的输出结果是否相等
        self.assertEqual(fn(z), fn_opt(z))

    # 定义测试函数，验证是否在编译中初始化后修改了张量并返回是否是同一对象的布尔值
    def test_is_init_in_compile_mutated_tensor_tensor(self):
        # 定义内部函数fn，对输入张量x进行克隆操作，然后在克隆张量上执行原地加法，并返回结果y是否与z是同一对象
        def fn(x):
            z = x.clone()
            y = z.add_(1)
            return y is z

        # 使用torch.compile进行函数fn的优化编译，设置后端为"eager"，全图模式为True，动态模式为True
        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        # 创建一个元素全为1的张量z，形状为(4, 1)
        z = torch.ones(4, 1)

        # 使用self.assertEqual断言原函数和优化函数在输入z时的输出结果是否相等
        self.assertEqual(fn(z), fn_opt(z))

    # 定义测试函数，验证是否在编译中使用vmap对张量进行批处理操作后修改了张量并返回是否是同一对象的布尔值
    def test_is_init_in_compile_vmapped_mutated_tensor_tensor(self):
        # 定义内部函数fn，对输入张量z进行克隆操作，然后使用vmap对张量中的每个元素执行acos_操作，并返回结果y是否与x是同一对象
        def fn(z):
            x = z.clone()
            y = torch.vmap(torch.Tensor.acos_)(x)
            _ = y is z
            return y is x

        # 使用torch.compile进行函数fn的优化编译，设置后端为"eager"，全图模式为True，动态模式为True
        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        # 创建一个元素全为1的张量z，形状为(4, 1)
        z = torch.ones(4, 1)

        # 使用self.assertEqual断言原函数和优化函数在输入z时的输出结果是否相等
        self.assertEqual(fn(z), fn_opt(z))

    # 定义测试函数，验证是否在使用vmap对张量进行批处理操作后修改了张量并返回是否是同一对象的布尔值
    def test_is_vmapped_mutated_tensor_tensor(self):
        # 定义内部函数fn，使用vmap对输入张量x中的每个元素执行acos_操作，并返回结果y是否与x是同一对象
        def fn(x):
            y = torch.vmap(torch.Tensor.acos_)(x)
            return y is x

        # 使用torch.compile进行函数fn的优化编译，设置后端为"eager"，全图模式为True，动态模式为True
        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        # 创建一个元素全为1的张量z，形状为(4, 1)
        z = torch.ones(4, 1)

        # 使用self.assertEqual断言原函数和优化函数在输入z时的输出结果是否相等
        self.assertEqual(fn(z), fn_opt(z))

    # 定义测试函数，验证是否在编译中使用vmap对多个张量进行批处理操作后修改了张量并返回是否是同一对象的布尔值
    def test_is_init_in_compile_vmapped_mutated_tensor_tensor_multi_arg(self):
        # 定义内部函数fn，对输入张量y和z进行克隆操作，然后使用vmap对它们执行函数g，在返回结果中判断多个张量是否是同一对象
        def fn(y, z):
            a = y.clone()
            b = z.clone()

            def g(a, b):
                return a.acos_(), b.acos_()

            c, d = torch.vmap(g)(a, b)
            return a is c is b is d

        # 使用torch.compile进行函数fn的优化编译，设置后端为"eager"，全图模式为True，动态模式为True
        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        # 创建元素全为1的张量y和z，形状分别为(4, 2)和(4, 10)
        y = torch.ones(4, 2)
        z = torch.ones(4, 10)

        # 使用self.assertEqual断言原函数和优化函数在输入y和z时的输出结果是否相等
        self.assertEqual(fn(y, z), fn_opt(y, z))
        self.assertEqual(fn(y, y), fn_opt(y, y))

    # 定义测试函数，验证集合中的张量与广播失败
    def test_in_set_would_fail_broadcast(self):
        # 创建形状为(5)和(5, 10)的全零张量param和param2
        param = torch.zeros(5)
        param2 = torch.zeros(5, 10)

        # 创建空集合tensor_list，并将param2添加到集合中
        tensor_list = set()
        tensor_list.add(param2)

        # 使用assert断言param不在tensor_list集合中
        assert param not in tensor_list

        # 定义内部函数fn，对输入的param进行原地加法操作，创建一个新的集合tensor_list包含param2，返回param是否在tensor_list中
        def fn(param, param2):
            param.add_(1)
            tensor_list = set([param2])
            return param in tensor_list

        # 创建torch._dynamo.testing.CompileCounter对象cnts
        cnts = torch._dynamo.testing.CompileCounter()

        # 使用torch._dynamo.optimize对函数fn进行优化，设置nopython=True
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)

        # 使用self.assertEqual断言优化后的函数opt_fn和原函数在输入param和param2时的输出结果是否相等
        self.assertEqual(opt_fn(param, param2), fn(param, param2))

        # 使用self.assertEqual断言优化过程中重新编译的帧数cnts.frame_count是否为1
        self.assertEqual(cnts.frame_count, 1)

        # 测试参数为相同张量的情况
        self.assertEqual(opt_fn(param, param), fn(param, param))

        # 使用self.assertEqual断言再次优化过程中重新编译的帧数cnts.frame_count是否为2
        self.assertEqual(cnts.frame_count, 2)  # Recompiles
    # 定义一个测试方法，用于测试 inplace 操作
    def test_in_set_inplace(self):
        # 创建一个全零的张量 param
        param = torch.zeros(5)
        # 创建一个全零的二维张量 param2
        param2 = torch.zeros(5, 10)

        # 创建一个空的集合 tensor_list，并将 param2 添加到集合中
        tensor_list = set()
        tensor_list.add(param2)
        # 断言 param 不在 tensor_list 中
        assert param not in tensor_list

        # 定义一个嵌套函数 fn，接受 param 和 param2 作为参数
        def fn(param, param2):
            # 对 param 进行 inplace 加法操作，并将结果赋给 y
            y = param.add_(1)  # 张量的方法
            # 使用 torch.Tensor.add_ 函数对 y 加 1，结果赋给 z
            z = torch.Tensor.add_(y, 1)  # torch 函数
            # 创建一个集合 tensor_list，包含 param2
            tensor_list = set([param2])
            # 返回 y 是否在 tensor_list 中，并且 z 是否在 tensor_list 中
            return y in tensor_list and z in tensor_list

        # 创建一个计数器对象 cnts
        cnts = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 对 fn 进行优化，启用 nopython 模式
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        # 断言优化后的函数 opt_fn 返回值与原始函数 fn 返回值相同
        self.assertEqual(opt_fn(param, param2), fn(param, param2))
        # 断言帧计数为 1
        self.assertEqual(cnts.frame_count, 1)

        # 测试别名情况
        # 断言优化后的函数 opt_fn 使用 param 自身调用时返回值与 fn 使用 param 自身调用时返回值相同
        self.assertEqual(opt_fn(param, param), fn(param, param))
        # 断言帧计数为 2，表示重新编译

    # 定义一个测试方法，用于测试重构名称
    def test_reconstructed_name(self):
        # 创建一个空列表 lst
        lst = []

        # 定义一个装饰器函数 disallowed，将 g 的名称追加到 lst 中
        @torch._dynamo.disable
        def disallowed(g):
            lst.append(g.__name__)

        # 定义一个嵌套函数 f
        def f():
            # 定义一个内部函数 g，返回空元组
            def g():
                return ()

            # 调用 disallowed 函数，传入 g 函数作为参数
            disallowed(g)

        # 使用 torch._dynamo 对函数 f 进行优化
        f_opt = torch._dynamo
        opt_f = torch._dynamo.optimize(backend="eager")(f)
        # 调用优化后的函数 opt_f 和原始函数 f
        opt_f()
        f()
        # 断言列表 lst 的长度为 2
        self.assertEqual(len(lst), 2)
        # 断言 lst 的第一个元素等于第二个元素
        self.assertEqual(lst[0], lst[1])

    # 跳过测试条件：Python 版本低于 3.10 时不实现 zip 的严格参数
    @unittest.skipIf(
        sys.version_info < (3, 10),
        "zip strict kwargs not implemented for Python < 3.10",
    )
    # 定义一个测试方法，用于测试严格模式的 zip 函数
    def test_zip_strict(self):
        # 定义一个函数 fn，接受 x、ys、zs 作为参数
        def fn(x, ys, zs):
            # 克隆 x，并将结果赋给 x
            x = x.clone()
            # 对 ys 和 zs 使用 zip 函数进行循环迭代，启用严格模式
            for y, z in zip(ys, zs, strict=True):
                x += y * z
            return x

        # 使用 torch._dynamo.optimize 对 fn 进行优化，使用 eager 模式
        opt_fn = torch._dynamo.optimize(backend="eager")(fn)
        # 使用 nopython 模式对 fn 进行优化
        nopython_fn = torch._dynamo.optimize(backend="eager", nopython=True)(fn)

        # 创建张量 x，全部初始化为 1
        x = torch.ones(3)
        # 创建列表 ys 和 zs，包含浮点数元素
        ys = [1.0, 2.0, 3.0]
        zs = [2.0, 5.0, 8.0]

        # 断言优化后的函数 opt_fn 返回值与原始函数 fn 返回值相同
        self.assertEqual(opt_fn(x, ys, zs), fn(x, ys, zs))

        # 如果启用 nopython 模式，应该引发 UserError
        with self.assertRaisesRegex(torch._dynamo.exc.UserError, "zip()"):
            nopython_fn(x, ys[:1], zs)

        # 应该引发 ValueError，表示允许图形破坏
        with self.assertRaisesRegex(ValueError, "zip()"):
            opt_fn(x, ys[:1], zs)
# 实例化带参数化的测试，传入 FunctionTests 对象
instantiate_parametrized_tests(FunctionTests)

# 如果当前脚本是作为主程序运行
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块导入 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 运行测试用例
    run_tests()
```