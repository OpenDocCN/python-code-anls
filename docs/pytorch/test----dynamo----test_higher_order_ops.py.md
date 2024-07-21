# `.\pytorch\test\dynamo\test_higher_order_ops.py`

```py
# Owner(s): ["module: dynamo"]

# 导入必要的模块和库
import enum
import functools
import pprint
import re
import unittest
import warnings

# 导入functorch的控制流模块
import functorch.experimental.control_flow as control_flow

# 导入PyTorch相关模块
import torch
import torch._dynamo.config as config
import torch._dynamo.test_case
import torch._functorch.config
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.checkpoint

# 导入特定的测试工具和类
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import (
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    normalize_gm,
)
from torch._dynamo.utils import counters, ifdynstaticdefault
from torch._higher_order_ops.wrap import wrap
from torch.testing._internal.common_utils import (
    munge_exc,
    TEST_WITH_TORCHDYNAMO,
    xfailIfTorchDynamo,
)
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test

# 定义一个装饰器，当没有CUDA时跳过测试
requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")


# 检查是否动态形状捕获已启用
def check_dynamic_shape_capture():
    # 这也反映了来自`test/dynamo/test_dynamic_shapes.py:make_dynamic_cls`的config配置
    if not config.assume_static_by_default:
        return True
    return False


# 统计图中特定操作的出现次数
def count_ops(gm, args, freq, op):
    actual = [node.target for node in gm.graph.nodes].count(op)
    assert actual == freq, f"expected={freq}, actual={actual}"
    return gm


# 定义一个空的类对象
class Obj:
    pass


# 定义一个简单的PyTorch模块，包含一个参数并实现前向传播
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.existing = torch.nn.Parameter(torch.ones([]))

    def forward(self, x):
        return self.existing * x


# 创建全局变量和对象
global_obj = Obj()
global_module = MyModule()
global_var = torch.randn(3)
global_num = 3.14
global_list = []


# 在计算图中查找第一个匹配给定函数的节点
def find_first_node(gm, func):
    for node in gm.graph.nodes:
        if node.target is func:
            return node
    return None


# 统计图中所有调用操作的数量
def op_count(gm):
    result = 0
    for node in gm.graph.nodes:
        if "call" in node.op:
            result += 1
    return result


# 断言一个字典与具有“正则表达式键”的字典匹配
def assert_dict_matches_regex(self, dct, dct_with_regex_keys):
    regex_keys = dct_with_regex_keys.keys()
    regex_key_to_actual_key = {}
    for regex_key in regex_keys:
        for key in dct:
            if re.match(regex_key, key):
                if regex_key in regex_key_to_actual_key:
                    raise AssertionError(
                        f"Single key regex mapped to multiple keys. Please improve your "
                        f"regex. Got: regex='{regex_key}' "
                        f"keys='{regex_key_to_actual_key[regex_key]}',"
                        f"'{key}'"
                    )
                regex_key_to_actual_key[regex_key] = key
    new_dct = {}
    # 遍历给定的正则表达式键列表
    for regex_key in regex_keys:
        # 检查当前正则表达式键是否存在于映射字典中
        if regex_key not in regex_key_to_actual_key:
            # 如果找不到匹配的实际键，则抛出断言错误，指明无法匹配的正则表达式键和字典中的所有键
            raise AssertionError(
                f"Got regex '{regex_key}' but could not match any key in dict with "
                f"keys {dct.keys()}"
            )
        # 将新字典中的键值对从原始字典中复制，通过映射字典将正则表达式键映射到实际键
        new_dct[regex_key_to_actual_key[regex_key]] = dct_with_regex_keys[regex_key]
    # 使用断言检查原始字典和新字典是否相等
    self.assertEqual(dct, new_dct)
# 定义一个生成默认参数的生成器函数，接受一个种子值作为输入参数
def default_args_generator(seed_value):
    # 使用 pytree 库的 tree_flatten 函数，将种子值展平并返回展平后的结果和结构描述
    flat_args, args_spec = pytree.tree_flatten(seed_value)
    
    # 循环3次，生成新的参数
    for i in range(3):
        # 创建一个新的空列表来存放新生成的展平参数
        new_flat_arg = []
        
        # 遍历展平后的参数列表
        for val in flat_args:
            # 根据值的类型进行不同的处理
            if isinstance(val, torch.Tensor):
                new_val = val + 0.1 * i  # 对于 torch.Tensor 类型，加上一个倍数为 i 的增量
            elif isinstance(val, int):
                new_val = val + 1 * i    # 对于整数类型，加上一个倍数为 i 的增量
            elif isinstance(val, float):
                new_val = val + 0.1 * i  # 对于浮点数类型，加上一个倍数为 i 的增量
            elif isinstance(val, enum.Enum):
                new_val = val             # 枚举类型保持不变
            else:
                raise AssertionError("unexpected arg type")  # 如果遇到未预期的类型，则引发断言错误
            
            # 将处理后的新值添加到新展平参数列表中
            new_flat_arg.append(new_val)
        
        # 使用 pytree 库的 tree_unflatten 函数，根据结构描述 args_spec 将展平后的参数列表还原为原始结构的参数
        new_args = pytree.tree_unflatten(new_flat_arg, args_spec)
        
        # 使用生成器的 yield 返回新生成的参数
        yield new_args


# 定义一个测试类 HigherOrderOpTests，继承自 torch._dynamo.test_case.TestCase
class HigherOrderOpTests(torch._dynamo.test_case.TestCase):
    
    # 定义一个私有方法 _assert_wrap_fallback，用于验证函数 func 在使用指定参数 args 时的行为
    def _assert_wrap_fallback(self, func, args, setup=lambda: None):
        # 清空计数器 counters
        counters.clear()
        
        # 创建一个 EagerAndRecordGraphs 类型的后端对象 backend
        backend = EagerAndRecordGraphs()
        
        # 创建一个 CompileCounterWithBackend 类型的计数器 cnt，使用指定的后端 backend
        cnt = CompileCounterWithBackend(backend)
        
        # 调用传入的 setup 函数来准备测试环境
        setup()
        
        # 调用 func 函数，使用参数 args，记录其期望的返回值
        expected = func(*args)
        
        # 再次调用 setup 函数来确保测试环境的一致性
        setup()
        
        # 调用 torch.compile 函数，编译 func 函数，使用计数器 cnt 和指定的后端 backend，返回编译后的结果
        result = torch.compile(func, backend=cnt, fullgraph=False)(*args)
        
        # 统计计数器中记录的图断点数量
        num_graph_breaks = len(counters["graph_break"].keys())
        
        # 断言至少有一个图断点被触发
        self.assertGreater(num_graph_breaks, 0)
        
        # 遍历 backend 中的所有图形对象
        for gm in backend.graphs:
            # 遍历每个图形对象中的节点
            for node in gm.graph.nodes:
                # 断言节点的目标不是 wrap
                self.assertFalse(node.target is wrap)
        
        # 断言编译后的结果与预期的结果相等
        self.assertEqual(result, expected)
    
    # 定义一个私有方法 _test_wrap_simple，用于测试简单的包装操作
    def _test_wrap_simple(
        self,
        func,
        args_generator,
        expected_num_wrap_args,
        expected_opcount=2,
        return_graph=False,
        ```
    ):
        # 给定一个函数 `func`，该函数只调用了 `wrap`，
        # 我们检查以下内容：
        # - 没有图形中断
        # - eager vs torch.compile 具有相同的结果（正确性）
        # - 其他编译指标，例如捕获图中的操作数，wrap 是否具有预期数量的参数等
        #
        # 我们通过 args_generator 中的每个参数运行一次或多次，
        # 并检查以下内容：
        # - 每次运行的正确性和无图形中断
        # - 仅对第一次运行检查其他编译指标，因为 automatic_dynamic_shapes 可能会为后续运行编译另一个动态版本图形
        graph = None
        for i, args in enumerate(args_generator):
            # 使用 EagerAndRecordGraphs 后端
            backend = EagerAndRecordGraphs()
            # 使用 CompileCounterWithBackend 计数器
            cnt = CompileCounterWithBackend(backend)
            # 获取 func(*args) 的期望结果
            expected = func(*args)
            # 使用 torch.compile 运行 func，并记录完整的图形
            result = torch.compile(func, fullgraph=True, backend=cnt)(*args)
            # 检查正确性和无图形中断
            self.assertEqual(result, expected)
            self.assertEqual(cnt.frame_count, 1)
            self.assertEqual(len(backend.graphs), 1)
            # 检查其他编译指标
            if i == 0:
                self.assertEqual(cnt.op_count, expected_opcount)
                # 获取第一个运行的图形
                graph = backend.graphs[0]
                # 查找图形中第一个 wrap 节点
                wrap_node = find_first_node(graph, wrap)
                self.assertEqual(len(wrap_node.args), expected_num_wrap_args)
        # 如果 return_graph = True，则始终返回/检查第一次运行的图形
        if return_graph:
            return normalize_gm(graph.print_readable(print_output=False))

    def test_error_message_sane(self):
        foo = []

        def inner(x):
            foo.append(x)
            return x.clone()

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            return wrap(inner, x)

        x = torch.randn(3)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"HigherOrderOperator: Mutating a variable not in the current scope \(SideEffects\)",
        ):
            f(x)

    def test_no_freevars(self):
        def f(x):
            return wrap(lambda x: torch.sin(x), x)

        x = torch.randn(3)
        self._test_wrap_simple(f, default_args_generator((x,)), 2)

    def test_enum_arg(self):
        class SomeEnum(enum.Enum):
            A = 0
            B = 1

        def g(x, val):
            if val == SomeEnum.A:
                return torch.sin(x)
            return torch.cos(x)

        def f(x, val):
            return wrap(g, x, val)

        x = torch.randn(3)
        self._test_wrap_simple(f, default_args_generator((x, SomeEnum.A)), 2)
    def test_return_captured_var(self):
        # 创建一个包含三个随机数的张量
        freevar = torch.randn(3)

        def test(x):
            # 返回在 test 函数内捕获的自由变量 freevar
            return freevar

        def fn(x):
            # 将 test 函数包装起来，并返回包装后的函数
            return wrap(test, x)

        x = torch.randn(3)

        # 因为 `x` 没有被使用，所以不将其作为输入提升
        self._test_wrap_simple(fn, default_args_generator((x,)), 2)

    def test_return_captured_vars(self):
        # 创建三个随机数的张量作为自由变量
        freevar1 = torch.randn(3)
        freevar2 = torch.randn(3)

        def test(x):
            # 返回在 test 函数内捕获的自由变量 freevar1 和 freevar2
            return freevar1, freevar2, freevar1

        def fn(x):
            # 将 test 函数包装起来，并返回包装后的函数
            return wrap(test, x)

        x = torch.randn(3)

        # 因为 `x` 没有被使用，所以不将其作为输入提升
        self._test_wrap_simple(fn, default_args_generator((x,)), 3, 4)

    def test_return_captured_var_used_multiple_times(self):
        # 创建一个包含三个随机数的张量
        freevar = torch.randn(3)

        def test(x):
            # 在 test 函数内使用捕获的自由变量 freevar
            y = x + freevar
            return y, freevar

        def fn(x):
            # 将 test 函数包装起来，并返回包装后的函数
            return wrap(test, x)

        x = torch.randn(3)
        self._test_wrap_simple(fn, default_args_generator((x,)), 3, 3)

    def test_capture_untracked_global(self):
        def f(x):
            # 使用 lambda 函数捕获未被跟踪的全局变量 global_var
            return wrap(lambda x: x + global_var, x)

        x = torch.randn(3)
        self._test_wrap_simple(f, default_args_generator((x,)), 3)

    def test_symint_input(self):
        def f(x):
            # 计算输入张量 x 的大小，并将其作为 lambda 函数的参数之一
            i = x.size(0)
            return wrap(lambda x, i: x.view(i), x, i)

        x = torch.randn(3, 1)
        self._test_wrap_simple(
            f,
            default_args_generator((x,)),
            ifdynstaticdefault(2, 3),
            expected_opcount=ifdynstaticdefault(2, 3),
        )

    def test_wrap_pytree_args_nested(self):
        def f(x, y, z):
            def fn(d):
                # 使用输入字典 d 的 "x" 键和 "y" 键来进行数学运算
                return d["x"].sin() + d["y"][0].cos() - d["y"][1][2].sin()

            return wrap(fn, d)

        x = torch.tensor(1.5)
        y = torch.tensor(2.0)
        z = torch.tensor(3.0)
        d = {"x": x, "y": (y, [x, y, z])}

        def my_args_generator(t):
            # 生成器函数，产生多种参数组合
            yield t
            yield t[0] + 0.1, t[1], t[2]
            yield t[0], t[1] + 0.1, t[2]

        actual_graph = self._test_wrap_simple(
            f,
            my_args_generator((x, y, z)),
            4,
            return_graph=True,
        )
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    # 定义一个自定义的 PyTorch 模块 GraphModule
    def forward(self, L_d_x_: "f32[]", L_d_y_0_: "f32[]", L_d_y_1_2_: "f32[]"):
        # 定义 forward 方法，接受三个输入参数 L_d_x_, L_d_y_0_, L_d_y_1_2_
        
        # 将输入参数分别赋值给本地变量
        l_d_x_ = L_d_x_
        l_d_y_0_ = L_d_y_0_
        l_d_y_1_2_ = L_d_y_1_2_
        
        # 获取 wrap_body_0 属性
        wrap_body_0 = self.wrap_body_0
        # 调用 torch._higher_order_ops.wrap.wrap 方法，传入 wrap_body_0 和输入参数，得到 wrap 对象
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_d_x_, l_d_y_0_, l_d_y_1_2_);
        # 清空部分变量以释放内存
        wrap_body_0 = l_d_x_ = l_d_y_0_ = l_d_y_1_2_ = None
        # 从 wrap 对象中取得索引为 0 的元素
        getitem: "f32[]" = wrap[0];
        # 清空 wrap 对象以释放内存
        wrap = None
        # 返回包含 getitem 的元组
        return (getitem,)

    class GraphModule(torch.nn.Module):
        # 再次定义一个嵌套的 GraphModule 类，继承自 torch.nn.Module
        def forward(self, l_d_x_: "f32[]", l_d_y_0_: "f32[]", l_d_y_1_2_: "f32[]"):
            # 定义 forward 方法，接受三个输入参数 l_d_x_, l_d_y_0_, l_d_y_1_2_
            
            # 对 l_d_x_ 执行 sin 函数操作，结果存入 sin 变量
            sin: "f32[]" = l_d_x_.sin();
            # 清空 l_d_x_ 以释放内存
            l_d_x_ = None
            # 对 l_d_y_0_ 执行 cos 函数操作，结果存入 cos 变量
            cos: "f32[]" = l_d_y_0_.cos();
            # 清空 l_d_y_0_ 以释放内存
            l_d_y_0_ = None
            # 将 sin 和 cos 相加，结果存入 add 变量
            add: "f32[]" = sin + cos;
            # 清空 sin 和 cos 以释放内存
            sin = cos = None
            # 对 l_d_y_1_2_ 执行 sin 函数操作，结果存入 sin_1 变量
            sin_1: "f32[]" = l_d_y_1_2_.sin();
            # 清空 l_d_y_1_2_ 以释放内存
            l_d_y_1_2_ = None
            # 将 add 和 sin_1 相减，结果存入 sub 变量
            sub: "f32[]" = add - sin_1;
            # 清空 add 和 sin_1 以释放内存
            add = sin_1 = None
            # 返回包含 sub 的元组
            return (sub,)
    # 定义测试函数，测试 wrap 函数的 kwargs 功能
    def test_wrap_pytree_kwargs(self):
        # 定义内部函数 f，接受参数 x, y, z
        def f(x, y, z):
            # 定义内部函数 fn，使用关键字参数 x, y, z
            def fn(*, x, y, z):
                # 解包 z 成 z1 和 z2
                z1, z2 = z
                # 返回计算结果
                return (x * 2) + y + z1

            # 调用 wrap 函数，传入 fn 函数和参数 x, y, z
            return wrap(fn, x=x, y=y, z=z)

        # 生成随机张量 x 和形状为 (3, 3) 的随机张量 y
        x = torch.randn(3)
        y = torch.randn(3, 3)

        # 定义生成器函数 my_args_generator，用于生成参数元组
        def my_args_generator(t):
            yield t
            x1 = t[0] + 0.1
            y1 = t[1] + 0.1
            yield (x1, y1, (x1, y1))
            x2 = t[0] + 0.2
            y2 = t[0] + 0.2
            yield (x2, y2, (x2, y2))

        # 调用 _test_wrap_simple 方法，测试 f 函数
        self._test_wrap_simple(f, my_args_generator((x, y, (x, y))), 3)

    # 定义测试函数，测试 wrap 函数对非常量和符号整数张量的支持
    def test_wrap_pytree_args_not_const_symint_tensor(self):
        # 定义 MyClass 类
        class MyClass:
            def __init__(self, x):
                self.val = x

        # 定义内部函数 f，接受参数 x, y
        def f(x, y):
            # 使用 wrap 包装 lambda 函数，该函数计算张量 z[0] 的正弦值与张量 y 的余弦值的乘积
            return wrap(lambda z: z[0].sin() * z[1].val.cos(), (x, y))

        # 创建张量 x 和 MyClass 实例 y
        x = torch.tensor(1.2)
        y = MyClass(torch.tensor(3.4))
        # 调用 _test_wrap_simple 方法，测试 f 函数
        self._test_wrap_simple(f, [(x, y)], 3)

    # 定义测试函数，测试 wrap 函数对常量的捕获
    def test_capture_constants(self):
        # 生成随机张量 x 和常量 y
        x = torch.randn(3, 3)
        y = 4.0

        # 定义函数 fn，根据条件返回 x + y 或 x * y
        def fn(x, y, z):
            if z:
                return x + y
            return x * y

        # 定义内部函数 f，使用 wrap 包装 fn 函数
        def f(x, y, z):
            return wrap(fn, x, y, z)

        # 设置参数 args 为 (x, 4.0, None)，编译 f 函数
        args = (x, 4.0, None)
        opt_f = torch.compile(f, fullgraph=True, backend=CompileCounter())
        expected = f(*args)
        result = opt_f(*args)
        # 断言编译结果与预期结果相等
        self.assertEqual(result, expected)

        # 再次设置参数 args 为 (x, 5.0, None)，重新编译 f 函数
        args = (x, 5.0, None)
        expected = f(*args)
        result = opt_f(*args)
        # 断言编译结果与预期结果相等
        self.assertEqual(result, expected)

    # 定义测试函数，测试 wrap 函数对未跟踪全局变量的捕获
    def test_capture_untracked_global_nested(self):
        # 创建 EagerAndRecordGraphs 实例作为 backend，创建 CompileCounterWithBackend 实例 cnt
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        # 使用 @torch.compile 注解函数 f
        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            # 包装 lambda 函数，该函数内嵌包装另一 lambda 函数，计算 x + global_var
            return wrap(lambda x: wrap(lambda x: x + global_var, x), x)

        # 生成随机张量 x
        x = torch.randn(3)
        # 调用 f 函数
        result = f(x)

        # 断言结果等于 x + global_var
        self.assertEqual(result, x + global_var)
        # 断言编译帧数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言操作数为 2
        self.assertEqual(cnt.op_count, 2)

        # 断言 backend.graphs 的长度为 1
        self.assertEqual(len(backend.graphs), 1)
        # 查找第一个 wrap 节点
        wrap_node = find_first_node(backend.graphs[0], wrap)
        # 断言 inner_wrap_node 的参数长度为 3
        self.assertTrue(len(wrap_node.args), 3)

        # 获取 backend.graphs[0] 的 body_function 函数
        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        # 断言 body_function 操作数为 2
        self.assertEqual(op_count(body_function), 2)
        # 查找 body_function 中的第一个 wrap 节点
        inner_wrap_node = find_first_node(body_function, wrap)
        # 断言 inner_wrap_node 的参数长度为 3
        self.assertTrue(len(inner_wrap_node.args), 3)

    # 定义测试函数，测试 wrap 函数对未跟踪非局部变量的捕获
    def test_capture_untracked_nonlocal(self):
        # 生成随机张量 x 和 y
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        # 定义内部函数 f，接受参数 x, y
        def f(x, y):
            # 定义内部函数 g，使用 wrap 包装 lambda 函数，计算 x + y
            def g(x):
                return wrap(lambda x: x + y, x)

            # 调用 _test_wrap_simple 方法，测试 g 函数
            self._test_wrap_simple(g, default_args_generator((x,)), 3)
            # 返回调用 g 函数的结果
            return g(x)

        # 调用 f 函数
        f(x, y)

    # 定义测试函数，测试 wrap 函数对跟踪变量的捕获
    def test_capture_tracked(self):
        # 生成随机张量 x 和 y
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        # 定义内部函数 f，使用 wrap 包装 lambda 函数，计算 x + y
        def f(x, y):
            return wrap(lambda x: x + y, x)

        # 调用 _test_wrap_simple 方法，测试 f 函数
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)
    # 定义测试函数 test_capture_tracked_nested，测试嵌套函数的捕获行为
    def test_capture_tracked_nested(self):
        # 创建随机张量 x 和 y
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        # 定义函数 f，该函数通过 wrap 封装了一个双重 lambda 函数
        def f(x, y):
            # 内部的 lambda 函数捕获了外部的 y，并与外部的 x 进行 wrap
            return wrap(lambda x: wrap(lambda x: x + y, x), x)

        # 调用 _test_wrap_simple 方法来测试 f 函数的行为
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    # 定义测试函数 test_inlined_functions，测试内联函数的行为
    def test_inlined_functions(self):
        # 定义函数 g，简单地返回两个张量的加法结果
        def g(x, y):
            return x + y

        # 定义函数 f，通过 wrap 封装了一个 lambda 函数，调用了 g 函数
        def f(x, y):
            return wrap(lambda x: g(x, y), x)

        # 创建随机张量 x 和 y
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        # 调用 _test_wrap_simple 方法来测试 f 函数的行为
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    # 定义测试函数 test_same_freevar_twice，测试捕获相同自由变量两次的行为
    def test_same_freevar_twice(self):
        # 创建一个随机的张量 free
        free = torch.randn(3)

        # 定义函数 g，使用自由变量 free 计算其 sin 和 cos
        def g(x):
            y = free.sin()
            z = free.cos()
            return y, z

        # 定义函数 f，通过 wrap 封装了 g 函数
        def f(x):
            return wrap(g, x)

        # 创建一个随机张量 x
        x = torch.randn(3)

        # 调用 _test_wrap_simple 方法来测试 f 函数的行为
        # 期望返回值的数量为 2，由于 x 未被使用，不会作为输入进行提升
        self._test_wrap_simple(f, default_args_generator((x,)), 2, 3)

    # 定义测试函数 test_capture_value_created_in_subgraph，测试捕获子图中创建的值的行为
    def test_capture_value_created_in_subgraph(self):
        # 创建具有 EagerAndRecordGraphs 后端的 backend 和 CompileCounterWithBackend 对象
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        # 创建随机张量 x 和 y
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        # 定义内部函数 inner，捕获 x 和 y，计算 z = x + y，并在其内部创建一个双重 lambda 函数的 wrap
        def inner(x, y):
            z = x + y
            return wrap(lambda x: wrap(lambda x: x + z, x), x)

        # 使用 backend 和 fullgraph=True 装饰器定义函数 f，捕获 x 和 y，并将其传递给 inner 函数
        @torch.compile(backend=cnt, fullgraph=True)
        def f(x, y):
            return wrap(inner, x, y)

        # 调用 f 函数，将结果与预期的 x + y + x 进行比较
        result = f(x, y)

        # 断言结果与预期的 x + y + x 相等
        self.assertEqual(result, x + y + x)

        # 断言 CompileCounterWithBackend 对象的帧计数为 1
        self.assertEqual(cnt.frame_count, 1)

        # 断言 CompileCounterWithBackend 对象的操作计数为 2
        self.assertEqual(cnt.op_count, 2)

        # 断言 backend 中的图的数量为 1
        self.assertEqual(len(backend.graphs), 1)

        # 断言外部 wrap 的参数未发生变化
        gm = backend.graphs[0]
        wrap_node = find_first_node(gm, wrap)
        self.assertTrue(len(wrap_node.args), 3)

        # 断言 z 已经被提升为内部 wrap 的参数
        body_function = getattr(gm, wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 3)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

        # 内部函数体：z 也被提升为参数
        body_function = getattr(body_function, inner_wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

    # 定义测试函数 test_side_effect_set_new_attr_global_obj，测试设置全局对象的新属性的副作用行为
    def test_side_effect_set_new_attr_global_obj(self):
        # 定义 setup 函数，初始化全局对象 global_obj
        def setup():
            global global_obj
            global_obj = Obj()

        # 定义函数 f，内部函数 h 和 g 捕获了全局对象 global_obj，并进行了修改和使用
        def f(x):
            def h(x):
                def g(x):
                    global_obj.foo = x + 1
                    return x.clone()

                # 使用 wrap 封装函数 g，并返回其结果与 global_obj.foo 的和
                y = wrap(g, x)
                return y + global_obj.foo

            return h(x)

        # 创建一个张量 x
        x = torch.zeros([])

        # 调用 _assert_wrap_fallback 方法来测试 f 函数的行为，传入张量 x 和 setup 函数
        self._assert_wrap_fallback(f, (x,), setup=setup)
    # 测试函数：test_side_effect_set_existing_attr_global_obj
    def test_side_effect_set_existing_attr_global_obj(self):
        # 设置初始化环境函数
        def setup():
            # 声明全局对象 global_obj，并赋值为 Obj 类的实例
            global global_obj
            global_obj = Obj()
            # 设置 global_obj 的属性 foo 为一个 Torch 参数张量值为 4.0
            global_obj.foo = nn.Parameter(torch.tensor(4.0))

        # 定义函数 f，接受参数 x
        def f(x):
            # 定义函数 h，接受参数 x
            def h(x):
                # 定义函数 g，接受参数 x
                def g(x):
                    # 设置全局对象 global_obj 的属性 foo 为 x + 1
                    global_obj.foo = x + 1
                    # 返回 x 的克隆
                    return x.clone()

                # 调用 wrap 函数，将 g 函数包装起来，传入参数 x，得到结果 y
                y = wrap(g, x)
                # 返回 y 加上 global_obj.foo 的结果
                return y + global_obj.foo

            # 调用 h 函数，传入参数 x，并返回其结果
            return h(x)

        # 初始化参数 x 为一个零维张量
        x = torch.zeros([])
        # 调用 self._assert_wrap_fallback 方法，传入函数 f 和参数元组 (x,)，并传入 setup 函数
        self._assert_wrap_fallback(f, (x,), setup=setup)

    # 测试函数：test_side_effect_del_existing_attr_global_obj
    def test_side_effect_del_existing_attr_global_obj(self):
        # 设置初始化环境函数
        def setup():
            # 声明全局对象 global_obj，并赋值为 Obj 类的实例
            global global_obj
            global_obj = Obj()
            # 设置 global_obj 的属性 foo 为一个 Torch 张量，值为 4.0
            global_obj.foo = torch.tensor(4.0)

        # 定义函数 f，接受参数 x
        def f(x):
            # 定义函数 h，接受参数 x
            def h(x):
                # 定义函数 g，接受参数 x
                def g(x):
                    # 删除全局对象 global_obj 的属性 foo
                    del global_obj.foo
                    # 返回 x 的克隆
                    return x.clone()

                # 调用 wrap 函数，将 g 函数包装起来，传入参数 x，得到结果 y
                y = wrap(g, x)
                # 返回 y 的结果
                return y

            # 调用 h 函数，传入参数 x，并返回其结果
            return h(x)

        # 初始化参数 x 为一个零维张量
        x = torch.zeros([])
        # 调用 self._assert_wrap_fallback 方法，传入函数 f 和参数元组 (x,)，并传入 setup 函数
        self._assert_wrap_fallback(f, (x,), setup=setup)

    # 测试函数：test_side_effect_set_new_attr_global_module
    def test_side_effect_set_new_attr_global_module(self):
        # 设置初始化环境函数
        def setup():
            # 声明全局模块 global_module，并赋值为 MyModule 的实例
            global global_module
            global_module = MyModule()

        # 定义函数 h，接受参数 x
        def h(x):
            # 定义函数 g，接受参数 x
            def g(x):
                # 设置全局模块 global_module 的属性 foo 为一个 Torch 参数张量，其值为 x + 1
                global_module.foo = nn.Parameter(x + 1)
                # 返回 x 的克隆
                return x.clone()

            # 调用 wrap 函数，将 g 函数包装起来，传入参数 x，得到结果 y
            y = wrap(g, x)
            # 返回 y 加上 global_module.foo 的结果
            return y + global_module.foo

        # 初始化参数 x 为一个零维张量
        x = torch.zeros([])
        # 调用 self._assert_wrap_fallback 方法，传入函数 h 和参数元组 (x,)，并传入 setup 函数
        self._assert_wrap_fallback(h, (x,), setup=setup)

    # 测试函数：test_side_effect_set_existing_attr_global_module
    def test_side_effect_set_existing_attr_global_module(self):
        # 设置初始化环境函数
        def setup():
            # 声明全局模块 global_module，并赋值为 MyModule 的实例
            global global_module
            global_module = MyModule()

        # 定义函数 h，接受参数 x
        def h(x):
            # 定义函数 g，接受参数 x
            def g(x):
                # 设置全局模块 global_module 的属性 existing 为一个 Torch 参数张量，其值为 4.0
                global_module.existing = nn.Parameter(torch.tensor(4.0))
                # 调用 global_module 的方法，传入参数 x，并返回结果
                return global_module(x)

            # 调用 wrap 函数，将 g 函数包装起来，传入参数 x，得到结果 y
            y = wrap(g, x)
            # 返回 y 的结果
            return y

        # 初始化参数 x 为一个零维张量
        x = torch.zeros([])
        # 调用 self._assert_wrap_fallback 方法，传入函数 h 和参数元组 (x,)，并传入 setup 函数
        self._assert_wrap_fallback(h, (x,), setup=setup)

    # 测试函数：test_side_effect_del_existing_attr_global_module
    def test_side_effect_del_existing_attr_global_module(self):
        # 设置初始化环境函数
        def setup():
            # 声明全局模块 global_module，并赋值为 MyModule 的实例
            global global_module
            global_module = MyModule()

        # 定义函数 h，接受参数 x
        def h(x):
            # 定义函数 g，接受参数 x
            def g(x):
                # 删除全局模块 global_module 的属性 existing
                del global_module.existing
                # 返回 x 的克隆
                return x.clone()

            # 调用 wrap 函数，将 g 函数包装起来，传入参数 x，得到结果 y
            y = wrap(g, x)
            # 返回 y 的结果
            return y

        # 初始化参数 x 为一个零维张量
        x = torch.zeros([])
        # 调用 self._assert_wrap_fallback 方法，传入函数 h 和参数元组 (x,)，并传入 setup 函数
        self._assert_wrap_fallback(h, (x,), setup=setup)

    # 测试函数：test_side_effect_mutate_global_num
    def test_side_effect_mutate_global_num(self):
        # 设置初始化环境函数
        def setup():
            # 声明全局变量 global_num，并赋值为 3.14
            global global_num
            global_num = 3.14

        # 定义函数 f，接受参数 x
        def f(x):
            # 定义函数 g，接受参数 x
            def g(x):
                # 声明全局变量 global_num，并对其进行加 1 操作
                global global_num
                global_num = global_num + 1
                # 返回 x 加上 global_num 的结果
                return x + global_num

            # 调用 wrap 函数，将 g 函数包装起来，传入参数 x，得到结果 y
            y = wrap(g, x)
            # 返回 y 加上 global_num 的结果
            return y + global_num

        # 初始化参数 x 为一个零维张量
        x = torch.zeros([])
        # 调用 self._assert_wrap_fallback 方法，传入函数 f 和参数元组 (x,)，并传入 setup 函数
        self._assert_wrap_fallback(f, (x,), setup=setup)
    # 测试函数：test_side_effect_mutate_global_num_builtin
    def test_side_effect_mutate_global_num_builtin(self):
        # 设置全局变量 global_num 为 3.14
        def setup():
            global global_num
            global_num = 3.14
        
        # 定义函数 f，接受参数 x
        def f(x):
            # 定义函数 g，接受参数 x
            def g(x):
                # 引用全局变量 global_num，并递增其值
                global global_num
                global_num += 1
                # 返回 x 与 global_num 相加的结果
                return x + global_num
            
            # 调用 wrap 函数，将 g 函数包装，传递参数 x
            y = wrap(g, x)
            # 返回 y 与 global_num 相加的结果
            return y + global_num
        
        # 创建一个空的 torch 张量 x
        x = torch.zeros([])
        # 调用 _assert_wrap_fallback 函数，传递函数 f 和参数 x，并设置初始化函数 setup
        self._assert_wrap_fallback(f, (x,), setup=setup)

    # 测试函数：test_side_effect_mutate_global_tensor
    def test_side_effect_mutate_global_tensor(self):
        # 设置全局变量 global_var 为包含三个元素的全 1 张量
        def setup():
            global global_var
            global_var = torch.ones(3)
        
        # 定义函数 f，接受参数 x
        def f(x):
            # 定义函数 g，接受参数 x
            def g(x):
                # 引用全局变量 global_var，并将其每个元素加 1
                global global_var
                global_var = global_var + 1
                # 返回 x 与 global_var 相加的结果
                return x + global_var
            
            # 调用 wrap 函数，将 g 函数包装，传递参数 x
            y = wrap(g, x)
            # 返回 y 与 global_var 相加的结果
            return y + global_var
        
        # 创建一个空的 torch 张量 x
        x = torch.zeros([])
        # 调用 _assert_wrap_fallback 函数，传递函数 f 和参数 x，并设置初始化函数 setup
        self._assert_wrap_fallback(f, (x,), setup=setup)

    # 测试函数：test_side_effect_mutate_global_tensor_builtin
    def test_side_effect_mutate_global_tensor_builtin(self):
        # 设置全局变量 global_var 为包含三个元素的全 1 张量
        def setup():
            global global_var
            global_var = torch.ones(3)
        
        # 定义函数 f，接受参数 x
        def f(x):
            # 定义函数 g，接受参数 x
            def g(x):
                # 引用全局变量 global_var，并将其每个元素加 1
                global global_var
                global_var += 1
                # 返回 x 与 global_var 相加的结果
                return x + global_var
            
            # 调用 wrap 函数，将 g 函数包装，传递参数 x
            y = wrap(g, x)
            # 返回 y 与 global_var 相加的结果
            return y + global_var
        
        # 创建一个空的 torch 张量 x
        x = torch.zeros([])
        # 调用 _assert_wrap_fallback 函数，传递函数 f 和参数 x，并设置初始化函数 setup
        self._assert_wrap_fallback(f, (x,), setup=setup)

    # 测试函数：test_side_effect_mutate_global_list
    def test_side_effect_mutate_global_list(self):
        # 设置全局变量 global_list 为空列表
        def setup():
            global global_list
            global_list = []
        
        # 定义函数 f，接受参数 x
        def f(x):
            # 定义函数 g，接受参数 x
            def g(x):
                # 将 x 加 1，并将结果添加到全局列表 global_list 中
                val = x + 1
                global_list.append(val)
                # 返回 global_list 的最后一个元素
                return global_list[-1]
            
            # 调用 wrap 函数，将 g 函数包装，传递参数 x
            y = wrap(g, x)
            # 计算 y 与 global_list 的最后一个元素之和
            z = y + global_list[-1]
            return z
        
        # 创建一个空的 torch 张量 x
        x = torch.zeros([])
        # 调用 _assert_wrap_fallback 函数，传递函数 f 和参数 x，并设置初始化函数 setup
        self._assert_wrap_fallback(f, (x,), setup=setup)

    # 测试函数：test_side_effect_mutate_nonlocal_num
    def test_side_effect_mutate_nonlocal_num(self):
        # 定义函数 f，接受参数 x
        def f(x):
            # 定义函数 h，接受参数 x
            def h(x):
                # 设置局部变量 val 为 1
                val = 1
                
                # 定义函数 g，接受参数 x
                def g(x):
                    # 引用外层函数 h 的局部变量 val，并将其加 1
                    nonlocal val
                    val = val + 1
                    # 返回 x 与 val 相加的结果
                    return x + val
                
                # 调用 wrap 函数，将 g 函数包装，传递参数 x
                y = wrap(g, x)
                # 计算 y 与 val 的和
                z = y + val
                return z
            
            # 调用 h 函数，传递参数 x
            return h(x)
        
        # 创建一个空的 torch 张量 x
        x = torch.zeros([])
        # 调用 _assert_wrap_fallback 函数，传递函数 f 和参数 x
        self._assert_wrap_fallback(f, (x,))

    # 测试函数：test_side_effect_set_new_attr_nonlocal_obj
    def test_side_effect_set_new_attr_nonlocal_obj(self):
        # 定义函数 f，接受参数 x
        def f(x):
            # 定义函数 h，接受参数 x
            def h(x):
                # 创建一个 Obj 对象
                obj = Obj()
                
                # 定义函数 g，接受参数 x
                def g(x):
                    # 设置 obj 的 val 属性为 x 的维度
                    obj.val = x.dim()
                    # 返回 x 的克隆
                    return x.clone()
                
                # 调用 wrap 函数，将 g 函数包装，传递参数 x
                y = wrap(g, x)
                # 计算 y 与 obj.val 的和
                z = y + obj.val
                return z
            
            # 调用 h 函数，传递参数 x
            return h(x)
        
        # 创建一个空的 torch 张量 x
        x = torch.zeros([])
        # 调用 _assert_wrap_fallback 函数，传递函数 f 和参数 x
        self._assert_wrap_fallback(f, (x,))
    # 定义一个测试函数，测试在设置现有属性时是否会对非局部对象产生副作用
    def test_side_effect_set_existing_attr_nonlocal_obj(self):
        # 定义函数 f，接受参数 x
        def f(x):
            # 定义函数 h，接受参数 x
            def h(x):
                # 创建一个 Obj 对象
                obj = Obj()
                # 设置对象的属性 val 为 3
                obj.val = 3

                # 定义函数 g，接受参数 x
                def g(x):
                    # 设置对象的属性 val 为 x.dim() 的结果
                    obj.val = x.dim()
                    # 克隆并返回 x
                    return x.clone()

                # 调用 wrap 函数，将 g 函数和 x 作为参数传入，得到 y
                y = wrap(g, x)
                # 计算 y 和 obj.val 的和，将结果赋给 z
                z = y + obj.val
                return z

            # 调用 h 函数，将 x 作为参数传入，返回其结果
            return h(x)

        # 创建一个 torch 的空张量 x
        x = torch.zeros([])
        # 调用 _assert_wrap_fallback 方法，传入 f 函数和参数 x
        self._assert_wrap_fallback(f, (x,))

    # 定义一个测试函数，测试在删除现有属性时是否会对非局部对象产生副作用
    def test_side_effect_del_existing_attr_nonlocal_obj(self):
        # 定义函数 f，接受参数 x
        def f(x):
            # 定义函数 h，接受参数 x
            def h(x):
                # 创建一个 Obj 对象
                obj = Obj()
                # 设置对象的属性 val 为 3
                obj.val = 3

                # 定义函数 g，接受参数 x
                def g(x):
                    # 删除对象的属性 val
                    del obj.val
                    # 克隆并返回 x
                    return x.clone()

                # 调用 wrap 函数，将 g 函数和 x 作为参数传入，得到 y
                y = wrap(g, x)
                # 返回 y
                return y

            # 调用 h 函数，将 x 作为参数传入，返回其结果
            return h(x)

        # 创建一个 torch 的空张量 x
        x = torch.zeros([])
        # 调用 _assert_wrap_fallback 方法，传入 f 函数和参数 x
        self._assert_wrap_fallback(f, (x,))

    # 定义一个测试函数，测试在设置新属性时是否会对非局部模块对象产生副作用
    def test_side_effect_set_new_attr_nonlocal_module(self):
        # 定义函数 h，接受参数 x
        def h(x):
            # 创建一个 MyModule 对象
            obj = MyModule()

            # 定义函数 g，接受参数 x
            def g(x):
                # 设置模块对象的属性 val 为 x.dim() 的结果
                obj.val = x.dim()
                # 克隆并返回 x
                return x.clone()

            # 调用 wrap 函数，将 g 函数和 x 作为参数传入，得到 y
            y = wrap(g, x)
            # 计算 y 和 obj.val 的和，将结果赋给 z
            z = y + obj.val
            return z

        # 创建一个 torch 的空张量 x
        x = torch.zeros([])
        # 调用 _assert_wrap_fallback 方法，传入 h 函数和参数 x
        self._assert_wrap_fallback(h, (x,))

    # 定义一个测试函数，测试在设置现有属性时是否会对非局部模块对象产生副作用
    def test_side_effect_set_existing_attr_nonlocal_module(self):
        # 定义函数 h，接受参数 x
        def h(x):
            # 创建一个 MyModule 对象
            obj = MyModule()

            # 定义函数 g，接受参数 x
            def g(x):
                # 设置模块对象的现有属性 existing 为一个 torch.tensor(3.14) 的 nn.Parameter
                obj.existing = nn.Parameter(torch.tensor(3.14))
                # 对模块对象 obj 调用，返回其结果
                return obj(x)

            # 调用 wrap 函数，将 g 函数和 x 作为参数传入，得到 y
            y = wrap(g, x)
            # 返回 y
            return y

        # 创建一个 torch 的空张量 x
        x = torch.zeros([])
        # 调用 _assert_wrap_fallback 方法，传入 h 函数和参数 x
        self._assert_wrap_fallback(h, (x,))

    # 定义一个测试函数，测试在删除现有属性时是否会对非局部模块对象产生副作用
    def test_side_effect_del_existing_attr_nonlocal_module(self):
        # 定义函数 h，接受参数 x
        def h(x):
            # 创建一个 MyModule 对象
            obj = MyModule()

            # 定义函数 g，接受参数 x
            def g(x):
                # 删除模块对象的属性 existing
                del obj.existing
                # 克隆并返回 x
                return x.clone()

            # 调用 wrap 函数，将 g 函数和 x 作为参数传入，得到 y
            y = wrap(g, x)
            # 返回 y
            return y

        # 创建一个 torch 的空张量 x
        x = torch.zeros([])
        # 调用 _assert_wrap_fallback 方法，传入 h 函数和参数 x
        self._assert_wrap_fallback(h, (x,))

    # 定义一个测试函数，测试在更改非局部张量时是否会产生副作用
    def test_side_effect_mutate_nonlocal_tensor(self):
        # 定义函数 f，接受参数 x
        def f(x):
            # 定义函数 h，接受参数 x
            def h(x):
                # 创建一个值为 1 的张量 val
                val = torch.tensor(1.0)

                # 定义函数 g，接受参数 x
                def g(x):
                    # 声明 val 为非局部变量，将 val 加上 1
                    nonlocal val
                    val = val + 1
                    # 返回 x 和增加后的 val 的和
                    return x + val

                # 调用 wrap 函数，将 g 函数和 x 作为参数传入，得到 y
                y = wrap(g, x)
                # 计算 y 和 val 的和，将结果赋给 z
                z = y + val
                return z

            # 调用 h 函数，将 x 作为参数传入，返回其结果
            return h(x)

        # 创建一个 torch 的空张量 x
        x = torch.zeros([])
        # 调用 _assert_wrap_fallback 方法，传入 f 函数和参数 x
        self._assert_wrap_fallback(f, (x,))

    # 定义一个测试函数，测试在更改非局部内置数字时是否会产生副作用
    def test_side_effect_mutate_nonlocal_num_builtin(self):
        # 定义函数 f，接受参数 x
        def f(x):
            # 定义函数 h，接受参数 x
            def h(x):
                # 创建一个值为 1 的数字 val
                val = 1

                # 定义函数 g，接受参数 x
                def g(x):
                    # 声明 val 为非局部变量，将 val 加上 1
                    nonlocal val
                    val += 1
                    # 返回 x 和增加后的 val 的和
                    return x + val

                # 调用 wrap 函数，将 g 函数和 x 作为参数传入，得到 y
                y = wrap(g, x)
                # 计算 y 和 val 的和，将结果赋给 z
                z = y + val
                return z

            # 调用 h 函数，将 x 作为参数传入，返回其结果
            return h(x)

        # 创建一个 torch 的空张量 x
        x = torch.zeros([])
        # 调用 _assert_wrap_fallback 方法，传入 f 函数和参数 x
        self._assert_wrap_fallback(f, (x,))
    # 测试函数，验证非局部张量和内置函数副作用
    def test_side_effect_mutate_nonlocal_tensor_builtin(self):
        # 内部函数，接受参数 x
        def f(x):
            # 内部函数 h，接受参数 x
            def h(x):
                # 创建一个值为 1.0 的张量 val
                val = torch.tensor(1.0)

                # 内部函数 g，接受参数 x
                def g(x):
                    nonlocal val  # 使用外部函数 h 中的 val 变量
                    val += 1  # 对 val 进行加法操作
                    return x + val  # 返回 x 与 val 的和

                # 使用 wrap 函数对 g 进行封装，传入参数 x，返回值为 y
                y = wrap(g, x)
                # 计算 z 为 y 与 val 的和
                z = y + val
                return z

            # 调用内部函数 h，传入参数 x，返回结果
            return h(x)

        # 创建一个形状为空的零张量 x
        x = torch.zeros([])
        # 调用 self._assert_wrap_fallback 函数，传入 f 和参数元组 (x,)
        self._assert_wrap_fallback(f, (x,))

    # 测试函数，验证非局部列表添加操作并破坏图
    def test_side_effect_nonlocal_list_append_graph_break(self):
        # 内部函数 g，接受参数 x
        def g(x):
            # 创建一个空列表 y
            y = []

            # 内部函数 f，接受参数 k
            def f(k):
                # 创建一个变量 m，其值为 k + 1
                m = k + 1
                # 向列表 y 中添加 m
                y.append(m)
                # 返回 k
                return k

            # 使用 wrap 函数对 f 进行封装，传入参数 x
            wrap(f, x)
            # 返回列表 y 中的第一个元素
            return y[0]

        # 创建一个形状为 (3, 3) 的随机张量 x
        x = torch.randn(3, 3)
        # 调用 self._assert_wrap_fallback 函数，传入 g 和参数元组 (x,)
        self._assert_wrap_fallback(g, (x,))

    # 测试函数，验证嵌套非局部列表添加操作并破坏图
    def test_side_effect_nested_nonlocal_list_append_graph_break(self):
        # 内部函数 g，接受参数 x
        def g(x):
            # 内部函数 h，接受参数 x
            def h(x):
                # 创建一个空列表 y
                y = []

                # 内部函数 f，接受参数 k
                def f(k):
                    # 创建一个变量 m，其值为 k + 1
                    m = k + 1
                    # 向列表 y 中添加 m
                    y.append(m)
                    # 返回 k
                    return k

                # 使用 wrap 函数对 f 进行封装，传入参数 x
                wrap(f, x)
                # 返回列表 y 中的第一个元素
                return y[0]

            # 调用内部函数 h，传入参数 x，返回结果
            return h(x)

        # 创建一个形状为 (3, 3) 的随机张量 x
        x = torch.randn(3, 3)
        # 调用 self._assert_wrap_fallback 函数，传入 g 和参数元组 (x,)
        self._assert_wrap_fallback(g, (x,))

    # 测试函数，验证局部列表添加操作不会破坏图
    def test_side_effect_local_list_append_no_graph_break(self):
        # 内部函数 g，接受参数 x
        def g(x):
            # 内部函数 f，接受参数 k
            def f(k):
                # 创建一个空列表 y
                y = []
                # 向列表 y 中添加 k + 1
                y.append(k + 1)
                # 返回列表 y 的第一个元素
                return y[0]

            # 使用 wrap 函数对 f 进行封装，传入参数 x
            return wrap(f, x)

        # 创建一个形状为 (3, 3) 的随机张量 x
        x = torch.randn(3, 3)
        # 调用 self._test_wrap_simple 函数，传入 g、default_args_generator((x,)) 和 2 作为参数
        self._test_wrap_simple(g, default_args_generator((x,)), 2)

    # 测试函数，验证使用关键字参数的 wrap 函数调用
    def test_wrap_kwarg(self):
        # 内部函数 f，接受参数 x 和 y
        def f(x, y):
            # 使用 wrap 函数对 lambda 函数进行封装，传入 x 和 y 作为参数
            return wrap(lambda x, y: x + y, x, y=y)

        # 创建一个形状为 (3,) 的随机张量 x
        x = torch.randn(3)
        # 创建一个形状为 (3, 3) 的随机张量 y
        y = torch.randn(3, 3)
        # 调用 self._test_wrap_simple 函数，传入 f、default_args_generator((x, y)) 和 3 作为参数
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    # 测试函数，验证使用关键字参数和整数作为 y 的 wrap 函数调用
    def test_wrap_kwarg_int(self):
        # 内部函数 f，接受参数 x 和 y
        def f(x, y):
            # 使用 wrap 函数对 lambda 函数进行封装，传入 x 和 y 作为参数
            return wrap(lambda x, y: x + y, x, y=y)

        # 创建一个形状为 (3,) 的随机张量 x
        x = torch.randn(3)
        # 创建一个整数值为 8 的 y
        y = 8

        # 调用 self._test_wrap_simple 函数，
        # 传入 f、default_args_generator((x, y)) 和 ifdynstaticdefault(2, 3) 作为参数
        self._test_wrap_simple(f, default_args_generator((x, y)), ifdynstaticdefault(2, 3))

    # 测试函数，验证使用所有关键字参数的 wrap 函数调用
    def test_wrap_all_kwarg(self):
        # 内部函数 f，接受参数 y 和 x
        def f(y, x):
            # 使用 wrap 函数对 lambda 函数进行封装，传入 x 和 y 作为参数
            return wrap(lambda x, y: (x * 2) + y, x=x, y=y)

        # 创建一个形状为 (3,) 的随机张量 x
        x = torch.randn(3)
        # 创建一个形状为 (3, 3) 的随机张量 y
        y = torch.randn(3, 3)

        # 调用 self._test_wrap_simple 函数，传入 f、default_args_generator((x, y)) 和 3 作为参数
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    # 测试函数，验证使用仅关键字参数的 wrap 函数调用
    def test_wrap_kwarg_only(self):
        # 内部函数 f，接受参数 x 和 y
        def f(x, y):
            # 内部函数 fn，使用关键字参数 x 和 y，返回结果 (x * 2) + y
            def fn(*, x, y):
                return (x * 2) + y

            # 使用 wrap 函数对 fn 进行封装，传入 x 和 y 作为关键字参数
            return wrap(fn, x=x, y=y)

        # 创建一个形状为 (3,) 的随机张量 x
        x = torch.randn(3)
        # 创建一个形状为 (3, 3) 的随机张量 y
        y = torch.randn(3, 3)

        # 调用 self._test_wrap_simple 函数，传入 f、default_args_generator((x, y)) 和 3 作为参数
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    # 测试函数，验证使用带默认值的关键字参数的 wrap 函数调用
    def test_wrap_kwarg_default(self):
        # 内部函数 f，接受参数 x 和 y
        def f(x, y):
            # 内部函数 fn，使用关键字参数 x、y 和默认值 z
    def test_wrap_kwarg_default_if_branch(self):
        # 定义一个函数 f，接受参数 x 和 y
        def f(x, y):
            # 定义一个函数 fn，使用命名关键字参数 x, y, z=None
            def fn(*, x, y, z=None):
                # 如果 z 为 None，则返回 (x * 2) + y
                if z is None:
                    return (x * 2) + y
                else:
                    # 否则返回 2 * x
                    return 2 * x

            # 调用 wrap 函数，将 fn 函数包装，使用 x 和 y 作为参数
            return wrap(fn, x=x, y=y)

        # 生成一个长度为 3 的随机张量 x
        x = torch.randn(3)
        # 生成一个 3x3 的随机张量 y
        y = torch.randn(3, 3)

        # 调用 _test_wrap_simple 方法，测试包装后的函数 f，使用生成的默认参数元组 (x, y)，预期结果为 3
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_wrap_kwarg_recompile(self):
        # 定义一个函数 f，接受参数 x, y, z
        def f(x, y, z=None):
            # 定义一个函数 fn，使用命名关键字参数 x, y, z=None
            def fn(*, x, y, z=None):
                # 如果 z 为 None，则返回 (x * 2) + y
                if z is None:
                    return (x * 2) + y
                else:
                    # 否则返回 2 * x
                    return 2 * x

            # 调用 wrap 函数，将 fn 函数包装，使用 x, y, z 作为参数
            return wrap(fn, x=x, y=y, z=z)

        # 生成一个长度为 3 的随机张量 x
        x = torch.randn(3)
        # 生成一个 3x3 的随机张量 y
        y = torch.randn(3, 3)

        # 清除计数器
        counters.clear()
        # 编译函数 f，使用 eager 后端，完整图形模式
        opt = torch.compile(f, backend="eager", fullgraph=True)
        # 调用编译后的函数 opt，传入参数 x, y
        opt(x, y)
        # 断言调用计数器中捕获的调用数为 2
        self.assertEqual(counters["stats"]["calls_captured"], 2)

        # 验证不会重新编译
        opt(x, y)
        # 断言调用计数器中捕获的调用数仍为 2
        self.assertEqual(counters["stats"]["calls_captured"], 2)

        # 调用 opt 函数，传入参数 x, y, 8
        output = opt(x, y, 8)
        # 断言调用计数器中捕获的调用数为 4
        self.assertEqual(counters["stats"]["calls_captured"], 4)
        # 断言输出结果为 2 * x
        self.assertEqual(output, 2 * x)

    def test_wrap_kwarg_default_else_branch(self):
        # 定义一个函数 f，接受参数 x, y, z
        def f(x, y, z):
            # 定义一个函数 fn，使用命名关键字参数 x, y, z=None
            def fn(*, x, y, z=None):
                # 如果 z 为 None，则返回 (x * 2) + y
                if z is None:
                    return (x * 2) + y
                else:
                    # 否则返回 2 * x
                    return 2 * x

            # 调用 wrap 函数，将 fn 函数包装，使用 x, y, z 作为参数
            return wrap(fn, x=x, y=y, z=z)

        # 生成一个长度为 3 的随机张量 x
        x = torch.randn(3)
        # 生成一个 3x3 的随机张量 y
        y = torch.randn(3, 3)

        # 调用 _test_wrap_simple 方法，测试包装后的函数 f，使用生成的默认参数元组 (x, y, 8)，预期结果为 2
        self._test_wrap_simple(f, default_args_generator((x, y, 8)), 2)

    def test_map_subgraph_name_is_valid(self):
        # 创建一个 EagerAndRecordGraphs 后端
        backend = EagerAndRecordGraphs()
        # 创建一个 CompileCounterWithBackend 计数器
        cnt = CompileCounterWithBackend(backend)

        # 生成一个形状为 (2, 3, 3) 的随机张量 xs
        xs = torch.randn(2, 3, 3)
        # 生成一个长度为 3 的随机张量 y
        y = torch.randn(3)

        # 定义一个函数 map_f，接受参数 xs, y
        def map_f(xs, y):
            # 定义一个函数 inner，接受参数 x, y
            def inner(x, y):
                # 定义一个函数 inner2，接受参数 x, y
                def inner2(x, y):
                    # 返回 x + y
                    return x + y

                # 使用 control_flow.map 调用 inner2 函数，对输入的 x, y 进行映射处理
                return control_flow.map(inner2, x, y)

            # 使用 control_flow.map 调用 inner 函数，对输入的 xs, y 进行映射处理
            return control_flow.map(inner, xs, y)

        # 调用 _check_map_graph_and_extract 方法，检查并提取 map_f 函数的图形，使用参数元组 (xs, y)
        graphs = self._check_map_graph_and_extract(map_f, (xs, y))
        # 如果检查到图形
        if graphs:
            # 提取 graph 和 body_graph
            graph, body_graph = graphs
            # 断言内联的预期结果与 graph 相符
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_xs_ : torch.Tensor, L_y_ : torch.Tensor):
    # 将输入参数重命名为局部变量
    l_xs_ = L_xs_
    l_y_ = L_y_
    # 获取 self 对象中的 map_body_1 属性
    map_body_1 = self.map_body_1
    # 调用外部库中的高阶函数实现，并传入参数
    map_impl = torch.ops.higher_order.map_impl(map_body_1, [l_xs_], [l_y_]);  map_body_1 = l_xs_ = l_y_ = None
    # 从返回结果中获取特定索引的元素
    getitem_1 = map_impl[0];  map_impl = None
    # 返回一个元组，包含上述获取的元素
    return (getitem_1,)

def test_map_multi_return(self):
    cnt = CompileCounter()

    def f(x):
        # 使用控制流函数映射 x 中的每个元素，并返回一个元组
        return control_flow.map(lambda x: (x.sin(), x.sin()), x)

    x = torch.randn(3)
    # 检查映射函数生成的图形，并提取其中的两个图形
    graphs = self._check_map_graph_and_extract(f, (x,))
    if graphs:
        graph, body_graph = graphs
        # 断言 graph 函数的预期输出
        self.assertExpectedInline(
            graph,
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    # 获取 self 对象中的 map_body_0 属性
    map_body_0 = self.map_body_0
    # 调用外部库中的高阶函数实现，并传入参数
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [l_x_], []);  map_body_0 = l_x_ = None
    # 从返回结果中获取特定索引的元素
    getitem_1 = map_impl[0]
    getitem_2 = map_impl[1];  map_impl = None
    # 返回一个元组，包含上述获取的元素
    return (getitem_1, getitem_2)""",
        )
        # 断言 body_graph 函数的预期输出
        self.assertExpectedInline(
            body_graph,
            """\
def forward(self, getitem):
    # 对获取的元素执行数学函数 sin()
    sin = getitem.sin()
    sin_1 = getitem.sin();  getitem = None
    # 返回一个元组，包含上述执行结果
    return (sin, sin_1)""",
        )

def test_map_pytree_return(self):
    cnt = CompileCounter()

    def _construct_pytree(a):
        # 构造一个复杂的 Python 树结构
        return (a, [[[a]]], a, (a, (a,), a), {"a": a})

    def f(x):
        def inner_f(xs):
            # 调用内部函数构造一个复杂的 Python 树结构
            return _construct_pytree(xs)

        # 使用控制流函数映射 x 中的每个元素，并返回一个复杂的 Python 树结构
        return control_flow.map(inner_f, x)

    x = torch.randn(3)
    # 检查映射函数生成的图形，并提取其中的两个图形
    graphs = self._check_map_graph_and_extract(f, (x,))
    if graphs:
        graph, body_graph = graphs
        # 断言 graph 函数的预期输出
        self.assertExpectedInline(
            graph,
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    # 获取 self 对象中的 map_body_0 属性
    map_body_0 = self.map_body_0
    # 调用外部库中的高阶函数实现，并传入参数
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [l_x_], []);  map_body_0 = l_x_ = None
    # 从返回结果中获取特定索引的元素
    getitem_1 = map_impl[0]
    getitem_2 = map_impl[1]
    getitem_3 = map_impl[2]
    getitem_4 = map_impl[3]
    getitem_5 = map_impl[4]
    getitem_6 = map_impl[5]
    getitem_7 = map_impl[6];  map_impl = None
    # 返回一个元组，包含上述获取的元素
    return (getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7)""",
        )
        # 断言 body_graph 函数的预期输出
        self.assertExpectedInline(
            body_graph,
            """\
def forward(self, getitem):
    # 直接返回获取的元素，构成一个复杂的 Python 树结构
    return (getitem, getitem, getitem, getitem, getitem, getitem, getitem)""",
        )
    def test_map_kwargs(self):
        # 创建编译计数器对象
        cnt = CompileCounter()

        # 使用装饰器设置编译后端为cnt，定义函数f，对输入x应用sin函数
        @torch.compile(backend=cnt)
        def f(x):
            return control_flow.map(lambda x: x.sin(), x=x)

        # 生成一个大小为3的随机张量x
        x = torch.randn(3)
        # 断言调用f(x)会引发TypeError异常
        self.assertRaises(TypeError, lambda: f(x))
        # 断言编译计数器对象的帧数为0
        self.assertEqual(cnt.frame_count, 0)

    def test_map_symint_input(self):
        # 创建EagerAndRecordGraphs后端对象
        backend = EagerAndRecordGraphs()
        # 创建带有指定后端的编译计数器对象
        cnt = CompileCounterWithBackend(backend)

        def fn(x, y):
            # 定义内部函数inner，返回torch.sin(x + y)的结果
            def inner(x, y):
                return torch.sin(x + y)

            # 对inner函数以x为输入，y.size(0)为大小进行映射
            return control_flow.map(inner, x, y.size(0))

        # 生成大小为(3, 1)的随机张量x和y
        x = torch.randn(3, 1)
        y = torch.randn(3, 1)
        # 检查映射的图表并提取结果
        graphs = self._check_map_graph_and_extract(fn, (x, y))
        # 如果存在图表
        if graphs:
            # 分别获取图表和主体图表
            graph, body_graph = graphs
            # 断言图表内容符合预期
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    # 将输入的张量赋值给局部变量 l_x_
    l_x_ = L_x_
    # 获取当前对象的 map_body_0 属性
    map_body_0 = self.map_body_0
    # 调用自定义的高阶函数 map_impl，传入 map_body_0 和 l_x_，执行映射操作
    # 这里 [3] 是传递给 map_impl 的常量参数列表
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [l_x_], [3]);  map_body_0 = l_x_ = None
    # 从 map_impl 结果中获取第一个元素作为 getitem_1
    getitem_1 = map_impl[0];  map_impl = None
    # 返回包含 getitem_1 的元组作为结果
    return (getitem_1,)""",



def test_map_lowers_to_graph(self):
    # 创建 EagerAndRecordGraphs 的实例作为后端
    backend = EagerAndRecordGraphs()
    # 创建 CompileCounterWithBackend 的计数器实例
    cnt = CompileCounterWithBackend(backend)

    def fn(x, y):
        def inner(x, y):
            # 返回 torch.sin(x + y) 的结果
            return torch.sin(x + y)

        # 使用 control_flow.map 调用内部函数 inner，对输入 x 应用映射操作
        return control_flow.map(inner, x, y.size(0))

    # 创建随机张量 x 和 y
    x = torch.randn(3, 1)
    y = torch.randn(3, 1)
    # 检查 map 函数生成的图并提取其中的函数图和主体图
    graphs = self._check_map_graph_and_extract(fn, (x, y))
    if graphs:
        graph, body_graph = graphs
        # 断言 graph 的预期内容
        self.assertExpectedInline(
            graph,
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [l_x_], [3]);  map_body_0 = l_x_ = None
    getitem_1 = map_impl[0];  map_impl = None
    return (getitem_1,)""",
        )
        # 断言 body_graph 的预期内容
        self.assertExpectedInline(
            body_graph,
            """\
def forward(self, getitem, const):
    add = getitem + 3;  getitem = None
    sin = torch.sin(add);  add = None
    return (sin,)""",
        )
    # 定义测试函数，用于验证 map_dense 函数的行为与期望的一致性
    def test_map_example_value_metadata_consistent_with_eager(self):
        # 导入 map_dense 函数
        from torch._higher_order_ops.map import map_dense

        # 创建 EagerAndRecordGraphs 的实例作为后端
        backend = EagerAndRecordGraphs()

        # 定义内部函数 inner，对输入张量进行一系列操作并返回多个张量
        def inner(x):
            return x.sin(), x.cos().T, x.sin().view(-1)

        # 创建多个随机张量作为输入
        rand_44 = torch.randn(4, 4)
        inps = [
            torch.randn(3),
            torch.randn(3, 4),
            torch.randn(3, 4, 5, requires_grad=True),
            torch.randn(3, 4, 5, requires_grad=True).permute((2, 0, 1)),
            torch.randn(3, 4, 5, requires_grad=True).detach(),
            torch.randn(3, 4, 5, requires_grad=True).narrow(1, 1, 2),
            rand_44.T,
            rand_44[::2],
            rand_44[::2, ::2],
            rand_44[1::3, 1::3],
            rand_44[1::3, 1::2].T,
            rand_44.unsqueeze(1),
            rand_44.squeeze(0),
            rand_44.reshape(2, 8),
        ]

        # 对每一个输入张量进行测试
        for x in inps:
            # 编译执行 map 函数，使用 backend 进行控制流的操作记录，获取完整的计算图
            compiled_ret = torch.compile(
                control_flow.map, backend=backend, fullgraph=True
            )(inner, x)

            # 使用 map_dense 函数直接计算 eager 模式下的结果
            eager_sin, eager_transpose, eager_view = map_dense(inner, (x,), tuple())

            # 在记录的计算图中查找包含 "map" 调用的节点
            map_node = next(
                node
                for node in backend.graphs[0].graph.nodes
                if node.op == "call_function" and "map" in node.name
            )

            # 从计算图节点的元数据中获取示例值对应的张量
            fake_sin, fake_transpose, fake_view = map_node.meta["example_value"]

            # 定义内部函数 _check_size_stride_contiguous，验证张量的大小、步幅、是否连续以及梯度属性是否一致
            def _check_size_stride_contiguous(x, y):
                self.assertEqual(y.size(), x.size())
                self.assertEqual(y.stride(), x.stride())
                self.assertEqual(y.requires_grad, x.requires_grad)
                self.assertEqual(x.is_contiguous(), True)
                self.assertEqual(y.is_contiguous(), True)

            # 对比 eager 模式计算得到的结果和示例值的结果
            _check_size_stride_contiguous(eager_sin, fake_sin)
            _check_size_stride_contiguous(eager_transpose, fake_transpose)
            _check_size_stride_contiguous(eager_view, fake_view)

            # 重置动态计算图状态，并清空后端记录的图
            torch._dynamo.reset()
            backend.graphs.clear()
    # 定义一个测试函数，用于验证条件子图的名称是否有效
    def test_cond_subgraph_name_is_valid(self):
        # 创建 EagerAndRecordGraphs 后端对象
        backend = EagerAndRecordGraphs()
        # 使用 CompileCounterWithBackend 类创建计数器对象
        cnt = CompileCounterWithBackend(backend)

        # 创建布尔张量 pred 和 pred2
        pred = torch.tensor(True)
        pred2 = torch.tensor(False)
        # 生成指定形状的随机张量 xs 和 y
        xs = torch.randn(2, 3, 3)
        y = torch.randn(3, 3)

        # 定义装饰器，使用 cnt 后端进行编译，全图模式为 True
        @torch.compile(backend=cnt, fullgraph=True)
        def cond_f(pred, pred2, x, y):
            # 定义条件函数 true_fn
            def true_fn(pred2, x, y):
                return x + y

            # 定义条件函数 false_fn
            def false_fn(pred2, x, y):
                # 内部嵌套条件函数 true_fn2 和 false_fn2
                def true_fn2(x, y):
                    return x.sin() - y.cos()

                def false_fn2(x, y):
                    return x.cos() - y.sin()

                # 调用 control_flow.cond 函数执行条件判断
                return control_flow.cond(pred2, true_fn2, false_fn2, [x, y])

            # 返回 control_flow.cond 的结果
            return control_flow.cond(pred, true_fn, false_fn, [pred2, x, y])

        # 调用 cond_f 函数，并断言结果与 xs + y 相等
        result = cond_f(pred, pred2, xs, y)
        self.assertEqual(result, xs + y)

        # 获取 backend 中记录的第一个图形对象 cond_gm
        cond_gm = backend.graphs[0]
        # 创建空集合 name_set
        name_set = set()
        # 更新 name_set，包含 cond_gm 中所有命名模块的名称
        name_set.update(name for name, _ in cond_gm.named_modules())
        # 断言 name_set 是否等于给定集合
        self.assertEqual(
            name_set,
            {
                "",
                "cond_true_1",
                "cond_false_1",
                "cond_false_1.cond_false_0",
                "cond_false_1.cond_true_0",
            },
        )

    # 使用 torch._dynamo.config.patch 进行配置修补
    @torch._dynamo.config.patch(
        assume_static_by_default=True,
        dynamic_shapes=True,
    )
    # 定义测试函数，用于验证条件图在一个分支中断开的情况
    def test_cond_graph_break_in_one_branch(self):
        # 创建 EagerAndRecordGraphs 后端对象
        backend = EagerAndRecordGraphs()
        # 使用 CompileCounterWithBackend 类创建计数器对象
        cnt = CompileCounterWithBackend(backend)

        # 定义 Foo 类，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个名为 buffer 的张量缓冲区，初始值为全为 1 的 6x4 张量
                self.register_buffer("buffer", torch.ones(6, 4))

            # 定义前向传播方法
            def forward(self, x):
                # 定义条件函数 true_fn
                def true_fn(x):
                    self.buffer += 1
                    return self.buffer.sum() + x.sum()

                # 定义条件函数 false_fn
                def false_fn(x):
                    return (x - 1).sum()

                # 调用 control_flow.cond 函数执行条件判断
                return control_flow.cond(x.shape[0] > 4, true_fn, false_fn, [x])

        # 使用 torch.compile 将 Foo 类对象编译为 mod_for_compile 模块
        mod_for_compile = torch.compile(Foo(), backend=cnt, dynamic=True)
        # 创建 Foo 类对象 mod_for_eager
        mod_for_eager = Foo()

        # 使用 self.assertRaisesRegex 断言调用 mod_for_eager 抛出指定异常
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Cond doesn't work unless it is captured completely with torch.compile",
        ):
            # 调用 mod_for_eager，传入一个全为 1 的 6x4 张量，期望抛出异常
            mod_for_eager(torch.ones(6, 4))

        # 使用 self.assertRaisesRegex 断言调用 mod_for_compile 抛出指定异常
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Cond doesn't work unless it is captured completely with torch.compile",
        ):
            # 调用 mod_for_compile，传入一个全为 1 的 3x4 张量，期望抛出异常
            mod_for_compile(torch.ones(3, 4))
    def test_cond_free_variable_in_both_branches(self):
        # 创建一个 EagerAndRecordGraphs 后端对象
        backend = EagerAndRecordGraphs()
        # 创建一个 CompileCounterWithBackend 对象，与后端对象关联
        cnt = CompileCounterWithBackend(backend)

        # 创建一个 4x4 的全为 1 的张量 z
        z = torch.ones(4, 4)

        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个名为 "buffer" 的缓冲区，内容为 6x4 的全为 1 的张量
                self.register_buffer("buffer", torch.ones(6, 4))

            # 前向传播方法，接受输入 x 和 y
            def forward(self, x, y):
                # 定义真条件下的函数 true_fn
                def true_fn(x):
                    # 返回 x 的总和 + self.buffer 的总和 + z 的总和
                    return x.sum() + self.buffer.sum() + z.sum()

                # 定义假条件下的函数 false_fn
                def false_fn(x):
                    # 返回 x 的总和 - z 的总和 - self.buffer 的总和
                    return x.sum() - z.sum() - self.buffer.sum()

                # 调用控制流函数 cond，根据 y 的值执行 true_fn 或 false_fn
                return control_flow.cond(y, true_fn, false_fn, [x])

        # 使用后端 cnt 编译 Foo 类，并指定动态图和完整图
        mod_for_compile = torch.compile(
            Foo(), backend=cnt, dynamic=True, fullgraph=True
        )
        # 创建一个 Foo 类的实例 mod_for_eager
        mod_for_eager = Foo()

        # 断言编译后的模型和直接调用的模型输出相等
        self.assertEqual(
            mod_for_compile(torch.tensor(True), torch.tensor(5)),
            mod_for_eager(torch.tensor(True), torch.tensor(5)),
        )

        # 遍历后端记录的第一个图中的节点
        for node in backend.graphs[0].graph.nodes:
            # 如果节点是调用函数且目标是 torch.ops.higher_order.cond
            if (
                node.op == "call_function"
                and node.target == torch.ops.higher_order.cond
            ):
                _, _, _, operands = node.args
                # 断言每个分支有3个输入（buffer, x, z）
                self.assertEqual(len(operands), 3)
            # 如果节点是获取属性
            if node.op == "get_attr":
                # 如果节点目标是 "cond_true_0" 或 "cond_false_0"
                if str(node.target) in ("cond_true_0, cond_false_0"):
                    # 统计占位符节点的数量
                    num_placeholders = len(
                        [
                            node
                            for node in getattr(
                                backend.graphs[0], str(node.target)
                            ).graph.nodes
                            if node.op == "placeholder"
                        ]
                    )
                    # 断言占位符的数量为3
                    self.assertEqual(num_placeholders, 3)

    def _check_cond_graph_and_extract(self, fn, args):
        # 创建一个 EagerAndRecordGraphs 后端对象
        backend = EagerAndRecordGraphs()
        # 创建一个 CompileCounterWithBackend 对象，与后端对象关联
        cnt = CompileCounterWithBackend(backend)
        # 使用后端 cnt 编译函数 fn，并返回结果
        out = torch.compile(fn, backend=cnt, fullgraph=True)(*args)
        # 断言编译后的输出与直接调用函数 fn 的输出相等
        self.assertEqual(out, fn(*args))
        # 断言帧计数为1
        self.assertEqual(cnt.frame_count, 1)
        # 断言后端记录的图数量为1
        self.assertEqual(len(backend.graphs), 1)

        # 如果检查动态形状捕获返回 True，则直接返回
        if check_dynamic_shape_capture():
            return

        # 获取后端记录的第一个图对象
        gm = backend.graphs[0]
        # 获取整个图的代码并去除首尾空白字符
        graph = gm.code.strip()
        # 获取真条件分支的代码并去除首尾空白字符
        true_graph = gm.cond_true_0.code.strip()
        # 获取假条件分支的代码并去除首尾空白字符
        false_graph = gm.cond_false_0.code.strip()
        # 返回整个图、真条件分支图、假条件分支图的代码
        return (graph, true_graph, false_graph)
    # 定义一个私有方法，用于检查映射图并提取信息
    def _check_map_graph_and_extract(self, fn, args):
        # 创建一个 EagerAndRecordGraphs 的后端实例
        backend = EagerAndRecordGraphs()
        # 创建一个带计数功能的编译计数器实例，使用上面创建的后端
        cnt = CompileCounterWithBackend(backend)
        # 对给定函数 fn 进行编译，并使用完整图形模式，返回结果
        out = torch.compile(fn, backend=cnt, fullgraph=True)(*args)
        # 断言编译后的结果与直接调用函数 fn 的结果相等
        self.assertEqual(out, fn(*args))
        # 断言编译帧数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言生成的图形列表长度为 1
        self.assertEqual(len(backend.graphs), 1)

        # 如果检查到动态形状捕获，则直接返回，不继续执行下面的代码
        if check_dynamic_shape_capture():
            return

        # 获取第一个后端生成的图形管理器实例
        gm = backend.graphs[0]
        # 获取图形的代码字符串并去除首尾空白字符
        graph = gm.code.strip()
        # 初始化子图列表
        subgraphs = []
        # 遍历图形管理器中的模块名称列表
        for module_name in gm._modules.keys():
            # 获取每个模块的代码字符串并去除首尾空白字符，添加到子图列表中
            subgraphs.append(getattr(gm, module_name).code.strip())
        # 返回主图和所有子图的元组
        return (graph, *subgraphs)

    # 定义一个测试方法，用于测试条件分支在无参数情况下的行为
    def test_cond_branches_no_arguments(self):
        # 定义一个内部函数 fn，接受一个参数 x
        def fn(x):
            # 内部函数 true_fn，返回 x 的正弦值
            def true_fn():
                return torch.sin(x)

            # 内部函数 false_fn，返回 x 的余弦值
            def false_fn():
                return torch.cos(x)

            # 使用 control_flow.cond 函数，根据 x 的和是否大于 0 来选择调用 true_fn 或 false_fn
            return control_flow.cond(x.sum() > 0, true_fn, false_fn, tuple())

        # 调用 _check_cond_graph_and_extract 方法，传入 fn 函数和参数 torch.randn(4, 5)，获取返回的图形信息
        graphs = self._check_cond_graph_and_extract(fn, (torch.randn(4, 5),))
        # 如果图形信息不为 None，则进行断言比较和期望内联代码的检查
        if graphs is not None:
            graph, true_graph, false_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    # 将输入张量赋值给局部变量 l_x_
    l_x_ = L_x_
    # 计算张量 l_x_ 的元素总和
    sum_1 = l_x_.sum()
    # 比较元素总和是否大于0，返回布尔值，并清空 sum_1 变量
    gt = sum_1 > 0;  sum_1 = None
    # 获取条件分支函数 true_fn 和 false_fn
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    # 调用外部定义的条件判断函数 torch.ops.higher_order.cond
    # 根据条件 gt 执行 true_fn 或 false_fn，并传入 l_x_ 作为参数
    cond = torch.ops.higher_order.cond(gt, cond_true_0, cond_false_0, [l_x_]);  gt = cond_true_0 = cond_false_0 = l_x_ = None
    # 获取条件判断的返回结果
    getitem = cond[0];  cond = None
    # 返回结果的元组
    return (getitem,)


def forward(self, l_x_):
    # 将输入张量赋值给局部变量 l_x__1
    l_x__1 = l_x_
    # 计算张量 l_x__1 的正弦值
    sin = torch.sin(l_x__1);  l_x__1 = None
    # 返回正弦值的元组
    return (sin,)


def forward(self, l_x_):
    # 将输入张量赋值给局部变量 l_x__1
    l_x__1 = l_x_
    # 计算张量 l_x__1 的余弦值
    cos = torch.cos(l_x__1);  l_x__1 = None
    # 返回余弦值的元组
    return (cos,)
    def test_cond_side_effect_in_one_branches(self):
        # 创建 EagerAndRecordGraphs 的实例作为后端
        backend = EagerAndRecordGraphs()
        # 创建 CompileCounterWithBackend 的实例
        cnt = CompileCounterWithBackend(backend)

        # 创建包含一个 4x4 全 1 的张量的列表
        z = [torch.ones(4, 4)]

        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, y, x):
                # 定义一个在 true 分支执行的函数 true_fn
                def true_fn(x):
                    # 向 z 列表添加 x 两次，然后移除最后一个元素
                    z.append(x)
                    z.append(x)
                    z.pop()
                    # 返回 x 和 z 列表中最后一个元素的和
                    return x.sum() + z[-1].sum()

                # 定义一个在 false 分支执行的函数 false_fn
                def false_fn(x):
                    # 返回 x 和 z 列表中第一个元素的差
                    return x.sum() - z[0].sum()

                # 调用 control_flow.cond 方法，根据 y 的值选择执行 true_fn 或 false_fn
                return control_flow.cond(y, true_fn, false_fn, [x])

        # 创建 Foo 类的实例 mod_for_eager
        mod_for_eager = Foo()
        # 使用 torch.compile 方法编译 Foo 类的实例，使用 cnt 作为后端
        mod_for_compile = torch.compile(
            Foo(), backend=cnt, dynamic=True, fullgraph=False
        )
        # 断言调用 mod_for_eager 会抛出 UncapturedHigherOrderOpError 异常
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Cond doesn't work unless it is captured completely with torch.compile",
        ):
            mod_for_eager(torch.tensor(True), torch.tensor(5))

        # 断言调用 mod_for_compile 会抛出 UncapturedHigherOrderOpError 异常
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Cond doesn't work unless it is captured completely with torch.compile",
        ):
            mod_for_compile(torch.tensor(True), torch.tensor(5))

    def test_cond_with_constant_pred(self):
        # 定义一个测试函数 test，根据 pred 的值选择执行 true_fn 或 false_fn
        def test(pred, x):
            # 定义一个在 true 分支执行的函数 true_fn
            def true_fn(x):
                return x

            # 定义一个在 false 分支执行的函数 false_fn
            def false_fn(x):
                return -x

            # 调用 control_flow.cond 方法，根据 pred 的值选择执行 true_fn 或 false_fn
            return control_flow.cond(pred, true_fn, false_fn, [x])

        # 使用 torch.compile 方法将 test 函数编译为 opt_test，使用 "eager" 作为后端
        opt_test = torch.compile(test, backend="eager")
        # 创建一个全 1 的 3x3 张量 inp
        inp = torch.ones(3, 3)
        # 断言调用 test 函数和 opt_test 函数的结果近似相等
        self.assertTrue(torch.allclose(test(True, inp), opt_test(True, inp)))
        self.assertTrue(torch.allclose(test(False, inp), opt_test(False, inp)))

    def test_map_graph_break(self):
        # 创建 EagerAndRecordGraphs 的实例作为后端
        backend = EagerAndRecordGraphs()
        # 创建 CompileCounterWithBackend 的实例
        cnt = CompileCounterWithBackend(backend)

        # 定义一个继承自 torch.nn.Module 的类 Module
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个形状为 (6, 4) 的全 1 缓冲区 self.w
                self.register_buffer("w", torch.ones(6, 4))

            def forward(self, xs):
                # 定义一个在 map 中执行的函数 body
                def body(x):
                    # self.w 中的每个元素加 1
                    self.w += 1
                    return x

                # 调用 control_flow.map 方法，对 xs 中的每个元素执行 body 函数
                return control_flow.map(body, xs)

        # 创建 Module 类的实例 mod
        mod = Module()

        # 使用 torch.compile 方法编译 Module 类的实例 mod，使用 cnt 作为后端
        mod_for_compile = torch.compile(mod, backend=cnt, dynamic=True, fullgraph=False)
        # 创建 Module 类的另一个实例 mod_for_eager
        mod_for_eager = Module()

        # 调用 mod_for_compile 方法，传入张量 [[6, 4, 5], [3, 4, 5], [6, 6, 6]]，并将结果保存到 res
        res = mod_for_compile(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))
        # 断言 backend.graphs 的长度为 0
        self.assertEqual(len(backend.graphs), 0)
        # 断言调用 mod_for_eager 方法得到的结果与 res 相等
        self.assertEqual(
            res, mod_for_eager(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))
        )
    def test_wrap_allow_local_assign_in_body_fn(self):
        # 定义一个函数 f，接受两个参数 arg1 和 arg2
        def f(arg1, arg2):
            # 定义一个内部函数 inner_f，接受相同的两个参数 arg1 和 arg2
            def inner_f(arg1, arg2):
                # 将参数 arg1 赋值给变量 a，参数 arg2 赋值给变量 b
                a = arg1
                b = arg2
                # 初始化一个空列表 ret
                ret = []
                # 遍历列表 a 中的每个元素 x，将 x+1 的结果追加到 ret 中
                for x in a:
                    ret.append(x + 1)
                # 遍历列表 b 中的每个元素 x，将 x+1 的结果追加到 ret 中
                for x in b:
                    ret.append(x + 1)
                # 返回结果列表 ret
                return ret

            # 调用 wrap 函数，将 inner_f 和参数 arg1、arg2 传入
            return wrap(inner_f, arg1, arg2)

        # 创建一个包含三个元素的张量 x，所有元素值为 1
        x = torch.ones(3)

        # 定义一个生成器函数 my_args_generator，生成器每次 yield 返回两个元组，每个元组包含一个张量和其 sin 函数的结果张量
        def my_args_generator():
            yield [x], [x.sin()]
            yield (x,), (x.sin(),)

        # 调用对象的 _test_wrap_simple 方法，传入函数 f、my_args_generator 函数、3 和 3 作为参数，并要求返回图形结果
        actual_graph = self._test_wrap_simple(
            f,
            my_args_generator(),
            3,
            3,
            return_graph=True,
        )

        # 检查是否需要捕获动态形状
        if check_dynamic_shape_capture():
            return

        # 断言实际生成的图形 actual_graph 与预期的内联字符串匹配
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    # 定义一个名为 GraphModule 的 Torch 模块类
    def forward(self, L_arg1_0_: "f32[3]", L_arg2_0_: "f32[3]"):
        # 定义前向传播方法，接受两个参数 L_arg1_0_ 和 L_arg2_0_，类型为 f32[3]
        l_arg1_0_ = L_arg1_0_
        # 将参数 L_arg1_0_ 赋给局部变量 l_arg1_0_
        l_arg2_0_ = L_arg2_0_
        # 将参数 L_arg2_0_ 赋给局部变量 l_arg2_0_

        wrap_body_0 = self.wrap_body_0
        # 将 self.wrap_body_0 赋给 wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_arg1_0_, l_arg2_0_);
        # 使用 wrap 函数对 wrap_body_0、l_arg1_0_ 和 l_arg2_0_ 进行包装
        wrap_body_0 = l_arg1_0_ = l_arg2_0_ = None
        # 清空 wrap_body_0、l_arg1_0_ 和 l_arg2_0_ 的值
        getitem: "f32[3]" = wrap[0]
        # 从 wrap 中获取索引为 0 的项，赋给 getitem，类型为 f32[3]
        getitem_1: "f32[3]" = wrap[1];
        # 从 wrap 中获取索引为 1 的项，赋给 getitem_1，类型为 f32[3]
        wrap = None
        # 清空 wrap
        return (getitem, getitem_1)
        # 返回 getitem 和 getitem_1

    class GraphModule(torch.nn.Module):
        # 定义内嵌的 GraphModule 类，继承自 torch.nn.Module
        def forward(self, l_arg1_0_: "f32[3]", l_arg2_0_: "f32[3]"):
            # 定义内嵌 GraphModule 类的前向传播方法，接受两个参数 l_arg1_0_ 和 l_arg2_0_，类型为 f32[3]
            add: "f32[3]" = l_arg1_0_ + 1;
            # 计算 l_arg1_0_ 加上 1，结果赋给 add，类型为 f32[3]
            l_arg1_0_ = None
            # 清空 l_arg1_0_
            add_1: "f32[3]" = l_arg2_0_ + 1;
            # 计算 l_arg2_0_ 加上 1，结果赋给 add_1，类型为 f32[3]
            l_arg2_0_ = None
            # 清空 l_arg2_0_
            return (add, add_1)
            # 返回 add 和 add_1

    def test_capture_global_num(self):
        # 定义测试函数 test_capture_global_num
        def f(x):
            # 定义内部函数 f，接受参数 x
            return wrap(lambda x: x + global_num, x)
            # 返回一个 lambda 函数，该函数对 x 加上 global_num 的值

        x = torch.zeros([])
        # 创建一个形状为空的全零张量 x
        # Numbers don't get lifted, so args is still 2.
        # 数字不会被提升，所以 args 仍然是 2
        self._test_wrap_simple(f, default_args_generator((x,)), 2)
        # 调用 self._test_wrap_simple 进行简单的包装测试，传入 f 函数和生成的默认参数组 (x,)，期望的结果是 2

    def test_capture_global_num_adds_guard(self):
        # 定义测试函数 test_capture_global_num_adds_guard
        @torch.compile(backend="eager", fullgraph=True)
        # 使用 eager 后端进行编译，完整图形模式为 True
        def f(x):
            # 定义内部函数 f，接受参数 x
            return wrap(lambda x: x + global_num, x)
            # 返回一个 lambda 函数，该函数对 x 加上 global_num 的值

        global global_num
        # 声明全局变量 global_num
        x = torch.zeros([])
        # 创建一个形状为空的全零张量 x
        result = f(x)
        # 调用 f 函数并传入 x，结果赋给 result
        self.assertEqual(result, x + global_num)
        # 断言 result 等于 x 加上 global_num 的值

        global_num = torch.randn([]).item()
        # 将 global_num 赋值为一个随机张量的标量值
        result = f(x)
        # 再次调用 f 函数并传入 x，结果赋给 result
        self.assertEqual(result, x + global_num)
        # 断言 result 等于 x 加上更新后的 global_num 的值

    def test_capture_input_num(self):
        # 定义测试函数 test_capture_input_num
        def f(x, y):
            # 定义内部函数 f，接受两个参数 x 和 y
            return wrap(lambda x: x + y, x)
            # 返回一个 lambda 函数，该函数对 x 加上 y 的值

        x = torch.zeros([])
        # 创建一个形状为空的全零张量 x
        y = 3.14
        # 定义变量 y 为 3.14
        # Numbers don't get lifted, so args is still 2.
        # 数字不会被提升，所以 args 仍然是 2
        self._test_wrap_simple(f, default_args_generator((x, y)), 2)
        # 调用 self._test_wrap_simple 进行简单的包装测试，传入 f 函数和生成的默认参数组 (x, y)，期望的结果是 2

    def test_side_effect_in_body(self):
        # 定义测试函数 test_side_effect_in_body
        counters.clear()
        # 清空计数器
        backend = EagerAndRecordGraphs()
        # 创建一个 EagerAndRecordGraphs 后端对象赋给 backend

        x = torch.randn([])
        # 创建一个形状为空的随机张量 x
        y = torch.randn([])
        # 创建一个形状为空的随机张量 y

        def inner(x):
            # 定义内部函数 inner，接受参数 x
            nonlocal y
            # 声明非局部变量 y
            y = x
            # 将 x 的值赋给 y
            return x.clone()
            # 返回 x 的克隆版本

        @torch.compile(backend=backend)
        # 使用指定的 backend 进行编译
        def f(x):
            # 定义内部函数 f，接受参数 x
            return wrap(inner, x)
            # 返回 wrap 函数对 inner 和 x 的结果

        f(x)
        # 调用 f 函数并传入 x
        self.assertEqual(y, x)
        # 断言 y 等于 x
        assert_dict_matches_regex(
            self,
            dict(counters["graph_break"]),
            {
                r".*HigherOrderOperator: Mutating a variable not in the current scope \(SideEffects\)": 1
            },
        )
        # 使用正则表达式断言 counters["graph_break"] 中包含指定信息的字典

    def test_fallback_on_graph_break_simple(self):
        # 定义测试函数 test_fallback_on_graph_break_simple
        # In the future, there should be a per-HigherOrderOperator switch
        # on whether or not to fallback or raise a loud error.
        # For now we just fallback by default.
        # 在未来，应该有一个每个 HigherOrderOperator 的开关，
        # 决定是回退还是抛出一个大声的错误。
        # 目前我们只是默认回退。

        cnt = CompileCounter()
        # 创建一个 CompileCounter 对象 cnt
        x = torch.randn([])
        # 创建一个形状为空的随机张量 x

        def inner(x):
            # 定义内部函数 inner，接受参数 x
            y = x.sin()
            # 计算 x 的正弦值，结果赋给 y
            torch._dynamo.graph_break()
            # 调用 _dynamo 模块中的 graph_break 函数
            z = y.sin()
            # 计算 y 的正弦值，结果赋给 z
            return z
            # 返回 z

        @torch.compile(backend=cnt
    def test_flat_list_output(self):
        # 定义函数 f，接受参数 x，返回对 x 应用 sin 和 cos 后的结果列表
        def f(x):
            return wrap(lambda x: [torch.sin(x), torch.cos(x)], x)

        # 生成一个 3 维度的随机张量 x
        x = torch.randn(3)
        # 使用 _test_wrap_simple 函数测试 f，传入默认参数生成器和期望的操作数为 3
        self._test_wrap_simple(f, default_args_generator((x,)), 2, expected_opcount=3)

    def test_fallback_on_python_primitives_output(self):
        # 清空计数器
        counters.clear()
        # 创建编译计数器 cnt
        cnt = CompileCounter()

        # 使用 torch.compile 装饰器定义函数 f，接受参数 x，返回一个包含 1、torch.sin(x) 和 2.0 的列表
        @torch.compile(backend=cnt)
        def f(x):
            return wrap(lambda x: [1, torch.sin(x), 2.0], x)

        # 生成一个 3 维度的随机张量 x
        x = torch.randn(3)
        # 调用函数 f，得到结果
        result = f(x)
        # 断言结果应该与 [1, torch.sin(x), 2.0] 相等
        self.assertEqual(result, [1, torch.sin(x), 2.0])
        # 断言计数器的帧数为 0
        self.assertEqual(cnt.frame_count, 0)
        # 使用 assert_dict_matches_regex 函数断言 counters["graph_break"] 字典与给定正则表达式匹配
        assert_dict_matches_regex(
            self,
            dict(counters["graph_break"]),
            {".*HigherOrderOperator body's output must consist of tensors only": 1},
        )
    def test_nested_tuple_output(self):
        # 定义内部函数 f，接受参数 x
        def f(x):
            # 使用 wrap 函数对 x 进行封装，返回的结果解构为 ((a, b),)
            ((a, b),) = wrap(lambda x: ((x.sin(), x.cos()),), x)
            # 返回 a + b 的结果
            return a + b

        # 生成一个 2x3 的随机张量
        x = torch.randn(2, 3)

        # 清空计数器 counters
        counters.clear()
        # 调用 self._test_wrap_simple 方法，传入 f 函数、生成默认参数、2 和 4，返回图形对象 graph
        graph = self._test_wrap_simple(
            f, default_args_generator((x,)), 2, 4, return_graph=True
        )
        # 断言 graph_break 在 counters 中的长度为 0
        self.assertEqual(len(counters["graph_break"]), 0)

        # 检查是否捕获了动态形状
        if check_dynamic_shape_capture():
            return

        # 断言 graph 与预期的内联字符串匹配
        self.assertExpectedInline(
            graph,
            """\
class GraphModule(torch.nn.Module):
    # 定义神经网络模块的类 GraphModule
    def forward(self, L_x_: "f32[2, 3]"):
        # 前向传播函数，接收类型为 f32[2, 3] 的输入 L_x_

        l_x_ = L_x_

        # 将输入赋值给新变量 l_x_

        wrap_body_0 = self.wrap_body_0
        # 获取 self 对象的 wrap_body_0 属性，并赋值给 wrap_body_0 变量

        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_);
        # 调用 torch._higher_order_ops.wrap.wrap 函数，对 wrap_body_0 和 l_x_ 进行包装操作，结果赋值给 wrap
        wrap_body_0 = l_x_ = None
        # 清空 wrap_body_0 和 l_x_ 的引用，释放内存空间

        a: "f32[2, 3]" = wrap[0]
        # 从 wrap 结果中获取第一个元素，赋值给变量 a，类型为 f32[2, 3]
        b: "f32[2, 3]" = wrap[1];  wrap = None
        # 从 wrap 结果中获取第二个元素，赋值给变量 b，类型为 f32[2, 3]，然后清空 wrap 的引用，释放内存空间

        add: "f32[2, 3]" = a + b;  a = b = None
        # 计算 a 和 b 的和，结果赋值给 add，类型为 f32[2, 3]，然后清空 a 和 b 的引用，释放内存空间
        return (add,)
        # 返回计算结果 add 的元组形式

    class GraphModule(torch.nn.Module):
        # 定义内部嵌套的神经网络模块类 GraphModule

        def forward(self, l_x_: "f32[2, 3]"):
            # 前向传播函数，接收类型为 f32[2, 3] 的输入 l_x_

            sin: "f32[2, 3]" = l_x_.sin()
            # 计算输入 l_x_ 的正弦值，结果赋值给 sin，类型为 f32[2, 3]
            cos: "f32[2, 3]" = l_x_.cos();  l_x_ = None
            # 计算输入 l_x_ 的余弦值，结果赋值给 cos，类型为 f32[2, 3]，然后清空 l_x_ 的引用，释放内存空间
            return (sin, cos)
            # 返回计算结果 sin 和 cos 的元组形式
    # 定义一个测试函数 test_make_closure(self)，测试闭包的创建和使用
    def test_make_closure(self):
        # 定义内部函数 f(x, y)，接受两个参数 x 和 y
        def f(x, y):
            # 定义内部函数 g(x)，接受参数 x，返回 x + y 的结果
            def g(x):
                return x + y

            # 返回调用 g(x) 的结果
            return g(x)

        # 定义内部函数 h(x, y)，接受两个参数 x 和 y，调用 wrap 函数并返回结果
        def h(x, y):
            return wrap(f, x, y)

        # 创建一个 3x3 的随机张量 x
        x = torch.randn(3, 3)
        # 创建一个 3x3 的随机张量 y
        y = torch.randn(3, 3)
        # 调用测试函数 _test_wrap_simple，并期望其结果为 3
        self._test_wrap_simple(h, default_args_generator((x, y)), 3)

    # 定义一个测试函数 test_internal_nonlocal(self)，测试闭包中使用 nonlocal 关键字
    def test_internal_nonlocal(self):
        # 定义内部函数 f(x, y)，接受两个参数 x 和 y
        def f(x, y):
            # 在 f 内部定义局部变量 w，并赋值为 1
            w = 1

            # 定义内部函数 g(x)，接受参数 x，使用 nonlocal 修改外部函数的局部变量 w，返回 x
            def g(x):
                nonlocal w
                w = x
                return x

            # 定义内部函数 h(x)，接受参数 x，使用 nonlocal 修改外部函数的局部变量 w，返回 x
            def h(x):
                nonlocal w
                w = w + 1
                return x

            # 调用 g(x) 和 h(x)，并返回 w + y 的结果
            g(x)
            h(x)
            return w + y

        # 定义内部函数 h(x, y)，接受两个参数 x 和 y，调用 wrap 函数并返回结果
        def h(x, y):
            return wrap(f, x, y)

        # 创建一个 3x3 的随机张量 x
        x = torch.randn(3, 3)
        # 创建一个 3x3 的随机张量 y
        y = torch.randn(3, 3)
        # 调用测试函数 _test_wrap_simple，并期望其结果为 3
        self._test_wrap_simple(h, default_args_generator((x, y)), 3)

    # 定义一个测试函数 test_capture_numpy_number(self)，测试捕获 numpy 数字的情况
    def test_capture_numpy_number(self):
        # 导入 numpy 库
        import numpy as np

        # 创建一个 numpy.float32 类型的常量 y，值为 1.0
        y = np.float32(1.0)

        # 定义内部函数 f(x)，接受参数 x，使用 wrap 函数捕获 lambda 函数 x + y，并返回结果
        def f(x):
            return wrap(lambda x: x + y, x)

        # 创建一个长度为 3 的随机张量 x
        x = torch.randn(3)
        # 调用测试函数 _test_wrap_simple，并期望其结果为 3
        # np.number 类型会被提升为图形输入
        self._test_wrap_simple(f, default_args_generator((x,)), 3)

    # 定义一个测试函数 test_freevars_as_inputs_to_wrap(self)，测试自由变量作为 wrap 函数的输入
    def test_freevars_as_inputs_to_wrap(self):
        # 创建一个长度为 3 的随机张量 y
        y = torch.randn(3)

        # 定义内部函数 f(x)，接受参数 x，使用 wrap 函数捕获 lambda 函数 x + y，并返回结果
        def f(x):
            return wrap(lambda x, y: x + y, x, y)

        # 创建一个长度为 3 的随机张量 x
        x = torch.randn(3)
        # 调用测试函数 _test_wrap_simple，并期望其结果为 3
        self._test_wrap_simple(f, default_args_generator((x,)), 3)

    # 定义一个测试函数 test_lift_tensor_constant(self)，测试提升张量常量的情况
    def test_lift_tensor_constant(self):
        # 定义内部函数 f(x)，接受参数 x
        def f(x):
            # 创建一个张量常量 y，值为 1.0
            y = torch.tensor(1.0)
            # 使用 wrap 函数捕获 lambda 函数 x + y，并返回结果
            return wrap(lambda x: x + y, x)

        # 创建一个长度为 3 的随机张量 x
        x = torch.randn(3)
        # 调用测试函数 _test_wrap_simple，并期望其结果为 3，同时预期操作计数为 3
        self._test_wrap_simple(f, default_args_generator((x,)), 3, expected_opcount=3)

    # 定义一个测试函数 test_nested_wrap(self)，测试嵌套 wrap 的情况
    def test_nested_wrap(self):
        # 定义一个 MockModule 类，继承自 torch.nn.Module
        class MockModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            # 前向传播方法
            def forward(self, x):
                return self.linear(x)

        # 创建 MockModule 类的实例 mod
        mod = MockModule()

        # 定义内部函数 gn(x)，接受参数 x，返回 torch.cos(x) + wrap(mod, x) 的结果
        def gn(x):
            return torch.cos(x) + wrap(mod, x)

        # 定义内部函数 fn(x)，接受参数 x，返回 wrap(gn, x) 的结果
        def fn(x):
            return wrap(gn, x)

        # 调用测试函数 _test_wrap_simple，并期望其结果为 4
        self._test_wrap_simple(fn, default_args_generator((torch.randn(10, 10),)), 4)

    # 定义一个测试函数 test_fn_with_kwargs_in_torch_ops(self)，测试 torch 操作中带有关键字参数的情况
    def test_fn_with_kwargs_in_torch_ops(self):
        # 定义内部函数 fn(x)，接受参数 x，返回 wrap(lambda z: torch.cos(input=z), x) 的结果
        def fn(x):
            return wrap(lambda z: torch.cos(input=z), x)

        # 创建一个长度为 3 的随机张量 x
        x = torch.randn(3)
        # 调用测试函数 _test_wrap_simple，并期望其结果为 2
        self._test_wrap_simple(fn, default_args_generator((x,)), 2)
    def test_hooks(self):
        # 定义一个简单的神经网络模型，包含一个线性层
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.net(x)

        # 创建一个ToyModel实例
        model = ToyModel()
        # 用于存储forward hook的字典和激活值的字典
        forward_handles = {}
        activations = dict()

        # 定义一个保存激活值的函数，并将其注册为forward hook
        def save_activations(mod, inp, out):
            activations[name] = inp

        # 遍历模型的每个子模块，注册forward hook，保存到forward_handles字典中
        for name, module in model.named_children():
            forward_handles[name] = module.register_forward_hook(save_activations)

        # 定义一个使用eager模式编译的函数fn
        @torch.compile(backend="eager")
        def fn(x):
            return wrap(lambda x: model(x), x)

        # 进行两次迭代
        for i in range(2):
            # 清空激活值字典，准备进行下一次迭代
            activations.clear()
            # 生成一个随机的输入张量
            x = torch.randn((10, 10))
            # 调用fn函数进行模型推断
            pred = fn(x)
            # 计算预测结果的总和作为损失
            loss = pred.sum()
            # 反向传播计算梯度
            loss.backward()

        # 断言激活值字典的键与forward_handles字典的键相同
        self.assertTrue(activations.keys() == forward_handles.keys())

    def _get_source_fn_stack(self, gm, node_names):
        # 定义一个函数，从计算图中获取指定节点的源函数堆栈信息
        ret = {}
        # 遍历计算图中的每个模块
        for mod in gm.modules():
            # 遍历模块的每个节点
            for node in mod.graph.nodes:
                # 如果节点的名称在给定的node_names集合中
                if node.name in node_names:
                    # 获取节点的源函数堆栈信息，并保存到ret字典中
                    actual_stack = [
                        name for name, _ in node.meta.get("source_fn_stack", [])
                    ]
                    ret[node.name] = actual_stack
        return ret

    def test_wrap_source_fn_stack(self):
        # 定义一个MockModule模型，包含一个线性层
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        # 创建MockModule实例
        mod = MockModule()

        # 定义一个简单的函数gn，包含一个cos函数和一个模型的调用
        def gn(x):
            return torch.cos(x) + wrap(mod, x)

        # 定义一个函数fn，调用gn函数
        def fn(x):
            return wrap(gn, x)

        # 使用EagerAndRecordGraphs作为后端，编译fn函数
        backend = EagerAndRecordGraphs()
        inp = torch.randn((4, 4))
        torch.compile(fn, backend=backend, fullgraph=True)(inp)

        # 获取编译后的计算图
        gm = backend.graphs[0]
        # 获取包含指定节点源函数堆栈信息的字典
        actual_stack = self._get_source_fn_stack(gm, {"cos", "add", "linear"})
        # 断言实际的源函数堆栈信息与预期值相符
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """\
    def test_cond_source_fn_stack(self):
        # 创建一个启用了“急切模式”和记录图的后端对象
        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        # 定义一个控制流函数，当参数pred为True时调用true_fn，否则调用false_fn
        def cond_f(pred, pred2, x, y):
            # 定义当pred2为True时的函数
            def true_fn(pred2, x, y):
                # 返回x和y的和
                return x + y

            # 定义当pred2为False时的函数
            def false_fn(pred2, x, y):
                # 定义一个内部函数true_fn2，返回x的正弦值减去y的余弦值
                def true_fn2(x, y):
                    return x.sin() - y.cos()

                # 定义一个内部函数false_fn2，返回x的余弦值减去y的正弦值
                def false_fn2(x, y):
                    return x.cos() - y.sin()

                # 根据pred2的值调用条件分支函数control_flow.cond，并返回结果
                return control_flow.cond(pred2, true_fn2, false_fn2, [x, y])

            # 根据pred的值调用条件分支函数control_flow.cond，并返回结果
            return control_flow.cond(pred, true_fn, false_fn, [pred2, x, y])

        # 定义一个torch.tensor对象，表示True
        pred = torch.tensor(True)
        # 定义一个torch.tensor对象，表示False
        pred2 = torch.tensor(False)
        # 创建一个形状为(2, 3, 3)的张量对象，填充随机值
        xs = torch.randn(2, 3, 3)
        # 创建一个形状为(3, 3)的张量对象，填充随机值
        y = torch.randn(3, 3)
        # 调用cond_f函数，传入参数，执行控制流程
        cond_f(pred, pred2, xs, y)

        # 获取记录的图对象
        gm = backend.graphs[0]
        # 获取包含的源函数栈，指定了要检查的函数集合{"cos", "add", "sin", "sub"}
        actual_stack = self._get_source_fn_stack(gm, {"cos", "add", "sin", "sub"})
        # 断言实际的源函数栈与预期的值相等
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """\
{'add': ['cond', 'add'],
 'cos': ['cond', 'cond', 'cos'],
 'sin': ['cond', 'cond', 'sin'],
 'sub': ['cond', 'cond', 'sub']}""",
        )

    def test_map_source_fn_stack(self):
        # 创建一个启用了“急切模式”和记录图的后端对象
        backend = EagerAndRecordGraphs()

        # 创建形状为(2, 3, 3)和(3,)的张量对象，填充随机值
        xs = torch.randn(2, 3, 3)
        y = torch.randn(3)

        @torch.compile(backend=backend, fullgraph=True)
        # 定义一个映射函数，对xs和y进行映射操作
        def map_f(xs, y):
            # 定义内部函数inner，对x和y进行操作
            def inner(x, y):
                # 定义内部函数inner2，返回x和y的和
                def inner2(x, y):
                    return x + y

                # 调用control_flow.map函数，对x和y进行映射操作，并乘以y的余弦值
                return control_flow.map(inner2, x, y) * y.cos()

            # 调用control_flow.map函数，对xs和y进行映射操作，并返回结果的正弦值
            return control_flow.map(inner, xs, y).sin()

        # 调用map_f函数，传入参数，执行映射操作
        result = map_f(xs, y)

        # 获取记录的图对象
        gm = backend.graphs[0]
        # 获取包含的源函数栈，指定了要检查的函数集合{"cos", "add", "sin"}
        actual_stack = self._get_source_fn_stack(gm, {"cos", "add", "sin"})
        # 断言实际的源函数栈与预期的值相等
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """{'add': ['map', 'map', 'add'], 'cos': ['map', 'cos'], 'sin': ['sin']}""",
        )

    def test_grad_source_fn_stack(self):
        # 创建一个启用了“急切模式”和记录图的后端对象
        backend = EagerAndRecordGraphs()

        # 定义一个函数fn，对x执行正弦函数后求和
        def fn(x):
            return x.sin().sum()

        @torch.compile(backend=backend, fullgraph=False)
        # 定义一个包装函数，对fn的二阶导数进行计算
        def wrapper_fn(x):
            return torch.func.grad(torch.func.grad(fn))(x)

        # 创建一个形状为()的张量对象，填充随机值
        x = torch.randn(())

        # 调用wrapper_fn函数，传入参数，执行函数包装
        wrapper_fn(x)
        
        # 获取记录的图对象
        gm = backend.graphs[0]
        # 获取包含的源函数栈，指定了要检查的函数集合{"sum_1", "sin"}
        actual_stack = self._get_source_fn_stack(gm, {"sum_1", "sin"})
        # 断言实际的源函数栈与预期的值相等
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """{'sin': ['sin']}""",
        )
    def test_vmap_source_fn_stack(self):
        backend = EagerAndRecordGraphs()  # 创建一个记录图形操作的后端对象

        def inner_fn(x):
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1))(x)
            # 使用 torch.func.vmap 对输入的 x 进行向量化映射，计算每个元素的列和与行和之和

        @torch.compile(backend=backend, fullgraph=True)
        def fn(x):
            return torch.func.vmap(lambda x: inner_fn(x.cos()))(x)
            # 使用 torch.func.vmap 对输入的 x 进行向量化映射，并在每个元素上应用 inner_fn 函数

        x = torch.randn(3, 3, 3, 3)  # 创建一个形状为 (3, 3, 3, 3) 的随机张量
        fn(x)  # 调用 fn 函数对张量 x 进行处理
        gm = backend.graphs[0]  # 获取记录的第一个计算图
        actual_stack = self._get_source_fn_stack(
            gm, {"sum_1", "sum_2", "batched_output"}
        )
        # 获取计算图中与指定键集合匹配的源函数堆栈
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """{'sum_1': ['sum_1'], 'sum_2': ['sum_2']}""",
        )

    def test_cond_pytree_operands(self):
        def _construct_pytree():
            a = torch.randn(3, 3)  # 创建一个形状为 (3, 3) 的随机张量 a
            b = torch.randn(3, 3)  # 创建一个形状为 (3, 3) 的随机张量 b
            c = torch.randn(3, 3)  # 创建一个形状为 (3, 3) 的随机张量 c
            d = torch.randn(3, 3)  # 创建一个形状为 (3, 3) 的随机张量 d
            e = torch.randn(3, 3)  # 创建一个形状为 (3, 3) 的随机张量 e
            f = torch.randn(3, 3)  # 创建一个形状为 (3, 3) 的随机张量 f
            g = torch.randn(3, 3)  # 创建一个形状为 (3, 3) 的随机张量 g
            return (a, [[[b]]], c, (d, (e,), f), {"g": g})  # 构建一个复杂的 Pytree 结构并返回

        pred = torch.tensor(True)  # 创建一个布尔张量 pred，值为 True
        inp = _construct_pytree()  # 通过调用 _construct_pytree 函数创建输入张量 inp

        def _reduce_sum(flattened):
            init = 0
            for val in flattened:
                init += val
            return init
            # 对扁平化的输入列表执行求和操作并返回结果

        def _reduce_max(flattened):
            init = flattened[0]
            for val in flattened:
                init = max(val, init)
            return init
            # 对扁平化的输入列表执行求最大值操作并返回结果

        def true_fn(pytree_in):
            flattened, spec = pytree.tree_flatten(pytree_in)
            return _reduce_sum(flattened)
            # 对输入的 pytree 结构进行扁平化并调用 _reduce_sum 函数

        def false_fn(pytree_in):
            flattened, spec = pytree.tree_flatten(pytree_in)
            return _reduce_max(flattened)
            # 对输入的 pytree 结构进行扁平化并调用 _reduce_max 函数

        def fn(pred, pytree_in):
            return torch.cond(pred, true_fn, false_fn, [pytree_in])
            # 根据预测值 pred 条件执行 true_fn 或 false_fn 函数

        backend = EagerAndRecordGraphs()  # 创建一个记录图形操作的后端对象
        cnt = CompileCounterWithBackend(backend)  # 使用指定后端创建编译计数器对象
        compiled_res = torch.compile(fn, backend=backend)(pred, inp)
        # 编译 fn 函数并使用给定的输入 pred 和 inp 运行，返回编译结果
        eager_res = fn(pred, inp)  # 直接调用 fn 函数获取非编译结果
        self.assertEqual(compiled_res, eager_res)
        graph = backend.graphs[0]  # 获取记录的第一个计算图

        # 动态形状生成略有不同的图形。
        if check_dynamic_shape_capture():
            return

        self.assertExpectedInline(
            graph.code.strip(),
            """\
# 定义一个方法，接受多个输入张量参数，执行条件分支操作
def forward(self, L_pred_ : torch.Tensor, L_pytree_in_0_ : torch.Tensor, L_pytree_in_1_0_0_0_ : torch.Tensor,
            L_pytree_in_2_ : torch.Tensor, L_pytree_in_3_0_ : torch.Tensor, L_pytree_in_3_1_0_ : torch.Tensor,
            L_pytree_in_3_2_ : torch.Tensor, L_pytree_in_4_g_ : torch.Tensor):
    # 将输入张量赋值给局部变量
    l_pred_ = L_pred_
    l_pytree_in_0_ = L_pytree_in_0_
    l_pytree_in_1_0_0_0_ = L_pytree_in_1_0_0_0_
    l_pytree_in_2_ = L_pytree_in_2_
    l_pytree_in_3_0_ = L_pytree_in_3_0_
    l_pytree_in_3_1_0_ = L_pytree_in_3_1_0_
    l_pytree_in_3_2_ = L_pytree_in_3_2_
    l_pytree_in_4_g_ = L_pytree_in_4_g_
    
    # 获取条件真值和假值操作符
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    
    # 调用 torch 高阶条件分支操作
    cond = torch.ops.higher_order.cond(l_pred_, cond_true_0, cond_false_0, [l_pytree_in_0_, l_pytree_in_1_0_0_0_,
                                                                            l_pytree_in_2_, l_pytree_in_3_0_,
                                                                            l_pytree_in_3_1_0_, l_pytree_in_3_2_,
                                                                            l_pytree_in_4_g_])
    
    # 解包条件分支的结果并清理局部变量引用
    l_pred_ = cond_true_0 = cond_false_0 = l_pytree_in_0_ = l_pytree_in_1_0_0_0_ = l_pytree_in_2_ = l_pytree_in_3_0_ = \
    l_pytree_in_3_1_0_ = l_pytree_in_3_2_ = l_pytree_in_4_g_ = None
    
    # 从条件分支的结果中获取指定索引的项
    getitem = cond[0]
    cond = None  # 清理条件变量的引用
    
    # 返回获取到的项作为元组
    return (getitem,)
    def test_grad_guard_fail(self, records):
        # 引入 torch.func.grad 并赋值给 grad 变量
        grad = torch.func.grad

        # 使用 eager 后端编译 fn 函数
        @torch.compile(backend="eager")
        def fn(x):
            # 计算 torch.sin 的梯度，并返回其关于 x.sum() 的梯度
            return grad(torch.sin)(x.sum())

        # 生成一个随机张量 x
        x = torch.randn([])
        # 调用 fn 函数
        fn(x)
        # 断言 records 长度为 0
        self.assertEqual(len(records), 0)

        # 再次调用 fn 函数，不应使图形无效
        fn(x)
        # 断言 records 长度为 0
        self.assertEqual(len(records), 0)

        # 调用 grad 应重新触发编译
        x = torch.randn(3)
        grad(fn)(x)
        # 断言 records 长度大于 0
        self.assertGreater(len(records), 0)
        # 获取记录中的消息，并检查是否包含特定的触发信息
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """\
    triggered by the following guard failure(s):
    - torch._functorch.pyfunctorch.compare_functorch_state([])""",
            munge_exc(record.getMessage()),
        )




    @make_logging_test(recompiles=True)
    def test_dual_level_guard(self, records):
        # 引入 torch.autograd.forward_ad 并赋值给 fwAD 变量
        fwAD = torch.autograd.forward_ad

        # 使用 eager 后端编译 fn 函数，完整图形模式为 True
        @torch.compile(backend="eager", fullgraph=True)
        def fn(foo, tangent):
            # 使用 dual_level 上下文，创建 foo 和 tangent 的双层自动微分对象 dual
            with fwAD.dual_level():
                dual = fwAD.make_dual(foo, tangent[1:])
                return dual

        # 生成随机张量 foo 和 tangent
        foo = torch.rand(2)
        tangent = torch.rand(3)
        # 调用 fn 函数
        fn(foo, tangent)
        # 断言 records 长度为 0
        self.assertEqual(len(records), 0)

        # 再次调用 fn 函数，不应使图形无效
        fn(foo, tangent)
        # 断言 records 长度为 0
        self.assertEqual(len(records), 0)

        # 使用 assertRaises 检查是否有内部错误异常
        with self.assertRaises(torch._dynamo.exc.InternalTorchDynamoError):
            with fwAD.dual_level():
                fn(foo, tangent)
        # 断言 records 长度大于 0
        self.assertGreater(len(records), 0)
        # 获取记录中的消息，并检查是否包含特定的触发信息
        record = self.getRecord(records, "forward_ad")
        self.assertIn(
            """\
    triggered by the following guard failure(s):
    - torch.autograd.forward_ad._current_level == -1""",
            munge_exc(record.getMessage()),
        )




    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_jvp_guard_fail(self, records):
        # 引入 torch.func.jvp 并赋值给 jvp 变量
        jvp = torch.func.jvp
        # 引入 torch.func.vmap 并赋值给 vmap 变量
        vmap = torch.func.vmap

        # 使用 eager 后端编译 fn 函数
        @torch.compile(backend="eager")
        def fn(x):
            # 对 torch.sin 进行 jvp 计算，使用 x 作为输入和变化量
            return jvp(torch.sin, (x,), (x,))

        # 生成随机张量 x
        x = torch.randn(3, 4)
        # 调用 fn 函数
        fn(x)
        # 断言 records 长度为 0
        self.assertEqual(len(records), 0)

        # 再次调用 fn 函数，不应使图形无效
        fn(x)
        # 断言 records 长度为 0
        self.assertEqual(len(records), 0)

        # 调用 jvp 应重新触发编译
        x = torch.randn(3, 4, 5)
        jvp(vmap(fn), (x,), (x,))

        # 断言 records 长度大于 0
        self.assertGreater(len(records), 0)
        # 如果记录中有 "pyfunctorch"，获取记录并检查是否包含特定的触发信息
        if self.hasRecord(records, "pyfunctorch"):
            record = self.getRecord(records, "pyfunctorch")
            self.assertIn(
                """\
    triggered by the following guard failure(s):
    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_vmap_guard_fail_different_state(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 4)
        y = torch.vmap(fn, randomness="same")(x)
        self.assertEqual(x.sin(), y)
        self.assertEqual(len(records), 0)

        # call vmap(vmap(fn))(x) should retrigger compilation
        y = torch.vmap(fn, randomness="different")(x)
        self.assertEqual(x.sin(), y)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        # 检查日志记录中是否包含以下触发守卫失败的信息：
        # - torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'same')])
        self.assertIn(
            """\
    triggered by the following guard failure(s):
    - torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'same')])""",
            record.getMessage(),
        )

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_vmap_guard_fail(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        y = torch.vmap(fn)(x)
        self.assertEqual(x.sin(), y)
        self.assertEqual(len(records), 0)

        # call vmap(vmap(fn))(x) should retrigger compilation as
        # _functorch.current_level() is not the same
        x = torch.zeros(3, 3, 3, 4, 5)
        y = torch.vmap(torch.vmap(fn))(x)
        self.assertEqual(x.sin(), y)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        # 检查日志记录中是否包含以下触发守卫失败的信息：
        # - torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'error')])
        self.assertIn(
            """\
    triggered by the following guard failure(s):
    - torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'error')])""",
            record.getMessage(),
        )
    # 使用装饰器 @make_logging_test(recompiles=True) 包装的测试函数，用于测试 vmap 和梯度计算中的异常情况
    def test_vmap_grad_vmap_guard_fail(self, records):
        # 导入 torch 的 vmap 和 grad 函数
        vmap = torch.vmap
        grad = torch.func.grad

        # 定义函数 g，对输入 x 应用 torch.sin 函数，然后对结果进行求和
        def g(x):
            y = vmap(torch.sin, randomness="same")(x)
            return y.sum(0)

        # 使用装饰器 @torch.compile(backend="eager") 包装的函数 fn，计算 g 的梯度
        @torch.compile(backend="eager")
        def fn(x):
            return grad(g)(x)

        # 创建随机张量 x，并对 fn 进行 vmap 操作，期望结果与 x 的余弦值相等
        x = torch.randn(3, 3)
        y = vmap(fn, randomness="error")(x)
        self.assertEqual(x.cos(), y)

        # 先前的 FX 图应该无效化
        x = torch.randn(3, 3, 4)
        # 对 fn 进行双重 vmap 操作，记录操作历史
        y = vmap(vmap(fn, randomness="different"))(x)
        self.assertGreater(len(records), 0)
        # 从日志记录中获取包含特定信息的记录
        record = self.getRecord(records, "pyfunctorch")
        # 断言记录消息包含特定的错误信息
        self.assertIn(
            """\
    triggered by the following guard failure(s):
    - torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'error')])""",
            munge_exc(record.getMessage()),
        )

    # 使用装饰器 @xfailIfTorchDynamo 和 @make_logging_test(recompiles=True) 包装的测试函数，测试不同状态下的 vmap 重新编译行为
    def test_vmap_recompile_different_states(self, records):
        # 使用 @torch.compile(backend="eager") 包装的函数 fn，对输入 x 应用 torch.sin 函数进行 vmap 操作
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        # 创建形状为 (3, 3, 4, 5) 的零张量 x
        x = torch.zeros(3, 3, 4, 5)
        # 对 fn 进行 vmap 操作，期望记录中不包含任何错误信息
        y = torch.vmap(fn, randomness="same")(x)
        self.assertEqual(len(records), 0)  # 确保记录为空

        # 对 fn 进行 vmap 操作，使用不同的随机性参数，记录 vmap 重新编译的历史
        y = torch.vmap(fn, randomness="different")(x)
        self.assertGreater(len(records), 0)
        # 从日志记录中获取包含特定信息的记录
        record = self.getRecord(records, "pyfunctorch")
        # 断言记录消息包含特定的错误信息
        self.assertIn(
            """\
    triggered by the following guard failure(s):
    - torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'same')])""",
            munge_exc(record.getMessage()),
        )

    # 使用装饰器 @config.patch(capture_func_transforms=True) 和 @make_logging_test(guards=True) 包装的测试函数，测试 funtorch 的 guard 机制
    def test_emit_functorch_guard_if_active(self, records):
        # 使用 @torch.compile(backend="eager") 包装的函数 fn，对输入 x 应用 torch.sin 函数
        @torch.compile(backend="eager")
        def fn(x):
            return torch.sin(x)

        # 创建随机张量 x，并计算 fn 的结果
        x = torch.randn(3, 4)
        _ = fn(x)
        self.assertFalse(self.hasRecord(records, "pyfunctorch"))  # 确保记录中不包含 funtorch 的信息

        # 对 fn 进行 vmap 操作，记录 funtorch 的 guard 机制激活的历史
        _ = torch.vmap(fn)(x)
        self.assertTrue(self.hasRecord(records, "pyfunctorch"))
        # 从日志记录中获取包含特定信息的记录
        record = self.getRecord(records, "pyfunctorch")
        # 断言记录消息包含特定的错误信息
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'error')])""",
            munge_exc(record.getMessage()),
        )

    # 使用装饰器 @make_logging_test(recompiles=True) 包装的测试函数，测试 fn 函数的重新编译行为
    def test_linearize_recompiles(self, records):
        # 使用 @torch.compile(backend="eager") 包装的函数 fn，对输入 x 应用 torch.sin 函数进行线性化处理
        @torch.compile(backend="eager")
        def fn(x):
            out, jvp_fn = torch.func.linearize(torch.sin, x)
            return out, jvp_fn(x)

        # 创建形状为 (2, 3) 的随机张量 x，并对 fn 进行计算，期望记录中不包含任何错误信息
        x = torch.randn(2, 3)
        fn(x)
        self.assertEqual(len(records), 0)

        # 创建形状为 (2, 3) 的随机张量 z，并对 fn 进行计算，期望记录中不包含任何错误信息
        z = torch.randn(2, 3)
        fn(z)
        self.assertEqual(len(records), 0)

        # 创建形状为 (3, 4) 的随机张量 y，并对 fn 进行计算，记录 fn 函数的重新编译历史
        y = torch.randn(3, 4)
        fn(y)
        self.assertGreater(len(records), 0)
    # 定义 FuncTorchHigherOrderOpTests 类，继承自 torch._dynamo.test_case.TestCase
    class FuncTorchHigherOrderOpTests(torch._dynamo.test_case.TestCase):
        
        # tearDown 方法，用于在每个测试结束后执行清理操作
        def tearDown(self):
            # 如果未启用 TORCH_TEST_WITH_DYNAMO 环境变量，则直接返回，不执行清理操作
            if not TEST_WITH_TORCHDYNAMO:
                return
            
            # 初始化警告标志
            warn = False
            
            # 循环检查 functorch 的解释器栈，确保不存在未撤销的 _vmap_increment_nesting 操作
            while ci := torch._C._functorch.peek_interpreter_stack():
                # 如果当前栈顶是 Vmap 类型的解释器
                if ci.key() == torch._C._functorch.TransformType.Vmap:
                    # 设置警告标志为 True，并执行 _vmap_decrement_nesting 操作
                    warn = True
                    torch._C._functorch._vmap_decrement_nesting()
                else:
                    break
            
            # 如果存在警告，则发出警告消息
            if warn:
                msg = (
                    "Interpreter stack is not empty. Test should have called "
                    "'torch._C._functorch._vmap_decrement_nesting()'"
                )
                warnings.warn(msg)
        
        # _compile_check 方法，编译检查函数，用于比较预期结果和实际结果
        def _compile_check(self, fn, inputs, fullgraph=True, graph_idx=0):
            # 创建 EagerAndRecordGraphs 后端对象
            backend = EagerAndRecordGraphs()
            
            # 调用输入函数 fn，并记录实际输出
            actual = fn(*inputs)
            
            # 使用 torch.compile 函数编译输入函数 fn，获取预期输出
            expected = torch.compile(fn, backend=backend, fullgraph=fullgraph)(*inputs)
            
            # 断言实际输出和预期输出相等
            self.assertEqual(actual, expected)
            
            # 获取包装后的图形模块对象
            wrapped_gm = backend.graphs[graph_idx]
            return wrapped_gm
        
        # test_hessian 方法，测试求解 Hessian 矩阵的函数
        def test_hessian(self):
            # 清空计数器
            counters.clear()
            
            # 定义 wrapper_fn 函数，用于包装求解 sin 函数的 Hessian 矩阵计算
            def wrapper_fn(x):
                return torch.func.hessian(torch.sin)(x)
            
            # 生成随机张量 x
            x = torch.randn(4, 3)
            
            # 执行 _compile_check 方法，进行编译检查
            wrapped_gm = self._compile_check(wrapper_fn, (x,))
            
            # 如果检测到动态形状捕获，则直接返回
            if check_dynamic_shape_capture():
                return
            
            # 标准化图形模块对象的输出，确保可以进行内联比较
            actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
            
            # 断言实际输出与预期输出（标准化后的图形模块表示）相等
            self.assertExpectedInline(
                actual,
                """\
class GraphModule(torch.nn.Module):
""",
            )
        
        # test_hessian_argnums 方法，测试带参数的 Hessian 矩阵计算
        def test_hessian_argnums(self):
            # 清空计数器
            counters.clear()
            
            # 定义 fn 函数，接受两个参数 x 和 y，返回 x 的正弦值
            def fn(x, y):
                return x.sin()
            
            # 定义 wrapper_fn 函数，用于包装带参数的 Hessian 矩阵计算
            def wrapper_fn(x, y):
                return torch.func.hessian(fn, argnums=(1,))(x, y)
            
            # 生成随机张量 x 和 y
            x = torch.randn(4, 3)
            y = torch.randn(3, 4)
            
            # 执行 _compile_check 方法，进行编译检查
            wrapped_gm = self._compile_check(wrapper_fn, (x, y))
            
            # 如果检测到动态形状捕获，则直接返回
            if check_dynamic_shape_capture():
                return
            
            # 标准化图形模块对象的输出，确保可以进行内联比较
            actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
            
            # 断言实际输出与预期输出（标准化后的图形模块表示）相等，但不包括最后两行
            self.assertExpectedInline(
                "\n".join(actual.split("\n")[:-2]),
                """\
class GraphModule(torch.nn.Module):
""",
            )
    def test_hessian_disable_capture(self):
        counters.clear()  # 清空计数器，准备开始测试

        with config.patch(capture_func_transforms=False):  # 使用配置上下文，禁用函数捕获变换
            # 在上面验证了这个函数编译通过后
            def wrapper_fn(x):
                return torch.func.hessian(torch.sin)(x)  # 计算 torch.sin 的 Hessian 矩阵

            x = torch.randn(3, 3, 3)  # 生成一个形状为 (3, 3, 3) 的随机张量
            actual = wrapper_fn(x)  # 使用 wrapper_fn 计算实际结果
            expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
                x
            )  # 使用编译后端生成预期结果
            self.assertEqual(len(counters["graph_break"]), 2)  # 断言 graph_break 计数为 2
            self.assertEqual(
                {
                    "torch.func.vmap capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 2,
                    "torch.func.hessian capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 1,
                },
                dict(counters["graph_break"]),  # 断言 graph_break 计数字典符合预期
            )
            self.assertEqual(actual, expected)  # 断言实际结果与预期结果相等

    def test_jacrev(self):
        counters.clear()  # 清空计数器，准备开始测试

        def wrapper_fn(x):
            return torch.func.jacrev(torch.sin)(x)  # 计算 torch.sin 的 Jacobian 矩阵

        x = torch.randn(4, 3)  # 生成一个形状为 (4, 3) 的随机张量
        wrapped_gm = self._compile_check(wrapper_fn, (x,))  # 编译检查 wrapper_fn
        # 动态形状生成一个略有不同的图形。
        if check_dynamic_shape_capture():  # 检查是否需要捕获动态形状
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))  # 标准化打印输出
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    # 定义一个继承自torch.nn.Module的图模块类

    def forward(self, L_x_: "f32[4, 3]"):
        # 前向传播函数，接受一个形状为"f32[4, 3]"的输入参数L_x_

        l_x_ = L_x_
        # 复制输入参数L_x_到局部变量l_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        # 禁用保存的张量钩子，用于特定的autograd操作

        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()
        # 增加梯度嵌套层级，用于functorch库的操作

        diff_primals = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        # 使用functorch库对l_x_进行梯度包装，生成diff_primals，然后将l_x_置为None释放内存

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)
        # 设置允许原地操作需要梯度

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_primals)
        # 设置diff_primals的张量需要梯度

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)
        # 设置不允许原地操作需要梯度

        o = torch.sin(diff_primals)
        # 对diff_primals进行正弦函数操作，得到输出o

        results: "f32[4, 3]" = torch._C._functorch._unwrap_for_grad(o, 1)
        # 使用functorch库对o进行梯度解包，得到results

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        # 减少梯度嵌套层级

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        # 启用保存的张量钩子

        tensor: "i64[1]" = torch.tensor((12,))
        # 创建一个形状为(1,)的长整型张量tensor，赋值为12

        cumsum: "i64[1]" = tensor.cumsum(dim=0);  tensor = None
        # 计算tensor在维度0上的累加和，得到cumsum，并释放tensor的内存

        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        # 对cumsum进行切片操作，得到getitem，并释放cumsum的内存

        neg: "i64[0]" = getitem.neg();  getitem = None
        # 对getitem进行取负操作，得到neg，并释放getitem的内存

        unbind = neg.unbind();  neg = None
        # 对neg进行解绑操作，得到unbind，并释放neg的内存

        chunk: "f32[12, 12]" = results.new_zeros(12, 12);  results = None
        # 使用results的设备和数据类型，创建一个全零的形状为(12, 12)的张量chunk，并释放results的内存

        diagonal: "f32[12]" = chunk.diagonal(0)
        # 提取chunk的主对角线元素到diagonal

        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = None
        # 使用1填充diagonal的元素，并释放diagonal的内存

        basis: "f32[12, 4, 3]" = chunk.view(12, 4, 3);  chunk = None
        # 将chunk重塑为形状为(12, 4, 3)的张量basis，并释放chunk的内存

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()
        # 加载vmap库的惰性分解函数

        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        # 再次禁用保存的张量钩子，用于特定的autograd操作

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(12, 'error')
        # 增加vmap的嵌套层级，用于批处理操作，如果出错则抛出异常

        _add_batch_dim = torch._C._functorch._add_batch_dim(basis, 0, 1);  basis = None
        # 在basis上添加批处理维度，得到_add_batch_dim，并释放basis的内存

        _vjp_treespec_compare = torch._functorch.eager_transforms._vjp_treespec_compare(o, _add_batch_dim)
        # 比较o和_add_batch_dim的VJP树结构规范

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([o], [diff_primals], [_add_batch_dim], retain_graph=True, create_graph=True);  o = diff_primals = _add_batch_dim = None
        # 计算o关于o、diff_primals和_add_batch_dim的梯度，同时释放它们的内存

        batched_outputs = _autograd_grad[0];  _autograd_grad = None
        # 从_autograd_grad中提取批处理的输出，然后释放_autograd_grad的内存

        chunked_result: "f32[12, 4, 3]" = torch._C._functorch._remove_batch_dim(batched_outputs, 1, 12, 0);  batched_outputs = None
        # 移除批处理的结果维度，得到chunked_result，并释放batched_outputs的内存

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        # 减少vmap的嵌套层级

        _saved_tensors_hooks_enable_1 = torch._C._autograd._saved_tensors_hooks_enable()
        # 再次启用保存的张量钩子

        split = chunked_result.split((12,), dim=0);  chunked_result = None
        # 对chunked_result在维度0上进行拆分，得到split，并释放chunked_result的内存

        split_1: "f32[12, 4, 3]" = split[0];  split = None
        # 从split中提取第一个部分，得到split_1，并释放split的内存

        output_input: "f32[4, 3, 4, 3]" = split_1.view((4, 3, 4, 3));  split_1 = None
        # 将split_1重塑为形状为(4, 3, 4, 3)的张量output_input，并释放split_1的内存

        return (output_input,)
        # 返回output_input作为前向传播的结果，以元组形式返回
    def test_vjp(self):
        counters.clear()

        # 定义一个函数 fn，计算输入张量 x 的正弦值的和
        def fn(x):
            return x.sin().sum()

        # 定义一个包装函数 wrapper_fn，使用 torch.func.vjp 对 fn 进行反向自动求导
        def wrapper_fn(x, v):
            # 调用 torch.func.vjp 函数，对 fn 进行反向自动求导，返回两个值：out 和 vjpfunc
            (out, vjpfunc) = torch.func.vjp(fn, x)
            return out

        # 生成一个形状为 [5] 的随机张量 x 和相同形状的随机张量 v
        x = torch.randn([5])
        v = torch.randn(5)

        # 编译和检查 wrapper_fn 函数
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # 如果捕获动态形状，则返回
        if check_dynamic_shape_capture():
            return

        # 规范化打印的图形表示
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))

        # 断言实际输出是否符合预期
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
""",
        )
    # 定义一个名为 GraphModule 的 PyTorch 模块类
    class GraphModule(torch.nn.Module):
        
        # 定义模块的前向传播函数，接受一个参数 L_x_，类型为 'f32[5]'
        def forward(self, L_x_: "f32[5]"):
            
            # 将 L_x_ 复制给 l_x_
            l_x_ = L_x_

            # 禁用保存的张量钩子，用于 Torch 函数转换，给出一条警告信息作为参数
            _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
            
            # 增加梯度计数嵌套层级
            _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

            # 使用 Functorch 提供的函数包装 l_x_，使其支持梯度计算；并清空 l_x_ 的引用
            child = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

            # 允许就地操作的张量需要梯度
            set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

            # 使用 Functorch 提供的函数设置 child 的 requires_grad 为 True
            child_1 = torch._functorch.eager_transforms._set_tensor_requires_grad(child)

            # 禁止就地操作的张量需要梯度
            set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

            # 计算 child 的 sin 值，并将 child 的引用置空
            sin = child.sin();  child = None

            # 对 sin 的结果进行求和操作，并将 sin 的引用置空
            o = sin.sum();  sin = None

            # 使用 Functorch 提供的函数将 o 解封以获取梯度，1 表示它需要梯度信息
            results: "f32[]" = torch._C._functorch._unwrap_for_grad(o, 1);  o = None

            # 减少梯度计数嵌套层级
            _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
            
            # 启用保存的张量钩子
            _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()

            # 返回包含 results 的元组作为输出
            return (results,)

    # 定义一个测试函数 test_vjp_multiple_outputs
    def test_vjp_multiple_outputs(self):
        
        # 清空计数器
        counters.clear()

        # 定义一个包装函数 wrapper_fn，接受 x 和 v 作为参数
        def wrapper_fn(x, v):
            # 定义一个 lambda 函数 fn，返回 x 的 sin 和 cos 值
            fn = lambda x: (x.sin(), x.cos())  # noqa: E731
            
            # 使用 Torch 的 func 模块进行反向传播求导，得到结果 out 和 vjpfunc
            (out, vjpfunc) = torch.func.vjp(fn, x)
            
            # 使用 vjpfunc 计算 v 的偏导数，得到 vjps
            vjps = vjpfunc((v, v))
            
            # 返回 out 和 vjps 作为结果
            return out, vjps

        # 生成一个形状为 [5] 的随机张量 x
        x = torch.randn([5])
        
        # 生成一个形状为 [5] 的随机张量 v
        v = torch.randn(5)
        
        # 编译和检查 wrapper_fn 函数，使用 x 和 v 作为输入
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # 如果检查动态形状捕获返回 True，则返回
        if check_dynamic_shape_capture():
            return

        # 标准化 wrapped_gm 的可读输出，并禁用打印输出
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        
        # 使用 self 断言预期的内联输出与实际输出相符
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5]", L_v_: "f32[5]"):
        # 将输入参数赋值给局部变量 l_x_ 和 l_v_
        l_x_ = L_x_
        l_v_ = L_v_

        # 禁用保存的张量钩子，用于特定的 Torch 函数转换
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        # 增加梯度嵌套层级，用于 Functorch
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        # 将 l_x_ 包装为支持梯度计算的张量
        child = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        # 设置允许就地操作需要梯度的张量
        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        # 设置 child 对象的 requires_grad 属性为 True
        child_3 = torch._functorch.eager_transforms._set_tensor_requires_grad(child)

        # 设置允许就地操作需要梯度的张量为 False
        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        # 计算 child 的 sin 和 cos 值，并释放 child 对象
        child_1 = child.sin()
        child_2 = child.cos();  child = None

        # 将 child_1 和 child_2 解包为不需要梯度的张量
        _unwrap_for_grad: "f32[5]" = torch._C._functorch._unwrap_for_grad(child_1, 1)
        _unwrap_for_grad_1: "f32[5]" = torch._C._functorch._unwrap_for_grad(child_2, 1)

        # 减少梯度嵌套层级
        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()

        # 启用保存的张量钩子
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()

        # 比较 VJP (Vector-Jacobian Product) 树规范
        _vjp_treespec_compare = torch._functorch.eager_transforms._vjp_treespec_compare((child_1, child_2), (l_v_, l_v_))

        # 执行自动求导过程，计算梯度并返回结果
        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([child_1, child_2], [child_3], [l_v_, l_v_], retain_graph = True, create_graph = True);  child_1 = child_2 = child_3 = l_v_ = None
        getitem: "f32[5]" = _autograd_grad[0];  _autograd_grad = None
        return (_unwrap_for_grad, _unwrap_for_grad_1, getitem)
# 清除计数器，用于跟踪函数调用次数或其他计数器
counters.clear()

# 定义包装函数 wrapper_fn，接受输入 x 和 v
def wrapper_fn(x, v):
    # 定义 lambda 函数 fn，返回一个字典，包含 x.sin() 和 x.cos() 的结果
    fn = lambda x: {"first": x.sin(), "second": x.cos()}  # noqa: E731

    # 调用 torch.func.vjp 函数，获取输出 out 和 vjpfunc
    (out, vjpfunc) = torch.func.vjp(fn, x)

    # 使用 vjpfunc 处理输入字典 {"first": v, "second": v.sin()}，获取对应的梯度信息
    vjps = vjpfunc({"first": v, "second": v.sin()})

    # 返回 out 和 vjps 作为结果
    return out, vjps

# 创建一个随机张量 x
x = torch.randn([5])

# 创建一个随机张量 v
v = torch.randn(5)

# 使用 self._compile_check 方法编译并检查 wrapper_fn 的执行结果
wrapped_gm = self._compile_check(wrapper_fn, (x, v))

# 如果检查动态形状捕获，返回
if check_dynamic_shape_capture():
    return

# 对 wrapped_gm 的可读输出进行规范化处理
actual = normalize_gm(wrapped_gm.print_readable(print_output=False))

# 断言实际输出与预期输出的内联格式是否相符
self.assertExpectedInline(
    actual,
    """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5]", L_v_: "f32[5]"):
        l_x_ = L_x_
        l_v_ = L_v_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        child = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        child_3 = torch._functorch.eager_transforms._set_tensor_requires_grad(child)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        child_1 = child.sin()
        child_2 = child.cos();  child = None

        _unwrap_for_grad: "f32[5]" = torch._C._functorch._unwrap_for_grad(child_1, 1)
        _unwrap_for_grad_1: "f32[5]" = torch._C._functorch._unwrap_for_grad(child_2, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()

        child_4: "f32[5]" = l_v_.sin()

        _vjp_treespec_compare = torch._functorch.eager_transforms._vjp_treespec_compare({'first': child_1, 'second': child_2}, {'first': l_v_, 'second': child_4})

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([child_1, child_2], [child_3], [l_v_, child_4], retain_graph = True, create_graph = True);  child_1 = child_2 = child_3 = l_v_ = child_4 = None
        getitem: "f32[5]" = _autograd_grad[0];  _autograd_grad = None
        return (_unwrap_for_grad, _unwrap_for_grad_1, getitem)
""",
)
        # 清空计数器，准备测试环境
        counters.clear()

        # 定义函数 fn，计算输入张量 x 的 sin 值的总和，并返回该总和与输入张量 x 本身
        def fn(x):
            return x.sin().sum(), x

        # 定义包装函数 wrapper_fn，接受输入张量 x 和梯度 v，调用 torch 的反向自动微分函数 vjp，并设置 has_aux=True
        def wrapper_fn(x, v):
            (out, vjpfunc, _) = torch.func.vjp(fn, x, has_aux=True)
            return out

        # 生成一个形状为 [5] 的随机张量 x 和其梯度 v
        x = torch.randn([5])
        v = torch.randn(5)

        # 使用 self._compile_check 方法编译和检查 wrapper_fn 的图形态，并保存为 wrapped_gm
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # 如果检测到动态形状捕获，则直接返回，不进行后续断言
        if check_dynamic_shape_capture():
            return

        # 调用 wrapped_gm 的 print_readable 方法，以获得可读的输出，并进行规范化处理
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))

        # 使用 self.assertExpectedInline 方法断言 actual 结果与预期输出的一致性
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    # 定义神经网络模块的类，继承自 torch.nn.Module

    def forward(self, L_x_: "f32[5]"):
        # 定义 forward 方法，接收参数 L_x_，类型为 f32[5]

        l_x_ = L_x_
        # 将 L_x_ 赋值给 l_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        # 调用 _saved_tensors_hooks_disable 函数，禁用保存的张量钩子

        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()
        # 调用 _grad_increment_nesting 函数，增加梯度嵌套层数

        child = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        # 调用 _wrap_for_grad 函数，为 l_x_ 创建梯度包装对象 child，然后将 l_x_ 置为 None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)
        # 设置允许原地操作的张量梯度需要

        child_1 = torch._functorch.eager_transforms._set_tensor_requires_grad(child)
        # 调用 _set_tensor_requires_grad 函数，设置 child 张量需要梯度

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)
        # 设置不允许原地操作的张量梯度需要

        sin = child.sin()
        # 对 child 执行正弦函数操作

        o = sin.sum();  sin = None
        # 对 sin 执行求和操作，并将 sin 置为 None

        aux: "f32[5]" = torch._C._functorch._unwrap_for_grad(child, 1);  child = None
        # 调用 _unwrap_for_grad 函数，从 child 中提取梯度信息，存储到 aux 中，然后将 child 置为 None

        results: "f32[]" = torch._C._functorch._unwrap_for_grad(o, 1);  o = None
        # 调用 _unwrap_for_grad 函数，从 o 中提取梯度信息，存储到 results 中，然后将 o 置为 None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        # 调用 _grad_decrement_nesting 函数，减少梯度嵌套层数

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        # 调用 _saved_tensors_hooks_enable 函数，启用保存的张量钩子

        return (results,)
        # 返回结果 results 的元组
    # 定义一个方法 `forward`，接受一个名为 `L_x_` 的参数，类型为 `f32[3, 3, 3]`
    def forward(self, L_x_: "f32[3, 3, 3]"):
        # 将参数赋值给局部变量 l_x_
        l_x_ = L_x_

        # 禁用保存的张量钩子，用于提示不支持保存张量钩子的警告信息
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        
        # 增加梯度嵌套计数
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        # 对 l_x_ 进行梯度封装，返回封装后的对象，同时将 l_x_ 置空
        diff_args = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        # 设置允许原地操作的张量需要梯度
        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        # 设置 diff_args 中的张量需要梯度
        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args)

        # 设置不允许原地操作的张量需要梯度
        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        # 计算 diff_args 中的 sin 函数，得到输出并求和
        sin = diff_args.sin()
        output = sin.sum();  sin = None

        # 执行 autograd 的梯度计算，并创建计算图
        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        # 对 grad_input 进行梯度解封，返回解封后的对象，同时将 grad_input 置空
        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        # 对 output 进行梯度解封，返回解封后的对象，同时将 output 置空
        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = None

        # 减少梯度嵌套计数
        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()

        # 启用保存的张量钩子
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()

        # 返回包含 grad_input_1 的元组作为输出
        return (grad_input_1,)
    def test_grad_freevar_tensor(self):
        # 清空计数器
        counters.clear()
        # 生成一个形状为 (3, 3) 的随机张量 y
        y = torch.randn(3, 3)

        def fn(x):
            # 定义一个函数 fn，计算 x 的正弦值加上 y 的和，并返回总和
            return (x.sin() + y).sum()

        def wrapper_fn(x):
            # 调用 torch.func.grad 函数，计算函数 fn 对 x 的梯度
            return torch.func.grad(fn)(x)

        # 生成一个形状为 (3, 3, 3) 的随机张量 x
        x = torch.randn(3, 3, 3)
        # 通过 AOT eager 模式编译 wrapper_fn 函数，并获得编译后的输出
        actual = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)(x)
        # 断言编译后的输出与预期结果相等
        self.assertEqual(actual, expected)

    def test_grad_freevar_python_scalar(self):
        # 清空计数器
        counters.clear()
        # 定义一个标量 y 等于 3
        y = 3

        def fn(x):
            # 定义一个函数 fn，计算 x 的正弦值加上标量 y 的和，并返回总和
            return (x.sin() + y).sum()

        def wrapper_fn(x):
            # 调用 torch.func.grad 函数，计算函数 fn 对 x 的梯度
            return torch.func.grad(fn)(x)

        # 生成一个形状为 (3, 3, 3) 的随机张量 x
        x = torch.randn(3, 3, 3)
        
        # 检查是否支持动态形状捕获，若支持则直接返回，不进行后续断言
        if check_dynamic_shape_capture():
            return

        # 对编译后的图模块进行标准化，获取可读的打印输出
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        # 断言标准化后的输出与内联预期结果相等
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        diff_args = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        sin = diff_args.sin()
        add = sin + 3;  sin = None
        output = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (grad_input_1,)
""",
        )
    def test_grad_capture_tensor(self):
        # 清空计数器
        counters.clear()

        def wrapper_fn(x):
            # 生成一个形状为(3,)的随机张量y
            y = torch.randn(3)

            def fn(x):
                # 计算函数 fn(x) = (sin(x) + y).sum() 的梯度
                return (x.sin() + y).sum()

            # 使用 torch.func.grad 计算 fn 的梯度
            return torch.func.grad(fn)(x)

        # 生成一个形状为(3, 3, 3)的随机张量x
        x = torch.randn(3, 3, 3)

        # 调用 _compile_check 方法进行编译和检查
        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # 如果检查动态形状捕获，则返回
        if check_dynamic_shape_capture():
            return

        # 对 wrapped_gm 进行可读性打印，并进行规范化
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        # 使用 self.assertExpectedInline 断言实际输出是否符合预期
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    # 定义神经网络模块类 GraphModule，继承自 torch.nn.Module

    def forward(self, L_x_: "f32[3, 3, 3]"):
        # 定义前向传播函数 forward，接受参数 L_x_，类型为 f32[3, 3, 3]

        l_x_ = L_x_
        # 将输入参数 L_x_ 赋值给局部变量 l_x_

        y: "f32[3]" = torch.randn(3)
        # 创建一个形状为 [3] 的随机张量 y

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        # 禁用保存张量钩子，用于在函数变换期间禁用保存张量的钩子

        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()
        # 增加梯度计算嵌套层数，用于跟踪嵌套的梯度计算

        diff_args = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        # 将 l_x_ 包装为可用于梯度计算的对象，并清空 l_x_

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)
        # 设置允许原地操作需要梯度的张量

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args)
        # 设置需要梯度的张量，用于变换中的张量操作

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)
        # 设置不允许原地操作需要梯度的张量

        sin = diff_args.sin()
        # 对包装后的对象进行正弦运算

        add = sin + y;  sin = None
        # 计算 sin 和 y 的和，并清空 sin

        output = add.sum();  add = None
        # 对加法结果进行求和操作，得到输出结果，并清空 add

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph=True);  diff_args = None
        # 对输出结果进行自动求导计算梯度，并清空 diff_args

        grad_input = _autograd_grad[0];  _autograd_grad = None
        # 获取输入梯度，并清空 _autograd_grad

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None
        # 解包梯度输入，得到最终的梯度张量 grad_input，并清空 grad_input

        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = None
        # 解包输出结果，得到最终的输出张量 output，并清空 output

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        # 减少梯度计算嵌套层数

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        # 启用保存张量的钩子

        return (y, grad_input_1)
        # 返回结果元组 (y, grad_input_1)，其中 y 是随机张量，grad_input_1 是输入的梯度张量
    # 定义一个方法 `forward`，接受一个名为 L_x_ 的参数，类型为 "f32[3, 3, 3]"
    def forward(self, L_x_: "f32[3, 3, 3]"):
        # 将参数 L_x_ 赋值给局部变量 l_x_
        l_x_ = L_x_

        # 禁用 Torch 中保存张量钩子的功能，以避免与函数转换冲突，并给出警告信息
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        # 增加梯度计算嵌套深度，用于函数转换
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        # 将 l_x_ 包装以进行梯度计算
        diff_args = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        # 允许就地操作的张量需要梯度
        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        # 设置 diff_args 中张量的梯度属性
        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args)

        # 禁止就地操作的张量需要梯度
        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        # 计算 diff_args 中张量 sin 的正弦值
        sin = diff_args.sin()
        # 将 sin 加上常数 3.14
        add = sin + 3.14;  sin = None
        # 计算 add 张量的所有元素之和
        output = add.sum();  add = None

        # 计算 output 的梯度，并创建一个梯度计算图
        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        # 提取梯度计算结果中的输入梯度
        grad_input = _autograd_grad[0];  _autograd_grad = None

        # 解包 grad_input，恢复其原始的梯度计算状态
        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        # 解包 output，恢复其原始的梯度计算状态
        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = None

        # 减少梯度计算嵌套深度，用于函数转换
        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        # 启用 Torch 中保存张量钩子的功能
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        
        # 返回最终的梯度输入作为一个元组
        return (grad_input_1,)
    def test_grad_has_aux(self):
        # 清空计数器
        counters.clear()

        # 定义常量 y
        y = 3.14

        # 定义计算函数 fn(x)，返回两个值的和及 x 的余弦值
        def fn(x):
            return ((x.sin() + y).sum(), x.cos())

        # 定义包装函数 wrapper_fn(x)，使用 torch.func.grad 计算 fn 的梯度，并指定支持辅助输出
        def wrapper_fn(x):
            return torch.func.grad(fn, has_aux=True)(x)

        # 生成随机张量 x
        x = torch.randn(3, 3, 3)
        
        # 编译检查 wrapper_fn 并获取编译后的图形模块
        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # 检查动态形状捕获
        if check_dynamic_shape_capture():
            return

        # 规范化图形模块的可读输出
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        
        # 断言实际输出与预期输出的内联文本相符
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        diff_args = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        sin = diff_args.sin()
        add = sin + 3.14;  sin = None
        output = add.sum();  add = None
        aux = diff_args.cos()

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = None

        aux_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (grad_input_1, aux_1)
""",
        )

    def test_grad_two_tensor_has_aux(self):
        # 清空计数器
        counters.clear()

        # 定义计算函数 fn(x, y)，返回两个值的和及 x 的余弦值
        def fn(x, y):
            return ((x.sin() + y).sum(), x.cos())

        # 定义包装函数 wrapper_fn(x, y)，使用 torch.func.grad 计算 fn 的梯度，并指定支持辅助输出
        def wrapper_fn(x, y):
            return torch.func.grad(fn, has_aux=True)(x, y)

        # 生成随机张量 y 和 x
        y = torch.randn(3, 3, 3)
        x = torch.randn(3, 3, 3)
        
        # 编译检查 wrapper_fn 并获取编译后的图形模块
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))

        # 检查动态形状捕获
        if check_dynamic_shape_capture():
            return

        # 规范化图形模块的可读输出
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        
        # 断言实际输出与预期输出的内联文本相符
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    # 定义一个前向传播函数，接受两个输入参数
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3, 3]"):
        # 将输入参数赋值给局部变量
        l_x_ = L_x_
        l_y_ = L_y_

        # 禁用保存张量钩子
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        # 增加梯度嵌套层数
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        # 将 l_x_ 包装为可求导张量
        diff_args = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        # 将 l_y_ 包装为可求导张量
        _wrap_for_grad_1 = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        # 设置允许原地操作需要梯度
        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        # 设置张量需要梯度
        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args)

        # 设置不允许原地操作需要梯度
        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        # 计算 diff_args 的正弦值
        sin = diff_args.sin()
        # 计算正弦值与 _wrap_for_grad_1 的和
        add = sin + _wrap_for_grad_1;  sin = _wrap_for_grad_1 = None
        # 对 add 求和
        output = add.sum();  add = None
        # 计算 diff_args 的余弦值
        aux = diff_args.cos()

        # 计算梯度
        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        # 将梯度张量解包
        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        # 将输出张量解包
        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = None

        # 将辅助张量解包
        aux_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        # 减少梯度嵌套层数
        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        # 启用保存张量钩子
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        # 返回梯度和辅助张量
        return (grad_input_1, aux_1)
# 在这里开始一个新的测试函数，测试梯度计算函数在所有梯度都具有辅助信息的情况下的行为
def test_grad_two_tensor_all_grad_has_aux(self):
    counters.clear()

    # 定义两个变量的序号
    nums = (0, 1)

    # 定义一个函数 fn，计算输入张量 x 的正弦和张量 y 的和的总和，同时返回张量 x 的余弦
    def fn(x, y):
        return ((x.sin() + y).sum(), x.cos())

    # 定义一个包装函数 wrapper_fn_const_var，用于计算 fn 的梯度，并指定其参数的序号，并声明有辅助信息
    def wrapper_fn_const_var(x, y):
        return torch.func.grad(fn, argnums=(0, 1), has_aux=True)(x, y)

    # 定义一个包装函数 wrapper_fn_tuple_var，同样用于计算 fn 的梯度，但这次指定参数序号为 nums，并声明有辅助信息
    def wrapper_fn_tuple_var(x, y):
        return torch.func.grad(fn, argnums=nums, has_aux=True)(x, y)

    # 生成随机张量 y 和 x
    y = torch.randn(3, 3, 3)
    x = torch.randn(3, 3, 3)

    # 编译和检查 wrapper_fn_const_var 和 wrapper_fn_tuple_var 的图表达式
    wrapped_gm_const_var = self._compile_check(wrapper_fn_const_var, (x, y))
    wrapped_gm_tuple_var = self._compile_check(wrapper_fn_tuple_var, (x, y))

    # 动态形状生成略有不同的图形。
    if check_dynamic_shape_capture():
        return

    # 标准化 wrapped_gm_const_var 和 wrapped_gm_tuple_var 的图形模块，使其输出可读性更好
    actual_const_var = normalize_gm(
        wrapped_gm_const_var.print_readable(print_output=False)
    )
    actual_tuple_var = normalize_gm(
        wrapped_gm_tuple_var.print_readable(print_output=False)
    )

    # 断言 actual_const_var 和 actual_tuple_var 与预期的内联字符串相匹配
    self.assertExpectedInline(
        actual_const_var,
        """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3, 3]"):
        # 接收输入的两个张量 L_x_ 和 L_y_
        l_x_ = L_x_
        l_y_ = L_y_

        # 禁用保存张量钩子，因为 torch.func 目前不支持保存张量钩子
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        
        # 增加梯度嵌套深度
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        # 将 l_x_ 包装为支持梯度的对象
        child = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        # 将 l_y_ 包装为支持梯度的对象
        child_1 = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        # 允许原地操作需要梯度
        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        # 设置张量 child 需要梯度
        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(child)

        # 不再允许原地操作需要梯度
        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)
        # 再次允许原地操作需要梯度
        set_inplace_requires_grad_allowed_2 = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        # 设置张量 child_1 需要梯度
        _set_tensor_requires_grad_1 = torch._functorch.eager_transforms._set_tensor_requires_grad(child_1)

        # 不再允许原地操作需要梯度
        set_inplace_requires_grad_allowed_3 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        # 对 child 执行 sin 操作
        sin = child.sin()
        # 将 sin 结果与 child_1 相加
        add = sin + child_1;  sin = None
        # 计算 add 的和
        output = add.sum();  add = None
        # 对 child 执行 cos 操作
        aux = child.cos()

        # 计算输出 output 和 aux 的梯度，同时保留计算图
        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [child, child_1], create_graph = True);  child = child_1 = None
        # 获取计算后的梯度张量 child_2 和 child_3
        child_2 = _autograd_grad[0]
        child_3 = _autograd_grad[1];  _autograd_grad = None

        # 将 child_2 解封，返回原始张量
        _unwrap_for_grad: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(child_2, 1);  child_2 = None
        # 将 child_3 解封，返回原始张量
        _unwrap_for_grad_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(child_3, 1);  child_3 = None

        # 将 output 解封，返回原始张量
        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = None

        # 将 aux 解封，返回原始张量
        aux_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        # 减少梯度嵌套深度
        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        # 启用保存张量钩子
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        # 返回解封后的张量
        return (_unwrap_for_grad, _unwrap_for_grad_1, aux_1)
class GraphModule(torch.nn.Module):
    # 定义神经网络模块，继承自PyTorch的Module类

    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3, 3]"):
        # 前向传播函数，接受两个输入参数 L_x_ 和 L_y_

        l_x_ = L_x_
        # 将输入 L_x_ 赋值给 l_x_

        l_y_ = L_y_
        # 将输入 L_y_ 赋值给 l_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        # 禁用保存的张量钩子，用于支持函数转换，如果有需要，会打开对应的issue

        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()
        # 增加梯度嵌套层级计数

        child = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        # 对 l_x_ 进行梯度包装，级别为1，同时清空 l_x_

        child_1 = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None
        # 对 l_y_ 进行梯度包装，级别为1，同时清空 l_y_

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)
        # 设置允许就地操作需要梯度为真

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(child)
        # 设置张量需要梯度，针对 child

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)
        # 设置允许就地操作需要梯度为假

        set_inplace_requires_grad_allowed_2 = torch._C._functorch.set_inplace_requires_grad_allowed(True)
        # 再次设置允许就地操作需要梯度为真

        _set_tensor_requires_grad_1 = torch._functorch.eager_transforms._set_tensor_requires_grad(child_1)
        # 设置张量需要梯度，针对 child_1

        set_inplace_requires_grad_allowed_3 = torch._C._functorch.set_inplace_requires_grad_allowed(False)
        # 再次设置允许就地操作需要梯度为假

        sin = child.sin()
        # 计算 child 张量的正弦值

        add = sin + child_1;  sin = None
        # 将 sin 与 child_1 相加，然后清空 sin

        output = add.sum();  add = None
        # 计算 add 张量的元素和，然后清空 add

        aux = child.cos()
        # 计算 child 的余弦值，并赋值给 aux

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [child, child_1], create_graph = True);  child = child_1 = None
        # 计算 output 的梯度，同时清空 child 和 child_1 的引用

        child_2 = _autograd_grad[0]
        # 取出 _autograd_grad 的第一个元素，赋值给 child_2

        child_3 = _autograd_grad[1];  _autograd_grad = None
        # 取出 _autograd_grad 的第二个元素，并清空 _autograd_grad

        _unwrap_for_grad: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(child_2, 1);  child_2 = None
        # 对 child_2 进行梯度解包，级别为1，同时清空 child_2

        _unwrap_for_grad_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(child_3, 1);  child_3 = None
        # 对 child_3 进行梯度解包，级别为1，同时清空 child_3

        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = None
        # 对 output 进行梯度解包，级别为1，同时清空 output

        aux_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None
        # 对 aux 进行梯度解包，级别为1，同时清空 aux

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        # 减少梯度嵌套层级计数

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        # 启用保存的张量钩子

        return (_unwrap_for_grad, _unwrap_for_grad_1, aux_1)
        # 返回梯度解包后的结果元组
    def forward(self, L_x_: "f32[]"):
        # 将输入参数复制给局部变量 l_x_
        l_x_ = L_x_

        # 禁用保存的张量钩子
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        # 增加梯度嵌套层级
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        # 对输入参数 l_x_ 进行梯度封装
        diff_args = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        # 允许就地操作需要梯度的张量
        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        # 设置张量需要梯度
        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args)

        # 禁用就地操作需要梯度的张量
        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        # 再次禁用保存的张量钩子
        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        # 再次增加梯度嵌套层级
        _grad_increment_nesting_1 = torch._C._functorch._grad_increment_nesting()

        # 对更新后的参数进行梯度封装
        diff_args_1 = torch._C._functorch._wrap_for_grad(diff_args, 2)

        # 再次允许就地操作需要梯度的张量
        set_inplace_requires_grad_allowed_2 = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        # 再次设置张量需要梯度
        _set_tensor_requires_grad_1 = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args_1)

        # 最后禁用就地操作需要梯度的张量
        set_inplace_requires_grad_allowed_3 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        # 对更新后的参数应用正弦函数
        sin = diff_args_1.sin()

        # 计算正弦函数的和作为输出
        output = sin.sum();  sin = None

        # 对输出进行自动微分，创建计算图
        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args_1], create_graph = True);  diff_args_1 = None

        # 获取输入的梯度
        grad_input = _autograd_grad[0];  _autograd_grad = None

        # 对梯度进行解封，恢复为原始梯度维度
        grad_input_1 = torch._C._functorch._unwrap_for_grad(grad_input, 2);  grad_input = None

        # 对输出进行解封，恢复为原始输出维度
        output_1 = torch._C._functorch._unwrap_for_grad(output, 2);  output = None

        # 减少梯度嵌套层级
        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()

        # 最后一次禁用保存的张量钩子
        _saved_tensors_hooks_disable_2 = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        # 再次进行自动微分，创建计算图
        _autograd_grad_1 = torch._functorch.eager_transforms._autograd_grad((grad_input_1,), [diff_args], create_graph = True);  diff_args = None

        # 获取输入的梯度
        grad_input_2 = _autograd_grad_1[0];  _autograd_grad_1 = None

        # 对梯度进行解封，恢复为原始梯度维度
        grad_input_3: "f32[]" = torch._C._functorch._unwrap_for_grad(grad_input_2, 1);  grad_input_2 = None

        # 对输出进行解封，恢复为原始输出维度
        output_2: "f32[]" = torch._C._functorch._unwrap_for_grad(grad_input_1, 1);  grad_input_1 = None

        # 减少梯度嵌套层级
        _grad_decrement_nesting_1 = torch._C._functorch._grad_decrement_nesting()

        # 启用保存的张量钩子
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()

        # 返回梯度输入作为结果
        return (grad_input_3,)
    """
    确保在图中插入一个中断点以进行梯度计算
    """
    def test_grad_with_graph_break(self):
        # 清空计数器
        counters.clear()

        def fn(x):
            # 在计算图中插入中断点
            torch._dynamo.graph_break()
            # 计算正弦函数的和
            return x.sin().sum()

        def wrapper_fn(x):
            # 返回函数 fn 的梯度
            return torch.func.grad(fn)(x)

        # 生成一个3x3x3的随机张量
        x = torch.randn(3, 3, 3)
        # 获取实际梯度值
        actual = wrapper_fn(x)
        # 使用预编译的方式获取期望的梯度值
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x)
        # 断言中断点计数为1
        self.assertEqual(len(counters["graph_break"]), 1)
        # 断言实际值与期望值相等
        self.assertEqual(actual, expected)

    """
    确保在计算梯度时没有副作用
    """
    def test_grad_with_side_effect(self):
        # 清空计数器
        counters.clear()

        foo = [1, 2]

        def fn(x):
            # 在函数中修改副作用变量 foo
            foo.append(3)
            # 计算正弦函数的和
            return x.sin().sum()

        def wrapper_fn(x):
            # 返回函数 fn 的梯度
            return torch.func.grad(fn)(x)

        # 生成一个3x3x3的随机张量
        x = torch.randn(3, 3, 3)
        # 获取实际梯度值
        actual = wrapper_fn(x)
        # 使用预编译的方式获取期望的梯度值
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x)
        # 断言中断点计数为0
        self.assertEqual(len(counters["graph_break"]), 0)
        # 断言实际值与期望值相等
        self.assertEqual(actual, expected)

    """
    确保在计算多元张量梯度时没有副作用
    """
    def test_grad_pytree(self):
        # 清空计数器
        counters.clear()

        def fn(x):
            x1, x2 = x
            # 计算第一个张量的正弦函数和第二个张量的和
            return x1.sin().sum() + x2

        def wrapper_fn(x):
            # 返回函数 fn 的梯度
            return torch.func.grad(fn)(x)

        # 生成一个3x3x3的随机张量和一个标量的随机张量
        x1 = torch.randn(3, 3, 3)
        x2 = torch.randn(())
        # 获取实际梯度值
        actual = wrapper_fn((x1, x2))
        # 使用预编译的方式获取期望的梯度值
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
            (x1, x2)
        )
        # 断言中断点计数为0
        self.assertEqual(len(counters["graph_break"]), 0)
        # 断言实际值与期望值相等
        self.assertEqual(actual, expected)

    """
    确保在计算张量和标量梯度时没有副作用
    """
    def test_grad_non_tensor_input(self):
        # 清空计数器
        counters.clear()

        def fn(x, y):
            # 计算第一个张量的正弦函数和第二个标量的和
            return x.sin().sum() + y

        def wrapper_fn(x, y):
            # 返回函数 fn 的梯度
            return torch.func.grad(fn)(x, y)

        # 生成一个3x3x3的随机张量和一个标量值
        x = torch.randn(3, 3, 3)
        y = 3.0
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))

        # 动态形状产生稍有不同的图形。
        if check_dynamic_shape_capture():
            return

        # 标准化图形模块以便进行比较
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        # 断言实际输出与预期输出相等
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    # 定义一个方法 `forward`，接受一个名为 L_x_ 的参数，其类型为 "f32[3, 3, 3]"
    def forward(self, L_x_: "f32[3, 3, 3]"):
        # 将 L_x_ 参数赋值给 l_x_
        l_x_ = L_x_

        # 禁用保存张量钩子的功能，因为 torch.func 目前不支持保存张量钩子
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        # 增加梯度嵌套计数器
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        # 使用 functorch 的函数将 l_x_ 包装以进行梯度计算
        diff_args = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        # 允许就地操作需要梯度的张量
        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        # 设置 diff_args 张量需要梯度
        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args)

        # 禁止就地操作需要梯度的张量
        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        # 计算 diff_args 张量的正弦值
        sin = diff_args.sin()
        # 对正弦值张量进行求和
        sum_1 = sin.sum();  sin = None
        # 计算输出值，为正弦值求和结果加上 3.0
        output = sum_1 + 3.0;  sum_1 = None

        # 使用 functorch 的函数计算 output 的梯度，并创建计算图
        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        # 获取计算得到的梯度输入
        grad_input = _autograd_grad[0];  _autograd_grad = None

        # 将梯度输入张量 unwrap 以用于梯度计算
        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        # 将输出张量 unwrap 以用于梯度计算
        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = None

        # 减少梯度嵌套计数器
        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        # 启用保存张量钩子功能
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        
        # 返回梯度输入作为元组的形式
        return (grad_input_1,)
    def test_grad_disable_capture(self):
        # 清空计数器
        counters.clear()

        # 禁用捕获函数变换的配置
        with config.patch(capture_func_transforms=False):
            # 在上面已验证此函数编译通过
            def fn(x):
                return x.sin().sum()

            def wrapper_fn(x):
                return torch.func.grad(fn)(x)

            # 创建一个随机张量
            x = torch.randn(3, 3)
            # 使用 wrapper_fn 计算梯度
            actual = wrapper_fn(x)
            # 使用 AOT eager 模式编译 wrapper_fn
            expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
                x
            )
            # 断言图断点的数量为 1
            self.assertEqual(len(counters["graph_break"]), 1)
            # 断言计数器中的内容
            self.assertEqual(
                dict(counters["graph_break"]),
                {
                    "torch.func.grad capture is disabled, it can be turned "
                    "on by setting `torch._dynamo.config.capture_func_transforms=True`": 2
                },
            )
            # 断言实际输出等于预期输出
            self.assertEqual(actual, expected)

    def test_grad_fn_with_kwargs(self):
        # 定义一个函数 fn，接受两个参数并返回它们的和的总和
        def fn(x, y):
            return (x + y).sum()

        def wrapper_fn(x, y):
            # 使用 torch.func.grad 获取 fn 的梯度函数
            return torch.func.grad(fn)(x, y=y)

        # 创建两个随机张量
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        # 使用 wrapper_fn 计算梯度
        actual = wrapper_fn(x, y)
        # 使用 AOT eager 模式编译 wrapper_fn
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x, y)
        # 断言图断点的数量为 0
        self.assertEqual(len(counters["graph_break"]), 0)
        # 断言实际输出等于预期输出
        self.assertEqual(actual, expected)

    def test_jacfwd(self):
        # 清空计数器
        counters.clear()

        def wrapper_fn(x):
            # 使用 torch.func.jacfwd 计算 torch.sin 的雅可比矩阵乘以输入 x
            return torch.func.jacfwd(torch.sin)(x)

        # 创建一个随机张量
        x = torch.randn(4, 3)
        # 进行编译检查并获取编译后的图形模块
        wrapped_gm = self._compile_check(wrapper_fn, (x,))
        # 动态形状会产生略有不同的图形
        if check_dynamic_shape_capture():
            return

        # 标准化图形模块的可读输出
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        # 断言实际输出与预期的内联字符串相等
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
""",
        )

    def test_jacfwd_two_tensors_argnums(self):
        # 清空计数器
        counters.clear()

        def fn(x, y):
            # 返回输入张量 y 的正弦值
            return y.sin()

        def wrapper_fn(x, y):
            # 使用 torch.func.jacfwd 获取 fn 函数的雅可比矩阵，并指定 argnums=1
            return torch.func.jacfwd(fn, argnums=1)(x, y)

        # 创建两个随机张量
        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        # 进行编译检查并获取编译后的图形模块
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # 动态形状会产生略有不同的图形
        if check_dynamic_shape_capture():
            return

        # 标准化图形模块的可读输出
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        # 断言实际输出与预期的内联字符串相等
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
""",
        )
    # 定义一个测试方法，用于验证 jacfwd 是否有辅助功能
    def test_jacfwd_has_aux(self):
        # 清空计数器
        counters.clear()

        # 定义一个函数 fn，接受两个参数 x 和 y，并返回 y.sin() 和 x
        def fn(x, y):
            return y.sin(), x

        # 定义一个包装函数 wrapper_fn，接受参数 x 和 y，调用 torch.func.jacfwd 函数，
        # 该函数使用 fn 作为输入函数，argnums=1 表示在第二个参数上计算 Jacobian，
        # has_aux=True 表示 fn 具有辅助输出
        def wrapper_fn(x, y):
            return torch.func.jacfwd(fn, argnums=1, has_aux=True)(x, y)

        # 生成随机张量 x 和 y
        x = torch.randn(4, 3)
        y = torch.randn(3, 4)

        # 编译并检查 wrapper_fn，返回编译后的图形式表示 wrapped_gm
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))

        # 检查是否需要捕捉动态形状，如果是，则直接返回
        if check_dynamic_shape_capture():
            return

        # 调用 wrapped_gm 的 print_readable 方法，打印可读性好的图形式表示，
        # 并将其标准化为规范形式，排除打印输出
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))

        # 断言实际输出与预期输出的内联形式相符
        self.assertExpectedInline(
            actual,
            """\
    def test_jacfwd_randomness(self):
        # 清空计数器
        counters.clear()

        def fn(x, y):
            # 返回 y 的正弦值和 x
            return y.sin(), x

        def wrapper_fn(x, y):
            # 使用相同的随机性对 fn 进行自动微分
            return torch.func.jacfwd(fn, randomness="same")(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # 动态形状会产生稍微不同的计算图
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
""",
        )

    def test_jacfwd_disable_capture(self):
        # 清空计数器
        counters.clear()

        with config.patch(capture_func_transforms=False):
            # 我们已经验证了这个函数可以编译通过
            def wrapper_fn(x):
                # 对 torch.sin 函数进行前向自动微分
                return torch.func.jacfwd(torch.sin)(x)

            x = torch.randn(3, 3, 3)
            actual = wrapper_fn(x)
            expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
                x
            )
            self.assertEqual(len(counters["graph_break"]), 2)
            self.assertEqual(
                dict(counters["graph_break"]),
                {
                    "torch.func.vmap capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 2,
                    "torch.func.jacfwd capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 1,
                },
            )
            self.assertEqual(actual, expected)

    def test_jvp_simple(self):
        # 清空计数器
        counters.clear()

        def fn(x):
            # 返回 x 的正弦值之和
            return x.sin().sum()

        def wrapper_fn(x, v):
            # 对 fn 函数进行雅可比向量积计算
            return torch.func.jvp(fn, (x,), (v,))

        x = torch.randn(3, 3)
        v = torch.randn(3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # 动态形状会产生稍微不同的计算图
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    # 定义一个方法 forward，接受两个参数 L_x_ 和 L_v_，类型为 "f32[3, 3]"
    def forward(self, L_x_: "f32[3, 3]", L_v_: "f32[3, 3]"):
        # 将参数 L_x_ 和 L_v_ 分别赋值给 l_x_ 和 l_v_
        l_x_ = L_x_
        l_v_ = L_v_

        # 禁用保存的张量钩子
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        # 比较 JVP (Jacobians with respect to vector products) 的树形结构
        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_x_,), (l_v_,))

        # 增加 JVP 嵌套层级
        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting()
        
        # 启用前向梯度
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True)
        
        # 进入双重层级
        _enter_dual_level = torch._C._enter_dual_level()

        # 可能加载分解
        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions()

        # 创建双重数值
        _make_dual = torch._make_dual(l_x_, l_v_, level=0); l_x_ = l_v_ = None

        # 计算正弦函数
        sin = _make_dual.sin(); _make_dual = None
        
        # 对结果进行求和
        result_duals = sin.sum(); sin = None

        # 解包双重数值
        _unpack_dual = torch._unpack_dual(result_duals, level=0); result_duals = None
        primal = _unpack_dual[0]
        dual = _unpack_dual[1]; _unpack_dual = None

        # 展开原始输出（用于梯度）
        primals_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(primal, 1); primal = None

        # 展开切线输出（用于梯度）
        tangents_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(dual, 1); dual = None

        # 退出双重层级
        _exit_dual_level = torch._C._exit_dual_level(0)
        # 再次启用前向梯度
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True)
        # 减少 JVP 嵌套层级
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting()
        # 启用保存的张量钩子
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        
        # 返回正向传播的结果，包括原始输出和切线输出
        return (primals_out_unflatten, tangents_out_unflatten)
# 在测试中创建一个新的测试用例来检查具有附加输出的 JVP 函数
def test_jvp_has_aux(self):
    # 清空计数器
    counters.clear()

    # 定义一个函数 fn，它对输入 x 执行 sin() 函数并返回结果及其自身
    def fn(x):
        return x.sin().sum(), x

    # 定义一个包装函数 wrapper_fn，它接收输入 x 和 v，并调用 torch.func.jvp 函数来计算 JVP，带有附加输出
    def wrapper_fn(x, v):
        return torch.func.jvp(fn, (x,), (v,), has_aux=True)

    # 生成随机输入数据 x 和 v
    x = torch.randn(3, 3)
    v = torch.randn(3, 3)

    # 编译和检查包装的图形模块
    wrapped_gm = self._compile_check(wrapper_fn, (x, v))

    # 如果检查动态形状捕获，则返回，因为动态形状会产生略有不同的图形
    if check_dynamic_shape_capture():
        return

    # 将包装后的图形模块打印为可读性更高的格式，并进行规范化处理
    actual = normalize_gm(wrapped_gm.print_readable(print_output=False))

    # 断言预期的内联输出
    self.assertExpectedInline(
        actual,
        """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_v_: "f32[3, 3]"):
        l_x_ = L_x_
        l_v_ = L_v_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_x_,), (l_v_,))

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting()
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True)
        _enter_dual_level = torch._C._enter_dual_level()

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions()

        aux = torch._make_dual(l_x_, l_v_, level = 0);  l_x_ = l_v_ = None

        sin = aux.sin()
        result_duals = sin.sum();  sin = None

        aux_1: "f32[3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal = _unpack_dual[0]
        dual = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(primal, 1);  primal = None

        tangents_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(dual, 1);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0)
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True)
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (primals_out_unflatten, tangents_out_unflatten, aux_1)
""",
    )
        # 清空计数器，准备测试
        counters.clear()

        # 定义一个函数 fn，接受两个参数 x 和 y，返回一个元组
        def fn(x, y):
            # 对 x 求正弦并求和，再加上 y 的余弦
            return (x.sin().sum() + y.cos()), x

        # 定义一个包装函数 wrapper_fn，接受三个参数 x、y、v
        def wrapper_fn(x, y, v):
            # 调用 torch.func.jvp 函数计算 fn 在 (x, y) 和 (v, v) 上的 JVP（Jacobian Vector Product）
            return torch.func.jvp(fn, (x, y), (v, v), has_aux=True)

        # 生成三个随机的 3x3 Tensor
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        v = torch.randn(3, 3)

        # 使用 self._compile_check 方法编译并检查 wrapper_fn
        wrapped_gm = self._compile_check(wrapper_fn, (x, y, v))

        # 如果需要检查动态形状捕获，则直接返回，不进行后续断言
        if check_dynamic_shape_capture():
            return

        # 对 wrapped_gm 进行可读性打印并进行规范化
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))

        # 使用 self.assertExpectedInline 进行断言比较
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    # 定义模块的前向传播方法
    def forward(self, L_x_: "f32[3, 3]", L_y_: "f32[3, 3]", L_v_: "f32[3, 3]"):
        # 复制输入张量到本地变量
        l_x_ = L_x_
        l_y_ = L_y_
        l_v_ = L_v_

        # 禁用保存的张量钩子功能，并返回相应的上下文管理器
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        # 使用 Functorch 的函数比较 JVP 树结构
        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_x_, l_y_), (l_v_, l_v_))

        # 增加 JVP 嵌套层级
        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting()
        # 启用前向梯度计算
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True)
        # 进入双重级别
        _enter_dual_level = torch._C._enter_dual_level()

        # 在前向自动求导中，可能加载分解
        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions()

        # 创建 l_x_ 的对偶张量
        aux = torch._make_dual(l_x_, l_v_, level=0); l_x_ = None

        # 再次检查是否需要加载分解
        _maybe_load_decompositions_1 = torch.autograd.forward_ad._maybe_load_decompositions()

        # 创建 l_y_ 和 l_v_ 的对偶张量
        _make_dual_1 = torch._make_dual(l_y_, l_v_, level=0); l_y_ = l_v_ = None

        # 计算 aux 的正弦值
        sin = aux.sin()
        # 对正弦值求和
        sum_1 = sin.sum(); sin = None
        # 计算 _make_dual_1 的余弦值
        cos = _make_dual_1.cos(); _make_dual_1 = None
        # 结果对偶张量为 sum_1 加上 cos
        result_duals = sum_1 + cos; sum_1 = cos = None

        # 解包 aux 的对偶张量，获取原始值和对偶值
        aux_1: "f32[3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1); aux = None

        # 解包结果对偶张量，获取原始值
        _unpack_dual = torch._unpack_dual(result_duals, level=0); result_duals = None
        primal = _unpack_dual[0]
        # 获取对偶值
        dual = _unpack_dual[1]; _unpack_dual = None

        # 解包原始值，获取未扁平化的原始输出
        primals_out_unflatten: "f32[3, 3]" = torch._C._functorch._unwrap_for_grad(primal, 1); primal = None

        # 解包对偶值，获取未扁平化的切线输出
        tangents_out_unflatten: "f32[3, 3]" = torch._C._functorch._unwrap_for_grad(dual, 1); dual = None

        # 退出双重级别
        _exit_dual_level = torch._C._exit_dual_level(0)
        # 再次启用前向梯度计算
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True)
        # 减少 JVP 嵌套层级
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting()
        # 启用保存的张量钩子功能
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()

        # 返回未扁平化的原始输出、切线输出和 aux_1
        return (primals_out_unflatten, tangents_out_unflatten, aux_1)
    # 定义一个方法 `forward`，接受两个参数 L_x_ 和 L_v_
    def forward(self, L_x_: "f32[3, 3]", L_v_: "f32[3, 3]"):
        # 复制输入参数作为本地变量
        l_x_ = L_x_
        l_v_ = L_v_

        # 设置前向梯度计算为关闭状态
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(False)

        # 禁用保存的张量钩子，显示消息提示用户不支持保存张量钩子
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        # 比较 JVP (Jacobians with respect to inputs and vectors) 的树形结构
        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_x_,), (l_v_,))

        # 增加 JVP 的嵌套层级
        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting()

        # 设置前向梯度计算为开启状态
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True)

        # 进入双重级别
        _enter_dual_level = torch._C._enter_dual_level()

        # 可能加载分解
        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions()

        # 创建双重数
        _make_dual = torch._make_dual(l_x_, l_v_, level=0); l_x_ = l_v_ = None

        # 计算正弦函数
        sin = _make_dual.sin(); _make_dual = None

        # 对正弦函数结果求和
        result_duals = sin.sum(); sin = None

        # 解包双重数，得到原始值和对偶值
        _unpack_dual = torch._unpack_dual(result_duals, level=0); result_duals = None
        primal = _unpack_dual[0]
        dual = _unpack_dual[1]; _unpack_dual = None

        # 将原始值展平为一维数组
        primals_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(primal, 1); primal = None

        # 将对偶值展平为一维数组
        tangents_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(dual, 1); dual = None

        # 退出双重级别
        _exit_dual_level = torch._C._exit_dual_level(0)

        # 设置前向梯度计算为关闭状态
        _set_fwd_grad_enabled_2 = torch._C._set_fwd_grad_enabled(False)

        # 减少 JVP 的嵌套层级
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting()

        # 启用保存的张量钩子
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()

        # 再次设置前向梯度计算为开启状态
        _set_fwd_grad_enabled_3 = torch._C._set_fwd_grad_enabled(True)

        # 返回展平后的原始值数组和对偶值数组
        return (primals_out_unflatten, tangents_out_unflatten)
        )

    def test_jvp_two_tensors_disable_enable_disable_grad(self):
        counters.clear()

        def fn(x):
            return x.sin().sum()  # 定义一个函数 fn，计算输入张量 x 的正弦值之和

        def wrapper_fn(x, v):
            with torch.autograd.forward_ad._set_fwd_grad_enabled(False):  # (1) 禁用前向自动求导
                with torch.autograd.forward_ad._set_fwd_grad_enabled(True):  # (2) 启用前向自动求导
                    with torch.autograd.forward_ad._set_fwd_grad_enabled(False):  # (3) 再次禁用前向自动求导
                        return torch.func.jvp(fn, (x,), (v,))  # (4) 对函数 fn 进行雅可比向量积计算

            # Start True
            # False      (1)
            #   True     (2)
            #     False  (3)
            #       True (4)
            #     True   (undo 3)
            #   False    (undo 2)
            # True       (undo 1)

        x = torch.randn(3, 3)  # 生成一个大小为 (3, 3) 的随机张量 x
        v = torch.randn(3, 3)  # 生成一个大小为 (3, 3) 的随机张量 v
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    # 定义一个类方法 `forward`，接受两个参数 L_x_ 和 L_v_
    def forward(self, L_x_: "f32[3, 3]", L_v_: "f32[3, 3]"):
        # 将输入参数赋值给局部变量 l_x_ 和 l_v_
        l_x_ = L_x_
        l_v_ = L_v_

        # 禁用前向梯度计算
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(False)
        # 启用前向梯度计算
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True)
        # 再次禁用前向梯度计算
        _set_fwd_grad_enabled_2 = torch._C._set_fwd_grad_enabled(False)
        # 禁用保存张量钩子，提示不支持保存张量钩子的转换函数
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        # 比较 JVP 树规范
        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_x_,), (l_v_,))

        # 增加 JVP 嵌套级别
        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting()
        # 再次启用前向梯度计算
        _set_fwd_grad_enabled_3 = torch._C._set_fwd_grad_enabled(True)
        # 进入双重级别
        _enter_dual_level = torch._C._enter_dual_level()

        # 可能加载分解
        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions()

        # 创建双重张量
        _make_dual = torch._make_dual(l_x_, l_v_, level=0);  l_x_ = l_v_ = None

        # 计算正弦函数
        sin = _make_dual.sin();  _make_dual = None
        # 求和得到结果双重张量
        result_duals = sin.sum();  sin = None

        # 解包双重张量，获取原始值和对偶值
        _unpack_dual = torch._unpack_dual(result_duals, level=0);  result_duals = None
        primal = _unpack_dual[0]
        dual = _unpack_dual[1];  _unpack_dual = None

        # 展开原始值，用于梯度计算
        primals_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(primal, 1);  primal = None

        # 展开对偶值，用于梯度计算
        tangents_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(dual, 1);  dual = None

        # 退出双重级别
        _exit_dual_level = torch._C._exit_dual_level(0)
        # 禁用前向梯度计算
        _set_fwd_grad_enabled_4 = torch._C._set_fwd_grad_enabled(False)
        # 减少 JVP 嵌套级别
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting()
        # 启用保存张量钩子
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        # 再次启用前向梯度计算
        _set_fwd_grad_enabled_5 = torch._C._set_fwd_grad_enabled(True)
        # 再次禁用前向梯度计算
        _set_fwd_grad_enabled_6 = torch._C._set_fwd_grad_enabled(False)
        # 再次启用前向梯度计算
        _set_fwd_grad_enabled_7 = torch._C._set_fwd_grad_enabled(True)
        # 返回展开的原始值和对偶值
        return (primals_out_unflatten, tangents_out_unflatten)
    def test_jvp_freevar_tensor(self):
        # 清空计数器，用于统计某些事件发生的次数
        counters.clear()
        # 生成一个 3x3 的随机张量 y
        y = torch.randn(3, 3)

        # 定义一个函数 fn，接受参数 x，计算 x 的正弦值与张量 y 的和，并返回总和
        def fn(x):
            return (x.sin() + y).sum()

        # 定义一个包装函数 wrapper_fn，用于计算 fn 函数对于输入 x 的 JVP（Jacobian Vector Product）
        def wrapper_fn(x):
            return torch.func.jvp(fn, (x,), (x,))

        # 生成一个 3x3 的随机张量 x
        x = torch.randn(3, 3)
        # 计算 wrapper_fn 的预期输出
        expected = wrapper_fn(x)
        # 编译 wrapper_fn 函数，使用 eager 模式的 AOT 编译器，并返回计算结果
        actual = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)(x)
        # 断言实际输出与预期输出相等
        self.assertEqual(actual, expected)

    def test_jvp_jvp(self):
        # 清空计数器，用于统计某些事件发生的次数
        counters.clear()

        # 如果检查到动态形状捕获，则跳过该测试
        if check_dynamic_shape_capture():
            self.skipTest("test fails with dynamic shapes")

        # 定义一个函数 fn，接受参数 x，计算 torch.sin 函数在 x 上的 JVP，并返回结果
        def fn(x):
            return torch.func.jvp(torch.sin, (x,), (x,))

        # 定义一个包装函数 wrapper_fn，用于计算 fn 函数对于输入 x 的 JVP
        def wrapper_fn(x):
            return torch.func.jvp(fn, (x,), (x,))

        # 生成一个 3x3x3 的随机张量 x
        x = torch.randn(3, 3, 3)
        # 使用 self._compile_check 方法编译 wrapper_fn 函数并返回结果
        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # 如果检查到动态形状捕获，则直接返回，因为生成的图形可能略有不同
        if check_dynamic_shape_capture():
            return

        # 标准化图形模块的输出，以便比较
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        # 断言实际输出与预期的内联代码字符串相等
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
""",
        )

    def test_jvp_freevar_python_scalar(self):
        # 清空计数器，用于统计某些事件发生的次数
        counters.clear()
        # 设置一个 Python 标量 y 的值为 3
        y = 3

        # 定义一个函数 fn，接受参数 x，计算 x 的正弦值与标量 y 的和，并返回总和
        def fn(x):
            return (x.sin() + y).sum()

        # 定义一个包装函数 wrapper_fn，用于计算 fn 函数对于输入 x 的 JVP
        def wrapper_fn(x):
            return torch.func.jvp(fn, (x,), (x,))

        # 生成一个 3x3x3 的随机张量 x
        x = torch.randn(3, 3, 3)
        # 计算 wrapper_fn 的预期输出
        expected = wrapper_fn(x)
        # 编译 wrapper_fn 函数，使用 eager 模式的 AOT 编译器，并返回计算结果
        actual = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)(x)
        # 断言实际输出与预期输出相等
        self.assertEqual(actual, expected)

    def test_jvp_disable_capture(self):
        # 清空计数器，用于统计某些事件发生的次数
        counters.clear()

        # 使用 config.patch 方法禁用捕获函数变换的配置
        with config.patch(capture_func_transforms=False):
            # 定义一个函数 wrapper_fn，接受参数 x，计算 torch.sin 函数在 x 上的 JVP，并返回结果
            def wrapper_fn(x):
                return torch.func.jvp(torch.sin, (x,), (x,))

            # 生成一个 3x3x3 的随机张量 x
            x = torch.randn(3, 3, 3)
            # 计算 wrapper_fn 的实际输出
            actual = wrapper_fn(x)
            # 编译 wrapper_fn 函数，使用 eager 模式的 AOT 编译器，并返回计算结果
            expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x)
            # 断言图形中断计数器中的条目数量为 1
            self.assertEqual(len(counters["graph_break"]), 1)
            # 断言图形中断计数器中的内容与预期相符
            self.assertEqual(
                dict(counters["graph_break"]),
                {
                    "torch.func.jvp capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 1
                },
            )
        # 断言实际输出与预期输出相等
        self.assertEqual(actual, expected)

    @config.patch(capture_func_transforms=True)
        counters.clear()

清空计数器，假设该函数用于清空某种计数状态，以保证测试的独立性。


        def wrapper_fn(x):

定义一个内部函数 `wrapper_fn`，接受参数 `x`。


            output, jvp_fn = torch.func.linearize(torch.sin, x)

调用 `torch.func.linearize` 函数，对 `torch.sin` 函数在 `x` 处进行线性化处理，返回 `output` 和 `jvp_fn`。


            return output, jvp_fn(x)

返回 `output` 和 `jvp_fn(x)` 的结果。


        x = torch.randn(3, 3, 3)

生成一个形状为 (3, 3, 3) 的随机张量 `x`。


        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False, graph_idx=0)

调用 `self._compile_check` 方法，将 `wrapper_fn` 编译为图模式并检查，传入参数 `(x,)`，关闭完整图模式 (`fullgraph=False`)，指定图索引为 0 (`graph_idx=0`)，将结果赋给 `wrapped_gm`。


        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

如果 `check_dynamic_shape_capture()` 返回真值，说明动态形状会产生略有不同的图形。在这种情况下，函数提前返回，不继续执行后续代码。


        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))

调用 `wrapped_gm.print_readable(print_output=False)` 打印 `wrapped_gm` 的可读格式，然后使用 `normalize_gm` 函数对其进行归一化处理，将结果赋给 `actual`。


        self.assertExpectedInline(
            actual,
            """\

使用 `self.assertExpectedInline` 方法断言 `actual` 的值与预期字符串相等，预期字符串是一个多行字符串的开头。
class GraphModule(torch.nn.Module):
    def forward(self, L_self_buffers_tensor_constant0_: "f32[3, 3, 3]"):
        # 接收参数并赋值给局部变量
        l_self_buffers_tensor_constant0_ = L_self_buffers_tensor_constant0_

        # 调用 torch.ops.aten.alias.default 方法，将结果赋给 alias_default 变量
        alias_default: "f32[3, 3, 3]" = torch.ops.aten.alias.default(l_self_buffers_tensor_constant0_);  l_self_buffers_tensor_constant0_ = None

        # 调用 torch.ops.aten.sin.default 方法，将结果赋给 sin_default 变量
        sin_default: "f32[3, 3, 3]" = torch.ops.aten.sin.default(alias_default)

        # 再次调用 torch.ops.aten.alias.default 方法，将结果赋给 alias_default_1 变量
        alias_default_1: "f32[3, 3, 3]" = torch.ops.aten.alias.default(alias_default)

        # 调用 torch.ops.aten.cos.default 方法，将结果赋给 cos_default 变量，并清空 alias_default_1
        cos_default: "f32[3, 3, 3]" = torch.ops.aten.cos.default(alias_default_1);  alias_default_1 = None

        # 再次调用 torch.ops.aten.alias.default 方法，将结果赋给 alias_default_2 变量
        alias_default_2: "f32[3, 3, 3]" = torch.ops.aten.alias.default(sin_default)
        
        # 返回计算结果的元组
        return (alias_default, cos_default, sin_default)
    # 定义测试函数，验证禁用捕获时的行为
    def test_linearize_disable_capture(self):
        # 清空计数器
        counters.clear()
        # 使用 config.patch 上下文管理器禁用捕获函数变换
        with config.patch(capture_func_transforms=False):
            # 在此处已经验证了该函数可以编译通过
            def wrapper_fn(x):
                # 调用 torch.func.linearize 对 torch.sin 进行线性化
                out, _ = torch.func.linearize(torch.sin, x)
                return out

            # 创建随机数张量 x
            x = torch.randn(2, 3)
            # 调用 wrapper_fn 得到实际输出
            actual = wrapper_fn(x)
            # 使用 torch.compile 编译 wrapper_fn，指定后端为 "aot_eager"，关闭完整图形模式
            expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
                x
            )
            # 断言 graph_break 计数为 1
            self.assertEqual(len(counters["graph_break"]), 1)
            # 断言 graph_break 中的特定消息计数为 1
            self.assertEqual(
                {
                    "torch.func.linearize capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 1,
                },
                dict(counters["graph_break"]),
            )
            # 断言实际输出与期望输出相等
            self.assertEqual(actual, expected)

    # 使用 config.patch 启用捕获函数变换，同时开启重新编译时报错选项
    @config.patch(capture_func_transforms=True)
    @config.patch(error_on_recompile=True)
    def test_vmap_recompile(self):
        # 使用 torch.compile 编译函数 fn，后端为 "eager"
        @torch.compile(backend="eager")
        def fn(x):
            # 对输入张量 x 的每个元素应用 torch.sin 函数，并使用 torch.vmap 向量化
            return torch.vmap(lambda x: x.sin())(x)

        # 创建形状为 (3, 3, 4, 5) 的零张量 x
        x = torch.zeros(3, 3, 4, 5)
        # 对 fn 使用 torch.vmap 向量化处理张量 x
        y = torch.vmap(fn)(x)
        # 第二次调用不应重新编译。参见 Pytorch 问题 #118493
        y = torch.vmap(fn)(x)

    # 使用 config.patch 设置为 xfailIfTorchDynamo，同时开启重新编译时报错选项
    @xfailIfTorchDynamo
    @config.patch(error_on_recompile=True)
    def test_vmap_recompile_different_config(self):
        # 使用 torch.compile 编译函数 fn，后端为 "eager"
        @torch.compile(backend="eager")
        def fn(x):
            # 对输入张量 x 的每个元素应用 torch.sin 函数，并使用 torch.vmap 向量化
            return torch.vmap(lambda x: x.sin())(x)

        # 创建形状为 (3, 3, 4, 5) 的零张量 x
        x = torch.zeros(3, 3, 4, 5)
        # 对 fn 使用 torch.vmap 向量化处理张量 x
        with self.assertRaises(torch._dynamo.exc.RecompileError):
            fn(x)

    # 使用 config.patch 启用捕获函数变换，同时开启重新编译时报错选项
    @config.patch(error_on_recompile=True)
    def test_vmap_recompile_same_config(self):
        # 使用 torch.compile 编译函数 fn，后端为 "eager"
        @torch.compile(backend="eager")
        def fn(x):
            # 对输入张量 x 的每个元素应用 torch.sin 函数，并使用 torch.vmap 向量化
            return torch.vmap(lambda x: x.sin())(x)

        # 创建形状为 (3, 3, 4, 5) 的零张量 x
        x = torch.zeros(3, 3, 4, 5)
        # 使用 torch.vmap 对 torch.vmap(fn, randomness="same") 进行向量化处理
        torch.vmap(torch.vmap(fn, randomness="same"), randomness="same")(x)
        # 使用 assertRaises 断言应引发 torch._dynamo.exc.RecompileError 异常
        with self.assertRaises(torch._dynamo.exc.RecompileError):
            torch.vmap(torch.vmap(fn, randomness="same"), randomness="error")(x)

    # 使用 config.patch 启用捕获函数变换，同时开启重新编译时报错选项
    @config.patch(error_on_recompile=True)
    def test_vmap_recompile_with_randomness(self):
        # 使用 torch.compile 编译函数 fn，后端为 "eager"
        @torch.compile(backend="eager")
        def fn(x):
            # 对输入张量 x 的每个元素应用 torch.sin 函数，并使用 torch.vmap 向量化
            return torch.vmap(lambda x: x.sin())(x)

        # 创建形状为 (3, 3, 4, 5) 的零张量 x
        x = torch.zeros(3, 3, 4, 5)
        # 使用 torch.vmap 对 fn 进行向量化处理，使用 randomness="same"
        torch.vmap(fn, randomness="same")(x)
        # 使用 assertRaises 断言应引发 torch._dynamo.exc.RecompileError 异常
        with self.assertRaises(torch._dynamo.exc.RecompileError):
            torch.vmap(fn, randomness="different")(x)

    # 使用 config.patch 启用捕获函数变换，同时开启重新编译时报错选项
    @config.patch(error_on_recompile=True)
    def test_grad_recompile(self):
        # 使用 torch.compile 编译函数 fn，后端为 "eager"
        @torch.compile(backend="eager")
        def fn(x):
            # 对 torch.sin 函数进行梯度计算
            return torch.func.grad(torch.sin)(x)

        # 创建形状为空的随机张量 x
        x = torch.randn([])
        # 对 fn 使用 torch.func.grad 进行梯度计算
        torch.func.grad(fn)(x)
        # 第二次调用不应重新编译
        torch.func.grad(fn)(x)
    # 清空计数器，准备测试环境
    counters.clear()

    # 定义一个函数 g(x)，对输入 x 执行 sin() 函数
    def g(x):
        return x.sin()

    # 使用 torch.compile 编译函数 fn()，指定后端为 "aot_eager"，并开启完整图形计算
    def fn():
        return torch.vmap(g)

    # 生成一个大小为 3x4 的随机张量 x
    x = torch.randn(3, 4)
    # 计算使用 vmap(g) 对 x 的预期结果
    expected = torch.vmap(g)(x)
    # 调用 fn() 获取函数的包装器
    wrapper = fn()
    # 对 x 应用 wrapper 函数
    got = wrapper(x)
    # 使用断言验证预期结果与计算结果的一致性
    self.assertEqual(expected, got)



    # 定义一个函数 g(x)，根据输入 x 的形状进行条件判断
    def g(x):
        # 如果 x 的维度小于 2，则触发图形中断
        if len(x.shape) < 2:
            torch._dynamo.graph_break()
            # 返回 x 的 sin() 函数值
            return x.sin()
        else:
            # 否则返回 x 的 cos() 函数值
            return x.cos()

    # 使用 torch.compile 编译函数 fn(x)，并应用 vmap(g) 到输入 x
    def fn(x):
        return torch.vmap(g)(x)

    # 清空计数器，准备测试环境
    counters.clear()
    # 生成一个大小为 2x3 的随机张量 x
    x = torch.randn(2, 3)
    # 计算预期的 x.sin() 结果
    expected = x.sin()
    # 对 x 应用 fn(x) 函数
    got = fn(x)
    # 使用断言验证预期结果与计算结果的一致性
    self.assertEqual(expected, got)
    # 使用断言验证 graph_break 计数为 1
    self.assertEqual(len(counters["graph_break"]), 1)

    # 清空计数器，准备测试环境
    counters.clear()
    # 生成一个大小为 2x3x4 的随机张量 y
    y = torch.randn(2, 3, 4)
    # 计算预期的 y.cos() 结果
    expected = y.cos()
    # 对 y 应用 fn(y) 函数
    got = fn(y)
    # 使用断言验证预期结果与计算结果的一致性
    self.assertEqual(expected, got)
    # 使用断言验证 graph_break 计数为 0
    self.assertEqual(len(counters["graph_break"]), 0)



    # 清空计数器，准备测试环境
    counters.clear()

    # 定义一个函数 g(x)，对输入 x 执行 cos()，然后对结果执行 sin()，并输出 "hi"
    def g(x):
        y = x.cos()
        print("hi")
        return y.sin()

    # 定义一个函数 fn(x)，应用 vmap(g) 到输入 x
    def fn(x):
        return torch.vmap(g)(x)

    # 生成一个大小为 3x4 的随机张量 x
    x = torch.randn(3, 4)
    # 使用 torch.compile 编译函数 fn，后端为 "aot_eager"，不使用完整图形计算
    opt = torch.compile(fn, backend="aot_eager", fullgraph=False)
    # 计算 fn(x) 的预期结果
    expected = fn(x)
    # 应用 opt(x) 计算结果
    got = opt(x)
    # 使用断言验证 graph_break 计数为 1
    self.assertEqual(len(counters["graph_break"]), 1)
    # 使用断言验证预期结果与计算结果的一致性
    self.assertEqual(expected, got)



    # 清空计数器，准备测试环境
    counters.clear()

    # 定义两个函数 cos(x) 和 sin(x)，分别执行 x 的 cos() 和 sin() 操作
    def cos(x):
        print("cos")
        return x.cos()

    def sin(x):
        print("sin")
        return x.sin()

    # 定义一个函数 g(x)，对输入 x 执行 cos()，然后对结果执行 sin()
    def g(x):
        y = cos(x)
        return sin(y)

    # 定义一个函数 fn(x)，应用 vmap(g, randomness="same") 到输入 x
    def fn(x):
        return torch.vmap(g, randomness="same")(x)

    # 生成一个大小为 3x4 的随机张量 x
    x = torch.randn(3, 4)
    # 使用 torch.compile 编译函数 fn，后端为 "aot_eager"，不使用完整图形计算
    opt = torch.compile(fn, backend="aot_eager", fullgraph=False)
    # 计算 fn(x) 的预期结果
    expected = fn(x)
    # 应用 opt(x) 计算结果
    got = opt(x)
    # 使用断言验证 graph_break 计数为 1
    self.assertEqual(len(counters["graph_break"]), 1)
    # 使用断言验证预期结果与计算结果的一致性
    self.assertEqual(expected, got)



    # 清空计数器，准备测试环境
    counters.clear()

    # 定义一个函数 sin(x)，对输入 x 执行 sin() 操作
    def sin(x):
        print("sin")
        return x.sin()

    # 定义一个函数 fn(x)，应用 vmap(lambda x: sin(x)) 到输入 x
    def fn(x):
        return torch.vmap(lambda x: sin(x))(x)

    # 生成一个大小为 3x4 的随机张量 x
    x = torch.randn(3, 4)
    # 使用 torch.compile 编译函数 fn，后端为 "aot_eager"，不使用完整图形计算
    opt = torch.compile(fn, backend="aot_eager", fullgraph=False)
    # 计算 fn(x) 的预期结果
    expected = fn(x)
    # 应用 opt(x) 计算结果
    got = opt(x)
    # 使用断言验证 graph_break 计数为 1
    self.assertEqual(len(counters["graph_break"]), 1)
    # 使用断言验证预期结果与计算结果的一致性
    self.assertEqual(expected, got)
    def test_vmap(self):
        # 定义一个函数 fn，使用 torch 的 vmap 函数对输入 x 进行映射处理
        def fn(x):
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1))(x)

        # 生成一个大小为 3x3x3 的随机张量 x
        x = torch.randn(3, 3, 3)
        # 调用 self._compile_check 方法对 fn 进行编译和检查
        wrapped_gm = self._compile_check(fn, (x,))

        # 如果检查动态形状捕获返回 True，则直接返回
        if check_dynamic_shape_capture():
            return

        # 调用 wrapped_gm 的 print_readable 方法，以禁止打印输出的方式返回可读的字符串表示
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        # 使用 self.assertExpectedInline 方法断言 actual 与预期字符串之间的匹配
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    # 定义神经网络模块的类
    def forward(self, L_x_: "f32[3, 3, 3]"):
        # 定义前向传播函数，输入参数 L_x_ 是一个形状为 [3, 3, 3] 的浮点数张量

        l_x_ = L_x_
        # 将输入张量 L_x_ 复制给 l_x_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()
        # 调用 functorch 库的 lazy_load_decompositions 函数并将结果赋给 lazy_load_decompositions 变量

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        # 禁用保存的张量钩子以避免在 torch.func 转换中使用。如果有问题，请提交问题报告。

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(3, 'error')
        # 增加 vmap 的嵌套深度到 3 层，如果出错则抛出错误信息

        _add_batch_dim = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None
        # 在 l_x_ 张量的第 0 维和第 1 维之间添加批处理维度，并释放 l_x_ 变量的引用

        sum_1 = _add_batch_dim.sum(0)
        # 对 _add_batch_dim 张量的第 0 维进行求和并赋给 sum_1

        sum_2 = _add_batch_dim.sum(1);  _add_batch_dim = None
        # 对 _add_batch_dim 张量的第 1 维进行求和并赋给 sum_2，并释放 _add_batch_dim 变量的引用

        batched_outputs = sum_1 + sum_2;  sum_1 = sum_2 = None
        # 计算 sum_1 和 sum_2 张量之和，并释放 sum_1 和 sum_2 变量的引用

        _remove_batch_dim: "f32[3, 3]" = torch._C._functorch._remove_batch_dim(batched_outputs, 1, 3, 0);  batched_outputs = None
        # 从 batched_outputs 张量中移除第 1 维，保留形状为 [3, 3] 的张量，并释放 batched_outputs 变量的引用

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        # 减少 vmap 的嵌套深度

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        # 启用保存的张量钩子

        return (_remove_batch_dim,)
        # 返回包含 _remove_batch_dim 张量的元组作为输出
    def test_vmap_free_tensor(self):
        # 创建一个3x3的随机张量y
        y = torch.randn(3, 3)

        def fn(x):
            # 使用torch的函数vmap对输入的x进行操作，对每个x.sum(0)和x.sum(1)加和，并加上y的结果
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1) + y)(x)

        # 创建一个3x3x3的随机张量x
        x = torch.randn(3, 3, 3)
        # 使用self._compile_check方法编译和检查fn函数，得到编译后的图
        wrapped_gm = self._compile_check(fn, (x,))

        # 如果检测到动态形状捕获，则返回
        if check_dynamic_shape_capture():
            return

        # 对编译后的图进行可读性打印，并标准化输出
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        # 使用self.assertExpectedInline方法断言实际输出与预期输出相符
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    # 定义一个自定义的 PyTorch 模块 GraphModule
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3]"):
        # 定义 forward 方法，接受两个输入参数 L_x_ 和 L_y_

        l_x_ = L_x_
        # 将输入参数 L_x_ 复制给局部变量 l_x_

        l_y_ = L_y_
        # 将输入参数 L_y_ 复制给局部变量 l_y_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()
        # 调用 torch._functorch.vmap.lazy_load_decompositions() 函数并赋值给 lazy_load_decompositions

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        # 调用 torch._C._autograd._saved_tensors_hooks_disable() 函数，传递一条警告信息作为参数，并赋值给 _saved_tensors_hooks_disable

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(3, 'error')
        # 调用 torch._C._functorch._vmap_increment_nesting() 函数，传递两个参数 3 和 'error'，并赋值给 _vmap_increment_nesting

        _add_batch_dim = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None
        # 调用 torch._C._functorch._add_batch_dim() 函数，将 l_x_ 添加批次维度，传递参数为 l_x_, 0, 1，并将 l_x_ 置为 None

        sum_1 = _add_batch_dim.sum(0)
        # 对 _add_batch_dim 在维度 0 上求和，结果赋给 sum_1

        sum_2 = _add_batch_dim.sum(1);  _add_batch_dim = None
        # 对 _add_batch_dim 在维度 1 上求和，结果赋给 sum_2，并将 _add_batch_dim 置为 None

        add = sum_1 + sum_2;  sum_1 = sum_2 = None
        # 计算 sum_1 和 sum_2 的和，结果赋给 add，并将 sum_1 和 sum_2 置为 None

        batched_outputs = add + l_y_;  add = l_y_ = None
        # 将 add 和 l_y_ 相加，结果赋给 batched_outputs，并将 add 和 l_y_ 置为 None

        _remove_batch_dim: "f32[3, 3]" = torch._C._functorch._remove_batch_dim(batched_outputs, 1, 3, 0);  batched_outputs = None
        # 调用 torch._C._functorch._remove_batch_dim() 函数，移除 batched_outputs 的批次维度，传递参数为 batched_outputs, 1, 3, 0，并将 batched_outputs 置为 None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        # 调用 torch._C._functorch._vmap_decrement_nesting() 函数，无参数传递，并赋值给 _vmap_decrement_nesting

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        # 调用 torch._C._autograd._saved_tensors_hooks_enable() 函数，无参数传递，并赋值给 _saved_tensors_hooks_enable

        return (_remove_batch_dim,)
        # 返回一个包含 _remove_batch_dim 的元组
    def test_vmap_two_inputs_tuple_in_dims(self):
        # 定义输入维度为元组 (0, 1)
        in_dims = (0, 1)

        def fn(x, y):
            # 使用 torch 的向量化映射 vmap，对输入 x 和 y 执行操作
            return torch.func.vmap(
                # Lambda 函数：对 x 按第0维和第1维求和，然后加上 y
                lambda x, y: x.sum(0) + x.sum(1) + y, in_dims=in_dims
            )(x, y)

        # 生成随机张量 x 和 y，形状分别为 (3, 3, 3) 和 (3, 3)
        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3)

        # 编译和检查函数 fn，并返回编译后的图形式
        wrapped_gm = self._compile_check(fn, (x, y))

        # 动态形状可能会产生稍有不同的图形
        if check_dynamic_shape_capture():
            return

        # 获取可读性良好的打印输出，然后进行规范化处理
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))

        # 断言实际输出与预期输出的内联形式一致
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    # 定义了一个继承自torch.nn.Module的图模块类

    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3]"):
        # 定义了模型的前向传播方法，接受两个输入参数L_x_和L_y_

        l_x_ = L_x_
        # 将输入参数L_x_赋值给局部变量l_x_

        l_y_ = L_y_
        # 将输入参数L_y_赋值给局部变量l_y_

        # 加载延迟分解模型
        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        # 禁用保存张量钩子
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        # 增加vmap嵌套层数
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(3, 'error')

        # 为l_x_添加批次维度
        _add_batch_dim = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        # 为l_y_添加批次维度
        _add_batch_dim_1 = torch._C._functorch._add_batch_dim(l_y_, 1, 1);  l_y_ = None

        # 计算_sum_1和sum_2的和
        sum_1 = _add_batch_dim.sum(0)
        sum_2 = _add_batch_dim.sum(1);  _add_batch_dim = None

        # 计算add和_add_batch_dim_1的和
        add = sum_1 + sum_2;  sum_1 = sum_2 = None

        # 计算batched_outputs和_add_batch_dim_1的和
        batched_outputs = add + _add_batch_dim_1;  add = _add_batch_dim_1 = None

        # 移除批次维度，得到_remove_batch_dim
        _remove_batch_dim: "f32[3, 3]" = torch._C._functorch._remove_batch_dim(batched_outputs, 1, 3, 0);  batched_outputs = None

        # 减少vmap嵌套层数
        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()

        # 启用保存张量钩子
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()

        # 返回移除批次维度后的结果
        return (_remove_batch_dim,)
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3, 3]"):
        # 将输入参数赋值给局部变量 l_x_ 和 l_y_
        l_x_ = L_x_
        l_y_ = L_y_

        # 获取懒加载分解的函数对象
        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        # 禁用保存张量钩子的功能，用于警告信息
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        # 增加 VMap 的嵌套层级，设置错误信息模式
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(3, 'error')

        # 在 l_x_ 上添加批处理维度，并释放 l_x_ 的引用
        child = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None
        # 在 l_y_ 上添加批处理维度，并释放 l_y_ 的引用
        child_1 = torch._C._functorch._add_batch_dim(l_y_, 0, 1);  l_y_ = None

        # 获取懒加载分解的函数对象（另一个实例）
        lazy_load_decompositions_1 = torch._functorch.vmap.lazy_load_decompositions()

        # 禁用保存张量钩子的功能，用于警告信息（另一个实例）
        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        # 增加 VMap 的嵌套层级，设置错误信息模式（另一个实例）
        _vmap_increment_nesting_1 = torch._C._functorch._vmap_increment_nesting(3, 'error')

        # 在 child 和 child_1 的结果上进行批处理加法，并释放 child 和 child_1 的引用
        _add_batch_dim_2 = torch._C._functorch._add_batch_dim(child, 1, 2);  child = None
        _add_batch_dim_3 = torch._C._functorch._add_batch_dim(child_1, 1, 2);  child_1 = None

        # 在批处理输出上去除指定的批处理维度，并释放 batched_outputs 的引用
        batched_outputs = _add_batch_dim_2 + _add_batch_dim_3;  _add_batch_dim_2 = _add_batch_dim_3 = None

        # 函数结束后减少 VMap 的嵌套层级
        batched_outputs_1 = torch._C._functorch._remove_batch_dim(batched_outputs, 2, 3, 0);  batched_outputs = None

        # 减少 VMap 的嵌套层级（另一个实例）
        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        # 启用保存张量钩子的功能
        _saved_tensors_hooks_disable_2 = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        # 在 batched_outputs_1 上去除指定的批处理维度，并释放 batched_outputs_1 的引用
        _remove_batch_dim_1: "f32[3, 3, 3]" = torch._C._functorch._remove_batch_dim(batched_outputs_1, 1, 3, 0);  batched_outputs_1 = None

        # 减少 VMap 的嵌套层级（另一个实例）
        _vmap_decrement_nesting_1 = torch._C._functorch._vmap_decrement_nesting()
        # 启用保存张量钩子的功能
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        # 返回去除批处理维度后的结果元组
        return (_remove_batch_dim_1,)
    def test_vmap_over_vmap_captured(self):
        x = torch.ones(2, 3)
        y = torch.ones(5, 3)

        def fn(x):
            # 定义了一个嵌套的 torch.vmap，首先对 y 应用 vmap，然后对结果再次应用 vmap
            return torch.func.vmap(torch.func.vmap(lambda y: x * y))(y)

        wrapped_gm = self._compile_check(fn, (x,))

        # 如果检测到动态形状捕获，则返回，因为会产生稍有不同的图形
        if check_dynamic_shape_capture():
            return

        # 规范化输出的图形模块，以便进行比较
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_y_: "f32[5, 3]", L_x_: "f32[2, 3]"):
        l_y_ = L_y_
        l_x_ = L_x_

        # 获取 lazy_load_decompositions 函数对象
        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        # 禁用 saved tensor hooks 的功能，提示不支持 torch.func 转换的用例
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        # 增加 vmap 的嵌套层级
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(5, 'error')

        # 在 l_y_ 上添加批次维度，生成 child，并清空 l_y_
        child = torch._C._functorch._add_batch_dim(l_y_, 0, 1);  l_y_ = None

        # 再次获取 lazy_load_decompositions 函数对象
        lazy_load_decompositions_1 = torch._functorch.vmap.lazy_load_decompositions()

        # 禁用 saved tensor hooks 的功能，提示不支持 torch.func 转换的用例
        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        # 增加 vmap 的嵌套层级
        _vmap_increment_nesting_1 = torch._C._functorch._vmap_increment_nesting(3, 'error')

        # 在 child 上添加批次维度，生成 _add_batch_dim_1，并清空 child
        _add_batch_dim_1 = torch._C._functorch._add_batch_dim(child, 0, 2);  child = None

        # 计算批次化的输出
        batched_outputs = l_x_ * _add_batch_dim_1;  l_x_ = _add_batch_dim_1 = None

        # 在 batched_outputs 上移除批次维度
        batched_outputs_1 = torch._C._functorch._remove_batch_dim(batched_outputs, 2, 3, 0);  batched_outputs = None

        # 减少 vmap 的嵌套层级
        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()

        # 再次禁用 saved tensor hooks 的功能，提示不支持 torch.func 转换的用例
        _saved_tensors_hooks_disable_2 = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        # 移除 _remove_batch_dim_1 上的批次维度，声明 _remove_batch_dim_1 的类型
        _remove_batch_dim_1: "f32[5, 3, 2, 3]" = torch._C._functorch._remove_batch_dim(batched_outputs_1, 1, 5, 0);  batched_outputs_1 = None

        # 减少 vmap 的嵌套层级
        _vmap_decrement_nesting_1 = torch._C._functorch._vmap_decrement_nesting()

        # 启用 saved tensor hooks 的功能
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()

        # 返回结果元组
        return (_remove_batch_dim_1,)
""",
        )

    def test_vmap_multiple_outputs(self):
        x = torch.ones(2, 4, 3)

        def fn(x):
            # 使用 torch.vmap 对输入 x 应用 lambda 函数，分别对第 0 维和第 1 维求和，返回元组
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)))(x)

        wrapped_gm = self._compile_check(fn, (x,))

        # 如果检测到动态形状捕获，则返回，因为会产生稍有不同的图形
        if check_dynamic_shape_capture():
            return

        # 规范化输出的图形模块，以便进行比较
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    # 定义一个方法 forward，接受一个名为 L_x_ 的参数，其类型为 f32[2, 4, 3]
    def forward(self, L_x_: "f32[2, 4, 3]"):
        # 将参数 L_x_ 赋值给局部变量 l_x_
        l_x_ = L_x_

        # 调用 torch._functorch.vmap.lazy_load_decompositions() 函数，获取惰性加载的分解
        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        # 禁用保存的张量钩子，给出错误信息，提示不支持函数转换中的张量钩子
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        # 增加 vmap 嵌套层级到 2，如果出错则抛出错误
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(2, 'error')

        # 在 l_x_ 张量的第 0 维上添加批处理维度
        _add_batch_dim = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        # 沿着新添加的批处理维度对 _add_batch_dim 进行求和，得到 child 张量
        child = _add_batch_dim.sum(0)

        # 沿着 _add_batch_dim 的第 1 维进行求和，得到 child_1 张量；释放 _add_batch_dim
        child_1 = _add_batch_dim.sum(1);  _add_batch_dim = None

        # 在 child 张量中去除批处理维度，将其形状调整为 f32[2, 3]，赋值给 _remove_batch_dim；释放 child
        _remove_batch_dim: "f32[2, 3]" = torch._C._functorch._remove_batch_dim(child, 1, 2, 0);  child = None

        # 在 child_1 张量中去除批处理维度，将其形状调整为 f32[2, 4]，赋值给 _remove_batch_dim_1；释放 child_1
        _remove_batch_dim_1: "f32[2, 4]" = torch._C._functorch._remove_batch_dim(child_1, 1, 2, 0);  child_1 = None

        # 减少当前 vmap 嵌套层级；清理内部状态并返回
        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()

        # 启用保存的张量钩子，返回内部状态
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()

        # 返回去除批处理维度后的两个张量 _remove_batch_dim 和 _remove_batch_dim_1
        return (_remove_batch_dim, _remove_batch_dim_1)
    def test_vmap_multiple_outputs_diff_dims(self):
        x = torch.ones(2, 4, 3)

        def fn(x):
            # 使用 torch.vmap 对输入 x 进行映射，对每个元素应用 lambda 函数
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)), out_dims=(1, 0))(x)

        # 将函数 fn 编译并检查其输出
        wrapped_gm = self._compile_check(fn, (x,))

        # 如果支持动态形状捕获，则直接返回，因为结果图形可能略有不同
        if check_dynamic_shape_capture():
            return

        # 标准化图形模块的输出以便进行比较
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        # 断言实际输出与预期的内联字符串相匹配
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 4, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(2, 'error')

        _add_batch_dim = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        child = _add_batch_dim.sum(0)
        child_1 = _add_batch_dim.sum(1);  _add_batch_dim = None

        _remove_batch_dim: "f32[3, 2]" = torch._C._functorch._remove_batch_dim(child, 1, 2, 1);  child = None
        _remove_batch_dim_1: "f32[2, 4]" = torch._C._functorch._remove_batch_dim(child_1, 1, 2, 0);  child_1 = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (_remove_batch_dim, _remove_batch_dim_1)
""",
        )
    # 定义一个方法 `forward`，接受一个名为 `L_x_` 的参数，类型为 "f32[2, 4, 3]"
    def forward(self, L_x_: "f32[2, 4, 3]"):
        # 将参数赋值给局部变量 l_x_
        l_x_ = L_x_

        # 调用 torch._functorch.vmap.lazy_load_decompositions() 函数，返回 lazy load decompositions 对象
        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        # 禁用保存张量钩子的功能，并返回一个消息提示
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        # 增加 vmap 的嵌套层数为 2，如果失败则抛出错误
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(2, 'error')

        # 在 l_x_ 的第 0 维度前添加一个批次维度
        _add_batch_dim = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        # 对 _add_batch_dim 沿第 0 维度求和，得到 child
        child = _add_batch_dim.sum(0)

        # 对 _add_batch_dim 沿第 1 维度求和，得到 child_1；清空 _add_batch_dim 变量
        child_1 = _add_batch_dim.sum(1);  _add_batch_dim = None

        # 从 child 中移除第 1 维度，保留原始维度 2 和 3
        _remove_batch_dim: "f32[3, 2]" = torch._C._functorch._remove_batch_dim(child, 1, 2, 1);  child = None

        # 从 child_1 中移除第 1 维度，保留原始维度 0 和 2
        _remove_batch_dim_1: "f32[2, 4]" = torch._C._functorch._remove_batch_dim(child_1, 1, 2, 0);  child_1 = None

        # 减少当前 vmap 的嵌套层数
        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()

        # 启用保存张量钩子的功能
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()

        # 返回移除批次维度后的两个结果 _remove_batch_dim 和 _remove_batch_dim_1
        return (_remove_batch_dim, _remove_batch_dim_1)
    def test_vmap_kwargs(self):
        # 清空计数器
        counters.clear()
        # 创建一个全是1的2x3张量x和一个随机值的2x3张量y
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        def fn(x, y):
            # 使用torch的vmap函数，将lambda函数应用于x和y的元素上，并返回结果
            return torch.func.vmap(lambda x, y: x + y)(x, y)

        # 执行vmap函数
        actual = fn(x, y)
        # 编译fn函数，使用"aot_eager"后端，不使用完整图形
        expected = torch.compile(fn, backend="aot_eager", fullgraph=False)(x, y)
        # 断言图形中断计数为0
        self.assertEqual(len(counters["graph_break"]), 0)
        # 断言实际结果等于预期结果
        self.assertEqual(actual, expected)

    def test_vmap_pytree_inputs(self):
        # 清空计数器
        counters.clear()
        # 创建一个全是1的2x3张量x和一个随机值的2x3张量y
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        def vmap_fn(inps):
            # 从输入字典中获取张量x和y
            x = inps["x"]
            y = inps["y"]
            # 返回x和y的元素求和结果
            return x + y

        def fn(x, y):
            # 使用torch的vmap函数，将vmap_fn函数应用于包含x和y的字典，并返回结果
            return torch.func.vmap(vmap_fn)({"x": x, "y": y})

        # 执行vmap函数
        actual = fn(x, y)
        # 编译fn函数，使用"aot_eager"后端，不使用完整图形
        expected = torch.compile(fn, backend="aot_eager", fullgraph=False)(x, y)
        # 断言图形中断计数为0
        self.assertEqual(len(counters["graph_break"]), 0)
        # 断言实际结果等于预期结果
        self.assertEqual(actual, expected)

    def test_vmap_side_effects(self):
        # 清空计数器
        counters.clear()
        # 创建一个全是1的2x3张量x和一个随机值的2x3张量y
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        some_list = []

        def f(x, y):
            # 将1添加到列表some_list中，并返回x和y的元素求和结果
            some_list.append(1)
            return x + y

        def wrapper_fn(x, y):
            # 使用torch的vmap函数，将f函数应用于x和y，并返回结果
            return torch.func.vmap(f)(x, y)

        # 执行wrapper_fn函数
        actual = wrapper_fn(x, y)
        # 编译wrapper_fn函数，使用"aot_eager"后端，不使用完整图形
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x, y)
        # 断言图形中断计数为0
        self.assertEqual(len(counters["graph_break"]), 0)
        # 断言实际结果等于预期结果
        self.assertEqual(actual, expected)
        # 断言some_list中的内容为[1, 1]
        self.assertEqual(some_list, [1, 1])

    @unittest.expectedFailure
    def test_vmap_side_effects_append_input(self):
        # 清空计数器
        counters.clear()
        # 创建一个全是1的2x3张量x和一个随机值的2x3张量y
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        some_list = []

        def f(x, y):
            # 将x添加到列表some_list中，并返回x和y的元素求和结果
            some_list.append(x)
            return x + y

        def wrapper_fn(x, y):
            # 使用torch的vmap函数，将f函数应用于x和y，并返回结果
            return torch.func.vmap(f)(x, y)

        # 执行wrapper_fn函数
        actual = wrapper_fn(x, y)
        # 编译wrapper_fn函数，使用"aot_eager"后端，不使用完整图形
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x, y)
        # 断言图形中断计数为0
        self.assertEqual(len(counters["graph_break"]), 0)
        # 断言实际结果等于预期结果
        self.assertEqual(actual, expected)

    def test_vmap_previous_illegal_op_no_graph_break(self):
        # 清空计数器
        counters.clear()

        # 调用.stride()将会导致先前的图形中断
        def bad_fn(x):
            y = x.view((4, 3))
            y.stride()
            return y

        def wrapper_fn(x):
            # 使用torch的vmap函数，将bad_fn函数应用于x，并返回结果
            return torch.func.vmap(bad_fn)(x)

        x = torch.randn(2, 3, 4)
        # 执行wrapper_fn函数
        actual = wrapper_fn(x)
        # 编译wrapper_fn函数，使用"aot_eager"后端，不使用完整图形
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x)
        # 断言图形中断计数为0
        self.assertEqual(len(counters["graph_break"]), 0)
        # 断言实际结果等于预期结果
        self.assertEqual(actual, expected)
    def test_vmap_disable_capture(self):
        counters.clear()  # 清空计数器对象

        with config.patch(capture_func_transforms=False):
            # 使用 config.patch() 上下文管理器禁用函数捕获变换
            # 这里已验证该函数编译成功
            def wrapper_fn(x):
                return torch.func.vmap(lambda x: x.sum(0) + x.sum(1))(x)

            x = torch.randn(3, 3, 3)
            # 编译 wrapper_fn 函数，使用 "aot_eager" 后端，关闭完整图
            actual = wrapper_fn(x)
            expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
                x
            )
            # 断言 graph_break 计数器中的条目数为 1
            self.assertEqual(len(counters["graph_break"]), 1)
            # 断言 graph_break 计数器中的具体内容
            self.assertEqual(
                dict(counters["graph_break"]),
                {
                    "torch.func.vmap capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 2
                },
            )
            # 断言实际输出与期望输出相等
            self.assertEqual(actual, expected)

    def test_vmap_multiple_invocation_in_dims(self):
        counters.clear()  # 清空计数器对象

        def wrapper_fn(x, in_dims):
            return torch.func.vmap(torch.sum, in_dims)(x)

        x = torch.randn(3, 3, 3, 3)
        cnt = CompileCounter()
        # 使用 torch.compile 编译 wrapper_fn 函数，使用动态模式
        opt = torch.compile(wrapper_fn, backend=cnt, fullgraph=False, dynamic=True)
        expected = wrapper_fn(x, 0), wrapper_fn(x, 1), wrapper_fn(x, 2)
        # 第三次调用 opt 使 in_dims 成为 SymInt
        actual = opt(x, 0), opt(x, 1), opt(x, 2)
        # 断言期望输出与实际输出相等
        self.assertEqual(expected, actual)
        # 断言帧计数与操作计数为 3 和 27
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 27)

    def test_vmap_multiple_invocation_out_dims(self):
        counters.clear()  # 清空计数器对象

        def wrapper_fn(x, out_dims):
            return torch.func.vmap(lambda x: torch.sum(x, 0), out_dims=out_dims)(x)

        x = torch.randn(3, 3, 3, 3)
        cnt = CompileCounter()
        # 使用 torch.compile 编译 wrapper_fn 函数，使用动态模式
        opt = torch.compile(wrapper_fn, backend=cnt, fullgraph=False, dynamic=True)
        expected = wrapper_fn(x, 0), wrapper_fn(x, 1), wrapper_fn(x, 2)
        # 第三次调用 opt 使 in_dims 成为 SymInt
        actual = opt(x, 0), opt(x, 1), opt(x, 2)
        # 断言期望输出与实际输出相等
        self.assertEqual(expected, actual)
        # 断言帧计数与操作计数为 3 和 27
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 27)

    def test_vmap_new_tensor_in_body(self):
        def fn(x):
            return x + torch.ones(3)

        def wrapper_fn(x):
            return torch.func.vmap(fn)(x)

        x = torch.randn(
            3,
        )
        # 使用 torch.compile 编译 wrapper_fn 函数，使用 "aot_eager" 后端，完整图模式
        opt = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)
        expected = wrapper_fn(x)
        actual = opt(x)
        # 断言期望输出与实际输出相等
        self.assertEqual(expected, actual)

    def test_vmap_new_tensor_unused_in_body(self):
        def fn(x):
            return torch.tensor(0.5)

        def wrapper_fn(x):
            return torch.func.vmap(fn)(x)

        x = torch.randn(3)
        # 使用 torch.compile 编译 wrapper_fn 函数，使用 "aot_eager" 后端，完整图模式
        opt = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)
        expected = wrapper_fn(x)
        actual = opt(x)
        # 断言期望输出与实际输出相等
        self.assertEqual(expected, actual)
    # 定义一个测试方法，用于测试通过操作隐式创建新张量的向量映射
    def test_vmap_new_tensor_implicit_via_op(self):
        # 定义一个包装函数，将输入张量中的每个元素加上0.5并返回结果
        def wrapper_fn(x):
            return torch.func.vmap(lambda t: torch.add(t, 0.5))(x)
    
        # 创建一个包含三个随机数的张量
        x = torch.randn(3)
        # 编译包装函数，使用AOT eager模式和完整图形模式
        opt = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)
        # 预期的结果，等于直接调用包装函数得到的结果
        expected = wrapper_fn(x)
        # 实际调用优化后的包装函数得到的结果
        actual = opt(x)
        # 使用断言检查预期结果和实际结果是否相等
        self.assertEqual(expected, actual)
# 定义测试类 `ActivationCheckpointingTests`，继承自 `torch._dynamo.test_case.TestCase`
class ActivationCheckpointingTests(torch._dynamo.test_case.TestCase):

    # 定义私有方法 `_validate`，用于验证函数在给定参数下的正确性
    def _validate(self, fn, backend, *args, skip_check=False, fullgraph=True):
        # 克隆参数，确保不影响原始数据并设置梯度追踪
        cloned_args = []
        for arg in args:
            cloned_args.append(arg.clone().detach().requires_grad_(arg.requires_grad))

        # 设置随机种子
        torch.manual_seed(0)
        # 计算预期输出
        expected = fn(*args)
        expected.sum().backward()

        # 编译优化函数
        opt_fn = torch.compile(fn, fullgraph=fullgraph, backend=backend)
        # 重新设置随机种子
        torch.manual_seed(0)
        # 执行优化函数
        result = opt_fn(*cloned_args)
        result.sum().backward()

        # 如果未跳过检查，则进行结果检查
        if not skip_check:
            # 断言优化函数结果与预期结果相等
            self.assertEqual(result, expected)
            # 检查每个参数的梯度是否正确
            for arg, cloned_arg in zip(args, cloned_args):
                self.assertEqual(arg.grad, cloned_arg.grad)

    # 标记为需要 CUDA 的测试方法
    @requires_cuda
    # 使用 `torch._functorch.config.patch` 装饰器配置功能
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    # 定义测试函数 `test_function`
    def test_function(self):
        # 定义计算函数 `gn`，计算输入张量 x 和 y 的 sigmoid 矩阵乘积
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        # 定义测试函数 `fn`，使用检查点技术来计算函数 `gn` 的结果
        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        # 创建需要梯度追踪的随机输入张量 x 和 y
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        # 配置前向和后向编译器
        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # 使用 `aot_autograd` 函数生成后端
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # 调用私有方法 `_validate` 验证函数 `fn` 的正确性
        self._validate(fn, backend, x, y)

    # 标记为需要 CUDA 的测试方法
    @requires_cuda
    # 使用 `torch._functorch.config.patch` 装饰器配置功能
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    # 定义带关键字参数的测试函数 `test_function_with_kwargs`
    def test_function_with_kwargs(self):
        # 定义计算函数 `gn`，计算输入张量 x 和 y 的 sigmoid 矩阵乘积
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        # 定义测试函数 `fn`，使用检查点技术来计算函数 `gn` 的结果，并指定关键字参数
        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                torch.sin(x),
                y,
                use_reentrant=True,
                preserve_rng_state=False,
            )

        # 创建需要梯度追踪的随机输入张量 x 和 y
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        # 配置前向和后向编译器
        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # 使用 `aot_autograd` 函数生成后端
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # 调用私有方法 `_validate` 验证函数 `fn` 的正确性
        self._validate(fn, backend, x, y)

    # 标记为需要 CUDA 的测试方法
    @requires_cuda
    # 使用 `torch._functorch.config.patch` 装饰器配置功能
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_dropout(self):
        def gn(x, y):
            return torch.nn.functional.dropout(torch.matmul(x, y), p=0.2)

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        # 定义前向编译器，设置频率和使用的操作
        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.rngprims.philox_rand.default
        )
        # 定义后向编译器，设置频率和使用的操作
        bw_compiler = functools.partial(
            count_ops, freq=0, op=torch.ops.rngprims.philox_rand.default
        )
        # 使用 AOT 自动微分的后端
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(
            fn, backend, x, y, skip_check=True
        )  # dropout decomp is known to diverge with eager

    @requires_cuda
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_dropout_inductor(self):
        def gn(x, y):
            return torch.nn.functional.dropout(torch.matmul(x, y), p=0.2)

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        # 使用 "inductor" 后端
        backend = "inductor"
        self._validate(
            fn, backend, x, y, skip_check=True
        )  # dropout decomp is known to diverge with eager

    @requires_cuda
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_fallback(self):
        def gn(x, y):
            torch._dynamo.graph_break()
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.cos(
                torch.utils.checkpoint.checkpoint(
                    gn, torch.sin(x), y, use_reentrant=True
                ),
            )

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        args = (x, y)

        # 使用 EagerAndRecordGraphs 后端
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        # 预期的函数计算结果
        expected = fn(*args)
        # 使用后端计算函数
        result = torch.compile(fn, backend=cnt)(*args)

        # 断言结果与预期相等
        self.assertEqual(result, expected)

        # torch.sin 和 torch.cos 分别生成一个图
        self.assertEqual(cnt.frame_count, 2)
        # 总操作计数为 2
        self.assertEqual(cnt.op_count, 2)
        # 后端记录了 2 个图
        self.assertEqual(len(backend.graphs), 2)
    # 定义一个测试方法，用于测试模块的功能
    def test_module(self):
        # 定义一个模拟的神经网络模块类
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在模块中添加一个线性层
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                # 定义模块的前向传播逻辑，使用sigmoid函数对线性层的输出进行处理
                return torch.sigmoid(self.linear(x))

        # 创建一个MockModule的实例
        mod = MockModule()

        # 定义一个函数fn，使用checkpoint机制对模块mod进行前向传播
        def fn(x):
            return torch.utils.checkpoint.checkpoint(
                mod, torch.sin(x), use_reentrant=True
            )

        # 创建一个形状为(10, 10)的随机张量x，并要求计算其梯度
        x = torch.randn(10, 10, requires_grad=True)

        # 配置前向传播计算器，计算sigmoid函数的操作次数
        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        # 配置反向传播计算器，将sigmoid函数的操作次数设为0
        bw_compiler = functools.partial(
            count_ops, freq=0, op=torch.ops.aten.sigmoid.default
        )
        # 使用aot_autograd函数配置自动微分及即时编译后端
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # 对模块fn进行验证，传入测试数据x和配置的后端
        self._validate(fn, backend, x)

    # 测试覆盖默认的fallthrough调度键
    def test_override_fallthrough_dispatch_key(self):
        # 创建一个测试操作对象test_op，使用高阶操作器"_fallthrough_test_only"
        test_op = torch._ops.HigherOrderOperator("_fallthrough_test_only")
        # 获取默认的fallthrough调度键列表
        default_keys = torch._ops._HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS
        # 断言测试操作对象的non_fallthrough_keys是否不包含任何默认调度键
        self.assertTrue(
            not any(test_op.non_fallthrough_keys.has(key) for key in default_keys)
        )

        # 创建一个包含lambda函数的列表foos，每个函数返回其索引
        foos = [lambda x=i: x for i, k in enumerate(default_keys)]
        # 遍历foos列表和default_keys列表，为test_op注册对应的Python实现
        for foo, fallthrough_key in zip(foos, default_keys):
            test_op.py_impl(fallthrough_key)(foo)

        # 断言测试操作对象的non_fallthrough_keys是否包含所有默认调度键
        self.assertTrue(
            all(test_op.non_fallthrough_keys.has(key) for key in default_keys)
        )
        # 断言test_op的Python核心函数返回值与默认调度键列表的索引是否一致
        self.assertEqual(
            list(range(len(default_keys))),
            [test_op.py_kernels[key]() for key in default_keys],
        )

    # 测试带有关键字参数的条件运算
    def test_cond_with_kwargs(self):
        # 导入条件运算的操作模块cond_op
        from torch._higher_order_ops.cond import cond_op

        # 定义一个测试函数test，根据输入的pred条件选择true_fn或false_fn进行处理
        def test(pred, x):
            # 定义真值情况下的处理函数true_fn，直接返回输入x
            def true_fn(x):
                return x

            # 定义假值情况下的处理函数false_fn，返回-x
            def false_fn(x):
                return -x

            # 调用cond_op进行条件运算，根据pred选择true_fn或false_fn处理x
            return cond_op(pred=pred, true_fn=true_fn, false_fn=false_fn, operands=[x])

        # 创建一个编译计数器实例cnt
        cnt = CompileCounter()
        # 使用torch.compile编译test函数，选择cnt作为后端
        opt_test = torch.compile(test, backend=cnt)
        # 创建一个形状为(3, 3)的全1张量inp
        inp = torch.ones(3, 3)
        # 断言使用test函数和opt_test函数对输入True和inp的输出是否近似
        self.assertTrue(torch.allclose(test(True, inp), opt_test(True, inp)))
        # 断言编译计数器的帧数为1
        self.assertEqual(cnt.frame_count, 1)
        # 断言使用test函数和opt_test函数对输入False和inp的输出是否近似
        self.assertTrue(torch.allclose(test(False, inp), opt_test(False, inp)))
        # 断言编译计数器的帧数为2
        self.assertEqual(cnt.frame_count, 2)
    # 定义一个测试方法，用于测试在无效参数设置下的条件操作函数
    def test_cond_with_invalid_kwargs(self):
        # 导入 torch._higher_order_ops.cond 模块中的 cond_op 函数
        from torch._higher_order_ops.cond import cond_op

        # 定义一个测试函数 test，接受条件 pred、模式 mode 和输入 x
        def test(pred, mode, x):
            # 定义一个返回输入的函数 true_fn
            def true_fn(x):
                return x

            # 定义一个返回输入取反的函数 false_fn
            def false_fn(x):
                return -x

            # 根据模式 mode 判断条件，选择相应的条件操作
            if mode:
                # 调用 cond_op 函数，传入预测条件 pred、真函数 true_fn、假函数 false_fn，
                # 操作数列表 [x]，并设置 invalid 参数为 True
                return cond_op(
                    pred=pred,
                    true_fn=true_fn,
                    false_fn=false_fn,
                    operands=[x],
                    invalid=True,
                )
            else:
                # 调用 cond_op 函数，传入预测条件 pred、真函数 true_fn、假函数 false_fn，
                # 操作数列表 [x]，无需设置 invalid 参数
                return cond_op(
                    pred,
                    pred=pred,
                    true_fn=true_fn,
                    false_fn=false_fn,
                    operands=[x],
                )

        # 创建一个编译计数器对象
        cnt = CompileCounter()
        # 使用 torch.compile 方法编译 test 函数，指定后端为 cnt
        opt_test = torch.compile(test, backend=cnt)
        # 创建一个全为 1 的输入张量 inp
        inp = torch.ones(3, 3)
        
        # 断言调用 opt_test(True, True, inp) 时会抛出 UncapturedHigherOrderOpError 异常
        with self.assertRaises(torch._dynamo.exc.UncapturedHigherOrderOpError):
            opt_test(True, True, inp)

        # 断言调用 opt_test(True, False, inp) 时会抛出 AssertionError 异常
        with self.assertRaises(AssertionError):
            opt_test(True, False, inp)

    # 定义一个测试方法，用于测试不同输入张量之间的别名检查
    def test_non_aliasing_util(self):
        # 导入 torch._dynamo.variables.higher_order_ops 模块中的 _assert_tensors_nonaliasing 函数
        from torch._dynamo.variables.higher_order_ops import _assert_tensors_nonaliasing

        # 创建包含两个元素的列表 a，第一个元素为 tensor，第二个元素为字典
        a = [torch.tensor(1), {"a": torch.tensor(1)}]
        # 创建包含一个元素的元组 b，元素为 tensor
        b = (torch.tensor(1),)
        
        # 调用 _assert_tensors_nonaliasing 函数，断言 a 和 b 中的张量不会相互别名
        _assert_tensors_nonaliasing(a, b)

        # 断言调用 _assert_tensors_nonaliasing(a, a) 时会抛出 AssertionError 异常，
        # 并提示"inputs to function body cannot alias outputs"
        with self.assertRaisesRegex(
            AssertionError, "inputs to function body cannot alias outputs"
        ):
            _assert_tensors_nonaliasing(a, a)
# 如果这个脚本作为主程序被执行
if __name__ == "__main__":
    # 导入 torch._dynamo.test_case 模块中的 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 运行测试函数，通常用于执行测试套件或单元测试
    run_tests()
```