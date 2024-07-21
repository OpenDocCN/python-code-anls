# `.\pytorch\test\dynamo\test_subclasses.py`

```py
# Owner(s): ["module: dynamo"]
# 导入必要的模块和函数
import functools  # 导入 functools 模块
import itertools  # 导入 itertools 模块
import unittest  # 导入 unittest 模块

from functools import partial  # 导入 functools 模块中的 partial 函数

import torch  # 导入 PyTorch 模块

# 导入 PyTorch 内部测试和配置模块
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._functorch.config

# 导入 PyTorch 工具模块
import torch.utils._pytree as pytree
import torch.utils.checkpoint

# 从 PyTorch 内部测试模块中导入 normalize_gm 函数
from torch._dynamo.testing import normalize_gm

# 从 PyTorch 高阶操作模块中导入 wrap 函数
from torch._higher_order_ops.wrap import wrap

# 从 PyTorch FX 实验性符号形状模块中导入相关类
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)

# 从 PyTorch 嵌套张量内部模块中导入相关函数
from torch.nested._internal.nested_tensor import (
    jagged_from_list,
    jagged_from_tensor_and_lengths,
    nested_view_from_values_offsets,
)

# 从 PyTorch 内部测试常用工具模块中导入相关函数
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)

# 从 PyTorch 内部感应器工具模块中导入 HAS_CUDA 常量
from torch.testing._internal.inductor_utils import HAS_CUDA

# 从 PyTorch 内部两张张量模块中导入 TwoTensor 类
from torch.testing._internal.two_tensor import TwoTensor


def traceable_subclass(c):
    # 返回通过 torch._dynamo.config.patch 函数进行的特定配置
    return torch._dynamo.config.patch("traceable_tensor_subclasses", {c})


def get_jagged_tensor(nested_size, offsets, requires_grad=True):
    # 创建一个不规则张量，由指定大小的子张量组成
    # nested_size: 嵌套大小，offsets: 偏移量，requires_grad: 是否需要梯度
    D = nested_size[1]
    out = []
    for s in nested_size[0]:
        # 创建具有指定形状、数据类型和梯度需求的随机张量
        out.append(torch.randn(s, D, requires_grad=requires_grad, dtype=torch.float64))
    # 使用 out 中的张量创建不规则张量
    return jagged_from_list(out, offsets)


def get_view_test_cases():
    # 返回包含视图测试案例的函数列表

    # 创建基础情况的测试案例函数
    def mk_basic(base_is_nt):
        # 获取一个不规则张量并进行克隆操作（如果基础是不规则张量）
        x, _ = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=True)
        x = x.clone() if base_is_nt else x
        # 断言 x 不是叶子节点
        assert not x.is_leaf
        # 返回张量的维度扩展后的结果
        return x.unsqueeze(-1)

    # 创建叶子情况的测试案例函数
    def mk_leaf(base_is_nt, requires_grad_1, requires_grad_2):
        # 获取一个不规则张量并根据需求梯度创建视图
        x, _ = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=requires_grad_1)
        x = x.clone() if base_is_nt else x
        # 使用 torch.no_grad() 上下文管理器创建视图并设置需求梯度
        with torch.no_grad():
            x_view = x.unsqueeze(-1)
            x_view.requires_grad_(requires_grad_2)
        # 返回创建的张量视图
        return x_view

    # 创建复杂情况的测试案例函数
    def mk_obscure(base_is_nt):
        # 获取一个不规则张量并创建多层次视图
        x, _ = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=False)
        x = x.clone() if base_is_nt else x
        # 使用 torch.no_grad() 上下文管理器创建中间叶子视图
        with torch.no_grad():
            x_view = x.unsqueeze(-1)
        # 设置叶子视图的需求梯度为 True
        x_view.requires_grad_(True)
        # 对视图的视图再次进行维度扩展
        x_view_view = x_view.unsqueeze(-1)
        # 返回创建的多层次视图
        return x_view_view
    # 对于每个 base_is_nt 值（False 和 True），执行以下操作：
    for base_is_nt in [False, True]:
        # 创建前缀字符串，格式为 "base_is_nt_<base_is_nt>"
        prefix = f"base_is_nt_{base_is_nt}"

        # 返回部分函数 mk_basic 的部分应用及其命名
        yield partial(mk_basic, base_is_nt), f"{prefix}_basic"

        # (2) leaf view case:
        # 视图必须是叶子（requires_grad 为 True 或 False）
        # base 也可以是 requires_grad 为 True 或 False
        # 遍历所有可能的 requires_grad 组合
        for requires_grad_1, requires_grad_2 in itertools.product(
            [True, False], repeat=2
        ):
            # 返回部分函数 mk_leaf 的部分应用及其命名，包括 base_is_nt 和 requires_grad 组合
            yield partial(
                mk_leaf, base_is_nt, requires_grad_1, requires_grad_2
            ), f"{prefix}_leaf_{requires_grad_1}_{requires_grad_2}"

        # (3) obscure case:
        # 视图不是叶子（意味着 requires_grad 为 True）
        # base 是 requires_grad 为 False
        # 返回部分函数 mk_obscure 的部分应用及其命名
        yield partial(mk_obscure, base_is_nt), f"{prefix}_obscure"

    # Subclass -> Dense
    # 返回一个函数，这个函数获取具有指定形状和 requires_grad=True 的张量的视图并进行克隆
    yield lambda: get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=True)[
        0
    ].clone(), "subclass_dense"

    # Dense -> Subclass -> Dense -> Subclass
    # 定义一个函数，创建并返回嵌套视图的值和偏移量
    def mk_dense_subclass_dense_subclass():
        values = torch.randn(10, 5)
        offsets = torch.tensor([0, 3, 6, 10])
        offsets2 = offsets.clone().detach()
        return nested_view_from_values_offsets(
            nested_view_from_values_offsets(values, offsets).values(), offsets
        )

    # 返回函数 mk_dense_subclass_dense_subclass 及其命名
    yield mk_dense_subclass_dense_subclass, "dense_subclass_dense_subclass"

    # 定义一个函数，获取具有指定形状和 requires_grad=True 的张量的视图并进行克隆，然后创建并返回嵌套视图的值和偏移量
    def mk_subclass_dense_subclass_dense():
        x = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=True)[0].clone()
        offsets2 = x.offsets().clone().detach()
        nt_view = nested_view_from_values_offsets(x.values(), offsets2).values()

    # 返回函数 mk_subclass_dense_subclass_dense 及其命名
    yield mk_subclass_dense_subclass_dense, "subclass_dense_subclass_dense"
# 创建 VIEW_TEST_CASES 字典，使用 get_view_test_cases 函数返回的键值对，将值作为键，键作为值
VIEW_TEST_CASES = {k: v for v, k in get_view_test_cases()}

# 如果没有 CUDA 支持，则跳过对应的单元测试，提示需要 CUDA
requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")

# 使用 torch.compile 函数编译后端为 "eager"，启用完整图形模式
compile_full_eager = torch.compile(backend="eager", fullgraph=True)


class BaseTorchFunction(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 如果 kwargs 为 None，则初始化为空字典
        if kwargs is None:
            kwargs = {}
        # 调用父类的 __torch_function__ 方法
        return super().__torch_function__(func, types, args, kwargs)


class MockSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 如果 kwargs 为 None，则初始化为空字典
        if kwargs is None:
            kwargs = {}
        # 直接调用传入的 func 函数，传入 args 和 kwargs
        return func(*args, **kwargs)


class AttrSubclass(torch.Tensor):
    x: int = 10
    size: int = 10

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 如果 kwargs 为 None，则初始化为空字典
        if kwargs is None:
            kwargs = {}
        # 调用传入的 func 函数，传入 args 和 kwargs
        return func(*args, **kwargs)


class DummyNDim(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 如果 kwargs 为 None，则初始化为空字典
        if kwargs is None:
            kwargs = {}

        # 如果 func 是 torch.Tensor.ndim 方法
        if func == torch.Tensor.ndim.__get__:
            # 返回一个固定的维度数 10
            return 10

        # 否则调用父类的 __torch_function__ 方法
        return super().__torch_function__(func, types, args, kwargs)


class WrapperSubclass:
    def __init__(self, tensor):
        self.tensor = tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 如果 kwargs 为 None，则初始化为空字典
        if kwargs is None:
            kwargs = {}

        # 将 args 和 kwargs 中的所有元素映射为 WrapperSubclass 对象的 tensor 属性
        args = pytree.tree_map_only(WrapperSubclass, lambda x: x.tensor, args)
        kwargs = pytree.tree_map_only(WrapperSubclass, lambda x: x.tensor, kwargs)

        # 调用 func 函数，传入处理后的 args 和 kwargs
        return func(*args, **kwargs)


class SigmoidToExpSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 如果 kwargs 为 None，则初始化为空字典
        if kwargs is None:
            kwargs = {}

        # 如果 func 是 torch.Tensor.sigmoid 方法
        if func == torch.Tensor.sigmoid:
            # 调用 torch.Tensor.exp 方法，并传入 types, args, kwargs
            return super().__torch_function__(torch.Tensor.exp, types, args, kwargs)

        # 否则调用父类的 __torch_function__ 方法
        return super().__torch_function__(func, types, args, kwargs)


# Wrapper subclass with two inner tensors: data and scale
# data has same shape as outer, and scale has single dim size
class ScaledTensor(torch.Tensor):
    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        *,
        constant: int = 0,
    ):
        # 调用 torch.Tensor._make_wrapper_subclass 方法创建一个新的包装子类
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=data.dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )

    def __init__(self, data: torch.Tensor, scale: torch.Tensor, constant: int = 0):
        self._data = data
        self._scale = scale
        self._constant = constant

    def __tensor_flatten__(self):
        # 返回 _data 和 _scale 作为扁平化的内容，同时返回包含常量 _constant 的上下文
        ctx = {"_constant": self._constant}
        return ["_data", "_scale"], ctx

    @staticmethod
    # 定义一个私有方法 __tensor_unflatten__，用于将内部张量解压缩为 ScaledTensor 对象
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        # 断言内部张量列表长度为2，确保参数正确
        assert len(inner_tensors) == 2
        # 返回一个新的 ScaledTensor 对象，使用给定的内部数据、尺度和元数据中的常数值
        return ScaledTensor(
            inner_tensors["_data"],
            inner_tensors["_scale"],
            constant=metadata["_constant"],
        )

    # 定义一个类方法 __torch_dispatch__，用于处理 Torch 函数的分发调用
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # 获取调用参数中的 scaled_tensor（一个 ScaledTensor 对象）
        scaled_tensor = args[0]
        # 调用给定的 Torch 函数 func，传入 scaled_tensor 的数据部分和其他参数
        out = func(scaled_tensor._data, *args[1:], **kwargs)
        # 返回一个新的 ScaledTensor 对象，使用调用结果 out、scaled_tensor 的尺度和常数值
        return ScaledTensor(out, scaled_tensor._scale, constant=scaled_tensor._constant)

    # 定义对象的 __repr__ 方法，返回对象的字符串表示形式，包括其数据和尺度信息
    def __repr__(self):
        return f"{self._data.__repr__()}\n{self._scale.__repr__()}"
# 定义一个继承自 torch.Tensor 的可选缩放张量子类
class OptionalScaledTensor(torch.Tensor):
    # __new__ 方法，用于创建新的张量实例
    def __new__(
        cls,
        data,  # 数据部分的张量
        scale,  # 缩放因子
        *,
        constant: int = 0,  # 常量值，默认为0
    ):
        # 调用父类的 _make_wrapper_subclass 方法创建实例
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),  # 张量的尺寸
            strides=data.stride(),  # 张量的步长
            storage_offset=data.storage_offset(),  # 张量的存储偏移量
            dtype=data.dtype,  # 张量的数据类型
            layout=data.layout,  # 张量的布局
            requires_grad=data.requires_grad,  # 张量是否需要梯度
            device=data.device,  # 张量所在设备
        )

    # __init__ 方法，初始化可选缩放张量实例
    def __init__(self, data: torch.Tensor, scale, constant: int = 0):
        self._data = data  # 存储原始数据的张量
        self._scale = scale  # 缩放因子
        self._constant = constant  # 常量值

    # __tensor_flatten__ 方法，返回用于展平张量的元数据
    def __tensor_flatten__(self):
        ctx = {"_constant": self._constant}  # 上下文中包含常量值
        if self._scale is not None:
            return ["_data", "_scale"], ctx  # 如果有缩放因子，返回数据和缩放因子的键名
        else:
            return ["_data"], ctx  # 否则只返回数据的键名

    # 静态方法 __tensor_unflatten__，用于从展平后的数据恢复张量实例
    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return OptionalScaledTensor(
            inner_tensors["_data"],  # 获取数据部分的张量
            inner_tensors["_scale"] if "_scale" in inner_tensors else None,  # 获取缩放因子
            constant=metadata["_constant"],  # 获取常量值
        )

    # 类方法 __torch_dispatch__，用于调度 Torch 函数的执行
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        scaled_tensor = args[0]  # 获取输入的可选缩放张量实例
        out = func(scaled_tensor._data, *args[1:], **kwargs)  # 执行指定函数
        if scaled_tensor._scale is not None:
            out = out * scaled_tensor._scale  # 若存在缩放因子，对输出进行缩放
        return OptionalScaledTensor(
            out, scaled_tensor._scale, constant=scaled_tensor._constant  # 返回新的可选缩放张量实例
        )

    # __repr__ 方法，返回可选缩放张量的字符串表示
    def __repr__(self):
        return (
            f"OptionalScaledTensor({self._data.__repr__()}\n{self._scale.__repr__()})"
        )


# 函数 func，对输入张量执行正弦函数
def func(a):
    return a.sin()


# 类 EagerRecordGraphAndInputs，用于记录图和输入数据
class EagerRecordGraphAndInputs:
    def __init__(self):
        self.graphs = []  # 存储图对象的列表
        self.example_inputs = []  # 存储示例输入数据的列表

    # __call__ 方法，用于将图对象和输入数据存储并返回图对象
    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.graphs.append(gm)  # 将图对象添加到列表中
        self.example_inputs.append(example_inputs)  # 将输入数据添加到列表中
        return gm  # 返回图对象本身


# 全局变量 GLOBAL_TEST_SUBCLASSES，用于存储测试子类的集合
GLOBAL_TEST_SUBCLASSES = {
    MockSubclass,
    DummyNDim,
    SigmoidToExpSubclass,
    BaseTorchFunction,
}


# 函数 _recompiles_for_inputs，用于检查函数在不同输入下是否重新编译
# 返回 True 如果在指定的动态设置下，函数在输入 inputs1 和 inputs2 之间重新编译
def _recompiles_for_inputs(fn, inputs1, inputs2, dynamic=True):
    compile_count = [0]  # 编译计数器初始化为0

    # 内部函数 counter，用于统计编译次数并返回图对象
    def counter(gm, example_inputs):
        compile_count[0] += 1  # 每次调用计数器加1
        return gm  # 返回图对象

    # 使用 torch.compile 编译函数 fn，设置回调函数为 counter
    compiled_f = torch.compile(fn, fullgraph=True, backend=counter, dynamic=dynamic)
    compiled_f(*inputs1)  # 编译并执行输入 inputs1
    compiled_f(*inputs2)  # 编译并执行输入 inputs2
    return compile_count[0] > 1  # 返回是否重新编译超过1次的布尔值


# 类 SubclassTests，用于测试子类的测试案例
class SubclassTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            torch._dynamo.config.patch(
                "traceable_tensor_subclasses", GLOBAL_TEST_SUBCLASSES
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._exit_stack.close()
    def test_no_call_to_new(self):
        class BadNewTorchFunction(torch.Tensor):
            def __new__(cls, *args, **kwargs):
                # 抛出运行时错误，禁止直接调用 __new__ 方法
                raise RuntimeError("Oops!")

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # 如果 kwargs 为 None，则设置为空字典
                if kwargs is None:
                    kwargs = {}
                # 调用父类的 __torch_function__ 方法
                return super().__torch_function__(func, types, args, kwargs)

        # 使用 torch._dynamo.config.patch 方法来配置 traceable_tensor_subclasses
        with torch._dynamo.config.patch(
            "traceable_tensor_subclasses", {BadNewTorchFunction}
        ):

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                # 对输入的张量执行加法操作
                return torch.add(x, 1)

            # 创建一个由 BadNewTorchFunction 子类化的张量作为输入
            input = torch.ones(2, 2).as_subclass(BadNewTorchFunction)

            # 调用函数 fn 处理输入张量
            res = fn(input)
            # 确保返回结果是 BadNewTorchFunction 的实例
            self.assertIsInstance(res, BadNewTorchFunction)

    def test_no_torch_function_recompiles(self):
        class NJT:
            def __repr__(self):
                # 返回该对象的字符串表示，包括 shape 属性
                return f"NJT(shape={self.shape})"

            def __init__(self, values, offsets):
                # 初始化 NJT 对象，使用给定的 values 和 offsets
                self._values = values
                self._offsets = offsets

            def sin(self):
                # 调用 torch.sin 函数处理对象自身的 _values 属性
                return torch.sin(self)

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # 如果 kwargs 为 None，则设置为空字典
                if kwargs is None:
                    kwargs = {}
                # 如果调用的是 torch.sin 函数，则处理参数并返回处理后的结果
                if func == torch.sin:
                    self = args[0]
                    return NJT(func(self._values), self._offsets)
                # 如果调用了未预期的函数，则引发断言错误
                raise AssertionError("should not get here")

        # 创建两个张量 values1 和 values2，及一个偏移张量 offsets
        values1 = torch.randn(10, 3, 4, requires_grad=True)
        values2 = torch.randn(10, 3, 4, requires_grad=True)
        offsets = torch.tensor([0, 3, 10])
        # 创建两个 NJT 对象，分别使用 values1 和 values2 作为值，offsets 作为偏移
        njt1 = NJT(values1, offsets)
        njt2 = NJT(values2, offsets)

        # 定义函数 f，对输入的对象调用 torch.sin 函数
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            return torch.sin(x)

        # 在配置 error_on_recompile 为 True 的环境中，分别使用 njt1 和 njt2 调用函数 f
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            f(njt1)
            f(njt2)

    def test_base_torch_function_tracing(self):
        def fn(x):
            # 对输入张量执行加法操作
            return torch.add(x, 1)

        # 创建一个由 BaseTorchFunction 子类化的张量作为输入
        input = torch.ones(2, 2).as_subclass(BaseTorchFunction)
        # 使用 fn 函数处理输入张量
        out = fn(input)
        # 使用 compile_full_eager 函数优化并处理输入张量
        out_opt = compile_full_eager(fn)(input)
        # 确保返回结果是 BaseTorchFunction 的实例，并且两种处理方式得到的结果相等
        self.assertIsInstance(out, BaseTorchFunction)
        self.assertEqual(out, out_opt)

    def test_torch_function_state_graph_break(self):
        # 定义函数 fn，对输入张量执行 torch.add 操作，并在其中断开 Torch 函数图
        @torch.compile(backend="eager")
        def fn(x):
            with torch._C.DisableTorchFunctionSubclass():
                torch._dynamo.graph_break()
                # 返回 Torch 函数是否启用的状态，以及对输入张量执行 torch.add 后的结果
                return torch._C._is_torch_function_enabled(), torch.add(x, 1.0)

        # 创建一个形状为 (2, 2) 的张量 input
        input = torch.ones(2, 2)
        # 调用函数 fn 处理 input，得到返回的结果 res
        res, _ = fn(input)
        # 确保 Torch 函数未启用
        self.assertFalse(res)
    def test_torch_function_state_nested(self):
        # 定义一个使用 Torch 编译器的测试函数，使用 eager 后端
        @torch.compile(backend="eager")
        def fn(x):
            # 进入上下文管理器，禁用 Torch 函数子类化
            with torch._C.DisableTorchFunctionSubclass():
                # 再次进入上下文管理器，禁用 Torch 函数子类化
                with torch._C.DisableTorchFunctionSubclass():
                    # 对输入张量 x 执行加法操作
                    x = x + 1
                # 退出上下文管理器后应恢复到外部的状态（禁用）
                return torch._C._is_torch_function_enabled(), torch.add(x, 1.0)

        # 创建一个全为1的2x2张量作为输入
        input = torch.ones(2, 2)
        # 调用函数 fn，并获取返回结果的第一个元素
        res, _ = fn(input)
        # 断言 Torch 函数是否被禁用
        self.assertFalse(res)

    def test_torch_function_state_tracing(self):
        # 定义一个使用 Torch 编译器的测试函数，使用 eager 后端，并启用完整图形模式
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # 进入上下文管理器，禁用 Torch 函数子类化
            with torch._C.DisableTorchFunctionSubclass():
                # 对输入张量 x 执行加法操作
                torch.add(x, 1.0)

        # 创建一个全为1的2x2张量作为输入
        input = torch.ones(2, 2)
        # 调用函数 fn
        res = fn(input)

    def test_torch_function_state_guards(self):
        # 创建一个编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()

        # 定义一个使用 Torch 编译器的测试函数，使用编译计数器作为后端，并启用完整图形模式
        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # 对输入张量 x 执行加法操作
            torch.add(x, 1.0)

        # 创建一个全为1的2x2张量作为输入
        input = torch.ones(2, 2)

        # 进入上下文管理器，禁用 Torch 函数子类化
        with torch._C.DisableTorchFunctionSubclass():
            # 调用函数 fn
            res = fn(input)

        # 再次调用函数 fn
        res = fn(input)

        # 断言编译帧计数是否为2
        self.assertEqual(cnt.frame_count, 2)

    def test_return_subclass(self):
        # 定义一个使用 Torch 编译器的测试函数，使用 eager 后端，并启用完整图形模式
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # 返回一个 MockSubclass 对象，包裹了对输入张量 x 执行加法操作的结果
            return MockSubclass(torch.add(x, 1.0))

        # 创建一个全为1的2x2张量作为输入
        input = torch.ones(2, 2)

        # 调用函数 fn
        res = fn(input)
        # 断言返回结果是否为 MockSubclass 的实例
        self.assertIsInstance(res, MockSubclass)

    def test_return_as_subclass(self):
        # 定义一个使用 Torch 编译器的测试函数，使用 eager 后端，并启用完整图形模式
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # 返回一个对输入张量 x 执行加法操作的结果，并转换为 MockSubclass 的子类
            return torch.add(x, 1.0).as_subclass(MockSubclass)

        # 创建一个全为1的2x2张量作为输入
        input = torch.ones(2, 2)

        # 调用函数 fn
        res = fn(input)
        # 断言返回结果是否为 MockSubclass 的实例
        self.assertIsInstance(res, MockSubclass)

    def test_return_local_subclass(self):
        # 定义一个本地的 Torch 子类 LocalSubclass，继承自 torch.Tensor
        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                # 调用原始函数 func，并返回结果
                return func(*args, **kwargs)

        # 使用 Torch 配置管理器，将 LocalSubclass 添加到可追踪的张量子类集合中
        with torch._dynamo.config.patch("traceable_tensor_subclasses", {LocalSubclass}):

            # 定义一个使用 Torch 编译器的测试函数，使用 eager 后端，并启用完整图形模式
            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                # 返回一个 LocalSubclass 对象，包裹了对输入张量 x 执行加法操作的结果
                return LocalSubclass(torch.add(x, 1.0))

            # 创建一个全为1的2x2张量作为输入
            input = torch.ones(2, 2)

            # 调用函数 fn
            res = fn(input)
            # 断言返回结果是否为 LocalSubclass 的实例
            self.assertIsInstance(res, LocalSubclass)
    # 定义一个空字典，用于存储处理过的函数
    HANDLED_FUNCTIONS = {}

    # 定义一个示例类 MyClass
    class MyClass:
        def __init__(self, foo):
            self.foo = foo

        # 类方法 __torch_function__，用于处理 torch 函数调用
        @classmethod
        def __torch_function__(
            cls,
            func,
            types,
            args=(),
            kwargs=None,
        ):
            # 如果 kwargs 为 None，则设为一个空字典
            if kwargs is None:
                kwargs = {}
            # 如果 func 不在 HANDLED_FUNCTIONS 中，或者 types 中的所有类型不是 torch.Tensor 或 MyClass 的子类，则返回 NotImplemented
            if func not in HANDLED_FUNCTIONS or not all(
                [
                    issubclass(t, (torch.Tensor, MyClass)) for t in types
                ]
            ):
                return NotImplemented
            # 调用 HANDLED_FUNCTIONS 中存储的处理函数来处理参数 args 和 kwargs，并返回结果
            return HANDLED_FUNCTIONS[func](*args, **kwargs)

    # 定义一个 _stack 函数，用于将 input 中 MyClass 实例的 foo 属性相加，并返回 MyClass 实例
    def _stack(input, dim=0, *, out=None):
        return MyClass(sum([x.foo for x in input]))

    # 将 _stack 函数注册到 HANDLED_FUNCTIONS 中以处理 torch.stack 函数
    HANDLED_FUNCTIONS[torch.stack] = _stack

    # 定义一个使用 torch.compile 装饰器的函数 fn，返回调用 torch.stack 函数的结果
    @torch.compile(backend="eager", fullgraph=True)
    def fn(v0, v1):
        return torch.stack([v0, v1])

    # 调用 fn 函数，传入 MyClass 实例，并验证返回值的 foo 属性是否为 2
    ret = fn(MyClass(1), MyClass(1))
    self.assertEqual(ret.foo, 2)

@parametrize(
    "comparison",
    [
        # 使用 subtest 创建一个测试实例，用于测试 isinstance 函数
        subtest(isinstance, "isinstance"),
        # 使用 subtest 创建一个测试实例，用 lambda 表达式测试 type(instance) == type_
        subtest(lambda instance, type_: type(instance) == type_, "equality"),
        # 使用 subtest 创建一个测试实例，用 lambda 表达式测试 type(instance) is type_
        subtest(lambda instance, type_: type(instance) is type_, "identity"),
    ],
)
@parametrize(
    "input_type",
    [
        # 使用 subtest 创建一个测试实例，测试 torch.Tensor 类型
        subtest(torch.Tensor, "tensor"),
        # 使用 subtest 创建一个测试实例，测试 DummyNDim 的子类
        subtest(DummyNDim, "subclass"),
    ],
)
def test_type_check(self, comparison, input_type):
    # 使用 torch._dynamo.config.patch 方法，将 traceable_tensor_subclasses 配置为 {DummyNDim}
    with torch._dynamo.config.patch("traceable_tensor_subclasses", {DummyNDim}):

        # 定义一个函数 fn，根据 comparison 测试结果返回 torch.ones 或 torch.zeros
        def fn(x):
            if comparison(x, DummyNDim):
                return torch.ones(1, 1)
            else:
                return torch.zeros(2, 2)

        # 创建一个输入张量 input，并使用 as_subclass 方法将其转换为 input_type 类型
        input = torch.ones(2, 2).as_subclass(input_type)
        # 调用 fn 函数，并使用 torch.compile 装饰器创建的函数对其进行编译和执行
        exp_res = fn(input)
        act_res = torch.compile(backend="eager", fullgraph=True)(fn)(input)
        # 验证预期结果 exp_res 和实际结果 act_res 是否相等
        self.assertEqual(exp_res, act_res)

def test_torch_function_call_on_method(self):
    # 创建三个张量 x, y, z，并使用 as_subclass 方法将 x, y 转换为 SigmoidToExpSubclass 类型
    x = torch.ones(2, 2)
    y = torch.ones(2, 2)
    z = torch.ones(2, 2)
    wrapped = x.as_subclass(SigmoidToExpSubclass)
    wrapped2 = y.as_subclass(SigmoidToExpSubclass)

    # 定义一个函数 fn，调用输入张量 w 的 sigmoid 方法
    def fn(w):
        return w.sigmoid()

    # 使用 compile_full_eager 函数将 fn 函数编译为优化版本 fn_opt
    fn_opt = compile_full_eager(fn)

    # 分别对 wrapped, wrapped2 执行 fn_opt 函数，并比较其结果是否与 z.exp() 的结果相等
    res_exp = fn(wrapped)
    res_act = fn_opt(wrapped2)
    res_exp2 = z.exp()

    # 验证 res_exp 和 res_act，res_exp 和 res_exp2 是否相等
    self.assertEqual(res_exp, res_act)
    self.assertEqual(res_exp, res_exp2)
    # 定义一个测试方法，用于测试不支持的用户重写方法
    def test_user_overidden_method_unsupported(self):
        # 创建一个本地子类，继承自 torch.Tensor
        class LocalSubclass(torch.Tensor):
            # 定义一个类方法 __torch_function__
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # 如果 kwargs 为 None，则设为一个空字典
                if kwargs is None:
                    kwargs = {}
                # 调用父类的 __torch_function__ 方法
                return super().__torch_function__(func, types, args, kwargs)

            # 定义一个不支持的方法 sigmoid
            def sigmoid(self):
                return None

        # 定义一个装饰器，编译函数 fn，使用 "eager" 后端和全图模式
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # 调用 x 的 sigmoid 方法
            x.sigmoid()

        # 错误信息字符串
        msg = (
            "Accessing overridden method/attribute sigmoid on a tensor"
            " subclass with a __torch_function__ override is not supported"
        )
        # 在特定配置下，验证是否抛出指定异常类型和消息
        with torch._dynamo.config.patch(
            "traceable_tensor_subclasses", {LocalSubclass}
        ), self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            # 创建一个 tensor，类型为 LocalSubclass，然后调用 fn 函数
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)

    # 定义一个测试方法，用于测试不支持的用户重写属性
    def test_user_overidden_attr_unsupported(self):
        # 创建一个本地子类，继承自 torch.Tensor
        class LocalSubclass(torch.Tensor):
            # 定义一个类方法 __torch_function__
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # 如果 kwargs 为 None，则设为一个空字典
                if kwargs is None:
                    kwargs = {}
                # 调用父类的 __torch_function__ 方法
                return super().__torch_function__(func, types, args, kwargs)

            # 定义一个不支持的属性 ndim
            ndim = 10

        # 定义一个装饰器，编译函数 fn，使用 "eager" 后端和全图模式
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # 返回 x 的 ndim 属性
            return x.ndim

        # 错误信息字符串
        msg = (
            "Accessing overridden method/attribute ndim on a tensor"
            " subclass with a __torch_function__ override is not supported"
        )
        # 在特定配置下，验证是否抛出指定异常类型和消息
        with torch._dynamo.config.patch(
            "traceable_tensor_subclasses", {LocalSubclass}
        ), self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            # 创建一个 tensor，类型为 LocalSubclass，然后调用 fn 函数
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)

    # 定义一个测试方法，用于测试不支持的用户重写属性（包括属性的 getter 和 setter）
    def test_user_overidden_property_unsupported(self):
        # 创建一个本地子类，继承自 torch.Tensor
        class LocalSubclass(torch.Tensor):
            # 类的初始化方法，设置 _ndim 初始值为 10
            def __init__(self):
                self._ndim = 10

            # 定义一个类方法 __torch_function__
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # 如果 kwargs 为 None，则设为一个空字典
                if kwargs is None:
                    kwargs = {}
                # 调用父类的 __torch_function__ 方法
                return super().__torch_function__(func, types, args, kwargs)

            # 定义一个属性 ndim 的 getter 方法
            @property
            def ndim(self):
                return self._ndim

            # 定义一个属性 ndim 的 setter 方法
            @ndim.setter
            def ndim(self, value):
                self._ndim = value

        # 定义一个装饰器，编译函数 fn，使用 "eager" 后端和全图模式
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # 返回 x 的 ndim 属性
            return x.ndim

        # 错误信息字符串
        msg = (
            "Accessing overridden method/attribute ndim on a tensor"
            " subclass with a __torch_function__ override is not supported"
        )
        # 在特定配置下，验证是否抛出指定异常类型和消息
        with torch._dynamo.config.patch(
            "traceable_tensor_subclasses", {LocalSubclass}
        ), self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            # 创建一个 tensor，类型为 LocalSubclass，然后调用 fn 函数
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)
    def test_overridden_method_guarding(self):
        # 定义本地子类，继承自 torch.Tensor
        class LocalSubclass(torch.Tensor):
            # 实现 __torch_function__ 方法作为类方法
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # 如果 kwargs 为 None，则设置为空字典
                if kwargs is None:
                    kwargs = {}
                # 调用父类的 __torch_function__ 方法
                return super().__torch_function__(func, types, args, kwargs)

        # 使用 torch.compile 注解定义函数 fn，使用 eager 后端编译
        @torch.compile(backend="eager")
        def fn(x):
            return x.sigmoid()

        # 在 torch._dynamo.config.patch 上下文中设置参数
        with torch._dynamo.config.patch(
            error_on_recompile=True, traceable_tensor_subclasses={LocalSubclass}
        ):
            # 创建 LocalSubclass 类型的张量 x
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            # 调用 fn 函数
            fn(x)
            # 再次调用 fn 函数
            fn(x)
            # 创建另一个 LocalSubclass 类型的张量 x
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            # 调用 fn 函数
            fn(x)

        # 在 torch._dynamo.config.patch 上下文中设置参数，同时检查是否抛出预期的 TypeError 异常
        with torch._dynamo.config.patch(
            traceable_tensor_subclasses={LocalSubclass}
        ), self.assertRaisesRegex(
            TypeError,
            "'bool' object is not callable",
        ):
            # 设置 LocalSubclass 的 sigmoid 属性为 False
            LocalSubclass.sigmoid = False
            # 调用 fn 函数
            fn(x)

    def test_torch_function_call_on_attr(self):
        # 创建形状为 (2, 2) 的张量 x
        x = torch.ones(2, 2)
        # 将 x 转换为 DummyNDim 类型的张量 wrapped
        wrapped = x.as_subclass(DummyNDim)

        # 定义函数 fn，接受一个参数 w
        def fn(w):
            return w.ndim + torch.ones(2)

        # 使用 compile_full_eager 编译函数 fn，得到 fn_opt
        fn_opt = compile_full_eager(fn)

        # 计算预期结果 res_exp 和实际结果 res_act
        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped)

        # 断言 res_exp 等于 res_act
        self.assertEqual(res_exp, res_act)
        # 断言 res_exp 等于 torch.ones(2) + 10
        self.assertEqual(res_exp, torch.ones(2) + 10)

    def test_torch_function_wrapper_class(self):
        # 创建形状为 (2, 2) 的张量 x
        x = torch.ones(2, 2)
        # 使用 WrapperSubclass 封装 x，得到 wrapped
        wrapped = WrapperSubclass(x)

        # 定义函数 fn，接受一个参数 w
        def fn(w):
            return torch.add(w, 1.0)

        # 使用 compile_full_eager 编译函数 fn，得到 fn_opt
        fn_opt = compile_full_eager(fn)

        # 计算预期结果 res_exp 和实际结果 res_act
        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped)

        # 断言 res_exp 等于 res_act
        self.assertEqual(res_exp, res_act)

    def test_torch_function_wrapper_class_with_kwargs(self):
        # 创建形状为 (2, 2) 的张量 x
        x = torch.ones(2, 2)
        # 使用 WrapperSubclass 封装 x，得到 wrapped
        wrapped = WrapperSubclass(x)

        # 定义函数 fn，接受一个参数 w
        def fn(w):
            return torch.add(w, 1.0, alpha=2.0)

        # 使用 compile_full_eager 编译函数 fn，得到 fn_opt
        fn_opt = compile_full_eager(fn)

        # 计算预期结果 res_exp 和实际结果 res_act
        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped)

        # 断言 res_exp 等于 res_act
        self.assertEqual(res_exp, res_act)

    def test_tensor_subclass_custom_attr(self):
        # 定义 AttrSubclass，继承自 torch.Tensor，带有自定义属性 x
        class AttrSubclass(torch.Tensor):
            x: int = 10

            # 实现 __torch_function__ 方法作为类方法
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # 如果 kwargs 为 None，则设置为空字典
                if kwargs is None:
                    kwargs = {}

                # 调用父类的 __torch_function__ 方法
                return super().__torch_function__(func, types, args, kwargs)

        # 使用 torch.compile 注解定义函数 fn，使用 eager 后端和 fullgraph=True
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x.x + torch.ones(2, 2)

        # 在 traceable_subclass 上下文中设置 AttrSubclass 为可追踪的子类
        with traceable_subclass(AttrSubclass):
            # 创建 AttrSubclass 类型的张量 input
            input = torch.ones(2, 2).as_subclass(AttrSubclass)
            # 使用 compile_full_eager 编译函数 fn，得到 fn_opt
            fn_opt = compile_full_eager(fn)

            # 计算预期结果 res_exp 和实际结果 res_act
            res_exp = fn(input)
            res_act = fn_opt(input)

            # 断言 res_exp 等于 res_act
            self.assertEqual(res_exp, res_act)
    # 定义一个测试函数，用于测试使用带有动态维度的虚拟张量编译
    def test_compile_with_fake_tensor_dynamic_dim(self):
        # 创建一个形状为 [3, 4] 的随机张量 x
        x = torch.randn([3, 4])

        # 定义一个简单的函数 f，对输入张量取正弦函数
        def f(x):
            return torch.sin(x)

        # 定义一个测试动态维度情况下编译后的函数测试方法
        def test_dynamic_dim(f, x, dim_dynamic, exp_frame_count, exp_op_count):
            # 重置 torch._dynamo 对象，用于模拟编译环境
            torch._dynamo.reset()
            # 创建一个编译计数器
            cnt = torch._dynamo.testing.CompileCounter()

            # 编译函数 f，并使用编译计数器作为后端，完整图形模式编译
            opt_f = torch.compile(f, backend=cnt, fullgraph=True)

            # 创建一个形状与 x 相同的随机张量 x1
            x1 = torch.rand_like(x)
            # 调用 f 函数两次，分别使用 x 和形状为 [4, 3] 的随机张量作为输入
            f(x)
            f(torch.randn([4, 3]))

            # 创建一个形状环境对象
            shape_env = ShapeEnv()
            # 使用 torch._subclasses.fake_tensor.FakeTensorMode 上下文管理器模拟虚拟张量模式
            with torch._subclasses.fake_tensor.FakeTensorMode(
                shape_env=shape_env
            ) as fake_mode:
                # 将真实张量 x 转换为虚拟张量，使用 StatelessSymbolicContext 指定动态尺寸
                x_fake = fake_mode.from_tensor(
                    x,
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[dim_dynamic for i in range(x.dim())]
                    ),
                )
                # 将真实张量 x1 转换为虚拟张量，使用相同的动态尺寸设置
                x1_fake = fake_mode.from_tensor(
                    x1,
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[dim_dynamic for i in range(x.dim())]
                    ),
                )
                # 调用经过优化的函数 opt_f 处理虚拟张量 x_fake 和 x1_fake
                opt_f(x_fake)
                opt_f(x1_fake)

            # 断言编译计数器中的帧数与操作数符合预期值
            self.assertEqual(cnt.frame_count, exp_frame_count)
            self.assertEqual(cnt.op_count, exp_op_count)

        # 使用不同的动态维度模式分别测试动态维度情况下编译后的函数
        test_dynamic_dim(f, x, DimDynamic.DYNAMIC, 1, 1)
        test_dynamic_dim(f, x, DimDynamic.DUCK, 1, 1)
        test_dynamic_dim(f, x, DimDynamic.STATIC, 1, 1)
    def test_compile_with_fake_tensor_automatic_dynamic(self):
        # 定义函数 f，用于计算输入张量 x 的正弦值
        def f(x):
            return torch.sin(x)

        # 定义测试函数 test_automatic_dynamic，用于测试动态编译的情况
        def test_automatic_dynamic(f, inps, dim_dynamic, exp_frame_count, exp_op_count):
            # 重置 Dynamo 状态
            torch._dynamo.reset()
            # 创建编译计数器
            cnt = torch._dynamo.testing.CompileCounter()
            # 使用 torch.compile 编译函数 f，指定后端为 cnt，并开启完整图形模式
            opt_f = torch.compile(f, backend=cnt, fullgraph=True)

            # 创建形状环境
            shape_env = ShapeEnv()
            # 进入假张量模式
            with torch._subclasses.fake_tensor.FakeTensorMode(
                shape_env=shape_env
            ) as fake_mode:
                # 对每个输入进行测试
                for inp in inps:
                    # 将真实张量 inp 转换为假张量 fake_inp
                    fake_inp = fake_mode.from_tensor(
                        inp,
                        symbolic_context=StatelessSymbolicContext(
                            [dim_dynamic for i in range(x.dim())]
                        ),
                    )
                    # 调用编译后的函数 opt_f
                    opt_f(fake_inp)
            # 断言编译帧数和操作数符合预期
            self.assertEqual(cnt.frame_count, exp_frame_count)
            self.assertEqual(cnt.op_count, exp_op_count)

        # 创建测试用例的输入张量
        x = torch.randn([3, 4])
        y = torch.randn([4, 5])
        z = torch.randn([5, 6])
        a = torch.randn([3, 5])
        b = torch.randn([4, 4])

        # 对于 DimDynamic.DYNAMIC 或 DimDynamic.DUCK 的情况进行测试
        for dim_dynamic in [DimDynamic.DYNAMIC, DimDynamic.DUCK]:
            test_automatic_dynamic(f, [x, y, z], dim_dynamic, 1, 1)
            test_automatic_dynamic(f, [x, a, z], dim_dynamic, 1, 1)
            test_automatic_dynamic(f, [x, b, z], dim_dynamic, 1, 1)

        # 对于 DimDynamic.STATIC 的情况进行测试
        for dim_dynamic in [DimDynamic.STATIC]:
            # 第一次编译，dim 0 和 1 变为 Dynamic
            test_automatic_dynamic(f, [x, y, z], dim_dynamic, 2, 2)
            # 第一次编译，dim 1 变为 Dynamic；第二次编译，dim 0 变为 Dynamic
            test_automatic_dynamic(f, [x, a, z], dim_dynamic, 3, 3)
            # 第一次编译，dim 0 变为 Dynamic；第二次编译，dim 1 变为 Dynamic
            test_automatic_dynamic(f, [x, b, z], dim_dynamic, 3, 3)

    def test_compile_with_functionalization(self):
        # 创建随机张量 x 和其克隆版本
        x = torch.randn([3, 4])
        x_clone = x.clone()
        x_clone2 = x.clone()
        # 创建记录图形和输入的后端
        backend = EagerRecordGraphAndInputs()
        # 创建编译计数器，关联后端
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        # 使用 torch.compile 为函数 f 编译，指定后端为 cnt，并开启完整图形模式
        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return x.add_(1.0) + torch.nn.functional.relu_(x)

        # 调用编译后的函数 f，得到输出 f_out
        f_out = f(x)
        # 断言编译帧数、操作数和图形数量符合预期
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)
        self.assertEqual(len(backend.graphs), 1)

        # 获取规范化后的图形表示
        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))
        # 断言规范化后的图形表示与预期一致
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    # 定义神经网络模块，继承自torch.nn.Module

    def forward(self, L_x_: "f32[3, 4]"):
        # 定义前向传播函数，接收名为L_x_的输入张量，其类型为f32[3, 4]

        l_x_ = L_x_
        # 将输入张量赋值给局部变量l_x_

        add_: "f32[3, 4]" = l_x_.add_(1.0)
        # 对l_x_执行原地加法操作，每个元素加1.0，并将结果赋值给add_

        relu_: "f32[3, 4]" = torch.relu_(l_x_);  l_x_ = None
        # 对l_x_执行原地ReLU操作，将非正元素置零，并将结果赋值给relu_，然后释放l_x_

        add: "f32[3, 4]" = add_ + relu_;  add_ = relu_ = None
        # 将add_与relu_对应元素相加，并将结果赋值给add_，然后释放add_和relu_

        return (add,)
        # 返回包含add的元组作为输出
# 测试功能化张量的函数
def test_compile_higher_order_with_functionalization(self):
    backend = EagerRecordGraphAndInputs()
    cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

    # 定义一个编译函数 f，使用指定的后端和完整图形模式
    @torch.compile(backend=cnt, fullgraph=True)
    def f(x):
        # 返回一个函数包装器，该函数对输入张量应用 add_(1.0) 操作
        return wrap(lambda x: x.add_(1.0), x)

    # 辅助函数，用于验证编译计数和图形
    def check_count_and_graph(exp_frame_count, exp_op_count, exp_n_graph, exp_graph):
        self.assertEqual(cnt.frame_count, exp_frame_count)
        self.assertEqual(cnt.op_count, exp_op_count)
        self.assertEqual(len(backend.graphs), exp_n_graph)
        # 获取并规范化最后一个图形的可读输出，并验证其与期望的图形是否匹配
        actual = normalize_gm(
            backend.graphs[exp_n_graph - 1].print_readable(print_output=False)
        )
        self.assertExpectedInline(actual, exp_graph, skip=1)

    # 创建一个随机张量 t
    t = torch.randn([3, 4])
    t_clone = t.clone()
    t_clone2 = t.clone()
    # 调用函数 f，并传入张量 t
    f(t)

    # 验证编译计数和图形的期望值
    check_count_and_graph(
        1,
        2,
        1,
        """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3, 4]" = wrap[0];  wrap = None
        return (getitem,)
    class GraphModule(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 4]"):
            add_: "f32[3, 4]" = l_x_.add_(1.0);  l_x_ = None
            return (add_,)
""",
    )

    # 使用 functionalize 函数将函数 f 转换为功能化函数 ff
    ff = torch.func.functionalize(f)
    # 对 t_clone 调用功能化函数 ff
    ff_out = ff(t_clone)
    # 由于重新编译，增加了帧计数和操作计数
    check_count_and_graph(
        2,
        4,
        2,
        """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem = wrap[0];  wrap = None
        return (getitem,)
    class GraphModule(torch.nn.Module):
        def forward(self, l_x_):
            add_ = l_x_.add_(1.0);  l_x_ = None
            return (add_,)
""",
    )
# 三个反引号开始注释代码块
""",
)

try:
    # 调用 torch._to_functional_tensor 函数，将 t_clone2 转换为 functional tensor x
    x = torch._to_functional_tensor(t_clone2)
    # 将 t_clone2 的 autograd 元数据镜像到 x 上
    torch._mirror_autograd_meta_to(t_clone2, x)
    # 启用 functionalization，禁止重新应用视图
    torch._enable_functionalization(reapply_views=False)
    # 对 x 应用函数 f，并将结果赋给 aot_f_out
    aot_f_out = f(x)
finally:
    # 禁用 functionalization
    torch._disable_functionalization()

# 由于重新编译，增加了帧数和操作数
# 检查帧数和操作图
check_count_and_graph(
    3,
    6,
    3,
    """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem = wrap[0];  wrap = None
        return (getitem,)

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_):
            add_ = l_x_.add_(1.0);  l_x_ = None
            return (add_,)
""",
)

def test_has_torch_function():
    class MyTensor:
        @classmethod
        # 实现 __torch_function__ 方法，用于 Torch 函数的自定义行为
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}

            if func is torch.max:
                # 如果调用的是 torch.max 函数，则返回固定的 torch.tensor(123)
                return torch.tensor(123)
            # 否则调用原始的 func 函数
            return func(*args, **kwargs)

    class LocalSubclass(torch.Tensor):
        @classmethod
        # 实现 __torch_function__ 方法，用于 Torch 函数的自定义行为
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}
            # 调用原始的 func 函数
            return func(*args, **kwargs)

    def fn(x):
        # 调用 torch.overrides.has_torch_function_unary 和 torch.overrides.has_torch_function_variadic 函数
        return torch.overrides.has_torch_function_unary(
            x
        ), torch.overrides.has_torch_function_variadic(x)

    # 测试 MyTensor 和 LocalSubclass 类的行为
    for test_class in [MyTensor, LocalSubclass]:
        # 创建 test_class 的实例 x
        x = test_class()
        # 调用 fn 函数得到 ref0 和 ref1 结果
        ref0 = fn(x)
        ref1 = fn(4)
        # 对 fn 函数应用 eager 模式的优化，并得到 res0 和 res1 结果
        opt_fn = torch._dynamo.optimize("eager")(fn)
        res0 = opt_fn(x)
        res1 = opt_fn(4)
        # 断言 ref0 和 res0 相等
        self.assertEqual(ref0, res0)
        # 断言 ref1 和 res1 相等
        self.assertEqual(ref1, res1)

# 与示例相同，用于检查 s1 的数学表达式
Eq(2*s1, s0)
# 检查 s1 是否小于 13
2*s1 < 13
# 检查 s1 是否大于 3
s1 > 3
    def test_wrapper_subclass_with_same_sized_inner_tensor(self):
        # 对于 dynamic=True，不应该因为不同的尺寸而重新编译
        sub1 = ScaledTensor(torch.randn(2, 4), torch.randn(6))
        sub2 = ScaledTensor(torch.randn(3, 5), torch.randn(7))
        self.assertFalse(_recompiles_for_inputs(func, (sub1,), (sub2,), dynamic=True))

        # 对于 dynamic=False，应该因为不同的数据尺寸而重新编译
        sub1 = ScaledTensor(torch.randn(2, 4), torch.randn(6))
        sub2 = ScaledTensor(torch.randn(3, 5), torch.randn(6))
        self.assertTrue(_recompiles_for_inputs(func, (sub1,), (sub2,), dynamic=False))

        # 使用 manual mark_dynamic() 来避免因为不同数据尺寸而重新编译
        sub1 = ScaledTensor(torch.randn(2, 4), torch.randn(6))
        # 注意：mark_dynamic() 在外层张量上调用，应该会影响到相同尺寸的内层张量
        torch._dynamo.mark_dynamic(sub1, 0)
        torch._dynamo.mark_dynamic(sub1, 1)
        sub2 = ScaledTensor(torch.randn(3, 5), torch.randn(6))
        self.assertFalse(_recompiles_for_inputs(func, (sub1,), (sub2,), dynamic=False))

    def test_wrapper_subclass_with_differently_sized_inner_tensor(self):
        # 对于 dynamic=False，应该因为不同的缩放尺寸而重新编译
        sub1 = ScaledTensor(torch.randn(2, 4), torch.randn(3))
        sub2 = ScaledTensor(torch.randn(2, 4), torch.randn(5))
        self.assertTrue(_recompiles_for_inputs(func, (sub1,), (sub2,), dynamic=False))

        # 使用 manual mark_dynamic() 在外层对不同缩放尺寸仍然会重新编译
        sub1 = ScaledTensor(torch.randn(2, 4), torch.randn(3))
        # 注意：mark_dynamic() 在外层张量上调用，不会传递到不同尺寸的内层张量
        torch._dynamo.mark_dynamic(sub1, 0)
        torch._dynamo.mark_dynamic(sub1, 1)
        sub2 = ScaledTensor(torch.randn(2, 4), torch.randn(5))
        self.assertTrue(_recompiles_for_inputs(func, (sub1,), (sub2,), dynamic=False))

    def test_recompiles_with_optional_inner_tensor(self):
        def f(x):
            return x + 1

        # sub1 没有指定可选张量，而 sub2 则有
        sub1 = OptionalScaledTensor(torch.randn(2, 4), None)
        sub2 = OptionalScaledTensor(torch.randn(2, 4), torch.randn(2, 4))

        # 检查基本情况；相同输入不应重新编译
        self.assertFalse(_recompiles_for_inputs(f, (sub1,), (sub1,), dynamic=True))
        self.assertFalse(_recompiles_for_inputs(f, (sub2,), (sub2,), dynamic=True))

        # 这些情况应该重新编译；可选张量在指定和未指定之间变化
        self.assertTrue(_recompiles_for_inputs(f, (sub1,), (sub2,), dynamic=True))
        self.assertTrue(_recompiles_for_inputs(f, (sub2,), (sub1,), dynamic=True))

        f_compiled = torch.compile(f, backend="aot_eager")
        self.assertEqual(f(sub1)._data, f_compiled(sub1)._data)
        self.assertEqual(f(sub2)._data, f_compiled(sub2)._data)
    # 定义一个测试函数，用于测试 torch_dispatch_subclass_guard_recompile 功能
    def test_torch_dispatch_subclass_guard_recompile(self):
        # 创建一个 2x2 的张量 x，所有元素为 1
        x = torch.ones(2, 2)
        # 创建一个 TwoTensor 对象，使用 x 的克隆作为两个张量的数据源
        x_two = TwoTensor(x.clone(), x.clone())

        # 定义一个函数 fn，接受一个张量 w，返回 w 加 1 的结果张量
        def fn(w):
            return torch.add(w, 1.0)

        # 使用 eager 模式编译函数 fn，得到编译后的版本 fn_opt
        fn_opt = torch.compile(backend="eager")(fn)

        # 对 x_two 调用原始函数 fn 和编译后的函数 fn_opt，比较结果是否相等
        ref = fn(x_two)
        res = fn_opt(x_two)
        self.assertEqual(ref, res)

        # 确保在相同输入类型上没有重新编译
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn_opt(TwoTensor(x + 1, x + 2))

        # 强制重新编译
        ref = fn(x)
        res = fn_opt(x)
        self.assertEqual(ref, res)
    def test_torch_function_subclass_survives_into_aot_autograd(self):
        # 定义一个测试函数，验证子类张量在 AOTAutograd 中是否能正确继承 torch 函数功能
        # 如果张量子类依赖于相同操作的调度而不是解包并调用 torch._C.DisableTorchFunctionSubclass()，
        # 则 torch 函数性将在 AOTAutograd 中保留下来。NestedTensor 目前就依赖于这种行为！因为
        # torch 函数逻辑在 AOTAutograd 运行，此测试确保没有逻辑依赖于在从子类的 torch 函数重新调度后
        # 意外禁用的 torch 函数。
        class SubTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, t):
                # 创建一个包装子类的新实例，继承自 torch.Tensor
                return torch.Tensor._make_wrapper_subclass(
                    cls,
                    t.shape,
                    t.stride(),
                    t.storage_offset(),
                    torch.contiguous_format,
                    t.dtype,
                    torch.strided,
                    t.device,
                    False,
                    t.requires_grad,
                    "sizes",
                    False,
                    False,
                    None,
                )

            def __init__(self, t):
                super().__init__()
                self._t = t

            def __tensor_flatten__(self):
                # 返回用于张量扁平化的信息，这里只展示了 _t 字段
                return ["_t"], {}

            @staticmethod
            def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
                # 从扁平化的张量信息中重建子类张量
                t = inner_tensors["_t"]
                return SubTensor(t)

            def __repr__(self):
                # 返回子类张量的字符串表示形式
                return f"SubTensor({self._t})"

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # 处理 torch 函数的静态方法，使用 torch._C.DisableTorchFunctionSubclass() 禁用 torch 函数子类功能
                if kwargs is None:
                    kwargs = {}

                with torch._C.DisableTorchFunctionSubclass():
                    return func(*args, **kwargs)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # 处理 torch 函数的调度方法，将输入参数转换为 SubTensor 类型，并将输出转换回 torch.Tensor
                kwargs = {} if kwargs is None else kwargs
                new_args = pytree.tree_map_only(SubTensor, lambda s: s._t, args)
                output = func(*new_args, **kwargs)
                output = pytree.tree_map_only(
                    torch.Tensor, lambda t: SubTensor(t), output
                )
                return output

        @torch.compile(dynamic=True)
        def f(x):
            # 编译动态函数 f，用于对输入 x 进行操作
            return x.unflatten(-1, [2, 5])

        s = SubTensor(torch.randn(3, 10))
        f(s)
    def test_recompile_with_symbool_inputs(self):
        # 定义内部函数 f，接受一个布尔型参数 pred
        def f(pred: bool):
            # 如果 pred 为真，则返回一个 3x4 的张量全为 1
            if pred:
                return torch.ones([3, 4])
            else:
                # 如果 pred 不为真，则返回一个 4x3 的张量全为 1
                return torch.ones([4, 3])

        # 定义测试函数 test_recompilation，接受函数 f、输入 x、尺寸 sizes、预期图表 exp_graphs、
        # 预期帧计数 exp_frame_count、预期形状环境保护条件 exp_shape_env_guards 作为参数
        def test_recompilation(
            f, x, sizes, exp_graphs, exp_frame_count, exp_shape_env_guards
        ):
            # 重置 Torch 的动态图
            torch._dynamo.reset()
            # 创建形状环境对象
            shape_env = ShapeEnv()
            # 创建 Torch 的 EagerAndRecordGraphs 后端对象
            backend = torch._dynamo.testing.EagerAndRecordGraphs()
            # 创建带计数器的后端对象
            cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
            # 使用给定的后端编译函数 f，并且启用全图模式
            f_cond = torch.compile(f, backend=cnt, fullgraph=True)
            # 进入假张量模式，使用给定的形状环境对象
            with torch._subclasses.fake_tensor.FakeTensorMode(
                shape_env=shape_env
            ) as fake_mode:
                # 将输入张量 x 转换为假张量，使用无状态符号上下文定义动态尺寸
                fake_inp = fake_mode.from_tensor(
                    x,
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[DimDynamic.DYNAMIC for i in range(x.dim())]
                    ),
                )
                # 遍历 sizes 中的尺寸
                for i, size in enumerate(sizes):
                    # 计算假输入张量的大小是否等于当前尺寸
                    pred = fake_inp.size(0) == size
                    # 调用编译后的函数 f_cond，并传入预测值 pred
                    f_cond(pred)
                    # 根据预期的帧计数，打印指定帧的可读图表
                    actual = normalize_gm(
                        backend.graphs[exp_frame_count[i] - 1].print_readable(
                            print_output=False
                        )
                    )
                    # 提取当前形状环境中所有保护条件的表达式字符串
                    actual_guard_str = [str(guard.expr) for guard in shape_env.guards]
                    # 断言实际输出与预期图表相匹配
                    self.assertExpectedInline(actual, exp_graphs[i])
                    # 断言计数器的帧计数与预期帧计数相等
                    self.assertEqual(cnt.frame_count, exp_frame_count[i])
                    # 断言实际保护条件字符串与预期保护条件字符串相等
                    self.assertEqual(actual_guard_str, exp_shape_env_guards[i])

        true_graph = """
class GraphModule(torch.nn.Module):
    # 定义神经网络模块的类 GraphModule，继承自 torch.nn.Module

    def forward(self):
        # 定义神经网络的前向传播函数 forward

        ones: "f32[3, 4]" = torch.ones([3, 4])
        # 创建一个形状为 [3, 4]，元素类型为 f32 的张量 ones，所有元素初始化为 1

        return (ones,)
        # 返回包含 ones 张量的元组，作为前向传播的输出结果

"""
        false_graph = """\
class GraphModule(torch.nn.Module):
    # 定义另一个神经网络模块的类 GraphModule，继承自 torch.nn.Module

    def forward(self):
        # 定义神经网络的前向传播函数 forward

        ones: "f32[4, 3]" = torch.ones([4, 3])
        # 创建一个形状为 [4, 3]，元素类型为 f32 的张量 ones，所有元素初始化为 1

        return (ones,)
        # 返回包含 ones 张量的元组，作为前向传播的输出结果

"""

        test_recompilation(
            f,
            torch.randn([3, 4]),
            [3, 3, 4, 5],
            exp_graphs=[true_graph, true_graph, false_graph, false_graph],
            exp_frame_count=[1, 1, 2, 2],
            exp_shape_env_guards=[
                [],
                # s0 在外部 shape_env 中特定化并受保护，当 dynamo 检查守卫时
                ["Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)"],
                [
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                    "Ne(Piecewise((1, Eq(s0, 4)), (0, True)), 1)",
                ],
                [
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                    "Ne(Piecewise((1, Eq(s0, 4)), (0, True)), 1)",
                    "Ne(Piecewise((1, Eq(s0, 5)), (0, True)), 1)",
                ],
            ],
        )
        # 调用 test_recompilation 函数进行测试重编译，验证以下参数：
        # - f 函数
        # - 形状为 [3, 4] 的随机张量
        # - 列表 [3, 3, 4, 5] 作为预期的图形
        # - 列表 [1, 1, 2, 2] 作为预期的帧数
        # - 列表包含各种 shape_env 守卫条件，用于形状环境保护

        test_recompilation(
            f,
            torch.randn([3, 4]),
            [4, 5, 3, 3],
            exp_graphs=[false_graph, false_graph, true_graph, true_graph],
            exp_frame_count=[1, 1, 2, 2],
            exp_shape_env_guards=[
                [],
                # s0 在外部 shape_env 中特定化并受保护，当 dynamo 检查守卫时
                ["Ne(Piecewise((1, Eq(s0, 5)), (0, True)), 1)"],
                [
                    "Ne(Piecewise((1, Eq(s0, 5)), (0, True)), 1)",
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                ],
                [
                    "Ne(Piecewise((1, Eq(s0, 5)), (0, True)), 1)",
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                ],
            ],
        )
        # 再次调用 test_recompilation 函数进行测试重编译，验证以下参数：
        # - f 函数
        # - 形状为 [3, 4] 的随机张量
        # - 列表 [4, 5, 3, 3] 作为预期的图形
        # - 列表 [1, 1, 2, 2] 作为预期的帧数
        # - 列表包含各种 shape_env 守卫条件，用于形状环境保护

    def test_wrapper_subclass_dynamo_attribute_access_on_intermediate(self):
        # 定义一个测试函数 test_wrapper_subclass_dynamo_attribute_access_on_intermediate

        def f(x_subclass):
            # 定义函数 f，参数为 x_subclass

            tmp_subclass = torch.add(x, 1)
            # 对输入张量 x 执行加法操作，并赋值给 tmp_subclass

            return torch.mul(tmp_subclass._scale, tmp_subclass._constant)
            # 返回 tmp_subclass 的 _scale 和 _constant 属性相乘的结果

        x = ScaledTensor(torch.randn(2, 4), torch.randn(3), constant=2)
        # 创建一个 ScaledTensor 对象 x，包含随机初始化的形状为 [2, 4] 的张量和随机初始化的常数 2

        out_ref = f(x)
        # 调用函数 f，传入 x，得到参考输出 out_ref

        out_test = torch.compile(f, backend="aot_eager", fullgraph=True)(x)
        # 使用 torch.compile 对函数 f 进行编译，使用 "aot_eager" 后端，并生成完整图形

        self.assertEqual(out_ref, out_test)
        # 使用断言检查 out_ref 和 out_test 是否相等
    # 定义一个测试方法，用于测试支持的基类
    def test_support_bases(self):
        # 导入必要的模块
        import abc
        import torch.fx._symbolic_trace
        
        # 定义一个元类，继承自 ABCMeta 和 ProxyableClassMeta
        class Meta(abc.ABCMeta, torch.fx._symbolic_trace.ProxyableClassMeta):
            # 自定义类实例化方法，添加额外属性 attr 并返回实例
            def __new__(cls, name, bases, dct):
                x = super().__new__(cls, name, bases, dct)
                x.attr = 100
                return x
        
        # 定义一个抽象基类 Multistreamable
        class Multistreamable(abc.ABC):  # noqa: B024
            pass
        
        # 定义一个类 Foo，继承自 Multistreamable，使用 Meta 元类
        class Foo(Multistreamable, metaclass=Meta):
            pass
        
        # 定义一个装饰函数 f，接受参数 x
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # 获取 Foo 实例的类型
            typ = type(Foo())
            # 返回 Foo 实例的基类元组
            return typ.__bases__
        
        # 断言调用 f 函数返回的类型基类是 (Multistreamable,)
        self.assertEqual(f(torch.randn(1)), (Multistreamable,))
    
    # 使用参数化装饰器 parametrize，测试子类视图的函数
    @parametrize("dynamic", [False, True])
    def test_subclass_views(self, dynamic):
        # 定义内部函数 _get_views，接受参数 t
        def _get_views(t):  # returns (view: Tensor, expects_raises_false)
            # 返回视图操作的生成器
            yield t.narrow(dim=-1, start=3, length=8), False
            yield t.split(5, -1)[2], False
            yield t.split_with_sizes([9, 6], -1)[1], False
            yield t.unsqueeze(-1).expand(4, 15, 10), False
            yield t.select(-1, 6), False
            # 给定切片操作引发的异常情况
            # https://github.com/pytorch/pytorch/issues/128649
            yield t[2:3, 5:9], dynamic
            yield t.view(-1, 15), False
        
        # 定义内部函数 f，接受参数 x
        def f(x):
            # 返回 x 的两倍
            return x * 2
        
        # 编译函数 f，使用 aot_eager 后端，全图模式，并根据 dynamic 参数决定是否动态
        compiled_f = torch.compile(
            f, backend="aot_eager", fullgraph=True, dynamic=dynamic
        )
        
        # 创建 TwoTensor 的实例 t，传入两个形状为 (4, 15) 的张量
        t = TwoTensor(torch.randn(4, 15), torch.randn(4, 15))
        
        # 遍历 _get_views 函数返回的视图操作和预期是否引发异常的信息
        for view, expects_raises in _get_views(t):
            # 重置 Torch 的动态图
            torch._dynamo.reset()
            # 对参考实现进行视图操作
            out_ref = f(view)
            # 如果预期引发异常，则使用断言检查是否引发了 AssertionError
            if expects_raises:
                with self.assertRaises(AssertionError):
                    out_test = compiled_f(view)
            else:
                # 否则，使用编译后的函数执行视图操作
                out_test = compiled_f(view)
                # 使用断言检查编译后的结果与参考实现结果是否一致
                self.assertEqual(out_ref, out_test)
# 实例化一个参数化的测试类，使用 SubclassTests 作为参数
instantiate_parametrized_tests(SubclassTests)


class TestNestedTensor(torch._dynamo.test_case.TestCase):
    # 定义获取嵌套张量的方法，返回具有指定大小和偏移的嵌套张量
    def _get_jagged_tensor(self, nested_size, offsets, requires_grad=True):
        return get_jagged_tensor(nested_size, offsets, requires_grad)

    # 定义获取带有内部维度、起始位置和长度的嵌套张量的方法
    def _get_nc_jagged_tensor(self, inner_dim, starts, lengths, requires_grad=True):
        # 创建具有指定形状和数据类型的随机张量
        max_dim = (starts + lengths).max()
        values_tensor = torch.randn(
            starts.shape[0],
            max_dim.item(),
            inner_dim,
            requires_grad=requires_grad,
            dtype=torch.float64,
        )
        # 返回基于张量和长度创建的嵌套张量
        return jagged_from_tensor_and_lengths(values_tensor, starts, lengths)

    # 定义检查给定函数在输入上是否重新编译的方法
    def _check_recompiles(self, fn, inputs1, inputs2, expected_recompiles):
        # 调用内部函数来计算实际重新编译次数
        actual_recompiles = _recompiles_for_inputs(fn, inputs1, inputs2)
        # 断言实际重新编译次数与期望值相等
        self.assertEqual(actual_recompiles, expected_recompiles)

    # 测试一元操作不会导致重新编译
    def test_unary_does_not_recompile(self):
        # 获取两个嵌套张量并进行一元操作检查
        nt1, _ = self._get_jagged_tensor(((2, 3, 4), 3), None)
        nt2, _ = self._get_jagged_tensor(((3, 4, 5, 6), 4), None)
        self._check_recompiles(lambda nt1: nt1.sin(), (nt1,), (nt2,), False)

    # 测试二元操作不会导致重新编译
    def test_binary_does_not_recompile(self):
        # 定义二元操作函数并获取相关嵌套张量
        def binary(nt1, nt2):
            if nt1.shape == nt2.shape:
                return nt1 + nt2
            else:
                return nt1.sin()

        # 获取多组嵌套张量并进行二元操作检查
        nt1, offsets = self._get_jagged_tensor(((2, 3, 4), 5), None)
        nt2, _ = self._get_jagged_tensor(((2, 3, 4), 5), offsets)
        nt3, offsets = self._get_jagged_tensor(((3, 4, 5), 4), None)
        nt4, _ = self._get_jagged_tensor(((3, 4, 5), 4), offsets)
        self._check_recompiles(binary, (nt1, nt2), (nt3, nt4), False)

    # 测试二元操作导致重新编译
    def test_binary_recompiles(self):
        # 定义二元操作函数并获取相关嵌套张量
        def binary(nt1, nt2):
            if nt1.shape == nt2.shape:
                return nt1 + nt2
            else:
                return nt1.sin()

        # 获取多组嵌套张量并进行二元操作检查
        nt1, offsets = self._get_jagged_tensor(((2, 3, 4), 5), None)
        nt2, _ = self._get_jagged_tensor(((2, 3, 4), 5), offsets)
        nt3, _ = self._get_jagged_tensor(((2, 3, 4), 5), None)
        self._check_recompiles(binary, (nt1, nt2), (nt1, nt3), True)

    # TODO: 由于某些原因无法为此测试类添加设备参数化
    # 定义一个测试方法，用于测试自动求导功能
    def _test_autograd(self, backend):
        # 创建随机张量 a, b, c，均为浮点数类型，需要梯度计算
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64)
        # 使用 nested.as_nested_tensor 方法将 a, b, c 转换为嵌套张量 nt，使用 jagged 布局
        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        # TODO: 当公共 API 可用时切换
        # 使用 jagged_from_list 方法创建 nt2，传入张量列表 [a, b, c] 和 nt 的偏移量
        nt2, _ = jagged_from_list([a, b, c], nt.offsets())

        # 定义一个函数 fn1，接受 nt1 和 nt2 作为参数，返回 (nt1 + nt2).sin().cos() 的结果
        def fn1(nt1, nt2):
            return (nt1 + nt2).sin().cos()

        # 使用 torch.compile 方法编译函数 fn1，指定使用指定的后端和动态计算
        compiled_f = torch.compile(fn1, fullgraph=True, backend=backend, dynamic=True)
        # 对 nt 和 nt2 调用编译后的函数 compiled_f，得到输出 out
        out = compiled_f(nt, nt2)
        # 提取输出 out 的值
        out_buffer = out.values()
        # 对 (a, b, c) 进行自动求导，求出梯度 ga, gb, gc
        ga, gb, gc = torch.autograd.grad(out_buffer.sum(), (a, b, c))

        # 对比使用函数 fn1 直接计算的参考输出 out_ref
        out_ref = fn1(nt, nt2)
        # 提取参考输出 out_ref 的值
        out_buffer_ref = out_ref.values()
        # 对参考输出的 (a, b, c) 进行自动求导，求出参考梯度 ga_ref, gb_ref, gc_ref
        ga_ref, gb_ref, gc_ref = torch.autograd.grad(out_buffer_ref.sum(), (a, b, c))

        # 使用 assertTrue 进行断言，检查计算得到的梯度与参考梯度的近似程度
        self.assertTrue(torch.allclose(ga, ga_ref))
        self.assertTrue(torch.allclose(gb, gb_ref))
        self.assertTrue(torch.allclose(gc, gc_ref))

    # 定义基本的自动求导测试方法，调用 _test_autograd 方法，使用 "aot_eager" 后端
    def test_basic_autograd(self):
        self._test_autograd("aot_eager")

    # 使用 @requires_cuda 装饰器定义 CUDA 环境下的自动求导测试方法，调用 _test_autograd 方法，使用 "inductor" 后端
    @requires_cuda
    def test_basic_autograd_inductor(self):
        self._test_autograd("inductor")

    # 定义测试子类中包含图中突变的方法
    def test_subclass_with_mutation_in_graph(self):
        # 在这个图中，存在图中的突变，即允许保留在图中的突变。
        # 通常情况下是允许的，但如果图处理子类，则不允许。
        # 突变是否允许在图中会改变正向图的输出数量。
        # 以前在这个处理中的错误意味着有时正向图的预期输出数量与实际数量不匹配，导致断言失败。

        # 定义函数 fn，接受 x 和 y 作为参数，对 x 进行 sin 计算，对 y 进行 in-place 的 sin 计算，然后返回 z.cos() 和 y.cos() 的结果
        def fn(x, y):
            z = x.sin()
            y.sin_()
            return z.cos(), y.cos()

        # 使用 torch.compile 方法编译函数 fn，指定使用 "inductor" 后端
        fn_c = torch.compile(fn, backend="inductor")

        # 创建随机张量列表 values 和它们的克隆列表 values_copy，均需要梯度计算
        values = [torch.rand((i, 8), requires_grad=True) for i in range(1, 6)]
        values_copy = [x.detach().clone().requires_grad_(True) for x in values]

        # 使用 jagged_from_list 方法将 values 转换为嵌套张量 nt，并获取偏移量 offsets
        nt, offsets = jagged_from_list(values, None)
        # 使用 jagged_from_list 方法将 values_copy 转换为嵌套张量 nt_copy，并使用相同的偏移量 offsets
        nt_copy, offsets = jagged_from_list(values_copy, offsets)
        
        # 创建随机张量 y 和它的克隆 y_copy
        y = torch.rand((4, 8))
        y_copy = y.clone()

        # 调用编译后的函数 fn_c，对 nt 和 y 进行计算，获取输出 ret
        ret = fn_c(nt, y)[0]
        # 使用函数 fn，对 nt_copy 和 y_copy 进行计算，获取参考输出 ref
        ref = fn(nt_copy, y_copy)[0]

        # 使用 assertEqual 进行断言，检查 ret 的值与 ref 的值是否相等
        self.assertEqual(ret.values(), ref.values())

        # 对 ret 和 ref 的值进行求和，并分别进行反向传播
        ret.values().sum().backward()
        ref.values().sum().backward()

        # 使用循环检查 values_copy 和 values 的梯度是否相等
        for ref_v, res_v in zip(values_copy, values):
            self.assertEqual(ref_v.grad, res_v.grad)
    def test_unbind(self):
        # NB: If we have shape e.g. (3, j0, 3), duck sizing will give us (s0, s1, s0).
        # This causes a recompile later on when it realizes the batch and last dim
        # should not always be equal. To avoid that, we use (3, j0, 5) here.
        # 警告: 如果我们的形状是 (3, j0, 3)，鸭子尺寸将给出 (s0, s1, s0)。
        # 这会导致稍后重新编译，当它意识到批次和最后一个维度不总是相等时。
        # 为了避免这种情况，这里我们使用 (3, j0, 5)。

        nt, _ = self._get_jagged_tensor(((2, 3, 4), 5), None)
        nt2, _ = self._get_jagged_tensor(((2, 3, 5), 2), None)
        nt3, _ = self._get_jagged_tensor(((2, 3, 4, 5), 3), None)

        def fn(x):
            return x.unbind()

        # 使用 Torch 编译函数 fn，使用完整图形和动态模式
        compiled_f = torch.compile(fn, fullgraph=True, backend="eager", dynamic=True)
        # 将 nt 作为输入调用编译后的函数
        out = compiled_f(nt)

        # 使用普通的 fn 函数计算参考输出
        out_ref = fn(nt)

        # 校验正确性
        self.assertEqual(len(out), len(out_ref))
        for x, x_ref in zip(out, out_ref):
            self.assertTrue(torch.allclose(x, x_ref))

        # 我们根据 offsets 的长度进行特化，例如 (1) 如果 offsets 的长度不同，我们重新编译。
        # (2) 如果 offsets 的长度相同，即使包含张量的大小不同，我们也不重新编译。
        self._check_recompiles(fn, (nt,), (nt2,), False)
        self._check_recompiles(fn, (nt,), (nt3,), True)

    def test_inline_nested_tensor_from_jagged(self):
        nt, _ = self._get_jagged_tensor(((2, 3, 4), 5), None)

        def fn(x):
            return torch.nested.nested_tensor_from_jagged(x.values() * 2, x.offsets())

        # 使用 Torch 编译函数 fn，使用完整图形和 AOT eager 模式
        torch.compile(fn, fullgraph=True, backend="aot_eager")(nt)

    # The test here: nn.Parameters that are secretly subclasses
    # have a metaclass that overrides __isinstance__,
    # that dynamo needs to respect when it inlines the if statement.
    # 这里的测试是: nn.Parameters 实际上是子类，
    # 它们有一个元类，重写了 __isinstance__，
    # 当 dynamo 内联 if 语句时需要尊重这一点。
    def test_param_subclass_isinstance_input(self):
        x_inner = torch.randn(16, 16, requires_grad=True)
        x = torch.nn.Parameter(TwoTensor(x_inner, x_inner))
        m = torch.nn.Linear(16, 16)
        m.weight = x

        def fn():
            if isinstance(m.weight, torch.nn.Parameter):
                return m.weight + 1
            else:
                return m.weight + 2

        # 计算 fn 的参考输出
        out_ref = fn()
        # 使用 Torch 编译函数 fn，使用 AOT eager 模式
        out_test = torch.compile(fn, backend="aot_eager")()
        self.assertEqual(out_ref, out_test)
    # 定义一个内部测试方法，用于测试指定名称的视图对象
    def _input_view_test(self, nt_view_name):
        # 从预定义的测试用例字典中获取并实例化指定名称的视图对象
        nt_view = VIEW_TEST_CASES[nt_view_name]()

        # 定义一个函数 fn，接受一个参数 x，并调用其 sin 方法后返回结果
        def fn(x):
            return x.sin()

        # 使用 fn 函数处理 nt_view 对象，并将结果保存到 out_ref 中
        out_ref = fn(nt_view)

        # 重置 Torch 的动态机制状态
        torch._dynamo.reset()

        # 使用 Torch 的编译功能编译 fn 函数，启用全图模式，并选择特定后端和动态模式
        compile_fn = torch.compile(
            fn, fullgraph=True, backend="aot_eager", dynamic=True
        )

        # 使用编译后的函数 compile_fn 处理 nt_view 对象，并将结果保存到 out 中
        out = compile_fn(nt_view)

        # 检查元数据和数值是否正确
        self.assertTrue(out.size() == out_ref.size())
        self.assertTrue(out.stride() == out_ref.stride())
        if out.is_nested:
            self.assertTrue(torch.allclose(out.values(), out_ref.values()))
        else:
            self.assertTrue(torch.allclose(out, out_ref))

        # 检查是否没有触发上/下界守卫
        def backend(gm, args):
            # 获取当前追踪上下文中的守卫表达式列表
            context = torch._guards.TracingContext.get()
            guards = [str(g.expr) for g in context.fake_mode.shape_env.guards]

            # 将守卫表达式列表转换为字符串形式
            guard_str = "\n".join(guards)

            # 根据视图类型的不同进行不同的断言
            if nt_view_name == "subclass_dense":
                self.assertExpectedInline(guard_str, """Eq(s3 - 1, s0)""")
            elif nt_view_name == "dense_subclass_dense_subclass":
                self.assertExpectedInline(
                    guard_str,
                    """\
        # 如果视图名称以 "subclass_dense_subclass_dense" 开头，则执行以下断言
        if nt_view_name.startswith("subclass_dense_subclass_dense"):
            self.assertExpectedInline(
                guard_str,
                """\
Eq(s5 - 1, s2)
Eq(s11 - 1, s6)
Eq(s10, s8)""",
            )
        # 如果视图名称以 "base_is_nt_True" 开头，则执行以下断言
        elif nt_view_name.startswith("base_is_nt_True"):
            self.assertExpectedInline(
                guard_str,
                """\
Eq(s3 - 1, s0)
Eq(zf1, zf6)""",
            )
        else:
            # 对于其它视图名称，执行以下断言
            self.assertExpectedInline(
                guard_str,
                """\
Eq(s4 - 1, s1)
Eq(s12 - 1, s7)
Eq(s11, s9)""",
            )
        # 返回修改后的图形
        return gm
```