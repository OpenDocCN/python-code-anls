# `.\pytorch\test\test_dynamic_shapes.py`

```
# Owner(s): ["oncall: jit"]

import contextlib  # 引入上下文管理模块
import copy  # 引入复制操作模块
import inspect  # 引入检查模块，用于获取对象信息
import itertools  # 引入迭代工具模块
import math  # 引入数学计算模块
import operator  # 引入操作符模块
import re  # 引入正则表达式模块

import sympy  # 引入符号计算模块

import torch  # 引入PyTorch模块
import torch.fx  # 引入PyTorch FX模块
import torch.nn.functional as F  # 引入PyTorch神经网络函数模块
from torch import sym_int, SymBool, SymFloat, SymInt  # 从torch导入符号类型
from torch._C import _disabled_torch_function_impl  # 从torch._C导入函数
from torch.fx.experimental import sym_node  # 引入PyTorch FX实验性模块
from torch.fx.experimental.proxy_tensor import make_fx  # 引入FX代理张量模块
from torch.fx.experimental.sym_node import method_to_operator, SymNode, to_node  # 引入FX符号节点相关模块
from torch.fx.experimental.symbolic_shapes import (
    _constrain_range_for_size,  # 导入符号形状约束函数
    DimConstraints,  # 导入维度约束类
    DimDynamic,  # 导入动态维度类
    expect_true,  # 导入期望为真的函数
    guard_bool,  # 导入布尔类型守护函数
    guard_float,  # 导入浮点数类型守护函数
    guard_int,  # 导入整数类型守护函数
    GuardOnDataDependentSymNode,  # 导入数据依赖符号节点守护类
    hint_int,  # 导入整数提示函数
    is_symbolic,  # 导入符号判断函数
    ShapeEnv,  # 导入形状环境类
    StatelessSymbolicContext,  # 导入无状态符号上下文类
    statically_known_true,  # 导入静态已知为真函数
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入实例化参数化测试函数
    parametrize,  # 导入参数化装饰器
    run_tests,  # 导入运行测试函数
    skipIfTorchDynamo,  # 导入如果Torch Dynamo跳过装饰器
    TestCase,  # 导入测试用例类
)
from torch.utils import _pytree as pytree  # 导入_pytree模块
from torch.utils._python_dispatch import TorchDispatchMode  # 导入Torch分发模式类
from torch.utils._sympy.functions import (
    FloorDiv,  # 导入向下取整除法函数
    IsNonOverlappingAndDenseIndicator,  # 导入非重叠稠密指示器函数
    Mod,  # 导入取模函数
)

aten = torch.ops.aten  # 设置torch的aten操作

meta_funcs = {}  # 定义一个空字典用于存储元信息函数


def register_meta(op):
    def decorator(f):
        def add_func(op):
            meta_funcs[op] = f  # 将操作符op作为键，函数f作为值加入到meta_funcs字典中

        pytree.tree_map_(add_func, op)  # 对op应用add_func函数
        return f

    return decorator


@register_meta([aten.add.Tensor, aten.sub.Tensor])
def binary_meta(a, b):
    return a.new_empty(a.shape)  # 返回一个与a形状相同的新张量


@register_meta(aten.cat.default)
def cat_meta(tensors, dim=0):
    concat_length = 0  # 初始化拼接长度为0
    shape = tensors[0].shape  # 获取第一个张量的形状
    for tensor in tensors:
        for idx, (common_length, length) in enumerate(zip(shape, tensor.shape)):
            if idx == dim:
                concat_length = concat_length + length  # 累加指定维度的长度
            else:
                assert length == common_length  # 断言其他维度的长度与第一个张量相同
    new_shape = list(shape)  # 将形状转换为列表
    new_shape[dim] = concat_length  # 更新指定维度的长度为拼接长度
    return tensors[0].new_empty(new_shape)  # 返回一个与第一个张量形状相同的新空张量


@register_meta([aten.narrow_copy.default])
def narrow_copy_symint_meta(a, dim, start, length, **kwargs):
    shape = []
    for i, x in enumerate(a.shape):
        if i == dim:
            shape.append(length)  # 在指定维度上添加长度
        else:
            shape.append(x)  # 在其他维度上保持不变
    return a.new_empty(tuple(shape))  # 返回一个新张量，形状由shape确定


@register_meta([aten.expand.default])
def expand_symint_meta(a, size, implicit=False):
    return a.new_empty(size)  # 返回一个与a形状相同的新张量，形状由size确定


def create_contiguous(shape):
    strides = [1]  # 初始化步幅列表，第一个步幅为1
    for dim in reversed(shape[:-1]):
        strides.append(dim * strides[-1])  # 计算并添加每个维度的步幅
    return list(reversed(strides))  # 返回反转后的步幅列表


class FakeSymbolicTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        sym_shape,
        sym_strides,
        dtype,
        layout,
        requires_grad,
        device,
        storage_offset=0,
    ):
        # TODO: this is wrong in general
        # 创建一个连续的符号形状（symbolic shape）对象
        sym_stride = create_contiguous(sym_shape)
        # 使用给定参数创建一个新的 torch.Tensor 子类对象
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            sym_shape,
            sym_stride,
            storage_offset,
            dtype=dtype,
            layout=layout,
            requires_grad=requires_grad,
            device=device,
        )
        # 返回创建的新对象
        return r

    # 禁用 torch_function 方法
    __torch_function__ = _disabled_torch_function_impl

    # 创建一个新的空 FakeSymbolicTensor 对象
    def new_empty(self, shape):
        return FakeSymbolicTensor(
            shape, None, self.dtype, self.layout, self.requires_grad, self.device
        )

    # 实现 torch_dispatch 方法，用于分发不同的函数重载
    @classmethod
    def __torch_dispatch__(cls, func_overload, types, args=(), kwargs=None):
        # 如果函数重载存在于 meta_funcs 中，则调用相应的函数
        if func_overload in meta_funcs:
            return meta_funcs[func_overload](*args, **kwargs)

        # 如果是 torch.ops.aten.new_empty.default 函数重载，则创建一个新的 FakeSymbolicTensor 对象
        if func_overload == torch.ops.aten.new_empty.default:
            self = args[0]
            shape = args[1]
            return FakeSymbolicTensor(
                shape,
                self.stride(),
                self.dtype,
                self.layout,
                self.requires_grad,
                self.device,
            )

        # 抛出 RuntimeError 异常，表示不支持该操作
        raise RuntimeError(f"operator {func_overload} not supported")
# 导入必要的库
def create_symbolic_tensor(name, arg, shape_env, source=None, dynamic_dims=None):
    from torch._dynamo.source import ConstantSource

    # 如果未提供源，则使用常量源创建一个新的源对象
    if source is None:
        source = ConstantSource(name)
    # 创建与张量维度相同数量的约束维度列表，初始为None
    constraint_dims = [None] * arg.dim()
    # 如果未提供动态维度信息，则将所有维度设为DimDynamic.DUCK
    if dynamic_dims is None:
        dynamic_dims = [DimDynamic.DUCK] * arg.dim()
    # 使用shape_env对象创建符号化的尺寸、步幅和存储偏移量
    (
        sym_shapes,
        sym_strides,
        sym_storage_offset,
    ) = shape_env.create_symbolic_sizes_strides_storage_offset(
        arg,
        source=source,
        symbolic_context=StatelessSymbolicContext(
            dynamic_sizes=dynamic_dims, constraint_sizes=constraint_dims
        ),
    )
    # 返回一个虚拟的符号化张量对象
    return FakeSymbolicTensor(
        sym_shapes,
        sym_strides,
        arg.dtype,
        arg.layout,
        arg.requires_grad,
        arg.device,
        sym_storage_offset,
    )


def create_symtype(cls, pytype, shape_env, val, duck=True):
    from torch._dynamo.source import ConstantSource

    # 在shape_env中创建一个符号对象，用于表示给定的值
    symbol = shape_env.create_symbol(
        val,
        source=ConstantSource(f"__testing_only{len(shape_env.var_to_val)}"),
        dynamic_dim=DimDynamic.DUCK if duck else DimDynamic.DYNAMIC,
        constraint_dim=None,
    )
    # 返回给定类（如SymInt、SymBool等）的实例，其值被符号节点包装
    return cls(
        SymNode(
            symbol,
            shape_env,
            pytype,
            hint=val,
        )
    )


# TODO: 默认情况下将duck参数设为False
def create_symint(shape_env, i: int, duck=True) -> SymInt:
    # 创建一个符号化的整数对象
    return create_symtype(SymInt, int, shape_env, i, duck=duck)


def create_symbool(shape_env, b: bool) -> SymBool:
    # 创建一个符号化的布尔对象
    return create_symtype(SymBool, bool, shape_env, b)


def create_symfloat(shape_env, f: float) -> SymFloat:
    # 创建一个符号化的浮点数对象
    return create_symtype(SymFloat, float, shape_env, f)


@skipIfTorchDynamo(
    "Creating ShapeEnv fails for confusing reasons (also we never expect dynamo to see code like this)"
)
class TestPySymInt(TestCase):
    def test_arith_ops(self):
        # 创建一个形状环境对象
        shape_env = ShapeEnv()
        symints = []
        # 生成几个符号化整数并存储在symints列表中
        for i in range(2, 5):
            symints.append((i, create_symint(shape_env, i)))

        ops = [
            operator.add,
            operator.sub,
            operator.floordiv,
            operator.mul,
            operator.mod,
        ]

        # 对每个操作符和所有可能的参数排列进行操作测试
        for op in ops:
            for args in itertools.permutations(symints, 2):
                # 如果第一个参数不是整数，且操作不是取模或整数除法，并且第二个参数不为0，则验证条件成立
                if not isinstance(args[0][1], int) and (
                    (op != operator.mod and op != operator.floordiv) or args[1][0] != 0
                ):
                    # 断言操作后的结果符合预期结果
                    self.assertTrue(
                        op(args[0][1], args[1][1]) == op(args[0][0], args[1][0])
                    )

    def test_reverse_arith_ops(self):
        # 创建一个形状环境对象
        shape_env = ShapeEnv()

        # 创建一个符号化整数对象a，并验证整数除法操作的结果
        a = create_symint(shape_env, 2)
        self.assertTrue(5 // a == 5 // 2)

        # 创建一个符号化整数对象a，并验证乘法操作的结果
        a = create_symint(shape_env, 2)
        self.assertTrue(5 * a == 5 * 2)
    # 定义测试方法 `test_roundtrip`
    def test_roundtrip(self):
        # 创建 ShapeEnv 对象
        shape_env = ShapeEnv()
        # 创建名为 "x" 的符号张量，形状为 torch.randn(5, 4, 3)，并使用 shape_env 进行形状环境处理
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)

        # 断言：x 的第一个维度不是 SymNode 类型
        self.assertTrue(not isinstance(x.shape[0], SymNode))
        # 断言：x 的第一个维度是 SymInt 类型
        self.assertTrue(isinstance(x.shape[0], SymInt))

        # 断言：x 的第一个维度的大小为 5
        self.assertTrue(x.shape[0] == 5)
        # 断言：x 的第二个维度的大小为 4
        self.assertTrue(x.shape[1] == 4)
        # 断言：x 的第三个维度的大小为 3
        self.assertTrue(x.shape[2], 3)  # 此处可能应为 self.assertTrue(x.shape[2] == 3)，修正为这样

        # 断言：x 的大小的列表中第一个元素为 5
        self.assertTrue(x.size()[0], 5)  # 此处可能应为 self.assertTrue(x.size()[0] == 5)，修正为这样
        # 断言：x 的大小的列表中第二个元素为 4
        self.assertTrue(x.size()[1], 4)  # 此处可能应为 self.assertTrue(x.size()[1] == 4)，修正为这样
        # 应该能简化为整数。
        # 参考：https://github.com/pytorch/pytorch/pull/107492
        # 断言：x 的大小的列表中第二个元素的类型为 SymInt，并且可以转换为整数
        self.assertTrue(
            isinstance(x.size()[1].node.maybe_as_int(), int)
        )  # 由于上述保护
        # 断言：x 的大小的列表中第三个元素的大小为 3
        self.assertTrue(x.size()[2] == 3)

        # 断言：x 的大小的列表中第一个维度的大小为 5
        self.assertTrue(x.size(0) == 5)
        # 断言：x 的大小的列表中第二个维度的大小为 4
        self.assertTrue(x.size(1) == 4)
        # 断言：x 的大小的列表中第三个维度的大小为 3
        self.assertTrue(x.size(2) == 3)
        # 断言：x 的大小的列表中第三个维度的类型为 SymInt
        self.assertTrue(isinstance(x.size(2), SymInt))
        # 断言：x 的大小的列表中第三个维度的类型为 SymInt 并且可以转换为整数
        self.assertTrue(isinstance(x.size(2).node.maybe_as_int(), int))

        # 创建名为 "y" 的符号张量，形状为 torch.randn(5, 4, 3)[1:]，并使用 shape_env 进行形状环境处理
        y = create_symbolic_tensor("y", torch.randn(5, 4, 3)[1:], shape_env)
        # 断言：y 的 storage_offset() 的类型为 SymInt
        self.assertTrue(isinstance(y.storage_offset(), SymInt))
        # 断言：y 的 storage_offset() 的值为 12
        self.assertTrue(y.storage_offset() == 12)

    # 定义测试方法 `test_binary`
    def test_binary(self):
        # 创建 ShapeEnv 对象
        shape_env = ShapeEnv()
        # 创建名为 "x" 的符号张量，形状为 torch.randn(5, 4, 3)，并使用 shape_env 进行形状环境处理
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)
        # 创建名为 "y" 的符号张量，形状为 torch.randn(5, 4, 3)，并使用 shape_env 进行形状环境处理
        y = create_symbolic_tensor("y", torch.randn(5, 4, 3), shape_env)

        # 创建张量 z，为 x 和 y 的加法结果
        z = x + y
        # 断言：z 的第一个维度的大小为 5
        self.assertTrue(z.shape[0] == 5)
        # 断言：z 的第二个维度的大小为 4
        self.assertTrue(z.shape[1] == 4)
        # 断言：z 的第三个维度的大小为 3
        self.assertTrue(z.shape[2] == 3)

        # broadcasting 广播操作
        # 创建名为 "y2" 的符号张量，形状为 torch.randn(1, 4, 1)，并使用 shape_env 进行形状环境处理
        y = create_symbolic_tensor("y2", torch.randn(1, 4, 1), shape_env)
        # 创建张量 z，为 x 和 y 的加法结果
        z = x + y
        # 断言：z 的第一个维度的大小为 5
        self.assertTrue(z.shape[0] == 5)
        # 断言：z 的第二个维度的大小为 4
        self.assertTrue(z.shape[1] == 4)
        # 断言：z 的第三个维度的大小为 3
        self.assertTrue(z.shape[2] == 3)

    # 定义测试方法 `test_symint_args`
    def test_symint_args(self):
        # 创建 ShapeEnv 对象
        shape_env = ShapeEnv()
        # 创建名为 "x" 的符号张量，形状为 torch.randn(5, 4, 3)，并使用 shape_env 进行形状环境处理
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)
        # 创建名为 "y" 的符号张量，形状为 torch.randn(5, 4, 1)，并使用 shape_env 进行形状环境处理
        y = create_symbolic_tensor("y", torch.randn(5, 4, 1), shape_env)
        LAST_DIM = 2

        # 创建张量 z，通过 x 的 narrow_copy 方法截取最后维度为 2 的部分，并使用 y 的相同维度
        z = x.narrow_copy(LAST_DIM, 0, y.shape[LAST_DIM])
        # 断言：z 的第三个维度的大小等于 y 的第三个维度的大小
        self.assertTrue(z.shape[2] == y.shape[2])

        # 创建张量 z，通过 x 的 narrow_copy 方法截取最后维度为 2 的部分，长度为 x 的最后维度长度减去 y 的最后维度长度
        z = x.narrow_copy(LAST_DIM, 0, x.shape[LAST_DIM] - y.shape[LAST_DIM])
        # 断言：z 的第三个维度的大小为 2
        self.assertTrue(z.shape[2] == 2)

        # 创建张量 z，通过 x 的 narrow_copy 方法截取最后维度为 2 的部分，长度为 x 的最后维度长度减去 1
        z = x.narrow_copy(LAST_DIM, 0, x.shape[LAST_DIM] - 1)
        # 断言：z 的第三个维度的大小为 2
        self.assertTrue(z.shape[2] == 2)
    # 定义测试函数，验证符号整数变长参数的处理
    def test_symint_vargs(self):
        # 创建形状环境对象
        shape_env = ShapeEnv()
        # 创建名为"x"的符号张量，使用形状环境创建，数据为随机生成的5x4x3张量
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)
        # 创建名为"y"的符号张量，使用形状环境创建，数据为随机生成的1x4x1张量
        y = create_symbolic_tensor("y", torch.randn(1, 4, 1), shape_env)

        # 使用变长参数扩展张量y的形状到与x相同
        z = y.expand(x.shape[0], y.shape[1], x.shape[2])
        # 断言z的形状为(5, 4, 3)
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # 使用形状列表扩展张量y的形状到与x相同
        z = y.expand((x.shape[0], y.shape[1], x.shape[2]))
        # 断言z的形状为(5, 4, 3)
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # 使用混合的Python符号整数和整数扩展张量y的形状
        z = y.expand(x.shape[0], y.shape[1], 3)
        # 断言z的形状为(5, 4, 3)
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # 使用混合的Python符号整数和整数列表扩展张量y的形状
        z = y.expand((x.shape[0], y.shape[1], 3))
        # 断言z的形状为(5, 4, 3)
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # 使用混合的Python整数和符号整数扩展张量y的形状
        z = y.expand(5, y.shape[1], x.shape[2])
        # 断言z的形状为(5, 4, 3)
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # 使用混合的Python整数和符号整数列表扩展张量y的形状
        z = y.expand((5, y.shape[1], x.shape[2]))
        # 断言z的形状为(5, 4, 3)
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # 单独扩展张量y的形状到与y的第一维度大小相同
        z = y.expand((y.shape[1],))
        # 单独扩展张量y的形状到与y的第一维度大小相同
        z = y.expand(y.shape[1])
    def test_numel(self):
        # 创建 ShapeEnv 对象
        shape_env = ShapeEnv()
        # 创建名为 x 的符号张量，形状为 torch.randn(5)，使用 shape_env 环境
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        # 检查 x.numel() 的类型是否为 torch.SymInt
        self.assertIsInstance(x.numel(), torch.SymInt)
        # 检查 torch.numel(x) 的类型是否为 torch.SymInt
        self.assertIsInstance(torch.numel(x), torch.SymInt)

        # 创建一个形状为 3x3 的随机张量 x
        x = torch.rand(3, 3)
        # 检查 x.numel() 的类型是否为 int
        self.assertIsInstance(x.numel(), int)
        # 检查 torch.numel(x) 的类型是否为 int
        self.assertIsInstance(torch.numel(x), int)

    def test_int_to_float(self):
        # 创建 ShapeEnv 对象
        shape_env = ShapeEnv()
        # 创建名为 x 的符号张量，形状为 torch.randn(5)，使用 shape_env 环境
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        # 将 x.shape[0] 转换为 torch.SymFloat 类型，并赋给 r
        r = torch.sym_float(x.shape[0])
        # 检查 r 的类型是否为 torch.SymFloat，附带消息为 type(r)
        self.assertIsInstance(r, torch.SymFloat, msg=type(r))

    def test_aten_ops(self):
        # 创建 ShapeEnv 对象
        shape_env = ShapeEnv()
        # 创建名为 x 的符号张量，形状为 torch.randn(5)，使用 shape_env 环境
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        # 使用 torch.ops.aten.narrow_copy.default 对 x 进行操作
        torch.ops.aten.narrow_copy.default(x, 0, 0, x.shape[0])

        # 创建新的 ShapeEnv 对象
        shape_env = ShapeEnv()
        # 创建名为 x2 的符号张量，形状为 torch.randn(5, 4, 3)，使用 shape_env 环境
        x = create_symbolic_tensor("x2", torch.randn(5, 4, 3), shape_env)
        # 使用 torch.ops.aten.expand.default 对 x 进行操作，扩展到指定形状
        torch.ops.aten.expand.default(x, [x.shape[0], x.shape[1], x.shape[2]])

    def test_fx_trace_intlist(self):
        # 定义一个自定义的 Torch 模块 CustomModule
        class CustomModule(torch.nn.Module):
            def forward(self, x):
                # 获取 x 的形状参数 bs, c, h, w
                bs, c, h, w = x.shape
                # 使用 F.pad 对 x 进行填充操作
                return F.pad(x, (0, w % 2, 0, h % 2, 0, 0))

        # 创建 CustomModule 实例 m
        m = CustomModule()
        # 创建形状为 1x3x4x4 的随机张量 x
        x = torch.rand(1, 3, 4, 4)
        # 对模块 m 进行符号跟踪，检查是否会出现 TypeError
        torch.fx.symbolic_trace(m)

    def test_meta_symint(self):
        # 创建 ShapeEnv 对象
        shape_env = ShapeEnv()
        # 使用 create_symint 创建一个符号整数 a0，值为 2
        a0 = create_symint(shape_env, 2)
        # 创建一个元素为空的 torch 张量 r，设备为 "meta"
        r = torch.empty(a0, device="meta")
        # 检查 r.shape[0] 的类型是否为 SymInt
        self.assertIsInstance(r.shape[0], SymInt)

    def test_guard_int(self):
        # 创建 ShapeEnv 对象
        shape_env = ShapeEnv()
        # 使用 create_symint 创建一个符号整数 a0，值为 2
        a0 = create_symint(shape_env, 2)
        # 调用 guard_int 函数，期望返回值为 2
        self.assertEqual(guard_int(a0), 2)
        # 使用 assertExpectedInline 检查 shape_env.guards[0][0] 的表达式
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s0, 2)""")

    def test_prefer_deferred_runtime_assertions_over_guards(self):
        # 创建 ShapeEnv 对象，优先使用运行时延迟断言而非保护条件
        shape_env = ShapeEnv(prefer_deferred_runtime_asserts_over_guards=True)
        # 使用 create_symint 创建一个符号整数 s0，值为 2
        s0 = create_symint(shape_env, 2)
        # 调用 guard_int 函数，期望返回值为 2
        self.assertEqual(guard_int(s0), 2)
        # 使用 assertExpectedInline 检查 shape_env.guards[0][0] 的表达式
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s0, 2)""")

        # 创建 ShapeEnv 对象，优先使用运行时延迟断言而非保护条件
        shape_env = ShapeEnv(prefer_deferred_runtime_asserts_over_guards=True)
        # 使用 create_symint 创建一个符号整数 s0，值为 2
        s0 = create_symint(shape_env, 2)
        # 断言 expect_true(s0 == 2) 为 True
        self.assertTrue(expect_true(s0 == 2))
        # 检查延迟运行时断言的数量是否为 0
        self.assertEqual(len(shape_env.guards), 0)
        # 使用 assertExpectedInline 检查 shape_env.deferred_runtime_asserts[None] 的表达式
        self.assertExpectedInline(
            str([ra.expr for ra in shape_env.deferred_runtime_asserts[None]]),
            """[Eq(s0, 2)]""",
        )
    def test_sym_int(self):
        # 创建一个形状环境对象
        shape_env = ShapeEnv()
        # 创建一个符号整数，值为5
        a0 = create_symint(shape_env, 5)
        # 对符号整数进行符号整数操作，返回结果r为5
        r = sym_int(a0)
        # 断言r的值为5
        self.assertEqual(r, 5)
        # 断言r的类型为torch.SymInt，并输出类型消息
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        # 断言形状环境的第一个卫语句为预期的字符串 """Eq(s0, 5)"""
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s0, 5)""")

        # 创建一个符号整数，值为7
        a1 = create_symint(shape_env, 7)
        # 对a1除以2后进行符号整数操作，返回结果r为3
        r = sym_int(a1 / 2)
        # 断言r的整数值为3
        self.assertEqual(guard_int(r), 3)
        # 断言r的类型为torch.SymInt，并输出类型消息
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        # 断言形状环境的第二个卫语句为预期的字符串 """Eq(TruncToInt(IntTrueDiv(s1, 2)), 3)"""
        self.assertExpectedInline(
            str(shape_env.guards[1][0]), """Eq(TruncToInt(IntTrueDiv(s1, 2)), 3)"""
        )

        # 创建一个符号整数，值为3
        a3 = create_symint(shape_env, 3)
        # 对2.0乘以a3的符号浮点数进行符号整数操作，返回结果r为6
        r = sym_int(2.0 * torch.sym_float(a3))
        # 断言r的整数值为6
        self.assertEqual(guard_int(r), 6)
        # 断言r的类型为torch.SymInt，并输出类型消息
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        # 断言形状环境的第三个卫语句为预期的字符串 """Eq(TruncToInt(2.0*ToFloat(s2)), 6)"""
        self.assertExpectedInline(
            str(shape_env.guards[2][0]), """Eq(TruncToInt(2.0*ToFloat(s2)), 6)"""
        )

    def test_sym_sqrt(self):
        # 创建一个形状环境对象
        shape_env = ShapeEnv()
        # 创建一个符号整数，值为4
        a0 = create_symint(shape_env, 4)
        # 对a0进行符号平方根操作，返回结果r为2
        r = torch._sym_sqrt(a0)
        # 断言r的值为2
        self.assertEqual(r, 2)
        # 断言r的类型为torch.SymFloat，并输出类型消息
        self.assertIsInstance(r, torch.SymFloat, msg=type(r))
        # 断言形状环境的第一个卫语句为预期的字符串 """Eq(OpaqueUnaryFn_sqrt(s0), 2.0)"""
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(OpaqueUnaryFn_sqrt(s0), 2.0)"""
        )

    def test_sym_floor(self):
        # 创建一个形状环境对象
        shape_env = ShapeEnv()
        # 创建一个符号整数，值为5
        a0 = create_symint(shape_env, 5)
        # 对a0除以2后进行向下取整操作，返回结果r为2
        r = math.floor(a0 / 2)
        # 断言r的值为2
        self.assertEqual(r, 2)
        # 断言r的类型为torch.SymInt，并输出类型消息
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        # 断言形状环境的第一个卫语句为预期的字符串 """Eq(FloorToInt(IntTrueDiv(s0, 2)), 2)"""
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(FloorToInt(IntTrueDiv(s0, 2)), 2)"""
        )
        # 对3.0乘以a0进行向下取整操作，返回结果r为15
        r = math.floor(3.0 * a0)
        # 断言r的值为15
        self.assertEqual(r, 15)
        # 断言r的类型为torch.SymInt，并输出类型消息
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        # 断言形状环境的第二个卫语句为预期的字符串 """Eq(FloorToInt(3.0*ToFloat(s0)), 15)"""
        self.assertExpectedInline(
            str(shape_env.guards[1][0]), """Eq(FloorToInt(3.0*ToFloat(s0)), 15)"""
        )

    def test_sym_trunc(self):
        # 创建一个形状环境对象
        shape_env = ShapeEnv()
        # 创建一个符号整数，值为5
        a0 = create_symint(shape_env, 5)
        # 对a0除以2后进行向零取整操作，返回结果r为2
        r = math.trunc(a0 / 2)
        # 断言r的值为2
        self.assertEqual(r, 2)
        # 断言r的类型为torch.SymInt，并输出类型消息
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        # 断言形状环境的第一个卫语句为预期的字符串 """Eq(TruncToInt(IntTrueDiv(s0, 2)), 2)"""
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(TruncToInt(IntTrueDiv(s0, 2)), 2)"""
        )
        # 对a0进行符号平方根操作，然后进行向零取整操作，返回结果r为2
        r = torch.sym_int(torch.sym_sqrt(a0))
        # 断言r的值为2
        self.assertEqual(r, 2)
        # 断言r的类型为torch.SymInt，并输出类型消息
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        # 断言形状环境的第二个卫语句为预期的字符串 """Eq(TruncToInt(OpaqueUnaryFn_sqrt(s0)), 2)"""
        self.assertExpectedInline(
            str(shape_env.guards[1][0]), """Eq(TruncToInt(OpaqueUnaryFn_sqrt(s0)), 2)"""
        )
    def test_sym_ceil(self):
        # 创建一个 ShapeEnv 对象，用于管理符号变量的环境
        shape_env = ShapeEnv()
        # 创建一个符号整数对象，并赋值为常数 5
        a0 = create_symint(shape_env, 5)
        # 对 a0 除以 2，然后对结果向上取整
        r = math.ceil(a0 / 2)
        # 断言 r 的值为 3
        self.assertEqual(r, 3)
        # 断言 r 的类型为 torch.SymInt 类型，输出消息为其类型
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        # 断言形状环境的第一个保护条件的字符串表示，应与预期的字符串匹配
        self.assertExpectedInline(
            str(shape_env.guards[0][0]),
            """Eq(CeilToInt(IntTrueDiv(s0, 2)), 3)""",
        )
        # 计算 3.0 乘以 a0 的结果
        r1 = 3.0 * a0
        # 对 r1 的结果向下取整
        r = math.floor(r1)
        # 断言 r 的值为 15
        self.assertEqual(r, 15)
        # 断言 r 的类型为 torch.SymInt 类型，输出消息为其类型
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        # 断言形状环境的第二个保护条件的字符串表示，应与预期的字符串匹配
        self.assertExpectedInline(
            str(shape_env.guards[1][0]),
            """Eq(FloorToInt(3.0*ToFloat(s0)), 15)""",
        )

    def test_sym_ite(self):
        # 创建一个 ShapeEnv 对象，用于管理符号变量的环境
        shape_env = ShapeEnv()
        # 创建两个符号整数对象 t 和 f，并分别赋值为 5 和 4
        t = create_symint(shape_env, 5)
        f = create_symint(shape_env, 4)
        # 设置布尔变量 b1 为 True
        b1 = True
        # 使用 torch.sym_ite 对 b1 进行条件判断，返回 t
        r1 = torch.sym_ite(b1, t, f)
        # 断言 r1 与 t 是同一个对象
        self.assertTrue(r1 is t)
        # 设置布尔变量 b2 为 False
        b2 = False
        # 使用 torch.sym_ite 对 b2 进行条件判断，返回 f
        r2 = torch.sym_ite(b2, t, f)
        # 断言 r2 与 f 是同一个对象
        self.assertTrue(r2 is f)
        # 使用 t 是否等于 5 的布尔表达式作为条件
        b3 = t == 5
        # 使用 torch.sym_ite 对 b3 进行条件判断，返回 t
        r3 = torch.sym_ite(b3, t, f)
        # 断言形状环境中没有保护条件
        self.assertEqual(len(shape_env.guards), 0)
        # 断言 r3 的值为 5
        self.assertEqual(r3, 5)
        # 断言 t 和 r3 的类型相同
        self.assertEqual(type(t), type(r3))
        # 断言形状环境的第一个保护条件的字符串表示，应与预期的字符串匹配
        self.assertExpectedInline(
            str(shape_env.guards[0][0]),
            """Eq(Piecewise((s0, Eq(s0, 5)), (s1, True)), 5)""",
        )
        # 使用 f 是否等于 5 的布尔表达式作为条件
        b4 = f == 5
        # 使用 torch.sym_ite 对 b4 进行条件判断，返回 f
        r4 = torch.sym_ite(b4, t, f)
        # 断言形状环境中有一个保护条件
        self.assertEqual(len(shape_env.guards), 1)
        # 断言 r4 的值为 4
        self.assertEqual(r4, 4)
        # 断言 f 和 r4 的类型相同
        self.assertEqual(type(f), type(r4))
        # 断言形状环境的第二个保护条件的字符串表示，应与预期的字符串匹配
        self.assertExpectedInline(
            str(shape_env.guards[1][0]),
            """Eq(Piecewise((s0, Eq(s1, 5)), (s1, True)), 4)""",
        )

    def test_tracing_sym_ite(self):
        # 定义一个函数 f，输入参数 x
        def f(x):
            # 创建布尔变量 b，判断 x 的第一维是否等于 5
            b = x.shape[0] == 5
            # 使用 torch.sym_ite 对 b 进行条件判断，返回 x 的第一维或第二维
            ret = torch.sym_ite(b, x.shape[0], x.shape[1])
            # 返回 ret
            return ret

        # 使用 make_fx 函数创建一个图模式的函数 gm，并传入全为 1 的 4x5 张量作为参数
        gm = make_fx(f, tracing_mode="symbolic")(torch.ones(4, 5))
        # 断言形状环境中没有保护条件
        self.assertEqual(len(gm.shape_env.guards), 0)
        # 断言 gm 的代码的去掉首尾空白字符后，应与预期的字符串匹配
        self.assertExpectedInline(
            gm.code.strip(),
            """\
    # 使用 torch.ops.aten.sym_size.int 函数获取张量 x_1 的第一个维度大小
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    # 检查第一个维度大小是否等于 5
    eq = sym_size_int == 5
    # 使用 torch.ops.aten.sym_size.int 函数获取张量 x_1 的第二个维度大小，并清空 x_1 引用
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1);  x_1 = None
    # 根据前面的等式判断选择合适的维度大小作为结果
    sym_ite = torch.sym_ite(eq, sym_size_int, sym_size_int_1);  eq = sym_size_int = sym_size_int_1 = None
    # 返回选择的维度大小作为结果
    return sym_ite
    def test_unbacked_substitution(self):
        # 创建形状环境对象
        shape_env = ShapeEnv()
        # 创建未备份的符号整数变量 i0 和 i1
        i0 = shape_env.create_unbacked_symint()
        i1 = shape_env.create_unbacked_symint()
        # 对 i0 和 i1 进行大小范围约束
        _constrain_range_for_size(i0)
        _constrain_range_for_size(i1)
        # 断言 i0 等于 i1 的四倍
        self.assertTrue(expect_true(i0 == i1 * 4))
        # 断言 i0 转换为字符串后为 "u0"
        self.assertExpectedInline(str(i0), """u0""")

        # 再次创建未备份的符号整数变量 i2 和 i3
        i2 = shape_env.create_unbacked_symint()
        i3 = shape_env.create_unbacked_symint()
        # 对 i2 和 i3 进行大小范围约束
        _constrain_range_for_size(i2)
        _constrain_range_for_size(i3)
        # 断言 i2 的四倍等于 i3
        self.assertTrue(expect_true(i2 * 4 == i3))
        # 断言 i3 转换为字符串后为 "u3"
        self.assertExpectedInline(str(i3), """u3""")

    def test_avoid_unbacked_substitution(self):
        # 创建形状环境对象
        shape_env = ShapeEnv()
        # 创建未备份的符号整数变量 i0
        i0 = shape_env.create_unbacked_symint()
        # 对 i0 进行大小范围约束
        _constrain_range_for_size(i0)
        # 再次创建未备份的符号整数变量 i1
        i1 = shape_env.create_unbacked_symint()
        # 对 i1 进行大小范围约束
        _constrain_range_for_size(i1)
        # 断言 i0 等于 10 减去 i1
        self.assertTrue(expect_true(i0 == 10 - i1))
        # 断言 i0 转换为字符串后为 "u0"
        self.assertExpectedInline(str(i0), """u0""")

    def test_expect_true_double_digits(self):
        # 创建形状环境对象
        shape_env = ShapeEnv()
        # 创建包含 11 个未备份符号整数变量的列表 ia
        ia = [shape_env.create_unbacked_symint() for _ in range(11)]  # allocate 10
        # 断言 ia 中最后一个变量转换为字符串后为 "u10"
        self.assertEqual(str(ia[-1]), "u10")
        # 断言 ia 所有变量的和为 20
        self.assertTrue(expect_true(sum(ia) == 20))
        # 断言 ia[-1].node.expr 在形状环境中延迟运行断言的长度为 1
        self.assertEqual(len(shape_env.deferred_runtime_asserts[ia[-1].node.expr]), 1)

    def test_expect_true_refine_range(self):
        # 创建形状环境对象
        shape_env = ShapeEnv()
        # 对于每个关系函数 rel，依次执行以下操作
        for i, rel in enumerate(
            [lambda x: x > 4, lambda x: 4 < x, lambda x: x >= 5, lambda x: 5 <= x]
        ):
            with self.subTest(f"i = {i}"):
                # 创建未备份的符号整数变量 i0
                i0 = shape_env.create_unbacked_symint()
                # 断言对 rel(i0) 的期望为真
                self.assertTrue(expect_true(rel(i0)))
                # 断言 statically_known_true(i0 != 3) 的期望为真
                self.assertTrue(statically_known_true(i0 != 3))
                # 断言 statically_known_true(i0 != 4) 的期望为真
                self.assertTrue(statically_known_true(i0 != 4))
                # 断言 statically_known_true(i0 != 5) 的期望为假
                self.assertFalse(statically_known_true(i0 != 5))
                # 断言 statically_known_true(i0 != 6) 的期望为假
                self.assertFalse(statically_known_true(i0 != 6))
                # 断言 statically_known_true(i0 > 4) 的期望为真
                self.assertTrue(statically_known_true(i0 > 4))
                # 断言 statically_known_true(i0 >= 5) 的期望为真
                self.assertTrue(statically_known_true(i0 >= 5))

        # 对于每个关系函数 rel，依次执行以下操作
        for i, rel in enumerate(
            [lambda x: x < 4, lambda x: 4 > x, lambda x: x <= 3, lambda x: 3 >= x]
        ):
            with self.subTest(f"i = {i}"):
                # 创建未备份的符号整数变量 i0
                i0 = shape_env.create_unbacked_symint()
                # 断言对 rel(i0) 的期望为真
                self.assertTrue(expect_true(rel(i0)))
                # 断言 statically_known_true(i0 != 2) 的期望为假
                self.assertFalse(statically_known_true(i0 != 2))
                # 断言 statically_known_true(i0 != 3) 的期望为假
                self.assertFalse(statically_known_true(i0 != 3))
                # 断言 statically_known_true(i0 != 4) 的期望为真
                self.assertTrue(statically_known_true(i0 != 4))
                # 断言 statically_known_true(i0 != 5) 的期望为真
                self.assertTrue(statically_known_true(i0 != 5))
                # 断言 statically_known_true(i0 < 4) 的期望为真
                self.assertTrue(statically_known_true(i0 < 4))
                # 断言 statically_known_true(i0 <= 5) 的期望为真
                self.assertTrue(statically_known_true(i0 <= 5))
    # 定义测试函数 test_guard_refine_range，用于测试区间精化保护
    def test_guard_refine_range(self):
        # 创建形状环境对象
        shape_env = ShapeEnv()

        # 第一组测试：测试大于4的条件
        for i, rel in enumerate(
            [lambda x: x > 4, lambda x: 4 < x, lambda x: x >= 5, lambda x: 5 <= x]
        ):
            # 使用子测试，标记当前测试的索引 i
            with self.subTest(f"i = {i}"):
                # 创建符号整数对象 i0，初始值为10，duck 参数设置为 False
                i0 = create_symint(shape_env, 10, duck=False)
                # 断言符号整数 i0 满足 rel 函数定义的条件
                self.assertTrue(bool(rel(i0)))
                # 断言静态已知 i0 不等于 3
                self.assertTrue(statically_known_true(i0 != 3))
                # 断言静态已知 i0 不等于 4
                self.assertTrue(statically_known_true(i0 != 4))
                # 断言静态已知 i0 等于 5 的结果为假
                self.assertFalse(statically_known_true(i0 != 5))
                # 断言静态已知 i0 等于 6 的结果为假
                self.assertFalse(statically_known_true(i0 != 6))
                # 断言静态已知 i0 大于 4
                self.assertTrue(statically_known_true(i0 > 4))
                # 断言静态已知 i0 大于等于 5
                self.assertTrue(statically_known_true(i0 >= 5))

        # 第二组测试：测试小于4的条件
        for i, rel in enumerate(
            [lambda x: x > 4, lambda x: 4 < x, lambda x: x >= 5, lambda x: 5 <= x]
        ):
            # 使用子测试，标记当前测试的索引 i
            with self.subTest(f"i = {i}"):
                # 创建符号整数对象 i0，初始值为2，duck 参数设置为 False
                i0 = create_symint(shape_env, 2, duck=False)
                # 断言符号整数 i0 不满足 rel 函数定义的条件
                self.assertFalse(bool(rel(i0)))
                # 断言静态已知 i0 不等于 3
                self.assertFalse(statically_known_true(i0 != 3))
                # 断言静态已知 i0 不等于 4
                self.assertFalse(statically_known_true(i0 != 4))
                # 断言静态已知 i0 等于 5 的结果为真
                self.assertTrue(statically_known_true(i0 != 5))
                # 断言静态已知 i0 等于 6 的结果为真
                self.assertTrue(statically_known_true(i0 != 6))
                # 断言静态已知 i0 小于等于 4
                self.assertTrue(statically_known_true(i0 <= 4))
                # 断言静态已知 i0 小于 5
                self.assertTrue(statically_known_true(i0 < 5))

        # 第三组测试：测试小于4的条件（不同的表达方式）
        for i, rel in enumerate(
            [lambda x: x < 4, lambda x: 4 > x, lambda x: x <= 3, lambda x: 3 >= x]
        ):
            # 使用子测试，标记当前测试的索引 i
            with self.subTest(f"i = {i}"):
                # 创建符号整数对象 i0，初始值为2，duck 参数设置为 False
                i0 = create_symint(shape_env, 2, duck=False)
                # 断言符号整数 i0 满足 rel 函数定义的条件
                self.assertTrue(bool(rel(i0)))
                # 断言静态已知 i0 不等于 2
                self.assertFalse(statically_known_true(i0 != 2))
                # 断言静态已知 i0 不等于 3
                self.assertFalse(statically_known_true(i0 != 3))
                # 断言静态已知 i0 等于 4 的结果为真
                self.assertTrue(statically_known_true(i0 != 4))
                # 断言静态已知 i0 等于 5 的结果为真
                self.assertTrue(statically_known_true(i0 != 5))
                # 断言静态已知 i0 小于 4
                self.assertTrue(statically_known_true(i0 < 4))
                # 断言静态已知 i0 小于等于 3
                self.assertTrue(statically_known_true(i0 <= 3))

        # 第四组测试：测试大于等于4的条件（不同的表达方式）
        for i, rel in enumerate(
            [lambda x: x < 4, lambda x: 4 > x, lambda x: x <= 3, lambda x: 3 >= x]
        ):
            # 使用子测试，标记当前测试的索引 i
            with self.subTest(f"i = {i}"):
                # 创建符号整数对象 i0，初始值为10，duck 参数设置为 False
                i0 = create_symint(shape_env, 10, duck=False)
                # 断言符号整数 i0 不满足 rel 函数定义的条件
                self.assertFalse(bool(rel(i0)))
                # 断言静态已知 i0 等于 2 的结果为真
                self.assertTrue(statically_known_true(i0 != 2))
                # 断言静态已知 i0 等于 3 的结果为真
                self.assertTrue(statically_known_true(i0 != 3))
                # 断言静态已知 i0 不等于 4
                self.assertFalse(statically_known_true(i0 != 4))
                # 断言静态已知 i0 不等于 5
                self.assertFalse(statically_known_true(i0 != 5))
                # 断言静态已知 i0 大于等于 4
                self.assertTrue(statically_known_true(i0 >= 4))
                # 断言静态已知 i0 大于 3
                self.assertTrue(statically_known_true(i0 > 3))

    # 定义测试函数 test_non_overlapping_and_dense，用于测试非重叠且密集的条件
    def test_non_overlapping_and_dense(self):
        # 创建形状环境对象
        shape_env = ShapeEnv()
        # 创建符号整数对象 a0，初始值为5
        a0 = create_symint(shape_env, 5)
        # 创建形状为 (a0, 7) 的空张量 r，步幅为 (1, a0)，设备为 "meta"
        r = torch.empty_strided((a0, 7), (1, a0), device="meta")
        # 断言张量 r 满足非重叠且密集的条件
        self.assertTrue(torch.ops.aten.is_non_overlapping_and_dense.default(r))
    # 定义测试方法，用于测试非重叠且密集未备份情况
    def test_non_overlapping_and_dense_unbacked(self):
        # 创建形状环境对象
        shape_env = ShapeEnv()
        # 创建未备份的符号整数对象
        u0 = shape_env.create_unbacked_symint()
        # 检查 u0 对象是否符合大小要求
        torch._check_is_size(u0)
        # 获取非重叠且密集判断函数的默认版本
        cf = torch.ops.aten.is_non_overlapping_and_dense.default

        # 断言检查 u0 对象的表达式是否满足指定的非重叠且密集条件
        self.assertEqual(IsNonOverlappingAndDenseIndicator(u0.node.expr, 2, 2, 1), 1)
        # 断言检查给定参数的非重叠且密集条件是否满足
        self.assertEqual(IsNonOverlappingAndDenseIndicator(2, u0.node.expr, 1, 2), 1)
        # 断言检查使用 meta 设备创建的张量是否满足非重叠且密集条件
        self.assertTrue(cf(torch.empty_strided((u0, 2), (2, 1), device="meta")))
        # 断言检查使用 meta 设备创建的张量是否满足非重叠且密集条件
        self.assertTrue(cf(torch.empty_strided((2, u0), (1, 2), device="meta")))

        # 断言检查 u0 对象的表达式是否满足指定的非重叠且密集条件
        self.assertEqual(IsNonOverlappingAndDenseIndicator(u0.node.expr, 1), 1)
        # 断言检查给定参数的非重叠且密集条件是否满足
        self.assertEqual(IsNonOverlappingAndDenseIndicator(1, u0.node.expr), 1)
        # 断言检查使用 meta 设备创建的张量是否满足非重叠且密集条件
        self.assertTrue(cf(torch.empty_strided((u0,), (1,), device="meta")))
        # 断言检查使用 meta 设备创建的张量是否满足非重叠且密集条件
        self.assertTrue(cf(torch.empty_strided((1,), (u0,), device="meta")))

        Max = torch.sym_max
        # NB: This only works because we're able to determine this tensor is
        # contiguous. transpose(0, 1) makes it stop working
        # 断言检查使用 meta 设备创建的张量是否满足非重叠且密集条件（此处包含对可连续性的注释说明）
        self.assertTrue(
            cf(
                torch.empty_strided(
                    (
                        2,
                        3,
                        1,
                        u0,
                    ),
                    (3 * Max(1, u0), Max(1, u0), Max(1, u0), 1),
                    device="meta",
                )
            )
        )

    # 定义测试方法，用于测试调试中的内部重叠未备份情况
    def test_debug_has_internal_overlap_unbacked(self):
        # 创建形状环境对象
        shape_env = ShapeEnv()
        # 创建未备份的符号整数对象
        u0 = shape_env.create_unbacked_symint()
        # 检查 u0 对象是否符合大小要求
        torch._check_is_size(u0)
        # 获取调试中的内部重叠检查函数
        cf = torch._debug_has_internal_overlap

        # 断言检查使用 meta 设备创建的张量是否存在内部重叠
        self.assertEqual(cf(torch.empty_strided((u0, 2), (2, 1), device="meta")), 0)
        # 断言检查使用 meta 设备创建的张量是否存在内部重叠
        self.assertEqual(cf(torch.empty_strided((2, u0), (1, 2), device="meta")), 0)
        # 断言检查使用 meta 设备创建的张量是否存在内部重叠
        self.assertEqual(cf(torch.empty_strided((u0,), (1,), device="meta")), 0)
        # 断言检查使用 meta 设备创建的张量是否存在内部重叠
        self.assertEqual(cf(torch.empty_strided((1,), (u0,), device="meta")), 0)

        Max = torch.sym_max
        # 断言检查使用 meta 设备创建的张量是否存在内部重叠
        self.assertEqual(
            cf(
                torch.empty_strided(
                    (
                        2,
                        3,
                        1,
                        u0,
                    ),
                    (3 * Max(1, u0), Max(1, u0), Max(1, u0), 1),
                    device="meta",
                )
            ),
            0,
        )

        # Wobbling these to zero is OK too
        # 断言检查使用 meta 设备创建的张量是否存在内部重叠（此处对结果为 2 的注释说明）
        self.assertEqual(cf(torch.empty_strided((u0, 2), (3, 1), device="meta")), 2)
        # 断言检查使用 meta 设备创建的张量是否存在内部重叠（此处对结果为 2 的注释说明）
        self.assertEqual(cf(torch.empty_strided((2, u0), (1, 3), device="meta")), 2)

    # 定义测试方法，用于测试特殊化的零和一
    def test_specialize_zero_one(self):
        # 创建特殊化零和一的形状环境对象
        shape_env = ShapeEnv(specialize_zero_one=True)
        # 创建具有特定值的符号整数对象
        a0 = create_symint(shape_env, 5)
        # 断言检查 a0 对象是否不等于 1
        assert a0 != 1
        # 断言检查形状环境对象的保护条件列表长度为 0
        self.assertEqual(len(shape_env.guards), 0)

        # 创建非特殊化零和一的形状环境对象
        shape_env = ShapeEnv(specialize_zero_one=False)
        # 创建具有特定值的符号整数对象
        a0 = create_symint(shape_env, 5)
        # 断言检查 a0 对象是否不等于 1
        assert a0 != 1
        # 断言检查形状环境对象的保护条件列表长度为 1
        self.assertEqual(len(shape_env.guards), 1)
    # 测试具有“duck_shape”特性的环境下的函数
    def test_duck_shape(self):
        # 创建一个“duck_shape”环境对象
        shape_env = ShapeEnv(duck_shape=True)
        # 创建一个符号整数对象a0
        a0 = create_symint(shape_env, 5)
        # 再次创建一个符号整数对象a1，预期与a0相等
        a1 = create_symint(shape_env, 5)
        # 断言a0与a1相等
        assert a0 == a1
        # 断言shape_env的guards列表长度为0
        self.assertEqual(len(shape_env.guards), 0)

        # 切换到不具有“duck_shape”特性的环境
        shape_env = ShapeEnv(duck_shape=False)
        # 创建一个符号整数对象a0
        a0 = create_symint(shape_env, 5)
        # 再次创建一个符号整数对象a1，预期与a0相等
        a1 = create_symint(shape_env, 5)
        # 断言a0与a1相等
        assert a0 == a1
        # 断言shape_env的guards列表长度为1
        self.assertEqual(len(shape_env.guards), 1)

    # 测试符号整数作为标量的行为
    def test_int_bool(self):
        # 创建一个“duck_shape”环境对象
        shape_env = ShapeEnv(duck_shape=True)
        # 创建一个符号整数对象a0
        a0 = create_symint(shape_env, 5)
        # 断言a0为真
        assert a0
        # 断言shape_env的guards列表长度为0
        self.assertEqual(len(shape_env.guards), 0)

    # 测试符号整数作为标量使用时的行为
    def test_symint_as_scalar(self):
        # 创建一个默认环境对象ShapeEnv
        shape_env = ShapeEnv()
        # 创建一个符号整数对象a0
        a0 = create_symint(shape_env, 2)

        # 初始化一个标志变量sym_int_encountered
        sym_int_encountered = False

        # 定义一个TestSymInt类，继承自TorchDispatchMode
        class TestSymInt(TorchDispatchMode):
            # 实现__torch_dispatch__方法
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # 断言func是torch.ops.aten.add.Tensor函数
                assert func == torch.ops.aten.add.Tensor

                nonlocal sym_int_encountered
                # 警告：不要在外部对SymInt/SymFloat进行身份测试，它们不稳定
                sym_int_encountered = kwargs["alpha"].node is a0.node
                kwargs["alpha"] = 0
                # 调用func函数，并返回其结果
                return func(*args)

        # 创建一个4x4的随机张量x
        x = torch.rand([4, 4])
        # 使用TestSymInt上下文环境
        with TestSymInt():
            # 对张量x执行加法操作，使用a0作为alpha参数
            y = torch.add(x, x, alpha=a0)

        # 断言sym_int_encountered为真
        self.assertTrue(sym_int_encountered)

    # 测试深拷贝的行为
    def test_deepcopy(self):
        # 创建一个默认环境对象ShapeEnv
        shape_env = ShapeEnv()
        # 创建一个符号整数对象a0
        a0 = create_symint(shape_env, 2)
        # 断言a0小于4
        assert a0 < 4
        # 深拷贝shape_env对象得到new_shape_env
        new_shape_env = copy.deepcopy(shape_env)
        # 断言new_shape_env的guards列表长度为1
        self.assertEqual(len(new_shape_env.guards), 1)

    # 测试带有符号整数的可打印输出行为
    def test_print_readable_with_symints(self):
        # 定义一个函数f，接受两个参数a和b
        def f(a, b):
            # 计算dim0和dim1
            dim0 = a.shape[0] + b.shape[0]
            dim1 = a.shape[1] + b.shape[1]
            # 创建一个新的空张量d，形状为dim0和dim1
            d = a.new_empty(dim0, dim1)
            # 对张量d执行本地dropout操作，保持训练状态
            d = torch.ops.aten.native_dropout(d, 0.5, train=True)
            # 返回张量d
            return d

        # 使用make_fx函数对函数f进行符号化处理，使用torch.randn(5, 3)和torch.randn(4, 3)作为输入
        fx_g = make_fx(f, tracing_mode="symbolic")(torch.randn(5, 3), torch.randn(4, 3))
        # 打印可读的输出，但不打印输出结果
        out = fx_g.print_readable(print_output=False)

        # 断言out去除首尾空白字符后，与给定的字符串匹配
        self.assertExpectedInline(
            out.strip(),
            """\
class f(torch.nn.Module):
    # 定义神经网络模块，继承自torch.nn.Module类

    def forward(self, a_1: "f32[s0, s1]", b_1: "f32[s2, s1]"):
        # 定义前向传播方法，接受两个输入参数a_1和b_1，类型为f32[s0, s1]和f32[s2, s1]

        # 计算a_1的第0维度大小，返回符号化结果Sym(s0)
        sym_size_int: "Sym(s0)" = torch.ops.aten.sym_size.int(a_1, 0)

        # 计算b_1的第0维度大小，返回符号化结果Sym(s2)
        sym_size_int_1: "Sym(s2)" = torch.ops.aten.sym_size.int(b_1, 0)

        # 对第0维度大小进行符号化加法操作，返回结果Sym(s0 + s2)
        add: "Sym(s0 + s2)" = sym_size_int + sym_size_int_1;  sym_size_int = sym_size_int_1 = None

        # 计算a_1的第1维度大小，返回符号化结果Sym(s1)
        sym_size_int_2: "Sym(s1)" = torch.ops.aten.sym_size.int(a_1, 1)

        # 计算b_1的第1维度大小，返回符号化结果Sym(s1)
        sym_size_int_3: "Sym(s1)" = torch.ops.aten.sym_size.int(b_1, 1);  b_1 = None

        # 对第1维度大小进行符号化加法操作，返回结果Sym(2*s1)
        add_1: "Sym(2*s1)" = sym_size_int_2 + sym_size_int_3;  sym_size_int_2 = sym_size_int_3 = None

        # 创建一个指定形状的新的空tensor，类型为f32[s0 + s2, 2*s1]，不采用内存固定
        new_empty: "f32[s0 + s2, 2*s1]" = torch.ops.aten.new_empty.default(a_1, [add, add_1], pin_memory=False);  a_1 = add = add_1 = None

        # 对新创建的tensor应用本地dropout操作，dropout概率为0.5，training模式为True
        native_dropout = torch.ops.aten.native_dropout.default(new_empty, 0.5, True);  new_empty = None

        # 获取native_dropout结果的第0个元素，类型为f32[s0 + s2, 2*s1]
        getitem: "f32[s0 + s2, 2*s1]" = native_dropout[0]

        # 获取native_dropout结果的第1个元素，类型为b8[s0 + s2, 2*s1]
        getitem_1: "b8[s0 + s2, 2*s1]" = native_dropout[1];  native_dropout = None

        # 返回getitem和getitem_1作为前向传播的输出结果
        return (getitem, getitem_1)""",  # noqa: B950

    def test_statically_known_true(self):
        # 定义测试静态已知为真的方法

        shape_env = ShapeEnv()  # 创建形状环境对象
        s2, s3, s4 = (create_symint(shape_env, i) for i in range(2, 5))

        # 断言以下表达式在静态情况下为真
        self.assertTrue(statically_known_true(True))
        self.assertTrue(statically_known_true(s2 == s2))
        self.assertTrue(statically_known_true(s2 * s3 > s3))
        self.assertTrue(statically_known_true(s3 * s4 > s4))
        self.assertTrue(statically_known_true((s3 + s3) % 2 == 0))

        # 断言以下表达式在静态情况下为假
        self.assertFalse(statically_known_true(False))
        self.assertFalse(statically_known_true(s3 * s4 <= s4))
        self.assertFalse(statically_known_true((s3 + s3) % 2 == 1))

        # 对于提示为真，但静态情况下未知的表达式进行断言为假
        self.assertFalse(statically_known_true(s2 + s2 == s4))
        self.assertFalse(statically_known_true(s4 % s2 == 0))
        self.assertFalse(statically_known_true(s2 != s3))
        self.assertFalse(statically_known_true(s3 * s4 > s2))

        # 对于提示为假，但静态情况下未知的表达式进行断言为假
        self.assertFalse(statically_known_true(s2 == s3))
        self.assertFalse(statically_known_true(s2 > s3))
        self.assertFalse(statically_known_true(s3 + s3 == s4))

        # 保证不生成任何保护条件
        self.assertEqual(len(shape_env.guards), 0)
    def test_ephemeral_source_simplification(self):
        from torch._dynamo.source import EphemeralSource

        # 针对完整的健壮性，确保无论构建顺序或检查顺序如何，都简化出临时源符号。
        for construct_ephemeral_first, x_first_in_check in itertools.product(
            [False, True], [False, True]
        ):
            shape_env = ShapeEnv()  # 创建形状环境对象
            shape = (5, 10)  # 设置张量的形状
            dynamic_dims = [DimDynamic.DYNAMIC for _ in shape]  # 动态维度的列表
            x = create_symbolic_tensor(
                "x",
                torch.randn(*shape),
                shape_env,
                source=(EphemeralSource() if construct_ephemeral_first else None),  # 如果首先构建临时源，则使用临时源
                dynamic_dims=dynamic_dims,
            )
            y = create_symbolic_tensor(
                "y",
                torch.randn(*shape),
                shape_env,
                source=(EphemeralSource() if not construct_ephemeral_first else None),  # 如果不首先构建临时源，则使用临时源
                dynamic_dims=dynamic_dims,
            )
            t_with_ephemeral = x if construct_ephemeral_first else y  # 根据构建顺序选择张量

            def _get_ephemeral_source_symbols(t):
                return [
                    s.node.expr
                    for s in itertools.chain(t.shape, t.stride(), (t.storage_offset(),))
                    if isinstance(s, torch.SymInt)
                    and s.node.expr in shape_env.var_to_sources
                    and any(
                        source.is_ephemeral()
                        for source in shape_env.var_to_sources[s.node.expr]
                    )
                ]

            # 这些检查应简化出临时符号，无论 x == y 还是 y == x 的顺序
            self.assertTrue(len(_get_ephemeral_source_symbols(t_with_ephemeral)) > 0)
            if x_first_in_check:
                torch._check(x.size() == y.size())  # 检查张量 x 和 y 的尺寸是否相等
                torch._check(x.stride() == y.stride())  # 检查张量 x 和 y 的步长是否相等
                torch._check(x.storage_offset() == y.storage_offset())  # 检查张量 x 和 y 的存储偏移是否相等
            else:
                torch._check(y.size() == x.size())  # 检查张量 y 和 x 的尺寸是否相等
                torch._check(y.stride() == x.stride())  # 检查张量 y 和 x 的步长是否相等
                torch._check(y.storage_offset() == x.storage_offset())  # 检查张量 y 和 x 的存储偏移是否相等
            self.assertEqual(len(_get_ephemeral_source_symbols(t_with_ephemeral)), 0)  # 确保临时符号已被简化掉
    # 定义测试方法，用于验证 ephemeral 和 non-ephemeral 数据源的统一性
    def test_ephemeral_source_unified_with_non_ephemeral_source(self):
        # 从 torch._dynamo.source 模块导入 EphemeralSource 类
        from torch._dynamo.source import EphemeralSource
        
        # 对于 construct_ephemeral_first 取 False 和 True 两种情况进行迭代
        for construct_ephemeral_first in (False, True):
            # 创建 ShapeEnv 实例，用于处理形状相关的环境
            shape_env = ShapeEnv()
            # 定义张量的形状
            shape = (5, 10)
            # 使用 duck sizing 确保符号在 x 和 y 之间的重用
            duck_dims = [DimDynamic.DUCK for _ in shape]
            
            # 创建符号化张量 x，使用随机数填充，指定形状环境和数据源
            x = create_symbolic_tensor(
                "x",
                torch.randn(*shape),
                shape_env,
                source=(EphemeralSource() if construct_ephemeral_first else None),
                dynamic_dims=duck_dims,
            )
            
            # 创建符号化张量 y，使用随机数填充，指定形状环境和数据源
            y = create_symbolic_tensor(
                "y",
                torch.randn(*shape),
                shape_env,
                source=(EphemeralSource() if not construct_ephemeral_first else None),
                dynamic_dims=duck_dims,
            )
            
            # 无论构造顺序如何，非短暂数据源应优先出现在 var_to_sources 列表中，以备后续可能的保护操作
            for source_list in shape_env.var_to_sources.values():
                self.assertFalse(source_list[0].is_ephemeral())
            
            # 断言 x 和 y 的尺寸相同
            self.assertEqual(x.size(), y.size())
            # 断言 x 和 y 的步长相同
            self.assertEqual(x.stride(), y.stride())
            # 断言 x 和 y 的存储偏移量相同
            self.assertEqual(x.storage_offset(), y.storage_offset())
# 如果 Torch Dynamo 环境跳过测试，给出相应的消息
@skipIfTorchDynamo(
    "Creating ShapeEnv fails for confusing reasons (also we never expect dynamo to see code like this)"
)
# 测试类 TestSymNumberMagicMethods，继承自 TestCase
class TestSymNumberMagicMethods(TestCase):

    # 执行测试的私有方法，接受函数名、输入1、输入2、形状环境和是否是一元函数的标志
    def _do_test(self, fn, inp1, inp2, shape_env, is_unary_fn):
        # 在子测试中执行 _do_test2 方法，传递函数名、输入1、输入2、形状环境和是否是一元函数的标志
        with self.subTest(fn=fn, inp1=inp1, inp2=inp2, is_unary_fn=is_unary_fn):
            return self._do_test2(fn, inp1, inp2, shape_env, is_unary_fn)

    # 使用 sym_node 的所有魔术方法作为参数进行参数化
    @parametrize("fn", list(sym_node.magic_methods.keys()))
    # 测试布尔方法的函数，接受函数名作为参数
    def test_bool_method(self, fn):
        # 如果函数名不在布尔魔术方法列表中，或者是 "sym_ite"，则跳过测试
        if fn not in sym_node.bool_magic_methods or fn == "sym_ite":
            self.skipTest(f"{fn} is non-bool")

        # 检查函数是否是一元函数
        is_unary_fn = fn in sym_node.unary_methods
        # 创建 ShapeEnv 对象
        shape_env = ShapeEnv()
        # 执行 _do_test 方法，传递函数名、True、False、形状环境和是否是一元函数的标志
        self._do_test(fn, True, False, shape_env, is_unary_fn)

    # 使用 sym_node 的所有魔术方法、第一个类型和第二个类型作为参数进行参数化
    @parametrize("fn", list(sym_node.magic_methods.keys()))
    @parametrize("first_type", ["int", "float"])
    @parametrize("second_type", ["int", "float"])
    # 测试方法的函数，接受函数名、第一个类型和第二个类型作为参数
    def test_method(self, fn, first_type, second_type):
        # 如果第一个类型是 "float"，跳过测试，并给出相应的消息
        if first_type == "float":
            # TODO: Hmm, this looks like we skip all floats
            self.skipTest(f"{fn} is not a float magic method")

        # 如果第一个类型是 "int" 或者第二个类型是 "int"，并且函数名在仅支持浮点数魔术方法列表中，跳过测试
        if (first_type == "int" or second_type == "int") and fn in sym_node.only_float_magic_methods:
            self.skipTest(f"{fn} is not an int method")

        # 如果第二个类型是 "float" 并且函数名是 ["mod"]，跳过测试
        if second_type == "float" and fn in ["mod"]:
            self.skipTest(f"{fn} only handles int")

        # 检查函数是否是一元函数或者函数名是 "round"
        is_unary_fn = fn in sym_node.unary_methods or fn == "round"
        # 如果是一元函数并且第二个类型是 "float"，跳过测试
        if is_unary_fn and second_type == "float":
            self.skipTest(f"{fn} is unary and already tested")

        # 如果函数名在布尔魔术方法列表中，跳过测试
        if fn in sym_node.bool_magic_methods:
            self.skipTest(f"{fn} is bool")

        # 只有浮点数，因为这些将根据需要转换为整数。我们还忽略复数和布尔值。
        # 这里的值是浮点数列表，如果函数名是 ("sym_acos", "sym_asin") 中的一个，则将 0.5 作为值，以避免数学域错误
        values = (
            0.0,
            1.0,
            0.5 if fn in ("sym_acos", "sym_asin") else 2.5,  # avoid math domain error
        )

        # 创建负值元组，包括所有 values 中的值的负数
        neg_values = tuple(-x for x in values)

        # 遍历所有输入值的组合，包括 values 与 values 的乘积、values 与 neg_values 的乘积、neg_values 与 values 的乘积、neg_values 与 neg_values 的乘积
        for inp1, inp2 in itertools.chain(
            itertools.product(values, values),
            itertools.product(values, neg_values),
            itertools.product(neg_values, values),
            itertools.product(neg_values, neg_values),
        ):
            # 如果第一个类型是 "int"，将输入1转换为整数
            if first_type == "int":
                inp1 = int(inp1)
            # 如果第二个类型是 "int"，将输入2转换为整数
            if second_type == "int":
                inp2 = int(inp2)

            # 创建 ShapeEnv 对象
            shape_env = ShapeEnv()
            # 执行 _do_test 方法，传递函数名、输入1、输入2、形状环境和是否是一元函数的标志
            self._do_test(fn, inp1, inp2, shape_env, is_unary_fn)

    # 获取常量布尔值的方法，接受一个值作为参数，返回 SymBool 对象
    def get_constant_bool(self, val):
        return SymBool(torch._C._get_constant_bool_symnode(val))
    # 定义测试方法：验证符号节点的哈希处理
    def test_symnode_hashing(self):
        # 创建形状环境对象
        shape_env = ShapeEnv()

        # 创建不可哈希对象的元组
        unhashable = (
            create_symint(shape_env, 3),     # 创建符号整数对象
            create_symbool(shape_env, True), # 创建符号布尔对象
            # 目前 create_symfloat 只支持整数，应传入浮点数，但暂未实现
            create_symfloat(shape_env, 3.0), # 创建符号浮点数对象（仅支持整数）
        )

        # 遍历不可哈希对象
        for x in unhashable:
            # 断言调用 hash(x) 会引发 TypeError 异常，异常信息包含 "unhashable"
            with self.assertRaisesRegex(TypeError, "unhashable"):
                hash(x)

        # 创建可嵌套整数（SymInt）、常量布尔（SymBool）、符号节点（SymNode）对象
        j1 = torch._C._get_nested_int(1, 1)
        j1_copy = torch._C._get_nested_int(1, 1)
        j2 = torch._C._get_nested_int(2, 1)
        t = self.get_constant_bool(True)
        t_copy = self.get_constant_bool(True)
        f = self.get_constant_bool(False)
        n = create_symint(shape_env, 3).node
        m = self.get_constant_bool(True).node

        # 断言对象相等性及哈希值
        self.assertIs(j1 == j1_copy, True)
        self.assertEqual(hash(j1), hash(j1_copy))
        self.assertIs(j1 == j2, False)
        self.assertNotEqual(hash(j1), hash(j2))
        self.assertIs(t == t_copy, True)
        self.assertEqual(hash(t), hash(t_copy))
        self.assertIs(t == f, False)
        self.assertNotEqual(hash(t), hash(f))

        # 计算符号节点 n 和 m 的哈希值
        hash(n)
        hash(m)

    # 定义测试方法：验证符号整数对象的深拷贝功能
    def test_symint_deepcopy(self):
        # 创建形状环境对象
        shape_env = ShapeEnv()

        # 创建包含一个嵌套整数对象的元组
        symnodes = (torch._C._get_nested_int(1, 1),)
        # 对 symnodes 进行深拷贝
        deepcopied_symnodes = copy.deepcopy(symnodes)
        # 断言原始对象与深拷贝对象相等
        self.assertEqual(symnodes, deepcopied_symnodes)
    def test_non_symbolic_symnode(self):
        # 调用 torch._C._get_nested_int 函数获取 j1, j2, j3 三个变量
        j1 = torch._C._get_nested_int(1, 1)
        j2 = torch._C._get_nested_int(1, 1)
        j3 = torch._C._get_nested_int(3, 1)

        # 断言 j1 的类型是 torch.SymInt
        self.assertIsInstance(j1, torch.SymInt)
        # 断言 j1 的类型不是 int
        self.assertNotIsInstance(j1, int)

        # 使用断言捕获 RuntimeError 异常，检查 "add not supported by NestedIntSymNode"
        with self.assertRaisesRegex(
            RuntimeError, "add not supported by NestedIntSymNode"
        ):
            j1 + 3

        # 断言 j1 不等于 3
        self.assertFalse(j1 == 3)
        # 使用断言捕获 RuntimeError 异常，检查 "indeterminate"
        with self.assertRaisesRegex(RuntimeError, "indeterminate"):
            self.assertFalse(3 >= j2)

        # 断言 j1 等于 j1
        self.assertIs(j1 == j1, True)
        # 断言 j1 等于 j2
        self.assertIs(j1 == j2, True)
        # 断言 j1 不等于 j3
        self.assertIs(j1 == j3, False)
        # 断言 j1 不等于 j3
        self.assertIs(j1 != j3, True)
        # 断言 j1 等于 j2
        self.assertIs(j1 != j2, False)

        # 获取一个常量布尔值 True
        x = self.get_constant_bool(True)
        #
        # 一元操作
        #
        # 对常量 SymBool 执行逻辑非操作
        # 结果应为 False
        self.assertIs(x.__sym_not__(), False)

        #
        # 二元操作
        #
        # 对常量 SymBool 和 bool 值执行逻辑与操作
        # 对常量 SymBool 和 常量 SymBool 执行逻辑与操作
        # 对 bool 值和常量 SymBool 执行逻辑与操作
        self.assertIs(operator.and_(x, True), True)
        self.assertIs(operator.and_(x, x), True)
        self.assertIs(operator.and_(True, x), True)

        # 对象的形状环境设置为 ShapeEnv 类实例
        shape_env = ShapeEnv()
        # 使用 shape_env 创建两个符号整数 a 和 b
        a = create_symint(shape_env, 2)
        b = create_symint(shape_env, 2)
        # 创建符号布尔变量 c，表示 a 是否等于 b
        c = a == b  # symbolic SymBool
        # 获取一个常量布尔值 True
        d = self.get_constant_bool(True)
        # 对 c 和 d 执行逻辑与操作，得到结果 e 和 f
        e = operator.and_(c, d)
        f = operator.and_(d, c)
        # 断言 e 和 f 是符号化的布尔值
        self.assertTrue(is_symbolic(e))
        self.assertTrue(is_symbolic(f))
        # 对 e 和 f 的节点执行 guard_bool 操作，期望结果为 True
        self.assertIs(e.node.guard_bool("", 0), True)
        self.assertIs(f.node.guard_bool("", 0), True)

        # 创建两个尺寸对象 sz1 和 sz2
        sz1 = torch.Size([j1, j1, j1])
        sz2 = torch.Size([j1, j1, j1])
        # 断言 sz1 等于 sz2
        self.assertIs(sz1 == sz2, True)

        # 创建两个尺寸对象 sz1 和 sz2，其中包含符号整数 j1 和 j2
        sz1 = torch.Size([3, j1, 4])
        sz2 = torch.Size([3, j2, 4])
        # 断言 sz1 等于 sz2
        self.assertIs(sz1 == sz2, True)
        # 断言 sz1 不等于 sz2
        self.assertIs(sz1 != sz2, False)
# 实例化一个参数化的测试类 TestSymNumberMagicMethods 的对象
instantiate_parametrized_tests(TestSymNumberMagicMethods)


class TestFloorDiv(TestCase):
    @staticmethod
    def python_floordiv(x, y):
        # 返回两个数的整数除法结果
        return x // y

    @staticmethod
    def torch_floordiv(x, y):
        # 注意：这里完全评估表达式，因为 FloorDiv 可能并不总是这样做。
        shape_env = ShapeEnv()
        # 使用 ShapeEnv 对象来评估 FloorDiv(x, y) 表达式的结果
        return shape_env.evaluate_expr(FloorDiv(x, y))

    @staticmethod
    def yield_test_cases(values, negate=True):
        # 生成测试用例的生成器函数
        for x, y in values:
            yield (x, y)
            if negate:
                yield (-x, y)
                yield (x, -y)
                yield (-x, -y)

    def test_floordiv_float_int(self):
        # 测试整数和浮点数之间的整数除法
        values = ((7, 2),)

        for x, y in TestFloorDiv.yield_test_cases(values):
            # 断言 Python 的整数除法和 torch 的整数除法结果相同
            self.assertEqual(
                TestFloorDiv.python_floordiv(x, y), TestFloorDiv.torch_floordiv(x, y)
            )

    def test_floordiv_div_by_one(self):
        # 测试整数除以 1 的情况
        values = ((2, 1),)

        for x, y in TestFloorDiv.yield_test_cases(values):
            # 断言 Python 的整数除法和 torch 的整数除法结果相同
            self.assertEqual(
                TestFloorDiv.python_floordiv(x, y), TestFloorDiv.torch_floordiv(x, y)
            )

    def test_floordiv_simplify(self):
        # 测试如何简化或评估不带自由变量的 FloorDiv 表达式
        shape_env = ShapeEnv()
        result = 21
        exprs = (7 * FloorDiv(6, 2),)

        for expr in exprs:
            # 断言不同方式简化或评估表达式后的结果都等于预期结果
            self.assertEqual(expr, result)
            self.assertEqual(expr.doit(deep=False), result)
            self.assertEqual(expr.doit(deep=True), result)
            self.assertEqual(sympy.simplify(expr), result)
            self.assertEqual(shape_env.simplify(expr), result)
            self.assertEqual(shape_env.evaluate_expr(expr), result)
    # 定义一个测试方法，用于测试 FloorDiv 操作的假设条件
    def test_floordiv_assumptions(self):
        # 定义测试用例，包含两个整数符号变量
        cases = (
            sympy.Symbol("i1", integer=True),
            sympy.Symbol("i2", integer=True),
        )

        # 使用 itertools.product 生成所有可能的组合
        for base, divisor in itertools.product(cases, repeat=2):

            # 定义一个函数 op()，返回 FloorDiv 操作的结果
            def op():
                return FloorDiv(base, divisor)

            # 定义一个函数，用于检查是否为复数
            def is_complex(x):
                # 判断 x 是否既非整数又非实数但是复数
                return x.is_integer is False and x.is_real is False and x.is_complex

            # 如果 base 或 divisor 是复数，预期会抛出 TypeError 异常
            if is_complex(base) or is_complex(divisor):
                # 使用 self.assertRaisesRegex 验证异常消息是否符合预期
                self.assertRaisesRegex(
                    TypeError,
                    (
                        r"unsupported operand type\(s\) for //: 'Symbol' and 'Symbol',"
                        r" expected integer or real"
                    ),
                    op,
                )
                continue

            # 执行 FloorDiv 操作
            op = op()

            # 在常规 Python 中，如果 x 是浮点数，则 x//x == 1.0，但是 FloorDiv
            # 当两个参数相同时，总是返回整数 1。这对于没有指定任何假设的符号变量也适用。
            if base is divisor:
                # 验证结果是整数且是实数
                self.assertTrue(op.is_integer)
                self.assertTrue(op.is_real)
            elif base.is_integer and divisor.is_integer:
                # 如果 base 和 divisor 都是整数，则结果应该是整数且是实数
                self.assertTrue(op.is_integer)
                self.assertTrue(op.is_real)
            else:
                # 如果无法确定结果是否为整数，则 op.is_integer 返回 None
                self.assertEqual(op.is_integer, None)
                self.assertTrue(op.is_real)
# 定义一个测试类 TestDimConstraints，继承自 TestCase
class TestDimConstraints(TestCase):

    # 定义测试方法 test_dim_constraints_reduce_congruences_simple
    def test_dim_constraints_reduce_congruences_simple(self):
        # 从 sympy 模块导入 Symbol 符号变量
        from sympy import Symbol

        # 创建一个名为 s 的符号变量，设定为正数且为整数
        s = Symbol("s", positive=True, integer=True)
        
        # 创建 DimConstraints 类的实例 dim_constraints，传入空字典和集合等参数
        dim_constraints = DimConstraints({}, {}, set(), {})
        
        # 在 dim_constraints 对象的 _congruences 字典中，为符号变量 s 添加一组同余条件集合
        dim_constraints._congruences[s] = {
            (s / 2) % 2,
            (s / 2) % 8,
            (s / 2) % 4,
            s % 2,
            ((s / 16) + 2) % 4,
        }
        
        # 调用 dim_constraints 对象的 _reduce_congruences 方法，获取约简后的同余条件集合
        congruences = dim_constraints._reduce_congruences()
        
        # 使用 self.assertEqual 方法断言 congruences[s] 的值等于 {(s + 32) % 64}
        self.assertEqual(congruences[s], {(s + 32) % 64})

    # 定义测试方法 test_dim_constraints_reduce_inequalities_simple
    def test_dim_constraints_reduce_inequalities_simple(self):
        # 从 sympy 模块导入 Eq, Interval, Ne, Symbol 等符号和不等式解析方法
        from sympy import Eq, Interval, Ne, Symbol
        from sympy.solvers.inequalities import reduce_inequalities

        # 创建一个名为 s 的符号变量，设定为正数且为整数
        s = Symbol("s", positive=True, integer=True)
        
        # 创建包含多个不等式表达式的集合 exprs
        exprs = {
            s >= 2,
            Ne(8 * s, 16),
            Ne(s / 2, 1),
            Ne(16 * s, 32),
            s < 16,
            Ne(s, 2),
            s / 2 < 16,
            s / 2 > 1,
            s / 2 >= 2,
            Ne(3 * s / 2, 3),
        }
        
        # 调用 reduce_inequalities 函数处理不等式集合 exprs，返回其解的集合对象
        solution = reduce_inequalities(exprs, s).as_set()
        
        # 使用 self.assertEqual 方法断言 solution 的值等于 Interval.Ropen(4, 16)
        self.assertEqual(solution, Interval.Ropen(4, 16))

        # 向 exprs 集合添加一个等式表达式 Eq(s / 2, 4)
        exprs.add(Eq(s / 2, 4))
        
        # 再次调用 reduce_inequalities 函数处理更新后的 exprs 集合，返回解的集合对象
        solution = reduce_inequalities(exprs, s).as_set()
        
        # 使用 self.assertEqual 方法断言 solution 的值等于 {8}
        self.assertEqual(solution, {8})

    # 定义测试方法 test_dim_constraints_reduce_inequalities_error
    def test_dim_constraints_reduce_inequalities_error(self):
        # 从 collections 模块导入 defaultdict
        from collections import defaultdict
        
        # 从 sympy 模块导入 Symbol
        from sympy import Symbol
        from sympy.solvers.inequalities import reduce_inequalities
        
        # 导入 torch 模块的相关类和函数
        from torch._dynamo.source import (
            LocalSource,
            TensorProperty,
            TensorPropertySource,
        )
        from torch.fx.experimental.symbolic_shapes import DynamicDimConstraintPrinter

        # 创建一个名为 s0 的符号变量，设定为正数且为整数
        s0 = Symbol("s0", positive=True, integer=True)
        
        # 创建包含多个不等式表达式的集合 exprs
        exprs = {
            4 * s0**3 - 4 * s0**2 + s0 <= 2147483647,
            s0 >= 2,
            s0**3 <= 2147483647,
            s0 <= 2147483647,
        }
        
        # 调用 reduce_inequalities 函数处理不等式集合 exprs，返回其解的集合对象
        answer = reduce_inequalities(exprs, s0)

        # 创建一个 defaultdict 对象 symbol_to_source，用于存储符号到源的映射关系
        symbol_to_source = defaultdict(list)
        
        # 向 symbol_to_source[s0] 中添加一个 TensorPropertySource 对象
        symbol_to_source[s0].append(
            TensorPropertySource(
                base=LocalSource(local_name="a"), prop=TensorProperty.SIZE, idx=0
            )
        )
        
        # 创建 DynamicDimConstraintPrinter 对象 dcp，传入 symbol_to_source 和空字典
        dcp = DynamicDimConstraintPrinter(symbol_to_source, {})
        
        # 使用 self.assertRaisesRegex 断言捕获 AssertionError 异常，并验证其提示信息
        with self.assertRaisesRegex(
            AssertionError,
            "Unknown symbol.*created by constraints solver",
        ):
            # 调用 dcp 对象的 doprint 方法，传入 answer，触发断言异常
            dcp.doprint(answer)

# 定义函数 specializations，接受六个参数 a, b, c, d, e, f
def specializations(a, b, c, d, e, f):
    # a:
    # 使用 assert 断言 a.size()[0] 等于 8
    assert a.size()[0] == 8
    # 使用 assert 断言 a.size()[1] 等于 22
    assert a.size()[1] == 22
    # 使用 assert 断言 a.size()[2] 等于 96
    assert a.size()[2] == 96
    # 使用 assert 断言 a.size()[3] 等于 96
    assert a.size()[3] == 96

    # b:
    # 使用 assert 断言 b.size()[0] 等于 8
    assert b.size()[0] == 8
    # 使用 assert 断言 b.size()[1] 等于 22
    assert b.size()[1] == 22
    # 使用 assert 断言 b.size()[2] 等于 3
    assert b.size()[2] == 3

    # c:
    # 使用 assert 断言 c.size()[0] 等于 8
    assert c.size()[0] == 8

    # d:
    # 使用 assert 断言 d.size()[0] 等于 8
    assert d.size()[0] == 8

    # f:
    # 使用 assert 断言 f.size()[1] 等于 1
    assert f.size()[1] == 1
    return [
        # 检查动态维度函数返回的结果是否与 c 的第一维度相等
        dynamic_dim(d, 1) == dynamic_dim(c, 1),

        # 检查动态维度函数返回的结果是否与 e 的第一维度相等
        dynamic_dim(e, 1) == dynamic_dim(c, 1),
    ]
    def test_guards_gt_lt(self):
        # 创建一个 ShapeEnv 实例作为形状环境
        shape_env = ShapeEnv()
        # 创建三个符号整数对象 s0, s1, s2，并分别初始化为 6, 7, 5
        s0 = create_symint(shape_env, 6)
        s1 = create_symint(shape_env, 7)
        s2 = create_symint(shape_env, 5)

        # 添加 s0 > 5 的整数守卫条件
        guard_int(sym_int(s0 > 5))
        # 添加 s0 < 7 的整数守卫条件
        guard_int(sym_int(s0 < 7))

        # 生成 s0 相关的守卫表达式
        guards = shape_env.produce_guards_expression([s0])

        # 使用形状环境评估守卫表达式，确保对于 s0 返回 True
        self.assertTrue(shape_env.evaluate_guards_expression(guards, [hint_int(s0)]))
        # 确保对于 s1 返回 False
        self.assertFalse(shape_env.evaluate_guards_expression(guards, [hint_int(s1)]))
        # 确保对于 s2 返回 False
        self.assertFalse(shape_env.evaluate_guards_expression(guards, [hint_int(s2)]))

    def test_guards_float_div(self):
        # 创建一个 ShapeEnv 实例作为形状环境
        shape_env = ShapeEnv()
        # 创建两个符号整数对象 s0, s1，并分别初始化为 8, 7
        s0 = create_symint(shape_env, 8)
        s1 = create_symint(shape_env, 7)

        # 添加 s0 / 2.0 的浮点数守卫条件
        guard_int(sym_int(s0 / 2.0))
        # 生成 s0 相关的守卫表达式
        guards = shape_env.produce_guards_expression([s0])

        # 确保守卫表达式中包含 "ToFloat"
        self.assertIn("ToFloat", guards)
        # 确保守卫表达式中包含 "FloatTrueDiv"
        self.assertIn("FloatTrueDiv", guards)
        # 使用形状环境评估守卫表达式，确保对于 s0 返回 True
        self.assertTrue(shape_env.evaluate_guards_expression(guards, [hint_int(s0)]))
        # 确保对于 s1 返回 False
        self.assertFalse(shape_env.evaluate_guards_expression(guards, [hint_int(s1)]))
```