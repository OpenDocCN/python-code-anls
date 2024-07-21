# `.\pytorch\torch\_inductor\graph.py`

```py
# 设置 mypy 来允许未类型化的函数定义
# 导入必要的模块和库
import functools  # 提供高阶函数操作
import itertools  # 提供迭代工具函数
import logging  # 提供日志记录功能
import operator  # 提供标准的运算符函数
import os  # 提供与操作系统交互的功能
import re  # 提供正则表达式匹配操作
import sys  # 提供与Python解释器交互的功能
import time  # 提供时间相关操作的功能
from collections import defaultdict  # 提供默认字典功能
from contextlib import contextmanager  # 提供上下文管理工具
from typing import (  # 提供类型提示支持
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import sympy  # 提供符号数学计算功能

import torch  # 提供深度学习框架功能
import torch._logging  # Torch的内部日志记录功能
import torch.fx  # Torch的函数式编程工具
from torch._decomp import get_decompositions  # 获取张量分解功能
from torch._dynamo.utils import defake, dynamo_timed  # 提供工具函数和时间测量装饰器
from torch._logging import LazyString, trace_structured  # 提供懒字符串和结构化跟踪日志功能
from torch._prims_common import make_channels_last_strides_for  # 提供通用的通道相关操作函数
from torch._subclasses.fake_tensor import FakeTensor  # 提供虚拟张量支持
from torch.fx.experimental._backward_state import BackwardState  # 提供反向状态管理工具
from torch.fx.experimental.sym_node import magic_methods, method_to_operator  # 提供符号节点功能
from torch.fx.experimental.symbolic_shapes import (  # 提供符号形状相关操作
    free_unbacked_symbols,
    has_free_symbols,
    resolve_unbacked_bindings,
    RuntimeAssert,
    ShapeEnv,
    SymTypes,
)
from torch.utils._mode_utils import no_dispatch  # 提供模式相关工具
from torch.utils._sympy.numbers import int_oo  # 提供无穷大符号

from . import config, ir  # 导入自定义模块和库
from .codegen.common import (  # 从代码生成模块导入通用函数
    BackendFeature,
    DeviceOpOverrides,
    get_backend_features,
    get_device_op_overrides,
    get_wrapper_codegen_for_device,
    init_backend_registration,
)
from .exc import (  # 导入自定义异常类
    CppWrapperCodeGenError,
    LoweringException,
    MissingOperatorWithDecomp,
    MissingOperatorWithoutDecomp,
)
from .ir import (  # 导入中间表示(IR)相关类和函数
    Constant,
    FixedLayout,
    get_device_type,
    InputBuffer,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
    TorchBindObject,
)
from .lowering import (  # 导入降低(IR到代码)相关函数和类
    constrain_to_fx_strides,
    FALLBACK_ALLOW_LIST,
    fallback_handler,
    fallback_node_due_to_unsupported_type,
    layout_constraints,
    lowerings,
    make_fallback,
    needs_realized_inputs,
    unsupported_output_tensor,
)
from .sizevars import SizeVarAllocator  # 导入大小变量分配器
from .utils import (  # 导入实用工具函数
    convert_shape_to_inductor,
    gather_origins,
    get_cloned_parameter_buffer_name,
    get_sympy_Expr_dtype,
    maybe_get_suppress_shape_guards_ctx,
    should_assume_input_aligned,
)
from .virtualized import NullHandler, V  # 导入虚拟化相关功能

if TYPE_CHECKING:
    from torch._higher_order_ops.effects import _EffectType  # 导入类型检查相关模块
    from .codegen.wrapper import WrapperCodeGen  # 导入代码生成包装器类

from torch._inductor.codecache import output_code_log  # 导入代码缓存输出日志

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")  # 获取性能提示相关的日志记录器对象

aten = torch.ops.aten  # 导入Torch操作模块的命名空间

_post_grad_graph_counter = itertools.count()  # 创建一个无限计数器用于后向梯度图

if config.is_fbcode():  # 如果运行在Facebook代码环境中
    from torch._inductor.fb.utils import log_module_code  # 导入Facebook特定的模块代码日志记录工具
else:
    # 定义一个占位符函数，用于在非Facebook环境中忽略模块代码的日志记录
    def log_module_code(*args, **kwargs):
        pass


def supported_dtype_of_cpp_wrapper(dtype, cuda):
    # 定义一个集合，包含所有支持的张量数据类型
    supported_dtype = {
        torch.float32,         # 32位浮点数
        torch.float64,         # 64位浮点数
        torch.int64,           # 64位有符号整数
        torch.int32,           # 32位有符号整数
        torch.int16,           # 16位有符号整数
        torch.int8,            # 8位有符号整数
        torch.uint8,           # 8位无符号整数
        torch.bool,            # 布尔类型
        torch.bfloat16,        # 16位浮点数（截断的浮点数）
        torch.complex32,       # 32位复数
        torch.complex64,       # 64位复数
        torch.complex128,      # 128位复数
        torch.float16,         # 16位浮点数
    }
    
    # 如果启用了 CUDA，添加额外的支持的张量数据类型
    if cuda:
        supported_dtype.add(torch.float8_e4m3fn)     # 自定义的8位浮点数格式1
        supported_dtype.add(torch.float8_e5m2)       # 自定义的8位浮点数格式2
        supported_dtype.add(torch.float8_e4m3fnuz)   # 自定义的8位浮点数格式3
        supported_dtype.add(torch.float8_e5m2fnuz)   # 自定义的8位浮点数格式4
    
    # 检查给定的数据类型是否在支持的数据类型集合中
    return dtype in supported_dtype
def may_get_constant_buffer_dtype(constant_buffer):
    # 检查 constant_buffer 是否是 sympy.Symbol, sympy.Expr 或 sympy.core.numbers.Integer 的实例
    assert isinstance(
        constant_buffer, (sympy.Symbol, sympy.Expr, sympy.core.numbers.Integer)
    ), "get_constant_buffer_dtype only supports input of sympy.Symbol, sympy.Expr or sympy.core.numbers.Integer"
    
    # 如果 constant_buffer 是 sympy.core.numbers.Integer 类型，则返回 torch.int64
    if isinstance(constant_buffer, sympy.core.numbers.Integer):
        return torch.int64
    
    # 如果 constant_buffer 是 sympy.Expr 类型，则调用 get_sympy_Expr_dtype 处理返回结果
    if isinstance(constant_buffer, sympy.Expr):
        return get_sympy_Expr_dtype(constant_buffer)
    
    # 如果 constant_buffer 是整数类型，则返回 torch.int64
    if constant_buffer.is_integer:
        return torch.int64
    # 如果 constant_buffer 是浮点数类型，则返回 torch.float32
    elif constant_buffer.is_float:
        return torch.float32
    else:
        # 否则返回 None
        return None


def is_magic_method(op):
    # 构建包含所有魔术方法的集合 magic_ops
    magic_ops = {method_to_operator(m) for m in magic_methods}
    # 检查 op 是否在 magic_ops 中
    return op in magic_ops


def getattr_recursive(obj, target):
    # 将目标字符串 target 按 '.' 分割成列表 target_atoms
    target_atoms = target.split(".")
    # 初始化属性迭代器 attr_itr 为 obj
    attr_itr = obj
    # 遍历目标列表中的每一个属性
    for i, atom in enumerate(target_atoms):
        # 检查 attr_itr 是否具有属性 atom
        if not hasattr(attr_itr, atom):
            # 如果不存在属性 atom，则引发 RuntimeError 异常
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        # 获取属性 atom 的值，更新 attr_itr
        attr_itr = getattr(attr_itr, atom)
    # 返回最终属性的值 attr_itr
    return attr_itr


def mark_nodes_dislike_padding(g):
    """
    标记不喜欢填充的节点，这些节点可能是在卷积或反向卷积中，希望其输入是密集的。
    如果我们对它们的输入进行填充，就会导致额外的内核复制调用！另一方面，填充通常有助于减少计算量。

    该函数找到不喜欢填充的节点。这些节点可以从卷积/反向卷积的反向方向到达，而不需要通过减少操作。
    """
    # 如果配置项 config.comprehensive_padding 不为真，则直接返回
    if not config.comprehensive_padding:
        return
    
    # 定义不喜欢填充的操作集合 ops_dislike_padding
    ops_dislike_padding = {
        aten.convolution,
        aten.convolution_backward,
    }
    
    # 定义喜欢填充的操作集合 ops_like_padding
    ops_like_padding = {
        aten.var_mean,
        aten.sum,
        aten.mean,
        aten.prod,
        aten.any,
        aten.amin,
        aten.amax,
        aten.min,
        aten.max,
        aten.argmin,
        aten.argmax,
        aten.scatter_reduce,
    }
    
    def _get_overload_packet(node):
        # 如果节点的操作是 "call_function" 并且节点的目标具有 "_overloadpacket" 属性，则返回该属性值
        return (
            node.target._overloadpacket
            if node.op == "call_function" and hasattr(node.target, "_overloadpacket")
            else None
        )
    
    # 反向遍历图中的节点 g.nodes
    for cur in reversed(g.nodes):
        # 获取当前节点的重载包
        op = _get_overload_packet(cur)
        # 如果没有重载包，则继续下一个节点
        if not op:
            continue
        # 如果当前节点的重载包在 ops_dislike_padding 中，则将其 meta 标记为不喜欢填充
        if op in ops_dislike_padding:
            cur.meta["dislike_padding"] = True
        
        # 如果当前节点被标记为不喜欢填充
        if cur.meta.get("dislike_padding", False):
            # 向前传播，将其所有输入节点标记为不喜欢填充
            for prior in cur.all_input_nodes:
                prior_op = _get_overload_packet(prior)
                if not prior_op:
                    continue
                # 如果输入节点的重载包不在 ops_like_padding 中，则将其标记为不喜欢填充
                if prior_op not in ops_like_padding:
                    prior.meta["dislike_padding"] = True


class GraphLowering(torch.fx.Interpreter):
    # 初始化时定义 graph_outputs 属性为 IRNode 类型的列表
    graph_outputs: List[ir.IRNode]
    def symbolic_sizes_strides(self, ex: torch.Tensor):
        """
        Support dynamic shapes and dynamic strides by assigning variables
        to each dimension.  We duck-shape tensors, so if two tensors
        have the same size they get assigned the same symbolic variable.
        """
        # 如果启用了形状环境重用，则将张量的大小和步长转换为符号表示
        if self.reuse_shape_env:
            return convert_shape_to_inductor(ex.size()), convert_shape_to_inductor(
                ex.stride()
            )
        else:
            from torch._dynamo.source import ConstantSource

            # TODO: this should not be needed once #93059 lands
            # https://github.com/pytorch/pytorch/pull/94031#discussion_r1096044816
            # TODO: make a dedicated UnknownSource for this?
            # NB: This is using the legacy default behavior from
            # create_symbolic_sizes_strides_storage_offset but we hope we can
            # just delete this entirely
            # 创建一个常量数据源用于未知张量大小和步长
            source = ConstantSource(
                f"__inductor_unknown_tensor_{len(self._shape_env.var_to_val)}"
            )
            (
                size,
                stride,
                _,
            ) = self._shape_env.create_symbolic_sizes_strides_storage_offset(
                ex,
                source,
            )

        # 将符号整数转换为表达式，以便返回
        size = [i.node.expr if isinstance(i, torch.SymInt) else i for i in size]
        stride = [i.node.expr if isinstance(i, torch.SymInt) else i for i in stride]
        return size, stride

    def static_sizes_strides(self, ex: torch.Tensor):
        """
        Primarily used to weights
        """
        # 将张量的静态大小和步长转换为整数类型
        size = [sympy.Integer(i) for i in ex.size()]
        stride = [sympy.Integer(i) for i in ex.stride()]
        return size, stride

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Optional[List[torch.Tensor]] = None,
        shape_env=None,
        graph_id=None,
        cpp_wrapper=False,
        aot_mode=False,
        user_visible_outputs=None,
        layout_opt=None,
        extern_node_serializer=None,
        is_inference=False,
        is_const_graph=False,
        const_output_index=None,
        const_code=None,
        const_module=None,
        name=None,
    ):
        # 初始化函数，接受多个参数来配置对象的状态和行为

    def has_feature(self, device, feature):
        assert isinstance(feature, BackendFeature), feature
        # 检查特定设备上是否存在指定的后端特性
        return feature in self.get_backend_features(get_device_type(device))

    @staticmethod
    def qualify_name(self, name: str) -> str:
        """Prepend the given name with the graph name if any."""
        # 如果存在图名称，则将给定名称前置为图名称 + 下划线
        if self.name is not None:
            return f"{self.name}_{name}"
        return name

    def make_subgraph(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
        subgraph_name: str,
    ) -> "GraphLowering":
        """
        返回一个名为 'GraphLowering' 的对象。此对象表示当前图的一个子图，包括所有继承的部分，
        但不包括图模块 (`gm`) 和 `example_inputs`。子图将单独降低，但预期会内联到父图的代码生成中。
        因此需要保持相同的 `shape_env` 和其他属性。子图的名称由父图的名称限定。
        """
        return GraphLowering(
            gm=gm,  # 使用参数 `gm` 初始化子图对象的图模块
            example_inputs=example_inputs,  # 使用参数 `example_inputs` 初始化子图对象的示例输入
            shape_env=self._shape_env,  # 使用当前对象的 `_shape_env` 属性初始化子图对象的形状环境
            cpp_wrapper=self.cpp_wrapper,  # 使用当前对象的 `cpp_wrapper` 属性初始化子图对象的 C++ 封装器
            aot_mode=self.aot_mode,  # 使用当前对象的 `aot_mode` 属性初始化子图对象的 AOT 模式
            extern_node_serializer=self.extern_node_serializer,  # 使用当前对象的 `extern_node_serializer` 属性初始化子图对象的外部节点序列化器
            is_inference=self.is_inference,  # 使用当前对象的 `is_inference` 属性初始化子图对象的推断模式
            name=self.qualify_name(subgraph_name),  # 使用方法 `qualify_name` 处理 `subgraph_name` 后作为子图对象的名称
        )
    def find_nodes_prefer_channels_last(self):
        """
        根据以下规则确定节点是否偏好通道在最后的布局：
        1. 如果节点是卷积操作的输入/输出
        2. 如果其任一用户偏好通道在最后的布局

        第一条规则存在的原因是，cudnn针对通道在最后的输入可以运行更快的卷积核；
        第二条规则同样重要，确保间接输入到卷积的节点也偏好通道在最后的布局。

        考虑以下场景：conv -> batch-norm -> relu -> conv
        如果没有第二条规则，batch-norm 的输出可能使用连续的布局。这会导致两次额外的拷贝：
        1. batch-norm 的输出初始应该是通道在最后的布局，因为其输入是卷积的输出。
           强制使 batch-norm 的输出连续会导致第一次拷贝。
        2. 第二个卷积的输入初始是连续的。这种布局从 batch-norm 的输出传播而来。
           我们需要将其转换为通道在最后的布局，导致第二次拷贝。
        有了第二条规则，我们确保链条中所有张量使用通道在最后的布局。这样可以避免这两次拷贝。

        返回一个集合，包含偏好通道在最后布局的节点。
        """
        output_set = set()
        for n in reversed(self.module.graph.nodes):
            if n.target == torch.ops.aten.convolution.default:
                output_set.add(n)
                continue

            for user in n.users:
                if user in output_set:
                    output_set.add(n)
                    break

        """
        需要第二次遍历，将这些偏好通道在最后布局的节点的下游节点也加入集合中。
        这次遍历特别需要，以避免在反向传播中混合布局的内核输入。

        例如，一个 conv-batchnorm 的输出传递给 relu，后者的输出从前向图中返回。
        如果没有这第二次遍历，我们会强制 relu 的输出是连续的。
        然后在反向传播中，relu 的连续输出可能与其他偏好通道在最后的张量混合，
        然后传递给一个内核，这会导致问题。

        这次遍历提高了 yolov3 的训练加速比从 1.116x （比禁用布局优化的加速比 1.196x 差）到 1.457x。
        同时也提高了 dla102 的训练加速比从 1.240x （比禁用布局优化的加速比 1.523x 差）到 1.835x 。
        这也对以下模型有帮助：
        - res2net101_26w_4s
        - res2net50_14w_8s
        - sebotnet33ts_256
        """
        for n in self.module.graph.nodes:
            if n in output_set:
                output_set.update(n.users)

        return output_set

    def warn_fallback(self, name):
        """
        如果尚未警告过名为 name 的后备内核，则发出警告并记录在 _warned_fallback 中。
        """
        if name not in self._warned_fallback:
            self._warned_fallback.add(name)
            perf_hint_log.info("Using FallbackKernel: %s", name)
    # 添加设备信息到相应的集合中，根据设备类型添加到 self.device_types 中
    # 如果设备有索引，也添加到 self.device_idxs 中
    # 如果当前图的当前节点存在且设备不在 self.device_node_mapping 中，则将设备映射到当前节点
    def add_device_info(self, device: torch.device):
        self.device_types.add(device.type)
        if device.index is not None:
            self.device_idxs.add(device.index)
        if V.graph.current_node and device not in self.device_node_mapping:
            self.device_node_mapping[device] = V.graph.current_node

    @property
    # 返回 V.fake_mode 的属性值
    def fake_mode(self):
        return V.fake_mode

    # 根据 buffer_name 获取对应的缓冲区对象
    def get_buffer(self, buffer_name: str):
        if buffer_name in self.name_to_buffer:
            return self.name_to_buffer[buffer_name]
        if buffer_name in self.graph_inputs:
            return self.graph_inputs[buffer_name]
        if buffer_name in self.constants:
            # 如果 buffer_name 在常量中，根据数据创建一个固定布局的常量缓冲区对象返回
            data = V.graph.constants[buffer_name]
            return ir.ConstantBuffer(
                buffer_name,
                ir.FixedLayout(
                    data.device, data.dtype, *V.graph.static_sizes_strides(data)
                ),
            )
        # 如果找不到对应的缓冲区，返回 None
        return None

    # 根据 buffer_name 获取对应的数据类型
    def get_dtype(self, buffer_name: str):
        if buffer_name in self.constants:
            return self.constants[buffer_name].dtype
        if buffer_name in self.name_to_buffer:
            return self.name_to_buffer[buffer_name].get_dtype()
        if buffer_name in self.graph_inputs:
            return self.graph_inputs[buffer_name].get_dtype()
        # 如果 buffer_name 符合特定模式，则使用正则表达式匹配获取数据类型
        m = re.match(r"(as_strided|reinterpret_tensor)\(([a-zA-Z0-9_]+),", buffer_name)
        if m:
            return self.get_dtype(m.group(1))
        # 抛出 KeyError 如果找不到对应的数据类型
        raise KeyError(f"could not find {buffer_name}")

    # 根据 buffer_name 获取对应的元素数量
    def get_numel(self, buffer_name: str):
        from .ir import MultiOutputLayout

        if buffer_name in self.constants:
            return self.constants[buffer_name].numel()
        if buffer_name in self.name_to_buffer:
            buf = self.name_to_buffer[buffer_name]
            if isinstance(getattr(buf, "layout", None), MultiOutputLayout):
                return 1
            return buf.get_numel()
        if buffer_name in self.graph_inputs:
            return self.graph_inputs[buffer_name].get_numel()
        # 抛出 KeyError 如果找不到对应的元素数量
        raise KeyError(f"could not find {buffer_name}")

    # 使用 dynamo_timed 装饰器来执行 run 方法，并返回其结果
    @dynamo_timed
    def run(self, *args):
        return super().run(*args)

    # 注册一个缓冲区对象，并返回其名称
    def register_buffer(self, buffer: ir.Buffer, *, set_name: bool = False):
        name = self.qualify_name(f"buf{len(self.buffers)}")
        self.buffers.append(buffer)
        self.name_to_buffer[name] = buffer
        # 如果缓冲区不是空的 CPU 张量，并且有设备信息，则将设备信息添加到相应集合中
        if (
            not (isinstance(buffer, ir.ComputedBuffer) and buffer.is_zero_elements())
            and buffer.get_device() is not None
        ):
            self.add_device_info(buffer.get_device())

        # 如果 set_name 为 True，则设置缓冲区的名称为生成的名称
        if set_name:
            buffer.name = name
        return name

    # 注册一个缓冲区名称列表，存储到 self.lists 中，并返回列表的名称
    def register_list(self, buffer_names: List[str]):
        name = self.qualify_name("list_" + "_".join(buffer_names))
        self.lists[name] = buffer_names
        return name
    # 注册与给定节点输出相关的用户
    def register_users_of(self, node_output):
        # 内部函数，递归注册值或者列表/元组中的每个元素
        def register(value):
            # 如果值是列表或元组，递归注册每个元素
            if isinstance(value, (list, tuple)):
                for x in value:
                    register(x)
            # 如果值是 IRNode 类型
            if isinstance(value, ir.IRNode):
                # 检查是否有 data 属性，并且 data 属性也是 IRNode 类型
                if (
                    not hasattr(value, "data")
                    or not isinstance(value.data, ir.IRNode)
                    or not (
                        hasattr(value.data, "data")
                        and isinstance(value.data.data, ir.IRNode)
                    )
                ):
                    return

                # 获取当前节点的所有读取名称，并将当前节点添加到相应名称的用户列表中
                for read_name in value.get_read_names():
                    self.name_to_users[read_name].append(value)

        # 调用 register 函数注册给定的节点输出
        register(node_output)

    # 标记缓冲区名称为 name 的缓冲区被修改
    def mark_buffer_mutated(self, name: str):
        """
        当缓冲区被修改时，确保在修改发生之前实现对旧版本的所有读取。
        """
        assert isinstance(name, str)
        # 将被修改的缓冲区名称添加到 mutated_buffers 集合中
        self.mutated_buffers.add(name)

        # 如果 name 不在 name_to_users 字典中，直接返回
        if name not in self.name_to_users:
            return

        # 遍历使用 name 缓冲区的所有用户，并调用其 realize 方法
        for user in self.name_to_users[name]:
            user.realize()

    # 获取常量 name 的原始值
    def get_original_value_of_constant(self, name: str):
        """
        在 AOTI 中，模块缓冲区可能在跟踪和编译过程中被修改。
        因此，我们需要从先前存储的原始缓冲区读取，以确保生成的 model.so 使用正确的初始值。
        """
        assert name in self.allocated_constant_name and name in self.constants, (
            "Can not find the original value for " + name
        )
        # 根据分配的常量名获取其克隆参数缓冲区的名称
        orig_name = get_cloned_parameter_buffer_name(self.allocated_constant_name[name])
        # 返回模块元数据中 orig_name 对应的值，否则返回常量字典中 name 对应的值
        return (
            self.module.meta[orig_name]
            if orig_name in self.module.meta
            else self.constants[name]
        )
    # 分配一个非重复的常量名，将给定的数据与名称关联起来
    def allocate_non_dup_const_name(self, name, data):
        # 保存原始名称
        orig_name = name
        # 如果禁用了运行时常量折叠，则检查是否存在相同的常量已经存在
        if not config.aot_inductor.use_runtime_constant_folding:
            for constant_name, value in self.constants.items():
                # 检查数据是否符合条件，如果是，则返回已存在的常量名
                if (
                    not data.is_mkldnn
                    and data.size() == value.size()
                    and data.stride() == value.stride()
                    and data.dtype == value.dtype
                    and data.device == value.device
                    and data.untyped_storage().data_ptr()
                    == value.untyped_storage().data_ptr()
                    and data.storage_offset() == value.storage_offset()
                ):
                    return constant_name

        # 如果名称为None，则生成一个新的常量名称
        if name is None:
            name = f"constant{len(self.constants)}"
        # 如果名称以数字开头，则添加下划线作为前缀
        if name[0].isdigit():
            name = f"constant_{name}"
        # 通过限定名称方法修正名称
        name = self.qualify_name(name)
        # 保留合理的字符作为名称前缀
        prefix = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        name = prefix
        cnt = 0
        # 确保生成的名称在常量字典中唯一
        while name in self.constants:
            name = f"{prefix}_{cnt}"
            cnt += 1
        # 将常量名称与数据关联存储到常量字典中
        self.constants[name] = data
        # 存储常量的表示形式，包括设备、数据类型、大小、步幅和哈希值
        self.constant_reprs[name] = (
            f"{data.device!r} {data.dtype!r} "
            f"{tuple(data.size())!r} {tuple(data.stride())!r} "
            f"{hash(data):x}"
        )
        # 记录分配的常量名称及其原始名称的关系
        self.allocated_constant_name[name] = orig_name
        # 返回分配的常量名称
        return name

    # 向常量集合中添加张量常量，并返回包装后的张量对象
    def add_tensor_constant(self, data, name=None):
        # 分配一个非重复的常量名
        new_name = self.allocate_non_dup_const_name(name, data)
        # 创建一个包含常量数据的张量对象
        return TensorBox.create(
            ir.ConstantBuffer(
                new_name,
                FixedLayout(data.device, data.dtype, *self.static_sizes_strides(data)),
            )
        )

    # 获取常量的名称，并根据需要复制到指定的设备
    def constant_name(self, name: str, device_override: Optional[torch.device]):
        """
        将常量复制到需要的设备上。
        如果设备不匹配，则复制数据并返回一个新的名称。
        """
        # 如果指定设备与常量的设备一致，或者设备未指定，则直接返回常量名称
        if self.constants[name].device == device_override or device_override is None:
            return name
        # 否则，根据设备复制数据，并生成一个新的常量名称
        with torch.utils._python_dispatch._disable_current_modes():
            # 调用者可能设置了假张量模式，会在调用 .to() 方法时创建一个假张量，因此在这里取消设置模式
            return self.allocate_non_dup_const_name(
                f"{name}_{device_override.type}{device_override.index or 0}",
                self.constants[name].to(device_override),
            )
    # 定义一个方法 `placeholder`，接受一个字符串类型的目标、位置参数和关键字参数
    def placeholder(self, target: str, args, kwargs):
        # 调用父类的 `placeholder` 方法，并将返回值赋给 `example`
        example = super().placeholder(target, args, kwargs)
        # 将目标添加到图的输入名称列表中
        self.graph_input_names.append(target)
        
        # 如果 `example` 是 `SymTypes` 类型的实例
        if isinstance(example, SymTypes):
            # 获取表达式节点并赋给 `expr`
            expr = example.node.expr
            # 将 `expr` 存储到图的输入字典中，键为 `target`
            self.graph_inputs[target] = expr
            # 返回 `expr`
            return expr
        
        # 如果 `example` 是整型、布尔型或浮点型之一
        elif isinstance(example, (int, bool, float)):
            # 将 `example` 转换为符号表达式并赋给 `expr`
            expr = sympy.sympify(example)
            # 将 `expr` 存储到图的输入字典中，键为 `target`
            self.graph_inputs[target] = expr
            # 返回 `expr`
            return expr
        
        # 如果 `example` 是 `BackwardState` 类型的实例
        if isinstance(example, BackwardState):
            # 忽略该参数，因为它应该未被使用
            # 另一种方法是在 AotAutograd 中将其过滤掉
            return None
        
        # 断言 `example` 是 `torch.Tensor` 类型，如果不是将抛出异常并显示 `example` 的值
        assert isinstance(example, torch.Tensor), example
        
        # 如果 `example` 的大小和步幅没有包含符号信息
        if not example._has_symbolic_sizes_strides:
            # 使用静态大小和步幅计算 `sizes` 和 `strides`
            sizes, strides = self.static_sizes_strides(example)
        else:
            # 使用符号大小和步幅计算 `sizes` 和 `strides`
            sizes, strides = self.symbolic_sizes_strides(example)
        
        # TODO(jansel): 处理输入的别名
        
        # 使用 `qualify_name` 方法确保 `target` 的命名符合要求
        target = self.qualify_name(target)
        
        # 创建一个 `TensorBox` 对象，内部包含一个 `InputBuffer` 对象
        tensor = TensorBox.create(
            InputBuffer(
                target,
                FixedLayout(example.device, example.dtype, sizes, strides),
            )
        )
        
        # 将 `tensor` 存储到图的输入字典中，键为 `target`
        self.graph_inputs[target] = tensor
        # 将 `tensor.data.data` 存储到图的原始输入字典中，键为 `target`
        self.graph_inputs_original[target] = tensor.data.data
        # 添加 `example` 的设备信息到设备信息集合中
        self.add_device_info(example.device)

        # 注意: [Inductor 中的输入对齐处理]
        # 输入对齐对于生成高效代码很重要。某些操作（如矢量化加载）只能在对齐的输入上执行。
        #
        # 但如果我们在编码时假设输入已对齐，然后在运行时获得未对齐的输入，我们将被迫克隆 - 这对性能和内存使用都不利。
        #
        # 一种选择是基于 storage_offset%ALIGNMENT 进行保护，然后根据此生成代码。但是发现 storage_offset 保护耗费高昂且导致重新编译；
        # 相反，我们基于示例输入的对齐生成代码而没有保护。
        with maybe_get_suppress_shape_guards_ctx():
            # 如果应该假设输入是对齐的，则将 `target` 添加到对齐输入集合中
            if should_assume_input_aligned(example):
                self.aligned_inputs.add(target)
        
        # 返回 `tensor`
        return tensor
    # 定义一个实例方法，用于调用目标函数，传入参数和关键字参数
    def call_function(self, target, args, kwargs):
        # 检查目标函数是否为 operator.getitem，并且第一个参数是 list、tuple 或 dict 类型
        if target is operator.getitem and isinstance(args[0], (list, tuple, dict)):
            # 如果是，调用父类的 call_function 方法处理
            return super().call_function(target, args, kwargs)

        # 检查目标函数是否具有 _inductor_lowering_function 属性
        if hasattr(target, "_inductor_lowering_function"):
            # 如果有，直接调用目标函数并传入参数和关键字参数
            # 该情况用于 .pattern_matcher 中的特定处理逻辑
            return target(*args, **kwargs)

        # 定义内部函数，用于获取自定义操作的布局约束
        def get_custom_op_layout_constraints(target, args, kwargs):
            # 如果目标操作需要保留步幅顺序，则需要对其参数进行布局约束
            layout_constraint = None
            if torch._C.Tag.needs_fixed_stride_order in target.tags:
                # 设置当前参数的固定步幅顺序约束
                args, kwargs = constrain_to_fx_strides(
                    self.current_node, *args, **kwargs
                )
                # 注册布局约束，以便在再次使用回退时，可以对参数施加相同的布局约束
                layout_constraint = constrain_to_fx_strides
            return layout_constraint, args, kwargs

        # 如果目标函数不在预定义的降级列表中
        if target not in lowerings:
            # 断言目标函数为 torch._ops.OpOverload 类型，否则引发异常
            assert isinstance(
                target, torch._ops.OpOverload
            ), f"{target} is not an OpOverload"
            base_name = target.name().split(".")[0]
            # 如果基础名称在 FALLBACK_ALLOW_LIST 中，执行降级处理
            if base_name in FALLBACK_ALLOW_LIST:
                make_fallback(target)
            # 如果启用了隐式回退
            elif config.implicit_fallbacks:
                # 获取自定义操作的布局约束
                layout_constraint, args, kwargs = get_custom_op_layout_constraints(
                    target, args, kwargs
                )
                # 判断目标操作是否有分解算子，选择相应的错误类型
                error = (
                    MissingOperatorWithDecomp
                    if get_decompositions([target])
                    else MissingOperatorWithoutDecomp
                )
                # 记录信息，创建隐式回退
                log.info(
                    "Creating implicit fallback for:\n%s",
                    error.operator_str(target, args, kwargs),
                )
                # 执行降级处理，并传入布局约束
                make_fallback(target, layout_constraint)

            # 如果存在分解算子
            elif get_decompositions([target]):
                # 抛出异常，指示缺少具有分解的操作
                raise MissingOperatorWithDecomp(target, args, kwargs)
            else:
                # 否则，抛出异常，指示缺少没有分解的操作
                raise MissingOperatorWithoutDecomp(target, args, kwargs)

        try:
            # 调试信息，显示通过哪种降级方式处理目标函数
            log.debug("  via %s", lowerings[target])
            # 调用降级函数，并传入参数和关键字参数
            out = lowerings[target](*args, **kwargs)
            return out
        except Exception as e:
            # 如果出现异常，包装成 LoweringException，并重新抛出
            raise LoweringException(e, target, args, kwargs).with_traceback(
                e.__traceback__
            ) from None
    def can_inline_constant(t: torch.Tensor) -> bool:
        """
        True if this is a small constant attr that will be inlined.
        """
        # 返回True，如果张量是一维且长度不超过8，表示可以内联
        return len(t.shape) == 1 and t.shape[0] <= 8

    def get_attr(self, target, args, kwargs):
        # 获取目标属性的值
        value = getattr_recursive(self.module, target)

        if isinstance(value, torch.fx.GraphModule):
            # 如果值是 torch.fx.GraphModule，返回其子图对象
            return ir.Subgraph(name=target, graph_module=value)

        if isinstance(value, torch._C.ScriptObject):
            # 如果值是 torch._C.ScriptObject，保存到torchbind_constants中并返回TorchBindObject对象
            self.torchbind_constants[target] = value
            self.constant_reprs[target] = ""
            return TorchBindObject(target, value)

        if (
            config.aot_inductor.use_runtime_constant_folding
            or config.always_keep_tensor_constants
            or unsupported_output_tensor(value)
        ):
            # 如果满足常量折叠条件，调用add_tensor_constant方法处理并返回
            return self.add_tensor_constant(value, target)

        with no_dispatch():
            if value.shape == ():
                # 如果值是标量，返回常量对象
                return Constant(value.item(), value.dtype, value.device)
            if self.can_inline_constant(value):
                # 如果可以内联常量，调用tensor函数处理
                # 引用来自.lowering的张量内联逻辑
                from .lowering import tensor

                return tensor(value.tolist(), dtype=value.dtype, device=value.device)

        # 默认情况下，调用add_tensor_constant方法处理并返回
        return self.add_tensor_constant(value, target)

    def call_module(self, target, args, kwargs):
        # 抛出断言错误，表明不支持调用模块的操作
        raise AssertionError

    def call_method(self, target, args, kwargs):
        # 抛出断言错误，表明不支持调用方法的操作
        raise AssertionError

    def finalize(self):
        # 遍历所有缓冲区，进行布局决策
        for buf in self.buffers:
            buf.decide_layout()

    @contextmanager
    def set_current_node(self, node: torch.fx.Node):
        # 管理当前节点的上下文管理器，用于在执行过程中设置和恢复当前节点
        old = self.current_node
        try:
            self.current_node = node
            yield
        finally:
            self.current_node = old

    def try_match_insignificant_strides(
        self,
        tensor,
        meta_strides_inp: Tuple[Union[int, torch.SymInt], ...],
    ) -> ir.TensorBox:
        """
        Tries to match the strides of the tensor to those in the meta_strides. Strides of insignificant
        dimensions - size 0 or 1 - will be updated.

        If there are real stride differences (NHWC vs NCHW) then the input will be returned.
        """

        # 应该已经被实现
        assert torch._inductor.ir.is_storage_and_layout(tensor)

        # 将 meta_strides_inp 中的每个元素转换为表达式，如果是 torch.SymInt 类型的话
        meta_strides = [
            s.node.expr if isinstance(s, torch.SymInt) else s for s in meta_strides_inp
        ]

        # 如果 tensor 的步幅与 meta_strides 中的步幅完全匹配，则直接返回 tensor
        if all(
            self.sizevars.statically_known_equals(s1, s2)
            for s1, s2 in zip(meta_strides, tensor.get_stride())
        ):
            return tensor

        # 定义一个函数，用于检查在显著维度上 tensor 的步幅是否与 meta_strides 匹配
        def significant_strides_equal(shape, meta_strides, tensor_strides):
            for dim, s1, s2 in zip(shape, meta_strides, tensor_strides):
                if self.sizevars.statically_known_leq(dim, 1):  # 忽略大小为 1 或 0 的维度
                    continue

                if not self.sizevars.statically_known_equals(s1, s2):
                    return False

            return True

        # 如果在显著维度上 tensor 的步幅与 meta_strides 不匹配，则直接返回 tensor
        if not significant_strides_equal(
            tensor.get_size(), meta_strides, tensor.get_stride()
        ):
            return tensor

        # 获取 tensor 的存储和旧布局
        storage, old_layout = torch._inductor.ir.as_storage_and_layout(tensor)
        new_stride = list(old_layout.stride)

        # 更新新的步幅，对于大小为 1 的维度，使用 meta_strides 中的步幅
        for i, s in enumerate(tensor.get_size()):
            if self.sizevars.statically_known_leq(s, 1):  # 忽略大小为 1 或 0 的维度
                new_stride[i] = meta_strides[i]

        # 创建新的布局对象
        new_layout = torch._inductor.ir.FixedLayout(
            old_layout.device,
            old_layout.dtype,
            old_layout.size,
            new_stride,
            old_layout.offset,
        )
        # 返回一个新的 TensorBox，其中包含重新解释视图的存储和新的布局
        return ir.TensorBox(torch._inductor.ir.ReinterpretView(storage, new_layout))

    def validate_can_generate_cpp_wrapper(self):
        # 检查是否禁用了 C++ 代码生成
        if config.disable_cpp_codegen:
            raise CppWrapperCodeGenError("C++ codegen is disabled")

        # 检查系统平台是否为 linux 或 darwin，否则抛出异常
        if sys.platform not in ["linux", "darwin"]:
            raise CppWrapperCodeGenError(f"Unsupported platform {sys.platform}")

        # 遍历图的输入值，检查是否支持生成对应的 C++ 包装器
        for value in self.graph_inputs.values():
            dtype = None
            if isinstance(value, TensorBox):
                dtype = value.get_dtype()
            elif isinstance(
                value, (sympy.Symbol, sympy.Expr, sympy.core.numbers.Integer)
            ):
                dtype = may_get_constant_buffer_dtype(value)

            # 如果不支持当前输入的数据类型，则抛出异常
            if not supported_dtype_of_cpp_wrapper(dtype, self.cuda):
                raise CppWrapperCodeGenError(f"Unsupported input dtype {dtype}")
    def init_wrapper_code(self):
        # 检查是否需要使用 CUDA
        self.cuda = "cuda" in self.device_types
        # 如果开启了 C++ 包装器，验证是否能生成 C++ 包装代码
        if self.cpp_wrapper:
            self.validate_can_generate_cpp_wrapper()

        # 复制设备类型集合，并移除 "cpu" 和 "meta" 类型
        device_types = self.device_types.copy()
        device_types.discard("cpu")
        device_types.discard("meta")
        # TODO(Eikan): 目前仅支持混合使用 CPU 和其他设备。
        assert len(device_types) <= 1, "Does not support mixing {}".format(
            "+".join(device_types)
        )
        # 检查是否只有 CPU 设备
        only_cpu = len(device_types) == 0
        # 确定使用的设备类型，如果只有 CPU 则选择 "cpu"，否则选择剩余的设备类型
        device_type = "cpu" if only_cpu else device_types.pop()

        # 获取指定设备类型的设备操作重载
        self.device_ops = get_device_op_overrides(device_type)
        # 获取生成指定设备类型的包装代码的类
        wrapper_code_gen_cls = get_wrapper_codegen_for_device(
            device_type, self.cpp_wrapper
        )
        # 断言是否找到了指定设备类型的包装代码生成类
        assert wrapper_code_gen_cls is not None, f"Device {device_type} not supported"
        # 实例化指定设备类型的包装代码生成器
        self.wrapper_code = wrapper_code_gen_cls()

        # 如果存在常量模块，则复用其包装代码生成器的状态和核心映射
        if self.const_module:
            # 复制常量模块的迭代器状态
            self.wrapper_code._names_iter = self.const_module.wrapper_code._names_iter
            # 复制常量模块的源码到内核的映射
            self.wrapper_code.src_to_kernel = (
                self.const_module.wrapper_code.src_to_kernel
            )

    def codegen(self):
        from .scheduler import Scheduler

        # 初始化包装代码生成器
        self.init_wrapper_code()

        # 创建调度器对象，并传入缓冲区列表
        self.scheduler = Scheduler(self.buffers)
        # 在调试模式下绘制原始图形和调度器节点
        V.debug.draw_orig_fx_graph(self.orig_gm, self.scheduler.nodes)

        # 将当前图的代码生成图形推送到包装代码生成器中
        self.wrapper_code.push_codegened_graph(self)
        # 调用调度器的代码生成方法
        self.scheduler.codegen()
        # 生成最终的包装代码
        result = self.wrapper_code.generate(self.is_inference)
        # 弹出当前图的代码生成图形
        self.wrapper_code.pop_codegened_graph()
        return result

    def codegen_subgraph(self, parent_graph):
        """
        This is a more compact version of the `codegen()` above
        where we codegen this graph as a subgraph of some parent
        graph. The parent graph is passed as an argument: the
        intention is to inline codegening of the subgraph in
        the parent graph's wrapper code (including the generated
        kerenls). The wrapper code is not finalized (via `.generate()`
        call), as this will be done in the parent graph's `codegen()`.
        """
        from .scheduler import Scheduler

        # 将父图的包装代码生成器、设备操作和 C++ 包装器状态传递给当前图
        self.wrapper_code = parent_graph.wrapper_code
        self.device_ops = parent_graph.device_ops
        self.cpp_wrapper = parent_graph.cpp_wrapper

        # 创建调度器对象，并传入缓冲区列表
        self.scheduler = Scheduler(self.buffers)
        # 在调度器中生成代码
        self.scheduler.codegen()

    def count_bytes(self):
        # 初始化字节总数、节点计数和节点运行时间列表
        total_bytes = 0
        node_counts = []
        node_runtimes = []
        # 遍历调度器中的每个节点
        for node in self.scheduler.nodes:
            # 获取节点读写缓冲区的大小
            num_bytes = node.get_read_write_buffers_sizes()
            total_bytes += num_bytes
            # 将节点和缓冲区大小（除以 4，因为假设每个值占4字节）添加到节点计数列表
            node_counts.append((node, num_bytes // 4))
            # 将节点和估计的运行时间添加到节点运行时间列表
            node_runtimes.append((node, node.get_estimated_runtime()))
        return total_bytes, node_counts, node_runtimes
    def save_output_code(code: str):
        # 仅用于单元测试中进行补丁操作，无实际功能
        pass

    @dynamo_timed(phase_name="code_gen", fwd_only=False)
    def compile_to_module(self):
        # 导入 PyCodeCache 类，用于代码缓存
        from .codecache import PyCodeCache

        # 调用 self.codegen_with_cpp_wrapper() 或 self.codegen() 生成代码和行映射
        code, linemap = (
            self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
        )

        # 将生成的代码保存，用于调试
        GraphLowering.save_output_code(code)
        # 调试输出生成的代码内容
        output_code_log.debug("Output code: \n%s", code)

        try:
            # 构建行映射列表
            linemap = [(line_no, node.stack_trace) for line_no, node in linemap]
            # 将代码写入 PyCodeCache，返回键和路径
            key, path = PyCodeCache.write(code)
        except Exception:
            # 发生异常时记录结构化跟踪，并传递生成的代码作为 payload
            trace_structured(
                "inductor_output_code",
                payload_fn=lambda: code,
            )
            raise
        else:
            # 记录生成代码的结构化跟踪和文件名
            trace_structured(
                "inductor_output_code",
                lambda: {"filename": path},
                payload_fn=lambda: code,
            )

        # 从 PyCodeCache 加载模块
        mod = PyCodeCache.load_by_key_path(
            key,
            path,
            linemap=linemap,
            attrs={**self.constants, **self.torchbind_constants},
        )
        # 设置缓存键、路径和行映射
        self.cache_key = key
        self.cache_path = path
        self.cache_linemap = linemap

        # 根据注释要求，此处是在两个地方记录模块代码的位置，用于调试
        assert mod.__file__ is not None
        log_module_code(mod.__file__)
        log.debug("Output code written to: %s", mod.__file__)
        output_code_log.info("Output code written to: %s", mod.__file__)

        # 如果启用了基准测试，则打印编译模块的路径
        if config.benchmark_kernel:
            print(f"Compiled module path: {mod.__file__}", file=sys.stderr)

        # 调试输出模块的代码位置
        V.debug.output_code(mod.__file__)
        # 复制调试文件
        V.debug.copy(os.path.splitext(mod.__file__)[0] + ".debug")

        # 返回加载的模块对象
        return mod
    # 如果处于AOT编译模式下
    def compile_to_fn(self):
        if self.aot_mode:
            # 导入AotCodeCompiler类，用于AOT编译
            from .codecache import AotCodeCompiler
            
            # 断言只有在使用C++包装器时才支持AOT模式
            assert self.cpp_wrapper, "AOT mode only supports C++ wrapper"
            
            # 生成带有C++包装器的代码和行映射
            code, linemap = self.codegen_with_cpp_wrapper()
            
            # 记录输出的代码到调试日志中
            output_code_log.debug("Output code: \n%s", code)

            # 初始化外部内核节点序列化的结果为None
            serialized_extern_kernel_nodes = None
            
            # 如果运行在fbcode环境中，且存在外部内核节点和节点序列化器
            if (
                config.is_fbcode()
                and self.extern_kernel_nodes
                and self.extern_node_serializer
            ):
                # 序列化外部内核节点
                serialized_extern_kernel_nodes = self.extern_node_serializer(
                    self.extern_kernel_nodes
                )
                # 记录序列化后的外部内核节点到调试日志中
                output_code_log.debug(
                    "Serialized Extern Kernel Nodes: \n%s",
                    serialized_extern_kernel_nodes,
                )

            # 直接返回编译后的代码文件路径，使用AotCodeCompiler进行编译
            return AotCodeCompiler.compile(
                self, code, serialized_extern_kernel_nodes, cuda=self.cuda
            )
        else:
            # 如果不处于AOT编译模式，返回编译后的模块的调用函数
            return self.compile_to_module().call

    # 获取输出节点的名称列表
    def get_output_names(self):
        return [
            node.get_name()
            for node in self.graph_outputs
            if not isinstance(node, ir.NoneAsConstantBuffer)
            and not isinstance(node, ir.ShapeAsConstantBuffer)
        ]

    # 判断给定名称的参数是否为未指定的参数
    def is_unspec_arg(self, name: str):
        # dynamo将未指定的变量包装为0维CPU张量，
        # 在代码生成期间需要转换为标量（仅适用于triton）
        return (
            name in self.graph_inputs.keys()
            and self.graph_inputs[name].get_numel() == 1
            and self.graph_inputs[name].get_device().type == "cpu"
        )
```