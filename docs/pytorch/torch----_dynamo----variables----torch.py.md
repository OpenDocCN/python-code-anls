# `.\pytorch\torch\_dynamo\variables\torch.py`

```
# 引入必要的模块和函数
# 设置 mypy 配置允许未标记类型的函数定义
# functools：提供高阶函数操作
# inspect：用于获取对象信息的工具
# logging：用于记录日志信息的模块

import functools
import inspect
import logging

# 引入数学和正则表达式模块
import math
import re

# 引入类型提示相关的模块和类
from typing import Dict, List

# 引入 PyTorch 的核心 C 模块和一些特定模块
import torch._C
import torch._refs
import torch.fx
import torch.nn
import torch.onnx.operators

# 引入用于记录警告信息的特定模块
from torch._logging import warning_once

# 引入 StreamBase 类
from torch._streambase import _StreamBase

# 引入跨包引用的保护机制
from ..._guards import TracingContext

# 引入当前包的特定子模块和函数
from .. import config, polyfill, variables

# 引入代码生成器模块
from ..codegen import PyCodegen

# 引入用于创建参数操作的相关模块和函数
from ..create_parameter_op import (
    can_convert_to_tracable_parameter,
    new_parameter_placeholder,
    tracable_create_parameter,
)

# 引入设备接口注册函数
from ..device_interface import get_registered_device_interfaces

# 引入未实现异常模块
from ..exc import unimplemented

# 引入守卫构建器和守卫安装函数
from ..guards import GuardBuilder, install_guard

# 引入合成本地来源模块
from ..source import SyntheticLocalSource

# 引入各种实用函数和工具
from ..utils import (
    check_unspec_or_constant_args,
    guard_if_dyn,
    has_torch_function,
    hashable,
    product,
    proxy_args_kwargs,
    unwrap_if_wrapper,
)

# 引入变量追踪器基类
from .base import VariableTracker

# 引入上下文管理器相关模块
from .ctx_manager import (
    AutocastModeVariable,
    NullContextVariable,
    TorchFunctionDisableVariable,
)

# 引入分布式相关变量模块
from .distributed import DistributedVariable, ProcessGroupVariable

# 引入列表和元组变量模块
from .lists import ListVariable, TupleVariable

# 引入 Torch 函数调度相关模块
from .torch_function import can_dispatch_torch_function, dispatch_torch_function

# 尝试引入 NumPy，如果未找到则设置为 None
try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

# 尝试引入分布式组件的特定模块，如果未找到则设置为 None
try:
    from torch.distributed._composable.fsdp import _fsdp_param_group
except ModuleNotFoundError:
    _fsdp_param_group = None  # type: ignore[assignment]

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)

# 支持的上下文管理器类字典，键为各种 Torch 函数和类
supported_ctx_manager_classes = dict.fromkeys(
    [
        torch.profiler.profiler.profile,
        torch.autograd.forward_ad._set_fwd_grad_enabled,
        torch.autograd.forward_ad.dual_level,
        torch.autograd.profiler.profile,
        torch.autograd.profiler.record_function,
        torch._C.DisableTorchFunctionSubclass,
        torch._functorch.vmap.vmap_increment_nesting,
        torch._functorch.eager_transforms.grad_increment_nesting,
        torch._functorch.eager_transforms.jvp_increment_nesting,
        torch._functorch.eager_transforms.enable_inplace_requires_grad,
        torch.amp.autocast_mode.autocast,
        torch.autograd.grad_mode.enable_grad,
        torch.autograd.grad_mode.inference_mode,
        torch.autograd.grad_mode.no_grad,
        torch.autograd.grad_mode.set_grad_enabled,
        torch.autograd.graph.disable_saved_tensors_hooks,
        torch.cpu.amp.autocast_mode.autocast,
        torch.cuda.amp.autocast_mode.autocast,
    ]
)

# 重写操作以使用张量大小方法的字典
REWRITE_OPS_TO_TENSOR_SIZE_METHOD = dict.fromkeys(
    [
        torch.onnx.operators.shape_as_tensor,
        torch._shape_as_tensor,
    ]
)

# 常量折叠函数列表
constant_fold_functions = [
    torch._assert,
    torch._utils._get_device_index,
    torch._C._get_cublas_allow_tf32,
    torch._C._is_any_autocast_enabled,
    torch.cuda.get_device_properties,
    torch.cuda.is_available,
    torch.distributed.is_available,
    torch.get_autocast_dtype,
    # 此处还有其他函数，但未完全列出
]
    # 获取自动混合精度模式下使用的 GPU 数据类型
    torch.get_autocast_gpu_dtype,
    
    # 获取默认的张量数据类型
    torch.get_default_dtype,
    
    # 检查自动混合精度缓存是否启用
    torch.is_autocast_cache_enabled,
    
    # 检查自动混合精度在 CPU 上是否启用
    torch.is_autocast_cpu_enabled,
    
    # 检查自动混合精度是否启用（无论在 CPU 还是 GPU 上）
    torch.is_autocast_enabled,
    
    # 检查张量是否是复数类型
    torch.is_complex,
    
    # 检查张量是否是浮点数类型
    torch.is_floating_point,
    
    # 获取用于 torch.nn.functional 中的 _Reduction 枚举值
    torch.nn.functional._Reduction.get_enum,  # type: ignore[attr-defined]
    
    # 将两个数据类型提升为它们中更高级别的类型
    torch.promote_types,
    
    # 获取私有用途的后端名称，通常用于内部调试或特定用途
    torch._C._get_privateuse1_backend_name,
# 如果 torch 分布式模块可用
if torch.distributed.is_available():
    # 将以下函数添加到 constant_fold_functions 列表中
    constant_fold_functions.extend(
        [
            torch.distributed.is_initialized,
            torch.distributed.get_rank,
            torch.distributed.get_world_size,
        ]
    )

# 将 constant_fold_functions 列表转换为字典，以便实现 O(1) 时间复杂度的访问
constant_fold_functions = dict.fromkeys(constant_fold_functions)

# 定义一个包含 torch 追踪状态的函数字典
tracing_state_functions = {
    torch.jit.is_scripting: False,
    torch.jit.is_tracing: False,
    torch._C._get_tracing_state: None,
    torch.fx._symbolic_trace.is_fx_tracing: False,
    torch.onnx.is_in_onnx_export: False,
    torch._dynamo.external_utils.is_compiling: True,
    torch._utils.is_compiling: True,
    torch.compiler.is_compiling: True,
    torch.compiler.is_dynamo_compiling: True,
    torch.nn.modules.activation._is_make_fx_tracing: False,
}

# 创建一个包含二元操作的字典，初始值为 None
bin_ops = dict.fromkeys(["add", "sub", "mul", "div", "sqrt"])

# 定义一个基础的 Torch 变量类，继承自 VariableTracker
class BaseTorchVariable(VariableTracker):
    """common base for all torch.* functions, classes, modules and other things"""

    @classmethod
    def create_with_source(cls, value, source):
        # 安装与 source 相关的函数匹配守卫
        install_guard(source.make_guard(GuardBuilder.FUNCTION_MATCH))
        # 使用给定的 value 和 source 创建类的实例
        return cls(
            value,
            source=source,
        )

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def reconstruct(self, codegen):
        try:
            # 尝试获取对象的完整名称
            name = f"{self.value.__module__}.{self.value.__name__}"
        except Exception:
            # 如果获取失败，使用对象的唯一标识符创建名称
            name = f"torch_obj_{id(self.value)}"
        # 根据名称生成唯一的变量名
        unique_var_name = "__" + re.sub(r"[^a-zA-Z0-9_]+", "_", name)
        # 扩展代码生成器的输出，设置全局缓存的变量
        codegen.extend_output(
            codegen.setup_globally_cached(unique_var_name, self.value)
        )

    def as_proxy(self):
        # 返回对象的值
        return self.value

    def python_type(self):
        # 返回对象的 Python 类型
        return type(self.value)

    def as_python_constant(self):
        # 返回对象的 Python 常量值
        return self.value

    def call_hasattr(self, tx, name):
        # 检查对象是否具有指定名称的属性
        result = hasattr(self.value, name)
        # 创建一个常量变量并返回结果
        return variables.ConstantVariable.create(result)

    def can_constant_fold_through(self):
        # 如果对象存在于 constant_fold_functions 中，则可以进行常量折叠
        if self.value in constant_fold_functions:
            return True
        # 否则检查对象的模块是否为 "math"
        return getattr(self.value, "__module__", None) == "math"


class TorchCtxManagerClassVariable(BaseTorchVariable):
    """Points to a context manager class in torch.* that dynamo has implementations"""

    def __repr__(self):
        # 返回包含值的 Torch 上下文管理器类变量的字符串表示形式
        return f"TorchCtxManagerClassVariable({self.value})"
    # 判断给定的值是否符合上下文管理器的条件
    def is_matching_cls(value):
        # 如果 value 是 functools.lru_cache 的包装器，则解包它
        value = unwrap_if_wrapper(value)
        # 无法使用 isinstance(value, type) 检查，因为某些上下文管理器
        # 是通过 contextlib.contextmanager 装饰的函数实现的，
        # 例如 torch._functorch.vmap.vmap_increment_nesting。
        return (
            # value 是可调用的
            callable(value)
            and (
                hashable(value)  # 访问 value.__hash__() 方法
                and value in supported_ctx_manager_classes
            )
        )

    # 调用函数并传递参数
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    # 定义一个继承自 BaseTorchVariable 的类，表示应该放入 FX 图中的 torch 函数或方法
    class TorchInGraphFunctionVariable(BaseTorchVariable):
        """Points to a torch function/method that should be put in FX graph"""

        # 返回对象的字符串表示，格式化为 "TorchInGraphFunctionVariable(值)"
        def __repr__(self):
            return f"TorchInGraphFunctionVariable({self.value})"

        # 返回存储的函数或方法对象
        def get_function(self):
            return self.value

        # 静态方法，使用 LRU 缓存调用函数的结果
        @staticmethod
        @functools.lru_cache(None)
        # 调用存储的函数或方法
        def call_function(
            self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
        ) -> "VariableTracker":
            # 导入所需的变量类型和模块
            from . import ConstantVariable, SymNodeVariable, TensorVariable
            from .builder import wrap_fx_proxy

            # 如果可以进行常量折叠，并且参数是未指定或常量，则进行常量折叠
            if self.can_constant_fold_through() and check_unspec_or_constant_args(
                args, kwargs
            ):
                return ConstantVariable.create(
                    self.as_python_constant()(
                        *[x.as_python_constant() for x in args],
                        **{k: v.as_python_constant() for k, v in kwargs.items()},
                    ),
                )

            # 获取特殊处理函数，如果存在则调用它处理函数或方法
            special_handler = self._get_handlers().get(self.value)
            if special_handler:
                result = special_handler(self, tx, *args, **kwargs)
                if result:
                    return result

            # 如果可以调度 torch 函数，则调度它
            if can_dispatch_torch_function(tx, args, kwargs):
                return dispatch_torch_function(tx, self, args, kwargs)
            else:
                # 检查是否存在 SymNodeVariable 类型的参数
                any_symints_or_symfloats = any(isinstance(x, SymNodeVariable) for x in args)

                # 检查所有参数是否都是 ConstantVariable 或 SymNodeVariable 类型
                all_ints_or_floats = all(
                    isinstance(x, (variables.ConstantVariable, variables.SymNodeVariable))
                    for x in args
                )

                # 如果函数或方法属于 torch 模块的二元操作，并且包含 SymNodeVariable 参数，则报错
                if (
                    getattr(self.value, "__module__", "") == "torch"
                    and self.value.__name__ in bin_ops
                    and any_symints_or_symfloats
                    and all_ints_or_floats
                ):
                    msg = f"""\
Calling {str(self.value)} on only torch.SymInt arguments is not yet supported.
To support this behavior, we need to allow const-propping tensors that store symint data.
For now, dynamo will explicitly graph break when it encounters user code with this behavior.
        """
        # 发出警告日志信息
        log.warning(msg)
        # 调用未实现的函数，传入消息作为参数
        unimplemented(msg)

    # TODO(voz): Replace w/ dynamic shape rewrite table.
    # Ideally, we would be able to do this at ctor time, but alas we need a combination
    # of value + args to determine this.
    # 获取当前实例的值
    fn_ = self.value
    # 如果存在任何符号整数或符号浮点数
    if any_symints_or_symfloats:
        # 构建 torch 符号操作的名称
        torch_sym_op = f"_sym_{self.value.__name__}"
        # 如果当前实例的模块为 "math" 并且 torch 包含对应的 torch_sym_op 方法
        if getattr(self.value, "__module__", None) == "math" and hasattr(
            torch, torch_sym_op
        ):
            # 更新 fn_ 为对应的 torch_sym_op 方法
            fn_ = getattr(torch, torch_sym_op)

    # 使用 wrap_fx_proxy 函数包装生成代理的调用函数
    tensor_variable = wrap_fx_proxy(
        tx=tx,
        # 使用 tx.output.create_proxy 创建代理对象，类型为 "call_function"
        # 参数包括函数 fn_ 和其余参数及关键字参数的代理
        proxy=tx.output.create_proxy(
            "call_function",
            fn_,
            *proxy_args_kwargs(args, kwargs),
        ),
    )

    # 如果 tensor_variable 是 TensorVariable 类型，并且 kwargs 中包含 "requires_grad" 字段并且其值可转为 Python 常量
    if (
        isinstance(tensor_variable, TensorVariable)
        and "requires_grad" in kwargs
        and kwargs["requires_grad"].as_python_constant()
    ):
        # 调用未实现的函数，传入错误消息
        unimplemented(
            """factory functions that return tensors that require grad are not supported.
    def _call_ntuple(self, tx, args, kwargs):
        """inline behavior of torch.nn.modules.utils._ntuple"""
        # 如果 self.value 是 torch.nn.modules.utils._ntuple 函数
        if self.value is torch.nn.modules.utils._ntuple:
            # 获取参数 args[0] 的 Python 常量值作为 count
            count = args[0].as_python_constant()
        else:
            # 否则，获取 self.value 的 __closure__ 中的第一个元素作为 count
            count = self.value.__closure__[0].cell_contents
        # 断言 count 是整数类型
        assert isinstance(count, int)
        # 断言 kwargs 为空

        # 定义处理 ntuple 的函数
        def handle_ntuple(value):
            # 如果 value 包含解包变量序列
            if value.has_unpack_var_sequence(tx):
                # 返回一个 TupleVariable 对象，包含解包后的变量列表
                return variables.TupleVariable(
                    list(value.unpack_var_sequence(tx)),
                )
            # 如果 value 是 Python 常量
            elif value.is_python_constant():
                # 常量传播
                return variables.ConstantVariable.create(
                    torch.nn.modules.utils._ntuple(count)(value.as_python_constant()),
                )
            else:
                # 调用未实现的函数，传入错误消息
                unimplemented(f"torch.nn.modules.utils._ntuple({value})")

        # 如果 self.value 是 torch.nn.modules.utils._ntuple 函数
        if self.value is torch.nn.modules.utils._ntuple:
            # 返回 LambdaVariable 包装的 handle_ntuple 函数
            return variables.LambdaVariable(handle_ntuple)
        else:
            # 否则，直接调用 handle_ntuple 处理 args[0]，并返回结果
            return handle_ntuple(args[0])

    @classmethod
    ```
    # 定义一个静态方法，用于创建或处理 torch.nn.Parameter 对象
    def call_nn_parameter(cls, tx, data=None, requires_grad=True):
        """A call to torch.nn.Parameter() gets lifted to before the graph"""
        # 如果 tx.export 标志位为真，暂不支持导出时创建 nn 参数
        if tx.export:
            unimplemented("nn parameter construction not supported with export")

        # 如果 requires_grad 是 VariableTracker 对象，则尝试获取其常量值
        if isinstance(requires_grad, variables.VariableTracker):
            try:
                requires_grad = requires_grad.as_python_constant()
            except NotImplementedError:
                unimplemented("Parameter(requires_grad=...) not constant")

        # 如果 data 不是 TensorVariable 对象，则暂不支持该类型的参数
        if not isinstance(data, variables.TensorVariable):
            unimplemented(f"Parameter(data={data}) not implemented")

        # 如果 data 来源已知，则通过指定前缀插入方式创建 nn 参数
        if data.source:
            return cls._nn_param_via_prefix_insert(tx, data, requires_grad)

        # 如果无法转换为可追踪的参数，则暂时不支持 nn_parameter 构建
        if not can_convert_to_tracable_parameter():
            unimplemented("Workaround for issues with nn_parameter construction")

        try:
            # 尝试获取 data 的形状、数据类型和设备信息
            shape = tuple(data.var_getattr(tx, "shape").as_python_constant())
            dtype = data.var_getattr(tx, "dtype").as_python_constant()
            device = data.var_getattr(tx, "device").as_python_constant()
        except NotImplementedError as e:
            unimplemented(f"Parameter not python_constant: {e}")

        # 在图中创建一个合成的参数占位符
        placeholder = tx.output.synthetic_graph_input(
            new_parameter_placeholder, [shape, dtype, device, requires_grad]
        )

        # 如果 data 需要梯度，则对其进行 detach 处理
        if data.requires_grad:
            data = data.call_method(tx, "detach", [], {})

        # 导入 wrap_fx_proxy 函数，并用其创建一个 fx 代理对象
        from .builder import wrap_fx_proxy

        result = wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_function",
                tracable_create_parameter,
                (data.as_proxy(), placeholder.as_proxy()),
                {},
            ),
        )
        assert isinstance(result, variables.TensorVariable)
        result.class_type = torch.nn.Parameter

        # 解决 tracable_create_paramter 存在的问题，强制将 has_grad_fn 设为 False
        result.has_grad_fn = False

        # 在 reconstruct() 中应使用原始的参数，返回的将是一个别名
        result.source = placeholder.source

        # TODO(jansel): 如果新参数超出作用域，当前实现下直到图的结束才会释放。需要修复这个问题。
        return result
    def _nn_param_via_prefix_insert(tx, data, requires_grad):
        # 如果有 .source，则使用备用版本
        from .builder import VariableBuilder  # 导入变量构建器模块

        varname = tx.output.new_var()  # 创建一个新变量名

        # 构建 nn.Parameter 并将其存储到 varname 中
        cg = PyCodegen(tx)  # 创建 PyCodegen 对象
        cg.add_push_null(lambda: cg.load_import_from("torch.nn", "Parameter"))  # 添加空值推送和加载 torch.nn 中的 Parameter
        cg(data.source)  # 添加数据源
        cg(variables.ConstantVariable(requires_grad))  # 添加需要梯度的常量变量
        cg.call_function(2, False)  # 调用函数，参数为 2 个，不使用返回值
        cg.store(varname)  # 存储结果到 varname
        tx.output.pregraph_bytecode.extend(cg.get_instructions())  # 扩展输出前图字节码

        data_node = data.as_proxy().node  # 获取数据的代理节点
        if data_node.op not in ("placeholder", "get_attr"):
            unimplemented(
                "Unexpected type of data placeholder op for parameter construction"
            )  # 如果数据占位符操作类型不是 "placeholder" 或 "get_attr"，则报未实现的错误

        # 将新构建的 nn.Parameter 作为图输入添加
        source = SyntheticLocalSource(varname)  # 创建合成本地源
        example_value = torch.nn.Parameter(
            tx.output.example_value_from_input_node(data.as_proxy().node)
        )  # 从输入节点生成示例值
        result = VariableBuilder(tx, source)(example_value)  # 使用变量构建器构建结果
        # 无需对此进行保护，因为我们已经在 `data` 上进行了保护。
        # 这些保护会失败，因为 varname 直到函数开始后才存在
        TracingContext.get().guards_context.dynamo_guards.remove_guards_with_source(
            source
        )  # 移除与源相关的守卫
        return result  # 返回结果
```