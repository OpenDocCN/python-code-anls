# `.\pytorch\torch\_subclasses\functional_tensor.py`

```
# mypy: allow-untyped-defs
# 引入 contextlib 模块，提供了用于管理上下文的实用工具
import contextlib
# 引入 warnings 模块，用于管理警告信息
import warnings
# 从 abc 模块中导入 ABC 抽象基类和 abstractmethod 抽象方法装饰器
from abc import ABC, abstractmethod
# 导入类型提示相关的模块和类型定义
from typing import Any, Callable, ContextManager, Dict, Optional, Tuple, Union

# 导入 PyTorch 模块
import torch
# 导入 PyTorch 内部模块 _pytree
import torch.utils._pytree as pytree
# 从 torch._C 模块导入 _functionalization_reapply_views_tls 和 _get_dispatch_mode_pre_dispatch
from torch._C import _functionalization_reapply_views_tls as _reapply_views
from torch._ops import _get_dispatch_mode_pre_dispatch
# 从 torch.utils._python_dispatch 模块导入多个函数和类
from torch.utils._python_dispatch import (
    _detect_infra_mode,
    _disable_infra_mode,
    return_and_correct_aliasing,
    TorchDispatchMode,
)

# 获取一个日志记录器，用于记录未实现的日志
not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")

# NOTE Some special handling for tensor conversion during export is needed.
# Normally, when tracing through the model with tensor.to(), the maybe-aliasing
# relationship between input and output tensors will be baked into the graph.
# For example, if we got a tensor with device cpu and call tensor.to("cpu"),
# it will become a no-op in the graph. For a whole graph capture, this is not
# sound so we need to do something different. Instead, in export we will try to
# preserve the tensor conversion by forcing a non-semantic-breaking aten::_to_copy
# operator to be traced in the graph, and subsequently banning mutations on all
# such converted tensors.
# In addition to patching .to() method call in functionalization, we will have to
# patch other similar methods like float() and cpu(), because they intentionally
# don't fall back to .to() methods, but have the same behavior as .to() according to
# pytorch document. https://pytorch.org/docs/stable/generated/torch.Tensor.float.html
# thus we simply force them to go through .to() call.
# 定义一个模板方法 _conversion_method_template，用于生成转换方法
def _conversion_method_template(**extra_kwargs):
    # 内部方法，接受 self 参数，并将其它参数传递给 tensor.to 方法
    def _(self, *args, **kwargs):
        return self.to(*args, **{**kwargs, **extra_kwargs})
    
    return _

# 定义 FunctionalTensor 类，继承自 torch.Tensor
class FunctionalTensor(torch.Tensor):
    """
    Functional tensors represent tensors that will remove mutations
    from a program. If you perform a mutable operation on a functional tensor,
    it will re-dispatch to the functional variant of that operation.

    Historically, functionalization is implemented in C++ in the dispatcher.
    This class is a lightweight python shim around the C++ functionalization logic.

    FunctionalTensor is required to be used with a corresponding
    FunctionalTensormode active, because it relies
    on using the mode for dispatch (which can properly handle factory functions).
    """
    
    elem: torch.Tensor
    # 用于 torch_dispatch 分发基础设施的标识符，表示这是一个“基础设施”模式，具有较低的分发优先级
    _mode_key = torch._C._TorchDispatchModeKey.FUNCTIONAL
    
    # Note: The reason we add these extra keys to our FunctionalTensor subclass
    # is to mirror the behavior of C++ functionalization (we can choose to change this
    # later, as long as it doesn't break anything).
    # FunctionalTensorWrapper copies **all** dispatch keys from the inner tensor
    # 将 ZeroTensor 作为一个额外的分发键添加到包装张量的属性中，但排除 functorch 和 python 分发键。
    # 这里我尝试重用 functorch 包装器子类复制的键集，不过它们不包括 ZeroTensor，所以我手动添加了它。
    _extra_dispatch_keys = torch._C._additional_keys_to_prop_for_wrapper_tensors.add(
        torch._C.DispatchKey.ZeroTensor
    )

    # 这些是所有对应于元数据查询的 aten 操作。
    # 我们希望 FunctionalTensor 能够直接处理它们。
    metadata_fns = [
        torch.ops.aten.is_contiguous.default,  # type: ignore[has-type]
        torch.ops.aten.is_contiguous.memory_format,  # type: ignore[has-type]
        torch.ops.aten.is_strides_like_format.default,  # type: ignore[has-type]
        torch.ops.aten.is_non_overlapping_and_dense.default,  # type: ignore[has-type]
        torch.ops.aten.size.default,  # type: ignore[has-type]
        torch.ops.aten.sym_size.default,  # type: ignore[has-type]
        torch.ops.aten.stride.default,  # type: ignore[has-type]
        torch.ops.aten.sym_stride.default,  # type: ignore[has-type]
        torch.ops.aten.storage_offset.default,  # type: ignore[has-type]
        torch.ops.aten.sym_storage_offset.default,  # type: ignore[has-type]
        torch.ops.aten.numel.default,  # type: ignore[has-type]
        torch.ops.aten.sym_numel.default,  # type: ignore[has-type]
        torch.ops.aten.dim.default,  # type: ignore[has-type]
        torch.ops.prim.device.default,  # type: ignore[has-type]
    ]

    # 这些操作声称是函数式的，但实际上可能是变异的或可能是别名的。
    # TODO（tmanlaibaatar）将其作为一个标记
    maybe_aliasing_or_mutating_ops = [
        torch.ops.aten.dropout.default,  # type: ignore[has-type]
        torch.ops.aten.batch_norm.default,  # type: ignore[has-type]
        torch.ops.aten.native_batch_norm.default,  # type: ignore[has-type]
        torch.ops.aten._batch_norm_impl_index.default,  # type: ignore[has-type]
        torch.ops.aten.cudnn_batch_norm.default,  # type: ignore[has-type]
        torch.ops.aten.miopen_batch_norm.default,  # type: ignore[has-type]
        torch.ops.aten.atleast_1d.default,  # type: ignore[has-type]
        torch.ops.aten.atleast_2d.default,  # type: ignore[has-type]
        torch.ops.aten.atleast_3d.default,  # type: ignore[has-type]
        torch.ops.aten.cartesian_prod.default,  # type: ignore[has-type]
        torch.ops.aten.conj_physical.default,  # type: ignore[has-type]
        torch.ops.aten.alpha_dropout.default,  # type: ignore[has-type]
        torch.ops.aten.feature_dropout.default,  # type: ignore[has-type]
        torch.ops.aten.feature_alpha_dropout.default,  # type: ignore[has-type]
        torch.ops.aten.unsafe_chunk.default,  # type: ignore[has-type]
    ]
    # 定义一个特殊方法 `__new__`，用于创建新的对象实例
    def __new__(cls, elem):
        # 断言传入的 elem 是一个 functional tensor，否则会触发 AssertionError
        assert torch._is_functional_tensor(elem)

        # 确定需要额外分发的关键字，这些关键字需要在内部和外部张量之间保持一致
        extra_dispatch_keys = (
            FunctionalTensor._extra_dispatch_keys & torch._C._dispatch_keys(elem)
        )

        # 使用 torch.Tensor._make_wrapper_subclass 方法创建一个张量的子类实例
        out = torch.Tensor._make_wrapper_subclass(  # type: ignore[arg-type, attr-defined]
            # TODO: 目前 _make_wrapper_subclass 对动态形状的交互支持不好。
            # 使用带有 kwargs 的重载会导致我们进入第一条重载路径，
            # 它将始终专门化尺寸。我们可能最终应该修复这个问题，
            # 以便第一条重载可以处理动态形状。
            cls,
            elem.shape,  # sizes，张量的形状
            elem.stride(),  # strides，张量的步幅
            elem.storage_offset(),  # storage_offset，张量的存储偏移量
            None,  # memory_format，内存格式，这里为 None
            elem.dtype,  # dtype，张量的数据类型
            elem.layout,  # layout，张量的布局
            elem.device,  # device，张量所在的设备
            False,  # pin_memory，是否锁定内存
            elem.requires_grad,  # requires_grad，张量是否需要梯度
            "sizes",  # dispatch_sizes_strides_policy，分发策略
            False,  # dispatch_device，分发设备
            False,  # dispatch_layout，分发布局
            extra_dispatch_keys,  # _extra_dispatch_keys，额外的分发关键字
        )
        
        # 设置 out 对象在有可变数据指针时抛出异常
        torch._C._set_throw_on_mutable_data_ptr(out)
        
        # 将传入的 elem 存储在 out 对象的 elem 属性中
        out.elem = elem
        
        # 返回创建的 out 对象实例
        return out
    # 实现 __torch_dispatch__ 方法，用于处理 Torch 框架中的函数调度
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 检查是否存在未识别的子类类型
        unrecognized_types = [
            t
            for t in types
            if t not in [torch.Tensor, torch._subclasses.FakeTensor, FunctionalTensor]
        ]
        # 如果存在未识别的子类类型，则记录调试信息并返回未实现
        if unrecognized_types:
            not_implemented_log.debug(
                "FunctionalTensor unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        # 如果 kwargs 为 None，则初始化为空字典
        if kwargs is None:
            kwargs = {}

        # 如果 func 在 metadata_fns 中，则处理所有的元数据请求到内部张量
        if func in FunctionalTensor.metadata_fns:
            # 所有的元数据访问应该传递给内部张量，以确保包装器和内部张量之间的元数据同步
            assert len(kwargs) == 0
            # 根据不同的 func 进行处理不同的操作
            if func in [
                torch.ops.aten.is_strides_like_format.default,
                torch.ops.aten.is_contiguous.memory_format,
            ]:
                assert len(args) == 2 and isinstance(args[0], FunctionalTensor)
                # 调用 func，传入内部张量和其他参数
                return func(torch._from_functional_tensor(args[0].elem), args[1])
            assert len(args) == 1 and isinstance(args[0], FunctionalTensor)
            # 调用 func，传入内部张量
            return func(torch._from_functional_tensor(args[0].elem))
        
        # 如果不是以上情况，则抛出运行时错误，提示使用正确的 FunctionalTensorMode()
        raise RuntimeError(
            "Attempting to use FunctionalTensor on its own. Instead, please use it with a corresponding FunctionalTensorMode()"
        )

    # 实现 __repr__ 方法，返回对象的字符串表示
    def __repr__(self):
        return f"FunctionalTensor({repr(self.elem)})"
    def to_functional(x):
        # 用户无需自行进行封装
        assert not torch._is_functional_tensor(x)
        # FunctionalTensor 上我们关心的唯一自动求导元数据是：
        # - requires_grad（使得自动求导运行）
        # - is_leaf（这样可以通过自动求导引擎允许对图输入的变异，即使它们不是叶子节点）
        #   这由 FunctionalTensor.to_functional 处理
        x_functional = torch._to_functional_tensor(x)
        # 这里的 FunctionalTensor 模式实际上是不必要的，
        # 但它避免了在 `ProxyTorchDispatchMode` 跟踪期间产生不必要的 NotImplemented 日志。
        # _mirror_autograd_meta_to 查询张量大小，
        # 并且除此之外，sym_size() 调用会进入代理模式，然后再进入 FunctionalTensor.__torch_dispatch__

        functional_mode = _detect_infra_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL)
        assert functional_mode is not None

        with functional_mode:
            torch._mirror_autograd_meta_to(x, x_functional)  # type: ignore[attr-defined]
            out = FunctionalTensor(x_functional)
            torch._mirror_autograd_meta_to(x_functional, out)  # type: ignore[attr-defined]
        return out

    def from_functional(self):
        torch._sync(self)
        return torch._from_functional_tensor(self.elem)

    def replace_(self, output) -> None:
        torch._functionalize_replace(self.elem, output)

    def commit_update(self) -> None:
        torch._functionalize_commit_update(self.elem)

    def sync(self) -> None:
        torch._functionalize_sync(self.elem)

    def mark_mutation_hidden_from_autograd(self) -> None:
        torch._functionalize_mark_mutation_hidden_from_autograd(self.elem)

    def tolist(self) -> Any:
        if self.elem.dim() == 0:
            return self.elem.item()
        elif self.elem.dim() == 1:
            return [elem.item() for elem in self.elem]
        else:
            return [elem.tolist() for elem in self.elem]

    def to(self, *args, **kwargs):
        if _detect_infra_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL).export:
            # 如果将复制指定为位置参数，则始终是第二个参数。
            if len([arg for arg in args if isinstance(arg, bool)]) <= 1:
                return super().to(*args, **{**kwargs, "copy": True})
        return super().to(*args, **kwargs)

    def cuda(self, device=None, *args, **kwargs):
        device = device or torch.cuda.current_device()
        if len(args) > 0:
            return self.to(device, *args, **kwargs)
        else:
            return self.to(device=device, **kwargs)

    char = _conversion_method_template(dtype=torch.int8)
    cpu = _conversion_method_template(device=torch.device("cpu"))
    bfloat16 = _conversion_method_template(dtype=torch.bfloat16)
    byte = _conversion_method_template(dtype=torch.uint8)
    double = _conversion_method_template(dtype=torch.float64)
    # 使用_conversion_method_template函数生成针对不同dtype的转换方法，返回一个针对torch.float32的转换方法
    float = _conversion_method_template(dtype=torch.float32)
    # 使用_conversion_method_template函数生成针对不同dtype的转换方法，返回一个针对torch.bool的转换方法
    bool = _conversion_method_template(dtype=torch.bool)
    # 使用_conversion_method_template函数生成针对不同dtype的转换方法，返回一个针对torch.float16的转换方法
    half = _conversion_method_template(dtype=torch.float16)
    # 使用_conversion_method_template函数生成针对不同dtype的转换方法，返回一个针对torch.int32的转换方法
    int = _conversion_method_template(dtype=torch.int32)
    # 使用_conversion_method_template函数生成针对不同dtype的转换方法，返回一个针对torch.int64的转换方法
    long = _conversion_method_template(dtype=torch.int64)
class FunctionalTensorMode(TorchDispatchMode):
    # 定义一个新的类，继承自TorchDispatchMode，用于功能张量模式

    def __init__(self, pre_dispatch=False, export=False, _allow_token_discovery=False):
        # 初始化函数，设置各种属性
        self.export = export
        self.is_on_stack = False
        self.enter_stack = []
        # 表示这是一个“infra”模式，具有较低的调度优先级
        self._mode_key = torch._C._TorchDispatchModeKey.FUNCTIONAL
        self.pre_dispatch = pre_dispatch
        # 这将稍后用于预调度功能化时关闭
        self._dispatch_key = torch._C.DispatchKey.PreDispatch if pre_dispatch else None  # type: ignore[attr-defined]
        # 效果类型映射到一个令牌的字典，令牌有助于跟踪有副作用操作之间的顺序
        self._tokens: Dict[Any, torch.Tensor] = {}

        # 功能化在AOTAutograd中运行两次，一次在`run_functionalized_fw_and_collect_metadata`中收集元数据，
        # 看哪些张量需要功能化和发现需要多少令牌，另一次在`make_fx`中实际追踪以替换操作及处理有副作用的操作。
        # 第二阶段不应有令牌发现。此标志区分这两个阶段。
        self._allow_token_discovery = _allow_token_discovery

    # 如果FunctionalTensorMode已在使用中，则无操作
    def __enter__(self):
        def _get_prev_mode():
            if self._dispatch_key == torch._C.DispatchKey.PreDispatch:
                return _get_dispatch_mode_pre_dispatch(
                    torch._C._TorchDispatchModeKey.FUNCTIONAL
                )
            return torch._C._get_dispatch_mode(
                torch._C._TorchDispatchModeKey.FUNCTIONAL
            )

        if _get_prev_mode() is None:
            self.enter_stack.append(True)
            return super().__enter__()
        else:
            self.enter_stack.append(False)
            return self

    def __exit__(self, a, b, c):
        # 退出函数，管理模式的堆栈状态
        is_on_stack = self.enter_stack.pop()
        if is_on_stack:
            super().__exit__(a, b, c)

@contextlib.contextmanager
def disable_functional_mode():
    # 上下文管理器，用于禁用功能模式
    return _disable_infra_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL)


# 这类似于torch.func.functionalize，但是：
# - 它使用FunctionalTensorMode和FunctionalTensor（Python子类）。
#   使用此模式的一个重要优势是，它允许我们在__torch_dispatch__下运行功能化，这在AOTAutograd中是必需的。
# - 这样做意味着它不会自动与其他functorch转换组合，因为这些转换始终在__torch_dispatch__上运行。
#   这就是为什么这个实用程序存在于这里，而不是functorch中的原因。
def dispatch_functionalize(func, mode: FunctionalTensorMode = FunctionalTensorMode()):
    # 分派功能化函数，使用FunctionalTensorMode进行功能化处理
    # TODO: 从AOT Autograd中提取这些信息
    def to_fun(t):
        # 如果输入是 torch.Tensor 对象，则转换为 FunctionalTensor 对象
        if isinstance(t, torch.Tensor):
            return FunctionalTensor.to_functional(t)
        # 否则直接返回输入对象
        return t

    def from_fun(t):
        # 如果输入不是 FunctionalTensor 对象
        if not isinstance(t, FunctionalTensor):
            # 快速进行健全性断言
            if isinstance(t, torch.Tensor):
                assert not torch._is_functional_tensor(t)
            # 直接返回输入对象
            return t
        # 对 FunctionalTensor 对象进行同步操作
        torch._sync(t)
        # 从 FunctionalTensor 中获取其元素的 torch.Tensor 对象，并返回
        return torch._from_functional_tensor(t.elem)

    def inner(*args, **kwargs):
        # 创建一个排除功能化 dispatch key 的上下文管理器
        disable_above = torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        )
        # 在 disable_above 上下文中执行功能化模式
        with disable_above, mode:
            # 将输入参数 args 中的所有 torch.Tensor 对象转换为 FunctionalTensor 对象
            func_args = pytree.tree_map_only(torch.Tensor, to_fun, args)
            # 将输入参数 kwargs 中的所有 torch.Tensor 对象转换为 FunctionalTensor 对象
            func_kwargs = pytree.tree_map_only(torch.Tensor, to_fun, kwargs)
            # 调用 func 函数，并传入转换后的参数和关键字参数，获取功能化输出
            func_outputs = func(*func_args, **func_kwargs)
            # 将 func_outputs 中的所有 FunctionalTensor 对象转换回 torch.Tensor 对象
            outputs = pytree.tree_map_only(FunctionalTensor, from_fun, func_outputs)

            # 返回转换后的输出
            return outputs

    # 返回 inner 函数作为最终的功能化处理函数
    return inner
# 定义一个抽象基类 BaseFunctionalizeAPI，继承自 ABC
class BaseFunctionalizeAPI(ABC):
    
    # 声明抽象方法 wrap_tensors，用于封装张量
    @abstractmethod
    def wrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        pass
    
    # 声明抽象方法 unwrap_tensors，用于解封张量
    @abstractmethod
    def unwrap_tensors(
        self, args: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        pass
    
    # 声明抽象方法 functionalize，接受一个内部函数并返回一个装饰后的函数
    @abstractmethod
    def functionalize(self, inner_f: Callable) -> Callable:
        pass
    
    # 声明抽象方法 redispatch_to_next，返回一个上下文管理器
    @abstractmethod
    def redispatch_to_next(self) -> ContextManager:
        pass
    
    # 声明抽象方法 replace，用于替换输入张量的内容
    @abstractmethod
    def replace(self, input_tensor, output_tensor) -> None:
        pass
    
    # 声明抽象方法 commit_update，用于提交更新到张量
    @abstractmethod
    def commit_update(self, tensor) -> None:
        pass
    
    # 声明抽象方法 sync，用于同步张量的状态
    @abstractmethod
    def sync(self, tensor) -> None:
        pass
    
    # 声明抽象方法 mark_mutation_hidden_from_autograd，用于标记张量的变异隐藏状态
    @abstractmethod
    def mark_mutation_hidden_from_autograd(self, tensor) -> None:
        pass


# 定义具体类 PythonFunctionalizeAPI，继承自 BaseFunctionalizeAPI
class PythonFunctionalizeAPI(BaseFunctionalizeAPI):
    
    # 初始化方法，接受一个可选的 FunctionalTensorMode 和一个布尔值 pre_dispatch
    def __init__(
        self, mode: Optional[FunctionalTensorMode] = None, pre_dispatch: bool = False
    ) -> None:
        super().__init__()
        # 如果未提供 mode，则使用默认的 FunctionalTensorMode 对象
        self.mode = mode if mode else FunctionalTensorMode()
        # 设置是否预调度的标志
        self.pre_dispatch = pre_dispatch
    
    # 实现 wrap_tensors 方法，用于封装传入的参数张量 args
    def wrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        # 使用 self.mode 上下文环境，封装张量成为功能张量
        with self.mode:
            return torch.utils._pytree.tree_map_only(
                torch.Tensor, FunctionalTensor.to_functional, args
            )
    
    # 实现 unwrap_tensors 方法，用于解封传入的参数张量 args
    def unwrap_tensors(
        self, args: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return torch.utils._pytree.tree_map_only(
            FunctionalTensor, FunctionalTensor.from_functional, args
        )
    
    # 实现 functionalize 方法，接受一个内部函数 inner_f 并返回装饰后的函数
    def functionalize(self, inner_f: Callable) -> Callable:
        return dispatch_functionalize(inner_f, self.mode)
    
    # 实现 redispatch_to_next 方法，返回一个上下文管理器，用于环境中没有执行操作
    def redispatch_to_next(self) -> ContextManager:
        # [注意] 这里不执行任何操作，因为在此路径上执行时，我们已经从模式堆栈中弹出 FunctionalTensorMode。
        # 由于 FunctionalTensorMode 现在是有状态的，最好直接传入正确的模式而不是全局设置。
        return contextlib.nullcontext()
    
    # 实现 replace 方法，用于替换输入张量 input_tensor 的内容为 output_tensor
    def replace(self, input_tensor, output_tensor) -> None:
        assert isinstance(input_tensor, FunctionalTensor)
        assert not isinstance(output_tensor, FunctionalTensor)
        input_tensor.replace_(output_tensor)
    
    # 实现 commit_update 方法，用于提交更新到张量 tensor
    def commit_update(self, tensor) -> None:
        assert isinstance(tensor, FunctionalTensor)
        tensor.commit_update()
    
    # 实现 sync 方法，用于同步张量 tensor 的状态
    def sync(self, tensor) -> None:
        assert isinstance(tensor, FunctionalTensor)
        tensor.sync()
    
    # 实现 mark_mutation_hidden_from_autograd 方法，用于标记张量 tensor 的变异隐藏状态
    def mark_mutation_hidden_from_autograd(self, tensor) -> None:
        assert isinstance(tensor, FunctionalTensor)
        tensor.mark_mutation_hidden_from_autograd()


# 定义具体类 CppFunctionalizeAPI，继承自 BaseFunctionalizeAPI，但未实现任何方法
class CppFunctionalizeAPI(BaseFunctionalizeAPI):
    pass
    # 将输入参数中的所有张量包装成函数化张量
    def wrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        from torch._functorch.eager_transforms import _wrap_all_tensors_to_functional
        
        return _wrap_all_tensors_to_functional(args, level=0)

    # 从函数化张量中解包输入参数中的所有张量
    def unwrap_tensors(
        self, args: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        from torch._functorch.eager_transforms import (
            _unwrap_all_tensors_from_functional,
        )
        
        return _unwrap_all_tensors_from_functional(args, reapply_views=_reapply_views())

    # 将指定的内部函数 functionalize 化，返回一个包装后的函数
    def functionalize(self, inner_f: Callable) -> Callable:
        return torch.func.functionalize(inner_f)

    # 返回一个上下文管理器，用于排除 DispatchKey 为 Functionalize 的操作
    def redispatch_to_next(self) -> ContextManager:
        return torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        )

    # 将输入张量中的部分或全部内容用输出张量替换
    def replace(self, input_tensor, output_tensor) -> None:
        torch._functionalize_replace(input_tensor, output_tensor)

    # 提交更新，将函数化张量的变化同步到原始张量
    def commit_update(self, tensor) -> None:
        torch._functionalize_commit_update(tensor)

    # 将函数化张量的变化同步到原始张量
    def sync(self, tensor) -> None:
        torch._functionalize_sync(tensor)

    # 标记张量的突变操作对自动求导不可见
    def mark_mutation_hidden_from_autograd(self, tensor) -> None:
        torch._functionalize_mark_mutation_hidden_from_autograd(tensor)
# 定义一个 FunctorchFunctionalizeAPI 类，继承自 BaseFunctionalizeAPI 类
class FunctorchFunctionalizeAPI(BaseFunctionalizeAPI):
    
    # 初始化方法，接受一个解释器对象作为参数
    def __init__(self, interpreter):
        self.interpreter = interpreter

    # wrap_tensors 方法，将输入的张量包装为功能化的张量
    def wrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        # 导入 _wrap_all_tensors_to_functional 函数，用于包装所有张量到功能化张量
        from torch._functorch.eager_transforms import _wrap_all_tensors_to_functional

        # 调用 _wrap_all_tensors_to_functional 函数，并传入 args 和解释器的级别作为参数
        return _wrap_all_tensors_to_functional(args, level=self.interpreter.level())

    # unwrap_tensors 方法，将功能化的张量解包为普通张量
    def unwrap_tensors(
        self, args: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # 导入 _unwrap_all_tensors_from_functional 函数，用于解包功能化张量
        from torch._functorch.eager_transforms import (
            _unwrap_all_tensors_from_functional,
        )

        # 调用 _unwrap_all_tensors_from_functional 函数，并传入 args 和是否重新应用视图作为参数
        return _unwrap_all_tensors_from_functional(
            args, reapply_views=self.interpreter.functionalize_add_back_views()
        )

    # functionalize 方法，对内部函数进行功能化处理，返回功能化后的可调用对象
    def functionalize(self, inner_f: Callable) -> Callable:
        # 调用 torch.func.functionalize 函数，将内部函数功能化，根据解释器的设置决定移除哪些操作
        return torch.func.functionalize(
            inner_f,
            remove="mutations_and_views"
            if self.interpreter.functionalize_add_back_views()
            else "mutations",
        )

    # redispatch_to_next 方法，返回解释器的下一级上下文管理器
    def redispatch_to_next(self) -> ContextManager:
        return self.interpreter.lower()

    # replace 方法，用新的输出张量替换输入张量
    def replace(self, input_tensor, output_tensor) -> None:
        # 调用 torch._functionalize_replace 函数，实现替换操作
        torch._functionalize_replace(input_tensor, output_tensor)

    # commit_update 方法，提交对张量的更新
    def commit_update(self, tensor) -> None:
        # 调用 torch._functionalize_commit_update 函数，提交对张量的更新
        torch._functionalize_commit_update(tensor)

    # sync 方法，同步张量的操作
    def sync(self, tensor) -> None:
        # 调用 torch._functionalize_sync 函数，同步张量的操作
        torch._functionalize_sync(tensor)

    # mark_mutation_hidden_from_autograd 方法，标记张量的变异操作对自动求导不可见
    def mark_mutation_hidden_from_autograd(self, tensor) -> None:
        # 调用 torch._functionalize_mark_mutation_hidden_from_autograd 函数，标记张量的变异操作对自动求导不可见
        torch._functionalize_mark_mutation_hidden_from_autograd(tensor)
```