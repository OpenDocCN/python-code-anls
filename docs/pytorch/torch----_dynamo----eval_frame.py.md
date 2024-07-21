# `.\pytorch\torch\_dynamo\eval_frame.py`

```py
# mypy: allow-untyped-defs
# mypy: disable-error-code="method-assign"

"""
Functions in this file are responsible for modifying the eval frame
handler at RUNTIME.  Therefore, all functions in this file are hot.
Functions that only execute at compile time should be placed
in torch._dynamo.convert_frame.
"""

# 引入必要的模块和库
import contextlib  # 上下文管理模块，用于创建上下文管理器
import functools  # 函数工具模块，提供了装饰器和函数修饰符
import inspect  # 检查对象模块，用于获取对象信息
import logging  # 日志记录模块，用于生成日志信息
import os  # 操作系统模块，提供了与操作系统交互的功能
import sys  # 系统特定参数和函数模块，用于操作 Python 运行时环境
import textwrap  # 文本包装模块，用于格式化和填充文本段落
import traceback  # 追溯异常模块，用于获取异常的回溯信息
import types  # 类型操作模块，提供了动态创建和操作 Python 类型的功能
import warnings  # 警告处理模块，用于管理警告信息
import weakref  # 弱引用模块，用于创建弱引用对象

from enum import Enum  # 枚举模块，用于创建枚举类型
from os.path import dirname, join  # 路径操作函数，用于获取文件路径的目录名和连接路径
from typing import (  # 类型提示模块，用于类型注解和类型检查
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)
from unittest.mock import patch  # 单元测试模块，用于模拟对象

import torch  # PyTorch 主模块
import torch.fx  # PyTorch FX 模块
import torch.utils._pytree as pytree  # PyTorch 工具模块，用于处理 Python 树形数据结构
import torch.utils.checkpoint  # PyTorch 工具模块，用于模型检查点管理
from torch import _guards  # PyTorch 内部工具模块，用于管理保护对象
from torch._utils_internal import justknobs_check, log_export_usage  # PyTorch 内部工具模块，用于导出日志和检查控制
from torch.export.dynamic_shapes import _process_dynamic_shapes  # PyTorch 导出模块，处理动态形状
from torch.fx.experimental.proxy_tensor import (  # PyTorch FX 实验模块，代理张量功能
    make_fx,
    maybe_disable_fake_tensor_mode,
)
from torch.fx.experimental.symbolic_shapes import (  # PyTorch FX 实验模块，符号形状
    ConstraintViolationError,
    DimDynamic,
    StatelessSymbolicContext,
)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo  # PyTorch FX 图形模块，Python 树形代码生成和信息

from ..fx import GraphModule  # 相对导入，引入上级目录中的 FX 模块中的 GraphModule 类
from .backends.registry import CompilerFn, lookup_backend  # 相对导入，引入当前目录中的后端注册和编译器函数
from .hooks import Hooks  # 相对导入，引入当前目录中的 Hooks 类

# see discussion at https://github.com/pytorch/pytorch/issues/120699
reset_code = torch._C._dynamo.eval_frame.reset_code  # 获取 eval frame 的重置代码
set_eval_frame = torch._C._dynamo.eval_frame.set_eval_frame  # 设置 eval frame 的帧
set_guard_error_hook = torch._C._dynamo.eval_frame.set_guard_error_hook  # 设置保护错误钩子
skip_code = torch._C._dynamo.eval_frame.skip_code  # 跳过 eval frame 中的代码
unsupported = torch._C._dynamo.eval_frame.unsupported  # eval frame 中的不支持操作

from . import config, convert_frame, external_utils, trace_rules, utils  # 相对导入，引入当前目录中的配置、帧转换、外部工具、跟踪规则和实用工具
from .code_context import code_context  # 相对导入，引入当前目录中的代码上下文
from .exc import (  # 相对导入，引入当前目录中的异常模块
    CondOpArgsMismatchError,
    UserError,
    UserErrorType,
)
from .mutation_guard import install_generation_tagging_init  # 相对导入，引入当前目录中的变异守护模块
from .utils import (  # 相对导入，引入当前目录中的实用工具
    common_constant_types,
    compile_times,
)

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

from torch._dispatch.python import enable_python_dispatcher  # PyTorch 分发模块，启用 Python 分发器

always_optimize_code_objects = utils.ExactWeakKeyDictionary()  # 使用实用工具中的弱引用字典

null_context = contextlib.nullcontext  # 空上下文管理器，用于不做任何操作的上下文

import sympy  # 符号计算模块，用于进行符号计算

if TYPE_CHECKING:
    from torch._subclasses import fake_tensor  # 类型检查中，导入伪张量
    from .types import CacheEntry, DynamoCallback  # 类型检查中，导入缓存条目和动态回调函数

# See https://github.com/python/typing/pull/240
class Unset(Enum):
    token = 0  # 未设置枚举类，包含一个 token 属性为 0

cached_backends: Dict[int, CompilerFn] = {}  # 缓存的后端字典，键为整数，值为编译器函数

unset = Unset.token  # 设置为未设置枚举类的 token 属性

def _reset_guarded_backend_cache():
    global cached_backends
    for backend in cached_backends.values():
        if hasattr(backend, "reset"):
            backend.reset()  # 如果后端具有 reset 方法，则重置它
    cached_backends.clear()  # 清空缓存的后端字典

DONT_WRAP_FILES = {
    # For tracing into fx modules
    inspect.getsourcefile(GraphModule),  # 获取 GraphModule 模块的源文件路径
    join(dirname(dirname(__file__)), "onnx/_internal/fx/dynamo_graph_extractor.py"),  # 拼接获取指定文件的路径
}
def _debug_get_cache_entry_list(
    code: Union[types.CodeType, Callable[..., Any]]
) -> List[CacheEntry]:
    """
    Given a code object or a callable object, retrieve the cache entries
     stored in this code.
    """
    # 如果传入的是可调用对象，则获取其对应的代码对象
    if callable(code):
        code = code.__code__
    # 调用 torch 库的私有函数 _debug_get_cache_entry_list，传入代码对象，返回缓存条目列表
    return torch._C._dynamo.eval_frame._debug_get_cache_entry_list(code)


class OptimizedModule(torch.nn.Module):
    """
    Wraps the original nn.Module object and later patches its
    forward method to optimized self.forward method.
    """

    _torchdynamo_orig_callable: Callable[..., Any]
    get_compiler_config: Callable[[], Any]

    _opt_mod_attributes = {
        "_orig_mod",
        "dynamo_ctx",
        "_torchdynamo_orig_callable",
        "get_compiler_config",
        "forward",
        "_forward",
        "__dict__",
        "named_children_walk",
    }

    def __init__(self, mod: torch.nn.Module, dynamo_ctx):
        super().__init__()
        # Installs the params/buffer
        # 将原始的 nn.Module 对象保存到 _orig_mod 属性中
        self._orig_mod = mod
        # 保存 dynamo_ctx 到对象属性中
        self.dynamo_ctx = dynamo_ctx
        # 调用初始化方法进行初始化操作
        self._initialize()

    def _initialize(self):
        # Do this stuff in constructor to lower overhead slightly
        # 如果 dynamo_ctx 是 DisableContext 类型，则设置 forward 方法为动态上下文的调用结果
        if isinstance(self.dynamo_ctx, DisableContext):
            # No need to check trace rules
            self.forward = self.dynamo_ctx(self._orig_mod.__call__)
        # 如果 _orig_mod 的 forward 方法是方法类型，并且符合 trace_rules.check 或者 _is_fsdp_managed_module 规则
        elif isinstance(self._orig_mod.forward, types.MethodType) and (
            trace_rules.check(self._orig_mod.forward)
            or getattr(self._orig_mod, "_is_fsdp_managed_module", False)
        ):
            # 设置 forward 方法为动态上下文的调用结果，包裹在 external_utils.wrap_inline 中
            self.forward = self.dynamo_ctx(external_utils.wrap_inline(self._orig_mod))
        else:
            # 在 dynamo 外部调用钩子，然后获取内部帧
            self.forward = self.dynamo_ctx(self._orig_mod.__call__)

        # 如果 _orig_mod 有 _initialize_hook 属性，则将 forward 方法保存到 _forward 属性中，并将 forward 方法设置为 _call_lazy_check 方法
        if hasattr(self._orig_mod, "_initialize_hook"):
            self._forward = self.forward
            self.forward = self._call_lazy_check

    def __reduce__(self):
        # 返回序列化对象时的参数元组，包含 _orig_mod 和 dynamo_ctx
        return (self.__class__, (self._orig_mod, self.dynamo_ctx))

    def __getstate__(self):
        # 返回对象状态的字典表示，删除 forward 和 __call__ 键
        state = dict(self.__dict__)
        state.pop("forward", None)
        state.pop("__call__", None)
        return state

    def __setstate__(self, state):
        # 设置对象状态为给定的状态字典，并重新初始化对象
        self.__dict__ = state
        self._initialize()

    def __getattr__(self, name):
        # 当访问属性为 _orig_mod 时，返回 _modules["_orig_mod"]，否则调用原始模块对象的 getattr 方法
        if name == "_orig_mod":
            return self._modules["_orig_mod"]
        return getattr(self._orig_mod, name)

    def __setattr__(self, name, val):
        # 允许覆盖类属性，如果属性名存在于 OptimizedModule._opt_mod_attributes 中，则调用父类的 setattr 方法设置属性值，否则调用原始模块对象的 setattr 方法
        if hasattr(type(self), name):
            return super().__setattr__(name, val)

        if name in OptimizedModule._opt_mod_attributes:
            return super().__setattr__(name, val)
        return setattr(self._orig_mod, name, val)
    # 调用懒加载检查的方法，接受任意位置参数和关键字参数
    def _call_lazy_check(self, *args, **kwargs):
        # 如果原始模块有 "_initialize_hook" 属性
        if hasattr(self._orig_mod, "_initialize_hook"):
            # 对于懒加载模块，我们希望运行预初始化钩子
            # 然后懒加载模块会删除其预初始化钩子，
            # 以避免在后续重新编译时将其视为懒加载。
            self._orig_mod._infer_parameters(self._orig_mod, args, kwargs)
        # 返回调用前向方法 "_forward" 处理后的结果
        return self._forward(*args, **kwargs)

    # 重写 "__dir__" 方法
    def __dir__(self):
        # 获取原始模块的所有属性列表
        orig_mod_attrs = self._orig_mod.__dir__()
        # 返回原始模块属性列表加上当前对象（super()）中不在原始模块属性列表中的属性
        return orig_mod_attrs + [
            attr for attr in super().__dir__() if attr not in orig_mod_attrs
        ]
# 定义一个函数，用于移除缓存中的函数代码，以强制重新编译
def remove_from_cache(f):
    # 如果 f 是 types.CodeType 类型的对象，则重置其代码
    if isinstance(f, types.CodeType):
        reset_code(f)
    # 如果 f 拥有 "__code__" 属性，则重置其代码
    elif hasattr(f, "__code__"):
        reset_code(f.__code__)
    # 如果 f 拥有 "forward" 属性，且其 forward 属性拥有 "__code__" 属性，则重置其 forward.__code__ 的代码
    elif hasattr(getattr(f, "forward", None), "__code__"):
        reset_code(f.forward.__code__)
    else:
        # 导入 reset 模块，并调用其功能（可能是重置相关的缓存）
        from . import reset  # type: ignore[attr-defined]
        reset()
        # 记录警告日志，指示无法确定 f 的 __code__ 属性
        log.warning("could not determine __code__ for %s", f)


# 定义一个空函数，不执行任何操作
def nothing():
    pass


# 定义一个总是返回 False 的函数
def always_false():
    return False


# 定义一个函数，用于查找嵌套 _TorchDynamoContext 调用中的最内层函数
def innermost_fn(fn):
    """
    In case of nesting of _TorchDynamoContext calls, find the innermost
    function. TorchDynamo caches on fn.__code__ object, so its necessary to find
    the innermost function to pass on the optimize, run, disable etc.
    """
    # 将未改变的原始函数设置为 fn
    unaltered_fn = fn
    # 循环直到找到没有 _torchdynamo_orig_callable 属性的函数
    while hasattr(unaltered_fn, "_torchdynamo_orig_callable"):
        unaltered_fn = unaltered_fn._torchdynamo_orig_callable
        assert callable(unaltered_fn)
    return unaltered_fn


# 定义一个函数，根据参数 enable 的值创建一个配置设置器
def make_set_enable_dynamic(enable: bool):
    assert isinstance(enable, bool)
    if enable:
        # 假设一切默认为动态
        return config._make_closure_patcher(assume_static_by_default=False)
    else:
        # 禁用自动动态形状，假设一切默认为静态
        return config._make_closure_patcher(
            automatic_dynamic_shapes=False, assume_static_by_default=True
        )


# 定义一个类 _TorchDynamoContext，用于管理 TorchDynamo 的上下文
class _TorchDynamoContext:
    def __init__(
        self,
        callback: DynamoCallback,
        on_enter=nothing,
        backend_ctx_ctor=null_context,
        patch_fn=nothing,
        first_ctx=False,
        *,
        export=False,
        dynamic=None,
        compiler_config=None,
    ):
        # 调用父类的构造方法
        super().__init__()
        # 断言回调函数是可调用的，或者为False或None
        assert callable(callback) or callback is False or callback is None
        # 将回调函数保存为DynamoCallback类型
        self.callback: DynamoCallback = callback
        # 保存后端上下文构造器
        self._backend_ctx_ctor = backend_ctx_ctor
        # 保存先前的DynamoCallback对象或unset（未设置）
        self.prior: Union[Unset, DynamoCallback] = unset
        # 保存第一个上下文对象
        self.first_ctx = first_ctx
        # 保存是否导出的标志
        self.export = export
        # 保存是否动态的标志
        self._dynamic = dynamic
        # 保存编译器配置
        self.compiler_config = compiler_config
        # 初始化清理函数列表
        self.cleanup_fns: List[Callable[[], Any]] = []
        # 初始化进入和退出钩子列表
        self.enter_exit_hooks = []
        # 执行patch_fn函数
        patch_fn()

        # 保存后端，以便在torch._dynamo.reset时重置它们
        backend = innermost_fn(callback)
        cached_backends.setdefault(id(backend), backend)

        # 如果dynamic不为None，则添加设置动态标志的钩子函数
        if dynamic is not None:
            self.enter_exit_hooks.append(make_set_enable_dynamic(dynamic))

        # 如果存在on_enter函数，则创建一个调用on_enter并返回nothing的函数，并添加到进入和退出钩子列表中
        if on_enter is not nothing:
            # 这种情况并不常见
            def call_on_enter():
                on_enter()
                return nothing

            self.enter_exit_hooks.append(call_on_enter)

        # 如果backend_ctx_ctor不是contextlib.nullcontext，则创建一个调用backend_ctx_ctor的函数，并将其添加到进入和退出钩子列表中
        if backend_ctx_ctor is not contextlib.nullcontext:
            # 这种情况并不常见
            def call_backend_ctx():
                ctx = backend_ctx_ctor()
                ctx.__enter__()
                return functools.partial(ctx.__exit__, None, None, None)

            self.enter_exit_hooks.append(call_backend_ctx)

    def __enter__(self):
        # 如果配置要求在上下文管理器中使用时引发异常，则引发异常
        if config.raise_on_ctx_manager_usage:
            raise RuntimeError(
                "torch._dynamo.optimize(...) is used with a context manager. "
                "Please refer to https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html "
                "to use torch._dynamo.optimize(...) as an annotation/decorator. "
            )
        # 执行所有进入钩子函数，并将返回的清理函数列表保存在cleanup_fns中
        self.cleanup_fns = [enter() for enter in self.enter_exit_hooks]
        # 设置当前回调函数为当前评估帧，并保存先前的评估帧
        self.prior = set_eval_frame(self.callback)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 断言先前的评估帧不是unset
        assert self.prior is not unset
        # 恢复先前的评估帧
        set_eval_frame(self.prior)
        # 将先前的评估帧标记为unset
        self.prior = unset
        # 依次执行所有清理函数
        for cleanup in self.cleanup_fns:
            cleanup()
        # 清空清理函数列表
        self.cleanup_fns.clear()
# 定义 OptimizeContext 类，继承自 _TorchDynamoContext 类
class OptimizeContext(_TorchDynamoContext):
    # 初始化方法
    def __init__(
        self,
        callback,  # 回调函数
        backend_ctx_ctor,  # 后端上下文构造器
        first_ctx=False,  # 是否是第一个上下文，默认为 False
        *,
        export=False,  # 是否导出，默认为 False
        dynamic=None,  # 是否动态，默认为 None
        compiler_config=None,  # 编译器配置，默认为 None
        rebuild_ctx: Optional[
            Callable[[], Union[OptimizeContext, _NullDecorator]]
        ] = None,  # 可选的重建上下文回调函数，返回类型为 OptimizeContext 或 _NullDecorator
    ):
        # 进入上下文时调用的方法
        def on_enter():
            install_generation_tagging_init()  # 安装生成标记初始化

        # 调用父类的初始化方法
        super().__init__(
            callback=callback,
            on_enter=on_enter,
            backend_ctx_ctor=backend_ctx_ctor,
            patch_fn=TorchPatcher.patch,  # 补丁函数来自 TorchPatcher.patch
            first_ctx=first_ctx,
            export=export,
            dynamic=dynamic,
            compiler_config=compiler_config,
        )

        # 如果配置为编译自动微分
        if config.compiled_autograd:
            
            # 定义调用编译自动微分的函数
            def call_compiled_autograd():
                assert rebuild_ctx is not None  # 断言重建上下文函数不为 None
                compiler_fn = rebuild_ctx()  # 调用重建上下文函数获取编译器函数
                ctx = torch._dynamo.compiled_autograd.enable(compiler_fn)  # 启用编译自动微分
                ctx.__enter__()  # 进入上下文
                return functools.partial(ctx.__exit__, None, None, None)

            # 将调用编译自动微分的函数添加到进入和退出钩子列表中
            self.enter_exit_hooks.append(call_compiled_autograd)

    # 实现对象的序列化方法
    def __reduce__(self):
        return (
            self.__class__,  # 返回类本身
            (self.callback, self._backend_ctx_ctor, self.first_ctx),  # 初始化参数
            {
                "export": self.export,  # 导出标志
                "dynamic": self._dynamic,  # 动态标志
                "compiler_config": self.compiler_config,  # 编译器配置
            },
        )


# 定义 RunOnlyContext 类，继承自 _TorchDynamoContext 类
class RunOnlyContext(_TorchDynamoContext):
    # 初始化方法
    def __init__(self):
        # 进入上下文时调用的方法
        def on_enter():
            torch._dynamo.mutation_guard.GenerationTracker.generation += 1  # 更新代数追踪的生成数

        # 调用父类的初始化方法
        super().__init__(callback=False, on_enter=on_enter)

    # 实现对象的序列化方法
    def __reduce__(self):
        return (self.__class__, ())  # 返回类本身


# 定义 DisableContext 类，继承自 _TorchDynamoContext 类
class DisableContext(_TorchDynamoContext):
    # 初始化方法
    def __init__(self):
        super().__init__(callback=None)  # 调用父类的初始化方法，设置回调为 None
    # 定义一个特殊方法 __call__，使对象可调用
    def __call__(self, fn):
        # 将原本在基类 _TorchDynamoContext 中的代码移到这里，以提升代码组织性。
        # 对于禁用操作，我们只需将回调函数设为 None。不需要检查 trace_rules 或创建任何包装器。
        fn = innermost_fn(fn)

        # 如果 fn 是 torch.nn.Module 类型
        if isinstance(fn, torch.nn.Module):
            mod = fn
            # 创建一个优化过的模块实例，并将 self 作为参数传入
            new_mod = OptimizedModule(mod, self)
            # 存储原始的模块的 forward 方法
            new_mod._torchdynamo_orig_callable = mod.forward
            return new_mod

        # 如果 fn 是一个类
        if inspect.isclass(fn):
            # 用户用编译/禁用装饰器包装了这个类。应用禁用到其 __init__/__call__ 方法上。
            cls_obj = fn
            # 对于重建字节码时禁用 __init__ 方法是有用的，可以防止 Dynamo 追踪到 __init__ 函数内部。
            cls_obj.__init__ = self(cls_obj.__init__)
            cls_obj.__call__ = self(cls_obj.__call__)
            # 如果 cls_obj 是 torch.nn.Module 的子类
            if issubclass(cls_obj, torch.nn.Module):
                # NN 模块变量跟踪直接内联 _call_impl。禁用它。
                cls_obj._call_impl = self(cls_obj._call_impl)
            return cls_obj

        # 如果 fn 是可调用的函数
        assert callable(fn)

        # 获取当前对象的 callback 属性
        callback = self.callback

        # 定义一个装饰器函数 _fn，用于包装 fn 函数
        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            # 设置当前评估帧为 callback 的返回值
            prior = set_eval_frame(callback)
            try:
                # 调用原始的 fn 函数，并返回其结果
                return fn(*args, **kwargs)
            finally:
                # 恢复先前的评估帧
                set_eval_frame(prior)

        # 将 _fn 标记为被 TorchDynamo 禁用的函数
        _fn._torchdynamo_disable = True  # type: ignore[attr-defined]

        # 存储函数指针以便在装饰器嵌套中找到原始的可调用对象
        _fn._torchdynamo_orig_callable = fn  # type: ignore[attr-defined]

        return _fn

    # 定义一个特殊方法 __reduce__，用于对象的序列化
    def __reduce__(self):
        return (self.__class__, ())
# 创建一个函数 _optimize_catch_errors，用于优化编译函数并处理可能的错误
def _optimize_catch_errors(
    compile_fn,  # 编译函数
    hooks: Hooks,  # 钩子函数集合
    backend_ctx_ctor=null_context,  # 后端上下文构造器，默认为 null_context
    export=False,  # 是否导出模型
    dynamic=None,  # 是否动态优化
    compiler_config=None,  # 编译器配置
    rebuild_ctx=None,  # 重建上下文
):
    # 返回一个 OptimizeContext 对象，包装了 catch_errors_wrapper 处理的 compile_fn 和 hooks
    return OptimizeContext(
        convert_frame.catch_errors_wrapper(compile_fn, hooks),
        backend_ctx_ctor=backend_ctx_ctor,
        first_ctx=True,
        export=export,
        dynamic=dynamic,
        compiler_config=compiler_config,
        rebuild_ctx=rebuild_ctx,
    )


# 创建一个函数 get_compiler_fn，用于获取编译器函数并进行后端调试包装
def get_compiler_fn(compiler_fn):
    # 导入 wrap_backend_debug 函数，用于后端调试包装
    from .repro.after_dynamo import wrap_backend_debug

    # 如果 compiler_fn 具有属性 compiler_name，则使用其编译器名称
    if hasattr(compiler_fn, "compiler_name"):
        compiler_str = compiler_fn.compiler_name
    # 否则，如果 compiler_fn 是字符串，则直接使用其作为编译器名称
    elif isinstance(compiler_fn, str):
        compiler_str = compiler_fn
    else:
        compiler_str = None  # 否则编译器名称为空
    # 查找并返回编译器函数的后端调试包装
    compiler_fn = lookup_backend(compiler_fn)
    return wrap_backend_debug(compiler_fn, compiler_str)


# 创建一个 _NullDecorator 类，继承自 contextlib.nullcontext，用于装饰器的空上下文处理
class _NullDecorator(contextlib.nullcontext):  # type: ignore[type-arg]
    # 定义 __call__ 方法，接受一个函数 fn，确保 fn 是可调用的
    def __call__(self, fn):
        assert callable(fn)
        return fn  # 返回原始函数 fn


# 创建一个函数 check_if_dynamo_supported，用于检查是否支持 Dynamo 引擎
def check_if_dynamo_supported():
    # 如果 Python 版本大于等于 3.13，则抛出运行时错误，提示不支持 Python 3.13+
    if sys.version_info >= (3, 13):
        raise RuntimeError("Python 3.13+ not yet supported for torch.compile")


# 创建一个函数 is_dynamo_supported，用于检查是否支持 Dynamo 引擎
def is_dynamo_supported():
    try:
        check_if_dynamo_supported()  # 检查 Dynamo 是否支持
        return True  # 如果通过检查，则返回 True
    except Exception:
        return False  # 捕获异常时返回 False


# 创建一个函数 check_if_inductor_supported，用于检查是否支持 Inductor 引擎
def check_if_inductor_supported():
    check_if_dynamo_supported()  # 首先检查是否支持 Dynamo

    # 如果运行在 Windows 平台，则抛出运行时错误，提示不支持 Windows 平台的 Inductor
    if sys.platform == "win32":
        raise RuntimeError("Windows not yet supported for inductor")


# 创建一个函数 is_inductor_supported，用于检查是否支持 Inductor 引擎
def is_inductor_supported():
    try:
        check_if_inductor_supported()  # 检查是否支持 Inductor
        return True  # 如果通过检查，则返回 True
    except Exception:
        return False  # 捕获异常时返回 False


# 创建一个 optimize 函数，用于优化操作，内部定义了 rebuild_ctx 函数
def optimize(*args, **kwargs):
    # 定义 rebuild_ctx 函数，用于重新构建上下文
    def rebuild_ctx():
        return optimize(*args, **kwargs)  # 返回 optimize 函数的调用结果

    # 调用 _optimize 函数，并传递 rebuild_ctx 函数以及其他参数和关键字参数
    return _optimize(rebuild_ctx, *args, **kwargs)


# 创建一个 _optimize 函数，作为 TorchDynamo 的主入口点，进行图捕获并调用后端进行优化
def _optimize(
    rebuild_ctx: Callable[[], Union[OptimizeContext, _NullDecorator]],  # 重建上下文的可调用函数类型
    backend="inductor",  # 后端名称，默认为 "inductor"
    *,
    nopython=False,  # 是否禁用 Python
    guard_export_fn=None,  # 导出函数的保护函数
    guard_fail_fn=None,  # 失败函数的保护函数
    disable=False,  # 是否禁用优化
    dynamic=None,  # 是否动态优化
) -> Union[OptimizeContext, _NullDecorator]:  # 返回类型为 OptimizeContext 或 _NullDecorator
    """
    TorchDynamo 的主入口点。进行图捕获并调用后端进行优化。
    """
    Args:
        backend: 优化器的后端，可以是以下两种形式之一：
            - 函数/可调用对象，接受 torch.fx.GraphModule 和 example_inputs 参数，并返回一个运行图形更快的 Python 可调用对象。
              也可以通过设置 backend_ctx_ctor 属性提供后端的额外上下文，如 torch.jit.fuser("fuser2")。
              参见 AOTAutogradMemoryEfficientFusionWithContext 的用法。
            - 字符串形式的后端名称，需在 `torch._dynamo.list_backends()` 中。
        nopython: 如果为 True，则图形中断将被视为错误，并且将生成一个整体程序图形。
        disable: 如果为 True，则将此装饰器变为无操作。
        dynamic: 如果为 True，则尽可能动态编译内核。如果为 False，则禁用所有动态形状支持（始终专门化）。
                 如果为 None，则在重新编译时自动检测大小变化并生成动态内核。
    
    Example Usage::
    
        @torch._dynamo.optimize()
        def toy_example(a, b):
            ...
    """
    检查是否支持 Dynamo 优化
    check_if_dynamo_supported()
    
    # 注意：hooks 对象可以是全局的，而不是传递的，但这会使 API 使用和管道故事变得混乱，
    # 在多个 .optimize 调用嵌套时会引发混淆。在此之前已有一些相关的先例，
    # 关于此事强制嵌套后端调用是相同编译器，但是对于回调和 hooks，我们更愿意在我们的一端处理一些更简单易懂的用户体验。
    hooks = Hooks(guard_export_fn=guard_export_fn, guard_fail_fn=guard_fail_fn)
    torch._C._log_api_usage_once("torch._dynamo.optimize")
    
    # 如果禁用或环境变量 TORCHDYNAMO_DISABLE 设置为 "1" 或未启用 "pytorch/compiler:enable_dynamo" 选项，
    # 则返回一个空装饰器 _NullDecorator()
    if (
        disable
        or os.environ.get("TORCHDYNAMO_DISABLE", "") == "1"
        or (not justknobs_check("pytorch/compiler:enable_dynamo"))
    ):
        return _NullDecorator()
    
    # 获取优化器函数
    backend = get_compiler_fn(backend)
    
    # 查找后端是否有任何额外的上下文管理器
    backend_ctx_ctor = getattr(backend, "backend_ctx_ctor", null_context)
    
    # 如果 nopython 为 True，则进行优化断言
    if nopython:
        return optimize_assert(
            backend,
            dynamic=dynamic,
            hooks=hooks,
            rebuild_ctx=rebuild_ctx,
        )
    
    # 否则，将 backend 函数存储在由 _optimize_catch_errors 返回的可调用对象中的 _torchdynamo_orig_callable 字段中。
    # 这可以被 eval_frame.c 使用，以在后端上插入保护。
    return _optimize_catch_errors(
        convert_frame.convert_frame(backend, hooks=hooks),
        hooks,
        backend_ctx_ctor,
        dynamic=dynamic,
        compiler_config=backend.get_compiler_config()
        if hasattr(backend, "get_compiler_config")
        else None,
        rebuild_ctx=rebuild_ctx,
    )
# TODO(voz): Consider making "explain" output alongside a run / part of a run
@patch("torch._dynamo.symbolic_convert.explain", True)
def explain(f, *extra_args, **extra_kwargs):
    def inner(*args, **kwargs):
        # TODO(voz): Do we want a decorator for this?
        from . import reset  # type: ignore[attr-defined]

        # Reset the state to ensure clean execution environment
        reset()

        # Initialize lists and counters to store graph information
        graphs: List[torch.fx.GraphModule] = []
        break_reasons: List[Any] = []
        op_count: int = 0
        ops_per_graph: List[torch.fx.Node] = []
        out_guards: List[_guards.Guard] = []

        # Function to accumulate graph information during compilation
        def dynamo_graph_accumulating_compiler(
            gm: torch.fx.GraphModule, example_inputs
        ):
            from .backends.debugging import _explain_graph_detail

            nonlocal graphs
            nonlocal op_count
            nonlocal ops_per_graph
            nonlocal break_reasons

            # Call to accumulate detailed graph information
            gm, graphs, op_count, ops_per_graph, break_reasons = _explain_graph_detail(
                gm, graphs, op_count, ops_per_graph, break_reasons
            )

            return gm.forward

        # Function to export guards during optimization
        def guard_export_print(guards):
            nonlocal out_guards
            out_guards.extend(guards)

        # Optimize the function using dynamo_graph_accumulating_compiler
        opt_f = optimize(
            dynamo_graph_accumulating_compiler,
            nopython=False,
            guard_export_fn=guard_export_print,
        )(f)

        # Execute the optimized function with given arguments
        opt_f(*args, **kwargs)

        # Calculate various counts and times related to the compiled graphs
        graph_count = len(graphs)
        graph_break_count = graph_count - 1
        compile_time = compile_times(repr="str")

        # Reset the state after execution
        reset()

        # Import ExplainOutput class for returning detailed explanation
        from .backends.debugging import ExplainOutput

        # Return an instance of ExplainOutput containing collected information
        return ExplainOutput(
            graphs,
            graph_count,
            graph_break_count,
            break_reasons,
            op_count,
            ops_per_graph,
            out_guards,
            compile_time,
        )

    # Warning if extra arguments are provided (deprecated usage)
    if extra_args or extra_kwargs:
        warnings.warn(
            "explain(f, *args, **kwargs) is deprecated, use explain(f)(*args, **kwargs) instead.  "
            "If you don't migrate, we may break your explain call in the future if your user defined kwargs "
            "conflict with future kwargs added to explain(f).",
            FutureWarning,
            stacklevel=2,
        )
        # Invoke inner function with extra arguments
        return inner(*extra_args, **extra_kwargs)
    else:
        # Return inner function without extra arguments
        return inner


class FlattenInputOutputSignature(torch.fx.interpreter.Transformer):
    def __init__(
        self,
        m: torch.fx.GraphModule,
        flat_args: Tuple[Any],
        matched_input_elements_positions: List[int],
        flat_results: List[Any],
        matched_output_elements_positions: List[int],
        example_fake_inputs: List[torch.Tensor],
        flat_args_dynamic_dims: List[Set[int]],
        fake_mode: Optional[fake_tensor.FakeTensorMode] = None,
        ):
            super().__init__(m)
            
            # 确保动态维度参数列表与静态参数列表长度一致
            assert len(flat_args_dynamic_dims) == len(flat_args)
            
            # 构建匹配的输入元素到虚假输入的映射字典
            matched_input_elements_to_fake = {
                val: example_fake_inputs[ix]
                for ix, val in enumerate(matched_input_elements_positions)
            }
    
            # 初始化新参数列表
            self.new_args = []
            for i in range(0, len(flat_args)):
                # 创建占位符参数对象
                arg = super().placeholder(f"arg{i}", (), {})
                
                # 如果当前参数索引存在于匹配的输入元素到虚假输入的映射中，则设置节点的值
                if i in matched_input_elements_to_fake:
                    arg.node.meta["val"] = matched_input_elements_to_fake[i]
                else:
                    # 如果未在匹配的输入元素位置中找到，则从输入中填充节点值
                    if fake_mode is not None and isinstance(flat_args[i], torch.Tensor):
                        # TODO(zhxchen17) 保留所有用户约束条件
                        # 使用 fake_mode 将 tensor 转换为值，并设置符号上下文
                        arg.node.meta["val"] = fake_mode.from_tensor(
                            flat_args[i],
                            symbolic_context=StatelessSymbolicContext(
                                dynamic_sizes=[
                                    DimDynamic.DYNAMIC
                                    if d in flat_args_dynamic_dims[i]
                                    else DimDynamic.STATIC
                                    for d in range(len(flat_args[i].shape))
                                ],
                                constraint_sizes=[None] * len(flat_args[i].shape),
                            ),
                        )
                
                # 将新参数添加到列表中
                self.new_args.append(arg)
            
            # 生成器，用于迭代旧参数列表中的元素
            self.old_args_gen = (self.new_args[i] for i in matched_input_elements_positions)
            
            # 设置匹配的输出元素位置和扁平化的结果
            self.matched_output_elements_positions = matched_output_elements_positions
            self.flat_results = flat_results
    
        # 返回 arg 对象
        def placeholder(self, target, args, kwargs):
            # 从旧参数生成器中获取下一个参数
            arg = next(self.old_args_gen)
            
            # 如果当前节点的 meta 中包含 "val"，则将其复制到参数节点的 meta 中
            if "val" in self.current_node.meta:
                arg.node.meta["val"] = self.current_node.meta["val"]
            
            # 如果当前节点的 meta 中包含 "tensor_dict"，则将其复制到参数节点的 meta 中
            if "tensor_dict" in self.current_node.meta:
                arg.node.meta["tensor_dict"] = self.current_node.meta["tensor_dict"]
            
            # 如果当前节点的 meta 中包含 "example_value"，则将其复制到参数节点的 meta 中
            # 注意：故意不使用 set_example_value
            if "example_value" in self.current_node.meta:
                arg.node.meta["example_value"] = self.current_node.meta["example_value"]
            
            # 如果当前节点的 meta 中包含 "unbacked_bindings"，则将其复制到参数节点的 meta 中
            if "unbacked_bindings" in self.current_node.meta:
                arg.node.meta["unbacked_bindings"] = self.current_node.meta["unbacked_bindings"]
            
            # 返回参数对象
            return arg
    # 从参数中获取动态结果的扁平化列表
    dynamo_result_flat = args[0]
    # 创建查找列表，包括动态结果扁平化列表和对象的新参数
    lookup = [*dynamo_result_flat, *self.new_args]
    # 初始化新的结果扁平化列表
    new_results_flat = []
    # 遍历扁平化结果列表的长度
    for i in range(len(self.flat_results)):
        # 如果匹配输出元素的位置不为 None
        if self.matched_output_elements_positions[i] is not None:
            # 根据匹配位置从查找列表中取值，并添加到新结果扁平化列表中
            new_results_flat.append(
                lookup[self.matched_output_elements_positions[i]]
            )
        else:
            # 否则，将扁平化结果列表中的常量值添加到新结果扁平化列表中
            const_val = self.flat_results[i]
            assert isinstance(const_val, tuple(common_constant_types))
            new_results_flat.append(const_val)
    # 调用父类的输出方法，传入目标、新结果扁平化列表和空字典作为参数，返回结果代理对象
    return super().output(target, (new_results_flat,), {})

# 设置当前节点并执行节点，返回结果代理对象
def run_node(self, n):
    self.current_node = n
    result_proxy = super().run_node(n)
    # 如果当前节点的元数据中包含 'val' 键
    if "val" in self.current_node.meta:
        # 将 'val' 值复制到结果代理节点的元数据中
        result_proxy.node.meta["val"] = self.current_node.meta["val"]
    # 如果当前节点的元数据中包含 'example_value' 键
    if "example_value" in self.current_node.meta:
        # 注意：故意不使用 set_example_value 方法
        # 将 'example_value' 值复制到结果代理节点的元数据中
        result_proxy.node.meta["example_value"] = self.current_node.meta[
            "example_value"
        ]
    # 如果当前节点的元数据中包含 'unbacked_bindings' 键
    if "unbacked_bindings" in self.current_node.meta:
        # 将 'unbacked_bindings' 值复制到结果代理节点的元数据中
        result_proxy.node.meta["unbacked_bindings"] = self.current_node.meta[
            "unbacked_bindings"
        ]
    # 如果当前节点的操作不是 'output'
    if self.current_node.op != "output":
        # 重命名结果代理节点的名称为当前节点的名称或者结果代理节点的名称
        result_proxy.node._rename(
            getattr(self.current_node, "name", result_proxy.node.name)
        )
    # 返回结果代理对象
    return result_proxy

# 执行转换操作并返回结果
def transform(self):
    # 调用父类的转换方法，返回结果图模型
    result_gm = super().transform()
    # 如果模块的元数据中包含 'dynamo_flat_name_to_original_fqn' 键
    if "dynamo_flat_name_to_original_fqn" in self.module.meta:
        # 将 'dynamo_flat_name_to_original_fqn' 值复制到结果图模型的元数据中
        result_gm.meta["dynamo_flat_name_to_original_fqn"] = self.module.meta[
            "dynamo_flat_name_to_original_fqn"
        ]
    # 返回结果图模型
    return result_gm
# 定义一个名为 ExportResult 的命名元组，包含两个字段：graph_module 和 guards
# graph_module: 一个 torch.fx.GraphModule 对象，表示图模块
# guards: 一个 _guards.GuardsSet 对象，表示守卫集合
# NB: 不要添加新的字段，除非同时重写 __iter__ 方法；因为有代码在解构元组，所以添加新字段会破坏向后兼容性

class ExportResult(NamedTuple):
    graph_module: torch.fx.GraphModule
    guards: _guards.GuardsSet
    # NB: Do not add new fields without overriding __iter__; people are
    # destructuring so it is BC-breaking


def check_signature_rewritable(graph):
    # 初始化一个空列表，用于存储输入错误信息
    input_errors = []
    # 遍历图中所有 op 为 "placeholder" 的节点
    for node in graph.graph.find_nodes(op="placeholder"):
        # 断言节点对象具有属性 "_dynamo_source"
        assert hasattr(node, "_dynamo_source")
        # 获取节点的源信息 _dynamo_source
        source = node._dynamo_source
        # 获取与该源信息相关的用户调用堆栈列表
        user_stacks = graph._source_to_user_stacks.get(source)
        # 如果没有用户调用堆栈信息，继续下一个节点
        if user_stacks is None:
            continue
        # 断言用户调用堆栈列表长度大于 0
        assert len(user_stacks) > 0
        # 初始化堆栈信息为 None
        stack = None
        # 在用户调用堆栈列表中查找第一个非空堆栈
        for s in user_stacks:
            if len(s) == 0:
                continue
            stack = s
            break
        # 如果未找到有效堆栈信息，生成一个关于源信息的错误消息
        if stack is None:
            msg = f"{source.name()}, a closed over free variable"
        else:
            # 将堆栈信息格式化为字符串
            tb = "".join(traceback.format_list(stack))
            extra = ""
            # 如果用户调用堆栈列表长度大于 1，添加省略信息
            if len(user_stacks) > 1:
                extra = f"(elided {len(user_stacks) - 1} more accesses)"
            msg = f"{source.name()}, accessed at:\n{tb}{extra}"
        # 将生成的错误消息添加到输入错误列表
        input_errors.append(msg)

    # 如果存在输入错误信息，抛出一个 UserError 异常
    if input_errors:
        raise UserError(
            UserErrorType.INVALID_INPUT,
            "Cannot export model which references tensors that are neither "
            "buffers/parameters/constants nor are direct inputs.  For each tensor, if you'd "
            "like this tensor to be an explicit input, add it as a dummy argument "
            "to the top-level model definition you are exporting; if you would "
            "like its value to be embedded as an exported constant, wrap its access "
            "in a function marked with @assume_constant_result.\n\n"
            + "\n\n".join(input_errors),
        )


def rewrite_signature(
    f_sig,
    graph,
    fake_mode,
    flat_args,
    in_spec,
    example_fake_inputs,
    graph_captured_input,
    graph_captured_output,
    dynamo_traced_result,
    flat_args_dynamic_dims,
):
    # 将平坦化的参数列表 flat_args 与输入规范 in_spec 进行反平铺操作，得到原始参数和关键字参数
    orig_args, orig_kwargs = pytree.tree_unflatten(flat_args, in_spec)
    def check_user_input_output(flat_values, error_type):
        # 支持的类型列表，包括 torch.Tensor 和其他特定类型
        supported_types = [
            torch.Tensor,
            torch.SymInt,
            torch.SymFloat,
            torch.SymBool,
            torch._C.ScriptObject,
        ] + list(common_constant_types)

        def is_supported_type(val):
            # 检查给定值是否属于支持的类型之一
            return isinstance(val, tuple(supported_types))

        # 确定错误类型对应的值类型是输入还是输出
        value_type = "input" if error_type == UserErrorType.INVALID_INPUT else "output"
        # 遍历扁平化后的值列表
        for v in flat_values:
            # 如果值不是支持的类型之一
            if not is_supported_type(v):
                # 对于无效输入错误类型，允许值为 None
                if error_type == UserErrorType.INVALID_INPUT and v is None:
                    continue

                # 抛出用户定义的错误，说明值类型不受支持或不能被pytree扁平化
                raise UserError(
                    error_type,
                    f"It looks like one of the {value_type}s with type `{type(v)}` "
                    "is not supported or pytree-flattenable. \n"
                    f"Exported graphs {value_type}s can only contain the "
                    f"following supported types: {supported_types}. \n"
                    "If you are using a custom class object, "
                    "please register a pytree_flatten/unflatten function "
                    "using `torch.utils._pytree.register_pytree_node` or "
                    "`torch.export.register_dataclass`.",
                )

    # 检查输入值的类型是否有效
    check_user_input_output(flat_args, UserErrorType.INVALID_INPUT)

    # 对动态追踪结果进行树形扁平化，获取扁平化结果和输出规范
    flat_results_traced, out_spec_traced = pytree.tree_flatten(dynamo_traced_result)

    # 检查输出值的类型是否有效
    check_user_input_output(flat_results_traced, UserErrorType.INVALID_OUTPUT)

    def produce_matching(debug_type, sources, candidates):
        # 初始化匹配的元素位置列表和源值字典
        matched_elements_positions: List[Optional[int]] = []
        dict_of_source_vals = {}
        # 枚举源值列表，构建源值字典，键为值的ID，值为索引
        for i, val in enumerate(sources):
            dict_of_source_vals[id(val)] = i

        # 遍历候选值列表
        for i, val in enumerate(candidates):
            # 如果值是通用常量类型之一的元组，则添加空值到匹配位置列表
            if isinstance(val, tuple(common_constant_types)):
                matched_elements_positions.append(None)
            # 如果候选值的ID不在源值字典中，则抛出断言错误
            elif id(val) not in dict_of_source_vals:
                raise AssertionError(
                    f"Unexpectedly found a {type(val)} in the {debug_type}.\n"
                    'Please file an issue along with a paste of the logs from TORCH_LOGS="+export"'
                )
            # 否则，将候选值对应的源值索引添加到匹配位置列表
            else:
                matched_elements_positions.append(dict_of_source_vals[id(val)])

        return matched_elements_positions

    # 执行输入元素的匹配
    matched_input_elements_positions = produce_matching(
        "inputs", flat_args, graph_captured_input
    )

    # 断言图捕获的输出不为None
    assert graph_captured_output is not None
    # 执行输出元素的匹配，包括图捕获的输出、扁平化后的结果和扁平化前的输入
    matched_output_elements_positions = produce_matching(
        "outputs", list(graph_captured_output) + flat_args, flat_results_traced
    )
    # 使用 FlattenInputOutputSignature 对象来转换给定的图形 graph，使其扁平化输入输出签名
    new_graph = FlattenInputOutputSignature(
        graph,  # 输入原始图形对象
        flat_args,  # 扁平化后的参数列表
        matched_input_elements_positions,  # 匹配到的输入元素位置
        flat_results_traced,  # 被跟踪的扁平化结果
        matched_output_elements_positions,  # 匹配到的输出元素位置
        example_fake_inputs,  # 示例假输入
        flat_args_dynamic_dims,  # 动态维度的扁平化参数
        fake_mode,  # 假模式标志
    ).transform()  # 对图形进行转换并返回新的图形对象

    # 设置新图形的代码生成器，使其具有与用户代码相同的输入/输出规范
    new_graph.graph._codegen = _PyTreeCodeGen(
        _PyTreeInfo(
            argument_names(f_sig, orig_args, orig_kwargs),  # 获取参数名称
            in_spec,  # 输入规范信息
            out_spec_traced,  # 被跟踪的输出规范信息
        )
    )

    # 重新编译新图形，确保其更新后的设置生效
    new_graph.recompile()

    # 返回重新设置后的新图形对象
    return new_graph
# 导出函数 f 到可以在 PyTorch 外部执行的 FX 图格式

def export(
    f: Callable[..., Any],  # 输入函数 f，可以接受任意参数并返回任意类型
    *extra_args,  # 任意数量的额外位置参数，以元组形式接收
    aten_graph: bool = False,  # 是否导出包含 ATen 操作符的图，默认为 False
    pre_dispatch: bool = False,  # 如果为 True，在运行任何 PyTorch 调度逻辑之前导出 ATen 操作符的图
    decomposition_table: Optional[Dict[torch._ops.OpOverload, Callable[..., Any]]] = None,
    # 操作符到其分解函数的映射表，当设定 aten_graph 或 tracing_mode 时需要提供，默认为 None
    tracing_mode: str = "symbolic",  # 追踪模式，"symbolic" 表示开启动态形状支持，默认为 "symbolic"
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]] = None,
    # 动态形状参数，可以是字典或元组/列表，指定动态形状规范
    assume_static_by_default: bool = False,  # 默认假设静态形状，默认为 False
    same_signature: bool = True,  # 如果为 True，重写返回图的签名为与 f 相同
    disable_constraint_solver: bool = False,  # 是否禁用约束求解器
    prefer_deferred_runtime_asserts_over_guards: bool = False,
    # 是否优先使用延迟运行时断言而不是守卫
    _allow_complex_guards_as_runtime_asserts: bool = False,
    _log_export_usage: bool = True,  # 是否记录导出使用情况，默认为 True
    **extra_kwargs,  # 任意数量的额外关键字参数，以字典形式接收
) -> Callable[..., ExportResult]:  # 返回一个可调用对象，接受任意参数并返回 ExportResult 类型的值
    """
    导出输入函数 f 到可以在 PyTorch 外部执行的 FX 图格式。

    Args:
        f (callable): 要导出的 PyTorch 函数。
        
        aten_graph (bool): 如果为 True，导出包含 ATen 操作符的图。如果为 False，导出包含 Python 操作符的图。默认为 False。
        
        pre_dispatch (bool): 如果为 True，在任何 PyTorch 调度逻辑运行之前导出 ATen 操作符的图。这在您希望在运行 autograd、autocast 或其他集成到调度器的功能之前对图应用进一步转换时非常有用。此标志仅在设置 aten_graph=True 时有效。默认为 False。
        
        decomposition_table (dict): 将操作符映射到其分解函数的字典。如果设置了 aten_graph 或 tracing_mode，则需要提供。默认为 None。
        
        tracing_mode (str): 如果为 "symbolic"，则开启动态形状支持。默认为 "symbolic"。
        
        dynamic_shapes: 可选参数，类型应为：
            1) 从 "f" 的参数名到它们的动态形状规范的字典，
            2) 指定每个原始输入的动态形状规范的元组。
            如果在关键字参数上指定动态性，需要按照原始函数签名中定义的顺序传递它们。
            张量参数的动态形状可以指定为：
                (1) 从动态维度索引到 Dim 类型的字典，其中不需要在此字典中包含静态维度索引，但在包含时应映射为 None；
                (2) 一组 Dim 类型或 None 的元组/列表，其中 Dim 类型对应于动态维度，静态维度用 None 表示。
                递归使用包含规范的映射或序列来指定字典或张量的元组/列表。
        
        same_signature (bool): 如果为 True，重写返回图的签名为与 f 相同。
        
        disable_constraint_solver (bool): 是否必须禁用维度约束求解器。
        
    Returns:
        callable[..., ExportResult]: 一个可调用对象，接受任意参数并返回 ExportResult 类型的值。
    """
    # 实现在这里
    # 如果 _log_export_usage 为真，则记录导出使用情况，包括事件和标志
    if _log_export_usage:
        log_export_usage(event="export.private_api", flags={"_dynamo"})

    # 解决“局部变量在赋值前引用”的问题，将参数 f 和 assume_static_by_default 复制到新变量
    _f = f
    _assume_static_by_default = assume_static_by_default

    # 如果存在额外的位置参数或关键字参数
    if extra_args or extra_kwargs:
        # 发出警告，提醒用户使用新的调用方式以替代过时的 export(f, *args, **kwargs)
        warnings.warn(
            "export(f, *args, **kwargs) is deprecated, use export(f)(*args, **kwargs) instead.  "
            "If you don't migrate, we may break your export call in the future if your user defined kwargs "
            "conflict with future kwargs added to export(f).",
            FutureWarning,
            stacklevel=2,
        )
        # 返回 inner 函数的调用结果，传入额外的位置参数和关键字参数
        return inner(*extra_args, **extra_kwargs)
    else:
        # 如果没有额外的参数，直接返回 inner 函数
        return inner
# 定义一个函数 optimize_assert，用于优化断言
# backend 参数表示后端，hooks 参数是钩子对象，默认为不指定的 Hooks 对象
# export 参数表示是否导出，export_constraints 参数表示导出约束，默认为 None
# dynamic 参数表示动态属性，默认为 None，rebuild_ctx 参数表示重建上下文，默认为 None
def optimize_assert(
    backend,
    *,
    hooks=Hooks(None, None),
    export=False,
    export_constraints=None,
    dynamic=None,
    rebuild_ctx=None,
):
    """
    The same as `torch._dynamo.optimize(backend, nopython=True)`
    """
    # 获取编译器函数，并将 backend 参数重新赋值为编译器函数
    backend = get_compiler_fn(backend)

    # 查找 backend 是否有额外的上下文管理器，如果没有，则使用 null_context
    backend_ctx_ctor = getattr(backend, "backend_ctx_ctor", null_context)

    # 调用 _optimize_catch_errors 函数，处理 convert_frame.convert_frame_assert 的结果
    # backend 参数传递给 convert_frame.convert_frame_assert 函数
    # hooks, backend_ctx_ctor, export, dynamic, rebuild_ctx 参数也传递给 _optimize_catch_errors 函数
    return _optimize_catch_errors(
        convert_frame.convert_frame_assert(
            backend, export=export, export_constraints=export_constraints
        ),
        hooks,
        backend_ctx_ctor,
        export=export,
        dynamic=dynamic,
        rebuild_ctx=rebuild_ctx,
    )


class TorchPatcher:
    @staticmethod
    @functools.lru_cache(None)
    def patch():
        # 导入用于禁用函数的装饰器，由于问题与内部torch.deploy冲突，此处未使用修饰器
        from .decorators import disable
        
        # 禁用torch.jit模块的函数trace
        torch.jit.trace = disable(torch.jit.trace)
        # 禁用torch.jit模块的函数trace_module
        torch.jit.trace_module = disable(torch.jit.trace_module)
        # 禁用torch.jit模块的函数_get_trace_graph
        torch.jit._get_trace_graph = disable(torch.jit._get_trace_graph)
        # 禁用torch.fx._symbolic_trace.Tracer类的方法trace
        torch.fx._symbolic_trace.Tracer.trace = disable(
            torch.fx._symbolic_trace.Tracer.trace
        )
        # 设置torch.distributions.Distribution类的默认validate_args参数为False
        torch.distributions.Distribution.set_default_validate_args(False)

        # 导入优化器模块
        from ..optim import (
            adadelta,
            adagrad,
            adam,
            adamax,
            adamw,
            asgd,
            lbfgs,
            nadam,
            radam,
            rmsprop,
            rprop,
            sgd,
            sparse_adam,
        )

        # 构建优化器模块的集合
        optimizer_modules = {
            adadelta,
            adagrad,
            adam,
            adamax,
            adamw,
            asgd,
            lbfgs,
            nadam,
            radam,
            rmsprop,
            rprop,
            sgd,
            sparse_adam,
        }

        # 遍历优化器模块集合
        for opt_mod in optimizer_modules:
            # 获取优化器模块的名称
            opt_name = opt_mod.__name__.split(".")[-1]
            # 构建融合优化函数名
            fused_fn_name = f"_fused_{opt_name}"
            # 构建单张量优化函数名
            single_tensor_fn_name = f"_single_tensor_{opt_name}"

            # 如果优化器模块有融合优化函数，禁用之
            if hasattr(opt_mod, fused_fn_name):
                setattr(
                    opt_mod, fused_fn_name, disable(getattr(opt_mod, fused_fn_name))
                )

        # 获取所有torch.optim模块中的优化器类
        optimizer_classes = [
            opt
            for opt in torch.optim.__dict__.values()
            if inspect.isclass(opt) and issubclass(opt, torch.optim.Optimizer)
        ]

        # 不支持稀疏优化器类或追踪反向传播
        excluded_optimizer_classes = {
            torch.optim.SparseAdam,
            torch.optim.LBFGS,
        }

        # 遍历优化器类列表
        for opt in optimizer_classes:
            # 如果优化器类在排除列表中，禁用其step方法
            if opt in excluded_optimizer_classes:
                opt.step = disable(opt.step)

            # 如果优化器类具有_init_group方法，禁用之
            if hasattr(opt, "_init_group"):
                opt._init_group = disable(opt._init_group)

    @staticmethod
    def suppress_torch_distributed_warnings(fn):
        # 内部函数用于忽略torch.distributed模块中的UserWarning警告
        def inner_fn(*args, **kwargs):
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="torch.distributed"
            )
            # 调用原始函数并返回其结果
            return fn(*args, **kwargs)

        return inner_fn
```