# `.\pytorch\torch\_export\wrappers.py`

```py
# 用于声明允许未类型化定义的标志，供 mypy 使用
# 从 contextlib 模块导入上下文管理器
from contextlib import contextmanager

# 导入 torch 库
import torch
# 导入 torch 自定义操作模块
import torch._custom_ops
# 从 torch._C 模块导入 DispatchKey 枚举
from torch._C import DispatchKey
# 从 torch._higher_order_ops.strict_mode 模块导入 strict_mode 函数
from torch._higher_order_ops.strict_mode import strict_mode
# 从 torch._higher_order_ops.utils 模块导入 autograd_not_implemented 函数
from torch._higher_order_ops.utils import autograd_not_implemented
# 从 torch._ops 模块导入 HigherOrderOperator 类
from torch._ops import HigherOrderOperator
# 从 torch._subclasses.fake_tensor 模块导入 FakeTensorMode 类
from torch._subclasses.fake_tensor import FakeTensorMode
# 从 torch.fx.experimental.proxy_tensor 模块导入 ProxyTorchDispatchMode 和 track_tensor_tree 函数
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
# 从 torch.utils 模块导入 _pytree 别名 pytree
from torch.utils import _pytree as pytree

# 声明 _export_tracepoint 作为 HigherOrderOperator 对象的实例
_export_tracepoint = HigherOrderOperator("_export_tracepoint")

# 使用 _export_tracepoint 实例的 py_impl 方法注册 dispatch_mode 为 ProxyTorchDispatchMode 的实现
@_export_tracepoint.py_impl(ProxyTorchDispatchMode)
def export_tracepoint_dispatch_mode(mode, *args, **kwargs):
    # 如果 mode.enable_tracing 为 False，直接调用 _export_tracepoint 函数
    if not mode.enable_tracing:
        return _export_tracepoint(*args, **kwargs)
    # 对 args 和 kwargs 应用 mode.tracer.unwrap_proxy 函数
    p_args, p_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, (args, kwargs))
    # 使用 mode.tracer.create_proxy 创建代理，调用 _export_tracepoint 函数，并返回结果
    proxy = mode.tracer.create_proxy(
        "call_function", _export_tracepoint, p_args, p_kwargs
    )
    # 使用 track_tensor_tree 函数追踪张量树的变化，返回结果
    return track_tensor_tree(args, proxy, constant=None, tracer=mode.tracer)

# 使用 _export_tracepoint 实例的 py_impl 方法注册 dispatch_mode 为 FakeTensorMode 的实现
@_export_tracepoint.py_impl(FakeTensorMode)
def export_tracepoint_fake_tensor_mode(mode, *args, **kwargs):
    # 使用 mode 上下文管理器包裹返回值 args
    with mode:
        return args

# 使用 _export_tracepoint 实例的 py_functionalize_impl 方法将函数注册为 functional 实现
@_export_tracepoint.py_functionalize_impl
def export_tracepoint_functional(ctx, *args, **kwargs):
    # 使用 ctx.unwrap_tensors 函数对 args 和 kwargs 进行解包
    unwrapped_args = ctx.unwrap_tensors(args)
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)

    # 使用 ctx.redispatch_to_next 上下文管理器，重新分派到下一个函数
    with ctx.redispatch_to_next():
        # 调用 _export_tracepoint 函数，并使用 ctx.wrap_tensors 包装结果
        out = _export_tracepoint(*unwrapped_args, **unwrapped_kwargs)
        return ctx.wrap_tensors(out)

# 使用 _export_tracepoint 实例的 py_impl 方法注册 dispatch_key 为 DispatchKey.Autograd 的实现
@_export_tracepoint.py_impl(DispatchKey.Autograd)
def export_tracepoint_autograd(*args, **kwargs):
    # 调用 autograd_not_implemented 函数处理 _export_tracepoint 函数的延迟错误
    return autograd_not_implemented(_export_tracepoint, deferred_error=True)

# 使用 _export_tracepoint 实例的 py_impl 方法注册 dispatch_key 为 DispatchKey.CPU 的实现
@_export_tracepoint.py_impl(DispatchKey.CPU)
def export_tracepoint_cpu(*args, **kwargs):
    # 直接返回参数 args
    return args

# 定义 _wrap_submodule 函数，用于封装子模块，检查路径和模块调用规范
def _wrap_submodule(mod, path, module_call_specs):
    # 断言 mod 是 torch.nn.Module 的实例
    assert isinstance(mod, torch.nn.Module)
    # 断言 path 不为空字符串
    assert path != ""
    # 将 submodule 初始化为 mod
    submodule = mod
    # 遍历路径中的每个名称，访问子模块
    for name in path.split("."):
        # 如果 submodule 没有名为 name 的属性，抛出运行时错误
        if not hasattr(submodule, name):
            raise RuntimeError(f"Couldn't find submodule at path {path}")
        submodule = getattr(submodule, name)

    # 定义 update_module_call_signatures 函数，更新模块调用签名
    def update_module_call_signatures(path, in_spec, out_spec):
        # 如果路径已经存在于 module_call_specs 中，则断言输入和输出规范相等
        if path in module_call_specs:
            assert module_call_specs[path]["in_spec"] == in_spec
            assert module_call_specs[path]["out_spec"] == out_spec
        # 更新 module_call_specs[path] 的值
        module_call_specs[path] = {"in_spec": in_spec, "out_spec": out_spec}

    # 定义 check_flattened 函数，检查扁平化参数的类型
    def check_flattened(flat_args):
        # 遍历 flat_args 中的每个参数 a
        for a in flat_args:
            # 如果 a 不是 torch.Tensor、str、int、float、bool 或 None 类型，则抛出断言错误
            if not (isinstance(a, (torch.Tensor, str, int, float, bool)) or a is None):
                raise AssertionError(
                    f"Only Tensors or scalars are supported as pytree flattened inputs, got: {a}"
                )
    def pre_hook(module, args, kwargs):
        # 将输入的 args 和 kwargs 扁平化，以便进行检查
        flat_args, in_spec = pytree.tree_flatten((args, kwargs))
        # 对扁平化后的参数进行检查
        check_flattened(flat_args)
        # 在导出跟踪点之前，为模块调用输入生成跟踪数据
        flat_args = _export_tracepoint(*flat_args, kind="module_call_inputs", path=path)
        # 将扁平化的参数重新组装成原始结构
        args, kwargs = pytree.tree_unflatten(flat_args, in_spec)
        # 返回处理后的参数和关键字参数
        return args, kwargs

    def post_hook(module, args, kwargs, res):
        # 将输入的 args 和 kwargs 扁平化，以便进行检查
        _, in_spec = pytree.tree_flatten((args, kwargs))
        # 将结果 res 扁平化，以便进行检查
        flat_res, out_spec = pytree.tree_flatten(res)
        # 对扁平化后的结果进行检查
        check_flattened(flat_res)
        # 在导出跟踪点之前，为模块调用输出生成跟踪数据
        flat_res = _export_tracepoint(*flat_res, kind="module_call_outputs", path=path)
        # 更新模块调用的签名
        update_module_call_signatures(path, in_spec, out_spec)
        # 将扁平化的结果重新组装成原始结构
        return pytree.tree_unflatten(flat_res, out_spec)

    # 注册前向预处理钩子，捕获模块调用前的参数和关键字参数
    pre_handle = submodule.register_forward_pre_hook(pre_hook, with_kwargs=True)
    # 注册后向处理钩子，捕获模块调用后的参数、关键字参数和结果
    post_handle = submodule.register_forward_hook(post_hook, with_kwargs=True)
    # 返回注册的预处理和后处理钩子
    return pre_handle, post_handle
# 定义一个上下文管理器函数，用于包装子模块
@contextmanager
def _wrap_submodules(f, preserve_signature, module_call_signatures):
    # 用于存储所有的处理句柄
    handles = []

    try:
        # 遍历需要保留签名的路径列表
        for path in preserve_signature:
            # 调用子模块包装函数，并将返回的句柄扩展到handles列表中
            handles.extend(_wrap_submodule(f, path, module_call_signatures))
        # 执行 yield，将控制权交给调用方
        yield
    finally:
        # 在退出前，移除所有的句柄
        for handle in handles:
            handle.remove()


# 给类添加一个名为__call__的方法，实现严格模式的实验功能
def _mark_strict_experimental(cls):
    # 定义一个新的call方法，接受任意数量的参数并返回调用strict_mode的结果
    def call(self, *args):
        return strict_mode(self, args)

    # 将新定义的call方法绑定到类的__call__属性上
    cls.__call__ = call
    # 返回修改后的类对象
    return cls
```