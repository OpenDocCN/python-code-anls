# `.\pytorch\torch\_dynamo\backends\registry.py`

```
# mypy: ignore-errors  # 忽略类型检查错误，用于标记该文件中不应进行类型检查的部分

import functools  # 导入 functools 模块，用于高阶函数支持
import sys  # 导入 sys 模块，用于系统相关操作
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple  # 导入类型提示相关的类和函数

import torch  # 导入 PyTorch 模块
from torch import fx  # 导入 PyTorch 的 FX 模块

# 定义一个 Protocol 类型，用于表示编译后的函数对象
class CompiledFn(Protocol):
    def __call__(self, *args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ...


# 定义一个类型别名，表示编译函数的类型
CompilerFn = Callable[[fx.GraphModule, List[torch.Tensor]], CompiledFn]

# 创建一个全局的字典，用于存储不同后端的编译函数
_BACKENDS: Dict[str, CompilerFn] = dict()


def register_backend(
    compiler_fn: Optional[CompilerFn] = None,
    name: Optional[str] = None,
    tags: Sequence[str] = (),
):
    """
    Decorator to add a given compiler to the registry to allow calling
    `torch.compile` with string shorthand.  Note: for projects not
    imported by default, it might be easier to pass a function directly
    as a backend and not use a string.

    Args:
        compiler_fn: Callable taking a FX graph and fake tensor inputs
        name: Optional name, defaults to `compiler_fn.__name__`
        tags: Optional set of string tags to categorize backend with
    """
    if compiler_fn is None:
        # 支持 @register_backend(name="") 语法的使用方式
        return functools.partial(register_backend, name=name, tags=tags)
    assert callable(compiler_fn)  # 断言编译函数是可调用的
    name = name or compiler_fn.__name__  # 使用给定名称或者默认使用编译函数的名称
    assert name not in _BACKENDS, f"duplicate name: {name}"  # 确保名称在字典中唯一
    _BACKENDS[name] = compiler_fn  # 将编译函数加入到后端字典中
    compiler_fn._tags = tuple(tags)  # 设置编译函数的标签元组
    return compiler_fn


# 创建一个特定标签的注册函数，用于调试目的
register_debug_backend = functools.partial(register_backend, tags=("debug",))
# 创建一个特定标签的注册函数，用于实验性质的后端
register_experimental_backend = functools.partial(
    register_backend, tags=("experimental",)
)


def lookup_backend(compiler_fn):
    """Expand backend strings to functions"""
    if isinstance(compiler_fn, str):  # 如果编译函数是字符串类型
        if compiler_fn not in _BACKENDS:  # 如果字符串不在后端字典中
            _lazy_import()  # 惰性导入相关模块
        if compiler_fn not in _BACKENDS:  # 如果字符串不在后端字典中
            _lazy_import_entry_point(compiler_fn)  # 尝试从入口点惰性导入
        if compiler_fn not in _BACKENDS:  # 如果字符串不在后端字典中
            from ..exc import InvalidBackend  # 导入自定义异常类

            raise InvalidBackend(name=compiler_fn)  # 抛出后端无效异常
        compiler_fn = _BACKENDS[compiler_fn]  # 获取后端函数对象
    return compiler_fn  # 返回编译函数对象


def list_backends(exclude_tags=("debug", "experimental")) -> List[str]:
    """
    Return valid strings that can be passed to:

        torch.compile(..., backend="name")
    """
    _lazy_import()  # 惰性导入相关模块
    exclude_tags = set(exclude_tags or ())  # 将排除标签转换为集合
    return sorted(
        [
            name
            for name, backend in _BACKENDS.items()  # 遍历后端字典
            if not exclude_tags.intersection(backend._tags)  # 根据标签排除特定后端
        ]
    )


@functools.lru_cache(None)
def _lazy_import():
    from .. import backends  # 导入后端模块
    from ..utils import import_submodule  # 导入子模块导入工具

    import_submodule(backends)  # 导入指定的后端模块

    from ..repro.after_dynamo import dynamo_minifier_backend  # 导入动态后端相关模块

    assert dynamo_minifier_backend is not None  # 断言动态后端不为空


@functools.lru_cache(None)
def _lazy_import_entry_point(backend_name: str):
    from importlib.metadata import entry_points  # 导入入口点相关模块

    compiler_fn = None  # 初始化编译函数对象
    group_name = "torch_dynamo_backends"  # 指定组名称
    # 检查 Python 版本是否低于 3.10
    if sys.version_info < (3, 10):
        # 获取所有的 entry points
        backend_eps = entry_points()
        # 筛选出与 backend_name 匹配的 entry points
        eps = [ep for ep in backend_eps.get(group_name, ()) if ep.name == backend_name]
        # 如果找到匹配的 entry points
        if len(eps) > 0:
            # 加载第一个匹配的 entry point 对应的编译器函数
            compiler_fn = eps[0].load()
    else:
        # 获取特定组名下的所有 entry points
        backend_eps = entry_points(group=group_name)
        # 检查指定的 backend_name 是否在 entry points 中
        if backend_name in backend_eps.names:
            # 加载指定 backend_name 对应的编译器函数
            compiler_fn = backend_eps[backend_name].load()

    # 如果成功加载了编译器函数并且 backend_name 不在已注册的后端列表中
    if compiler_fn is not None and backend_name not in list_backends(tuple()):
        # 注册该编译器函数作为指定 backend_name 的后端
        register_backend(compiler_fn=compiler_fn, name=backend_name)
```