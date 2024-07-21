# `.\pytorch\torch\fx\graph_module.py`

```py
# mypy: allow-untyped-defs
# 引入需要的模块和库
import contextlib  # 提供了一些用于管理上下文的工具，如关闭文件、锁的创建等
import copy  # 提供了用于复制对象的功能
import itertools  # 提供了用于创建和操作迭代器的函数
import linecache  # 缓存模块，用于按行存储和检索源代码
import os  # 提供了访问操作系统服务的功能
import sys  # 提供了对 Python 解释器的访问和控制
import traceback  # 提供了追踪异常和记录追踪信息的功能
import warnings  # 用于管理警告信息的模块
from pathlib import Path  # 提供了处理路径名的类
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union  # 提供了类型提示的支持

import torch  # PyTorch 深度学习库
import torch.nn as nn  # PyTorch 中定义神经网络层的模块
import torch.overrides  # 提供了在模型中覆盖方法的支持
from torch.nn.modules.module import _addindent  # 用于缩进字符串的辅助函数
from torch.package import Importer, PackageExporter, PackageImporter, sys_importer  # 与 PyTorch 包相关的导入器和导出器

from ._compatibility import compatibility  # 导入兼容性相关模块
from .graph import _custom_builtins, _is_from_torch, _PyTreeCodeGen, Graph, PythonCode  # 导入与图形相关的模块和类

__all__ = [  # 模块中公开的符号列表
    "reduce_graph_module",
    "reduce_package_graph_module",
    "reduce_deploy_graph_module",
    "GraphModule",
]

_USER_PRESERVED_ATTRIBUTES_KEY = "_user_preserved_attributes"

# Normal exec loses the source code, however we can work with
# the linecache module to recover it.
# Using _exec_with_source will add it to our local cache
# and then tools like TorchScript will be able to get source info.
class _EvalCacheLoader:
    def __init__(self):
        self.eval_cache = {}  # 初始化空的源代码缓存
        self.next_id = 0  # 初始化源代码缓存键的序号

    def cache(self, src: str, globals: Dict[str, Any], co_fields=None):
        """Store the source in a private cache, and add a lazy entry in linecache
        that allows the source to be retrieved by 'filename'.

        Args:
            src (str): The module source to cache
            globals (dict): The module globals

        Returns:
            str: The cache key (and dummy filename) generated for src.
        """

        key = self._get_key()  # 获取一个新的缓存键
        if co_fields:
            key += f" from {co_fields['co_filename']}:{co_fields['co_firstlineno']} in {co_fields['co_name']}"
        self.eval_cache[key] = src  # 将源代码存储到缓存中

        # 不要修改 globals，这样该加载器只用于填充 linecache，并且不与可能检查 `__loader__` 的其他模块交互
        globals_copy = globals.copy()
        globals_copy["__file__"] = key
        globals_copy["__name__"] = key
        globals_copy["__loader__"] = self
        linecache.lazycache(key, globals_copy)  # 向 linecache 中添加延迟加载条目

        return key  # 返回生成的缓存键

    # Part of the loader protocol (PEP 302)
    # linecache 在尝试查找源代码时会使用该方法
    def get_source(self, module_name) -> Optional[str]:
        if module_name in self.eval_cache:
            return self.eval_cache[module_name]  # 如果存在缓存的源代码，返回源代码字符串
        return None  # 否则返回 None

    def _get_key(self):
        key = f"<eval_with_key>.{self.next_id}"  # 生成新的缓存键
        self.next_id += 1  # 更新键的序号
        return key  # 返回生成的缓存键


_loader = _EvalCacheLoader()  # 创建一个源代码缓存加载器的实例


def _exec_with_source(src: str, globals: Dict[str, Any], co_fields=None):
    key = _loader.cache(src, globals, co_fields)  # 使用加载器缓存源代码
    exec(compile(src, key, "exec"), globals)  # 执行编译后的源代码，并将结果存储在 globals 中


def _forward_from_src(src: str, globals: Dict[str, Any], co_fields=None):
    return _method_from_src(
        method_name="forward", src=src, globals=globals, co_fields=co_fields
    )


def _method_from_src(
    method_name: str, src: str, globals: Dict[str, Any], co_fields=None
):
    # 从源代码中获取特定方法的实现
# 复制全局变量字典以避免修改传入的字典
def _exec_with_globals(src: str, globals: Dict[str, Any], method_name: str, co_fields: Dict[str, Any]) -> Callable:
    globals_copy = globals.copy()  # 复制全局变量字典，避免直接修改原始字典
    _exec_with_source(src, globals_copy, co_fields)  # 使用给定源代码、复制后的全局变量字典和字段字典执行操作
    fn = globals_copy[method_name]  # 从复制后的全局变量字典中获取指定方法名对应的函数
    del globals_copy[method_name]  # 从复制后的全局变量字典中删除指定方法名对应的函数
    return fn  # 返回获取的函数


def _format_import_statement(name: str, obj: Any, importer: Importer) -> str:
    if name in _custom_builtins:
        return _custom_builtins[name].import_str  # 如果名称在自定义内置对象字典中，返回其导入字符串
    if _is_from_torch(name):
        return "import torch"  # 如果名称来自torch，返回torch导入语句
    module_name, attr_name = importer.get_name(obj)  # 使用导入器获取对象的模块名和属性名
    return f"from {module_name} import {attr_name} as {name}"  # 返回格式化的导入语句


def _format_import_block(globals: Dict[str, Any], importer: Importer):
    import_strs: Set[str] = {_format_import_statement(name, obj, importer) for name, obj in globals.items()}
    # 获取所有全局变量的导入语句并存储在集合中
    # 对导入语句进行排序，以确保稳定的导入块，这样可以对图模块进行哈希并获得一致的缓存键。
    return "\n".join(sorted(import_strs))  # 返回按照名称排序后的导入语句块


@compatibility(is_backward_compatible=True)
def reduce_graph_module(body: Dict[Any, Any], import_block: str) -> torch.nn.Module:
    # BC: attribute name was changed from `code` to `_code` to facilitate
    # making `code` into a property and adding a docstring to it
    fn_src = body.get("_code") or body["code"]  # 获取函数源码，优先使用'_code'，否则使用'code'
    forward = _forward_from_src(import_block + fn_src, {})  # 使用导入块和函数源码生成前向方法
    return _deserialize_graph_module(forward, body)  # 反序列化图模块并返回


@compatibility(is_backward_compatible=True)
def reduce_package_graph_module(
    importer: PackageImporter, body: Dict[Any, Any], generated_module_name: str
) -> torch.nn.Module:
    forward = importer.import_module(generated_module_name).forward  # 使用包导入器导入生成的模块并获取前向方法
    return _deserialize_graph_module(forward, body)  # 反序列化图模块并返回


@compatibility(is_backward_compatible=True)
def reduce_deploy_graph_module(
    importer: PackageImporter, body: Dict[Any, Any], import_block: str
) -> torch.nn.Module:
    ns = {}
    ns["__builtins__"] = importer.patched_builtins  # 设置命名空间的内置对象为补丁后的内置对象
    fn_src = body.get("_code")
    assert fn_src is not None
    forward = _forward_from_src(import_block + fn_src, ns)  # 使用导入块和函数源码生成前向方法
    return _deserialize_graph_module(forward, body)  # 反序列化图模块并返回


# We create a dummy class here because symbolic_trace pulls the forward()
# function off of the class, rather than the instance. This class is used
# in _deserialize_graph_module() below.
class _CodeOnlyModule(torch.nn.Module):
    def __init__(self, body):
        super().__init__()
        self.__dict__ = body  # 将类的__dict__属性设置为给定的body字典，用于反序列化图模块


def _deserialize_graph_module(forward, body: Dict[Any, Any], graph_module_cls=None) -> torch.nn.Module:
    """
    Deserialize a GraphModule given the dictionary of the original module,
    using the code to reconstruct the graph. We delete the actual graph before
    saving the dictionary so that changes to the in-memory graph format do not
    get serialized.
    """
    _CodeOnlyModule.forward = forward  # 将前向方法设置为_CodeOnlyModule类的类属性

    tracer_cls = body.get("_tracer_cls")  # 获取追踪器类（tracer class）
    if tracer_cls is None:
        from ._symbolic_trace import Tracer

        tracer_cls = Tracer

    graphmodule_cls_name = body.get("_graphmodule_cls_name", "GraphModule")

    # This is a workaround for a mypy linter issue related to
    # passing base class as an argument - https://github.com/python/mypy/issues/5865.
    # 设置一个类型标记，用于解决 mypy 检查器在将基类作为参数传递时的问题

    cls_tracer: Any = tracer_cls

    class KeepModules(cls_tracer):
        # we shouldn't trace into any of the submodules,
        # because they were not traced in the original GraphModule
        # 定义一个子类 KeepModules，用于控制追踪行为，保持模块的结构不变
        def is_leaf_module(self, _: torch.nn.Module, __: str) -> bool:
            return True

    com = _CodeOnlyModule(body)

    tracer_extras = body.get("_tracer_extras", {})
    # 通过 KeepModules 类来追踪 _CodeOnlyModule 对象 com，传入额外的追踪参数 tracer_extras
    graph = KeepModules().trace(com, **tracer_extras)

    # Manually set Tracer class on the reconstructed Graph, to avoid
    # referencing the private local subclass KeepModules.
    # 手动设置重建的图形对象 graph 的追踪器类，避免引用私有的局部子类 KeepModules
    graph._tracer_cls = tracer_cls
    from ._lazy_graph_module import _make_graph_module
    # 使用 _make_graph_module 函数创建图形模块 gm，基于 _CodeOnlyModule com 和追踪后的 graph，
    # 指定类名为 graphmodule_cls_name，图形模块类为 graph_module_cls
    gm = _make_graph_module(com, graph, class_name=graphmodule_cls_name, graph_module_cls=graph_module_cls)

    # The GraphModule constructor only retains attributes referenced by the graph.
    # In this case, our goal is return a GraphModule as close to identical as the one
    # put into the package. If any additional attributes were present in body,
    # we should keep them.
    # 将 body 中未在 gm 中存在的属性添加到 gm 中，以确保返回的 GraphModule 尽可能与原始包中的相同
    for k, v in body.items():
        if not hasattr(gm, k):
            setattr(gm, k, v)
    return gm
# copy an attribute value with qualified name 'target' from 'from_module' to 'to_module'
# This installs empty Modules where none exist yet if they are subpaths of target
def _copy_attr(from_module: torch.nn.Module, to_module: torch.nn.Module, target: str):
    *prefix, field = target.split(".")  # Split the target string into parts separated by '.'
    for item in prefix:
        f = getattr(from_module, item)  # Get attribute 'item' from 'from_module'
        t = getattr(to_module, item, None)  # Get attribute 'item' from 'to_module', or None if not exists
        if f is t:
            # Skip copying if 'item' from 'from_module' is already installed in 'to_module'
            # (e.g., target = root.linear.weight, but root.linear is already installed)
            # Once a parent is installed, no need to copy children since properties are already present
            return

        if t is None:
            t = torch.nn.Module()  # Create an empty Module if 'item' does not exist in 'to_module'
            setattr(to_module, item, t)  # Set the newly created Module to 'item' in 'to_module'
        from_module, to_module = f, t  # Move to the next level in 'from_module' and 'to_module'

    orig = getattr(from_module, field)  # Get the final attribute specified by 'field' from 'from_module'
    # If it is a tensor and not a parameter attribute of a module, register it as a named buffer in 'to_module'
    if isinstance(orig, torch.Tensor) and not isinstance(orig, torch.nn.Parameter):
        to_module.register_buffer(field, orig)
    else:
        setattr(to_module, field, orig)  # Set the attribute 'field' in 'to_module' to 'orig'


# Assign attribute 'from_obj' to the qualified name 'target' on 'to_module'
# This installs empty Modules where none exist yet if they are subpaths of target
def _assign_attr(from_obj: Any, to_module: torch.nn.Module, target: str):
    *prefix, field = target.split(".")  # Split the target string into parts separated by '.'
    for item in prefix:
        t = getattr(to_module, item, None)  # Get attribute 'item' from 'to_module', or None if not exists

        if t is None:
            t = torch.nn.Module()  # Create an empty Module if 'item' does not exist in 'to_module'
            setattr(to_module, item, t)  # Set the newly created Module to 'item' in 'to_module'
        to_module = t  # Move to the next level in 'to_module'

    # If it is a tensor and not a parameter attribute of a module, register it as a named buffer in 'to_module'
    if isinstance(from_obj, torch.Tensor) and not isinstance(
        from_obj, torch.nn.Parameter
    ):
        to_module.register_buffer(field, from_obj)
    else:
        setattr(to_module, field, from_obj)  # Set the attribute 'field' in 'to_module' to 'from_obj'


class _WrappedCall:
    def __init__(self, cls, cls_call):
        self.cls = cls  # Store the class object in self.cls
        self.cls_call = cls_call  # Store the callable class function in self.cls_call

    # Previously, if an error occurred when valid
    # symbolically-traced code was run with an invalid input, the
    # user would see the source of the error as coming from
    # `File "<eval_with_key_N">`, where N is some number. We use
    # this function to generate a more informative error message. We
    # return the traceback itself, a message explaining that the
    # error occurred in a traced Module's generated forward
    # function, and five lines of context surrounding the faulty
    # line
    @staticmethod
    # 生成错误消息的辅助函数，接收一个 traceback.FrameSummary 对象作为参数，并返回错误消息的字符串形式
    def _generate_error_message(frame_summary: traceback.FrameSummary) -> str:
        # 获取错误发生的行号
        err_lineno = frame_summary.lineno
        assert err_lineno is not None  # 确保行号不为空
        # 获取出错的代码行内容
        line = frame_summary.line
        assert line is not None  # 确保代码行内容不为空
        # 获取出错代码行的长度
        err_line_len = len(line)
        # 获取所有源代码行
        all_src_lines = linecache.getlines(frame_summary.filename)

        # 构建错误消息的各部分字符串
        tb_repr = torch._dynamo.disable(traceback.format_exc)()  # 获取当前 traceback 的字符串表示
        custom_msg = (
            "Call using an FX-traced Module, "
            f"line {err_lineno} of the traced Module's "
            "generated forward function:"
        )  # 自定义的错误消息
        before_err = "".join(all_src_lines[err_lineno - 2 : err_lineno])  # 出错行之前的两行代码
        marker = "~" * err_line_len + "~~~ <--- HERE"  # 标记出错位置的符号
        err_and_after_err = "\n".join(all_src_lines[err_lineno : err_lineno + 2])  # 出错行及其后续两行代码

        # 拼接所有部分，形成最终的错误消息
        return "\n".join([tb_repr, custom_msg, before_err, marker, err_and_after_err])

    # 实现类的调用操作符重载，用于处理对象的调用操作
    def __call__(self, obj, *args, **kwargs):
        try:
            # 如果定义了 cls_call 方法，则调用它来处理对象的调用
            if self.cls_call is not None:
                return self.cls_call(obj, *args, **kwargs)
            else:
                # 否则，调用父类的调用操作符重载方法来处理对象的调用
                return super(self.cls, obj).__call__(*args, **kwargs)  # type: ignore[misc]
        except Exception as e:
            assert e.__traceback__  # 确保异常对象有 traceback 信息
            # 获取异常堆栈中最顶层的 FrameSummary 对象
            topmost_framesummary: traceback.FrameSummary = (
                traceback.StackSummary.extract(traceback.walk_tb(e.__traceback__))[-1]
            )  # type: ignore[arg-type]
            # 如果异常发生在 "eval_with_key" 的文件中
            if "eval_with_key" in topmost_framesummary.filename:
                # 打印生成的错误消息到标准错误流中
                print(
                    _WrappedCall._generate_error_message(topmost_framesummary),
                    file=sys.stderr,
                )
                # 重新引发异常，但清除 traceback 信息
                raise e.with_traceback(None)  # noqa: B904
            else:
                # 否则，直接重新引发异常
                raise e
# 根据兼容性装饰器定义一个带有自定义属性的类 GraphModule
@compatibility(is_backward_compatible=True)
class GraphModule(torch.nn.Module):
    """
    GraphModule is an nn.Module generated from an fx.Graph. Graphmodule has a
    ``graph`` attribute, as well as ``code`` and ``forward`` attributes generated
    from that ``graph``.

    .. warning::

        When ``graph`` is reassigned, ``code`` and ``forward`` will be automatically
        regenerated. However, if you edit the contents of the ``graph`` without reassigning
        the ``graph`` attribute itself, you must call ``recompile()`` to update the generated
        code.
    """

    # 使用 __new__ 方法来生成 GraphModuleImpl 的单例类
    def __new__(cls: "Type[GraphModule]", *args, **kwargs):
        # 每个 GraphModule 实例都需要有其自己的 forward 方法
        # 因此为每个实例创建一个新的单例类
        # 它是用户定义类的子类，唯一的区别是多了一层用来安装 forward 方法

        # 解决 https://github.com/pytorch/pytorch/issues/63883 中描述的问题
        # 换句话说，遍历类层次结构以解决多余的类定义问题
        for t in cls.__mro__:
            c = t.__qualname__.split(".")[-1]
            if c != "GraphModuleImpl":
                cls = t
                break

        class GraphModuleImpl(cls):  # type: ignore[misc, valid-type]
            pass

        return super().__new__(GraphModuleImpl)

    # 初始化方法，接受 root、graph 和 class_name 三个参数
    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: Graph,
        class_name: str = "GraphModule",
    # TorchScript 因为持续的字符串字面量尝试编译 graph 设置器而中断
    # 问题在于：https://github.com/pytorch/pytorch/issues/44842
    #
    # 由于这些方法本来就不应该在 TorchScript 中使用，所以不应该是一个问题
    __jit_unused_properties__ = ["graph"]

    # 返回当前 GraphModule 的 graph 属性
    @property
    def graph(self) -> Graph:
        """
        Return the ``Graph`` underlying this ``GraphModule``
        """
        return self._graph

    # 设置当前 GraphModule 的 graph 属性，重新编译以确保生成的 forward 方法与新的 graph 对应
    @graph.setter
    def graph(self, g: Graph) -> None:
        """
        Set the underlying ``Graph`` for this ``GraphModule``. This will internally
        recompile the ``GraphModule`` so that the generated ``forward()`` function
        corresponds to ``g``
        """
        assert isinstance(g, Graph), f"Expected a Graph instance, but got {type(g)}"
        self._graph = g
        g.owning_module = self
        self.recompile()

    # 定义一个不兼容的装饰器，用于某些特定情况
    @compatibility(is_backward_compatible=False)
        def to_folder(self, folder: Union[str, os.PathLike], module_name: str = "FxModule"):
            """将模块转储到指定的文件夹，使用指定的模块名，以便可以通过 `from <folder> import <module_name>` 导入

            Args:
                folder (Union[str, os.PathLike]): 要写入代码的目标文件夹路径
                module_name (str): 写出代码时要使用的模块的顶层名称
            """
            # 将输入的文件夹路径转换为Path对象
            folder = Path(folder)
            # 如果文件夹不存在，则创建它
            Path(folder).mkdir(exist_ok=True)
            # 将当前模型的状态字典保存为名为 "state_dict.pt" 的文件
            torch.save(self.state_dict(), folder / "state_dict.pt")
            # 创建一个包含4个空格的制表符字符串
            tab = " " * 4
            # 获取自定义内置对象的导入字符串，每个字符串以换行符分隔
            custom_builtins = "\n".join([v.import_str for v in _custom_builtins.values()])
            # 创建一个模块字符串，包含换行符，用于生成目标文件夹中的 Python 模块文件
            model_str = f"""
import torch
{custom_builtins}

from torch.nn import *

class {module_name}(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化函数，用于定义模型结构和初始化参数

        def _gen_model_repr(module_name: str, module: torch.nn.Module) -> Optional[str]:
            # 生成模型的字符串表示，用于保存模型或者在需要时输出模型信息
            safe_reprs = [
                nn.Linear,
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
            ]
            if type(module) in safe_reprs:
                return f"{module.__repr__()}"
            else:
                return None

        blobified_modules = []
        for module_name, module in self.named_children():
            # 遍历模型的子模块，生成各子模块的字符串表示或保存为文件
            module_str = _gen_model_repr(module_name, module)
            if module_str is None:
                module_file = folder / f"{module_name}.pt"
                torch.save(module, module_file)
                blobified_modules.append(module_name)
                module_repr = module.__repr__().replace("\r", " ").replace("\n", " ")
                module_str = f"torch.load(r'{module_file}') # {module_repr}"
            model_str += f"{tab*2}self.{module_name} = {module_str}\n"

        for buffer_name, buffer in self._buffers.items():
            if buffer is None:
                continue
            # 注册模型的缓冲区
            model_str += f"{tab*2}self.register_buffer('{buffer_name}', torch.empty({list(buffer.shape)}, dtype={buffer.dtype}))\n"

        for param_name, param in self._parameters.items():
            if param is None:
                continue
            # 设置模型的参数
            model_str += f"{tab*2}self.{param_name} = torch.nn.Parameter(torch.empty({list(param.shape)}, dtype={param.dtype}))\n"

        # 加载模型的状态字典
        model_str += (
            f"{tab*2}self.load_state_dict(torch.load(r'{folder}/state_dict.pt'))\n"
        )
        model_str += f"{_addindent(self.code, 4)}\n"  # 可能是代码生成的部分，需要进一步确认

        module_file = folder / "module.py"
        module_file.write_text(model_str)

        init_file = folder / "__init__.py"
        init_file.write_text("from .module import *")

        if len(blobified_modules) > 0:
            # 如果有子模块无法直接生成字符串表示，则保存为pickle文件，同时发出警告
            warnings.warn(
                "Was not able to save the following children modules as reprs -"
                f"saved as pickled files instead: {blobified_modules}"
            )

    @compatibility(is_backward_compatible=True)
    # 标记函数兼容性信息，这里指示函数是向后兼容的
    def add_submodule(self, target: str, m: torch.nn.Module) -> bool:
        """
        Adds the given submodule to ``self``.

        This installs empty Modules where none exist yet if they are
        subpaths of ``target``.

        Args:
            target: The fully-qualified string name of the new submodule
                (See example in ``nn.Module.get_submodule`` for how to
                specify a fully-qualified string.)
            m: The submodule itself; the actual object we want to
                install in the current Module

        Return:
            bool: Whether or not the submodule could be inserted. For
                this method to return True, each object in the chain
                denoted by ``target`` must either a) not exist yet,
                or b) reference an ``nn.Module`` (not a parameter or
                other attribute)
        """
        # 将目标字符串按点号分隔，最后一个元素作为字段名，前面部分作为路径前缀
        *prefix, field = target.split(".")
        # 将当前对象初始化为根模块
        mod: torch.nn.Module = self

        # 遍历路径前缀
        for item in prefix:
            # 获取当前模块下的子模块，如果不存在则创建一个空的 Module
            submod = getattr(mod, item, None)
            if submod is None:
                submod = torch.nn.Module()
                setattr(mod, item, submod)

            # 如果获取到的不是 Module 类型，则返回 False
            if not isinstance(submod, torch.nn.Module):
                return False

            # 将当前模块更新为子模块，准备处理下一个路径部分
            mod = submod

        # 将给定的子模块添加到最终模块中
        mod.add_module(field, m)
        return True

    @compatibility(is_backward_compatible=True)
    def delete_submodule(self, target: str) -> bool:
        """
        Deletes the given submodule from ``self``.

        The module will not be deleted if ``target`` is not a valid
        target.

        Args:
            target: The fully-qualified string name of the new submodule
                (See example in ``nn.Module.get_submodule`` for how to
                specify a fully-qualified string.)

        Returns:
            bool: Whether or not the target string referenced a
                submodule we want to delete. A return value of ``False``
                means that the ``target`` was not a valid reference to
                a submodule.
        """
        # 将目标字符串按点号分隔，最后一个元素作为子模块名，前面部分作为路径
        atoms = target.split(".")
        path, target_submod = atoms[:-1], atoms[-1]
        # 将当前对象初始化为根模块
        mod: torch.nn.Module = self

        # 遍历路径部分，找到最终子模块的父模块
        for item in path:
            # 如果当前模块不存在指定的属性，则返回 False
            if not hasattr(mod, item):
                return False

            # 获取当前属性对应的模块
            mod = getattr(mod, item)

            # 如果获取到的不是 Module 类型，则返回 False
            if not isinstance(mod, torch.nn.Module):
                return False

        # 检查最终子模块是否存在于父模块中
        if not hasattr(mod, target_submod):
            return False

        # 如果最终子模块不是 Module 类型，则返回 False
        if not isinstance(getattr(mod, target_submod), torch.nn.Module):
            return False

        # 删除最终子模块
        delattr(mod, target_submod)
        return True

    @compatibility(is_backward_compatible=True)
    def delete_all_unused_submodules(self) -> None:
        """
        Deletes all unused submodules from ``self``.

        A Module is considered "used" if any one of the following is
        true:
        1. It has children that are used
        2. Its forward is called directly via a ``call_module`` node
        3. It has a non-Module attribute that is used from a
        ``get_attr`` node

        This method can be called to clean up an ``nn.Module`` without
        manually calling ``delete_submodule`` on each unused submodule.
        """
        used: List[str] = []  # 用于存放所有被使用的模块名列表

        for node in self.graph.nodes:
            # 检查节点操作类型是否为调用模块或获取属性
            if node.op == "call_module" or node.op == "get_attr":

                # 将目标路径按点分隔为各部分字符串列表，如 `foo.bar.baz` 变为 ["foo", "bar", "baz"]
                fullpath = node.target.split(".")

                # 函数用于将两个字符串连接成路径，处理节点目标的路径拼接
                def join_fn(x: str, y: str) -> str:
                    return ".".join([x, y] if y else [x])

                # 逐步收集所有中间模块的名称，例如，对于目标 `foo.bar.baz`，将添加 `foo`、`foo.bar` 和 `foo.bar.baz` 到列表中
                used.extend(itertools.accumulate(fullpath, join_fn))

                # 对于 `call_module` 节点，还需要注册所有递归子模块作为已使用
                if node.op == "call_module":
                    try:
                        submod = self.get_submodule(node.target)

                        # 遍历子模块的所有命名模块，将命名添加到已使用列表中
                        for submod_name, _ in submod.named_modules():
                            if submod_name != "":
                                used.append(".".join([node.target, submod_name]))
                    except AttributeError:
                        # 节点引用不存在的子模块，不需要处理任何内容
                        pass

        # 确定哪些模块未被使用需要删除
        to_delete = [name for name, _ in self.named_modules() if name not in used]

        # 删除所有未被使用的模块
        for name in to_delete:
            self.delete_submodule(name)

    @property
    def code(self) -> str:
        """
        Return the Python code generated from the ``Graph`` underlying this
        ``GraphModule``.
        """
        if not hasattr(self, "_code"):
            raise RuntimeError(
                "Code has not been generated! Please report a bug to PyTorch"
            )
        return self._code

    @compatibility(is_backward_compatible=True)
    # 重新编译该 GraphModule，使用其当前的 graph 属性。如果编辑了包含的 graph，需要调用此方法，
    # 否则生成的代码将过时。
    def recompile(self) -> PythonCode:
        if isinstance(self._graph._codegen, _PyTreeCodeGen):
            # 如果 _graph._codegen 是 _PyTreeCodeGen 类型，则获取其 pytree_info 的输入规范和输出规范
            self._in_spec = self._graph._codegen.pytree_info.in_spec
            self._out_spec = self._graph._codegen.pytree_info.out_spec
        # 生成当前 graph 的 Python 代码
        python_code = self._graph.python_code(root_module="self")
        # 将生成的 Python 代码保存到 _code 属性中
        self._code = python_code.src
        # 将行号映射保存到 _lineno_map 属性中
        self._lineno_map = python_code._lineno_map

        # 获取当前类的类型
        cls = type(self)
        # 如果 _graph 拥有 _co_fields 属性，则使用其值；否则使用空字典
        co_fields = self._graph._co_fields if hasattr(self._graph, "_co_fields") else {}
        # 使用 _forward_from_src 函数从 _code 中生成 forward 方法
        cls.forward = _forward_from_src(self._code, python_code.globals, co_fields)

        # 确定该类是否显式定义了 __call__ 方法，以便包装调用
        cls_call = cls.__call__ if "__call__" in vars(cls) else None

        # 如果当前类没有定义 _wrapped_call 属性，则创建 _WrappedCall 实例并赋值给 _wrapped_call
        if "_wrapped_call" not in vars(cls):
            cls._wrapped_call = _WrappedCall(cls, cls_call)  # type: ignore[attr-defined]

        # 定义一个 call_wrapped 方法，将 _wrapped_call 应用到实例上
        def call_wrapped(self, *args, **kwargs):
            return self._wrapped_call(self, *args, **kwargs)

        # 将 call_wrapped 方法赋值给当前类的 __call__ 方法
        cls.__call__ = call_wrapped  # type: ignore[method-assign]

        # 返回生成的 PythonCode 对象
        return python_code

    # 用于部署时将 GraphModule 序列化，去除 _graph 属性后返回其余属性和导入代码块
    def __reduce_deploy__(self, importer: Importer):
        # 复制对象的 __dict__，去除 "_graph" 属性，添加 "_graphmodule_cls_name" 属性
        dict_without_graph = self.__dict__.copy()
        dict_without_graph["_graphmodule_cls_name"] = self.__class__.__name__
        del dict_without_graph["_graph"]

        # 重新编译并获取生成的 Python 代码
        python_code = self.recompile()
        # 格式化导入代码块，用于导入生成代码中的全局变量
        import_block = _format_import_block(python_code.globals, importer)
        # 返回序列化时需要的元组，包括去除 _graph 后的对象属性字典和导入代码块
        return (reduce_deploy_graph_module, (dict_without_graph, import_block))

    # 用于打包时将 GraphModule 序列化，去除 _graph 属性后返回其余属性和生成的模块名
    def __reduce_package__(self, exporter: PackageExporter):
        # 复制对象的 __dict__，去除 "_graph" 属性，添加 "_graphmodule_cls_name" 属性
        dict_without_graph = self.__dict__.copy()
        dict_without_graph["_graphmodule_cls_name"] = self.__class__.__name__
        del dict_without_graph["_graph"]

        # 生成一个唯一的模块名
        generated_module_name = f"fx-generated._{exporter.get_unique_id()}"
        # 重新编译并获取生成的 Python 代码
        python_code = self.recompile()
        # 格式化导入代码块，用于导入生成代码中的全局变量
        import_block = _format_import_block(python_code.globals, exporter.importer)
        # 将导入代码块与实例的代码合并，并保存为模块代码
        module_code = import_block + self.code
        # 使用导出器保存生成的模块代码
        exporter.save_source_string(generated_module_name, module_code)
        # 返回序列化时需要的元组，包括去除 _graph 后的对象属性字典和生成的模块名
        return (
            reduce_package_graph_module,
            (dict_without_graph, generated_module_name),
        )
    # 定义 __reduce__ 方法，用于序列化 GraphModule。
    # 仅序列化生成的 Python 代码，而不是底层的 Graph 对象。
    # 这是因为 Graph 对象没有关于向后兼容性的硬盘保证，而 Python 源代码有。
    # 在反序列化时，通过生成的代码进行符号化跟踪，以重新生成底层的 Graph。
    def __reduce__(self):
        dict_without_graph = self.__dict__.copy()  # 复制对象的字典属性

        # 重新编译对象生成的 Python 代码
        python_code = self.recompile()
        # 格式化导入块
        import_block = _format_import_block(python_code.globals, sys_importer)
        # 删除字典中的 "_graph" 属性
        del dict_without_graph["_graph"]
        # 返回元组，包含自定义的 reduce 函数和其参数
        return (reduce_graph_module, (dict_without_graph, import_block))

    # 定义 __deepcopy__ 方法，用于深度复制对象
    # 因为定义了 __reduce__ 用于序列化，所以需要定义 __deepcopy__ 避免每次复制对象时都调用 __reduce__ 导致符号化跟踪
    def __deepcopy__(self, memo):
        res = type(self).__new__(type(self))  # 创建新的对象
        memo[id(self)] = res  # 将原对象 ID 映射到新对象
        # 创建一个仅包含代码的假模块，用于深度复制
        fake_mod = _CodeOnlyModule(copy.deepcopy(self.__dict__, memo))
        # 调用 _deepcopy_init 方法来初始化新对象
        self._deepcopy_init()(res, fake_mod, fake_mod.__dict__["_graph"])

        # 复制额外保留的属性，例如 hooks
        extra_preserved_attrs = [
            "_state_dict_hooks",
            "_load_state_dict_pre_hooks",
            "_load_state_dict_post_hooks",
            "_replace_hook",
            "_create_node_hooks",
            "_erase_node_hooks"
        ]
        for attr in extra_preserved_attrs:
            if attr in self.__dict__:
                setattr(res, attr, copy.deepcopy(self.__dict__[attr], memo))
        
        # 深度复制 meta 属性
        res.meta = copy.deepcopy(getattr(self, "meta", {}), memo)
        # 如果存在用户保留属性，则复制到新对象中
        if _USER_PRESERVED_ATTRIBUTES_KEY in res.meta:
            for attr_name, attr in res.meta[_USER_PRESERVED_ATTRIBUTES_KEY].items():
                setattr(res, attr_name, attr)
        
        return res

    # 定义 __copy__ 方法，用于浅复制对象
    def __copy__(self):
        from ._lazy_graph_module import _make_graph_module
        # 使用 _make_graph_module 函数创建一个新的 GraphModule 对象
        res = _make_graph_module(self, self.graph)
        # 复制 meta 属性
        res.meta = getattr(self, "meta", {})
        return res

    # 用于标记对象不向后兼容的装饰器，无实际代码作用
    @compatibility(is_backward_compatible=False)
    def print_readable(self, print_output=True, include_stride=False, include_device=False, colored=False):
        """
        Return the Python code generated for current GraphModule and its children GraphModules
        """
        # 生成当前 GraphModule 及其子 GraphModule 的 Python 代码
        verbose_python_code = self._graph.python_code(
            root_module="self", verbose=True, include_stride=include_stride, include_device=include_device, colored=colored
        )
        # 获取 Python 代码字符串
        module_code = verbose_python_code.src
        # 去除代码开头的换行符
        module_code = module_code.lstrip("\n")
        # 将 Python 代码包装为类，并添加类名
        module_code = f"class {self._get_name()}(torch.nn.Module):\n" + module_code
        # 添加适当的缩进
        module_code = _addindent(module_code, 4)

        # 收集子模块的代码
        submodule_code_list = [""]
        for submodule in self.children():
            if isinstance(submodule, GraphModule):
                submodule_code_list.append(submodule.print_readable(print_output=False))
        submodule_code = "\n".join(submodule_code_list)
        # 添加适当的缩进
        submodule_code = _addindent(submodule_code, 4)

        # 合并主模块和子模块的代码
        output = module_code + submodule_code
        # 如果需要打印输出，则打印代码
        if print_output:
            print(module_code + submodule_code)
        # 返回生成的代码字符串
        return output

    def __str__(self) -> str:
        orig_str = super().__str__()
        # 添加打印可读性代码的提醒信息
        print_readable_reminder = (
            "# To see more debug info, please use `graph_module.print_readable()`"
        )
        # 返回对象的原始字符串、代码和提醒信息的组合
        return "\n".join([orig_str, self._code, print_readable_reminder])

    def _replicate_for_data_parallel(self):
        # 复制当前对象以用于数据并行处理
        new_gm = self.__copy__()
        new_gm._is_replica = True
        return new_gm

    @contextlib.contextmanager
    def _set_replace_hook(self, f):
        """
        Takes a callable which will be called everytime when we replace a node
        to a new node, or change the node's name. Callable takes three arguments:
        the old node we're changing, and NAME of the new node, followed by the
        user node which consumes the old node to be replaced.
        """
        # 设置替换钩子，用于在替换或更改节点时调用指定的可调用对象
        assert callable(f), "Replace hook must be a callable."
        prev, self._replace_hook = self._replace_hook, f
        try:
            yield
        finally:
            # 恢复之前的替换钩子
            self._replace_hook = prev

    def _register_create_node_hook(self, f):
        """
        Takes a callable which will be called after we create a new node. The
        callable takes the newly created node as input and returns None.
        """
        # 注册创建节点钩子，用于在创建新节点后调用指定的可调用对象
        assert callable(f), "create_node hook must be a callable."
        self._create_node_hooks.append(f)

    def _unregister_create_node_hook(self, f):
        """
        Takes a callable which was previously registered to be called after we create a node.
        This function will unregister that callable so it is no longer invoked on node creation.
        """
        # 取消注册创建节点钩子，用于在创建节点后取消之前注册的可调用对象
        assert callable(f), "create_node hook must be a callable."
        self._create_node_hooks.remove(f)
    # 注册一个回调函数，用于在擦除节点后调用。这个函数接受被擦除的节点作为输入，并返回None。
    def _register_erase_node_hook(self, f):
        """
        Takes a callable which will be called after we erase a node. The
        callable takes the node that is being erased as input and returns None.
        """
        assert callable(f), "erase_node hook must be a callable."
        # 将可调用对象 f 添加到擦除节点后调用的钩子列表中
        self._erase_node_hooks.append(f)

    # 取消注册一个在擦除节点后调用的回调函数。
    def _unregister_erase_node_hook(self, f):
        """
        Takes a callable which was previously registered to be called after we erase a node.
        This function will unregister that callable so it is no longer invoked on node erasure.
        """
        assert callable(f), "erase_node hook must be a callable."
        # 从擦除节点后调用的钩子列表中移除可调用对象 f
        self._erase_node_hooks.remove(f)
# 解决 __torch_function__ 中的问题的临时解决方案

# 对于 __torch_function__ 不能处理张量列表的问题，
# 修复方案位于 https://github.com/pytorch/pytorch/pull/34725
# 保存原始的 torch.cat 函数引用
orig_cat = torch.cat
# 定义一个新的 patched_cat 函数来替代原始的 torch.cat 函数
def patched_cat(*args, **kwargs):
    # 获取参数中的张量列表
    tensors = args[0]
    # 遍历张量列表
    for t in tensors:
        # 如果张量是 Proxy 类型
        if isinstance(t, Proxy):
            # 调用 Proxy 对象的 __torch_function__ 方法来处理 patched_cat 函数
            return t.__torch_function__(patched_cat, (), args, kwargs)
    # 如果没有 Proxy 对象，则调用原始的 torch.cat 函数
    return orig_cat(*args, **kwargs)

# 设置 patched_cat 函数的模块名为 'torch'
patched_cat.__module__ = 'torch'
# 设置 patched_cat 函数的名称为 'cat'
patched_cat.__name__ = 'cat'
# 将 torch.cat 替换为 patched_cat 函数
torch.cat = patched_cat
```