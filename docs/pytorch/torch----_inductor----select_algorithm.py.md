# `.\pytorch\torch\_inductor\select_algorithm.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和库
import builtins                 # 内建模块
import contextlib               # 上下文管理工具
import functools               # 函数装饰器和高阶函数
import inspect                 # 解析源码
import itertools               # 创建迭代器的函数
import json                    # JSON 编码和解码
import logging                 # 日志记录

import math                    # 数学函数
import operator                # 操作符函数
import os                      # 操作系统功能
import sys                     # Python 解释器参数和函数
import textwrap                # 文本包装和填充
import time                    # 时间函数
from collections import namedtuple   # 命名元组
from concurrent.futures import as_completed, ThreadPoolExecutor   # 并发执行
from io import StringIO       # 文本 I/O

from typing import Any, Callable, Dict, List, Optional, Tuple, Union   # 类型提示
from unittest.mock import patch   # 单元测试模拟

import sympy                   # 符号计算库
from filelock import FileLock  # 文件锁

import torch                   # PyTorch 深度学习库
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._dynamo.testing import rand_strided   # 动态生成随机步幅数据
from torch._dynamo.utils import counters, identity, preserve_rng_state   # 助手函数

from . import config, ir   # 导入当前目录下的 config 和 ir 模块
from .autotune_process import TensorMeta, TritonBenchmarkRequest   # 自动调优相关
from .codecache import code_hash, PersistentCache, PyCodeCache   # 代码缓存
from .codegen.common import IndentedBuffer, KernelTemplate   # 代码生成通用功能

from .codegen.triton import (
    gen_common_triton_imports,
    texpr,
    TritonKernel,
    TritonPrinter,
    TritonScheduling,
)   # Triton 代码生成器相关函数

from .codegen.triton_utils import config_of, signature_to_meta   # Triton 工具函数
from .exc import CUDACompileError   # CUDA 编译错误异常
from .ir import ChoiceCaller, PrimitiveInfoType   # IR 相关
from .runtime.hints import DeviceProperties   # 设备属性提示
from .runtime.runtime_utils import do_bench   # 运行时工具函数
from .utils import (
    FakeIndentedBuffer,
    get_dtype_size,
    Placeholder,
    restore_stdout_stderr,
    sympy_dot,
    sympy_index_symbol,
    sympy_product,
    unique,
)   # 实用工具函数

from .virtualized import V   # 虚拟化相关

log = logging.getLogger(__name__)   # 获取当前模块的日志记录器

# correctness checks struggle with fp16/tf32
VERIFY: Dict[str, Any] = dict()   # 用于存储验证相关信息的字典
PRINT_AUTOTUNE = True   # 是否打印自动调优信息
DEBUG = False   # 是否开启调试模式

class KernelNamespace:
    pass

# these objects are imported from the generated wrapper code
extern_kernels = KernelNamespace()   # 从生成的包装代码中导入的对象

class PartialRender:
    """
    Some parts of a template need to be generated at the end, but
    inserted into the template at the start.  This allows doing a bunch
    of replacements after the initial render.
    """

    def __init__(self, code, replacement_hooks):
        super().__init__()
        self.code = code   # 模板代码
        self.replacement_hooks = replacement_hooks   # 替换钩子函数

    def finalize_hook(self, hook_key: str) -> None:
        assert (
            hook_key in self.replacement_hooks
        ), f"{hook_key} not registered in self.replacement_hooks"
        assert (
            self.replacement_hooks[hook_key] is not None
        ), "hook_key can only be called once"
        self.code = self.code.replace(hook_key, self.replacement_hooks[hook_key]())   # 根据钩子键替换模板中的内容
        self.replacement_hooks[hook_key] = None

    def finalize_all(self) -> str:
        for key, fn in self.replacement_hooks.items():
            self.code = self.code.replace(key, fn())   # 替换所有钩子键对应的内容
        return self.code

# This is used to store info needed for lowering each subgraph in triton
# templates
SubgraphInfo = namedtuple(
    "SubgraphInfo",
    [
        "body",
        "template_mask",
        "template_out",
    ],
)   # Triton 模板中降低每个子图所需信息的命名元组
# TritonTemplateKernel 类，继承自 TritonKernel 类
class TritonTemplateKernel(TritonKernel):
    # 初始化方法，接收多个参数
    def __init__(
        self,
        kernel_name,                 # 内核名称
        input_nodes,                 # 输入节点
        output_node,                 # 输出节点
        defines,                     # 定义
        num_stages,                  # 阶段数
        num_warps,                   # Warp 数量
        grid_fn,                     # 网格函数
        meta,                        # 元信息
        call_sizes,                  # 调用大小
        use_jit=False,               # 是否使用 JIT，默认为 False
        prefix_args=0,               # 前缀参数数量，默认为 0
        suffix_args=0,               # 后缀参数数量，默认为 0
        epilogue_fn=identity,        # 结尾函数，默认为 identity 函数
        subgraphs: Optional[List[ir.ComputedBuffer]] = None,  # 子图列表，可选的计算缓冲列表
        *,
        index_dtype,                 # 索引数据类型，必填参数
    ):
        # 调用父类的初始化方法
        super().__init__(
            sympy_product(output_node.get_size()),  # 使用输出节点大小的乘积作为参数调用父类初始化
            sympy.Integer(1),                      # 使用整数 1 作为参数调用父类初始化
            index_dtype=index_dtype,               # 索引数据类型作为参数调用父类初始化
        )
        # 设置对象属性
        self.input_nodes = input_nodes                   # 输入节点列表
        self.output_node = output_node                   # 输出节点
        self.named_input_nodes = {}                     # 命名输入节点的空字典，类型不确定
        self.defines = defines                           # 定义
        self.kernel_name = kernel_name                   # 内核名称
        self.use_jit = use_jit                           # 是否使用 JIT
        self.num_stages = num_stages                     # 阶段数
        self.num_warps = num_warps                       # Warp 数量
        self.grid_fn = grid_fn                           # 网格函数
        self.meta = meta                                 # 元信息
        self.call_sizes = call_sizes                     # 调用大小
        self.prefix_args = prefix_args                   # 前缀参数数量
        self.suffix_args = suffix_args                   # 后缀参数数量
        self.epilogue_fn = epilogue_fn                   # 结尾函数
        self.render_hooks = dict()                       # 渲染钩子的空字典，类型不确定
        self.triton_meta: Optional[Dict[str, object]] = None  # Triton 元数据，可选的字符串到对象的字典
        self.subgraphs: Optional[List[ir.ComputedBuffer]] = subgraphs  # 子图列表，可选的计算缓冲列表

        # 下面的属性 (body, template_mask, output_val) 都用于 Triton 内核代码生成。
        # 它们通过 `set_subgraph_body` 方法在 TritonTemplateKernel 对象上交换。
        self.subgraph_bodies: Dict[str, SubgraphInfo] = {}  # 子图体信息的空字典，键为字符串，值为 SubgraphInfo 对象

        self.body: IndentedBuffer = FakeIndentedBuffer()    # 主体缓冲区，类型为 IndentedBuffer 的 FakeIndentedBuffer 对象
        self.template_mask: Optional[str] = None             # 模板掩码，可选的字符串
        self.template_out: Optional[str] = None              # 模板输出，可选的字符串

    # 上下文管理器，用于设置子图主体
    @contextlib.contextmanager
    def set_subgraph_body(self, body_name: str):
        # 保存旧的 body、template_mask 和 template_out
        old_body, old_mask, old_out = self.body, self.template_mask, self.template_out
        # 断言子图名称存在于 subgraph_bodies 中
        assert body_name in self.subgraph_bodies, body_name
        # 将当前对象的 body、template_mask 和 template_out 设置为指定子图名称的值
        self.body, self.template_mask, self.template_out = self.subgraph_bodies[body_name]
        yield  # 执行操作的代码块
        # 将子图名称及其对应的 SubgraphInfo 存储回 subgraph_bodies
        self.subgraph_bodies[body_name] = SubgraphInfo(
            self.body, self.template_mask, self.template_out
        )
        # 恢复旧的 body、template_mask 和 template_out
        self.body, self.template_mask, self.template_out = old_body, old_mask, old_out

    # 上下文管理器，用于创建子图主体
    @contextlib.contextmanager
    def create_subgraph_body(self, body_name: str):
        # 断言子图名称不存在于 subgraph_bodies 中
        assert body_name not in self.subgraph_bodies
        # 将子图名称及其对应的 SubgraphInfo 存储为空缓冲区的对象
        self.subgraph_bodies[body_name] = SubgraphInfo(IndentedBuffer(), None, None)
        # 使用 set_subgraph_body 方法设置子图主体
        with self.set_subgraph_body(body_name):
            yield  # 执行操作的代码块

    # 返回是否需要 numel 参数，始终返回 False
    def need_numel_args(self):
        return False
    # 估算该内核所占用的总字节数
    def estimate_kernel_num_bytes(self):
        """
        Estimate the total number of bytes this kernel takes.
        For in/out nodes, sizes are counted twice: once for reading and
        once for writing.
        """
        # 获取不同的inplace缓冲区值的数量
        ninplace_args = len(unique(self.args.inplace_buffers.values()))
        num_bytes = []
        # 遍历输入节点和输出节点（最后一个节点），计算每个节点的大小
        for i, inp in enumerate(itertools.chain(self.input_nodes, (self.output_node,))):
            # 获取节点大小的估算值
            size = V.graph.sizevars.size_hints(inp.get_size())
            # 计算节点中元素的数量
            numel = functools.reduce(operator.mul, size)
            # 获取数据类型的字节大小
            dtype_size = get_dtype_size(inp.get_dtype())
            # 计算节点所占用的字节数，考虑是否为inplace参数
            num_bytes.append(numel * dtype_size * (1 + int(i < ninplace_args)))
        # 返回所有节点所占用的总字节数之和
        return sum(num_bytes)

    # 生成与JIT编译相关的代码行
    def jit_lines(self):
        if self.use_jit:
            return "@triton.jit"

        # 获取Python函数的参数默认值、签名等信息
        argdefs, _, signature, _ = self.args.python_argdefs()
        # 根据函数签名创建元数据，指定大小和数据类型
        triton_meta = {
            "signature": signature_to_meta(signature, size_dtype=self.index_dtype),
            "device": DeviceProperties.create(self.output_node.get_device()),
            "constants": {},
        }
        # 获取函数签名的配置信息
        triton_meta["configs"] = [config_of(signature)]
        # 对于标记为equal_to_1的参数编号，设置常量值为1
        for arg_num in triton_meta["configs"][0].equal_to_1:  # type: ignore[index]
            triton_meta["constants"][arg_num] = 1  # type: ignore[index]
        # 获取元数据中的矩阵指令非K维度
        matrix_instr_nonkdim = self.meta.get("matrix_instr_nonkdim", 0)
        if matrix_instr_nonkdim != 0:
            triton_meta["matrix_instr_nonkdim"] = matrix_instr_nonkdim

        # 将生成的Triton元数据赋值给对象的triton_meta属性
        self.triton_meta = triton_meta

        # 创建感应器元数据，包括内核名称和常用的Triton感应器元数据
        inductor_meta = {
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            **TritonKernel.inductor_meta_common(),
        }
        # 如果配置了带宽分析或者内核基准测试，计算内核所占用的GB数量
        if config.profile_bandwidth or config.benchmark_kernel:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            inductor_meta["kernel_num_gb"] = num_gb
        # 返回包含triton_heuristics模板的字符串
        return f"""
            @triton_heuristics.template(
                num_stages={self.num_stages},
                num_warps={self.num_warps},
                triton_meta={triton_meta!r},
                inductor_meta={inductor_meta!r},
            )
            @triton.jit
        """
    def def_kernel(self, *argnames):
        """
        Hook called from template code to generate function def and
        needed args.
        """
        # 确保所有参数都是字符串类型
        assert all(isinstance(x, str) for x in argnames)
        # 创建一个缩进的代码缓冲区
        renames = IndentedBuffer(initial_indent=1)

        # 获取输入节点中用于生成函数定义和参数的部分
        named_args = self.input_nodes[
            self.prefix_args : len(self.input_nodes) - self.suffix_args
        ]

        # 确保输入节点数量与参数名称数量相等
        assert len(argnames) == len(named_args), (
            len(argnames),
            len(named_args),
            self.prefix_args,
            len(self.input_nodes),
        )

        # 处理前缀参数之前的输入节点
        for input_node in self.input_nodes[: self.prefix_args]:
            # 按正确顺序获取参数
            self.args.input(input_node.get_name())

        # 将参数名称与命名的输入节点关联，并将其存储在字典中
        for name, input_node in zip(argnames, named_args):
            arg_name = f"arg_{name}"
            self.named_input_nodes[name] = input_node
            self.args.input_buffers[input_node.get_name()] = arg_name

        # 参数可能存在重复，因此重命名必须在去重参数之后进行
        for name in argnames:
            input_node = self.named_input_nodes[name]
            arg_name = self.args.input_buffers[input_node.get_name()]
            if input_node.get_layout().offset == 0:
                # 如果偏移为0，则直接赋值参数名称
                renames.writeline(f"{name} = {arg_name}")
            else:
                # 否则，使用偏移量重新赋值参数名称
                offset = texpr(self.rename_indexing(input_node.get_layout().offset))
                renames.writeline(f"{name} = {arg_name} + {offset}")

        # 处理后缀参数之后的输入节点
        for input_node in self.input_nodes[len(self.input_nodes) - self.suffix_args :]:
            # 按正确顺序获取参数
            self.args.input(input_node.get_name())

        def hook():
            """
            Generate the function definition for the kernel function.
            """
            # 等到模板的其余部分懒惰地添加更多参数之后才能运行 python_argdefs()
            arg_defs, *_ = self.args.python_argdefs()
            code = IndentedBuffer()
            # 插入通用的 Triton 导入语句
            code.splice(gen_common_triton_imports())
            # 插入 JIT 编译的代码行
            code.splice(self.jit_lines())
            # 写入生成的函数定义
            code.writeline(f"def {self.kernel_name}({', '.join(arg_defs)}):")
            with code.indent():
                # 插入定义部分的代码缓冲区内容
                code.splice(self.defines)
                code.splice(renames.getvalue())
            return code.getvalue()

        # 确保在渲染钩子中没有重复定义 "<DEF_KERNEL>"
        assert "<DEF_KERNEL>" not in self.render_hooks
        # 将 "<DEF_KERNEL>" 渲染钩子与生成函数定义的 hook 关联起来
        self.render_hooks["<DEF_KERNEL>"] = hook
        # 返回渲染钩子的名称 "<DEF_KERNEL>"
        return "<DEF_KERNEL>"

    def size(self, name: str, index: int):
        """
        Hook called from template code to get the size of an arg.
        Will add needed args to pass it in if it is dynamic.
        """
        # 确保索引是整数类型
        assert isinstance(index, int)
        if name is None:
            # 如果名称为 None，则获取输出节点的指定索引处的大小
            val = self.output_node.get_size()[index]
        else:
            # 否则，获取命名输入节点的指定索引处的大小
            assert isinstance(name, str)
            val = self.named_input_nodes[name].get_size()[index]
        # 返回重新命名后的索引表达式
        return texpr(self.rename_indexing(val))
    def stride(self, name, index):
        """
        Hook called from template code to get the stride of an arg.
        Will add needed args to pass it in if it is dynamic.
        """
        # 断言确保索引是整数类型
        assert isinstance(index, int)
        # 如果名称为None，从输出节点获取指定索引的步长
        if name is None:
            val = self.output_node.get_stride()[index]
        else:
            # 断言确保名称是字符串类型
            assert isinstance(name, str)
            # 根据名称从命名输入节点字典中获取指定索引的步长
            val = self.named_input_nodes[name].get_stride()[index]
        # 对获取的步长值进行索引重命名并返回
        return texpr(self.rename_indexing(val))

    def modification(
        self, subgraph_number: int, output_name: str, **fixed_inputs
    ) -> str:
        """This creates a modification function for a subgraph.
        To use this inside a template, the first argument should specify which subgraph to codegen for

        Args:
            subgraph_number (int): The index of the subgraph in self.subgraphs
        """
        # 初始化计数器
        num = 0
        # 检查生成的修改函数名称是否已存在，确保名称唯一性
        while f"mod_{subgraph_number}_{num}" in self.subgraph_bodies:
            num += 1
        # 使用创建子图主体的上下文管理器
        with self.create_subgraph_body(f"mod_{subgraph_number}_{num}"):
            # 断言确保子图编号是整数类型
            assert isinstance(subgraph_number, int)
            # 断言确保self.subgraphs是列表类型
            assert isinstance(self.subgraphs, list)
            # 断言主体内容为空，确保在添加修改之前主体是清空的
            assert (
                self.body.getvalue() == ""
            ), "Body should be clear before adding a modification"
            # 断言子图编号小于self.subgraphs的长度，确保提供的子图编号有效
            assert subgraph_number < len(
                self.subgraphs
            ), f"Invalid subgraph number provided to create_modification, {subgraph_number} must be < {len(self.subgraphs)}"

            # 获取指定索引的子图
            subgraph = self.subgraphs[subgraph_number]

            # 定义添加输入的函数
            def add_input(name):
                return self.args.input(name)

            # 创建占位符替换类
            name = f"PlaceholderSubstitution_{subgraph_number}"

            class PlaceholderSubstitution(V.WrapperHandler):  # type: ignore[name-defined]
                self.name = name

                # 加载函数，用于加载名称和索引处的表达式
                def load(self, name: str, index: sympy.Expr):
                    if name not in fixed_inputs:
                        # 如果不是固定输入，则从捕获的张量加载
                        var = add_input(name)
                        return f"tl.load({var} + {index})"
                    # 返回固定输入的值
                    return f"({fixed_inputs[name]})"

                # 间接索引函数，用于间接索引操作
                def indirect_indexing(self, index_var, size, check):
                    return sympy_index_symbol(str(index_var))

            # 使用占位符替换设置操作处理器
            with V.set_ops_handler(PlaceholderSubstitution(V.ops)):
                # 断言确保子图是ComputedBuffer类型
                assert isinstance(
                    subgraph, ir.ComputedBuffer
                ), f"Expected the subgraph to be a ComputedBuffer, got {type(subgraph)}"
                # 如果子图数据是InputBuffer类型，则创建加载器
                if isinstance(subgraph.data, ir.InputBuffer):
                    out = subgraph.data.make_loader()((1,))
                else:
                    out = subgraph.data.inner_fn((1,))

            # 生成代码主体
            self.codegen_body()
            # 将结果赋值给指定输出名称
            self.body.writeline(f"{output_name} = {out.value}")

            # 获取生成的代码主体内容
            body_val = self.body.getvalue()
            # 使公共子表达式缓存无效
            self.cse.invalidate(set())
            # 返回生成的代码主体内容
            return body_val
    def store_output(
        self,
        indices: Union[List[Any], Tuple[Any]],
        val: str,
        mask: Optional[str] = None,
        indent_width: int = 4,
    ):
        """
        存储输出的方法，将指定索引的值存储到对象中。
        """
        def render(self, template, kwargs):
            """
            使用模板和关键字参数进行渲染，返回部分渲染对象。
            """
            return PartialRender(
                template.render(**self.template_env(), **kwargs),
                self.render_hooks,
            )

        def make_load(self, name, indices, mask):
            """
            辅助方法，从模板代码中调用以生成加载张量所需的代码。
            """
            assert isinstance(indices, (list, tuple))
            assert isinstance(name, str)
            assert isinstance(mask, str)
            stride = self.named_input_nodes[name].get_stride()
            indices = list(map(TritonPrinter.paren, indices))
            assert len(indices) == len(stride)
            index = " + ".join(
                f"{texpr(self.rename_indexing(s))} * {i}" for s, i in zip(stride, indices)
            )
            return f"tl.load({name} + ({index}), {mask}, other=0.0)"

        def template_env(self):
            """
            生成模板可见的命名空间。
            """
            return {
                fn.__name__: fn
                for fn in [
                    self.def_kernel,
                    self.size,
                    self.stride,
                    self.store_output,
                    self.make_load,
                    self.modification,
                ]
            }

        def indexing(
            self,
            index: sympy.Expr,
            *,
            dense_indexing=False,
            copy_shape=None,
            override_mask=None,
            block_ptr=False,
        ):
            """
            覆盖默认的索引方法，使用自定义掩码和强制密集索引。
            """
            return super().indexing(
                index,
                dense_indexing=False,
                copy_shape=self.template_out,
                override_mask=self.template_mask,
                block_ptr=block_ptr,
            )

        def codegen_range_tree(self):
            """
            生成范围树代码的方法，目前忽略默认的代码生成。
            """
            pass  # ignore default codegen
    def call_kernel(self, name: str, node: Optional[ir.IRNode] = None):
        # 获取当前图形对象的包装器代码
        wrapper = V.graph.wrapper_code
        # 解析参数定义，获取调用参数、参数类型等信息
        _, call_args, _, arg_types = self.args.python_argdefs()
        if V.graph.cpp_wrapper:
            # 在 cpp_wrapper 情况下，需要在运行时计算 CUDA 启动网格
            # 如果涉及到任何动态维度。我们依赖于 Python 版本的 grid 函数来生成这些网格配置，
            # 这些配置可能包含符号值。包装器将使用 cexpr 适当地打印出 C++ 代码以适配这些网格配置。
            grid = self.call_sizes + [self.meta]
            wrapper.generate_kernel_call(
                name,
                call_args,
                grid=self.grid_fn(*grid),
                arg_types=arg_types,
                triton_meta=self.triton_meta,
            )
        else:
            # 添加一次性导入语句，导入 grid_fn 所在模块
            wrapper.add_import_once(f"import {self.grid_fn.__module__}")
            # 添加一次性元数据到包装器中
            meta = wrapper.add_meta_once(self.meta)
            grid = self.call_sizes + [meta]
            wrapper.generate_kernel_call(
                name,
                call_args,
                grid=grid,
                grid_fn=f"{self.grid_fn.__module__}.{self.grid_fn.__name__}",
                arg_types=arg_types,
                triton_meta=self.triton_meta,
            )
# 使用 functools 模块的 lru_cache 装饰器，将 _jinja2_env 函数结果缓存起来，不限制缓存大小
@functools.lru_cache(None)
def _jinja2_env():
    try:
        # 尝试导入 jinja2 库
        import jinja2

        # 返回一个 jinja2 的环境对象，使用 StrictUndefined 严格模式处理未定义变量
        return jinja2.Environment(
            undefined=jinja2.StrictUndefined,
        )
    except ImportError:
        # 如果导入失败则返回 None
        return None


# 定义 TritonTemplate 类，继承自 KernelTemplate 类
class TritonTemplate(KernelTemplate):
    # 类变量，用于为每个实例生成唯一的索引值
    index_counter = itertools.count()
    # 字典类型的类变量，用于存储所有 TritonTemplate 实例，键为模板名称，值为实例对象
    all_templates: Dict[str, "TritonTemplate"] = dict()

    # 初始化方法，接受模板名称 name、网格 grid、模板源码 source 和调试标志 debug
    def __init__(self, name: str, grid: Any, source: str, debug=False):
        # 调用父类 KernelTemplate 的初始化方法，传入模板名称
        super().__init__(name)
        # 将传入的 grid 赋值给实例变量 self.grid
        self.grid = grid
        # 调用 self._template_from_string 方法，根据 source 创建模板对象，并赋值给 self.template
        self.template = self._template_from_string(source)
        # 断言确保名称 name 在 all_templates 字典中不存在，避免重复模板名称
        assert name not in self.all_templates, "duplicate template name"
        # 将当前模板实例添加到 all_templates 字典中，键为模板名称，值为当前实例对象
        self.all_templates[name] = self
        # 设置调试模式标志
        self.debug = debug

    # 生成方法，接受多个参数用于生成内核调用
    def generate(
        self,
        input_nodes,
        layout,
        num_stages,
        num_warps,
        prefix_args=0,
        suffix_args=0,
        epilogue_fn=identity,
        subgraphs=None,
        mutated_inputs=None,
        call_sizes=None,
        **kwargs,
    ):
        # 调用父类的初始化方法，传入名称、输入节点和布局
        super().__init__(name, input_nodes, layout)
        # 设置是否生成内核渲染器的标志
        self.make_kernel_render = make_kernel_render
        # 设置额外调试信息的标志
        self.debug_extra = debug_extra
        # 设置 TritonBenchmarkRequest 对象
        self.bmreq: TritonBenchmarkRequest = bmreq
        # 如果 log_info 为 None，则初始化为空字典
        if log_info is None:
            log_info = {}
        # 设置日志信息字典，并添加 Triton 后端的相关信息
        self.log_info: Dict[str, Any] = log_info
        self.log_info.update(
            {
                "backend": "Triton",
                "grid": str(self.bmreq.grid),
                "num_stages": self.bmreq.num_stages,
                "num_warps": self.bmreq.num_warps,
            }
        )
        # 设置变异后的输入数据
        self.mutated_inputs = mutated_inputs

    def benchmark(self, *args, out):
        # 断言确保 bmreq 不为 None
        assert self.bmreq is not None
        # 调用 TritonBenchmarkRequest 对象的 benchmark 方法
        return self.bmreq.benchmark(*args, output_tensor=out)

    def precompile(self):
        # 断言确保 bmreq 不为 None
        assert self.bmreq is not None
        # 调用 TritonBenchmarkRequest 对象的 precompile 方法
        self.bmreq.precompile()

    def __str__(self):
        # 返回对象的字符串表示，包括模块路径和调试额外信息
        return f"TritonTemplateCaller({self.bmreq.module_path}, {self.debug_extra})"

    def call_name(self):
        # 返回模板内核的调用名称
        return f"template_kernels.{self.name}"

    def hash_key(self):
        # 返回对象的哈希键，由名称的前部分和模块缓存键组成
        return "-".join(
            [
                self.name.rsplit("_", 1)[0],
                self.bmreq.module_cache_key,
            ]
        )

    def output_node(self):
        # 创建 TritonTemplateBuffer 对象，并封装在 TensorBox 中返回
        return ir.TensorBox.create(
            ir.TritonTemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
                debug_extra=self.debug_extra,
                mutated_inputs=self.mutated_inputs,
            )
        )

    def info_dict(self) -> Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        # 返回用于自动调优日志的信息字典
        return self.log_info

    def get_make_kernel_render(self):
        # 返回是否生成内核渲染器的标志
        return self.make_kernel_render
# 定义一个外部内核调用器类，继承自ChoiceCaller类
class ExternKernelCaller(ChoiceCaller):
    
    # 初始化方法
    def __init__(
        self,
        choice: ExternKernelChoice,  # 外部内核选择对象
        input_nodes,  # 输入节点
        layout,  # 布局
        kwargs=None,  # 关键字参数，默认为None
        *,
        has_out_variant=True,  # 是否有输出变体，默认为True
    ):
        # 调用父类ChoiceCaller的初始化方法，传入选择对象的名称、输入节点和布局
        super().__init__(choice.name, input_nodes, layout)
        self.choice = choice  # 设置选择对象
        self.kwargs = kwargs or {}  # 设置关键字参数，若kwargs为None则设为空字典
        self.has_out_variant = has_out_variant  # 设置是否有输出变体标志

    # 返回对象的字符串表示
    def __str__(self):
        return f"ExternKernelCaller({self.choice.call_name()})"

    # 基准测试方法，用于测量性能
    def benchmark(self, *args, out):
        if out.numel() == 0:
            # 如果输出张量大小为0，则不需要运行内核，直接返回0.0
            return 0.0
        if self.has_out_variant:
            # 如果有输出变体，则调用父类ChoiceCaller的benchmark方法，传入参数和输出张量
            return super().benchmark(*args, out=out)
        else:
            # 否则获取可调用对象algo，并用其计算结果替换输出张量，用于正确性检查
            algo = self.to_callable()
            out_new = algo(*args)
            # 使用torch._C._dynamo.guards.assert_size_stride检查张量大小和步幅是否正确
            torch._C._dynamo.guards.assert_size_stride(
                out_new, tuple(out.size()), tuple(out.stride())
            )
            out.copy_(out_new)  # 将计算结果复制到输出张量，用于正确性检查
            return do_bench(algo, args, {})  # 执行benchmark操作，测量性能

    # 将选择对象转换为可调用对象
    def to_callable(self):
        fn = self.choice.to_callable()  # 获取选择对象的可调用版本
        if self.kwargs:
            return functools.partial(fn, **self.kwargs)  # 如果有关键字参数，则使用偏函数返回
        else:
            return fn  # 否则直接返回可调用函数

    # 生成对象的哈希键
    def hash_key(self):
        return "-".join(
            [
                self.choice.name,
                *[
                    f"{kwarg}={repr(self.kwargs[kwarg])}"
                    for kwarg in sorted(self.kwargs.keys())
                ],
                self.choice.hash_key(),  # 将选择对象的哈希键添加到列表中
            ]
        )

    # 返回输出节点
    def output_node(self):
        if config.abi_compatible and self.choice.use_fallback_kernel:
            assert (
                self.choice.op_overload is not None
            ), "Please provide an op_overload to use ir.FallbackKernel"
            # 如果ABI兼容且使用回退内核，则创建一个FallbackKernel对象
            inner = ir.FallbackKernel.create(
                self.choice.op_overload, *self.input_nodes, **self.kwargs
            )
        elif self.choice.kernel_creator is not None:
            # 否则如果存在内核创建器，则调用它创建内核对象
            inner = self.choice.kernel_creator(*self.input_nodes, **self.kwargs)
        else:
            # 否则根据是否有输出变体选择创建ExternKernelOut或ExternKernelAlloc对象
            cls = ir.ExternKernelOut if self.has_out_variant else ir.ExternKernelAlloc
            inner = cls(
                layout=self.layout,
                inputs=self.input_nodes,
                python_kernel_name=self.choice.call_name(),
                cpp_kernel_name=self.choice.cpp_kernel_name,
                ordered_kwargs_for_cpp_kernel=self.choice.ordered_kwargs_for_cpp_kernel,
                op_overload=self.choice.op_overload,
                kwargs=self.kwargs,
            )

        return ir.TensorBox.create(inner)  # 创建并返回TensorBox对象

    # 返回信息字典，用于记录到自动调优日志文件中
    def info_dict(self) -> Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        return {
            "backend": "extern",  # 后端类型为extern
            "kernel_call_name": self.choice.call_name(),  # 内核调用名称
        }


# 使用functools.lru_cache(None)装饰器定义一个获取mm日志文件名的函数，返回可选的字符串类型文件名
@functools.lru_cache(None)
def get_mm_log_filename() -> Optional[str]:
    # 从环境变量中获取 TORCHINDUCTOR_MM_LOGGING_FILE 的值作为日志文件名，如果不存在则返回 None
    mm_file_name = os.environ.get("TORCHINDUCTOR_MM_LOGGING_FILE", None)
    # 如果 mm_file_name 为空，则直接返回 None，表示没有找到日志文件名
    if not mm_file_name:
        return None
    
    # 如果 mm_file_name 不包含 "json" 字符串，则将其加上 ".json" 后缀
    if "json" not in mm_file_name:
        mm_file_name = f"{mm_file_name}.json"
    
    # 返回处理后的 mm_file_name，可能加上了 ".json" 后缀
    return mm_file_name
def append_to_log(filename, data):
    # 根据日志文件名生成对应的锁文件名
    lock_file = filename.replace(".json", ".lock")
    # 创建文件锁对象
    lock = FileLock(lock_file)
    # 使用文件锁保护以下代码块，确保并发安全
    with lock:
        try:
            # 尝试打开日志文件并加载其中的 JSON 数据
            with open(filename) as f:
                log_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或者解析 JSON 失败，则初始化空日志数据
            log_data = []

        # 将新数据追加到日志数据中
        log_data.append(data)

        # 将更新后的日志数据写回到文件中，格式化输出，保持缩进
        with open(filename, "w") as f:
            json.dump(log_data, f, indent=4)


class DataProcessorChoiceCallerWrapper:
    def __init__(self, wrapped, preprocessor, postprocessor):
        # 初始化包装对象和预处理、后处理函数
        self._wrapped = wrapped
        # 如果预处理函数未提供，则默认为将参数原样返回
        if preprocessor is not None:
            self._preprocessor = preprocessor
        else:
            self._preprocessor = lambda x, y: (x, y)
        # 如果后处理函数未提供，则默认为将结果值原样返回
        if postprocessor is not None:
            self._postprocessor = postprocessor
        else:
            self._postprocessor = lambda x: x

    def __getattr__(self, name):
        # 委托未知属性调用给被包装对象
        return getattr(self._wrapped, name)

    def benchmark(self, *args, out) -> float:
        # 调用前处理函数处理参数和输出
        new_args, new_out = self._preprocessor(args, out)
        # 调用被包装对象的 benchmark 方法，并获取结果
        result = self._wrapped.benchmark(*new_args, out=new_out)
        # 调用后处理函数处理输出结果
        new_out = self._postprocessor(new_out)
        # 如果原始输出和新输出不同，则将新输出复制到原始输出中
        if out is not new_out:
            out.copy_(new_out)
        # 返回 benchmark 方法的结果
        return result

    def output_node(self) -> ir.TensorBox:
        # 调用被包装对象的 output_node 方法，并对结果应用后处理函数
        result = self._wrapped.output_node()
        return self._postprocessor(result)

    def __repr__(self) -> str:
        # 返回对象的字符串表示，指定包装对象的信息
        return f"DataProcessorChoiceCallerWrapper({self._wrapped})"


class DataProcessorTemplateWrapper:
    """
    A wrapper class for a kernel template.

    This class together with `DataProcessorChoiceCallerWrapper` provides a convenient way to
    preprocess and postprocess data before and after using the wrapped template. A typical
    usage is to reorder or filter the input nodes in order to match the expected input of other
    kernel choices like a ATen kernel. A more complicated usage is to prepack the weights.
    See the example from :mod:`cpp_gemm_template` for more details.
    """

    def __init__(
        self,
        wrapped_template_cls,
        preprocessor,
        postprocessor,
        **kwargs,
    ):
        # 如果预处理函数未提供，则默认为将参数原样返回
        if preprocessor is not None:
            self._preprocessor = preprocessor
        else:
            self._preprocessor = lambda x, y: (x, y)
        # 如果后处理函数未提供，则默认为将结果值原样返回
        if postprocessor is not None:
            self._postprocessor = postprocessor
        else:
            self._postprocessor = lambda x: x
        # 确保输入节点和布局参数在初始化之前经过预处理
        assert "input_nodes" in kwargs
        assert "layout" in kwargs
        kwargs["input_nodes"], kwargs["layout"] = preprocessor(
            kwargs["input_nodes"], kwargs["layout"]
        )
        # 使用经过预处理的参数初始化被包装的模板类对象
        self._wrapped = wrapped_template_cls(**kwargs)

    def __getattr__(self, name):
        # 委托未知属性调用给被包装对象
        return getattr(self._wrapped, name)

    def maybe_append_choice(self, choices, **kwargs):
        # 调用被包装对象的 maybe_append_choice 方法，并传递自身对象和其他参数
        return type(self._wrapped).maybe_append_choice(self, choices, **kwargs)
    # 定义一个方法 `generate`，接受任意关键字参数 `kwargs`
    def generate(self, **kwargs):
        # 调用 `self._wrapped` 对象的 `generate` 方法，传入关键字参数 `kwargs`，获取返回值
        choice_caller = self._wrapped.generate(**kwargs)
        # 返回一个 `DataProcessorChoiceCallerWrapper` 对象，包装 `choice_caller`、预处理器和后处理器
        return DataProcessorChoiceCallerWrapper(
            choice_caller, self._preprocessor, self._postprocessor
        )

    # 定义一个特殊方法 `__repr__`，返回描述对象的字符串表示
    def __repr__(self) -> str:
        # 返回包含 `_wrapped` 属性字符串表示的 `DataProcessorTemplateWrapper` 对象的字符串表示
        return f"DataProcessorTemplateWrapper({self._wrapped})"
class ErrorFromChoice(RuntimeError):
    # 自定义异常类，用于表示选择时发生的错误
    def __init__(self, msg, choice: ChoiceCaller, inputs_str):
        # 在异常消息中包含选择器和输入信息
        msg += f"\nFrom choice {choice}\n{inputs_str}"
        super().__init__(msg)
        self.choice = choice


class NoValidChoicesError(RuntimeError):
    # 自定义异常类，用于表示没有有效选择的错误
    pass


@functools.lru_cache(None)
def get_env_num_workers() -> Optional[int]:
    # 获取环境变量中的线程数配置，如果存在则返回整数值，否则返回None
    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        return int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"])
    return None


class AlgorithmSelectorCache(PersistentCache):
    def __init__(self, *args, **kwargs):
        # 继承自持久化缓存类，初始化方法
        super().__init__(*args, **kwargs)

        # 算法选择器缓存类，存储预编译函数的缓存字典
        # 用于存储特定键的所有降低操作的共享预编译函数
        self.precompile_cache: Dict[str, Callable[[], None]] = {}

    def __call__(
        self,
        name,
        choices: List[ChoiceCaller],
        input_nodes,
        layout,
        # 可选参数：字典，将参数索引映射到生成 torch.Tensor 的函数
        # 如果传递给定的参数，则用函数生成 torch.Tensor 而不是为基准测试生成随机 torch.Tensor
        input_gen_fns: Optional[Dict[int, Callable[[ir.Buffer], torch.Tensor]]] = None,
        precompilation_timeout_seconds: int = 60 * 60,
        return_multi_template=False,
    ):
        # 算法选择器缓存类的调用方法，用于选择算法并进行预编译
        ...

    @classmethod
    def make_benchmark_fn(
        cls,
        choices,
        input_nodes,
        layout,
        input_gen_fns=None,
    ):
        # 类方法：创建基准测试函数，用于评估不同选择的性能
        ...

    @staticmethod
    def log_results(
        name: str,
        input_nodes: List[ir.IRNode],
        timings: Dict[ChoiceCaller, float],
        elapse: float,
        precompile_elapse: float,
    ):
        # 静态方法：记录基准测试结果，包括名称、输入节点、时间等信息
        ...
    # 当函数被调用时，记录自动调优的结果到调试日志中
    def log_autotuning_results(
        name, input_nodes, timings, elapse, precompile_elapse
    ):
        # 调试日志记录自动调优的结果，包括名称、输入节点、时间信息、预编译时间
        V.debug.log_autotuning_results(
            name, input_nodes, timings, elapse, precompile_elapse
        )
        # 如果没有设置最大自动调优或最大自动调优GEMM，或者不打印自动调优信息，则直接返回
        if not (config.max_autotune or config.max_autotune_gemm) or not PRINT_AUTOTUNE:
            return
        # 生成输入节点的尺寸信息字符串，用于日志记录
        sizes = ", ".join(
            [
                "x".join(
                    map(
                        str,
                        V.graph.sizevars.size_hints(
                            n.get_size(), fallback=config.unbacked_symint_fallback
                        ),
                    )
                )
                for n in input_nodes
            ]
        )
    
        # 如果日志的有效级别是DEBUG，则只显示前10个最佳结果，否则不限制显示数量
        n = None if log.getEffectiveLevel() == logging.DEBUG else 10
        top_k = sorted(timings, key=timings.__getitem__)[:n]
        best = top_k[0]
    
        # 定义函数，根据选择获取调优信息
        def get_choice_info(choice):
            # 如果选择是ExternKernelCaller类型，则返回相关信息
            if isinstance(choice, torch._inductor.select_algorithm.ExternKernelCaller):
                return {"type": "cublas", "time": timings[choice]}
    
            # 否则，选择应该是TritonTemplateCaller类型，返回其详细信息
            assert isinstance(
                choice, torch._inductor.select_algorithm.TritonTemplateCaller
            )
    
            # 获取模板信息中的块形状数据，并解析为具体数值
            info = choice.info_dict()
            tile = info["tile_shape"]
            tile_vals = eval(tile)  # type: ignore[arg-type]
            BLOCK_M = tile_vals[0]
            BLOCK_K = tile_vals[1]
            BLOCK_N = tile_vals[2]
    
            return {
                "type": "triton",
                "time": timings[choice],
                "BLOCK_M": BLOCK_M,
                "BLOCK_K": BLOCK_K,
                "BLOCK_N": BLOCK_N,
                "num_stages": info["num_stages"],
                "num_warps": info["num_warps"],
            }
    
        # 获取保存矩阵乘法调优日志的文件名，并且如果存在的话，将相关信息追加到日志中
        mm_filename = get_mm_log_filename()
        if mm_filename and "mm" in name:
            M, K = input_nodes[-2].get_size()[:2]
            N = input_nodes[-1].get_size()[-1]
    
            out_dict = {
                str((M, K, N)): [get_choice_info(choice) for choice in timings.keys()]
            }
    
            append_to_log(mm_filename, out_dict)
    
        # 获取最佳执行时间，并输出调优结果信息到标准错误流
        best_time = timings[best]
        sys.stderr.write(f"AUTOTUNE {name}({sizes})\n")
        for choice in top_k:
            result = timings[choice]
            if result:
                kernel_info = (
                    choice.debug_extra if hasattr(choice, "debug_extra") else ""
                )
                sys.stderr.write(
                    f"  {choice.name} {result:.4f} ms {best_time / result:.1%} {kernel_info}\n"
                )
            else:
                sys.stderr.write(
                    f"  {choice.name} {result:.4f} ms <DIVIDED BY ZERO ERROR>\n"
                )
    
        # 输出自动调优类型和相应的时间信息到标准错误流
        autotune_type_str = (
            "SubProcess" if config.autotune_in_subproc else "SingleProcess"
        )
        sys.stderr.write(
            f"{autotune_type_str} AUTOTUNE benchmarking takes {elapse:.4f} seconds and {precompile_elapse:.4f}"
            " seconds precompiling\n"
        )
    # 定义一个静态方法，用于将 ir.Buffer 转换为可以用于基准测试的具体 torch.Tensor
    def benchmark_example_value(node):
        """
        Convert an ir.Buffer into a concrete torch.Tensor we can use for
        benchmarking.
        """
        # 如果传入的 node 是 ir.Layout 类型，则创建一个名为 "fake" 的 ir.Buffer 对象
        if isinstance(node, ir.Layout):
            node = ir.Buffer("fake", node)
        # 如果传入的 node 是 ir.BaseView 类型，则获取其基础 tensor
        # triton templates 要求使用基础 tensor。
        if isinstance(node, ir.BaseView):
            node = node.unwrap_view()
        # 调用 AlgorithmSelectorCache 类的 generate_example_value 方法，
        # 生成基准测试所需的例子值
        return AlgorithmSelectorCache.generate_example_value(
            V.graph.sizevars.size_hints(
                node.get_size(),
                fallback=config.unbacked_symint_fallback,
            ),
            V.graph.sizevars.size_hints(
                node.get_stride(),
                fallback=config.unbacked_symint_fallback,
            ),
            node.get_device(),
            node.get_dtype(),
            node.layout.offset,
        )

    @staticmethod
    # 定义一个静态方法，生成基准测试所需的例子值
    def generate_example_value(size, stride, device, dtype, extra_size):
        # 保存随机数生成器状态，以避免 rand_strided 调用修改真实模型代码的随机数生成器状态。
        with preserve_rng_state():
            # 调用 rand_strided 方法生成指定参数的随机张量
            return rand_strided(
                size,
                stride,
                device=device,
                dtype=dtype,
                extra_size=extra_size,
            )

    @staticmethod
    # 定义一个静态方法，返回用于在缓存中无效化自动调优结果的 ir.Buffer 片段
    def key_of(node):
        """
        Extract the pieces of an ir.Buffer that we should invalidate cached
        autotuning results on.
        """
        # 获取 V.graph.sizevars 对象
        sizevars = V.graph.sizevars
        # 返回一个元组，包含 node 的设备类型、数据类型字符串表示、大小提示和步长提示，
        # 以及布局偏移量的大小提示
        return (
            node.get_device().type,
            str(node.get_dtype()),
            *sizevars.size_hints(
                node.get_size(),
                fallback=config.unbacked_symint_fallback,
            ),
            *sizevars.size_hints(
                node.get_stride(),
                fallback=config.unbacked_symint_fallback,
            ),
            sizevars.size_hint(
                node.get_layout().offset,
                fallback=config.unbacked_symint_fallback,
            ),
        )
# 全局变量，用于缓存算法选择器对象，类型为 AlgorithmSelectorCache 或 None
_ALGORITHM_SELECTOR_CACHE: Optional[AlgorithmSelectorCache] = None

# 自动调优选择算法的函数，根据参数选择最优算法
def autotune_select_algorithm(*args, **kwargs):
    global _ALGORITHM_SELECTOR_CACHE
    # 如果缓存为空，则实例化一个 AlgorithmSelectorCache 对象并赋给缓存
    if _ALGORITHM_SELECTOR_CACHE is None:
        _ALGORITHM_SELECTOR_CACHE = AlgorithmSelectorCache()

    # 如果参数中未指定 "return_multi_template"，则使用 torch._inductor.config.benchmark_epilogue_fusion 的值
    if "return_multi_template" not in kwargs:
        kwargs[
            "return_multi_template"
        ] = torch._inductor.config.benchmark_epilogue_fusion

    # 调用缓存中的算法选择器对象，传入参数和关键字参数，返回选择的算法结果
    return _ALGORITHM_SELECTOR_CACHE(*args, **kwargs)


# 实现输入数据的函数，接受一个或多个参数，并确保它们被转换为特定格式
def realize_inputs(*args):
    # 如果参数个数为1，调用 extern_kernels.ir.ExternKernel.realize_input 函数处理并返回
    if len(args) == 1:
        return ir.ExternKernel.require_stride1(ir.ExternKernel.realize_input(args[0]))
    # 如果参数个数大于1，对每个参数递归调用 realize_inputs 函数，并返回结果列表
    return [realize_inputs(x) for x in args]


# 确保 lowering 模块被导入，以便 `extern_kernels.*` 被正确填充
from . import lowering  # noqa: F401
```