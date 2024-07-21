# `.\pytorch\torch\_inductor\codegen\cpp_template_kernel.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和类型定义
import itertools  # 导入 itertools 模块，用于迭代器操作
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型定义

import sympy  # 导入 sympy 模块，用于符号计算
from sympy.parsing.sympy_parser import parse_expr  # 从 sympy.parsing.sympy_parser 导入 parse_expr 函数

import torch  # 导入 torch 模块，用于深度学习
from torch.utils._sympy.symbol import SymT  # 导入 SymT 类型
from .. import config, cpp_builder, ir, lowering as L  # 导入本地模块

from ..autotune_process import CppBenchmarkRequest  # 导入 CppBenchmarkRequest 类
from ..select_algorithm import PartialRender  # 导入 PartialRender 类
from ..utils import sympy_index_symbol, sympy_index_symbol_with_prefix  # 导入工具函数
from ..virtualized import V  # 导入 V 类
from .cpp import CppKernel, CppKernelProxy, KernelGroup  # 导入本地 CppKernel 相关类
from .cpp_utils import cexpr_index, DTYPE_TO_CPP, LocalBufferScope  # 导入本地工具函数和常量

def parse_expr_with_index_symbols(expr):
    # 如果 expr 已经是 sympy.Expr 类型，则直接返回
    if isinstance(expr, sympy.Expr):
        return expr
    # 如果 expr 是列表或元组，则递归调用 parse_expr_with_index_symbols 处理每个元素
    elif isinstance(expr, (list, tuple)):
        return [parse_expr_with_index_symbols(e) for e in expr]
    # 否则，将 expr 转换为字符串并解析成 sympy.Expr 类型的表达式
    else:
        expr = parse_expr(str(expr))
        # 为表达式中的自由符号创建索引符号，返回替换后的表达式
        int_symbols = {sym: sympy_index_symbol(sym.name) for sym in expr.free_symbols}
        return expr.subs(int_symbols)

def wrap_with_tensorbox(node) -> ir.TensorBox:
    # 如果 node 是 ir.Buffer 类型，则使用 ir.TensorBox.create 方法创建 TensorBox 对象
    if isinstance(node, ir.Buffer):
        return ir.TensorBox.create(node)
    # 否则，直接使用 ir.TensorBox 构造函数创建 TensorBox 对象
    else:
        return ir.TensorBox(node)

class CppTemplateKernel(CppKernel):
    def __init__(self, kernel_name, num_threads):
        # 调用父类 CppKernel 的初始化方法，设置 kernel_name 和 num_threads
        super().__init__(None, num_threads)
        self.kernel_name = kernel_name  # 设置内核名称
        self.render_hooks = {}  # 初始化渲染钩子字典
        self.local_buffers = {}  # 初始化本地缓冲区字典

    def render(self, template, **kwargs):
        # 使用 PartialRender 类对模板进行渲染，传入 kernel 和其他关键字参数
        return PartialRender(
            template.render(kernel=self, **kwargs), self.render_hooks
        ).finalize_all()

    def def_kernel(
        self,
        inputs: Dict[str, ir.Buffer],
        outputs: Dict[str, ir.Buffer],
        aliases: Optional[List[Tuple[ir.Buffer, ir.Buffer]]] = None,
        # 定义内核函数，接受输入和输出缓冲区的字典，以及可选的别名列表
    ) -> str:
        # 遍历输入项字典，将非空输入项名称映射到其对应的参数名称
        for name, inp in inputs.items():
            if inp is not None:
                self.args.input_buffers[inp.get_name()] = name
        # 遍历输出项字典，将输出项名称映射到其对应的参数名称
        for name, out in outputs.items():
            self.args.output_buffers[out.get_name()] = name
        # 如果存在别名列表，则处理每对别名，将原始名称映射到别名名称
        if aliases is not None:
            for alias, orig in aliases:
                orig_name = orig.get_name()
                alias_name = alias.get_name()
                # 如果原始名称在输入缓冲区中，将别名映射到相同的输入缓冲区位置
                if orig_name in self.args.input_buffers:
                    self.args.input_buffers[alias_name] = self.args.input_buffers[
                        orig_name
                    ]
                # 如果原始名称在输出缓冲区中，将别名映射到相同的输出缓冲区位置
                if orig_name in self.args.output_buffers:
                    self.args.output_buffers[alias_name] = self.args.output_buffers[
                        orig_name
                    ]

        # 收集所有输入项和输出项中的唯一大小变量
        unique_sizevars = {
            s
            for input in inputs.values()
            if input is not None
            for sym in itertools.chain(input.get_size(), input.get_stride())
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        }
        # 合并所有输出项中的唯一大小变量到已收集的集合中
        unique_sizevars |= {
            s
            for output in outputs.values()
            for sym in itertools.chain(output.get_size(), output.get_stride())
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        }
        # 将唯一大小变量按其字符串形式排序，并存储在self.args.sizevars中
        sizevars = sorted(unique_sizevars, key=str)
        for sizevar in sizevars:
            self.args.sizevars[sizevar] = f"k{sizevar}"

        def hook():
            # 在生成函数定义之前，删除所有别名映射
            if aliases is not None:
                for alias, _ in aliases:
                    alias_name = alias.get_name()
                    # 将别名在输入缓冲区中的映射标记为"REMOVED"
                    if alias_name in self.args.input_buffers:
                        self.args.input_buffers[alias_name] = "REMOVED"
                    # 将别名在输出缓冲区中的映射标记为"REMOVED"
                    if alias_name in self.args.output_buffers:
                        self.args.output_buffers[alias_name] = "REMOVED"
            # 获取CPP参数定义，用于生成函数签名
            cpp_argdefs, _, _ = self.args.cpp_argdefs()
            return f"void {self.kernel_name}({', '.join(cpp_argdefs)})"

        # 定义一个占位符字符串用于标记生成的内核函数
        placeholder = "<DEF_KERNEL>"
        # 确保占位符不在self.render_hooks中
        assert placeholder not in self.render_hooks
        # 将占位符与hook函数关联存储在self.render_hooks中
        self.render_hooks[placeholder] = hook
        # 返回占位符字符串
        return placeholder

    def call_kernel(self, name: str, node: ir.CppTemplateBuffer):
        # 获取图形包装器代码
        wrapper = V.graph.wrapper_code
        # 获取CPP参数定义、调用参数、参数类型，并调用包装器生成内核调用代码
        _, call_args, arg_types = self.args.cpp_argdefs()
        wrapper.generate_kernel_call(name, call_args, cuda=False, arg_types=arg_types)

    def dtype(self, node: ir.Buffer) -> str:
        # 返回节点的数据类型对应的CPP类型定义
        return DTYPE_TO_CPP[node.get_dtype()]

    def acc_dtype(self, node: ir.Buffer) -> str:
        # 根据节点的数据类型返回对应的加速数据类型定义，如果不支持则引发异常
        if node.get_dtype() in [torch.float32, torch.bfloat16, torch.half]:
            return "float"
        else:
            raise NotImplementedError(f"Unsupported dtype: {node.get_dtype()}")

    def size(self, node: ir.Buffer, dim: int) -> str:
        # 返回节点在指定维度上的大小，经过重命名后的索引表达式
        return cexpr_index(self.rename_indexing(node.get_size()[dim]))
    def stride(self, node: ir.Buffer, dim: int) -> str:
        # 获取节点在指定维度上的步长，并将其转换为对应的 C 表达式
        return cexpr_index(self.rename_indexing(node.get_stride()[dim]))

    def index(self, node: ir.Buffer, indices: List[Any]) -> str:
        # 获取节点的布局作为固定布局，并创建索引器
        indexer = node.layout.as_fixed().make_indexer()
        # 解析带有索引符号的表达式，并使用索引器进行索引
        index = indexer(parse_expr_with_index_symbols(indices))
        # 重命名索引，并获取节点的外部名称
        index = self.rename_indexing(index)
        outer_name = node.get_name()
        # 根据节点的外部名称选择内部名称，如果外部名称在本地缓冲中则直接使用，否则调用输入参数处理方法
        inner_name = (
            outer_name
            if outer_name in self.local_buffers
            else self.args.input(node.get_name())
        )
        # 返回格式化后的索引表达式
        return f"{inner_name}[{cexpr_index(index)}]"

    def slice_nd(self, node, ranges: List[Tuple[Any, Any]]) -> ir.ReinterpretView:
        """
        使用给定的范围列表（起始和结束）对节点进行切片，对应其维度。
        如果对应范围为空，则不对该维度进行切片。
        """
        assert len(ranges) == len(node.get_size())
        # 使用 TensorBox 封装节点
        sliced = wrap_with_tensorbox(node)
        for dim, _range in enumerate(ranges):
            if len(_range) == 0:
                continue
            assert len(_range) == 2
            # 解析带有索引符号的起始和结束表达式，并在不限制范围的情况下对节点进行切片
            start, end = parse_expr_with_index_symbols(_range)
            sliced = L.slice_(sliced, dim, start, end, clamp=False)
        assert isinstance(sliced.data, ir.ReinterpretView), sliced.data
        return sliced.data

    def view(self, node, sizes: List[Any]) -> ir.View:
        # 使用 TensorBox 封装节点，并解析带有索引符号的尺寸表达式
        node = wrap_with_tensorbox(node)
        sizes = parse_expr_with_index_symbols(sizes)
        # 返回视图数据
        return L.view(node, sizes).data

    def permute(self, node, dims):
        # 使用 TensorBox 封装节点，并对节点进行指定维度的排列
        node = wrap_with_tensorbox(node)
        permuted = L.permute(node, dims).data
        assert isinstance(permuted, ir.ReinterpretView)
        # 返回排列后的数据
        return permuted

    def maybe_codegen_profile(self) -> str:
        # 如果配置允许生成内核性能分析代码，则生成记录内核函数调用的 C++ 代码
        if config.cpp.enable_kernel_profile:
            graph_id = V.graph.graph_id
            prefix = "graph_" + str(graph_id) + "_" if graph_id is not None else ""
            return f'RECORD_FUNCTION("{prefix}{self.kernel_name}", c10::ArrayRef<c10::IValue>({{}}));'
        else:
            return ""

    def unroll_pragma(self, unroll):
        # 如果当前使用的是 GCC 编译器，则生成 GCC 风格的展开指令，否则生成通用的展开指令
        if cpp_builder.is_gcc():
            return f"#pragma GCC unroll {unroll}"
        else:
            return f"#pragma unroll {unroll}"

    def define_buffer(self, name, sizes: List[Any], dtype=torch.float) -> str:
        """定义内核本地缓冲区"""
        # 解析带有索引符号的尺寸表达式，并根据名称、尺寸和数据类型创建缓冲区对象
        sizes = parse_expr_with_index_symbols(sizes)
        buf = ir.Buffer(name, ir.FixedLayout(torch.device("cpu"), dtype, sizes))
        self.local_buffers[name] = buf
        # 生成对应的 C++ 代码，定义本地缓冲区，并返回定义语句
        ctype = f"{DTYPE_TO_CPP[dtype]}"
        numel = f"{cexpr_index(buf.get_numel())}"
        return f"auto _{name} = std::make_unique<{ctype}[]>({numel}); auto {name} = _{name}.get();"

    def store_pointwise_nodes(
        self,
        dst: ir.Buffer,
        nodes: List[ir.IRNode],
        offsets: Optional[List[sympy.Expr]] = None,
        reindexers: Optional[List[Optional[Callable[[List[Any]], List[Any]]]]] = None,
    ):
        # 略
        pass
    ) -> str:
        # 定义变量 var_sizes 为目标缓冲区的大小元组及空元组
        var_sizes = (tuple(dst.get_size()), ())
        # 使用 enumerate() 函数遍历 var_sizes[0] 中的每个元素，并为其创建带有前缀的 sympy 索引符号
        var_ranges = {
            sympy_index_symbol_with_prefix(SymT.INDEX, i): sz
            for i, sz in enumerate(var_sizes[0])
        }
        # 如果 offsets 为空列表，则将其设为与 var_sizes[0] 同长的零列表
        if not offsets:
            offsets = [sympy.Integer(0)] * len(var_sizes[0])
        # 如果 reindexers 为空列表，则将其设为与 nodes 列表同长的 None 列表
        if not reindexers:
            reindexers = [None] * len(nodes)
        # 断言 offsets 的长度与 var_sizes[0] 相等
        assert len(offsets) == len(var_sizes[0])
        # 获取目标缓冲区的布局，并生成输出索引
        output_index = dst.get_layout().make_indexer()(var_ranges.keys())
        # 创建 KernelGroup 对象
        kernel_group = KernelGroup()
        # 设置 kernel_group 的参数为 self.args
        kernel_group.args = self.args
        # 创建 CppKernelProxy 对象并将 kernel_group 作为参数传入
        cpp_kernel_proxy = CppKernelProxy(kernel_group)
        # 初始化空列表 bodies
        bodies = []
        # 初始化 var_sizes_list 为 var_sizes 的复制列表
        var_sizes_list = []
        # 遍历 nodes 列表的每个元素及其索引值 i
        for i, node in enumerate(nodes):
            # 如果 i 小于 nodes 列表长度减 1，则 output_name 为 node 的名称；否则为 dst 的名称
            output_name = node.get_name() if i < len(nodes) - 1 else dst.get_name()
            # 如果 node 是 ir.ComputedBuffer 实例，则将其转换为其 data 属性
            node = node.data if isinstance(node, ir.ComputedBuffer) else node
            # 断言 node 是 ir.Pointwise 类型
            assert isinstance(node, ir.Pointwise), node

            # 定义函数 fn，接受任意数量的参数
            def fn(*args):
                # 断言参数 args 的长度为 2
                assert len(args) == 2
                # 断言 args[0] 的长度与 var_sizes[0] 相等
                assert len(args[0]) == len(var_sizes[0])
                # 断言 args[1] 的长度为 0
                assert len(args[1]) == 0
                # 将每个 args[0] 中的元素与 offsets 中的对应元素相加，生成 new_args
                new_args = [arg + offset for arg, offset in zip(args[0], offsets)]  # type: ignore[arg-type]
                # 如果 reindexers[i] 不为 None，则调用 reindexers[i] 处理 new_args
                if reindexers[i] is not None:
                    new_args = reindexers[i](new_args)  # type: ignore[misc]
                # 将 node 根据 new_args 的值加载，并将其值存储到 output_name 处
                V.ops.store(
                    output_name,
                    output_index,
                    node.make_loader()(new_args).value,
                )

            # 创建 LoopBody 对象 body，将 fn 函数及其参数传入
            body = ir.LoopBody(fn, (list(var_ranges.keys()), ()), var_ranges)
            # 将 body 添加到 bodies 列表中
            bodies.append(body)
            # 将 var_sizes 添加到 var_sizes_list 中
            var_sizes_list.append(var_sizes)

        # 调用 cpp_kernel_proxy 的 codegen_loop_bodies 方法，传入 bodies 和 var_sizes_list 参数
        cpp_kernel_proxy.codegen_loop_bodies(bodies, var_sizes_list)
        # 调用 kernel_group 的 finalize_kernel 方法，传入 cpp_kernel_proxy 和空列表
        kernel_group.finalize_kernel(cpp_kernel_proxy, [])
        # 返回 kernel_group 的循环代码的值
        return kernel_group.loops_code.getvalue()
    ):
        """
        Store the `src` buffer to the `dst` buffer. The size of `src` and `dst` should match.
        If `epilogue_nodes` is provided, the `src` buffer is firstly computed with the epilogues
        before stored to `dst`. The `epilogues_nodes` are all pointwise.

        Notes:
        1. `src` and `dst` buffer could be the same buffer in which case we are doing in-place compute
           and stores. In case `epilogue_nodes` are not provided, we do nothing.
        2. The `epilogue_nodes`, if exist, have computations on `src` before storing to `dst` but since
           they come form the original Inductor IR, they might need to be adjusted before working with
           `src` and `dst` as outlined below:
           a) `src` or `dst` buffer could be a sub-slice of the ranges the `epilogue_nodes`work on.
              In this case, the `offsets` could be provided to adjust the indices passed to
              `epilogue_nodes` during codegen and the data ranges are also configured according to
              the sizes of `src` and `dst`.
           b) `dst` might be indexed in a different way as the `epilogue_nodes`, hence a `reindexer` is
              needed on the indices to `epilogue_nodes` to match the indexing of `dst`.
           c) If `src` is local, we need to add a local buffer for it and localize the `orig_src` buffer
              in `epilogue_nodes` with `src`.
        """
        # 确保 `src` 和 `dst` 的大小匹配
        assert dst.get_size() == src.get_size()
        # 如果提供了 offsets，将其解析为表达式并使用带索引符号的表达式解析器进行处理
        if offsets:
            offsets = parse_expr_with_index_symbols(offsets)
        # 如果提供了 epilogue_nodes
        if epilogue_nodes:
            # 在本地缓冲作用域内处理局部缓冲
            with LocalBufferScope(self) as scope:
                # 确保 orig_src 不为空
                assert orig_src is not None
                # 如果原始的 src 名称与当前 src 名称不同，添加局部缓冲
                if orig_src.get_name() != src.get_name():
                    scope.add_local_buffer(src)
                    # 通过局部缓冲范围化 epilogue_nodes
                    epilogue_nodes = scope.localize_buffer(
                        orig_src, src, epilogue_nodes
                    )
                # 调用存储点对点节点的方法，传入 dst、epilogue_nodes、offsets 和 reindexers
                return self.store_pointwise_nodes(
                    dst, epilogue_nodes, offsets, reindexers  # type: ignore[arg-type]
                )
        else:
            # 如果 dst 的名称与 src 的名称不同
            if dst.get_name() != src.get_name():
                # 如果 src 是本地的，进行复制操作，并在本地缓冲作用域内添加局部缓冲
                copy = L.copy(dst, src).data.data
                with LocalBufferScope(self) as scope:
                    scope.add_local_buffer(src)
                    # 调用存储点对点节点的方法，传入 dst 和复制的数据
                    return self.store_pointwise_nodes(dst, [copy])
            else:
                # 确保 dst 的布局与 src 的布局相同
                assert dst.layout == src.layout
                # 如果以上条件都不符合，返回空字符串
                return ""
# 定义一个名为 CppTemplateCaller 的类，它是 ir.ChoiceCaller 的子类，用于调用 CPP 模板内核。
class CppTemplateCaller(ir.ChoiceCaller):
    """
    CppTemplateCaller

    This class represents a caller for CPP template kernels. It is a subclass of ir.ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (CppBenchmarkRequest): The benchmark request for the caller.
        template_buffer (ir.CppTemplateBuffer): The template buffer for the caller.
    """

    # 初始化方法，接受多个参数来设置实例属性
    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: List[ir.Buffer],
        layout: ir.Layout,
        make_kernel_render: Callable[
            [ir.CppTemplateBuffer, Optional[List[ir.IRNode]]], str
        ],
        bmreq: CppBenchmarkRequest,
        template: "CppTemplate",  # type: ignore[name-defined]  # noqa: F821
        info_kwargs: Optional[
            Dict[str, Union[ir.PrimitiveInfoType, List[ir.PrimitiveInfoType]]]
        ] = None,
    ):
        # 调用父类的初始化方法，设置 name 和 layout
        super().__init__(name, input_nodes, layout)
        # 设置类特有的属性：category, make_kernel_render, bmreq, template, info_kwargs
        self.category = category
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.template = template
        self.info_kwargs = info_kwargs

    # 执行预编译操作的方法，确保 bmreq 不为空，然后调用其 precompile 方法
    def precompile(self) -> None:
        assert self.bmreq is not None
        self.bmreq.precompile()

    # 进行基准测试的方法，确保 bmreq 不为空，然后调用其 benchmark 方法
    def benchmark(self, *args, out) -> float:
        assert self.bmreq is not None
        return self.bmreq.benchmark(*args, output_tensor=out)

    # 返回对象的哈希键，由 category 和 bmreq 的哈希键组成
    def hash_key(self) -> str:
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
            ]
        )

    # 返回一个信息字典，说明后端为 CPP，操作类型为未知
    def info_dict(
        self,
    ) -> Dict[str, Union[ir.PrimitiveInfoType, List[ir.PrimitiveInfoType]]]:
        return {"backend": "CPP", "op_type": "unknown"}

    # 返回输出节点的 TensorBox 对象，使用给定的参数初始化 ir.CppTemplateBuffer
    def output_node(self) -> ir.TensorBox:
        return ir.TensorBox.create(
            ir.CppTemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
                template=self.template,
                choice=self,
            )
        )
```