# `.\pytorch\torch\_inductor\codegen\rocm\rocm_kernel.py`

```py
# mypy: allow-untyped-defs
import logging  # 导入日志模块，用于记录程序运行信息
from typing import Callable, Dict, List, Optional, TYPE_CHECKING, Union  # 导入类型提示相关的模块

from ...ir import Buffer, ChoiceCaller, IRNode, Layout, PrimitiveInfoType, TensorBox  # 导入IR相关的模块
from ...utils import sympy_product  # 导入数学计算相关的模块
from ...virtualized import V  # 导入虚拟化相关的模块
from ..common import IndentedBuffer, Kernel, OpOverrides  # 导入通用工具模块

from ..cpp_utils import CppPrinter  # 导入用于C++代码打印的工具

from .rocm_benchmark_request import ROCmBenchmarkRequest  # 导入ROCm基准请求相关模块
from .rocm_template_buffer import ROCmTemplateBuffer  # 导入ROCm模板缓冲区相关模块

if TYPE_CHECKING:
    from torch._inductor.codegen.rocm.rocm_template import ROCmTemplate  # 条件导入ROCm模板类，仅用于类型检查

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

cexpr = CppPrinter().doprint  # 创建一个用于打印C++代码的表达式对象


def _normalize_idx(index: int, total_length: int) -> int:
    """
    Normalize a negative index to positive within the range of total_length.
    """
    return index if index >= 0 else index + total_length  # 如果索引为负数，则转换为对应正数索引


class ROCmKernel(Kernel):
    """
    Baseclass for ROCm based Kernels
    """

    overrides = OpOverrides  # 设置覆盖参数为OpOverrides，用于类型检查


class ROCmTemplateKernel(ROCmKernel):
    """
    Template kernels defined by ROCm in C++.
    """

    _EXTRA_CPP_ARGS = "size_t* workspace_size, uint8_t* workspace, hipStream_t stream"

    def __init__(self, kernel_name):
        """
        Initializes a new instance of the ROCmTemplateKernel class.

        Args:
            kernel_name (str): The name of the kernel.
        """
        super().__init__()  # 调用父类的初始化方法
        self.kernel_name = kernel_name  # 设置当前对象的内核名称
        # Mapping from arg name to IRNode.
        self.named_nodes: Dict[str, IRNode] = {}  # 初始化一个空的字典，用于存储参数名称到IRNode的映射

    def arg_name(self, node: IRNode) -> Optional[str]:
        """
        Returns arg name of a given input or output node.
        """
        if node is None:
            return None
        return {**self.args.input_buffers, **self.args.output_buffers}.get(
            node.get_name(), None
        )  # 返回给定输入或输出节点的参数名称，从args的input_buffers和output_buffers中获取

    def check_not_null(self, node: IRNode) -> str:
        """
        Generates code to check that a node is not null.
        """

        if node is None:
            return ""  # 如果节点为空，则返回空字符串

        size_str = self.size(node, 0, -1)  # 获取节点的大小信息
        name_str = self.arg_name(node)  # 获取节点的参数名称

        if name_str is None:
            return ""  # 如果参数名称为空，则返回空字符串

        res = IndentedBuffer(initial_indent=8)  # 创建一个缩进的代码缓冲区
        res.tabwidth = 1  # 设置缩进宽度为1

        # 生成检查节点非空的代码块
        res.splice(
            f"""
            if (!{name_str}) {{
                int64_t {name_str}_size = {size_str};
                if ({name_str}_size > 0) {{
                    throw std::runtime_error("input {name_str} is null but size is not 0!");
                }}
            }}
            """
        )
        return res.getvalue()  # 返回生成的代码字符串

    def def_kernel(
        self,
        inputs: List[IRNode],
        outputs: List[IRNode],
        names_str: str = "",
        input_reorder: Optional[List[int]] = None,
    ) -> str:
        """
        Hook called from template code to generate function definition and
        needed args.

        Args:
            inputs: List of input IRNodes
                输入参数：输入的 IRNode 列表
            outputs: List of output IRNodes
                输出参数：输出的 IRNode 列表
            names_str: Comma separated list of input + output argument names.
                参数名称字符串：逗号分隔的输入和输出参数名称列表字符串
            input_reorder: The actual order of input nodes.
                输入重排序：实际输入节点的顺序。
                           例如，模板可能定义输入参数为 [X, W, Bias]，
                           而实际传入模板的输入可能是 [Bias, X, W]。
                           在这种情况下，input_reorder 将会是 [2, 0, 1]。
        """

        names = [x.strip() for x in names_str.strip().split(",")]
        # 拆分参数名称字符串为列表，并去除每个名称两端的空格

        if len(inputs) + len(outputs) != len(names):
            # 如果输入参数和输出参数的总数与参数名称列表的长度不匹配，抛出运行时错误
            raise RuntimeError(
                f"{len(inputs) + len(outputs)=} != {len(names)=}, {inputs=}, {outputs=}, {names=}"
            )

        if input_reorder is not None:
            # 如果输入重排序列表不为空，则检查其长度与输入节点列表长度是否相同
            assert len(inputs) == len(input_reorder)
        else:
            # 如果输入重排序列表为空，则创建一个默认顺序的列表
            input_reorder = list(range(len(inputs)))

        for idx in input_reorder:
            # 遍历输入重排序列表
            name = names[idx]
            # 获取当前索引对应的名称
            node = inputs[idx]
            # 获取当前索引对应的输入节点
            if node is not None:
                # 如果节点不为空，则将节点添加到命名节点字典中，并将节点名称与参数名映射关系存储起来
                self.named_nodes[name] = node
                self.args.input_buffers[node.get_name()] = name

        for name, node in zip(names[len(inputs) : len(inputs) + len(outputs)], outputs):
            # 遍历输出参数名称列表和输出节点列表的对应部分
            if node is not None:
                # 如果节点不为空，则将节点添加到命名节点字典中，并将节点名称与参数名映射关系存储起来
                self.named_nodes[name] = node
                self.args.output_buffers[node.get_name()] = name

        arg_defs, *_ = self.args.cpp_argdefs()
        # 获取参数定义的字符串表示
        return f"PT_EXPORT int {self.kernel_name}({', '.join(arg_defs)}, {self._EXTRA_CPP_ARGS})"
        # 返回生成的函数定义字符串，包括函数名称、参数定义和额外的 C++ 参数
    ) -> None:
        """
        Generates code to call the kernel through V.graph.wrapper_code.
        used from within torch._inductor.wrapper.WrapperCodeGen

        name: Name of kernel function.
        node: The ROCmTemplateBuffer node which contains information about the kernel, it's fused epilogue nodes
        as well as all required inputs and outputs.
        """
        # 获取 kernel 调用代码的包装器对象
        wrapper = V.graph.wrapper_code
        # 从 self.args.python_argdefs() 中获取参数的默认值、调用参数等信息
        _, call_args, _, arg_types = self.args.python_argdefs()
        
        # 将未指定参数包装为 0 维 CPU 张量，需要转换为标量
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"
            else:
                call_args[i] = f"c_void_p({call_args[i]}.data_ptr())"

        # workspace_size ptr 为 NULL 表示此次调用不用于获取 workspace_size
        # workspace_size 应在此调用之前已被检索到
        call_args.append("None")

        if node.get_workspace_size() > 0:
            # 生成工作空间分配的代码
            wrapper.generate_workspace_allocation(
                node.get_workspace_size(), V.graph.scheduler.current_device, False
            )
            call_args.append("c_void_p(workspace.data_ptr())")
        else:
            call_args.append("None")

        # 获取当前设备的索引
        current_device = V.graph.scheduler.get_current_device_or_throw()
        # 生成调用内核函数的代码
        wrapper.generate_kernel_call(
            name,
            call_args,
            device_index=current_device.index,
            cuda=True,
            triton=False,
            arg_types=arg_types,
        )
        if node.get_workspace_size() > 0:
            # 生成释放工作空间的代码
            wrapper.writeline(wrapper.make_free_by_names(["workspace"]))

    def size(
        self,
        node: IRNode,
        start_index: int,
        end_index: Optional[int] = None,
        default_value: int = 0,
    ) -> str:
        """
        Hook called from template code to get the size of an arg.
        Generates code which represents size of a given node in [start_index, end_index).
        If node is None, returns default_value.

        TODO: Will add needed args to pass it in if it is dynamic.
        """

        if node is None:
            return str(default_value)

        # 标准化起始索引和结束索引，确保它们在有效范围内
        start_index = _normalize_idx(start_index, len(node.get_size()))
        if end_index is None:
            end_index = start_index
        end_index = _normalize_idx(end_index, len(node.get_size()))

        # 获取节点在指定范围内的尺寸信息
        sizes = node.get_size()[start_index : end_index + 1]
        if len(sizes) == 0:
            return str(default_value)

        # 计算尺寸的乘积作为结果
        val = sympy_product(sizes)
        return cexpr(self.rename_indexing(val))
class ROCmTemplateCaller(ChoiceCaller):
    """
    ROCmTemplateCaller

    This class represents a caller for ROCm template kernels. It is a subclass of ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (ROCmBenchmarkRequest): The benchmark request for the caller.
        template_buffer (ROCmTemplateBuffer): The template buffer for the caller.
    """

    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: List[Buffer],
        layout: Layout,
        make_kernel_render: Callable[[ROCmTemplateBuffer, Optional[List[IRNode]]], str],
        bmreq: ROCmBenchmarkRequest,
        template: "ROCmTemplate",  # type: ignore[name-defined]
        info_kwargs: Optional[Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]],  # type: ignore[type-arg]
    ):
        # 调用父类构造函数初始化名称、输入节点和布局
        super().__init__(name, input_nodes, layout)
        # 设置分类
        self.category = category
        # 设置用于生成内核渲染的回调函数
        self.make_kernel_render = make_kernel_render
        # 设置 ROCmBenchmarkRequest 对象
        self.bmreq = bmreq
        # 设置 ROCmTemplate 对象
        self.template = template
        # 设置信息关键字参数
        self.info_kwargs = info_kwargs

    def precompile(self) -> None:
        # 确保 bmreq 不为空，预编译 benchmark 请求
        assert self.bmreq is not None
        self.bmreq.precompile()

    def benchmark(self, *args, out) -> float:
        # 确保 bmreq 不为空，执行 benchmark 计算
        assert self.bmreq is not None
        return self.bmreq.benchmark(*args, output_tensor=out)

    def __str__(self):
        # 返回调用对象的字符串表示形式，包括源文件和信息字典
        return f"ROCmTemplateCaller(source_file={self.bmreq.source_file}, {self.info_dict()})"

    def call_name(self) -> str:
        # 返回调用名称，用于 ROCm 模板内核
        return f"rocm_template_kernels.{self.name}"

    def hash_key(self) -> str:
        # 返回用于唯一标识调用对象的哈希键
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
            ]
        )

    def info_dict(self) -> Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        # 返回包含后端、名称和信息参数的字典，用于自动调优日志记录
        return {
            "backend": "ROCm",
            "name": self.name,
            **dict(self.info_kwargs["op"].dict_items()),  # 获取并添加信息关键字参数
        }

    def output_node(self) -> TensorBox:
        # 更新 benchmark 请求的工作空间大小
        self.bmreq.update_workspace_size()
        # 创建并返回 TensorBox 对象，包含 ROCmTemplateBuffer 实例
        return TensorBox.create(
            ROCmTemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
                workspace_size=self.bmreq.workspace_size,
                template=self.template,
            )
        )
```