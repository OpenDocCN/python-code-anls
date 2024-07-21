# `.\pytorch\torch\_inductor\codegen\cuda\cuda_kernel.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和类型声明
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

# 导入相关模块和类
from ...autotune_process import CUDABenchmarkRequest
from ...ir import (
    Buffer,
    ChoiceCaller,
    CUDATemplateBuffer,
    IRNode,
    Layout,
    PrimitiveInfoType,
    TensorBox,
)
# 导入工具函数
from ...utils import sympy_product
# 导入虚拟化模块
from ...virtualized import V
# 导入其他模块
from ..common import IndentedBuffer, Kernel, OpOverrides
# 导入 C++ 相关工具
from ..cpp_utils import CppPrinter, DTYPE_TO_CPP

# 如果是类型检查阶段，则导入相关类
if TYPE_CHECKING:
    from torch._inductor.codegen.cuda.cuda_template import CUDATemplate

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)

# 获取用于生成 C++ 代码的表达式对象
cexpr = CppPrinter().doprint


def _normalize_idx(index: int, total_length: int) -> int:
    """
    将索引规范化为非负索引

    Args:
        index (int): 输入索引值
        total_length (int): 总长度

    Returns:
        int: 规范化后的索引值
    """
    return index if index >= 0 else index + total_length


class CUDAKernel(Kernel):
    """
    CUDA / Cutlass 基础 Kernel 类
    """

    overrides = OpOverrides  # type: ignore[assignment]


class CUDATemplateKernel(CUDAKernel):
    """
    使用 CUDA / Cutlass 在 C++ 中定义的模板 Kernel 类
    """

    _EXTRA_CPP_ARGS = "size_t* workspace_size, uint8_t* workspace, cudaStream_t stream"

    def __init__(self, kernel_name):
        """
        初始化 CUDATemplateKernel 类的新实例。

        Args:
            kernel_name (str): Kernel 的名称
        """
        super().__init__()
        self.kernel_name = kernel_name
        # 从参数名到 IRNode 的映射
        self.named_nodes: Dict[str, IRNode] = {}

    def arg_name(self, node: IRNode) -> Optional[str]:
        """
        返回给定输入或输出节点的参数名。

        Args:
            node (IRNode): 输入或输出节点

        Returns:
            Optional[str]: 参数名或 None
        """
        if node is None:
            return None
        return {**self.args.input_buffers, **self.args.output_buffers}.get(
            node.get_name(), None
        )

    def check_not_null(self, node: IRNode) -> str:
        """
        生成检查节点不为空的代码。

        Args:
            node (IRNode): 要检查的节点

        Returns:
            str: 生成的代码字符串
        """
        if node is None:
            return ""

        size_str = self.size(node, 0, -1)
        name_str = self.arg_name(node)
        if name_str is None:
            return ""

        # 生成检查节点不为空的代码块
        res = IndentedBuffer(initial_indent=2)
        res.tabwidth = 1
        res.splice(
            f"""
            {{
              if (!{name_str}) {{
                int64_t {name_str}_size = {size_str};
                if ({name_str}_size > 0) {{
                  throw std::runtime_error("input {name_str} is null but size is not 0!");
                }}
              }}
            }}
            """
        )
        return res.getvalue()

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
            outputs: List of output IRNodes
            names_str: Comma separated list of input + output argument names.
            input_reorder: The actual order of input nodes.
                           e.g. The template might have input argument defined as [X, W, Bias],
                           and the actual input passed into this template could be [Bias, X, W].
                           In this case, the `input_reorder` would be [2, 0, 1].
        """

        # 将输入输出参数名以逗号分隔的字符串转换成列表，去除首尾空格
        names = [x.strip() for x in names_str.strip().split(",")]

        # 检查输入和输出节点的总数是否与参数名列表长度相等，如果不相等则抛出运行时错误
        if len(inputs) + len(outputs) != len(names):
            raise RuntimeError(
                f"{len(inputs) + len(outputs)=} != {len(names)=}, {inputs=}, {outputs=}, {names=}"
            )

        # 如果指定了输入节点的重排顺序，检查输入节点的数量与重排顺序列表长度相等，否则创建默认顺序列表
        if input_reorder is not None:
            assert len(inputs) == len(input_reorder)
        else:
            input_reorder = list(range(len(inputs)))

        # 根据重排顺序将节点名和节点对象添加到命名节点字典和输入缓冲区字典中
        for idx in input_reorder:
            name = names[idx]
            node = inputs[idx]
            if node is not None:
                self.named_nodes[name] = node
                self.args.input_buffers[node.get_name()] = name

        # 将输出节点名和节点对象添加到命名节点字典和输出缓冲区字典中
        for name, node in zip(names[len(inputs) : len(inputs) + len(outputs)], outputs):
            if node is not None:
                self.named_nodes[name] = node
                self.args.output_buffers[node.get_name()] = name

        # 获取参数定义列表并返回导出的函数定义字符串
        arg_defs, *_ = self.args.cpp_argdefs()
        return f"PT_EXPORT int {self.kernel_name}({', '.join(arg_defs)}, {self._EXTRA_CPP_ARGS})"
    ) -> None:
        """
        Generates code to call the kernel through V.graph.wrapper_code.
        used from within torch._inductor.wrapper.WrapperCodeGen

        name: Name of kernel function.
        node: The CUDATemplateBuffer node which contains information about the kernel, it's fused epilogue nodes
        as well as all required inputs and outputs.
        """
        # 获取调用内核的代码生成器对象
        wrapper = V.graph.wrapper_code
        # 获取函数调用的参数、参数类型等信息
        _, call_args, _, arg_types = self.args.python_argdefs()
        
        # 动态生成代码以处理未指定的变量作为0维CPU张量的情况，需要转换为标量
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                # 将未指定的参数转换为标量
                call_args[i] = call_args[i] + ".item()"
            else:
                # 将参数转换为 c_void_p 类型
                call_args[i] = f"c_void_p({call_args[i]}.data_ptr())"

        # workspace_size 指针为 NULL 表示此调用不用于获取 workspace_size。
        # workspace_size 应该在此调用之前已经被检索到。
        call_args.append("None")

        # 如果节点需要的 workspace_size 大于 0，则生成 workspace 的分配代码
        if node.get_workspace_size() > 0:
            wrapper.generate_workspace_allocation(
                node.get_workspace_size(), V.graph.scheduler.current_device, False
            )
            call_args.append("c_void_p(workspace.data_ptr())")
        else:
            call_args.append("None")

        # 生成调用内核的代码
        wrapper.generate_kernel_call(
            name,
            call_args,
            cuda=True,
            triton=False,
            arg_types=arg_types,
        )
        # 如果节点需要的 workspace_size 大于 0，则生成释放 workspace 的代码
        if node.get_workspace_size() > 0:
            wrapper.writeline(wrapper.make_free_by_names(["workspace"]))

    def dtype(self, node: IRNode) -> Optional[str]:
        """
        Generates code which represents dtype of a given node.
        """

        if node is None:
            return "void"
        # 返回给定节点的数据类型对应的 C++ 类型
        return DTYPE_TO_CPP.get(node.get_layout().dtype)

    def cutlass_dtype(self, node: IRNode, default_dtype="void") -> Optional[str]:
        # Helper method, called into from CUTLASSGemmTemplate
        if node is None:
            return default_dtype
        # 返回给定节点的数据类型对应的 CUTLASS 类型
        from torch._inductor.codegen.cuda.cuda_template import CUTLASSTemplate

        return CUTLASSTemplate._DTYPE_TO_CUTLASS[node.get_layout().dtype]

    def max_valid_index(self, node: IRNode, default=-1):
        # Helper method, called into from CUTLASSGemmTemplate
        if node is None:
            return default
        # 计算给定节点的最大有效索引
        max_valid_offset = 0
        for i in range(len(node.get_size())):
            max_valid_offset += (node.get_size()[i] - 1) * node.get_stride()[i]
        return max_valid_offset

    def offset(self, node: IRNode) -> str:
        """
        Generates code which represents offset of a given node.
        """

        if node is None:
            return "0"
        # 返回给定节点的偏移量
        return str(node.get_layout().offset)
    # 生成表示给定节点指针的代码。
    def ptr(self, node: IRNode) -> str:
        """
        Generates code which represents pointer of a given node.
        """

        # 如果节点为空，返回 "nullptr"
        if node is None:
            return "nullptr"
        # 获取节点的参数名
        arg_name = self.arg_name(node)
        # 如果参数名为空，返回 "nullptr"
        if arg_name is None:
            return "nullptr"
        # 获取节点的偏移量
        offset = self.offset(node)
        # 如果偏移量为 "0"，则返回参数名，否则返回带偏移量的参数名
        return arg_name if offset == "0" else f"{arg_name} + {offset}"

    # 从模板代码中调用的钩子，用于获取参数大小。
    # 生成表示给定节点在[start_index, end_index)范围内大小的代码。
    # 如果节点为空，返回默认值。
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

        # 如果节点为空，返回默认值的字符串表示
        if node is None:
            return str(default_value)

        # 标准化起始索引，确保在节点大小的有效范围内
        start_index = _normalize_idx(start_index, len(node.get_size()))
        # 如果结束索引未指定，默认与起始索引相同
        if end_index is None:
            end_index = start_index
        # 标准化结束索引，确保在节点大小的有效范围内
        end_index = _normalize_idx(end_index, len(node.get_size()))

        # 获取节点在[start_index, end_index]范围内的大小
        sizes = node.get_size()[start_index : end_index + 1]
        # 如果大小列表为空，返回默认值的字符串表示
        if len(sizes) == 0:
            return str(default_value)

        # 计算大小的乘积
        val = sympy_product(sizes)
        # 返回重命名索引后的表达式
        return cexpr(self.rename_indexing(val))

    # 从模板代码中调用的钩子，用于获取参数步长。
    # 生成表示给定节点在指定索引处的步长的代码。
    # 如果节点为空，返回默认值。
    def stride(self, node: IRNode, index: int, default_value: int = 0) -> str:
        """
        Hook called from template code to get the stride of an arg.
        Generates code which represents stride of a given node at index.
        If node is None, returns default_value.

        TODO: Will add needed args to pass it in if it is dynamic.
        """

        # 如果节点为空，返回默认值的字符串表示
        if node is None:
            return str(default_value)

        # 标准化索引，确保在节点大小的有效范围内
        index = _normalize_idx(index, len(node.get_size()))
        # 如果索引小于0，返回默认值的字符串表示
        if index < 0:
            return str(default_value)

        # 获取节点在指定索引处的步长
        stride = node.get_stride()[index]
        # 返回重命名索引后的步长表达式
        return cexpr(self.rename_indexing(stride))

    # 从模板代码中调用的钩子，用于获取参数的行或列步长。
    # 返回行或列步长的代码，用于某些 CUTLASS 2.X API。
    # 如果节点为空或步长列表长度小于2，返回默认值。
    def row_or_column_stride(self, node: IRNode, default_value: int = 0) -> str:
        """
        Hook called from template code to get the row or column stride of an arg.
        This is required by some CUTLASS 2.X APIs.
        If the node is in row_major, it returns stride[-2].
        If the node is in column_major, it returns stride[-1].

        TODO: Will add needed args to pass it in if it is dynamic.
        """

        # 如果节点为空或步长列表长度小于2，返回默认值的字符串表示
        if node is None or len(node.get_stride()) < 2:
            return str(default_value)

        # 获取最后两个步长值
        stride0 = node.get_stride()[-1]
        stride1 = node.get_stride()[-2]
        # 如果最后一个步长为1，返回重命名索引后的第二个步长表达式
        if stride0 == 1:
            return cexpr(self.rename_indexing(stride1))
        # 如果倒数第二个步长为1，返回重命名索引后的第一个步长表达式
        elif stride1 == 1:
            return cexpr(self.rename_indexing(stride0))
        # 否则，抛出运行时错误，要求至少有一个步长为1
        else:
            raise RuntimeError(
                f"At least 1 stride should be 1. Strides: {node.get_stride()=}"
            )
class CUDATemplateCaller(ChoiceCaller):
    """
    CUDATemplateCaller

    This class represents a caller for CUDA template kernels. It is a subclass of ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (CUDABenchmarkRequest): The benchmark request for the caller.
        template_buffer (CUDATemplateBuffer): The template buffer for the caller.
    """

    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: List[Buffer],
        layout: Layout,
        make_kernel_render: Callable[[CUDATemplateBuffer, Optional[List[IRNode]]], str],
        bmreq: CUDABenchmarkRequest,
        template: "CUDATemplate",  # type: ignore[name-defined]
        info_kwargs: Optional[Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]],  # type: ignore[type-arg]
    ):
        # 调用父类的初始化方法，传入名称、输入节点列表和布局信息
        super().__init__(name, input_nodes, layout)
        # 设置调用者的类别
        self.category = category
        # 设置用于生成内核的回调函数
        self.make_kernel_render = make_kernel_render
        # 设置CUDA基准请求对象
        self.bmreq = bmreq
        # 设置CUDA模板对象
        self.template = template
        # 设置附加信息关键字参数
        self.info_kwargs = info_kwargs

    def precompile(self) -> None:
        # 断言基准请求对象不为空，预编译基准请求
        assert self.bmreq is not None
        self.bmreq.precompile()

    def benchmark(self, *args, out) -> float:
        # 断言基准请求对象不为空，执行基准测试并返回性能评估结果
        assert self.bmreq is not None
        return self.bmreq.benchmark(
            *args, output_tensor=out
        )  # @TODO: Hack for ensuring that Cutlass Kernel is preferred

    def __str__(self):
        # 返回调用者对象的字符串表示，包括基准请求的源文件名
        return f"CUDATemplateCaller(source_file={self.bmreq.source_file})"

    def call_name(self) -> str:
        # 返回CUDA模板内核函数的调用名称
        return f"cuda_template_kernels.{self.name}"

    def hash_key(self) -> str:
        # 返回调用者对象的哈希键，包括类别和基准请求的哈希键
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
            ]
        )

    def info_dict(self) -> Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        if self.info_kwargs is not None and "op" in self.info_kwargs:
            op: Any = self.info_kwargs["op"]
            # 返回包含操作相关信息的字典，用于记录到自动调优日志文件中
            return {
                "backend": "CUDA",
                "op_type": type(op).__name__,
                "op_conf_name": str(op.configuration_name()),
                "op_arch": str(op.arch),
                "tile_shape": str(op.tile_description.tile_shape),
                "epilogue_schedule": str(op.epilogue_schedule),
                "kernel_schedule": str(op.kernel_schedule),
                "element_accumulator": str(op.accumulator_type()),
                "op_name": str(op.procedural_name()),
                "instruction_shape": str(
                    op.tile_description.math_instruction.instruction_shape
                ),
            }
        else:
            # 如果没有附加信息关键字参数或者没有 "op" 键，则返回默认的未知信息字典
            return {"backend": "CUDA", "op_type": "unknown"}
    # 定义一个方法 `output_node`，返回类型为 `TensorBox`
    def output_node(self) -> TensorBox:
        # 更新 `bmreq` 对象的工作空间大小
        self.bmreq.update_workspace_size()
        # 创建一个 `TensorBox` 对象，使用 `CUDATemplateBuffer` 构造函数
        return TensorBox.create(
            CUDATemplateBuffer(
                layout=self.layout,  # 使用给定的布局参数
                inputs=self.input_nodes,  # 使用 `input_nodes` 作为输入节点
                make_kernel_render=self.make_kernel_render,  # 使用 `make_kernel_render` 函数进行核函数渲染
                workspace_size=self.bmreq.workspace_size,  # 设置工作空间大小为 `bmreq` 对象中计算的大小
                template=self.template,  # 使用给定的模板
            )
        )
```