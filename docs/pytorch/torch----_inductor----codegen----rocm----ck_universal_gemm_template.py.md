# `.\pytorch\torch\_inductor\codegen\rocm\ck_universal_gemm_template.py`

```
# 设置日志记录器，用于当前模块中的日志输出
import logging
# 导入随机模块，用于生成随机数和随机选择
import random
# 导入 List 和 Optional 类型提示
from typing import List, Optional

# 导入 sympy 库，用于处理符号运算
import sympy

# 导入 PyTorch 库
import torch
# 导入 PyTorch 中私有模块的配置
from torch._inductor import config
# 导入 PyTorch 中 ROCm 相关模块的模板和代码生成器
from torch._inductor.codegen.rocm.ck_template import CKTemplate
from torch._inductor.codegen.rocm.rocm_kernel import ROCmTemplateKernel
# 导入 PyTorch 中 IR 层次结构相关模块
from torch._inductor.ir import Buffer, Layout

# 导入相对路径下的工具函数和变量
from ...utils import IndentedBuffer, try_import_ck_lib

# 解构导入的结果，包括 '_'，生成操作库，预选操作，和 CKGemmOperation
_, gen_ops_library, gen_ops_preselected, CKGemmOperation = try_import_ck_lib()

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


# 定义函数用于检查是否为静态整数
def is_static_int(number):
    return isinstance(number, (int, sympy.Integer))


# 将 PyTorch 的布局转换为 CK 的布局类型
def torch_layout_to_ck_layout(torch_layout):
    if torch_layout.stride[-1] == 1:
        return "Row"
    elif torch_layout.stride[-2] == 1:
        return "Col"
    else:
        return None


# 定义一个继承自 CKTemplate 的 CKGemmTemplate 类
class CKGemmTemplate(CKTemplate):
    # 定义 JINJA 模板，用于渲染 CK 通用 GEMM
    gemm_template = r"""{{version_comment}}
    {{headers}}
    {{globals}}
    {{instance_definition}}
    extern "C" {
    {{kernel_definition}} {
        auto gemm = {{instance_type}} {};
        auto invoker = gemm.MakeInvoker();

        constexpr auto M = {{M}};
        constexpr auto N = {{N}};
        constexpr auto K = {{K}};
        constexpr auto StrideA = std::is_same_v<{{a_layout}}, Row> ? K : M;
        constexpr auto StrideB = std::is_same_v<{{b_layout}}, Row> ? N : K;
        constexpr auto StrideC = std::is_same_v<{{c_layout}}, Row> ? N : M;
        constexpr auto KBatch = 1; // split k into batches

        auto argument = gemm.MakeArgument(
            reinterpret_cast<const {{a_element_dtype}}*>(X),
            reinterpret_cast<const {{b_element_dtype}}*>(W),
            reinterpret_cast<{{c_element_dtype}}*>(Y),
            M,
            N,
            K,
            StrideA,
            StrideB,
            StrideC,
            KBatch,
            {{a_elementwise_op}} {},
            {{b_elementwise_op}} {},
            {{c_elementwise_op}} {}
        );
        if (!gemm.IsSupportedArgument(argument)) {
            // 我们尽力静态避免这种情况发生在 `CKGemmTemplate.filter_op` 中
            std::cerr << "invalid argument for gemm instance " << gemm.GetTypeString() << std::endl;
            argument.Print();
            return -23;
        }
        if (workspace_size) {
            *workspace_size = gemm.GetWorkSpaceSize(&argument);
            return 0;
        }
        {{null_checks}}
        // 运行内核
        float elapsed_time = invoker.Run(argument, StreamConfig{stream, /* time kernel */ false, /* log level */ kDEBUG_LOG});
        return 0;
    } // kernel definition
    } // extern C
    """

    # 初始化函数，接受一系列输入节点、布局、alpha 和 beta 值，并可选地接受输入重排序列表
    def __init__(
        self,
        input_nodes: List[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: Optional[List[int]] = None,
    ):
        super().__init__(
            "ck_gemm_template",
            input_nodes=input_nodes,
            layout=layout,
            input_reorder=input_reorder,
        )
        self.alpha = alpha
        self.beta = beta


        # 调用父类构造函数，初始化模板名称、输入节点、布局和输入重新排序选项
        super().__init__(
            "ck_gemm_template",
            input_nodes=input_nodes,
            layout=layout,
            input_reorder=input_reorder,
        )
        # 设置对象属性 alpha 和 beta，用于后续的计算
        self.alpha = alpha
        self.beta = beta



    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                // CK GEMM header(s)

                #include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"
            """
        )
        return res


    # 生成继承父类方法 header 的重写版本，添加特定的头文件声明
    def header(self) -> IndentedBuffer:
        res = super().header()  # 调用父类的 header 方法，获取基础缓冲区
        res.splice(
            """
                // CK GEMM header(s)

                #include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"
            """
        )
        return res  # 返回更新后的缓冲区



    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                // CK GEMM globals

                using Row = ck::tensor_layout::gemm::RowMajor;
                using Col = ck::tensor_layout::gemm::ColumnMajor;

                using BlockGemmPipelineScheduler = ck::BlockGemmPipelineScheduler;
                using GemmSpecialization = ck::tensor_operation::device::GemmSpecialization;
                using BlockGemmPipelineVersion = ck::BlockGemmPipelineVersion;
            """
        )
        return res


    # 生成继承父类方法 globals 的重写版本，添加特定的全局声明
    def globals(self) -> IndentedBuffer:
        res = super().globals()  # 调用父类的 globals 方法，获取基础缓冲区
        res.splice(
            """
                // CK GEMM globals

                using Row = ck::tensor_layout::gemm::RowMajor;
                using Col = ck::tensor_layout::gemm::ColumnMajor;

                using BlockGemmPipelineScheduler = ck::BlockGemmPipelineScheduler;
                using GemmSpecialization = ck::tensor_operation::device::GemmSpecialization;
                using BlockGemmPipelineVersion = ck::BlockGemmPipelineVersion;
            """
        )
        return res  # 返回更新后的缓冲区



    def emit_ck_instance(self, op: "CKGemmOperation"):
        # The Jinja template for generating a C++ type alias *definition* for a Universal GEMM instance
        template_definition = r"""
    // Gemm operator {{operation_name}}
    using Operation_{{operation_name}} =
        ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
            {{template_params}}>;




    # 生成一个用于生成通用 GEMM 实例的 C++ 类型别名定义的 Jinja 模板
    def emit_ck_instance(self, op: "CKGemmOperation"):
        template_definition = r"""
    // Gemm operator {{operation_name}}
    using Operation_{{operation_name}} =
        ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
            {{template_params}}>;
        """
        # The Jinja template for generating a C++ type alias *usage* for a Universal GEMM instance
        template_type = r"""
    Operation_{{operation_name}}
"""
        # Empty list to store template parameters
        template_params = []
        # Iterate over key-value pairs in op's dictionary representation
        for field_name, field_value in op.dict_items():
            # Check if the field_value is a tuple
            if isinstance(field_value, tuple):
                # Append formatted string with field_name and tuple elements as template parameters
                template_params.append(
                    f"/* {field_name} */ S<{', '.join(map(str, iter(field_value)))}>"
                )
            else:
                # If field_value is not None, append it with field_name as template parameter
                if field_value is not None:
                    template_params.append(f"/* {field_name} */ {field_value}")
        # Render the template for instance definition using stored template parameters
        return self._template_from_string(template_definition).render(
            operation_name=op.name(),
            template_params=(",\n" + 12 * " ").join(template_params),
        ), self._template_from_string(template_type).render(operation_name=op.name())

    # Method to render the template with ROCmTemplateKernel kernel and CKGemmOperation op
    def render(self, kernel: ROCmTemplateKernel, op: "CKGemmOperation", **kwargs) -> str:  # type: ignore[override]
        """
        The primary entry point for the code rendering process used in this template.
        """
        # Retrieve epilogue_nodes from kwargs, expect it to be None or an empty list
        epilogue_nodes = kwargs.get("epilogue_nodes", None)
        assert epilogue_nodes is None or 0 == len(epilogue_nodes)
        # Retrieve template_buffer_node from kwargs and assign it to self.output_node if not None
        template_buffer_node = kwargs.get("template_buffer_node", None)
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        # Generate CK instance definition and type using op
        instance_definition, instance_type = self.emit_ck_instance(op)
        # Assign input and output nodes
        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None  # TBD support gemm_bias

        # Generate version_comment string with details about generated code
        version_comment = rf"""/**
* Generated code for CK inductor backend
* See {type(self).__module__}.{type(self).__qualname__}
*
* Problem size M={X.get_layout().size[-2]} N={W.get_layout().size[-1]} K={X.get_layout().size[-1]}
* Template instance {op}
*
* {torch.__version__=}
* {torch.version.git_version=}
*/
    # 使用给定的 gemm_template 渲染模板，并传入以下参数：
    # - headers: 获取 header() 方法返回的字符串
    # - globals: 获取 globals() 方法返回的字符串
    # - instance_definition: 实例定义
    # - kernel_definition: 调用 kernel.def_kernel() 方法定义核函数
    #   - inputs: 输入参数列表 [X, W, Bias]
    #   - outputs: 输出参数列表 [Y]
    #   - names_str: 参数名字符串 "X, W, Bias, Y"
    #   - input_reorder: 输入重新排序方式
    # - instance_type: 实例类型
    # - M: X 张量在倒数第二维的大小
    # - K: X 张量在最后一维的大小
    # - N: W 张量在最后一维的大小
    # - a_elementwise_op, b_elementwise_op, c_elementwise_op: 元素操作类型
    # - a_element_dtype, b_element_dtype, c_element_dtype: 元素数据类型
    # - a_layout, b_layout, c_layout: 布局类型
    # - null_checks: 对 X, W, Y 节点进行非空检查，并将结果拼接成字符串
    # - version_comment: 版本注释信息

def _is_rcr_f16(self):
    # 获取输入节点和输出节点的布局信息
    X_meta, W_meta, Y_meta = (
        T.get_layout() for T in [*self.input_nodes, self.output_node]
    )
    # 获取输入节点和输出节点的数据类型，并映射到相应的 Ck 数据类型
    X_dtype, W_dtype, Y_dtype = (
        self._TORCH_DTYPE_TO_CK[m.dtype] for m in (X_meta, W_meta, Y_meta)
    )
    # 将输入节点和输出节点的布局信息转换为 Ck 的布局类型
    X_layout, W_layout, Y_layout = (
        torch_layout_to_ck_layout(m) for m in (X_meta, W_meta, Y_meta)
    )
    
    # 返回布局为 "Row" 的 X，布局为 "Col" 的 W，布局为 "Row" 的 Y，
    # 并且数据类型为 "F16" 的 X，W，Y 的布尔值结果
    return (
        X_dtype == "F16"
        and W_dtype == "F16"
        and Y_dtype == "F16"
        and X_layout == "Row"
        and W_layout == "Col"
        and Y_layout == "Row"
    )
    def gen_ops(self):
        """
        Creates a list of `CKGemmOperation` instances that match the GEMM operation this template represents.
        The instances are guaranteed to have the correct layout, dtype and dimension padding for the GEMM input arguments.

        An instance may invalidate the GEMM configuration at runtime.
        Such instances will be assigned +inf runtime by the autotune process.
        """
        # 根据配置选择预选的实例或库中生成的实例，根据条件进行筛选
        unfiltered_instances = (
            gen_ops_preselected()
            if config.rocm.use_preselected_instances and self._is_rcr_f16()
            else gen_ops_library()
        )
        # 对未经筛选的实例列表应用筛选函数，得到符合条件的实例列表
        filtered_instances = list(
            filter(lambda op: self.filter_op(op), unfiltered_instances)
        )
        # 当使用固定的实例列表顺序时，通常会选择相似的实例子集。随机化选择解决了这个问题。
        random.seed(-11)
        # 从筛选后的实例中随机选择一定数量的实例，或者选择全部实例（如果数量小于配置的最大配置数）
        chosen_instances = (
            random.sample(
                filtered_instances,
                min(len(filtered_instances), config.rocm.n_max_profiling_configs),
            )
            if config.rocm.n_max_profiling_configs
            else filtered_instances
        )
        # 记录调试信息，输出筛选后的实例数量和实例列表
        log.debug(
            "generated %d ck instances after filter: %s",
            len(chosen_instances),
            chosen_instances,
        )
        # 返回选定的实例列表
        return chosen_instances

    @staticmethod
    def add_ck_gemm_choices(
        choices,
        layout,
        input_nodes,
        alpha=1,
        beta=0,
        input_reorder=None,
    ):
        """
        Add Composable Kernel Universal GEMM instance choices to the auto-tuning list.
        """
        # 创建一个 CKGemmTemplate 实例，用于生成操作实例列表
        template = CKGemmTemplate(
            input_nodes,
            layout,
            alpha=alpha,
            beta=beta,
            input_reorder=input_reorder,
        )
        # 生成符合模板的操作实例列表
        ops = template.gen_ops()
        # 遍历每个操作实例，将其作为选择项添加到给定的选择列表中
        for op in ops:
            template.maybe_append_choice(
                choices,
                op=op,
            )
```