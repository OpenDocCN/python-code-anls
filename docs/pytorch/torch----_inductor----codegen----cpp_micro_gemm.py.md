# `.\pytorch\torch\_inductor\codegen\cpp_micro_gemm.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和库
import dataclasses  # 导入dataclasses模块，用于数据类的支持
from enum import Enum  # 导入Enum枚举类型，用于定义枚举类型
from typing import Callable, Dict, List, Optional, Type  # 导入类型注解需要的各种类型

import sympy  # 导入sympy库，用于符号计算

import torch  # 导入torch库，用于深度学习相关的计算

from .. import ir  # 从当前目录的上级目录导入ir模块
from ..cpu_vec_isa import pick_vec_isa, VecAMX, VecAVX2, VecAVX512, VecISA  # 从cpu_vec_isa模块导入特定的向量指令集
from ..utils import IndentedBuffer, parallel_num_threads  # 从utils模块导入IndentedBuffer和parallel_num_threads工具类
from ..virtualized import V  # 从virtualized模块导入V类
from .common import KernelTemplate  # 从当前目录的common模块导入KernelTemplate类
from .cpp_template_kernel import CppTemplateKernel  # 从当前目录的cpp_template_kernel模块导入CppTemplateKernel类
from .cpp_utils import DTYPE_TO_CPP, GemmBlocking, value_to_cpp  # 从当前目录的cpp_utils模块导入各种工具类和函数

class LayoutType(Enum):
    """
    枚举类型LayoutType，定义了NORMAL和VNNI2两种布局类型
    """
    NORMAL = 0  # NORMAL布局，值为0
    VNNI2 = 1  # VNNI2布局，值为1

class CppMicroGemm:
    """
    一个类，用于生成计算小型矩阵乘法的内核代码。

    微型GEMM内核负责寄存器阻塞、指令选择和其他特定于CPU架构的优化。

    子类需要重写`codegen_define`方法来定义由`codegen_call`生成的代码调用的内核函数。
    """

    # TODO(jgong5): support constant shapes and lds as template args.
    # 定义内核声明的字符串模板
    DECLARE_KERNEL = r"""
template <bool accum>
inline void {{kernel_name}}(
{%- if kernel_extra_args_declare %}
    {{kernel_extra_args_declare}}
{%- endif %}
    const {{input_t}}* __restrict__ A,
    const {{input_t}}* __restrict__ B,
    {{output_t}}* __restrict__ C,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc
)
"""

    def __init__(
        self,
        name,
        input_dtype,
        output_dtype,
        compute_dtype,
        register_blocking,
        alpha=1,
    ):
        """
        初始化方法，设置微型GEMM内核的基本属性。

        Args:
            name: 内核名称
            input_dtype: 输入数据类型
            output_dtype: 输出数据类型
            compute_dtype: 计算数据类型
            register_blocking: 寄存器阻塞配置
            alpha: 计算参数alpha，默认为1
        """
        self.name = name  # 设置内核名称
        self.input_dtype = input_dtype  # 设置输入数据类型
        self.output_dtype = output_dtype  # 设置输出数据类型
        self.compute_dtype = compute_dtype  # 设置计算数据类型
        self.register_blocking = register_blocking  # 设置寄存器阻塞配置
        self.alpha = alpha  # 设置计算参数alpha

    def get_common_options(self):
        """
        返回常用选项的字典，用于生成内核代码。

        Returns:
            dict: 包含常用选项的字典
        """
        return {
            "torch": torch,
            "kernel_name": self.name,
            "input_dtype": self.input_dtype,
            "output_dtype": self.output_dtype,
            "compute_dtype": self.compute_dtype,
            "input_t": DTYPE_TO_CPP[self.input_dtype],
            "output_t": DTYPE_TO_CPP[self.output_dtype],
            "compute_t": DTYPE_TO_CPP[self.compute_dtype],
            "alpha": self.alpha,
            "kernel_extra_args_declare": self.get_kernel_extra_args_declare(),
        }

    def get_kernel_declaration(self):
        """
        生成内核声明的字符串。

        Returns:
            str: 内核声明的字符串
        """
        options = self.get_common_options()
        return KernelTemplate._template_from_string(self.DECLARE_KERNEL).render(options)

    def get_kernel_extra_args_declare(self) -> str:
        """
        返回额外的内核参数声明字符串。

        Returns:
            str: 额外的内核参数声明字符串
        """
        return ""

    def get_kernel_extra_args(self) -> str:
        """
        返回额外的内核参数字符串。

        Returns:
            str: 额外的内核参数字符串
        """
        return ""

    def codegen_define(self, kernel: CppTemplateKernel) -> str:
        """
        生成内核定义的代码。

        Args:
            kernel: CppTemplateKernel对象，表示要生成的内核

        Returns:
            str: 内核定义的代码
        """
        raise NotImplementedError

    def codegen_call(
        self,
        kernel: CppTemplateKernel,
        A: ir.Buffer,
        B: ir.Buffer,
        C: ir.Buffer,
        accum: bool,
    ) -> str:
        """
        Generate the code for calling the templated kernel that computes
        `C += alpha * A @ B` if `accum` is True, or `C = alpha * A @ B` otherwise.
        """
        # 获取 A 在内核中的指针
        A_ptr = f"&({kernel.index(A, [0, 0])})"
        # 获取 B 在内核中的指针
        B_ptr = f"&({kernel.index(B, [0, 0])})"
        # 获取 C 在内核中的指针
        C_ptr = f"&({kernel.index(C, [0, 0])})"
        # 获取 C 的行数
        M = kernel.size(C, 0)
        # 获取 C 的列数
        N = kernel.size(C, 1)
        # 获取 A 的列数（同时也是 B 的行数）
        K = kernel.size(A, 1)
        # 获取 A 的列偏移
        lda = kernel.stride(A, 0)
        # 获取 B 的列偏移
        ldb = kernel.stride(B, 0)
        # 获取 C 的列偏移
        ldc = kernel.stride(C, 0)
        # 创建一个缩进的代码缓冲区
        res = IndentedBuffer()
        # 写入调用内核函数的代码行，并使用内核名称和累加标志构建函数调用
        res.writeline(f"{self.name}<{value_to_cpp(accum, 'bool')}>(")
        with res.indent():
            # 获取内核的额外参数字符串
            extra_args = self.get_kernel_extra_args()
            if extra_args:
                res.writeline(extra_args)
            # 依次写入 A_ptr, B_ptr, C_ptr, M, N, K, lda, ldb, ldc 的参数
            res.writeline(f"{A_ptr},")
            res.writeline(f"{B_ptr},")
            res.writeline(f"{C_ptr},")
            res.writeline(f"{M},")
            res.writeline(f"{N},")
            res.writeline(f"{K},")
            res.writeline(f"{lda},")
            res.writeline(f"{ldb},")
            res.writeline(f"{ldc}")
        # 写入调用结束的分号并返回整个生成的代码字符串
        res.writeline(");")
        return res.getvalue()

    # 初始化代码生成，目前为空字符串
    def codegen_init(
        self,
        kernel: CppTemplateKernel,
    ) -> str:
        return ""

    # 结束代码生成，目前为空字符串
    def codegen_finalize(
        self,
        kernel: CppTemplateKernel,
    ) -> str:
        return ""

    # 获取布局类型，这里返回 LayoutType.NORMAL
    def get_b_layout(self) -> LayoutType:
        return LayoutType.NORMAL
@dataclasses.dataclass
class CppMicroGemmConfig:
    input_dtype: torch.dtype
    output_dtype: torch.dtype
    compute_dtype: torch.dtype
    vec_isa_cls: Type[VecISA]
    register_blocking: GemmBlocking
    extra_check: Optional[Callable[..., bool]] = None



micro_gemm_configs: Dict[Type[CppMicroGemm], List[CppMicroGemmConfig]] = {}



def register_micro_gemm(*configs):
    def inner(cls):
        # 检查是否已经注册了该微小GEMM类，确保唯一性
        assert (
            cls not in micro_gemm_configs
        ), f"Duplicate micro_gemm registration for {cls}"
        # 检查提供给该微小GEMM类的配置列表不为空
        assert len(configs) > 0, f"No micro_gemm configs provided for {cls}"
        # 将该微小GEMM类与其配置列表关联存储
        micro_gemm_configs[cls] = list(configs)
        return cls

    return inner



def generate_gemm_config(
    vec_isa_cls,
    register_blockings,
    input_dtype=torch.float,
    output_dtype=None,
    compute_dtype=None,
    extra_check=None,
):
    if output_dtype is None:
        output_dtype = input_dtype
    if compute_dtype is None:
        compute_dtype = output_dtype
    # 根据提供的参数生成多个CppMicroGemmConfig实例的列表
    return [
        CppMicroGemmConfig(
            input_dtype,
            output_dtype,
            compute_dtype,
            vec_isa_cls,
            GemmBlocking(*blocking),
            extra_check,
        )
        for blocking in register_blockings
    ]



class CppMicroGemmRef(CppMicroGemm):
    """
    A reference implementation of the CppMicroGemm class with naive C++ code.
    It is used for correctness debugging.
    """

    TEMPLATE_ENTRY = r"""
{{declare_kernel}} {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            {{compute_t}} result = accum ? C[m * ldc + n] : 0;
            for (int64_t k = 0; k < K; ++k) {
                result += ({{compute_t}})A[m * lda + k] * ({{compute_t}})B[k * ldb + n] * {{alpha}};
            }
            C[m * ldc + n] = result;
        }
    }
}
"""

    def __init__(self, name, input_dtype, output_dtype, compute_dtype, alpha):
        super().__init__(
            name, input_dtype, output_dtype, compute_dtype, GemmBlocking(1, 1, 1), alpha
        )

    def codegen_define(self, kernel: CppTemplateKernel) -> str:
        options = {
            "declare_kernel": self.get_kernel_declaration(),
            **self.get_common_options(),
        }
        # 使用模板引擎渲染模板字符串，并返回生成的代码字符串
        return KernelTemplate._template_from_string(self.TEMPLATE_ENTRY).render(options)



def check_fp32_vec_extra(config, m, n, k, alpha, num_threads):
    # TODO(jgong5): support n % n_block_size != 0
    # 检查是否满足特定条件以支持额外的浮点32位向量化操作
    return n % config.register_blocking.block_n == 0



@register_micro_gemm(
    *generate_gemm_config(
        VecAVX512,
        [(8, 48, 1), (8, 32, 1), (16, 16, 1)],
        input_dtype=torch.float,
        extra_check=check_fp32_vec_extra,
    ),
    *generate_gemm_config(
        VecAVX512,
        [(8, 48, 1), (8, 32, 1), (16, 16, 1)],
        input_dtype=torch.bfloat16,
        output_dtype=torch.float,
        extra_check=check_fp32_vec_extra,
    ),



    *generate_gemm_config(
        VecAVX512,
        [(8, 48, 1), (8, 32, 1), (16, 16, 1)],
        input_dtype=torch.bfloat16,
        output_dtype=torch.float,
        extra_check=check_fp32_vec_extra,
    ),



)


### 注释完毕。
    *generate_gemm_config(
        VecAVX512,                               # 调用 generate_gemm_config 函数，使用 VecAVX512 作为参数
        [(8, 48, 1), (8, 32, 1), (16, 16, 1)],    # 传入元组列表作为参数，定义不同的配置
        input_dtype=torch.half,                  # 指定输入数据类型为 torch.half
        output_dtype=torch.float,                # 指定输出数据类型为 torch.float
        extra_check=check_fp32_vec_extra,        # 指定额外检查函数为 check_fp32_vec_extra
    ),
    *generate_gemm_config(
        VecAVX2,                                 # 调用 generate_gemm_config 函数，使用 VecAVX2 作为参数
        [(4, 24, 1), (4, 16, 1), (8, 8, 1)],      # 传入元组列表作为参数，定义不同的配置
        input_dtype=torch.float,                 # 指定输入数据类型为 torch.float
        extra_check=check_fp32_vec_extra,        # 指定额外检查函数为 check_fp32_vec_extra
    ),
    *generate_gemm_config(
        VecAVX2,                                 # 调用 generate_gemm_config 函数，再次使用 VecAVX2 作为参数
        [(4, 24, 1), (4, 16, 1), (8, 8, 1)],      # 传入元组列表作为参数，定义不同的配置
        input_dtype=torch.bfloat16,               # 指定输入数据类型为 torch.bfloat16
        output_dtype=torch.float,                # 指定输出数据类型为 torch.float
        extra_check=check_fp32_vec_extra,        # 指定额外检查函数为 check_fp32_vec_extra
    ),
    *generate_gemm_config(
        VecAVX2,                                 # 调用 generate_gemm_config 函数，再次使用 VecAVX2 作为参数
        [(4, 24, 1), (4, 16, 1), (8, 8, 1)],      # 传入元组列表作为参数，定义不同的配置
        input_dtype=torch.half,                  # 指定输入数据类型为 torch.half
        output_dtype=torch.float,                # 指定输出数据类型为 torch.float
        extra_check=check_fp32_vec_extra,        # 指定额外检查函数为 check_fp32_vec_extra
    ),
# 定义 CppMicroGemmFP32Vec 类，继承自 CppMicroGemm 类
class CppMicroGemmFP32Vec(CppMicroGemm):
    """
    This class generates the code for micro gemm using fp32 vec instructions for compute.
    It supports input types of torch.float, torch.bfloat16, and torch.half with fp32 output.
    """

    # 定义 TEMPLATE_ENTRY 常量，包含生成的模板代码字符串
    TEMPLATE_ENTRY = r"""
{{declare_kernel}} {
    TORCH_CHECK(N % {{block_n}} == 0, "N dimension must be multiple of {{block_n}}");
    TORCH_CHECK(K % {{block_k}} == 0, "K dimension must be multiple of {{block_k}}");
    // TODO(jgong5): loop unroll for M and N
    for (int64_t m = 0; m < M; m += {{block_m}}) {
        int64_t block_m = std::min<int64_t>(M - m, {{block_m}});
        for (int64_t n = 0; n < N; n += {{block_n}}) {
            // 根据 block_m 的大小选择不同的 kernel 函数进行调用
            if (block_m == {{block_m}}) {
                {{kernel_name}}_kernel<{{block_m}}, {{block_n}}, accum>(
                    A + m * lda,
                    B + n,
                    C + m * ldc + n,
                    K,
                    lda,
                    ldb,
                    ldc
                );
            } else {
                // 根据 block_m 的不同情况选择不同的 case 处理
                switch (block_m) {
                {%- for b in range(block_m - 1, 0, -1) %}
                case {{b}}:
                    {{kernel_name}}_kernel<{{b}}, {{block_n}}, accum>(
                        A + m * lda,
                        B + n,
                        C + m * ldc + n,
                        K,
                        lda,
                        ldb,
                        ldc
                    );
                    break;
                {%- endfor %}
                // 处理未覆盖到的 block_m 值
                default:
                    {{kernel.assert_function}}(false, "Unsupported block_m: ", block_m);
                }
            }
        }
    }
}
"""

    # 定义 TEMPLATE_KERNEL 常量，包含生成的模板 kernel 函数代码字符串
    TEMPLATE_KERNEL = r"""
template <int64_t BLOCK_M, int64_t BLOCK_N, bool accum>
inline void {{kernel_name}}_kernel(
    const {{input_t}}* __restrict__ A,
    const {{input_t}}* __restrict__ B,
    {{output_t}}* __restrict__ C,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc
) {
    // 使用 at::vec::Vectorized 类型进行向量化计算
    using Vectorized = at::vec::Vectorized<{{compute_t}}>;
    using VectorizedIn = at::vec::Vectorized<{{input_t}}>;
    constexpr auto VLEN = Vectorized::size();
    constexpr auto ROWS = BLOCK_M;  // 定义行数为 BLOCK_M
    constexpr auto COLS = BLOCK_N / VLEN;  // 定义列数为 BLOCK_N / VLEN

    Vectorized va;  // 定义向量化对象 va
    at::vec::VectorizedN<{{compute_t}}, COLS> vb;  // 定义包含 COLS 列的向量化对象 vb
    at::vec::VectorizedN<{{compute_t}}, ROWS*COLS> vc;  // 定义包含 ROWS*COLS 个元素的向量化对象 vc

    // 定义 loadc 函数，用于加载 C 中的数据
    auto loadc = [&](auto i) {
        // 如果需要累加，则加载 C 中指定位置的向量化数据
        if constexpr (accum) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
            vc[i] = Vectorized::loadu(C + row * ldc + col * VLEN);
        } else {
            vc[i] = Vectorized(0.0f);  // 否则初始化为零向量
        }
    };
    c10::ForcedUnroll<ROWS * COLS>{}(loadc);  // 使用 c10::ForcedUnroll 进行强制展开
    // 定义一个 lambda 函数 compute，用于计算矩阵乘法中的每个元素
    auto compute = [&, COLS](auto i, int k) {
        // 计算当前索引对应的行号和列号
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;

        // 如果列号为0，则根据条件选择性地计算 va
        if constexpr (col == 0) {
            {%- if alpha != 1 %}
            // 如果 alpha 不等于 1，则用 A 中的元素乘以 alpha 创建一个 Vectorized 对象 va
            va = Vectorized(static_cast<{{compute_t}}>(A[row * lda + k]) * {{alpha}});
            {%- else %}
            // 否则，直接用 A 中的元素创建一个 Vectorized 对象 va
            va = Vectorized(static_cast<{{compute_t}}>(A[row * lda + k]));
            {%- endif %}
        }

        // 如果行号为0，则根据数据类型的不同加载 B 的数据到 vb 数组中的列
        if constexpr (row == 0) {
            {%- if input_dtype == torch.bfloat16 or input_dtype == torch.float16 %}
            // 如果输入数据类型是 torch.bfloat16 或 torch.float16，则将 B 中的数据加载并转换为指定的 compute_t 类型
            auto b = VectorizedIn::loadu(B + k * ldb + col * VLEN, VLEN);
            vb[col] = at::vec::convert<{{compute_t}}>(b);
            {%- else %}
            // 否则，直接加载 B 中的数据到 vb 数组中的列
            vb[col] = Vectorized::loadu(B + k * ldb + col * VLEN);
            {%- endif %}
        }

        // 计算当前元素在结果矩阵 C 中的索引，并用 va 和 vb[col] 计算并累加到 vc[idx]
        constexpr int idx = row * COLS + col;
        vc[idx] = at::vec::fmadd(va, vb[col], vc[idx]);
    };

    // 使用指定的展开方式展开循环，执行计算
    {{kernel.unroll_pragma(4)}}
    for (int k = 0; k < K; ++k) {
        // 使用 ForcedUnroll 对象强制展开 ROWS * COLS 次循环，每次调用 compute 函数计算一次
        c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
    }

    // 将计算结果存储到矩阵 C 中
    // 定义一个 lambda 函数 storec，用于将 vc[i] 中的数据存储到 C 中对应的位置
    auto storec = [&](auto i) {
        // 计算当前索引对应的行号和列号
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;
        // 将 vc[i] 中的数据存储到 C 中的对应位置
        vc[i].store(C + row * ldc + col * VLEN);
    };
    // 使用 ForcedUnroll 对象强制展开 ROWS * COLS 次循环，每次调用 storec 函数存储一次结果
    c10::ForcedUnroll<ROWS * COLS>{}(storec);
class CppMicroGemmAMX(CppMicroGemm):
    """
    This class generates the code for micro gemm using Advanced Matrix eXtention (AMX)
    instructions available in 4th generation Intel Xeon for compute.
    It supports input types of torch.bfloat16 with fp32 output.
    TODO(jgong5): support int8 data type.
    """

    TEMPLATE_ENTRY = r"""
{{declare_kernel}} {
    TORCH_CHECK(N % {{block_n}} == 0, "N dimension must be multiple of {{block_n}}");
    TORCH_CHECK(K % 2 == 0, "K dimension must be multiple of 2");
    // TODO(jgong5): loop unroll for M and N
    for (int64_t m = 0; m < M; m += {{block_m}}) {
        int64_t block_m = std::min<int64_t>(M - m, {{block_m}});
        int64_t m_tail = m;
        for (int64_t n = 0; n < N; n += {{block_n}}) {
            {%- for num_rows in range(block_m, 0, -16) %}
            {%- if num_rows != block_m %}
            else
            {%- endif %}
            if (block_m >= {{num_rows}}) {
                {{kernel_name}}_amx_kernel_{{num_rows}}_{{num_columns}}<accum>(
                    amx_state,
                    A + m * lda,
                    B + n,
                    C + m * ldc + n,
                    K,
                    lda,
                    ldb,
                    ldc,
                    16
                );
                block_m -= {{num_rows}};
                m_tail += {{num_rows}};
            }
            {%- endfor %}
            if (block_m > 0) {
                {{kernel_name}}_amx_kernel_16_{{num_columns}}<accum>(
                    amx_state,
                    A + m_tail * lda,
                    B + n,
                    C + m_tail * ldc + n,
                    K,
                    lda,
                    ldb,
                    ldc,
                    block_m
                );
            }
        }
    }
}
"""

    TEMPLATE_KERNEL = r"""
template <bool accum>


注释：
- `TEMPLATE_ENTRY`: 定义了一个字符串模板，用于生成微型矩阵乘法（micro gemm）的代码，使用了Advanced Matrix eXtention (AMX)指令，适用于第四代Intel Xeon处理器。支持torch.bfloat16输入类型和fp32输出。包含了一个TODO注释，提醒添加对int8数据类型的支持。
- `TEMPLATE_KERNEL`: 定义了一个模板函数，用于生成带有累加标志的微型矩阵乘法内核函数。
// 定义内联函数，执行 AMX 加速的矩阵乘法核心计算，处理输入矩阵 A, B 和输出矩阵 C
inline void {{kernel_name}}_amx_kernel_{{num_rows}}_{{num_columns}}(
    // AMX 状态结构体的引用
    AMXState& amx_state,
    // 输入矩阵 A，限定指针，存储类型为 input_t
    const {{input_t}}* __restrict__ A,
    // 输入矩阵 B，限定指针，存储类型为 input_t
    const {{input_t}}* __restrict__ B,
    // 输出矩阵 C，限定指针，存储类型为 output_t
    {{output_t}}* __restrict__ C,
    // 矩阵乘法的 K 维度大小
    int64_t K,
    // 矩阵 A 的列偏移
    int64_t lda,
    // 矩阵 B 的列偏移
    int64_t ldb,
    // 矩阵 C 的列偏移
    int64_t ldc,
    // AMX 瓦片配置行数
    uint8_t tilecfg_rows
) {
    // 添加 A, B, C 的预取提示
    // TODO(jgong5): add prefetch hint for A, B, C

    // 定义加载配置的 lambda 函数，配置 AMX 加速瓦片
    auto loadconfig = [](const amx_tilecfg& cfg) {
        _tile_loadconfig(&cfg);
    };

    // 计算最后一个 K 的偏移量和剩余的 K 大小
    const auto last_k_offset = K / {{block_k}} * {{block_k}};
    const auto tail_k_size = K - last_k_offset;

    // 根据最后一个 K 的偏移量来配置 AMX 状态
    if C10_LIKELY (last_k_offset > 0) {
        amx_state.configure(tilecfg_rows, 64, {{num_rows}} / 16, {{num_columns}}, loadconfig);
    } else {
        amx_state.configure(tilecfg_rows, tail_k_size * sizeof({{input_t}}), {{num_rows}} / 16, {{num_columns}}, loadconfig);
    }

    // 定义加载矩阵 C 的 lambda 函数
    auto load_c = [&]() {
        // 循环加载矩阵 C 的瓦片数据
        {%- for tile_row in range(num_rows // 16) %}
            {%- for tile_col in range(num_columns) %}
            {%- set tile_idx = tile_row * num_columns + tile_col %}
            _tile_loadd({{tile_idx}}, C + {{tile_row * 16}} * ldc + {{tile_col * 16}}, ldc * sizeof({{output_t}}));
            {%- endfor %}
        {%- endfor %}
    };

    // 定义将矩阵 C 置零的 lambda 函数
    auto zero_c = [&]() {
        // 循环将矩阵 C 的每个瓦片清零
        {%- for tile_row in range(num_rows // 16) %}
            {%- for tile_col in range(num_columns) %}
            {%- set tile_idx = tile_row * num_columns + tile_col %}
            _tile_zero({{tile_idx}});
            {%- endfor %}
        {%- endfor %}
    };

    // 根据累加标志选择加载矩阵 C 还是清零
    if constexpr (accum) {
        load_c();
    } else {
        zero_c();
    }

    // 定义计算矩阵乘法的 lambda 函数
    auto compute = [&](int k) {
        // 计算矩阵乘法的核心部分
        {%- set tile_offset_a = num_rows // 16 * num_columns %}
        {%- set tile_offset_b = tile_offset_a + num_rows // 16 %}
        {%- for tile_row in range(num_rows // 16) %}
            {%- for tile_col in range(num_columns) %}
            {%- set tile_idx_a = tile_offset_a + tile_row %}
            {%- set tile_idx_b = tile_offset_b + tile_col %}
            {%- set tile_idx_c = tile_row * num_columns + tile_col %}
            {%- if tile_col == 0 %}
            _tile_loadd({{tile_idx_a}}, A + {{tile_row * 16}} * lda + k, lda * sizeof({{input_t}}));
            {%- endif %}
            {%- if tile_row == 0 %}
            _tile_loadd({{tile_idx_b}}, B + k * ldb + {{tile_col * 16 * 2}}, ldb * 2 * sizeof({{input_t}}));
            {%- endif %}
            _tile_dpbf16ps({{tile_idx_c}}, {{tile_idx_a}}, {{tile_idx_b}});
            {%- endfor %}
        {%- endfor %}
    };

    // 根据 {{kernel.unroll_pragma(4)}} 指令展开循环，处理每个 k 块
    {{kernel.unroll_pragma(4)}}
    for (int k = 0; k < last_k_offset; k += {{block_k}}) {
        compute(k);
    }

    // 定义将结果存储回矩阵 C 的 lambda 函数
    auto store_c = [&]() {
        // 将计算结果存储回矩阵 C
        {%- for tile_row in range(num_rows // 16) %}
            {%- for tile_col in range(num_columns) %}
            {%- set tile_idx = tile_row * num_columns + tile_col %}
            _tile_stored({{tile_idx}}, C + {{tile_row * 16}} * ldc + {{tile_col * 16}}, ldc * sizeof({{output_t}}));
            {%- endfor %}
        {%- endfor %}
    };

    // TODO(jgong5): move tail k computation to separate loopnest to save tile configuration overhead
    // 将尾部 k 的计算移到单独的循环嵌套中以节省瓦片配置的开销
}
    # 如果 tail_k_size 大于 0，执行以下代码块
    if C10_UNLIKELY (tail_k_size > 0) {
        # 如果 last_k_offset 大于 0，执行以下代码块
        if C10_LIKELY (last_k_offset > 0) {
            # 存储 C 状态
            store_c();
            # 使用 AMX 状态配置，设置行数为 tilecfg_rows，每个元素大小为 {{input_t}} 的字节大小乘以 tail_k_size，行数为 {{num_rows}} 除以 16，列数为 {{num_columns}}，加载配置为 loadconfig
            amx_state.configure(tilecfg_rows, tail_k_size * sizeof({{input_t}}), {{num_rows}} / 16, {{num_columns}}, loadconfig);
            # 加载 C 状态
            load_c();
        }
        # 计算 last_k_offset
        compute(last_k_offset);
    }

    # 存储 C 状态
    store_c();
    def codegen_define(self, kernel: CppTemplateKernel) -> str:
        # 获取注册的块大小（m、n、k）
        block_m, block_n, block_k = self.register_blocking
        # 断言块大小 block_m 要求是 16 的倍数，适用于 AMX
        assert block_m % 16 == 0, "Only support block_m % 16 == 0 for AMX"
        # 断言块大小 block_n 要求是 16 的倍数，适用于 AMX
        assert block_n % 16 == 0, "Only support block_n % 16 == 0 for AMX"
        # 断言块大小 block_k 要求为 32，适用于 AMX
        assert block_k == 32, "Only support block_k = 32 for AMX"
        # 计算每个块中列数
        num_columns = block_n // 16
        # 准备代码生成的选项
        options = {
            "declare_kernel": self.get_kernel_declaration(),
            "kernel": kernel,
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "num_columns": num_columns,
            **self.get_common_options(),
        }
        # 初始化结果字符串
        result = ""
        # 从 block_m 开始向下循环，每次减少 16
        for num_rows in range(block_m, 0, -16):
            # 准备 AMX 内核的选项
            amx_kernel_options = {**options, "num_rows": num_rows}
            # 使用内核模板渲染 AMX 内核代码，并添加到结果中
            result += KernelTemplate._template_from_string(self.TEMPLATE_KERNEL).render(
                amx_kernel_options
            )
        # 使用入口模板渲染代码，并添加到结果中
        result += KernelTemplate._template_from_string(self.TEMPLATE_ENTRY).render(
            options
        )
        # 返回最终生成的代码字符串
        return result

    def codegen_init(
        self,
        kernel: CppTemplateKernel,
    ) -> str:
        # 初始化 AMX 状态
        return "AMXState amx_state;"

    def codegen_finalize(
        self,
        kernel: CppTemplateKernel,
    ) -> str:
        # 完成 AMX 操作后释放 AMX 状态
        return "amx_state.release([]() { _tile_release(); });"

    def get_kernel_extra_args_declare(self) -> str:
        # 返回声明内核额外参数的字符串
        return "AMXState& amx_state,"

    def get_kernel_extra_args(self) -> str:
        # 返回内核额外参数的字符串
        return "amx_state,"

    def get_b_layout(self):
        # 返回布局类型 VNNI2
        return LayoutType.VNNI2


def create_micro_gemm(
    name,
    m,
    n,
    k,
    input_dtype,
    output_dtype=None,
    compute_dtype=None,
    alpha=1,
    num_threads=-1,
    use_ref=True,
) -> Optional[CppMicroGemm]:
    def create_from_config(cls, config: CppMicroGemmConfig):
        # 根据配置创建微型 GEMM 实例
        return cls(
            name,
            config.input_dtype,
            config.output_dtype,
            config.compute_dtype,
            config.register_blocking,
            alpha,
        )

    # 断言 n 和 k 是整数或者可以转换为数字的类型
    assert isinstance(n, int) or n.is_number, n
    assert isinstance(k, int) or k.is_number, k
    # 如果 m 是符号表达式，则使用默认大小提示进行计算
    m = V.graph.sizevars.size_hint(m, fallback=1) if isinstance(m, sympy.Expr) else m
    # 断言 m 是整数
    assert isinstance(m, int), m
    # 如果未指定输出数据类型，则使用输入数据类型
    if output_dtype is None:
        output_dtype = input_dtype
    # 如果未指定计算数据类型，则使用输出数据类型
    if compute_dtype is None:
        compute_dtype = output_dtype
    # 如果线程数小于 0，则使用并行线程数
    if num_threads < 0:
        num_threads = parallel_num_threads()
    # 选择向量指令集
    vec_isa = pick_vec_isa()
    # 匹配的配置列表
    matched_configs = []
    for cls, configs in micro_gemm_configs.items():
        for config in configs:
            # 检查向量指令集是否符合配置要求
            if not issubclass(vec_isa.__class__, config.vec_isa_cls):
                continue
            # 检查配置的数据类型是否与输入输出和计算数据类型匹配
            if (
                config.input_dtype == input_dtype
                and config.output_dtype == output_dtype
                and config.compute_dtype == compute_dtype
            ):
                # 如果存在额外的检查函数，并且检查未通过，则跳过此配置
                if config.extra_check is not None and not config.extra_check(
                    config, m, n, k, alpha, num_threads
                ):
                    continue
                # 获取配置的寄存器阻塞大小
                block_m, block_n, block_k = config.register_blocking
                # 根据配置的准则对配置进行排名：
                # 1. ISA：AMX > VEC
                # 2. 可以被块大小 (block_m, block_n, block_k) 整除
                # 3. mxn 块的数量足够大以占据所有线程
                # 4. 寄存器块更大
                isa_score = 0
                if config.vec_isa_cls == VecAMX:
                    isa_score += 1
                dividable_score = 0
                if m % block_m == 0:
                    dividable_score += 1
                if n % block_n == 0:
                    dividable_score += 1
                if k % block_k == 0:
                    dividable_score += 1
                occupancy_score = 0
                n_blocks = (n + block_n - 1) // block_n
                total_mxn_blocks = n_blocks * ((m + block_m - 1) // block_m)
                if n_blocks >= num_threads:
                    occupancy_score += 1
                if total_mxn_blocks >= num_threads:
                    occupancy_score += 1
                # 计算寄存器的字节数
                register_bytes = (
                    block_m * block_n * config.compute_dtype.itemsize
                    + (block_m * block_k + block_k * block_n)
                    * config.input_dtype.itemsize
                )
                # 将匹配的配置加入到列表中
                matched_configs.append(
                    (
                        (isa_score, dividable_score, occupancy_score, register_bytes),
                        cls,
                        config,
                    )
                )
    # 如果没有匹配的配置，则根据 use_ref 返回参考的 CppMicroGemmRef 或 None
    if len(matched_configs) == 0:
        if use_ref:
            return CppMicroGemmRef(
                name, input_dtype, output_dtype, compute_dtype, alpha
            )
        else:
            return None
    # TODO(jgong5): 允许在配置选择上进行自动调优
    # 返回具有最高评分的配置创建的对象
    return create_from_config(*max(matched_configs, key=lambda x: x[0])[1:])
```