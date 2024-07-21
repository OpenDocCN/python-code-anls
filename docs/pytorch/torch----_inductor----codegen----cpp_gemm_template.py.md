# `.\pytorch\torch\_inductor\codegen\cpp_gemm_template.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型定义：Any, Callable, cast, List, Optional, Union
from typing import Any, Callable, cast, List, Optional, Union

# 引入 PyTorch 库
import torch
import torch.utils

# 引入本地模块
from ..._dynamo.utils import counters
from .. import ir, lowering as L

# 引入计算矩阵乘法所需的模块和函数
from ..kernel.mm_common import mm_args
from ..select_algorithm import DataProcessorTemplateWrapper
from ..utils import cache_on_self, has_free_symbols, parallel_num_threads
from ..virtualized import V

# 引入 C++ 实现的微内核矩阵乘法相关模块
from .cpp_micro_gemm import CppMicroGemmAMX, create_micro_gemm, LayoutType
from .cpp_template import CppTemplate

# 引入 C++ 模板和工具类
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import GemmBlocking

# 定义 GEMM 模板字符串
GEMM_TEMPLATE = r"""
{{template.header().getvalue()}}

{{micro_gemm.codegen_define(kernel)}}

extern "C"
{{kernel.def_kernel(inputs={"X": X, "W": W, "inp": inp}, outputs={"Y": Y}, aliases=buffer_aliases)}}
{
    {{kernel.maybe_codegen_profile()}}
    # 定义线程数
    constexpr int64_t num_threads = {{num_threads}};
    # 获取矩阵维度
    constexpr int64_t N = {{kernel.size(GemmOut, 1)}};
    constexpr int64_t K = {{kernel.size(X, 1)}};
    # 定义微内核的块大小
    constexpr int64_t M0 = {{micro_gemm.register_blocking.block_m}};
    constexpr int64_t N0 = {{micro_gemm.register_blocking.block_n}};
    constexpr int64_t K0 = {{micro_gemm.register_blocking.block_k}};
    # 计算分块数量
    constexpr int64_t N0_blocks = (N + N0 - 1) / N0;
    constexpr int64_t K0_blocks = (K + K0 - 1) / K0;

    # 断言 N 必须是 N0 的倍数
    static_assert(N % N0 == 0, "N dimension must be multiple of N0");

    // TODO(jgong5): improve cache blocking with CPU info (Mc, Kc)
    {%- if is_dynamic_M %}
    # 如果 M 是动态的，则动态计算 M 的分块数量
    const int64_t M = {{kernel.size(GemmOut, 0)}};
    const int64_t M0_blocks = (M + M0 - 1) / M0;
    {%- if num_threads > 1 %}
    # 如果有多个线程，调用函数计算每个线程的分块数量
    int64_t Mt_blocks, Nt_blocks, Kt_blocks;
    mm_get_thread_blocking(num_threads, M, N, K, M0, N0, K0, Mt_blocks, Nt_blocks, Kt_blocks);
    {%- else %}
    # 否则使用默认的分块数量
    const auto Mt_blocks = M0_blocks;
    const auto Nt_blocks = N0_blocks;
    const auto Kt_blocks = K0_blocks;
    {%- endif %}
    # 计算缓存分块数量
    const int64_t Mc_blocks = Mt_blocks;
    const int64_t Kc_blocks = Kt_blocks;
    {%- else %}
    # 如果 M 是静态的，则使用静态计算得到的分块数量
    constexpr int64_t M = {{kernel.size(GemmOut, 0)}};
    constexpr int64_t M0_blocks = (M + M0 - 1) / M0;
    constexpr int64_t Mt_blocks = {{template.thread_blocking().block_m}};
    constexpr int64_t Nt_blocks = {{template.thread_blocking().block_n}};
    constexpr int64_t Kt_blocks = {{template.thread_blocking().block_k}};
    constexpr int64_t Mc_blocks = {{template.cache_blocking().block_m}};
    constexpr int64_t Kc_blocks = {{template.cache_blocking().block_k}};
    {%- endif %}

    // TODO(jgong5): support k-slicing
    # 断言 Kt_blocks 与 K0_blocks 相等，目前不支持 k 切片
    {{kernel.assert_function}}(Kt_blocks == K0_blocks, "Do not support k slicing yet.");
    // 确保所有分区都被分配到
    {{kernel.assert_function}}(
        Mt_blocks * Nt_blocks * Kt_blocks * {{num_threads}} >= M0_blocks * N0_blocks * K0_blocks,
        "Not all partitions are assigned."
    );

    {%- if num_threads > 1 %}
    #pragma omp parallel num_threads({{num_threads}})
    {
        # 获取当前 OpenMP 线程的线程编号
        int tid = omp_get_thread_num();
        # 声明并初始化用于存储分块范围的变量
        int64_t m_block_start, m_block_end, n_block_start, n_block_end, k_block_start, k_block_end;
        # 调用函数 mm_get_thread_blocks 获取当前线程的分块范围
        mm_get_thread_blocks(
            tid, M0_blocks, N0_blocks, K0_blocks, Mt_blocks, Nt_blocks, Kt_blocks,
            m_block_start, m_block_end, n_block_start, n_block_end, k_block_start, k_block_end);
    {%- else %}
    {
        # 当不使用 OpenMP 时，设置默认的分块范围
        int64_t m_block_start = 0;
        int64_t m_block_end = M0_blocks;
        int64_t n_block_start = 0;
        int64_t n_block_end = N0_blocks;
        int64_t k_block_start = 0;
        int64_t k_block_end = K0_blocks;
    {%- endif %}
            {{ micro_gemm.codegen_init(kernel) }}
            for (int64_t mc = m_block_start; mc < m_block_end; mc += Mc_blocks) {
                const int64_t m_start = mc * M0;
                const int64_t m_end = std::min((mc + Mc_blocks) * M0, M);
                const int64_t m_size = m_end - m_start;
                {%- if use_local_acc %}
                {{ kernel.define_buffer(acc_buf_name, ["m_end - m_start", "N0"]) }}
                {%- endif %}
                for (int64_t nc = n_block_start; nc < n_block_end; ++nc) {
                    const int64_t n_start = nc * N0;
                    const int64_t n_size = N0;
                    {%- if use_local_acc %}
                    {%- set acc = kernel.local_buffers[acc_buf_name] %}
                    {%- else %}
                    {%- set acc = kernel.slice_nd(GemmOut, [("m_start", "m_end"), ("n_start", "n_start + N0")]) %}
                    {%- endif %}
                    {%- if inp is not none and beta != 0 %}
                    for (int64_t m = 0; m < m_size; ++m) {
                        // 使用 OpenMP SIMD 指令并行计算累加
                        #pragma omp simd
                        for (int64_t n = 0; n < n_size; ++n) {
                            // 计算加权输入值与累加器值
                            {{kernel.index(acc, ["m", "n"])}} = {{beta}} * {{kernel.index(inp, ["m + m_start", "n + n_start"])}};
                        }
                    }
                    {%- endif %}
                    for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                        int64_t k_start = kc * K0;
                        int64_t k_end = std::min((kc + Kc_blocks) * K0, K);
                        // 提取输入矩阵和权重矩阵的切片
                        {%- set tile_X = kernel.slice_nd(X, [("m_start", "m_end"), ("k_start", "k_end")]) %}
                        {%- set tile_W_3d = kernel.slice_nd(W, [("nc", "nc + 1"), ("k_start", "k_end"), ()]) %}
                        {%- set tile_W = kernel.view(tile_W_3d, ["k_end - k_start", micro_gemm.register_blocking.block_n]) %}
                        {%- if inp is not none and beta != 0 %}
                        // 调用微内核函数，累加到累加器
                        {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, acc, accum=True)|indent(20, false) }}
                        {%- else %}
                        // 根据是否是第一个块，选择进行累加或者直接计算
                        if (kc == k_block_start) {
                            {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, acc, accum=False)|indent(24, false) }}
                        } else {
                            {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, acc, accum=True)|indent(24, false) }}
                        }
                        {%- endif %}
                    }
                    // 将累加器的值写回输出矩阵的对应位置
                    {%- set tile_Y = kernel.slice_nd(Y_2d, [("m_start", "m_end"), ("n_start", "n_start + N0")]) %}
                    {{ kernel.store_output(
                          tile_Y, acc, GemmOut, epilogue_nodes, offsets=("m_start", "n_start"), reindexers=reindexers
                       )|indent(16, false)
                    }}
                }
            }
            // 完成代码生成后的收尾工作
            {{ micro_gemm.codegen_finalize(kernel) }}
        }
# 定义一个 CppPackedGemmTemplate 类，继承自 CppTemplate 类
class CppPackedGemmTemplate(CppTemplate):
    # 初始化方法，接受多个参数来配置矩阵乘法模板
    def __init__(
        self,
        input_nodes,  # 输入节点
        layout: ir.Layout,  # 布局对象，描述数据的排列方式
        num_threads: int,  # 使用的线程数
        register_blocking: GemmBlocking,  # 寄存器阻塞配置对象
        beta=1,  # beta 参数，默认为 1
        alpha=1,  # alpha 参数，默认为 1
        has_bias=False,  # 是否有偏置项，默认为 False
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,  # 后处理函数创建器，可选
    ):
        # 断言布局的数据类型在 [torch.float, torch.bfloat16, torch.half] 中
        assert layout.dtype in [torch.float, torch.bfloat16, torch.half]
        # 调用父类的初始化方法，传递名称 "packed_gemm"、输入节点、布局对象、线程数和后处理函数创建器
        super().__init__(
            "packed_gemm",
            input_nodes,
            layout,
            num_threads,
            epilogue_creator=epilogue_creator,
        )
        # 初始化对象的属性
        self.beta = beta  # 初始化 beta
        self.alpha = alpha  # 初始化 alpha
        self.has_bias = has_bias  # 初始化是否有偏置项的标志
        self.register_blocking = register_blocking  # 初始化寄存器阻塞配置对象
        m, n = layout.size  # 从布局对象获取矩阵的尺寸
        _, k = input_nodes[0].get_size()  # 从输入节点获取矩阵的尺寸
        self.m, self.n, self.k = m, n, k  # 初始化矩阵的尺寸属性
        self.is_dynamic_M = has_free_symbols((m,))  # 检查矩阵的尺寸是否动态

    # 缓存装饰器，将 thread_blocking 方法的结果缓存到对象的属性上
    @cache_on_self
    def thread_blocking(self) -> GemmBlocking:
        # TODO(jgong5): allow tuning various blocking options
        # 内部函数，获取给定数值的因子列表
        def get_factors(number):
            factors = []
            # 优先考虑更均匀分布的因子
            for i in range(int(number**0.5), 0, -1):
                if number % i == 0:
                    factors.append(number // i)
                    factors.append(i)
            return factors

        # 内部函数，根据给定参数计算阻塞配置对象
        def get_blocking(num_threads, factor, m_blocks, n_blocks, k_blocks):
            thread_block_n = (n_blocks + factor - 1) // factor
            cofactor = num_threads // factor
            thread_block_m = (m_blocks + cofactor - 1) // cofactor
            return GemmBlocking(thread_block_m, thread_block_n, k_blocks)

        # 断言不是动态矩阵尺寸，否则无法确定线程阻塞配置
        assert (
            not self.is_dynamic_M
        ), "Unable to determine thread blocking for dynamic M."
        
        # 获取寄存器阻塞配置对象
        register_blocking = self.register_blocking
        # 计算 m_blocks、n_blocks、k_blocks
        m_blocks = (self.m + register_blocking.block_m - 1) // register_blocking.block_m
        n_blocks = (self.n + register_blocking.block_n - 1) // register_blocking.block_n
        k_blocks = (self.k + register_blocking.block_k - 1) // register_blocking.block_k
        
        # 获取线程数的因子列表
        factors = get_factors(self.num_threads)
        assert len(factors) > 0
        
        # 遍历因子列表，找到合适的阻塞配置
        for factor in factors:
            if n_blocks % factor == 0 and m_blocks % (self.num_threads // factor) == 0:
                return get_blocking(
                    self.num_threads, factor, m_blocks, n_blocks, k_blocks
                )
        for factor in factors:
            if n_blocks % factor == 0:
                return get_blocking(
                    self.num_threads, factor, m_blocks, n_blocks, k_blocks
                )
            cofactor = self.num_threads // factor
            if m_blocks % cofactor == 0:
                return get_blocking(
                    self.num_threads, factor, m_blocks, n_blocks, k_blocks
                )
        
        # 如果找不到合适的阻塞配置，抛出断言错误
        raise AssertionError("Should not reach here.")

    # 缓存装饰器，将结果缓存到对象的属性上
    @cache_on_self
    # 定义一个实例方法，用于优化缓存块处理
    def cache_blocking(self) -> GemmBlocking:
        # TODO(jgong5): 使用 CPU 信息来改进缓存块处理
        # 如果 M 是动态的，则无法确定缓存块大小
        assert (
            not self.is_dynamic_M
        ), "Unable to determine cache blocking for dynamic M."
        # 调用 thread_blocking 方法获取线程块配置
        thread_blocking = self.thread_blocking()
        # 返回一个 GemmBlocking 对象，其中包含了 M 和 K 的块大小
        return GemmBlocking(thread_blocking.block_m, 1, thread_blocking.block_k)

    @staticmethod
    # 静态方法：用于向选择列表中添加选项
    def add_choices(
        choices,
        layout,
        input_nodes,
        beta=1,
        alpha=1,
        has_bias=False,
        trans_w=False,
        input_indices=None,
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,
    ):
        # 这个方法没有返回值，只是用来修改传入的选择列表

    # 方法重写声明：render 方法，用于生成 C++ 模板的代码
    def render(  # type: ignore[override]
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        **kwargs,
    ):
        # 这个方法负责将模板应用于代码生成过程中，生成 C++ 的模板代码
```