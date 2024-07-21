# `.\pytorch\aten\src\ATen\native\transformers\cuda\mem_eff_attention\kernels\generate_kernels.py`

```
# 版权声明及许可信息
#
# 此源代码使用 BSD 许可证授权，详见源代码根目录中的 LICENSE 文件。

# 生成内核组合 - 实现和注册表

# 内核按顺序排列（参见 `sort_index`），在调度时，我们选择支持输入的第一个内核

import argparse  # 导入命令行参数解析模块
import collections  # 导入集合类的模块
import itertools  # 导入迭代器模块
from dataclasses import dataclass, field  # 导入数据类和域模块
from pathlib import Path  # 导入路径类模块
from typing import Dict, List, Optional, Tuple, TypeVar  # 导入类型提示模块

DTYPES = {  # 数据类型映射字典
    "f32": "float",
    "f16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

SM = [50, 70, 75, 80, 100]  # Sm80 内核支持最高到 Sm100

KERNEL_IMPL_TEMPLATE = """__global__ void __launch_bounds__(
    {CPP_CLASS}::kNumThreads,
    {CPP_CLASS}::kMinBlocksPerSm)
{NAME}(typename {CPP_CLASS}::Params p) {{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= {SM}0
#if __CUDA_ARCH__ < {SM_MAX}0
  if (!p.advance_to_block()) {{  // 如果未能成功进入块
    return;  // 直接返回
  }}
  {CPP_CLASS}::attention_kernel(p);  // 调用指定的注意力内核函数
  return;  // 返回
#endif
#endif
    printf(  // 打印错误消息
        "FATAL: kernel `{NAME}` is for sm{SM}-sm{SM_MAX}, but was built for sm%d\\n",
        int(__CUDA_ARCH__ + 0) / 10);  // 输出错误信息，指出内核不支持当前计算能力
#endif
}}
"""

@dataclass(order=True)  # 使用 dataclass 装饰器创建排序数据类
class FwdKernel:
    sort_index: Tuple[int, ...] = field(init=False, repr=False)  # 不可初始化和显示在表示中
    aligned: bool  # 是否对齐的标志
    dtype: str  # 数据类型
    sm_range: Tuple[int, int]  # 支持的计算能力范围
    q: int  # 量化参数 q
    k: int  # 量化参数 k
    max_k: int  # 最大量化参数 k
    supports_dropout: bool = True  # 是否支持 dropout，默认为 True
    supports_bias: bool = True  # 是否支持偏置，默认为 True
    dispatch_cond: Optional[str] = None  # 调度条件，可选的字符串类型

    def __post_init__(self) -> None:
        # 设置内核选择优先级
        # 匹配输入的最低值将被选中
        self.sort_index = (
            # 首先选择对齐内核
            0 if self.aligned else 1,
            # 然后保持输出在寄存器文件中
            self.max_k,
            self.k,
            # 如果有可能，更喜欢没有 dropout/bias 的内核
            1 if self.supports_dropout else 0,
            1 if self.supports_bias else 0,
        )

    @property
    def _aligned_suffix(self) -> str:
        return "aligned" if self.aligned else "notaligned"  # 返回对齐标志的后缀字符串

    @property
    def name(self) -> str:
        acc = "rf" if self.max_k <= self.k else "gmem"  # 根据 k 和 max_k 决定是否在寄存器文件中保留输出
        return f"fmha_cutlassF_{self.dtype}_{self._aligned_suffix}_{self.q}x{self.k}_{acc}_sm{self.sm_range[0]}"

    @property
    def cpp_class(self) -> str:
        # 构造 C++ 类的模板参数字符串
        template_args = ", ".join(
            [
                DTYPES[self.dtype],
                f"cutlass::arch::Sm{self.sm_range[0]}",
                "true" if self.aligned else "false",
                str(self.q),
                str(self.k),
                str(self.max_k),
                "true" if self.supports_dropout else "false",
                "true" if self.supports_bias else "false",
            ]
        )
        return f"AttentionKernel<{template_args}>"
    # 返回一个字符串，表示实现组的文件名，包含数据类型和对齐后缀
    def impl_group(self) -> str:
        # Maps to file which will contain the implementation
        return f"{self.dtype}_{self._aligned_suffix}"

    # 返回一个字符串，表示用于 C++ 实现的模板，填充了类名、名称和 SM 范围等信息
    @property
    def cpp_impl(self) -> str:
        return KERNEL_IMPL_TEMPLATE.format(
            CPP_CLASS=self.cpp_class,
            NAME=self.name,
            SM=self.sm_range[0],
            SM_MAX=self.sm_range[1],
        )

    # 返回一个包含所有 FwdKernel 实例的列表
    @classmethod
    def get_all(cls) -> List["FwdKernel"]:
        # 初始化一个空列表，用于存储 FwdKernel 实例
        kernels: List[FwdKernel] = []
        # 使用 itertools.product 生成 aligned、dtype 和 (sm, sm_max) 的组合
        for aligned, dtype, (sm, sm_max) in itertools.product(
            [True, False], DTYPES.keys(), zip(SM, SM[1:])
        ):
            # 过滤掉不需要使用的内核
            if dtype == "bf16" and sm < 80:
                continue
            if not aligned and sm >= 80:
                continue
            # 使用不同的 q, k, max_k 组合创建 FwdKernel 实例，并添加到 kernels 列表中
            for q, k, max_k in [
                (64, 64, 64),
                # A100 上使用 64x128 可获得更好的性能
                (64 if sm > 75 else 32, 128, 128),
                (32, 128, 2**16),
            ]:
                kernels.append(
                    cls(
                        aligned=aligned,
                        dtype=dtype,
                        sm_range=(sm, sm_max),
                        q=q,
                        k=k,
                        max_k=max_k,
                    )
                )
        # 返回填充了 FwdKernel 实例的列表
        return kernels
# 使用 dataclass 装饰器创建一个具有排序功能的类 BwdKernel
@dataclass(order=True)
class BwdKernel:
    # Tuple，表示排序索引，不在 __repr__ 中显示
    sort_index: Tuple[int, ...] = field(init=False, repr=False)
    # Tuple，表示处理范围
    sm_range: Tuple[int, int]
    # 字符串，指定数据类型
    dtype: str
    # 布尔值，表示是否对齐
    aligned: bool
    # 布尔值，表示是否应用 dropout
    apply_dropout: bool
    # 布尔值，表示是否预加载 MMAs
    preload_mmas: bool
    # 整数，表示块的 i 维度大小
    block_i: int
    # 整数，表示块的 j 维度大小
    block_j: int
    # 整数，表示最大的 k 值
    max_k: int
    # 可选的条件字符串，用于调度
    dispatch_cond: Optional[str] = None
    # 布尔值，表示是否对齐 keys 和 queries 到块大小
    keys_queries_aligned_to_blocksizes: bool = False

    def __post_init__(self) -> None:
        # 初始化方法，设置核函数的选择优先级
        # 选择优先级由 sort_index 决定，根据属性设置顺序排序
        self.sort_index = (
            # 首先选择对齐的核函数
            0 if self.aligned else 1,
            # 如果可能的话，选择没有 dropout 的核函数
            1 if self.apply_dropout else 0,
            # 然后选择最小的 max_k
            self.max_k,
            # 接着选择最大的 block_i 的负值
            -self.block_i,
            # 最后避免边界检查，如果可能的话
            0 if self.keys_queries_aligned_to_blocksizes else 1,
        )

    @property
    def _aligned_suffix(self) -> str:
        # 根据 self.aligned 属性返回对齐或非对齐的后缀字符串
        return "aligned" if self.aligned else "notaligned"

    @property
    def name(self) -> str:
        # 根据类的属性生成一个名称字符串，用于标识 BwdKernel 的实例
        dropout_suffix = "_dropout" if self.apply_dropout else ""
        seqlen_aligned_suffix = (
            "_seqaligned" if self.keys_queries_aligned_to_blocksizes else ""
        )
        return (
            f"fmha_cutlassB_{self.dtype}_{self._aligned_suffix}"
            f"_{self.block_i}x{self.block_j}_k{self.max_k}{dropout_suffix}{seqlen_aligned_suffix}_sm{self.sm_range[0]}"
        )

    @property
    def cpp_class(self) -> str:
        # 返回一个描述 C++ 类模板参数的字符串
        template_args = ", ".join(
            [
                f"cutlass::arch::Sm{self.sm_range[0]}",
                DTYPES[self.dtype],
                "true" if self.aligned else "false",
                "true" if self.apply_dropout else "false",
                "true" if self.preload_mmas else "false",
                str(self.block_i),
                str(self.block_j),
                str(self.max_k),
            ]
        )
        if self.keys_queries_aligned_to_blocksizes:
            template_args += ", true"
        return f"AttentionBackwardKernel<{template_args}>"

    @property
    def impl_group(self) -> str:
        # 返回一个标识实现组的字符串，用于映射到包含实现的文件
        dropout_suffix = "_dropout" if self.apply_dropout else ""
        return f"{self.dtype}_{self._aligned_suffix}_k{self.max_k}{dropout_suffix}"

    @property
    def cpp_impl(self) -> str:
        # 返回一个格式化好的 C++ 实现模板，用于生成具体的 C++ 代码
        return KERNEL_IMPL_TEMPLATE.format(
            CPP_CLASS=self.cpp_class,
            NAME=self.name,
            SM=self.sm_range[0],
            SM_MAX=self.sm_range[1],
        )

    @classmethod
    # 定义一个类方法，用于获取所有后向传播核函数对象的列表
    def get_all(cls) -> List["BwdKernel"]:
        # 初始化空的核函数对象列表
        kernels: List[BwdKernel] = []
        # 使用 itertools.product 迭代生成不同参数组合
        for aligned, dtype, (sm, sm_max), apply_dropout, max_k in itertools.product(
            [True, False],  # 对齐标志的两种可能取值
            DTYPES.keys(),  # 遍历预定义数据类型的键
            zip(SM, SM[1:]),  # 遍历 SM 列表和其后一项的元组
            [True, False],  # 应用 dropout 的两种可能取值
            [32, 64, 128, 2**16],  # 最大 K 值的几种可能取值
        ):
            # 根据条件过滤不符合要求的参数组合
            if dtype == "bf16" and sm < 80:
                continue
            if not aligned and sm >= 80:
                continue
            # 检查是否是半精度数据类型
            is_half = dtype in ["bf16", "f16"]

            bi_values = [64]
            # 一些架构具有更多的共享内存，可以使用 128
            # 但对于共享内存较少的 GPU（如 Sm75、Sm86 ...），仍然需要回退到 64
            if sm >= 80 or (sm >= 70 and is_half):
                if max_k > 64:
                    bi_values.append(128)
            # 遍历 bi_values 列表
            for bi in bi_values:
                # 计算是否输出在寄存器文件中
                output_in_rf = is_half and max_k <= bi
                # 是否预加载 MMAs
                preload_mmas = is_half and sm >= 80 and output_in_rf
                # 根据条件设置 bj 的值
                bj = 128 if (preload_mmas and max_k > 64) else 64
                # 将新创建的核函数对象添加到 kernels 列表中
                kernels.append(
                    cls(
                        aligned=aligned,
                        dtype=dtype,
                        sm_range=(sm, sm_max),
                        apply_dropout=apply_dropout,
                        preload_mmas=preload_mmas,
                        block_i=bi,
                        block_j=bj,
                        max_k=max_k,
                    )
                )
                # 一些专门的核函数，执行速度更快
                if apply_dropout or max_k > 128 or not is_half or not aligned:
                    continue
                if sm not in [70, 80]:
                    continue
                # 将特定条件下的核函数对象添加到 kernels 列表中
                kernels.append(
                    cls(
                        aligned=aligned,
                        dtype=dtype,
                        sm_range=(sm, sm_max),
                        apply_dropout=apply_dropout,
                        preload_mmas=preload_mmas,
                        block_i=bi,
                        block_j=bj,
                        max_k=max_k,
                        keys_queries_aligned_to_blocksizes=True,
                    )
                )
        # 为稳定扩散 BW（K=80）添加一些专门的核函数
        # 这是唯一一个可以在 Sm86/Sm89 上将输出保持在 RF 上的核函数，比 64x64 的要快得多
        for dtype in ["f16", "bf16"]:
            # 将新创建的核函数对象添加到 kernels 列表中
            kernels.append(
                cls(
                    aligned=True,
                    dtype=dtype,
                    sm_range=(80, SM[SM.index(80) + 1]),
                    apply_dropout=False,
                    preload_mmas=True,
                    block_i=128,
                    block_j=64,
                    max_k=96,
                    # Sm80 的情况下有一个更快的核函数
                    dispatch_cond="cc == 86 || cc == 89",
                )
            )
        # 返回所有生成的核函数对象列表
        return kernels
# 定义一个类型变量 T，可以是 FwdKernel 或 BwdKernel 类型之一
T = TypeVar("T", FwdKernel, BwdKernel)

# 写入声明的实现函数，用于生成自动生成的 C++ 头文件和 CUDA 文件
def write_decl_impl(
    # kernels 参数是一个包含 FwdKernel 或 BwdKernel 对象的列表
    kernels: List[T],
    # family_name 是字符串，表示内核族列的名称
    family_name: str,
    # impl_file 是字符串，表示实现文件的路径
    impl_file: str,
    # autogen_dir 是 Path 对象，表示自动生成文件的目录路径
    autogen_dir: Path,
    # disable_def 是可选的字符串参数，默认为 None，用于禁用定义
    disable_def: str = None,
) -> None:
    # cpp_file_header 包含版权声明和文件自动生成的注释
    cpp_file_header = """/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
"""

    # 对 kernels 列表进行排序
    kernels.sort()

    # 创建两个字典，用于将实现文件映射到内核和将类别映射到内核
    implfile_to_kernels: Dict[str, List[T]] = collections.defaultdict(list)
    cat_to_kernels: Dict[Tuple[str, int, int], List[T]] = collections.defaultdict(list)

    # 初始化 dispatch_all 字符串
    dispatch_all = ""
    # 初始化声明字符串，包括版权声明和#pragma once
    declarations = cpp_file_header + "#pragma once\n"
    # declarations += f"#ifndef {disable_def}\n"
    declarations += f"""#include {impl_file}\n"""
    declarations += """using namespace PyTorchMemEffAttention;\n"""

    # 遍历 kernels 列表，将每个 kernel 对象添加到相应的字典中
    for k in kernels:
        implfile_to_kernels[k.impl_group].append(k)
        cat_to_kernels[(k.dtype, k.sm_range[0], k.sm_range[1])].append(k)

    # 遍历 cat_to_kernels 字典，生成每个类别的内核声明
    for (cat_dt, cat_sm, cat_sm_max), kernels in cat_to_kernels.items():
        declarations += f"// ======== {cat_dt} / sm{cat_sm} ========\n"
        declarations += "\n".join(
            k.cpp_impl.split("{")[0].rstrip() + ";" for k in kernels
        )
        dispatch_category_fn = f"dispatch_{family_name}_{cat_dt}_sm{cat_sm}"
        declarations += (
            f"\n\ntemplate <typename T> void {dispatch_category_fn}(T cb, int cc) {{\n"
        )
        # 为每个 kernel 生成调度函数的代码
        for k in kernels:
            _call = f"cb({k.cpp_class}(), {k.name});\n"
            if k.dispatch_cond is not None:
                _call = f"if ({k.dispatch_cond}) {_call}"
            declarations += f"    {_call}"
        declarations += "}\n\n"
        # 为 dispatch_all 添加条件语句，根据内核类别生成调度函数的调用
        dispatch_all += f"""
    if (std::is_same<DT, {DTYPES[cat_dt]}>::value && {cat_sm} <= cc && cc < {cat_sm_max}) {{
        {dispatch_category_fn}(cb, cc);
    }}"""

    # 生成 dispatch_{family_name} 的模板函数，根据 dispatch_all 字符串生成条件语句
    declarations += f"""
template <typename DT, typename T>
void dispatch_{family_name}(T cb, int cc = 0) {{
{dispatch_all}
}}
"""
    # declarations += f"#endif // {disable_def}\n"

    # 将声明写入到自动生成的头文件中
    (autogen_dir / f"{family_name}.h").write_text(declarations)

    # 将每个 implfile_to_kernels 字典中的内核写入到对应的实现文件中
    for f, f_kernels in implfile_to_kernels.items():
        impl_cu = cpp_file_header
        # impl_cu += f"#ifndef {disable_def}\n"
        impl_cu += f"""#include {impl_file}\n"""
        impl_cu += """using namespace PyTorchMemEffAttention;\n"""
        for k in f_kernels:
            impl_cu += k.cpp_impl
        # impl_cu += f"#endif // {disable_def}\n"
        # 将实现代码写入到自动生成的 CUDA 文件中
        (autogen_dir / f"{family_name}_{f}.cu").write_text(impl_cu)


# 主函数，根据输出目录路径生成自动生成的文件
def main(output_dir: Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)
    # 调用函数 `write_decl_impl`，生成前向传播（FwdKernel）相关的声明和实现
    write_decl_impl(
        FwdKernel.get_all(),  # 获取所有前向传播核心函数对象
        "cutlassF",            # 生成的前向传播声明和实现的标识符
        impl_file="<ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>",  # 指定实现文件路径
        autogen_dir=output_dir,  # 指定自动生成的目录
    )
    
    # 调用函数 `write_decl_impl`，生成反向传播（BwdKernel）相关的声明和实现
    write_decl_impl(
        BwdKernel.get_all(),    # 获取所有反向传播核心函数对象
        "cutlassB",              # 生成的反向传播声明和实现的标识符
        impl_file="<ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>",  # 指定实现文件路径
        autogen_dir=output_dir,  # 指定自动生成的目录
    )
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser(
        prog="generate_kernels",  # 程序的名称，显示在帮助消息中
        description="Generate the mem-eff kernels template instantiations",  # 程序的描述，显示在帮助消息中
    )

    # 添加一个可选的输出目录参数
    parser.add_argument(
        "-o",  # 参数的短标记
        "--output_dir",  # 参数的长标记
        required=False,  # 参数是否必需
        help="Where to generate the kernels "  # 参数的帮助信息，显示在帮助消息中
        " will default to <ATen/native/transformers/cuda/mem_eff_attention/kernels/> ",  # 帮助信息的延续
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数，并传入输出目录参数
    main(args.output_dir)
```