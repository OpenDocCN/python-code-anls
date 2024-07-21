# `.\pytorch\torchgen\static_runtime\gen_static_runtime_ops.py`

```py
# 从未来版本导入注解支持
from __future__ import annotations

# 导入命令行参数解析模块
import argparse
# 导入迭代工具模块
import itertools
# 导入操作系统接口模块
import os
# 导入类型提示模块
from typing import Sequence, TypeVar, Union

# 导入日志设置函数（类型: 忽略导入错误）
from libfb.py.log import set_simple_logging  # type: ignore[import]

# 导入 torchgen 中的生成器和上下文管理器
from torchgen import gen
from torchgen.context import native_function_manager
# 导入 torchgen 中的模型相关内容：调度键、本地函数组、本地函数视图组
from torchgen.model import DispatchKey, NativeFunctionsGroup, NativeFunctionsViewGroup
# 导入 torchgen 中的静态运行时配置和生成器
from torchgen.static_runtime import config, generator


# 定义类型变量 NativeGroupT，可以是 NativeFunctionsGroup 或 NativeFunctionsViewGroup 的子类
NativeGroupT = TypeVar(
    "NativeGroupT",
    bound=Union[NativeFunctionsGroup, NativeFunctionsViewGroup],
)


# 根据操作名对分组的本地函数列表进行分组，返回分组后的列表
def group_functions_by_op_name(
    grouped_native_functions: Sequence[NativeGroupT],
) -> Sequence[Sequence[NativeGroupT]]:
    # 如果给定的本地函数列表为空，则返回空列表
    if not grouped_native_functions:
        return []
    # 初始化一个空列表来存储分组后的结果
    groups = []

    # 定义一个函数判断本地函数组是否支持
    def is_supported(g: NativeFunctionsGroup | NativeFunctionsViewGroup) -> bool:
        # 使用本地函数管理器进行上下文管理，并判断是否支持生成
        with native_function_manager(g):
            return generator.is_supported(g)

    # 筛选出所有支持生成的本地函数组
    eligible_ops = (g for g in grouped_native_functions if is_supported(g))
    # 使用 itertools.groupby 函数对支持生成的本地函数组进行按操作名分组
    groups = [
        list(group)
        for k, group in (
            itertools.groupby(
                eligible_ops,
                key=config.func_name_base_str,
            )
        )
    ]

    # 返回按操作名分组后的本地函数组列表
    return groups


# 调用 clang-format 命令对给定的 C++ 文件路径进行格式化
def clang_format(cpp_file_path: str) -> None:
    # 导入子进程模块
    import subprocess

    # 调用 clang-format 命令对指定的 C++ 文件进行格式化，修改文件内容
    subprocess.check_call(["clang-format", "-i", cpp_file_path])


# 将生成的 C++ 操作代码写入到指定路径的文件中
def write_cpp(cpp_ops: Sequence[str], file_path: str) -> None:
    # 将所有 C++ 操作代码连接成一个字符串，每行之间用换行符分隔
    code = "\n".join(cpp_ops)
    # 生成 C++ 文件的头部注释和包含的库文件，使用生成器注释自动生成的信息
    generated = f"""// @lint-ignore-every CLANGTIDY HOWTOEVEN
// AUTO-GENERATED FROM: torchgen/static_runtime/gen_static_runtime_ops.py
#include <torch/csrc/jit/runtime/static/ops.h>

#include <ATen/CPUFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/EmbeddingBag.h>
#include <ATen/native/Fill.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/cpu/SerialStackImpl.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qembeddingbag.h>
#include <ATen/native/quantized/cpu/qembeddingbag_prepack.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/ScalarType.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/te_wrapper.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <torch/csrc/jit/tensorexpr/ir.h>

"""

# 输出代码块需包含完整的函数定义及注释
# 导入必要的库文件
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>

# 定义命名空间 torch::jit
namespace torch {
namespace jit {

# 定义函数 write_generated_cpp，用于生成并格式化 C++ 代码文件
def write_generated_cpp(generated_code: str, file_path: str) -> None:
    # 将生成的代码写入指定路径的文件中
    with open(file_path, "w") as f:
        f.write(generated_code)
    # 调用 clang_format 函数格式化生成的代码文件
    clang_format(file_path)

# 定义函数 write_test_cpp，用于生成测试用例的 C++ 代码文件
def write_test_cpp(cpp_ops: Sequence[str], file_path: str) -> None:
    # 将给定的 C++ 代码操作序列连接成一个字符串
    code = "\n".join(cpp_ops)
    # 生成头部注释和包含的声明
    generated = f"""// @lint-ignore-every CLANGTIDY HOWTOEVEN
// AUTO-GENERATED FROM: torchgen/static_runtime/gen_static_runtime_ops.py
#include <gtest/gtest.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/torch.h>

#include "test_utils.h"

using namespace caffe2;
using namespace torch;
using namespace torch::jit;
using namespace torch::jit::test;
using c10::IValue;

{code}

"""
    # 将生成的测试代码写入指定路径的文件中
    with open(file_path, "w") as f:
        f.write(generated)

    # 调用 clang_format 函数格式化生成的测试代码文件
    clang_format(file_path)

# 定义主函数 main，用于生成 ATen 源文件
def main() -> None:
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Generate ATen source files")
    
    # 添加命令行参数选项
    parser.add_argument(
        "-s",
        "--source-path",
        help="path to source directory for ATen",
        default="caffe2/aten/src/ATen",
    )
    parser.add_argument(
        "-p",
        "--generated-ops-cpp-path",
        help="path to directory to generate op dispatcher .cpp file",
        default="caffe2/torch/csrc/jit/runtime/static/generated_ops.cpp",
    )
    parser.add_argument(
        "-t",
        "--generated-ops-test-cpp-path",
        help="path to directory to generate op dispatcher .cpp file",
        default="caffe2/benchmarks/static_runtime/test_generated_ops.cc",
    )
    
    # 解析命令行参数
    options = parser.parse_args()
    
    # 构建 native_yaml_path 和 tags_yaml_path
    native_yaml_path = os.path.join(options.source_path, "native/native_functions.yaml")
    tags_yaml_path = os.path.join(options.source_path, "native/tags.yaml")
    
    # 解析 native_functions.yaml 和 tags.yaml 文件，生成解析结果
    parsed_yaml = gen.parse_native_yaml(native_yaml_path, tags_yaml_path)
    
    # 提取 native_functions 和 backend_indices
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )
    
    # 创建操作分发器和测试用例生成器的实例
    op_generator = generator.GenOpDispatcher()
    test_case_generator = generator.GenOpTestCase()
    
    # 过滤出 NativeFunctionsGroup 类型的本地函数分组
    native_functions_groups = [
        g
        for g in gen.get_grouped_native_functions(native_functions)
        if isinstance(g, NativeFunctionsGroup)
    ]
    
    # 将支持的函数按操作名称分组
    supported_functions_groups = group_functions_by_op_name(native_functions_groups)
    
    # 生成输出变体操作结果列表
    out_variant_op_result = [
        op_generator.out_variant(groups, backend_indices[DispatchKey.CPU])
        for groups in supported_functions_groups
    ]
    
    # 生成输出变体测试结果列表
    out_variant_test_result = [
        test_case_generator.out_variant(groups) for groups in supported_functions_groups
    ]
    
    # 过滤出 NativeFunctionsViewGroup 类型的本地函数视图分组
    native_functions_view_groups = [
        g
        for g in gen.get_grouped_by_view_native_functions(native_functions)
        if isinstance(g, NativeFunctionsViewGroup)
    ]
    
    # 将支持的视图函数按操作名称分组
    supported_functions_view_groups = group_functions_by_op_name(
        native_functions_view_groups
    )
    # 生成包含操作结果视图的列表，使用 op_generator.view 方法处理每个 groups 中的数据，
    # 这些数据由 backend_indices[DispatchKey.CPU] 索引指定。
    view_op_result = [
        op_generator.view(groups, backend_indices[DispatchKey.CPU])
        for groups in supported_functions_view_groups
    ]
    
    # 生成包含测试用例结果视图的列表，使用 test_case_generator.view 方法处理每个 groups 中的数据。
    view_test_result = [
        test_case_generator.view(groups) for groups in supported_functions_view_groups
    ]

    # 将输出变体操作结果、换行符和操作视图结果合并为一个列表。
    op_result = out_variant_op_result + ["\n\n"] + view_op_result
    
    # 将输出变体测试结果、换行符和测试视图结果合并为一个列表。
    test_result = out_variant_test_result + ["\n\n"] + view_test_result

    # 调用 write_cpp 函数将操作结果列表写入指定的生成操作的 C++ 文件路径。
    write_cpp(op_result, options.generated_ops_cpp_path)
    
    # 调用 write_test_cpp 函数将测试结果列表写入指定的生成测试的 C++ 文件路径。
    write_test_cpp(test_result, options.generated_ops_test_cpp_path)

    # 打印输出：总共分组的本地操作数量，通过 gen.get_grouped_native_functions 函数获取。
    print(
        "\ntotal grouped native ops: %d"
        % len(gen.get_grouped_native_functions(native_functions))
    )

    # 打印输出：具有输出变体的分组本地操作数量。
    print("grouped native ops with out variant: %d" % len(native_functions_groups))
    
    # 计算并打印输出：不带输出变体生成的函数分组数量。
    supported_functions_num = sum(len(groups) for groups in supported_functions_groups)
    print("generated functions groups with out variant: %d" % supported_functions_num)

    # 打印输出：视图分组的本地操作数量。
    print("\nview grouped native ops: %d" % len(native_functions_view_groups))
    
    # 计算并打印输出：生成的视图函数分组数量。
    supported_view_functions_num = sum(
        len(groups) for groups in supported_functions_view_groups
    )
    print("generated functions view groups: %d" % supported_view_functions_num)

    # 打印输出：总体生成的函数数量，包括带输出变体和视图的。
    print(
        "\noverall generated : %d"
        % (supported_functions_num + supported_view_functions_num)
    )
# 如果当前模块被直接执行（而不是被导入），则执行以下代码
if __name__ == "__main__":
    # 调用一个函数来设置简单的日志记录，禁用对换行符的转义
    set_simple_logging(escape_newlines=False)
    # 调用主函数执行程序的主要逻辑
    main()
```