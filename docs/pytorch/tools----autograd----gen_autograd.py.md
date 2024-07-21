# `.\pytorch\tools\autograd\gen_autograd.py`

```
"""
To run this file by hand from the root of the PyTorch
repository, run:

python -m tools.autograd.gen_autograd \
       aten/src/ATen/native/native_functions.yaml \
       aten/src/ATen/native/tags.yaml \
       $OUTPUT_DIR \
       tools/autograd

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch/csrc/autograd/generated/
"""
# 从 PyTorch 仓库的根目录手动运行此文件时，执行以下命令：
#
# python -m tools.autograd.gen_autograd \
#        aten/src/ATen/native/native_functions.yaml \
#        aten/src/ATen/native/tags.yaml \
#        $OUTPUT_DIR \
#        tools/autograd
#
# 其中 $OUTPUT_DIR 是希望生成文件的目标目录。在完整的构建系统中，OUTPUT_DIR 是 torch/csrc/autograd/generated/

# gen_autograd.py 生成 C++ 自动微分函数和 Python 绑定。

# It delegates to the following scripts:
# 它委托以下脚本：

#  gen_autograd_functions.py: 生成 torch::autograd::Node 的子类
#  gen_variable_type.py: 生成包含所有张量方法的 VariableType.h
#  gen_python_functions.py: 生成 THPVariable 的 Python 绑定

from __future__ import annotations

import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统相关功能

from torchgen.api import cpp  # 导入 cpp 模块
from torchgen.api.autograd import (  # 导入自动微分相关模块
    match_differentiability_info,
    NativeFunctionWithDifferentiabilityInfo,
)
from torchgen.gen import parse_native_yaml  # 导入解析本地 YAML 文件的函数
from torchgen.selective_build.selector import SelectiveBuilder  # 导入选择性构建模块

from . import gen_python_functions  # 导入生成 Python 函数的模块
from .gen_autograd_functions import (  # 导入生成自动微分函数的模块
    gen_autograd_functions_lib,
    gen_autograd_functions_python,
)
from .gen_inplace_or_view_type import gen_inplace_or_view_type  # 导入生成就地操作或视图类型的模块
from .gen_trace_type import gen_trace_type  # 导入生成追踪类型的模块
from .gen_variable_factories import gen_variable_factories  # 导入生成变量工厂的模块
from .gen_variable_type import gen_variable_type  # 导入生成变量类型的模块
from .gen_view_funcs import gen_view_funcs  # 导入生成视图函数的模块
from .load_derivatives import load_derivatives  # 导入加载导数信息的模块


def gen_autograd(
    native_functions_path: str,
    tags_path: str,
    out: str,
    autograd_dir: str,
    operator_selector: SelectiveBuilder,
    disable_autograd: bool = False,
) -> None:
    # 解析并加载 derivatives.yaml
    differentiability_infos, used_dispatch_keys = load_derivatives(
        os.path.join(autograd_dir, "derivatives.yaml"), native_functions_path, tags_path
    )

    template_path = os.path.join(autograd_dir, "templates")

    native_funcs = parse_native_yaml(native_functions_path, tags_path).native_functions
    fns = sorted(
        filter(
            operator_selector.is_native_function_selected_for_training, native_funcs
        ),
        key=lambda f: cpp.name(f.func),
    )
    fns_with_diff_infos: list[
        NativeFunctionWithDifferentiabilityInfo
    ] = match_differentiability_info(fns, differentiability_infos)

    # 生成 VariableType.h/cpp
    if not disable_autograd:
        gen_variable_type(
            out,
            native_functions_path,
            tags_path,
            fns_with_diff_infos,
            template_path,
            used_dispatch_keys,
        )

        gen_inplace_or_view_type(
            out, native_functions_path, tags_path, fns_with_diff_infos, template_path
        )

        # 操作符过滤器未应用，因为追踪源在选择性构建中被排除
        gen_trace_type(out, native_funcs, template_path)
    # 生成 Functions.h/cpp
    # 调用函数生成自动微分函数库
    gen_autograd_functions_lib(out, differentiability_infos, template_path)
    
    # 调用函数生成 variable_factories.h 文件
    gen_variable_factories(out, native_functions_path, tags_path, template_path)
    
    # 调用函数生成 ViewFuncs.h/cpp 文件
    gen_view_funcs(out, fns_with_diff_infos, template_path)
# 生成自动求导 Python 代码的函数，根据给定路径加载导数信息并生成相关文件
def gen_autograd_python(
    native_functions_path: str,
    tags_path: str,
    out: str,
    autograd_dir: str,
) -> None:
    # 加载导数信息和未使用的返回值信息
    differentiability_infos, _ = load_derivatives(
        os.path.join(autograd_dir, "derivatives.yaml"), native_functions_path, tags_path
    )

    # 设置模板文件的路径
    template_path = os.path.join(autograd_dir, "templates")

    # 生成 Functions.h/cpp 文件
    gen_autograd_functions_python(out, differentiability_infos, template_path)

    # 生成 Python 绑定代码
    deprecated_path = os.path.join(autograd_dir, "deprecated.yaml")
    gen_python_functions.gen(
        out, native_functions_path, tags_path, deprecated_path, template_path
    )


# 主函数，用于解析命令行参数并调用生成自动求导 Python 代码的函数
def main() -> None:
    # 创建参数解析器，描述为“生成自动求导 C++ 文件的脚本”
    parser = argparse.ArgumentParser(description="Generate autograd C++ files script")
    
    # 添加位置参数：native_functions（路径到 native_functions.yaml 文件）
    parser.add_argument(
        "native_functions", metavar="NATIVE", help="path to native_functions.yaml"
    )
    
    # 添加位置参数：tags（路径到 tags.yaml 文件）
    parser.add_argument("tags", metavar="NATIVE", help="path to tags.yaml")
    
    # 添加位置参数：out（输出目录的路径）
    parser.add_argument("out", metavar="OUT", help="path to output directory")
    
    # 添加位置参数：autograd（自动求导目录的路径）
    parser.add_argument(
        "autograd", metavar="AUTOGRAD", help="path to autograd directory"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用生成自动求导 Python 代码的函数，传递解析后的参数和一个默认选择器
    gen_autograd(
        args.native_functions,
        args.tags,
        args.out,
        args.autograd,
        SelectiveBuilder.get_nop_selector(),
    )


# 如果当前脚本被直接运行，则调用主函数
if __name__ == "__main__":
    main()
```