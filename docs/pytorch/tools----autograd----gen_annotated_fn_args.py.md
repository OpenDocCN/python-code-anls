# `.\pytorch\tools\autograd\gen_annotated_fn_args.py`

```py
"""
For procedural tests needed for __torch_function__, we use this function
to export method names and signatures as needed by the tests in
test/test_overrides.py.

python -m tools.autograd.gen_annotated_fn_args \
       aten/src/ATen/native/native_functions.yaml \
       aten/src/ATen/native/tags.yaml \
       $OUTPUT_DIR \
       tools/autograd

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch/testing/_internal/generated
"""

# 导入必要的模块和函数
from __future__ import annotations

import argparse  # 用于解析命令行参数
import os  # 提供与操作系统交互的功能
import textwrap  # 用于格式化文本块，包括缩进
from collections import defaultdict  # 默认字典，支持设置默认值的字典
from typing import Any, Sequence, TYPE_CHECKING  # 引入类型提示

import torchgen.api.python as python  # 导入torchgen库中的Python API
from torchgen.context import with_native_function  # 导入装饰器函数
from torchgen.gen import parse_native_yaml  # 导入解析本地YAML文件的函数
from torchgen.utils import FileManager  # 导入文件管理器类

from .gen_python_functions import (  # 导入本地的函数判断和生成模块
    is_py_fft_function,
    is_py_linalg_function,
    is_py_nn_function,
    is_py_special_function,
    is_py_torch_function,
    is_py_variable_method,
    should_generate_py_binding,
)

# 如果处于类型检查模式，导入特定类型
if TYPE_CHECKING:
    from torchgen.model import Argument, BaseOperatorName, NativeFunction


# 定义生成带注释函数的主函数
def gen_annotated(
    native_yaml_path: str, tags_yaml_path: str, out: str, autograd_dir: str
) -> None:
    # 解析本地YAML文件获取原生函数信息
    native_functions = parse_native_yaml(
        native_yaml_path, tags_yaml_path
    ).native_functions

    # 定义函数判断和命名空间的映射关系
    mappings = (
        (is_py_torch_function, "torch._C._VariableFunctions"),
        (is_py_nn_function, "torch._C._nn"),
        (is_py_linalg_function, "torch._C._linalg"),
        (is_py_special_function, "torch._C._special"),
        (is_py_fft_function, "torch._C._fft"),
        (is_py_variable_method, "torch.Tensor"),
    )

    # 初始化带注释的参数列表
    annotated_args: list[str] = []

    # 遍历映射关系
    for pred, namespace in mappings:
        # 使用默认字典存储按功能分组的本地函数
        groups: dict[BaseOperatorName, list[NativeFunction]] = defaultdict(list)
        for f in native_functions:
            # 如果不应生成Python绑定或者函数不符合当前预测条件，则跳过
            if not should_generate_py_binding(f) or not pred(f):
                continue
            groups[f.func.name.name].append(f)
        # 将分组后的函数列表添加到带注释的参数列表中
        for group in groups.values():
            for f in group:
                annotated_args.append(f"{namespace}.{gen_annotated_args(f)}")

    # 定义模板路径
    template_path = os.path.join(autograd_dir, "templates")
    # 初始化文件管理器实例
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    # 使用模板生成带注释函数的文件
    fm.write_with_template(
        "annotated_fn_args.py",
        "annotated_fn_args.py.in",
        lambda: {
            "annotated_args": textwrap.indent("\n".join(annotated_args), "    "),
        },
    )


# 装饰器函数，用于处理本地函数
@with_native_function
def gen_annotated_args(f: NativeFunction) -> str:
    # 定义排除关键字参数的函数列表
    def _get_kwargs_func_exclusion_list() -> list[str]:
        return [
            "diagonal",
            "round_",
            "round",
            "scatter_",
        ]

    def _add_out_arg(
        out_args: list[dict[str, Any]], args: Sequence[Argument], *, is_kwarg_only: bool
    ):
        # 此处省略函数的具体实现
        pass
    ) -> None:
        # 遍历参数列表
        for arg in args:
            # 如果参数有默认值，则跳过
            if arg.default is not None:
                continue
            # 初始化输出参数字典
            out_arg: dict[str, Any] = {}
            # 设置关键字参数标志
            out_arg["is_kwarg_only"] = str(is_kwarg_only)
            # 记录参数名
            out_arg["name"] = arg.name
            # 获取参数的简单类型描述
            out_arg["simple_type"] = python.argument_type_str(
                arg.type, simple_type=True
            )
            # 获取参数的大小信息
            size_t = python.argument_type_size(arg.type)
            if size_t:
                out_arg["size"] = size_t
            # 将处理后的参数字典添加到输出参数列表
            out_args.append(out_arg)

    # 初始化输出参数列表
    out_args: list[dict[str, Any]] = []
    # 处理位置参数
    _add_out_arg(out_args, f.func.arguments.flat_positional, is_kwarg_only=False)
    # 如果函数名不在排除列表中，则处理关键字参数
    if f"{f.func.name.name}" not in _get_kwargs_func_exclusion_list():
        _add_out_arg(out_args, f.func.arguments.flat_kwarg_only, is_kwarg_only=True)

    # 返回格式化后的字符串，包含函数名和处理后的参数列表的表示
    return f"{f.func.name.name}: {repr(out_args)},"
# 定义程序的主函数
def main() -> None:
    # 创建参数解析器对象，并设置程序描述信息
    parser = argparse.ArgumentParser(description="Generate annotated_fn_args script")
    
    # 添加必需的位置参数定义，用于指定输入的文件路径
    parser.add_argument(
        "native_functions", metavar="NATIVE", help="path to native_functions.yaml"
    )
    parser.add_argument("tags", metavar="TAGS", help="path to tags.yaml")
    parser.add_argument("out", metavar="OUT", help="path to output directory")
    parser.add_argument(
        "autograd", metavar="AUTOGRAD", help="path to template directory"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用 gen_annotated 函数，传入解析后的参数对象，执行生成带注解的函数参数脚本的操作
    gen_annotated(args.native_functions, args.tags, args.out, args.autograd)


# 当该脚本直接运行时，执行主函数
if __name__ == "__main__":
    main()
```