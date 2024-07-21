# `.\pytorch\tools\jit\gen_unboxing.py`

```py
# Generates RegisterCodegenUnboxedKernels.cpp, UnboxingFunctions.h and UnboxingFunctions.cpp.
from __future__ import annotations  # 允许在注解中使用类型提示的未来语法

import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统功能模块
import sys  # 导入系统相关的功能模块
from dataclasses import dataclass  # 导入数据类装饰器
from pathlib import Path  # 导入处理文件路径的模块
from typing import Literal, Sequence, TYPE_CHECKING  # 导入类型提示相关的功能

import yaml  # 导入处理 YAML 格式的模块

from torchgen.api import cpp, unboxing  # 导入 torchgen 库中的 API：cpp 和 unboxing
from torchgen.api.translate import translate  # 导入 torchgen 库中的翻译功能
from torchgen.api.types import CppSignatureGroup  # 导入 torchgen 库中的类型定义
from torchgen.api.unboxing import convert_arguments  # 导入 torchgen 库中的参数转换功能
from torchgen.context import method_with_native_function  # 导入 torchgen 库中的本地函数上下文装饰器
from torchgen.gen import cpp_string, get_custom_build_selector, parse_native_yaml  # 导入 torchgen 库中的代码生成相关功能
from torchgen.model import Argument, NativeFunction, NativeFunctionsGroup, Variant  # 导入 torchgen 库中的模型定义
from torchgen.utils import FileManager, make_file_manager, mapMaybe, Target  # 导入 torchgen 库中的实用工具

if TYPE_CHECKING:
    from torchgen.selective_build.selector import SelectiveBuilder  # 在类型检查模式下，导入选择性构建器类


# Generates UnboxingFunctions.h & UnboxingFunctions.cpp.
@dataclass(frozen=True)
class ComputeUnboxingFunctions:
    target: Literal[Target.DECLARATION, Target.DEFINITION]  # 定义目标类型的字面量，可以是声明或定义
    selector: SelectiveBuilder  # 选择性构建器的实例

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        if not self.selector.is_root_operator(f"aten::{f.func.name}"):
            return ""  # 如果函数不是根运算符，则返回空字符串

        if self.target is Target.DECLARATION:
            # Note [The ATen Codegen Unboxing API]
            # Similar to the ATen Operators API, ATen Codegen Unboxing API lives in the at::unboxing namespace, and
            # will be used by codegen unboxing wrappers (CodegenUnboxingWrappers.cpp).
            # The Wrappers will be registered into torch::jit::OperatorRegistry using RegisterOperators API.
            #
            # Important characteristics about the Codegen Unboxing API:
            # (1) It follows the OperatorRegistry API.
            #     This is kind of necessary to avoid overhead.
            #     For example: if it followed the C++ API, then all of the faithful C++ factory functions
            #     would need to wrap their arguments into TensorOptions only to unwrap them again.
            # (2) Under the hood it calls C++ API.
            return f"""
// aten::{f.func}
TORCH_API void {f.func.name.unambiguous_name()}(Stack & stack);
"""
生成 RegisterCodegenUnboxedKernels.cpp 文件的函数

@dataclass(frozen=True)
# 定义 ComputeCodegenUnboxedKernels 类，包含 selector 属性
class ComputeCodegenUnboxedKernels:
    # 构造函数，接收一个 SelectiveBuilder 对象作为参数
    selector: SelectiveBuilder

    # 使用 method_with_native_function 装饰器
    @method_with_native_function
    else:
        # 使用 CppSignatureGroup.from_native_function 从原生函数生成签名组
        sig_group = CppSignatureGroup.from_native_function(
            f, method=(Variant.method in f.variants)
        )
        # 获取最符合预期的签名
        sig = sig_group.most_faithful_signature()
        # 将参数解析为 C++ 代码
        binding_list, code_list = convert_arguments(f)

        # 为每个 C++ 参数生成转换代码
        code_connector = "\n\t"
        arg_connector = ", "

        # 如果签名是方法，使用 self_base. 前缀，否则使用 at:: 前缀
        prefix = "self_base." if sig.method else "at::"

        # 翻译参数列表
        translated_args = translate(
            binding_list, sig.arguments(), method=sig.method
        )

        # 构建参数字符串
        args_str = f"{arg_connector.join(e.expr for e in translated_args)}"

        # 如果函数没有返回值
        if len(f.func.returns) == 0:
            ret_str = ""
            push_str = ""
        else:
            # 否则，设置结果自动变量
            ret_str = "auto result_ = "
            # 推送结果到堆栈
            push_str = """
    pack(stack, std::move(result_));
            """

        # 返回生成的 C++ 函数定义字符串
        return f"""
// aten::{f.func}
TORCH_API void {f.func.name.unambiguous_name()}(Stack & stack) {{
    {code_connector.join(code_list)}

    drop(stack, {len(binding_list)});

    {ret_str}{prefix}{sig.name()}({args_str});
    {push_str}
}}
"""
    # 定义 __call__ 方法，用于将 NativeFunction 转换为字符串表示
    def __call__(self, f: NativeFunction) -> str:
        # 如果函数不是根运算符，则返回空字符串
        if not self.selector.is_root_operator(f"aten::{f.func.name}"):
            return ""
        
        # 无条件生成函数包装器
        sig_group = CppSignatureGroup.from_native_function(f, method=False)
        
        # 获取最符合的函数签名
        sig = sig_group.most_faithful_signature()
        
        # 转义模式中的双引号，并去掉额外的双引号
        schema = cpp_string(str(sig.func))[1:-1]
        
        # 处理函数参数
        args = sig.arguments()
        connector = ",\n\t\t"
        args_code = []
        for arg in args:
            # 使用 method=False 的忠实 C++ API，因此不应出现 SelfArgument/TensorOptionsArgument
            assert isinstance(arg.argument, Argument)
            # 如果参数没有默认值，设置为 c10::IValue(c10::nullopt)
            if not arg.argument.default:
                arg_cpp = "c10::IValue(c10::nullopt)"
            else:
                # 解包代码使用忠实的 C++ API 来避免 TensorOptions 的包装/解包开销
                # 但是，我们希望在模式解析中包含默认参数
                # 默认参数只出现在非忠实的 C++ API 中，
                arg_default = cpp.default_expr(
                    arg.argument.default, arg.argument.type, symint=False
                )
                # 如果默认表达式以 '{' 开头，将其视为 c10::IntArrayRef
                if arg_default.startswith("{"):
                    arg_cpp = f"c10::IntArrayRef({arg_default})"
                else:
                    arg_cpp = f"c10::IValue({arg_default})"
            # 构建参数代码列表
            args_code.append(
                f"""c10::Argument("{arg.name}", nullptr, c10::nullopt, {arg_cpp})"""
            )
        
        # 处理返回值
        returns = f.func.returns
        returns_code = []
        for ret in returns:
            returns_code.append(f"""c10::Argument("{ret.name if ret.name else ""}")""")
        
        # 返回生成的 C++ 代码块
        return f"""
# 为给定的 ATen 操作生成代码，包括函数名和重载名称
OperatorGenerator(
    "aten::{f.func.name.name}",
    "{f.func.name.overload_name}",
    {{
        {connector.join(args_code)}
    }},
    {{
        {connector.join(returns_code)}
    }},
    # 记录函数调用，但不记录参数
    [](Stack & stack) {{
        RECORD_FUNCTION("{sig.name()}", std::vector<c10::IValue>());
        # 使用 ATen 的解包函数处理堆栈中的数据
        at::unboxing::{unboxing.name(f)}(stack);
    }},
    # 使用从模式中提取的别名分析
    aliasAnalysisFromSchema()
),
"""

def gen_unboxing(
    *,
    native_functions: Sequence[NativeFunction],
    cpu_fm: FileManager,
    selector: SelectiveBuilder,
) -> None:
    # 用于从函数或函数组获取键的函数
    def key_func(fn: NativeFunction | NativeFunctionsGroup) -> str:
        return fn.root_name

    # 选择的操作数量
    selected_op_num: int = len(selector.operators)
    # 启用分片的最佳实践阈值
    sharding_threshold: int = 100
    # 将生成的代码写入到单个文件中，根据函数进行分组
    cpu_fm.write_sharded(
        "UnboxingFunctions.cpp",
        native_functions,
        key_fn=key_func,
        env_callable=lambda fn: {
            "definitions": [ComputeUnboxingFunctions(Target.DEFINITION, selector)(fn)]
        },
        # 根据选择的操作数量确定分片数量
        num_shards=1 if selected_op_num < sharding_threshold else 5,
        sharded_keys={"definitions"},
    )
    # 将声明写入到头文件中，根据选择的函数映射而来
    cpu_fm.write(
        "UnboxingFunctions.h",
        lambda: {
            "declarations": list(
                mapMaybe(
                    ComputeUnboxingFunctions(Target.DECLARATION, selector),
                    native_functions,
                )
            ),
        },
    )
    # 将无盒内核注册代码写入到多个分片中
    cpu_fm.write_sharded(
        "RegisterCodegenUnboxedKernels.cpp",
        native_functions,
        key_fn=key_func,
        env_callable=lambda fn: {
            "unboxed_ops": [ComputeCodegenUnboxedKernels(selector)(fn)]
        },
        # 根据选择的操作数量确定分片数量
        num_shards=1 if selected_op_num < sharding_threshold else 10,
        sharded_keys={"unboxed_ops"},
    )


def main(args: list[str]) -> None:
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Generate unboxing source files")
    # 添加路径参数：源代码目录路径
    parser.add_argument(
        "-s",
        "--source-path",
        help="path to source directory for ATen",
        default="aten/src/ATen",
    )
    # 添加路径参数：安装目录路径
    parser.add_argument(
        "-d",
        "--install-dir",
        "--install_dir",
        help="output directory",
        default="build/aten/src/ATen",
    )
    # 添加可选参数：输出依赖项列表到指定文件
    parser.add_argument(
        "-o",
        "--output-dependencies",
        help="output a list of dependencies into the given file and exit",
    )
    # 添加标志参数：仅运行而不写入任何文件（仍然更新输出）
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run without writing any files (still updates outputs)",
    )
    parser.add_argument(
        "--op-selection-yaml-path",
        "--op_selection_yaml_path",
        help="Provide a path to the operator selection (for custom build) YAML "
        "that contains the information about the set of selected operators "
        "and their categories (training, ...). Each operator is either a "
        "full operator name with overload or just a bare operator name. "
        "The operator names also contain the namespace prefix (e.g. aten::)",
    )
    parser.add_argument(
        "--op-registration-allowlist",
        "--op_registration_allowlist",
        nargs="*",
        help="filter op registrations by the allowlist (if set); "
        "each item is `namespace`::`operator name` without overload name; "
        "e.g.: aten::empty aten::conv2d ...",
    )
    parser.add_argument(
        "--TEST-ONLY-op-registration-allowlist-yaml-path",
        "--TEST_ONLY_op_registration_allowlist_yaml_path",
        help="Provide a path to the operator selection (for custom build) YAML "
        "which contains a list of operators. It is to serve testing purpose and "
        "each item is `namespace`::`operator name` without overload name; "
        "e.g.: aten::empty aten::conv2d ...",
    )


    # 添加参数定义：用于指定操作符选择的 YAML 文件路径，这个 YAML 文件包含了所选操作符集合及其类别信息（如训练等）。
    # 每个操作符可以是完整的操作符名称（包括重载）或者只是操作符名称。
    # 操作符名称还包含命名空间前缀（例如 aten::）。

    # 添加参数定义：用于根据允许列表（如果设置了）过滤操作符注册。
    # 每个条目的格式为 `namespace`::`operator name`，不包括重载名称，例如 aten::empty aten::conv2d ...

    # 添加参数定义：用于指定测试专用的操作符注册允许列表的 YAML 文件路径。
    # 此 YAML 文件包含操作符列表，用于测试目的。
    # 每个条目的格式为 `namespace`::`operator name`，不包括重载名称，例如 aten::empty aten::conv2d ...


    options = parser.parse_args(args)


    # 解析命令行参数，并存储在 options 对象中。


    if options.op_registration_allowlist:
        op_registration_allowlist = options.op_registration_allowlist
    elif options.TEST_ONLY_op_registration_allowlist_yaml_path:
        with open(options.TEST_ONLY_op_registration_allowlist_yaml_path) as f:
            op_registration_allowlist = yaml.safe_load(f)
    else:
        op_registration_allowlist = None


    # 根据命令行参数设置操作符注册允许列表。
    # 如果指定了 op_registration_allowlist 参数，则直接使用命令行提供的列表。
    # 否则，如果指定了 TEST_ONLY_op_registration_allowlist_yaml_path 参数，则从对应的 YAML 文件中加载列表。
    # 如果都没有指定，则将 op_registration_allowlist 设置为 None。


    selector = get_custom_build_selector(
        op_registration_allowlist,
        options.op_selection_yaml_path,
    )


    # 调用 get_custom_build_selector 函数，传入操作符注册允许列表和操作符选择 YAML 文件路径作为参数，获取选择器对象。


    native_yaml_path = os.path.join(options.source_path, "native/native_functions.yaml")
    tags_yaml_path = os.path.join(options.source_path, "native/tags.yaml")
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )


    # 构建 native_yaml_path 和 tags_yaml_path 变量，分别指向 native_functions.yaml 和 tags.yaml 的路径。
    # 调用 parse_native_yaml 函数，解析这两个 YAML 文件，返回一个解析后的 YAML 对象 parsed_yaml。
    # 从 parsed_yaml 对象中获取 native_functions 和 backend_indices 两个属性。


    cpu_fm = make_file_manager(options=options)
    gen_unboxing(native_functions=native_functions, cpu_fm=cpu_fm, selector=selector)


    # 调用 make_file_manager 函数，传入 options 参数，创建一个文件管理器对象 cpu_fm。
    # 调用 gen_unboxing 函数，传入 native_functions、cpu_fm 和 selector 作为参数，执行一些未指定的操作。


    if options.output_dependencies:
        depfile_path = Path(options.output_dependencies).resolve()
        depfile_name = depfile_path.name
        depfile_stem = depfile_path.stem

        path = depfile_path.parent / depfile_name
        cpu_fm.write_outputs(depfile_stem, str(path))


    # 如果指定了 options.output_dependencies 参数，则执行以下操作：
    # 解析 options.output_dependencies 路径并获取其父路径及文件名。
    # 将文件名与父路径合并，构成完整路径。
    # 调用 cpu_fm 对象的 write_outputs 方法，将 depfile_stem 和完整路径作为参数，执行输出操作。
# 如果当前脚本被直接执行（而不是被导入到其他模块中执行），则执行以下代码块
if __name__ == "__main__":
    # 调用 main 函数，传递命令行参数列表（去掉第一个参数，即当前脚本文件名）
    main(sys.argv[1:])
```