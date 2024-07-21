# `.\pytorch\tools\setup_helpers\generate_code.py`

```
from __future__ import annotations  # 导入未来的注解语法支持

import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统功能模块
import sys  # 导入系统相关模块
from pathlib import Path  # 导入处理路径的模块
from typing import Any, cast  # 导入类型提示相关模块

import yaml  # 导入YAML文件解析模块

try:
    # 如果可用，使用更快的C加载器
    from yaml import CSafeLoader as YamlLoader
except ImportError:
    from yaml import SafeLoader as YamlLoader  # 如果不可用，使用Python加载器

NATIVE_FUNCTIONS_PATH = "aten/src/ATen/native/native_functions.yaml"  # 设置原生函数定义路径常量
TAGS_PATH = "aten/src/ATen/native/tags.yaml"  # 设置标签定义路径常量


def generate_code(
    gen_dir: Path,  # 生成代码的目录路径
    native_functions_path: str | None = None,  # 原生函数定义路径或None
    tags_path: str | None = None,  # 标签定义路径或None
    install_dir: str | None = None,  # 安装目录路径或None
    subset: str | None = None,  # 子集名称或None
    disable_autograd: bool = False,  # 是否禁用自动微分，默认为False
    force_schema_registration: bool = False,  # 是否强制模式注册，默认为False
    operator_selector: Any = None,  # 操作选择器，可以是任何类型，默认为None
) -> None:
    from tools.autograd.gen_annotated_fn_args import gen_annotated  # 导入生成带注释函数参数的工具函数
    from tools.autograd.gen_autograd import gen_autograd, gen_autograd_python  # 导入生成自动微分相关的工具函数
    from torchgen.selective_build.selector import SelectiveBuilder  # 导入选择构建器

    # 构建基于ATen的变量类
    if install_dir is None:
        install_dir = os.fspath(gen_dir / "torch/csrc")  # 如果安装目录为空，使用默认生成目录
        python_install_dir = os.fspath(gen_dir / "torch/testing/_internal/generated")  # Python安装目录为测试生成目录
    else:
        python_install_dir = install_dir
    autograd_gen_dir = os.path.join(install_dir, "autograd", "generated")  # 自动微分生成目录
    for d in (autograd_gen_dir, python_install_dir):
        os.makedirs(d, exist_ok=True)  # 创建目录，如果存在则忽略
    autograd_dir = os.fspath(Path(__file__).parent.parent / "autograd")  # 自动微分目录的路径

    if subset == "pybindings" or not subset:
        gen_autograd_python(
            native_functions_path or NATIVE_FUNCTIONS_PATH,  # 使用给定或默认的原生函数路径
            tags_path or TAGS_PATH,  # 使用给定或默认的标签路径
            autograd_gen_dir,  # 自动微分生成目录
            autograd_dir,  # 自动微分目录
        )

    if operator_selector is None:
        operator_selector = SelectiveBuilder.get_nop_selector()  # 获取默认的操作选择器

    if subset == "libtorch" or not subset:
        gen_autograd(
            native_functions_path or NATIVE_FUNCTIONS_PATH,  # 使用给定或默认的原生函数路径
            tags_path or TAGS_PATH,  # 使用给定或默认的标签路径
            autograd_gen_dir,  # 自动微分生成目录
            autograd_dir,  # 自动微分目录
            disable_autograd=disable_autograd,  # 是否禁用自动微分
            operator_selector=operator_selector,  # 操作选择器
        )

    if subset == "python" or not subset:
        gen_annotated(
            native_functions_path or NATIVE_FUNCTIONS_PATH,  # 使用给定或默认的原生函数路径
            tags_path or TAGS_PATH,  # 使用给定或默认的标签路径
            python_install_dir,  # Python安装目录
            autograd_dir,  # 自动微分目录
        )


def get_selector_from_legacy_operator_selection_list(
    selected_op_list_path: str,  # 选择的运算符列表文件路径
) -> Any:
    with open(selected_op_list_path) as f:
        # 剥离重载部分
        # 这仅适用于旧配置 - 不要复制此代码！
        selected_op_list = {
            opname.split(".", 1)[0] for opname in yaml.load(f, Loader=YamlLoader)
        }  # 使用YAML加载器加载文件并处理成操作符名称的集合

    # 内部构建不再使用此标志。只用于OSS构建。
    # 每个操作符应被视为根操作符
    # （因此为其生成解箱代码，这与当前行为一致），并且还被视为使用
    # 设置标志，表示这是用于训练的运算符
    is_root_operator = True
    # 设置标志，表示这些运算符将用于训练
    is_used_for_training = True

    # 从选择构建器中导入SelectiveBuilder类
    from torchgen.selective_build.selector import SelectiveBuilder

    # 使用SelectiveBuilder类创建选择构建器对象，从旧操作注册允许列表中
    # 选取操作列表，并指定是否是根操作符和是否用于训练
    selector = SelectiveBuilder.from_legacy_op_registration_allow_list(
        selected_op_list,
        is_root_operator,
        is_used_for_training,
    )

    # 返回构建好的选择器对象
    return selector
# 定义函数 get_selector，用于获取选择器对象，根据输入的路径参数来确定选择哪种方式构建选择器
def get_selector(
    selected_op_list_path: str | None,
    operators_yaml_path: str | None,
) -> Any:
    # cwrap 依赖于 pyyaml，因此我们不能提前导入它
    # 获取当前脚本文件的上三级目录作为根目录
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # 将根目录添加到系统路径中
    sys.path.insert(0, root)
    # 从 torchgen.selective_build.selector 模块导入 SelectiveBuilder 类
    from torchgen.selective_build.selector import SelectiveBuilder

    # 确保 selected_op_list_path 和 operators_yaml_path 中最多只能有一个被设置
    assert not (
        selected_op_list_path is not None and operators_yaml_path is not None
    ), (
        "Expected at most one of selected_op_list_path and "
        + "operators_yaml_path to be set."
    )

    # 如果 selected_op_list_path 和 operators_yaml_path 都为 None，则返回默认的 nop selector
    if selected_op_list_path is None and operators_yaml_path is None:
        return SelectiveBuilder.get_nop_selector()
    # 如果 selected_op_list_path 不为 None，则调用特定的函数根据遗留的操作列表路径获取选择器
    elif selected_op_list_path is not None:
        return get_selector_from_legacy_operator_selection_list(selected_op_list_path)
    # 否则，根据 operators_yaml_path 调用 SelectiveBuilder 类的 from_yaml_path 方法获取选择器
    else:
        return SelectiveBuilder.from_yaml_path(cast(str, operators_yaml_path))


# 定义主函数 main，用于解析命令行参数并执行相关操作
def main() -> None:
    # 创建参数解析器对象 parser，并设置程序的描述信息
    parser = argparse.ArgumentParser(description="Autogenerate code")
    # 添加命令行参数 --native-functions-path 和 --tags-path
    parser.add_argument("--native-functions-path")
    parser.add_argument("--tags-path")
    # 添加命令行参数 --gen-dir，类型为 Path，默认为当前工作目录，用于指定文件安装的根目录
    parser.add_argument(
        "--gen-dir",
        type=Path,
        default=Path("."),
        help="Root directory where to install files. Defaults to the current working directory.",
    )
    # 添加命令行参数 --install-dir（已废弃），建议使用 --gen-dir 替代
    parser.add_argument(
        "--install-dir",
        "--install_dir",
        help=(
            "Deprecated. Use --gen-dir instead. The semantics are different, do not change "
            "blindly."
        ),
    )
    # 添加命令行参数 --subset，用于指定要生成的源文件的子集，可选值为 "libtorch" 或 "pybindings"
    parser.add_argument(
        "--subset",
        help='Subset of source files to generate. Can be "libtorch" or "pybindings". Generates both when omitted.',
    )
    # 添加命令行参数 --disable-autograd，默认为 False，设置为 True 可跳过生成与自动求导相关的代码
    parser.add_argument(
        "--disable-autograd",
        default=False,
        action="store_true",
        help="It can skip generating autograd related code when the flag is set",
    )
    # 添加命令行参数 --selected-op-list-path，用于指定包含自定义构建操作列表的 YAML 文件路径
    parser.add_argument(
        "--selected-op-list-path",
        help="Path to the YAML file that contains the list of operators to include for custom build.",
    )
    # 添加命令行参数 --operators-yaml-path（已废弃），建议使用 --selected-op-list-path 替代
    parser.add_argument(
        "--operators-yaml-path",
        "--operators_yaml_path",
        help="Path to the model YAML file that contains the list of operators to include for custom build.",
    )
    # 添加命令行参数 --force-schema-registration（已废弃），建议使用 --selected-op-list-path 替代
    parser.add_argument(
        "--force-schema-registration",
        "--force_schema_registration",
        action="store_true",
        help="force it to generate schema-only registrations for ops that are not"
        "listed on --selected-op-list",
    )
    # 添加命令行参数 --gen-lazy-ts-backend，启用生成 torch::lazy TorchScript 后端的支持
    parser.add_argument(
        "--gen-lazy-ts-backend",
        "--gen_lazy_ts_backend",
        action="store_true",
        help="Enable generation of the torch::lazy TorchScript backend",
    )
    # 添加命令行参数 --per-operator-headers，构建 lazy tensor ts 后端时使用每个操作符的 ATen 头文件
    parser.add_argument(
        "--per-operator-headers",
        "--per_operator_headers",
        action="store_true",
        help="Build lazy tensor ts backend with per-operator ATen headers, must match how ATen was built",
    )
    # 解析命令行参数，并将解析结果存储在 options 中
    options = parser.parse_args()
    # 调用生成代码的函数，传入多个参数以配置生成过程
    generate_code(
        options.gen_dir,                     # 生成代码的目录路径
        options.native_functions_path,       # 原生函数定义文件的路径
        options.tags_path,                   # 标签文件的路径
        options.install_dir,                 # 安装目录的路径
        options.subset,                      # 生成的子集
        options.disable_autograd,            # 是否禁用自动求导
        options.force_schema_registration,   # 是否强制模式注册
        # options.selected_op_list           # 选定的操作列表（已注释）
        operator_selector=get_selector(      # 操作符选择器，根据给定的路径获取操作列表
            options.selected_op_list_path, options.operators_yaml_path
        ),
    )

    # 如果设置了生成延迟执行的 TorchScript 后端
    if options.gen_lazy_ts_backend:
        # 获取 aten 的路径
        aten_path = os.path.dirname(os.path.dirname(options.native_functions_path))
        # 定义 TorchScript 后端的 YAML 文件路径
        ts_backend_yaml = os.path.join(aten_path, "native/ts_native_functions.yaml")
        # TorchScript 后端的 cpp 文件路径
        ts_native_functions = "torch/csrc/lazy/ts_backend/ts_native_functions.cpp"
        # TorchScript 后端的节点基类头文件路径
        ts_node_base = "torch/csrc/lazy/ts_backend/ts_node.h"
        # 设置安装目录，如果未指定则使用默认路径
        install_dir = options.install_dir or os.fspath(options.gen_dir / "torch/csrc")
        # 设置生成的延迟执行安装目录
        lazy_install_dir = os.path.join(install_dir, "lazy/generated")
        # 确保生成的延迟执行安装目录存在，如果不存在则创建
        os.makedirs(lazy_install_dir, exist_ok=True)

        # 断言检查 TorchScript 后端的 YAML 文件是否存在
        assert os.path.isfile(
            ts_backend_yaml
        ), f"Unable to access ts_backend_yaml: {ts_backend_yaml}"
        # 断言检查 TorchScript 后端的 cpp 文件是否存在
        assert os.path.isfile(
            ts_native_functions
        ), f"Unable to access {ts_native_functions}"
        # 导入生成 TorchScript 懒执行 IR 所需的模块
        from torchgen.dest.lazy_ir import GenTSLazyIR
        # 导入运行 TorchScript 懒执行张量生成的模块
        from torchgen.gen_lazy_tensor import run_gen_lazy_tensor

        # 调用函数生成 TorchScript 懒执行张量
        run_gen_lazy_tensor(
            aten_path=aten_path,                        # aten 路径
            source_yaml=ts_backend_yaml,                # TorchScript 后端的 YAML 文件路径
            backend_name="TorchScript",                 # 后端名称
            output_dir=lazy_install_dir,                # 生成文件的输出目录
            dry_run=False,                              # 是否是测试运行
            impl_path=ts_native_functions,              # TorchScript 后端的 cpp 文件路径
            node_base="TsNode",                         # 节点基类名称
            node_base_hdr=ts_node_base,                 # 节点基类头文件路径
            build_in_tree=True,                         # 是否在树中构建
            lazy_ir_generator=GenTSLazyIR,              # TorchScript 懒执行 IR 生成器
            per_operator_headers=options.per_operator_headers,  # 每个操作符的头文件
            gen_forced_fallback_code=True,              # 是否生成强制回退代码
        )
# 如果当前脚本作为主程序运行（而不是作为模块被导入），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 main 的函数，这通常是主程序的入口点
    main()
```