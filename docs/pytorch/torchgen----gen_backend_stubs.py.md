# `.\pytorch\torchgen\gen_backend_stubs.py`

```py
# 从__future__模块导入annotations特性，使得可以在函数参数和返回值中使用类型提示
from __future__ import annotations

# 导入命令行参数解析模块
import argparse
# 导入操作系统功能模块
import os
# 导入正则表达式模块
import re
# 导入计数器和默认字典模块
from collections import Counter, defaultdict, namedtuple
# 导入路径操作模块
from pathlib import Path
# 导入类型提示模块
from typing import Sequence

# 导入YAML格式解析模块
import yaml

# 导入torchgen库中的模块和类
import torchgen.api.dispatcher as dispatcher
import torchgen.dest as dest
# 导入类型提示相关的类
from torchgen.api.types import DispatcherSignature
# 导入代码模板类
from torchgen.code_template import CodeTemplate
# 导入本地函数管理上下文模块
from torchgen.context import native_function_manager
# 导入本地函数分组和解析本地YAML函数的函数
from torchgen.gen import get_grouped_native_functions, parse_native_yaml
# 导入模型相关类
from torchgen.model import (
    BackendIndex,
    BackendMetadata,
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
)
# 导入选择性构建模块
from torchgen.selective_build.selector import SelectiveBuilder
# 导入工具函数和类
from torchgen.utils import concatMap, context, FileManager, NamespaceHelper, Target
# 导入YAML加载器
from torchgen.yaml_utils import YamlLoader


# 定义解析外部后端YAML文件的返回类型，包括后端关键字、自动微分关键字、类名、C++命名空间和更新后的后端索引映射
ParsedExternalYaml = namedtuple(
    "ParsedExternalYaml",
    ["backend_key", "autograd_key", "class_name", "cpp_namespace", "backend_indices"],
)


# 解析外部后端YAML文件，添加新的后端索引到后端调度关键字中
def parse_backend_yaml(
    backend_yaml_path: str,  # 后端YAML文件的路径
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],  # 本地函数和函数组的序列
    backend_indices: dict[DispatchKey, BackendIndex],  # 后端索引字典，映射调度关键字到后端索引
) -> ParsedExternalYaml:  # 返回解析后的外部YAML内容的ParsedExternalYaml对象
    # 创建本地函数名称到本地函数对象的映射字典
    native_functions_map: dict[OperatorName, NativeFunction] = {
        f.func.name: f
        for f in concatMap(
            lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()),
            grouped_native_functions,
        )
    }

    # 使用YAML加载器读取并解析指定路径的YAML文件
    with open(backend_yaml_path) as f:
        yaml_values = yaml.load(f, Loader=YamlLoader)
    assert isinstance(yaml_values, dict)  # 确保YAML值为字典类型

    # 定义有效的YAML键列表
    valid_keys = [
        "backend",
        "class_name",
        "cpp_namespace",
        "extra_headers",
        "supported",
        "autograd",
        "full_codegen",
        "non_native",
        "ir_gen",
        "symint",
    ]

    # 弹出并获取YAML中的后端关键字
    backend = yaml_values.pop("backend", None)
    assert backend is not None, 'You must provide a value for "backend"'  # 确保后端关键字不为空

    # 弹出并获取YAML中的类名
    class_name = yaml_values.pop("class_name", None)

    # 弹出并获取YAML中的C++命名空间
    cpp_namespace = yaml_values.pop("cpp_namespace", None)
    assert cpp_namespace is not None, 'You must provide a value for "cpp_namespace"'  # 确保C++命名空间不为空

    # 弹出并获取YAML中的use_out_as_primary标志，默认为False
    use_out_as_primary = yaml_values.pop("use_out_as_primary", False)
    assert isinstance(
        use_out_as_primary, bool
    ), f"You must provide either True or False for use_out_as_primary. Provided: {use_out_as_primary}"  # 确保use_out_as_primary为布尔值

    # 弹出并获取YAML中的device_guard标志，默认为False
    use_device_guard = yaml_values.pop("device_guard", False)
    assert isinstance(
        use_device_guard, bool
    ), f"You must provide either True or False for device_guard. Provided: {use_device_guard}"  # 确保device_guard为布尔值

    # 弹出并获取YAML中的supported操作列表，如果为None则设置为空列表
    supported = yaml_values.pop("supported", [])
    if supported is None:
        supported = []  # 允许支持操作列表为空列表
    # 检查 supported 是否为列表类型，如果不是则抛出异常
    assert isinstance(
        supported, list
    ), f'expected "supported" to be a list, but got: {supported} (of type {type(supported)})'

    # 从 yaml_values 字典中弹出键为 "symint" 的值，如果为 None，则设置为空列表
    symint = yaml_values.pop("symint", [])
    if symint is None:
        symint = []  # 允许 symint 操作的空列表
    # 检查 symint 是否为列表类型，如果不是则抛出异常
    assert isinstance(
        symint, list
    ), f'expected "symint" to be a list, but got: {supported} (of type {type(supported)})'
    # 将 symint 转换为集合
    symint_set = set(symint)

    # 从 yaml_values 字典中弹出键为 "autograd" 的值，默认为空列表
    supported_autograd = yaml_values.pop("autograd", [])
    # 检查 supported_autograd 是否为列表类型，如果不是则抛出异常
    assert isinstance(
        supported_autograd, list
    ), f'expected "autograd" to be a list, but got: {supported_autograd}'

    # 从 yaml_values 字典中弹出键为 "full_codegen" 的值，默认为空列表
    full_codegen = yaml_values.pop("full_codegen", [])
    # 将 full_codegen 的内容添加到 supported 列表中
    supported.extend(full_codegen)

    # 从 yaml_values 字典中弹出键为 "non_native" 的值，默认为空字典，但在此处无用
    yaml_values.pop("non_native", {})

    # 从 yaml_values 字典中弹出键为 "ir_gen" 的值，默认为空字典，但在此处无用
    yaml_values.pop("ir_gen", {})

    # 检查 yaml_values 字典是否不含任何键，如果有多余的键则抛出异常
    assert (
        len(yaml_values.keys()) == 0
    ), f'{backend_yaml_path} contains unexpected keys: {", ".join(yaml_values.keys())}. \
    # 定义函数 create_backend_index，用于创建后端索引对象
    def create_backend_index(
        backend_ops: list[str],  # 接受后端操作列表
        symint_ops: set[str],   # 接受符号整数操作的集合
        dispatch_key: DispatchKey,  # 指定调度键的类型
        *,
        use_out_as_primary: bool,   # 是否将输出用作主要
        use_device_guard: bool,     # 是否使用设备保护
    ) -> BackendIndex:   # 返回类型为 BackendIndex 对象
        metadata: dict[OperatorName, BackendMetadata] = {}  # 创建空字典，用于存储操作符名到后端元数据的映射
        for op in backend_ops:
            op_name = OperatorName.parse(op)  # 解析操作符名称
            assert (
                op_name in native_functions_map
            ), f"Found an invalid operator name: {op_name}"  # 断言操作符名存在于本地函数映射中
            # 获取调度器函数的名称
            kernel_name = dispatcher.name(native_functions_map[op_name].func)
            if op in symint_ops:
                kernel_name += "_symint"  # 若操作符在符号整数操作集合中，则添加 "_symint" 后缀
            # 创建后端元数据对象
            m = BackendMetadata(
                kernel=kernel_name, structured=False, cpp_namespace=cpp_namespace
            )
            metadata[op_name] = m  # 将操作符名及其对应的后端元数据存入字典中
        return BackendIndex(
            dispatch_key=dispatch_key,  # 设置调度键
            use_out_as_primary=use_out_as_primary,  # 设置是否使用输出作为主要
            external=True,  # 表示为外部后端
            device_guard=use_device_guard,  # 设置是否使用设备保护
            index=metadata,  # 将元数据字典作为索引的一部分
        )

    backend_key: DispatchKey | None = None  # 初始化后端调度键为 None
    if len(supported) > 0:  # 如果支持的后端列表非空
        with context(
            lambda: f'The provided value for "backend" must be a valid DispatchKey, but got {backend}.'
        ):
            backend_key = DispatchKey.parse(backend)  # 解析并设置后端调度键

        backend_idx = create_backend_index(
            supported,  # 支持的后端操作列表
            symint_set,  # 符号整数操作集合
            backend_key,  # 后端调度键
            use_out_as_primary=use_out_as_primary,  # 是否使用输出作为主要
            use_device_guard=use_device_guard,  # 是否使用设备保护
        )
        assert backend_key not in backend_indices  # 确保后端调度键不在后端索引字典中
        backend_indices[backend_key] = backend_idx  # 将后端调度键及其索引存入后端索引字典中

    autograd_key: DispatchKey | None = None  # 初始化自动微分调度键为 None
    if len(supported_autograd) > 0:  # 如果支持的自动微分列表非空
        with context(
            lambda: f'The "autograd" key was specified, which indicates that you would like to override \
the behavior of autograd for some operators on your backend. However "Autograd{backend}" is not a valid DispatchKey.'
        ):
            autograd_key = DispatchKey.parse(f"Autograd{backend}")  # 解析并设置自动微分调度键

        autograd_idx = create_backend_index(
            supported_autograd,  # 支持的自动微分操作列表
            symint_set,  # 符号整数操作集合
            autograd_key,  # 自动微分调度键
            use_out_as_primary=use_out_as_primary,  # 是否使用输出作为主要
            use_device_guard=use_device_guard,  # 是否使用设备保护
        )
        assert autograd_key not in backend_indices  # 确保自动微分调度键不在后端索引字典中
        backend_indices[autograd_key] = autograd_idx  # 将自动微分调度键及其索引存入后端索引字典中
    # 遍历分组后的原生函数列表
    for g in grouped_native_functions:
        # 检查当前元素是否为 NativeFunction 类型
        if isinstance(g, NativeFunction):
            # 如果是 NativeFunction 类型，根据 backend_key 获取前向传播的内核函数列表
            forward_kernels = (
                []
                if backend_key is None  # 如果 backend_key 为 None，则前向传播内核函数列表为空
                else [
                    m
                    for m in [backend_indices[backend_key].get_kernel(g)]
                    if m is not None
                ]
            )
            # 根据 autograd_key 获取后向传播的内核函数列表
            backward_kernels = (
                []
                if autograd_key is None  # 如果 autograd_key 为 None，则后向传播内核函数列表为空
                else [
                    m
                    for m in [backend_indices[autograd_key].get_kernel(g)]
                    if m is not None
                ]
            )
        else:
            # 如果当前元素不是 NativeFunction 类型，根据 backend_key 获取前向传播的内核函数列表
            forward_kernels = (
                []
                if backend_key is None  # 如果 backend_key 为 None，则前向传播内核函数列表为空
                else [
                    m
                    for m in [
                        backend_indices[backend_key].get_kernel(f)  # 遍历 g 的每个函数 f，获取其前向传播内核函数
                        for f in g.functions()
                    ]
                    if m is not None
                ]
            )
            # 根据 autograd_key 获取后向传播的内核函数列表
            backward_kernels = (
                []
                if autograd_key is None  # 如果 autograd_key 为 None，则后向传播内核函数列表为空
                else [
                    m
                    for m in [
                        backend_indices[autograd_key].get_kernel(f)  # 遍历 g 的每个函数 f，获取其后向传播内核函数
                        for f in g.functions()
                    ]
                    if m is not None
                ]
            )

        # 过滤掉列表中的 None 值，确保前向传播和后向传播内核函数列表中不含有 None
        forward_kernels = [f for f in forward_kernels if f is not None]
        backward_kernels = [f for f in backward_kernels if f is not None]
        
        # 断言：每个操作的所有变体必须要么都注册到一个 backend key，要么都注册到一个 autograd key
        assert (
            len(forward_kernels) == 0 or len(backward_kernels) == 0
        ), f'Currently, all variants of an op must either be registered to a backend key, or to a backend\'s \
autograd function'
    # 创建一个 ParsedExternalYaml 对象，用于解析外部 YAML 文件内容并返回
    return ParsedExternalYaml(
        backend_key, autograd_key, class_name, cpp_namespace, backend_indices
    )


def error_on_missing_kernels(
    # 检查缺失的内核并抛出错误
    native_functions: Sequence[NativeFunction],
    # 包含不同后端索引的字典，用于查找内核定义
    backend_indices: dict[DispatchKey, BackendIndex],
    # 后端的调度键
    backend_key: DispatchKey,
    # 自动求导的调度键（可选）
    autograd_key: DispatchKey | None,
    # 类名，用于查找内核定义
    class_name: str,
    # 内核定义文件的路径
    kernel_defn_file_path: str,
    # 全部代码生成的操作符名称列表（可选）
    full_codegen: list[OperatorName] | None = None,
) -> None:
    try:
        # 打开并读取内核定义文件
        with open(kernel_defn_file_path) as f:
            backend_defns = f.read()
    except OSError as e:
        # 如果无法读取文件，抛出 AssertionError
        raise AssertionError(
            f"Unable to read from the specified impl_path file: {kernel_defn_file_path}"
        ) from e

    # 如果未提供全部代码生成列表，则初始化为空列表
    if full_codegen is None:
        full_codegen = []

    # 获取指定后端和自动求导对应的索引列表
    indices = [backend_indices[backend_key].index] + (
        [] if autograd_key is None else [backend_indices[autograd_key].index]
    )

    # 期望的外部后端操作符名称到后端内核名称的映射
    expected_backend_op_names: dict[OperatorName, str] = dict(
        list(
            concatMap(
                lambda index: [
                    (op_name, metadata.kernel) for op_name, metadata in index.items()
                ],
                indices,
            )
        )
    )

    # 过滤出预期的后端原生函数列表
    expected_backend_native_funcs: list[NativeFunction] = [
        f
        for f in native_functions
        if f.func.name in expected_backend_op_names.keys()
        and f.func.name not in full_codegen
    ]

    # 按预期的后端内核名称对原生函数进行分组计数
    expected_backend_kernel_name_counts: dict[str, list[NativeFunction]] = defaultdict(
        list
    )
    for native_f in expected_backend_native_funcs:
        expected_backend_kernel_name_counts[
            expected_backend_op_names[native_f.func.name]
        ].append(native_f)

    # 使用正则表达式查找内核定义中实际的后端内核名称及其计数
    kernel_defn_regex = rf"(.*){class_name}::\s*([\w\d]*)\("
    actual_backend_kernel_name_counts = Counter(
        [
            y
            for (x, y) in re.findall(kernel_defn_regex, backend_defns)
            if not x.endswith(":")
        ]
    )

    # 初始化缺失内核的错误消息
    missing_kernels_err_msg = ""
    # 遍历 expected_backend_kernel_name_counts 字典中的每一对键值对
    for expected_name, funcs in expected_backend_kernel_name_counts.items():
        # 获取当前函数名在 expected_backend_kernel_name_counts 中出现的次数
        expected_overload_count = len(funcs)
        # 获取当前函数名在 actual_backend_kernel_name_counts 中实际出现的次数
        actual_overload_count = actual_backend_kernel_name_counts[expected_name]
        
        # 检查预期的重载次数和实际的重载次数是否不相等
        if expected_overload_count != actual_overload_count:

            # 定义一个内部函数 create_decl，用于创建函数声明字符串
            def create_decl(f: NativeFunction) -> str:
                # 将函数 f 注册到 native_function_manager 中，并获取函数声明字符串
                with native_function_manager(f):
                    return DispatcherSignature.from_schema(f.func).decl()

            # 生成所有不符合预期的函数声明字符串，并拼接到 missing_kernels_err_msg 中
            expected_schemas_str = "\n".join([create_decl(f) for f in funcs])
            # 将错误信息添加到 missing_kernels_err_msg 中
            missing_kernels_err_msg += f"""
    {class_name} is missing a kernel definition for {expected_name}. We found {actual_overload_count} kernel(s) with that name,
    but expected {expected_overload_count} kernel(s). The expected function schemas for the missing operator are:
    {expected_schemas_str}

    """
    # 断言确保没有缺失的内核定义信息
    assert missing_kernels_err_msg == "", missing_kernels_err_msg


def main() -> None:
    # 创建命令行解析器对象，用于生成后端存根文件
    parser = argparse.ArgumentParser(description="Generate backend stub files")
    # 添加命令行参数：源 YAML 文件路径，包含操作符的外部定义
    parser.add_argument(
        "-s",
        "--source-yaml",
        "--source_yaml",
        help="path to source yaml file containing operator external definitions",
    )
    # 添加命令行参数：输出目录路径
    parser.add_argument("-o", "--output-dir", "--output_dir", help="output directory")
    # 添加命令行参数：是否仅演示运行，类型为布尔，默认为 False
    parser.add_argument(
        "--dry-run", "--dry_run", type=bool, default=False, help="output directory"
    )
    # 添加命令行参数：实现路径，源 C++ 文件包含内核定义
    parser.add_argument(
        "--impl-path",
        "--impl_path",
        type=str,
        default=None,
        help="path to the source C++ file containing kernel definitions",
    )
    # 解析命令行参数
    options = parser.parse_args()

    # 运行生成函数，传递解析得到的命令行参数
    run(options.source_yaml, options.output_dir, options.dry_run, options.impl_path)


def gen_dispatchkey_nativefunc_headers(
    fm: FileManager,
    class_name: str,
    cpp_namespace: str,
    backend_indices: dict[DispatchKey, BackendIndex],
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    backend_dispatch_key: DispatchKey,
    autograd_dispatch_key: DispatchKey | None,
    backend_name: str = "",
) -> None:
    # 断言确保类名不为空
    assert class_name is not None
    # 自动生成的注释，指示此文件由 gen_backend_stubs.py 自动生成，不要直接编辑
    generated_comment = (
        "Autogenerated file by gen_backend_stubs.py. Do not edit directly!"
    )

    # 转换成集合以去除重复的内核名称
    # 后端允许重复内核名称；只生成一次声明即可！
    # 排序以确保输出的确定性
    backend_declarations = sorted(
        set(
            concatMap(
                lambda f: dest.compute_native_function_declaration(
                    f, backend_indices[backend_dispatch_key]
                ),
                grouped_native_functions,
            )
        )
    )
    # 如果自动微分分发键为 None，则不生成自动微分声明
    autograd_declarations = sorted(
        set(
            concatMap(
                lambda f: []
                if autograd_dispatch_key is None
                else dest.compute_native_function_declaration(
                    f, backend_indices[autograd_dispatch_key]
                ),
                grouped_native_functions,
            )
        )
    )

    # 创建命名空间助手对象，用于处理 C++ 命名空间
    ns_helper = NamespaceHelper(cpp_namespace)
    # 使用 fm 对象调用 write_with_template 方法，生成特定文件的内容
    fm.write_with_template(
        # 生成的文件名，基于 backend_dispatch_key 参数生成
        f"{backend_dispatch_key}NativeFunctions.h",
        # 使用的模板文件名，固定为 "DispatchKeyNativeFunctions.h"
        "DispatchKeyNativeFunctions.h",
        # lambda 函数用于动态生成模板中需要的内容
        lambda: {
            # 自动生成的注释部分
            "generated_comment": generated_comment,
            # 命名空间引入部分
            "namespace_prologue": ns_helper.prologue,
            # 类名
            "class_name": class_name,
            # 命名空间尾部
            "namespace_epilogue": ns_helper.epilogue,
            # 分发声明，包括后端和自动微分声明
            "dispatch_declarations": backend_declarations + autograd_declarations,
            # 后端名称
            "BackendName": backend_name,
            # 分发键
            "DispatchKey": backend_dispatch_key,
        },
    )
# 定义一个函数，生成分发器注册信息
def gen_dispatcher_registrations(
    fm: FileManager,                                      # 文件管理器对象
    output_dir: str,                                      # 输出目录路径
    class_name: str,                                      # 类名
    backend_indices: dict[DispatchKey, BackendIndex],     # 后端索引字典
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],  # 分组的本地函数序列
    backend_dispatch_key: DispatchKey,                     # 后端分发键
    dispatch_key: DispatchKey,                             # 分发键
    selector: SelectiveBuilder,                            # 选择构建器对象
    # build_in_tree is true for lazy TS backend and affects include paths, not used for external backends
    build_in_tree: bool = False,                           # 是否在构建树中，默认为假，影响包含路径
    per_operator_headers: bool = False,                    # 每个操作符头文件，默认为假
    backend_name: str = "",                                # 后端名称，默认为空字符串
    eager_registration: bool = True,                       # 是否急于注册，默认为真
) -> None:
    # 构建头文件路径列表
    headers = [
        f"{output_dir}/{backend_dispatch_key}NativeFunctions.h",
    ]
    # 根据 build_in_tree 的值选择包含路径格式
    if build_in_tree:
        external_backend_headers_str = "\n".join(f"#include <{h}>" for h in headers)
    else:
        external_backend_headers_str = "\n".join(f'#include "{h}"' for h in headers)

    # 断言确保 class_name 不为 None
    assert class_name is not None
    # 获取指定 dispatch_key 对应的后端索引
    backend_index = backend_indices[dispatch_key]

    # 生成分发注册主体列表
    dispatch_registrations_body = list(
        concatMap(
            dest.RegisterDispatchKey(
                backend_index,
                Target.REGISTRATION,
                selector,
                rocm=False,
                symint=True,
                class_method_name=f"{class_name}",
                skip_dispatcher_op_registration=False,
            ),
            grouped_native_functions,
        )
    )
    
    # 新行字符
    newline = "\n"
    # 命名空间辅助器对象
    ns_helper = NamespaceHelper(namespace_str="at")
    # 延迟分发注册的静态和延迟模板
    deferred_dispatch_registrations = ""
    static_init_dispatch_registrations = ""
    
    # 根据 eager_registration 的值选择静态或延迟注册模板
    if eager_registration:
        static_template = CodeTemplate(
            """\
TORCH_LIBRARY_IMPL(aten, $dispatch_key, m) {
    $dispatch_registrations_body
};"""
        )
        static_init_dispatch_registrations = static_template.substitute(
            dispatch_key=dispatch_key,
            dispatch_registrations_body=dispatch_registrations_body,
        )
    else:
        deferred_template = CodeTemplate(
            """\
TORCH_API void Register${backend_name}${dispatch_key}NativeFunctions();
TORCH_API void Register${backend_name}${dispatch_key}NativeFunctions() {
    static auto m = MAKE_TORCH_LIBRARY_IMPL(aten, $dispatch_key);
    $dispatch_registrations_body
}"""
        )
        deferred_dispatch_registrations = deferred_template.substitute(
            backend_name=backend_name,
            dispatch_key=dispatch_key,
            dispatch_registrations_body=dispatch_registrations_body,
        )
    fm.write_with_template(
        f"Register{dispatch_key}.cpp",  # 生成以 dispatch_key 命名的注册文件名
        "RegisterDispatchKey.cpp",  # 使用 "RegisterDispatchKey.cpp" 作为模板文件
        lambda: {  # 使用 lambda 函数生成模板所需的参数字典
            "extra_cuda_headers": "",  # 额外的 CUDA 头文件为空字符串
            "external_backend_headers": external_backend_headers_str,  # 外部后端头文件的字符串表示
            "ops_headers": "#include <ATen/Functions.h>" if not per_operator_headers else "",  # 根据条件选择操作头文件
            "DispatchKey": dispatch_key,  # 注册的 dispatch key
            "dispatch_namespace": dispatch_key.lower(),  # 小写形式的 dispatch key 命名空间
            "dispatch_headers": dest.gen_registration_headers(
                backend_index, per_operator_headers=per_operator_headers, rocm=False
            ),  # 生成并返回注册头文件列表
            "dispatch_definitions": fm.substitute_with_template(
                "RegisterDispatchDefinitions.ini",  # 使用 "RegisterDispatchDefinitions.ini" 作为模板文件
                lambda: {  # 使用 lambda 函数生成模板所需的参数字典
                    "ns_prologue": ns_helper.prologue,  # 命名空间前言
                    "ns_epilogue": ns_helper.epilogue,  # 命名空间尾声
                    "static_init_dispatch_registrations": static_init_dispatch_registrations,  # 静态初始化分发注册
                    "deferred_dispatch_registrations": deferred_dispatch_registrations,  # 延迟分发注册
                    "dispatch_helpers": dest.gen_registration_helpers(backend_index),  # 生成并返回注册辅助函数
                    "dispatch_namespace": dispatch_key.lower(),  # 小写形式的 dispatch key 命名空间
                    "dispatch_namespaced_definitions": "",  # 命名空间内部的定义为空字符串
                    "dispatch_anonymous_definitions": list(
                        concatMap(
                            dest.RegisterDispatchKey(
                                backend_index,
                                Target.ANONYMOUS_DEFINITION,
                                selector,
                                rocm=False,
                                symint=True,
                                class_method_name=f"{class_name}",
                                skip_dispatcher_op_registration=False,
                            ),  # 调用 RegisterDispatchKey 函数，生成匿名定义的列表
                            grouped_native_functions,
                        )  # 对分组的本地函数列表进行操作
                    ),
                },
            ).split(newline),  # 使用 newline 分割并返回字符串模板生成的结果列表
        },
    )
# 定义一个函数 `run`，用于生成后端存根代码。
def run(
    source_yaml: str, output_dir: str, dry_run: bool, impl_path: str | None = None
) -> None:
    # 假设此文件位于 PYTORCH_ROOT/torchgen/gen_backend_stubs.py，获取 PyTorch 根目录
    pytorch_root = Path(__file__).parent.parent.absolute()
    # 设置模板目录为 PYTORCH_ROOT/aten/src/ATen/templates
    template_dir = os.path.join(pytorch_root, "aten/src/ATen/templates")

    # 定义函数 `make_file_manager`，返回一个 FileManager 实例
    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(
            install_dir=install_dir, template_dir=template_dir, dry_run=dry_run
        )

    # 使用 `make_file_manager` 函数创建 FileManager 对象 `fm`，输出目录为 `output_dir`
    fm = make_file_manager(output_dir)

    # 设置本地 YAML 文件路径
    native_yaml_path = os.path.join(
        pytorch_root, "aten/src/ATen/native/native_functions.yaml"
    )
    # 设置标签 YAML 文件路径
    tags_yaml_path = os.path.join(pytorch_root, "aten/src/ATen/native/tags.yaml")
    # 解析本地 YAML 文件，获取解析后的结果 `parsed_yaml`
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
    # 获取本地函数列表和后端索引
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )
    # 将本地函数列表按组分组
    grouped_native_functions = get_grouped_native_functions(native_functions)
    # 解析后端 YAML 文件，获取解析后的结果 `parsed_backend_yaml`
    parsed_backend_yaml = parse_backend_yaml(
        source_yaml, grouped_native_functions, backend_indices
    )
    # 获取后端键值
    backend_key = parsed_backend_yaml.backend_key
    # 获取自动求导键值
    autograd_key = parsed_backend_yaml.autograd_key
    # 获取 C++ 命名空间
    cpp_namespace = parsed_backend_yaml.cpp_namespace
    # 获取类名
    class_name = parsed_backend_yaml.class_name
    # 获取后端索引
    backend_indices = parsed_backend_yaml.backend_indices

    # 创建一个 nop selector 对象 `selector`
    selector = SelectiveBuilder.get_nop_selector()

    # 如果后端键值为 None，则返回，用于快速设置空的 YAML 文件
    if backend_key is None:
        return

    # 如果类名为 None，则指定默认值或使用后端索引中的默认类名
    if class_name is None:
        class_name = backend_indices[backend_key].native_function_class_name()
    # 断言类名不为 None
    assert class_name is not None

    # 如果实现路径不为 None，则检查缺失的内核函数
    if impl_path is not None:
        error_on_missing_kernels(
            native_functions,
            backend_indices,
            backend_key,
            autograd_key,
            class_name,
            impl_path,
        )

    # 生成调度键本地函数头部
    gen_dispatchkey_nativefunc_headers(
        fm,
        class_name,
        cpp_namespace,
        backend_indices,
        grouped_native_functions,
        backend_key,
        autograd_key,
    )

    # 针对每个调度键生成分发器注册信息
    for dispatch_key in (
        [backend_key] if autograd_key is None else [backend_key, autograd_key]
    ):
        gen_dispatcher_registrations(
            fm,
            output_dir,
            class_name,
            backend_indices,
            grouped_native_functions,
            backend_key,
            dispatch_key,
            selector,
        )

# 如果运行文件为主程序，则执行 `main` 函数
if __name__ == "__main__":
    main()
```