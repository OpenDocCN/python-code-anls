# `.\pytorch\torchgen\gen_lazy_tensor.py`

```
from __future__ import annotations
# 从未来导入 annotations，以支持使用类名称作为类型提示

import argparse
# 导入 argparse 库，用于命令行参数解析

import os
# 导入 os 库，提供了操作系统相关的功能

from collections import namedtuple
# 导入 namedtuple，用于创建命名元组

from pathlib import Path
# 导入 Path 类，用于处理文件路径

from typing import Any, Callable, Iterable, Iterator, Sequence
# 导入类型提示，用于声明函数参数和返回值的类型

import yaml
# 导入 yaml 库，用于读取和解析 YAML 文件

import torchgen.dest as dest
# 导入 torchgen.dest 库中的模块 dest

from torchgen.api.lazy import setValueT
# 从 torchgen.api.lazy 中导入 setValueT 函数

from torchgen.api.types import BaseCppType
# 从 torchgen.api.types 中导入 BaseCppType 类型

from torchgen.dest.lazy_ir import GenLazyIR, GenLazyNativeFuncDefinition, GenTSLazyIR
# 从 torchgen.dest.lazy_ir 导入 GenLazyIR, GenLazyNativeFuncDefinition, GenTSLazyIR 类

from torchgen.gen import get_grouped_native_functions, parse_native_yaml
# 从 torchgen.gen 导入 get_grouped_native_functions 和 parse_native_yaml 函数

from torchgen.gen_backend_stubs import (
    error_on_missing_kernels,
    gen_dispatcher_registrations,
    gen_dispatchkey_nativefunc_headers,
    parse_backend_yaml,
)
# 从 torchgen.gen_backend_stubs 导入多个函数：
# error_on_missing_kernels, gen_dispatcher_registrations,
# gen_dispatchkey_nativefunc_headers, parse_backend_yaml

from torchgen.model import NativeFunction, NativeFunctionsGroup, OperatorName
# 从 torchgen.model 导入 NativeFunction, NativeFunctionsGroup, OperatorName 类

from torchgen.selective_build.selector import SelectiveBuilder
# 从 torchgen.selective_build.selector 导入 SelectiveBuilder 类

from torchgen.utils import FileManager, NamespaceHelper
# 从 torchgen.utils 导入 FileManager 和 NamespaceHelper 类

from torchgen.yaml_utils import YamlLoader
# 从 torchgen.yaml_utils 导入 YamlLoader 类
# Each native function is modeled as an object with a schema, and each schema has objects representing their
# arguments.  Much of the codegen is manipulation of the arguments and their types.  For example, lazy tensor
# backends need to transform 'at::Tensor' arguments into 'lazy::Value' objects, as well as replacing reference
# types (stringref) with actual string objects, and this is done by manipulating the data model objects.
# - see api/lazy.py for the lazy data model
#
# Once the data model is set up, the rest of this script processes a number of templates for output CPP file
# and fills in the template values using helpers in `dest/lazy_ir.py` and `dest/lazy_ts_lowering.py`.  These
# helpers mostly iterate over functions and their arguments, outputting different c++ snippets.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Parses the external backend's yaml, and adds a new BackendIndex for the backend's dispatch key.
# Returns a Tuple of (backend_key, autograd_key, cpp_namespace, updated BackendIndex mapping, full_codegen)
ParsedExternalYaml = namedtuple(
    "ParsedExternalYaml",
    ["backend_key", "autograd_key", "cpp_namespace", "backend_indices", "full_codegen"],
)


def parse_native_functions_keys(
    backend_yaml_path: str,
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
) -> tuple[list[OperatorName], list[Any], list[OperatorName]]:
    # Open and load the specified backend YAML file
    with open(backend_yaml_path) as f:
        yaml_values = yaml.load(f, Loader=YamlLoader)
        
    # Ensure that the loaded YAML content is a dictionary
    assert isinstance(yaml_values, dict)

    # Extract specific lists from the YAML content
    full_codegen = yaml_values.pop("full_codegen", [])
    non_native = yaml_values.pop("non_native", [])
    ir_gen = yaml_values.pop("ir_gen", [])
    
    # Validate types of extracted lists
    assert isinstance(full_codegen, list)
    assert isinstance(non_native, list)
    assert isinstance(ir_gen, list)
    
    # Parse operator names from extracted lists
    full_codegen_opnames = [OperatorName.parse(name) for name in full_codegen]
    ir_gen_opnames = [OperatorName.parse(name) for name in ir_gen]
    
    return full_codegen_opnames, non_native, ir_gen_opnames


def validate_shape_inference_header(
    shape_inference_hdr: str, expected_shape_infr_decls: list[str]
) -> None:
    try:
        # Attempt to open and read the specified shape inference header file
        with open(shape_inference_hdr) as f:
            shape_infr_decls = f.read()
            shape_infr_decl_lines = set(shape_infr_decls.split("\n"))
    except OSError as e:
        # Handle file reading errors by raising an AssertionError
        raise AssertionError(
            f"Unable to read from the specified shape_inference_hdr file: {shape_inference_hdr}"
        ) from e

    # TODO(whc) add a check for shape inference functions that have meta kernels implement and should be retired.

    # Identify any missing shape inference function declarations
    missing_decls = [
        decl for decl in expected_shape_infr_decls if decl not in shape_infr_decl_lines
    ]
    if missing_decls:
        # Raise an exception if there are missing declarations
        raise Exception(
            f"""Missing shape inference function.\n
Please add declare this function in {shape_inference_hdr}:\n
and implement it in the corresponding shape_inference.cpp file.\n
            """
        )
# 为缺失声明生成一个字符串，每个声明之间用操作系统的换行符分隔
f"""{os.linesep.join(missing_decls)}"""

# 一些用于代码生成的辅助函数
def get_ltc_helper_fns() -> str:
    return """\
at::Tensor to_meta(const at::Tensor& tensor) {
  # 如果张量未定义，则无法转换为元设备，因为它们没有大小/步长信息
  if (!tensor.defined()) return tensor;
  # 根据输入张量的符号大小和步长创建一个新的元设备张量
  auto out = at::native::empty_strided_meta_symint(tensor.sym_sizes(), tensor.sym_strides(), \
/*dtype=*/std::make_optional(tensor.scalar_type()), /*layout=*/std::make_optional(tensor.layout()), \
/*device=*/std::make_optional(c10::Device(c10::kMeta)), /*pin_memory=*/std::nullopt);
  # 如果张量是包装数字，需要设置包装数字标志，以确保数据类型提升正常工作
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    out.unsafeGetTensorImpl()->set_wrapped_number(true);
  }
  return out;
}
# 将可选的张量转换为元设备张量，如果输入为非空则进行转换，否则返回空值
std::optional<at::Tensor> to_meta(const std::optional<at::Tensor>& tensor) {
  if (tensor.has_value()) {
    return to_meta(*tensor);
  }
  return std::nullopt;
}

# 将张量列表转换为元设备张量列表
std::vector<at::Tensor> to_meta(at::ITensorListRef t_list) {
  std::vector<at::Tensor> outs;
  outs.reserve(t_list.size());
  for (const auto& tensor : t_list) {
    outs.push_back(to_meta(tensor));
  }
  return outs;
}
"""

class default_args:
    node_base: str = "Node"
    node_base_hdr: str | None = None
    shape_inference_hdr: str = "torch/csrc/lazy/core/shape_inference.h"
    tensor_class: str = "torch::lazy::LazyTensor"
    tensor_class_hdr: str = "torch/csrc/lazy/core/tensor.h"
    lazy_ir_generator: type[GenLazyIR] = GenLazyIR
    native_func_definition_generator: type[
        GenLazyNativeFuncDefinition
    ] = GenLazyNativeFuncDefinition
    backend_name: str = "TorchScript"

# 主函数入口
def main() -> None:
    # 创建命令行参数解析器，用于生成懒惰张量后端文件
    parser = argparse.ArgumentParser(description="Generate Lazy Tensor backend files")
    # 添加命令行参数选项
    parser.add_argument(
        "-s",
        "--source-yaml",
        "--source_yaml",
        help="path to source yaml file containing operator external definitions",
    )
    parser.add_argument("-o", "--output-dir", "--output_dir", help="output directory")
    parser.add_argument(
        "--dry-run", "--dry_run", type=bool, default=False, help="output directory"
    )
    parser.add_argument(
        "--impl-path",
        "--impl_path",
        type=str,
        default=None,
        help="path to the source C++ file containing kernel definitions",
    )
    parser.add_argument(
        "--gen-ts-lowerings",
        "--gen_ts_lowerings",
        action="store_true",
        help="Generate TorchScript lowerings in addition to Lazy IR and NativeFunctions",
    )
    parser.add_argument(
        "--node-base",
        "--node_base",
        type=str,
        default=default_args.node_base,
        help="Name of backend specific custom Lazy IR Node base class",
    )
    parser.add_argument(
        "--node-base-hdr",
        "--node_base_hdr",
        type=str,
        default=default_args.node_base_hdr,
        help="Path to header file defining custom Lazy IR Node base class",
    )
    # 添加一个名为 "--shape-inference-hdr" 或 "--shape_inference_hdr" 的命令行参数
    # 用于指定自定义懒惰形状推断函数的头文件路径
    parser.add_argument(
        "--shape-inference-hdr",
        "--shape_inference_hdr",
        type=str,
        default=default_args.shape_inference_hdr,
        help="Path to header file defining custom Lazy shape inference functions",
    )

    # 添加一个名为 "--tensor-class" 或 "--tensor_class" 的命令行参数
    # 用于指定后端特定的自定义懒惰张量类的名称
    parser.add_argument(
        "--tensor-class",
        "--tensor_class",
        type=str,
        default=default_args.tensor_class,
        help="Name of backend specific custom Lazy Tensor class",
    )

    # 添加一个名为 "--tensor-class-hdr" 或 "--tensor_class_hdr" 的命令行参数
    # 用于指定自定义懒惰张量类的头文件路径
    parser.add_argument(
        "--tensor-class-hdr",
        "--tensor_class_hdr",
        type=str,
        default=default_args.tensor_class_hdr,
        help="Path to header file defining custom Lazy Tensor class",
    )

    # 添加一个名为 "--backend-name" 或 "--backend_name" 的命令行参数
    # 用于指定要生成的后端的名称
    parser.add_argument(
        "--backend-name",
        "--backend_name",
        type=str,
        default=default_args.backend_name,
        help="Name of the backend to generate",
    )

    # 解析命令行参数，并将结果存储在 options 中
    options = parser.parse_args()

    # 假设该文件位于 PYTORCH_ROOT/torchgen/gen_backend_stubs.py
    # 确定 PyTorch 的根目录路径
    torch_root = Path(__file__).parent.parent.parent.absolute()

    # 构建 ATen 源码路径
    aten_path = str(torch_root / "aten" / "src" / "ATen")

    # 设置默认的懒惰 IR 生成器
    lazy_ir_generator: type[GenLazyIR] = default_args.lazy_ir_generator

    # 如果用户选择生成 TS 降级的代码，则使用 GenTSLazyIR 作为懒惰 IR 生成器
    if options.gen_ts_lowerings:
        lazy_ir_generator = GenTSLazyIR

    # 设置默认的本地函数定义生成器
    native_func_definition_generator: type[GenLazyNativeFuncDefinition] = default_args.native_func_definition_generator

    # 运行生成懒惰张量的函数
    run_gen_lazy_tensor(
        aten_path,
        options.source_yaml,
        options.output_dir,
        options.dry_run,
        options.impl_path,
        options.node_base,
        options.node_base_hdr,
        options.tensor_class,
        options.tensor_class_hdr,
        options.shape_inference_hdr,
        lazy_ir_generator,
        native_func_definition_generator,
        options.backend_name,
    )
def run_gen_lazy_tensor(
    aten_path: str,
    source_yaml: str,
    output_dir: str,
    dry_run: bool,
    impl_path: str | None,
    node_base: str = default_args.node_base,
    node_base_hdr: str | None = default_args.node_base_hdr,
    tensor_class: str = default_args.tensor_class,
    tensor_class_hdr: str = default_args.tensor_class_hdr,
    shape_inference_hdr: str = default_args.shape_inference_hdr,
    lazy_ir_generator: type[GenLazyIR] = default_args.lazy_ir_generator,
    native_func_definition_generator: type[
        GenLazyNativeFuncDefinition
    ] = default_args.native_func_definition_generator,
    # build_in_tree is true for TS backend and affects include paths
    build_in_tree: bool = False,
    # per_operator_headers changes whether ATen/Functions.h or individual operator headers are used
    # it must match how ATen was built
    per_operator_headers: bool = False,
    backend_name: str = default_args.backend_name,
    gen_forced_fallback_code: bool = False,
    use_lazy_shape: bool = True,
    # the following arguments are temporary customization points for xla backend migration.
    # do not rely on them otherwise, they should be removed once migration is complete
    backend_namespace: str = "torch::lazy",
    get_tensorlist: str = "GetTensorList",
    get_tensor_or_wrap_number: str = "GetLtcTensorOrCreateForWrappedNumber",
    try_get_tensor: str = "TryGetLtcTensor",
    metrics_counter: str = 'TORCH_LAZY_FN_COUNTER("lazy::")',
    create_tensor: str = "LazyTensor::Create",
    create_from_first_tensor: bool = False,
    create_aten_from_ltc_tensor: str = "torch::lazy::CreateAtenFromLtcTensor",
    tuple_aten_from_ltc_tensors: str = "torch::lazy::TupleAtenFromLtcTensors",
    lazy_value_class: str = "torch::lazy::Value",
    lazy_tensor_ptr: str = "LazyTensorPtr",
    get_device_fn: str = "torch::lazy::GetBackendDevice",
) -> None:
    # Split the lazy_value_class string by "::" and determine the namespace and class name
    lv_tokens = lazy_value_class.split("::")
    lv_class = lv_tokens[-1]
    lv_ns = "::".join(lv_tokens[:-1])
    # Set the base C++ type for the lazy value class
    setValueT(BaseCppType(lv_ns, lv_class))
    # Construct the template directory path within the ATen installation path
    template_dir = os.path.join(aten_path, "templates")

    # Define a function to create a FileManager instance for handling file operations
    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(
            install_dir=install_dir, template_dir=template_dir, dry_run=dry_run
        )

    # Create a FileManager instance for the specified output directory
    fm = make_file_manager(output_dir)

    # Determine the paths to the native functions YAML and tags YAML files
    native_yaml_path = os.path.join(aten_path, "native/native_functions.yaml")
    tags_yaml_path = os.path.join(aten_path, "native/tags.yaml")
    # Parse the native YAML files to extract native functions and backend indices
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )
    # Group the native functions based on their categories
    grouped_native_functions = get_grouped_native_functions(native_functions)
    def sort_native_function(f: NativeFunctionsGroup | NativeFunction) -> str:
        """
        We sort the native function because of the note in concat_map_codegen.
        TODO(alanwaketan): Remove this sorting hack once all ops are grouped properly.
        """
        # 如果参数 f 是 NativeFunctionsGroup 类型，则获取其 functional 属性的函数名；否则直接获取 f 的函数名
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        # 返回函数名的字符串表示形式
        return str(func.name.name)

    # 对 grouped_native_functions 列表进行排序，排序依据为 sort_native_function 函数返回的字符串
    grouped_native_functions = sorted(
        grouped_native_functions, key=sort_native_function
    )

    # 解析后端 YAML 文件，返回解析后的对象
    parsed_backend_yaml = parse_backend_yaml(
        source_yaml, grouped_native_functions, backend_indices
    )
    # 获取解析后的后端关键信息
    backend_key = parsed_backend_yaml.backend_key
    autograd_key = parsed_backend_yaml.autograd_key
    cpp_namespace = parsed_backend_yaml.cpp_namespace
    backend_indices = parsed_backend_yaml.backend_indices

    # 解析 native functions keys，返回三个不同的键
    full_codegen, non_native, ir_gen = parse_native_functions_keys(
        source_yaml, grouped_native_functions
    )

    def concat_map_codegen(
        func: Callable[[NativeFunction], Sequence[str]],
        xs: Iterable[NativeFunctionsGroup | NativeFunction],
        ops_list: list[OperatorName] = full_codegen,
    ) -> Iterator[str]:
        """
        We code-gen for the functional variant, which is all we need for IR classes/lowerings/shape inferences, but we
        only code-gen additional entries for the inplace variant for the native functions.
        """
        # 遍历 xs 中的每个元素 x
        for x in xs:
            # 如果 x 是 NativeFunctionsGroup 类型，则获取其所有函数；否则将 x 作为单个函数列表
            fs = list(x.functions()) if isinstance(x, NativeFunctionsGroup) else [x]
            # 遍历 fs 列表中的每个函数 f
            for f in fs:
                # 如果 f 的函数名在 ops_list 中
                if f.func.name in ops_list:
                    # 使用 func 函数生成代码，并以迭代器方式返回
                    yield from func(f)

    # 获取一个 nop selector 实例
    selector = SelectiveBuilder.get_nop_selector()

    # 断言 backend_key 不为 None
    assert backend_key is not None
    # 获取 backend_key 对应的后端类名
    class_name = backend_indices[backend_key].native_function_class_name()

    # 如果 impl_path 不为 None，则检查缺失的 kernel 错误
    if impl_path is not None:
        error_on_missing_kernels(
            native_functions,
            backend_indices,
            backend_key,
            autograd_key,
            class_name,
            impl_path,
            full_codegen,
        )

    """ Validate Shape Inference Definitions

    Generated lazy native functions all perform shape inference, by first using a meta:: kernel
    if available for that op, and otherwise using a 'compute_shape_{op}' function instead.  The generator
    knows the call signature for compute_shape_{op} because it matches the nativefunction (and meta::) signature,
    so it just has to check whether the op is structured and generate a call for one or the other.  It's up to the dev
    to supply the missing compute_shape_{op} function, but the codegen at least warns you about this and provides
    the expected signature which can be copy-pasted into shape_inference.h.
    compute_shape_{op} functions are handwritten and should be replaced over time as ops get ported
    to structured kernels.

    See torch/csrc/lazy/core/shape_inference.cpp #READ THIS! for more information.
    """
    如果给定 shape_inference_hdr 不为 None：
        生成预期的形状推断声明列表，这些声明是通过 concat_map_codegen 函数生成的，用于懒惰推断定义。
        使用 backend_indices[backend_key] 和 tensor_class 作为参数，
        并结合 grouped_native_functions 生成延迟形状推断定义。
    """
    if shape_inference_hdr is not None:
        expected_shape_infr_decls = list(
            concat_map_codegen(
                dest.GenLazyShapeInferenceDefinition(
                    backend_indices[backend_key], tensor_class
                ),
                grouped_native_functions,
            )
        )

        # 验证给定的形状推断头文件是否符合预期的形状推断声明列表
        validate_shape_inference_header(shape_inference_hdr, expected_shape_infr_decls)
    # 断言 class_name 不为 None
    assert class_name is not None

    # 生成本地函数声明
    # 注意，对于懒惰 TS 后端，eager registrations 被设置为 False，因为其他 LTC 后端可能希望注册自己的懒惰内核而不是 TS 的内核。
    # 当调用 init_ts_backend 时，注册将被惰性执行。
    gen_dispatchkey_nativefunc_headers(
        fm,
        class_name,
        cpp_namespace,
        backend_indices,
        grouped_native_functions,
        backend_key,
        autograd_key,
        backend_name,
    )

    # 生成 Dispatcher 注册，用于连接本地函数
    # 对于 dispatch_key，如果 autograd_key 为 None，则只使用 [backend_key]；否则使用 [backend_key, autograd_key]。
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
            build_in_tree=build_in_tree,
            per_operator_headers=per_operator_headers,
            backend_name=backend_name,
            eager_registration=False,
        )

    # 生成构建 IR 节点的本地函数实现
    ns_helper = NamespaceHelper(cpp_namespace)
    fm.write_with_template(
        f"{backend_key}NativeFunctions.cpp",
        "DispatchKeyNativeFunctions.cpp",
        lambda: {
            "includes": [
                f"#include <{path}>"
                for path in [
                    tensor_class_hdr,                          # 包含张量类的头文件
                    shape_inference_hdr,                       # 包含形状推断的头文件
                    "ATen/Functions.h",                        # 包含ATen函数的头文件
                    "ATen/native/TensorConversions.h",         # 包含ATen本地张量转换的头文件
                    "ATen/NativeFunctions.h",                  # 包含ATen本地函数的头文件
                    "ATen/CompositeExplicitAutogradNonFunctionalFunctions.h",  # 包含复合显式自动微分非功能函数的头文件
                    "ATen/MetaFunctions.h",                    # 包含ATen元函数的头文件
                    "ATen/Operators.h",                        # 包含ATen操作符的头文件
                    "ATen/native/CPUFallback.h",               # 包含ATen本地CPU回退的头文件
                    "torch/csrc/lazy/core/ir_builder.h",       # 包含IR构建器的头文件
                    "torch/csrc/lazy/core/lazy_graph_executor.h",  # 包含延迟图执行器的头文件
                    "torch/csrc/lazy/core/metrics.h",          # 包含度量标准的头文件
                    "torch/csrc/lazy/core/shape.h",            # 包含形状的头文件
                    f"{output_dir}/{backend_key}NativeFunctions.h",  # 输出目录下的后端本地函数头文件
                    f"{output_dir}/LazyIr.h",                  # 输出目录下的延迟IR头文件
                ]
                + (
                    ["torch/csrc/lazy/ts_backend/ts_eager_fallback.h"]  # 如果生成强制回退代码，则包含张量流后端急切回退的头文件
                    if gen_forced_fallback_code
                    else []                                    # 否则为空列表
                )
            ],
            "helper_fns": get_ltc_helper_fns(),               # 获取LTC（懒惰张量编译器）辅助函数
            "native_functions_include": "",                   # 包含本地函数的声明
            "namespace_prologue": ns_helper.prologue,         # 命名空间的前言
            "namespace_epilogue": ns_helper.epilogue,         # 命名空间的结尾
            "native_function_definitions": list(
                concat_map_codegen(
                    native_func_definition_generator(
                        f"{backend_key}NativeFunctions",       # 后端本地函数名
                        backend_indices[backend_key],          # 后端索引
                        tensor_class,                         # 张量类
                        gen_forced_fallback_code,             # 是否生成强制回退代码的标志
                        backend_namespace,                    # 后端命名空间
                        get_tensorlist,                       # 获取张量列表的函数
                        get_tensor_or_wrap_number,            # 获取张量或包装数字的函数
                        try_get_tensor,                       # 尝试获取张量的函数
                        metrics_counter,                      # 度量计数器
                        create_tensor,                        # 创建张量的函数
                        create_from_first_tensor,             # 从第一个张量创建的函数
                        create_aten_from_ltc_tensor,          # 从LTC张量创建ATen张量的函数
                        tuple_aten_from_ltc_tensors,          # 从LTC张量元组创建ATen张量的函数
                        lazy_tensor_ptr,                      # 懒惰张量指针
                        get_device_fn,                        # 获取设备函数
                    ),
                    grouped_native_functions,                  # 分组的本地函数
                )
            ),
        },
    )
    # 生成IR节点类
    lazy_ir_obj = lazy_ir_generator(
        backend_indices[backend_key],                       # 后端索引
        backend_name,                                      # 后端名称
        node_base,                                         # 节点基类
        use_lazy_shape                                     # 是否使用延迟形状的标志
    )
    # 使用文件管理器(fm)的方法write_with_template写入内容到文件LazyIr.h，使用模板"LazyIr.h"
    fm.write_with_template(
        "LazyIr.h",
        "LazyIr.h",
        lambda: {
            # "lazy_ir_sysinc"对应的值是一个列表，包含了多个以路径为模板的包含指令
            "lazy_ir_sysinc": [
                f"#include <{path}>"  # 每个路径都会被格式化为#include指令
                for path in [  # 路径列表包括了多个头文件路径
                    "ATen/core/Formatting.h",
                    "c10/core/ScalarType.h",
                    "c10/util/Optional.h",
                    "torch/csrc/lazy/core/hash.h",
                    "torch/csrc/lazy/core/ir.h",
                    "torch/csrc/lazy/core/shape.h",
                    "vector",  # C++标准库中的vector头文件
                ]
            ],
            # 如果node_base_hdr不是None，则"lazy_ir_inc"包含一个以node_base_hdr为路径的包含指令列表；否则为空列表
            "lazy_ir_inc": [f'#include "{node_base_hdr}"']
            if node_base_hdr is not None
            else [],
            # "ir_declarations"对应的值是一个由concat_map_codegen生成的列表，处理lazy_ir_obj、grouped_native_functions和full_codegen + ir_gen
            "ir_declarations": list(
                concat_map_codegen(
                    lazy_ir_obj, grouped_native_functions, full_codegen + ir_gen
                )
            ),
            # "namespace_prologue"和"namespace_epilogue"分别引用ns_helper对象的prologue和epilogue属性
            "namespace_prologue": ns_helper.prologue,
            "namespace_epilogue": ns_helper.epilogue,
        },
    )

    # 使用文件管理器(fm)的方法write_with_template写入内容到文件LazyNonNativeIr.h，使用模板"LazyNonNativeIr.h"
    fm.write_with_template(
        "LazyNonNativeIr.h",
        "LazyNonNativeIr.h",
        lambda: {
            # "lazy_non_native_ir_inc"对应的值是一个列表，包含了多个以路径为模板的包含指令
            "lazy_non_native_ir_inc": [
                f"#include <{path}>"  # 每个路径都会被格式化为#include指令
                for path in [
                    "torch/csrc/lazy/core/ir.h",
                    "torch/csrc/lazy/core/ir_builder.h",
                    "torch/csrc/lazy/core/internal_ops/ltc_ops.h",
                    "torch/csrc/lazy/core/shape_inference.h",
                ]
                + ([node_base_hdr] if node_base_hdr else [])  # 如果node_base_hdr存在，则添加到路径列表中
                if path  # 过滤空路径
            ],
            # "non_native_ir_nodes"对应的值是dest对象生成的非本地惰性IR节点
            "non_native_ir_nodes": dest.generate_non_native_lazy_ir_nodes(
                non_native, lazy_ir_obj
            ),
            # "namespace_prologue"和"namespace_epilogue"分别引用ns_helper对象的prologue和epilogue属性
            "namespace_prologue": ns_helper.prologue,
            "namespace_epilogue": ns_helper.epilogue,
        },
    )
# 如果这个脚本作为主程序被执行，那么执行 main() 函数
if __name__ == "__main__":
    main()
```