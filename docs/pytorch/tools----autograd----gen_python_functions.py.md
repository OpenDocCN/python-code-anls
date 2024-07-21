# `.\pytorch\tools\autograd\gen_python_functions.py`

```py
# 为 ATen 函数生成 Python 绑定
#
# 这些绑定生成为 python_variable 上的方法，或者作为 torch._C._nn、torch._C._fft、torch._C._linalg、
# torch._C._nested、torch._C._sparse 或 torch._C._special 对象上的函数。
#

# 代码遵循以下规则：
#
# - 模板应与使用它们的函数放置在一起。
#   模板目前没有在函数之间共享，但如果发生这种情况，可能会将模板与第一个函数放在一起。
#
# - 调用 template.substitute() 时不要使用环境字典。
#   对所有内容直接传递命名参数，否则将很难跟踪实际被使用的内容及其使用者。
#
# - 将新的黑客/调整与现有的同类项放置在一起，最好在数据结构而不是代码中。参见例如 SCHEMA_DEFAULT_CONVERSION_HACKS 等。
#
# - 类似地，从一种格式到另一种格式的转换应该在一个地方一次性完成。
#
# - 不要使用复杂的嵌套函数。几行代码是可以的，但请尽量避免读取/写入远处定义的外部变量的函数。
#
# - 抛出 RuntimeError 而不是使用断言，并尽可能将尽可能多的信息放入消息中。
#   即，不需要传递新参数，其唯一目的是填充错误消息，而是使用已有的参数。
#

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from typing import Callable, Iterable, Sequence

import yaml

from torchgen.api import cpp
from torchgen.api.python import (
    arg_parser_output_exprs,
    cpp_dispatch_exprs,
    cpp_dispatch_target,
    dispatch_lambda_args,
    dispatch_lambda_exprs,
    dispatch_lambda_return_str,
    has_tensor_options,
    PythonSignature,
    PythonSignatureDeprecated,
    PythonSignatureGroup,
    PythonSignatureNativeFunctionPair,
    signature,
    signature_from_schema,
    structseq_fieldnames,
)
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.gen import cpp_string, parse_native_yaml, parse_tags_yaml
from torchgen.model import (
    Argument,
    BaseOperatorName,
    FunctionSchema,
    NativeFunction,
    SchemaKind,
    Type,
    Variant,
)
from torchgen.utils import FileManager, split_name_params
from torchgen.yaml_utils import YamlLoader

from .gen_inplace_or_view_type import is_tensor_list_type
from .gen_trace_type import should_trace


#
# 声明块列表
# 我们跳过这些函数的代码生成，出于各种原因。
# 未来的 PR 将对此列表进行分类，消除或提升它们出现在急切生成代码之外。
# 参见 https://github.com/pytorch/pytorch/issues/30788
#

# 这些函数需要手动的 Python 绑定，或者未暴露给 Python
_SKIP_PYTHON_BINDINGS = [
    "alias",
    "contiguous",
    "is_cuda",
    "is_sparse",
    "is_sparse_csr",
    "size",
    "stride",
    "sym_size",
    "sym_stride",
    # 定义一个字符串列表，用于存储匹配指定模式的字符串
    patterns = [
        "sym_storage_offset",                           # 匹配字符串 "sym_storage_offset"
        "sym_numel",                                    # 匹配字符串 "sym_numel"
        ".*_backward",                                  # 匹配以 "_backward" 结尾的字符串
        ".*_backward_(out|input|weight|bias)",           # 匹配以 "_backward_" 开头并接特定词尾的字符串
        ".*_forward",                                   # 匹配以 "_forward" 结尾的字符串
        ".*_forward_out",                               # 匹配以 "_forward_out" 结尾的字符串
        ".*_jvp",                                       # 匹配以 "_jvp" 结尾的字符串
        "_unsafe_view",                                 # 匹配字符串 "_unsafe_view"
        "tensor",                                       # 匹配字符串 "tensor"
        "_?sparse_(coo|compressed|csr|csc|bsr|bsc)_tensor.*",  # 匹配包含特定格式的稀疏张量相关字符串
        "_range.*",                                     # 匹配以 "_range" 开头的字符串
        "_sparse_add_out",                              # 匹配字符串 "_sparse_add_out"
        "_sparse_div.*",                                # 匹配以 "_sparse_div" 开头的字符串
        "_sparse_mul.*",                                # 匹配以 "_sparse_mul" 开头的字符串
        "_sparse_sub.*",                                # 匹配以 "_sparse_sub" 开头的字符串
        "_sparse_dense_add_out",                        # 匹配字符串 "_sparse_dense_add_out"
        "index",                                        # 匹配字符串 "index"
        "index_out",                                    # 匹配字符串 "index_out"
        "unique_dim_consecutive",                       # 匹配字符串 "unique_dim_consecutive"
        "_cumsum.*",                                    # 匹配以 "_cumsum" 开头的字符串
        "_cumprod.*",                                   # 匹配以 "_cumprod" 开头的字符串
        "_sum.*",                                       # 匹配以 "_sum" 开头的字符串
        "_prod.*",                                      # 匹配以 "_prod" 开头的字符串
        "_th_.*",                                       # 匹配以 "_th_" 开头的字符串
        "_thnn_.*",                                     # 匹配以 "_thnn_" 开头的字符串
        "range.*",                                      # 匹配以 "range" 开头的字符串
        "_solve.*",                                     # 匹配以 "_solve" 开头的字符串
        "_inverse.*",                                   # 匹配以 "_inverse" 开头的字符串
        "_cholesky.*",                                  # 匹配以 "_cholesky" 开头的字符串
        "_triangular_solve.*",                          # 匹配以 "_triangular_solve" 开头的字符串
        "_qr.*",                                        # 匹配以 "_qr" 开头的字符串
        "_svd.*",                                       # 匹配以 "_svd" 开头的字符串
        "slice",                                        # 匹配字符串 "slice"
        "item",                                         # 匹配字符串 "item"
        "_local_scalar_dense",                          # 匹配字符串 "_local_scalar_dense"
        "to",                                            # 匹配字符串 "to"
        "_to_copy",                                     # 匹配字符串 "_to_copy"
        "_to_copy_out",                                 # 匹配字符串 "_to_copy_out"
        "_reshape_copy",                                # 匹配字符串 "_reshape_copy"
        "_reshape_copy_out",                            # 匹配字符串 "_reshape_copy_out"
        "copy_sparse_to_sparse_",                       # 匹配字符串 "copy_sparse_to_sparse_"
        "copy_",                                        # 匹配字符串 "copy_"
        "_foreach_copy",                                # 匹配字符串 "_foreach_copy"
        "numpy_T",                                      # 匹配字符串 "numpy_T"
        "matrix_H",                                     # 匹配字符串 "matrix_H"
        "mT",                                           # 匹配字符串 "mT"
        "mH",                                           # 匹配字符串 "mH"
        "nonzero(_(out|numpy))?",                       # 匹配 "nonzero" 或 "nonzero_out" 或 "nonzero_numpy"
        "set_data",                                     # 匹配字符串 "set_data"
        ".*_overrideable",                              # 匹配以 "_overrideable" 结尾的字符串
        "data",                                         # 匹配字符串 "data"
        "is_leaf",                                      # 匹配字符串 "is_leaf"
        "output_nr",                                    # 匹配字符串 "output_nr"
        "_version",                                     # 匹配字符串 "_version"
        "requires_grad_",                               # 匹配字符串 "requires_grad_"
        "retains_grad",                                 # 匹配字符串 "retains_grad"
        "set_",                                         # 匹配字符串 "set_"
        "_fw_primal",                                   # 匹配字符串 "_fw_primal"
        "fake_quantize_per_tensor_affine_cachemask",    # 匹配字符串 "fake_quantize_per_tensor_affine_cachemask"
        "fake_quantize_per_channel_affine_cachemask",   # 匹配字符串 "fake_quantize_per_channel_affine_cachemask"
        "_new_zeros_with_same_feature_meta",             # 匹配字符串 "_new_zeros_with_same_feature_meta"
        "_has_same_storage_numel",                      # 匹配字符串 "_has_same_storage_numel"
        "_reshape_alias",                               # 匹配字符串 "_reshape_alias"
        "replace_",                                     # 匹配字符串 "replace_"
        "copy",                                         # 匹配字符串 "copy"
        "fill.Tensor",                                  # 匹配字符串 "fill.Tensor"
        "fill.Scalar",                                  # 匹配字符串 "fill.Scalar"
        "lift.*",                                       # 匹配以 "lift" 开头的字符串
        "normal_functional",                            # 匹配字符串 "normal_functional"
        "nbytes",                                       # 匹配字符串 "nbytes"
        "itemsize",                                     # 匹配字符串 "itemsize"
        "_batch_norm_with_update",                      # 匹配字符串 "_batch_norm_with_update"
        "_batch_norm_with_update_out",                  # 匹配字符串 "_batch_norm_with_update_out"
        "_batch_norm_no_update",                        # 匹配字符串 "_batch_norm_no_update"
    ]
]

# 使用正则表达式根据预定义的模式生成跳过的 Python 绑定列表
SKIP_PYTHON_BINDINGS = [
    re.compile(rf"^{pattern}$") for pattern in _SKIP_PYTHON_BINDINGS
]

# 这些函数签名不会暴露给 Python。注意这个签名列表不支持正则表达式。
SKIP_PYTHON_BINDINGS_SIGNATURES = [
    "add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
    "sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
    "mul.Scalar(Tensor self, Scalar other) -> Tensor",
    "mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
    "div.Scalar(Tensor self, Scalar other) -> Tensor",
    "div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
]

# 使用装饰器 @with_native_function 标记的函数，判断是否应该生成 Python 绑定
@with_native_function
def should_generate_py_binding(f: NativeFunction) -> bool:
    # 对于完全由代码生成的 NativeFunction，不应该生成 Python 绑定，
    # 因为这些代码生成的实现通常效率低下。少数像 view_copy 风格操作是意外暴露的，
    # 因为它们之前是手写的，现在我们因为向后兼容的原因将它们移到代码生成。
    if "generated" in f.tags and "view_copy" not in f.tags:
        return False

    name = cpp.name(f.func)
    # 检查函数名是否匹配跳过 Python 绑定的正则表达式
    for skip_regex in SKIP_PYTHON_BINDINGS:
        if skip_regex.match(name):
            return False

    signature = str(f.func)
    # 检查函数签名是否在跳过 Python 绑定的签名列表中
    for pattern in SKIP_PYTHON_BINDINGS_SIGNATURES:
        if pattern == signature:
            return False
    return True


# 根据 BaseOperatorName 返回相应的 Python/C++ 操作名
def get_pycname(name: BaseOperatorName) -> str:
    return f"THPVariable_{name}"


# 检查函数重载是否是无参数的
def is_noarg(overloads: Sequence[PythonSignatureNativeFunctionPair]) -> bool:
    return len(overloads) == 1 and overloads[0].signature.arguments_count() == 0


# 检查 NativeFunction 是否是 Python 变量方法
def is_py_variable_method(f: NativeFunction) -> bool:
    return f.python_module is None and Variant.method in f.variants


# 检查 NativeFunction 是否是 Python Torch 函数
def is_py_torch_function(f: NativeFunction) -> bool:
    return f.python_module is None and Variant.function in f.variants


# 检查 NativeFunction 是否是 Python nn 模块函数
def is_py_nn_function(f: NativeFunction) -> bool:
    return f.python_module == "nn"


# 检查 NativeFunction 是否是 Python fft 模块函数
def is_py_fft_function(f: NativeFunction) -> bool:
    return f.python_module == "fft"


# 检查 NativeFunction 是否是 Python linalg 模块函数
def is_py_linalg_function(f: NativeFunction) -> bool:
    return f.python_module == "linalg"


# 检查 NativeFunction 是否是 Python nested 模块函数
def is_py_nested_function(f: NativeFunction) -> bool:
    return f.python_module == "nested"


# 检查 NativeFunction 是否是 Python sparse 模块函数
def is_py_sparse_function(f: NativeFunction) -> bool:
    return f.python_module == "sparse"


# 检查 NativeFunction 是否是 Python special 模块函数
def is_py_special_function(f: NativeFunction) -> bool:
    return f.python_module == "special"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                            Main Function
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# 生成函数，负责生成指定的 Python 绑定文件
def gen(
    out: str,
    native_yaml_path: str,
    tags_yaml_path: str,
    deprecated_yaml_path: str,
    template_path: str,
    *,
    symint: bool = True,


# 定义一个布尔类型的变量 symint，并初始化为 True
symint: bool = True,
def generate_bindings(
    out: Path, 
    template_path: Path, 
    native_yaml_path: Path, 
    deprecated_yaml_path: Path, 
    tags_yaml_path: Path, 
    symint: bool
) -> None:
    # 创建一个文件管理器实例，指定安装目录、模板目录和是否为干预运行模式
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    
    # 解析原生 YAML 文件中的原生函数信息，并进行过滤获取原生函数列表
    native_functions = parse_native_yaml(
        native_yaml_path, tags_yaml_path
    ).native_functions
    native_functions = list(filter(should_generate_py_binding, native_functions))

    # 加载原生函数的签名信息，生成 Python 绑定方法
    methods = load_signatures(native_functions, deprecated_yaml_path, method=True)
    create_python_bindings(
        fm,
        methods,
        is_py_variable_method,
        None,
        "python_variable_methods.cpp",
        method=True,
        symint=symint,
    )

    # 加载原生函数的签名信息，生成分片的 Python 绑定方法（用于 Torch 函数）
    functions = load_signatures(native_functions, deprecated_yaml_path, method=False)
    create_python_bindings_sharded(
        fm,
        functions,
        is_py_torch_function,
        "torch",
        "python_torch_functions.cpp",
        method=False,
        num_shards=3,
        symint=symint,
    )

    # 生成 Python 绑定方法（用于 torch.nn 函数）
    create_python_bindings(
        fm,
        functions,
        is_py_nn_function,
        "torch.nn",
        "python_nn_functions.cpp",
        method=False,
        symint=symint,
    )

    # 生成 Python 绑定方法（用于 torch.fft 函数）
    create_python_bindings(
        fm,
        functions,
        is_py_fft_function,
        "torch.fft",
        "python_fft_functions.cpp",
        method=False,
        symint=symint,
    )

    # 生成 Python 绑定方法（用于 torch.linalg 函数）
    create_python_bindings(
        fm,
        functions,
        is_py_linalg_function,
        "torch.linalg",
        "python_linalg_functions.cpp",
        method=False,
        symint=symint,
    )

    # 生成 Python 绑定方法（用于 torch.nested 函数）
    create_python_bindings(
        fm,
        functions,
        is_py_nested_function,
        "torch.nested",
        "python_nested_functions.cpp",
        method=False,
    )

    # 生成 Python 绑定方法（用于 torch.sparse 函数）
    create_python_bindings(
        fm,
        functions,
        is_py_sparse_function,
        "torch.sparse",
        "python_sparse_functions.cpp",
        method=False,
        symint=symint,
    )

    # 生成 Python 绑定方法（用于 torch.special 函数）
    create_python_bindings(
        fm,
        functions,
        is_py_special_function,
        "torch.special",
        "python_special_functions.cpp",
        method=False,
        symint=symint,
    )

    # 使用 functions 生成返回类型的 Python 绑定
    create_python_return_type_bindings(
        fm, functions, lambda fn: True, "python_return_types.cpp"
    )

    # 生成返回类型的 Python 绑定头文件
    create_python_return_type_bindings_header(
        fm, functions, lambda fn: True, "python_return_types.h"
    )

    # 解析标签 YAML 文件，获取有效标签
    valid_tags = parse_tags_yaml(tags_yaml_path)

    # 定义一个函数，生成标签的枚举字典
    def gen_tags_enum() -> dict[str, str]:
        return {
            "enum_of_valid_tags": (
                "".join(
                    [f'\n.value("{tag}", at::Tag::{tag})' for tag in sorted(valid_tags)]
                )
            )
        }
    # 向文件流 fm 写入生成的标签枚举数据，文件名为 "python_enum_tag.cpp"
    fm.write("python_enum_tag.cpp", gen_tags_enum)
# 将给定函数签名和预测函数对按照操作符名称进行分组过滤，并返回一个以操作符名称为键，值为对应函数对列表的字典
def group_filter_overloads(
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
) -> dict[BaseOperatorName, list[PythonSignatureNativeFunctionPair]]:
    # 使用 defaultdict 创建一个空的字典，值为列表，用于存储按操作符名称分组后的函数对列表
    grouped: dict[
        BaseOperatorName, list[PythonSignatureNativeFunctionPair]
    ] = defaultdict(list)
    # 遍历给定的函数对列表
    for pair in pairs:
        # 如果预测函数对该函数的 NativeFunction 返回 True
        if pred(pair.function):
            # 将该函数对添加到以操作符名称为键的字典值中的列表中
            grouped[pair.function.func.name.name].append(pair)
    # 返回分组后的字典
    return grouped


# 生成 ATen 函数的 Python 绑定
def create_python_bindings(
    fm: FileManager,
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
    module: str | None,
    filename: str,
    *,
    method: bool,
    symint: bool = True,
) -> None:
    """Generates Python bindings to ATen functions"""
    # 初始化存储不同类型的内容的列表
    py_methods: list[str] = []
    ops_headers: list[str] = []
    py_method_defs: list[str] = []
    py_forwards: list[str] = []

    # 将函数对按操作符名称分组过滤
    grouped = group_filter_overloads(pairs, pred)

    # 遍历按操作符名称分组后的字典的键（按字母顺序排序）
    for name in sorted(grouped.keys(), key=str):
        # 获取当前操作符名称下的函数对列表
        overloads = grouped[name]
        # 生成方法实现并添加到 py_methods 列表中
        py_methods.append(
            method_impl(name, module, overloads, method=method, symint=symint)
        )
        # 生成方法定义并添加到 py_method_defs 列表中
        py_method_defs.append(method_def(name, module, overloads, method=method))
        # 生成前向声明并扩展到 py_forwards 列表中
        py_forwards.extend(forward_decls(name, overloads, method=method))
        # 生成操作符头文件包含语句并添加到 ops_headers 列表中
        ops_headers.append(f"#include <ATen/ops/{name.base}.h>")

    # 使用 FileManager 写入模板文件，填充以下内容：
    # - 生成的注释信息
    # - ATen 操作符头文件包含列表
    # - 前向声明列表
    # - 方法实现列表
    # - 方法定义列表
    fm.write_with_template(
        filename,
        filename,
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/{filename}",
            "ops_headers": ops_headers,
            "py_forwards": py_forwards,
            "py_methods": py_methods,
            "py_method_defs": py_method_defs,
        },
    )


# 生成返回类型为 Python 的绑定函数
def create_python_return_type_bindings(
    fm: FileManager,
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
    filename: str,
) -> None:
    """
    Generate function to initialize and return named tuple for native functions
    which returns named tuple and registration invocations in `python_return_types.cpp`.
    """
    # 初始化存储返回类型定义和注册语句的列表
    py_return_types_definition: list[str] = []
    py_return_types_registrations: list[str] = []

    # 将函数对按操作符名称分组过滤
    grouped = group_filter_overloads(pairs, pred)

    # 遍历按操作符名称分组后的字典的键（按字母顺序排序）
    for name in sorted(grouped.keys(), key=str):
        # 获取当前操作符名称下的函数对列表
        overloads = grouped[name]
        # 生成返回类型的定义和注册语句，并分别添加到对应列表中
        definitions, registrations = generate_return_type_definition_and_registrations(
            overloads
        )
        py_return_types_definition.append(
            "" if not definitions else "\n".join(definitions)
        )
        py_return_types_registrations.append(
            "" if not registrations else "\n".join(registrations)
        )
    fm.write_with_template(
        filename,
        filename,
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/{filename}",
            "py_return_types": py_return_types_definition,
            "py_return_types_registrations": py_return_types_registrations,
        },
    )



# 使用模板文件管理器(fm)的write_with_template方法生成文件
# 生成的文件名和模板文件名相同
# 使用lambda函数生成包含以下内容的字典：
# - "generated_comment": 包含生成文件的注释信息，格式为 "@generated from {模板目录}/{filename}"
# - "py_return_types": Python返回类型的定义
# - "py_return_types_registrations": Python返回类型的注册信息
# 生成用于 Python 返回类型绑定头文件的函数，用于初始化和返回本地函数的命名元组及相关映射条目
def create_python_return_type_bindings_header(
    fm: FileManager,
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
    filename: str,
) -> None:
    """
    生成函数，用于初始化和返回本地函数的命名元组及相关映射条目，在 `python_return_types.cpp` 中使用。
    """
    # 初始化 Python 返回类型声明列表
    py_return_types_declarations: list[str] = []

    # 将函数根据名称分组并过滤
    grouped = group_filter_overloads(pairs, pred)

    # 遍历已排序的函数名称列表
    for name in sorted(grouped.keys(), key=str):
        # 获取该函数名下的所有函数重载列表
        overloads = grouped[name]
        # 生成返回类型声明，并将其添加到声明列表中
        declarations = generate_return_type_declarations(overloads)
        py_return_types_declarations.append(
            "" if not declarations else "\n".join(declarations)
        )

    # 使用文件管理器写入带有模板的文件
    fm.write_with_template(
        filename,
        filename,
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/{filename}",
            "py_return_types_declarations": py_return_types_declarations,
        },
    )


# 生成 Python 绑定到 ATen 函数的分片版本
def create_python_bindings_sharded(
    fm: FileManager,
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
    module: str | None,
    filename: str,
    *,
    method: bool,
    num_shards: int,
    symint: bool = True,
) -> None:
    """生成 Python 绑定到 ATen 函数"""
    # 根据名称分组和过滤函数重载
    grouped = group_filter_overloads(pairs, pred)

    # 定义用于键值排序的函数
    def key_func(
        kv: tuple[BaseOperatorName, list[PythonSignatureNativeFunctionPair]]
    ) -> str:
        return kv[0].base

    # 定义用于生成环境字典的函数
    def env_func(
        kv: tuple[BaseOperatorName, list[PythonSignatureNativeFunctionPair]]
    ) -> dict[str, list[str]]:
        name, fn_pairs = kv
        return {
            "ops_headers": [f"#include <ATen/ops/{name.base}.h>"],
            "py_forwards": list(forward_decls(name, fn_pairs, method=method)),
            "py_methods": [
                method_impl(name, module, fn_pairs, method=method, symint=symint)
            ],
            "py_method_defs": [method_def(name, module, fn_pairs, method=method)],
        }

    # 使用文件管理器写入分片文件
    fm.write_sharded(
        filename,
        grouped.items(),
        base_env={
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/{filename}",
        },
        key_fn=key_func,
        env_callable=env_func,
        num_shards=num_shards,
        sharded_keys={"ops_headers", "py_forwards", "py_methods", "py_method_defs"},
    )


# 加载函数签名
def load_signatures(
    native_functions: list[NativeFunction],
    deprecated_yaml_path: str,
    *,
    method: bool,
    skip_deprecated: bool = False,
    pyi: bool = False,
) -> Sequence[PythonSignatureNativeFunctionPair]:
    @with_native_function
    def gen_signature_pairs(f: NativeFunction) -> PythonSignatureNativeFunctionPair:
        return PythonSignatureNativeFunctionPair(
            signature=signature(f, method=method, pyi=pyi),
            function=f,
        )
    # 使用 map 函数将 gen_signature_pairs 应用到 native_functions 列表的每个元素上，
    # 生成包含签名对的列表 pairs
    pairs = list(map(gen_signature_pairs, native_functions))
    
    # 载入包含废弃签名信息的文件，生成废弃签名列表 deprecated
    deprecated = load_deprecated_signatures(
        pairs, deprecated_yaml_path, method=method, pyi=pyi
    )
    
    # 如果 skip_deprecated 为真，则直接返回 pairs 列表；否则返回 pairs 列表与 deprecated 列表的合并结果
    return pairs if skip_deprecated else pairs + deprecated
# 加载被废弃的签名信息，并将其与原始 ATen 签名进行整合，以生成完整的 Python 签名。
def load_deprecated_signatures(
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    deprecated_yaml_path: str,
    *,
    method: bool,
    pyi: bool,
) -> list[PythonSignatureNativeFunctionPair]:
    # 废弃的 YAML 文件中缺乏完整的类型信息，需要查找并利用原始的 ATen 签名（将其委派给调用）来生成完整的 Python 签名。
    # 使用类型信息形式，将废弃的签名与原始签名进行合并。
    
    # 按名称将原始 ATen 签名分组
    grouped: dict[str, list[PythonSignatureNativeFunctionPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.signature.name].append(pair)

    # 查找每个废弃签名对应的原始签名
    results: list[PythonSignatureNativeFunctionPair] = []

    with open(deprecated_yaml_path) as f:
        deprecated_defs = yaml.load(f, Loader=YamlLoader)

    return results


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                         Named Tuple Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@with_native_function
# 生成用于结构化元组类型名称的键
def gen_structseq_typename_key(f: NativeFunction) -> str:
    name = cpp.name(f.func)
    fieldnames = structseq_fieldnames(f.func.returns)
    return "_".join([name] + fieldnames)


# 生成结构化元组调用的代码块
def emit_structseq_call(
    overloads: Sequence[PythonSignatureNativeFunctionPair],
) -> tuple[list[str], dict[str, str]]:
    """
    生成命名元组类型定义的初始化块，并将用到它们的声明添加类型引用片段
    """
    typenames: dict[
        str, str
    ] = {}  # 映射唯一名称和字段名称列表到类型定义名称
    typedefs: list[str] = []  # 类型定义声明和初始化代码

    for overload in overloads:
        fieldnames = structseq_fieldnames(overload.function.func.returns)
        if not fieldnames:
            continue

        name = cpp.name(overload.function.func)
        tn_key = gen_structseq_typename_key(overload.function)
        typename = typenames.get(tn_key)
        if typename is None:
            typename = f'NamedTuple{"" if not typedefs else len(typedefs)}'
            typenames[tn_key] = typename
            typedefs.append(
                f"""\
static PyTypeObject* {typename} = generated::get_{name}_structseq();"""
            )

    return typedefs, typenames


# 生成返回类型定义和在文件中注册的代码块
def generate_return_type_definition_and_registrations(
    overloads: Sequence[PythonSignatureNativeFunctionPair],
) -> tuple[list[str], list[str]]:
    """
    生成 `python_return_types.cpp` 文件中用于初始化和返回命名元组的本地函数的块，
    以及在同一文件中进行的注册调用。
    """
    typenames: dict[
        str, str
    ] = {}  # 映射唯一名称和字段名称列表到类型定义名称
    definitions: list[str] = []  # 函数定义用于注册类型定义
    registrations: list[str] = []  # 创建一个空列表registrations，用于存储typedef的注册信息

    for overload in overloads:
        fieldnames = structseq_fieldnames(overload.function.func.returns)
        # 获取当前overload函数的返回类型的字段名列表

        if not fieldnames:
            continue
        # 如果字段名列表为空，则跳过当前循环

        fields = ", ".join(f'{{"{fn}", ""}}' for fn in fieldnames)
        # 将字段名列表转换成字符串形式，每个字段名作为字典的键，并使用空字符串作为值

        name = cpp.name(overload.function.func)  # 获取overload函数对应的C++函数名
        tn_key = gen_structseq_typename_key(overload.function)
        # 生成structseq类型名的键，用于标识当前overload函数的类型

        typename = typenames.get(tn_key)
        # 从typenames字典中获取当前overload函数类型的名称

        if typename is None:
            # 如果typename为None，表示当前overload函数类型未注册过

            typename = f'{name}NamedTuple{"" if not definitions else len(definitions)}'
            # 构造新的类型名称，类似于"nameNamedTupleX"的形式，其中X是已注册过的数量

            typenames[tn_key] = typename
            # 将新生成的typename注册到typenames字典中

            definitions.append(
                f"""\
static PyObject * ${pycname}(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  // ${name} 方法的 Python 绑定实现
  ${method_header}
  // 创建 PythonArgParser 对象，用于解析 Python 参数
  static PythonArgParser parser({
    ${signatures}
  }, /*traceable=*/${traceable});

  // 存储解析后的参数
  ParsedArgs<${max_args}> parsed_args;
  // 执行参数解析过程
  auto _r = parser.parse(${self_}, args, kwargs, parsed_args);
  // 检查是否存在 torch 函数重载
  ${check_has_torch_function}
  // 根据解析结果进行分发
  switch (_r.idx) {
    ${dispatch}
  }
  ${method_footer}
}



case ${overload_index}: {
  // ${name} 方法的单个重载实现
  ${body}
}



// ${name}


注释：
// 生成用于 Python 绑定的方法实现函数，接收多个参数，支持关键字参数
static PyObject * ${pycname}(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  // 定义 Python 参数解析器对象，初始化支持的签名列表和跟踪信息
  static PythonArgParser parser({
    ${signatures}
  }, /*traceable=*/${traceable});

  // 定义用于存储解析后参数的结构体对象
  ParsedArgs<${max_args}> parsed_args;
  // 调用解析器对象的解析方法，解析 Python 传入的参数
  auto _r = parser.parse(${self_}, args, kwargs, parsed_args);
  // 检查是否存在 Torch 函数重载，用于跟踪调试
  ${check_has_torch_function}
  // 根据参数解析结果进行函数分发调用
  ${dispatch}
  // 方法执行结束的清理工作，如异常处理
  ${method_footer}
}

"""
)

// 用于无参数的方法 Python 绑定，简化参数解析过程
PY_VARIABLE_METHOD_NOARGS = CodeTemplate(
    """\
// ${name}
static PyObject * ${pycname}(PyObject* self_, PyObject* args)
{
  // 方法执行开始的头部
  ${method_header}
  // 检查是否存在 Torch 函数重载，用于跟踪调试
  ${check_has_torch_function}
  // 根据参数进行函数分发调用
  ${dispatch}
  // 方法执行结束的清理工作，如异常处理
  ${method_footer}
}

"""
)


def method_impl(
    name: BaseOperatorName,
    module: str | None,
    overloads: Sequence[PythonSignatureNativeFunctionPair],
    *,
    method: bool,
    symint: bool = True,
) -> str:
    """
    为一个操作符生成所有重载的 Python 绑定。
    """
    pycname = get_pycname(name)
    noarg = is_noarg(overloads)
    // 生成结构序列的初始化代码和类型名
    structseq_inits, structseq_typenames = emit_structseq_call(overloads)

    // 方法的头部代码初始化，包含错误处理的开始
    method_header = ["HANDLE_TH_ERRORS"]
    method_header += structseq_inits
    // 如果是成员方法，则解包 self_ 参数为 Tensor 对象
    method_header += (
        ["const Tensor& self = THPVariable_Unpack(self_);"] if method else []
    )

    // 方法的尾部代码，如果没有参数则返回 None，结束错误处理
    method_footer = ([] if noarg else ["Py_RETURN_NONE;"]) + ["END_HANDLE_TH_ERRORS"]

    // 根据是否需要跟踪调试，设置跟踪信息为 true 或 false
    traceable = "true" if all(should_trace(o.function) for o in overloads) else "false"

    // 根据重载函数的分组，生成不同签名的 Python 方法和函数分发逻辑
    grouped_overloads: Sequence[PythonSignatureGroup] = group_overloads(
        overloads, symint=symint
    )
    is_singleton = len(grouped_overloads) == 1
    signatures: list[str] = []
    dispatch: list[str] = []
    for overload_index, overload in enumerate(grouped_overloads):
        signature = overload.signature.signature_str(symint=symint)
        // 将每个重载的签名转换为 C++ 字符串形式，并添加到签名列表中
        signatures.append(f"{cpp_string(str(signature))},")
        // 生成特定重载函数的分发逻辑，并添加到分发列表中
        dispatch_body = emit_dispatch_case(overload, structseq_typenames, symint=symint)
        dispatch.append(
            PY_VARIABLE_CASE.substitute(
                overload_index=overload_index, body=dispatch_body
            )
            if not is_singleton
            else dispatch_body
        )

    // 根据参数是否为空、是否为单例选择不同的模板进行 Python 绑定生成
    if noarg:
        template = PY_VARIABLE_METHOD_NOARGS
    elif is_singleton:
        template = PY_VARIABLE_METHOD_VARARGS_SINGLETON
    else:
        template = PY_VARIABLE_METHOD_VARARGS

    // 返回生成的 Python 绑定代码
    return template.substitute(
        name=name,
        pycname=pycname,
        method_header=method_header,
        max_args=max(o.signature.arguments_count() for o in overloads),
        signatures=signatures,
        traceable=traceable,
        check_has_torch_function=gen_has_torch_function_check(
            name=name,
            module=module,
            noarg=noarg,
            method=method,
        ),
        dispatch=dispatch,
        method_footer=method_footer,
        self_="self_" if method else "nullptr",
    )
    # 如果 noarg 参数为真（即非空），进入条件判断
    if noarg:
        # 如果 method 参数存在且为真（即非空字符串），进入条件判断
        if method:
            # 返回一个空的格式化字符串
            return f"""\
# 如果 self_ 拥有 torch function，调用 handle_torch_function 处理，并返回结果
if(check_has_torch_function(self_)) {{
  return handle_torch_function(self_, "{name}");
}}
"""
        else:
            return ""

# 根据方法是否存在确定 self_ 的值，nullptr 表示空指针
    self_ = "self_" if method else "nullptr"

    # 根据模块选择相应的命名空间
    namespace = (
        {
            "torch": "THPVariableFunctionsModule",
            "torch.nn": "THPNNVariableFunctionsModule",
            "torch.fft": "THPFFTVariableFunctionsModule",
            "torch.linalg": "THPLinalgVariableFunctionsModule",
            "torch.nested": "THPNestedVariableFunctionsModule",
            "torch.sparse": "THPSparseVariableFunctionsModule",
            "torch.special": "THPSpecialVariableFunctionsModule",
        }[module]
        if module
        else "THPVariableClass"
    )

    # 返回根据模块是否存在的情况下的格式化字符串
    return f"""\
if(_r.has_torch_function()) {{
  return handle_torch_function(_r, {self_}, args, kwargs, {namespace}, "{module or "torch.Tensor"}");
}}
"""


# 处理带输出和无输出重载对的代码模板
PY_VARIABLE_OUT = CodeTemplate(
    """\
if (_r.isNone(${out_idx})) {
  ${call_dispatch}
} else {
  ${call_dispatch_out}
}
"""
)


def emit_dispatch_case(
    overload: PythonSignatureGroup,
    structseq_typenames: dict[str, str],
    *,
    symint: bool = True,
) -> str:
    """
    生成单个解析签名的调度代码。对应于单个本地函数，或者仅在输出参数不同的情况下。在后一种情况下，使用单个Python签名来处理调度，并切换到传递的输出参数的存在/不存在。
    """
    if overload.outplace is not None:
        # 调度输出和无输出变体，根据 _r.isNone(<out_idx>) 分支
        return PY_VARIABLE_OUT.substitute(
            out_idx=overload.signature.output_idx(),
            call_dispatch=emit_single_dispatch(
                overload.signature, overload.base, structseq_typenames, symint=symint
            ),
            call_dispatch_out=emit_single_dispatch(
                overload.signature,
                overload.outplace,
                structseq_typenames,
                symint=symint,
            ),
        )
    else:
        # 只有无输出版本
        return emit_single_dispatch(
            overload.signature, overload.base, structseq_typenames, symint=symint
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                    Forward Declarations Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def forward_decls(
    name: BaseOperatorName,
    overloads: Sequence[PythonSignatureNativeFunctionPair],
    *,
    method: bool,
) -> tuple[str, ...]:
    """
    生成前向声明的代码。如果是方法，则返回空元组。

    Args:
    - name: 操作符名称
    - overloads: PythonSignatureNativeFunctionPair 序列
    - method: 是否是方法

    Returns:
    - tuple[str, ...]: 包含生成的前向声明代码的元组
    """
    if method:
        return ()

    pycname = get_pycname(name)
    if is_noarg(overloads):
        return (
            f"""\
static PyObject * {pycname}(PyObject* self_, PyObject* args);
""",
        )
    else:
        return (
            f"""\
static PyObject * {pycname}(PyObject* self_, PyObject* args, PyObject* kwargs);
""",
        )
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#              Method Def (Binding Table Entry) Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def method_def(
    name: BaseOperatorName,
    module: str | None,
    overloads: Sequence[PythonSignatureNativeFunctionPair],
    *,
    method: bool,
) -> str:
    """
    Generate method def entry.
    """
    # 根据操作符名获取其对应的 Python/C API 函数名
    pycname = get_pycname(name)

    if name.dunder_method:
        # 如果是双下划线方法（如 __add__），生成一个抛出未实现错误的 PyMethodDef 条目
        pycname = f"TypeError_to_NotImplemented_<{pycname}>"

    if is_noarg(overloads):
        # 如果没有参数的重载函数，设置方法标志为 METH_NOARGS，否则为 METH_VARARGS | METH_KEYWORDS
        flags = "METH_NOARGS" if method else "METH_VARARGS | METH_KEYWORDS"
    else:
        # 如果有参数的重载函数，将函数指针转换为带关键字参数的 PyCFunction 对象
        pycname = f"castPyCFunctionWithKeywords({pycname})"
        flags = "METH_VARARGS | METH_KEYWORDS"

    if module == "torch":
        # 如果模块是 torch，添加 METH_STATIC 标志到 flags 中
        flags += " | METH_STATIC"

    # 返回生成的 PyMethodDef 条目字符串
    return f'{{"{name}", {pycname}, {flags}, NULL}},'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                   Overload Sorting and Grouping
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def group_overloads(
    overloads: Sequence[PythonSignatureNativeFunctionPair], *, symint: bool = True
) -> Sequence[PythonSignatureGroup]:
    bases: dict[str, PythonSignatureNativeFunctionPair] = {}
    outplaces: dict[str, PythonSignatureNativeFunctionPair] = {}

    # 首先按照函数签名（忽略输出参数）对重载函数进行分组
    for overload in overloads:
        sig = overload.signature.signature_str(skip_outputs=True, symint=symint)
        if overload.function.func.is_out_fn():
            # 如果是输出函数，则将其存储在 outplaces 字典中
            if sig in outplaces:
                raise RuntimeError(
                    f"Found duplicated function definition:\n- {overload.function.func}.\n"
                    f"Existing definition:\n- {outplaces[sig].function.func}."
                )
            outplaces[sig] = overload
        else:
            # 否则存储在 bases 字典中
            if sig in bases:
                raise RuntimeError(
                    f"Found duplicated function definition:\n- {overload.function.func}.\n"
                    f"Existing definition:\n- {bases[sig].function.func}."
                )
            bases[sig] = overload
    # 遍历输出位置字典中的每个项目，sig为键，out为值
    for sig, out in outplaces.items():
        # 如果sig不在基础函数字典bases中
        if sig not in bases:
            # 初始化候选列表
            candidates: list[str] = []
            # 遍历所有重载函数列表overloads
            for overload in overloads:
                # 如果重载函数的名称与输出函数名称匹配，并且重载函数不是输出函数，且签名未被弃用
                if (
                    str(overload.function.func.name.name)
                    == str(out.function.func.name.name)
                    and not overload.function.func.is_out_fn()
                    and not overload.signature.deprecated
                ):
                    # 将符合条件的重载函数签名字符串添加到候选列表中
                    candidates.append(
                        overload.signature.signature_str(
                            skip_outputs=True, symint=symint
                        )
                    )
            # 获取输出函数的签名字符串
            out_sig = out.signature.signature_str(symint=symint)
            # 抛出运行时异常，指示找到一个输出模式(out schema)没有对应的非输出变体
            raise RuntimeError(
                f"While identifying overloads, we found an out schema {out_sig} without a corresponding non-out variant. "
                f"We expected the non-out variant to have schema: \n- {sig}\nPlease check that you spelled the schema "
                "correctly in native_functions.yaml. We discovered the following candidate(s): \n"
                + "\n".join(f"- {candidate}" for candidate in candidates)
            )

    # 根据基础函数字典bases中的每个项，创建PythonSignatureGroup对象，并从输出位置字典outplaces获取相应的输出函数
    grouped = [
        PythonSignatureGroup.from_pairs(
            functional=base,
            out=outplaces.get(sig),
        )
        for sig, base in bases.items()
    ]
    # 返回排序后的重载组列表
    return sort_overloads(grouped, symint=symint)
# This function declares a partial order on declarations, and sorts them according
# to its linear extension. This is necessary, because there's some ambiguity in the
# choice of overload, and we want a different order.
#
# See Note[Order of overloads matters]
#
# A few examples of ambiguous python signature pairs.
#
#   All parameters have the same type, except one taking Tensor the other taking
#   Scalar. A numeric PyObject can be casted into Tensor, and a zero-dim Tensor
#   object can be accepted as Scalar type parameter (see python_arg_parser.cpp).
#   Therefore, same input arguments might be accepted by either python signature.
#   We want to always parse the one taking Tensor first.
#
#     bitwise_and(Tensor input, Tensor other, *, Tensor out=None)
#     bitwise_and(Tensor input, Scalar other, *, Tensor out=None)
#
#   If they have different number of parameters then they are not ambiguous - but
#   the difference on output param can be ignored as it's optional.
#
#     multiply(Tensor input, Tensor other, *, Tensor out=None)
#     multiply(Tensor input, Scalar other)
#
#   Both positional args and keyword-only args are considered together.
#
#     subtract(Tensor other, *, Scalar alpha=1)
#     subtract(Scalar other, Scalar alpha=1)
#
# A few ambiguous cases which it does NOT handle yet.
#
#   If there is any difference in other parameters besides the Tensor/Scalar
#   difference, then they are not considered ambiguous by this method anymore.
#   However, the difference could be too trivial to disambiguate.
#
#     foo(Tensor input, Scalar other, Scalar bar)
#     foo(Tensor input, Tensor other, double bar)
#
#   If they are taking different number of parameters then they are not considered
#   ambiguous anymore, even if the difference is only on optional kwargs.
#
#     foo(Scalar other, Scalar alpha=1)
#     foo(Tensor other, *, Scalar alpha=1, Scalar beta=1)

def sort_overloads(
    grouped_overloads: Sequence[PythonSignatureGroup], *, symint: bool = True
) -> Sequence[PythonSignatureGroup]:
    # NB: Smaller here means lower priority
    # Placeholder function body; actual implementation of overload sorting is not provided in this snippet.
    pass
    # 定义一个函数，用于比较两个类型是否满足特定的小于关系
    def is_arg_smaller(t1: Type, t2: Type) -> bool:
        return (
            str(t1) == "Scalar"  # 如果 t1 是 Scalar
            and str(t2) == "Tensor"  # 并且 t2 是 Tensor
            or str(t1) == "Scalar?"  # 或者 t1 是 Scalar?
            and str(t2) == "Tensor?"  # 并且 t2 是 Tensor?
            or "Dimname" in str(t1)  # 或者 t1 包含 "Dimname"
            and "Dimname" not in str(t2)  # 并且 t2 不包含 "Dimname"
            or
            # 在讨论 https://github.com/pytorch/pytorch/issues/54555 中讨论了为什么优先处理 int/int? 而不是 int[]
            str(t1) == "int[]"  # 如果 t1 是 int[]
            and (str(t2) == "int" or str(t2) == "int?")  # 并且 t2 是 int 或 int?
            or
            # TensorList 在参数解析时会抛出错误，因此它需要放在签名排序的最后。参见讨论：https://github.com/pytorch/pytorch/issues/58087
            str(t1) == "Tensor[]"  # 如果 t1 是 Tensor[]
            and str(t2).find("[]") != -1  # 并且 t2 包含 "[]"
            or
            # 优先处理 SymInt[] 的重载，而不是 int[]
            str(t1) == "SymInt[]"  # 如果 t1 是 SymInt[]
            and str(t2) == "int[]"  # 并且 t2 是 int[]
            or
            # 确保在 int、SymInt 与 Tensor 之间有一致的排序，因为 Tensor 可以隐式转换为 int 或 SymInt。优先处理 Tensor 的重载以避免被遮蔽。
            (str(t1) == "SymInt" or str(t1) == "int")  # 如果 t1 是 SymInt 或 int
            and str(t2) == "Tensor"  # 并且 t2 是 Tensor
        )

    # 定义一个函数，判断两个 PythonSignature 是否在部分顺序上 s1 < s2
    def is_smaller(s1: PythonSignature, s2: PythonSignature) -> bool:
        """Returns True if s1 < s2 in the partial order."""
        # 获取两个签名的参数列表（排除输出参数）
        args1, args2 = s1.arguments(skip_outputs=True), s2.arguments(skip_outputs=True)
        if len(args1) != len(args2):
            return False
        # 检查所有参数的类型是否相等
        equal = all(arg1.type == arg2.type for arg1, arg2 in zip(args1, args2))
        # 检查是否存在部分顺序上的小于等于关系
        smaller_or_equal = all(
            str(arg1.type) == str(arg2.type) or is_arg_smaller(arg1.type, arg2.type)
            for arg1, arg2 in zip(args1, args2)
        )
        return smaller_or_equal and not equal

    # 根据函数签名进行排序
    grouped_overloads = sorted(
        grouped_overloads, key=lambda x: x.signature.signature_str(symint=symint)
    )

    # 构建关系图
    larger_than: dict[int, set[int]] = defaultdict(set)
    for i1, overload1 in enumerate(grouped_overloads):
        for i2, overload2 in enumerate(grouped_overloads):
            # 如果 overload1 的签名小于 overload2 的签名，则在 larger_than 中建立连接
            if is_smaller(overload1.signature, overload2.signature):
                larger_than[i1].add(i2)

    if not larger_than:
        return list(grouped_overloads)

    # 使用拓扑排序根据部分顺序对重载进行排序
    N = len(grouped_overloads)
    sorted_ids: list[int] = list(filter(lambda x: x not in larger_than, range(N)))
    # 遍历从 0 到 N-1 的索引，其中 N 是 sorted_ids 的长度
    for idx in range(N):
        # 从 sorted_ids 中获取索引为 idx 的元素 i
        i = sorted_ids[idx]
        # 遍历 larger_than 字典中键按顺序排序的所有键 j
        for j in sorted(larger_than.keys()):
            # 获取键 j 对应的值 larger
            larger = larger_than[j]
            # 从 larger 集合中移除元素 i
            larger.discard(i)
            # 如果 larger 集合变为空集
            if not larger:
                # 则从 larger_than 字典中删除键为 j 的项
                del larger_than[j]
                # 将 j 添加到 sorted_ids 列表末尾
                sorted_ids.append(j)

    # 根据 sorted_ids 中的元素顺序，构建结果列表并返回
    return [grouped_overloads[x] for x in sorted_ids]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                       Codegen API Integration
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# 定义函数 emit_single_dispatch，生成单个原生函数的分发代码
def emit_single_dispatch(
    ps: PythonSignature,  # 参数 ps 是 PythonSignature 类型，用于描述 Python 函数签名
    f: NativeFunction,    # 参数 f 是 NativeFunction 类型，表示原生函数对象
    structseq_typenames: dict[str, str],  # 参数 structseq_typenames 是字典类型，映射结构序列类型名
    *,
    symint: bool = True,  # 关键字参数 symint，默认为 True，控制是否支持符号整数
) -> str:
    """
    Emit dispatch code for a single native function.
    """

    # 嵌套函数定义，使用装饰器 with_native_function
    @with_native_function
        # 定义函数 `go`，参数为 `f`，返回类型为字符串
        def go(f: NativeFunction) -> str:
            # 如果 `ps` 是 `PythonSignatureDeprecated` 类型
            if isinstance(ps, PythonSignatureDeprecated):
                # 使用过时模式的注释，格式为 "// [deprecated] aten::" 后接 `ps.deprecated_schema`
                schema_comment = f"// [deprecated] aten::{ps.deprecated_schema}"
            else:
                # 否则，使用正常函数的注释，格式为 "// aten::" 后接 `f.func`
                schema_comment = f"// aten::{f.func}"

            # 如果 `ps` 被标记为过时，则在注释中加上 "[deprecated] "
            deprecated = "[deprecated] " if ps.deprecated else ""

            # 生成调度 lambda 函数的签名
            name = cpp.name(f.func)
            # 将调度 lambda 函数的参数类型和名称用逗号连接成字符串
            lambda_formals = ", ".join(
                f"{a.type_str} {a.name}" for a in dispatch_lambda_args(ps, f, symint=symint)
            )
            # 调度 lambda 函数的返回类型字符串
            lambda_return = dispatch_lambda_return_str(f)

            # 生成调度 lambda 函数的主体
            # 调度的目标函数名
            dispatch_callee = cpp_dispatch_target(f)
            # 生成调度 lambda 函数的参数表达式，用逗号连接
            dispatch_args = ", ".join(cpp_dispatch_exprs(f, python_signature=ps))

            # 将参数解析器的输出转换为调度 lambda 函数的参数
            parser_outputs = arg_parser_output_exprs(ps, f, symint=symint)
            lambda_arg_exprs = dispatch_lambda_exprs(ps, f, symint=symint)
            # 将初始化表达式用换行符连接成字符串
            inits = "\n".join(lambda_arg_exprs.inits)
            # 将调度 lambda 函数的参数表达式用逗号连接成字符串
            lambda_args = ", ".join(lambda_arg_exprs.exprs)

            # 处理 scatter fields 部分
            # TODO: 对于张量方法，检查是否需要设置 'requires_grad' 参数，当前实现方式是一个权宜之计
            need_set_requires_grad = ps.tensor_options_args and (
                not has_tensor_options(f)
                or (ps.method and ("requires_grad" in parser_outputs))
            )
            # 如果需要设置 'requires_grad' 参数，则生成相应的设置语句
            set_requires_grad = (
                f'.set_requires_grad({parser_outputs["requires_grad"].expr})'
                if need_set_requires_grad
                else ""
            )

            # 如果调度 lambda 函数的返回类型为 "void"
            if lambda_return == "void":
                # 在 python-binding 层面，使得原地 foreach 返回 `self`
                # 参考：https://github.com/pytorch/pytorch/pull/118622#pullrequestreview-1904804954
                self_arg = f.func.arguments.self_arg
                return_stmt: str
                # 如果函数名以 "_foreach_" 开头，并且是原地操作的模式
                if (
                    str(f.func.name).startswith("_foreach_")
                    and f.func.kind() == SchemaKind.inplace
                ):
                    # 注意：`_foreach_pow.ScalarAndTensor` 没有其原地操作的变体，未来也不太可能有
                    # 所以可以安全地进行以下断言
                    assert self_arg is not None and is_tensor_list_type(
                        self_arg.argument.type
                    )
                    # 返回语句是一个 CPython 对象指针的定义
                    return_stmt = """PyObject* self_tensorlist = _r.args[0];
"""
根据函数的返回类型生成不同的代码段
"""
            else:
                // 如果返回类型为 None，则返回 Py_RETURN_NONE
                return_stmt = "Py_RETURN_NONE;"
            // 返回生成的 C++ Lambda 函数
            return f"""\
{schema_comment}
{inits}
auto dispatch_{name} = []({lambda_formals}) -> {lambda_return} {{
  pybind11::gil_scoped_release no_gil;
  // 调用 C++ 函数并释放 GIL
  {dispatch_callee}({dispatch_args});
}};
dispatch_{name}({lambda_args}){set_requires_grad};
{return_stmt}
"""
        else:
            // 获取结构化序列的类型名称
            typename = structseq_typenames.get(gen_structseq_typename_key(f))
            // 如果存在类型名称，则生成对应的结构化序列类型引用
            structseq_typeref = f"{typename}, " if typename is not None else ""
            // 返回生成的 C++ Lambda 函数，并封装结果
            return f"""\
{schema_comment}
{inits}
auto dispatch_{name} = []({lambda_formals}) -> {lambda_return} {{
  pybind11::gil_scoped_release no_gil;
  // 调用 C++ 函数并释放 GIL
  return {dispatch_callee}({dispatch_args});
}};
// 包装并返回调用结果
return wrap({structseq_typeref}dispatch_{name}({lambda_args}){set_requires_grad});
"""

    // 执行函数并返回结果
    return go(f)
```