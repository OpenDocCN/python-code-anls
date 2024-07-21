# `.\pytorch\torchgen\gen.py`

```
# 导入必要的模块和类
from __future__ import annotations

import argparse  # 解析命令行参数的模块
import functools  # 函数工具模块，提供了一些高阶函数
import json  # JSON 编解码模块
import os  # 提供了访问操作系统功能的接口
from collections import defaultdict, namedtuple, OrderedDict  # 默认字典、命名元组和有序字典
from dataclasses import dataclass, field  # 用于创建和操作数据类的模块
from pathlib import Path  # 提供了处理路径的类
from typing import Any, Callable, Literal, Sequence, TypeVar  # 类型提示相关的类和函数

import yaml  # YAML 格式处理模块

# 导入其他自定义模块
import torchgen.api.dispatcher as dispatcher  # 调度器 API
import torchgen.api.meta as meta  # 元数据 API
import torchgen.api.native as native  # 原生 API
import torchgen.api.structured as structured  # 结构化 API
import torchgen.dest as dest  # 目标位置
from torchgen.aoti.fallback_ops import inductor_fallback_ops  # AOTI 回退操作
from torchgen.api import cpp  # C++ API
from torchgen.api.translate import translate  # 翻译 API
from torchgen.api.types import (  # 类型相关的导入
    Binding,
    CppSignature,
    CppSignatureGroup,
    DispatcherSignature,
    NamedCType,
    NativeSignature,
    SpecialArgName,
)
from torchgen.context import (  # 上下文相关的导入
    method_with_native_function,
    native_function_manager,
    with_native_function,
    with_native_function_and_indices,
)
from torchgen.gen_aoti_c_shim import (  # AOTI C Shim 生成相关的导入
    gen_aoti_c_shim,
    gen_static_dispatch_backend_call_signature,
    get_fallback_op_name,
    get_header_for_aoti,
)
from torchgen.gen_functionalization_type import (  # 函数化类型生成相关的导入
    gen_functionalization_definition,
    gen_functionalization_registration,
    gen_functionalization_view_inverse_declaration,
    GenCompositeViewCopyKernel,
)
from torchgen.gen_vmap_plumbing import gen_all_vmap_plumbing  # VMap 布线生成
from torchgen.model import (  # 模型相关的导入
    Argument,
    BackendIndex,
    BackendMetadata,
    BaseOperatorName,
    DEFAULT_KERNEL_NAMESPACE,
    DispatchKey,
    FRAGMENT_NAMESPACES,
    FunctionSchema,
    is_cuda_dispatch_key,
    is_generic_dispatch_key,
    is_ufunc_dispatch_key,
    Location,
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
    OperatorName,
    OptionalType,
    SchemaKind,
    SelfArgument,
    STRUCTURED_DISPATCH_KEYS,
    TensorOptionsArguments,
    Type,
    Variant,
    ViewSchemaKind,
)
from torchgen.native_function_generation import (  # 原生函数生成相关的导入
    add_generated_native_functions,
    gen_composite_functional_kernel,
    gen_composite_out_kernel,
    pre_group_native_functions,
)
from torchgen.selective_build.selector import SelectiveBuilder  # 选择性构建器
from torchgen.utils import (  # 实用工具相关的导入
    assert_never,
    concatMap,
    context,
    FileManager,
    make_file_manager,
    mapMaybe,
    NamespaceHelper,
    Target,
)
from torchgen.yaml_utils import YamlDumper, YamlLoader  # YAML 格式处理工具

T = TypeVar("T")  # 泛型类型变量 T

# 欢迎来到 ATen 代码生成器 v2！ATen 代码生成器负责解析 native_functions.yaml，并基于此文件定义的运算符生成各种生成文件（例如 TypeDefault.cpp）。
# 这意味着代码生成器知道如何解析函数模式，然后将其转换为各种 C++ 类型和样板代码。
#
# 修改此文件时需注意的一些事项：
#
# - 该文件进行了严格的 mypy 类型检查。在根源代码目录中使用 `mypy --config mypy-strict.ini` 进行类型检查。
#
# - 大部分重要工作都存在于外部模块中：
#   - 'model' 包含了 native_functions.yaml 的数据模型。这些文件中的类表示查看 native_functions.yaml 时看到的内容
#   - 'api' 包含了如何将 JIT schema 转换为代码生成器交互的各种 C++ API 的转换。事实上有三种不同的 C++ API：公共 C++ API、调度程序 API 和旧调度程序 API。请查看各自的文件以获取更多信息

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                         HELPER FUNCTIONS
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# 用于 YAML 的自定义加载器，允许我们同时跟踪 YAML 文件中每个条目的行号
class LineLoader(YamlLoader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        # 调用父类的构造函数来构建映射
        mapping = super().construct_mapping(node, deep=deep)  # type: ignore[no-untyped-call]
        # 添加 1 以使行号从 1 开始
        mapping["__line__"] = node.start_mark.line + 1
        return mapping


# 将 native_functions.yaml 解析为 NativeFunctions 和 Backend Indices 的序列。
ParsedYaml = namedtuple("ParsedYaml", ["native_functions", "backend_indices"])

# 全局缓存，存储解析过的 native_functions.yaml 和 tags.yaml 的内容
_GLOBAL_PARSE_NATIVE_YAML_CACHE: dict[str, ParsedYaml] = {}
_GLOBAL_PARSE_TAGS_YAML_CACHE: dict[str, set[str]] = {}


def parse_native_yaml_struct(
    es: object,
    valid_tags: set[str],
    ignore_keys: set[DispatchKey] | None = None,
    path: str = "<stdin>",
    skip_native_fns_gen: bool = False,
) -> ParsedYaml:
    # 断言输入参数 es 是一个列表
    assert isinstance(es, list)
    rs: list[NativeFunction] = []  # 存储解析后的 NativeFunction 对象
    bs: dict[DispatchKey, dict[OperatorName, BackendMetadata]] = defaultdict(dict)  # 存储后端元数据
    for e in es:
        # 断言每个元素 e 是一个字典
        assert isinstance(e, dict), f"expected to be dict: {e}"
        # 断言字典中包含 '__line__' 键并且其值是整数类型
        assert isinstance(e.get("__line__"), int), e
        # 创建 Location 对象，用于指示在文件中的位置
        loc = Location(path, e["__line__"])
        funcs = e.get("func")
        # 断言 funcs 不为 None，确保字典中包含 'func' 键
        assert funcs is not None, f"missed 'func' in {e}"
        # 使用上下文管理器显示当前解析的函数名称和位置信息
        with context(lambda: f"in {loc}:\n  {funcs}"):
            # 从 YAML 中创建 NativeFunction 对象和元数据
            func, m = NativeFunction.from_yaml(e, loc, valid_tags, ignore_keys)
            rs.append(func)  # 将解析的 NativeFunction 对象添加到 rs 列表中
            BackendIndex.grow_index(bs, m)  # 更新后端索引
    error_check_native_functions(rs)  # 对解析得到的 NativeFunctions 进行错误检查
    # 默认字典用于在没有内核的情况下防止代码生成器报错
    indices: dict[DispatchKey, BackendIndex] = defaultdict(
        lambda: BackendIndex(
            dispatch_key=DispatchKey.Undefined,
            use_out_as_primary=True,
            external=False,
            device_guard=False,
            index={},
        )
    )
    if not skip_native_fns_gen:
        add_generated_native_functions(rs, bs)  # 如果不跳过生成本地函数，添加生成的本地函数到 bs 中
    for k, v in bs.items():
        # 遍历字典 bs 的每个键值对，k 是键，v 是值
        indices[k] = BackendIndex(
            dispatch_key=k,
            use_out_as_primary=True,
            external=False,
            # 只有类似 CUDA 的设备需要设备保护
            device_guard=is_cuda_dispatch_key(k),
            index=v,
        )
    # 返回解析后的 YAML 对象，包含 rs 和 indices
    return ParsedYaml(rs, indices)
# 解析带有结构化标签的 YAML 数据，返回标签集合
def parse_tags_yaml_struct(es: object, path: str = "<stdin>") -> set[str]:
    # 断言 es 是一个列表对象
    assert isinstance(es, list)
    # 初始化结果集合
    rs: set[str] = set()
    
    # 遍历列表中的每个元素 e
    for e in es:
        # 断言每个元素 e 中包含 "__line__" 键且其对应的值是整数类型
        assert isinstance(e.get("__line__"), int), e
        # 构建位置对象 loc，表示在指定路径的指定行数
        loc = Location(path, e["__line__"])
        # 获取标签列表 tags
        tags = e.get("tag")
        
        # 使用上下文管理器输出当前位置和标签信息
        with context(lambda: f"in {loc}:\n  {tags}"):
            # 复制当前元素 e，并弹出键为 "tag" 的值作为 name
            e_i = e.copy()
            name = e_i.pop("tag")
            # 弹出键为 "desc" 的值作为描述 desc
            desc = e_i.pop("desc", "")
            # 确保每个标签具有非空描述
            assert desc != ""
            # 将标签名 name 添加到结果集合 rs 中
            rs.add(name)
    
    # 返回结果集合 rs
    return rs


# 使用 functools 提供的 LRU 缓存装饰器，解析指定路径下的 YAML 文件中的标签信息
@functools.lru_cache(maxsize=None)
def parse_tags_yaml(path: str) -> set[str]:
    # 全局变量，缓存解析后的 YAML 数据
    global _GLOBAL_PARSE_TAGS_YAML_CACHE
    # 如果指定路径不在缓存中
    if path not in _GLOBAL_PARSE_TAGS_YAML_CACHE:
        # 打开指定路径的文件并加载 YAML 数据，使用 LineLoader 作为加载器
        with open(path) as f:
            es = yaml.load(f, Loader=LineLoader)
            # 将解析得到的数据存入缓存
            _GLOBAL_PARSE_TAGS_YAML_CACHE[path] = parse_tags_yaml_struct(es, path=path)

    # 返回缓存中指定路径的结果集合
    return _GLOBAL_PARSE_TAGS_YAML_CACHE[path]


# 解析本地 YAML 文件，生成 ParsedYaml 对象
def parse_native_yaml(
    path: str,
    tags_yaml_path: str,
    ignore_keys: set[DispatchKey] | None = None,
    *,
    skip_native_fns_gen: bool = False,
    loaded_yaml: object | None = None,
) -> ParsedYaml:
    # 全局变量，缓存解析后的本地 YAML 数据
    global _GLOBAL_PARSE_NATIVE_YAML_CACHE
    # 如果指定路径不在缓存中
    if path not in _GLOBAL_PARSE_NATIVE_YAML_CACHE:
        # 解析标签 YAML 文件，获取有效标签集合
        valid_tags = parse_tags_yaml(tags_yaml_path)

        # 如果提供了已加载的 YAML 数据，则使用该数据而不是从文件中读取
        if loaded_yaml is None:
            # 打开指定路径的文件并加载 YAML 数据，使用 LineLoader 作为加载器
            with open(path) as f:
                es = yaml.load(f, Loader=LineLoader)
        else:
            es = loaded_yaml
        
        # 将解析得到的数据存入缓存
        _GLOBAL_PARSE_NATIVE_YAML_CACHE[path] = parse_native_yaml_struct(
            es,
            valid_tags,
            ignore_keys,
            path=path,
            skip_native_fns_gen=skip_native_fns_gen,
        )

    # 返回缓存中指定路径的 ParsedYaml 对象
    return _GLOBAL_PARSE_NATIVE_YAML_CACHE[path]


# 检查本地函数定义中的错误，跨多个 NativeFunction 进行断言检查
def error_check_native_functions(funcs: Sequence[NativeFunction]) -> None:
    # 初始化函数映射字典和基本函数映射字典
    func_map: dict[OperatorName, NativeFunction] = {}
    base_func_map: dict[BaseOperatorName, list[NativeFunction]] = defaultdict(list)
    
    # 遍历函数列表 funcs 中的每个 NativeFunction 对象 f
    for f in funcs:
        # 将函数名 f.func.name 作为键，函数对象 f 作为值存入 func_map 中
        func_map[f.func.name] = f
        # 将函数名的基础操作符名 f.func.name.name 作为键，将函数对象 f 添加到对应列表中
        base_func_map[f.func.name.name].append(f)
    # 遍历函数列表中的每个函数对象
    for f in funcs:
        # 检查当前函数是否有结构化代理
        if f.structured_delegate is not None:
            # 获取结构化代理函数对象
            delegate_func = func_map.get(f.structured_delegate)
            # 断言确保代理函数存在
            assert delegate_func is not None, (
                f"{f.func.name} is marked as a structured_delegate pointing to "
                f"{f.structured_delegate}, but {f.structured_delegate} is missing."
            )
            # 断言确保代理函数标记为结构化
            assert delegate_func.structured, (
                f"{f.func.name} is marked as a structured_delegate pointing to "
                f"{f.structured_delegate}, but {f.structured_delegate} is not marked as structured. "
                f"Consider adding 'structured=True' to the delegated operator"
            )
        
        # 查看笔记 [resize_ in Functionalization]
        # resize_() 是一个就地视图操作（因此需要标记），但添加一个真正的“视图”变体的开销太大。
        # 因此，resize_() 在功能化中得到特殊处理，
        # 我们有一个 resize() 操作，它既不会别名也是功能化的。
        if (
            "inplace_view" in f.tags
            and str(f.func.name) != "resize_"  # 排除 resize_ 函数
            and str(f.func.name) != "resize_as_"  # 排除 resize_as_ 函数
            and str(f.func.name.name) != "set_"  # 排除 set_ 函数
        ):
            # 获取基础函数名
            base_name = f.func.name.name
            # 断言确保基础函数名是就地操作的命名约定
            assert base_name.inplace, (
                f"{f.func.name} is marked with tag: inplace_view, but it doesn't follow the naming "
                "convention for inplace ops - the codegen expects the base name to have a trailing underscore. "
            )
            # 创建非就地操作的基础函数名对象
            out_of_place_base_name = BaseOperatorName(
                base_name.base, False, base_name.dunder_method
            )
            # 断言确保存在对应的非就地视图操作
            assert len(base_func_map[out_of_place_base_name]) > 0, (
                f"{f.func.name} is marked with tag: inplace_view. The codegen expects there to be a corresponding "
                f"out-of-place view op with the name '{base_name}' and matching schema, but it didn't find one. "
            )
# 将 Python 字符串转换为 C++ 字符串字面值
def cpp_string(s: str) -> str:
    s = s.replace("\\", "\\\\")    # 转义反斜杠为双反斜杠
    s = s.replace('"', '\\"')      # 转义双引号为反斜杠加双引号
    s = s.replace("\a", "\\a")     # 转义响铃符为反斜杠加'a'
    s = s.replace("\b", "\\b")     # 转义退格符为反斜杠加'b'
    s = s.replace("\f", "\\f")     # 转义换页符为反斜杠加'f'
    s = s.replace("\n", "\\n")     # 转义换行符为反斜杠加'n'
    s = s.replace("\v", "\\v")     # 转义垂直制表符为反斜杠加'v'
    s = s.replace("\t", "\\t")     # 转义水平制表符为反斜杠加't'
    return f'"{s}"'                # 返回包装在双引号内的 C++ 字符串字面值


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                        C++ CODE GENERATION
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# 大多数函数都是柯里化的：它们由一个接受某些参数（例如要生成的内容）的函数组成，
# 它本身返回一个将 NativeFunction 映射到生成的代码的函数。
# 这种模式使得使用 map、concatMap 和类似的函数组合器非常方便。

# 返回给定后端列表中的静态调度键列表
def static_dispatch_keys(backends: list[BackendIndex]) -> list[DispatchKey]:
    if len(backends) == 0:
        return []
    else:
        return [backend.dispatch_key for backend in backends] + [
            DispatchKey.CompositeImplicitAutograd,
            DispatchKey.CompositeImplicitAutogradNestedTensor,
            DispatchKey.CompositeExplicitAutograd,
            DispatchKey.CompositeExplicitAutogradNonFunctional,
        ]


# 获取静态调度后端对于给定 NativeFunction 的调度键，如果无法确定则返回 None
def get_static_dispatch_backend(
    f: NativeFunction, backend_index: BackendIndex
) -> DispatchKey | None:
    if f.structured_delegate is not None or backend_index.has_kernel(f):
        # 如果有 structured_delegate 或者后端具有该函数的内核，则返回后端的调度键
        return backend_index.dispatch_key
    elif f.has_composite_explicit_autograd_kernel:
        return DispatchKey.CompositeExplicitAutograd
    elif f.has_composite_explicit_autograd_non_functional_kernel:
        return DispatchKey.CompositeExplicitAutogradNonFunctional
    elif f.has_composite_implicit_autograd_kernel:
        return DispatchKey.CompositeImplicitAutograd
    elif f.has_composite_implicit_autograd_nested_tensor_kernel:
        return DispatchKey.CompositeImplicitAutogradNestedTensor
    return None


# 生成静态调度操作的头文件包含内容
def static_dispatch_ops_header(
    f: NativeFunction, backend_index: list[BackendIndex]
) -> str | None:
    if backend_index is None or f.manual_kernel_registration:
        return None

    output = []
    for index in backend_index:
        dispatch_key = get_static_dispatch_backend(f, index)
        if dispatch_key is not None:
            output.append(
                f"#include <ATen/ops/{f.root_name}_{dispatch_key.lower()}_dispatch.h>"
            )
    return "\n".join(output)


# 返回额外的静态调度头文件包含列表
def static_dispatch_extra_headers(backends: list[BackendIndex]) -> list[str]:
    # 返回一个列表，列表的每个元素都是一个字符串，形如 "#include <ATen/{dispatch_key}Functions.h>"
    # 其中 dispatch_key 是由 static_dispatch_keys(backends) 返回的静态分发键
    return [
        f"#include <ATen/{dispatch_key}Functions.h>"
        for dispatch_key in static_dispatch_keys(backends)
    ]
# 将 `sig` 的参数翻译为 CppSignature 绑定。
# 注意，对于 `memory_format` 参数有一个特殊情况，并且此情况尚未被 tools.codegen.api.translate() 覆盖，
# 因为其应用仅限于静态调度。
def translate_args(
    sig: CppSignature | DispatcherSignature,
    cpp_sig: CppSignature,
) -> str:
    # 添加用于 memory_format 绑定的 SpecialArgName.possibly_redundant_memory_format NamedCType
    def add_spl_memory_format_binding(input_bindings: list[Binding]) -> list[Binding]:
        output_bindings: list[Binding] = []
        for binding in input_bindings:
            if binding.name == "memory_format":
                # 创建一个特殊的 memory_format 绑定
                spl_mem_format_binding = Binding(
                    nctype=NamedCType(
                        SpecialArgName.possibly_redundant_memory_format,
                        binding.nctype.type,
                    ),
                    name=binding.name,
                    default=binding.default,
                    argument=binding.argument,
                )
                output_bindings.append(spl_mem_format_binding)
            else:
                output_bindings.append(binding)
        return output_bindings

    # 获取 sig 的参数列表
    src_bindings = list(sig.arguments())
    # 获取 cpp_sig 的参数列表
    goal_bindings = list(cpp_sig.arguments())
    
    # 如果 CPP 签名的最后一个参数具有 SpecialArgName.possibly_redundant_memory_format NCType，
    # 则将调度器签名的 memory_format 绑定也设置为相同的 NCType
    for arg in goal_bindings:
        if arg.nctype.name == SpecialArgName.possibly_redundant_memory_format:
            src_bindings = add_spl_memory_format_binding(src_bindings)
            break
    
    # 翻译参数表达式以匹配目标签名
    exprs = translate(src_bindings, goal_bindings)
    
    # 返回逗号分隔的表达式字符串
    return ", ".join(a.expr for a in exprs)


# 生成静态调度后端调用的字符串表示
def generate_static_dispatch_backend_call(
    sig: CppSignature | DispatcherSignature,
    f: NativeFunction,
    backend_index: BackendIndex,
) -> str:
    # 生成静态调度后端调用的签名
    cpp_sig = gen_static_dispatch_backend_call_signature(sig, f)
    # 获取函数名
    name = cpp_sig.name()
    # 翻译参数表达式
    exprs = translate_args(sig, cpp_sig)
    # 获取后端元数据
    backend_metadata = backend_index.get_kernel(f)
    # 获取内核命名空间，如果不存在则使用默认命名空间
    kernel_ns = (
        backend_metadata.cpp_namespace
        if backend_metadata and backend_metadata.cpp_namespace
        else DEFAULT_KERNEL_NAMESPACE
    )
    # 去除 "::native" 后的命名空间部分
    ns = kernel_ns.replace("::native", "")
    # 返回调用后端函数的字符串表示
    return f"return {ns}::{backend_index.dispatch_key.lower()}::{name}({exprs});"


# 生成静态调度回退调用的字符串表示
def generate_static_dispatch_fallback_call(
    sig: CppSignature | DispatcherSignature,
    f: NativeFunction,
    backend_indices: list[BackendIndex],
) -> str:
    # 从原生函数生成 CppSignatureGroup
    cpp_sigs = CppSignatureGroup.from_native_function(
        f, method=False, fallback_binding=False
    )
    # 根据情况选择合适的 CPP 签名
    if sig.symint and f.func.has_symint():
        cpp_sig = cpp_sigs.symint_signature
    else:
        cpp_sig = cpp_sigs.signature
    # 确保 cpp_sig 不为空
    assert cpp_sig is not None
    # 获取函数名
    name = cpp_sig.name()
    # 翻译参数表达式
    exprs = translate_args(sig, cpp_sig)
    # 获取默认内核命名空间并去除 "::native" 后的部分
    ns = DEFAULT_KERNEL_NAMESPACE.replace("::native", "")
    # 检查函数对象 `f` 是否具有复合显式自动微分内核
    if f.has_composite_explicit_autograd_kernel:
        # 返回一个格式化字符串，调用具有复合显式自动微分的函数
        return f"return {ns}::{DispatchKey.CompositeExplicitAutograd.lower()}::{name}({exprs});"
    # 检查函数对象 `f` 是否具有复合显式自动微分非功能内核
    elif f.has_composite_explicit_autograd_non_functional_kernel:
        # 返回一个格式化字符串，调用具有复合显式自动微分非功能的函数
        return f"return {ns}::{DispatchKey.CompositeExplicitAutogradNonFunctional.lower()}::{name}({exprs});"
    # 检查函数对象 `f` 是否具有复合隐式自动微分内核
    elif f.has_composite_implicit_autograd_kernel:
        # 返回一个格式化字符串，调用具有复合隐式自动微分的函数
        return f"return {ns}::{DispatchKey.CompositeImplicitAutograd.lower()}::{name}({exprs});"
    # 检查函数对象 `f` 是否具有复合隐式自动微分嵌套张量内核
    elif f.has_composite_implicit_autograd_nested_tensor_kernel:
        # 返回一个格式化字符串，调用具有复合隐式自动微分嵌套张量的函数
        return f"return {ns}::{DispatchKey.CompositeImplicitAutogradNestedTensor.lower()}::{name}({exprs});"
    else:
        # 如果没有上述情况，返回一个包含错误消息的格式化字符串
        return f"""TORCH_CHECK(false, "Static dispatch does not support {name} for\
# 构造一个字符串，其中包含所有 backend_indices 的 dispatch_key 属性的字符串形式，用逗号和空格分隔
{', '.join([str(index.dispatch_key) for index in backend_indices])} ");"""

def static_dispatch(
    sig: CppSignature | DispatcherSignature,
    f: NativeFunction,
    backend_indices: list[BackendIndex],
) -> str:
    """
    根据给定的 `NativeFunction`，找到对应的后端并进行静态调度。如果存在多个后端，通过输入确定 dispatch key 进行静态调度。
    参数:
        sig: 这个本地函数的 CppSignature 或 DispatcherSignature。
        f: 生成静态调度的 NativeFunction。
        backend_indices: 所有可用的后端列表。
    返回:
        调用特定后端函数的 C++ 代码，例如，"return at::cpu::add(self, other, scale);"
    """
    if len(backend_indices) == 0 or f.manual_kernel_registration:
        return ""

    # 根据条件过滤出具有内核的后端或符合结构化调度键的后端
    keys = [
        b
        for b in backend_indices
        if b.has_kernel(f)
        or (
            f.structured_delegate is not None
            and b.dispatch_key in STRUCTURED_DISPATCH_KEYS
        )
    ]

    if len(keys) == 1:
        # 如果只有一个后端符合条件，则生成调用该后端的静态调度代码
        return generate_static_dispatch_backend_call(sig, f, keys[0])
    elif len(keys) == 0:
        # 如果没有后端符合条件，则生成回退调度的代码
        return generate_static_dispatch_fallback_call(sig, f, backend_indices)

    # 确定需要作为 tensor 参数传递的本地张量参数
    native_tensor_args = [
        a.name
        for a in sig.arguments()
        if isinstance(a.argument, SelfArgument)
        or isinstance(a.argument, Argument)
        and a.argument.type.is_tensor_like()
    ]
    tensor_args = ", ".join(native_tensor_args)
    tensor_opts = f.func.arguments.tensor_options

    stmts = []
    subexprs: list[str] = []

    if tensor_opts is not None:
        # 如果存在 tensor_opts，则添加相应的子表达式
        subexprs.append(
            "DispatchKeySet(c10::computeDispatchKey(dtype, layout, device))"
        )
    if tensor_args != "":
        # 如果存在 tensor 参数，则添加多重 dispatch key set 子表达式
        subexprs.append(f"c10::detail::multi_dispatch_key_set({tensor_args})")

    # 生成声明 DispatchKeySet 的语句
    stmts.append(f"""DispatchKeySet _dk_set = {' | '.join(subexprs)};""")
    # 生成声明 DispatchKey 的语句
    stmts.append("DispatchKey _dk = c10::highestPriorityBackendTypeId(_dk_set);")

    dispatch_code = []
    for index in keys:
        # 为每个符合条件的后端生成对应的 case 语句
        dispatch_code.append(f"""case DispatchKey::{index.dispatch_key}:""")
        dispatch_code.append(
            f"""\t{generate_static_dispatch_backend_call(sig, f, index)};"""
        )

    # 生成回退调度的代码
    fallback = generate_static_dispatch_fallback_call(sig, f, backend_indices)
    connector = "\n\t\t"

    # 返回最终的 C++ 代码块，包括声明语句、switch 语句和默认回退
    return f"""
    {connector.join(stmts)}
    switch (_dk) {{
        {connector.join(dispatch_code)}
        default:
            {fallback}
    }}
    """

# 生成 RegisterSchema.cpp。根据选择器，要么注册所有模式，要么仅注册部分模式（选择性构建）
@dataclass(frozen=True)
class RegisterSchema:
    selector: SelectiveBuilder
    known_tags: dict[str, int] = field(default_factory=dict)

    @method_with_native_function
    # 定义一个方法 __call__，接受一个 NativeFunction 参数并返回一个字符串或 None
    def __call__(self, f: NativeFunction) -> str | None:
        # 如果选择器未选择当前的原生函数 f，则返回 None
        if not self.selector.is_native_function_selected(f):
            return None
        
        # 构建标签字符串，格式为 "{at::Tag::tag1, at::Tag::tag2, ...}"
        tags = "{" + ", ".join(f"at::Tag::{tag}" for tag in sorted(f.tags)) + "}"
        
        # 如果标签为空集，则返回函数名字符串和空的 m.def() 调用
        if tags == "{}":
            return f"m.def({cpp_string(str(f.func))}, {{}});\n"
        
        # 初始化 maybe_tags 字符串为空
        maybe_tags = ""
        
        # 如果当前标签不在已知标签集合中
        if tags not in self.known_tags:
            # 计算当前标签集合的索引号
            idx = len(self.known_tags)
            # 将当前标签集合和其索引号存入 known_tags 字典中
            self.known_tags[tags] = idx
            # 构建 tags_索引号 的标签数组声明语句
            maybe_tags = f"const std::vector<at::Tag> tags_{idx} = {tags};\n"
        
        # 返回最终的 m.def() 调用语句，包含可能的标签声明
        return f"{maybe_tags}m.def({cpp_string(str(f.func))}, tags_{self.known_tags[tags]});\n"
# 生成 Operators.h 和 Operators.cpp 文件。
# 这些文件提供了宏定义，允许用户根据操作符和重载名称访问“非重载”函数版本的操作符。
# 这对于希望（1）使用 decltype 获取操作符类型和（2）不想担心仅限于方法的操作符的扩展编写者非常有用。
@dataclass(frozen=True)
class ComputeOperators:
    # 表示生成的目标类型，可以是声明（Target.DECLARATION）或定义（Target.DEFINITION）
    target: Literal[Target.DECLARATION, Target.DEFINITION]
    # 静态分发的后端索引列表
    static_dispatch_backend_indices: list[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        # 从函数模式创建调度器签名
        sig = DispatcherSignature.from_schema(f.func)
        # 获取函数的不含歧义的名称
        name = f.func.name.unambiguous_name()

        if self.target is Target.DECLARATION:
            # 注释 [The ATen Operators API]
            # ATen 操作符 API 存在于 at::_ops 命名空间中，包含每个操作符的编译时元数据以及进入调度器的入口点。
            # C++ 函数、方法和重新调度 API 都作为对这里定义的各种结构的包装器实现。
            #
            # 关于操作符 API 的重要特性：
            # (1) 它遵循调度器 API。
            #     这是为了避免开销而必需的。
            #     例如：如果它遵循了 C++ API，则所有忠实的 C++ 工厂函数都需要将其参数包装成 TensorOptions，然后再解包。
            # (2) 重载名称是唯一的。
            #     这对于希望 decltype() 一个有多个重载的 aten 操作符的 PyTorch 扩展者非常有帮助，例如 decltype(at::_ops::mul_Tensor::call)。
            # (3) 不允许参数默认值。
            #     这更多是一个实现细节，以避免 #include 循环，因为定义了 Tensor 类的 TensorBody.h 需要包含这个文件。
            # (4) 手动的 C++ 绑定和忠实的名称不包含在 API 中。
            #     这适用于类似 __dispatch__is_complex() 和 add_outf() 这样的东西。
            #     这些不是真正的 aten 操作符，它们只是由 C++ API 提供的额外函数。
            #     它们作为 Functions.h 中的包装器实现，调用实际上在这里定义的操作符，即 at::_ops::is_complex::call() 和 at::_ops::add_out::call()。
            #     这意味着 ATEN_OP(is_complex) 将不会进行快速路径，而是通过调度器。
            return f"""
// 定义一个结构体 TORCH_API，名称为 {name}
struct TORCH_API {name} {
  // 使用 schema 类型作为别名，其值为 {sig.type()} 的结果
  using schema = {sig.type()};
  // 使用 ptr_schema 类型作为 schema* 的别名
  using ptr_schema = schema*;
  // 针对 Windows NVCC 的注释，见 Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::{f.func.name.name}")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "{f.func.name.overload_name}")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, {cpp_string(str(f.func))})
  
  // 定义名为 call 的静态函数，非 redispatching 类型
  static {sig.defn(name="call", is_redispatching_fn=False)};
  // 定义名为 redispatch 的静态函数，redispatching 类型
  static {sig.defn(name="redispatch", is_redispatching_fn=True)};
};

// 如果 self.target 是 Target.DEFINITION
elif self.target is Target.DEFINITION:
    // 定义一系列静态常量字符串，用于 Windows NVCC
    defns = f"""
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, name, "aten::{f.func.name.name}")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, overload_name, "{f.func.name.overload_name}")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, schema_str, {cpp_string(str(f.func))})

// aten::{f.func}
// 定义名为 create_{name}_typed_handle 的静态函数，返回 c10::TypedOperatorHandle<{name}::schema>
static C10_NOINLINE c10::TypedOperatorHandle<{name}::schema> create_{name}_typed_handle() {{
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow({name}::name, {name}::overload_name)
      .typed<{name}::schema>();
}}

// 遍历 is_redispatching_fn 的值 [False, True]
"""
// 如果是 redispatching 函数
for is_redispatching_fn in [False, True]:
    // 如果是 redispatching 函数，构建调度表达式字符串
    if is_redispatching_fn:
        dispatcher_exprs_str = ", ".join(
            ["dispatchKeySet"] + [a.name for a in sig.arguments()]
        )
        method_base = "redispatch"
    else:
        dispatcher_exprs_str = ", ".join([a.name for a in sig.arguments()])
        method_base = "call"

    dispatcher_call = method_base
    method_name = f"{name}::{method_base}"

    // 构建函数体 fn_body
    fn_body = f"""
static auto op = create_{name}_typed_handle();
return op.{dispatcher_call}({dispatcher_exprs_str});"""

    // 如果不是 redispatching 函数且 self.static_dispatch_backend_indices 的长度大于零
    if (
        not is_redispatching_fn
        and len(self.static_dispatch_backend_indices) > 0
    ):
        // call() 应通过静态分派执行
        fn_body = static_dispatch(
            sig, f, backend_indices=self.static_dispatch_backend_indices
        )
    defns += f"""
// aten::{f.func}
{sig.defn(name=method_name, is_redispatching_fn=is_redispatching_fn)} {{
    {fn_body}
}}
"""
return defns

// 如果 self.target 不是 Target.DEFINITION，则断言不应该发生，应该是 assert_never(self.target)
else:
    assert_never(self.target)
    # 定义一个方法，将其作为对象的调用接口，接受一个 NativeFunction 参数并返回字符串或空值
    def __call__(self, f: NativeFunction) -> str | None:
        # 从 NativeFunction 创建 CppSignatureGroup 对象，用于处理 C++ 的签名组
        sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=f.manual_cpp_binding
        )
        # 检查函数是否具有 symint 特性
        has_symint = f.func.has_symint()

        # 初始化结果字符串
        result = ""
        # 遍历签名组中的每一个签名
        for sig in sig_group.signatures():
            # 创建 DispatcherSignature 对象，从函数的 schema 中获取目标签名
            target_sig = DispatcherSignature.from_schema(f.func)
            # 将当前签名的参数列表翻译成目标签名的参数表达式
            exprs = translate(sig.arguments(), target_sig.arguments())
            # 将参数表达式列表转换为字符串，用逗号分隔
            exprs_str = ", ".join([e.expr for e in exprs])

            # 根据当前签名是否包含 symint 特性来选择 intlike_t 的类型
            if sig.symint:
                intlike_t = "c10::SymInt"
            else:
                intlike_t = "int64_t"

            # 如果函数的 variant 中包含 Variant.function，将下面的内容添加到结果字符串中
            if Variant.function in f.variants:
                result += f"""
"""
// aten::{f.func}
inline {sig.decl()} {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
}}"""

# 生成内联函数，调用 ATen 操作的具体函数，根据签名声明和表达式生成相应的代码块

            # 可以在模板情况下使用模板函数
            # 根据模板参数决定是否使用 symint 版本或非 symint 版本
            #
            # 注意：即使是方法，我们也总是生成这个函数。但我们将其放在
            # 这个头文件中，以便利用每个操作的头文件
            if has_symint:
                result += f"""
namespace symint {{
  template <typename T, typename = std::enable_if_t<std::is_same<T, {intlike_t}>::value>>
  {sig.decl(suppress_symint_suffix=True)} {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
  }}
}}
"""
            # 根据目标类型返回声明或定义的代码段
        return result


# 生成 TensorBody.h。该文件提供了面向对象（基于方法的）的公共 C++ API，
# 以及从这些函数调用分发器的支架。
@dataclass(frozen=True)
class ComputeTensorMethod:
    target: Literal[Target.DECLARATION, Target.DEFINITION]
    static_dispatch_backend_indices: list[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str | None:
        if Variant.method not in f.variants:
            return None

        assert not f.func.is_out_fn()
        assert f.func.arguments.self_arg is not None

        sig_group = CppSignatureGroup.from_native_function(
            f, method=True, fallback_binding=f.manual_cpp_binding
        )

        if self.target is Target.DECLARATION:
            result = ""
            for sig in sig_group.signatures():
                result += f"{sig.decl()} const;\n"
            return result

        if self.target is not Target.DEFINITION:
            assert_never(self.target)

        result = ""

        for sig in sig_group.signatures():
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments(), method=True)
            exprs_str = ", ".join([e.expr for e in exprs])

            result += f"""
// aten::{f.func}
inline {sig.defn(prefix="Tensor::")} const {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
}}
"""

        return result


# 生成 RedispatchFunctions.h。
# 这类似于 Functions.h 中定义的 C++ API，但提供了访问分发器的 redispatch API。
@dataclass(frozen=True)
class ComputeRedispatchFunction:
    @method_with_native_function
"""
    def __call__(self, f: NativeFunction) -> str | None:
        # 我们无条件地生成重新调度 API 的函数变体。
        # 主要原因是我们可以单独命名空间函数，但不能单独命名空间方法。
        # 使用传入的 NativeFunction 对象创建一个 CppSignatureGroup 对象，不处理方法，根据需要使用手动的 C++ 绑定作为后备绑定。
        sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=f.manual_cpp_binding
        )

        # 初始化一个空字符串来存储结果
        result = ""
        # 遍历 CppSignatureGroup 中的每一个签名
        for sig in sig_group.signatures():
            # 根据函数原型创建 DispatcherSignature 对象
            target_sig = DispatcherSignature.from_schema(f.func)
            # 将当前签名的参数转换为目标签名的参数表达式
            exprs = translate(sig.arguments(), target_sig.arguments())
            # 将表达式转换为字符串，并使用逗号连接每个表达式，添加 dispatchKeySet 作为第一个参数
            exprs_str = ", ".join(["dispatchKeySet"] + [a.expr for a in exprs])

            # 将结果字符串按格式附加到 result 变量中
            result += f"""
# 定义内联函数，根据给定的函数签名重新调度至对应的 ATen 操作
inline {sig.decl(is_redispatching_fn=True)} {{
    return at::_ops::{f.func.name.unambiguous_name()}::redispatch({exprs_str});
}}
"""

# 返回生成的 ATenOpList.cpp 文件内容，其中包含所有 aten 操作符的列表
# 此列表在运行时可以访问
@with_native_function
def compute_aten_op(f: NativeFunction) -> str:
    return f'{{"aten::{f.func.name.name}", "{f.func.name.overload_name}"}},'


# 生成 MetaFunctions.h 文件的函数声明
# 若不是结构化的函数组，则返回 None
def compute_meta_function_declaration(g: NativeFunctionsGroup) -> str | None:
    if not g.structured:
        return None

# 定义结构化函数的 C++ 结构体
# 继承自给定的父类，包含预计算的声明、元函数返回类型定义和元函数声明
struct TORCH_API structured_{name} : public {parent_class} {{
    {precomputed_decl}
    {meta_return_typedef}
    {meta_return} meta({args_str});
}};
"""


# 判断给定的 NativeFunction 是否需要选择后端
# 根据函数名判断是否以 "_like" 结尾或者以 "new_" 开头，若是则返回 False
# 若函数的 tensor_options 为 None，则返回 False
# 否则使用 selector 判断该函数是否被选择
def needs_backend_select(f: NativeFunction, selector: SelectiveBuilder) -> bool:
    name = str(f.func.name.name)
    if name.endswith("_like") or name.startswith("new_"):
        return False
    if f.func.arguments.tensor_options is None:
        return False
    return selector.is_native_function_selected(f)


# 生成 RegisterBackendSelect.cpp 文件的数据类
# 包含专门计算操作符分发密钥的内核序列
@dataclass(frozen=True)
class ComputeBackendSelect:
    target: Literal[Target.DEFINITION, Target.REGISTRATION]

    # 用于确定哪些操作符生成注册代码的选择器对象
    selector: SelectiveBuilder

    @method_with_native_function
        def __call__(self, f: NativeFunction) -> str | None:
            # 如果函数不需要后端选择，则返回 None
            if not needs_backend_select(f, self.selector):
                return None
            
            # 获取函数的名称
            name = native.name(f.func)
            
            # 根据函数的原生签名创建 NativeSignature 对象，保留符号整数
            # BackendSelect 可以进入 Meta，因此必须保留符号整数
            native_sig = NativeSignature(f.func, symint=True)

            # 提取原生签名中的张量类型参数
            native_tensor_args = [
                a
                for a in native_sig.arguments()
                if isinstance(a.argument, Argument) and a.argument.type.is_tensor_like()
            ]

            # 从函数的 schema 中创建 DispatcherSignature 对象
            dispatcher_sig = DispatcherSignature.from_schema(f.func)

            # 根据目标选择使用 NativeSignature 还是 DispatcherSignature
            sig: NativeSignature | DispatcherSignature
            sig = dispatcher_sig

            # 获取 DispatcherSignature 的表达式列表
            dispatcher_exprs = dispatcher_sig.exprs()

            # 设置调度键的字符串表示
            dispatch_key = "c10::computeDispatchKey(dtype, layout, device)"

            # 如果目标是 Target.DEFINITION
            if self.target is Target.DEFINITION:
                # 我认为实际上没有理由以不同方式生成这两种情况
                # 第一个情况可能可以改进- 它调用 computeDispatchKeySet()，
                # 它查看 TLS 调度键- 在我们到达后端选择时不应存在任何调度键。
                if native_tensor_args:
                    # 断言函数参数中有张量参数
                    assert f.func.arguments.has_tensor_arg()
                    # 将张量参数名称连接为字符串
                    tensor_args = ", ".join(a.name for a in native_tensor_args)
                    # 生成计算调度键的表达式
                    compute_dk = f"""\
// 创建包含指定调度键的调度键集合 _dk_set，包括 dispatch_key 和 tensor_args
DispatchKeySet _dk_set = c10::DispatchKeySet({dispatch_key}) | c10::detail::multi_dispatch_key_set({tensor_args});
// 创建用于后续计算的调度键集合 _dk_mask，指定为全集并且移除 BackendSelect 调度键
DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
// 计算最终的调度键集合 _dk，基于给定的 _dk_set 和 _dk_mask
DispatchKeySet _dk = c10::impl::computeDispatchKeySet(_dk_set, _dk_mask);"""
            else:
                assert not f.func.arguments.has_tensor_arg()
                // 如果函数没有张量参数，直接使用 dispatch_key 创建调度键集合 _dk
                compute_dk = (
                    f"DispatchKeySet _dk = c10::DispatchKeySet({dispatch_key});"
                )
            return f"""\
// 定义 aten::{f.func} 函数的实现
C10_ALWAYS_INLINE
{sig.defn(name)} {{
  {compute_dk}
  // 调用 at::_ops::{f.func.name.unambiguous_name()}::redispatch 函数，重新调度操作
  return at::_ops::{f.func.name.unambiguous_name()}::redispatch(
      _dk, {', '.join(a.expr for a in dispatcher_exprs)});
}}
"""
        elif self.target is Target.REGISTRATION:
            // 在模块中注册 aten::{f.func.name} 函数的实现
            return f"""m.impl("aten::{f.func.name}", TORCH_FN({name}));"""
        else:
            // 断言，用于确保代码路径不会到达此处，因为目标类型未知
            assert_never(self.target)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                       YAML CODE GENERATION
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def format_yaml(data: object) -> str:
    // 忽略 Dumper 中的别名处理
    YamlDumper.ignore_aliases = lambda self, data: True  # type: ignore[assignment]

    // 支持序列化 OrderedDict
    def dict_representer(dumper: Any, data: Any) -> Any:
        return dumper.represent_dict(data.items())

    // 添加 OrderedDict 的序列化处理器
    YamlDumper.add_representer(OrderedDict, dict_representer)  # type: ignore[no-untyped-call]
    // 生成 YAML 格式的字符串，排除默认流样式，指定宽度避免可选的换行
    return yaml.dump(data, default_flow_style=False, Dumper=YamlDumper, width=1e9)  # type: ignore[no-any-return, call-overload]


// 有些默认值在写入 YAML 时会作为原生 YAML 对象写入，而非字符串。此函数检测这些情况并转换为原生 Python 对象。
def pythonify_default(s: str) -> object:
    if s == "true":
        return True
    elif s == "false":
        return False

    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


// dynamic_type 函数用于返回类型 t 的动态类型描述，对于 OptionalType 类型递归处理其元素类型，对于 Tensor 返回 "at::Tensor"
// TODO: 在更新代码生成框架后，移除 dynamic_type 函数的使用
def dynamic_type(t: Type) -> str:
    if isinstance(t, OptionalType):
        return dynamic_type(t.elem)
    // 注意这里不使用 t.is_tensor_like()，因为它也包括 Tensor[]
    if str(t) == "Tensor":
        return "at::Tensor";
    # 返回一个 C++ 类型的表示，根据给定的参数类型 t
    # mutable 参数设置为 False，表示类型不可变
    # binds 参数设置为 "__placeholder__"，指定绑定信息
    # symint 参数设置为 False，表示不使用 SymInt（可能是某种符号整数的概念）
    return cpp.argumenttype_type(
        t, mutable=False, binds="__placeholder__", symint=False
    ).cpp_type()
# 计算 YAML 条目的返回字段
def compute_method_of_yaml(variants: set[Variant]) -> list[str]:
    # 这里明确指定顺序，确保 Type 和 namespace 被正确放入列表
    method_of = ["Type"]
    if Variant.method in variants:
        method_of.append("Tensor")
    if Variant.function in variants:
        method_of.append("namespace")
    return method_of


def compute_returns_yaml(
    f: NativeFunction,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    # Note [name and field_name]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 要理解 name_to_field_name，我们首先需要讨论这个模式:
    # lstsq.X(Tensor self, Tensor A, *, Tensor(a!) X, Tensor(b!) qr) -> (Tensor(a!) solution, Tensor(b!) QR)
    # 这个模式有一个很奇怪的地方: 它是函数的输出变体（换句话说，在 C++ API 中会转换为 at::lstsq_out()），但输出返回参数的名称与输入的关键字参数名称不匹配。
    # 实际上，在这种情况下，我们想要输出的历史 Declarations.yaml 如下（仅显示相关字段）:
    # arguments:
    #   ...
    # - field_name: solution
    #   name: X
    # - field_name: QR
    #   name: qr
    # returns:
    # - field_name: solution
    #   name: X
    # - field_name: QR
    #   name: qr
    # 返回字段的名称存储在 'field_name' 中，参数的名称存储在 'name' 中。因此，在处理返回参数时，我们需要一种方法来获取对应的返回值。目前，最方便的方法是在处理返回参数时构建从名称（参数概念）到字段名称（返回概念）的映射。
    # 参见 https://github.com/pytorch/pytorch/issues/43114
    name_to_field_name: dict[str, str] = {}

    # 计算 YAML 条目的返回部分
    names = cpp.return_names(f)
    returns = []
    for i, (r, name) in enumerate(zip(f.func.returns, names)):
        ret = {
            "dynamic_type": dynamic_type(r.type),
            "name": name,
            # legacy, report ints
            "type": cpp.return_type(r, symint=False).cpp_type(),
        }

        if r.name:
            # See Note [name and field_name]
            # 如果 r.name 存在，将其作为 'field_name' 添加到返回条目中
            ret["field_name"] = r.name
            if f.func.is_out_fn():
                # 如果函数是输出函数，将参数名映射到字段名
                name_to_field_name[f.func.arguments.out[i].name] = r.name

        returns.append(ret)

    return returns, name_to_field_name


# arguments in yaml 大致对应于公共的 C++ API
def compute_cpp_argument_yaml(
    cpp_a: Binding,
    *,
    schema_order: bool,
    kwarg_only_set: set[str],
    out_arg_set: set[str],
    name_to_field_name: dict[str, str],
) -> object:
    # 检查 cpp_a.argument 是否属于 TensorOptionsArguments 类型
    if isinstance(cpp_a.argument, TensorOptionsArguments):
        # 如果是 TensorOptionsArguments 类型，创建一个字典 arg 来存储参数信息
        arg: dict[str, object] = {
            "annotation": None,  # 参数的注释设为 None
            "dynamic_type": "at::TensorOptions",  # 参数的动态类型为 at::TensorOptions
            "is_nullable": False,  # 参数不可为空
            "name": cpp_a.name,  # 参数的名称
            "type": cpp_a.type,  # 参数的类型
            "kwarg_only": True,  # 参数仅能作为关键字参数传递
        }
        # 如果 cpp_a.default 不为 None，则将其作为默认值存入 arg 字典中
        if cpp_a.default is not None:
            arg["default"] = cpp_a.default
        # 返回构建好的参数字典 arg
        return arg
    # 如果 cpp_a.argument 属于 SelfArgument 类型，则抛出 AssertionError
    elif isinstance(cpp_a.argument, SelfArgument):
        raise AssertionError
    # 如果 cpp_a.argument 属于 Argument 类型，则调用 compute_argument_yaml 函数处理
    elif isinstance(cpp_a.argument, Argument):
        return compute_argument_yaml(
            cpp_a.argument,
            schema_order=schema_order,  # 将 schema_order 参数传递给 compute_argument_yaml 函数
            kwarg_only_set=kwarg_only_set,  # 将 kwarg_only_set 参数传递给 compute_argument_yaml 函数
            out_arg_set=out_arg_set,  # 将 out_arg_set 参数传递给 compute_argument_yaml 函数
            name_to_field_name=name_to_field_name,  # 将 name_to_field_name 参数传递给 compute_argument_yaml 函数
        )
# 计算函数参数的 YAML 表示
def compute_argument_yaml(
    a: Argument,
    *,
    schema_order: bool,
    kwarg_only_set: set[str],
    out_arg_set: set[str],
    name_to_field_name: dict[str, str],
) -> object:
    # 初始化参数字典
    arg: dict[str, object] = {
        "annotation": str(a.annotation) if a.annotation else None,  # 存储参数的注解信息，如果有的话
        "dynamic_type": dynamic_type(a.type),  # 计算参数的动态类型
        "is_nullable": a.type.is_nullable(),  # 检查参数是否可空
        "name": a.name,  # 存储参数的名称
        # legacy, report ints
        "type": cpp.argument_type(a, binds="__placeholder__", symint=False).cpp_type(),  # 获取参数的类型（整数报告的遗留部分）
    }
    # 如果参数有默认值，将其转换为 Python 可读形式并存储
    if a.default is not None:
        arg["default"] = pythonify_default(
            cpp.default_expr(a.default, a.type, symint=False)
        )
    # 如果参数在 kwarg_only_set 中，标记为仅限关键字参数
    if a.name in kwarg_only_set:
        arg["kwarg_only"] = True
    # 如果参数在 out_arg_set 中，标记为输出参数，并需分配内存
    if a.name in out_arg_set:
        arg["output"] = True
        arg["allocate"] = True
        # 查找参数名称在 name_to_field_name 中的映射关系并存储到 field_name 中
        # See Note [name and field_name]
        if a.name in name_to_field_name:
            arg["field_name"] = name_to_field_name[a.name]
    # 历史上，布尔类型不记录其大小，因为它已经内置在 cpp 类型中（例如 std::array<bool, 4>）
    l = a.type.is_list_like()
    if l is not None and l.size is not None and str(l.elem) != "bool":
        arg["size"] = l.size  # 存储列表类型的大小信息
    return arg  # 返回参数字典


# 计算函数声明的 YAML 表示
@with_native_function
def compute_declaration_yaml(f: NativeFunction) -> object:
    returns, name_to_field_name = compute_returns_yaml(f)

    # 用于快速测试参数是否是仅限关键字参数或输出参数的集合
    kwarg_only_set = {a.name for a in f.func.arguments.flat_kwarg_only}
    out_arg_set = {a.name for a in f.func.arguments.out}

    # 从原生函数创建 C++ 签名组
    sig_group = CppSignatureGroup.from_native_function(
        f, method=False, fallback_binding=False
    )
    cpp_args = sig_group.signature.arguments()

    # 计算所有参数的 YAML 表示
    arguments = [
        compute_cpp_argument_yaml(
            cpp_a,
            schema_order=False,
            kwarg_only_set=kwarg_only_set,
            out_arg_set=out_arg_set,
            name_to_field_name=name_to_field_name,
        )
        for cpp_a in cpp_args
    ]

    # 获取按照 schema_order 排序的参数列表
    schema_order_jit_arguments = list(f.func.schema_order_arguments())

    # 计算按照 schema_order 排序的参数的 YAML 表示
    schema_order_arguments = [
        compute_argument_yaml(
            a,
            schema_order=True,
            kwarg_only_set=kwarg_only_set,
            out_arg_set=out_arg_set,
            name_to_field_name=name_to_field_name,
        )
        for a in schema_order_jit_arguments
    ]

    # 获取按照 schema_order 排序的参数对应的 C++ 类型
    cpp_schema_order_types = [
        # NB: method here doesn't matter
        r.type
        for a in schema_order_jit_arguments
        for r in cpp.argument(
            a,
            method=False,
            cpp_no_default_args=set(),
            faithful=False,
            symint=False,
            has_tensor_options=False,
        )
    ]

    # legacy, report ints
    # 获取函数返回值的 C++ 类型
    cpp_returns = cpp.returns_type(f.func.returns, symint=False).cpp_type()
    schema_order_cpp_signature = f"{cpp_returns} ({', '.join(cpp_schema_order_types)})"  # 构建函数的 C++ 签名
    # 检查函数是否是工厂方法，条件为：至少有一个参数是TensorOptionsArguments类型，且该方法不属于Variant.method
    is_factory_method = (
        any(isinstance(a.argument, TensorOptionsArguments) for a in cpp_args)
        and Variant.method not in f.variants
    )

    # 构建有序字典，包含以下字段：
    # - 'name': 使用cpp.name方法获取函数的名称
    # - 'operator_name': 将函数的名称转换为字符串
    # - 'overload_name': 将函数的重载名称转换为字符串
    # - 'manual_kernel_registration': 指示是否进行手动内核注册
    # - 'category_override': 如果f.category_override不为None，则使用其值；否则为空字符串
    # - 'schema_string': 构建表示函数schema的字符串
    # - 'arguments': 函数的参数列表
    # - 'schema_order_cpp_signature': C++签名的schema顺序
    # - 'schema_order_arguments': schema顺序的参数列表
    # - 'method_of': 计算方法属于哪个yaml配置的函数
    # - 'mode': 设置为'native'，表示本地模式
    # - 'python_module': 如果f.python_module为None，则设置为空字符串；否则使用其值
    # - 'returns': 函数的返回类型
    # - 'inplace': 函数是否为原地操作的标志
    # - 'is_factory_method': 上面计算得到的工厂方法标志
    # - 'abstract': 函数是否为抽象方法
    # - 'device_guard': 函数的设备保护设置
    # - 'with_gil': 是否需要全局解释器锁(GIL)
    # - 'deprecated': 函数是否已经废弃
    # - 'has_math_kernel': 是否具有数学内核
    return OrderedDict(
        [
            ("name", cpp.name(f.func)),
            ("operator_name", str(f.func.name.name)),
            ("overload_name", str(f.func.name.overload_name)),
            ("manual_kernel_registration", f.manual_kernel_registration),
            (
                "category_override",
                f.category_override if f.category_override is not None else "",
            ),
            ("schema_string", f"aten::{f.func}"),
            ("arguments", arguments),
            ("schema_order_cpp_signature", schema_order_cpp_signature),
            ("schema_order_arguments", schema_order_arguments),
            ("method_of", compute_method_of_yaml(f.variants)),
            ("mode", "native"),
            ("python_module", "" if f.python_module is None else f.python_module),
            ("returns", returns),
            ("inplace", f.func.name.name.inplace),
            ("is_factory_method", is_factory_method),
            ("abstract", f.is_abstract),
            ("device_guard", f.device_guard),
            ("with_gil", False),
            ("deprecated", False),
            ("has_math_kernel", f.has_composite_implicit_autograd_kernel),
        ]
    )
# 标记一个函数，用于检查是否具有自动生成的复合内核
def has_autogenerated_composite_kernel(f: NativeFunction) -> bool:
    # 返回条件表达式，检查结构化属性或者结构化委托是否不为None，并且函数类型为functional或inplace之一
    return (f.structured or f.structured_delegate is not None) and (
        f.func.kind() == SchemaKind.functional or f.func.kind() == SchemaKind.inplace
    )


# 使用装饰器，处理原生函数和索引
@with_native_function_and_indices
# 生成注册声明计算函数，返回类型为字符串
def compute_registration_declarations(
    f: NativeFunction, backend_indices: dict[DispatchKey, BackendIndex]
) -> str:
    # 获取函数名称
    name = dispatcher.name(f.func)
    # 获取返回类型的注册声明的C++类型
    returns_type = dispatcher.returns_type(
        f.func.returns
    ).cpp_type_registration_declarations()
    # 获取函数参数
    args = dispatcher.arguments(f.func)
    # 将参数转换为字符串形式，用逗号分隔
    args_str = ", ".join(a.no_default().decl_registration_declarations() for a in args)
    # 准备评论数据的字典
    comment_data: dict[str, str] = {
        "schema": f"aten::{f.func}",
        # TODO: 'dispatch'字段的语义是什么？
        "dispatch": str(
            {k for k, v in backend_indices.items() if v.has_kernel(f)}
            != {DispatchKey.CompositeImplicitAutograd}
            and {k for k, v in backend_indices.items() if v.has_kernel(f)}
            != {
                DispatchKey.CompositeImplicitAutograd,
                DispatchKey.CompositeImplicitAutogradNestedTensor,
            }
        ),
        # 判断是否具有复合内核或自动生成的复合内核
        "default": str(f.has_composite_kernel or has_autogenerated_composite_kernel(f)),
    }
    # 返回格式化的注册声明字符串，包含注释数据的JSON表示
    return f"""{returns_type} {name}({args_str}); // {json.dumps(comment_data)}
"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           RUN IT ALL
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# 获取自定义构建选择器
def get_custom_build_selector(
    provided_op_registration_allowlist: list[str] | None,
    op_selection_yaml_path: str | None,
) -> SelectiveBuilder:
    # 断言不能同时提供provided_op_registration_allowlist和op_selection_yaml_path
    assert not (
        provided_op_registration_allowlist is not None
        and op_selection_yaml_path is not None
    ), (
        "Both provided_op_registration_allowlist and "
        + "op_selection_yaml_path can NOT be provided at the "
        + "same time."
    )

    # 初始化操作注册允许列表为None
    op_registration_allowlist: set[str] | None = None
    # 如果提供了provided_op_registration_allowlist，则设置为其集合形式
    if provided_op_registration_allowlist is not None:
        op_registration_allowlist = set(provided_op_registration_allowlist)

    # 根据条件选择性地创建选择器
    if op_registration_allowlist is not None:
        selector = SelectiveBuilder.from_legacy_op_registration_allow_list(
            op_registration_allowlist,
            True,
            False,
        )
    elif op_selection_yaml_path is not None:
        selector = SelectiveBuilder.from_yaml_path(op_selection_yaml_path)
    else:
        selector = SelectiveBuilder.get_nop_selector()

    # 返回选择器对象
    return selector


# 获取按视图分组的本地函数
def get_grouped_by_view_native_functions(
    native_functions: Sequence[NativeFunction],
) -> Sequence[NativeFunction | NativeFunctionsViewGroup]:
    # 定义内部函数，用于可能创建视图组
    def maybe_create_view_group(
        d: dict[ViewSchemaKind | SchemaKind, NativeFunction]
        # （此处省略了部分函数体）
    ) -> list[NativeFunction | NativeFunctionsViewGroup]:
        funcs: list[NativeFunction | NativeFunctionsViewGroup] = []
        if ViewSchemaKind.aliasing in d:
            # 如果别名视图在字典中，则从字典中取出别名视图
            view = d.pop(ViewSchemaKind.aliasing)
            # 同时尝试取出就地别名视图和函数性别名视图
            view_inplace = d.pop(ViewSchemaKind.aliasing_inplace, None)
            view_copy = d.pop(SchemaKind.functional, None)

            # 将取出的视图信息封装成 NativeFunctionsViewGroup 对象并添加到 funcs 列表中
            funcs.append(
                NativeFunctionsViewGroup(
                    view=view,
                    view_copy=view_copy,
                    view_inplace=view_inplace,
                )
            )
        # 将剩余未添加到视图组的函数添加到 funcs 列表中
        funcs.extend(d.values())
        return funcs

    grouped_by_views: dict[
        FunctionSchema, dict[SchemaKind | ViewSchemaKind, NativeFunction]
    ] = defaultdict(dict)
    # 遍历 native_functions 列表，为每个函数按照视图分组
    for f in native_functions:
        # 调用函数的 view_signature 方法获取其视图模式
        schema = f.func.view_signature()
        # 获取函数的视图模式类型
        view_kind: ViewSchemaKind = f.view_schema_kind
        
        # 根据视图模式类型将函数分组：
        # 如果视图模式类型为非别名视图，则按照功能类型分组
        if view_kind == ViewSchemaKind.non_aliasing:
            kind = f.func.kind()
            assert kind not in grouped_by_views[schema]
            grouped_by_views[schema][kind] = f
        else:
            # 否则，按照视图模式类型分组
            assert (
                view_kind not in grouped_by_views[schema]
            ), f"{view_kind} already in {grouped_by_views[schema].keys()}"
            grouped_by_views[schema][view_kind] = f

    # 对每个分组创建视图组并将结果扁平化后返回列表
    return list(concatMap(maybe_create_view_group, grouped_by_views.values()))
# 根据给定的原生函数列表，获取分组后的原生函数或原生函数组合成的序列
def get_grouped_native_functions(
    native_functions: Sequence[NativeFunction],
) -> Sequence[NativeFunction | NativeFunctionsGroup]:
    # 将原生函数预先分组并扁平化为原生函数或原生函数组的序列
    def flatten_pre_group(
        d: dict[SchemaKind, NativeFunction]
    ) -> Sequence[NativeFunction | NativeFunctionsGroup]:
        # 将字典转换为原生函数组，若转换失败则返回原生函数列表
        r = NativeFunctionsGroup.from_dict(d)
        if r is None:
            # 不变式: 任何被代码生成的原生函数应该已经被分组成原生函数组对象
            assert not any("generated" in f.tags for f in d.values())
            return list(d.values())
        else:
            return [r]

    # TODO: ValuesView 怎么不是 Sequence 呢
    # 预先分组的原生函数
    pre_grouped_native_functions = pre_group_native_functions(native_functions)
    # 返回扁平化后的原生函数或原生函数组的列表
    return list(
        concatMap(flatten_pre_group, list(pre_grouped_native_functions.values()))
    )


def get_ns_grouped_kernels(
    *,
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    backend_indices: dict[DispatchKey, BackendIndex],
    native_function_decl_gen: Callable[
        [NativeFunctionsGroup | NativeFunction, BackendIndex], list[str]
    ] = dest.compute_native_function_declaration,
) -> dict[str, list[str]]:
    # 命名空间分组的内核函数字典
    ns_grouped_kernels: dict[str, list[str]] = defaultdict(list)
    # 遍历分组后的原生函数或原生函数组
    for f in grouped_native_functions:
        # 原生函数的命名空间集合和调度键集合
        native_function_namespaces = set()
        dispatch_keys = set()
        # 遍历后端索引中的调度键和后端索引
        for dispatch_key, backend_idx in backend_indices.items():
            # 获取后端元数据中的内核函数信息
            backend_metadata = backend_idx.get_kernel(f)
            if backend_metadata:
                # 获取内核函数所在的 C++ 命名空间
                namespace = backend_metadata.cpp_namespace
                dispatch_keys.add(dispatch_key)
                native_function_namespaces.add(namespace)
            else:
                namespace = DEFAULT_KERNEL_NAMESPACE
            # 确保每个操作符只支持一个命名空间，否则报错
            assert (
                len(native_function_namespaces) <= 1
            ), f"Codegen only supports one namespace per operator, got {native_function_namespaces} from {dispatch_keys}"
            # 将内核函数声明生成并加入对应命名空间的列表
            ns_grouped_kernels[namespace].extend(
                native_function_decl_gen(f, backend_idx)
            )
    # 返回命名空间分组的内核函数字典
    return ns_grouped_kernels


def get_native_function_declarations_from_ns_grouped_kernels(
    *,
    ns_grouped_kernels: dict[str, list[str]],
) -> list[str]:
    # 函数声明列表
    declarations: list[str] = []
    newline = "\n"
    # 遍历命名空间分组的内核函数字典
    for namespace, kernels in ns_grouped_kernels.items():
        # 创建命名空间助手对象
        ns_helper = NamespaceHelper(
            namespace_str=namespace,
            entity_name="",
            max_level=4,
        )
        # 去除重复的内核函数名称并按顺序加入声明列表
        ordered_kernels = list(OrderedDict.fromkeys(kernels))
        # 将命名空间助手的前言和后言与内核函数列表拼接成声明列表并加入结果中
        declarations.extend(
            f"""
{ns_helper.prologue}
{newline.join(ordered_kernels)}
{ns_helper.epilogue}
        """.split(
                newline
            )
        )
    # 返回所有命名空间的函数声明列表
    return declarations
# 返回由其命名空间分组的本地函数声明。
def get_native_function_declarations(
    *,
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    backend_indices: dict[DispatchKey, BackendIndex],
    native_function_decl_gen: Callable[
        [NativeFunctionsGroup | NativeFunction, BackendIndex], list[str]
    ] = dest.compute_native_function_declaration,
) -> list[str]:
    """
    在 `NativeFunction(s).h` 中生成内核声明。
    :param grouped_native_functions: 包含 `NativeFunction` 或 `NativeFunctionGroup` 的序列。
    :param backend_indices: 根据调度键分组的内核集合。
    :param native_function_decl_gen: 用于为每个 `NativeFunction` 生成内核声明的可调用对象。
    :return: 字符串列表，包含所有声明的字符串，按命名空间分组，通过换行符分隔。
    """

    # 获取按命名空间分组的内核
    ns_grouped_kernels = get_ns_grouped_kernels(
        grouped_native_functions=grouped_native_functions,
        backend_indices=backend_indices,
        native_function_decl_gen=native_function_decl_gen,
    )
    # 返回从命名空间分组的内核中获取本地函数声明的结果
    return get_native_function_declarations_from_ns_grouped_kernels(
        ns_grouped_kernels=ns_grouped_kernels
    )


def get_kernel_namespace(
    *, f: NativeFunction | NativeFunctionsGroup, backend_idx: BackendIndex
) -> str:
    # 获取函数的后端元数据
    backend_metadata = backend_idx.get_kernel(f)
    # 断言后端元数据不存在或者其 C++ 命名空间中包含 "::native"
    assert not backend_metadata or "::native" in backend_metadata.cpp_namespace, (
        f"The kernel for function {f.func.name if isinstance(f, NativeFunction) else f.functional.func.name} "
        f"with dispatch key {backend_idx.dispatch_key}"
        f" has a namespace {backend_metadata.cpp_namespace} and it's not ending with '::native'."
    )
    # 返回后端元数据的 C++ 命名空间，如果不存在则返回默认的内核命名空间
    return (
        backend_metadata.cpp_namespace if backend_metadata else DEFAULT_KERNEL_NAMESPACE
    )


# 返回按调度键和自定义命名空间分组的本地函数定义。
# 用于 RegisterDispatchKey.cpp 等文件。
def get_native_function_definitions(
    *,
    fm: FileManager,
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    dispatch_key: DispatchKey,
    backend_idx: BackendIndex,
    selector: SelectiveBuilder,
    rocm: bool,
    symint: bool,
    skip_dispatcher_op_registration: bool,
    gen_dispatch_helpers: bool,
) -> list[str]:
    # 函数定义列表
    definitions: list[str] = []
    # 按命名空间分组的定义字典
    ns_definitions: dict[str, list[str]] = defaultdict(list)
    # 匿名定义字典
    anonymous_definitions: dict[str, list[str]] = defaultdict(list)
    # 注册字典
    registrations: dict[str, dict[str, list[str]]] = defaultdict(dict)
    # 换行符
    newline = "\n"
    # 创建注册调度键的对象
    ns_gen = dest.RegisterDispatchKey(
        backend_idx,
        Target.NAMESPACED_DEFINITION,
        selector,
        rocm=rocm,
        symint=symint,
        class_method_name=None,
        skip_dispatcher_op_registration=skip_dispatcher_op_registration,
    )
    # 使用目标对象的 RegisterDispatchKey 方法注册匿名定义的调度键
    anonymous_gen = dest.RegisterDispatchKey(
        backend_idx,
        Target.ANONYMOUS_DEFINITION,
        selector,
        rocm=rocm,
        symint=symint,
        class_method_name=None,
        skip_dispatcher_op_registration=skip_dispatcher_op_registration,
    )
    # 使用目标对象的 RegisterDispatchKey 方法注册注册类型的调度键
    reg_gen = dest.RegisterDispatchKey(
        backend_idx,
        Target.REGISTRATION,
        selector,
        rocm=rocm,
        symint=symint,
        class_method_name=None,
        skip_dispatcher_op_registration=skip_dispatcher_op_registration,
    )
    # 遍历分组的原生函数列表
    for f in grouped_native_functions:
        # 获取与函数相关的内核命名空间，并移除 "::native" 后缀
        kernel_namespace = get_kernel_namespace(f=f, backend_idx=backend_idx).replace(
            "::native", ""
        )

        # 将命名空间生成器生成的内容扩展到命名空间定义字典中
        ns_definitions[kernel_namespace].extend(
            ns_gen(f),
        )
        # 将匿名定义生成器生成的内容扩展到匿名定义字典中
        anonymous_definitions[kernel_namespace].extend(
            anonymous_gen(f),
        )
        # 确定命名空间，如果是 NativeFunction 则使用其命名空间，否则使用函数功能的命名空间
        namespace = (
            f.namespace if isinstance(f, NativeFunction) else f.functional.namespace
        )
        # 如果命名空间不在注册字典中，添加一个空列表作为其值
        if namespace not in registrations[kernel_namespace]:
            registrations[kernel_namespace][namespace] = defaultdict(list)
        # 将注册生成器生成的内容扩展到注册字典中
        registrations[kernel_namespace][namespace].extend(
            reg_gen(f),
        )

    # 遍历命名空间定义字典中的每个命名空间
    for kernel_namespace in ns_definitions:
        # 如果命名空间定义内容为空，跳过当前迭代
        if len(ns_definitions[kernel_namespace]) == 0:
            continue
        # 创建命名空间助手对象，使用当前命名空间字符串初始化
        ns_helper = NamespaceHelper(namespace_str=kernel_namespace)
        registration_body = ""
        # 遍历注册字典中当前命名空间的每个命名空间
        for namespace in registrations[kernel_namespace]:
            # 如果注册列表为空，跳过当前迭代
            if not registrations[kernel_namespace][namespace]:
                continue
            # 将当前命名空间的注册内容添加到注册体中
            registration_body += f"""
# 定义 TORCH_LIBRARY_IMPL 宏，用于实现特定命名空间和调度键的注册
TORCH_LIBRARY_IMPL({namespace}, {dispatch_key}, m) {{
    # 插入注册列表中的每个注册代码
    {newline.join(registrations[kernel_namespace][namespace])}
}};"""

definitions.extend(
    # 使用模板替换来生成注册声明的代码段
    fm.substitute_with_template(
        "RegisterDispatchDefinitions.ini",
        lambda: {
            "ns_prologue": ns_helper.prologue,  # 命名空间辅助函数的开始部分
            "ns_epilogue": ns_helper.epilogue,  # 命名空间辅助函数的结束部分
            "dispatch_helpers": dest.gen_registration_helpers(backend_idx)
            if gen_dispatch_helpers
            else [],  # 生成调度助手代码（如果需要）
            "dispatch_anonymous_definitions": anonymous_definitions[
                kernel_namespace
            ],  # 匿名定义的调度代码
            "static_init_dispatch_registrations": ""
            if skip_dispatcher_op_registration
            else registration_body,  # 静态初始化调度注册（如果不跳过调度操作注册）
            "deferred_dispatch_registrations": "",  # 延迟调度注册
            "dispatch_namespace": dispatch_key.lower(),  # 调度命名空间
            "dispatch_namespaced_definitions": ns_definitions[kernel_namespace],  # 调度命名空间的定义
        },
    ).split(newline)
)

# 返回注册定义列表
return definitions


# 返回按调度键和自定义命名空间分组的本机函数声明
# 用于 CPUFunctions_inl.h 等文件
def get_namespaced_declaration(
    *,
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    dispatch_key: DispatchKey,
    backend_idx: BackendIndex,
    selector: SelectiveBuilder,
    rocm: bool,
    symint: bool,
) -> list[str]:
    # 本机函数声明列表
    declarations: list[str] = []
    # 自定义命名空间分组的内核函数字典
    ns_grouped_kernels: dict[str, list[str]] = defaultdict(list)
    newline = "\n"
    # 目标调度键的注册声明
    func = dest.RegisterDispatchKey(
        backend_idx,
        Target.NAMESPACED_DECLARATION,
        selector,
        rocm=rocm,
        class_method_name=None,
        skip_dispatcher_op_registration=False,
        symint=symint,
    )
    for f in grouped_native_functions:
        # 获取内核函数的命名空间，并替换为小写的调度键
        namespace = get_kernel_namespace(f=f, backend_idx=backend_idx).replace(
            "native", dispatch_key.lower()
        )

        ns_grouped_kernels[namespace].extend(
            func(f),
        )

    for namespace, kernels in ns_grouped_kernels.items():
        if len(kernels) == 0:
            continue
        # 命名空间辅助函数
        ns_helper = NamespaceHelper(
            namespace_str=namespace, entity_name="", max_level=3
        )
        # 按格式排列内核函数
        ordered_kernels = list(OrderedDict.fromkeys(kernels))
        declarations.extend(
            f"""
{ns_helper.prologue}
{newline.join(ordered_kernels)}
{ns_helper.epilogue}
        """.split(
                newline
            )
        )
    # 返回声明列表
    return declarations


# 返回 aten 和其他命名空间的本机函数模式注册代码
def get_native_function_schema_registrations(
    *,
    native_functions: Sequence[NativeFunction],
    schema_selector: SelectiveBuilder,
) -> tuple[list[str], str]:
    # 各命名空间的本机函数列表字典
    ns_native_functions: dict[str, list[NativeFunction]] = defaultdict(list)
    # 遍历 native_functions 列表，其中每个元素是一个原生函数对象
    for native_function in native_functions:
        # 将每个原生函数对象按其命名空间分类，添加到 ns_native_functions 字典中
        ns_native_functions[native_function.namespace].append(native_function)
    # 初始化空字符串，用于存储模式注册信息
    schema_registrations = ""
    # 初始化空列表，用于存储 ATen 命名空间的模式注册信息
    aten_schema_registrations = []
    # 初始化自定义命名空间为 None
    custom_namespace = None
    # 遍历 ns_native_functions 字典，其中键是命名空间，值是对应命名空间的原生函数列表
    for namespace, funcs in ns_native_functions.items():
        # 调用 RegisterSchema(schema_selector) 函数，根据条件映射注册模式的结果列表
        schema_registrations_body = list(
            mapMaybe(RegisterSchema(schema_selector), funcs)
        )
        # 如果命名空间是 "aten"
        if namespace == "aten":
            # 将 ATen 命名空间的模式注册信息存储在 aten_schema_registrations 列表中
            aten_schema_registrations = schema_registrations_body
        else:
            # 记录当前处理的自定义命名空间
            custom_namespace = namespace
            tab = "\t"
            # 如果命名空间是预定义的，使用 TORCH_LIBRARY_FRAGMENT 宏定义库片段
            # 否则使用 TORCH_LIBRARY 宏定义新库
            torch_library_macro = (
                "TORCH_LIBRARY_FRAGMENT"
                if namespace in FRAGMENT_NAMESPACES
                else "TORCH_LIBRARY"
            )
            # 将自定义命名空间的模式注册信息追加到 schema_registrations 字符串中
            schema_registrations += f"""
def gen_aggregated_headers(
    *,
    native_functions: Sequence[NativeFunction],
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    structured_native_functions: Sequence[NativeFunctionsGroup],
    static_dispatch_idx: list[BackendIndex],
    selector: SelectiveBuilder,
    backend_indices: dict[DispatchKey, BackendIndex],
    cpu_fm: FileManager,
    cuda_fm: FileManager,
    functions_keys: set[DispatchKey],
    dispatch_keys: Sequence[DispatchKey],
    rocm: bool,
) -> None:
    # Buck doesn't support dynamic output files, so we aggregate all operator
    # headers into a single file
    cpu_fm.write(
        "NativeMetaFunctions.h",
        lambda: {
            "NativeMetaFunctions_includes": [],
            # Generate meta function declarations for structured native functions
            "NativeMetaFunctions_declarations": list(
                mapMaybe(compute_meta_function_declaration, structured_native_functions)
            ),
        },
    )
    # Filter native functions that are methods
    method_native_functions = [
        fn for fn in native_functions if Variant.method in fn.variants
    ]
    # Filter native functions that are not methods
    non_method_native_functions = [
        fn for fn in native_functions if fn not in method_native_functions
    ]
    cpu_fm.write(
        "MethodOperators.h",
        lambda: {
            "MethodOperators_includes": [],
            # Generate method operator declarations using static dispatch indices
            "MethodOperators_declarations": list(
                mapMaybe(
                    ComputeOperators(
                        Target.DECLARATION,
                        static_dispatch_backend_indices=static_dispatch_idx,
                    ),
                    method_native_functions,
                )
            ),
        },
    )
    cpu_fm.write(
        "Operators.h",
        lambda: {
            "Operators_includes": ["#include <ATen/MethodOperators.h>"],
            # Generate operator declarations for non-method native functions
            "Operators_declarations": list(
                mapMaybe(
                    ComputeOperators(
                        Target.DECLARATION,
                        static_dispatch_backend_indices=static_dispatch_idx,
                    ),
                    non_method_native_functions,
                )
            ),
        },
    )
    cpu_fm.write(
        "Functions.h",
        lambda: {
            # Include static dispatch extra headers based on static dispatch indices
            "static_dispatch_extra_headers": static_dispatch_extra_headers(
                static_dispatch_idx
            ),
            "Functions_includes": ["#include <ATen/Operators.h>"],
            # Generate function declarations for all native functions
            "Functions_declarations": list(
                mapMaybe(
                    ComputeFunction(),
                    native_functions,
                )
            ),
        },
    )
    # Retrieve native function declarations for grouped native functions
    declarations = get_native_function_declarations(
        grouped_native_functions=grouped_native_functions,
        backend_indices=backend_indices,
    )
    # 将内容写入 cpu_fm 或 cuda_fm 的 "NativeFunctions.h" 文件
    cpu_fm.write(
        "NativeFunctions.h",
        # 创建一个 lambda 函数，返回一个包含 NativeFunctions 的声明和头文件包含信息的字典
        lambda: {
            "NativeFunctions_includes": ["#include <ATen/NativeMetaFunctions.h>"],
            "NativeFunctions_declarations": declarations,
        },
    )

    # 遍历 dispatch_keys 列表中的每个 dispatch_key
    for dispatch_key in dispatch_keys:
        # 根据 dispatch_key 判断使用 cpu_fm 还是 cuda_fm
        fm = cuda_fm if is_cuda_dispatch_key(dispatch_key) else cpu_fm
        # 如果 dispatch_key 在 functions_keys 中
        if dispatch_key in functions_keys:
            # 生成 inline 头文件路径
            inl_headers = f"#include <ATen/{dispatch_key}Functions_inl.h>"

            # 向文件 "{dispatch_key}Functions.h" 写入内容，使用模板 "DispatchKeyFunctions.h"
            fm.write_with_template(
                f"{dispatch_key}Functions.h",
                "DispatchKeyFunctions.h",
                # 创建 lambda 函数，返回一个字典，包含 dispatch_key 和 inline_headers
                lambda: {
                    "dispatch_key": str(dispatch_key),
                    "inline_headers": inl_headers,
                },
            )
            # 向文件 "{dispatch_key}Functions_inl.h" 写入内容，使用模板 "DispatchKeyFunctions_inl.h"
            fm.write_with_template(
                f"{dispatch_key}Functions_inl.h",
                "DispatchKeyFunctions_inl.h",
                # 创建 lambda 函数，返回一个字典，包含 DispatchKeyFunctions_inl 的相关信息
                lambda: {
                    "DispatchKeyFunctions_inl_includes": [],
                    "dispatch_namespace": dispatch_key.lower(),
                    "dispatch_namespaced_declarations": get_namespaced_declaration(
                        grouped_native_functions=grouped_native_functions,
                        dispatch_key=dispatch_key,
                        backend_idx=backend_indices[dispatch_key],
                        selector=selector,
                        rocm=rocm,
                        symint=True,
                    ),
                },
            )

        # 删除 fm 对象的引用，释放资源
        del fm
# 定义函数，生成每个运算符的头文件
def gen_per_operator_headers(
    *,
    native_functions: Sequence[NativeFunction],  # 原生函数的列表
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],  # 分组的原生函数列表，可以是单个函数或函数组
    static_dispatch_idx: list[BackendIndex],  # 静态调度索引的列表
    selector: SelectiveBuilder,  # 选择性构建器对象
    backend_indices: dict[DispatchKey, BackendIndex],  # 分发键与后端索引的字典映射
    cpu_fm: FileManager,  # CPU 文件管理器
    cuda_fm: FileManager,  # CUDA 文件管理器
    ops_fm: FileManager,  # 运算符文件管理器
    functions_keys: set[DispatchKey],  # 分发键的集合
    dispatch_keys: Sequence[DispatchKey],  # 分发键的序列
    rocm: bool,  # 是否为 ROCm 架构
) -> None:
    # 对于 CMake 构建，将运算符声明分割到 ATen/ops 文件夹中的单独头文件，以减少头文件依赖关系
    functions_by_root_name: dict[str, list[NativeFunction]] = defaultdict(list)  # 以根名称为键的原生函数列表的默认字典
    for fn in native_functions:
        functions_by_root_name[fn.root_name].append(fn)  # 将原生函数按根名称分组存储到字典中

    grouped_functions_by_root_name: dict[
        str, list[NativeFunction | NativeFunctionsGroup]
    ] = defaultdict(list)  # 以根名称为键的原生函数或函数组列表的默认字典
    for group in grouped_native_functions:
        name = group.root_name
        grouped_functions_by_root_name[name].append(group)  # 将分组的原生函数或函数组按根名称分组存储到字典中
    for name, functions in functions_by_root_name.items():
        # 遍历 functions_by_root_name 字典中的每个键值对，name 是根名称，functions 是相关的函数列表

        ops_fm.write_with_template(
            f"{name}_ops.h",
            "Operator.h",
            lambda: {
                "declarations": list(
                    mapMaybe(
                        ComputeOperators(
                            Target.DECLARATION,
                            static_dispatch_backend_indices=static_dispatch_idx,
                        ),
                        functions,
                    )
                ),
            },
        )
        # 使用 ops_fm 对象的 write_with_template 方法，生成名为 {name}_ops.h 的文件，模板文件为 "Operator.h"
        # 生成文件的内容由 lambda 表达式返回的字典决定，包括声明部分的运算符定义

        ops_fm.write_with_template(
            f"{name}.h",
            "Function.h",
            lambda: {
                "static_dispatch_ops_headers": list(
                    mapMaybe(
                        lambda fn: static_dispatch_ops_header(
                            fn, backend_index=static_dispatch_idx
                        ),
                        functions,
                    )
                ),
                "operator_includes": f"#include <ATen/ops/{name}_ops.h>",
                "function_definitions": list(
                    mapMaybe(
                        ComputeFunction(),
                        functions,
                    )
                ),
            },
        )
        # 使用 ops_fm 对象的 write_with_template 方法，生成名为 {name}.h 的文件，模板文件为 "Function.h"
        # 生成文件的内容由 lambda 表达式返回的字典决定，包括静态分发操作头、运算符头文件包含和函数定义部分

        grouped_functions = grouped_functions_by_root_name.get(name, [])
        structured_functions = [
            fn
            for fn in grouped_functions
            if isinstance(fn, NativeFunctionsGroup) and fn.structured
        ]
        is_structured = len(structured_functions) > 0

        if is_structured:
            ops_fm.write_with_template(
                f"{name}_meta.h",
                "NativeMetaFunction.h",
                lambda: {
                    "meta_function_declarations": list(
                        mapMaybe(
                            compute_meta_function_declaration, structured_functions
                        )
                    ),
                },
            )
        # 如果存在结构化函数，则使用 ops_fm 对象的 write_with_template 方法
        # 生成名为 {name}_meta.h 的文件，模板文件为 "NativeMetaFunction.h"
        # 生成文件的内容由 lambda 表达式返回的字典决定，包括元函数声明部分

        declarations = get_native_function_declarations(
            grouped_native_functions=grouped_functions,
            backend_indices=backend_indices,
            native_function_decl_gen=dest.compute_native_function_declaration,
        )
        ops_fm.write_with_template(
            f"{name}_native.h",
            "NativeFunction.h",
            lambda: {
                "extra_includes": (
                    f"#include <ATen/ops/{name}_meta.h>" if is_structured else []
                ),
                "native_function_declarations": declarations,
            },
        )
        # 使用 ops_fm 对象的 write_with_template 方法，生成名为 {name}_native.h 的文件，模板文件为 "NativeFunction.h"
        # 生成文件的内容由 lambda 表达式返回的字典决定，包括额外的包含（如果有结构化函数）和本地函数声明部分

    for category, suffix in [
        ("Functions", ""),
        ("Operators", "_ops"),
        ("NativeMetaFunctions", "_meta"),
        ("NativeFunctions", "_native"),
    ]:
        # 遍历包含类别名称和后缀的列表，对每个类别生成相应的文件名后缀
    ]:
        cpu_fm.write(
            f"{category}.h",
            lambda: {
                f"{category}_includes": [
                    f"#include <ATen/ops/{name}{suffix}.h>"
                    for name in sorted(functions_by_root_name.keys())
                ],
                f"{category}_declarations": [],
            },
        )



    for dispatch_key in dispatch_keys:
        if dispatch_key not in functions_keys:
            continue

        dispatch_namespace = dispatch_key.lower()
        dispatch_names = []

        for name, functions in functions_by_root_name.items():
            grouped_functions = grouped_functions_by_root_name.get(name, [])
            declarations = list(
                concatMap(
                    dest.RegisterDispatchKey(
                        backend_indices[dispatch_key],
                        Target.NAMESPACED_DECLARATION,
                        selector,
                        rocm=rocm,
                        symint=True,
                        class_method_name=None,
                        skip_dispatcher_op_registration=False,
                    ),
                    grouped_functions,
                )
            )

            if len(declarations) == 0:
                continue

            dispatch_names.append(name)
            ops_fm.write_with_template(
                f"{name}_{dispatch_namespace}_dispatch.h",
                "DispatchKeyFunction.h",
                lambda: {
                    "dispatch_namespace": dispatch_namespace,
                    "dispatch_namespaced_declarations": declarations,
                },
            )



        fm = cuda_fm if is_cuda_dispatch_key(dispatch_key) else cpu_fm
        inl_headers = f"#include <ATen/{dispatch_key}Functions_inl.h>"

        fm.write_with_template(
            f"{dispatch_key}Functions.h",
            "DispatchKeyFunctions.h",
            lambda: {
                "dispatch_key": str(dispatch_key),
                "inline_headers": inl_headers,
            },
        )
        fm.write_with_template(
            f"{dispatch_key}Functions_inl.h",
            "DispatchKeyFunctions_inl.h",
            lambda: {
                "dispatch_namespace": dispatch_namespace,
                "DispatchKeyFunctions_inl_includes": [
                    f"#include <ATen/ops/{name}_{dispatch_namespace}_dispatch.h>"
                    for name in sorted(dispatch_names)
                ],
                "dispatch_namespaced_declarations": [],
            },
        )
        del fm



    cpu_fm.write(
        "MethodOperators.h",
        lambda: {
            "MethodOperators_includes": sorted(
                f"#include <ATen/ops/{name}_ops.h>"
                for name, functions in functions_by_root_name.items()
                if any(Variant.method in fn.variants for fn in functions)
            ),
            "MethodOperators_declarations": [],
        },
    )
# 定义函数 gen_headers，生成头文件的函数
def gen_headers(
    *,
    native_functions: Sequence[NativeFunction],  # 原生函数列表
    valid_tags: set[str],  # 有效标签集合
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],  # 分组的原生函数列表
    structured_native_functions: Sequence[NativeFunctionsGroup],  # 结构化的原生函数列表
    static_dispatch_idx: list[BackendIndex],  # 静态分发索引列表
    selector: SelectiveBuilder,  # 选择器对象
    backend_indices: dict[DispatchKey, BackendIndex],  # 后端索引字典
    core_fm: FileManager,  # 核心文件管理器对象
    cpu_fm: FileManager,  # CPU 文件管理器对象
    cuda_fm: FileManager,  # CUDA 文件管理器对象
    ops_fm: FileManager,  # 操作文件管理器对象
    dispatch_keys: Sequence[DispatchKey],  # 分发键序列
    functions_keys: set[DispatchKey],  # 函数键集合
    rocm: bool,  # 是否为 ROCm
    per_operator_headers: bool,  # 是否为每个运算符生成头文件
) -> None:
    # 如果需要为每个运算符生成头文件
    if per_operator_headers:
        # 调用 gen_per_operator_headers 函数生成每个运算符的头文件
        gen_per_operator_headers(
            native_functions=native_functions,
            grouped_native_functions=grouped_native_functions,
            static_dispatch_idx=static_dispatch_idx,
            selector=selector,
            backend_indices=backend_indices,
            cpu_fm=cpu_fm,
            cuda_fm=cuda_fm,
            ops_fm=ops_fm,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            rocm=rocm,
        )
    else:
        # 否则调用 gen_aggregated_headers 函数生成聚合的头文件
        gen_aggregated_headers(
            native_functions=native_functions,
            grouped_native_functions=grouped_native_functions,
            structured_native_functions=structured_native_functions,
            static_dispatch_idx=static_dispatch_idx,
            selector=selector,
            backend_indices=backend_indices,
            cpu_fm=cpu_fm,
            cuda_fm=cuda_fm,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            rocm=rocm,
        )

    # 将张量方法声明和定义写入核心文件管理器
    core_fm.write(
        "TensorBody.h",
        lambda: {
            "tensor_method_declarations": list(
                mapMaybe(
                    ComputeTensorMethod(
                        target=Target.DECLARATION,
                        static_dispatch_backend_indices=static_dispatch_idx,
                    ),
                    native_functions,
                )
            ),
            "tensor_method_definitions": list(
                mapMaybe(
                    ComputeTensorMethod(
                        target=Target.DEFINITION,
                        static_dispatch_backend_indices=static_dispatch_idx,
                    ),
                    native_functions,
                )
            ),
        },
    )

    # 将重新分发函数的定义写入 CPU 文件管理器
    cpu_fm.write(
        "RedispatchFunctions.h",
        lambda: {
            "function_redispatch_definitions": list(
                mapMaybe(ComputeRedispatchFunction(), native_functions)
            ),
        },
    )

    # 将注册声明写入 CPU 文件管理器
    cpu_fm.write(
        "RegistrationDeclarations.h",
        lambda: {
            "registration_declarations": [
                compute_registration_declarations(f, backend_indices)
                for f in native_functions
            ],
        },
    )

    # 将 Vmap 生成的管道代码写入 CPU 文件管理器
    cpu_fm.write(
        "VmapGeneratedPlumbing.h", lambda: gen_all_vmap_plumbing(native_functions)
    )
    # 定义一个函数，生成 ATen 库中的函数名和参数名的符号表
    def gen_aten_interned_strings() -> dict[str, str]:
        # 存储所有函数参数名的集合
        attrs: set[str] = set()  # All function argument names
        # 存储所有 ATen 函数名的集合
        names = set()  # All ATen function names
        
        # 遍历 native_functions 中的每个函数对象
        for func in native_functions:
            # 将函数对象的名称添加到 names 集合中
            names.add(str(func.func.name.name))
            # 对于一些没有功能变体的运算符，我们仍然创建一个没有下划线的符号
            names.add(func.func.name.name.base)

            # 更新 attrs 集合，包含每个函数的参数名称
            attrs.update(arg.name for arg in func.func.schema_order_arguments())

        # 这些是 C++ 中的关键字，因此不是有效的符号名
        # https://en.cppreference.com/w/cpp/language/operator_alternative
        # 从 names 集合中移除这些关键字
        names -= {
            "and",
            "and_eq",
            "bitand",
            "bitor",
            "compl",
            "not",
            "not_eq",
            "or",
            "or_eq",
            "xor",
            "xor_eq",
        }

        # 返回一个包含两个键值对的字典
        return {
            # 生成 ATen 符号的字符串，按名称排序并用空格和换行符连接
            "aten_symbols": " \\\n".join(
                [f"_(aten, {name})" for name in sorted(names)]
            ),
            # 生成属性符号的字符串，按名称排序并用空格和换行符连接
            "attr_symbols": " \\\n".join(
                [f"_(attr, {name})" for name in sorted(attrs)]
            ),
        }

    # 将生成的 ATen 符号字符串写入 core_fm 对象的文件 "aten_interned_strings.h"
    core_fm.write("aten_interned_strings.h", gen_aten_interned_strings)

    # 定义一个函数，生成有效标签的枚举字符串
    def gen_tags_enum() -> dict[str, str]:
        # 返回一个包含单一键值对的字典，枚举有效标签的字符串按名称排序并用逗号和换行符连接
        return {"enum_of_valid_tags": (",\n".join(sorted(valid_tags)))}

    # 将生成的枚举有效标签字符串写入 core_fm 对象的文件 "enum_tag.h"
    core_fm.write("enum_tag.h", gen_tags_enum)
def gen_source_files(
    *,
    native_functions: Sequence[NativeFunction],  # 传入原生函数列表
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],  # 分组后的原生函数列表
    structured_native_functions: Sequence[NativeFunctionsGroup],  # 结构化的原生函数组列表
    view_groups: Sequence[NativeFunctionsViewGroup],  # 视图组列表
    selector: SelectiveBuilder,  # 选择构建器对象
    static_dispatch_idx: list[BackendIndex],  # 静态调度索引列表
    backend_indices: dict[DispatchKey, BackendIndex],  # 后端索引字典
    aoti_fm: FileManager,  # AOTInductor 文件管理器
    core_fm: FileManager,  # 核心文件管理器
    cpu_fm: FileManager,  # CPU 文件管理器
    cpu_vec_fm: FileManager,  # CPU 向量文件管理器
    cuda_fm: FileManager,  # CUDA 文件管理器
    dispatch_keys: Sequence[DispatchKey],  # 调度键序列
    functions_keys: set[DispatchKey],  # 函数键集合
    rocm: bool,  # 是否为 ROCm 平台
    force_schema_registration: bool,  # 是否强制模式注册
    per_operator_headers: bool,  # 是否使用每个操作员的头文件
    skip_dispatcher_op_registration: bool,  # 是否跳过分发操作注册
    update_aoti_c_shim: bool,  # 是否更新 AOTInductor C 语言接口的 shim 文件
) -> None:
    # 额外的 CUDA 头文件字符串
    extra_cuda_headers = """\
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/cuda/CUDAContext.h>"""
    
    # 如果是 ROCm 平台，更新额外的 CUDA 头文件字符串为 ROCm 版本
    if rocm:
        extra_cuda_headers = """\
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/ATenHIPGeneral.h>
#include <ATen/hip/HIPDevice.h>
#include <ATen/hip/HIPContext.h>"""
                        """
                # 尝试生成此操作的正确的 C++ 调用代码。如需帮助，请联系 AOTInductor 团队。
                except FileNotFoundError:
                    # 如果文件未找到，打印相应的错误信息
                    print(
                        f"{os.path.join(aoti_fm.install_dir, header_file_name)} not found"
                    )

            # cpp 文件总是即时生成的
            def headers_for_aoti() -> str:
                # 存储所有的头文件
                headers = []
                # 遍历所有后备的本地函数
                for func in fallback_native_functions:
                    # 获取针对 AOTInductor 的头文件
                    header = get_header_for_aoti(
                        func, structured_func_group_dict, dispatch_key, backend_indices
                    )
                    # 如果头文件不为空，则添加到列表中
                    if header is not None:
                        headers.append(header)
                # 返回所有头文件的排序且去重的字符串形式
                return "\n".join(sorted(set(headers)))

            # 根据是否是 CUDA 分发键决定是否使用额外的 CUDA 头文件
            extra_headers = (
                extra_cuda_headers if is_cuda_dispatch_key(dispatch_key) else ""
            )

            # 写入 C++ Shim 文件
            aoti_fm.write(
                f"c_shim_{dispatch_key.lower()}.cpp",
                lambda: gen_aoti_c_shim(
                    fallback_native_functions,
                    structured_func_group_dict,
                    dispatch_key,
                    backend_indices,
                    header=False,
                    includes=headers_for_aoti() + "\n" + extra_headers,
                ),
            )

        # 删除文件管理器对象 fm
        del fm

    # BackendSelect 是特别生成的
    def gen_backend_select() -> dict[str, list[str]]:
        # 选择需要后端选择的本地函数
        relevant_fns = [
            fn for fn in native_functions if needs_backend_select(fn, selector)
        ]
        # 返回包含后端选择相关信息的字典
        return {
            "ops_headers": [
                f"#include <ATen/ops/{fn.root_name}_ops.h>" for fn in relevant_fns
            ],
            "backend_select_method_definitions": list(
                mapMaybe(
                    ComputeBackendSelect(Target.DEFINITION, selector), relevant_fns
                )
            ),
            "backend_select_function_registrations": list(
                mapMaybe(
                    ComputeBackendSelect(Target.REGISTRATION, selector), relevant_fns
                )
            ),
        }

    # 写入 RegisterBackendSelect.cpp 文件
    cpu_fm.write("RegisterBackendSelect.cpp", gen_backend_select)

    # 设置 schema_selector 为选择器
    schema_selector = selector
    # 如果需要强制注册 schema，则设置 schema_selector 为 nop_selector
    if force_schema_registration:
        schema_selector = SelectiveBuilder.get_nop_selector()

    # 获取本地函数的 schema 注册信息
    (
        aten_schema_registrations,
        schema_registrations,
    ) = get_native_function_schema_registrations(
        native_functions=native_functions, schema_selector=schema_selector
    )
    # 写入 RegisterSchema.cpp 文件
    cpu_fm.write(
        "RegisterSchema.cpp",
        lambda: {
            "aten_schema_registrations": []
            if skip_dispatcher_op_registration
            else aten_schema_registrations,
            "schema_registrations": []
            if skip_dispatcher_op_registration
            else schema_registrations,
        },
    )

    # 定义用于排序的 key_func 函数
    def key_func(
        fn: NativeFunction | NativeFunctionsGroup | NativeFunctionsViewGroup,
    ) -> str:
        return fn.root_name


# 定义一个函数，接受一个函数对象 fn 作为参数，并返回 fn 的 root_name 属性



    cpu_fm.write_sharded(
        "Operators.cpp",
        native_functions,
        key_fn=key_func,
        env_callable=lambda fn: {
            "operator_headers": [f"#include <ATen/ops/{fn.root_name}.h>"],
            "definitions": [
                ComputeOperators(
                    Target.DEFINITION,
                    static_dispatch_backend_indices=static_dispatch_idx,
                )(fn)
            ],
        },
        base_env={
            "static_dispatch_extra_headers": static_dispatch_extra_headers(
                static_dispatch_idx
            ),
        },
        num_shards=5,
        sharded_keys={
            "operator_headers",
            "definitions",
            "static_dispatch_extra_headers",
        },
    )


# 使用 cpu_fm 对象的 write_sharded 方法，将数据写入多个分片文件
# 参数说明:
# - "Operators.cpp": 目标文件名
# - native_functions: 包含本地函数的数据结构
# - key_fn=key_func: 用于生成数据分片键的函数
# - env_callable=lambda fn: {...}: 用于生成每个分片环境的回调函数，fn 是每个本地函数对象
# - base_env: 基础环境字典，包含静态分发额外头文件的配置
# - num_shards=5: 分片数量
# - sharded_keys: 分片的键集合，用于指定分片的内容



    cpu_fm.write("Functions.cpp", dict)


# 使用 cpu_fm 对象的 write 方法，将字典 dict 写入文件 "Functions.cpp"



    core_fm.write("TensorMethods.cpp", dict)


# 使用 core_fm 对象的 write 方法，将字典 dict 写入文件 "TensorMethods.cpp"



    core_fm.write(
        "ATenOpList.cpp",
        lambda: {
            "aten_ops": list(mapMaybe(compute_aten_op, native_functions)),
        },
    )


# 使用 core_fm 对象的 write 方法，将由 lambda 函数生成的字典写入文件 "ATenOpList.cpp"
# lambda 函数生成的字典包含键 "aten_ops"，其值是对 native_functions 中每个函数应用 compute_aten_op 函数后的结果列表



    def functionalization_env_callable(
        g: NativeFunction | NativeFunctionsGroup | NativeFunctionsViewGroup,


# 定义一个名为 functionalization_env_callable 的函数，接受参数 g，类型可以是 NativeFunction、NativeFunctionsGroup 或 NativeFunctionsViewGroup
    ) -> dict[str, list[str]]:
        # 定义一个内部函数，用于生成操作的头文件列表
        def gen_op_headers(
            g: NativeFunction | NativeFunctionsGroup | NativeFunctionsViewGroup,
        ) -> list[str]:
            # 如果操作组是视图操作组（NativeFunctionsViewGroup）
            if isinstance(g, NativeFunctionsViewGroup):
                # 视图操作总是包含一个功能化内核
                headers = [
                    f"#include <ATen/ops/{g.view.root_name}_native.h>",
                    f"#include <ATen/ops/{g.view.root_name}_ops.h>",
                ]
                # 如果视图操作组有复制视图，则添加其头文件
                if g.view_copy is not None:
                    headers += [
                        f"#include <ATen/ops/{g.view_copy.root_name}_native.h>",
                        f"#include <ATen/ops/{g.view_copy.root_name}_ops.h>",
                    ]
                return headers
            # 如果操作组是普通操作组（NativeFunctionsGroup）
            elif isinstance(g, NativeFunctionsGroup):
                headers = [
                    f"#include <ATen/ops/{g.functional.root_name}_native.h>",
                    f"#include <ATen/ops/{g.functional.root_name}_ops.h>",
                    f"#include <ATen/ops/{g.out.root_name}_native.h>",
                    f"#include <ATen/ops/{g.out.root_name}_ops.h>",
                ]
                # 如果操作组有原地操作，则添加其头文件
                if g.inplace is not None:
                    headers += [
                        f"#include <ATen/ops/{g.inplace.root_name}_native.h>",
                        f"#include <ATen/ops/{g.inplace.root_name}_ops.h>",
                    ]
                # 如果操作组有可变操作，则添加其头文件
                if g.mutable is not None:
                    headers += [
                        f"#include <ATen/ops/{g.mutable.root_name}_native.h>",
                        f"#include <ATen/ops/{g.mutable.root_name}_ops.h>",
                    ]
                return headers
            # 如果不是以上两种操作组，则默认为单个操作（NativeFunction）
            else:
                return [
                    f"#include <ATen/ops/{g.root_name}_native.h>",
                    f"#include <ATen/ops/{g.root_name}_ops.h>",
                ]

        # 返回一个字典，包含生成的操作头文件列表、功能化定义和功能化注册
        return {
            "ops_headers": gen_op_headers(g),
            "func_definitions": gen_functionalization_definition(
                selector,
                g,
            ),
            "func_registrations": gen_functionalization_registration(
                selector,
                g,
                backend_indices[DispatchKey.CompositeImplicitAutograd],
            ),
        }

    # 获取所有操作组成的列表，包括 NativeFunction、NativeFunctionsGroup 和 NativeFunctionsViewGroup
    all_groups: list[
        NativeFunction | NativeFunctionsGroup | NativeFunctionsViewGroup
    ] = list(structured_native_functions) + list(
        view_groups  # type: ignore[assignment, arg-type, operator]
    )
    # 注意：所有需要功能化处理的操作（包括可变和别名操作）应当被适当地分组。
    # 我们需要直接处理 NativeFunction 而不是组，唯一的原因是：
    # (1) 我们可以提供更好的错误检查（如果有人引入不符合分组逻辑的可变操作，则报错）。
    # (2) 功能化需要手动注册 CompositeImplicitAutograd 内核，这些内核可能未被分组。
    # 创建结构化映射，将操作符名称映射到对应的原生函数对象
    structured_map: dict[OperatorName, NativeFunction] = {
        f.func.name: f
        for f in concatMap(lambda g: list(g.functions()), structured_native_functions)
    }
    # 创建视图映射，将操作符名称映射到对应的原生函数对象
    view_map: dict[OperatorName, NativeFunction] = {
        f.func.name: f for f in concatMap(lambda g: list(g.functions()), view_groups)
    }
    # 遍历所有原生函数，如果函数既不在结构化映射中也不在视图映射中，则将其加入到all_groups列表中
    for f in native_functions:
        if f.func.name not in structured_map and f.func.name not in view_map:
            all_groups.append(f)

    # 将所有分组写入到名为"RegisterFunctionalization.cpp"的文件中
    cpu_fm.write_sharded(
        "RegisterFunctionalization.cpp",
        all_groups,
        key_fn=key_func,
        env_callable=functionalization_env_callable,
        num_shards=4,
        sharded_keys={
            "ops_headers",
            "func_definitions",
            "func_registrations",
            "func_add_back_views_definitions",
            "func_add_back_views_registrations",
        },
    )

    # 将视图逆声明写入到名为"FunctionalInverses.h"的文件中
    cpu_fm.write(
        "FunctionalInverses.h",
        lambda: {
            "view_inverse_declarations": list(
                mapMaybe(
                    lambda g: gen_functionalization_view_inverse_declaration(
                        selector, g
                    ),
                    view_groups,
                )
            )
        },
    )

    # 注意 [view_copy NativeFunctions]
    # 对于native_functions.yaml中的每个视图操作符，除了CompositeImplicitAutograd，都需要对应的非别名视图副本{view}_copy。
    # 使用功能化的后端，不知道如何处理别名操作符的操作将会使用这些{view}_copy内核。
    # 核心代码中的{view}_copy操作符代码非常重复，因此我们进行代码生成：
    # (1) 为每个{view}_copy操作符生成一个CompositeExplicitAutogradNonFunctional内核。
    #     这些内核从不被功能化通道显式调用，但理论上可以从用户代码中调用（我为了完整性添加了这些内核，因为这些操作符是公共API的一部分）。
    # (2) 为每个{view}_copy操作符生成一个导数公式。
    #     {view}_copy操作符可以重用其{view}操作符对应的导数公式，因此我们不需要在derivatives.yaml中为它们生成所有条目，
    #     而是在代码中进行代码生成。
    #     这类似于自动求导代码生成不要求原地操作必须在derivatives.yaml中有条目的方式。
    cpu_fm.write(
        "CompositeViewCopyKernels.cpp",
        # 将内容写入名为 "CompositeViewCopyKernels.cpp" 的文件
        lambda: {
            "ops_headers": [
                "\n".join(
                    f"#include <ATen/ops/{f.root_name}_ops.h>\n"
                    # 注意：这个包含很重要，它确保我们正确设置生成的视图复制内核的可见性
                    f"#include <ATen/ops/{f.root_name}_native.h>"
                    for f in (
                        [g.view] if g.view_copy is None else [g.view, g.view_copy]
                    )
                )
                for g in view_groups
            ]
            + [
                "\n".join(
                    f"#include <ATen/ops/{f.root_name}_ops.h>"
                    # 对于每个结构化的本地函数组，如果条件满足，包含其头文件
                    for f in [g.inplace, g.mutable, g.functional]
                    if f is not None and "generated" not in f.tags
                )
                for g in structured_native_functions
            ],
            "CompositeViewCopyKernel_Definitions": list(
                mapMaybe(
                    # 生成复合视图复制内核的定义
                    GenCompositeViewCopyKernel(
                        backend_indices[
                            DispatchKey.CompositeExplicitAutogradNonFunctional
                        ]
                    ),
                    view_groups,
                )
            ),
            "GeneratedCompositeFunctional_Definitions": list(
                mapMaybe(
                    # 生成复合功能内核的定义
                    gen_composite_functional_kernel,
                    structured_native_functions,
                )
            ),
            "GeneratedCompositeOut_Definitions": list(
                mapMaybe(
                    # 生成复合输出内核的定义
                    gen_composite_out_kernel,
                    structured_native_functions,
                )
            ),
        },
    )
# 生成 Declarations.yaml 文件，将给定的原生函数列表转换成 YAML 格式并写入文件
def gen_declarations_yaml(
    cpu_fm: FileManager, native_functions: Sequence[NativeFunction]
) -> None:
    # 使用 FileManager 对象写入 Declarations.yaml 文件，内容为原生函数列表转换成的 YAML 格式
    cpu_fm.write(
        "Declarations.yaml",
        lambda: format_yaml([compute_declaration_yaml(f) for f in native_functions]),
    )


# 获取当前脚本文件的父目录的绝对路径，用于定位 torchgen 的根目录
def get_torchgen_root() -> Path:
    """
    If you're depending on torchgen out-of-tree, you can use the root to figure
    out the path to native_functions.yaml
    """
    return Path(__file__).parent.resolve()


# 主函数，用于生成 ATen 源文件
def main() -> None:
    # 创建命令行解析对象
    parser = argparse.ArgumentParser(description="Generate ATen source files")
    
    # 添加命令行参数：源目录路径
    parser.add_argument(
        "-s",
        "--source-path",
        help="path to source directory for ATen",
        default="aten/src/ATen",
    )
    
    # 添加命令行参数：输出依赖列表到指定文件
    parser.add_argument(
        "-o",
        "--output-dependencies",
        help="output a list of dependencies into the given file and exit",
    )
    
    # 添加命令行参数：仅运行而不写入文件
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run without writing any files (still updates outputs)",
    )
    
    # 添加命令行参数：每个操作符生成单独的头文件
    parser.add_argument(
        "--per-operator-headers",
        action="store_true",
        help="generate separate headers per operator in ATen/ops",
    )
    
    # 添加命令行参数：输出目录，默认为 build/aten/src/ATen
    parser.add_argument(
        "-d",
        "--install-dir",
        "--install_dir",
        help="output directory",
        default="build/aten/src/ATen",
    )
    
    # 添加命令行参数：AOTInductor shim 的输出目录，默认为 torch/csrc/inductor/aoti_torch/generated
    parser.add_argument(
        "--aoti-install-dir",
        "--aoti_install_dir",
        help="output directory for AOTInductor shim",
        default="torch/csrc/inductor/aoti_torch/generated",
    )
    
    # 添加命令行参数：将 CUDA 视为 ROCm/HIP 并相应调整文件路径
    parser.add_argument(
        "--rocm",
        action="store_true",
        help="reinterpret CUDA as ROCm/HIP and adjust filepaths accordingly",
    )
    
    # 添加命令行参数：生成 MPS 注册代码
    parser.add_argument(
        "--mps",
        action="store_true",
        help="Generate MPS registration code when set",
    )
    
    # TODO: --op-registration-whitelist 将在所有调用 gen.py 的地方都转移到使用操作符 YAML 文件进行移动自定义构建时被移除。
    # 添加命令行参数：操作符注册白名单，用于过滤操作符的注册
    parser.add_argument(
        "--op-registration-whitelist",
        "--op_registration_whitelist",
        nargs="*",
        help="filter op registrations by the whitelist (if set); "
        "each item is `namespace`::`operator name` without overload name; "
        "e.g.: aten::empty aten::conv2d ...",
    )
    
    # 添加命令行参数：操作符选择 YAML 文件路径，用于自定义构建时选择操作符
    parser.add_argument(
        "--op-selection-yaml-path",
        "--op_selection_yaml_path",
        help="Provide a path to the operator selection (for custom build) YAML "
        "that contains the information about the set of selected operators "
        "and their categories (training, ...). Each operator is either a "
        "full operator name with overload or just a bare operator name. "
        "The operator names also contain the namespace prefix (e.g. aten::)",
    )
    # 添加命令行参数 `--backend-whitelist` 或 `--backend_whitelist`，接受任意数量的参数作为白名单，用于过滤分发后端
    parser.add_argument(
        "--backend-whitelist",
        "--backend_whitelist",
        nargs="*",
        help="filter dispatch backend by the whitelist (if set), "
        "e.g.: CPU CUDA QuantizedCPU ...",
    )
    
    # 添加命令行参数 `--static-dispatch-backend` 或 `--static_dispatch_backend`，接受任意数量的参数作为静态分发后端，用于生成特定后端的静态分发代码
    parser.add_argument(
        "--static-dispatch-backend",
        "--static_dispatch_backend",
        nargs="*",
        help="generate static dispatch code for the specific backend (if set)",
    )
    
    # 添加命令行参数 `--skip-dispatcher-op-registration` 或 `--skip_dispatcher_op_registration`，设置为真时避免将运算符注册到分发器中
    parser.add_argument(
        "--skip-dispatcher-op-registration",
        "--skip_dispatcher_op_registration",
        action="store_true",
        help="Avoid registering operators into the dispatcher.",
    )
    
    # 添加命令行参数 `--force-schema-registration` 或 `--force_schema_registration`，设置为真时强制生成所有运算符的仅模式注册，即使它们未在白名单中列出
    parser.add_argument(
        "--force-schema-registration",
        "--force_schema_registration",
        action="store_true",
        help="force it to generate schema-only registrations for all ops, including"
        "those that are not listed on --op-registration-whitelist",
    )
    
    # 添加命令行参数 `--generate`，接受一组选择（headers, sources, declarations_yaml）作为生成文件的子集，默认为全部生成
    parser.add_argument(
        "--generate",
        type=str,
        nargs="*",
        choices=["headers", "sources", "declarations_yaml"],
        default=["headers", "sources", "declarations_yaml"],
        help="Generate only a subset of files",
    )
    
    # 添加命令行参数 `--update-aoti-c-shim`，设置为真时更新 AOTInductor C shim，在向 torchgen/aoti/fallback_ops.py 的 inductor_fallback_ops 添加条目后使用，警告：除非确切知道操作，否则不要使用
    parser.add_argument(
        "--update-aoti-c-shim",
        action="store_true",
        help="Update AOTInductor C shim after adding an entry to inductor_fallback_ops in torchgen/aoti/fallback_ops.py. "
        "WARNING: Do not use this unless you are sure what you are doing!!!",
    )

    # 解析命令行参数并将其保存在 options 变量中
    options = parser.parse_args()

    # 根据 options 中的参数设置选择器，用于获取定制构建的选择器对象
    selector = get_custom_build_selector(
        options.op_registration_whitelist,
        options.op_selection_yaml_path,
    )

    # 拼接本地函数的 YAML 路径
    native_yaml_path = os.path.join(options.source_path, "native/native_functions.yaml")
    tags_yaml_path = os.path.join(options.source_path, "native/tags.yaml")

    # 从 torchgen.model 中导入 dispatch_keys
    from torchgen.model import dispatch_keys

    # 如果 options.mps 为假，则添加 DispatchKey.MPS 到忽略键集合，并从 dispatch_keys 中删除相应键
    ignore_keys = set()
    if not options.mps:
        ignore_keys.add(DispatchKey.MPS)

        if DispatchKey.MPS in dispatch_keys:
            del dispatch_keys[dispatch_keys.index(DispatchKey.MPS)]

    # 解析本地 YAML 文件，忽略指定的键，并获取全局解析标签缓存中的有效标签
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path, ignore_keys)
    valid_tags = _GLOBAL_PARSE_TAGS_YAML_CACHE[tags_yaml_path]
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )

    # 获取分组后的本地函数列表，仅包括 NativeFunctionsGroup 类型的分组
    grouped_native_functions = get_grouped_native_functions(native_functions)

    # 获取分组后的具有视图组的本地函数列表
    native_functions_with_view_groups = get_grouped_by_view_native_functions(
        native_functions
    )

    # 获取仅包含 NativeFunctionsViewGroup 类型的视图组列表
    view_groups = [
        g
        for g in native_functions_with_view_groups
        if isinstance(g, NativeFunctionsViewGroup)
    ]

    # 注意：在此处强制禁止使用 os.path.join，因为安装目录是特定的
    # 将会被 cmake 摄取，而 cmake 不支持 Windows 风格的路径斜杠。
    # 如果你改用 os.path.join，会出现以下错误：
    #
    #   cmake 代码解析字符串时出现语法错误
    #
    #     C:/Jenkins/workspace/pytorch-builds/pytorch-win-ws2016-cuda9-cudnn7-py3-build/build/aten/src/ATen\core/TensorMethods.h
    #
    #   无效的字符转义 '\c'。
    core_install_dir = f"{options.install_dir}/core"
    # 使用 Path 创建目录，如果父目录不存在则创建，确保目录存在
    Path(core_install_dir).mkdir(parents=True, exist_ok=True)
    ops_install_dir = f"{options.install_dir}/ops"
    # 使用 Path 创建目录，如果父目录不存在则创建，确保目录存在
    Path(ops_install_dir).mkdir(parents=True, exist_ok=True)
    aoti_install_dir = f"{options.aoti_install_dir}"
    # 使用 Path 创建目录，如果父目录不存在则创建，确保目录存在
    Path(aoti_install_dir).mkdir(parents=True, exist_ok=True)

    # 创建文件管理器对象，用于指定目录的文件操作
    core_fm = make_file_manager(options=options, install_dir=core_install_dir)
    # 创建文件管理器对象，用于默认目录的文件操作
    cpu_fm = make_file_manager(options=options)
    # 创建文件管理器对象，用于默认目录的文件操作
    cpu_vec_fm = make_file_manager(options=options)
    # 创建文件管理器对象，用于默认目录的文件操作
    cuda_fm = make_file_manager(options=options)
    # 创建文件管理器对象，用于指定目录的文件操作
    ops_fm = make_file_manager(options=options, install_dir=ops_install_dir)
    # 创建文件管理器对象，用于指定目录的文件操作
    aoti_fm = make_file_manager(options=options, install_dir=aoti_install_dir)

    # 仅为有限一组分发键生成 CPUFunctions.h 头文件；这是该组合
    functions_keys = {
        DispatchKey.CPU,
        DispatchKey.CUDA,
        DispatchKey.CompositeImplicitAutograd,
        DispatchKey.CompositeImplicitAutogradNestedTensor,
        DispatchKey.CompositeExplicitAutograd,
        DispatchKey.CompositeExplicitAutogradNonFunctional,
        DispatchKey.Meta,
    }

    # 如果 options 中包含 mps，则添加 DispatchKey.MPS 到 functions_keys
    if options.mps:
        functions_keys.add(DispatchKey.MPS)

    # 如果 options 包含 backend_whitelist，则仅保留支持的 dispatch keys
    if options.backend_whitelist:
        dispatch_keys = [
            k
            for k in dispatch_keys
            if is_generic_dispatch_key(k) or str(k) in options.backend_whitelist
        ]

    # 定义静态分发索引列表
    static_dispatch_idx: list[BackendIndex] = []
    # 如果 options 包含 static_dispatch_backend，则解析并加入 static_dispatch_idx
    if options.static_dispatch_backend:
        static_dispatch_idx = [
            backend_indices[DispatchKey.parse(key)]
            for key in options.static_dispatch_backend
        ]
        # 遍历 static_dispatch_backend，将其转换为 DispatchKey 并加入 functions_keys
        for key in options.static_dispatch_backend:
            dp_key = DispatchKey.parse(key)
            if dp_key not in functions_keys:
                functions_keys.add(dp_key)
    # 如果在生成选项中包含 "sources"，则生成源文件
    if "sources" in options.generate:
        # 调用函数生成源文件，传入多个参数
        gen_source_files(
            native_functions=native_functions,  # 原生函数列表
            grouped_native_functions=grouped_native_functions,  # 分组的原生函数
            structured_native_functions=structured_native_functions,  # 结构化的原生函数
            view_groups=view_groups,  # 视图组
            selector=selector,  # 选择器
            static_dispatch_idx=static_dispatch_idx,  # 静态分发索引
            backend_indices=backend_indices,  # 后端索引
            aoti_fm=aoti_fm,  # aoti_fm
            core_fm=core_fm,  # core_fm
            cpu_fm=cpu_fm,  # cpu_fm
            cpu_vec_fm=cpu_vec_fm,  # cpu_vec_fm
            cuda_fm=cuda_fm,  # cuda_fm
            dispatch_keys=dispatch_keys,  # 分发键
            functions_keys=functions_keys,  # 函数键
            rocm=options.rocm,  # 是否为 ROCm
            force_schema_registration=options.force_schema_registration,  # 强制模式注册
            per_operator_headers=options.per_operator_headers,  # 每个操作符的头文件
            skip_dispatcher_op_registration=options.skip_dispatcher_op_registration,  # 跳过分发操作注册
            update_aoti_c_shim=options.update_aoti_c_shim,  # 更新 aoti_c_shim
        )

    # 如果在生成选项中包含 "headers"，则生成头文件
    if "headers" in options.generate:
        # 调用函数生成头文件，传入多个参数
        gen_headers(
            native_functions=native_functions,  # 原生函数列表
            valid_tags=valid_tags,  # 有效标签
            grouped_native_functions=grouped_native_functions,  # 分组的原生函数
            structured_native_functions=structured_native_functions,  # 结构化的原生函数
            static_dispatch_idx=static_dispatch_idx,  # 静态分发索引
            selector=selector,  # 选择器
            backend_indices=backend_indices,  # 后端索引
            core_fm=core_fm,  # core_fm
            cpu_fm=cpu_fm,  # cpu_fm
            cuda_fm=cuda_fm,  # cuda_fm
            ops_fm=ops_fm,  # ops_fm
            dispatch_keys=dispatch_keys,  # 分发键
            functions_keys=functions_keys,  # 函数键
            rocm=options.rocm,  # 是否为 ROCm
            per_operator_headers=options.per_operator_headers,  # 每个操作符的头文件
        )

    # 如果在生成选项中包含 "declarations_yaml"，则生成声明的 YAML 文件
    if "declarations_yaml" in options.generate:
        # 调用函数生成声明的 YAML 文件，传入原生函数列表和 cpu_fm
        gen_declarations_yaml(native_functions=native_functions, cpu_fm=cpu_fm)

    # 如果输出依赖文件为真
    if options.output_dependencies:
        # 解析输出依赖文件路径
        depfile_path = Path(options.output_dependencies).resolve()
        depfile_name = depfile_path.name
        depfile_stem = depfile_path.stem

        # 对于每个文件管理器 fm 和对应的前缀
        for fm, prefix in [
            (cpu_fm, ""),  # 对应 cpu_fm
            (cpu_vec_fm, "cpu_vec_"),  # 对应 cpu_vec_fm
            (core_fm, "core_"),  # 对应 core_fm
            (cuda_fm, "cuda_"),  # 对应 cuda_fm
            (ops_fm, "ops_"),  # 对应 ops_fm
        ]:
            # 构造变量名
            varname = prefix + depfile_stem
            # 构造文件路径
            path = depfile_path.parent / (prefix + depfile_name)
            # 将变量名和路径写入文件管理器 fm 的输出
            fm.write_outputs(varname, str(path))
# 如果当前模块被直接执行（而不是被导入），则执行以下代码
if __name__ == "__main__":
    # 调用主函数 main()
    main()
```