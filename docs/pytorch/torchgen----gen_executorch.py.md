# `.\pytorch\torchgen\gen_executorch.py`

```
# 导入必要的模块和类
from __future__ import annotations  # 允许类型注解中使用类型本身（Python 3.10 之前需要）
import argparse  # 命令行参数解析模块
import os  # 系统操作模块
from collections import defaultdict  # 默认字典模块，支持默认值的字典
from dataclasses import dataclass  # 数据类装饰器，用于定义简单的类
from pathlib import Path  # 处理文件路径的模块
from typing import Any, Callable, Sequence, TextIO, TYPE_CHECKING  # 类型注解相关的模块

import yaml  # YAML 文件解析库

# 从 torchgen 库中导入各种功能模块和类
from torchgen import dest  # 从 torchgen 库中导入 dest 模块
from torchgen.api import cpp as aten_cpp  # 导入 torchgen.api.cpp 模块，并重命名为 aten_cpp
from torchgen.api.types import (  # 导入 torchgen.api.types 模块中的多个类和函数
    CppSignature,  # C++ 函数签名类
    CppSignatureGroup,  # C++ 函数签名组类
    CType,  # C 类型类
    NamedCType  # 命名 C 类型类
)
from torchgen.context import (  # 导入 torchgen.context 模块中的多个函数
    method_with_native_function,  # 带有原生函数的方法
    method_with_nested_native_function,  # 嵌套原生函数的方法
    with_native_function_and_index  # 带有原生函数和索引的方法
)
from torchgen.executorch.api import et_cpp  # 导入 torchgen.executorch.api.et_cpp 模块
from torchgen.executorch.api.custom_ops import (  # 导入 torchgen.executorch.api.custom_ops 模块中的多个类和函数
    ComputeNativeFunctionStub,  # 计算原生函数存根类
    gen_custom_ops_registration  # 生成自定义操作注册函数
)
from torchgen.executorch.api.types import contextArg, ExecutorchCppSignature  # 导入 torchgen.executorch.api.types 模块中的多个类
from torchgen.executorch.api.unboxing import Unboxing  # 导入 torchgen.executorch.api.unboxing 模块中的 Unboxing 类
from torchgen.executorch.model import (  # 导入 torchgen.executorch.model 模块中的多个类
    ETKernelIndex,  # ET 内核索引类
    ETKernelKey,  # ET 内核键类
    ETParsedYaml  # ET 解析后的 YAML 类
)
from torchgen.executorch.parse import (  # 导入 torchgen.executorch.parse 模块中的多个函数
    ET_FIELDS,  # ET 字段
    parse_et_yaml,  # 解析 ET YAML 函数
    parse_et_yaml_struct  # 解析 ET YAML 结构函数
)
from torchgen.gen import (  # 导入 torchgen.gen 模块中的多个函数
    get_custom_build_selector,  # 获取自定义构建选择器函数
    get_native_function_declarations,  # 获取原生函数声明函数
    get_native_function_declarations_from_ns_grouped_kernels,  # 从命名空间分组内核获取原生函数声明函数
    get_native_function_schema_registrations,  # 获取原生函数模式注册函数
    LineLoader,  # 行加载器类
    parse_native_yaml  # 解析原生 YAML 函数
)
from torchgen.model import (  # 导入 torchgen.model 模块中的多个类
    BackendIndex,  # 后端索引类
    BackendMetadata,  # 后端元数据类
    DEFAULT_KERNEL_NAMESPACE,  # 默认内核命名空间
    DispatchKey,  # 调度键类
    FunctionSchema,  # 函数模式类
    Location,  # 位置类
    NativeFunction,  # 原生函数类
    NativeFunctionsGroup,  # 原生函数组类
    OperatorName,  # 操作符名称类
    Variant  # 变体类
)
from torchgen.utils import (  # 导入 torchgen.utils 模块中的多个函数和类
    context,  # 上下文相关函数
    FileManager,  # 文件管理器类
    make_file_manager,  # 创建文件管理器函数
    mapMaybe,  # 映射可选项函数
    NamespaceHelper  # 命名空间助手类
)

if TYPE_CHECKING:
    from torchgen.selective_build.selector import SelectiveBuilder  # 如果是类型检查，导入 SelectiveBuilder 类


def _sig_decl_wrapper(sig: CppSignature | ExecutorchCppSignature) -> str:
    """
    包装函数，用于获取 `sig.decl(include_context=True)`。
    对于 ATen 内核，代码生成器对 ET contextArg 没有了解，因此我们使用这个包装器来添加它。
    """
    if isinstance(sig, ExecutorchCppSignature):
        return sig.decl()  # 返回 ExecutorchCppSignature 类型的声明字符串

    returns_type = aten_cpp.returns_type(sig.func.returns).cpp_type()  # 获取返回类型的 C++ 类型字符串
    cpp_args = [a.decl() for a in sig.arguments()]  # 获取参数列表的声明字符串
    cpp_args_str = ", ".join([contextArg.decl()] + cpp_args)  # 组合上下文参数和其他参数的声明字符串
    sig_decl = f"{returns_type} {sig.name()}({cpp_args_str})"  # 构建函数签名声明字符串
    return sig_decl  # 返回函数签名声明字符串


def static_dispatch(
    sig: CppSignature | ExecutorchCppSignature,
    f: NativeFunction,
    backend_indices: list[BackendIndex],
) -> str:
    """
    对于给定的 `NativeFunction`，找出相应的原生函数并进行静态分派。如果不存在或存在多个原生函数，则报错。
    这是 register_dispatch_key.py 的简化版本。
    参数:
        sig: 要使用的原生函数的 CppSignature。
        f: 要生成静态分派的 NativeFunction。
        backend_indices: 所有可用的后端索引列表。
    """
    Return:
        C++ code to call backend-specific functions, e.g., "return at::native::add(self, other, scale);"
    """
    # 如果没有指定后端索引或者需要手动注册内核，则返回空字符串
    if len(backend_indices) == 0 or f.manual_kernel_registration:
        return ""

    # 从所有后端中筛选出具有当前函数内核的后端列表
    backends = [b for b in backend_indices if b.has_kernel(f)]
    static_block = None
    # 如果只有一个后端具有当前函数的内核
    if len(backends) == 1:
        # 获取该后端的内核元数据
        backend_metadata = backends[0].get_kernel(f)
        if backend_metadata:
            # 构建参数列表字符串
            args = ", ".join(a.name for a in sig.arguments())
            # 构建静态代码块，调用后端特定的 C++ 函数
            static_block = f"return ::{backend_metadata.cpp_namespace}::{backend_metadata.kernel}({args});"
    else:
        # 如果有多个后端具有当前函数的内核，构建静态代码块
        static_block = f"""
# 校验断言不可达消息，展示绑定到 {f.func.name} 的本地函数数量为 {len(backends)}.
"""
返回一个字符串，包含如下内容：
// {f.namespace}::{f.func}
TORCH_API inline {_sig_decl_wrapper(sig)} {{
    {static_block}
}}
"""

# 生成 Functions.h 文件，提供公共的 C++ API 功能，并提供调用分发器的框架。
@dataclass(frozen=True)
class ComputeFunction:
    static_dispatch_backend_indices: list[BackendIndex]  # 静态分发后端索引列表

    selector: SelectiveBuilder  # 选择性构建器实例

    use_aten_lib: bool  # 是否使用 Aten 库

    is_custom_op: Callable[[NativeFunction], bool]  # 判断是否为自定义操作的函数

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str | None:  # 调用实例时传入本地函数参数并返回字符串或 None
        is_method_variant = False  # 初始化是否为方法变体的标志为 False
        if not self.selector.is_root_operator(f"{f.namespace}::{f.func.name}"):
            return None  # 如果不是根操作符，则返回 None

        if Variant.function not in f.variants and Variant.method in f.variants:
            is_method_variant = True  # 如果函数没有 function 变体但有 method 变体，则设置为方法变体

        # 如果既没有只有 function 变体也没有只有 method 变体，则抛出异常
        elif not (Variant.function in f.variants and Variant.method not in f.variants):
            raise Exception(
                f"Can't handle native function {f.func} with the following variant specification {f.variants}."
            )

        # 根据是否使用 Aten 库选择不同的 C++ 签名
        sig: CppSignature | ExecutorchCppSignature = (
            CppSignatureGroup.from_native_function(
                f, method=False, fallback_binding=f.manual_cpp_binding
            ).most_faithful_signature()
            if self.use_aten_lib
            else ExecutorchCppSignature.from_native_function(f)
        )

        # 如果使用 Aten 库且不是自定义操作，则生成对应的内联函数字符串
        if self.use_aten_lib and not self.is_custom_op(f):
            comma = ", "  # 定义参数分隔符为逗号和空格

            if is_method_variant:
                return f"""
// {f.namespace}::{f.func}
TORCH_API inline {_sig_decl_wrapper(sig)} {{
    return {sig.arguments()[0].name}.{sig.name()}({comma.join(e.name for e in sig.arguments()[1:])});
}}
"""
            else:
                return f"""
// {f.namespace}::{f.func}
TORCH_API inline {_sig_decl_wrapper(sig)} {{
    return at::{sig.name()}({comma.join(e.name for e in sig.arguments())});
}}
"""

        else:
            return static_dispatch(
                sig,
                f,
                backend_indices=self.static_dispatch_backend_indices,
            )  # 否则进行静态分发调用

# 生成 RegisterCodegenUnboxedKernels.cpp 文件。
@dataclass(frozen=True)
class ComputeCodegenUnboxedKernels:
    selector: SelectiveBuilder  # 选择性构建器实例

    use_aten_lib: bool  # 是否使用 Aten 库

    @method_with_nested_native_function
    def __call__(
        self,
        unbox_kernel_entry: tuple[NativeFunction, tuple[ETKernelKey, BackendMetadata]],
        newline + '"' + (k + '",') if k != 'default' else ''
    ):
        return Kernel(
            "{f.namespace}::{f.func.name}",{newline + '"' + (k + '",') if k != 'default' else ''}
        )
    []({contextArg.defn()}, EValue** stack) {{
        // 定义一个无名函数，接受上下文参数和堆栈数组作为参数
        {code_connector.join(code_list)}
        
        // 创建事件跟踪器作用域，用于跟踪本地调用函数的性能和行为
        internal::EventTracerProfileScope event_tracer_scope(context.internal_event_tracer(), "native_call_{f.func.name}");
        
        // 执行函数调用性能分析作用域
        EXECUTORCH_SCOPE_PROF("native_call_{f.func.name}");
        
        // 调用内核函数进行实际的计算操作，传入上下文和参数列表
        {ret_prefix}{kernel_call}(context, {args_str});
        
        // 记录事件跟踪器输出日志
        {event_tracer_output_logging}
        
        // 返回执行结果给调用者
        {return_assignment}
    }}
def gen_unboxing(
    *,
    native_functions: Sequence[NativeFunction],  # 传入的原生函数序列
    cpu_fm: FileManager,  # 文件管理器对象，用于文件操作
    selector: SelectiveBuilder,  # 选择构建器对象，用于选择性地构建内容
    use_aten_lib: bool,  # 是否使用 ATen 库的标志
    kernel_index: ETKernelIndex,  # ETKernelIndex 对象，包含内核索引信息
    manual_registration: bool,  # 是否手动注册内核的标志
) -> None:
    # Iterable type for write_sharded is a Tuple of (native_function, (kernel_key, metadata))
    # 定义一个键函数，接收一个元组参数，返回一个字符串
    def key_func(
        item: tuple[NativeFunction, tuple[ETKernelKey, BackendMetadata]]
    ) -> str:
        return item[0].root_name + ":" + item[1][0].to_native_string()

    # 生成一个列表 items，包含所有原生函数及其相关内核键和元数据的元组
    items: list[tuple[NativeFunction, tuple[ETKernelKey, BackendMetadata]]] = [
        (native_function, (kernel_key, metadata))
        for native_function in native_functions
        for kernel_key, metadata in kernel_index.get_kernels(native_function).items()
    ]

    # 根据 use_aten_lib 决定使用的头文件
    header = ["Functions.h" if use_aten_lib else "NativeFunctions.h"]

    # 根据 manual_registration 决定文件名
    filename = (
        "RegisterKernels.cpp"
        if manual_registration
        else "RegisterCodegenUnboxedKernels.cpp"
    )

    # 调用 cpu_fm 的 write_sharded 方法，将数据分片写入文件
    cpu_fm.write_sharded(
        filename,
        items,  # 写入的数据项列表
        key_fn=key_func,  # 键函数，用于确定每个项的键
        env_callable=lambda unbox_kernel_entry: {
            "unboxed_kernels": [  # 环境回调函数返回的字典，包含解包内核和头文件
                ComputeCodegenUnboxedKernels(selector, use_aten_lib)(unbox_kernel_entry)
            ],
            "fn_header": header if unbox_kernel_entry == items[0] else [],  # 只在第一项写入头文件
        },
        num_shards=1,  # 分片数为 1
        sharded_keys={"unboxed_kernels", "fn_header"},  # 分片键名集合
    )
    # 将 kernel_index 转换为 BackendIndex。这是因为目前不能处理 ETKernelIndex。
    # TODO larryliu: 评估是否仍然需要此代码。如果是，则让其处理 ETKernelIndex。

    # 将 kernel_index 转换为后端索引对象 BackendIndex。
    backend_index = kernel_index._to_backend_index()

    # 使用默认字典 defaultdict 创建一个命名空间到函数列表的映射。
    ns_grouped_functions = defaultdict(list)
    for native_function in native_functions:
        # 根据 native_function 的命名空间将其分组存储。
        ns_grouped_functions[native_function.namespace].append(native_function)
    
    # 初始化一个空字符串，用于存储函数声明的字符串表示。
    functions_declarations = ""
    
    # 定义一个换行符字符串。
    newline = "\n"
    
    # 遍历每个命名空间中分组的函数列表。
    for namespace in ns_grouped_functions:
        # 使用 NamespaceHelper 类，生成命名空间的帮助器对象。
        ns_helper = NamespaceHelper(
            namespace_str=namespace,
            entity_name="",
            max_level=3,
        )
        
        # 使用 mapMaybe 函数，根据条件筛选并映射每个命名空间下的函数。
        declarations = list(
            mapMaybe(
                ComputeFunction(
                    static_dispatch_backend_indices=[backend_index],
                    selector=selector,
                    use_aten_lib=use_aten_lib,
                    is_custom_op=lambda f: custom_ops_native_functions is not None
                                          and f in custom_ops_native_functions,
                ),
                ns_grouped_functions[namespace],
            )
        )
        
        # 将每个命名空间的函数声明字符串拼接到总的函数声明字符串中。
        functions_declarations += f"""
{ns_helper.prologue}
{newline.join(declarations)}
{ns_helper.epilogue}
"""

# 返回 functions_declarations 变量，包含 C++ 声明的函数和类
def get_ns_grouped_kernels(
    *,
    native_functions: Sequence[NativeFunction],
    kernel_index: ETKernelIndex,
    native_function_decl_gen: Callable[
        [
            NativeFunctionsGroup | NativeFunction,
            ETKernelIndex,
        ],
        list[str],
    ],
) -> dict[str, list[str]]:
    # 初始化一个默认字典，用于存储命名空间到声明列表的映射关系
    ns_grouped_kernels: dict[str, list[str]] = defaultdict(list)
    
    # 遍历 native_functions 中的每个 NativeFunction 对象
    for f in native_functions:
        # 初始化一个空集合，用于存储每个 NativeFunction 对象所属的命名空间
        native_function_namespaces = set()
        
        # 获取当前 NativeFunction 对象 f 对应的所有内核
        op_kernels = kernel_index.get_kernels(f)
        
        # 遍历 op_kernels 中的每个 backend_metadata 对象
        for backend_metadata in op_kernels.values():
            # 如果 backend_metadata 不为空
            if backend_metadata:
                # 获取 backend_metadata 对象的 cpp_namespace 属性作为命名空间
                namespace = backend_metadata.cpp_namespace
                # 将该命名空间加入 native_function_namespaces 集合
                native_function_namespaces.add(namespace)
            else:
                # 如果 backend_metadata 为空，则使用默认的 KERNEL_NAMESPACE 命名空间
                namespace = DEFAULT_KERNEL_NAMESPACE
            
            # 断言 native_function_namespaces 集合中的元素数量不超过 1
            assert (
                len(native_function_namespaces) <= 1
            ), f"Codegen only supports one namespace per operator, got {native_function_namespaces}"
            
            # 将生成的函数声明添加到命名空间对应的列表中
            ns_grouped_kernels[namespace].extend(
                native_function_decl_gen(f, kernel_index)
            )
    
    # 返回命名空间到声明列表的映射关系字典
    return ns_grouped_kernels


# 生成头文件
def gen_headers(
    *,
    native_functions: Sequence[NativeFunction],
    gen_custom_ops_header: bool,
    custom_ops_native_functions: Sequence[NativeFunction],
    selector: SelectiveBuilder,
    kernel_index: ETKernelIndex,
    cpu_fm: FileManager,
    use_aten_lib: bool,
) -> None:
    """Generate headers.

    Args:
        native_functions (Sequence[NativeFunction]): a collection of NativeFunction for ATen ops.
        gen_custom_ops_header (bool): whether we should generate CustomOpsNativeFunctions.h
        custom_ops_native_functions (Sequence[NativeFunction]): a collection of NativeFunction for custom ops.
        kernel_index (ETKernelIndex): kernel collection
        cpu_fm (FileManager): file manager manages output stream
        use_aten_lib (bool): whether we are generating for PyTorch types or Executorch types.
    """
    # 初始化 ATen 头文件列表
    aten_headers = ["#include <ATen/Functions.h>"]
    
    # 初始化后端索引字典，将 CPU DispatchKey 映射到相应的后端索引
    backend_indices = {DispatchKey.CPU: kernel_index._to_backend_index()}
    
    # 如果需要生成自定义操作的头文件
    if gen_custom_ops_header:
        # 使用 cpu_fm 写入自定义操作头文件
        cpu_fm.write_with_template(
            "CustomOpsNativeFunctions.h",
            "NativeFunctions.h",
            lambda: {
                # 调用函数获取自定义操作的函数声明
                "nativeFunctions_declarations": get_native_function_declarations(
                    grouped_native_functions=custom_ops_native_functions,
                    backend_indices=backend_indices,
                    native_function_decl_gen=dest.compute_native_function_declaration,
                ),
                # 添加额外的头文件引用
                "headers": [
                    "#include <ATen/ATen.h>",
                    "#include <torch/torch.h>",
                ],
            },
        )
        # 将自定义操作头文件引用添加到 ATen 头文件列表中
        aten_headers.append('#include "CustomOpsNativeFunctions.h"')
    # 在 cpu_fm 对象上调用 write 方法，将以下内容写入文件 "Functions.h"
    cpu_fm.write(
        "Functions.h",
        # 使用 lambda 函数生成一个字典，包含静态调度额外的头文件或者包含 "NativeFunctions.h"
        lambda: {
            "static_dispatch_extra_headers": aten_headers if use_aten_lib else ['#include "NativeFunctions.h"'],
            # 生成函数声明，包括原生函数、核心索引、选择器等信息
            "Functions_declarations": gen_functions_declarations(
                native_functions=native_functions,
                kernel_index=kernel_index,
                selector=selector,
                use_aten_lib=use_aten_lib,
                custom_ops_native_functions=custom_ops_native_functions,
            ),
        },
    )
    
    # 在 cpu_fm 对象上调用 write 方法，将以下内容写入文件 "RegisterKernels.h"
    cpu_fm.write(
        "RegisterKernels.h",
        # 使用 lambda 函数生成一个字典，包含生成注释信息的内容
        lambda: {
            "generated_comment": "@" + "generated by torchgen/gen_executorch.py",
        },
    )
    
    # 定义头文件的字典
    headers = {
        "headers": [
            "#include <executorch/runtime/core/exec_aten/exec_aten.h> // at::Tensor etc.",
            "#include <executorch/codegen/macros.h> // TORCH_API",
            "#include <executorch/runtime/kernel/kernel_runtime_context.h>",
        ],
    }
    
    # 如果使用 Aten 库，则在 cpu_fm 对象上调用 write 方法，将以下内容写入文件 "NativeFunctions.h"
    if use_aten_lib:
        cpu_fm.write(
            "NativeFunctions.h",
            # 使用 lambda 函数生成一个字典，包含原生函数声明以及头文件信息
            lambda: dict(
                {
                    "nativeFunctions_declarations": get_native_function_declarations(
                        grouped_native_functions=native_functions,
                        backend_indices=backend_indices,
                        native_function_decl_gen=dest.compute_native_function_declaration,
                    ),
                },
                **headers,  # 包含预定义的头文件
            ),
        )
    else:
        # 否则，获取按命名空间分组的内核函数，然后在 cpu_fm 对象上调用 write 方法，将以下内容写入文件 "NativeFunctions.h"
        ns_grouped_kernels = get_ns_grouped_kernels(
            native_functions=native_functions,
            kernel_index=kernel_index,
            native_function_decl_gen=compute_native_function_declaration,  # type: ignore[arg-type]
        )
        cpu_fm.write(
            "NativeFunctions.h",
            # 使用 lambda 函数生成一个字典，包含按命名空间分组的内核函数声明以及头文件信息
            lambda: dict(
                {
                    "nativeFunctions_declarations": get_native_function_declarations_from_ns_grouped_kernels(
                        ns_grouped_kernels=ns_grouped_kernels,
                    ),
                },
                **headers,  # 包含预定义的头文件
            ),
        )
# 定义生成自定义操作的函数，没有返回值
def gen_custom_ops(
    *,
    native_functions: Sequence[NativeFunction],  # 输入参数：原生函数序列
    selector: SelectiveBuilder,  # 输入参数：选择器对象
    kernel_index: ETKernelIndex,  # 输入参数：ETKernelIndex 对象
    cpu_fm: FileManager,  # 输入参数：文件管理器对象，用于写文件
    rocm: bool,  # 输入参数：布尔值，表示是否支持 ROCm
) -> None:
    # 设置分发键为 CPU
    dispatch_key = DispatchKey.CPU
    # 调用函数生成自定义操作的注册信息，返回匿名定义和静态初始化分发注册
    (
        anonymous_definition,
        static_init_dispatch_registrations,
    ) = gen_custom_ops_registration(
        native_functions=native_functions,
        selector=selector,
        kernel_index=kernel_index,
        rocm=rocm,
    )
    # 使用 CPU 文件管理器写入自定义操作的注册文件，使用模板
    cpu_fm.write_with_template(
        f"Register{dispatch_key}CustomOps.cpp",
        "RegisterDispatchKeyCustomOps.cpp",
        lambda: {
            "ops_headers": '#include "CustomOpsNativeFunctions.h"',
            "DispatchKey": dispatch_key,
            "dispatch_namespace": dispatch_key.lower(),
            "dispatch_namespaced_definitions": "",
            "dispatch_anonymous_definitions": anonymous_definition,
            "static_init_dispatch_registrations": static_init_dispatch_registrations,
        },
    )
    # 使用 CPU 文件管理器写入自定义操作的存根文件，使用模板
    cpu_fm.write_with_template(
        f"Register{dispatch_key}Stub.cpp",
        "RegisterDispatchKeyCustomOps.cpp",
        lambda: {
            "ops_headers": "",
            "DispatchKey": dispatch_key,
            "dispatch_namespace": dispatch_key.lower(),
            "dispatch_namespaced_definitions": "",
            "dispatch_anonymous_definitions": list(
                mapMaybe(ComputeNativeFunctionStub(), native_functions)
            ),
            "static_init_dispatch_registrations": static_init_dispatch_registrations,
        },
    )

    # 调用函数获取原生函数的模式注册信息，返回 ATen 模式和 schema 注册信息
    (
        aten_schema_registrations,
        schema_registrations,
    ) = get_native_function_schema_registrations(
        native_functions=native_functions,
        schema_selector=selector,
    )
    # 使用 CPU 文件管理器写入模式注册文件，不使用模板
    cpu_fm.write(
        "RegisterSchema.cpp",
        lambda: {
            "schema_registrations": schema_registrations,
            "aten_schema_registrations": aten_schema_registrations,
        },
    )


# 将 Executorch DSL 方言的 YAML 转换为与 native_functions.yaml 相同的语法
def translate_native_yaml(
    tags_yaml_path: str,  # 输入参数：标签 YAML 文件路径
    aten_yaml_path: str,  # 输入参数：ATen YAML 文件路径
    native_yaml_path: str | None,  # 输入参数：原生 YAML 文件路径或 None
    use_aten_lib: bool,  # 输入参数：布尔值，表示是否使用 ATen 库
    out_file: TextIO,  # 输入参数：输出文件对象
) -> None:
    """Translates Executorch DSL dialect to use the same syntax as
    native_functions.yaml. The major difference is that Executorch DSL dialect
    supports "op" key, where it refers to the operator name in native_functions.yaml.

    For example, a functions.yaml may have the following entry:

    - op: add.out
      ...

    It needs to be translated to the following:

    - func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
      ...

    We go in aten_yaml_path and find the operator schema for "add.out" and add it
    to the original functions.yaml. We also add required field "variants", where for
    Executorch it will always be "function".

    For ATen mode we don't have to do the translation because native_yaml_path is
    the same as native_functions.yaml.
    """
    def generate_native_functions(tags_yaml_path: str, aten_yaml_path: str,
                                  native_yaml_path: Optional[str], use_aten_lib: bool,
                                  out_file: IO) -> None:
        """
        Generate native functions based on YAML inputs and write them into the output file.
    
        Args:
            tags_yaml_path: Path to a tags.yaml file to satisfy codegen parsing.
                It is not optional.
            aten_yaml_path: Path to ATen operator yaml file native_functions.yaml.
            native_yaml_path: Path to a functions.yaml file to parse.
                If the path does not exist in the filesystem, it is treated as an
                empty file. If `custom_ops_yaml_path` exists, the contents of that
                file are appended to the yaml input to be parsed.
            use_aten_lib: We use this flag to determine if we want to generate native
                functions. In ATen mode we should generate out= variants.
            out_file: The IO object that we are writing into.
        Returns:
            None
        """
    
        # If use_aten_lib is True, directly read and write ATen YAML content to out_file
        if use_aten_lib:
            with open(aten_yaml_path) as aten_yaml:
                out_file.writelines(aten_yaml.readlines())
            return
    
        # Parse ATen YAML and tags YAML to get native functions and persisted fields
        native_functions, persisted_fields = parse_et_yaml(
            aten_yaml_path,
            tags_yaml_path,
            None,  # No specific tag to parse
            skip_native_fns_gen=False,  # Ensure native functions generation is not skipped
        )
    
        # Create mapping from FunctionSchema to scoped name
        func_to_scoped_name: dict[FunctionSchema, str] = {
            f.func: f"{f.namespace}::{f.func.name}" for f in native_functions
        }
    
        # Create mapping from OperatorName to scoped name
        op_to_scoped_name: dict[OperatorName, str] = {
            func.name: name for func, name in func_to_scoped_name.items()
        }
    
        # Create dictionary of function schemas mapped to their string representations
        schema_dict = {name: str(func) for func, name in func_to_scoped_name.items()}
    
        # Create dictionary of persisted kernel information
        kernel_persist_dict: dict[str, dict[str, Any]] = {
            op_to_scoped_name[op]: v for op, v in persisted_fields.items()
        }
    
        # If native_yaml_path is not provided or is empty, return early
        if (
            not native_yaml_path
            or not os.path.exists(native_yaml_path)
            or os.stat(native_yaml_path).st_size == 0
        ):
            return
    
        # Read native YAML content and modify entries according to schema and kernel persistence
        with open(native_yaml_path) as native_yaml:
            native_es = yaml.load(native_yaml, Loader=LineLoader)
            if not native_es:
                return
            for e in native_es:
                assert isinstance(e.get("__line__"), int), e
                loc = Location(native_yaml_path, e.pop("__line__"))
                with context(lambda: f"in {loc}:\n  "):
                    if "variants" not in e:
                        e["variants"] = "function"
                    if "func" in e:
                        continue
                    assert isinstance(e.get("op"), str), e
                    opname = e.pop("op")
                    if "::" not in opname:
                        opname = "aten::" + opname
                    assert opname in schema_dict
                    e["func"] = schema_dict.get(opname)
    
                    # Write out persisted kernel information
                    if opname in kernel_persist_dict:
                        for k, v in kernel_persist_dict[opname].items():
                            e[k] = v
    
            # Dump modified native ES to the output file
            yaml.dump(native_es, out_file, width=1000)
# 解析 YAML 文件，返回解析结果作为元组的一部分
def parse_yaml(
    path: str | None,                                      # 参数：YAML 文件路径或 None
    tags_yaml_path: str,                                    # 参数：标签 YAML 文件路径
    function_filter: Callable[[NativeFunction], bool],       # 参数：用于过滤函数的回调函数
    skip_native_fns_gen: bool = False,                      # 参数：是否跳过生成本地函数的标志，默认为 False
) -> tuple[
    list[NativeFunction],                                  # 返回：本地函数列表
    dict[DispatchKey, dict[OperatorName, BackendMetadata]] | ETKernelIndex,  # 返回：后端索引或 ET 核心索引
]:
    if path and os.path.exists(path) and os.stat(path).st_size > 0:  # 检查路径有效性和文件大小
        with open(path) as f:                                # 打开文件句柄
            es = yaml.load(f, Loader=LineLoader)             # 使用 LineLoader 加载 YAML 文件内容

        # 检查是否存在核心索引结构
        kernel_index = (
            parse_et_yaml_struct(es) if any("kernels" in e for e in es) else None
        )

        # 为了与旧版本兼容，从条目中删除 ET 特定字段
        for entry in es:
            for field in ET_FIELDS:
                entry.pop(field, None)

        # 解析本地 YAML 文件，生成本地函数列表
        parsed_yaml = parse_native_yaml(
            path,
            tags_yaml_path,
            None,
            skip_native_fns_gen=skip_native_fns_gen,
            loaded_yaml=es,
        )
        native_functions = list(filter(function_filter, parsed_yaml.native_functions))  # 过滤本地函数列表
        op_names = [f.func.name for f in native_functions]  # 提取本地函数的操作名称列表

        # (1) 如果存在核心索引，则返回 ETKernelIndex
        if kernel_index is not None:
            filtered_index = {
                op_name: kernel_mapping
                for op_name, kernel_mapping in kernel_index.index.items()
                if op_name in op_names
            }
            return native_functions, ETKernelIndex(index=filtered_index)

        # (2) 如果核心索引不存在，则返回后端索引
        def map_index(
            m: dict[OperatorName, BackendMetadata]
        ) -> dict[OperatorName, BackendMetadata]:
            return {op: m[op] for op in m if op in op_names}

        backend_indices = {
            k: map_index(b.index) for (k, b) in parsed_yaml.backend_indices.items()
        }

        return native_functions, backend_indices
    else:
        return [], {}  # 如果路径无效或文件为空，则返回空列表和空字典


def parse_yaml_files(
    tags_yaml_path: str,                                    # 参数：标签 YAML 文件路径
    aten_yaml_path: str,                                    # 参数：ATen YAML 文件路径
    native_yaml_path: str | None,                           # 参数：本地 YAML 文件路径或 None
    custom_ops_yaml_path: str | None,                       # 参数：自定义操作 YAML 文件路径或 None
    selector: SelectiveBuilder,                             # 参数：选择器对象
    use_aten_lib: bool,                                     # 参数：是否使用 ATen 库的标志
) -> tuple[ETParsedYaml, ETParsedYaml | None]:              # 返回：解析的 ET YAML 对象和可选的解析的 ET YAML 对象
    """Parses functions.yaml and custom_ops.yaml files.
    解析 functions.yaml 和 custom_ops.yaml 文件。
    """
    Args:
        tags_yaml_path: 需要用来满足代码生成解析的 tags.yaml 文件路径。
            这个参数是必须的。
        aten_yaml_path: ATen 操作符 yaml 文件 native_functions.yaml 的路径。
        native_yaml_path: 需要解析的 functions.yaml 文件的路径。
            如果该路径在文件系统中不存在，则视为空文件。
            如果 custom_ops_yaml_path 存在，则将其内容附加到要解析的 yaml 输入中。
        custom_ops_yaml_path: 需要解析的 custom_ops.yaml 文件的路径。
            如果该路径在文件系统中不存在，则忽略该文件。
        selector: 用于选择性构建的选择器。
        use_aten_lib: 我们使用此标志来确定是否生成本地函数。
            在 ATen 模式下，我们应该生成带有 out= 变体的函数。
    Returns:
        返回一个包含两个元素的元组：
        [0]: 连接 `native_yaml_path` 和 `custom_ops_yaml_path` 内容后的解析结果。
        [1]: `custom_ops_yaml_path` 内容的解析结果，如果存在的话；如果不存在，则为 None。
    """
    import tempfile

    # 只包括选定的操作，这是因为我们希望避免...
    def function_filter(f: NativeFunction) -> bool:
        return selector.is_native_function_selected(f)

    with tempfile.TemporaryDirectory() as tmpdirname:
        translated_yaml_path = os.path.join(tmpdirname, "translated.yaml")
        with open(translated_yaml_path, "w") as translated:
            translate_native_yaml(
                tags_yaml_path,
                aten_yaml_path,
                native_yaml_path,
                use_aten_lib,
                translated,
            )

        # 解析翻译后的 YAML 文件
        translated_functions, translated_indices = parse_yaml(
            translated_yaml_path, tags_yaml_path, function_filter, not use_aten_lib
        )

        # 解析 custom_ops YAML 文件（如果存在）
        custom_ops_functions, custom_ops_indices = parse_yaml(
            custom_ops_yaml_path, tags_yaml_path, function_filter, True
        )

        # 将 BackendIndices 转换为 ETKernelIndex
        if not isinstance(translated_indices, ETKernelIndex):
            translated_indices = ETKernelIndex.from_backend_indices(translated_indices)
        if not isinstance(custom_ops_indices, ETKernelIndex):
            custom_ops_indices = ETKernelIndex.from_backend_indices(custom_ops_indices)

        # 合并函数列表和索引
        combined_functions = translated_functions + custom_ops_functions
        combined_kernel_index = ETKernelIndex.merge_indices(
            translated_indices, custom_ops_indices
        )
        combined_yaml = ETParsedYaml(combined_functions, combined_kernel_index)
        custom_ops_parsed_yaml = ETParsedYaml(custom_ops_functions, custom_ops_indices)

    return combined_yaml, custom_ops_parsed_yaml
def main() -> None:
    # 创建参数解析器，描述为“生成操作符源文件”
    parser = argparse.ArgumentParser(description="Generate operator source files")
    
    # 添加命令行参数 "-s" 或 "--source-path"，用于指定包含模板文件夹的源目录路径
    parser.add_argument(
        "-s",
        "--source-path",
        help="path to source directory for kernel templates",
    )
    
    # 添加命令行参数 "--functions-yaml-path" 或 "--functions_yaml_path"，用于指定 functions.yaml 文件路径
    # 至少指定 "--functions-yaml-path" 或 "--custom-ops-yaml-path" 之一
    parser.add_argument(
        "--functions-yaml-path",
        "--functions_yaml_path",
        help="path to the functions.yaml file to use. Optional, but at least "
        "one of --functions-yaml-path and --custom-ops-yaml-path must be "
        "specified.",
    )
    
    # 添加命令行参数 "--custom-ops-yaml-path" 或 "--custom_ops_yaml_path"，用于指定 custom_ops.yaml 文件路径
    # 至少指定 "--functions-yaml-path" 或 "--custom-ops-yaml-path" 之一
    parser.add_argument(
        "--custom-ops-yaml-path",
        "--custom_ops_yaml_path",
        help="path to the custom_ops.yaml file to use. Optional, but at least "
        "one of --functions-yaml-path and --custom-ops-yaml-path must be "
        "specified.",
    )
    
    # 添加命令行参数 "--aten-yaml-path" 或 "--aten_yaml_path"，用于指定 native_functions.yaml 文件路径
    parser.add_argument(
        "--aten-yaml-path",
        "--aten_yaml_path",
        help="path to native_functions.yaml file.",
    )
    
    # 添加命令行参数 "-d" 或 "--install-dir" 或 "--install_dir"，用于指定输出目录，默认为 "build/generated"
    parser.add_argument(
        "-d",
        "--install-dir",
        "--install_dir",
        help="output directory",
        default="build/generated",
    )
    
    # 添加命令行参数 "-o" 或 "--output-dependencies"，用于指定输出依赖列表到给定文件中
    parser.add_argument(
        "-o",
        "--output-dependencies",
        help="output a list of dependencies into the given file and exit",
    )
    
    # 添加命令行参数 "--dry-run"，设置为 True 表示运行时不写入任何文件，但仍然更新输出
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run without writing any files (still updates outputs)",
    )
    
    # 添加命令行参数 "--static-dispatch-backend" 或 "--static_dispatch_backend"，用于生成特定后端的静态调度代码
    parser.add_argument(
        "--static-dispatch-backend",
        "--static_dispatch_backend",
        nargs="*",
        help="generate static dispatch code for the specific backend (if set)",
    )
    
    # 添加命令行参数 "--op-registration-whitelist" 或 "--op_registration_whitelist"，用于通过白名单筛选操作注册
    parser.add_argument(
        "--op-registration-whitelist",
        "--op_registration_whitelist",
        nargs="*",
        help="filter op registrations by the whitelist (if set); "
        "each item is `namespace`::`operator name` without overload name; "
        "e.g.: aten::empty aten::conv2d ...",
    )
    
    # 添加命令行参数 "--op-selection-yaml-path" 或 "--op_selection_yaml_path"，提供操作符选择的 YAML 文件路径
    parser.add_argument(
        "--op-selection-yaml-path",
        "--op_selection_yaml_path",
        help="Provide a path to the operator selection (for custom build) YAML "
        "that contains the information about the set of selected operators "
        "and their categories (training, ...). Each operator is either a "
        "full operator name with overload or just a bare operator name. "
        "The operator names also contain the namespace prefix (e.g. aten::)",
    )
    
    # 添加命令行参数 "--tags-path"，用于指定 tags.yaml 文件路径，由代码生成系统中的 YAML 解析要求
    parser.add_argument(
        "--tags-path",
        help="Path to tags.yaml. Required by yaml parsing in codegen system.",
    )
    parser.add_argument(
        "--rocm",
        action="store_true",
        help="reinterpret CUDA as ROCm/HIP and adjust filepaths accordingly",
    )
    parser.add_argument(
        "--use-aten-lib",
        "--use_aten_lib",
        action="store_true",
        help="a boolean flag to indicate whether we use ATen kernels or not, in the future this flag will be per "
        "operator",
    )
    parser.add_argument(
        "--manual_registration",
        "--manual-registration",
        action="store_true",
        help="a boolean flag to indicate whether we want to manually call"
        "register_kernels() or rely on static init. ",
    )
    parser.add_argument(
        "--generate",
        type=str,
        nargs="*",
        choices=["headers", "sources"],
        default=["headers", "sources"],
        help="Generate only a subset of files",
    )
    options = parser.parse_args()
    assert options.tags_path, "tags.yaml is required by codegen yaml parsing."

    # 根据参数配置获取自定义构建选择器
    selector = get_custom_build_selector(
        options.op_registration_whitelist,
        options.op_selection_yaml_path,
    )

    # 解析 YAML 文件，获取解析后的信息和自定义操作的解析信息（如果存在）
    parsed_yaml, custom_ops_parsed_yaml = parse_yaml_files(
        aten_yaml_path=options.aten_yaml_path,
        tags_yaml_path=options.tags_path,
        native_yaml_path=options.functions_yaml_path,
        custom_ops_yaml_path=options.custom_ops_yaml_path,
        selector=selector,
        use_aten_lib=options.use_ aten_lib,
    )
    # 获取原生函数和内核索引
    native_functions, kernel_index = (
        parsed_yaml.native_functions,
        parsed_yaml.kernel_index,
    )
    # 获取自定义操作的原生函数（如果存在）
    custom_ops_native_functions = (
        custom_ops_parsed_yaml.native_functions if custom_ops_parsed_yaml else []
    )

    # 根据选项配置创建文件管理器
    cpu_fm = make_file_manager(options=options)

    if "headers" in options.generate:
        # 当存在 custom_ops.yaml 文件时，生成 CustomOpsNativeFunctions.h，以匹配构建系统
        gen_headers(
            native_functions=native_functions,
            gen_custom_ops_header=options.custom_ops_yaml_path,
            custom_ops_native_functions=custom_ops_native_functions,
            selector=selector,
            kernel_index=kernel_index,
            cpu_fm=cpu_fm,
            use_aten_lib=options.use_aten_lib,
        )

    if "sources" in options.generate:
        # 生成解箱代码
        gen_unboxing(
            native_functions=native_functions,
            cpu_fm=cpu_fm,
            selector=selector,
            use_aten_lib=options.use_aten_lib,
            kernel_index=kernel_index,
            manual_registration=options.manual_registration,
        )
        # 如果存在自定义操作的原生函数，生成自定义操作代码
        if custom_ops_native_functions:
            gen_custom_ops(
                native_functions=custom_ops_native_functions,
                selector=selector,
                kernel_index=kernel_index,
                cpu_fm=cpu_fm,
                rocm=options.rocm,
            )
    # 如果设置了输出依赖项选项，则执行以下代码块
    if options.output_dependencies:
        # 解析并获取输出依赖项文件的路径
        depfile_path = Path(options.output_dependencies).resolve()
        # 获取输出依赖项文件的名称
        depfile_name = depfile_path.name
        # 获取输出依赖项文件的基本名称（不带扩展名）
        depfile_stem = depfile_path.stem

        # 对于每个文件管理器 fm，以指定的前缀遍历
        for fm, prefix in [
            (cpu_fm, ""),
        ]:
            # 构建变量名，可能包含指定的前缀和输出依赖项文件的基本名称
            varname = prefix + depfile_stem
            # 构建文件的完整路径，包括可能的前缀和输出依赖项文件的名称
            path = depfile_path.parent / (prefix + depfile_name)
            # 将文件管理器 fm 的输出写入指定的变量名和路径
            fm.write_outputs(varname, str(path))
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行 main() 函数
if __name__ == "__main__":
    main()
```