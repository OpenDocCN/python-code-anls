# `.\pytorch\torchgen\dest\register_dispatch_key.py`

```
# 从未来模块导入注释
from __future__ import annotations

# 导入迭代工具模块
import itertools
# 导入文本包装模块
import textwrap
# 导入数据类模块
from dataclasses import dataclass
# 导入字面量类型模块
from typing import Literal, TYPE_CHECKING

# 导入 TorchGen 中的 C++ API
import torchgen.api.cpp as cpp
# 导入 TorchGen 中的元信息 API
import torchgen.api.meta as meta
# 导入 TorchGen 中的结构化 API
import torchgen.api.structured as structured
# 导入 TorchGen 中的翻译函数
from torchgen.api.translate import translate
# 导入 TorchGen 中的类型模块
from torchgen.api.types import (
    BaseCType,
    Binding,
    ConstRefCType,
    CppSignature,
    CppSignatureGroup,
    DispatcherSignature,
    Expr,
    kernel_signature,
    MutRefCType,
    NamedCType,
    NativeSignature,
    tensorT,
)
# 导入 TorchGen 中的上下文模块
from torchgen.context import method_with_native_function, native_function_manager
# 导入 TorchGen 中的模型模块
from torchgen.model import (
    Argument,
    BackendIndex,
    DeviceCheckType,
    DispatchKey,
    gets_generated_out_inplace_wrapper,
    is_cuda_dispatch_key,
    NativeFunction,
    NativeFunctionsGroup,
    SchemaKind,
    TensorOptionsArguments,
)
# 导入 TorchGen 中的实用工具模块
from torchgen.utils import assert_never, mapMaybe, Target

# 如果是类型检查环境，导入选择构建器模块
if TYPE_CHECKING:
    from torchgen.selective_build.selector import SelectiveBuilder

# 生成注册头文件函数，返回头文件列表
def gen_registration_headers(
    backend_index: BackendIndex,
    per_operator_headers: bool,
    rocm: bool,
) -> list[str]:
    # 根据 per_operator_headers 决定是否使用特定操作符的头文件
    if per_operator_headers:
        headers = ["#include <ATen/ops/as_strided_native.h>"]
    else:
        headers = ["#include <ATen/NativeFunctions.h>"]

    # 根据后端索引的分发键选择对应的头文件
    if backend_index.dispatch_key in (DispatchKey.CPU, DispatchKey.Meta):
        headers.append("#include <ATen/EmptyTensor.h>")
    elif backend_index.dispatch_key == DispatchKey.CUDA:
        # 如果是 CUDA 分发键，根据 rocm 判断使用哪种 EmptyTensor 头文件
        if rocm:
            headers.append("#include <ATen/hip/EmptyTensor.h>")
        else:
            headers.append("#include <ATen/cuda/EmptyTensor.h>")
    elif backend_index.dispatch_key == DispatchKey.MPS:
        headers.append("#include <ATen/mps/EmptyTensor.h>")
    elif per_operator_headers:
        # 如果使用特定操作符的头文件，添加多个特定操作符的头文件
        headers += [
            "#include <ATen/ops/empty.h>",
            "#include <ATen/ops/empty_strided.h>",
            "#include <ATen/ops/_copy_from_and_resize.h>",
            "#include <ATen/ops/_copy_from.h>",
        ]
    else:
        headers.append("#include <ATen/Functions.h>")

    # 添加通用的宏定义头文件
    headers.append("#include <c10/macros/Macros.h>")
    # 返回头文件列表
    return headers

# 生成空实现函数名的元组
def gen_empty_impl_names(
    backend_index: BackendIndex,
) -> tuple[str | None, str | None]:
    # 初始化空实现名称变量
    empty_impl = None
    empty_strided_impl = None

    # 根据后端索引的分发键设置对应的空实现名称
    if backend_index.dispatch_key in (
        DispatchKey.Meta,
        DispatchKey.CPU,
        DispatchKey.CUDA,
        DispatchKey.MPS,
    ):
        dispatch = str(backend_index.dispatch_key).lower()
        empty_impl = f"at::detail::empty_{dispatch}"
        empty_strided_impl = f"at::detail::empty_strided_{dispatch}"
    elif backend_index.dispatch_key in (
        DispatchKey.CompositeExplicitAutogradNonFunctional,
        DispatchKey.QuantizedCPU,
        DispatchKey.QuantizedCUDA,
    ):
        empty_impl = "at::empty"
        empty_strided_impl = "at::empty_strided"

    # 返回空实现名称的元组
    return empty_impl, empty_strided_impl
# 生成 create_out 函数的帮助器代码，根据给定的后端索引生成相应的函数列表
def gen_create_out_helper(backend_index: BackendIndex) -> list[str]:
    # 根据后端索引的调度键决定空选项的字符串表示
    if backend_index.dispatch_key == DispatchKey.Meta:
        empty_options = "options.device(at::kMeta)"
    else:
        empty_options = "options"

    # 生成空实现和空步进实现的名称
    empty_impl, empty_strided_impl = gen_empty_impl_names(backend_index)
    # 如果空实现为 None，返回空列表
    if empty_impl is None:
        return []

    # 返回包含函数定义的列表，使用 f-string 插入空实现和空选项的字符串表示
    return [
        f"""
Tensor create_out(IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {{
  # 如果步进为空，则调用空实现函数
  if (strides.empty()) {{
      return {empty_impl}(sizes, {empty_options});
  }} else {{
      # 否则调用空步进实现函数
      return {empty_strided_impl}(sizes, strides, {empty_options});
  }}
}}
"""
    ]


# 生成 maybe_create_proxy 函数的帮助器代码，根据给定的后端索引生成相应的函数列表
def gen_maybe_create_proxy_helper(backend_index: BackendIndex) -> list[str]:
    _, empty_strided_impl = gen_empty_impl_names(backend_index)
    # 如果空步进实现为 None，返回空列表
    return (
        []
        if empty_strided_impl is None
        else [
            f"""
std::optional<Tensor> maybe_create_proxy(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {{
  # 如果输出张量的步进与给定的步进不同，则调用空步进实现函数
  if (out.strides() != strides) {{
    return {empty_strided_impl}(sizes, strides, options);
  }}
  return std::nullopt;
}}
"""
        ]
    )


# 生成 resize_out 函数的帮助器代码，根据给定的后端索引生成相应的函数列表
def gen_resize_out_helper(backend_index: BackendIndex) -> list[str]:
    if backend_index.dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
        # 对于该调度键，函数未被使用，因此返回空列表
        return []
    # 返回包含函数定义的列表，使用三重引号字符串插入相应的代码和选项
    return [
        """
void resize_out(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {
  # 检查输出张量的数据类型是否与给定选项匹配
  TORCH_CHECK(options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ", options.dtype(), ", but got ", out.dtype(), " instead");
  # 检查输出张量的设备是否与给定选项匹配
  TORCH_CHECK(options.device() == out.device(),
      "Expected out tensor to have device ", options.device(), ", but got ", out.device(), " instead");
  # 尝试调整输出张量的大小，并记录是否进行了调整
  const bool resized = at::native::resize_output(out, sizes);
  # 只有在进行了调整时才重新调整步进；否则直接使用输出张量的现有步进
  if (resized) {
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      // TODO: 避免在此处重新调度
      out.as_strided_(sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      out.unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
    }
  }
}
"""
    ]


# 生成 gen_check_inplace_helper 函数的帮助器代码，根据给定的后端索引生成相应的函数列表
def gen_check_inplace_helper(backend_index: BackendIndex) -> list[str]:
    # 返回包含函数定义的列表，使用三重引号字符串插入相应的代码和选项
    return [
        """
# 确保输入张量和输出张量的数据类型匹配
# 这些检查适用于不使用 'TensorIterator' 的运算符（如 'addmm' 和 'baddbmm'）
# 以及具有特定类型规则的运算符（如 'cumsum' 和 'cumprod'）
# 对于其他运算符（如 'add'），'TensorIterator' 已经单独检查了这些内容。
void check_inplace(const Tensor &self, IntArrayRef sizes, const TensorOptions &options) {
    # 检查输入张量和输出张量的数据类型是否一致
    TORCH_CHECK(options.dtype() == self.dtype(),
        "Bad in-place call: ",
        "input tensor dtype ", self.dtype(), " and output tensor dtype ", options.dtype(), " should match");
    # 检查输入张量和输出张量的设备是否一致
    TORCH_CHECK(options.device() == self.device(),
        "Bad in-place call: ",
        "input tensor device ", self.device(), " and output tensor device ", options.device(), " should match");
    # 检查输入张量和输出张量的尺寸是否一致
    TORCH_CHECK(sizes == self.sizes(),
        "Bad in-place call: ",
        "input tensor size ", self.sizes(), " and output tensor size ", sizes, " should match");
}

# 生成注册助手函数列表，用于特定后端索引
def gen_registration_helpers(backend_index: BackendIndex) -> list[str]:
    # 忽略编译警告 "-Wunused-function"
    return [
        'C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-function")',
        # 生成创建输出张量助手函数
        *gen_create_out_helper(backend_index),
        # 生成调整大小输出张量助手函数
        *gen_resize_out_helper(backend_index),
        # 生成检查原位操作助手函数
        *gen_check_inplace_helper(backend_index),
        # 生成可能创建代理的助手函数
        *gen_maybe_create_proxy_helper(backend_index),
        "C10_DIAGNOSTIC_POP()",
    ]

# 生成注册调度键文件（例如 RegisterCPU.cpp）
#
# - 该文件的主要功能是将给定调度键的所有实现注册到调度程序中，
#   以便它们可以在PyTorch中使用。如果调度键为None，我们生成模式（def）注册和
#   捕获所有注册。
# - 该文件的次要功能是生成函数的包装器。在CPUType中，这些包装器不起作用
#   （应该被删除），但在其他情况下，它们处理DeviceGuard。包装器的一个小额外好处是，
#   它们不是重载的，因此可以在注册API中使用，而不必区分要注册的是哪个重载
#   （如果直接注册native::函数的话会有这种情况）。
# - 该文件的第三个功能是生成*静态*的cpp API绑定，可以用于绕过调度程序，
#   直接访问内核，但使用用户友好的cpp风格API
@dataclass(frozen=True)
class RegisterDispatchKey:
    backend_index: BackendIndex

    target: Literal[
        Target.ANONYMOUS_DEFINITION,
        Target.NAMESPACED_DEFINITION,
        Target.NAMESPACED_DECLARATION,
        Target.REGISTRATION,
    ]

    # 选择器对象，用于确定要为其生成注册代码的操作符
    selector: SelectiveBuilder

    # 是否实际为 ROCm 代码生成
    rocm: bool

    # 是否生成 symint 注册或不生成。对于不关心 symint 的代码生成的外部用户，
    # 可以将此设置为 false，以获取非 SymInt 代码生成
    symint: bool
    # 所有非结构化原生函数所在的类。用于在核函数编写者添加错误签名的情况下改进编译器错误消息。
    # 这仅在非结构化内核中使用，因为结构化内核已经位于一个类中。
    # 最后，此字段当前为可选，因为它仅由外部后端使用。
    # 如果我们可以将相同的逻辑添加到内核中，那将非常好，但这需要更新位于aten/src/ATen/native中的所有现有内核签名。
    class_method_name: str | None

    # 仅在轻量级分发中设置为true。如果启用轻量级分发，我们将操作符注册到JIT操作注册表中，因此我们需要避免生成代码以注册到分发器中。
    skip_dispatcher_op_registration: bool

    @staticmethod
    def gen_device_check(
        type: DeviceCheckType, args: list[Argument], method_name: str
    ) -> str:
        if type == DeviceCheckType.NoCheck:
            return "  // No device check\n"

        # 生成设备检查的C++代码
        device_check = "std::optional<Device> common_device = std::nullopt;\n"
        device_check += "(void)common_device; // Suppress unused variable warning\n"
        for arg in args:
            # 仅适用于类似张量的参数
            if arg.type.is_tensor_like():
                device_check += f"""
  c10::impl::check_and_update_common_device(common_device, {arg.name}, "{method_name}", "{arg.name}");"""
        return device_check

    @method_with_native_function
    def __call__(self, f: NativeFunctionsGroup | NativeFunction) -> list[str]:
        if isinstance(f, NativeFunctionsGroup):
            g: NativeFunctionsGroup = f
            # 注意：如果操作符标记为结构化，则调用gen_structured()，不管后端如何。
            # gen_structured()具有处理自动生成内核的特殊逻辑。
            if g.structured:
                return self.gen_structured(g)
            else:
                return list(
                    mapMaybe(lambda f: self.gen_unstructured(f, g), g.functions())
                )
        elif isinstance(f, NativeFunction):
            r = self.gen_unstructured(f)
            return [] if r is None else [r]
        else:
            assert_never(f)

    def wrapper_kernel_sig(
        self, f: NativeFunction
    ) -> NativeSignature | DispatcherSignature:
        # 前缀仅用于确保唯一性。调度程序API不保证唯一的内核名称。
        return DispatcherSignature.from_schema(
            f.func,
            prefix=f"wrapper_{self.backend_index.dispatch_key}_{f.func.name.overload_name}_",
            symint=self.symint,
        )

    def gen_out_inplace_wrapper(
        self, f: NativeFunction, g: NativeFunctionsGroup | None
    ) -> str | None:
        # 如果 g 为空，则返回 None
        if g is None:
            return None
        # 获取函数对象的类型
        k = f.func.kind()
        # 根据函数对象的类型选择不同的复制操作
        if k is SchemaKind.inplace:
            copy_op = "at::_copy_from"
        elif k is SchemaKind.out:
            copy_op = "at::_copy_from_and_resize"
        else:
            # 如果函数类型不是 inplace 或 out，则抛出断言错误
            raise AssertionError("gen_out_inplace_wrapper called on a functional op")

        # 获取函数的包装器签名
        sig = self.wrapper_kernel_sig(f)
        # 获取函数签名的名称
        name = sig.name()

        # 构建函数的临时结果名称
        func_res = f"{name}_tmp"
        # 获取 C++ 返回值的名称列表
        return_names = cpp.return_names(f)
        
        # 如果返回值列表的长度大于 1，则需要生成多个更新语句
        if len(return_names) > 1:
            updates = "\n  ".join(
                # 生成多个复制操作语句
                f"{copy_op}(std::get<{i}>({func_res}), {ret_name});"
                for i, ret_name in enumerate(return_names)
            )
            # 构建返回值的 C++ 表达式
            returns = f'{sig.returns_type().cpp_type()}({", ".join(return_names)})'
        # 如果返回值列表长度为 1，则只需生成单个复制操作语句和返回值
        elif len(return_names) == 1:
            ret_name = return_names[0]
            updates = f"{copy_op}({func_res}, {ret_name});"
            returns = ret_name
        else:
            # 否则，确保函数的输出参数列表中有且仅有一个参数
            assert len(f.func.arguments.out) == 1
            returns = ""
            out_arg = f.func.arguments.out[0]
            # 如果输出参数类型类似于列表，则生成循环复制操作语句
            if out_arg.type.is_list_like():
                updates = f"""\
    for (int64_t i = 0; i < {func_res}.size(); ++i) {{
        {copy_op}({func_res}[i], {out_arg.name}[i]);
    }}"""
            else:
                # 否则，生成单个复制操作语句
                updates = f"{copy_op}({func_res}, {out_arg.name});"

        # 获取函数式操作的包装器签名
        functional_sig = self.wrapper_kernel_sig(g.functional)
        # 获取函数签名的名称
        wrapper_name = sig.name()

        # 返回生成的 C++ 代码块
        return f"""\
{sig.defn(name=wrapper_name)} {{
  auto {func_res} = {functional_sig.name()}({", ".join(e.expr for e in translate(sig.arguments(), functional_sig.arguments()))});
  {updates}
  return {returns};
}}
"""

在生成代码中插入函数定义，并返回结果。


def gen_structured(self, g: NativeFunctionsGroup) -> list[str]:

定义一个方法 `gen_structured`，接受一个 `NativeFunctionsGroup` 类型的参数 `g`，返回一个 `list[str]`。


metadata = self.backend_index.get_kernel(g)

从 `backend_index` 中获取与 `g` 相关的内核元数据。


if self.backend_index.dispatch_key == DispatchKey.Meta:

如果 `backend_index` 的调度键为 `DispatchKey.Meta`：


assert not self.backend_index.has_kernel(g.out), (
    "Do not explicitly specify Meta dispatch key on structured "
    "functions, they will be automatically generated for you"
)

确保 `backend_index` 中不存在 `g.out` 的内核，并给出相应的错误消息。


elif (
    self.backend_index.dispatch_key
    == DispatchKey.CompositeExplicitAutogradNonFunctional
):

否则，如果 `backend_index` 的调度键为 `DispatchKey.CompositeExplicitAutogradNonFunctional`：


assert not self.backend_index.has_kernel(g.out), (
    "Do not explicitly specify CompositeExplicitAutograd dispatch key on structured "
    "functions, they will be automatically generated for you"
)

确保 `backend_index` 中不存在 `g.out` 的内核，并给出相应的错误消息。


elif metadata is None or not metadata.structured:

否则，如果 `metadata` 是 `None` 或者不是结构化的：


return list(mapMaybe(lambda f: self.gen_unstructured(f, g), g.functions()))

返回通过 `mapMaybe` 处理 `g.functions()` 中每个函数 `f` 调用 `gen_unstructured(f, g)` 后的结果列表。


structured_gen = StructuredRegisterDispatchKey(
    self.backend_index,
    self.target,
    self.selector,
    self.rocm,
    self.symint,
    self.class_method_name,
    self.skip_dispatcher_op_registration,
    g,
)

创建一个 `StructuredRegisterDispatchKey` 对象 `structured_gen`，并传入多个参数，包括 `self.backend_index`、`self.target` 等。


return list(mapMaybe(structured_gen.gen_one, g.functions()))

返回通过 `mapMaybe` 处理 `g.functions()` 中每个函数调用 `structured_gen.gen_one` 后的结果列表。


def gen_unstructured(
    self, f: NativeFunction, g: NativeFunctionsGroup | None = None

定义一个方法 `gen_unstructured`，接受一个 `NativeFunction` 类型的参数 `f`，以及一个可选的 `NativeFunctionsGroup` 类型参数 `g`。
    ) -> str | None:
        # 定义方法签名，并指定返回类型为字符串或空值
        with native_function_manager(f):
            # 进入本地函数管理器的上下文
            inplace_meta = False
            gets_out_inplace_wrapper = False
            # 如果函数 f 没有对应的后端内核
            if not self.backend_index.has_kernel(f):
                # 如果当前后端调度键为 DispatchKey.Meta
                if (
                    self.backend_index.dispatch_key == DispatchKey.Meta
                    and f.func.kind() is SchemaKind.inplace
                    and
                    # 延迟到组合体用于元实现
                    not f.has_composite_kernel
                    and
                    # 不支持原地列表操作
                    len(f.func.returns) == 1
                ):
                    inplace_meta = True
                # 否则，如果不使用 out 参数作为主要参数，并且 g 不为 None，并且获取生成的原地包装器
                elif (
                    not self.backend_index.use_out_as_primary
                    and g is not None
                    and gets_generated_out_inplace_wrapper(f, g, self.backend_index)
                ):
                    # 我们希望生成原地/out包装器，后端没有内核
                    gets_out_inplace_wrapper = True
                else:
                    # 否则返回空
                    return None
            # 如果需要手动注册内核，则返回空
            if f.manual_kernel_registration:
                return None

            # 如果目标为注册，并且没有选择本地函数，则返回空
            if (
                self.target is Target.REGISTRATION
                and not self.selector.is_native_function_selected(f)
            ):
                return None

            # 生成包装器内核的签名
            sig = self.wrapper_kernel_sig(f)

            # 获取签名的名称、返回类型和参数
            name = sig.name()
            returns_type = sig.returns_type().cpp_type()
            args = sig.arguments()
            args_str = ", ".join(a.defn() for a in args)

            # 查看[直接调度绑定]注释
            cpp_sig_group = CppSignatureGroup.from_native_function(
                f, method=False, fallback_binding=False
            )

            # 如果目标是命名空间声明
            if self.target is Target.NAMESPACED_DECLARATION:
                result = ""
                # 对于每个 CPP 签名组中的签名
                for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                    # 生成 TORCH_API 声明
                    result += f"TORCH_API {cpp_sig.decl()};\n"
                return result
            # 否则，如果目标是命名空间定义
            elif self.target is Target.NAMESPACED_DEFINITION:

                def generate_defn(cpp_sig: CppSignature) -> str:
                    return f"""
# 定义一个字符串模板，包含一个C++函数的签名和调用
defn = f"""
return {sig.name()}({', '.join(e.expr for e in translate(cpp_sig.arguments(), sig.arguments()))});
"""

# 生成所有C++签名的定义并拼接成一个字符串
result = ""
for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
    result += generate_defn(cpp_sig)
return result

elif self.target is Target.ANONYMOUS_DEFINITION:
    # 对于 inplace_meta 的快捷通道
    if inplace_meta:
        assert f.func.arguments.self_arg is not None
        self_arg_name = f.func.arguments.self_arg.argument.name
        # TODO: 处理对张量列表的原地操作
        return f"""
{returns_type} {name}({args_str}) {{
  TORCH_CHECK_NOT_IMPLEMENTED({self_arg_name}.is_meta(),
    "Cannot inplace into non-meta tensor with meta tensor argument");
  return {self_arg_name};
}}
namespace {{

# 定义匿名命名空间内的函数
{returns_type} {name}({args_str}) {{
  # 检查设备兼容性
  {device_check}

  # 设备保护
  {device_guard}
  return {impl_name}({args_exprs_str});
}}

}} // 匿名命名空间结束
"""

elif self.target is Target.REGISTRATION:
    # 如果是注册目标且不是手动注册内核或跳过调度操作注册
    if f.manual_kernel_registration or self.skip_dispatcher_op_registration:
        return None
    else:
        payload = f"TORCH_FN({name})"
        return f'm.impl("{f.func.name}",\n{payload});\n'

else:
    assert_never(self.target)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           STRUCTURED
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@dataclass(frozen=True)
class StructuredRegisterDispatchKey(RegisterDispatchKey):
    g: NativeFunctionsGroup

    # 生成类的设置输出函数
    def gen_class_set_output_functions(
        self, k: SchemaKind, parent_class: str, generate_super: bool
    ) -> str:
        if generate_super:
            set_output_super = f"{parent_class}::set_output_raw_strided(output_idx, sizes, strides, options, names);"
        else:
            set_output_super = ""

        # 生成设置输出函数的具体实现
        def gen_set_output_function(name: str, maybe_create_proxy: bool) -> str:
            return f"""
void set_output_{name}(
    int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
    TensorOptions options, DimnameList names
) override {{
{textwrap.indent(self.gen_class_set_output_body(k, maybe_create_proxy), "    ")}
    if (!names.empty()) {{
      namedinference::propagate_names(outputs_[output_idx], names);
    }}
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
{textwrap.indent(set_output_super, "    ")}
}}
"""

        # 返回生成的两个设置输出函数的字符串
        return f"""
{gen_set_output_function("strided", maybe_create_proxy=True)}
{gen_set_output_function("raw_strided", maybe_create_proxy=False)}
"""
    # 定义一个方法，生成类设置输出的主体部分，返回一个字符串
    def gen_class_set_output_body(self, k: SchemaKind, maybe_create_proxy: bool) -> str:
        # 如果当前对象的后端索引调度键在以下列表中
        if self.backend_index.dispatch_key in [
            DispatchKey.CUDA,
            DispatchKey.MPS,
            DispatchKey.CompositeExplicitAutogradNonFunctional,
        ]:
            # 可能设置守卫条件的字符串
            maybe_set_guard = """
# 获取当前设备
auto current_device = guard_.current_device();
# 如果当前设备存在
if (C10_UNLIKELY(current_device.has_value())) {
    # 断言当前设备与选项中的设备相同，否则抛出错误
    TORCH_INTERNAL_ASSERT(*current_device == options.device(),
        "structured kernels don't support multi-device outputs");
} else {
    # 重置当前设备为选项中的设备
    guard_.reset_device(options.device());
}
"""
maybe_set_guard_line = maybe_set_guard + "\n"
else:
    maybe_set_guard_line = maybe_set_guard = ""

if maybe_create_proxy:
    # 如果可能创建代理
    create_proxy = """
auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
# 如果代理存在
if (C10_UNLIKELY(maybe_proxy.has_value())) {
    # 将代理移动到相应的位置
    proxy_outputs_[output_idx] = std::move(maybe_proxy).value();
}
"""
else:
    create_proxy = ""

if k is SchemaKind.functional:
    # 如果是函数式模式
    assert self.backend_index.dispatch_key in (
        DispatchKey.Meta,
        DispatchKey.CPU,
        DispatchKey.CUDA,
        DispatchKey.MPS,
        DispatchKey.CompositeExplicitAutogradNonFunctional,
    )
    # 返回创建输出的代码块
    return f"""{maybe_set_guard_line}
outputs_[output_idx] = create_out(sizes, strides, options);"""
elif k is SchemaKind.inplace:
    # 如果是原地操作模式
    return f"""{maybe_set_guard_line}
const auto& out = outputs_[output_idx].get();
# 检查是否可以原地操作
check_inplace(out, sizes, options);
{create_proxy}"""
elif k is SchemaKind.out:
    # 如果是输出模式
    return f"""{maybe_set_guard_line}
const auto& out = outputs_[output_idx].get();
# 调整输出的大小
resize_out(out, sizes, strides, options);
{create_proxy}"""
elif k is SchemaKind.mutable or k is SchemaKind.scratch:
    # 如果是可变模式或者临时模式，抛出断言错误
    raise AssertionError(
        f"{k} structured operators are currently not supported"
    )
else:
    # 如果类型未知，触发断言错误
    assert_never(k)

# 返回一个构造函数的定义，以及如何构造这个类到一个名为op的变量中
def gen_class_ctor(self, k: SchemaKind, class_name: str, returns: int) -> str:
    if k is SchemaKind.functional:
        # 如果是函数式模式，返回空字符串
        return ""
    elif k is SchemaKind.inplace:
        # 如果是原地操作模式
        # TODO: 确保out参数保证是self
        return f"{class_name}(Tensor& self) : outputs_{{std::ref(self)}} {{}}"
    elif k is SchemaKind.out:
        # 如果是输出模式
        out_args = ", ".join(f"Tensor& out{i}" for i in range(returns))
        out_refs = ", ".join(f"std::ref(out{i})" for i in range(returns))
        return f"{class_name}({out_args}) : outputs_{{ {out_refs} }} {{}}"
    elif k is SchemaKind.mutable or k is SchemaKind.scratch:
        # 如果是可变模式或者临时模式，抛出断言错误
        raise AssertionError(
            f"{k} structured operators are currently not supported"
        )
    else:
        # 如果类型未知，触发断言错误
        assert_never(k)

# 定义一个生成类的方法
def gen_class(
    self,
    f: NativeFunction,
    k: SchemaKind,
    *,
    class_name: str,
    parent_class: str,
    generate_super: bool,
    ) -> str:
        if k is SchemaKind.functional:
            output_type = "Tensor"
            output_value = "outputs_[output_idx]"
            proxy_field = ""
        elif k is SchemaKind.inplace:
            output_type = "std::reference_wrapper<Tensor>"
            output_value = "proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get()"
            proxy_field = f"std::array<::std::optional<Tensor>, {len(f.func.returns)}> proxy_outputs_;"
        elif k is SchemaKind.out:
            output_type = "std::reference_wrapper<Tensor>"
            output_value = "proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get()"
            proxy_field = f"std::array<::std::optional<Tensor>, {len(f.func.returns)}> proxy_outputs_;"
        else:
            raise RuntimeError(f"Unsupported SchemaKind {k}")

        if self.backend_index.dispatch_key == DispatchKey.CUDA:
            if self.rocm:
                guard_field = "c10::hip::OptionalHIPGuardMasqueradingAsCUDA guard_;"
            else:
                guard_field = "c10::cuda::OptionalCUDAGuard guard_;"
        elif (
            self.backend_index.dispatch_key
            == DispatchKey.CompositeExplicitAutogradNonFunctional
        ):
            guard_field = "c10::OptionalDeviceGuard guard_;"
        elif self.backend_index.dispatch_key == DispatchKey.MPS:
            # TODO: Move to OptionalMPSGuard.
            guard_field = "c10::OptionalDeviceGuard guard_;"
        else:
            guard_field = ""

        indent = " " * 4
        # 生成类的构造函数字符串
        class_ctor_str = self.gen_class_ctor(k, class_name, len(f.func.returns))
        # 组装类定义的各行代码
        lines = (
            f"struct {class_name} final : public {parent_class} {{",
            f"{textwrap.indent(class_ctor_str, indent)}",
            f"{textwrap.indent(self.gen_class_set_output_functions(k, parent_class, generate_super), indent)}",
            "    const Tensor& maybe_get_output(int64_t output_idx) override {",
            f"      return {output_value};\n",  # 返回给定输出索引的输出值  # TODO: audit
            "    }",
            f"    std::array<{output_type}, {len(f.func.returns)}> outputs_;",  # 输出值的数组类型定义  # TODO: audit
            f"{textwrap.indent(proxy_field, indent)}",  # 代理输出的字段定义  # TODO: audit
            f"{textwrap.indent(guard_field, indent)}",  # 设备保护字段定义
            "};",
        )
        return "\n".join(line for line in lines if line)

    @method_with_native_function
        # 确保函数不是手动注册的内核
        assert not f.manual_kernel_registration

        # 如果目标是注册并且选择器未选择本地函数，则返回 None
        if (
            self.target is Target.REGISTRATION
            and not self.selector.is_native_function_selected(f)
        ):
            return None

        # TODO: 现在这里有一些有趣的事情。在下面的代码中，
        # 我们基于 out 实现生成 CompositeExplicitAutogradNonFunctional 的功能和原地功能实现。
        # 但事实上，out 也可以由功能定义（只是效率不高），
        # 这实际上更可能是后端实现者的情况。
        # 我们如何选择？借鉴 Haskell 类型类和默认方法的思想，
        # 我们可以注册一个循环定义（out 以功能方式定义，功能以 out 方式定义），
        # 并且只要求某人实现其中一个。我们需要做一些工作，
        # 以便在 DAG 中没有强定义时不注册这些“弱”定义！因此目前尚未实现。

        # 如果后端索引的分发键是 CompositeExplicitAutogradNonFunctional，
        # 并且函数的类型是 SchemaKind.out，则返回 None，不生成 out 的默认实现
        if (
            self.backend_index.dispatch_key
            == DispatchKey.CompositeExplicitAutogradNonFunctional
            and f.func.kind() is SchemaKind.out
        ):
            return None

        # 注释 [Direct dispatch bindings]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 生成我们将在头文件中公开的非分发函数的签名（例如 at::cpu::add）。
        # 我们不生成方法（TODO: 当 CPUTensor 类存在时可以生成方法）；
        # 我们也不为 manual_cpp_binding 函数生成回退绑定。
        cpp_sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=False
        )

        # 生成将注册到分发器的包装函数的签名
        kern = self.backend_index.get_kernel(f)
        sig = NativeSignature(
            f.func,
            prefix=f"wrapper_{self.backend_index.dispatch_key}_",
            symint=kern is not None and kern.supports_symint(),
        )

        # 如果目标是命名空间声明
        if self.target is Target.NAMESPACED_DECLARATION:
            result = ""
            # 对于每个 cpp_sig，在支持 symint 的情况下生成 TORCH_API 声明
            for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                result += f"TORCH_API {cpp_sig.decl()};\n"
            return result

        # 如果目标是命名空间定义
        elif self.target is Target.NAMESPACED_DEFINITION:

            def generate_defn(cpp_sig: CppSignature) -> str:
                return f"""
{
cpp_sig.defn()} {{
return {sig.name()}({', '.join(e.expr for e in translate(cpp_sig.arguments(), sig.arguments()))});
}}
{self.gen_class(
f, k,
class_name=class_name,
parent_class=parent_class,
generate_super=self.g.out.structured_inherits is not None
)}



# 返回 C++ 函数定义字符串，调用目标函数签名生成的函数，并传递翻译后的参数列表
{
cpp_sig.defn()} {{
return {sig.name()}({', '.join(e.expr for e in translate(cpp_sig.arguments(), sig.arguments()))});
}}
{self.gen_class(
f, k,
class_name=class_name,
parent_class=parent_class,
generate_super=self.g.out.structured_inherits is not None
)}



{sig.defn()} {{
{sig_body_str}
}}



# 返回当前函数签名的定义字符串和函数体内容
{sig.defn()} {{
{sig_body_str}
}}



"""

elif self.target is Target.REGISTRATION:
return f'm.impl("{f.func.name}", TORCH_FN({sig.name()}));'
else:
assert_never(self.target)
# Silence mypy's "Missing return statement" error
return None



# 如果 self.target 是 REGISTRATION，返回注册模块的字符串表示，包括函数名和对应的 Torch 函数名
elif self.target is Target.REGISTRATION:
return f'm.impl("{f.func.name}", TORCH_FN({sig.name()}));'
# 否则，如果 self.target 无效，触发断言以显示错误信息，并在静态类型检查中防止 "Missing return statement" 错误
else:
assert_never(self.target)
# Silence mypy's "Missing return statement" error
return None
```