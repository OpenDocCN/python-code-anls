# `.\pytorch\tools\autograd\gen_variable_factories.py`

```py
# 生成将 ATen 张量工厂方法包装为变量的 C++ 函数。
# 
# 这段代码用于生成一个文件：variable_factories.h

from __future__ import annotations  # 允许在类型注解中使用类名

import re  # 导入正则表达式模块

import torchgen.api.python as python  # 导入 TorchGen 中的 Python API
from torchgen.api import cpp  # 导入 TorchGen 中的 C++ API
from torchgen.api.types import CppSignatureGroup  # 导入 TorchGen 中的 C++ 签名组
from torchgen.context import with_native_function  # 导入 TorchGen 中的本地函数上下文管理器
from torchgen.gen import parse_native_yaml  # 导入 TorchGen 中的解析本地 YAML 的功能
from torchgen.model import NativeFunction, TensorOptionsArguments, Variant  # 导入 TorchGen 中的数据模型
from torchgen.utils import FileManager, mapMaybe  # 导入 TorchGen 中的文件管理器和 mapMaybe 函数

OPTIONAL_TYPE_PATTERN = re.compile(r"c10::optional<(.+)>" )  # 匹配可选类型模式的正则表达式
TYPE_PATTERN = re.compile(r"(?:const\s+)?([A-Z]\w+)" )  # 匹配类型模式的正则表达式


# 将 ATen 命名空间中定义的类型（例如 Tensor、TensorList、IntArrayRef 等）添加 'at::' 前缀。
# TODO: 可能更新 cpp 参数 API 来接受可选的命名空间参数？
def fully_qualified_type(argument_type: str) -> str:
    # 内部函数：如果是可选类型，则添加 c10::optional<> 封装
    def maybe_optional_type(type: str, is_opt: bool) -> str:
        return f"c10::optional<{type}>" if is_opt else type

    opt_match = OPTIONAL_TYPE_PATTERN.match(argument_type)  # 尝试匹配可选类型模式
    is_opt = opt_match is not None  # 判断是否是可选类型
    if opt_match:
        argument_type = argument_type[opt_match.start(1) : opt_match.end(1)]  # 获取可选类型内部的具体类型
    match = TYPE_PATTERN.match(argument_type)  # 尝试匹配一般类型模式
    if match is None:
        return maybe_optional_type(argument_type, is_opt)  # 如果不匹配，直接返回可能的可选类型
    index = match.start(1)  # 获取匹配到的类型的起始索引
    qualified_type = f"{argument_type[:index]}at::{argument_type[index:]}"  # 添加 'at::' 前缀
    return maybe_optional_type(qualified_type, is_opt)  # 返回带有可选类型封装的完整类型名


# 生成变量工厂的函数，输出到指定路径的文件中
def gen_variable_factories(
    out: str, native_yaml_path: str, tags_yaml_path: str, template_path: str
) -> None:
    # 解析本地 YAML 文件以获取本地函数列表
    native_functions = parse_native_yaml(
        native_yaml_path, tags_yaml_path
    ).native_functions
    # 过滤出所有的工厂函数
    factory_functions = [fn for fn in native_functions if is_factory_function(fn)]
    # 创建文件管理器
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    # 使用模板写入文件 variable_factories.h
    fm.write_with_template(
        "variable_factories.h",
        "variable_factories.h",
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/variable_factories.h",
            # 生成所有工厂函数对应的头文件包含语句
            "ops_headers": [
                f"#include <ATen/ops/{fn.root_name}.h>" for fn in factory_functions
            ],
            # 处理每个工厂函数，生成函数定义
            "function_definitions": list(mapMaybe(process_function, factory_functions)),
        },
    )


# 本地函数装饰器：判断给定的本地函数是否是工厂函数
@with_native_function
def is_factory_function(f: NativeFunction) -> bool:
    # 如果本地函数不是函数变体，则返回 False
    if Variant.function not in f.variants:
        return False

    name = cpp.name(f.func)  # 获取本地函数的名称
    has_tensor_options = python.has_tensor_options(f)  # 判断本地函数是否具有张量选项
    # 如果具有张量选项或者名称以 '_like' 结尾，则认为是工厂函数
    return has_tensor_options or name.endswith("_like")


# 本地函数装饰器：处理给定的本地函数，返回其 C++ 函数定义字符串或者 None
@with_native_function
def process_function(f: NativeFunction) -> str | None:
    name = cpp.name(f.func)  # 获取本地函数的名称
    has_tensor_options = python.has_tensor_options(f)  # 判断本地函数是否具有张量选项
    is_factory = has_tensor_options or name.endswith("_like")  # 判断是否是工厂函数

    # 如果不是函数变体或者不是工厂函数，则返回 None
    if Variant.function not in f.variants or not is_factory:
        return None

    # 从本地函数生成 C++ 签名组
    cpp_sigs = CppSignatureGroup.from_native_function(f, method=False)
    # 返回 C++ 函数定义字符串
    return ""
    sigs = [cpp_sigs.signature]
    # 初始化一个包含主签名的签名列表
    if cpp_sigs.symint_signature is not None:
        # 如果存在符号整数签名，则将其添加到签名列表中
        sigs.append(cpp_sigs.symint_signature)
    # 初始化结果字符串
    r = ""
    # 遍历每个签名
    for sig in sigs:
        # 初始化形参和表达式列表
        formals: list[str] = []
        exprs: list[str] = []
        # 默认设置梯度不需要
        requires_grad = "false"
        # 遍历签名的参数
        for arg in sig.arguments():
            # 获取参数的完全限定类型
            qualified_type = fully_qualified_type(arg.type)
            # 如果参数有默认值，则将其添加到形参列表中
            if arg.default:
                formals.append(f"{qualified_type} {arg.name} = {arg.default}")
            else:
                formals.append(f"{qualified_type} {arg.name}")

            # 如果参数类型是TensorOptionsArguments
            if isinstance(arg.argument, TensorOptionsArguments):
                # 注意：我们从TensorOptions中移除requires_grad设置，
                # 因为它会被忽略（我们有一个断言，如果设置了这个值会失败）。
                # 我们在这里明确地处理requires_grad，而不是将其传递给内核。
                exprs.append(
                    f"at::TensorOptions({arg.name}).requires_grad(c10::nullopt)"
                )
                # 手动设置结果张量的requires_grad位
                requires_grad = f"{arg.name}.requires_grad()"
            else:
                # 否则，将参数名添加到表达式列表中
                exprs.append(arg.name)

        # 将当前签名的结果格式化添加到结果字符串r中
        r += f"""\
# 定义一个名为 `{sig.name()}` 的内联函数，接受参数 `{', '.join(formals)}`
inline at::Tensor {sig.name()}({', '.join(formals)}) {{
  # 在当前作用域下禁用自动分发以支持自动微分中的就地操作或视图
  at::AutoDispatchBelowADInplaceOrView guard;
  # 调用 `at::{sig.name()}` 函数，并使用 `autograd::make_variable` 包装其结果作为变量返回
  return autograd::make_variable(at::{sig.name()}({', '.join(exprs)}), /*requires_grad=*/{requires_grad});
}}
"""
# 返回格式化后的字符串作为结果
return r
```