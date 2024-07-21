# `.\pytorch\torchgen\gen_vmap_plumbing.py`

```
# 导入所需模块和类
from __future__ import annotations
import textwrap
from dataclasses import dataclass
from typing import Sequence

# 从torchgen.api.translate模块导入translate函数
from torchgen.api.translate import translate
# 从torchgen.api.types模块导入DispatcherSignature类
from torchgen.api.types import DispatcherSignature
# 从torchgen.context模块导入method_with_native_function函数
from torchgen.context import method_with_native_function
# 从torchgen.model模块导入各种类和类型
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    OptionalType,
    Return,
    SchemaKind,
    Type,
)
# 从torchgen.utils模块导入mapMaybe函数
from torchgen.utils import mapMaybe


# 定义函数is_tensor，判断给定的Type对象是否为BaseType类型且名称为BaseTy.Tensor
def is_tensor(typ: Type) -> bool:
    return isinstance(typ, BaseType) and typ.name == BaseTy.Tensor


# 定义函数is_optional_tensor，判断给定的Type对象是否为OptionalType类型且其元素是否为Tensor类型
def is_optional_tensor(typ: Type) -> bool:
    return isinstance(typ, OptionalType) and is_tensor(typ.elem)


# 定义函数is_tensor_list，判断给定的Type对象是否为ListType类型且其元素是否为Tensor类型
def is_tensor_list(typ: Type) -> bool:
    return isinstance(typ, ListType) and is_tensor(typ.elem)


# 定义函数unwrap_tensor，生成解包Tensor的代码片段
def unwrap_tensor(name: str, cur_level_var: str) -> list[str]:
    result = f"""\
    Tensor {name}_value;
    optional<int64_t> {name}_bdim;
    std::tie({name}_value, {name}_bdim) = unwrapTensorAtLevel({name}, {cur_level_var});"""
    return textwrap.dedent(result).split("\n")


# 定义函数unwrap_optional_tensor，生成解包Optional[Tensor]的代码片段
def unwrap_optional_tensor(name: str, cur_level_var: str) -> list[str]:
    result = f"""\
    optional<Tensor> {name}_value;
    optional<int64_t> {name}_bdim;
    if ({name}) {{
        std::tie({name}_value, {name}_bdim) = unwrapTensorAtLevel({name}.value(), {cur_level_var});
    }}"""
    return textwrap.dedent(result).split("\n")


# 定义函数gen_unwraps，生成解包所有Tensor和Optional[Tensor]的代码片段和解包后的参数列表
def gen_unwraps(
    flat_arguments: Sequence[Argument], cur_level_var: str
) -> tuple[str, list[str]]:
    arg_names = [a.name for a in flat_arguments]
    arg_types = [a.type for a in flat_arguments]

    tensors = [name for typ, name in zip(arg_types, arg_names) if is_tensor(typ)]
    optional_tensors = [
        name for typ, name in zip(arg_types, arg_names) if is_optional_tensor(typ)
    ]

    unwraps = []
    for tensor in tensors:
        unwraps += unwrap_tensor(tensor, cur_level_var)

    for opt_tensor in optional_tensors:
        unwraps += unwrap_optional_tensor(opt_tensor, cur_level_var)
    unwrap_code = "\n".join(unwraps)

    unwrapped_arg_list = []
    for arg in arg_names:
        if arg in tensors or arg in optional_tensors:
            unwrapped_arg_list += [f"{arg}_value", f"{arg}_bdim"]
        else:
            unwrapped_arg_list.append(arg)
    return unwrap_code, unwrapped_arg_list


# 定义函数gen_case_where_all_bdims_are_none，生成检查所有Tensor的批处理维度是否为None的代码片段
def gen_case_where_all_bdims_are_none(
    outer_sig: DispatcherSignature, schema: FunctionSchema, cur_level_var: str
) -> str:
    conditions = []
    flat_args = schema.arguments.flat_all
    for arg in flat_args:
        if not arg.type.is_tensor_like():
            continue
        conditions.append(f"!isBatchedAtLevel({arg.name}, {cur_level_var})")

    sig = DispatcherSignature.from_schema(schema)
    translated_args = ", ".join(
        e.expr for e in translate(outer_sig.arguments(), sig.arguments())
    )
    return f"""\
if ({' && '.join(conditions)}) {{
  return at::_ops::{sig.func.name.unambiguous_name()}::call({translated_args});
# 定义一个生成返回语句的函数，用于生成批处理规则下的返回语句
def gen_returns(
    returns: tuple[Return, ...], cur_level_var: str, results_var: str
) -> str:
    # 初始化索引为0
    idx = 0
    # 存储包装后的返回语句列表
    wrapped_returns = []
    # 遍历所有返回值
    for ret in returns:
        # 如果返回值是张量类型
        if is_tensor(ret.type):
            # 构建返回语句，调用makeBatched函数，将结果包装进批处理中
            wrapped_returns.append(
                f"makeBatched(std::get<{idx}>({results_var}), std::get<{idx + 1}>({results_var}), {cur_level_var})"
            )
            idx += 2
        # 如果返回值是张量列表类型
        elif is_tensor_list(ret.type):
            # 构建返回语句，调用makeBatchedVector函数，将结果包装进批处理中
            wrapped_returns.append(
                f"makeBatchedVector(std::get<{idx}>({results_var}), std::get<{idx+1}>({results_var}), {cur_level_var})"
            )
            idx += 2
        else:
            # 其他类型的返回值，直接获取
            wrapped_returns.append(f"std::get<{idx}>({results_var})")
            idx += 1
    # 如果只有一个返回值，返回语句格式为return <wrapped_return>;
    if len(wrapped_returns) == 1:
        result = f"return {wrapped_returns[0]};"
    else:
        # 如果有多个返回值，返回语句格式为return std::make_tuple(<wrapped_returns>);
        result = f'return std::make_tuple({", ".join(wrapped_returns)});'
    return result


# 判断给定函数的参数中是否至少包含一个张量类型的输入
def accepts_at_least_one_tensor_input(schema: FunctionSchema) -> bool:
    return any(a.type.is_tensor_like() for a in schema.arguments.flat_all)


# 判断给定参数是否被修改
def is_mutated_arg(argument: Argument) -> bool:
    return argument.annotation is not None and argument.annotation.is_write


# 生成处理inplace操作的辅助代码
def gen_vmap_inplace_plumbing(native_function: NativeFunction) -> str | None:
    # 假设：
    # - 只有一个参数被原地修改
    # - 被原地修改的参数是第一个参数
    # - 所有返回值要么是张量，要么是张量元组，要么是张量列表
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    returns = schema.returns

    # 检查假设条件，如果条件不符合则返回None
    assert schema.kind() == SchemaKind.inplace
    if not is_mutated_arg(schema.arguments.flat_all[0]):
        return None
    if not len([arg for arg in schema.arguments.flat_all if is_mutated_arg(arg)]) == 1:
        return None

    # 只支持所有返回值都是张量或者张量列表的情况
    if len(returns) == 0:
        return None
    if not all(is_tensor(ret.type) or is_tensor_list(ret.type) for ret in returns):
        return None
    if not accepts_at_least_one_tensor_input(schema):
        return None

    # 当前层级的变量名
    cur_level_var = "cur_level"

    # 生成解包和解包参数列表的代码
    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    # 生成所有批次维度都为None的情况的代码
    bdims_all_none_case = gen_case_where_all_bdims_are_none(sig, schema, cur_level_var)

    return f"""\
# 定义生成的inplace操作的模板函数
template <typename batch_rule_t, batch_rule_t batch_rule>
{sig.decl(name=schema.name.unambiguous_name() + '_generated_plumbing')} {{
  # 排除FuncTorchBatched分发键的守卫
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  # 获取当前可能的动态层
  auto maybe_layer = maybeCurrentDynamicLayer();
  # 检查vmap是否逃逸，用于调试和错误处理
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  # 初始化当前层级的变量
  int64_t {cur_level_var} = maybe_layer->layerId();
  {textwrap.indent(bdims_all_none_case, "  ")}
# 生成 VMap 相关的 C++ 代码，不返回结果的情况下的模板函数
def gen_vmap_plumbing_no_returns(native_function: NativeFunction) -> str:
    # 获取函数的模式定义
    schema = native_function.func
    # 根据模式生成调度签名
    sig = DispatcherSignature.from_schema(schema)
    # 当前层级变量
    cur_level_var = "cur_level"

    # 生成解开参数和解包参数列表
    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    # 生成所有批次维度都为 None 的情况
    bdims_all_none_case = gen_case_where_all_bdims_are_none(sig, schema, cur_level_var)

    # 构建函数返回值
    return f"""\
template <typename batch_rule_t, batch_rule_t batch_rule>
{sig.decl(name=schema.name.unambiguous_name() + '_generated_plumbing')} {{
  // 排除 FuncTorchBatched 调度键
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  // 获取当前动态层
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 检查 VMap 是否逃逸
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  // 设置当前层级变量
  int64_t {cur_level_var} = maybe_layer->layerId();
  // 插入所有批次维度都为 None 的情况代码块
  {textwrap.indent(bdims_all_none_case, "  ")}
  // 插入解开参数代码块
  {textwrap.indent(unwraps, "  ")}
  // 执行批次规则
  batch_rule({', '.join(unwrapped_arg_list)});
}}"""


# 生成 VMap 相关的 C++ 代码，包含返回结果的模板函数
def gen_vmap_plumbing(native_function: NativeFunction) -> str | None:
    # 获取函数的模式定义
    schema = native_function.func
    # 根据模式生成调度签名
    sig = DispatcherSignature.from_schema(schema)
    # 获取函数的返回值
    returns = schema.returns

    # 只支持所有返回值为 Tensor 或 vector<Tensor> 的情况
    if not accepts_at_least_one_tensor_input(schema):
        return None
    if len(returns) == 0:
        return gen_vmap_plumbing_no_returns(native_function)
    if not all(ret.type.is_tensor_like() for ret in returns):
        return None
    # 特殊处理原地视图
    if "inplace_view" in native_function.tags:
        return None

    # 如果模式是 inplace 类型，则生成相应的处理代码
    if schema.kind() == SchemaKind.inplace:
        return gen_vmap_inplace_plumbing(native_function)

    # 不支持 mutable、out、scratch 这些类型
    if schema.kind() != SchemaKind.functional:
        return None

    # 结果变量名
    results_var = "results"
    # 当前层级变量
    cur_level_var = "cur_level"

    # 生成解开参数和解包参数列表
    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    # 生成所有批次维度都为 None 的情况
    bdims_all_none_case = gen_case_where_all_bdims_are_none(sig, schema, cur_level_var)
    # 生成包装返回值的代码
    wrapped_returns = gen_returns(returns, cur_level_var, results_var)

    # 构建函数返回值
    return f"""\
template <typename batch_rule_t, batch_rule_t batch_rule>
{sig.decl(name=schema.name.unambiguous_name() + '_generated_plumbing')} {{
  // 排除 FuncTorchBatched 调度键
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  // 获取当前动态层
  auto maybe_layer = maybeCurrentDynamicLayer();
  // 检查 VMap 是否逃逸
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  // 设置当前层级变量
  int64_t {cur_level_var} = maybe_layer->layerId();
  // 插入所有批次维度都为 None 的情况代码块
  {textwrap.indent(bdims_all_none_case, "  ")}
  // 插入解开参数代码块
  {textwrap.indent(unwraps, "  ")}
  // 执行批次规则，并获取结果
  auto {results_var} = batch_rule({', '.join(unwrapped_arg_list)});
  // 插入包装返回值的代码
  {wrapped_returns}
}}"""
# 生成所有原生函数的映射相关代码，返回一个字符串
def gen_all_vmap_plumbing(native_functions: Sequence[NativeFunction]) -> str:
    # 将原生函数序列映射为可能有值的列表，并连接成字符串
    body = "\n".join(list(mapMaybe(ComputeBatchRulePlumbing(), native_functions)))
    # 构建包含特定头文件的C++头文件声明
    return f"""
#pragma once
#include <ATen/Operators.h>
#include <ATen/functorch/PlumbingHelper.h>

namespace at {{ namespace functorch {{

{body}  # 插入映射生成的代码段

}}}} // namespace at::functorch
"""
```