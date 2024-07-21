# `.\pytorch\tools\autograd\gen_view_funcs.py`

```py
# 生成 ViewFuncs.h/cpp 文件

# 注意事项：
# 如果对 ViewFunc 代码生成进行任何更改，请同时检查 torch/csrc/autograd/autograd_not_implemented_fallback.cpp 是否需要更新。
# 预期回退操作应模仿此代码生成过程，因此两者应保持同步。

from __future__ import annotations

from typing import TYPE_CHECKING

import torchgen.api.dispatcher as dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
    BaseCType,
    Binding,
    NamedCType,
    SymIntT,
    tensorT,
    VectorCType,
)
from torchgen.code_template import CodeTemplate
from torchgen.model import Argument, NativeFunction, OptionalType
from torchgen.utils import FileManager

from .gen_inplace_or_view_type import (
    CALL_DISPATCH,
    extract_bindings,
    get_view_info,
    modifies_arguments,
    use_derived,
)

if TYPE_CHECKING:
    from torchgen.api.autograd import NativeFunctionWithDifferentiabilityInfo

# 定义函数声明模板
FUNCTION_DECLARATION = CodeTemplate(
    """\
#define ${uppercase_op}_AVAILABLE
struct ${op} : public ${superclass} {
  ${op}(${constructor_args}) ${initializer_list}
  {};
  virtual ~${op}() override {};
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = c10::nullopt,
      std::optional<std::vector<at::Tensor>> = c10::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  ${state}
};

"""
)

# 定义函数定义模板
FUNCTION_DEFINITION = CodeTemplate(
    """\
std::vector<c10::SymInt> ${op}::get_symints() const {
  ${get_symints}
}

size_t ${op}::num_symints() const {
  return static_cast<size_t>(${num_symints});
}

void ${op}::set_symints(std::vector<c10::SymInt> ${symints_vec}) {
  TORCH_INTERNAL_ASSERT(${symints_vec}.size() == num_symints());
  ${set_symints}
}

std::vector<at::Tensor> ${op}::get_tensors() const {
  ${get_tensors}
}

size_t ${op}::num_tensors() const {
  return static_cast<size_t>(${num_tensors});
}

void ${op}::set_tensors(std::vector<at::Tensor> ${tensors_vec}) {
  TORCH_INTERNAL_ASSERT(${tensors_vec}.size() == num_tensors());
  ${set_tensors}
}

at::Tensor ${op}::operator()(const at::Tensor& ${call_input_name}) const {
  return ${op_call};
}

std::unique_ptr<ViewFunc> ${op}::clone_and_set(
    std::optional<std::vector<c10::SymInt>> ${symints_vec},
    std::optional<std::vector<at::Tensor>> ${tensors_vec}) const {
  auto output = std::make_unique<${op}>(${clone_args});
  if (${symints_vec}.has_value()) {
    output->set_symints(std::move(*(${symints_vec})));
  }
  if (${tensors_vec}.has_value()) {
    // 将指针所指的张量向量移动到output对象中
    output->set_tensors(std::move(*(${tensors_vec})));
  }
  // 返回output对象作为函数的结果
  return output;
}

"""
)


# e.g. as_strided -> AsStridedViewFunc for camel case or
# as_strided_view_func otherwise
# 定义一个函数，根据给定的函数对象和参数生成视图函数的名称字符串
def view_func_name(
    f: NativeFunction, include_namespace: bool = False, camel_case: bool = True
) -> str:
    # 获取函数的唯一非模糊名称
    name = f.func.name.unambiguous_name()
    # 根据是否包含命名空间选择命名空间前缀
    namespace = "torch::autograd::generated::" if include_namespace else ""
    # 将函数名中的点号替换为下划线，生成视图函数的基本名称
    view_func_name = f"{name.replace('.', '_')}_view_func"
    if camel_case:
        # 检查是否私有函数，若是，在生成的函数名前加下划线
        is_private = view_func_name.startswith("_")
        # 根据下划线分隔单词，每个单词首字母大写（驼峰命名法）
        view_func_name = "".join(
            [p.title() for p in view_func_name.replace(".", "_").split("_")]
        )
        if is_private:
            # 若原函数名是私有的，重新加上下划线
            view_func_name = f"_{view_func_name}"
    # 返回完整的视图函数名称
    return f"{namespace}{view_func_name}"


# 检查参数是否是 Tensor 类型或符号整数类型
def is_symint_or_tensor(arg: Argument) -> bool:
    return arg.type.is_tensor_like() or arg.type.is_symint_like()


# 移除参数绑定中的 const 引用修饰符
def remove_const_ref(binding: Binding) -> Binding:
    return Binding(
        name=binding.name,
        nctype=binding.nctype.remove_const_ref(),
        argument=binding.argument,
        default=binding.default,
    )


# 检查函数是否返回多个张量对象
def returns_multi_tensor(fn: NativeFunction) -> bool:
    returns = fn.func.returns
    assert len(returns) == 1  # 确保返回值列表长度为1
    # 检查返回值类型是否为列表类对象且其中包含张量
    returns_list_like = returns[0].type.is_list_like() is not None
    returns_tensor_like = returns[0].type.is_tensor_like()
    return returns_list_like and returns_tensor_like


# 生成处理特定类型状态获取和设置逻辑的字符串
#
# Args:
#   bindings (list): 需要处理的状态绑定列表（可以为空）
#   state_vec_type (NamedCType): 要返回或从中复制的向量类型
#
# Returns:
#   tuple: (获取逻辑字符串列表，设置逻辑字符串列表，包含项数表达式的字符串)
def generate_state_getter_setter(
    bindings: list[Binding],
    state_vec_type: NamedCType,
) -> tuple[list[str], list[str], str]:
    getter_logic = []
    setter_logic = []

    state_vec = state_vec_type.name
    # 生成状态向量的声明
    getter_logic.append(f"{state_vec_type.cpp_type()} {state_vec};")
    if len(bindings) > 0:
        # 如果有绑定项，设置迭代计数器
        setter_logic.append("auto i = 0;")

    num_exprs = []
    for i, b in enumerate(bindings):
        # 确保 b.argument 是 Argument 类型的对象
        assert isinstance(b.argument, Argument)
        if b.argument.type.is_list_like():
            # 处理列表类型的参数
            num_expr = f"{b.name}.size()"  # 计算列表长度的表达式
            num_exprs.append(num_expr)  # 将长度表达式添加到列表中
            getter = f"{state_vec}.insert({state_vec}.end(), {b.name}.begin(), {b.name}.end());"
            # 生成获取列表数据的逻辑语句
            setter = f"std::copy({state_vec}.begin() + i, {state_vec}.begin() + i + {b.name}.size(), {b.name}.begin());"
            # 生成设置列表数据的逻辑语句
        elif isinstance(b.argument.type, OptionalType):
            # 处理可选类型的参数
            num_expr = f"({b.name}.has_value() ? 1 : 0)"  # 判断可选值是否存在的表达式
            num_exprs.append(num_expr)  # 将判断表达式添加到列表中
            conditional = f"if({b.name}.has_value())"  # 可选值存在时的条件语句
            getter = (
                f"{conditional} {state_vec}.insert({state_vec}.end(), *({b.name}));"
            )
            # 生成获取可选值数据的逻辑语句
            setter = f"{conditional} {b.name} = {state_vec}[i];"
            # 生成设置可选值数据的逻辑语句
        else:
            num_expr = "1"  # 默认的单个元素的表达式
            num_exprs.append(num_expr)  # 将默认表达式添加到列表中
            getter = f"{state_vec}.push_back({b.name});"
            # 生成获取单个元素数据的逻辑语句
            setter = f"{b.name} = {state_vec}[i];"
            # 生成设置单个元素数据的逻辑语句

        getter_logic.append(getter)  # 将获取数据的逻辑语句添加到获取逻辑列表中
        setter_logic.append(setter)  # 将设置数据的逻辑语句添加到设置逻辑列表中
        if i < len(bindings) - 1:
            setter_logic.append(f"i += {num_expr};")  # 如果不是最后一个绑定，增加索引的逻辑语句

    # 根据所有项的总数表达式进行保留或断言
    num_items = "0" if len(num_exprs) == 0 else " + ".join(num_exprs)
    if len(bindings) > 0:
        getter_logic.insert(1, f"{state_vec}.reserve({num_items});")
        # 如果有绑定，插入保留容量的逻辑语句到获取逻辑列表的第二个位置

    getter_logic.append(f"return {state_vec};")  # 添加返回结果的逻辑语句到获取逻辑列表末尾

    return getter_logic, setter_logic, num_items
    # 返回获取逻辑列表、设置逻辑列表和总数表达式
# 处理给定的函数，生成一个字符串表示其结构化处理结果
def process_function(fn: NativeFunction, template: CodeTemplate) -> str:
    # 提取函数的绑定信息
    bindings = extract_bindings(fn)
    # 从绑定中过滤掉自身 ("self") 参数
    non_self_bindings = [b for b in bindings if b.name != "self"]

    # 获取除自身参数外的所有参数
    non_self_args = fn.func.arguments.flat_all[1:]
    # 生成非自身值绑定列表，用于生成结构体的构造函数和克隆参数
    non_self_value_bindings = [
        dispatcher.argument(a, remove_non_owning_ref_types=True) for a in non_self_args
    ]

    # 为生成的结构体生成构造函数和克隆参数
    constructor_args = [b.defn() for b in non_self_bindings]
    clone_args = [b.name for b in non_self_bindings]

    # 为生成的结构体生成状态变量声明
    state_variables = [
        f"{remove_const_ref(b).defn()};" for b in non_self_value_bindings
    ]

    # 为生成的结构体生成初始化列表表达式
    # allow_expensive_conversions=True 因为需要存储例如 SymIntArrayRefs 作为 vector<SymInt>。
    init_exprs = translate(
        non_self_bindings, non_self_value_bindings, allow_expensive_conversions=True
    )
    initializers = []
    for b, init_expr in zip(non_self_bindings, init_exprs):
        name = b.nctype.name
        assert isinstance(name, str)
        initializers.append(f"{name}({init_expr.expr})")

    # 生成对底层视图操作的调用
    call_input_name = "input_base"
    op_call_args = [call_input_name, *(b.name for b in non_self_bindings)]
    op_call = CALL_DISPATCH.substitute(
        unambiguous_name=fn.func.name.unambiguous_name(),
        unpacked_args=op_call_args,
    )

    # 如果函数返回多个张量视图，则需要一个 view_idx 来消除歧义
    if returns_multi_tensor(fn):
        view_idx_name = "view_idx"
        view_idx_typename = "int64_t"
        view_idx_decl = f"{view_idx_typename} {view_idx_name}"
        constructor_args.append(view_idx_decl)
        clone_args.append(view_idx_name)
        state_variables.append(f"{view_idx_decl};")
        initializers.append(f"{view_idx_name}({view_idx_name})")
        op_call += f"[{view_idx_name}]"

    # 生成生成的结构体的初始化列表
    initializer_list = f": {', '.join(initializers)}" if len(initializers) > 0 else ""

    # 为任何 symints 生成 getter / setter 逻辑
    symint_bindings = [
        b
        for b in non_self_bindings
        if isinstance(b.argument, Argument) and b.argument.type.is_symint_like()
    ]
    symints_vec_type = NamedCType("symints", VectorCType(BaseCType(SymIntT)))
    get_symints, set_symints, num_symints = generate_state_getter_setter(
        symint_bindings, symints_vec_type
    )

    # 为任何张量生成 getter / setter 逻辑
    tensor_bindings = [
        b
        for b in non_self_bindings
        if isinstance(b.argument, Argument) and b.argument.type.is_tensor_like()
    ]
    tensors_vec_type = NamedCType("tensors", VectorCType(BaseCType(tensorT)))
    get_tensors, set_tensors, num_tensors = generate_state_getter_setter(
        tensor_bindings, tensors_vec_type
    )
    # 使用模板字符串替换变量，生成函数模板的实现代码
    return template.substitute(
        # 替换操作函数的名称
        op=view_func_name(fn),
        # 替换操作函数的名称并转换为大写形式
        uppercase_op=view_func_name(fn, camel_case=False).upper(),
        # 设置超类为 torch::autograd::ViewFunc
        superclass="torch::autograd::ViewFunc",
        # 替换初始化列表的内容
        initializer_list=initializer_list,
        # 替换状态变量的内容
        state=state_variables,
        # 替换构造函数参数的内容
        constructor_args=constructor_args,
        # 替换克隆函数参数的内容
        clone_args=clone_args,
        # 替换符号整数向量的类型名称
        symints_vec=symints_vec_type.name,
        # 替换获取符号整数的函数
        get_symints=get_symints,
        # 替换设置符号整数的函数
        set_symints=set_symints,
        # 替换符号整数的数量
        num_symints=num_symints,
        # 替换张量向量的类型名称
        tensors_vec=tensors_vec_type.name,
        # 替换获取张量的函数
        get_tensors=get_tensors,
        # 替换设置张量的函数
        set_tensors=set_tensors,
        # 替换张量的数量
        num_tensors=num_tensors,
        # 替换调用输入名称
        call_input_name=call_input_name,
        # 替换操作调用的名称
        op_call=op_call,
    )
def gen_view_funcs(
    out: str,
    fns_with_infos: list[NativeFunctionWithDifferentiabilityInfo],
    template_path: str,
) -> None:
    # 提取不需要信息部分，只保留函数本身
    fns = [fn.func for fn in fns_with_infos if use_derived(fn)]
    
    # 只保留 out-of-place 视图函数
    view_fns = [
        fn for fn in fns if get_view_info(fn) is not None and not modifies_arguments(fn)
    ]

    # 生成视图函数的声明
    declarations = [process_function(fn, FUNCTION_DECLARATION) for fn in view_fns]
    
    # 生成视图函数的定义
    definitions = [process_function(fn, FUNCTION_DEFINITION) for fn in view_fns]
    
    # 生成操作头文件引用
    ops_headers = [f"#include <ATen/ops/{fn.root_name}_ops.h>" for fn in view_fns]

    # 文件基本名
    file_basename = "ViewFuncs"
    
    # 文件管理器实例化
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    
    # 针对每个后缀生成文件
    for suffix in [".h", ".cpp"]:
        fname = file_basename + suffix
        fm.write_with_template(
            fname,
            fname,
            lambda: {
                # 自动生成的注释，指明来源模板路径
                "generated_comment": "@"
                + f"generated from {fm.template_dir_for_comments()}/"
                + fname,
                # 视图函数声明
                "view_func_declarations": declarations,
                # 视图函数定义
                "view_func_definitions": definitions,
                # 操作头文件引用
                "ops_headers": ops_headers,
            },
        )
```