# `.\pytorch\torchgen\dest\lazy_ts_lowering.py`

```py
from torchgen.api.lazy import LazyArgument, LazyIrSchema
from torchgen.api.types import OptionalCType


def ts_lowering_body(schema: LazyIrSchema) -> str:
    # 目前，我们只需一个 IR 类声明和之后的方法定义，而且我们使用的是函数版本而不是 inplace 版本。
    emplace_arguments = []

    def get_value(arg: LazyArgument) -> str:
        # 如果参数的类型是 OptionalCType，则返回条件表达式，否则返回普通获取操作数的表达式
        if isinstance(arg.lazy_type, OptionalCType):
            return f"has_{arg.name} ? loctx->GetOutputOp(operand(i++)) : nullptr"
        return "loctx->GetOutputOp(operand(i++))"

    # 遍历位置参数列表
    for arg in schema.positional_args:
        if arg.is_lazy_value:
            # 如果参数是 lazy_value，则获取其值并添加到 emplace_arguments
            emplace_arguments.append(get_value(arg))
            continue
        # 否则，将参数名和参数值格式化添加到 emplace_arguments
        emplace_arguments.append(f'"{arg.name}", {arg.name}')

    # 将 emplace_arguments 转换为字符串，并格式化为向量初始化语句
    emplace_arguments_str = "\n    ".join(
        [f"arguments.emplace_back({a});" for a in emplace_arguments]
    )

    # 构建关键字参数的值列表和标量列表
    emplace_kwarg_values = [
        f'"{arg.name}", {get_value(arg)}' for arg in schema.keyword_values
    ]
    emplace_kwarg_scalars = [
        f'"{arg.name}", {arg.name}' for arg in schema.keyword_scalars
    ]

    # 合并关键字参数值和标量的 emplace 语句
    emplace_kwarguments = "\n    ".join(
        [
            f"kwarguments.emplace_back({a});"
            for a in emplace_kwarg_values + emplace_kwarg_scalars
        ]
    )

    # 返回语句，包括初始化 arguments 和 kwarguments 向量，并调用 LowerTSBuiltin 函数
    return f"""\
    std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::NamedValue> kwarguments;
    arguments.reserve({len(emplace_arguments)});
    kwarguments.reserve({len(emplace_kwarg_values + emplace_kwarg_scalars)});
    size_t i = 0;
    {emplace_arguments_str}
    {emplace_kwarguments}
    torch::lazy::TSOpVector {schema.aten_name}_out = torch::lazy::LowerTSBuiltin(function, op().op, arguments, kwarguments);
    TORCH_CHECK_EQ({schema.aten_name}_out.size(), {len(schema.returns)});

    return {schema.aten_name}_out;
"""
```