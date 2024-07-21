# `.\pytorch\torch\jit\supported_ops.py`

```py
# mypy: allow-untyped-defs
# 引入inspect模块，用于检查和获取活动对象的信息
import inspect
# 引入textwrap模块，用于对文本进行自动换行和缩进处理

# 引入torch.jit模块，用于PyTorch的即时编译功能
import torch.jit
# 从torch.jit._builtins模块中导入_find_builtin函数
from torch.jit._builtins import _find_builtin

# 此文件用于使用sphinx autodoc生成文档
# > help(torch.jit.supported_ops)也会通过编程方式列出支持的操作

# 定义一个函数，判断名称是否以"_"开头且不以"__"开头
def _hidden(name):
    return name.startswith("_") and not name.startswith("__")

# 定义一个函数，返回类型的字符串表示
def _emit_type(type):
    return str(type)

# 定义一个函数，生成参数的字符串表示
def _emit_arg(indent, i, arg):
    v = f"{arg.name} : {_emit_type(arg.type)}"
    default = arg.default_value
    if default is not None:
        v = f"{v}={str(default)}"
    if i > 0:
        v = f"\n{' ' * indent}{v}"
    return v

# 定义一个函数，生成所有参数的字符串表示
def _emit_args(indent, arguments):
    return ",".join(_emit_arg(indent, i, arg) for i, arg in enumerate(arguments))

# 定义一个函数，生成返回值的字符串表示
def _emit_ret(ret):
    return _emit_type(ret.type)

# 定义一个函数，生成所有返回值的字符串表示
def _emit_rets(returns):
    if len(returns) == 1:
        return _emit_ret(returns[0])
    return f"Tuple[{', '.join(_emit_ret(r) for r in returns)}]"

# 定义一个函数，生成模块、方法和其对应schema的字符串表示
def _emit_schema(mod, name, schema, arg_start=0, padding=4):
    if mod is None:
        qualified_name = name
    else:
        qualified_name = f"{mod}.{name}"
    schema_str = (
        f"{qualified_name}"
        f"({_emit_args(len(qualified_name) + 1 + padding, schema.arguments[arg_start:])}) "
        f"-> {_emit_rets(schema.returns)}"
    )
    return schema_str

# 定义一个函数，判断是否为张量方法
def _get_tensor_ops():
    def is_tensor_method(schema):
        if len(schema.arguments) == 0:
            return False
        self = schema.arguments[0]
        if self.name != "self":
            return False
        if not self.type.isSubtypeOf(torch._C.TensorType.get()):
            return False
        return True

    methods = []
    # 发现方法
    for elem in dir(torch.Tensor):
        if not _hidden(elem):
            # 获取运算符的schema
            schemas = torch._C._jit_get_schemas_for_operator("aten::" + elem)
            for schema in schemas:
                if is_tensor_method(schema):
                    methods.append(_emit_schema("Tensor", elem, schema, arg_start=1))

    return "Supported Tensor Methods", methods

# 定义一个函数，获取torch.nn.functional模块中的操作
def _get_nn_functional_ops():
    functions = []

    # 迭代torch.nn.functional模块
    mod = torch.nn.functional
    name = mod.__name__
    # 遍历 torch.nn.functional 模块中的所有元素（函数名）
    for elem in dir(torch.nn.functional):
        # 获取模块中元素的属性
        attr = getattr(mod, elem)
        # 如果不是函数或者是内部方法（以 _ 开头），则忽略
        if not inspect.isfunction(attr) or _hidden(elem[0]):
            # 忽略非函数和内部方法
            continue

        # 获取函数所在的模块
        attr_module = inspect.getmodule(attr)
        # 如果找不到模块，则抛出运行时错误
        if not attr_module:
            raise RuntimeError(f"Module for {attr} not found")

        # 如果函数不属于 torch.nn.functional 模块，则忽略
        if "torch.nn.functional" not in attr_module.__name__:
            # 忽略来自 torch.nn.functional 之外的函数
            continue

        try:
            # 编译函数并获取其 schema
            scripted = torch.jit.script(attr)
            scripted_schema = scripted.schema
            # 发射（emit）该函数的 schema，并将其添加到 functions 列表中
            functions.append(_emit_schema(name, elem, scripted_schema))
        except:  # noqa: B001,E722
            # 捕获所有异常，跳过 interpolate / boolean dispatched 的内容
            pass

    # 遍历包含许多内置函数的模块
    for mod in torch.jit._builtins._modules_containing_builtins:
        name = mod.__name__
        # 遍历模块中的所有元素（函数名）
        for elem in dir(mod):
            # 获取元素对应的内置函数
            builtin = _find_builtin(getattr(mod, elem))
            # 如果找到了内置函数
            if builtin is not None:
                # 获取该内置函数的所有 schema
                schemas = torch._C._jit_get_schemas_for_operator(builtin)
                # 遍历每个 schema
                for schema in schemas:
                    # 如果元素名不是以 _ 开头，则将其 schema 添加到 functions 列表中
                    if not _hidden(elem):
                        functions.append(_emit_schema(name, elem, schema))
    # 返回字符串和收集到的函数信息列表 functions
    return "Supported PyTorch Functions", functions
# 获取内置函数的辅助函数
def _get_builtins_helper():
    # 初始化空列表，用于存储内置函数和内置操作的元组
    builtins = []
    # 遍历 torch.jit._builtins._builtin_ops 中的元组 (fn, _builtin_name)
    for fn, _builtin_name in torch.jit._builtins._builtin_ops:
        # 获取函数 fn 所属的模块
        mod = inspect.getmodule(fn)

        # 如果 fn 没有 "__name__" 属性，通常是类型定义类，跳过
        if not hasattr(fn, "__name__"):
            continue
        # 如果模块为空，跳过
        if not mod:
            continue
        # 如果函数名、限定函数名或模块名被标记为隐藏，跳过
        if _hidden(fn.__name__) or _hidden(fn.__qualname__) or _hidden(mod.__name__):
            continue

        # 如果模块名包含 "torch._C"，跳过
        if "torch._C" in mod.__name__:
            continue

        # 将合格的内置函数和内置操作添加到 builtins 列表中
        builtins.append((fn, _builtin_name))

    return builtins


# 检查函数是否为数学函数
def _is_math_fn(fn):
    # 获取函数 fn 所属的模块
    mod = inspect.getmodule(fn)
    # 如果模块为空，引发运行时错误，指明找不到 fn 的模块
    if not mod:
        raise RuntimeError(f"Module for {fn} not found")

    # 返回布尔值，判断函数 fn 是否属于 "math" 模块
    return mod.__name__ == "math"


# 获取 TorchScript 内置函数
def _get_torchscript_builtins():
    # 初始化空列表，用于存储 TorchScript 内置函数
    functions = []
    # 从 _get_builtins_helper() 返回的列表中过滤掉数学函数，得到内置函数列表 builtins
    builtins = filter(lambda fn: not _is_math_fn(fn[0]), _get_builtins_helper())
    # 将过滤后的内置函数列表转换为列表
    builtins_list = list(builtins)
    
    # 遍历内置函数列表 builtins_list
    for fn, _builtin_name in builtins_list:
        # 获取函数 fn 所属的模块
        mod = inspect.getmodule(fn)
        # 如果模块为空，引发运行时错误，指明找不到 fn 的模块
        if not mod:
            raise RuntimeError(f"Module for {fn} not found")
        
        # 查找内置函数 fn 对应的函数实现
        builtin = _find_builtin(fn)
        
        # 如果找到内置函数
        if builtin is not None:
            # 获取内置函数的所有 schema
            schemas = torch._C._jit_get_schemas_for_operator(builtin)
            # 遍历每个 schema
            for schema in schemas:
                # 根据模块名、函数名和 schema 生成字符串，添加到 functions 列表中
                functions.append(_emit_schema(mod.__name__, fn.__name__, schema))
                pass

    # 返回 TorchScript 内置函数的标题和 functions 列表
    return "TorchScript Builtin Functions", functions


# 获取数学函数的内置函数
def _get_math_builtins():
    # 初始化空列表，用于存储数学函数的内置函数
    functions = []
    # 从 _get_builtins_helper() 返回的列表中过滤出数学函数，得到内置函数列表 builtins
    builtins = filter(lambda fn: _is_math_fn(fn[0]), _get_builtins_helper())
    # 将过滤后的内置函数列表转换为列表
    builtins_list = list(builtins)
    
    # 遍历内置函数列表 builtins_list
    for fn, _builtin_name in builtins_list:
        # 获取函数 fn 所属的模块
        mod = inspect.getmodule(fn)
        # 如果模块为空，引发运行时错误，指明找不到 fn 的模块
        if not mod:
            raise RuntimeError(f"Module for {fn} not found")
        
        # 查找内置函数 fn 对应的函数实现
        builtin = _find_builtin(fn)
        
        # 如果找到内置函数
        if builtin is not None:
            # 获取内置函数的所有 schema
            schemas = torch._C._jit_get_schemas_for_operator(builtin)
            # 遍历每个 schema
            for schema in schemas:
                # 根据模块名、函数名和 schema 生成字符串
                schema_str = _emit_schema(mod.__name__, fn.__name__, schema)
                # 如果 schema_str 中包含 "Tensor"，跳过该 schema
                if "Tensor" in schema_str:
                    continue
                # 否则将 schema 添加到 functions 列表中
                functions.append(schema)
                pass

    # 返回数学函数模块的标题和 functions 列表
    return "``math`` Module", functions


# 获取全局内置函数
def _get_global_builtins():
    # 从 torch/csrc/jit/frontend/ir_emitter.cpp 的 'globals' 映射中获取支持的内置函数列表
    supported_builtins = [
        "print",
        "tuple",
        "float",
        "complex",
        "int",
        "bool",
        "str",
        "getattr",
        "hasattr",
        "isinstance",
        "len",
        "hex",
        "oct",
        "round",
        "hash",
        "min",
        "max",
        "abs",
        "all",
        "divmod",
        "list",
        "ord",
        "chr",
        "bin",
        "range",
        "zip",
        "enumerate",
        "sorted",
    ]
    # 定义操作重命名的映射表，将指定的操作名映射到对应的 ATen 格式
    op_renames = {
        "bool": "aten::Bool",
        "int": "aten::Int",
        "float": "aten::Float",
        "complex": "aten::Complex",
        "abs": "prim::abs",
        "max": "prim::max",
        "min": "prim::min",
        "range": "fake::does_not_exist",
    }

    # 定义不需要特定模式的操作的说明字典
    schemaless_op_explanations = {
        "print": "Print any value",
        "tuple": "Lists cannot be converted to tuples with this method since their size is not statically known",
        "getattr": "Attribute name must be a literal string",
        "hasattr": "Attribute name must be a literal string",
        "isinstance": "Result is static",
        "zip": "Arguments must be iterable. See :ref:`Iterables <jit_iterables>` for details.",
        "enumerate": "Arguments must be iterable. See :ref:`Iterables <jit_iterables>` for details.",
        "range": "Can only be used as an iterator in a for loop",
    }

    # 定义包含特殊魔术方法名称的列表
    magic_methods = [
        ("complex", "__complex__"),
        ("float", "__float__"),
        ("int", "__int__"),
        ("bool", "__bool__"),
        ("str", "__str__"),
        ("len", "__len__"),
        ("hex", "__hex__"),
        ("oct", "__oct__"),
    ]

    # 生成魔术方法名称和其对应文档的字符串列表
    magic_methods_rows = []
    for fn, magic_method in magic_methods:
        magic_methods_rows.append(f'"{fn}", "``{magic_method}``"')

    # 初始化存放有模式的操作和无模式的操作的空列表
    schematized_ops = []
    schemaless_ops = []

    # 遍历支持的内置函数列表，并处理每个函数
    for fn in supported_builtins:
        # 默认使用 ATen 格式的操作名
        op_name = f"aten::{fn}"
        # 如果操作名在重命名映射表中，使用映射后的名字
        if fn in op_renames:
            op_name = op_renames[fn]
        # 获取给定操作名的所有模式
        schemas = torch._C._jit_get_schemas_for_operator(op_name)
        # 遍历每个模式并生成相应的模式描述
        for s in schemas:
            schematized_ops.append(_emit_schema(None, fn, s, padding=0))
        # 如果没有找到模式，则生成表格行描述无模式的操作
        if len(schemas) > 0:
            schematized_ops.append("")
        else:
            table_row = (
                f'":external+python:py:obj:`{fn}`", "{schemaless_op_explanations[fn]}"'
            )
            schemaless_ops.append(table_row)

    # 将列表转换为字符串，并用制表符进行缩进
    schematized_ops_str = "\n".join(schematized_ops)
    schemaless_ops_str = "\n".join(schemaless_ops)
    magic_methods_rows_str = "\n".join(magic_methods_rows)
    schematized_ops_str = textwrap.indent(schematized_ops_str, "\t")
    schemaless_ops_str = textwrap.indent(schemaless_ops_str, "\t")
    magic_methods_rows_str = textwrap.indent(magic_methods_rows_str, "\t")

    # 构建最终的文档段落
    section = f"""
#`
"""
以下表格中的函数是支持的，但没有静态模式

.. csv-table::
    :header: "Function", "Note"

{schemaless_ops_str}

以下函数将在 :any:`TorchScript` 类上使用相应的魔术方法

.. csv-table::
    :header: "Function", "Magic Method"

{magic_methods_rows_str}

这些内置函数使用模式

.. rst-class:: codeblock-height-limiter

::

{schematized_ops_str}
    """

    # 返回标题和段落内容
    return "Python Built-in Functions", section


def _list_supported_ops():
    # 内部函数，生成格式化的块字符串
    def emit_block(decls):
        return "\n.. rst-class:: codeblock-height-limiter\n\n::\n\n{}\n".format(
            "".join(f"    {d}\n\n" for d in decls)
        )

    # 初始化一个空字符串用于存储所有段落内容
    body = ""
    # 定义操作收集函数列表
    op_gathering_fns = (
        _get_tensor_ops,              # 获取张量操作
        _get_nn_functional_ops,       # 获取神经网络函数操作
        _get_torchscript_builtins,    # 获取 TorchScript 内置函数
        _get_global_builtins,         # 获取全局内置函数
        _get_math_builtins,           # 获取数学内置函数
    )
    # 遍历操作收集函数
    for fn in op_gathering_fns:
        header, items = fn()                      # 调用函数获取标题和内容
        link_target = header.replace("`", "").replace("-", "").lower().replace(" ", "-")  # 格式化链接目标
        if isinstance(items, str):
            # 如果内容是字符串，格式化成段落形式
            section = f"{header}\n{'~' * len(header)}\n{items}\n"
        else:
            # 如果内容是列表，使用 emit_block 函数格式化成块字符串
            section = f"{header}\n{'~' * len(header)}\n{emit_block(items)}"
        # 为每个段落添加链接目标前缀
        section = f".. _{link_target}:" + "\n\n" + section
        # 将段落添加到主体内容中
        body += section

    return body


# 设置文档字符串为生成的内容
__doc__ = _list_supported_ops()
```