# `.\pytorch\torch\jit\_dataclass_impls.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import ast  # 导入抽象语法树模块
import dataclasses  # 导入数据类模块
import inspect  # 导入检查模块
import os  # 导入操作系统相关功能
from functools import partial  # 导入 partial 函数
from typing import Callable, Dict, List  # 导入类型提示相关模块

from torch._jit_internal import FAKE_FILENAME_PREFIX, is_optional  # 从 torch 内部 JIT 模块导入必要函数
from torch._sources import ParsedDef, SourceContext  # 从 torch 源码模块导入 ParsedDef 和 SourceContext 类

# 定义函数以生成虚拟文件名
def _get_fake_filename(cls, method_name):
    return os.path.join(FAKE_FILENAME_PREFIX, cls.__name__, method_name)

# 定义函数以合成函数体并返回解析后的函数定义
def compose_fn(cls, name: str, body_lines: List[str], signature: str) -> ParsedDef:
    # 将函数体行合并为字符串，每行前缀添加两个空格以保持缩进
    body = "\n".join(f"  {b}" for b in body_lines)
    # 构建函数声明字符串
    decl = f"def {name}{signature}:\n{body}"

    # 尝试解析函数声明字符串为抽象语法树
    try:
        py_ast = ast.parse(decl)
    except SyntaxError as e:
        # 如果解析失败，抛出运行时错误，提醒提交 Bug 报告
        raise RuntimeError(
            f"TorchScript failed to synthesize dataclass method '{name}' for class '{cls.__name__}'. "
            "Please file a bug report at <https://github.com/pytorch/pytorch/issues>"
        ) from e

    # 生成虚拟文件名
    fake_filename = _get_fake_filename(cls, name)
    # 返回解析后的函数定义对象 ParsedDef
    return ParsedDef(
        py_ast,
        ctx=SourceContext(
            source=decl, filename=fake_filename, file_lineno=0, leading_whitespace_len=0
        ),
        source=decl,
        filename=fake_filename,
        file_lineno=0,
    )

# 定义函数以合成 __init__ 方法的函数定义并返回
def synthesize__init__(cls) -> ParsedDef:
    # 如果数据类中存在使用默认工厂函数初始化的字段，则抛出 NotImplementedError
    if any(
        field.default_factory is not dataclasses.MISSING
        for field in dataclasses.fields(cls)
    ):
        raise NotImplementedError(
            "Default factory initializers are not supported in TorchScript dataclasses"
        )

    # 获取生成的 __init__ 方法的签名，注意处理 InitVar 注解
    signature = inspect.signature(cls.__init__)

    # 处理 InitVar 注解，将其替换为其类型
    init_vars: List[str] = []
    params = []
    for name, param in signature.parameters.items():
        ann = param.annotation

        if isinstance(ann, dataclasses.InitVar):
            # TorchScript 解释器无法处理 InitVar 注解，因此在此处解开其底层类型
            init_vars.append(name)
            params.append(param.replace(annotation=ann.type))  # type: ignore[attr-defined]
        else:
            params.append(param)

    # 更新签名，替换掉处理后的参数列表
    signature = signature.replace(parameters=params)

    # 返回生成的 __init__ 方法的函数定义对象 ParsedDef
    return ParsedDef(
        py_ast,  # 使用之前解析的抽象语法树
        ctx=SourceContext(
            source=decl, filename=fake_filename, file_lineno=0, leading_whitespace_len=0
        ),
        source=decl,  # 函数声明字符串
        filename=fake_filename,  # 虚拟文件名
        file_lineno=0,
    )
    body = [
        # 遍历数据类的所有字段
        f"self.{field.name} = {field.name}"
        # 仅对那些允许初始化并且不在init_vars列表中的字段进行操作
        for field in dataclasses.fields(cls)
        if field.init and field.name not in init_vars
    ]
    # 检查是否存在用户定义的__post_init__方法
    if hasattr(cls, "__post_init__"):
        # 如果存在，则将调用用户实现的__post_init__方法，并传入init_vars参数
        body.append("self.__post_init__(" + ", ".join(init_vars) + ")")

    # 调用compose_fn函数，生成__init__方法的实现体，如果body为空则使用"pass"语句作为默认值
    return compose_fn(cls, "__init__", body or ["pass"], signature=str(signature))
# 定义一个函数用于合成类的 __repr__ 方法
def synthesize__repr__(cls) -> ParsedDef:
    # 调用 compose_fn 函数，生成 __repr__ 方法的函数体
    return compose_fn(
        cls,
        "__repr__",
        [
            # 返回一个字符串，表示类的字符串表示形式，包括所有字段的名称和值
            f"return '{cls.__name__}("
            + ", ".join(
                [
                    f"{field.name}=self.{field.name}"
                    for field in dataclasses.fields(cls)
                    if field.repr
                ]
            )
            + ")'"
        ],
        signature="(self) -> str",
    )


# 定义一个函数用于合成类的 __hash__ 方法
def synthesize__hash__(cls) -> ParsedDef:
    # 调用 compose_fn 函数，生成 __hash__ 方法的函数体
    return compose_fn(
        cls,
        "__hash__",
        [
            # 由于 TorchScript 解释器不调用自定义的 __hash__ 方法，这里只是一个占位符，防止编译失败
            "raise NotImplementedError('__hash__ is not supported for dataclasses in TorchScript')"
        ],
        signature="(self) -> int",
    )


# 实现类的 __eq__ 和 __ne__ 方法
def synthesize_equality(cls, name: str, converse: str) -> ParsedDef:
    # 调用 synthesize_comparison 函数，生成 __eq__ 或 __ne__ 方法的函数体
    return synthesize_comparison(
        cls,
        name,
        allow_eq=True,
        raise_on_none=False,
        inner=[f"if val1 {converse} val2: return False"],
    )


# 实现类的比较方法（__eq__ 或 __ne__）
def synthesize_inequality(cls, name: str, op: str, allow_eq: bool) -> ParsedDef:
    # 调用 synthesize_comparison 函数，生成比较方法的函数体
    return synthesize_comparison(
        cls,
        name,
        allow_eq,
        raise_on_none=True,
        inner=[
            f"if val1 {op} val2: return True",
            f"elif val2 {op} val1: return False",
        ],
    )


# 合成类的比较方法（__eq__ 或 __ne__）的通用函数
def synthesize_comparison(
    cls, name: str, allow_eq: bool, raise_on_none: bool, inner: List[str]
) -> ParsedDef:
    # 生成函数体的主体部分
    body = []
    for field in dataclasses.fields(cls):
        if not field.compare:
            continue

        # 添加字段值的比较语句
        body.extend(
            [
                f"val1 = self.{field.name}",
                f"val2 = other.{field.name}",
            ]
        )
        
        # 处理可选字段的类型细化
        body.extend(
            inner
            if not is_optional(field.type)
            else [
                # 对可选字段进行类型细化，以避免解释器的类型错误
                "if val1 is not None and val2 is not None:",
                *["  " + line for line in inner],
                "elif (val1 is None) != (val2 is None):",
                f"  raise TypeError('Cannot compare {cls.__name__} with None')"
                if raise_on_none
                else "  return False",
            ]
        )

    # 返回比较结果
    body.append(f"return {allow_eq}")
    return compose_fn(
        cls, name, body, signature=f"(self, other: {cls.__name__}) -> bool"
    )


# 定义一个字典，用于存储数据类的魔术方法和对应的合成函数
DATACLASS_MAGIC_METHODS: Dict[str, Callable] = {
    "__init__": synthesize__init__,
    "__repr__": synthesize__repr__,
    "__hash__": synthesize__hash__,
    "__eq__": partial(synthesize_equality, name="__eq__", converse="!="),
    # 创建特殊方法 "__ne__"，其实现是调用 synthesize_equality 函数，生成一个不等于操作的方法
    "__ne__": partial(synthesize_equality, name="__ne__", converse=="=="),
    
    # 创建特殊方法 "__lt__"，其实现是调用 synthesize_inequality 函数，生成一个小于操作的方法，不包括等于
    "__lt__": partial(synthesize_inequality, name="__lt__", op="<", allow_eq=False),
    
    # 创建特殊方法 "__le__"，其实现是调用 synthesize_inequality 函数，生成一个小于等于操作的方法，包括等于
    "__le__": partial(synthesize_inequality, name="__le__", op="<", allow_eq=True),
    
    # 创建特殊方法 "__gt__"，其实现是调用 synthesize_inequality 函数，生成一个大于操作的方法，不包括等于
    "__gt__": partial(synthesize_inequality, name="__gt__", op=">", allow_eq=False),
    
    # 创建特殊方法 "__ge__"，其实现是调用 synthesize_inequality 函数，生成一个大于等于操作的方法，包括等于
    "__ge__": partial(synthesize_inequality, name="__ge__", op=">", allow_eq=True),
}


注释：


# 这行代码表示一个代码块的结束，通常与一个以前的语句（如 if、for、while 等）配对使用，
# 表示这些语句内的代码执行结束。
```