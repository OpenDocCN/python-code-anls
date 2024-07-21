# `.\pytorch\torch\_export\db\case.py`

```
# 设置 mypy 参数，允许未类型化的函数定义
# 导入 inspect 模块，用于检查对象
import inspect
# 导入 re 模块，用于正则表达式操作
import re
# 导入 string 模块，包含字符串相关的常量和函数
import string
# 导入 dataclass 和 field 函数，用于创建数据类
from dataclasses import dataclass, field
# 导入 Enum 类，用于创建枚举类型
from enum import Enum
# 导入 Any, Dict, List, Optional, Set, Tuple, Union 类型提示
from typing import Any, Dict, List, Optional, Set, Tuple, Union
# 导入 ModuleType 类型，用于模块对象
from types import ModuleType

# 导入 torch 库，用于深度学习框架
import torch

# 定义全局变量 _TAGS，存储各类标签和与之关联的信息
_TAGS: Dict[str, Dict[str, Any]] = {
    "torch": {
        "cond": {},
        "dynamic-shape": {},
        "escape-hatch": {},
        "map": {},
        "dynamic-value": {},
        "operator": {},
        "mutation": {},
    },
    "python": {
        "assert": {},
        "builtin": {},
        "closure": {},
        "context-manager": {},
        "control-flow": {},
        "data-structure": {},
        "standard-library": {},
        "object-model": {},
    },
}

# 定义枚举类 SupportLevel，表示功能支持的阶段
class SupportLevel(Enum):
    """
    Indicates at what stage the feature
    used in the example is handled in export.
    """
    SUPPORTED = 1
    NOT_SUPPORTED_YET = 0

# 定义 ExportArgs 类，用于包装导出函数的参数
class ExportArgs:
    __slots__ = ("args", "kwargs")

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

# 定义函数 check_inputs_type，用于验证输入参数的类型
def check_inputs_type(x):
    if not isinstance(x, (ExportArgs, tuple)):
        raise ValueError(
            f"Expecting inputs type to be either a tuple, or ExportArgs, got: {type(x)}"
        )

# 定义函数 _validate_tag，用于验证标签的有效性
def _validate_tag(tag: str):
    parts = tag.split(".")
    t = _TAGS
    for part in parts:
        assert set(part) <= set(
            string.ascii_lowercase + "-"
        ), f"Tag contains invalid characters: {part}"
        if part in t:
            t = t[part]
        else:
            raise ValueError(f"Tag {tag} is not found in registered tags.")

# 定义数据类 ExportCase，表示导出用例
@dataclass(frozen=True)
class ExportCase:
    example_inputs: InputsType
    description: str  # A description of the use case.
    model: torch.nn.Module
    name: str
    extra_inputs: Optional[InputsType] = None  # For testing graph generalization.
    # Tags associated with the use case. (e.g dynamic-shape, escape-hatch)
    tags: Set[str] = field(default_factory=set)
    support_level: SupportLevel = SupportLevel.SUPPORTED
    dynamic_shapes: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        check_inputs_type(self.example_inputs)
        if self.extra_inputs is not None:
            check_inputs_type(self.extra_inputs)

        for tag in self.tags:
            _validate_tag(tag)

        if not isinstance(self.description, str) or len(self.description) == 0:
            raise ValueError(f'Invalid description: "{self.description}"')

# 定义全局变量 _EXAMPLE_CASES，存储导出用例的字典
_EXAMPLE_CASES: Dict[str, ExportCase] = {}
# 定义全局变量 _MODULES，存储模块对象的集合
_MODULES: Set[ModuleType] = set()
# 定义全局变量 _EXAMPLE_CONFLICT_CASES，存储冲突导出用例的字典
_EXAMPLE_CONFLICT_CASES: Dict[str, List[ExportCase]] = {}
# 定义全局变量 _EXAMPLE_REWRITE_CASES，存储重写导出用例的字典
_EXAMPLE_REWRITE_CASES: Dict[str, List[ExportCase]] = {}

# 定义函数 register_db_case，注册用户提供的导出用例到用例库中
def register_db_case(case: ExportCase) -> None:
    """
    Registers a user provided ExportCase into example bank.
    """
    # 如果 case.name 在 _EXAMPLE_CASES 中存在
    if case.name in _EXAMPLE_CASES:
        # 如果 case.name 不在 _EXAMPLE_CONFLICT_CASES 中
        if case.name not in _EXAMPLE_CONFLICT_CASES:
            # 将 case.name 添加到 _EXAMPLE_CONFLICT_CASES 中作为键，其值为包含 _EXAMPLE_CASES[case.name] 的列表
            _EXAMPLE_CONFLICT_CASES[case.name] = [_EXAMPLE_CASES[case.name]]
        # 向 _EXAMPLE_CONFLICT_CASES[case.name] 列表中添加当前 case 对象
        _EXAMPLE_CONFLICT_CASES[case.name].append(case)
        # 无论是否已存在，直接返回，函数执行结束
        return
    
    # 如果 case.name 不在 _EXAMPLE_CASES 中，则将当前 case 对象添加到 _EXAMPLE_CASES 中
    _EXAMPLE_CASES[case.name] = case
# 将给定名称转换为蛇形命名法（例如 "CamelCase" -> "camel_case"）
def to_snake_case(name):
    # 使用正则表达式将大写字母前加下划线，然后将连续的大写字母与小写字母或数字分开
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # 将小写字母或数字后的大写字母加下划线
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _make_export_case(m, name, configs):
    # 如果 m 不是 torch.nn.Module 类型，则抛出类型错误
    if not isinstance(m, torch.nn.Module):
        raise TypeError("Export case class should be a torch.nn.Module.")

    # 如果 configs 中缺少 "description" 键，则使用文档字符串作为描述的回退
    if "description" not in configs:
        assert (
            m.__doc__ is not None
        ), f"Could not find description or docstring for export case: {m}"
        configs = {**configs, "description": m.__doc__}
    
    # 返回一个 ExportCase 对象，包括给定的模型 m、名称 name 和配置信息 configs
    return ExportCase(**{**configs, "model": m, "name": name})


def export_case(**kwargs):
    """
    Decorator for registering a user provided case into example bank.
    """

    def wrapper(m):
        # 将关键字参数 kwargs 存储到 configs 中
        configs = kwargs
        # 获取模块对象
        module = inspect.getmodule(m)
        # 检查模块是否已经在 _MODULES 集合中，如果是则抛出运行时错误
        if module in _MODULES:
            raise RuntimeError("export_case should only be used once per example file.")

        assert module is not None
        # 将模块添加到 _MODULES 集合中
        _MODULES.add(module)
        # 获取模块的名称，并创建相应的 ExportCase 对象
        module_name = module.__name__.split(".")[-1]
        case = _make_export_case(m, module_name, configs)
        # 注册案例到数据库中
        register_db_case(case)
        # 返回创建的 ExportCase 对象
        return case

    return wrapper


def export_rewrite_case(**kwargs):
    def wrapper(m):
        # 将关键字参数 kwargs 存储到 configs 中
        configs = kwargs

        # 从 configs 中弹出 "parent" 键，并确保其为 ExportCase 类型
        parent = configs.pop("parent")
        assert isinstance(parent, ExportCase)
        
        # 以 parent 的名称为键将案例添加到 _EXAMPLE_REWRITE_CASES 字典中
        key = parent.name
        if key not in _EXAMPLE_REWRITE_CASES:
            _EXAMPLE_REWRITE_CASES[key] = []

        # 将 parent 的示例输入添加到 configs 中，并创建相应的 ExportCase 对象
        configs["example_inputs"] = parent.example_inputs
        case = _make_export_case(m, to_snake_case(m.__name__), configs)
        
        # 将创建的 ExportCase 对象添加到对应键的案例列表中
        _EXAMPLE_REWRITE_CASES[key].append(case)
        # 返回创建的 ExportCase 对象
        return case

    return wrapper


def normalize_inputs(x: InputsType) -> ExportArgs:
    # 如果 x 是元组，则创建一个 ExportArgs 对象并返回
    if isinstance(x, tuple):
        return ExportArgs(*x)

    # 如果 x 是 ExportArgs 类型，则直接返回 x
    assert isinstance(x, ExportArgs)
    return x
```