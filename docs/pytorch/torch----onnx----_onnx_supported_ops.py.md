# `.\pytorch\torch\onnx\_onnx_supported_ops.py`

```
# mypy: allow-untyped-defs
# 导入模块 inspect，用于检查对象的特征
import inspect
# 导入类型定义 Dict, List, Union
from typing import Dict, List, Union
# 导入 torch 的 C 模块
from torch import _C
# 导入 torch.onnx 的 _constants 模块
from torch.onnx import _constants
# 导入 torch.onnx._internal 的 registration 模块
from torch.onnx._internal import registration

# 定义 TorchSchema 类
class _TorchSchema:
    def __init__(self, schema: Union[_C.FunctionSchema, str]) -> None:
        # 如果 schema 是 _C.FunctionSchema 类型
        if isinstance(schema, _C.FunctionSchema):
            # 设置对象的名称和重载名称
            self.name: str = schema.name
            self.overload_name: str = schema.overload_name
            # 获取参数列表中每个参数的名称
            self.arguments: List[str] = [arg.name for arg in schema.arguments]
            # 初始化可选参数列表为空
            self.optional_arguments: List[str] = []
            # 获取返回值列表中每个返回值的名称
            self.returns: List[str] = [ret.name for ret in schema.returns]
            # 初始化 opsets 列表为空
            self.opsets: List[int] = []
        else:
            # 如果 schema 是字符串类型，则直接设置名称，并初始化其他属性为空列表或空字符串
            self.name = schema
            self.overload_name = ""
            self.arguments = []
            self.optional_arguments = []
            self.returns = []
            self.opsets = []

    # 定义 __str__ 方法，返回对象的字符串表示
    def __str__(self) -> str:
        # 构建对象的字符串表示，包括名称、重载名称、参数列表、返回值列表和 opsets 列表
        s = (
            f"{self.name}.{self.overload_name}("
            + ", ".join(self.arguments)
            + ") -> ("
            + ", ".join(self.returns)
            + ")"
            + " in opsets "
            + ", ".join(str(opset) for opset in self.opsets)
        )
        return s

    # 定义 __hash__ 方法，返回对象的哈希值
    def __hash__(self):
        # TODO(thiagocrepaldi): 处理重载名称？
        return hash(self.name)

    # 定义 __eq__ 方法，判断对象是否相等
    def __eq__(self, other) -> bool:
        # 如果 other 不是 _TorchSchema 类型，则返回 False
        if not isinstance(other, _TorchSchema):
            return False
        # TODO(thiagocrepaldi): 处理重载名称？
        # 比较对象的名称是否相等
        return self.name == other.name

    # 定义 is_aten 方法，判断对象是否为 "aten::" 开头
    def is_aten(self) -> bool:
        return self.name.startswith("aten::")

    # 定义 is_backward 方法，判断对象是否包含 "backward"
    def is_backward(self) -> bool:
        return "backward" in self.name

# 定义 _symbolic_argument_count 函数，返回函数的参数列表
def _symbolic_argument_count(func):
    # 初始化参数列表为空
    params = []
    # 获取函数的参数签名
    signature = inspect.signature(func)
    # 初始化可选参数列表为空
    optional_params = []
    # 遍历参数签名中的每个参数
    for name, parameter in signature.parameters.items():
        # 如果参数名是 {"_outputs", "g"} 中的一个，跳过
        if name in {"_outputs", "g"}:
            continue
        # 如果参数没有默认值，则将其添加到可选参数列表中
        if parameter.default is parameter.empty:
            optional_params.append(parameter)
        else:
            # 否则，将参数名添加到参数列表中
            params.append(str(parameter))
    return params

# 定义 all_forward_schemas 函数，返回所有 TorchScript 前向操作的模式
def all_forward_schemas() -> Dict[str, _TorchSchema]:
    """Returns schemas for all TorchScript forward ops."""
    # 获取所有 TorchScript 操作的模式，并创建 _TorchSchema 对象列表
    torch_schemas = [_TorchSchema(s) for s in _C._jit_get_all_schemas()]
    # 返回以模式名称为键，_TorchSchema 对象为值的字典，不包含后向操作
    return {schema.name: schema for schema in torch_schemas if not schema.is_backward()}

# 定义 all_symbolics_schemas 函数，返回所有支持的 ONNX 操作的模式
def all_symbolics_schemas() -> Dict[str, _TorchSchema]:
    """Returns schemas for all onnx supported ops."""
    # 创建一个空字典用于存储 symbolics 模式的 _TorchSchema 对象
    symbolics_schemas = {}
    # 遍历注册表中所有函数的名称
    for name in registration.registry.all_functions():
        # 获取函数名称对应的函数组
        func_group = registration.registry.get_function_group(name)
        # 确保函数组不为空
        assert func_group is not None
        
        # 创建一个 TorchSchema 对象，用于表示符号化模式
        symbolics_schema = _TorchSchema(name)
        
        # 尝试从函数组中获取指定的函数，此处使用 ONNX_MAX_OPSET 对应的函数
        func = func_group.get(_constants.ONNX_MAX_OPSET)
        if func is not None:
            # 如果找到了对应的函数，则设置符号化模式的参数数量和支持的 opset 范围
            symbolics_schema.arguments = _symbolic_argument_count(func)
            symbolics_schema.opsets = list(
                range(func_group.get_min_supported(), _constants.ONNX_MAX_OPSET + 1)
            )
        else:
            # 如果未找到对应 ONNX_MAX_OPSET 的函数，只支持 opset 小于 9 的情况
            func = func_group.get(7)
            symbolics_schema.arguments = _symbolic_argument_count(func)
            symbolics_schema.opsets = list(range(7, _constants.ONNX_BASE_OPSET))
        
        # 将构建好的 symbolics_schema 对象存入 symbolics_schemas 字典中
        symbolics_schemas[name] = symbolics_schema

    # 返回所有函数名称及其对应的符号化模式字典
    return symbolics_schemas
```