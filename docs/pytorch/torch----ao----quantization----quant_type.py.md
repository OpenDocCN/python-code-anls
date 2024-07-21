# `.\pytorch\torch\ao\quantization\quant_type.py`

```py
import enum  # 导入枚举模块

__all__ = [
    "QuantType",  # 在模块中暴露的公共接口列表，仅包含 QuantType
]

# Quantization type (dynamic quantization, static quantization).
# Should match the c++ enum in quantization_type.h
class QuantType(enum.IntEnum):
    DYNAMIC = 0  # 动态量化类型，对应整数值 0
    STATIC = 1   # 静态量化类型，对应整数值 1
    QAT = 2      # QAT（训练后量化）类型，对应整数值 2
    WEIGHT_ONLY = 3  # 仅权重量化类型，对应整数值 3

# Map from QuantType enum values to corresponding string representations
_quant_type_to_str = {
    QuantType.STATIC: "static",          # 静态量化映射到字符串 "static"
    QuantType.DYNAMIC: "dynamic",        # 动态量化映射到字符串 "dynamic"
    QuantType.QAT: "qat",                # QAT 类型映射到字符串 "qat"
    QuantType.WEIGHT_ONLY: "weight_only",  # 仅权重量化映射到字符串 "weight_only"
}

# TODO: make this private
def _get_quant_type_to_str(quant_type: QuantType) -> str:
    # 返回给定量化类型对应的字符串表示
    return _quant_type_to_str[quant_type]

def _quant_type_from_str(name: str) -> QuantType:
    # 根据字符串名称获取对应的 QuantType 枚举值
    for quant_type, s in _quant_type_to_str.items():
        if name == s:
            return quant_type
    # 如果未找到匹配的量化类型名称，则引发异常
    raise ValueError(f"Unknown QuantType name '{name}'")
```