# `.\pytorch\torch\backends\quantized\__init__.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和类型
import sys  # 导入 sys 模块
import types  # 导入 types 模块
from typing import List  # 从 typing 模块导入 List 类型

import torch  # 导入 torch 库


# This function should correspond to the enums present in c10/core/QEngine.h
# 根据输入的 qengine 字符串返回对应的整数 ID
def _get_qengine_id(qengine: str) -> int:
    if qengine == "none" or qengine == "" or qengine is None:
        ret = 0  # 如果 qengine 是 "none"、空字符串或者 None，返回 0
    elif qengine == "fbgemm":
        ret = 1  # 如果 qengine 是 "fbgemm"，返回 1
    elif qengine == "qnnpack":
        ret = 2  # 如果 qengine 是 "qnnpack"，返回 2
    elif qengine == "onednn":
        ret = 3  # 如果 qengine 是 "onednn"，返回 3
    elif qengine == "x86":
        ret = 4  # 如果 qengine 是 "x86"，返回 4
    else:
        ret = -1  # 如果 qengine 不匹配任何已知值，抛出错误
        raise RuntimeError(f"{qengine} is not a valid value for quantized engine")
    return ret


# This function should correspond to the enums present in c10/core/QEngine.h
# 根据整数 ID 返回对应的 qengine 字符串
def _get_qengine_str(qengine: int) -> str:
    all_engines = {0: "none", 1: "fbgemm", 2: "qnnpack", 3: "onednn", 4: "x86"}
    return all_engines.get(qengine, "*undefined")  # 返回对应 ID 的 qengine 字符串，若无匹配返回 "*undefined"


# Class representing the descriptor for the _QEngineProp attribute
class _QEngineProp:
    # Getter method for the engine property
    def __get__(self, obj, objtype) -> str:
        return _get_qengine_str(torch._C._get_qengine())  # 获取当前 quantized engine 的字符串表示

    # Setter method for the engine property
    def __set__(self, obj, val: str) -> None:
        torch._C._set_qengine(_get_qengine_id(val))  # 设置 quantized engine 的值


# Class representing the descriptor for the _SupportedQEnginesProp attribute
class _SupportedQEnginesProp:
    # Getter method for the supported_engines property
    def __get__(self, obj, objtype) -> List[str]:
        qengines = torch._C._supported_qengines()  # 获取所有支持的 quantized engines 列表
        return [_get_qengine_str(qe) for qe in qengines]  # 返回这些 engines 的字符串表示列表

    # Setter method for the supported_engines property
    def __set__(self, obj, val) -> None:
        raise RuntimeError("Assignment not supported")  # 不支持对 supported_engines 进行赋值操作的错误提示


# Class representing the module type for QuantizedEngine
class QuantizedEngine(types.ModuleType):
    # Constructor for QuantizedEngine
    def __init__(self, m, name):
        super().__init__(name)
        self.m = m  # 初始化父模块

    # Method to get attributes dynamically
    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)  # 动态获取属性

    engine = _QEngineProp()  # 定义 engine 属性，使用 _QEngineProp 描述符
    supported_engines = _SupportedQEnginesProp()  # 定义 supported_engines 属性，使用 _SupportedQEnginesProp 描述符


# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
# 用 QuantizedEngine 替换当前模块，以支持动态属性访问
sys.modules[__name__] = QuantizedEngine(sys.modules[__name__], __name__)
engine: str  # engine 属性的类型声明
supported_engines: List[str]  # supported_engines 属性的类型声明
```