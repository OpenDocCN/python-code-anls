# `.\pytorch\torch\_C\_onnx.pyi`

```
# 在 torch/csrc/onnx/init.cpp 中定义了以下内容

# 导入枚举(Enum)类，用于定义不同的数据类型
from enum import Enum

# 定义一个全局变量，用于表示生产者版本的字符串
PRODUCER_VERSION: str

# 定义了一个枚举类 TensorProtoDataType，用于表示 ONNX 中的数据类型
class TensorProtoDataType(Enum):
    UNDEFINED = ...  # 未定义的数据类型
    FLOAT = ...       # 单精度浮点数
    UINT8 = ...       # 无符号 8 位整数
    INT8 = ...        # 有符号 8 位整数
    UINT16 = ...      # 无符号 16 位整数
    INT16 = ...       # 有符号 16 位整数
    INT32 = ...       # 有符号 32 位整数
    INT64 = ...       # 有符号 64 位整数
    STRING = ...      # 字符串
    BOOL = ...        # 布尔值
    FLOAT16 = ...     # 半精度浮点数
    DOUBLE = ...      # 双精度浮点数
    UINT32 = ...      # 无符号 32 位整数
    UINT64 = ...      # 无符号 64 位整数
    COMPLEX64 = ...   # 复数，32 位浮点数
    COMPLEX128 = ...  # 复数，64 位浮点数
    BFLOAT16 = ...    # Brain Floating Point 16，用于深度学习
    FLOAT8E5M2 = ...  # 浮点数，8 位小数，5 位整数，2 位尾数
    FLOAT8E4M3FN = ...# 浮点数，8 位小数，4 位整数，3 位尾数
    FLOAT8E5M2FNUZ = ...# 未压缩数据浮点数，8 位小数，5 位整数，2 位尾数
    FLOAT8E4M3FNUZ = ...# 未压缩数据浮点数，8 位小数，4 位整数，3 位尾数

# 定义了一个枚举类 OperatorExportTypes，用于表示导出操作的类型
class OperatorExportTypes(Enum):
    ONNX = ...                # 基本的 ONNX 导出类型
    ONNX_ATEN = ...           # ATen 操作的 ONNX 导出类型
    ONNX_ATEN_FALLBACK = ...  # ATen 操作的后备 ONNX 导出类型
    ONNX_FALLTHROUGH = ...    # ONNX 的穿透导出类型

# 定义了一个枚举类 TrainingMode，用于表示训练模式
class TrainingMode(Enum):
    EVAL = ...       # 评估模式
    PRESERVE = ...   # 保留模式
    TRAINING = ...   # 训练模式
```