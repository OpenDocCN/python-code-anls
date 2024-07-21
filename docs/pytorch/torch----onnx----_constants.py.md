# `.\pytorch\torch\onnx\_constants.py`

```py
# 定义 ON`
# 定义用于 ONNX 的常量值

# ONNX 模型协议文件名常量
ONNX_ARCHIVE_MODEL_PROTO_NAME = "__MODEL_PROTO"

# ONNX 支持的基础操作集版本号
ONNX_BASE_OPSET = 9
# ONNX 支持的最低操作集版本号
ONNX_MIN_OPSET = 7
# ONNX 支持的最高操作集版本号
ONNX_MAX_OPSET = 20
# 使用 TorchScript 导出的最高操作集版本号
ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET = 20
# 默认的 ONNX 操作集版本号
# 由 tools/onnx/update_default_opset_version.py 生成
ONNX_DEFAULT_OPSET = 17
# 支持常量折叠的最低操作集版本号
ONNX_CONSTANT_FOLDING_MIN_OPSET = 9

# PyTorch GitHub 问题报告链接
PYTORCH_GITHUB_ISSUES_URL = "https://github.com/pytorch/pytorch/issues"

# 64位整数的最大值
INT64_MAX = 9223372036854775807
# 32位整数的最大值
INT32_MAX = 2147483647
# 16位整数的最大值
INT16_MAX = 32767
# 8位整数的最大值
INT8 = 255

# 定义不同整数类型的最小值常量
INT64_MIN = -9223372036854775808
INT32_MIN = -2147483648
INT16_MIN = -32768
INT8_MIN = -128
UINT8_MIN = 0
```