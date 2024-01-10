# `Bert-VITS2\onnx_modules\V200\__init__.py`

```
# 从当前目录的text模块中导入symbols变量
from .text.symbols import symbols
# 从当前目录的models_onnx模块中导入SynthesizerTrn类
from .models_onnx import SynthesizerTrn
# 定义当前模块对外暴露的变量和类
__all__ = ["symbols", "SynthesizerTrn"]
```