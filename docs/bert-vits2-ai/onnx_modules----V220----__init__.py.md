# `d:/src/tocomm/Bert-VITS2\onnx_modules\V220\__init__.py`

```
# 从text.symbols模块中导入symbols变量
from .text.symbols import symbols
# 从models_onnx模块中导入SynthesizerTrn类
from .models_onnx import SynthesizerTrn
# 定义__all__列表，包含symbols和SynthesizerTrn，用于模块导入时指定可导入的内容
__all__ = ["symbols", "SynthesizerTrn"]
```