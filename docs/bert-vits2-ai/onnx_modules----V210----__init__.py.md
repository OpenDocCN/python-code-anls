# `d:/src/tocomm/Bert-VITS2\onnx_modules\V210\__init__.py`

```
# 从text.symbols模块中导入symbols变量
from .text.symbols import symbols
# 从models_onnx模块中导入SynthesizerTrn类
from .models_onnx import SynthesizerTrn

# 将symbols和SynthesizerTrn添加到__all__列表中，表示它们是该模块的公开接口
__all__ = ["symbols", "SynthesizerTrn"]
```