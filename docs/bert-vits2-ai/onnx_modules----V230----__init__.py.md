# `Bert-VITS2\onnx_modules\V230\__init__.py`

```
# 从当前目录的text文件夹中导入symbols变量
from .text.symbols import symbols
# 从当前目录的models_onnx文件中导入SynthesizerTrn类
from .models_onnx import SynthesizerTrn
# 定义当前模块中可以被导入的变量和类
__all__ = ["symbols", "SynthesizerTrn"]
```