# `Bert-VITS2\onnx_modules\V220\__init__.py`

```

# 从当前目录下的text文件夹中的symbols模块中导入symbols变量
from .text.symbols import symbols
# 从当前目录下的models_onnx模块中导入SynthesizerTrn类
from .models_onnx import SynthesizerTrn
# 定义__all__变量，包含symbols和SynthesizerTrn，用于在其他模块中导入时指定导入的内容
__all__ = ["symbols", "SynthesizerTrn"]

```