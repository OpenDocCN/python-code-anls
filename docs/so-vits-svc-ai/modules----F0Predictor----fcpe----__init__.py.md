# `so-vits-svc\modules\F0Predictor\fcpe\__init__.py`

```py
# 从model模块中导入FCPEInfer类，禁止Flake8检查F401错误
from .model import FCPEInfer  # noqa: F401
# 从nvSTFT模块中导入STFT类，禁止Flake8检查F401错误
from .nvSTFT import STFT  # noqa: F401
# 从pcmer模块中导入PCmer类，禁止Flake8检查F401错误
from .pcmer import PCmer  # noqa: F401
```