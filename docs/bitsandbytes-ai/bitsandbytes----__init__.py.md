# `bitsandbytes\bitsandbytes\__init__.py`

```py
# 导入相关模块和包
from . import cuda_setup, research, utils
from .autograd._functions import (
    MatmulLtState,
    bmm_cublas,
    matmul,
    matmul_4bit,
    matmul_cublas,
    mm_cublas,
)
from .cextension import COMPILED_WITH_CUDA
from .nn import modules

# 如果编译时包含了 CUDA，则导入 adam 优化器
if COMPILED_WITH_CUDA:
    from .optim import adam

# 隐藏指定模块或类的文档
__pdoc__ = {
    "libbitsandbytes": False,
    "optim.optimizer.Optimizer8bit": False,
    "optim.optimizer.MockArgs": False,
}

# 定义当前版本号
__version__ = "0.43.0.dev"

# 定义包的 GitHub 地址
PACKAGE_GITHUB_URL = "https://github.com/TimDettmers/bitsandbytes"
```