# `.\diffusers\utils\constants.py`

```py
# 版权所有 2024 HuggingFace Inc.团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非根据适用法律或书面协议另有约定，
# 否则根据许可证分发的软件是按“原样”基础提供的，
# 不提供任何形式的明示或暗示的担保或条件。
# 有关许可证所涉及的特定权限和限制，请参见许可证。
import importlib  # 导入动态导入模块的库
import os  # 导入操作系统功能模块

from huggingface_hub.constants import HF_HOME  # 从 huggingface_hub.constants 导入 HF_HOME 常量
from packaging import version  # 从 packaging 导入版本管理模块

from ..dependency_versions_check import dep_version_check  # 从父目录导入版本检查函数
from .import_utils import ENV_VARS_TRUE_VALUES, is_peft_available, is_transformers_available  # 从当前目录导入环境变量和库可用性检查函数

MIN_PEFT_VERSION = "0.6.0"  # 定义 PEFT 库的最低版本要求
MIN_TRANSFORMERS_VERSION = "4.34.0"  # 定义 Transformers 库的最低版本要求
_CHECK_PEFT = os.environ.get("_CHECK_PEFT", "1") in ENV_VARS_TRUE_VALUES  # 检查环境变量以确定是否进行 PEFT 检查

CONFIG_NAME = "config.json"  # 定义配置文件的名称
WEIGHTS_NAME = "diffusion_pytorch_model.bin"  # 定义 PyTorch 模型权重文件的名称
WEIGHTS_INDEX_NAME = "diffusion_pytorch_model.bin.index.json"  # 定义权重索引文件的名称
FLAX_WEIGHTS_NAME = "diffusion_flax_model.msgpack"  # 定义 Flax 模型权重文件的名称
ONNX_WEIGHTS_NAME = "model.onnx"  # 定义 ONNX 模型权重文件的名称
SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"  # 定义 Safetensors 模型权重文件的名称
SAFE_WEIGHTS_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"  # 定义 Safetensors 权重索引文件的名称
SAFETENSORS_FILE_EXTENSION = "safetensors"  # 定义 Safetensors 文件扩展名
ONNX_EXTERNAL_WEIGHTS_NAME = "weights.pb"  # 定义外部 ONNX 权重文件的名称
HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")  # 获取 Hugging Face 端点环境变量，默认值为 https://huggingface.co
DIFFUSERS_DYNAMIC_MODULE_NAME = "diffusers_modules"  # 定义 Diffusers 动态模块名称
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(HF_HOME, "modules"))  # 获取 HF 模块缓存路径，默认为 HF_HOME/modules
DEPRECATED_REVISION_ARGS = ["fp16", "non-ema"]  # 定义不推荐使用的修订参数列表

# 以下条件为真，表示当前 PEFT 和 Transformers 版本与 PEFT 后端兼容。
# 如果可用的库版本正确，将自动回退到 PEFT 后端。
# 对于 PEFT，它必须大于或等于 0.6.0，对于 Transformers，它必须大于或等于 4.34.0。
_required_peft_version = is_peft_available() and version.parse(  # 检查 PEFT 是否可用且版本符合要求
    version.parse(importlib.metadata.version("peft")).base_version  # 获取 PEFT 的版本并进行解析
) >= version.parse(MIN_PEFT_VERSION)  # 比较 PEFT 版本与最低要求
_required_transformers_version = is_transformers_available() and version.parse(  # 检查 Transformers 是否可用且版本符合要求
    version.parse(importlib.metadata.version("transformers")).base_version  # 获取 Transformers 的版本并进行解析
) >= version.parse(MIN_TRANSFORMERS_VERSION)  # 比较 Transformers 版本与最低要求

USE_PEFT_BACKEND = _required_peft_version and _required_transformers_version  # 确定是否使用 PEFT 后端

if USE_PEFT_BACKEND and _CHECK_PEFT:  # 如果满足条件则进行版本检查
    dep_version_check("peft")  # 执行 PEFT 库的依赖版本检查
```