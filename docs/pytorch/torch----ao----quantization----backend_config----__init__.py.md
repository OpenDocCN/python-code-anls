# `.\pytorch\torch\ao\quantization\backend_config\__init__.py`

```
# 导入所需的模块和函数，包括从其他文件导入的类和函数
from .backend_config import BackendConfig, BackendPatternConfig, DTypeConfig, DTypeWithConstraints, ObservationType
from .fbgemm import get_fbgemm_backend_config
from .native import get_native_backend_config, get_native_backend_config_dict
from .qnnpack import get_qnnpack_backend_config
from .tensorrt import get_tensorrt_backend_config, get_tensorrt_backend_config_dict
from .executorch import get_executorch_backend_config
from .onednn import get_onednn_backend_config

# 定义一个列表，包含了所有公开的接口和类，便于模块外部使用
__all__ = [
    "get_fbgemm_backend_config",               # 获取 fbgemm 后端配置的函数
    "get_native_backend_config",               # 获取 native 后端配置的函数
    "get_native_backend_config_dict",          # 获取 native 后端配置字典的函数
    "get_qnnpack_backend_config",              # 获取 qnnpack 后端配置的函数
    "get_tensorrt_backend_config",             # 获取 tensorrt 后端配置的函数
    "get_tensorrt_backend_config_dict",        # 获取 tensorrt 后端配置字典的函数
    "get_executorch_backend_config",           # 获取 executorch 后端配置的函数
    "BackendConfig",                           # 后端配置类
    "BackendPatternConfig",                    # 后端模式配置类
    "DTypeConfig",                             # 数据类型配置类
    "DTypeWithConstraints",                    # 带约束的数据类型配置类
    "ObservationType",                         # 观察类型类
    "get_onednn_backend_config",               # 获取 onednn 后端配置的函数
]
```