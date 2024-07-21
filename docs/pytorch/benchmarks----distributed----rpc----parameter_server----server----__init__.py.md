# `.\pytorch\benchmarks\distributed\rpc\parameter_server\server\__init__.py`

```py
# 导入自定义模块中的 AverageBatchParameterServer 和 AverageParameterServer 类
from .server import AverageBatchParameterServer, AverageParameterServer

# 创建一个字典，映射服务器类型的字符串到相应的服务器类
server_map = {
    "AverageParameterServer": AverageParameterServer,  # 将字符串 "AverageParameterServer" 映射到 AverageParameterServer 类
    "AverageBatchParameterServer": AverageBatchParameterServer,  # 将字符串 "AverageBatchParameterServer" 映射到 AverageBatchParameterServer 类
}
```