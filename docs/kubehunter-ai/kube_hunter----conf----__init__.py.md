# `kubehunter\kube_hunter\conf\__init__.py`

```
# 从 kube_hunter.conf.parser 模块中导入 parse_args 函数
from kube_hunter.conf.parser import parse_args
# 从 kube_hunter.conf.logging 模块中导入 setup_logger 函数
from kube_hunter.conf.logging import setup_logger

# 调用 parse_args 函数解析配置参数，并将结果赋值给 config 变量
config = parse_args()
# 调用 setup_logger 函数设置日志记录器，使用 config.log 参数
setup_logger(config.log)

# 将 config 变量添加到 __all__ 列表中
__all__ = [config]
```