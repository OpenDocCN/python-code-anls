# `.\pytorch\torch\distributed\elastic\utils\api.py`

```
# 指定 Python 解释器路径
#!/usr/bin/env python3

# 版权声明：版权归 Facebook 及其关联公司所有。
# 保留所有权利。
#
# 此源代码根据 BSD 风格许可证授权，许可证文件可在源代码根目录中的 LICENSE 文件中找到。

# 导入标准库模块
import os
import socket
from string import Template
from typing import Any, List

# 定义函数，根据环境变量名称获取其值，如果找不到则抛出 ValueError 异常
def get_env_variable_or_raise(env_name: str) -> str:
    """
    Tries to retrieve environment variable. Raises ``ValueError``
    if no environment variable found.

    Args:
        env_name (str): Name of the env variable
    """
    # 获取环境变量的值，如果不存在则为 None
    value = os.environ.get(env_name, None)
    if value is None:
        # 如果值不存在，抛出异常，提示缺少对应的环境变量
        msg = f"Environment variable {env_name} expected, but not set"
        raise ValueError(msg)
    return value

# 定义函数，创建并返回一个带有端口的 socket 对象
def get_socket_with_port() -> socket.socket:
    # 获取地址信息列表
    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )
    for addr in addrs:
        # 解析地址信息
        family, type, proto, _, _ = addr
        # 创建 socket 对象
        s = socket.socket(family, type, proto)
        try:
            # 尝试绑定到 localhost 的随机端口
            s.bind(("localhost", 0))
            s.listen(0)
            return s
        except OSError as e:
            # 如果绑定失败，则关闭 socket 对象，继续尝试下一个地址
            s.close()
    # 如果所有地址都尝试失败，则抛出运行时异常
    raise RuntimeError("Failed to create a socket")

# 定义一个类 macros，用于定义简单的宏以便在 caffe2.distributed.launch 命令行参数中进行替换
class macros:
    """
    Defines simple macros for caffe2.distributed.launch cmd args substitution
    """

    # 定义一个类属性 local_rank，其值为 "${local_rank}"
    local_rank = "${local_rank}"

    # 定义静态方法 substitute，用于替换参数列表中的字符串中的 "${local_rank}" 为给定的 local_rank 值
    @staticmethod
    def substitute(args: List[Any], local_rank: str) -> List[str]:
        args_sub = []
        for arg in args:
            if isinstance(arg, str):
                # 如果参数是字符串类型，则进行模板替换，将 "${local_rank}" 替换为实际的 local_rank 值
                sub = Template(arg).safe_substitute(local_rank=local_rank)
                args_sub.append(sub)
            else:
                # 如果参数不是字符串类型，则直接添加到替换后的参数列表中
                args_sub.append(arg)
        return args_sub
```