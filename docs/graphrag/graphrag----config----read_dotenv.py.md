# `.\graphrag\graphrag\config\read_dotenv.py`

```py
# 版权所有 (c) 2024 微软公司。
# 根据 MIT 许可证授权

"""一个包含 read_dotenv 实用程序的模块。"""

# 导入日志记录、操作系统和路径处理相关的模块
import logging
import os
from pathlib import Path

# 从 dotenv 模块中导入 dotenv_values 函数
from dotenv import dotenv_values

# 获取记录器对象
log = logging.getLogger(__name__)


# 定义 read_dotenv 函数，接受一个字符串类型的参数 root，返回空值
def read_dotenv(root: str) -> None:
    """读取给定根路径下的 .env 文件。"""
    # 将 .env 文件路径构建为路径对象
    env_path = Path(root) / ".env"
    # 如果 .env 文件存在
    if env_path.exists():
        # 记录信息，指示正在加载管道的 .env 文件
        log.info("Loading pipeline .env file")
        # 从 .env 文件中获取配置信息
        env_config = dotenv_values(f"{env_path}")
        # 遍历配置信息的键值对
        for key, value in env_config.items():
            # 如果当前键不在操作系统的环境变量中
            if key not in os.environ:
                # 将键值对添加到操作系统的环境变量中，如果值为空字符串则设为""
                os.environ[key] = value or ""
    else:
        # 记录信息，指示在根路径下找不到 .env 文件
        log.info("No .env file found at %s", root)
```