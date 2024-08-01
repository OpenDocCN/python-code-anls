# `.\DB-GPT-src\dbgpt\serve\core\config.py`

```py
from dataclasses import dataclass  # 导入 dataclass 模块，用于创建数据类

from dbgpt.component import AppConfig  # 导入 AppConfig 类，用于应用配置
from dbgpt.util import BaseParameters  # 导入 BaseParameters 类，用于基础参数

@dataclass
class BaseServeConfig(BaseParameters):
    """Base configuration class for serve"""
    
    @classmethod
    def from_app_config(cls, config: AppConfig, config_prefix: str):
        """Create a configuration object from a dictionary
        
        Args:
            config (AppConfig): Application configuration object
            config_prefix (str): Prefix for configuration keys
        """
        global_prefix = "dbgpt.app.global."
        # 获取以 global_prefix 开头的全局配置字典
        global_dict = config.get_all_by_prefix(global_prefix)
        # 获取以 config_prefix 开头的配置字典
        config_dict = config.get_all_by_prefix(config_prefix)
        
        # 移除配置字典中的前缀
        config_dict = {
            k[len(config_prefix):]: v
            for k, v in config_dict.items()
            if k.startswith(config_prefix)
        }
        
        # 将全局配置中不存在于 config_dict 中的键值对添加到 config_dict 中
        for k, v in global_dict.items():
            if k not in config_dict and k[len(global_prefix):] in cls().__dict__:
                config_dict[k[len(global_prefix):]] = v
        
        # 使用解析后的 config_dict 参数初始化当前类的实例，并返回
        return cls(**config_dict)
```