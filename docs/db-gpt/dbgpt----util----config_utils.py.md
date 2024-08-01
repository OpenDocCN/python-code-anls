# `.\DB-GPT-src\dbgpt\util\config_utils.py`

```py
import os  # 导入操作系统相关的模块
from functools import cache  # 导入 functools 模块中的 cache 装饰器
from typing import Any, Dict, Optional, cast  # 导入类型提示相关的模块

class AppConfig:
    def __init__(self, configs: Optional[Dict[str, Any]] = None) -> None:
        self.configs = configs or {}  # 初始化 AppConfig 实例时，如果 configs 为 None 则设为一个空字典

    def set(self, key: str, value: Any, overwrite: bool = False) -> None:
        """Set config value by key
        
        Args:
            key (str): The key of config
            value (Any): The value of config
            overwrite (bool, optional): Whether to overwrite the value if key exists. Defaults to False.
        """
        if key in self.configs and not overwrite:
            raise KeyError(f"Config key {key} already exists")  # 如果 key 已存在且不允许覆盖，则抛出 KeyError
        self.configs[key] = value  # 设置 key 对应的配置值为 value

    def get(self, key, default: Optional[Any] = None) -> Any:
        """Get config value by key
        
        Args:
            key (str): The key of config
            default (Optional[Any], optional): The default value if key not found. Defaults to None.
        
        Returns:
            Any: The value corresponding to the key in configs dictionary, or default if key is not found
        """
        return self.configs.get(key, default)  # 返回 key 对应的配置值，如果 key 不存在则返回 default

    @cache
    def get_all_by_prefix(self, prefix) -> Dict[str, Any]:
        """Get all config values by prefix
        
        Args:
            prefix (str): The prefix of config
        
        Returns:
            Dict[str, Any]: A dictionary containing all key-value pairs from configs that start with the given prefix
        """
        return {k: v for k, v in self.configs.items() if k.startswith(prefix)}  # 返回所有以给定前缀开头的配置项及其对应的值

    def get_current_lang(self, default: Optional[str] = None) -> str:
        """Get current language
        
        Args:
            default (Optional[str], optional): The default language if not found. Defaults to None.
        
        Returns:
            str: The language of user running environment
        """
        env_lang = (
            "zh"  # 如果系统环境变量中的 LANG 存在且以 "zh" 开头，则设定 env_lang 为 "zh"
            if os.getenv("LANG") and cast(str, os.getenv("LANG")).startswith("zh")
            else default  # 否则设定为默认值 default
        )
        return self.get("dbgpt.app.global.language", env_lang)  # 返回配置中指定的语言或者根据环境设置的语言
```