# `.\graphrag\graphrag\config\environment_reader.py`

```py
# 版权声明和许可证声明
# Licensed under the MIT License

# 导入必要的模块和类
"""A configuration reader utility class."""
from collections.abc import Callable  # 导入Callable类，用于类型注解
from contextlib import contextmanager  # 导入contextmanager，用于创建上下文管理器
from enum import Enum  # 导入Enum枚举类，用于处理枚举类型数据
from typing import Any, TypeVar  # 导入类型相关的模块和类

from environs import Env  # 导入Env类，用于处理环境变量

T = TypeVar("T")  # 定义类型变量T，用于泛型类型注解

KeyValue = str | Enum  # 定义KeyValue类型，可以是str或Enum类型
EnvKeySet = str | list[str]  # 定义EnvKeySet类型，可以是str或str列表


def read_key(value: KeyValue) -> str:
    """Read a key value."""
    # 如果value不是str类型，则返回其枚举值的小写形式
    if not isinstance(value, str):
        return value.value.lower()
    # 否则返回value的小写形式
    return value.lower()


class EnvironmentReader:
    """A configuration reader utility class."""

    _env: Env  # 类属性_env，类型为Env
    _config_stack: list[dict]  # 类属性_config_stack，类型为字典列表

    def __init__(self, env: Env):
        """Initialize EnvironmentReader with an Env instance."""
        self._env = env  # 初始化_env属性为传入的Env实例
        self._config_stack = []  # 初始化_config_stack为空列表

    @property
    def env(self):
        """Get the environment object."""
        return self._env  # 返回_env属性，即Env实例

    def _read_env(
        self, env_key: str | list[str], default_value: T, read: Callable[[str, T], T]
    ) -> T | None:
        """Read from environment variables or return default_value."""
        # 如果env_key是str类型，则转换为单元素列表
        if isinstance(env_key, str):
            env_key = [env_key]

        # 遍历env_key列表
        for k in env_key:
            result = read(k.upper(), default_value)  # 调用read函数读取环境变量值
            # 如果result不等于default_value，则返回result
            if result is not default_value:
                return result

        return default_value  # 返回default_value

    def envvar_prefix(self, prefix: KeyValue):
        """Set the environment variable prefix."""
        prefix = read_key(prefix)  # 调用read_key函数处理prefix
        prefix = f"{prefix}_".upper()  # 将prefix转换为大写并添加下划线后缀
        return self._env.prefixed(prefix)  # 返回带有前缀的Env实例

    def use(self, value: Any | None):
        """Create a context manager to push the value into the config_stack."""

        @contextmanager
        def config_context():
            self._config_stack.append(value or {})  # 将value或空字典压入_config_stack
            try:
                yield  # 执行上下文管理器的主体部分
            finally:
                self._config_stack.pop()  # 弹出_config_stack中的最后一个元素

        return config_context()  # 返回config_context上下文管理器

    @property
    def section(self) -> dict:
        """Get the current section."""
        return self._config_stack[-1] if self._config_stack else {}  # 返回_config_stack中的最后一个字典或空字典

    def str(
        self,
        key: KeyValue,
        env_key: EnvKeySet | None = None,
        default_value: str | None = None,
    ) -> str | None:
        """Read a configuration value as a string."""
        key = read_key(key)  # 调用read_key函数处理key
        # 如果当前section非空且key在section中，则返回section[key]
        if self.section and key in self.section:
            return self.section[key]

        # 否则调用_read_env函数读取环境变量值
        return self._read_env(
            env_key or key, default_value, (lambda k, dv: self._env(k, dv))
        )

    def int(
        self,
        key: KeyValue,
        env_key: EnvKeySet | None = None,
        default_value: int | None = None,
    ) -> int | None:
        """Read a configuration value as an integer."""
        key = read_key(key)  # 调用read_key函数处理key
        # 如果当前section非空且key在section中，则将section[key]转换为整数并返回
        if self.section and key in self.section:
            return int(self.section[key])
        # 否则调用_read_env函数读取环境变量值，并将其转换为整数返回
        return self._read_env(
            env_key or key, default_value, lambda k, dv: self._env.int(k, dv)
        )

    def bool(
        self,
        key: KeyValue,
        env_key: EnvKeySet | None = None,
        default_value: bool | None = None,
    def bool(
        self,
        key: KeyValue,
        env_key: EnvKeySet | None = None,
        default_value: bool | None = None,
    ) -> bool | None:
        """Read a boolean configuration value."""
        # 调用 read_key 函数，获取配置键的值
        key = read_key(key)
        
        # 检查是否存在配置节，并且键在配置节中
        if self.section and key in self.section:
            # 返回配置键对应的布尔值
            return bool(self.section[key])

        # 调用 _read_env 方法，尝试从环境变量中读取布尔值
        return self._read_env(
            env_key or key, default_value, lambda k, dv: self._env.bool(k, dv)
        )

    def float(
        self,
        key: KeyValue,
        env_key: EnvKeySet | None = None,
        default_value: float | None = None,
    ) -> float | None:
        """Read a float configuration value."""
        # 调用 read_key 函数，获取配置键的值
        key = read_key(key)
        
        # 检查是否存在配置节，并且键在配置节中
        if self.section and key in self.section:
            # 返回配置键对应的浮点数值
            return float(self.section[key])
        
        # 调用 _read_env 方法，尝试从环境变量中读取浮点数值
        return self._read_env(
            env_key or key, default_value, lambda k, dv: self._env.float(k, dv)
        )

    def list(
        self,
        key: KeyValue,
        env_key: EnvKeySet | None = None,
        default_value: list | None = None,
    ) -> list | None:
        """Parse an list configuration value."""
        # 调用 read_key 函数，获取配置键的值
        key = read_key(key)
        
        result = None
        # 检查是否存在配置节，并且键在配置节中
        if self.section and key in self.section:
            result = self.section[key]
            # 如果结果是列表，则直接返回
            if isinstance(result, list):
                return result
        
        # 如果结果仍为 None，则调用 str 方法尝试获取字符串格式的值
        if result is None:
            result = self.str(key, env_key)
        
        # 如果结果存在，则以逗号分隔并去除空格形成列表返回
        if result:
            return [s.strip() for s in result.split(",")]
        
        # 返回默认值
        return default_value
```