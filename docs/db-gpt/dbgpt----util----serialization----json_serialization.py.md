# `.\DB-GPT-src\dbgpt\util\serialization\json_serialization.py`

```py
import json  # 导入 json 库，用于 JSON 数据的序列化和反序列化
from abc import ABC, abstractmethod  # 导入 ABC 抽象基类和抽象方法装饰器
from typing import Dict, Type  # 导入类型提示，用于声明字典和类型

from dbgpt.core.awel.flow import ResourceCategory, register_resource  # 导入注册资源函数和资源类别枚举
from dbgpt.core.interface.serialization import Serializable, Serializer  # 导入序列化接口和序列化器接口
from dbgpt.util.i18n_utils import _  # 导入国际化翻译函数

JSON_ENCODING = "utf-8"  # JSON 编码格式设为 utf-8


class JsonSerializable(Serializable, ABC):
    @abstractmethod
    def to_dict(self) -> Dict:
        """Return the dict of current serializable object"""
        # 抽象方法，子类需实现将当前对象转换为字典的功能


    def serialize(self) -> bytes:
        """Convert the object into bytes for storage or transmission."""
        return json.dumps(self.to_dict(), ensure_ascii=False).encode(JSON_ENCODING)
        # 将对象转换为 JSON 格式的字节流，用于存储或传输


@register_resource(
    label=_("Json Serializer"),
    name="json_serializer",
    category=ResourceCategory.SERIALIZER,
    description=_("The serializer for serializing data with json format."),
)
class JsonSerializer(Serializer):
    """The serializer abstract class for serializing cache keys and values."""

    def serialize(self, obj: Serializable) -> bytes:
        """Serialize a cache object.

        Args:
            obj (Serializable): The object to serialize
        """
        return json.dumps(obj.to_dict(), ensure_ascii=False).encode(JSON_ENCODING)
        # 序列化缓存对象为 JSON 格式的字节流

    def deserialize(self, data: bytes, cls: Type[Serializable]) -> Serializable:
        """Deserialize data back into a cache object of the specified type.

        Args:
            data (bytes): The byte array to deserialize
            cls (Type[Serializable]): The type of current object

        Returns:
            Serializable: The serializable object
        """
        # Convert bytes back to JSON and then to the specified class
        json_data = json.loads(data.decode(JSON_ENCODING))
        # 假定 cls 类型有一个接受字典参数的 __init__ 方法
        obj = cls(**json_data)
        obj.set_serializer(self)
        return obj
        # 反序列化数据为指定类型的缓存对象，并设置序列化器
```