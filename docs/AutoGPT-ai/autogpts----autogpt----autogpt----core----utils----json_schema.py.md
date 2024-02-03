# `.\AutoGPT\autogpts\autogpt\autogpt\core\utils\json_schema.py`

```py
# 导入必要的模块和类
import enum
from logging import Logger
from textwrap import indent
from typing import Literal, Optional

from jsonschema import Draft7Validator
from pydantic import BaseModel

# 定义 JSONSchema 类，用于表示 JSON Schema
class JSONSchema(BaseModel):
    # 定义 JSONSchema 类型枚举
    class Type(str, enum.Enum):
        STRING = "string"
        ARRAY = "array"
        OBJECT = "object"
        NUMBER = "number"
        INTEGER = "integer"
        BOOLEAN = "boolean"

    # JSONSchema 的描述信息
    description: Optional[str] = None
    # JSONSchema 的类型
    type: Optional[Type] = None
    # JSONSchema 的枚举值
    enum: Optional[list] = None
    # JSONSchema 是否必须
    required: bool = False
    # JSONSchema 的子项
    items: Optional["JSONSchema"] = None
    # JSONSchema 的属性
    properties: Optional[dict[str, "JSONSchema"]] = None
    # JSONSchema 的最小值
    minimum: Optional[int | float] = None
    # JSONSchema 的最大值
    maximum: Optional[int | float] = None
    # JSONSchema 的最小子项数
    minItems: Optional[int] = None
    # JSONSchema 的最大子项数
    maxItems: Optional[int] = None

    # 将 JSONSchema 转换为字典格式
    def to_dict(self) -> dict:
        # 初始化 schema 字典
        schema: dict = {
            "type": self.type.value if self.type else None,
            "description": self.description,
        }
        # 处理数组类型的 JSONSchema
        if self.type == "array":
            if self.items:
                schema["items"] = self.items.to_dict()
            schema["minItems"] = self.minItems
            schema["maxItems"] = self.maxItems
        # 处理对象类型的 JSONSchema
        elif self.type == "object":
            if self.properties:
                schema["properties"] = {
                    name: prop.to_dict() for name, prop in self.properties.items()
                }
                schema["required"] = [
                    name for name, prop in self.properties.items() if prop.required
                ]
        # 处理枚举类型的 JSONSchema
        elif self.enum:
            schema["enum"] = self.enum
        else:
            schema["minumum"] = self.minimum
            schema["maximum"] = self.maximum

        # 过滤掉值为 None 的键值对
        schema = {k: v for k, v in schema.items() if v is not None}

        return schema

    @staticmethod
    # 从字典对象创建 JSONSchema 对象
    def from_dict(schema: dict) -> "JSONSchema":
        # 返回一个 JSONSchema 对象，根据传入的字典对象设置相应属性
        return JSONSchema(
            description=schema.get("description"),
            type=schema["type"],
            enum=schema["enum"] if "enum" in schema else None,
            items=JSONSchema.from_dict(schema["items"]) if "items" in schema else None,
            properties=JSONSchema.parse_properties(schema)
            if schema["type"] == "object"
            else None,
            minimum=schema.get("minimum"),
            maximum=schema.get("maximum"),
            minItems=schema.get("minItems"),
            maxItems=schema.get("maxItems"),
        )

    # 解析 JSONSchema 中的属性
    @staticmethod
    def parse_properties(schema_node: dict) -> dict[str, "JSONSchema"]:
        # 如果 schema_node 中包含 properties 属性，则遍历创建 JSONSchema 对象
        properties = (
            {k: JSONSchema.from_dict(v) for k, v in schema_node["properties"].items()}
            if "properties" in schema_node
            else {}
        )
        # 如果 schema_node 中包含 required 属性，则设置对应属性的 required 值
        if "required" in schema_node:
            for k, v in properties.items():
                v.required = k in schema_node["required"]
        return properties

    # 验证对象是否符合 JSONSchema 规范
    def validate_object(
        self, object: object, logger: Logger
    ) -> tuple[Literal[True], None] | tuple[Literal[False], list]:
        """
        Validates a dictionary object against the JSONSchema.

        Params:
            object: The dictionary object to validate.
            schema (JSONSchema): The JSONSchema to validate against.

        Returns:
            tuple: A tuple where the first element is a boolean indicating whether the
                object is valid or not, and the second element is a list of errors found
                in the object, or None if the object is valid.
        """
        # 创建 Draft7Validator 对象，传入 JSONSchema 对象的字典表示
        validator = Draft7Validator(self.to_dict())

        # 如果存在错误，则返回错误列表
        if errors := sorted(validator.iter_errors(object), key=lambda e: e.path):
            return False, errors

        # 如果没有错误，则返回验证通过的信息
        return True, None
    # 将 JSONSchema 转换为 TypeScript 对象接口
    def to_typescript_object_interface(self, interface_name: str = "") -> str:
        # 如果 JSONSchema 类型不是 OBJECT，则抛出异常
        if self.type != JSONSchema.Type.OBJECT:
            raise NotImplementedError("Only `object` schemas are supported")

        # 如果存在属性
        if self.properties:
            # 初始化属性列表
            attributes: list[str] = []
            # 遍历属性字典
            for name, property in self.properties.items():
                # 如果属性有描述，则添加注释
                if property.description:
                    attributes.append(f"// {property.description}")
                # 添加属性名和 TypeScript 类型到属性列表
                attributes.append(f"{name}: {property.typescript_type};")
            # 将属性列表转换为字符串
            attributes_string = "\n".join(attributes)
        else:
            # 如果不存在属性，则设置默认属性字符串
            attributes_string = "[key: string]: any"

        # 返回 TypeScript 接口字符串，包括接口名和属性列表
        return (
            f"interface {interface_name} " if interface_name else ""
        ) + f"{{\n{indent(attributes_string, '  ')}\n}}"

    # 获取 JSONSchema 对应的 TypeScript 类型
    @property
    def typescript_type(self) -> str:
        # 根据 JSONSchema 类型返回对应的 TypeScript 类型
        if self.type == JSONSchema.Type.BOOLEAN:
            return "boolean"
        elif self.type in {JSONSchema.Type.INTEGER, JSONSchema.Type.NUMBER}:
            return "number"
        elif self.type == JSONSchema.Type.STRING:
            return "string"
        elif self.type == JSONSchema.Type.ARRAY:
            return f"Array<{self.items.typescript_type}>" if self.items else "Array"
        elif self.type == JSONSchema.Type.OBJECT:
            # 如果是 OBJECT 类型，根据属性生成 TypeScript 对象接口
            if not self.properties:
                return "Record<string, any>"
            return self.to_typescript_object_interface()
        elif self.enum:
            # 如果存在枚举值，则返回枚举值的字符串表示
            return " | ".join(repr(v) for v in self.enum)
        else:
            # 抛出异常，表示不支持该类型的转换
            raise NotImplementedError(
                f"JSONSchema.typescript_type does not support Type.{self.type.name} yet"
            )
```