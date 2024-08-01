# `.\DB-GPT-src\dbgpt\_private\pydantic.py`

```py
# 导入必要的模块和函数
from typing import get_origin

import pydantic

# 检查 pydantic 的版本
if pydantic.VERSION.startswith("1."):
    # 如果版本是 1.x，则抛出未实现错误
    PYDANTIC_VERSION = 1
    raise NotImplementedError("pydantic 1.x is not supported, please upgrade to 2.x.")
else:
    # 如果版本是 2.x，则设置版本号为 2
    PYDANTIC_VERSION = 2
    # 导入 pydantic 2.x 所需的类和函数
    from pydantic import (
        BaseModel,
        ConfigDict,
        Extra,
        Field,
        NonNegativeFloat,
        NonNegativeInt,
        PositiveFloat,
        PositiveInt,
        PrivateAttr,
        ValidationError,
        WithJsonSchema,
        field_validator,
        model_validator,
        root_validator,
        validator,
    )

    # 定义禁止额外字段的常量
    EXTRA_FORBID = "forbid"


def model_to_json(model, **kwargs) -> str:
    """将 pydantic 模型转换为 JSON 字符串."""
    if PYDANTIC_VERSION == 1:
        return model.json(**kwargs)
    else:
        # 对于版本 2.x，如果参数中有 'ensure_ascii'，则删除它
        if "ensure_ascii" in kwargs:
            del kwargs["ensure_ascii"]
        return model.model_dump_json(**kwargs)


def model_to_dict(model, **kwargs) -> dict:
    """将 pydantic 模型转换为字典."""
    if PYDANTIC_VERSION == 1:
        return model.dict(**kwargs)
    else:
        return model.model_dump(**kwargs)


def model_fields(model):
    """返回 pydantic 模型的字段."""
    if PYDANTIC_VERSION == 1:
        return model.__fields__
    else:
        return model.model_fields


def field_is_required(field) -> bool:
    """返回字段是否为必需的."""
    if PYDANTIC_VERSION == 1:
        return field.required
    else:
        return field.is_required()


def field_outer_type(field):
    """返回字段的外部类型."""
    if PYDANTIC_VERSION == 1:
        return field.outer_type_
    else:
        # 获取字段注解的原始类型，如果不存在则返回注解本身
        origin = get_origin(field.annotation)
        if origin is None:
            return field.annotation
        return origin


def field_description(field):
    """返回字段的描述."""
    if PYDANTIC_VERSION == 1:
        return field.field_info.description
    else:
        return field.description


def field_default(field):
    """返回字段的默认值."""
    if PYDANTIC_VERSION == 1:
        return field.field_info.default
    else:
        return field.default
```