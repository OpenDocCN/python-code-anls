# `.\AutoGPT\autogpts\autogpt\autogpt\core\configuration\schema.py`

```py
# 导入必要的模块
import abc
import os
import typing
from typing import Any, Callable, Generic, Optional, Type, TypeVar, get_args

# 导入必要的类和函数
from pydantic import BaseModel, Field, ValidationError
from pydantic.fields import ModelField, Undefined, UndefinedType
from pydantic.main import ModelMetaclass

# 定义类型变量
T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)

# 定义一个装饰器函数，用于配置用户可配置的字段
def UserConfigurable(
    default: T | UndefinedType = Undefined,
    *args,
    default_factory: Optional[Callable[[], T]] = None,
    from_env: Optional[str | Callable[[], T | None]] = None,
    description: str = "",
    **kwargs,
) -> T:
    # TODO: use this to auto-generate docs for the application configuration
    return Field(
        default,
        *args,
        default_factory=default_factory,
        from_env=from_env,
        description=description,
        **kwargs,
        user_configurable=True,
    )

# 定义系统配置类
class SystemConfiguration(BaseModel):
    # 获取用户配置的方法
    def get_user_config(self) -> dict[str, Any]:
        return _recurse_user_config_values(self)

    # 从环境变量初始化配置对象
    @classmethod
    def from_env(cls):
        """
        Initializes the config object from environment variables.

        Environment variables are mapped to UserConfigurable fields using the from_env
        attribute that can be passed to UserConfigurable.
        """

        # 推断字段值的函数
        def infer_field_value(field: ModelField):
            field_info = field.field_info
            # 获取字段的默认值
            default_value = (
                field.default
                if field.default not in (None, Undefined)
                else (field.default_factory() if field.default_factory else Undefined)
            )
            # 如果字段有from_env属性，则从环境变量中获取值
            if from_env := field_info.extra.get("from_env"):
                val_from_env = (
                    os.getenv(from_env) if type(from_env) is str else from_env()
                )
                if val_from_env is not None:
                    return val_from_env
            return default_value

        return _recursive_init_model(cls, infer_field_value)
    # 定义一个名为 Config 的类
    class Config:
        # 设置额外字段不允许存在
        extra = "forbid"
        # 使用枚举值
        use_enum_values = True
        # 验证赋值
        validate_assignment = True
# 定义一个类型变量 SC，限定为 SystemConfiguration 类型
SC = TypeVar("SC", bound=SystemConfiguration)

# 定义一个 SystemSettings 类，作为所有系统设置的基类
class SystemSettings(BaseModel):
    """A base class for all system settings."""
    
    # 设置系统设置的名称和描述
    name: str
    description: str

    class Config:
        # 禁止额外的字段
        extra = "forbid"
        # 使用枚举值
        use_enum_values = True
        # 验证赋值
        validate_assignment = True

# 定义一个类型变量 S，限定为 SystemSettings 类型
S = TypeVar("S", bound=SystemSettings)

# 定义一个 Configurable 类，作为所有可配置对象的基类
class Configurable(abc.ABC, Generic[S]):
    """A base class for all configurable objects."""
    
    # 设置前缀和默认设置
    prefix: str = ""
    default_settings: typing.ClassVar[S]

    @classmethod
    def get_user_config(cls) -> dict[str, Any]:
        # 递归获取用户配置值
        return _recurse_user_config_values(cls.default_settings)

    @classmethod
    def build_agent_configuration(cls, overrides: dict = {}) -> S:
        """Process the configuration for this object."""
        
        # 更新用户配置并构建代理配置
        base_config = _update_user_config_from_env(cls.default_settings)
        final_configuration = deep_update(base_config, overrides)

        return cls.default_settings.__class__.parse_obj(final_configuration)

# 更新 Pydantic 模型实例的配置字段从环境变量中
def _update_user_config_from_env(instance: BaseModel) -> dict[str, Any]:
    """
    Update config fields of a Pydantic model instance from environment variables.

    Precedence:
    1. Non-default value already on the instance
    2. Value returned by `from_env()`
    3. Default value for the field

    Params:
        instance: The Pydantic model instance.

    Returns:
        The user config fields of the instance.
    """

    # 推断字段值
    def infer_field_value(field: ModelField, value):
        field_info = field.field_info
        default_value = (
            field.default
            if field.default not in (None, Undefined)
            else (field.default_factory() if field.default_factory else None)
        )
        if value == default_value and (from_env := field_info.extra.get("from_env")):
            val_from_env = os.getenv(from_env) if type(from_env) is str else from_env()
            if val_from_env is not None:
                return val_from_env
        return value
    # 定义一个函数，初始化子配置对象
    def init_sub_config(model: Type[SC]) -> SC | None:
        # 尝试从环境变量中创建配置对象
        try:
            return model.from_env()
        # 捕获验证错误异常
        except ValidationError as e:
            # 优雅地处理缺失字段
            if all(e["type"] == "value_error.missing" for e in e.errors()):
                return None
            # 如果不是缺失字段错误，则重新抛出异常
            raise

    # 递归处理用户配置字段
    return _recurse_user_config_fields(instance, infer_field_value, init_sub_config)
# 递归初始化 Pydantic 模型的用户配置字段
def _recursive_init_model(
    model: Type[M],  # 模型类型参数
    infer_field_value: Callable[[ModelField], Any],  # 推断字段值的回调函数
) -> M:  # 返回类型为 M
    """
    Recursively initialize the user configuration fields of a Pydantic model.

    Parameters:
        model: The Pydantic model type.
        infer_field_value: A callback function to infer the value of each field.
            Parameters:
                ModelField: The Pydantic ModelField object describing the field.

    Returns:
        BaseModel: An instance of the model with the initialized configuration.
    """
    user_config_fields = {}  # 初始化用户配置字段字典
    for name, field in model.__fields__.items():  # 遍历模型的字段
        if "user_configurable" in field.field_info.extra:  # 如果字段可配置
            user_config_fields[name] = infer_field_value(field)  # 推断字段值并存储
        elif type(field.outer_type_) is ModelMetaclass and issubclass(
            field.outer_type_, SystemConfiguration
        ):  # 如果字段是 SystemConfiguration 类型
            try:
                user_config_fields[name] = _recursive_init_model(
                    model=field.outer_type_,
                    infer_field_value=infer_field_value,
                )  # 递归初始化子配置模型
            except ValidationError as e:
                # Gracefully handle missing fields
                if all(e["type"] == "value_error.missing" for e in e.errors()):
                    user_config_fields[name] = None  # 处理缺失字段
                raise

    user_config_fields = remove_none_items(user_config_fields)  # 移除空值项

    return model.parse_obj(user_config_fields)  # 解析并返回初始化后的模型实例


def _recurse_user_config_fields(
    model: BaseModel,  # Pydantic 模型实例
    infer_field_value: Callable[[ModelField, Any], Any],  # 推断字段值的回调函数
    init_sub_config: Optional[
        Callable[[Type[SystemConfiguration]], SystemConfiguration | None]
    ] = None,  # 初始化子配置的回调函数，默认为 None
) -> dict[str, Any]:  # 返回类型为字典
    """
    Recursively process the user configuration fields of a Pydantic model instance.
    ```
    # 定义函数参数：
    # model: 要迭代的 Pydantic 模型。
    # infer_field_value: 用于处理每个字段的回调函数。
    #     Params:
    #         ModelField: 描述字段的 Pydantic ModelField 对象。
    #         Any: 字段的当前值。
    # init_sub_config: 用于初始化子配置的可选回调函数。
    #     Params:
    #         Type[SystemConfiguration]: 要初始化的子配置的类型。
    #
    # 返回结果:
    # dict[str, Any]: 实例的处理过的用户配置字段。
    """
    # 初始化空字典，用于存储处理过的用户配置字段
    user_config_fields = {}
    # 遍历模型的字段和对应的值
    for name, field in model.__fields__.items():
        # 获取模型中字段的值
        value = getattr(model, name)

        # 处理单个字段
        if "user_configurable" in field.field_info.extra:
            # 如果字段标记为可配置，则根据字段值推断字段值类型并存储到用户配置字段中
            user_config_fields[name] = infer_field_value(field, value)

        # 递归处理嵌套的配置对象
        elif isinstance(value, SystemConfiguration):
            # 如果字段值是 SystemConfiguration 类型，则递归处理其字段
            user_config_fields[name] = _recurse_user_config_fields(
                model=value,
                infer_field_value=infer_field_value,
                init_sub_config=init_sub_config,
            )

        # 递归处理可选的嵌套配置对象
        elif value is None and init_sub_config:
            # 如果字段值为空且需要初始化子配置，则处理可选的嵌套配置对象
            field_type = get_args(field.annotation)[0]  # Optional[T] -> T
            if type(field_type) is ModelMetaclass and issubclass(
                field_type, SystemConfiguration
            ):
                sub_config = init_sub_config(field_type)
                if sub_config:
                    # 递归处理可选的嵌套配置对象
                    user_config_fields[name] = _recurse_user_config_fields(
                        model=sub_config,
                        infer_field_value=infer_field_value,
                        init_sub_config=init_sub_config,
                    )

        # 处理列表中的 SystemConfiguration 对象
        elif isinstance(value, list) and all(
            isinstance(i, SystemConfiguration) for i in value
        ):
            # 遍历列表中的 SystemConfiguration 对象并递归处理
            user_config_fields[name] = [
                _recurse_user_config_fields(i, infer_field_value, init_sub_config)
                for i in value
            ]

        # 处理字典中的 SystemConfiguration 对象
        elif isinstance(value, dict) and all(
            isinstance(i, SystemConfiguration) for i in value.values()
        ):
            # 遍历字典中的 SystemConfiguration 对象并递归处理
            user_config_fields[name] = {
                k: _recurse_user_config_fields(v, infer_field_value, init_sub_config)
                for k, v in value.items()
            }

    # 返回处理后的用户配置字段
    return user_config_fields
def _recurse_user_config_values(
    instance: BaseModel,
    get_field_value: Callable[[ModelField, T], T] = lambda _, v: v,
) -> dict[str, Any]:
    """
    This function recursively traverses the user configuration values in a Pydantic
    model instance.

    Params:
        instance: A Pydantic model instance.
        get_field_value: A callback function to process each field. Parameters:
            ModelField: The Pydantic ModelField object that describes the field.
            Any: The current value of the field.

    Returns:
        A dictionary containing the processed user configuration fields of the instance.
    """
    user_config_values = {}

    # Iterate over each attribute in the instance
    for name, value in instance.__dict__.items():
        # Get the field information for the attribute
        field = instance.__fields__[name]
        # Check if the field is user configurable
        if "user_configurable" in field.field_info.extra:
            # Process the field value using the provided callback function
            user_config_values[name] = get_field_value(field, value)
        # Recursively process nested SystemConfiguration instances
        elif isinstance(value, SystemConfiguration):
            user_config_values[name] = _recurse_user_config_values(
                instance=value, get_field_value=get_field_value
            )
        # Recursively process lists of SystemConfiguration instances
        elif isinstance(value, list) and all(
            isinstance(i, SystemConfiguration) for i in value
        ):
            user_config_values[name] = [
                _recurse_user_config_values(i, get_field_value) for i in value
            ]
        # Recursively process dictionaries of SystemConfiguration instances
        elif isinstance(value, dict) and all(
            isinstance(i, SystemConfiguration) for i in value.values()
        ):
            user_config_values[name] = {
                k: _recurse_user_config_values(v, get_field_value)
                for k, v in value.items()
            }

    return user_config_values


def _get_non_default_user_config_values(instance: BaseModel) -> dict[str, Any]:
    """
    Get the non-default user config fields of a Pydantic model instance.

    Params:
        instance: The Pydantic model instance.

    Returns:
        dict[str, Any]: The non-default user config values on the instance.
    """
    # 定义一个函数，用于获取字段的值
    def get_field_value(field: ModelField, value):
        # 如果字段有默认工厂函数，则调用该函数获取默认值，否则使用字段的默认值
        default = field.default_factory() if field.default_factory else field.default
        # 如果传入的值不等于默认值，则返回该值
        if value != default:
            return value
    
    # 调用递归函数_recurse_user_config_values，传入实例和获取字段值的函数get_field_value，并移除值为None的项后返回结果
    return remove_none_items(_recurse_user_config_values(instance, get_field_value))
# 递归更新一个字典
def deep_update(original_dict: dict, update_dict: dict) -> dict:
    """
    Recursively update a dictionary.

    Params:
        original_dict (dict): The dictionary to be updated.
        update_dict (dict): The dictionary to update with.

    Returns:
        dict: The updated dictionary.
    """
    # 遍历更新字典的键值对
    for key, value in update_dict.items():
        # 如果键存在于原始字典中，并且原始字典中的值和更新字典中的值都是字典类型
        if (
            key in original_dict
            and isinstance(original_dict[key], dict)
            and isinstance(value, dict)
        ):
            # 递归更新原始字典中的值
            original_dict[key] = deep_update(original_dict[key], value)
        else:
            # 直接更新原始字典中的值
            original_dict[key] = value
    # 返回更新后的原始字典
    return original_dict


# 移除字典中值为 None 或 Undefined 的项
def remove_none_items(d):
    # 如果输入是字典类型
    if isinstance(d, dict):
        # 使用字典推导式移除值为 None 或 Undefined 的项
        return {
            k: remove_none_items(v) for k, v in d.items() if v not in (None, Undefined)
        }
    # 如果输入不是字典类型，则直接返回输入
    return d
```