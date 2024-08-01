# `.\DB-GPT-src\dbgpt\util\parameter_utils.py`

```py
# 导入必要的模块
import argparse  # 用于解析命令行参数
import os  # 提供与操作系统交互的功能
from collections import OrderedDict  # 提供有序字典的实现
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass  # 数据类相关功能
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union  # 类型提示相关功能

# 如果在类型检查模式下
if TYPE_CHECKING:
    from dbgpt._private.pydantic import BaseModel  # 导入特定模块

# 缺失值的默认字符串表示
MISSING_DEFAULT_VALUE = "__MISSING_DEFAULT_VALUE__"

# 参数描述的数据类定义
@dataclass
class ParameterDescription:
    required: bool = False  # 参数是否必需，默认为 False
    param_class: Optional[str] = None  # 参数所属类名，可选
    param_name: Optional[str] = None  # 参数名称，可选
    param_type: Optional[str] = None  # 参数类型，可选
    description: Optional[str] = None  # 参数描述，可选
    default_value: Optional[Any] = None  # 参数默认值，可选
    valid_values: Optional[List[Any]] = None  # 参数的有效值列表，可选
    ext_metadata: Optional[Dict[str, Any]] = None  # 扩展元数据，可选

# 基础参数数据类定义
@dataclass
class BaseParameters:
    @classmethod
    def from_dict(
        cls, data: dict, ignore_extra_fields: bool = False
    ) -> "BaseParameters":
        """从字典创建数据类的实例。

        Args:
            data: 包含数据类字段值的字典。
            ignore_extra_fields: 如果为 True，将忽略数据字典中不属于数据类字段的额外字段。
                如果为 False，额外字段将引发错误。默认为 False。

        Returns:
            从给定字典中填充值后的数据类实例。

        Raises:
            TypeError: 如果 `ignore_extra_fields` 是 False 并且数据字典中有不属于数据类字段的字段。
        """
        all_field_names = {f.name for f in fields(cls)}  # 获取所有字段名的集合
        if ignore_extra_fields:
            data = {key: value for key, value in data.items() if key in all_field_names}  # 若忽略额外字段，只保留存在的字段
        else:
            extra_fields = set(data.keys()) - all_field_names  # 找出额外的字段
            if extra_fields:
                raise TypeError(f"Unexpected fields: {', '.join(extra_fields)}")  # 若有额外字段，引发错误
        return cls(**data)  # 使用字典中的值创建并返回数据类实例
    def update_from(self, source: Union["BaseParameters", dict]) -> bool:
        """
        Update the attributes of this object using the values from another object (of the same or parent type) or a dictionary.
        Only update if the new value is different from the current value and the field is not marked as "fixed" in metadata.

        Args:
            source (Union[BaseParameters, dict]): The source to update from. Can be another object of the same type or a dictionary.

        Returns:
            bool: True if at least one field was updated, otherwise False.
        """
        updated = False  # Flag to indicate whether any field was updated
        
        # Check if the source is either an instance of BaseParameters or a dictionary
        if isinstance(source, (BaseParameters, dict)):
            # Iterate through all fields defined in the current object
            for field_info in fields(self):
                # Check if the field has a "fixed" tag in metadata
                tags = field_info.metadata.get("tags")
                tags = [] if not tags else tags.split(",")
                if tags and "fixed" in tags:
                    continue  # Skip updating this field if it's marked as "fixed"
                
                # Determine the new value for the field from the source object
                new_value = (
                    getattr(source, field_info.name)
                    if isinstance(source, BaseParameters)
                    else source.get(field_info.name, None)
                )
                
                # If the new value is not None and differs from the current value, update the field
                if new_value is not None and new_value != getattr(self, field_info.name):
                    setattr(self, field_info.name, new_value)
                    updated = True  # Set the updated flag to True
        
        else:
            raise ValueError(
                "Source must be an instance of BaseParameters (or its derived class) or a dictionary."
            )

        return updated  # Return True if any field was updated, otherwise False

    def __str__(self) -> str:
        """
        Return a string representation of the object using a helper function.
        """
        return _get_dataclass_print_str(self)

    def to_command_args(self, args_prefix: str = "--") -> List[str]:
        """
        Convert the fields of the dataclass to a list of command line arguments.

        Args:
            args_prefix (str): Prefix to use for each command line argument.

        Returns:
            List[str]: A list where each field is represented by two items:
                1. The field name prefixed by args_prefix.
                2. The corresponding field value.
        """
        return _dict_to_command_args(asdict(self), args_prefix=args_prefix)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the dataclass object to a dictionary representation.

        Returns:
            Dict[str, Any]: A dictionary containing field names as keys and their corresponding values.
        """
        return asdict(self)
# 使用 dataclass 装饰器创建一个类，用于定义服务器参数，继承自 BaseParameters 类
@dataclass
class BaseServerParameters(BaseParameters):
    # 主机 IP 地址，可选参数，默认为 "0.0.0.0"，用于绑定服务器
    host: Optional[str] = field(
        default="0.0.0.0", metadata={"help": "The host IP address to bind to."}
    )
    # 端口号，可选参数，默认为 None，用于指定服务器监听的端口
    port: Optional[int] = field(
        default=None, metadata={"help": "The port number to bind to."}
    )
    # 是否作为守护进程运行，可选参数，默认为 False
    daemon: Optional[bool] = field(
        default=False, metadata={"help": "Run the server as a daemon."}
    )
    # 日志级别，可选参数，默认为 None，指定日志记录的详细程度
    log_level: Optional[str] = field(
        default=None,
        metadata={
            "help": "Logging level",
            "valid_values": [
                "FATAL",
                "ERROR",
                "WARNING",
                "WARNING",
                "INFO",
                "DEBUG",
                "NOTSET",
            ],
        },
    )
    # 日志文件名，可选参数，默认为 None，指定日志输出到的文件名
    log_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The filename to store log",
        },
    )
    # 跟踪器记录文件名，可选参数，默认为 None，指定跟踪器记录的文件名
    tracer_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The filename to store tracer span records",
        },
    )
    # 是否将跟踪器记录发送到 OpenTelemetry，可选参数，默认根据环境变量设置
    tracer_to_open_telemetry: Optional[bool] = field(
        default=os.getenv("TRACER_TO_OPEN_TELEMETRY", "False").lower() == "true",
        metadata={
            "help": "Whether send tracer span records to OpenTelemetry",
        },
    )
    # OpenTelemetry 导出器 OTLP 跟踪数据的目标端点，可选参数，默认为 None
    otel_exporter_otlp_traces_endpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` target to which the span "
            "exporter is going to send spans. The endpoint MUST be a valid URL host, "
            "and MAY contain a scheme (http or https), port and path. A scheme of https"
            " indicates a secure connection and takes precedence over this "
            "configuration setting.",
        },
    )
    # 是否启用 OTLP 跟踪数据的不安全连接，可选参数，默认为 None
    otel_exporter_otlp_traces_insecure: Optional[bool] = field(
        default=None,
        metadata={
            "help": "OTEL_EXPORTER_OTLP_TRACES_INSECURE` represents whether to enable "
            "client transport security for gRPC requests for spans. A scheme of https "
            "takes precedence over the this configuration setting. Default: False"
        },
    )
    # OTLP 跟踪数据的 TLS 证书文件路径，可选参数，默认为 None
    otel_exporter_otlp_traces_certificate: Optional[str] = field(
        default=None,
        metadata={
            "help": "`OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE` stores the path to the "
            "certificate file for TLS credentials of gRPC client for traces. "
            "Should only be used for a secure connection for tracing",
        },
    )
    # 用于跟踪的 gRPC 或 HTTP 请求关联的键值对头信息，可选参数，默认为 None
    otel_exporter_otlp_traces_headers: Optional[str] = field(
        default=None,
        metadata={
            "help": "`OTEL_EXPORTER_OTLP_TRACES_HEADERS` contains the key-value pairs "
            "to be used as headers for spans associated with gRPC or HTTP requests.",
        },
    )
    # 定义一个可选的整数类型字段 `otel_exporter_otlp_traces_timeout`，默认为 None
    otel_exporter_otlp_traces_timeout: Optional[int] = field(
        default=None,
        metadata={
            "help": "`OTEL_EXPORTER_OTLP_TRACES_TIMEOUT` is the maximum time the OTLP "
            "exporter will wait for each batch export for spans.",
        },
    )
    
    # 定义一个可选的字符串类型字段 `otel_exporter_otlp_traces_compression`，默认为 None
    otel_exporter_otlp_traces_compression: Optional[str] = field(
        default=None,
        metadata={
            "help": "`OTEL_EXPORTER_OTLP_COMPRESSION` but only for the span exporter. "
            "If both are present, this takes higher precedence.",
        },
    )
# 获取数据类对象的打印字符串
def _get_dataclass_print_str(obj):
    # 获取类名
    class_name = obj.__class__.__name__
    # 初始化参数列表
    parameters = [
        f"\n\n=========================== {class_name} ===========================\n"
    ]
    # 遍历数据类对象的字段信息
    for field_info in fields(obj):
        # 获取字段值
        value = _get_simple_privacy_field_value(obj, field_info)
        # 添加字段名和值到参数列表
        parameters.append(f"{field_info.name}: {value}")
    # 添加结束标记到参数列表
    parameters.append(
        "\n======================================================================\n\n"
    )
    # 返回参数列表拼接成的字符串
    return "\n".join(parameters)


def _dict_to_command_args(obj: Dict, args_prefix: str = "--") -> List[str]:
    """将字典转换为命令行参数列表

    Args:
        obj: 字典
    Returns:
        一个字符串列表，其中每个字段由两个项表示：
        一个是字段名，前缀为args_prefix，另一个是其值。
    """
    args = []
    # 遍历字典的键值对
    for key, value in obj.items():
        # 如果值为None，则跳过
        if value is None:
            continue
        # 添加字段名和值到参数列表
        args.append(f"{args_prefix}{key}")
        args.append(str(value))
    # 返回参数列表
    return args


def _get_simple_privacy_field_value(obj, field_info):
    """从数据类实例中检索字段的值，如果需要，应用隐私规则。

    Args:
        obj: 数据类实例。
        field_info: 包含有关数据类字段信息的Field对象。

    Returns:
    根据隐私规则返回字段的原始或修改后的值。

    示例用法:
    @dataclass
    class Person:
        name: str
        age: int
        ssn: str = field(metadata={"tags": "privacy"})
    p = Person("Alice", 30, "123-45-6789")
    print(_get_simple_privacy_field_value(p, Person.ssn))  # A******9
    """
    # 获取字段的标签
    tags = field_info.metadata.get("tags")
    tags = [] if not tags else tags.split(",")
    is_privacy = False
    if tags and "privacy" in tags:
        is_privacy = True
    # 获取字段值
    value = getattr(obj, field_info.name)
    # 如果不需要隐私处理或值为空，则返回原始值
    if not is_privacy or not value:
        return value
    # 获取字段类型
    field_type = EnvArgumentParser._get_argparse_type(field_info.type)
    # 根据字段类型应用隐私规则
    if field_type is int:
        return -999
    if field_type is float:
        return -999.0
    if field_type is bool:
        return False
    # 字符串
    if len(value) > 5:
        return value[0] + "******" + value[-1]
    return "******"


def _genenv_ignoring_key_case(
    env_key: str, env_prefix: Optional[str] = None, default_value: Optional[str] = None
):
    """忽略键的大小写从环境变量中获取值"""
    if env_prefix:
        env_key = env_prefix + env_key
    # 返回指定环境变量的值，支持大小写不同形式的环境变量名查找
    return os.getenv(
        env_key,             # 尝试获取环境变量名为 env_key 的值
        os.getenv(            # 如果未找到，尝试获取环境变量名为 env_key 全大写的值
            env_key.upper(),
            os.getenv(        # 如果还未找到，尝试获取环境变量名为 env_key 全小写的值
                env_key.lower(),
                default_value  # 如果以上都未找到，则返回默认值 default_value
            )
        )
    )
def _genenv_ignoring_key_case_with_prefixes(
    env_key: str,
    env_prefixes: Optional[List[str]] = None,
    default_value: Optional[str] = None,
) -> str:
    # 如果提供了环境变量前缀列表
    if env_prefixes:
        # 遍历每个前缀
        for env_prefix in env_prefixes:
            # 调用忽略键大小写的环境变量获取函数，使用当前前缀
            env_var_value = _genenv_ignoring_key_case(env_key, env_prefix)
            # 如果获取到了值，直接返回
            if env_var_value:
                return env_var_value
    # 如果没有获取到值，使用默认值继续获取环境变量
    return _genenv_ignoring_key_case(env_key, default_value=default_value)


class EnvArgumentParser:
    @staticmethod
    def get_env_prefix(env_key: str) -> Optional[str]:
        # 如果环境变量键不存在，返回空
        if not env_key:
            return None
        # 将环境变量键中的 "-" 替换为 "_"，并加上下划线后缀
        env_key = env_key.replace("-", "_")
        return env_key + "_"

    def parse_args_into_dataclass(
        self,
        dataclass_type: Type,
        env_prefixes: Optional[List[str]] = None,
        command_args: Optional[List[str]] = None,
        **kwargs,
    ) -> Any:
        """Parse parameters from environment variables and command lines and populate them into data class"""
        # 创建参数解析器对象
        parser = argparse.ArgumentParser(allow_abbrev=False)
        # 遍历数据类字段
        for field in fields(dataclass_type):
            # 从环境变量中获取字段对应的值
            env_var_value: Any = _genenv_ignoring_key_case_with_prefixes(
                field.name, env_prefixes
            )
            # 如果获取到值
            if env_var_value:
                # 去除首尾空白字符
                env_var_value = env_var_value.strip()
                # 根据字段类型进行类型转换
                if field.type is int or field.type == Optional[int]:
                    env_var_value = int(env_var_value)
                elif field.type is float or field.type == Optional[float]:
                    env_var_value = float(env_var_value)
                elif field.type is bool or field.type == Optional[bool]:
                    env_var_value = env_var_value.lower() == "true"
                elif field.type is str or field.type == Optional[str]:
                    pass
                else:
                    # 如果字段类型不支持，抛出异常
                    raise ValueError(f"Unsupported parameter type {field.type}")
            # 如果未获取到值，尝试从 kwargs 中获取
            if not env_var_value:
                env_var_value = kwargs.get(field.name)

            # 添加命令行参数选项
            EnvArgumentParser._build_single_argparse_option(
                parser, field, env_var_value
            )

        # 解析命令行参数
        cmd_args, cmd_argv = parser.parse_known_args(args=command_args)
        # 遍历数据类字段
        for field in fields(dataclass_type):
            # 如果命令行参数中包含字段名
            if field.name in cmd_args:
                # 获取命令行参数值
                cmd_line_value = getattr(cmd_args, field.name)
                # 如果命令行参数值不为空，则更新 kwargs 中的对应字段值
                if cmd_line_value is not None:
                    kwargs[field.name] = cmd_line_value

        # 使用 kwargs 创建并返回数据类对象
        return dataclass_type(**kwargs)

    @staticmethod
    # 定义一个静态方法，用于创建一个 argparse.ArgumentParser 对象
    def _create_arg_parser(dataclass_type: Type) -> argparse.ArgumentParser:
        # 创建一个 argparse.ArgumentParser 对象，并使用数据类的文档字符串作为描述信息
        parser = argparse.ArgumentParser(description=dataclass_type.__doc__)
        # 遍历数据类的每个字段
        for field in fields(dataclass_type):
            # 获取字段的帮助文本和有效值（如果有）
            help_text = field.metadata.get("help", "")
            valid_values = field.metadata.get("valid_values", None)
            # 准备传递给 argparse.ArgumentParser.add_argument 方法的参数字典
            argument_kwargs = {
                "type": EnvArgumentParser._get_argparse_type(field.type),  # 参数类型
                "help": help_text,  # 参数帮助文本
                "choices": valid_values,  # 参数可选值
                "required": EnvArgumentParser._is_require_type(field.type),  # 是否必需参数
            }
            # 如果字段有默认值且不是 MISSING 标记，则设置默认值并将 required 参数设为 False
            if field.default != MISSING:
                argument_kwargs["default"] = field.default
                argument_kwargs["required"] = False
            # 向 argparse.ArgumentParser 对象添加命令行参数
            parser.add_argument(f"--{field.name}", **argument_kwargs)
        # 返回创建好的 argparse.ArgumentParser 对象
        return parser

    # 定义一个静态方法，根据字段信息创建一个 click.Option 或 click.option 对象
    def _create_click_option_from_field(field_name: str, field: Type, is_func=True):
        import click  # 导入 click 库

        # 获取字段的帮助文本和有效值（如果有）
        help_text = field.metadata.get("help", "")
        valid_values = field.metadata.get("valid_values", None)
        # 准备传递给 click.option 或 click.Option 构造函数的参数字典
        cli_params = {
            "default": None if field.default is MISSING else field.default,  # 默认值
            "help": help_text,  # 参数帮助文本
            "show_default": True,  # 是否显示默认值
            "required": field.default is MISSING,  # 是否必需参数
        }
        # 如果字段有有效值，则设置类型为 click.Choice，并指定可选值
        if valid_values:
            cli_params["type"] = click.Choice(valid_values)
        # 根据字段的实际类型确定 click.Option 的类型
        real_type = EnvArgumentParser._get_argparse_type(field.type)
        if real_type is int:
            cli_params["type"] = click.INT
        elif real_type is float:
            cli_params["type"] = click.FLOAT
        elif real_type is str:
            cli_params["type"] = click.STRING
        elif real_type is bool:
            cli_params["is_flag"] = True  # 如果是布尔型，则设置为标志类型
        # 构建命令行参数名
        name = f"--{field_name}"
        # 根据 is_func 参数决定返回 click.option 或 click.Option 对象
        if is_func:
            return click.option(
                name,
                **cli_params,
            )
        else:
            return click.Option([name], **cli_params)

    # 定义一个静态方法，用于创建 click.Option 或 click.option 对象的工厂方法
    @staticmethod
    def create_click_option(
        *dataclass_types: Type,
        _dynamic_factory: Optional[Callable[[], List[Type]]] = None,
        ```
    ):
        # 导入必要的模块：functools 用于函数操作，OrderedDict 用于有序字典
        import functools
        from collections import OrderedDict

        # 创建一个空的有序字典，用于存储合并后的字段
        combined_fields = OrderedDict()

        # 如果存在动态工厂函数 _dynamic_factory，则调用它获取数据类类型列表 _types
        if _dynamic_factory:
            _types = _dynamic_factory()
            # 如果 _types 不为空，则将其转换为列表 dataclass_types，忽略类型检查
            if _types:
                dataclass_types = list(_types)  # type: ignore

        # 遍历每个数据类类型 dataclass_type
        for dataclass_type in dataclass_types:
            # type: ignore
            # 遍历数据类类型中的字段 field
            for field in fields(dataclass_type):
                # 如果字段名 field.name 不在 combined_fields 中，则将其添加进去
                if field.name not in combined_fields:
                    combined_fields[field.name] = field

        # 定义一个装饰器 decorator，用于添加命令行选项到函数 func
        def decorator(func):
            # 反向遍历 combined_fields 中的字段，为每个字段创建对应的点击选项装饰器
            for field_name, field in reversed(combined_fields.items()):
                option_decorator = EnvArgumentParser._create_click_option_from_field(
                    field_name, field
                )
                func = option_decorator(func)

            # 定义 wrapper 函数，用于包装 func，保持其原有函数签名和行为
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        # 返回装饰器函数 decorator
        return decorator

    @staticmethod
    def _create_raw_click_option(
        *dataclass_types: Type,
        _dynamic_factory: Optional[Callable[[], List[Type]]] = None,
    ):
        # 合并多个数据类类型的字段到一个字典 combined_fields 中
        combined_fields = _merge_dataclass_types(
            *dataclass_types, _dynamic_factory=_dynamic_factory
        )
        options = []

        # 反向遍历 combined_fields 中的字段，为每个字段创建原始的点击选项并添加到 options 列表
        for field_name, field in reversed(combined_fields.items()):
            options.append(
                EnvArgumentParser._create_click_option_from_field(
                    field_name, field, is_func=False
                )
            )

        # 返回创建的选项列表 options
        return options

    @staticmethod
    def create_argparse_option(
        *dataclass_types: Type,
        _dynamic_factory: Optional[Callable[[], List[Type]]] = None,
    ) -> argparse.ArgumentParser:
        # 合并多个数据类类型的字段到一个字典 combined_fields 中
        combined_fields = _merge_dataclass_types(
            *dataclass_types, _dynamic_factory=_dynamic_factory
        )
        # 创建一个 argparse.ArgumentParser 对象 parser
        parser = argparse.ArgumentParser()

        # 反向遍历 combined_fields 中的字段，为每个字段构建对应的 argparse 选项并添加到 parser 中
        for _, field in reversed(combined_fields.items()):
            EnvArgumentParser._build_single_argparse_option(parser, field)

        # 返回创建的 argparse.ArgumentParser 对象 parser
        return parser

    @staticmethod
    def _build_single_argparse_option(
        parser: argparse.ArgumentParser, field, default_value=None
    ):
        # 为此字段添加一个命令行参数
        help_text = field.metadata.get("help", "")  # 获取字段元数据中的帮助文本，若无则为空字符串
        valid_values = field.metadata.get("valid_values", None)  # 获取字段元数据中的有效值列表，若无则为 None
        short_name = field.metadata.get("short", None)  # 获取字段元数据
    # 定义函数声明，指定函数接受一个前缀字符串参数并返回一个字符串列表
    def get_env_args(prefix: str) -> List[str]:
        # 初始化一个空列表用于存储环境变量对应的命令行参数
        env_args = []
        # 遍历操作系统环境变量字典中的每一项
        for key, value in os.environ.items():
            # 检查环境变量键是否以给定前缀开头
            if key.startswith(prefix):
                # 构造命令行参数键，去掉前缀并添加 "--"
                arg_key = "--" + key.replace(prefix, "")
                # 检查环境变量的值是否为 "true" 或 "1"，若是则将参数键加入列表
                if value.lower() in ["true", "1"]:
                    # 标志性参数
                    env_args.append(arg_key)
                # 如果环境变量值不是 "false" 或 "0"，则将参数键和值加入列表
                elif not value.lower() in ["false", "0"]:
                    env_args.extend([arg_key, value])
        # 返回最终的命令行参数列表
        return env_args
# 合并给定的数据类类型，并返回一个有序字典，其中包含所有字段的合并视图
def _merge_dataclass_types(
    *dataclass_types: Type, _dynamic_factory: Optional[Callable[[], List[Type]]] = None
) -> OrderedDict:
    # 创建一个空的有序字典，用于存储合并后的字段信息
    combined_fields = OrderedDict()

    # 如果存在动态工厂函数，则使用其返回的类型列表替换输入的数据类类型列表
    if _dynamic_factory:
        _types = _dynamic_factory()
        if _types:
            dataclass_types = list(_types)  # type: ignore

    # 遍历每个数据类类型
    for dataclass_type in dataclass_types:
        # 遍历数据类类型中的每个字段
        for field in fields(dataclass_type):
            # 如果字段名不在已合并字段中，则将其添加到合并字段字典中
            if field.name not in combined_fields:
                combined_fields[field.name] = field

    # 返回合并后的有序字段字典
    return combined_fields


# 将字符串类型表示的类型转换为对应的 Python 类型，并返回
def _type_str_to_python_type(type_str: str) -> Type:
    # 定义类型字符串到 Python 类型的映射关系
    type_mapping: Dict[str, Type] = {
        "int": int,
        "float": float,
        "bool": bool,
        "str": str,
    }
    # 根据类型字符串返回对应的 Python 类型，如果找不到则返回 str 类型
    return type_mapping.get(type_str, str)


# 获取数据类类型的参数描述列表，并根据传入的 kwargs 可选参数进行处理
def _get_parameter_descriptions(
    dataclass_type: Type, **kwargs
) -> List[ParameterDescription]:
    # 初始化一个空列表用于存储参数描述对象
    descriptions = []

    # 遍历数据类类型中的每个字段
    for field in fields(dataclass_type):
        # 提取字段的扩展元数据，排除 "help" 和 "valid_values" 键
        ext_metadata = {
            k: v for k, v in field.metadata.items() if k not in ["help", "valid_values"]
        }

        # 获取字段的默认值，如果默认值为 MISSING 则设置为 None
        default_value = field.default if field.default != MISSING else None

        # 如果字段名在 kwargs 中存在，则将其默认值设置为 kwargs 中对应的值
        if field.name in kwargs:
            default_value = kwargs[field.name]

        # 构造参数描述对象，并添加到描述列表中
        descriptions.append(
            ParameterDescription(
                param_class=f"{dataclass_type.__module__}.{dataclass_type.__name__}",
                param_name=field.name,
                param_type=EnvArgumentParser._get_argparse_type_str(field.type),
                description=field.metadata.get("help", None),
                required=field.default is MISSING,
                default_value=default_value,
                valid_values=field.metadata.get("valid_values", None),
                ext_metadata=ext_metadata,
            )
        )

    # 返回参数描述列表
    return descriptions


# 根据参数描述列表构建参数类，并返回其类型
def _build_parameter_class(desc: List[ParameterDescription]) -> Type:
    # 导入必要的模块函数
    from dbgpt.util.module_utils import import_from_string

    # 如果描述列表为空，则抛出 ValueError 异常
    if not desc:
        raise ValueError("Parameter descriptions cant be empty")

    # 获取第一个参数描述对象的参数类字符串表示
    param_class_str = desc[0].param_class
    class_name = None

    # 如果参数类字符串不为空
    if param_class_str:
        # 尝试根据参数类字符串导入对应的类
        param_class = import_from_string(param_class_str, ignore_import_error=True)
        # 如果成功导入了类，则直接返回该类
        if param_class:
            return param_class
        # 否则从参数类字符串中提取模块名和类名
        module_name, _, class_name = param_class_str.rpartition(".")

    # 初始化用于存储字段名及其默认值或 field() 的字典
    fields_dict = {}
    # 初始化用于存储字段类型注解的字典
    annotations = {}

    # 遍历描述列表中的每个参数描述对象
    for d in desc:
        # 提取扩展元数据，如果不存在则初始化为空字典
        metadata = d.ext_metadata if d.ext_metadata else {}
        # 设置元数据中的 "help" 和 "valid_values" 键
        metadata["help"] = d.description
        metadata["valid_values"] = d.valid_values

        # 根据参数类型字符串获取对应的 Python 类型，并设置为类型注解
        annotations[d.param_name] = _type_str_to_python_type(
            d.param_type  # type: ignore
        )
        # 将字段名及其默认值或 field() 添加到字段字典中
        fields_dict[d.param_name] = field(default=d.default_value, metadata=metadata)

    # 创建新的参数类，并设置 __annotations__ 以提供类型提示
    # 使用 type() 函数动态创建一个新的类对象，类名为 class_name，继承自 object 类
    new_class = type(
        class_name,  # type: ignore  # 类名参数，类型注释忽略（type: ignore）
        (object,),   # 继承的基类，这里是 object 类
        {**fields_dict, "__annotations__": annotations},  # type: ignore
        # 类的属性字典，包括 fields_dict 中的所有键值对和额外的 "__annotations__" 键，类型注释忽略（type: ignore）
    )
    # 将新创建的类对象转换为 dataclass 类型
    result_class = dataclass(new_class)  # type: ignore  # 类型注释忽略（type: ignore）

    # 返回创建的 dataclass 类对象
    return result_class
# 从 argparse.ArgumentParser 对象中提取参数的详细描述，返回一个 ParameterDescription 对象的列表
def _extract_parameter_details(
    parser: argparse.ArgumentParser,
    param_class: Optional[str] = None,
    skip_names: Optional[List[str]] = None,
    overwrite_default_values: Optional[Dict[str, Any]] = None,
) -> List[ParameterDescription]:
    # 如果 overwrite_default_values 为 None，则设为空字典
    if overwrite_default_values is None:
        overwrite_default_values = {}
    
    # 初始化描述列表
    descriptions = []

    # 遍历 parser._actions 中的每一个动作
    for action in parser._actions:
        # 如果 action.default 等于 argparse.SUPPRESS，则通常表示未提供该参数
        if action.default == argparse.SUPPRESS:
            continue

        # 确定参数类别（flag 或 option）
        flag_or_option = (
            "flag" if isinstance(action, argparse._StoreConstAction) else "option"
        )

        # 提取参数名（使用第一个选项字符串，通常是长格式）
        param_name = action.option_strings[0] if action.option_strings else action.dest
        if param_name.startswith("--"):
            param_name = param_name[2:]
        if param_name.startswith("-"):
            param_name = param_name[1:]

        # 将参数名中的连字符替换为下划线
        param_name = param_name.replace("-", "_")

        # 如果 skip_names 存在且 param_name 存在于 skip_names 中，则跳过该参数
        if skip_names and param_name in skip_names:
            continue

        # 收集其他详细信息
        default_value = action.default
        if param_name in overwrite_default_values:
            default_value = overwrite_default_values[param_name]
        arg_type = (
            action.type
            if not callable(action.type)
            else str(action.type.__name__)  # type: ignore
        )
        description = action.help

        # 确定参数是否必需
        required = action.required

        # 提取选择项的有效值（如果提供了选择项）
        valid_values = list(action.choices) if action.choices is not None else None

        # 将 ext_metadata 设置为一个空字典，稍后可以更新
        ext_metadata: Dict[str, Any] = {}

        # 将参数描述添加到 descriptions 列表中
        descriptions.append(
            ParameterDescription(
                param_class=param_class,
                param_name=param_name,
                param_type=arg_type,
                default_value=default_value,
                description=description,
                required=required,
                valid_values=valid_values,
                ext_metadata=ext_metadata,
            )
        )

    # 返回描述列表
    return descriptions


# 从对象中获取字典表示，如果对象为空则返回默认值 default_value
def _get_dict_from_obj(obj, default_value=None) -> Optional[Dict]:
    if not obj:
        return None
    # 如果 obj 是数据类，则提取其字段信息并转换为字典
    if is_dataclass(type(obj)):
        params = {}
        for field_info in fields(obj):
            value = _get_simple_privacy_field_value(obj, field_info)
            params[field_info.name] = value
        return params
    # 如果 obj 是字典，则直接返回
    if isinstance(obj, dict):
        return obj
    # 否则返回默认值
    return default_value


# 从 BaseModel 类中获取基础模型的参数描述列表
def _get_base_model_descriptions(model_cls: "BaseModel") -> List[ParameterDescription]:
    # 导入必要的模块
    from dbgpt._private import pydantic

    # 提取 Pydantic 版本号的主要部分
    version = int(pydantic.VERSION.split(".")[0])  # type: ignore
    # 根据模型类获取模型的 JSON Schema 或者 Schema 对象
    schema = model_cls.model_json_schema() if version >= 2 else model_cls.schema()

    # 获取模型 JSON Schema 中的必填字段集合
    required_fields = set(schema.get("required", []))

    # 初始化参数描述列表
    param_descs = []

    # 遍历模型属性定义中的每一个字段及其对应的 Schema
    for field_name, field_schema in schema.get("properties", {}).items():
        # 获取模型字段对象
        field = model_cls.model_fields[field_name]

        # 获取字段的类型
        param_type = field_schema.get("type")

        # 如果字段类型未定义且存在 "anyOf" 定义，则尝试获取有效的类型
        if not param_type and "anyOf" in field_schema:
            for any_of in field_schema["anyOf"]:
                if any_of["type"] != "null":
                    param_type = any_of["type"]
                    break

        # 根据版本判断字段的默认值
        if version >= 2:
            default_value = (
                field.default
                if hasattr(field, "default")
                and str(field.default) != "PydanticUndefined"
                else None
            )
        else:
            default_value = (
                field.default
                if not field.allow_none
                else (
                    field.default_factory() if callable(field.default_factory) else None
                )
            )

        # 获取字段的描述信息
        description = field_schema.get("description", "")

        # 判断字段是否为必填字段
        is_required = field_name in required_fields

        # 初始化有效值和扩展元数据
        valid_values = None
        ext_metadata = None

        # 如果字段具有额外的元数据信息，则获取有效值和扩展元数据
        if hasattr(field, "field_info"):
            valid_values = (
                list(field.field_info.choices)
                if hasattr(field.field_info, "choices")
                else None
            )
            ext_metadata = (
                field.field_info.extra if hasattr(field.field_info, "extra") else None
            )

        # 构造参数描述对象的类名
        param_class = f"{model_cls.__module__}.{model_cls.__name__}"

        # 创建参数描述对象并添加到参数描述列表中
        param_desc = ParameterDescription(
            param_class=param_class,
            param_name=field_name,
            param_type=param_type,
            default_value=default_value,
            description=description,
            required=is_required,
            valid_values=valid_values,
            ext_metadata=ext_metadata,
        )
        param_descs.append(param_desc)

    # 返回构建好的参数描述列表
    return param_descs
# 定义一个私有类 _SimpleArgParser，用于解析命令行参数
class _SimpleArgParser:
    # 初始化方法，接收多个参数并将其转换为以 '-' 替换 '_' 的格式，初始值设为 None
    def __init__(self, *args):
        self.params = {arg.replace("_", "-"): None for arg in args}

    # 解析方法，接收一个参数列表 args，默认为 sys.argv[1:]，遍历参数列表并处理
    def parse(self, args=None):
        import sys

        # 如果未提供参数列表则使用系统参数列表 sys.argv[1:]
        if args is None:
            args = sys.argv[1:]
        else:
            args = list(args)
        
        prev_arg = None  # 前一个参数的占位符初始化为 None
        for arg in args:
            if arg.startswith("--"):  # 处理长参数（以 '--' 开头）
                if prev_arg:  # 如果有前一个参数，则将其值设为 None
                    self.params[prev_arg] = None
                prev_arg = arg[2:]  # 当前参数去掉前两个字符后作为新的参数名
            else:
                if prev_arg:  # 如果有前一个参数，则将其与当前参数关联
                    self.params[prev_arg] = arg
                    prev_arg = None  # 处理完后重置前一个参数占位符

        if prev_arg:  # 如果最后还有未处理的前一个参数，则设为 None
            self.params[prev_arg] = None

    # 根据参数名获取参数值，支持以 '-' 替换 '_' 的格式
    def _get_param(self, key):
        return self.params.get(key.replace("_", "-")) or self.params.get(key)

    # 获取属性的特殊方法，实现属性访问接口
    def __getattr__(self, item):
        return self._get_param(item)

    # 获取元素的特殊方法，实现下标访问接口
    def __getitem__(self, key):
        return self._get_param(key)

    # 获取参数值的方法，如果参数不存在则返回默认值
    def get(self, key, default=None):
        return self._get_param(key) or default

    # 转换为字符串的特殊方法，返回所有参数名及其值的格式化字符串
    def __str__(self):
        return "\n".join(
            [f'{key.replace("-", "_")}: {value}' for key, value in self.params.items()]
        )


# 构建懒加载点击命令的函数，接收多个数据类类型和可选的动态工厂函数参数
def build_lazy_click_command(*dataclass_types: Type, _dynamic_factory=None):
    import click

    # 定义一个懒加载命令类，继承自点击命令类
    class LazyCommand(click.Command):
        # 初始化方法，接收点击命令类的标准参数和关键字参数
        def __init__(self, *args, **kwargs):
            super(LazyCommand, self).__init__(*args, **kwargs)
            self.dynamic_params_added = False  # 标记是否已添加动态参数

        # 获取参数的方法，如果上下文存在且未添加动态参数，则创建原始点击选项
        def get_params(self, ctx):
            if ctx and not self.dynamic_params_added:
                dynamic_params = EnvArgumentParser._create_raw_click_option(
                    *dataclass_types, _dynamic_factory=_dynamic_factory
                )
                for param in reversed(dynamic_params):
                    self.params.append(param)  # 将动态参数逆序添加到参数列表中
                self.dynamic_params_added = True
            return super(LazyCommand, self).get_params(ctx)

    return LazyCommand  # 返回懒加载命令类作为结果
```