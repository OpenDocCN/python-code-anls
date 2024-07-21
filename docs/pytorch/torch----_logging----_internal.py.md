# `.\pytorch\torch\_logging\_internal.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块
import functools                   # 导入 functools 模块，用于高阶函数（Higher-order functions）操作
import hashlib                     # 导入 hashlib 模块，用于安全哈希和消息摘要算法的实现
import itertools                   # 导入 itertools 模块，用于创建和操作迭代器的函数
import json                        # 导入 json 模块，用于 JSON 数据的编码和解码
import logging                     # 导入 logging 模块，用于记录日志信息
import os                          # 导入 os 模块，提供了与操作系统交互的功能
import os.path                     # 导入 os.path 模块，用于处理文件路径相关操作
import re                          # 导入 re 模块，用于正则表达式操作
import tempfile                    # 导入 tempfile 模块，用于创建临时文件和目录
from dataclasses import dataclass, field  # 导入 dataclass 和 field 函数，用于创建数据类和定义字段默认值
from importlib import __import__  # 导入 __import__ 函数，用于动态导入模块
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union  # 导入各种类型提示

from weakref import WeakSet       # 导入 WeakSet 类，提供了弱引用集合的支持

import torch._logging.structured  # 导入 torch._logging.structured 模块
from torch.utils._traceback import CapturedTraceback  # 导入 CapturedTraceback 类

log = logging.getLogger(__name__)  # 获取当前模块的 logger 对象

# This is a synthetic logger which doesn't correspond to an actual logger,
# but handles all of our "tracing" logging, which is structured and doesn't go
# to stderr but always goes to a dedicated log file.  We don't put these
# loggers in the classic module hierarchy, because we don't want a suppression
# of logs to also cause a trace to get suppressed (traces typically are not
# collected, unless we are in prod, in which case they always are collected.)
#
# TODO: Maybe we should allow for some sub-hierarchy so you can control which
# traces you want to collect, for performance reasons.
#
# See https://docs.google.com/document/d/1CX_hJ0PNy9f3R1y8TJrfkSeLkvGjjjLU84BSXgS2AZ8/edit
trace_log = logging.getLogger("torch.__trace")

DEFAULT_LOG_LEVEL = logging.WARNING  # 设置默认日志级别为 WARNING
LOG_ENV_VAR = "TORCH_LOGS"          # 日志环境变量名称
LOG_OUT_ENV_VAR = "TORCH_LOGS_OUT"  # 日志输出环境变量名称
LOG_FORMAT_ENV_VAR = "TORCH_LOGS_FORMAT"  # 日志格式环境变量名称
TRACE_ENV_VAR = "TORCH_TRACE"       # 跟踪环境变量名称

@dataclass
class LogRegistry:
    # shorthand name to log qualified name
    # Note: this only contains loggers registered
    # from register_log
    # e.g. "dynamo" -> "torch._dynamo"
    log_alias_to_log_qnames: Dict[str, List[str]] = field(default_factory=dict)  # 日志别名到完整日志名的映射

    # artifact logger qualified names,
    # this is populated lazily, as calls to getArtifactLogger
    # currently formatted as <module>.__<artifact_name>
    # e.g. "torch._dynamo.convert_frame.__guards"
    artifact_log_qnames: Set[str] = field(default_factory=set)  # 已注册的 artifact 日志的限定名集合

    # child logs of registered logs if specified via open
    # registration by the user (ie placing "torch._dynamo.output_graph" in the env var)
    # these need to be tracked so their levels can be reset properly
    # e.g. "torch._dynamo.output_graph"
    child_log_qnames: Set[str] = field(default_factory=set)  # 已注册日志的子日志限定名集合，由用户通过环境变量进行开放注册

    # artifact names, populated by register_artifact
    # e.g. "guards"
    artifact_names: Set[str] = field(default_factory=set)  # 已注册的 artifact 名称集合

    # Artifacts that should be visible by default in the error message
    visible_artifacts: Set[str] = field(default_factory=set)  # 默认情况下在错误消息中可见的 artifacts 集合

    # A short description of each artifact
    artifact_descriptions: Dict[str, str] = field(default_factory=dict)  # 每个 artifact 的简短描述映射

    # artifacts which are not displayed unless explicitly named in the
    # settings. Ex. output_code is NOT displayed even if the inductor
    # log level is set to DEBUG. It must be explicitly named in the settings
    off_by_default_artifact_names: Set[str] = field(default_factory=set)  # 除非在设置中显式命名，否则不显示的 artifacts 名称集合

    # logging format string for artifacts
    # 用于存储日志格式化器的字典，键为字符串，值为 logging.Formatter 对象
    artifact_log_formatters: Dict[str, logging.Formatter] = field(default_factory=dict)

    # 检查给定的名称是否为已注册的工件名称
    def is_artifact(self, name):
        return name in self.artifact_names

    # 检查给定的别名是否为已注册的日志别名
    def is_log(self, alias):
        return alias in self.log_alias_to_log_qnames

    # 注册一个日志别名及其对应的日志资格名称列表
    def register_log(self, alias, log_qnames: Union[str, List[str]]):
        if isinstance(log_qnames, str):
            log_qnames = [log_qnames]
        self.log_alias_to_log_qnames[alias] = log_qnames

    # 注册一个工件名称，并设置其描述、可见性、是否默认关闭以及日志格式
    def register_artifact_name(
        self, name, description, visible, off_by_default, log_format
    ):
        self.artifact_names.add(name)
        if visible:
            self.visible_artifacts.add(name)
        self.artifact_descriptions[name] = description

        # 如果设置为默认关闭，则将其添加到默认关闭的工件名称集合中
        if off_by_default:
            self.off_by_default_artifact_names.add(name)

        # 如果提供了日志格式，则创建并存储对应的日志格式化器对象
        if log_format is not None:
            self.artifact_log_formatters[name] = logging.Formatter(log_format)

    # 注册一个工件日志的资格名称，用于确定需要在日志状态更改时重置哪些日志
    def register_artifact_log(self, artifact_log_qname):
        self.artifact_log_qnames.add(artifact_log_qname)

    # 注册子日志的资格名称
    def register_child_log(self, log_qname):
        self.child_log_qnames.add(log_qname)

    # 获取所有日志资格名称的集合（扁平化处理，可能考虑进行记忆化优化）
    def get_log_qnames(self) -> Set[str]:
        return {
            qname
            for qnames in self.log_alias_to_log_qnames.values()
            for qname in qnames
        }

    # 获取所有工件日志资格名称的集合
    def get_artifact_log_qnames(self):
        return set(self.artifact_log_qnames)

    # 获取所有子日志资格名称的集合
    def get_child_log_qnames(self):
        return set(self.child_log_qnames)

    # 检查给定工件资格名称是否默认为关闭状态
    def is_off_by_default(self, artifact_qname):
        return artifact_qname in self.off_by_default_artifact_names
@dataclass
class LogState:
    # qualified log names -> currently set log level
    # 日志资格名称到当前设置的日志级别的字典
    log_qname_to_level: Dict[str, str] = field(default_factory=dict)

    # the set of currently enabled artifacts
    # 当前启用的工件集合
    artifact_names: Set[str] = field(default_factory=set)

    def enable_artifact(self, artifact_name):
        # 添加指定的工件名称到集合中
        self.artifact_names.add(artifact_name)

    def is_artifact_enabled(self, name):
        # 检查指定名称的工件是否已启用
        return name in self.artifact_names

    def enable_log(self, log_qnames, log_level):
        # 将指定的日志资格名称列表或单个名称与日志级别关联起来
        if isinstance(log_qnames, str):
            log_qnames = [log_qnames]
        for log_qname in log_qnames:
            self.log_qname_to_level[log_qname] = log_level

    def get_log_level_pairs(self):
        """Returns all qualified module names for which the user requested
        explicit logging settings.

        .. warning:

            This function used to return all loggers, regardless of whether
            or not the user specified them or not; it now only returns logs
            which were explicitly mentioned by the user (and torch, which
            always is implicitly requested when we initialize our logging
            subsystem.)
        """
        # 返回用户明确请求的所有资格模块名称及其日志设置
        return self.log_qname_to_level.items()

    def clear(self):
        # 清空日志资格名称与日志级别的字典以及工件名称集合
        self.log_qname_to_level.clear()
        self.artifact_names.clear()


log_registry = LogRegistry()  # 创建日志注册表对象
log_state = LogState()  # 创建日志状态对象

# sample usage: torch._logging.set_logs(**torch._logging.DEFAULT_LOGGING)
# 示例用法：设置默认日志记录级别的字典
DEFAULT_LOGGING = {
    "dynamo": logging.DEBUG,
    "aot": logging.DEBUG,
    "inductor": logging.DEBUG,
    "fsdp": logging.DEBUG,
    "ddp_graphs": True,
    "graph_breaks": True,
    "guards": True,
    "recompiles": True,
    "dynamic": logging.INFO,
}


def set_logs(
    *,
    all: Optional[int] = None,
    dynamo: Optional[int] = None,
    aot: Optional[int] = None,
    autograd: Optional[int] = None,
    dynamic: Optional[int] = None,
    inductor: Optional[int] = None,
    distributed: Optional[int] = None,
    dist_c10d: Optional[int] = None,
    dist_ddp: Optional[int] = None,
    dist_fsdp: Optional[int] = None,
    onnx: Optional[int] = None,
    bytecode: bool = False,
    aot_graphs: bool = False,
    aot_joint_graph: bool = False,
    ddp_graphs: bool = False,
    graph: bool = False,
    graph_code: bool = False,
    graph_breaks: bool = False,
    graph_sizes: bool = False,
    guards: bool = False,
    recompiles: bool = False,
    recompiles_verbose: bool = False,
    trace_source: bool = False,
    trace_call: bool = False,
    trace_bytecode: bool = False,
    output_code: bool = False,
    kernel_code: bool = False,
    schedule: bool = False,
    perf_hints: bool = False,
    post_grad_graphs: bool = False,
    onnx_diagnostics: bool = False,
    fusion: bool = False,
    overlap: bool = False,
    export: Optional[int] = None,
    modules: Optional[Dict[str, Union[int, bool]]] = None,
    cudagraphs: bool = False,
    sym_node: bool = False,
    compiled_autograd_verbose: bool = False,
):
    # 设置日志记录级别和功能开关的函数，具体参数参见函数定义和文档注释
    # 定义一个名为 `fsdp` 的变量，类型为 `Optional[int]`，初始赋值为 `None`
    fsdp: Optional[int] = None,
    """
    Sets the log level for individual components and toggles individual log
    artifact types.

    .. warning:: This feature is a prototype and may have compatibility
        breaking changes in the future.

    .. note:: The ``TORCH_LOGS`` environment variable has complete precedence
        over this function, so if it was set, this function does nothing.

    A component is a set of related features in PyTorch. All of the log
    messages emitted from a given component have their own log levels. If the
    log level of a particular message has priority greater than or equal to its
    component's log level setting, it is emitted. Otherwise, it is suppressed.
    This allows you to, for instance, silence large groups of log messages that
    are not relevant to you and increase verbosity of logs for components that
    are relevant. The expected log level values, ordered from highest to lowest
    priority, are:

        * ``logging.CRITICAL``
        * ``logging.ERROR``
        * ``logging.WARNING``
        * ``logging.INFO``
        * ``logging.DEBUG``
        * ``logging.NOTSET``

    See documentation for the Python ``logging`` module for more information on
    log levels: `<https://docs.python.org/3/library/logging.html#logging-levels>`_

    An artifact is a particular type of log message. Each artifact is assigned
    to a parent component. A component can emit many different kinds of
    artifacts. In general, an artifact is emitted if either its corresponding
    setting in the argument list below is turned on or if its parent component
    is set to a log level less than or equal to the log level of the artifact.

    Example::

        >>> # xdoctest: +SKIP
        >>> import logging

        # The following changes the "dynamo" component to emit DEBUG-level
        # logs, and to emit "graph_code" artifacts.

        >>> torch._logging.set_logs(dynamo=logging.DEBUG, graph_code=True)

        # The following enables the logs for a different module

        >>> torch._logging.set_logs(modules={"unregistered.module.name": logging.DEBUG})
    """
    # ignore if env var is set
    if LOG_ENV_VAR in os.environ:
        log.warning(
            "Using TORCH_LOGS environment variable for log settings, ignoring call to set_logs"
        )
        return

    # 清空日志状态
    log_state.clear()

    # 初始化模块字典
    modules = modules or {}
    # 定义一个内部函数 _set_logs，接受关键字参数 kwargs
    def _set_logs(**kwargs):
        # 遍历 kwargs 和 modules 字典中的键值对
        for alias, val in itertools.chain(kwargs.items(), modules.items()):  # type: ignore[union-attr]
            # 如果值为 None，则跳过当前循环
            if val is None:
                continue
            
            # 如果 alias 是一个记录对象的标识
            if log_registry.is_artifact(alias):
                # 如果值不是布尔类型，抛出 ValueError 异常
                if not isinstance(val, bool):
                    raise ValueError(
                        f"Expected bool to enable artifact {alias}, received {val}"
                    )
                # 如果值为 True，则启用对应的记录对象
                if val:
                    log_state.enable_artifact(alias)
            # 如果 alias 是一个日志记录或者在子日志记录 QNames 中
            elif log_registry.is_log(alias) or alias in log_registry.child_log_qnames:
                # 如果值不在 logging._levelToName 的键集合中，抛出 ValueError 异常
                if val not in logging._levelToName:
                    raise ValueError(
                        f"Unrecognized log level for log {alias}: {val}, valid level values "
                        f"are: {','.join([str(k) for k in logging._levelToName.keys()])}"
                    )
                # 启用指定别名或者 QNames 的日志记录，并设置日志级别为 val
                log_state.enable_log(
                    log_registry.log_alias_to_log_qnames.get(alias, alias), val
                )
            # 如果 alias 不是已知的日志或者记录对象名称，抛出 ValueError 异常
            else:
                raise ValueError(
                    f"Unrecognized log or artifact name passed to set_logs: {alias}"
                )
        
        # 调用内部函数 _init_logs()，用于初始化日志设置
        _init_logs()
    
    # 调用 _set_logs 函数，并传入一系列参数来配置日志系统
    _set_logs(
        torch=all,
        dynamo=dynamo,
        aot=aot,
        autograd=autograd,
        inductor=inductor,
        dynamic=dynamic,
        bytecode=bytecode,
        aot_graphs=aot_graphs,
        aot_joint_graph=aot_joint_graph,
        ddp_graphs=ddp_graphs,
        distributed=distributed,
        dist_c10d=dist_c10d,
        dist_ddp=dist_ddp,
        dist_fsdp=dist_fsdp,
        graph=graph,
        graph_code=graph_code,
        graph_breaks=graph_breaks,
        graph_sizes=graph_sizes,
        guards=guards,
        recompiles=recompiles,
        recompiles_verbose=recompiles_verbose,
        trace_source=trace_source,
        trace_call=trace_call,
        trace_bytecode=trace_bytecode,
        output_code=output_code,
        kernel_code=kernel_code,
        schedule=schedule,
        perf_hints=perf_hints,
        post_grad_graphs=post_grad_graphs,
        onnx=onnx,
        onnx_diagnostics=onnx_diagnostics,
        fusion=fusion,
        overlap=overlap,
        sym_node=sym_node,
        export=export,
        cudagraphs=cudagraphs,
        compiled_autograd_verbose=compiled_autograd_verbose,
        fsdp=fsdp,
    )
# 返回所有已注册的日志记录器的列表
def get_loggers():
    """
    Returns: a list of all registered loggers
    """
    return [logging.getLogger(qname) for qname in log_registry.get_log_qnames()]


# 使用环境变量和用户API控制日志注册名和日志名的关联
def register_log(setting_name, log_name):
    """
    Enables a log to be controlled by the env var and user API with the setting_name
    Args:
        setting_name:  the shorthand name used in the env var and user API
        log_name:  the log name that the setting_name is associated with
    """
    log_registry.register_log(setting_name, log_name)


# 使用环境变量和用户API控制artifact注册名及其详细描述、可见性、默认关闭状态及日志格式
def register_artifact(
    setting_name, description, visible=False, off_by_default=False, log_format=None
):
    """
    Enables an artifact to be controlled by the env var and user API with name
    Args:
        setting_name: the shorthand name used in the env var and user API
        description: A description of what this outputs
        visible: Whether it gets suggested to users by default
        off_by_default: whether this artifact should be logged when the ancestor loggers
            are enabled at level DEBUG
    """
    log_registry.register_artifact_name(
        setting_name, description, visible, off_by_default, log_format
    )


# 获取特定artifact的日志记录器，如果artifact未注册则抛出异常
def getArtifactLogger(module_qname, artifact_name):
    if artifact_name not in log_registry.artifact_names:
        raise ValueError(
            f"Artifact name: {repr(artifact_name)} not registered,"
            f"please call register_artifact({repr(artifact_name)}) in torch._logging.registrations."
        )
    qname = module_qname + f".__{artifact_name}"
    log = logging.getLogger(qname)
    log.artifact_name = artifact_name  # type: ignore[attr-defined]
    log_registry.register_artifact_log(qname)
    configure_artifact_log(log)
    return log


# 增加和减少verbosity级别的正则表达式
INCR_VERBOSITY_CHAR = "+"
DECR_VERBOSITY_CHAR = "-"
VERBOSITY_REGEX = (
    "("
    + "|".join([re.escape(INCR_VERBOSITY_CHAR), re.escape(DECR_VERBOSITY_CHAR)])
    + "?)"
)


# 配置artifact的日志记录方式，根据是否默认关闭设置是否向上传播日志
def configure_artifact_log(log):
    # If the artifact is off by default, then it should only be logged when explicitly
    # enabled; set propagate to False so that this artifact is not propagated
    # to its ancestor logger
    if log_registry.is_off_by_default(log.artifact_name):
        log.propagate = False

    # enable artifact logging when explicitly enabled
    if log_state.is_artifact_enabled(log.artifact_name):
        log.setLevel(logging.DEBUG)
        log.propagate = True


# 匹配逗号分隔的可记录名称列表（允许逗号后面有空格）
def _gen_settings_regex():
    return re.compile(r"((\+|-)?[\w\.]+,\s*)*(\+|-)?[\w\.]+?")


# 验证设置字符串是否符合特定格式
def _validate_settings(settings):
    return re.fullmatch(_gen_settings_regex(), settings) is not None


# 输出帮助信息，根据verbose参数确定是否输出所有artifact名称
def help_message(verbose=False):
    def pad_to(s, length=30):
        assert len(s) <= length
        return s + " " * (length - len(s))

    if verbose:
        printed_artifacts = log_registry.artifact_names
    else:
        printed_artifacts = log_registry.visible_artifacts
    # 如果 verbose 为真，则设置标题为 "All registered names"
    if verbose:
        heading = "All registered names"
    # 否则，设置标题为 "Visible registered names (use TORCH_LOGS='+help' for full list)"
    else:
        heading = "Visible registered names (use TORCH_LOGS='+help' for full list)"
    
    # 从 log_registry 中获取所有日志别名到完全限定名的映射，并按字母顺序排序后，作为列表的第一个元素 "all"
    lines = (
        ["all"]
        + sorted(log_registry.log_alias_to_log_qnames.keys())
        # 对于 printed_artifacts 中的每个名称，构建格式化的字符串，包含名称和其对应的描述信息
        + sorted(
            [
                f"{pad_to(name)}\t{log_registry.artifact_descriptions[name]}"
                for name in printed_artifacts
            ]
        )
    )
    
    # 将 lines 列表中的每一行用换行符连接成一个字符串，并在每一行前面添加两个空格，作为设置信息
    setting_info = "  " + "\n  ".join(lines)
    
    # 定义一个示例字符串，多行字符串，用三个双引号包围
    examples = """
"""
Examples:
  TORCH_LOGS="+dynamo,aot" will set the log level of TorchDynamo to
  logging.DEBUG and AOT to logging.INFO

  TORCH_LOGS="-dynamo,+inductor" will set the log level of TorchDynamo to
  logging.ERROR and TorchInductor to logging.DEBUG

  TORCH_LOGS="aot_graphs" will enable the aot_graphs artifact

  TORCH_LOGS="+dynamo,schedule" will enable set the log level of TorchDynamo
  to logging.DEBUG and enable the schedule artifact

  TORCH_LOGS="+some.random.module,schedule" will set the log level of
  some.random.module to logging.DEBUG and enable the schedule artifact

  TORCH_LOGS_FORMAT="%(levelname)s: %(message)s" or any provided format
  string will set the output format
  Valid keys are "levelname", "message", "pathname", "levelno", "lineno",
  "filename" and "name".

  TORCH_LOGS_OUT=/tmp/output.txt will output the logs to /tmp/output.txt as
  well. This is useful when the output is long.
"""  # flake8: noqa: B950

# 构建包含示例和设置信息的消息字符串
msg = f"""
TORCH_LOGS Info
{examples}

{heading}
{setting_info}
"""
# 返回构建的消息字符串
return msg


def _invalid_settings_err_msg(settings, verbose=False):
    # 获取所有有效设置的字符串表示，包括模块名和日志名称
    valid_settings = ", ".join(
        ["all"]
        + list(log_registry.log_alias_to_log_qnames.keys())
        + list(log_registry.artifact_names)
    )
    # 构建无效设置的错误消息字符串
    msg = f"""
Invalid log settings: {settings}, must be a comma separated list of fully
qualified module names, registered log names or registered artifact names.
For more info on various settings, try TORCH_LOGS="help"
Valid settings:
{valid_settings}
"""
# 返回构建的错误消息字符串
return msg


@functools.lru_cache
def _parse_log_settings(settings):
    if settings == "":
        return dict()

    if settings == "help":
        raise ValueError(help_message(verbose=False))
    elif settings == "+help":
        raise ValueError(help_message(verbose=True))
    # 验证设置的有效性，如果无效则引发 ValueError
    if not _validate_settings(settings):
        raise ValueError(_invalid_settings_err_msg(settings))

    # 移除所有空白字符
    settings = re.sub(r"\s+", "", settings)
    # 按逗号分隔设置项
    log_names = settings.split(",")

    def get_name_level_pair(name):
        # 清理设置项的名称，移除增减 verbosity 字符
        clean_name = name.replace(INCR_VERBOSITY_CHAR, "")
        clean_name = clean_name.replace(DECR_VERBOSITY_CHAR, "")

        # 根据设置项的第一个字符确定日志级别
        if name[0] == INCR_VERBOSITY_CHAR:
            level = logging.DEBUG
        elif name[0] == DECR_VERBOSITY_CHAR:
            level = logging.ERROR
        else:
            level = logging.INFO

        return clean_name, level

    # 创建日志状态对象
    log_state = LogState()
    # 遍历日志名称列表，依次处理每个日志名称
    for name in log_names:
        # 解析日志名称和日志级别的对，返回名称和级别
        name, level = get_name_level_pair(name)

        # 如果日志名称为"all"，将其替换为"torch"
        if name == "all":
            name = "torch"

        # 检查日志注册表中是否存在该日志名称
        if log_registry.is_log(name):
            # 断言级别不为None
            assert level is not None
            # 获取日志别名对应的全限定名列表
            log_qnames = log_registry.log_alias_to_log_qnames[name]
            # 启用日志记录，并指定级别
            log_state.enable_log(log_qnames, level)
        # 如果日志名称是一个artifact（工件）
        elif log_registry.is_artifact(name):
            # 启用指定名称的artifact记录
            log_state.enable_artifact(name)
        # 如果日志名称是有效模块
        elif _is_valid_module(name):
            # 如果该模块没有已注册的父模块
            if not _has_registered_parent(name):
                # 注册该模块作为自身的日志
                log_registry.register_log(name, name)
            else:
                # 注册该模块作为父模块的子日志
                log_registry.register_child_log(name)
            # 启用指定名称的日志，并指定级别
            log_state.enable_log(name, level)
        # 如果以上情况均不符合，抛出值错误异常，包含错误消息
        else:
            raise ValueError(_invalid_settings_err_msg(settings))

    # 返回处理后的日志状态对象
    return log_state
# 检查给定的模块名是否有效
def _is_valid_module(qname):
    try:
        # 尝试动态导入指定名称的模块
        __import__(qname)
        # 如果成功导入模块，则返回 True
        return True
    except ImportError:
        # 如果导入失败，则返回 False
        return False


# 从环境变量中更新日志状态
def _update_log_state_from_env():
    global log_state
    # 获取环境变量中的日志设置信息
    log_setting = os.environ.get(LOG_ENV_VAR, None)
    if log_setting is not None:
        # 解析日志设置，并更新全局的日志状态
        log_state = _parse_log_settings(log_setting)


# 检查指定日志名称是否有已注册的父级日志记录器
def _has_registered_parent(log_qname):
    # 获取当前日志记录器对象
    cur_log = logging.getLogger(log_qname)

    # 获取已注册的日志记录器名称列表
    registered_log_qnames = log_registry.get_log_qnames()

    # 逐级检查当前日志记录器的父级，直到找到已注册的父级或达到顶级日志记录器
    while cur_log.parent:
        if cur_log.name in registered_log_qnames:
            # 如果当前日志记录器的名称在已注册的列表中，则返回 True
            return True
        cur_log = cur_log.parent

    # 如果没有找到已注册的父级日志记录器，则返回 False
    return False


# 应用自定义格式到需要时的日志记录器
class TorchLogsFormatter(logging.Formatter):
    def __init__(self, *, trace: bool = False):
        super().__init__()
        # 初始化时接收一个 trace 参数，表示是否追踪
        self._is_trace = trace
    # 定义一个方法，用于格式化日志记录
    def format(self, record):
        # 获取记录器名称对应的日志工件名称（如果有的话）
        artifact_name = getattr(logging.getLogger(record.name), "artifact_name", None)
        # 如果存在工件名称，则获取其对应的日志格式化器
        if artifact_name is not None:
            artifact_formatter = log_registry.artifact_log_formatters.get(
                artifact_name, None
            )
            # 如果找到了对应的格式化器，则使用该格式化器格式化记录
            if artifact_formatter is not None:
                return artifact_formatter.format(record)

        # 设置记录的消息文本
        record.message = record.getMessage()
        # 格式化记录的时间戳
        record.asctime = self.formatTime(record, "%m%d %H:%M:%S")

        # 异常处理 - 从 logging.Formatter.format 方法复制而来
        s = record.message
        # 如果记录中包含异常信息
        if record.exc_info:
            # 缓存异常的回溯信息，以避免多次转换（因为它是不变的）
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        # 如果存在异常文本，则将其追加到消息末尾
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        # 如果记录包含堆栈信息，则将其追加到消息末尾
        if record.stack_info:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + self.formatStack(record.stack_info)

        # 设置记录的 rankprefix 字段，用于标识日志级别
        record.rankprefix = ""
        # 如果不处于追踪状态，并且分布式环境可用且已初始化，则设置 rankprefix 字段
        if not self._is_trace and dist.is_available() and dist.is_initialized():
            record.rankprefix = f"[rank{dist.get_rank()}]:"

        # 设置记录的 traceid 字段，用于标识追踪 ID
        record.traceid = ""
        # 如果不处于追踪状态，并且当前存在追踪 ID，则设置 traceid 字段
        if (
            not self._is_trace
            and (trace_id := torch._guards.CompileContext.current_trace_id())
            is not None
        ):
            record.traceid = f" [{trace_id}]"

        # 定义日志级别到简写符号的映射关系
        glog_level_to_abbr = {
            "DEBUG": "V",  # V 代表 glog 中的 VERBOSE
            "INFO": "I",
            "WARNING": "W",
            "ERROR": "E",
            "CRITICAL": "C",
        }

        # 获取记录的简写日志级别
        shortlevel = glog_level_to_abbr.get(record.levelname, record.levelname)

        # 设置记录的 artifactprefix 字段，用于标识工件名称
        record.artifactprefix = ""
        # 如果存在工件名称，则设置 artifactprefix 字段
        if artifact_name is not None:
            record.artifactprefix = f" [__{artifact_name}]"

        # 构建记录的前缀字符串，包含日志级别、时间戳、线程信息、文件位置等
        prefix = (
            f"{record.rankprefix}{shortlevel}{record.asctime}.{int(record.msecs*1000):06d} {record.thread} "
            f"{os.path.relpath(record.pathname, os.path.dirname(os.path.dirname(torch.__file__)))}:"
            f"{record.lineno}]{record.traceid}{record.artifactprefix}"
        )

        # 如果处于追踪状态，则构建包含元数据和可选 payload 的记录字符串
        if self._is_trace:
            assert s == ""
            try:
                r = f"{prefix} {json.dumps(record.metadata)}"
            except TypeError:
                # 如果序列化元数据失败，则记录警告并抛出异常
                log.warning("failing metadata: %r", record.metadata)
                raise
            # 如果存在 payload，则将其添加到记录字符串中
            if record.payload is not None:
                r += "".join(f"\n\t{l}" for l in record.payload.split("\n"))
            return r
        else:
            # 如果不处于追踪状态，则将消息分行格式化，每行加上前缀信息，并返回结果
            lines = s.split("\n")
            return "\n".join(f"{prefix} {l}" for l in lines)
def _default_formatter():
    # 获取环境变量中的日志格式设置，若未设置则使用默认的 TorchLogsFormatter 格式
    fmt = os.environ.get(LOG_FORMAT_ENV_VAR, None)
    if fmt is None:
        return TorchLogsFormatter()
    else:
        # 如果环境变量中指定了 "short" 或者 "basic" 格式，则使用相应的基本日志格式
        if fmt in ("short", "basic"):
            fmt = logging.BASIC_FORMAT
        # 返回指定格式的日志 Formatter 对象
        return logging.Formatter(fmt)


DEFAULT_FORMATTER = _default_formatter()


def _setup_handlers(create_handler_fn, log):
    # 创建调试日志处理器，并进行跟踪记录
    debug_handler = _track_handler(create_handler_fn())
    # 设置处理器的格式为默认格式
    debug_handler.setFormatter(DEFAULT_FORMATTER)
    # 设置处理器的日志级别为 DEBUG
    debug_handler.setLevel(logging.DEBUG)
    # 将处理器添加到指定的日志对象中
    log.addHandler(debug_handler)


handlers = WeakSet()  # type: ignore[var-annotated]


# 标记我们创建的处理器，以免修改用户的处理器
def _track_handler(handler):
    # 将处理器添加到全局的处理器集合中
    handlers.add(handler)
    return handler


def _is_torch_handler(handler):
    # 判断给定的处理器是否在全局处理器集合中
    return handler in handlers


# 清除所有指定日志对象上的 Torch 处理器
def _clear_handlers(log):
    # 找出所有是 Torch 处理器的处理器并从日志对象中移除
    to_remove = [handler for handler in log.handlers if _is_torch_handler(handler)]
    for handler in to_remove:
        log.removeHandler(handler)


def _reset_logs():
    # 重置所有注册的日志对象
    for log_qname in log_registry.get_log_qnames():
        log = logging.getLogger(log_qname)
        # 将日志级别设为 WARNING
        log.setLevel(logging.WARNING)
        # 停止向上传播日志消息
        log.propagate = False
        # 清除该日志对象上的所有 Torch 处理器
        _clear_handlers(log)

    # 重置所有工件和子日志
    for artifact_log_qname in itertools.chain(
        log_registry.get_artifact_log_qnames(), log_registry.get_child_log_qnames()
    ):
        log = logging.getLogger(artifact_log_qname)
        # 将日志级别设为 NOTSET
        log.setLevel(logging.NOTSET)
        # 允许向上传播日志消息
        log.propagate = True

    # 停止向上传播追踪日志消息
    trace_log.propagate = False
    # 清除追踪日志对象上的所有 Torch 处理器
    _clear_handlers(trace_log)


def _get_log_state():
    # 返回当前的日志状态
    return log_state


def _set_log_state(state):
    # 设置全局的日志状态
    global log_state
    log_state = state


def _init_logs(log_file_name=None):
    # 重置所有日志设置
    _reset_logs()
    # 从环境变量更新日志状态
    _update_log_state_from_env()

    out = os.environ.get(LOG_OUT_ENV_VAR, None)
    if out is not None:
        log_file_name = out

    # 首先，将所有已知（注册的）日志记录器重置为 NOTSET 级别，以便它们遵循其父日志级别
    for log_qname in log_registry.get_log_qnames():
        # 但不包括顶层的 torch 日志：这默认为 WARNING 级别，以防止日志消息泄漏到更低级别
        if log_qname == "torch":
            continue
        log = logging.getLogger(log_qname)
        log.setLevel(logging.NOTSET)

    # 然后，对用户请求具有非标准日志行为的所有日志记录器修改其日志级别
    for log_qname, level in log_state.get_log_level_pairs():
        log = logging.getLogger(log_qname)
        log.setLevel(level)

    # 最后，为所有注册的日志记录器设置处理器
    # 遍历日志注册表中的所有日志名称
    for log_qname in log_registry.get_log_qnames():
        # 根据日志名称获取日志对象
        log = logging.getLogger(log_qname)
        # 设置日志处理程序，这里是一个流处理程序
        _setup_handlers(
            logging.StreamHandler,
            log,
        )

        # 如果指定了日志文件名，设置文件处理程序
        if log_file_name is not None:
            _setup_handlers(
                lambda: logging.FileHandler(log_file_name),
                log,
            )

    # 配置艺术品日志记录器，注意：此步骤必须在最后进行，
    # 因为祖先日志记录器的级别会影响到它们
    for artifact_log_qname in log_registry.get_artifact_log_qnames():
        # 根据艺术品日志名称获取日志对象
        log = logging.getLogger(artifact_log_qname)
        # 配置艺术品日志记录
        configure_artifact_log(log)

    # 设置特殊追踪日志的处理程序，具有不同的默认配置
    trace_dir_name = os.environ.get(TRACE_ENV_VAR, None)
    # 如果 trace_dir_name 为 None 并且不处于 FB 环境中，则此处理程序可能会自行移除。
    # 这样做可以推迟初始化，直到需要记录日志。这一点很重要，因为 JK 初始化了一个 C++ 单例，
    # 如果我们稍后 fork 进程，这将影响我们的进程。
    handler = LazyTraceHandler(trace_dir_name)
    # trace_log 总是处于 DEBUG 级别。在实际调用 logging 之前，我们会额外检查是否存在任何处理程序。
    trace_log.setLevel(logging.DEBUG)
    # 跟踪 trace_log 的处理程序，并设置格式化程序为 TorchLogsFormatter，开启跟踪功能。
    trace_log_handler = _track_handler(handler)
    trace_log_handler.setFormatter(TorchLogsFormatter(trace=True))
    trace_log.addHandler(trace_log_handler)
class LazyTraceHandler(logging.StreamHandler):
    """Like FileHandler, but the file is allocated lazily only upon the first log message"""

    def __init__(self, root_dir: Optional[str]):
        # 初始化方法，设置根目录属性，继承父类 logging.Handler 的初始化
        self.root_dir = root_dir
        logging.Handler.__init__(self)
        # 初始化流对象为 None
        self.stream = None
        # 保存内置的 open 函数的引用
        self._builtin_open = open

    # 从 CPython 的 FileHandler 克隆而来
    def close(self):
        # 获取锁，确保线程安全
        self.acquire()
        try:
            try:
                # 如果流对象存在
                if self.stream:
                    try:
                        # 刷新流中的数据
                        self.flush()
                    finally:
                        # 保存流对象的引用并清空
                        stream = self.stream
                        self.stream = None
                        # 如果流对象有 close 方法，则关闭流
                        if hasattr(stream, "close"):
                            stream.close()
            finally:
                # 无条件调用父类的 close 方法，防止延迟设置时的处理器泄漏
                logging.StreamHandler.close(self)
        finally:
            # 释放锁
            self.release()
    # 定义一个方法 `emit`，用于处理日志记录
    def emit(self, record):
        # 如果输出流未设置
        if self.stream is None:
            # 初始化一个标记为 False 的变量 ok
            ok = False
            # 如果根目录未设置
            if self.root_dir is None:
                # 设置日志目录为 "/logs"
                TRACE_LOG_DIR = "/logs"
                # 设置打开文件的函数为内置打开函数
                open_func = self._builtin_open

                # 导入 torch 版本信息
                import torch.version as torch_version

                # 如果 torch_version 具有属性 "git_version"
                if hasattr(torch_version, "git_version"):
                    # 记录日志信息，指示 LazyTraceHandler 被禁用，因为不是 fbcode
                    log.info("LazyTraceHandler: disabled because not fbcode")
                # 如果 justknobs_check('pytorch/trace:enable') 返回 False
                elif not torch._utils_internal.justknobs_check("pytorch/trace:enable"):
                    # 记录日志信息，指示 LazyTraceHandler 被禁用
                    log.info(
                        "LazyTraceHandler: disabled because justknobs_check('pytorch/trace:enable') returned False"
                    )
                # 如果 TRACE_LOG_DIR 目录不存在
                elif not os.path.exists(TRACE_LOG_DIR):
                    # 记录日志信息，指示 LazyTraceHandler 被禁用
                    log.info(
                        "LazyTraceHandler: disabled because %s does not exist",
                        TRACE_LOG_DIR,
                    )
                # 如果 TRACE_LOG_DIR 目录不可写
                elif not os.access(TRACE_LOG_DIR, os.W_OK):
                    # 记录日志信息，指示 LazyTraceHandler 被禁用
                    log.info(
                        "LazyTraceHandler: disabled because %s is not writeable",
                        TRACE_LOG_DIR,
                    )
                else:
                    # 如果以上条件均不满足，则将根目录设置为 TRACE_LOG_DIR
                    self.root_dir = TRACE_LOG_DIR

            # 如果根目录已设置
            if self.root_dir is not None:
                # 确保根目录存在，如果不存在则创建
                os.makedirs(self.root_dir, exist_ok=True)
                ranksuffix = ""
                # 如果分布式包可用且已初始化，则设置 ranksuffix 为当前进程的分布式排名
                if dist.is_available() and dist.is_initialized():
                    ranksuffix = f"rank_{dist.get_rank()}_"
                # 创建一个命名临时文件，用于写入日志
                self.stream = tempfile.NamedTemporaryFile(
                    mode="w+",
                    suffix=".log",
                    prefix=f"dedicated_log_torch_trace_{ranksuffix}",
                    dir=self.root_dir,
                    delete=False,
                )
                # 记录日志信息，指示正在将日志记录到该临时文件
                log.info("LazyTraceHandler: logging to %s", self.stream.name)
            else:
                # 如果根目录未设置，则移除当前处理器并返回
                trace_log.removeHandler(self)
                return
        
        # 如果输出流已设置，则调用父类的 emit 方法输出记录
        if self.stream:
            super().emit(record)
# 使用 functools.lru_cache(None) 装饰器缓存 warning_once 函数的调用结果，以优化重复调用性能
@functools.lru_cache(None)
def warning_once(logger_obj, *args, **kwargs):
    """
    This function is similar to `logger.warning()`, but will emit the warning with the same message only once
    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    # 调用 logger_obj 的 warning 方法，传递函数的参数和关键字参数
    logger_obj.warning(*args, **kwargs)


class LazyString:
    def __init__(self, func, *args, **kwargs):
        # 初始化 LazyString 类的实例，接受一个函数 func 和其余参数和关键字参数
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        # 当将 LazyString 实例转换为字符串时，调用 func 函数，并传递初始化时的参数和关键字参数
        return self.func(*self.args, **self.kwargs)


def trace_structured(
    name: str,
    # NB: metadata expected to be dict so adding more info is forward compatible
    # Tuple[str, int] is a special case for string interning
    metadata_fn: Callable[[], Union[Dict[str, Any], Tuple[str, int]]] = dict,
    *,
    payload_fn: Callable[[], Optional[Union[str, object]]] = lambda: None,
    suppress_context: bool = False,
):
    """
    metadata is an arbitrary JSON compatible struct, but it's expected to not be
    too long (e.g., less than 1MB)

    payload is an arbitrary string, which can be arbitrarily long (but expected to have
    newlines so no lines are too long)
    """
    # 断言确保 name 不在 ["rank", "frame_id", "frame_compile_id", "attempt"] 中
    assert "name" not in ["rank", "frame_id", "frame_compile_id", "attempt"]
    # 断言确保 metadata_fn 是可调用的函数类型
    assert callable(
        metadata_fn
    ), f"metadata_fn should be callable, but got {type(metadata_fn)}"
    # 断言确保 payload_fn 是可调用的函数类型
    assert callable(
        payload_fn
    ), f"payload_fn should be callable, but got {type(payload_fn)}"
    # trace_log 永不传播并且始终是 DEBUG 级别，因此检查是否存在处理程序而不是检查日志级别
    # 检查传播时，不要抑制上下文
    # 如果存在追踪日志的处理器
    if trace_log.handlers:
        # 创建一个空字典来存储日志记录的元数据
        record: Dict[str, object] = {}
        # 将给定的名字作为键，调用元数据函数并将结果存储为值
        record[name] = metadata_fn()
        # 如果不需要抑制上下文信息
        if not suppress_context:
            # TODO: 实际上，rank 可能只应该在顶部被发出一次，而不是在所有日志中重复出现，
            # 因为它永远不会改变，我们假设没有交叉。
            # 如果分布式环境可用且已初始化
            if dist.is_available() and dist.is_initialized():
                # 获取当前进程的排名并存储在记录中的 "rank" 键下
                record["rank"] = dist.get_rank()
            # 检查是否存在当前的跟踪 ID
            if (
                trace_id := torch._guards.CompileContext.current_trace_id()
            ) is not None:
                # 如果有跟踪 ID，则记录相关的帧 ID、帧编译 ID 和尝试次数
                record["frame_id"] = trace_id.compile_id.frame_id
                record["frame_compile_id"] = trace_id.compile_id.frame_compile_id
                record["attempt"] = trace_id.attempt
            else:
                # 否则记录日志调用的堆栈，以便更好地诊断为何没有帧 ID
                record["stack"] = torch._logging.structured.from_traceback(
                    CapturedTraceback.extract(skip=1).summary()
                )
        # 调用 payload_fn 函数获取日志记录的有效载荷
        payload = payload_fn()
        # 如果有效载荷不为空
        if payload is not None:
            # 如果有效载荷不是字符串类型
            if not isinstance(payload, str):
                # 如果有效载荷是列表类型，则将其格式化为 JSON 数组的字符串
                if isinstance(payload, list):
                    payload = "[\n" + ",\n".join(json.dumps(i) for i in payload) + "\n]"
                else:
                    # 否则强制换行，以避免溢出行限制
                    payload = json.dumps(payload, indent=0)
            # 计算有效载荷的 MD5 散列值，并将其存储在记录中的 "has_payload" 键下
            h = hashlib.md5()
            h.update(payload.encode("utf-8"))
            record["has_payload"] = h.hexdigest()
        # 向追踪日志记录调试信息，传递额外的 "metadata" 和 "payload" 参数
        trace_log.debug(
            "", extra={"metadata": record, "payload": payload}, stacklevel=2
        )
# 导入 torch 内部的 _guards 模块
import torch._guards
# 导入 torch 内部的 _utils_internal 模块
import torch._utils_internal
# 导入 torch 分布式模块，提供分布式训练支持
import torch.distributed as dist
```