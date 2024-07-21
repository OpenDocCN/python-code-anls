# `.\pytorch\tools\stats\upload_metrics.py`

```
# 使用 __future__ 模块中的 annotations 特性，使得类型注解中的类型可以使用字符串表示法
from __future__ import annotations

import datetime  # 导入处理日期时间的模块
import inspect  # 导入用于检查对象的内省模块
import os  # 导入操作系统相关功能的模块
import time  # 导入处理时间的模块
import uuid  # 导入生成 UUID 的模块
from decimal import Decimal  # 从 decimal 模块导入 Decimal 类
from typing import Any  # 导入类型提示相关的模块
from warnings import warn  # 导入警告相关的模块


# boto3 是一个可选依赖项，如果未安装，将不会发出指标
# 保持此逻辑在此处，以便调用者无需担心
EMIT_METRICS = False
try:
    import boto3  # type: ignore[import]

    EMIT_METRICS = True  # 若成功导入 boto3，则置为 True
except ImportError as e:
    # 如果导入失败，打印错误信息并指出原因
    print(f"Unable to import boto3. Will not be emitting metrics.... Reason: {e}")

# 有时我们的运行机器位于一个 AWS 帐户中，而指标表可能位于另一个帐户中，因此需要显式指定表的 ARN
TORCHCI_METRICS_TABLE_ARN = (
    "arn:aws:dynamodb:us-east-1:308535385114:table/torchci-metrics"
)


class EnvVarMetric:
    name: str  # 指标名称
    env_var: str  # 环境变量名
    required: bool = True  # 是否必需的标志，默认为 True
    # 用于将环境变量值转换为正确类型的函数（默认为 str）
    type_conversion_fn: Any = None

    def __init__(
        self,
        name: str,
        env_var: str,
        required: bool = True,
        type_conversion_fn: Any = None,
    ) -> None:
        self.name = name  # 初始化指标名称
        self.env_var = env_var  # 初始化环境变量名
        self.required = required  # 初始化是否必需的标志
        self.type_conversion_fn = type_conversion_fn  # 初始化类型转换函数

    def value(self) -> Any:
        # 获取环境变量的值
        value = os.environ.get(self.env_var)

        # GitHub CI 会将某些环境变量设置为空字符串
        DEFAULT_ENVVAR_VALUES = [None, ""]
        if value in DEFAULT_ENVVAR_VALUES:
            if not self.required:
                return None  # 如果不是必需的且值为空，则返回 None

            # 如果值为空且是必需的，则抛出 ValueError
            raise ValueError(
                f"Missing {self.name}. Please set the {self.env_var} "
                "environment variable to pass in this value."
            )

        if self.type_conversion_fn:
            return self.type_conversion_fn(value)  # 如果定义了类型转换函数，则进行转换并返回值

        return value  # 否则直接返回环境变量的值


global_metrics: dict[str, Any] = {}  # 全局指标字典，用于存储每个进程应该发出的指标信息


def add_global_metric(metric_name: str, metric_value: Any) -> None:
    """
    添加应该由当前进程发出的每个指标的统计信息。
    如果 emit_metrics 方法指定了具有相同名称的指标，则会覆盖此值。
    """
    global_metrics[metric_name] = metric_value  # 将指定的指标名称和值添加到全局指标字典中


def emit_metric(
    metric_name: str,
    metrics: dict[str, Any],
) -> None:
    """
    将指标上传到 DynamoDB（然后到 Rockset）。

    即使 EMIT_METRICS 设置为 False，此函数仍将运行用于验证和格式化指标的代码，只是跳过上传过程。

    Parameters:
        metric_name:
            指标的名称。每个唯一的指标应具有不同的名称，并且应在每次运行尝试中仅发出一次。
            指标通过其模块和发出它们的函数进行命名空间化。
        metrics: 要记录的实际数据。
    """
    # 实现上传指标到 DynamoDB 的功能，跳过上传部分仅保留验证和格式化逻辑
    """
    
    # 如果 metrics 参数为 None，则抛出 ValueError 异常
    if metrics is None:
        raise ValueError("You didn't ask to upload any metrics!")

    # 将给定的 metrics 合并到全局 metrics 中，覆盖任何重复的指标
    # 使用 ** 操作符进行字典合并
    metrics = {**global_metrics, **metrics}

    # 使用环境变量来确定工作流运行的基本信息
    # 使用环境变量可以避免将这些信息传递到每个函数中
    # 同时也确保仅在持续集成期间发出指标
    env_var_metrics = [
        EnvVarMetric("repo", "GITHUB_REPOSITORY"),
        EnvVarMetric("workflow", "GITHUB_WORKFLOW"),
        EnvVarMetric("build_environment", "BUILD_ENVIRONMENT", required=False),
        EnvVarMetric("job", "GITHUB_JOB"),
        EnvVarMetric("test_config", "TEST_CONFIG", required=False),
        EnvVarMetric("pr_number", "PR_NUMBER", required=False, type_conversion_fn=int),
        EnvVarMetric("run_id", "GITHUB_RUN_ID", type_conversion_fn=int),
        EnvVarMetric("run_number", "GITHUB_RUN_NUMBER", type_conversion_fn=int),
        EnvVarMetric("run_attempt", "GITHUB_RUN_ATTEMPT", type_conversion_fn=int),
        EnvVarMetric("job_id", "JOB_ID", type_conversion_fn=int),
        EnvVarMetric("job_name", "JOB_NAME"),
    ]

    # 使用调用此函数的函数信息作为命名空间和过滤指标的方式
    calling_frame = inspect.currentframe().f_back  # 获取调用栈的上一层帧
    calling_frame_info = inspect.getframeinfo(calling_frame)  # 获取调用帧的信息
    calling_file = os.path.basename(calling_frame_info.filename)  # 获取调用文件名
    calling_module = inspect.getmodule(calling_frame).__name__  # 获取调用模块名
    calling_function = calling_frame_info.function  # 获取调用函数名

    # 尝试构建保留的指标字典，包括 metric_name、调用文件名、调用模块名、调用函数名和时间戳等信息
    try:
        reserved_metrics = {
            "metric_name": metric_name,
            "calling_file": calling_file,
            "calling_module": calling_module,
            "calling_function": calling_function,
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
            **{m.name: m.value() for m in env_var_metrics if m.value()},
        }
    except ValueError as e:
        # 如果构建保留指标字典时发生 ValueError 异常，则发出警告并返回
        warn(f"Not emitting metrics for {metric_name}. {e}")
        return

    # 使用 metric_name 和时间戳作为前缀以降低 uuid1 名称冲突的风险
    reserved_metrics[
        "dynamo_key"
    ] = f"{metric_name}_{int(time.time())}_{uuid.uuid1().hex}"

    # 确保 metrics 字典不包含任何保留的键
    for key in reserved_metrics.keys():
        used_reserved_keys = [k for k in metrics.keys() if k == key]
        if used_reserved_keys:
            # 如果 metrics 字典中包含保留的键，则抛出 ValueError 异常
            raise ValueError(f"Metrics dict contains reserved keys: [{', '.join(key)}]")

    # 将 metrics 字典中的浮点数值转换为 Decimal 类型，以便能够上传到 DynamoDB
    metrics = _convert_float_values_to_decimals(metrics)
    # 如果设置了 EMIT_METRICS 标志，则执行以下操作
    if EMIT_METRICS:
        try:
            # 创建一个连接到 AWS DynamoDB 的会话
            session = boto3.Session(region_name="us-east-1")
            # 获取 DynamoDB 表并将指定的指标项上传到表中
            session.resource("dynamodb").Table(TORCHCI_METRICS_TABLE_ARN).put_item(
                Item={
                    **reserved_metrics,  # 添加预留的指标项到上传的项中
                    **metrics,           # 添加当前收集的指标项到上传的项中
                }
            )
        except Exception as e:
            # 如果上传指标时出现异常，记录警告信息但不中断程序
            warn(f"Error uploading metric {metric_name} to DynamoDB: {e}")
            return
    else:
        # 如果未设置 EMIT_METRICS 标志，则打印未导入 Boto 库的提示信息
        print(f"Not emitting metrics for {metric_name}. Boto wasn't imported.")
# 将输入的数据字典中的所有浮点数值转换为 Decimal 类型
def _convert_float_values_to_decimals(data: dict[str, Any]) -> dict[str, Any]:
    # 定义一个辅助函数，用于递归地转换数据类型
    def _helper(o: Any) -> Any:
        # 如果是浮点数，则转换为 Decimal 类型并返回
        if isinstance(o, float):
            return Decimal(str(o))
        # 如果是列表，则递归地对列表中的每个元素调用辅助函数
        if isinstance(o, list):
            return [_helper(v) for v in o]
        # 如果是字典，则递归地对字典中的每个键值对调用辅助函数
        if isinstance(o, dict):
            return {_helper(k): _helper(v) for k, v in o.items()}
        # 如果是元组，则递归地对元组中的每个元素调用辅助函数，并返回转换后的元组
        if isinstance(o, tuple):
            return tuple(_helper(v) for v in o)
        # 对于其他类型的数据，保持不变直接返回
        return o

    # 对输入数据字典中的每个键值对应用辅助函数 _helper 进行转换，并返回结果字典
    return {k: _helper(v) for k, v in data.items()}
```