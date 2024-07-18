# `.\graphrag\graphrag\index\cli.py`

```py
# 版权声明和许可信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Main definition."""

# 引入必要的库和模块
import asyncio  # 异步编程库
import json  # JSON 数据处理库
import logging  # 日志记录库
import platform  # 平台信息库
import sys  # 系统相关库
import time  # 时间处理库
import warnings  # 警告处理库
from pathlib import Path  # 路径处理库

# 引入 graphrag 库中的配置和模块
from graphrag.config import (
    create_graphrag_config,
)
from graphrag.index import PipelineConfig, create_pipeline_config
from graphrag.index.cache import NoopPipelineCache
from graphrag.index.progress import (
    NullProgressReporter,
    PrintProgressReporter,
    ProgressReporter,
)
from graphrag.index.progress.rich import RichProgressReporter
from graphrag.index.run import run_pipeline_with_config

# 引入本地 emit 模块中的 TableEmitterType 类型
from .emit import TableEmitterType
# 引入 graph.extractors 中不同模块中的提示信息常量
from .graph.extractors.claims.prompts import CLAIM_EXTRACTION_PROMPT
from .graph.extractors.community_reports.prompts import COMMUNITY_REPORT_PROMPT
from .graph.extractors.graph.prompts import GRAPH_EXTRACTION_PROMPT
from .graph.extractors.summarize.prompts import SUMMARIZE_PROMPT
# 引入 init_content 模块中的初始化常量
from .init_content import INIT_DOTENV, INIT_YAML

# 忽略 numba 发出的特定警告信息
warnings.filterwarnings("ignore", message=".*NumbaDeprecationWarning.*")

# 设置日志记录器
log = logging.getLogger(__name__)


def redact(input: dict) -> str:
    """Sanitize the config json."""
    # 对配置文件进行敏感信息的处理
    def redact_dict(input: dict) -> dict:
        if not isinstance(input, dict):
            return input

        result = {}
        for key, value in input.items():
            if key in {
                "api_key",
                "connection_string",
                "container_name",
                "organization",
            }:
                if value is not None:
                    result[key] = f"REDACTED, length {len(value)}"
            elif isinstance(value, dict):
                result[key] = redact_dict(value)
            elif isinstance(value, list):
                result[key] = [redact_dict(i) for i in value]
            else:
                result[key] = value
        return result

    # 调用内部函数对输入进行敏感信息处理，并将处理后的结果转换为格式化 JSON 字符串
    redacted_dict = redact_dict(input)
    return json.dumps(redacted_dict, indent=4)


def index_cli(
    root: str,
    init: bool,
    verbose: bool,
    resume: str | None,
    memprofile: bool,
    nocache: bool,
    reporter: str | None,
    config: str | None,
    emit: str | None,
    dryrun: bool,
    overlay_defaults: bool,
    cli: bool = False,
):
    """Run the pipeline with the given config."""
    # 生成运行标识符，如果没有提供则使用当前时间戳
    run_id = resume or time.strftime("%Y%m%d-%H%M%S")
    # 启用日志记录
    _enable_logging(root, run_id, verbose)
    # 获取进度报告器对象
    progress_reporter = _get_progress_reporter(reporter)
    
    # 如果初始化标志为 True，则在指定根目录下初始化项目并退出程序
    if init:
        _initialize_project_at(root, progress_reporter)
        sys.exit(0)
    
    # 如果 overlay_defaults 标志为 True，则创建带有默认值的流水线配置对象
    if overlay_defaults:
        pipeline_config: str | PipelineConfig = _create_default_config(
            root, config, verbose, dryrun or False, progress_reporter
        )
    else:
        # 否则，使用提供的配置文件或者创建默认的流水线配置对象
        pipeline_config: str | PipelineConfig = config or _create_default_config(
            root, None, verbose, dryrun or False, progress_reporter
        )
    # 根据条件创建一个空的缓存对象或者使用默认的缓存对象
    cache = NoopPipelineCache() if nocache else None
    # 根据逗号分割字符串创建发射器列表，或者设为None
    pipeline_emit = emit.split(",") if emit else None
    # 标记是否遇到任何错误
    encountered_errors = False

    def _run_workflow_async() -> None:
        import signal

        # 处理信号的回调函数
        def handle_signal(signum, _):
            # 处理接收到的信号
            progress_reporter.info(f"Received signal {signum}, exiting...")
            # 停止进度报告器
            progress_reporter.dispose()
            # 取消所有异步任务
            for task in asyncio.all_tasks():
                task.cancel()
            # 输出信息，表示所有任务已取消，准备退出
            progress_reporter.info("All tasks cancelled. Exiting...")

        # 注册信号处理器用于 SIGINT 和 SIGHUP
        signal.signal(signal.SIGINT, handle_signal)
        if sys.platform != "win32":
            signal.signal(signal.SIGHUP, handle_signal)

        # 异步执行函数
        async def execute():
            nonlocal encountered_errors
            # 异步迭代运行流水线配置
            async for output in run_pipeline_with_config(
                pipeline_config,
                run_id=run_id,
                memory_profile=memprofile,
                cache=cache,
                progress_reporter=progress_reporter,
                emit=(
                    [TableEmitterType(e) for e in pipeline_emit]
                    if pipeline_emit
                    else None
                ),
                is_resume_run=bool(resume),
            ):
                # 如果输出中包含错误信息
                if output.errors and len(output.errors) > 0:
                    # 标记已经遇到了错误
                    encountered_errors = True
                    # 输出错误信息到进度报告器
                    progress_reporter.error(output.workflow)
                else:
                    # 输出成功信息到进度报告器
                    progress_reporter.success(output.workflow)

                # 输出结果信息到进度报告器
                progress_reporter.info(str(output.result))

        # 根据平台选择不同的事件循环处理方式
        if platform.system() == "Windows":
            import nest_asyncio  # type: ignore Ignoring because out of windows this will cause an error

            # 应用nest_asyncio兼容性方案
            nest_asyncio.apply()
            # 获取事件循环并运行异步执行函数
            loop = asyncio.get_event_loop()
            loop.run_until_complete(execute())
        elif sys.version_info >= (3, 11):
            import uvloop  # type: ignore Ignoring because on windows this will cause an error

            # 使用uvloop作为事件循环的工厂函数运行异步执行函数
            with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:  # type: ignore Ignoring because minor versions this will throw an error
                runner.run(execute())
        else:
            import uvloop  # type: ignore Ignoring because on windows this will cause an error

            # 安装uvloop作为默认的事件循环
            uvloop.install()
            # 运行异步执行函数
            asyncio.run(execute())

    # 执行异步工作流函数
    _run_workflow_async()
    # 停止进度报告器
    progress_reporter.stop()
    # 如果遇到了错误，输出错误信息到进度报告器
    if encountered_errors:
        progress_reporter.error(
            "Errors occurred during the pipeline run, see logs for more details."
        )
    else:
        # 否则，输出成功信息到进度报告器
        progress_reporter.success("All workflows completed successfully.")

    # 如果是命令行接口，根据是否遇到错误退出程序
    if cli:
        sys.exit(1 if encountered_errors else 0)
def _initialize_project_at(path: str, reporter: ProgressReporter) -> None:
    """Initialize the project at the given path."""
    # 打印信息，指示正在初始化项目的路径
    reporter.info(f"Initializing project at {path}")

    # 将路径转换为 Path 对象
    root = Path(path)

    # 如果根目录不存在，则创建之，包括所有必要的父目录
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    # 设置 YAML 文件的路径
    settings_yaml = root / "settings.yaml"

    # 如果 settings.yaml 文件已经存在，则抛出 ValueError 异常
    if settings_yaml.exists():
        msg = f"Project already initialized at {root}"
        raise ValueError(msg)

    # 设置 .env 文件的路径
    dotenv = root / ".env"

    # 如果 .env 文件不存在，则写入初始的初始化 YAML 内容
    if not dotenv.exists():
        with settings_yaml.open("w") as file:
            file.write(INIT_YAML)

    # 写入初始的 .env 文件内容
    with dotenv.open("w") as file:
        file.write(INIT_DOTENV)

    # 设置 prompts 目录的路径
    prompts_dir = root / "prompts"

    # 如果 prompts 目录不存在，则创建之，包括所有必要的父目录
    if not prompts_dir.exists():
        prompts_dir.mkdir(parents=True, exist_ok=True)

    # 设置 entity_extraction.txt 文件的路径
    entity_extraction = prompts_dir / "entity_extraction.txt"

    # 如果 entity_extraction.txt 文件不存在，则写入 GRAPH_EXTRACTION_PROMPT 的内容
    if not entity_extraction.exists():
        with entity_extraction.open("w") as file:
            file.write(GRAPH_EXTRACTION_PROMPT)

    # 设置 summarize_descriptions.txt 文件的路径
    summarize_descriptions = prompts_dir / "summarize_descriptions.txt"

    # 如果 summarize_descriptions.txt 文件不存在，则写入 SUMMARIZE_PROMPT 的内容
    if not summarize_descriptions.exists():
        with summarize_descriptions.open("w") as file:
            file.write(SUMMARIZE_PROMPT)

    # 设置 claim_extraction.txt 文件的路径
    claim_extraction = prompts_dir / "claim_extraction.txt"

    # 如果 claim_extraction.txt 文件不存在，则写入 CLAIM_EXTRACTION_PROMPT 的内容
    if not claim_extraction.exists():
        with claim_extraction.open("w") as file:
            file.write(CLAIM_EXTRACTION_PROMPT)

    # 设置 community_report.txt 文件的路径
    community_report = prompts_dir / "community_report.txt"

    # 如果 community_report.txt 文件不存在，则写入 COMMUNITY_REPORT_PROMPT 的内容
    if not community_report.exists():
        with community_report.open("w") as file:
            file.write(COMMUNITY_REPORT_PROMPT)


def _create_default_config(
    root: str,
    config: str | None,
    verbose: bool,
    dryrun: bool,
    reporter: ProgressReporter,
) -> PipelineConfig:
    """Overlay default values on an existing config or create a default config if none is provided."""
    # 如果提供了配置文件且文件不存在，则抛出 ValueError 异常
    if config and not Path(config).exists():
        msg = f"Configuration file {config} does not exist"
        raise ValueError

    # 如果根目录不存在，则抛出 ValueError 异常
    if not Path(root).exists():
        msg = f"Root directory {root} does not exist"
        raise ValueError(msg)

    # 读取配置参数
    parameters = _read_config_parameters(root, config, reporter)

    # 记录默认配置信息到日志中
    log.info(
        "using default configuration: %s",
        redact(parameters.model_dump()),
    )

    # 如果 verbose 或 dryrun 标志为 True，则输出使用的默认配置信息到报告中
    if verbose or dryrun:
        reporter.info(f"Using default configuration: {redact(parameters.model_dump())}")

    # 创建管道配置，并根据 verbose 参数决定是否输出最终配置信息到报告中
    result = create_pipeline_config(parameters, verbose)

    # 如果 verbose 或 dryrun 标志为 True，则输出最终配置信息到报告中
    if verbose or dryrun:
        reporter.info(f"Final Config: {redact(result.model_dump())}")

    # 如果 dryrun 标志为 True，则输出完成 dry run 操作的信息，并退出程序
    if dryrun:
        reporter.info("dry run complete, exiting...")
        sys.exit(0)

    # 返回最终的管道配置对象
    return result


def _read_config_parameters(root: str, config: str | None, reporter: ProgressReporter):
    # 将根目录路径转换为 Path 对象
    _root = Path(root)

    # 根据提供的配置文件路径或默认的 settings.yaml/.yml 路径来确定 settings 文件的路径
    settings_yaml = (
        Path(config)
        if config and Path(config).suffix in [".yaml", ".yml"]
        else _root / "settings.yaml"
    )

    # 如果设置文件不存在，则尝试使用 .yml 后缀的文件名
    if not settings_yaml.exists():
        settings_yaml = _root / "settings.yml"
    # 如果指定了配置文件并且其后缀为 .json，则使用指定路径的配置文件
    settings_json = (
        Path(config)
        if config and Path(config).suffix == ".json"
        else _root / "settings.json"
    )

    # 如果 settings_yaml 存在，则读取其中的配置信息
    if settings_yaml.exists():
        # 输出成功消息，指示正在从 settings_yaml 文件读取设置
        reporter.success(f"Reading settings from {settings_yaml}")
        with settings_yaml.open("r") as file:
            import yaml

            # 使用 yaml.safe_load 安全加载 YAML 文件内容
            data = yaml.safe_load(file)
            # 调用 create_graphrag_config 函数创建配置对象，传入数据和根目录参数
            return create_graphrag_config(data, root)

    # 如果 settings_json 存在，则读取其中的配置信息
    if settings_json.exists():
        # 输出成功消息，指示正在从 settings_json 文件读取设置
        reporter.success(f"Reading settings from {settings_json}")
        with settings_json.open("r") as file:
            import json

            # 使用 json.loads 解析 JSON 文件内容
            data = json.loads(file.read())
            # 调用 create_graphrag_config 函数创建配置对象，传入数据和根目录参数
            return create_graphrag_config(data, root)

    # 如果以上两个文件都不存在，则从环境变量中读取配置信息
    reporter.success("Reading settings from environment variables")
    # 调用 create_graphrag_config 函数创建配置对象，只传入根目录参数
    return create_graphrag_config(root_dir=root)
# 根据报告类型获取进度报告器对象
def _get_progress_reporter(reporter_type: str | None) -> ProgressReporter:
    # 如果报告类型为空或者为"rich"，返回一个 RichProgressReporter 对象
    if reporter_type is None or reporter_type == "rich":
        return RichProgressReporter("GraphRAG Indexer ")
    # 如果报告类型为"print"，返回一个 PrintProgressReporter 对象
    if reporter_type == "print":
        return PrintProgressReporter("GraphRAG Indexer ")
    # 如果报告类型为"none"，返回一个 NullProgressReporter 对象
    if reporter_type == "none":
        return NullProgressReporter()

    # 如果报告类型不在已知的类型中，抛出一个值错误异常
    msg = f"Invalid progress reporter type: {reporter_type}"
    raise ValueError(msg)


# 启用日志记录功能
def _enable_logging(root_dir: str, run_id: str, verbose: bool) -> None:
    # 设置日志文件路径
    logging_file = (
        Path(root_dir) / "output" / run_id / "reports" / "indexing-engine.log"
    )
    # 确保日志文件的父目录存在，如果不存在则创建
    logging_file.parent.mkdir(parents=True, exist_ok=True)

    # 创建日志文件，如果文件已存在则不做任何操作
    logging_file.touch(exist_ok=True)

    # 配置日志基本设置
    logging.basicConfig(
        filename=str(logging_file),  # 设置日志文件名
        filemode="a",  # 追加模式写入日志文件
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",  # 日志格式
        datefmt="%H:%M:%S",  # 时间格式
        level=logging.DEBUG if verbose else logging.INFO,  # 日志级别
    )
```