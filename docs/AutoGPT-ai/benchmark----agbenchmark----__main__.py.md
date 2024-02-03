# `.\AutoGPT\benchmark\agbenchmark\__main__.py`

```py
# 导入日志、操作系统、系统模块，日期时间、时区、路径模块，以及类型提示模块
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# 导入 click 模块和 DefaultGroup 类，以及 dotenv 模块和 AgentBenchmarkConfig 类
import click
from click_default_group import DefaultGroup
from dotenv import load_dotenv

# 从 agbenchmark 包中导入 configure_logging 函数
from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.utils.logging import configure_logging

# 加载环境变量文件
load_dotenv()

# 尝试导入 helicone 模块，如果存在 HELICONE_API_KEY 环境变量，则启用 helicone
# 如果导入失败或者没有 HELICONE_API_KEY 环境变量，则禁用 helicone
# try:
#     if os.getenv("HELICONE_API_KEY"):
#         import helicone  # noqa

#         helicone_enabled = True
#     else:
#         helicone_enabled = False
# except ImportError:
#     helicone_enabled = False

# 定义自定义异常类 InvalidInvocationError
class InvalidInvocationError(ValueError):
    pass

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)

# 获取当前时间的 datetime 对象，并转换为 UTC 时间字符串
BENCHMARK_START_TIME_DT = datetime.now(timezone.utc)
BENCHMARK_START_TIME = BENCHMARK_START_TIME_DT.strftime("%Y-%m-%dT%H:%M:%S+00:00")

# 如果 helicone_enabled 为 True，则从 helicone.lock 模块导入 HeliconeLockManager 类
# 并写入自定义属性 "benchmark_start_time"，值为 BENCHMARK_START_TIME
# if helicone_enabled:
#     from helicone.lock import HeliconeLockManager

#     HeliconeLockManager.write_custom_property(
#         "benchmark_start_time", BENCHMARK_START_TIME
#     )

# 创建命令行应用程序组，设置默认日志级别为 DEBUG 或 INFO
@click.group(cls=DefaultGroup, default_if_no_args=True)
@click.option("--debug", is_flag=True, help="Enable debug output")
def cli(
    debug: bool,
) -> Any:
    configure_logging(logging.DEBUG if debug else logging.INFO)

# 定义名为 start 的命令，抛出 DeprecationWarning 异常，提示使用 "agbenchmark run" 替代 "agbenchmark start"
@cli.command(hidden=True)
def start():
    raise DeprecationWarning(
        "`agbenchmark start` is deprecated. Use `agbenchmark run` instead."
    )

# 定义默认命令，包含多个选项参数，用于运行挑战测试
@cli.command(default=True)
@click.option(
    "-N", "--attempts", default=1, help="Number of times to run each challenge."
)
@click.option(
    "-c",
    "--category",
    multiple=True,
    help="(+) Select a category to run.",
)
@click.option(
    "-s",
    "--skip-category",
    multiple=True,
    help="(+) Exclude a category from running.",
)
@click.option("--test", multiple=True, help="(+) Select a test to run.")
@click.option("--maintain", is_flag=True, help="Run only regression tests.")
@click.option("--improve", is_flag=True, help="Run only non-regression tests.")
@click.option(
    "--explore",
    is_flag=True,
    # 设置命令行参数 help，用于指定只运行从未被击败过的挑战
# 添加命令行选项，表示是否忽略挑战之间的依赖关系，无需成功/失败即运行所有（选定的）挑战
@click.option(
    "--no-dep",
    is_flag=True,
    help="Run all (selected) challenges, regardless of dependency success/failure.",
)
# 添加命令行选项，用于覆盖挑战的时间限制（秒）
@click.option("--cutoff", type=int, help="Override the challenge time limit (seconds).")
# 添加命令行选项，禁用挑战的时间限制
@click.option("--nc", is_flag=True, help="Disable the challenge time limit.")
# 添加命令行选项，运行时使用模拟数据
@click.option("--mock", is_flag=True, help="Run with mock")
# 添加命令行选项，保留答案
@click.option("--keep-answers", is_flag=True, help="Keep answers")
# 添加命令行选项，将日志输出到文件而不是终端
@click.option(
    "--backend",
    is_flag=True,
    help="Write log output to a file instead of the terminal.",
)
# 定义函数 run，接受多个参数，包括布尔值、元组和可选的整数和布尔值
def run(
    maintain: bool,
    improve: bool,
    explore: bool,
    mock: bool,
    no_dep: bool,
    nc: bool,
    keep_answers: bool,
    test: tuple[str],
    category: tuple[str],
    skip_category: tuple[str],
    attempts: int,
    cutoff: Optional[int] = None,
    backend: Optional[bool] = False,
    # agent_path: Optional[Path] = None,
) -> None:
    """
    Run the benchmark on the agent in the current directory.

    Options marked with (+) can be specified multiple times, to select multiple items.
    """
    # 导入必要的模块和函数
    from agbenchmark.main import run_benchmark, validate_args

    # 加载 AgentBenchmarkConfig 配置
    agbenchmark_config = AgentBenchmarkConfig.load()
    logger.debug(f"agbenchmark_config: {agbenchmark_config.agbenchmark_config_dir}")
    try:
        # 验证参数的有效性
        validate_args(
            maintain=maintain,
            improve=improve,
            explore=explore,
            tests=test,
            categories=category,
            skip_categories=skip_category,
            no_cutoff=nc,
            cutoff=cutoff,
        )
    except InvalidInvocationError as e:
        # 捕获异常并输出错误信息
        logger.error("Error: " + "\n".join(e.args))
        sys.exit(1)

    original_stdout = sys.stdout  # 保存原始标准输出
    exit_code = None
    # 如果指定了后端，则将标准输出重定向到文件"backend/backend_stdout.txt"
    if backend:
        with open("backend/backend_stdout.txt", "w") as f:
            sys.stdout = f
            # 运行基准测试，并获取退出码
            exit_code = run_benchmark(
                config=agbenchmark_config,
                maintain=maintain,
                improve=improve,
                explore=explore,
                mock=mock,
                no_dep=no_dep,
                no_cutoff=nc,
                keep_answers=keep_answers,
                tests=test,
                categories=category,
                skip_categories=skip_category,
                attempts_per_challenge=attempts,
                cutoff=cutoff,
            )
        # 恢复标准输出
        sys.stdout = original_stdout

    # 如果未指定后端，则直接运行基准测试
    else:
        # 运行基准测试，并获取退出码
        exit_code = run_benchmark(
            config=agbenchmark_config,
            maintain=maintain,
            improve=improve,
            explore=explore,
            mock=mock,
            no_dep=no_dep,
            no_cutoff=nc,
            keep_answers=keep_answers,
            tests=test,
            categories=category,
            skip_categories=skip_category,
            attempts_per_challenge=attempts,
            cutoff=cutoff,
        )
        # 退出程序并返回退出码
        sys.exit(exit_code)
# 定义一个 CLI 命令，用于启动 API 服务，可以通过 --port 参数指定端口号
@cli.command()
@click.option("--port", type=int, help="Port to run the API on.")
def serve(port: Optional[int] = None):
    """Serve the benchmark frontend and API on port 8080."""
    # 导入 uvicorn 模块
    import uvicorn
    # 导入 setup_fastapi_app 函数
    from agbenchmark.app import setup_fastapi_app

    # 加载 AgentBenchmarkConfig 配置
    config = AgentBenchmarkConfig.load()
    # 设置 FastAPI 应用
    app = setup_fastapi_app(config)

    # 运行 FastAPI 应用，使用 uvicorn
    port = port or int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)


# 定义一个 CLI 命令，用于显示当前 AGBenchmark 配置的信息
@cli.command()
def config():
    """Displays info regarding the present AGBenchmark config."""
    try:
        # 加载 AgentBenchmarkConfig 配置
        config = AgentBenchmarkConfig.load()
    except FileNotFoundError as e:
        # 如果配置文件不存在，则输出错误信息
        click.echo(e, err=True)
        return 1

    # 计算配置项的最大键长度
    k_col_width = max(len(k) for k in config.dict().keys())
    # 遍历配置项字典，输出键值对信息
    for k, v in config.dict().items():
        click.echo(f"{k: <{k_col_width}} = {v}")


# 定义一个 CLI 命令，用于打印 AGBenchmark 应用的版本信息
@cli.command()
def version():
    """Print version info for the AGBenchmark application."""
    # 导入 toml 模块
    import toml

    # 获取包的根目录
    package_root = Path(__file__).resolve().parent.parent
    # 加载 pyproject.toml 文件
    pyproject = toml.load(package_root / "pyproject.toml")
    # 获取版本号
    version = pyproject["tool"]["poetry"]["version"]
    # 输出版本信息
    click.echo(f"AGBenchmark version {version}")


# 如果当前脚本作为主程序运行，则执行 CLI 命令
if __name__ == "__main__":
    cli()
```