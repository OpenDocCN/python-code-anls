# `.\graphrag\tests\smoke\test_fixtures.py`

```py
# 导入必要的库和模块
import asyncio
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock

# 从自定义库中导入 BlobPipelineStorage 类
from graphrag.index.storage.blob_pipeline_storage import BlobPipelineStorage

# 设置日志记录器
log = logging.getLogger(__name__)

# 检查是否启用了调试模式
debug = os.environ.get("DEBUG") is not None

# 检查是否在 GH_PAGES 环境中
gh_pages = os.environ.get("GH_PAGES") is not None

# 定义 Azure Azurite 连接字符串常量，用于测试目的
# cspell:disable-next-line well-known-key
WELL_KNOWN_AZURITE_CONNECTION_STRING = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1"

def _load_fixtures():
    """从 tests/data 文件夹加载所有的测试数据."""
    params = []
    fixtures_path = Path("./tests/fixtures/")

    # 如果在 GH_PAGES 环境下，只使用 'min-csv' 文件夹，否则使用所有文件夹
    subfolders = ["min-csv"] if gh_pages else sorted(os.listdir(fixtures_path))

    for subfolder in subfolders:
        if not os.path.isdir(fixtures_path / subfolder):
            continue

        config_file = fixtures_path / subfolder / "config.json"
        with config_file.open() as f:
            params.append((subfolder, json.load(f)))

    return params


def pytest_generate_tests(metafunc):
    """为本模块中的所有测试函数生成测试用例."""
    run_slow = metafunc.config.getoption("run_slow")
    configs = metafunc.cls.params[metafunc.function.__name__]

    if not run_slow:
        # 只运行未标记为慢速的测试
        configs = [config for config in configs if not config[1].get("slow", False)]

    funcarglist = [params[1] for params in configs]
    id_list = [params[0] for params in configs]

    # 根据参数生成测试用例
    argnames = sorted(arg for arg in funcarglist[0] if arg != "slow")
    metafunc.parametrize(
        argnames,
        [[funcargs[name] for name in argnames] for funcargs in funcarglist],
        ids=id_list,
    )


def cleanup(skip: bool = False):
    """装饰器，用于在每个测试后清理输出和缓存文件夹."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AssertionError:
                raise
            finally:
                if not skip:
                    root = Path(kwargs["input_path"])
                    shutil.rmtree(root / "output", ignore_errors=True)
                    shutil.rmtree(root / "cache", ignore_errors=True)

        return wrapper

    return decorator


async def prepare_azurite_data(input_path: str, azure: dict) -> Callable[[], None]:
    """为 Azurite 测试准备数据."""
    input_container = azure["input_container"]
    input_base_dir = azure.get("input_base_dir")
    #`
    # 使用 Path 类创建表示输入路径的根对象
    root = Path(input_path)
    # 创建 BlobPipelineStorage 对象，连接到 Azure 存储，并指定容器名称
    input_storage = BlobPipelineStorage(
        connection_string=WELL_KNOWN_AZURITE_CONNECTION_STRING,
        container_name=input_container,
    )
    # 如果容器存在，则删除以清除旧的运行数据
    input_storage.delete_container()
    # 创建新的空容器
    input_storage.create_container()
    
    # 上传数据文件
    # 获取所有 .txt 文件列表和 .csv 文件列表
    txt_files = list((root / "input").glob("*.txt"))
    csv_files = list((root / "input").glob("*.csv"))
    data_files = txt_files + csv_files
    for data_file in data_files:
        # 使用 UTF-8 编码打开文件并读取内容
        with data_file.open(encoding="utf8") as f:
            text = f.read()
        # 构建文件在 Azure 存储中的路径
        file_path = (
            str(Path(input_base_dir) / data_file.name)
            ifath(input_base_dir) / data_file.name)
            if input_base_dir
            else data_file.name
        )
        
        # 异步上传文件到 Blob 存储，指定 UTF-8 编码
        await input_storage.set(file_path, text, encoding="utf-8")
    
    # 返回一个 lambda 函数，用于删除整个容器及其内容
    return lambda: input_storage.delete_container()
# 定义 TestIndexer 类，用于测试索引器功能
class TestIndexer:
    # 类变量 params，包含测试数据和配置，使用了类型注解说明数据结构
    params: ClassVar[dict[str, list[tuple[str, dict[str, Any]]]]] = {
        "test_fixture": _load_fixtures()  # 调用 _load_fixtures() 函数加载测试数据
    }

    # 私有方法 __run_indexer，运行索引器命令
    def __run_indexer(
        self,
        root: Path,  # 根路径对象
        input_file_type: str,  # 输入文件类型
    ):
        # 构建命令行参数列表
        command = [
            "poetry",
            "run",
            "poe",
            "index",
            "--verbose" if debug else None,  # 添加 verbose 参数如果 debug 为真
            "--root",
            root.absolute().as_posix(),  # 获取根路径的绝对路径并转为字符串
            "--reporter",
            "print",  # 设置报告器为 print
        ]
        command = [arg for arg in command if arg]  # 过滤掉空参数
        log.info("running command ", " ".join(command))  # 记录运行的命令
        # 使用 subprocess 模块运行命令，设置环境变量 GRAPHRAG_INPUT_FILE_TYPE
        completion = subprocess.run(
            command, env={**os.environ, "GRAPHRAG_INPUT_FILE_TYPE": input_file_type}
        )
        # 断言索引器返回码为 0，否则抛出异常
        assert (
            completion.returncode == 0
        ), f"Indexer failed with return code: {completion.returncode}"

    # 私有方法 __assert_indexer_outputs，断言索引器的输出
    def __assert_indexer_outputs(
        self, root: Path, workflow_config: dict[str, dict[str, Any]]
    ):
        # 方法体未完整给出，缺少实现细节

    # 私有方法 __run_query，运行查询命令
    def __run_query(self, root: Path, query_config: dict[str, str]):
        # 构建命令行参数列表
        command = [
            "poetry",
            "run",
            "poe",
            "query",
            "--root",
            root.absolute().as_posix(),  # 获取根路径的绝对路径并转为字符串
            "--method",
            query_config["method"],  # 设置查询方法
            "--community_level",
            str(query_config.get("community_level", 2)),  # 获取社区级别，默认为 2
            query_config["query"],  # 查询内容
        ]

        log.info("running command ", " ".join(command))  # 记录运行的命令
        # 使用 subprocess 模块运行命令，捕获输出并以文本形式返回
        return subprocess.run(command, capture_output=True, text=True)

    # 测试方法 test_fixture，使用装饰器配置测试环境和参数
    @cleanup(skip=debug)  # 如果 debug 为真则跳过清理
    @mock.patch.dict(
        os.environ,  # 使用 mock.patch.dict 装饰器模拟环境变量
        {
            **os.environ,  # 复制当前环境变量
            "BLOB_STORAGE_CONNECTION_STRING": os.getenv(
                "GRAPHRAG_CACHE_CONNECTION_STRING", WELL_KNOWN_AZURITE_CONNECTION_STRING
            ),  # 设置 BLOB_STORAGE_CONNECTION_STRING
            "LOCAL_BLOB_STORAGE_CONNECTION_STRING": WELL_KNOWN_AZURITE_CONNECTION_STRING,
            "GRAPHRAG_CHUNK_SIZE": "1200",  # 设置数据块大小为 1200
            "GRAPHRAG_CHUNK_OVERLAP": "0",  # 设置数据块重叠为 0
            "AZURE_AI_SEARCH_URL_ENDPOINT": os.getenv("AZURE_AI_SEARCH_URL_ENDPOINT"),  # 设置 Azure AI 搜索 URL
            "AZURE_AI_SEARCH_API_KEY": os.getenv("AZURE_AI_SEARCH_API_KEY"),  # 设置 Azure AI 搜索 API 密钥
        },
        clear=True,  # 清除当前环境变量
    )
    @pytest.mark.timeout(600)  # 设置测试超时时间为 600 秒（10 分钟）
    def test_fixture(
        self,
        input_path: str,  # 输入路径
        input_file_type: str,  # 输入文件类型
        workflow_config: dict[str, dict[str, Any]],  # 工作流配置
        query_config: list[dict[str, str]],  # 查询配置列表
    ):
        ):
            # 检查是否需要跳过此次 smoke test
            if workflow_config.get("skip", False):
                # 如果需要跳过，打印跳过消息并返回
                print(f"skipping smoke test {input_path})")
                return

            # 获取 Azure 配置信息
            azure = workflow_config.get("azure")
            # 解析输入路径为 Path 对象
            root = Path(input_path)
            # 初始化 dispose 为 None，用于存储清理函数
            dispose = None
            # 如果存在 Azure 配置，则运行异步函数准备 Azurite 数据
            if azure is not None:
                dispose = asyncio.run(prepare_azurite_data(input_path, azure))

            # 打印运行索引器消息
            print("running indexer")
            # 调用私有方法 __run_indexer，执行索引操作
            self.__run_indexer(root, input_file_type)
            # 打印索引器完成消息
            print("indexer complete")

            # 如果 dispose 不为 None，则执行清理操作
            if dispose is not None:
                dispose()

            # 如果不需要跳过断言步骤
            if not workflow_config.get("skip_assert", False):
                # 打印执行数据集断言消息
                print("performing dataset assertions")
                # 调用私有方法 __assert_indexer_outputs，进行断言检查
                self.__assert_indexer_outputs(root, workflow_config)

            # 打印运行查询消息
            print("running queries")
            # 遍历查询配置列表，依次执行查询
            for query in query_config:
                # 调用私有方法 __run_query，执行查询并获取结果
                result = self.__run_query(root, query)
                # 打印查询及其响应信息
                print(f"Query: {query}\nResponse: {result.stdout}")

                # 检查 stderr，因为 lancedb 将路径创建日志记录为 WARN，可能导致误报
                stderror = (
                    result.stderr if "No existing dataset at" not in result.stderr else ""
                )

                # 断言，确保 stderr 为空，否则输出错误信息
                assert stderror == "", f"Query failed with error: {stderror}"
                # 断言，确保查询结果 stdout 不为 None
                assert result.stdout is not None, "Query returned no output"
                # 断言，确保查询结果 stdout 非空
                assert len(result.stdout) > 0, "Query returned empty output"
```