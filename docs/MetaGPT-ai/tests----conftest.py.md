# `MetaGPT\tests\conftest.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/1 12:10
@Author  : alexanderwu
@File    : conftest.py
"""

import asyncio  # 引入异步IO库
import json  # 引入json库
import logging  # 引入日志库
import os  # 引入操作系统库
import re  # 引入正则表达式库
import uuid  # 引入uuid库

import pytest  # 引入pytest库

from metagpt.config import CONFIG, Config  # 从metagpt.config中引入CONFIG和Config
from metagpt.const import DEFAULT_WORKSPACE_ROOT, TEST_DATA_PATH  # 从metagpt.const中引入DEFAULT_WORKSPACE_ROOT和TEST_DATA_PATH
from metagpt.llm import LLM  # 从metagpt.llm中引入LLM
from metagpt.logs import logger  # 从metagpt.logs中引入logger
from metagpt.utils.git_repository import GitRepository  # 从metagpt.utils.git_repository中引入GitRepository
from tests.mock.mock_llm import MockLLM  # 从tests.mock.mock_llm中引入MockLLM

RSP_CACHE_NEW = {}  # 用于生成新的有用响应缓存的全局变量
ALLOW_OPENAI_API_CALL = os.environ.get(
    "ALLOW_OPENAI_API_CALL", True
)  # 注意：一旦模拟完成，应更改为默认值False

@pytest.fixture(scope="session")
def rsp_cache():
    # model_version = CONFIG.openai_api_model
    rsp_cache_file_path = TEST_DATA_PATH / "rsp_cache.json"  # 读取提供的存储响应缓存的文件路径
    new_rsp_cache_file_path = TEST_DATA_PATH / "rsp_cache_new.json"  # 导出一个新的副本
    if os.path.exists(rsp_cache_file_path):
        with open(rsp_cache_file_path, "r") as f1:
            rsp_cache_json = json.load(f1)
    else:
        rsp_cache_json = {}
    yield rsp_cache_json
    with open(rsp_cache_file_path, "w") as f2:
        json.dump(rsp_cache_json, f2, indent=4, ensure_ascii=False)
    with open(new_rsp_cache_file_path, "w") as f2:
        json.dump(RSP_CACHE_NEW, f2, indent=4, ensure_ascii=False)

# Hook to capture the test result
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        item.test_outcome = rep

@pytest.fixture(scope="function", autouse=True)
def llm_mock(rsp_cache, mocker, request):
    llm = MockLLM(allow_open_api_call=ALLOW_OPENAI_API_CALL)
    llm.rsp_cache = rsp_cache
    mocker.patch("metagpt.provider.base_llm.BaseLLM.aask", llm.aask)
    mocker.patch("metagpt.provider.base_llm.BaseLLM.aask_batch", llm.aask_batch)
    yield mocker
    if hasattr(request.node, "test_outcome") and request.node.test_outcome.passed:
        if llm.rsp_candidates:
            for rsp_candidate in llm.rsp_candidates:
                cand_key = list(rsp_candidate.keys())[0]
                cand_value = list(rsp_candidate.values())[0]
                if cand_key not in llm.rsp_cache:
                    logger.info(f"Added '{cand_key[:100]} ... -> {cand_value[:20]} ...' to response cache")
                    llm.rsp_cache.update(rsp_candidate)
                RSP_CACHE_NEW.update(rsp_candidate)

class Context:
    def __init__(self):
        self._llm_ui = None
        self._llm_api = LLM(provider=CONFIG.get_default_llm_provider_enum())

    @property
    def llm_api(self):
        # 1. 初始化llm，带有缓存结果
        # 2. 如果缓存query，那么直接返回缓存结果
        # 3. 如果没有缓存query，那么调用llm_api，返回结果
        # 4. 如果有缓存query，那么更新缓存结果
        return self._llm_api

@pytest.fixture(scope="package")
def llm_api():
    logger.info("Setting up the test")
    _context = Context()

    yield _context.llm_api

    logger.info("Tearing down the test")

@pytest.fixture
def proxy():
    pattern = re.compile(
        rb"(?P<method>[a-zA-Z]+) (?P<uri>(\w+://)?(?P<host>[^\s\'\"<>\[\]{}|/:]+)(:(?P<port>\d+))?[^\s\'\"<>\[\]{}|]*) "
    )

    async def pipe(reader, writer):
        while not reader.at_eof():
            writer.write(await reader.read(2048))
        writer.close()

    async def handle_client(reader, writer):
        data = await reader.readuntil(b"\r\n\r\n")
        print(f"Proxy: {data}")  # 使用capfd fixture检查
        infos = pattern.match(data)
        host, port = infos.group("host"), infos.group("port")
        port = int(port) if port else 80
        remote_reader, remote_writer = await asyncio.open_connection(host, port)
        if data.startswith(b"CONNECT"):
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        else:
            remote_writer.write(data)
        await asyncio.gather(pipe(reader, remote_writer), pipe(remote_reader, writer))

    async def proxy_func():
        server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
        return server, "http://{}:{}".format(*server.sockets[0].getsockname())

    return proxy_func()

# see https://github.com/Delgan/loguru/issues/59#issuecomment-466591978
@pytest.fixture
def loguru_caplog(caplog):
    class PropogateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    logger.add(PropogateHandler(), format="{message}")
    yield caplog

# init & dispose git repo
@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown_git_repo(request):
    CONFIG.git_repo = GitRepository(local_path=DEFAULT_WORKSPACE_ROOT / f"unittest/{uuid.uuid4().hex}")
    CONFIG.git_reinit = True

    # Destroy git repo at the end of the test session.
    def fin():
        CONFIG.git_repo.delete_repository()

    # Register the function for destroying the environment.
    request.addfinalizer(fin)

@pytest.fixture(scope="session", autouse=True)
def init_config():
    Config()

@pytest.fixture(scope="function")
def new_filename(mocker):
    # NOTE: Mock new filename to make reproducible llm aask, should consider changing after implementing requirement segmentation
    mocker.patch("metagpt.utils.file_repository.FileRepository.new_filename", lambda: "20240101")
    yield mocker

@pytest.fixture
def aiohttp_mocker(mocker):
    class MockAioResponse:
        async def json(self, *args, **kwargs):
            return self._json

        def set_json(self, json):
            self._json = json

    response = MockAioResponse()

    class MockCTXMng:
        async def __aenter__(self):
            return response

        async def __aexit__(self, *args, **kwargs):
            pass

        def __await__(self):
            yield
            return response

    def mock_request(self, method, url, **kwargs):
        return MockCTXMng()

    def wrap(method):
        def run(self, url, **kwargs):
            return mock_request(self, method, url, **kwargs)

        return run

    mocker.patch("aiohttp.ClientSession.request", mock_request)
    for i in ["get", "post", "delete", "patch"]:
        mocker.patch(f"aiohttp.ClientSession.{i}", wrap(i))

    yield response

```