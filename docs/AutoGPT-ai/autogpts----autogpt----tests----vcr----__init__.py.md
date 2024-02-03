# `.\AutoGPT\autogpts\autogpt\tests\vcr\__init__.py`

```py
# 导入 logging 模块
import logging
# 导入 os 模块
import os
# 导入 hashlib 模块中的 sha256 函数
from hashlib import sha256

# 导入 pytest 模块
import pytest
# 导入 OpenAI 类
from openai import OpenAI
# 导入 FinalRequestOptions 类
from openai._models import FinalRequestOptions
# 导入 Omit 类
from openai._types import Omit
# 导入 is_given 函数
from openai._utils import is_given
# 导入 pytest_mock 模块中的 MockerFixture 类
from pytest_mock import MockerFixture

# 导入 vcr_filter 模块中的 PROXY、before_record_request、before_record_response、freeze_request_body 变量
from .vcr_filter import (
    PROXY,
    before_record_request,
    before_record_response,
    freeze_request_body,
)

# 设置默认录制模式为 "new_episodes"
DEFAULT_RECORD_MODE = "new_episodes"
# 基础 VCR 配置
BASE_VCR_CONFIG = {
    "before_record_request": before_record_request,
    "before_record_response": before_record_response,
    "filter_headers": [
        "Authorization",
        "AGENT-MODE",
        "AGENT-TYPE",
        "Cookie",
        "OpenAI-Organization",
        "X-OpenAI-Client-User-Agent",
        "User-Agent",
    ],
    "match_on": ["method", "headers"],
}

# 定义 vcr_config 会话级别的 fixture
@pytest.fixture(scope="session")
def vcr_config(get_base_vcr_config):
    return get_base_vcr_config

# 定义 get_base_vcr_config 会话级别的 fixture
@pytest.fixture(scope="session")
def get_base_vcr_config(request):
    # 获取录制模式，默认为 "new_episodes"
    record_mode = request.config.getoption("--record-mode", default="new_episodes")
    config = BASE_VCR_CONFIG

    # 如果录制模式为 None，则设置为默认录制模式
    if record_mode is None:
        config["record_mode"] = DEFAULT_RECORD_MODE

    return config

# 定义 vcr_cassette_dir fixture
@pytest.fixture()
def vcr_cassette_dir(request):
    # 获取测试用例名称
    test_name = os.path.splitext(request.node.name)[0]
    # 返回 VCR 卡带目录路径
    return os.path.join("tests/vcr_cassettes", test_name)

# 定义 cached_openai_client fixture
@pytest.fixture
def cached_openai_client(mocker: MockerFixture) -> OpenAI:
    # 创建 OpenAI 客户端对象
    client = OpenAI()
    # 备份客户端的 _prepare_options 方法
    _prepare_options = client._prepare_options
    # 定义一个内部方法，用于准备请求选项
    def _patched_prepare_options(self, options: FinalRequestOptions):
        # 调用_prepare_options方法，准备请求选项
        _prepare_options(options)

        # 初始化一个字典类型的headers变量，如果options中包含headers，则复制一份，否则为空字典
        headers: dict[str, str | Omit] = (
            {**options.headers} if is_given(options.headers) else {}
        )
        # 将headers赋值给options的headers属性
        options.headers = headers
        # 初始化一个data变量，存储options中的json_data
        data: dict = options.json_data

        # 如果存在代理PROXY
        if PROXY:
            # 设置headers中的"AGENT-MODE"为环境变量"AGENT_MODE"的值，如果不存在则为Omit()
            headers["AGENT-MODE"] = os.environ.get("AGENT_MODE", Omit())
            # 设置headers中的"AGENT-TYPE"为环境变量"AGENT_TYPE"的值，如果不存在则为Omit()
            headers["AGENT-TYPE"] = os.environ.get("AGENT_TYPE", Omit())

        # 记录日志，输出请求头和数据
        logging.getLogger("cached_openai_client").debug(
            f"Outgoing API request: {headers}\n{data if data else None}"
        )

        # 为快速匹配录音带播放添加哈希头
        # 计算请求数据的哈希值，设置到headers中的"X-Content-Hash"中
        headers["X-Content-Hash"] = sha256(
            freeze_request_body(data), usedforsecurity=False
        ).hexdigest()

    # 如果存在代理PROXY，则设置client的base_url为"{PROXY}/v1"
    if PROXY:
        client.base_url = f"{PROXY}/v1"
    # 使用mocker.patch.object方法，将client的_prepare_options方法替换为_patched_prepare_options方法
    mocker.patch.object(
        client,
        "_prepare_options",
        new=_patched_prepare_options,
    )

    # 返回client对象
    return client
```