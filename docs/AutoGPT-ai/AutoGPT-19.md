# AutoGPT源码解析 19

# `autogpts/autogpt/tests/unit/models/test_base_open_api_plugin.py`

这段代码的作用是定义了一个名为 `DummyPlugin` 的自定义插件类，该插件实现了 `BaseOpenAIPlugin` 的接口。这个插件插在名为 `test_pytest` 的测试套件中，用于为测试提供一些自定义的行为和功能。

具体来说，这段代码做以下几件事情：

1. 引入了 `pytest` 模块，以便用于编写测试。
2. 从 `autogpt.models.base_open_ai_plugin` 类中继承了一个名为 `BaseOpenAIPlugin` 的类，并覆盖了其 `__init__` 方法，用于初始化一些默认的行为。
3. 在自定义插件类中定义了一个名为 `DummyPlugin` 的类，覆盖了 `BaseOpenAIPlugin` 类中的 `manifests_specs_clients` 属性，用于定义插件需要支持的几种 OpenAPI 规格。
4. 在 `DummyPlugin` 类中，重写了 `__init__` 方法，用于初始化时检查 OpenAPI 规格并执行相应的操作。
5. 在 `dummy_plugin` 函数中，创建了一个新的 `DummyPlugin` 实例，并返回该实例，以便在测试中使用。


```py
import pytest

from autogpt.models.base_open_ai_plugin import BaseOpenAIPlugin


class DummyPlugin(BaseOpenAIPlugin):
    """A dummy plugin for testing purposes."""


@pytest.fixture
def dummy_plugin():
    """A dummy plugin for testing purposes."""
    manifests_specs_clients = {
        "manifest": {
            "name_for_model": "Dummy",
            "schema_version": "1.0",
            "description_for_model": "A dummy plugin for testing purposes",
        },
        "client": None,
        "openapi_spec": None,
    }
    return DummyPlugin(manifests_specs_clients)


```



这段代码是一个测试用例，用于测试 DummyPlugin 类是否继承自 BaseOpenAIPPlugin 类。该代码包括三个函数，每个函数使用 assert 语句来验证是否符合预期。如果不符合，则会输出具体原因，否则则输出 "Success!"。

具体来说，第一个函数 `test_dummy_plugin_inheritance` 是测试 DummyPlugin 类是否继承自 BaseOpenAIPPlugin 类，该函数使用 `assertIsInstance` 函数来验证是否为 `BaseOpenAIPPlugin` 类。如果验证失败，则会输出一个错误消息并退出。如果验证成功，则会继续执行下一个函数。

第二个函数 `test_dummy_plugin_name` 是测试 DummyPlugin 类是否具有正确的名称，该函数使用 `assertIsInstance` 函数来验证是否为 `BaseOpenAIPPlugin` 类，并使用 `_name` 属性来获取名称。如果验证失败，则会输出一个错误消息并退出。如果验证成功，则会继续执行下一个函数。

第三个函数 `test_dummy_plugin_version` 是测试 DummyPlugin 类是否具有正确的版本号，该函数使用 `assertIsInstance` 函数来验证是否为 `BaseOpenAIPPlugin`` 类，并使用 `_version` 属性来获取版本号。如果验证失败，则会输出一个错误消息并退出。如果验证成功，则会继续执行下一个函数。


```py
def test_dummy_plugin_inheritance(dummy_plugin):
    """Test that the DummyPlugin class inherits from the BaseOpenAIPlugin class."""
    assert isinstance(dummy_plugin, BaseOpenAIPlugin)


def test_dummy_plugin_name(dummy_plugin):
    """Test that the DummyPlugin class has the correct name."""
    assert dummy_plugin._name == "Dummy"


def test_dummy_plugin_version(dummy_plugin):
    """Test that the DummyPlugin class has the correct version."""
    assert dummy_plugin._version == "1.0"


```

This is a Python test case for the DummyPlugin class, which is a Python class designed to handle natural language text messages and perform actions in a virtual robot.

The test case starts by checking that the plugin cannot handle post-instruction, pre-command, post-command, and chat-completion tasks. It then checks that the plugin responds correctly to different inputs, including a response to the "hello" command and text embeddings.

Finally, the test case checks that the plugin correctly handles post-instruction, post-command, and chat-completion tasks.


```py
def test_dummy_plugin_description(dummy_plugin):
    """Test that the DummyPlugin class has the correct description."""
    assert dummy_plugin._description == "A dummy plugin for testing purposes"


def test_dummy_plugin_default_methods(dummy_plugin):
    """Test that the DummyPlugin class has the correct default methods."""
    assert not dummy_plugin.can_handle_on_response()
    assert not dummy_plugin.can_handle_post_prompt()
    assert not dummy_plugin.can_handle_on_planning()
    assert not dummy_plugin.can_handle_post_planning()
    assert not dummy_plugin.can_handle_pre_instruction()
    assert not dummy_plugin.can_handle_on_instruction()
    assert not dummy_plugin.can_handle_post_instruction()
    assert not dummy_plugin.can_handle_pre_command()
    assert not dummy_plugin.can_handle_post_command()
    assert not dummy_plugin.can_handle_chat_completion(None, None, None, None)
    assert not dummy_plugin.can_handle_text_embedding(None)

    assert dummy_plugin.on_response("hello") == "hello"
    assert dummy_plugin.post_prompt(None) is None
    assert dummy_plugin.on_planning(None, None) is None
    assert dummy_plugin.post_planning("world") == "world"
    pre_instruction = dummy_plugin.pre_instruction(
        [{"role": "system", "content": "Beep, bop, boop"}]
    )
    assert isinstance(pre_instruction, list)
    assert len(pre_instruction) == 1
    assert pre_instruction[0]["role"] == "system"
    assert pre_instruction[0]["content"] == "Beep, bop, boop"
    assert dummy_plugin.on_instruction(None) is None
    assert dummy_plugin.post_instruction("I'm a robot") == "I'm a robot"
    pre_command = dummy_plugin.pre_command("evolve", {"continuously": True})
    assert isinstance(pre_command, tuple)
    assert len(pre_command) == 2
    assert pre_command[0] == "evolve"
    assert pre_command[1]["continuously"] == True
    post_command = dummy_plugin.post_command("evolve", "upgraded successfully!")
    assert isinstance(post_command, str)
    assert post_command == "upgraded successfully!"
    assert dummy_plugin.handle_chat_completion(None, None, None, None) is None
    assert dummy_plugin.handle_text_embedding(None) is None

```

# `autogpts/autogpt/tests/vcr/vcr_filter.py`

这段代码使用了多个第三方库，包括 contextlib、json、os、re、以及 htmlclick。它的主要目的是从 BytesIO 对象中读取 JSON 文件内容，并使用正则表达式进行替换。

具体来说，这段代码的作用是：

1. 导入需要的库
2. 从 PROXY 环境变量中获取后，创建一个 BytesIO 对象
3. 使用 vcr.request.Request 类从文件中读取 JSON 内容
4. 使用正则表达式替换 REPLACEMENTS 列表中的所有内容，这些内容通常包含 HTML 页面中的链接
5. 将替换后的内容写回 BytesIO 对象
6. 最后，将 BytesIO 对象关闭

REPLACEMENTS 列表中的每个元素都表示了一个匹配项，它由一个正则表达式、一个或多个 replacement 字段和一个或多个路径参数组成。正则表达式用于匹配 HTML 页面中的链接，而 replacement 字段用于替换匹配到的链接。路径参数用于将匹配到的链接替换为新的内容。


```py
import contextlib
import json
import os
import re
from io import BytesIO
from typing import Any, Dict, List

from vcr.request import Request

PROXY = os.environ.get("PROXY")

REPLACEMENTS: List[Dict[str, str]] = [
    {
        "regex": r"\w{3} \w{3} {1,2}\d{1,2} \d{2}:\d{2}:\d{2} \d{4}",
        "replacement": "Tue Jan  1 00:00:00 2000",
    },
    {
        "regex": r"<selenium.webdriver.chrome.webdriver.WebDriver[^>]*>",
        "replacement": "",
    },
]

```

这段代码的作用是设置一个名为 "ALLOWED_HOSTNAMES" 的列表，包含多个允许的域名，用于设置代理的设置。同时，它还设置了一个名为 "PROXY" 的变量，如果当前设置了代理，则将 "PROXY" 的值添加到 "ALLOWED_HOSTNAMES" 中，否则将 "ORIGINAL_URL" 的值设置为 "no_ci"。最后，它将 "NEW_URL" 设置为 "api.openai.com"。


```py
ALLOWED_HOSTNAMES: List[str] = [
    "api.openai.com",
    "localhost:50337",
    "duckduckgo.com",
]

if PROXY:
    ALLOWED_HOSTNAMES.append(PROXY)
    ORIGINAL_URL = PROXY
else:
    ORIGINAL_URL = "no_ci"

NEW_URL = "api.openai.com"


```

这两段代码都是Python中的函数，它们的作用是：

1. replace_message_content：该函数接收一个字符串content，以及一个列表replacements，该列表包含两个字典，第一个字典包含一个正则表达式，第二个字典包含一个字符串replacement。该函数通过使用replacements中的正则表达式和replacement字典中的replacement字符串，在content中查找匹配正则表达式的匹配项，并替换掉匹配项。最后，函数返回替换后的内容。

2. freeze_request_body：该函数接收一个JSON或字节数据，并将其转换为Python中的bytes类型。该函数可以去除请求体中任何动态的数据，如列表、字典和字段等。如果输入的JSON数据存在格式错误，该函数会将其转换为bytes类型并返回。

replace_message_content函数的作用是，当从请求的JSON数据中获取到消息内容时，根据replacements列表中的regex字符串和replacement字典中的replacement字符串，对消息内容进行正则表达式替换，以符合要求并返回替换后的内容。

freeze_request_body函数的作用是，将请求体中的JSON数据转换为bytes类型，并去除其中的动态数据。如果输入的JSON数据存在格式错误，该函数会将其转换为bytes类型并返回。


```py
def replace_message_content(content: str, replacements: List[Dict[str, str]]) -> str:
    for replacement in replacements:
        pattern = re.compile(replacement["regex"])
        content = pattern.sub(replacement["replacement"], content)

    return content


def freeze_request_body(json_body: str | bytes) -> bytes:
    """Remove any dynamic items from the request body"""

    try:
        body = json.loads(json_body)
    except ValueError:
        return json_body if type(json_body) == bytes else json_body.encode()

    if "messages" not in body:
        return json.dumps(body, sort_keys=True).encode()

    if "max_tokens" in body:
        del body["max_tokens"]

    for message in body["messages"]:
        if "content" in message and "role" in message:
            if message["role"] == "system":
                message["content"] = replace_message_content(
                    message["content"], REPLACEMENTS
                )

    return json.dumps(body, sort_keys=True).encode()


```

这两函数是使用Python编写的网络请求工具，其主要作用是保护应用程序免受一些潜在的错误。

首先，定义了一个名为 `freeze_request` 的函数，其接收一个 HTTP 请求对象（Request）作为参数，并返回经过处理后的请求对象。

这个函数的作用是通过创建一个名为 `freeze_request_body` 的函数，将输入的请求主体（body）进行处理，并返回一个新的请求主体。处理的方式是在函数内部使用 `with contextlib.suppress(ValueError)` 语法来捕获可能抛出的 `ValueError` 异常。

如果输入的请求主体已经是 BytesIO 类型，那么先将其转换成正常的字符串，否则不做任何处理。

然后，返回处理后的请求主体，并将其作为参数传入到 `freeze_request` 函数中，最终返回处理后的请求对象。

接下来，定义了一个名为 `before_record_response` 的函数，其接收一个 HTTP 响应对象（Response）作为参数，并返回一个新的响应对象。

这个函数的作用是在响应头中查找 "Transfer-Encoding" 字段，并将其删除。这个字段通常在 HTTP 响应头中出现，它指定了 HTTP 响应的内容在传输过程中采用的编码方式。

经过测试，正确的作用应该是：

1. 如果 "Transfer-Encoding" 字段不存在，则不做任何处理。
2. 如果 "Transfer-Encoding" 字段存在，则将其删除。


```py
def freeze_request(request: Request) -> Request:
    if not request or not request.body:
        return request

    with contextlib.suppress(ValueError):
        request.body = freeze_request_body(
            request.body.getvalue()
            if isinstance(request.body, BytesIO)
            else request.body
        )

    return request


def before_record_response(response: Dict[str, Any]) -> Dict[str, Any]:
    if "Transfer-Encoding" in response["headers"]:
        del response["headers"]["Transfer-Encoding"]
    return response


```

这段代码定义了一个名为 "before_record_request" 的函数，用于在发起请求之前对请求进行处理。函数接收一个 "Request" 对象作为参数，并返回一个 "Request" 对象或 None。

函数的主要作用是先对请求的原始URL进行修改，使得新的请求URL符合要求，然后对请求进行去动态数据操作，最后返回处理后的 "Request" 对象。

具体来说，函数首先调用一个名为 "replace_request_hostname" 的函数，该函数接收一个 "Request" 对象和两个字符串参数 "original_url" 和 "new_hostname"，并返回修改后的 "Request" 对象。这个函数的作用是将要修改的请求的 "hostname" 属性替换为新的 "new_hostname"。

然后，函数调用一个名为 "filter_hostnames" 的函数，该函数接收一个 "Request" 对象作为参数，并返回一个筛选后的 "Request" 对象，该对象中的所有 "hostname" 属性被保留。这个函数的作用是对请求进行主机名筛选，只返回原始请求的主机名，而忽略 "proxy_function" 动态数据操作。

最后，函数调用一个名为 "freeze_request" 的函数，该函数接收一个 "Request" 对象作为参数，并返回一个 Freeze 版本的 "Request" 对象，该对象序列化后不可见任何修改操作。这个函数的作用是使得请求对象在将来的某个时间点被序列化之前不可见任何修改操作，有助于防止在请求序列化后执行额外的操作。

最终，函数返回修改后的或 Freeze 版本的 "Request" 对象，根据 "Request" 对象是否有效(即，修改后的 "Request" 对象是否仍然符合期望)来决定返回哪个值。


```py
def before_record_request(request: Request) -> Request | None:
    request = replace_request_hostname(request, ORIGINAL_URL, NEW_URL)

    filtered_request = filter_hostnames(request)
    if not filtered_request:
        return None

    filtered_request_without_dynamic_data = freeze_request(filtered_request)
    return filtered_request_without_dynamic_data


from urllib.parse import urlparse, urlunparse


def replace_request_hostname(
    request: Request, original_url: str, new_hostname: str
) -> Request:
    parsed_url = urlparse(request.uri)

    if parsed_url.hostname in original_url:
        new_path = parsed_url.path.replace("/proxy_function", "")
        request.uri = urlunparse(
            parsed_url._replace(netloc=new_hostname, path=new_path, scheme="https")
        )

    return request


```

这段代码定义了一个名为 `filter_hostnames` 的函数，用于对请求中的主机名进行过滤。函数接收一个名为 `request` 的请求对象作为参数，并返回一个经过过滤后的请求对象或者 `None`。

函数内部的实现为：首先，定义了一个 `ALLOWED_HOSTNAMES` 列表，用于存储允许包含在主机名中的字符串列表。接着，使用一个循环来遍历 `request.url` 中的每个主机名，并将其与 `ALLOWED_HOSTNAMES` 中的主机名进行比较。如果当前主机名在 `ALLOWED_HOSTNAMES` 中，则返回请求对象；否则，返回 `None`。

这段代码的作用是用于对请求中的主机名进行匹配和过滤，以便只有允许的主机名可以被接受并传递下去。


```py
def filter_hostnames(request: Request) -> Request | None:
    # Add your implementation here for filtering hostnames
    if any(hostname in request.url for hostname in ALLOWED_HOSTNAMES):
        return request
    else:
        return None

```

# `autogpts/autogpt/tests/vcr/__init__.py`

这段代码使用了Python的一些标准库函数和第三方库，其作用是进行VRR（Virtual Room Real-time）技术的开发。下面是具体的解释：

1. `import logging`：这是Python的一个标准库，用于引入日志记录的功能。
2. `import os`：这也是Python的一个标准库，用于引入操作系统功能的支持。
3. `from hashlib import sha256`：这是一个第三方库，用于实现哈希算法，生成SHA-256签名。
4. `import openai.api_requestor`：这也是一个第三方库，用于与OpenAI的API进行交互。
5. `import pytest`：这是Python的一个标准库，用于支持pytest测试框架。
6. `from pytest_mock import MockerFixture`：这是pytest的一个支持，用于在测试过程中模拟Mocker环境。
7. `from .vcr_filter import ( PROXY, before_record_request, before_record_response, freeze_request_body )`：这是VRR技术的核心部分，包括了代理、请求和响应的过滤、请求体冻结等功能。


```py
import logging
import os
from hashlib import sha256

import openai.api_requestor
import pytest
from pytest_mock import MockerFixture

from .vcr_filter import (
    PROXY,
    before_record_request,
    before_record_response,
    freeze_request_body,
)

```

这段代码是一个Python配置文件中的代码片段，配置了一个VCR(Video Cassette Recorder)的BASE_VCR_CONFIG。

具体来说，代码定义了一个DEFAULT_RECORD_MODE变量，其值为"new_episodes"，表示使用新节目模式。

接着，定义了一个名为BASE_VCR_CONFIG的字典，其中包含了一些配置项。具体来说，BASE_VCR_CONFIG包含以下配置项：

- before_record_request：一个函数，用于在请求录制之前执行的操作。
- before_record_response：一个函数，用于在响应录制之前执行的操作。
- filter_headers：一个列表，其中包含了一些过滤头部的规则。这些规则可以被附加到请求或响应的头部，以便选择性地排除某些流量。
- match_on：一个列表，其中包含了一些匹配数据以过滤请求和响应的配置项。

这些配置项可以用来控制VCR的录制行为，例如选择要录制的内容、在录制前进行筛选、选择录制方式等。


```py
DEFAULT_RECORD_MODE = "new_episodes"
BASE_VCR_CONFIG = {
    "before_record_request": before_record_request,
    "before_record_response": before_record_response,
    "filter_headers": [
        "Authorization",
        "AGENT-MODE",
        "AGENT-TYPE",
        "OpenAI-Organization",
        "X-OpenAI-Client-User-Agent",
        "User-Agent",
    ],
    "match_on": ["method", "headers"],
}


```

这段代码使用了Python中的pytest库来进行断言测试，并定义了两个fixture，用于模拟和控制对VCR的配置。

`@pytest.fixture(scope="session")`是一个用于声明测试函数fixture的方法，用于将一个函数的作用域限定为测试函数所在的会话中。在本例中，该方法定义了一个名为`vcr_config`的fixture，它的作用域为`scope="session"`，这意味着该fixture将整个测试函数作为一个参数传递，并且该fixture将保持其作用域在测试函数会话中。

`def vcr_config(get_base_vcr_config):`是用于定义`vcr_config`fixture的方法。该方法使用`get_base_vcr_config`参数获取基础VCR配置，并将其返回。该方法的`getoption`函数用于从用户输入中读取`--record-mode`选项的值，如果没有该选项，则使用默认值`new_episodes`。然后，该方法将`BASE_VCR_CONFIG`作为基础VCR配置，并根据`record_mode`选项的值更新该配置。最后，该方法返回修改后的基础VCR配置。

`@pytest.fixture(scope="session")`另一个用于定义`get_base_vcr_config`fixture的方法，与上面类似，但它的作用域为测试函数之外的全局作用域。该方法使用`request`参数获取测试函数的配置，并将其返回。在本例中，该方法使用`getoption`函数从用户输入中读取`--record-mode`选项的值，如果没有该选项，则使用默认值`new_episodes`。然后，该方法将`BASE_VCR_CONFIG`作为基础VCR配置，并根据`record_mode`选项的值更新该配置。最后，该方法返回修改后的基础VCR配置。


```py
@pytest.fixture(scope="session")
def vcr_config(get_base_vcr_config):
    return get_base_vcr_config


@pytest.fixture(scope="session")
def get_base_vcr_config(request):
    record_mode = request.config.getoption("--record-mode", default="new_episodes")
    config = BASE_VCR_CONFIG

    if record_mode is None:
        config["record_mode"] = DEFAULT_RECORD_MODE

    return config


```

It looks like you are providing example code that uses the `openai.api_requestor` module to interact with the Azure Virtual Canary Service (VCS). The code defines a `patched_api_requestor` fixture that patches the `APIRequestor` class to include a hash header for improved matching during playback.

The `patched_api_requestor` fixture takes an `openai.api_requestor.APIRequestor` object as its argument and returns a `requestor` object that has been patched to include the hash header. The `openai.api_requestor.APIRequestor` class is the main class for interacting with the VCS, and the `patched_api_requestor` fixture allows for easier testing and manipulation of the API requests.

The `patched_api_requestor` fixture is implemented as a Python fixture using the `MockerFixture` pattern. This allows for the use of the fixture to manage the resources and lifecycle of the objects it depends on. The `MockerFixture` is used to create an instance of the `patched_api_requestor` fixture, which is then used to patch the `APIRequestor` class.


```py
@pytest.fixture()
def vcr_cassette_dir(request):
    test_name = os.path.splitext(request.node.name)[0]
    return os.path.join("tests/vcr_cassettes", test_name)


def patch_api_base(requestor: openai.api_requestor.APIRequestor):
    new_api_base = f"{PROXY}/v1"
    requestor.api_base = new_api_base
    return requestor


@pytest.fixture
def patched_api_requestor(mocker: MockerFixture):
    init_requestor = openai.api_requestor.APIRequestor.__init__
    prepare_request = openai.api_requestor.APIRequestor._prepare_request_raw

    def patched_init_requestor(requestor, *args, **kwargs):
        init_requestor(requestor, *args, **kwargs)
        patch_api_base(requestor)

    def patched_prepare_request(self, *args, **kwargs):
        url, headers, data = prepare_request(self, *args, **kwargs)

        if PROXY:
            headers["AGENT-MODE"] = os.environ.get("AGENT_MODE")
            headers["AGENT-TYPE"] = os.environ.get("AGENT_TYPE")

        logging.getLogger("patched_api_requestor").debug(
            f"Outgoing API request: {headers}\n{data.decode() if data else None}"
        )

        # Add hash header for cheap & fast matching on cassette playback
        headers["X-Content-Hash"] = sha256(
            freeze_request_body(data), usedforsecurity=False
        ).hexdigest()

        return url, headers, data

    if PROXY:
        mocker.patch.object(
            openai.api_requestor.APIRequestor,
            "__init__",
            new=patched_init_requestor,
        )
    mocker.patch.object(
        openai.api_requestor.APIRequestor,
        "_prepare_request_raw",
        new=patched_prepare_request,
    )

```

# 🚀 **AutoGPT-Forge**: Build Your Own AutoGPT Agent! 🧠 

### 🌌 Dive into the Universe of AutoGPT Creation! 🌌

Ever dreamt of becoming the genius behind an AI agent? Dive into the *Forge*, where **you** become the creator!

---

### 🛠️ **Why AutoGPT-Forge?**
- 💤 **No More Boilerplate!** Don't let the mundane tasks stop you. Fork and build without the headache of starting from scratch!
- 🧠 **Brain-centric Development!** All the tools you need so you can spend 100% of your time on what matters - crafting the brain of your AI!
- 🛠️ **Tooling ecosystem!** We work with the best in class tools to bring you the best experience possible!
---

### 🚀 **Get Started!**

The getting started [tutorial series](https://aiedge.medium.com/autogpt-forge-e3de53cc58ec) will guide you through the process of setting up your project all the way through to building a generalist agent.  

1. [AutoGPT Forge: A Comprehensive Guide to Your First Steps](https://aiedge.medium.com/autogpt-forge-a-comprehensive-guide-to-your-first-steps-a1dfdf46e3b4)
2. [AutoGPT Forge: The Blueprint of an AI Agent](https://aiedge.medium.com/autogpt-forge-the-blueprint-of-an-ai-agent-75cd72ffde6)
3. [AutoGPT Forge: Interacting with your Agent](https://aiedge.medium.com/autogpt-forge-interacting-with-your-agent-1214561b06b)
4. [AutoGPT Forge: Crafting Intelligent Agent Logic](https://medium.com/@aiedge/autogpt-forge-crafting-intelligent-agent-logic-bc5197b14cb4)




# `autogpts/forge/__init__.py`

很抱歉，我需要看到你的代码才能为你解释它的作用。请提供你的代码，我会尽力解释它的作用。


```py

```

Advanced commands to develop on the forge and the benchmark.
Stability not guaranteed.


# `autogpts/forge/forge/agent.py`

这段代码的作用是使用Forge SDK中的各个模块，实现一个智能对话系统。具体来说，它包括以下组件：

1. Agent: 定义了聊天机器人的基本行为，如获取用户输入、执行上下文处理等。
2. AgentDB: 实现了与数据库的交互，用于存储用户对话的历史记录。
3. ForgeLogger: 记录机器人行动的日志，可以方便地与后端服务器同步。
4. Step: 用于表示用户的每一个输入，通过StepRequestBody传递给Agent。
5. Task: 用于代表一个任务，可以包含多个Step。
6. TaskRequestBody: 定义了任务的请求体，包括用户输入的格式等。
7. Workspace: 保存了当前对话的工作空间，即对话历史。
8. PromptEngine: 实现了对话提示功能，根据用户输入给出合适的回复。
9. chat_completion_request: 实现了聊天完成标记的请求，用于结束对话。
10. ChromaMemStore: 用于存储聊天记录到内存中的数据结构。

通过这些组件，这段代码可以构建一个智能对话系统，可以与用户进行自然语言交互，完成各种任务。


```py
from forge.sdk import (
    Agent,
    AgentDB,
    ForgeLogger,
    Step,
    StepRequestBody,
    Task,
    TaskRequestBody,
    Workspace,    
    PromptEngine,	
    chat_completion_request,	
    ChromaMemStore	
)
import json	
import pprint

```

This is a Python class that implements the Forge agent protocol. The Forge is a benchmarking platform that allows users to create and execute tasks, such as solving青年主权的问题。The protocol is the core of the Forge, and it works by creating a task and then executing steps for that task.

The task that is created contains an input string, which is the task the agent has been asked to solve, as well as additional input, which is a dictionary and could contain anything.

If you want to get the task, you can use the `db.get_task(task_id)` method, which returns the task with the given ID.

The step request body is essentially the same as the task request and contains an input string, which is the task the agent has been asked to solve, as well as additional input, which is a dictionary and could contain anything.

You need to implement logic that will take in this step input and output the completed step as a step object. You can do everything in a single step or you can break it down into multiple steps. Returning a request to continue in the step output, the user can then decide if they want the agent to continue or not.

Note that the code also implements some logging and error handling.


```
LOG = ForgeLogger(__name__)


class ForgeAgent(Agent):
    """
    The goal of the Forge is to take care of the boilerplate code, so you can focus on
    agent design.

    There is a great paper surveying the agent landscape: https://arxiv.org/abs/2308.11432
    Which I would highly recommend reading as it will help you understand the possabilities.

    Here is a summary of the key components of an agent:

    Anatomy of an agent:
         - Profile
         - Memory
         - Planning
         - Action

    Profile:

    Agents typically perform a task by assuming specific roles. For example, a teacher,
    a coder, a planner etc. In using the profile in the llm prompt it has been shown to
    improve the quality of the output. https://arxiv.org/abs/2305.14688

    Additionally, based on the profile selected, the agent could be configured to use a
    different llm. The possibilities are endless and the profile can be selected
    dynamically based on the task at hand.

    Memory:

    Memory is critical for the agent to accumulate experiences, self-evolve, and behave
    in a more consistent, reasonable, and effective manner. There are many approaches to
    memory. However, some thoughts: there is long term and short term or working memory.
    You may want different approaches for each. There has also been work exploring the
    idea of memory reflection, which is the ability to assess its memories and re-evaluate
    them. For example, condensing short term memories into long term memories.

    Planning:

    When humans face a complex task, they first break it down into simple subtasks and then
    solve each subtask one by one. The planning module empowers LLM-based agents with the ability
    to think and plan for solving complex tasks, which makes the agent more comprehensive,
    powerful, and reliable. The two key methods to consider are: Planning with feedback and planning
    without feedback.

    Action:

    Actions translate the agent's decisions into specific outcomes. For example, if the agent
    decides to write a file, the action would be to write the file. There are many approaches you
    could implement actions.

    The Forge has a basic module for each of these areas. However, you are free to implement your own.
    This is just a starting point.
    """

    def __init__(self, database: AgentDB, workspace: Workspace):
        """
        The database is used to store tasks, steps and artifact metadata. The workspace is used to
        store artifacts. The workspace is a directory on the file system.

        Feel free to create subclasses of the database and workspace to implement your own storage
        """
        super().__init__(database, workspace)

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to create
        a task.

        We are hooking into function to add a custom log message. Though you can do anything you
        want here.
        """
        task = await super().create_task(task_request)
        LOG.info(
            f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )
        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        For a tutorial on how to add your own logic please see the offical tutorial series:
        https://aiedge.medium.com/autogpt-forge-e3de53cc58ec

        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to execute
        a step.

        The task that is created contains an input string, for the benchmarks this is the task
        the agent has been asked to solve and additional input, which is a dictionary and
        could contain anything.

        If you want to get the task use:

        ```py
        task = await self.db.get_task(task_id)
        ```

        The step request body is essentially the same as the task request and contains an input
        string, for the benchmarks this is the task the agent has been asked to solve and
        additional input, which is a dictionary and could contain anything.

        You need to implement logic that will take in this step input and output the completed step
        as a step object. You can do everything in a single step or you can break it down into
        multiple steps. Returning a request to continue in the step output, the user can then decide
        if they want the agent to continue or not.
        """
        # An example that
        step = await self.db.create_step(
            task_id=task_id, input=step_request, is_last=True
        )

        self.workspace.write(task_id=task_id, path="output.txt", data=b"Washington D.C")

        await self.db.create_artifact(
            task_id=task_id,
            step_id=step.step_id,
            file_name="output.txt",
            relative_path="",
            agent_created=True,
        )

        step.output = "Washington D.C"

        LOG.info(f"\t✅ Final Step completed: {step.step_id}. \n" +
                 f"Output should be placeholder text Washington D.C. You'll need to \n" +
                 f"modify execute_step to include LLM behavior. Follow the tutorial " +
                 f"if confused. ")

        return step

```py

# `autogpts/forge/forge/app.py`

这段代码的作用是使用Python和Forge SDK创建一个名为“MyAgentsApp”的Forge Agent应用程序，并使用指定的数据库。

具体来说，它首先通过`os.getenv()`函数获取当前工作目录中的数据库字符串，然后使用这个环境变量来创建一个本地数据库。接着，它使用`LocalWorkspace`类创建一个名为“MyAgentsWorkspace”的工作空间，并将数据库对象与工作空间绑定。

接下来，它使用`ForgeDatabase`类将本地数据库与指定的Forge Agent应用程序的数据库绑定，并使用`ForgeAgent`类创建一个Forge Agent实例，将该代理与上面创建的数据库绑定。

最后，它使用`agent.get_agent_app()`方法获取指定的Forge Agent应用程序，并将其用于将应用程序中的命令转换为Forge Agent可以理解的命令，并返回应用程序对象。


```
import os

from forge.agent import ForgeAgent
from forge.sdk import LocalWorkspace
from .db import ForgeDatabase

database_name = os.getenv("DATABASE_STRING")
workspace = LocalWorkspace(os.getenv("AGENT_WORKSPACE"))
database = ForgeDatabase(database_name, debug_enabled=False)
agent = ForgeAgent(database=database, workspace=workspace)

app = agent.get_agent_app()

```py

# `autogpts/forge/forge/db.py`

这段代码定义了一个名为 ChatModel 的类，它继承自 Base 类，并提供了用于存储聊天记录的功能。

ChatModel 类包括以下列：

- msg_id: 用于标识每条聊天记录的 UUID，设置为 Primary Key 和 Index。
- task_id: 用于标识任务单的 UUID，与 ChatModel 无关。
- role: 用户在对话中的角色，如管理员、参与者等。
- content: 用户发送的内容，可以包含文本、图片、文件等。
- created_at: 记录创建聊天记录的时间戳。
- modified_at: 记录最后一次修改聊天记录的时间戳，用于在用户忙碌时自动保存聊天记录。

此外，还定义了一个 SQLAlchemy 对象被视为 ChatModel 的表，并且在代码中导入了 .sdk 包。

该代码的作用是创建一个数据库表来存储聊天记录，并为每条记录添加了一个 UUID 作为主键，以便于在查询和修改时进行唯一标识。同时，还支持将内容类型添加到聊天记录中，以便于存储各种类型的数据。最后，还定义了一个 created_at 和 modified_at 列，用于记录聊天记录的创建和修改时间戳，以便于在查询和修改时进行排序和时间排序。


```
from .sdk import AgentDB, ForgeLogger, NotFoundError, Base
from sqlalchemy.exc import SQLAlchemyError

import datetime
from sqlalchemy import (
    Column,
    DateTime,
    String,
)
import uuid

LOG = ForgeLogger(__name__)

class ChatModel(Base):
    __tablename__ = "chat"
    msg_id = Column(String, primary_key=True, index=True)
    task_id = Column(String)
    role = Column(String)
    content = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    modified_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

```py

This is a Python class that inherits from `行动窃是一款网络内容创作工具`
```scss
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.utils import counterparty

from .exceptions import SQLAlchemyError, NotFoundError
from .dependencies import create_action, get_action_history

Base = declarative_base()

class ActionModel(Base):
   task_id = Column(Integer, primary_key=True)
   name = Column(String)
   args = Column(String)
   created_at = Column(DateTime)

class ActionDao(Base):
   __tablename__ = 'action_model'

   session_factory = sessionmaker(bind=Session)
   action_model = ActionModel

   def create_action(self, task_id, name, args):
       try:
           with self.Session() as session:
               new_action = ActionModel(
                   action_id=str(uuid.uuid4()),
                   task_id=task_id,
                   name=name,
                   args=str(args),
               )
               session.add(new_action)
               session.commit()
               session.refresh(new_action)
               if self.debug_enabled:
                   LOG.debug(f"Created new Action with task_id: {new_action.action_id}")
               return new_action
       except SQLAlchemyError as e:
           LOG.error(f"SQLAlchemy error while creating action: {e}")
           raise
       except NotFoundError as e:
           raise
       except Exception as e:
           LOG.error(f"Unexpected error while creating action: {e}")
           raise

   async def get_action_history(self, task_id):
       if self.debug_enabled:
           LOG.debug(f"Getting action history with task_id: {task_id}")
       try:
           with self.Session() as session:
               if actions := (
                   session.query(ActionModel)
                   .filter(ActionModel.task_id == task_id)
                   .order_by(ActionModel.created_at)
                   .all()
               ):
                   return [{"name": a.name, "args": a.args} for a in actions]

               else:
                   LOG.error(
                       f"Action history not found with task_id: {task_id}"
                   )
                   raise NotFoundError("Action history not found")
       except SQLAlchemyError as e:
           LOG.error(f"SQLAlchemy error while getting action history: {e}")
           raise
       except NotFoundError as e:
           raise
       except Exception as e:
           LOG.error(f"Unexpected error while getting action history: {e}")
           raise
```py


```sql
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.utils import counterparty

from .exceptions import SQLAlchemyError, NotFoundError
from .dependencies import create_action, get_action_history

Base = declarative_base()

class ActionModel(Base):
   task_id = Column(Integer, primary_key=True)
   name = Column(String)
   args = Column(String)
   created_at = Column(DateTime)

class ActionDao(Base):
   __tablename__ = 'action_model'

   session_factory = sessionmaker(bind=Session)
   action_model = ActionModel

   def create_action(self, task_id, name, args):
       try:
           with self.Session() as session:
               new_action = ActionModel(
                   action_id=str(uuid.uuid4()),
                   task_id=task_id,
                   name=name,
                   args=str(args),
               )
               session.add(new_action)
               session.commit()
               session.refresh(new_action)
               if self.debug_enabled:
                   LOG.debug(f"Created new Action with task_id: {new_action.action_id}")
               return new_action
       except SQLAlchemyError as e:
           LOG.error(f"SQLAlchemy error while creating action: {e}")
           raise
       except NotFoundError as e:
           raise
       except Exception as e:
           LOG.error(f
```py


```
class ActionModel(Base):
    __tablename__ = "action"
    action_id = Column(String, primary_key=True, index=True)
    task_id = Column(String)
    name = Column(String)
    args = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    modified_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )


class ForgeDatabase(AgentDB):

    async def add_chat_history(self, task_id, messages):
        for message in messages:
            await self.add_chat_message(task_id, message['role'], message['content'])

    async def add_chat_message(self, task_id, role, content):
        if self.debug_enabled:
            LOG.debug("Creating new task")
        try:
            with self.Session() as session:
                mew_msg = ChatModel(
                    msg_id=str(uuid.uuid4()),
                    task_id=task_id,
                    role=role,
                    content=content,
                )
                session.add(mew_msg)
                session.commit()
                session.refresh(mew_msg)
                if self.debug_enabled:
                    LOG.debug(f"Created new Chat message with task_id: {mew_msg.msg_id}")
                return mew_msg
        except SQLAlchemyError as e:
            LOG.error(f"SQLAlchemy error while creating task: {e}")
            raise
        except NotFoundError as e:
            raise
        except Exception as e:
            LOG.error(f"Unexpected error while creating task: {e}")
            raise
       
    async def get_chat_history(self, task_id):
        if self.debug_enabled:
            LOG.debug(f"Getting chat history with task_id: {task_id}")
        try:
            with self.Session() as session:
                if messages := (
                    session.query(ChatModel)
                    .filter(ChatModel.task_id == task_id)
                    .order_by(ChatModel.created_at)
                    .all()
                ):
                    return [{"role": m.role, "content": m.content} for m in messages]

                else:
                    LOG.error(
                        f"Chat history not found with task_id: {task_id}"
                    )
                    raise NotFoundError("Chat history not found")
        except SQLAlchemyError as e:
            LOG.error(f"SQLAlchemy error while getting chat history: {e}")
            raise
        except NotFoundError as e:
            raise
        except Exception as e:
            LOG.error(f"Unexpected error while getting chat history: {e}")
            raise
    
    async def create_action(self, task_id, name, args):
        try:
            with self.Session() as session:
                new_action = ActionModel(
                    action_id=str(uuid.uuid4()),
                    task_id=task_id,
                    name=name,
                    args=str(args),
                )
                session.add(new_action)
                session.commit()
                session.refresh(new_action)
                if self.debug_enabled:
                    LOG.debug(f"Created new Action with task_id: {new_action.action_id}")
                return new_action
        except SQLAlchemyError as e:
            LOG.error(f"SQLAlchemy error while creating action: {e}")
            raise
        except NotFoundError as e:
            raise
        except Exception as e:
            LOG.error(f"Unexpected error while creating action: {e}")
            raise

    async def get_action_history(self, task_id):
        if self.debug_enabled:
            LOG.debug(f"Getting action history with task_id: {task_id}")
        try:
            with self.Session() as session:
                if actions := (
                    session.query(ActionModel)
                    .filter(ActionModel.task_id == task_id)
                    .order_by(ActionModel.created_at)
                    .all()
                ):
                    return [{"name": a.name, "args": a.args} for a in actions]

                else:
                    LOG.error(
                        f"Action history not found with task_id: {task_id}"
                    )
                    raise NotFoundError("Action history not found")
        except SQLAlchemyError as e:
            LOG.error(f"SQLAlchemy error while getting action history: {e}")
            raise
        except NotFoundError as e:
            raise
        except Exception as e:
            LOG.error(f"Unexpected error while getting action history: {e}")
            raise

```py

# `autogpts/forge/forge/__init__.py`

我需要您提供需要解释的代码，才能为您提供解释和帮助。


```

```py

# `autogpts/forge/forge/__main__.py`

这段代码使用了多个模块和函数，具体解释如下：

1. 导入`os`模块，用于操作系统相关操作。
2. 导入`uvicorn`模块，用于创建一个`uvicorn`风格的HTTP服务器。
3. 导入`dotenv`模块，用于加载`dotenv`环境变量。
4. 导入`forge.sdk.forge_log`模块，用于在程序运行时记录日志。
5. 定义了一个名为`LOG`的函数，用于设置默认的日志等级（日志输出颜色）。
6. 定义了一个名为`logo`的常量，该常量定义了一个`d8888`的图案，用于显示在日志输出中。
7. 加载环境变量，通过调用`dotenv.config()`函数来实现。
8. 创建一个`ForgeLogger`实例，用于记录整个应用程序的日志，该实例的`__name__`属性被设置为`__name__`，以便在日志输出中显示出来。
9. 创建一个`logo_url`常量，该常量将`base_url`和`logo_url`组合在一起，用于将日志输出发送到指定的URL。
10. 创建一个名为`handler`的函数，该函数用于处理用户HTTP请求。
11. 在`handler`函数中，调用`uvicorn.get_蓝色`函数，创建一个HTTP服务器实例，并指定`blueprint`参数，用于指定服务器启动时加载的模块。
12. 调用`forge.sdk.forge_log.B锚`函数，设置日志输出等级为`INFO`，并记录日志信息，该函数将日志信息记录到`LOG`实例中，并返回一个`ForgeLogger`实例，该实例用于记录`INFO`级别的日志信息。
13. 创建一个名为`APSHealthChecker`的函数，该函数用于检查服务器应用程序的健康状态，如果应用程序运行时出现错误，将返回一个`500`的响应，否则返回一个`200`的响应。
14. 创建一个名为`绪论`的函数，该函数用于打印程序的版本信息，包括`__name__`，`__init__`和`__call__`函数的名称。


```
import os

import uvicorn
from dotenv import load_dotenv

import forge.sdk.forge_log

LOG = forge.sdk.forge_log.ForgeLogger(__name__)


logo = """\n\n
       d8888          888             .d8888b.  8888888b. 88888888888 
      d88888          888            d88P  Y88b 888   Y88b    888     
     d88P888          888            888    888 888    888    888     
    d88P 888 888  888 888888 .d88b.  888        888   d88P    888     
   d88P  888 888  888 888   d88""88b 888  88888 8888888P"     888     
  d88P   888 888  888 888   888  888 888    888 888           888     
 d8888888888 Y88b 888 Y88b. Y88..88P Y88b  d88P 888           888     
```py

这段代码是一个批处理脚本，会输出一个字符串"Y888888888"，其中"Y88888"是一个常量，表示为"888888888"，占用8个字节。脚本的主要目的是在字符串中插入一个新的字符'P"，并将其后的所有字符串连接起来，形成一个新的字符串"Y888888888P"。最终的结果是在字符串"Y8888888888"中插入了一个字符'P'，使得原始字符串变成了"Y8888888888P"。


```
d88P     888  "Y88888  "Y888 "Y88P"   "Y8888P88 888           888     
                                                                      
                                                                      
                                                                      
                8888888888                                            
                888                                                   
                888                                                   
                8888888  .d88b.  888d888 .d88b.   .d88b.              
                888     d88""88b 888P"  d88P"88b d8P  Y8b             
                888     888  888 888    888  888 88888888             
                888     Y88..88P 888    Y88b 888 Y8b.                 
                888      "Y88P"  888     "Y88888  "Y8888              
                                             888                      
                                        Y8b d88P                      
                                         "Y88P"                v0.1.0
```py

这段代码是一个Python脚本，它的作用是：

1. 定义了一个函数 `__main__`，当程序作为主函数运行时，它会被执行。
2. 输出一个名为 `logo` 的图像。
3. 获取操作系统中的环境变量 `PORT`，如果该变量没有被设置，它的默认值为 8000。
4. 输出一个日志信息，指出 Agent 服务器开始在本地运行的 HTTP 端口。
5. 加载 .env 文件中的环境变量。
6. 设置 Agent 服务器的日志记录器。
7. 使用 `uvicorn` 库的 `run` 函数，启动 Agent 服务器应用程序。应用程序的参数包括：`"forge.app:app"`，指定应用程序的类名为 `"forge.app"`，指定应用程序运行时的主机为 `"localhost"`，指定应用程序运行时的端口为 `port`（通过环境变量 `PORT` 获取），指定日志级别为 `"error"`，指定程序是否自动重新加载为最新版本。


```
\n"""

if __name__ == "__main__":
    print(logo)
    port = os.getenv("PORT", 8000)
    LOG.info(f"Agent server starting on http://localhost:{port}")
    load_dotenv()
    forge.sdk.forge_log.setup_logger()

    uvicorn.run(
        "forge.app:app", host="localhost", port=port, log_level="error", reload=True
    )

```py

# `autogpts/forge/forge/sdk/agent.py`

这段代码实现了一个FastAPI应用，用于处理网络身份验证和授权，以及实现其他相关功能。下面是这段代码的一些作用解释：

1. 引入所需的模块和库：os、pathlib、BytesIO、uuid4、uvicorn、FastAPI、uvicorn.middleware.cors、fastapi.responses.RedirectResponse、fastapi.responses.StreamingResponse、fastapi.staticfiles.StaticFiles、able.registry、db、errors、ForgeLogger。

2. 通过 从 IO 对象中读取和写入 BytesIO 对象，实现了从文件或其他数据源中读取和写入数据的功能。

3. 通过调用 uuid4.uuid4() 生成一个唯一的 UUID，用于身份验证和授权。

4. 通过创建一个 uvicorn.middleware.cors.CORSMiddleware 中间件，实现了跨域资源共享（CORS）的功能。

5. 通过创建一个 FastAPI 和 uvicorn.responses.RedirectResponse 类，实现了一个 HTTP 状态码为 302 的重定向。

6. 通过创建一个 uvicorn.responses.StreamingResponse 类，实现了一个 HTTP 状态码为 200 的流式传输。

7. 通过创建一个 StaticFiles 类，实现了一个静态文件系统，可以用于 served files。

8. 通过从 forge_log 模块导入一些方法和函数，用于实现日志记录和错误处理。

9. 通过从 .abilities.registry 和 .db 模块导入一些方法和函数，用于实现用户身份验证和数据库操作。

10. 通过创建一个 ForgeLogger 类，实现了日志的记录和输出。


```
import os
import pathlib
from io import BytesIO
from uuid import uuid4

import uvicorn
from fastapi import APIRouter, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .abilities.registry import AbilityRegister
from .db import AgentDB
from .errors import NotFoundError
from .forge_log import ForgeLogger
```py

This is a simple implementation of an Artifactory client in Python. It provides two tasks:

1. `create_artifact`: This task creates an artifact for a given task. It reads the contents of the file specified by the `file_name` parameter, and creates a new artifact with the specified file name in its `relative_path`.
2. `get_artifact`: This task retrieves an artifact by its ID. It reads the contents of the file specified by the `file_name` parameter, and returns it as a StreamingResponse.

The `create_artifact` task uses the `BytesIO` class to write the artifact data to disk, and the `get_artifact` task returns the artifact as a byte stream.

Note: This implementation is for educational purposes only, and should be adapted to fit your specific use case.


```
from .middlewares import AgentMiddleware
from .routes.agent_protocol import base_router
from .schema import *
from .workspace import Workspace

LOG = ForgeLogger(__name__)


class Agent:
    def __init__(self, database: AgentDB, workspace: Workspace):
        self.db = database
        self.workspace = workspace
        self.abilities = AbilityRegister(self)

    def get_agent_app(self, router: APIRouter = base_router):
        """
        Start the agent server.
        """

        app = FastAPI(
            title="AutoGPT Forge",
            description="Modified version of The Agent Protocol.",
            version="v0.4",
        )

        # Add CORS middleware
        origins = [
            "http://localhost:5000",
            "http://127.0.0.1:5000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            # Add any other origins you want to whitelist
        ]

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.include_router(router, prefix="/ap/v1")
        script_dir = os.path.dirname(os.path.realpath(__file__))
        frontend_path = pathlib.Path(
            os.path.join(script_dir, "../../../../frontend/build/web")
        ).resolve()

        if os.path.exists(frontend_path):
            app.mount("/app", StaticFiles(directory=frontend_path), name="app")

            @app.get("/", include_in_schema=False)
            async def root():
                return RedirectResponse(url="/app/index.html", status_code=307)

        else:
            LOG.warning(
                f"Frontend not found. {frontend_path} does not exist. The frontend will not be served"
            )
        app.add_middleware(AgentMiddleware, agent=self)

        return app

    def start(self, port):
        uvicorn.run(
            "forge.app:app", host="localhost", port=port, log_level="error", reload=True
        )

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        Create a task for the agent.
        """
        try:
            task = await self.db.create_task(
                input=task_request.input,
                additional_input=task_request.additional_input,
            )
            return task
        except Exception as e:
            raise

    async def list_tasks(self, page: int = 1, pageSize: int = 10) -> TaskListResponse:
        """
        List all tasks that the agent has created.
        """
        try:
            tasks, pagination = await self.db.list_tasks(page, pageSize)
            response = TaskListResponse(tasks=tasks, pagination=pagination)
            return response
        except Exception as e:
            raise

    async def get_task(self, task_id: str) -> Task:
        """
        Get a task by ID.
        """
        try:
            task = await self.db.get_task(task_id)
        except Exception as e:
            raise
        return task

    async def list_steps(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskStepsListResponse:
        """
        List the IDs of all steps that the task has created.
        """
        try:
            steps, pagination = await self.db.list_steps(task_id, page, pageSize)
            response = TaskStepsListResponse(steps=steps, pagination=pagination)
            return response
        except Exception as e:
            raise

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        Create a step for the task.
        """
        raise NotImplementedError

    async def get_step(self, task_id: str, step_id: str) -> Step:
        """
        Get a step by ID.
        """
        try:
            step = await self.db.get_step(task_id, step_id)
            return step
        except Exception as e:
            raise

    async def list_artifacts(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskArtifactsListResponse:
        """
        List the artifacts that the task has created.
        """
        try:
            artifacts, pagination = await self.db.list_artifacts(
                task_id, page, pageSize
            )
            return TaskArtifactsListResponse(artifacts=artifacts, pagination=pagination)

        except Exception as e:
            raise

    async def create_artifact(
        self, task_id: str, file: UploadFile, relative_path: str
    ) -> Artifact:
        """
        Create an artifact for the task.
        """
        data = None
        file_name = file.filename or str(uuid4())
        try:
            data = b""
            while contents := file.file.read(1024 * 1024):
                data += contents
            # Check if relative path ends with filename
            if relative_path.endswith(file_name):
                file_path = relative_path
            else:
                file_path = os.path.join(relative_path, file_name)

            self.workspace.write(task_id, file_path, data)

            artifact = await self.db.create_artifact(
                task_id=task_id,
                file_name=file_name,
                relative_path=relative_path,
                agent_created=False,
            )
        except Exception as e:
            raise
        return artifact

    async def get_artifact(self, task_id: str, artifact_id: str) -> Artifact:
        """
        Get an artifact by ID.
        """
        try:
            artifact = await self.db.get_artifact(artifact_id)
            if artifact.file_name not in artifact.relative_path:
                file_path = os.path.join(artifact.relative_path, artifact.file_name)
            else:
                file_path = artifact.relative_path
            retrieved_artifact = self.workspace.read(task_id=task_id, path=file_path)
        except NotFoundError as e:
            raise
        except FileNotFoundError as e:
            raise
        except Exception as e:
            raise

        return StreamingResponse(
            BytesIO(retrieved_artifact),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={artifact.file_name}"
            },
        )

```py

# `autogpts/forge/forge/sdk/agent_test.py`

该代码使用了现代Python中的pytest库，用于编写和运行测试。作用是定义了一个名为Agent的类，以及从db、db和schema模块中导出的相关类。此外，还定义了用于测试的fixture装饰器，用于在测试过程中初始化和清理数据库和 workspace。

具体来说，该代码的作用是创建一个带有数据库和workspace的agent对象，以便在测试中进行使用。通过使用fixture装饰器，可以在测试代码中轻松地调用Agent对象，而无需在每个测试函数中显式地创建和销毁Agent实例。这种方法有助于提高测试代码的可读性和可维护性。


```
import pytest

from .agent import Agent
from .db import AgentDB
from .schema import StepRequestBody, Task, TaskListResponse, TaskRequestBody
from .workspace import LocalWorkspace


@pytest.fixture
def agent():
    db = AgentDB("sqlite:///test.db")
    workspace = LocalWorkspace("./test_workspace")
    return Agent(db, workspace)


```py

这段代码使用了两个 Pytest 标记，分别是 `@pytest.mark.skip` 和 `@pytest.mark.asyncio`。它们表示该代码为 skip 测试，即这个测试将会跳过，而不是运行。

第一个测试函数 `test_create_task` 使用了 `@pytest.mark.asyncio` 标记，表示该函数使用了异步测试（asyncio）。在这个函数中，使用了 `TaskRequestBody` 类来创建一个任务请求，该类需要 `input` 和 `additional_input` 参数，分别表示任务接收到的输入和附加的输入。

第二个测试函数 `test_list_tasks` 同样使用了 `@pytest.mark.asyncio` 标记，表示该函数使用了异步测试。在这个函数中，同样使用了 `TaskRequestBody` 类来创建一个任务请求，该类需要 `input` 参数，表示任务接收到的输入。然后，使用了 `agent.create_task` 方法来创建一个任务，并使用了 `await` 关键字来等待任务执行完成。接着，使用了 `agent.list_tasks` 方法来获取任务列表，并使用了 `isinstance` 函数来判断列表是否为 `TaskListResponse` 类型。

这段代码的作用是测试一个简单的 asyncio 任务，包括创建任务和获取任务列表的功能。


```
@pytest.mark.skip
@pytest.mark.asyncio
async def test_create_task(agent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task: Task = await agent.create_task(task_request)
    assert task.input == "test_input"


@pytest.mark.skip
@pytest.mark.asyncio
async def test_list_tasks(agent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)
    tasks = await agent.list_tasks()
    assert isinstance(tasks, TaskListResponse)


```py

这段代码使用了Python的pytest库来进行断言，其中包含两个测试函数，分别是：

1. test_get_task(agent)：使用mark.asyncio标记为asyncio函数，这个mark表示这个函数使用异步编程技术来执行。函数内部使用了TaskRequestBody类来创建一个任务请求，并使用agent.create_task(task_request)来创建任务。然后，使用agent.get_task(task.task_id)来获取已经创建好的任务，并断言返回的结果与任务Id是否匹配。
2. test_create_and_execute_step(agent)：使用mark.asyncio标记为asyncio函数，这个mark表示这个函数使用异步编程技术来执行。函数内部使用了StepRequestBody类来创建一个步骤请求，并使用agent.create_and_execute_step(task.task_id, step_request)来创建步骤。然后，使用step.input和step.additional_input属性来获取步骤的输入和附加输入，并断言输入是否为"step_input"，附加输入是否包含"additional_test_input"。


```
@pytest.mark.skip
@pytest.mark.asyncio
async def test_get_task(agent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)
    retrieved_task = await agent.get_task(task.task_id)
    assert retrieved_task.task_id == task.task_id


@pytest.mark.skip
@pytest.mark.asyncio
async def test_create_and_execute_step(agent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)
    step_request = StepRequestBody(
        input="step_input", additional_input={"input": "additional_test_input"}
    )
    step = await agent.create_and_execute_step(task.task_id, step_request)
    assert step.input == "step_input"
    assert step.additional_input == {"input": "additional_test_input"}


```py

这段代码使用了Python的异步编程库(asyncio)以及pytest库进行测试。

该代码定义了一个名为`test_get_step`的测试函数，使用了mark.asyncio和mark.skip注解来标记为异步函数和允许该函数跳过测试。

函数体中，首先创建了一个名为`task_request`的任务请求对象，其中包含一个输入参数`input`和一个附加的输入参数`additional_input`。这个任务请求将被作为异步任务的一个参数传递给`agent.create_task()`方法，并且在函数内部使用`await`来等待该任务的完成。

然后，定义了一个名为`step_request`的步骤请求对象，其中包含一个输入参数`input`和一个附加的输入参数`additional_input`。这个步骤请求同样将被作为异步任务的一个参数传递给`agent.create_and_execute_step()`方法，并且在函数内部使用`await`来等待该步骤的完成。

接下来，使用`agent.get_step()`方法来获取异步任务`task.task_id`的步骤`step_id`。这个方法返回一个包含当前步骤的`Step`对象，其中包含一个`step_id`属性，该属性将用于检查所获取到的步骤是否与期望一致。

最后，使用`assert`语句来验证`agent.get_step()`方法返回的步骤对象中`step_id`是否与之前创建的步骤`step_request`中的输入参数`input`和`additional_input`完全匹配。如果不匹配，那么函数将返回错误，表明出现了问题。


```
@pytest.mark.skip
@pytest.mark.asyncio
async def test_get_step(agent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)
    step_request = StepRequestBody(
        input="step_input", additional_input={"input": "additional_test_input"}
    )
    step = await agent.create_and_execute_step(task.task_id, step_request)
    retrieved_step = await agent.get_step(task.task_id, step.step_id)
    assert retrieved_step.step_id == step.step_id


```py

这两行代码是使用Python的pytest库中定义的一个测试框架。`@pytest.mark.skip`和`@pytest.mark.asyncio`是装饰器，用于标记测试函数的作用，表示该函数将会暂停执行该测试套件中的所有测试函数，以便其他测试函数能够运行。

`asyncio`是一个Python库，定义了异步编程的概念和工具，如异步/await组合，可以让代码更加简洁易懂。

`asyncdef test_list_artifacts(agent):`定义了一个异步函数，名为`test_list_artifacts`，使用`asyncio`中的异步编程特性来实现。该函数使用`agent.list_artifacts()`方法获取代理对象的指令，并将其存储在函数内部，然后使用`assert`语句验证返回的结果是否为列表类型。

`@pytest.mark.skip`和`@pytest.mark.asyncio`是装饰器，用于标记`test_list_artifacts`函数的作用。表示该函数将会暂停执行该测试套件中的所有测试函数，以便其他测试函数能够运行。


```
@pytest.mark.skip
@pytest.mark.asyncio
async def test_list_artifacts(agent):
    artifacts = await agent.list_artifacts()
    assert isinstance(artifacts, list)


@pytest.mark.skip
@pytest.mark.asyncio
async def test_create_artifact(agent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)
    artifact_request = ArtifactRequestBody(file=None, uri="test_uri")
    artifact = await agent.create_artifact(task.task_id, artifact_request)
    assert artifact.uri == "test_uri"


```py

这段代码使用了Python的pytest库进行测试，并且使用了Mark.skip和Mark.asyncio注解来标记测试的功能，asyncio注解指定了该测试是一个异步测试，使用await关键字来挂起并等待异步操作的结果。

具体来说，这段代码的作用是测试异步操作中获取 artifacts（依赖物）的能力，主要包括创建一个Task Request，设置其输入参数，执行异步任务，获取生成的Artifact，然后检查获取到的Artifact是否与期望的结果一致。

以下是代码的更详细解释：

```python
@pytest.mark.skip
@pytest.mark.asyncio
async def test_get_artifact(agent):
   # 创建Task Request
   task_request = TaskRequestBody(
       input="test_input",
       additional_input={"input": "additional_test_input"}
   )

   # 创建Task
   task = await agent.create_task(task_request)

   # 创建Artifact Request
   artifact_request = ArtifactRequestBody(file=None, uri="test_uri")

   # 创建Artifact
   artifact = await agent.create_artifact(task.task_id, artifact_request)

   # 检查获取到的Artifact是否与期望的结果一致
   retrieved_artifact = await agent.get_artifact(task.task_id, artifact.artifact_id)
   assert retrieved_artifact.artifact_id == artifact.artifact_id
```py

该测试的输入参数是一个异步的Task Request，其中包含一个输入参数和一个或多个附加输入。在这个例子中，输入参数中传递了一个字符串（test_input）和一个元组的键值对（additional_test_input）。附加输入是一个Python的元组，其中包含一个文件（file）和一个URI（uri）。

测试创建了一个异步任务之后，使用await关键字将异步操作的结果（即生成的Artifact）获取出来，并检查获取到的Artifact是否与期望的结果一致。如果一致，则使用assert关键字进行断言，否则输出错误。


```
@pytest.mark.skip
@pytest.mark.asyncio
async def test_get_artifact(agent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)
    artifact_request = ArtifactRequestBody(file=None, uri="test_uri")
    artifact = await agent.create_artifact(task.task_id, artifact_request)
    retrieved_artifact = await agent.get_artifact(task.task_id, artifact.artifact_id)
    assert retrieved_artifact.artifact_id == artifact.artifact_id

```