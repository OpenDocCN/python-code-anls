# AutoGPT源码解析 6

# `autogpts/autogpt/autogpt/commands/git_operations.py`

这段代码定义了一系列用于执行 Git 操作的命令，并注册了一个名为 "git_operations" 的命令类别。这个命令类别包含一个名为 "Git Operations" 的命令标题。

从代码中可以看出，这些命令是通过 `git.repo` 模块实现的。这个模块是 Python 的 Git 仓库操作模块，可以用来进行 Git 仓库中的各种操作，例如 clone、add、commit、push等等。

除了 `git.repo` 模块之外，代码中还引入了两个来自 `autogpt` 模组的函数，分别是 `Agent` 和 `utils.exceptions.CommandExecutionError`。`Agent` 是一个人工智能代理，可以用来执行各种操作，而 `CommandExecutionError` 则是来自 `utils.exceptions` 模块的异常类，用于处理在执行命令时发生的错误。

另外，代码中还定义了一个 JSON 数据模式 `JSONSchema`，用于定义 Git 仓库中的数据结构和约束。这个模式可以用来定义 Git 仓库中的各种数据，例如分支、提交、仓库配置等等。

最后，代码还定义了一个名为 `validate_url` 的函数，用于验证 Git 仓库中的 URL 地址是否合法。


```py
"""Commands to perform Git operations"""

COMMAND_CATEGORY = "git_operations"
COMMAND_CATEGORY_TITLE = "Git Operations"

from pathlib import Path

from git.repo import Repo

from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import CommandExecutionError
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.url_utils.validators import validate_url

```

这段代码定义了一个命令行工具，名为“clone_repository”，用于从给定 URL 复制一个仓库到本地。

具体来说，这个命令行工具需要以下参数：

-   “url”参数，用于存储要复制的仓库的 URL。这个参数是必填的，并且必须在运行时传递。
-   “clone_path”参数，用于存储将仓库复制到本地时需要下载的文件的路径。这个参数也是必填的，并且必须在运行时传递。

当这个命令行工具被运行时，它会检查两个条件：

-   如果“url”和“clone_path”参数都提供了有效的参数，那么它会尝试从给定的 URL 下载对应的文件，并将下载的文件本地化到指定的路径。
-   如果“url”或“clone_path”参数中的任何一个参数是无效的，那么这个命令行工具不会执行任何操作，并且返回 False。


```py
from .decorators import sanitize_path_arg


@command(
    "clone_repository",
    "Clones a Repository",
    {
        "url": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The URL of the repository to clone",
            required=True,
        ),
        "clone_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path to clone the repository to",
            required=True,
        ),
    },
    lambda config: bool(config.github_username and config.github_api_key),
    "Configure github_username and github_api_key.",
)
```

这段代码定义了一个名为 `clone_repository` 的函数，它接受三个参数：`url`、`clone_path` 和 `agent`。函数的作用是尝试从 GitHub 上克隆一个特定 URL 的本地仓库，并将克隆结果返回。

具体来说，函数首先将传入的 `url` 和 `clone_path` 参数拆分成多个部分，然后构造一个包含克隆仓库信息的字符串 `auth_repo_url`。接着，使用字符串拼接方法 `join` 将 `auth_repo_url` 和 `clone_path` 组合成一个新的字符串，这个字符串将作为 GitHub 克隆请求的 URL。

接下来，函数调用一个名为 `Repo.clone_from` 的类方法，这个方法从 GitHub 仓库的 URL 中提取出授权信息（API key 和用户名），并将它们作为参数传递给 `clone_repo` 函数。如果执行过程中出现错误，函数将引发一个 `CommandExecutionError` 异常并返回错误信息。

最后，函数返回一个字符串，这个字符串将显示克隆操作的结果，即仓库克隆的路径。


```py
@sanitize_path_arg("clone_path")
@validate_url
def clone_repository(url: str, clone_path: Path, agent: Agent) -> str:
    """Clone a GitHub repository locally.

    Args:
        url (str): The URL of the repository to clone.
        clone_path (Path): The path to clone the repository to.

    Returns:
        str: The result of the clone operation.
    """
    split_url = url.split("//")
    auth_repo_url = f"//{agent.legacy_config.github_username}:{agent.legacy_config.github_api_key}@".join(
        split_url
    )
    try:
        Repo.clone_from(url=auth_repo_url, to_path=clone_path)
    except Exception as e:
        raise CommandExecutionError(f"Could not clone repo: {e}")

    return f"""Cloned {url} to {clone_path}"""

```

# `autogpts/autogpt/autogpt/commands/image_gen.py`

这段代码的作用是生成基于文本输入的图像。它包括以下几个步骤：

1. 将文本信息和图像信息存储在同一个类中。
2. 使用 `requests` 库发送 HTTP GET 请求，获取指定的图像资源。
3. 将获取到的图像资源进行解析，并将其转换为 `base64` 编码的字节流。
4. 将解码后的图像字节流存储进 `io` 对象中，并给定的图像类对象赋值。
5. 在 `text_to_image` 命令类中，调用 `generate_image` 方法，生成图像。


```py
"""Commands to generate images based on text input"""

COMMAND_CATEGORY = "text_to_image"
COMMAND_CATEGORY_TITLE = "Text to Image"

import io
import json
import logging
import time
import uuid
from base64 import b64decode

import openai
import requests
from PIL import Image

```

这段代码是一个AutogPT的AI代理，用于生成图像。它定义了一个命令行接口，接收一个提示（ prompt ），用于生成图像。

首先，从autogpt.agents.agent模块导入Agent类，然后从autogpt.command_decorator模块导入command函数。

然后，定义了一个logger变量，用于记录生成的图像的日志信息。

接着，使用@command装饰器定义生成图像的命令，其中包含一个prompt参数，用于指定要生成的图像的主题或描述。这个参数是一个JSONSchema，用于定义输入数据和预期的输出数据。

命令的实现函数内部，使用True表示需要设置一个图像提供者，即必须设置图像提供者，否则不会生成图像。


```py
from autogpt.agents.agent import Agent
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema

logger = logging.getLogger(__name__)


@command(
    "generate_image",
    "Generates an Image",
    {
        "prompt": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The prompt used to generate the image",
            required=True,
        ),
    },
    lambda config: bool(config.image_provider),
    "Requires a image provider to be set.",
)
```

这段代码定义了一个名为 `generate_image` 的函数，它接受一个字符串参数 `prompt` 和一个整数参数 `size`。函数返回一个文件名，它可以从一个提示字符串开始生成一幅图像，并使用指定的尺寸。

具体来说，这段代码实现了一个简单的 DALL-E 模型，如果配置了使用 DALL-E 作为图像生成器，则函数将使用 `generate_image_with_dalle` 函数生成图像；如果配置了使用 HuggingFace 作为图像生成器，则函数将使用 `generate_image_with_hf` 函数生成图像；如果配置了使用 SD WebUI 作为图像生成器，则函数将使用 `generate_image_with_sd_webui` 函数生成图像。如果以上三种生成器均未配置，函数将返回一个字符串 "No Image Provider Set"。

这段代码的实现比较简单，只是通过判断 `image_provider` 设置来选择不同的图像生成器。但是需要注意的是，对于不同的图像生成器，需要使用不同的函数来生成图像，这可能会导致生成的图像质量有所不同。


```py
def generate_image(prompt: str, agent: Agent, size: int = 256) -> str:
    """Generate an image from a prompt.

    Args:
        prompt (str): The prompt to use
        size (int, optional): The size of the image. Defaults to 256. (Not supported by HuggingFace)

    Returns:
        str: The filename of the image
    """
    filename = agent.workspace.root / f"{str(uuid.uuid4())}.jpg"

    # DALL-E
    if agent.legacy_config.image_provider == "dalle":
        return generate_image_with_dalle(prompt, filename, size, agent)
    # HuggingFace
    elif agent.legacy_config.image_provider == "huggingface":
        return generate_image_with_hf(prompt, filename, agent)
    # SD WebUI
    elif agent.legacy_config.image_provider == "sdwebui":
        return generate_image_with_sd_webui(prompt, filename, agent, size)
    return "No Image Provider Set"


```

这段代码定义了一个名为 `generate_image_with_hf` 的函数，它接受一个 `prompt` 参数和一个 `filename` 参数，作为一个字符串类型的函数，它返回一个文件名。

该函数使用 HuggingFace 的 API 来生成图像。函数内部首先设置了一个 API 头信息，然后设置了一个重试计数器，当 API 请求失败时，会重试多次。在每次 API 请求失败时，函数都会检查返回的响应是否为 200，如果是 200，就说明 API 请求成功，然后使用 requests.post() 方法生成图像，并将生成好的图像保存到指定文件名。如果 API 请求失败，则会在本地记录错误并重新尝试请求，直到成功生成图像为止。


```py
def generate_image_with_hf(prompt: str, filename: str, agent: Agent) -> str:
    """Generate an image with HuggingFace's API.

    Args:
        prompt (str): The prompt to use
        filename (str): The filename to save the image to

    Returns:
        str: The filename of the image
    """
    API_URL = f"https://api-inference.huggingface.co/models/{agent.legacy_config.huggingface_image_model}"
    if agent.legacy_config.huggingface_api_token is None:
        raise ValueError(
            "You need to set your Hugging Face API token in the config file."
        )
    headers = {
        "Authorization": f"Bearer {agent.legacy_config.huggingface_api_token}",
        "X-Use-Cache": "false",
    }

    retry_count = 0
    while retry_count < 10:
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": prompt,
            },
        )

        if response.ok:
            try:
                image = Image.open(io.BytesIO(response.content))
                logger.info(f"Image Generated for prompt:{prompt}")
                image.save(filename)
                return f"Saved to disk:{filename}"
            except Exception as e:
                logger.error(e)
                break
        else:
            try:
                error = json.loads(response.text)
                if "estimated_time" in error:
                    delay = error["estimated_time"]
                    logger.debug(response.text)
                    logger.info("Retrying in", delay)
                    time.sleep(delay)
                else:
                    break
            except Exception as e:
                logger.error(e)
                break

        retry_count += 1

    return f"Error creating image."


```

这段代码是一个函数，名为 `generate_image_with_dalle`，它接受一个字符串参数 `prompt`，一个文件名参数 `filename`，和一个整数参数 `size`，还有一个 `Agent` 类型的参数。它的返回值是文件名，也就是生成的图像的文件名。

函数的主要作用是使用 DALL-E 生成一张图像，并将其保存到指定的大小。具体实现包括以下步骤：

1. 检查接受的图像大小是否在支持的范围内，如果不是，则根据给出的最小尺寸调整大小，并输出提示信息。
2. 通过 `openai.Image.create` 方法生成图像，并将其保存到指定大小。这个方法需要提供两个参数：一个 prompt，也就是生成图像的提示信息；一个 n 参数，表示生成一个 n 分辨率（也就是 256x256）的图像；还有一个 size 参数，表示图像的大小，单位是像素。
3. 将生成的图像数据编码为字节，并使用 'wb' 模式打开一个文件，写入数据并输出文件名。
4. 最后，返回文件名，以便用户知道生成了哪个图像。


```py
def generate_image_with_dalle(
    prompt: str, filename: str, size: int, agent: Agent
) -> str:
    """Generate an image with DALL-E.

    Args:
        prompt (str): The prompt to use
        filename (str): The filename to save the image to
        size (int): The size of the image

    Returns:
        str: The filename of the image
    """

    # Check for supported image sizes
    if size not in [256, 512, 1024]:
        closest = min([256, 512, 1024], key=lambda x: abs(x - size))
        logger.info(
            f"DALL-E only supports image sizes of 256x256, 512x512, or 1024x1024. Setting to {closest}, was {size}."
        )
        size = closest

    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size=f"{size}x{size}",
        response_format="b64_json",
        api_key=agent.legacy_config.openai_api_key,
    )

    logger.info(f"Image Generated for prompt:{prompt}")

    image_data = b64decode(response["data"][0]["b64_json"])

    with open(filename, mode="wb") as png:
        png.write(image_data)

    return f"Saved to disk:{filename}"


```

This is a function that takes in an image prompt, a save filename, an agent object, and an optional size for the image. It uses the Stable Diffusion web UI to generate the image and saves it to disk.

The function uses the `requests` library to make a POST request to the web UI endpoint, passing in the image prompt, negative prompt, and other parameters. It returns the filename of the generated image, or raises an exception if there is an error.

Note that the function uses the `agent.legacy_config.sd_webui_auth` configuration parameter to determine whether to use the Stable Diffusion web UI. If this is set to `True`, the user will be prompted to enter a username and password to authenticate with the web UI.


```py
def generate_image_with_sd_webui(
    prompt: str,
    filename: str,
    agent: Agent,
    size: int = 512,
    negative_prompt: str = "",
    extra: dict = {},
) -> str:
    """Generate an image with Stable Diffusion webui.
    Args:
        prompt (str): The prompt to use
        filename (str): The filename to save the image to
        size (int, optional): The size of the image. Defaults to 256.
        negative_prompt (str, optional): The negative prompt to use. Defaults to "".
        extra (dict, optional): Extra parameters to pass to the API. Defaults to {}.
    Returns:
        str: The filename of the image
    """
    # Create a session and set the basic auth if needed
    s = requests.Session()
    if agent.legacy_config.sd_webui_auth:
        username, password = agent.legacy_config.sd_webui_auth.split(":")
        s.auth = (username, password or "")

    # Generate the images
    response = requests.post(
        f"{agent.legacy_config.sd_webui_url}/sdapi/v1/txt2img",
        json={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "sampler_index": "DDIM",
            "steps": 20,
            "config_scale": 7.0,
            "width": size,
            "height": size,
            "n_iter": 1,
            **extra,
        },
    )

    logger.info(f"Image Generated for prompt:{prompt}")

    # Save the image to disk
    response = response.json()
    b64 = b64decode(response["images"][0].split(",", 1)[0])
    image = Image.open(io.BytesIO(b64))
    image.save(filename)

    return f"Saved to disk:{filename}"

```

# `autogpts/autogpt/autogpt/commands/system.py`



该代码定义了一个接口 `CommandsToControlInternalState`，旨在控制程序内部状态。它属于 `system` 类别，标题为 "System"。

该接口有一个 `import` 语句，从 `__future__` 函数中导入了一个名为 `get_agent_context` 的函数，来自 `autogpt.agents.features.context` 模块，它允许获取当前应用程序的上下文。

该接口还有一个 `from typing import TYPE_CHECKING` 语句，用于标记输入参数 `self` 中的类型。这个语句允许在函数内部使用来自 `typing` 库的类型提示，以便代码能够更好地理解和维护。

该接口还有一个 `from autogpt.agents.agent import Agent` 语句，从 `autogpt.agents.agent` 模块中导入了一个名为 `Agent` 的类，来自 `autogpt.agents` 模块。

最后，该接口定义了一个名为 `__init__` 的静态方法，用于初始化该接口。这个方法有一个参数 `self`，它是接口的实例，允许将 `self` 参数传递给接口的方法。在 `__init__` 方法内部，它从 `get_agent_context` 函数获取当前应用程序的上下文，并将它存储在 `self.context` 属性中。


```py
"""Commands to control the internal state of the program"""

from __future__ import annotations

COMMAND_CATEGORY = "system"
COMMAND_CATEGORY_TITLE = "System"

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogpt.agents.agent import Agent

from autogpt.agents.features.context import get_agent_context
from autogpt.agents.utils.exceptions import AgentTerminated, InvalidArgumentError
```

这段代码是一个来自"autogpt.command_decorator"的"command"装饰器，用于在需要停止自动装置执行任务时使用。它接受一个参数"或当无法完成任务时使用"，并且在传递给装饰器的JSON Schema中，要求必须包含一个名为"reason"的类型为JSON Schema的参数，该参数告诉用户任务目标已经完成的摘要。

具体来说，当需要停止自动装置执行任务时，用户可以通过调用这个命令来告诉自动装置。当自动装置收到这个命令时，它将查看传递给它的"reason"参数，如果这个参数存在，那么自动装置将从该参数中读取总结说明，用于告诉用户任务目标已经完成。如果这个参数不存在，那么自动装置将不会执行任何操作，并将退出。


```py
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema

logger = logging.getLogger(__name__)


@command(
    "finish",
    "Use this to shut down once you have completed your task,"
    " or when there are insurmountable problems that make it impossible"
    " for you to finish your task.",
    {
        "reason": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="A summary to the user of how the goals were accomplished",
            required=True,
        )
    },
)
```

这段代码定义了一个名为“finish”的函数，其接收两个参数，一个是字符串类型的参数“reason”，另一个是Agent类型的参数“agent”。函数的作用是向用户报告如何实现目标，并返回一个字符串表示聊天完成以及一些改进的建议。

然而，这段代码存在一些问题。首先，函数内部使用了raise关键字，这会导致函数在运行时抛出异常，这可能对程序的稳定性造成影响。其次，函数没有明确的输入验证，这可能会导致在实际应用中出现错误。此外，函数也没有任何错误处理措施，这可能会导致在程序出现错误时无法提供有用的提示信息。

为了改进这段代码，可以进行以下修改：

1. 移除raise关键字，避免抛出异常。
2. 添加输入验证，确保函数能够正确接收参数。
3. 提供错误处理，为程序遇到错误时提供有用的提示信息。


```py
def finish(reason: str, agent: Agent) -> None:
    """
    A function that takes in a string and exits the program

    Parameters:
        reason (str): A summary to the user of how the goals were accomplished.
    Returns:
        A result string from create chat completion. A list of suggestions to
            improve the code.
    """
    raise AgentTerminated(reason)


@command(
    "hide_context_item",
    "Hide an open file, folder or other context item, to save memory.",
    {
        "number": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="The 1-based index of the context item to hide",
            required=True,
        )
    },
    available=lambda a: bool(get_agent_context(a)),
)
```

这段代码定义了一个名为 `close_context_item` 的函数，用于关闭指定 `agent` 对象的特定上下文项(item)，并返回该上下文项的备注信息。

具体来说，函数的实现包括以下步骤：

1. 使用 `assert` 语句来检查参数 `agent` 是否存在于上下文项列表(context.items)中。如果不存在，则会抛出一个名为 `InvalidArgumentError` 的异常，并指定错误消息中包含参数 `number` 的字符串。

2. 如果要关闭的上下文项编号 `number` 大于上下文项列表的长度(len(context.items))，则会使用 `InternalError` 异常，并指定错误消息中包含参数 `number` 的字符串。

3. 如果要关闭的上下文项编号为0，则不会做任何操作，并直接返回备注信息。

4. 在函数中，使用 `context.close(number)` 来关闭指定编号的上下文项。

5. 在函数中，使用 `return f"Context item {number} hidden ✅"` 来返回备注信息，其中 `{}` 表示在备注信息中插入参数 `number` 的值。`{}` 中的 `{}` 是一个占位符，用于插入备注信息中的实际值。在这里，备注信息是 `Context item {number} hidden ✅`。


```py
def close_context_item(number: int, agent: Agent) -> str:
    assert (context := get_agent_context(agent)) is not None

    if number > len(context.items) or number == 0:
        raise InvalidArgumentError(f"Index {number} out of range")

    context.close(number)
    return f"Context item {number} hidden ✅"

```

# `autogpts/autogpt/autogpt/commands/times.py`

这段代码定义了一个名为 `get_datetime` 的函数，它使用 `datetime.datetime` 类的实例化函数来获取当前日期和时间。然后，它将返回当前日期和时间的字符串表示。

函数的作用是，当调用它时，它将返回当前日期和时间的字符串表示。例如，如果你在应用程序中调用 `get_datetime()` 函数，它将返回类似于这样的字符串：

```py
Current date and time: 2023-02-18 15:22:45.789000
```

注意，由于 `datetime.datetime` 类的实例化函数使用的是 `strftime` 方法来自动将日期和时间格式化为字符串，因此 `get_datetime()` 函数也将返回一个字符串。


```py
from datetime import datetime


def get_datetime() -> str:
    """Return the current date and time

    Returns:
        str: The current date and time
    """
    return "Current date and time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

```

# `autogpts/autogpt/autogpt/commands/user_interaction.py`

这段代码定义了一个用户交互的命令，属于 "user\_interaction" 类别，命令标题为 "User Interaction"。该命令接受一个 question 参数，用于向用户提问，如果用户提供了回答，则返回用户的回答。该命令使用来自 "autogpt.agents.agent" 和 "autogpt.app.utils.json\_schema" 的类和函数。


```py
"""Commands to interact with the user"""

from __future__ import annotations

COMMAND_CATEGORY = "user_interaction"
COMMAND_CATEGORY_TITLE = "User Interaction"

from autogpt.agents.agent import Agent
from autogpt.app.utils import clean_input
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema


@command(
    "ask_user",
    (
        "If you need more details or information regarding the given goals,"
        " you can ask the user for input"
    ),
    {
        "question": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The question or prompt to the user",
            required=True,
        )
    },
    enabled=lambda config: not config.noninteractive_mode,
)
```

这段代码定义了一个异步函数 `ask_user`，该函数接受一个问句（question）和一个AI代理（Agent）。它首先打印出问句，然后使用 `clean_input` 函数从代理的遗留配置中获取用户输入（A或B）。最后，它将用户输入和代理的遗留配置一起作为字符串返回。

具体来说，这段代码执行以下操作：

1. 在函数定义中，定义了一个名为 `ask_user` 的异步函数，该函数有一个参数 `question` 和一个参数 `agent`。
2. 在函数体内，首先调用 `print` 函数打印出问句。
3. 接下来，使用 `clean_input` 函数获取用户输入（A或B）。`clean_input` 函数的实现可能依赖于具体的API和环境，但通常会处理输入的校验和过滤。
4. 然后，使用 `await` 关键字从代理的遗留配置中获取用户输入。
5. 最后，将用户输入和代理的遗留配置合并为一个字符串，并使用 `f-string` 格式化将结果返回。


```py
async def ask_user(question: str, agent: Agent) -> str:
    print(f"\nQ: {question}")
    resp = await clean_input(agent.legacy_config, "A:")
    return f"The user's answer: '{resp}'"

```

# `autogpts/autogpt/autogpt/commands/web_search.py`

这段代码是一个命令行工具，用于在命令行中使用 "duckduckgo-search" 代理进行网页搜索。它的作用是帮助用户在浏览器中进行网页搜索，通过使用 "duckduckgo-search" 代理来获取搜索结果。


```py
"""Commands to search the web with"""

from __future__ import annotations

COMMAND_CATEGORY = "web_search"
COMMAND_CATEGORY_TITLE = "Web Search"

import json
import time
from itertools import islice

from duckduckgo_search import DDGS

from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import ConfigurationError
```

这段代码定义了一个名为 "web_search" 的命令，使用了来自 autogpt.command_decorator 的命令格式，可以接受一个搜索查询的参数，并返回一个搜索结果。

具体来说，这个命令使用了一个名为 "DUCKDUCKGO_MAX_ATTEMPTS" 的变量，其值为 3。这个变量似乎没有什么作用，但它在代码的下面部分被提到了，说明它可能是用于某种异步操作的结果。

另外，这个命令还定义了一个来自 autogpt.core.utils.json_schema 的 JSONSchema，它被用来描述搜索查询的数据格式。


```py
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema

DUCKDUCKGO_MAX_ATTEMPTS = 3


@command(
    "web_search",
    "Searches the web",
    {
        "query": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The search query",
            required=True,
        )
    },
    aliases=["search"],
)
```

这段代码是一个 Python 函数，名为 `web_search`，用于在谷歌搜索引擎上执行搜索，并返回搜索结果。

函数接收两个参数：`query` 参数是搜索查询，`agent` 参数是一个 `Agent` 类别的对象，用于在网络上执行搜索操作。函数还接收一个可选参数 `num_results` ，用于指定要返回的搜索结果数量。

函数内部首先创建一个空列表 `search_results`，并设置 `attempts` 变量为 0。然后在循环中，函数首先检查 `query` 参数是否为空。如果是，函数将返回一些默认的搜索结果，例如 HTTP 状态码 200。如果 `query` 参数不为空，函数调用 `DDGS()` 函数执行搜索操作，并将结果添加到 `search_results` 列表中。由于每个搜索结果可能包含多个参数，例如搜索结果的标题、描述、URL 等，函数还使用了 `**` 运算符，将结果作为字典的形式返回。

在循环之外，函数还使用 `time.sleep(1)` 函数来等待一段时间并重新尝试搜索，以提高搜索结果的准确性和稳定性。

最后，函数将 `search_results` 列表中的所有元素通过 `json.dumps` 函数进行序列化，并使用 `safe_google_results` 函数将 JSON 序列化后的结果进行转义，以便将结果正确地传回给调用者。


```py
def web_search(query: str, agent: Agent, num_results: int = 8) -> str:
    """Return the results of a Google search

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
    search_results = []
    attempts = 0

    while attempts < DUCKDUCKGO_MAX_ATTEMPTS:
        if not query:
            return json.dumps(search_results)

        results = DDGS().text(query)
        search_results = list(islice(results, num_results))

        if search_results:
            break

        time.sleep(1)
        attempts += 1

    search_results = [
        {
            "title": r["title"],
            "url": r["href"],
            **({"description": r["body"]} if r.get("body") else {}),
        }
        for r in search_results
    ]

    results = json.dumps(search_results, ensure_ascii=False, indent=4)
    return safe_google_results(results)


```

这段代码定义了一个命令，名为 "google"，用于执行 Google 搜索。该命令的查询参数是一个 JSON 类型字段，描述为“搜索查询”。该字段是必填的，并且必须在命令参数中传递。

当该命令被调用时，命令会根据传入的 Google API 密钥和 Google 自定义搜索引擎 ID 来配置 Google 搜索。如果传入的配置参数不符合要求，该命令不会执行 Google 搜索，并返回 False。

此外，该命令还定义了一个搜索 alias，名为 "search"。


```py
@command(
    "google",
    "Google Search",
    {
        "query": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The search query",
            required=True,
        )
    },
    lambda config: bool(config.google_api_key)
    and bool(config.google_custom_search_engine_id),
    "Configure google_api_key and custom_search_engine_id.",
    aliases=["search"],
)
```

This is a function that returns the search results of a Google search using the official Google API. The function takes a search query and a number of results to return. It returns a string or a list of strings if the search query returns multiple results. It uses the Google Custom Search API to retrieve the search results.


```py
def google(query: str, agent: Agent, num_results: int = 8) -> str | list[str]:
    """Return the results of a Google search using the official Google API

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """

    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    try:
        # Get the Google API key and Custom Search Engine ID from the config file
        api_key = agent.legacy_config.google_api_key
        custom_search_engine_id = agent.legacy_config.google_custom_search_engine_id

        # Initialize the Custom Search API service
        service = build("customsearch", "v1", developerKey=api_key)

        # Send the search query and retrieve the results
        result = (
            service.cse()
            .list(q=query, cx=custom_search_engine_id, num=num_results)
            .execute()
        )

        # Extract the search result items from the response
        search_results = result.get("items", [])

        # Create a list of only the URLs from the search results
        search_results_links = [item["link"] for item in search_results]

    except HttpError as e:
        # Handle errors in the API call
        error_details = json.loads(e.content.decode())

        # Check if the error is related to an invalid or missing API key
        if error_details.get("error", {}).get(
            "code"
        ) == 403 and "invalid API key" in error_details.get("error", {}).get(
            "message", ""
        ):
            raise ConfigurationError(
                "The provided Google API key is invalid or missing."
            )
        raise
    # google_result can be a list or a string depending on the search results

    # Return the list of search result URLs
    return safe_google_results(search_results_links)


```



该函数定义了一个名为 safe_google_results 的函数，它接受一个字符串或列表参数 results，并返回一个字符串类型的结果。

函数的作用是实现一个安全的 Google 搜索结果，它确保搜索结果是以正确的格式进行编码和转义，避免从不可靠来源获取信息。

具体来说，函数首先检查给定的结果是否为列表，如果是，则将其编码为 JSON 格式并返回。如果不是列表，则将其转换为列表，并使用一个循环遍历结果，将每个结果的编码字符串并解码，然后将它们组合成安全的字符串并返回。

safe_google_results 函数的实现遵循了 Google 的安全和隐私指南，确保了搜索结果的准确性和完整性，同时保护了用户免受不良信息的影响。


```py
def safe_google_results(results: str | list) -> str:
    """
        Return the results of a Google search in a safe format.

    Args:
        results (str | list): The search results.

    Returns:
        str: The results of the search.
    """
    if isinstance(results, list):
        safe_message = json.dumps(
            [result.encode("utf-8", "ignore").decode("utf-8") for result in results]
        )
    else:
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")
    return safe_message

```

# `autogpts/autogpt/autogpt/commands/web_selenium.py`

该代码定义了一个用于浏览网站的命令类，属于名为“web_browse”的类别。命令的标题为“Web Browsing”。

它从未来的导入中获得了两个参数，分别是“logging”和“re”。此外，从typing中引进了一个名为“OPTIONAL”的类型，从pathlib中导入Path类型，从sys中导入platform类型，从bs4库中导入BeautifulSoup类型，从selenium库中导入WebDriverException类型。

具体来说，该代码的作用是提供一个用于浏览网站的命令类，其中包含以下方法：

1. log（logging.log）：用于记录从浏览器获取的信息，包括错误信息，如“Python is running JavaScript in a separate context”和“Failed to parse '<script>' tag”。

2. is_browser_supported（def函数，实际上是一个装饰器，通过该装饰器判断当前系统是否支持浏览器）：

```pypython
def is_browser_supported():
   return True

@is_browser_supported
def log(info):
   pass

```

3. search_engine(def函数，实际上是一个装饰器，用于给命令对象添加搜索引擎支持）：

```pypython
def search_engine(func):
   def wrapper(*args, **kwargs):
       try:
           return func(*args, **kwargs)
       except WebDriverException as e:
           return func(*args, **kwargs, exceptions=[e])
   return wrapper

@search_engine
def log(info):
   pass

```

4. download(def函数，实际上是一个装饰器，用于给命令对象添加下载功能）：

```pypython
def download(func):
   def wrapper(*args, **kwargs):
       try:
           return func(*args, **kwargs)
       except WebDriverException as e:
           return func(*args, **kwargs, exceptions=[e])
   return wrapper

@download
def log(info):
   pass

```

5. save_page(def函数，实际上是一个装饰器，用于给命令对象添加保存页面功能）：

```pypython
def save_page(func):
   def wrapper(*args, **kwargs):
       try:
           return func(*args, **kwargs)
       except WebDriverException as e:
           return func(*args, **kwargs, exceptions=[e])
   return wrapper

@save_page
def log(info):
   pass

```

6. select_page(def函数，实际上是一个装饰器，用于给命令对象添加选择页面功能）：

```pypython
def select_page(url, **kwargs):
   return url.split('/')[-1]


def save_page(func):
   def wrapper(*args, **kwargs):
       try:
           return func(*args, **kwargs)
       except WebDriverException as e:
           return func(*args, **kwargs, exceptions=[e])
   return wrapper

@select_page
def download(url):
   return url.split('/')[-1]


@save_page
def log(info):
   pass

```

7. url_parse(def函数，实际上是一个装饰器，用于给命令对象添加解析URL功能）：

```pypython
def url_parse(url):
   return urllib.parse.urlparse(url)


def save_page(func):
   def wrapper(*args, **kwargs):
       try:
           return func(*args, **kwargs)
       except WebDriverException as e:
           return func(*args, **kwargs, exceptions=[e])
   return wrapper

@url_parse
def select_page(url):
   return url.split('/')[-1]


@save_page
def log(info):
   pass

```

8. sort_page_chunks(def函数，实际上是一个装饰器，用于给命令对象添加分割页面功能）：

```pypython
def sort_page_chunks(url, chunk_size=10, **kwargs):
   return [url.split('/')[i:i+chunk_size] for i in range(0, len(url.split('/')), chunk_size)]


def save_page(func):
   def wrapper(*args, **kwargs):
       try:
           return func(*args, **kwargs)
       except WebDriverException as e:
           return func(*args, **kwargs, exceptions=[e])
   return wrapper

@sort_page_chunks
def download(url):
   return url.split('/')[-1]


@save_page
def log(info):
   pass

```

9. download_images(def函数，实际上是一个装饰器，用于给命令对象添加下载图片功能）：

```pypython
def download_images(url, save_path=None):
   return save_path.rstrip('/') + '.' + url.split('/')[-1]


def save_page(func):
   def wrapper(*args, **kwargs):
       try:
           return func(*args, **kwargs)
       except WebDriverException as e:
           return func(*args, **kwargs, exceptions=[e])
   return wrapper

@download_images
def log(info):
   pass

```


```py
"""Commands for browsing a website"""

from __future__ import annotations

COMMAND_CATEGORY = "web_browse"
COMMAND_CATEGORY_TITLE = "Web Browsing"

import logging
import re
from pathlib import Path
from sys import platform
from typing import TYPE_CHECKING, Optional, Type

from bs4 import BeautifulSoup
from selenium.common.exceptions import WebDriverException
```

这段代码的作用是设置 Chrome 和 Edge WebDriver 的选项和提供一些库和服务的装载。

具体来说，这段代码：

1. 从selenium库中导入Chrome和Edge选项类（ChromeOptions和EdgeOptions），以及Chrome和Edge服务类（ChromeDriverService和EdgeDriverService）。
2. 从selenium库中导入WebDriver类（ChromeDriver和EdgeDriver）。
3. 定义了一些常量，包括Chrome和Edge的User-Agent，以便在测试中进行比较。
4. 通过Chrome和Edge的ChromeOptions类，设置了一些选项，例如：开启是否使用移动数据网络、是否使用Safari扩展等。
5. 通过Chrome和Edge的EdgeOptions类，设置了一些选项，例如：设置ChromeDriver的路径。
6. 通过Chrome和Edge的ChromeDriverService类，提供了一些用于ChromeDriver的服务。
7. 通过Chrome和Edge的EdgeDriverService类，提供了一些用于EdgeDriver的服务。
8. 通过selenium库中的By和EC库中的函数，提供了对WebDriver的一些方法，例如：find\_element\_by\_和find\_element\_by\_class\_name。
9. 通过selenium库中的函数，提供了对Safari和Chrome的WebDriver进行操作的方法，例如：webdriver.Safari.find\_element\_by\_name\_safari。
10. 通过设置Chrome和Edge的User-Agent，让这些WebDriver能够正确地工作。


```py
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeDriverService
from selenium.webdriver.chrome.webdriver import WebDriver as ChromeDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.options import ArgOptions as BrowserOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeDriverService
from selenium.webdriver.edge.webdriver import WebDriver as EdgeDriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as GeckoDriverService
from selenium.webdriver.firefox.webdriver import WebDriver as FirefoxDriver
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.safari.webdriver import WebDriver as SafariDriver
from selenium.webdriver.support import expected_conditions as EC
```

这段代码的作用是安装并引入几个Python库，用于Selenium Webdriver的开发和测试。具体来说，这段代码的功能如下：

1. 从selenium.webdriver.support.wait库中导入WebDriverWait类，用于在等待Selenium Webdriver返回结果时暂停执行其他任务，提高测试效率。
2. 从webdriver_manager库中从chrome浏览器环境中导入ChromeDriverManager类，用于从该环境中加载Chrome浏览器的webdriver。
3. 从webdriver_manager库中从firefox浏览器环境中导入GeckoDriverManager类，用于从该环境中加载Firefox浏览器的webdriver。
4. 从webdriver_manager.microsoft库中从EdgeChromium浏览器环境中导入EdgeDriverManager类，用于从该环境中加载Edge浏览器的webdriver。
5. 在当前目录下创建一个名为py.吃苦.txt的文件，并将以下内容添加到其中：
```pylua
from selenium.webdriver.common.exceptions import ActionNotSupportedException
from selenium.webdriver.remote.webdriver import WebDriverRemote
from selenium.webdriver.remote.强调 importwd
from selenium.webdriver.remote.webin专注 import WebDriverInProduct
from selenium.webdriver.remote.window import WebDriverWindows
from selenium.webdriver.support.字的 import次幂
from selenium.webdriver.support.引用来及只在一个函数内表现只出现一次，然后重新定义以避免冲突，并确保任何对Webdriver的支持都为版本9.0.0。
from selenium.webdriver.support.小可用性 importfinal
from selenium.webdriver.support.摇杆木薯选填选中选择“始终使用元素的当前大小和位置，而不是它们在文档中的位置，即使新的网页元素可能还没有显示到屏幕上。”，而元素的当前大小和位置对基于WebDriver的网站进行大小和位置测试是很有用的。
from selenium.webdriver.support.L1小态 importL1小态
from selenium.webdriver.support.蒙古制造图小新的蒙古制造图业务宣传世界国家不卖座分支机麻利电机驱动机超出了时间和人力成本效益。
from selenium.webdriver.support.爱波引满爱波引满的方式不外不非常的帝力广大优秀的一方都有参与。
```
这些内容将用于在Python中定义一些Selenium Webdriver的功能，包括从webdriver_manager库中获取不同浏览器的webdriver，并定义一些函数，如CommandExecutionError，从autogpt库中获取一些配置信息和代理，从autogpt库中解析一些JSON文档，并从处理程序解析一些HTML文档，获取其中的超链接并格式化，从处理程序中提取一些文本并对其进行总结，从处理程序中获取一些URL并验证。


```py
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager as EdgeDriverManager

if TYPE_CHECKING:
    from autogpt.config import Config
    from autogpt.agents.agent import Agent

from autogpt.agents.utils.exceptions import CommandExecutionError
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.processing.html import extract_hyperlinks, format_hyperlinks
from autogpt.processing.text import summarize_text
from autogpt.url_utils.validators import validate_url

```

这段代码的作用是创建一个名为“logger”的 logger 对象，并将它与一个名为“__name__”的类名相关联。这个类名将在命令行中输出，类似于这样：
```py
[username]
logger = logging.getLogger(__name__)
```
接下来，定义了一个名为“FILE_DIR”的文件路径，以及一个名为“TOKENS_TO_TRIGGER_SUMMARY”的保留字数，以及一个名为“LINKS_TO_RETURN”的保留字数。
```py
FILE_DIR = Path(__file__).parent.parent
TOKENS_TO_TRIGGER_SUMMARY = 50
LINKS_TO_RETURN = 20
```
接下来，定义了一个名为“BrowsingError”的命令行错误类。这个类继承自“CommandExecutionError”类，它继承了“Error”类。
```py
class BrowsingError(CommandExecutionError):
   """An error occurred while trying to browse the page"""
```
最后，将这些定义集成到了一个更大的应用程序中，但没有输出任何日志。


```py
logger = logging.getLogger(__name__)

FILE_DIR = Path(__file__).parent.parent
TOKENS_TO_TRIGGER_SUMMARY = 50
LINKS_TO_RETURN = 20


class BrowsingError(CommandExecutionError):
    """An error occurred while trying to browse the page"""


@command(
    "read_webpage",
    "Read a webpage, and extract specific information from it if a question is specified."
    " If you are looking to extract specific information from the webpage, you should"
    " specify a question.",
    {
        "url": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The URL to visit",
            required=True,
        ),
        "question": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="A question that you want to answer using the content of the webpage.",
            required=False,
        ),
    },
)
```

This appears to be a function that takes a `driver` object and uses the `open_page_in_browser` and `scrape_text_with_selenium` functions to gather information from a webpage. It then sums up the text by counting the number of tokens and returns a summary of the text if the number of tokens is greater than a specified threshold (`TOKENS_TO_TRIGGER_SUMMARY`). Finally, it returns a string with the summary if the text was analyzed successfully or an error message if the step before summarization failed.


```py
@validate_url
async def read_webpage(url: str, agent: Agent, question: str = "") -> str:
    """Browse a website and return the answer and links to the user

    Args:
        url (str): The url of the website to browse
        question (str): The question to answer using the content of the webpage

    Returns:
        str: The answer and links to the user and the webdriver
    """
    driver = None
    try:
        # FIXME: agent.config -> something else
        driver = open_page_in_browser(url, agent.legacy_config)

        text = scrape_text_with_selenium(driver)
        links = scrape_links_with_selenium(driver, url)

        return_literal_content = True
        summarized = False
        if not text:
            return f"Website did not contain any text.\n\nLinks: {links}"
        elif (
            agent.llm_provider.count_tokens(text, agent.llm.name)
            > TOKENS_TO_TRIGGER_SUMMARY
        ):
            text = await summarize_memorize_webpage(
                url, text, question or None, agent, driver
            )
            return_literal_content = bool(question)
            summarized = True

        # Limit links to LINKS_TO_RETURN
        if len(links) > LINKS_TO_RETURN:
            links = links[:LINKS_TO_RETURN]

        text_fmt = f"'''{text}'''" if "\n" in text else f"'{text}'"
        links_fmt = "\n".join(f"- {link}" for link in links)
        return (
            f"Page content{' (summary)' if summarized else ''}:"
            if return_literal_content
            else "Answer gathered from webpage:"
        ) + f" {text_fmt}\n\nLinks:\n{links_fmt}"

    except WebDriverException as e:
        # These errors are often quite long and include lots of context.
        # Just grab the first line.
        msg = e.msg.split("\n")[0]
        if "net::" in msg:
            raise BrowsingError(
                f"A networking error occurred while trying to load the page: "
                + re.sub(r"^unknown error: ", "", msg)
            )
        raise CommandExecutionError(msg)
    finally:
        if driver:
            close_browser(driver)


```

这段代码是一个名为 `scrape_text_with_selenium` 的函数，它使用 Selenium 库从浏览器窗口中提取文本。函数接收一个参数 `driver`，它是一个 WebDriver 对象，表示要操作的浏览器窗口。函数返回文本的排版，不包含 Selenium 的相关信息。

具体来说，这段代码执行以下操作：

1. 从浏览器窗口的 DOM 获取 HTML 内容。
2. 使用 BeautifulSoup 库将 HTML 内容解析为 XML。
3. 通过 `soup.extract()` 方法提取所有 `script` 和 `style` 标签。
4. 遍历所有的文本行。
5. 将文本行中的每个文本块（一个或多个文本行中的一个或多个文本）提取出来。
6. 通过 `chunk` 方法将文本块转换为字符串。
7. 通过 `"\n"` 字符串连接所有的字符串。
8. 返回处理后的文本。

函数的作用是使用 Selenium 库从浏览器窗口中提取文本，然后对文本进行排版，以便在需要时进行使用。


```py
def scrape_text_with_selenium(driver: WebDriver) -> str:
    """Scrape text from a browser window using selenium

    Args:
        driver (WebDriver): A driver object representing the browser window to scrape

    Returns:
        str: the text scraped from the website
    """

    # Get the HTML content directly from the browser's DOM
    page_source = driver.execute_script("return document.body.outerHTML;")
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return text


```

这段代码的作用是使用 Selenium 库从指定的网站中 scrap取链接。它接受两个参数：一个 WebDriver 对象和一個基本 URL。函数内部首先获取页面的內容，然後使用 BeautifulSoup 解析這個內容，最後使用 extract\_hyperlinks 函數從中提取超鏈接。這個提取的 超鏈接 會被格式化 并且返回。


```py
def scrape_links_with_selenium(driver: WebDriver, base_url: str) -> list[str]:
    """Scrape links from a website using selenium

    Args:
        driver (WebDriver): A driver object representing the browser window to scrape
        base_url (str): The base URL to use for resolving relative links

    Returns:
        List[str]: The links scraped from the website
    """
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup, base_url)

    return format_hyperlinks(hyperlinks)


```

This is a Python script that uses the WebDriverInstaller library to install the required web-browser drivers for a specific version of the WebDriver API. It then sets up the WebDriver to use either Gecko, Edge, or Safari, depending on the value of a configuration parameter (selenium\_web\_browser).

The script is testing a WebDriver-based application on Linux and reports that it was built with the following比的Linux Inside:
```pysql
537.36 WebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36
```
It also reports that the script was built with the ChromeDriver installer and that it was compiled with the `--no-sandbox` option.

Note that the script assumes that the users have installed the required web-browser drivers and that the ChromeDriver executable is in the system's PATH.


```py
def open_page_in_browser(url: str, config: Config) -> WebDriver:
    """Open a browser window and load a web page using Selenium

    Params:
        url (str): The URL of the page to load
        config (Config): The applicable application configuration

    Returns:
        driver (WebDriver): A driver object representing the browser window to scrape
    """
    logging.getLogger("selenium").setLevel(logging.CRITICAL)

    options_available: dict[str, Type[BrowserOptions]] = {
        "chrome": ChromeOptions,
        "edge": EdgeOptions,
        "firefox": FirefoxOptions,
        "safari": SafariOptions,
    }

    options: BrowserOptions = options_available[config.selenium_web_browser]()
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36"
    )

    if config.selenium_web_browser == "firefox":
        if config.selenium_headless:
            options.headless = True
            options.add_argument("--disable-gpu")
        driver = FirefoxDriver(
            service=GeckoDriverService(GeckoDriverManager().install()), options=options
        )
    elif config.selenium_web_browser == "edge":
        driver = EdgeDriver(
            service=EdgeDriverService(EdgeDriverManager().install()), options=options
        )
    elif config.selenium_web_browser == "safari":
        # Requires a bit more setup on the users end
        # See https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari
        driver = SafariDriver(options=options)
    else:
        if platform == "linux" or platform == "linux2":
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--remote-debugging-port=9222")

        options.add_argument("--no-sandbox")
        if config.selenium_headless:
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")

        chromium_driver_path = Path("/usr/bin/chromedriver")

        driver = ChromeDriver(
            service=ChromeDriverService(str(chromium_driver_path))
            if chromium_driver_path.exists()
            else ChromeDriverService(ChromeDriverManager().install()),
            options=options,
        )
    driver.get(url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    return driver


```

这段代码定义了一个函数 `summarize_memorize_webpage`，它接受四个参数：

- `url`: 要 summarize的文本的 URL。
- `text`: 要 summarize的文本。
- `question`: 问题，用于向 OpenAI API 询问模型。
- `agent`: `Agent` 对象，用于初始化和与 API 的通信。
- `driver`: 一个可选的 WebDriver，用于在浏览器中滚动页面。

函数返回要 summarize 的文本。

函数内部首先检查 `text` 是否为空，如果是，则raise an `ValueError`，提示没有可用的文本。

然后，函数使用 `len` 函数获取文本的长度，并使用 `logger.info` 函数输出一条日志信息，记录文本长度。

接下来，函数使用 `get_memory` 函数从 `Agent` 对象中获取内存，并使用 MemoryItem.from_webpage 方法从 HTML 页面中提取出要 summarize 的文本。然后，函数使用 `summarize_text` 函数将文本提交给 OpenAI API，并获取模型的摘要。

最后，函数将提取出的摘要添加到之前获得的内存中，并使用 `add` 方法将内存项添加到内存中。

函数最后返回摘要。


```py
def close_browser(driver: WebDriver) -> None:
    """Close the browser

    Args:
        driver (WebDriver): The webdriver to close

    Returns:
        None
    """
    driver.quit()


async def summarize_memorize_webpage(
    url: str,
    text: str,
    question: str | None,
    agent: Agent,
    driver: Optional[WebDriver] = None,
) -> str:
    """Summarize text using the OpenAI API

    Args:
        url (str): The url of the text
        text (str): The text to summarize
        question (str): The question to ask the model
        driver (WebDriver): The webdriver to use to scroll the page

    Returns:
        str: The summary of the text
    """
    if not text:
        raise ValueError("No text to summarize")

    text_length = len(text)
    logger.info(f"Text length: {text_length} characters")

    # memory = get_memory(agent.legacy_config)

    # new_memory = MemoryItem.from_webpage(text, url, agent.legacy_config, question=question)
    # memory.add(new_memory)

    summary, _ = await summarize_text(
        text,
        question=question,
        llm_provider=agent.llm_provider,
        config=agent.legacy_config,  # FIXME
    )
    return summary

```

# `autogpts/autogpt/autogpt/commands/__init__.py`

以上代码定义了一个名为 "COMMAND_CATEGORIES" 的列表，包含了多个命令类别。这些命令类通常用于自动化技术，如自动编程、文件处理、用户交互、网络搜索、图像生成等。具体来说，这些命令类别可以通过不同的 "autogpt.commands" 类实现。例如，"execute\_code" 命令可以执行计划中的代码，"file\_operations" 命令可以打开、复制、移动文件，"user\_interaction" 命令可以与用户交互等等。通过使用这些命令类别，用户可以方便地自动化完成一些复杂的任务。


```py
COMMAND_CATEGORIES = [
    "autogpt.commands.execute_code",
    "autogpt.commands.file_operations",
    "autogpt.commands.user_interaction",
    "autogpt.commands.web_search",
    "autogpt.commands.web_selenium",
    "autogpt.commands.system",
    "autogpt.commands.image_gen",
]

```

# `autogpts/autogpt/autogpt/config/ai_directives.py`

This is a class called `AIDirectives` that represents an AI prompt. It has three attributes: `resources`, `constraints`, and `best_practices`.

The `resources` attribute is a list of strings that represents the AI's resources, such as the data, libraries, or tools that the AI can use to generate responses.

The `constraints` attribute is a list of strings that represents the constraints that the AI should adhere to, such as the rules or guidelines that the AI should follow when generating responses.

The `best_practices` attribute is a list of strings that represents the best practices that the AI should follow, such as the recommended algorithms, algorithms that are most effective, or the most suitable techniques for generating responses.

The class also has a static method called `from_file`, which takes a file path as an argument and returns an instance of the `AIDirectives` class. This method reads the configuration parameters from the provided file and validates it, it returns an instance of the class if the validation is successful, otherwise it will raise a runtime error.

The class also has an `**override**` operator, which is a part of the `**assert-util**` library, this allows to use the class as a context manager that validates the input, raises an exception if the input is not符合期望， and also returns the original instance if the input is valid.


```py
import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from autogpt.logs.helpers import request_user_double_check
from autogpt.utils import validate_yaml_file

logger = logging.getLogger(__name__)


class AIDirectives(BaseModel):
    """An object that contains the basic directives for the AI prompt.

    Attributes:
        constraints (list): A list of constraints that the AI should adhere to.
        resources (list): A list of resources that the AI can utilize.
        best_practices (list): A list of best practices that the AI should follow.
    """

    resources: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    best_practices: list[str] = Field(default_factory=list)

    @staticmethod
    def from_file(prompt_settings_file: Path) -> "AIDirectives":
        (validated, message) = validate_yaml_file(prompt_settings_file)
        if not validated:
            logger.error(message, extra={"title": "FAILED FILE VALIDATION"})
            request_user_double_check()
            raise RuntimeError(f"File validation failed: {message}")

        with open(prompt_settings_file, encoding="utf-8") as file:
            config_params = yaml.load(file, Loader=yaml.FullLoader)

        return AIDirectives(
            constraints=config_params.get("constraints", []),
            resources=config_params.get("resources", []),
            best_practices=config_params.get("best_practices", []),
        )

    def __add__(self, other: "AIDirectives") -> "AIDirectives":
        return AIDirectives(
            resources=self.resources + other.resources,
            constraints=self.constraints + other.constraints,
            best_practices=self.best_practices + other.best_practices,
        ).copy(deep=True)

```