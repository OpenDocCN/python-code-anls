# AutoGPT源码解析 21

# `autogpts/forge/forge/sdk/llm.py`

这段代码使用了多种不同的技巧来实现对用户的交互。首先，它使用`typing`模块中的`Union`类型来尝试从不同的来源获取响应，这将生成一个带有或不带参数的任意类型的响应。

其次，它使用`openai`库与用户的交互。具体来说，它使用`openai.Prompt`类来生成一个自然语言 prompt，然后使用`openai.Response`类来生成一个响应。

接着，它使用`forge_log`类来记录对话中的信息。`ForgeLogger`类是一个简单的日志库，它使用`__name__`作为其唯一标识符。

此外，它还使用`litellm`库来实现自动完成。`completion`函数使用Prompt V2类型来生成一个自然语言 prompt，然后使用`acquisition`函数来获取用户的回答。如果回答正确，该函数将返回一个`completion.Completion`对象。

最后，它使用`tw common`库中的`stop_after_attempt`函数来限制尝试次数。`stop_after_attempt`函数接受两个参数：一个是尝试次数，另一个是超时时间(以秒为单位)。如果尝试次数达到了设置的值，它将停止尝试，防止一直以来的错误。


```py
import typing

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .forge_log import ForgeLogger
from litellm import completion, acompletion, AuthenticationError, InvalidRequestError

LOG = ForgeLogger(__name__)


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def chat_completion_request(
    model, messages, **kwargs
) -> typing.Union[typing.Dict[str, typing.Any], Exception]:
    """Generate a response to a list of messages using OpenAI's API"""
    try:
        kwargs["model"] = model
        kwargs["messages"] = messages

        resp = await acompletion(**kwargs)
        return resp
    except AuthenticationError as e:
        LOG.exception("Authentication Error")
    except InvalidRequestError as e:
        LOG.exception("Invalid Request Error")
    except Exception as e:
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise


```

这段代码是一个异步函数，名为 `create_embedding_request`，它接受一个名为 `messages` 的参数。该函数使用 OpenAI 的 API 来为传入的每个消息生成一个嵌入。

具体来说，函数内部首先定义了一个参数 `min` 和 `max`，它们指定了在生成嵌入时允许的最小和最大值。然后，定义了一个参数 `stop_after_attempt`，它指定了在尝试多少次失败后停止尝试。接着，定义了一个名为 `wait_random_exponential` 的函数，该函数使用 `min` 和 `max` 参数来生成一个随机数，用于在生成失败时停止尝试。最后，定义了函数内部的 `try`/`except` 块，用于处理在生成嵌入时可能发生的异常。

函数内部首先将传入的每个消息转换为一个字串，然后使用 `openai.Embedding.acreate` 函数来生成消息的嵌入。如果生成的嵌入发生失败或者失败后已经尝试了 `stop_after_attempt` 次，函数将记录到日志中并抛出异常。


```py
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def create_embedding_request(
    messages, model="text-embedding-ada-002"
) -> typing.Union[typing.Dict[str, typing.Any], Exception]:
    """Generate an embedding for a list of messages using OpenAI's API"""
    try:
        return await openai.Embedding.acreate(
            input=[f"{m['role']}: {m['content']}" for m in messages],
            engine=model,
        )
    except Exception as e:
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise


```

这段代码定义了一个名为 `transcribe_audio` 的函数，它接受一个音频文件路径作为参数，并使用 OpenAI 的 API 对该文件进行文本转录。

函数内部使用 `@retry` 装饰器来处理可能出现的错误。`@retry` 装饰器会尝试运行函数一次，如果失败，则会根据定义的 `min` 和 `max` 参数设置超时时间，即在一定时间内再次尝试。

函数内部使用 `wait_random_exponential` 函数来生成一个介于 1 和 40 之间的随机数，用于生成新的超时时间。

函数内部使用 `asyncio` 中的 `await` 关键字来等待 API 请求的结果。函数的参数 `audio_file` 是字符串类型，表示要转录的音频文件路径。

函数最终返回一个 `typing.Union` 类型，其中包含两个可能的结果：成功转录的 `dict` 类型，或者异常信息。如果函数出现异常，则将捕获并打印异常信息，然后返回 `Exception` 异常。如果函数成功转录，则返回转录结果的 `dict` 类型。


```py
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def transcribe_audio(
    audio_file: str,
) -> typing.Union[typing.Dict[str, typing.Any], Exception]:
    """Transcribe an audio file using OpenAI's API"""
    try:
        return await openai.Audio.transcribe(model="whisper-1", file=audio_file)
    except Exception as e:
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise

```

# `autogpts/forge/forge/sdk/middlewares.py`

这段代码定义了一个AgentMiddleware类，用于在FastAPI应用程序中注入一个代理实例，并将该代理实例注入到请求上下文中。

具体来说，在__init__方法中，首先创建一个FastAPI应用程序和一个代理实例，然后将代理实例注入到应用程序的代理实例中。这个注入的代理实例可以在应用程序的上下文中使用，从而使应用程序能够访问代理实例中完成的中间件逻辑。

在__call__方法中，异步执行代理实例中执行的中间件逻辑。这个方法将上下文中的代理实例、请求和响应对象作为参数传递给代理人，然后执行代理实例中的业务逻辑。在异步执行完业务逻辑后，将结果返回给应用程序。


```py
from fastapi import FastAPI


class AgentMiddleware:
    """
    Middleware that injects the agent instance into the request scope.
    """

    def __init__(self, app: FastAPI, agent: "Agent"):
        """

        Args:
            app: The FastAPI app - automatically injected by FastAPI.
            agent: The agent instance to inject into the request scope.

        Examples:
            >>> from fastapi import FastAPI, Request
            >>> from agent_protocol.agent import Agent
            >>> from agent_protocol.middlewares import AgentMiddleware
            >>> app = FastAPI()
            >>> @app.get("/")
            >>> async def root(request: Request):
            >>>     agent = request["agent"]
            >>>     task = agent.db.create_task("Do something.")
            >>>     return {"task_id": a.task_id}
            >>> agent = Agent()
            >>> app.add_middleware(AgentMiddleware, agent=agent)
        """
        self.app = app
        self.agent = agent

    async def __call__(self, scope, receive, send):
        scope["agent"] = self.agent
        await self.app(scope, receive, send)

```

# `autogpts/forge/forge/sdk/prompting.py`

这段代码定义了一个名为 `PromptEngine` 的类，用于加载和使用 prompt 模板。

在 `__init__` 方法中，没有做任何初始化操作，因为题目中没有给出需要加载的 prompt 模板。

`load_prompt` 方法接受两个参数，一个是模型目录（在代码中使用 `model` 表示），另一个是提示名称（在代码中使用 `prompt_name` 表示）和提示标记（在代码中使用 `prompt_ags` 表示）。

根据题目描述，这段代码可能是一个自动编程工具，可以根据一个特定的模型名称和提示信息加载相应的 prompt 模板，并返回模板字符串。


```py
"""
Relative to this file I will have a prompt directory its located ../prompts
In this directory there will be a techniques directory and a directory for each model - gpt-3.5-turbo gpt-4, llama-2-70B, code-llama-7B etc

Each directory will have jinga2 templates for the prompts.
prompts in the model directories can use the techniques in the techniques directory.

Write the code I'd need to load and populate the templates.

I want the following functions:

class PromptEngine:

    def __init__(self, model):
        pass

    def load_prompt(model, prompt_name, prompt_ags) -> str:
        pass
```



This class appears to be a implementation of a的自然语言处理中的一个 clos尾模式。模式中包含一个训练好的语言模型，用来对传入的文本进行分析和匹配。当需要对一个特定的文本进行处理时，会首先查找该文本在所有的训练好的模型中，哪一篇模型的描述最贴近于需要处理的那篇文本，然后就从那篇模型的输出开始进行处理，直到得到需要的输出或者可以检测到模型没有输出为止。

具体来说，它包含以下方法：

- `get_closest_match(target, model_dirs, n=1, cutoff=0.1)`: returns与传入目标最相似的模型的描述。
- `load_prompt(template, **kwargs)`：加载并处理传入的模板。其中`template`是模板的名称，`**kwargs`是模板中需要填写的参数。返回处理后的模板。

该类还有一个`model_dirs`参数，它是一个包含多个模型目录的列表。在加载模板时，先从这些目录中查找需要的模型，如果找不到需要的模型，会在模型目录中继续查找，直到找到需要的模型或者可以检测到模型没有输出为止。


```py
"""

import glob
import os
from difflib import get_close_matches
from typing import List

from jinja2 import Environment, FileSystemLoader

from .forge_log import ForgeLogger

LOG = ForgeLogger(__name__)


class PromptEngine:
    """
    Class to handle loading and populating Jinja2 templates for prompts.
    """

    def __init__(self, model: str, debug_enabled: bool = False):
        """
        Initialize the PromptEngine with the specified model.

        Args:
            model (str): The model to use for loading prompts.
            debug_enabled (bool): Enable or disable debug logging.
        """
        self.model = model
        self.debug_enabled = debug_enabled
        if self.debug_enabled:
            LOG.debug(f"Initializing PromptEngine for model: {model}")

        try:
            # Get the list of all model directories
            models_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../prompts")
            )
            model_names = [
                os.path.basename(os.path.normpath(d))
                for d in glob.glob(os.path.join(models_dir, "*/"))
                if os.path.isdir(d) and "techniques" not in d
            ]

            self.model = self.get_closest_match(self.model, model_names)

            if self.debug_enabled:
                LOG.debug(f"Using the closest match model for prompts: {self.model}")

            self.env = Environment(loader=FileSystemLoader(models_dir))
        except Exception as e:
            LOG.error(f"Error initializing Environment: {e}")
            raise

    @staticmethod
    def get_closest_match(target: str, model_dirs: List[str]) -> str:
        """
        Find the closest match to the target in the list of model directories.

        Args:
            target (str): The target model.
            model_dirs (list): The list of available model directories.

        Returns:
            str: The closest match to the target.
        """
        try:
            matches = get_close_matches(target, model_dirs, n=1, cutoff=0.1)
            if matches:
                matches_str = ", ".join(matches)
                LOG.debug(matches_str)
            for m in matches:
                LOG.info(m)
            return matches[0]
        except Exception as e:
            LOG.error(f"Error finding closest match: {e}")
            raise

    def load_prompt(self, template: str, **kwargs) -> str:
        """
        Load and populate the specified template.

        Args:
            template (str): The name of the template to load.
            **kwargs: The arguments to populate the template with.

        Returns:
            str: The populated template.
        """
        try:
            template = os.path.join(self.model, template)
            if self.debug_enabled:
                LOG.debug(f"Loading template: {template}")
            template = self.env.get_template(f"{template}.j2")
            if self.debug_enabled:
                LOG.debug(f"Rendering template: {template} with args: {kwargs}")
            return template.render(**kwargs)
        except Exception as e:
            LOG.error(f"Error loading or rendering template: {e}")
            raise

```

# `autogpts/forge/forge/sdk/schema.py`

这段代码定义了一个名为ArtifactUpload的类，其父类是EnableCase一行文。

这个类的定义了一个文件上传的API，模型使用了从f-{区间}区中选择的一行文，并包含一个文件路径字段和一个相对路径字段。

文件上传到附件可以在附件根目录中找到，可以通过执行cv2.imread()函数中的cv2.VideoCapture()函数从摄像头中读取视频流。

这个类的定义对于在 FastAPI 应用程序中处理文件上传附件，提供了简单而一致的接口。


```py
# generated by fastapi-codegen:
#   filename:  ../../postman/schemas/openapi.yaml
#   timestamp: 2023-08-25T10:36:11+00:00

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ArtifactUpload(BaseModel):
    file: str = Field(..., description="File to upload.", format="binary")
    relative_path: str = Field(
        ...,
        description="Relative path of the artifact in the agent's workspace.",
        example="python/code",
    )


```



这个代码定义了两个模型，Pagination 和 Artifact。它们都是基于SQLAlchemy ORM框架的数据模型，使用了Python的五星模型(Pydantic)来定义这些模型的同时具备了数据验证和自动补全功能。

Pagination模型定义了分页组件中的各种属性，包括总共有多少条记录，每页有多少条记录，以及当前页的编号。这些属性都是整数类型，使用了Field来定义，使用了description来描述这些属性。

Artifact模型定义了记录组件中的各种属性，包括记录的创建时间、修改时间、ID、是否由客户端创建、以及文件在客户端 workspace 中的相对路径。这些属性也都是整数类型，使用了Field来定义，使用了description来描述这些属性。与Pagination模型不同的是，Artifact模型的FileName属性使用了FileField来定义。

这两个模型都使用了datetime和datetime疑惑来处理日期时间类型的数据，并且使用了lambda函数来将日期时间格式化为标准的日期时间格式。


```py
class Pagination(BaseModel):
    total_items: int = Field(..., description="Total number of items.", example=42)
    total_pages: int = Field(..., description="Total number of pages.", example=97)
    current_page: int = Field(..., description="Current_page page number.", example=1)
    page_size: int = Field(..., description="Number of items per page.", example=25)


class Artifact(BaseModel):
    created_at: datetime = Field(
        ...,
        description="The creation datetime of the task.",
        example="2023-01-01T00:00:00Z",
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    modified_at: datetime = Field(
        ...,
        description="The modification datetime of the task.",
        example="2023-01-01T00:00:00Z",
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    artifact_id: str = Field(
        ...,
        description="ID of the artifact.",
        example="b225e278-8b4c-4f99-a696-8facf19f0e56",
    )
    agent_created: bool = Field(
        ...,
        description="Whether the artifact has been created by the agent.",
        example=False,
    )
    relative_path: str = Field(
        ...,
        description="Relative path of the artifact in the agents workspace.",
        example="/my_folder/my_other_folder/",
    )
    file_name: str = Field(
        ...,
        description="Filename of the artifact.",
        example="main.py",
    )


```

这段代码定义了一个名为 "StepOutput" 的类 "StepOutput"，它是从名为 "BaseModel" 的基类派生的。但是，它没有定义任何方法或变量。

再定义了一个名为 "TaskRequestBody" 的类 "TaskRequestBody"，它也来自 "BaseModel"，并且包含一个名为 "input" 的字段。这个字段有一个最小长度为 1 的输入 prompt，描述为 "Input prompt for the task."，以及一个示例值 "Write the words you receive to the file 'output.txt'."，同时它还包含一个名为 "additional_input" 的可选字典，用于提供更多的输入数据。

最后，定义了一个名为 "Task" 的类 "Task"，它是 "TaskRequestBody" 的实例。它包含了一些描述性的字段，包括创建日期和修改日期，以及一个名为 "artifacts" 的可选列表 "TaskArtifacts"。它还包含一个名为 "created_at" 和 "modified_at" 的字段，用于记录创建和修改日期。


```py
class StepOutput(BaseModel):
    pass


class TaskRequestBody(BaseModel):
    input: str = Field(
        ...,
        min_length=1,
        description="Input prompt for the task.",
        example="Write the words you receive to the file 'output.txt'.",
    )
    additional_input: Optional[dict] = {}


class Task(TaskRequestBody):
    created_at: datetime = Field(
        ...,
        description="The creation datetime of the task.",
        example="2023-01-01T00:00:00Z",
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    modified_at: datetime = Field(
        ...,
        description="The modification datetime of the task.",
        example="2023-01-01T00:00:00Z",
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    task_id: str = Field(
        ...,
        description="The ID of the task.",
        example="50da533e-3904-4401-8a07-c49adf88b5eb",
    )
    artifacts: Optional[List[Artifact]] = Field(
        [],
        description="A list of artifacts that the task has produced.",
        example=[
            "7a49f31c-f9c6-4346-a22c-e32bc5af4d8e",
            "ab7b4091-2560-4692-a4fe-d831ea3ca7d6",
        ],
    )


```

这段代码定义了一个名为 "StepRequestBody" 的类，它继承自 "BaseModel" 类。这个类定义了一个 "StepRequestBody" 对象的属性和元数据，用于请求一个步骤的提交。

具体来说，这个类定义了一个名为 "name" 的字段，它的类型是一个可选的字符串，用于指定请求的步骤名称。它还定义了一个名为 "input" 的字段，它的类型也是可选的字符串，用于指定请求步骤的输入提示。最后，它定义了一个名为 "additional_input" 的字典，用于存储除了名称和输入提示之外的其他信息，以帮助请求更多的上下文信息。

此外，这个类还定义了一个名为 "Status" 的枚举类型，它包含了一些与请求状态相关的枚举值，如 "created"、"running" 和 "completed"。

整个类定义了一个 StepRequestBody 类对象，该对象可以使用 name、input 和 additional_input 属性来请求一个步骤的提交。此外，它还定义了一个名为 Status 的枚举类型，用于指定请求的状态，该类型包含了一些枚举值，如 "created"、"running" 和 "completed"。


```py
class StepRequestBody(BaseModel):
    name: Optional[str] = Field(
        None, description="The name of the task step.", example="Write to file"
    )
    input: Optional[str] = Field(
        None,
        description="Input prompt for the step.",
        example="Washington",
    )
    additional_input: Optional[dict] = {}


class Status(Enum):
    created = "created"
    running = "running"
    completed = "completed"


```



This is a Python `Step` class that represents a task step. The `Step` class has several fields that provide information about the task step, such as its creation and modification dates, its task and step IDs, its name and status, its output and any additional output, its artifacts, and whether it is the last step in the task or not.

The `Step` class also has a `created_at` and `modified_at` field that provide the creation and modification dates of the task step, respectively. These fields are defined as a `datetime` type and are stored as ISO 8601 formatted strings in Python.

The `get_义词` 函数 is used to convert the `name` field of the `Step` class to lowercase.

The `Column` class is used to create the columns of the `Step` class. The `datetime` column is defined as a subclass of the `不想实现为空` 的 `Field` class and is used to store the creation and modification dates of the task step.


```py
class Step(StepRequestBody):
    created_at: datetime = Field(
        ...,
        description="The creation datetime of the task.",
        example="2023-01-01T00:00:00Z",
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    modified_at: datetime = Field(
        ...,
        description="The modification datetime of the task.",
        example="2023-01-01T00:00:00Z",
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    task_id: str = Field(
        ...,
        description="The ID of the task this step belongs to.",
        example="50da533e-3904-4401-8a07-c49adf88b5eb",
    )
    step_id: str = Field(
        ...,
        description="The ID of the task step.",
        example="6bb1801a-fd80-45e8-899a-4dd723cc602e",
    )
    name: Optional[str] = Field(
        None, description="The name of the task step.", example="Write to file"
    )
    status: Status = Field(
        ..., description="The status of the task step.", example="created"
    )
    output: Optional[str] = Field(
        None,
        description="Output of the task step.",
        example="I am going to use the write_to_file command and write Washington to a file called output.txt <write_to_file('output.txt', 'Washington')",
    )
    additional_output: Optional[StepOutput] = Field(default_factory=dict)
    artifacts: Optional[List[Artifact]] = Field(
        [], description="A list of artifacts that the step has produced."
    )
    is_last: bool = Field(
        ..., description="Whether this is the last step in the task.", example=True
    )


```

这段代码定义了三个类：TaskListResponse、TaskStepsListResponse和TaskArtifactsListResponse。它们都属于BaseModel类，使用了Optional类型来表示可能存在的某个元素，并使用了Pagination类型来表示分页信息。

具体来说，TaskListResponse类有两个可选的属性tasks和pagination，分别表示任务列表和分页信息。TaskStepsListResponse类和TaskArtifactsListResponse类也有类似的可选属性。

这些类的定义可能会被用于对API的响应进行处理和解析。例如，一个后端API可能会返回一个包含任务列表、步骤列表和 artifacts 的响应，那么可以使用这些类来定义这些数据结构，并从API的返回数据中获取具体的信息。


```py
class TaskListResponse(BaseModel):
    tasks: Optional[List[Task]] = None
    pagination: Optional[Pagination] = None


class TaskStepsListResponse(BaseModel):
    steps: Optional[List[Step]] = None
    pagination: Optional[Pagination] = None


class TaskArtifactsListResponse(BaseModel):
    artifacts: Optional[List[Artifact]] = None
    pagination: Optional[Pagination] = None

```

# `autogpts/forge/forge/sdk/workspace.py`

这段代码定义了一个名为 "Workspace" 的类，其继承自 "abc"(即 "ABC") 类，使用 "ABC" 作为其基类。这个 "Workspace" 类提供了一些方法，用于读取、写入、删除和存在等操作。这些方法的具体实现由子类 "WorkspaceABC" 实现。

具体来说，这个 "Workspace" 类包含以下方法：

- `__init__`：用于初始化 "Workspace" 对象，需要提供一个 "base_path" 参数，即对象的基础路径。
- `read`：用于读取文件，需要传入两个参数，一个是 "task_id"，表示任务编号，另一个是 "path"，表示需要读取的文件路径。这个方法的具体实现由子类 "WorkspaceABC" 实现。
- `write`：用于写入文件，需要传入两个参数，一个是 "task_id"，表示任务编号，另一个是 "path"，表示需要写入的文件路径和数据。这个方法的具体实现由子类 "WorkspaceABC" 实现。
- `delete`：用于删除文件或目录，需要传入四个参数，分别是 "task_id"、"path"、"directory" 和 "recursive"(表示是否递归删除)，分别表示任务编号、需要删除的文件或目录的路径是否为目录、是否递归。这个方法的具体实现由子类 "WorkspaceABC" 实现。
- `__bool__`：用于存在判断，需要传入一个参数 "task_id"，表示需要判断的任务编号。这个方法的具体实现由子类 "WorkspaceABC" 实现。
- `__list__`：用于列出目录内容，需要传入一个参数 "task_id"，表示需要列出的目录内容。这个方法的具体实现由子类 "WorkspaceABC" 实现。


```py
import abc
import os
import typing
from pathlib import Path


class Workspace(abc.ABC):
    @abc.abstractclassmethod
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path

    @abc.abstractclassmethod
    def read(self, task_id: str, path: str) -> bytes:
        pass

    @abc.abstractclassmethod
    def write(self, task_id: str, path: str, data: bytes) -> None:
        pass

    @abc.abstractclassmethod
    def delete(
        self, task_id: str, path: str, directory: bool = False, recursive: bool = False
    ) -> None:
        pass

    @abc.abstractclassmethod
    def exists(self, task_id: str, path: str) -> bool:
        pass

    @abc.abstractclassmethod
    def list(self, task_id: str, path: str) -> typing.List[str]:
        pass


```

This is a class that provides a simple file system interface for Python programs. It allows for directory traversal (reading, writing, and deleting files and folders) and provides an easy way to resolve file paths based on a base path.

The `File` class has methods for reading, writing, and deleting files. It also provides methods for directory traversal, which allows it to resolve file paths relative to the base path.

The `File` class has a `__init__` method that takes a base path and a task ID. It also has a `_resolve_path` method that resolves a file path relative to the base path.

The `File` class has a `exists` method that checks if a file exists in a given directory. It also has a `list` method that returns a list of files in a given directory.

Overall, this class provides a simple and intuitive file system interface for Python programs.


```py
class LocalWorkspace(Workspace):
    def __init__(self, base_path: str):
        self.base_path = Path(base_path).resolve()

    def _resolve_path(self, task_id: str, path: str) -> Path:
        path = str(path)
        path = path if not path.startswith("/") else path[1:]
        abs_path = (self.base_path / task_id / path).resolve()
        if not str(abs_path).startswith(str(self.base_path)):
            print("Error")
            raise ValueError(f"Directory traversal is not allowed! - {abs_path}")
        try:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass
        return abs_path

    def read(self, task_id: str, path: str) -> bytes:
        with open(self._resolve_path(task_id, path), "rb") as f:
            return f.read()

    def write(self, task_id: str, path: str, data: bytes) -> None:
        file_path = self._resolve_path(task_id, path)
        with open(file_path, "wb") as f:
            f.write(data)

    def delete(
        self, task_id: str, path: str, directory: bool = False, recursive: bool = False
    ) -> None:
        path = self.base_path / task_id / path
        resolved_path = self._resolve_path(task_id, path)
        if directory:
            if recursive:
                os.rmdir(resolved_path)
            else:
                os.removedirs(resolved_path)
        else:
            os.remove(resolved_path)

    def exists(self, task_id: str, path: str) -> bool:
        path = self.base_path / task_id / path
        return self._resolve_path(task_id, path).exists()

    def list(self, task_id: str, path: str) -> typing.List[str]:
        path = self.base_path / task_id / path
        base = self._resolve_path(task_id, path)
        if not base.exists() or not base.is_dir():
            return []
        return [str(p.relative_to(self.base_path / task_id)) for p in base.iterdir()]

```

# `autogpts/forge/forge/sdk/workspace_test.py`

这段代码的作用是测试本地工作区（workspace）中的一个名为"workspace.py"的类。首先，它导入了名为"os"的模块，这是一个Python标准库，用于操作系统相关操作。接下来，它从名为"workspace.py"的类中导入了名为"LocalWorkspace"的类。这个类可能是定义在另一个名为"workspace.py"的文件中，它可能是用来管理测试任务和数据等。

接下来，它定义了一些常量，包括：

- `TEST_BASE_PATH`：测试工作区的基目录。
- `TEST_FILE_CONTENT`：一个包含"Hello World"字符串的 bytes 对象。
- `TEST_TASK_ID`：一个字符串，用于标识一个测试任务。

最后，它通过 `pytest` 库的 `setup` 和 `teardown` 函数，实现了测试工作的安装和删除。这些函数通常在测试过程中使用，以确保代码的整洁和易于维护。


```py
import os

import pytest

# Assuming the classes are defined in a file named workspace.py
from .workspace import LocalWorkspace

# Constants
TEST_BASE_PATH = "/tmp/test_workspace"
TEST_FILE_CONTENT = b"Hello World"
TEST_TASK_ID = "1234"


# Setup and Teardown for LocalWorkspace


```

这段代码定义了一个名为 `setup_local_workspace` 的测试 fixture，它会在测试执行之前创建一个名为 `TEST_BASE_PATH` 的目录，并定义了三个测试函数：`test_local_read_write_delete_exists`，用于测试本地文件的操作。

具体来说，这段代码的作用如下：

1. 创建一个名为 `TEST_BASE_PATH` 的目录，如果该目录不存在，则会自动创建。
2. 通过 `yield` 语句让测试函数使用这个已经创建好的目录。
3. 在测试函数内部，使用 `os.system` 函数对创建的目录进行清理，保证了测试函数每次运行后，都会清除之前的创建的目录。
4. 通过 `LocalWorkspace` 类对目录中的文件进行读取、写入和删除操作。其中，`write` 方法将文件内容写入到指定的任务文件中，`exists` 方法判断文件是否存在，`read` 方法返回文件内容，`delete` 方法删除指定的任务文件以及目录中的所有文件。


```py
@pytest.fixture
def setup_local_workspace():
    os.makedirs(TEST_BASE_PATH, exist_ok=True)
    yield
    os.system(f"rm -rf {TEST_BASE_PATH}")  # Cleanup after tests


def test_local_read_write_delete_exists(setup_local_workspace):
    workspace = LocalWorkspace(TEST_BASE_PATH)

    # Write
    workspace.write(TEST_TASK_ID, "test_file.txt", TEST_FILE_CONTENT)

    # Exists
    assert workspace.exists(TEST_TASK_ID, "test_file.txt")

    # Read
    assert workspace.read(TEST_TASK_ID, "test_file.txt") == TEST_FILE_CONTENT

    # Delete
    workspace.delete(TEST_TASK_ID, "test_file.txt")
    assert not workspace.exists(TEST_TASK_ID, "test_file.txt")


```

这段代码是一个测试用例，它的作用是测试 `LocalWorkspace` 类是否可以成功创建并写入一个测试任务的目标文件。

具体来说，代码首先创建一个名为 `LocalWorkspace` 的类，这个类可能是一个测试框架的本地 workspace，用于管理测试任务和其相关的文件。然后，它创建一个名为 `TEST_BASE_PATH` 的属性，用于存储测试任务的根目录。

接着，代码使用 `write` 方法将两个名为 `test1.txt` 和 `test2.txt` 的文件写入到测试任务的根目录下，这两个文件的内容相同。然后，它使用 `list` 方法获取根目录下的所有文件，并使用 `assert` 方法检查返回的文件列表是否包含 `test1.txt` 和 `test2.txt`。如果返回的文件列表正确，那么测试用例就可以通过，否则就会失败。


```py
def test_local_list(setup_local_workspace):
    workspace = LocalWorkspace(TEST_BASE_PATH)
    workspace.write(TEST_TASK_ID, "test1.txt", TEST_FILE_CONTENT)
    workspace.write(TEST_TASK_ID, "test2.txt", TEST_FILE_CONTENT)

    files = workspace.list(TEST_TASK_ID, ".")
    assert set(files) == {"test1.txt", "test2.txt"}

```

# `autogpts/forge/forge/sdk/__init__.py`

这段代码是一个Forge SDK，包含了Forge的核心协议和几个类，用于实现机器人代理的训练和评估。

具体来说，这段代码定义了以下类：

- Agent类：这是机器人的基本类，包含了与外界的交互操作以及一些通用的方法。
- AgentDB类：这是数据库的接口类，用于和数据库进行交互。
- ForgeLogger类：这是日志的接口类，用于记录机器人在训练和测试过程中的一些信息。
- chat_completion_request和create_embedding_request分别是与聊天相关的接口，用于实现聊天对话的上下文和词汇。
- transcribe_audio接口用于实现将音频转录为文本的功能。
- PromptEngine类是用于生成 prompt 的接口类，Prompt 是一种用于生成人类对话的交互式接口。
- Article类定义了机器人任务的文章，包含任务的相关信息，如任务的目标，任务的时间等。
- ArticleUpload类是用于将文章上传到数据库的接口类。
- Pagination类是用于分页统计数据的接口类，用于从服务器上获取分页结果。
- Status类定义了机器人的状态，包括机器人的运行状态和任务的状态等。
- Step类定义了机器人的步骤，包含任务执行的结果等信息。
- StepOutput类是用于返回 Step 执行结果的接口类。
- StepRequestBody类是用于描述 Step 请求的接口类，其中包含了 Step 的目标和执行结果等信息。
- Task类定义了机器人的任务，包含任务的目标、任务的时间等属性。
- TaskArtifactsListResponse类是用于返回任务艺术品的列表的接口类，其中包含了任务完成后的奖品列表。
- TaskListResponse类是用于返回任务列表的接口类，其中包含了机器人的任务列表。
- TaskRequestBody类是用于描述任务需求的接口类，其中包含了任务的目标、任务的时间等属性。
- TaskStepsListResponse类是用于返回任务步骤的接口类，其中包含了任务完成后的步骤列表。


```py
"""
The Forge SDK. This is the core of the Forge. It contains the agent protocol, which is the
core of the Forge.
"""
from .agent import Agent
from .db import AgentDB, Base
from .forge_log import ForgeLogger
from .llm import chat_completion_request, create_embedding_request, transcribe_audio
from .prompting import PromptEngine
from .schema import (
    Artifact,
    ArtifactUpload,
    Pagination,
    Status,
    Step,
    StepOutput,
    StepRequestBody,
    Task,
    TaskArtifactsListResponse,
    TaskListResponse,
    TaskRequestBody,
    TaskStepsListResponse,
)
```

这段代码定义了三个类：LocalWorkspace 和 Workspace，以及一个名为ChromaMemStore的类，它们都是与内存管理相关的类。

LocalWorkspace 和 Workspace 应该是从 workspace_sdk 库中导入的，但具体哪个类被使用了，我无法确定。这两个类的作用我没有找到相关信息，因此无法提供更多帮助。

ChromaMemStore 是一个用来存储 Chroma 引擎中数据（如艺术品、音乐等）的内存管理类。它可能是为了方便在 memory_row 和 memory_stub 两个内存区域之间同步数据而设计的。

MemStore 是一个继承自 memory_row 的类，它可能用于在 LocalWorkspace 和 ChromaMemStore 之间同步数据。但是，具体它的作用我没有找到相关信息，因此无法提供更多帮助。


```py
from .workspace import LocalWorkspace, Workspace
from .errors import *
from .memory.chroma_memstore import ChromaMemStore
from .memory.memstore import MemStore
```

# `autogpts/forge/forge/sdk/abilities/finish.py`

这段代码是一个Python脚本，它导入了ForgeLogger和registry库，并创建了一个名为"finish"的能力。该能力的作用是当用户完成了所有目标并且无法继续完成任务时，向用户显示一个提示信息，或者在有无法逾越的问题时使用。

具体来说，这段代码首先导入了ForgeLogger库，并创建了一个名为"finish"的能力。在这个能力中，定义了一个名为"reason"的参数，它是一个字符串，用于向用户描述如何完成任务。当这个能力被调用时，它会根据所提供的“reason”参数生成一个提示信息，或者在有无法逾越的问题时显示该提示信息。

此外，这段代码还定义了一个output_type参数，将其设置为None，意味着该能力不会输出任何信息。


```py
from ..forge_log import ForgeLogger
from .registry import ability

logger = ForgeLogger(__name__)


@ability(
    name="finish",
    description="Use this to shut down once you have accomplished all of your goals,"
    " or when there are insurmountable problems that make it impossible"
    " for you to finish your task.",
    parameters=[
        {
            "name": "reason",
            "description": "A summary to the user of how the goals were accomplished",
            "type": "string",
            "required": True,
        }
    ],
    output_type="None",
)
```

这段代码定义了一个名为 `finish` 的异步函数，它接受三个参数：`agent`、`task_id` 和 `reason`。

当这个函数被调用时，它会执行以下操作：

1. 它会在日志中记录下传入的 `reason` 参数，并添加一个名为 "Shutting down..." 的额外信息。
2. 它会输出一个字符串，其中包含一个摘要，告诉用户如何完成任务以及如何实现目标。
3. 它会返回一个字符串，表示任务完成的情况。

此外，该函数还包含一个参数 `reason`，该参数在函数内部被传递。


```py
async def finish(
    agent,
    task_id: str,
    reason: str,
) -> str:
    """
    A function that takes in a string and exits the program

    Parameters:
        reason (str): A summary to the user of how the goals were accomplished.
    Returns:
        A result string from create chat completion. A list of suggestions to
            improve the code.
    """
    logger.info(reason, extra={"title": "Shutting down...\n"})
    return reason

```

# `autogpts/forge/forge/sdk/abilities/registry.py`

这段代码使用了多个 Python 标准库中的函数和类，包括 glob、importlib、inspect、os 和typing。它还引入了一个自定义的 AbilityParameter 类，该类使用 pydantic 库进行类型检查。

具体来说，这段代码的作用是定义了一个名为 "AbilityParameter" 的类，用于表示一个参数，该参数属于一个名为 "Ability" 的模型类。这个模型类定义了一个包含 name、description 和 type 字段的属性，其中 name 和 description 是不可变的，而 type 和 required 则是可变的。

此外，还定义了一个名为 "AbilityMutator" 的类，用于实现一个 ability 的 mutator，该 mutator 可以接受一个参数，该参数也属于一个名为 "AbilityParameter" 的类。在 "AbilityMutator" 中，使用了 importlib 的 functions，将 "AbilityParameter" 的实例赋值给 "ability" 对象的一个属性，从而实现了 ability 的能力参数。


```py
import glob
import importlib
import inspect
import os
from typing import Any, Callable, List

import pydantic


class AbilityParameter(pydantic.BaseModel):
    """
    This class represents a parameter for an ability.

    Attributes:
        name (str): The name of the parameter.
        description (str): A brief description of what the parameter does.
        type (str): The type of the parameter.
        required (bool): A flag indicating whether the parameter is required or optional.
    """

    name: str
    description: str
    type: str
    required: bool


```

这段代码定义了一个名为 Ability 的类，属于 pydantic.BaseModel 类。这个类表示系统中的一个能力，包含了名称、描述、实现方法以及参数列表和返回类型等信息。

这个类中包含了一个特殊的方法 `__call__`，允许将类实例作为函数进行调用。在 `__call__` 方法中，通过 `*args` 和 `**kwds` 获取输入参数，并调用类的 `method` 方法，实现输入参数的传递给方法的机制。

另外，这个类还包含一个名为 `__str__` 的方法，用于将对象转换为字符串并输出。这个方法在对象被创建时自动调用，将对象作为参数传入，返回对象字符串形象表示。

整个类的实例代表了一个具有特定能力和参数列表的 Pydantic Model，可以在模型定义中使用 `pydantic.Model` 或 `@pydantic.model` 语法来创建。


```py
class Ability(pydantic.BaseModel):
    """
    This class represents an ability in the system.

    Attributes:
        name (str): The name of the ability.
        description (str): A brief description of what the ability does.
        method (Callable): The method that implements the ability.
        parameters (List[AbilityParameter]): A list of parameters that the ability requires.
        output_type (str): The type of the output that the ability returns.
    """

    name: str
    description: str
    method: Callable
    parameters: List[AbilityParameter]
    output_type: str
    category: str | None = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        This method allows the class instance to be called as a function.

        Args:
            *args: Variable length argument list.
            **kwds: Arbitrary keyword arguments.

        Returns:
            Any: The result of the method call.
        """
        return self.method(*args, **kwds)

    def __str__(self) -> str:
        """
        This method returns a string representation of the class instance.

        Returns:
            str: A string representation of the class instance.
        """
        func_summary = f"{self.name}("
        for param in self.parameters:
            func_summary += f"{param.name}: {param.type}, "
        func_summary = func_summary[:-2] + ")"
        func_summary += f" -> {self.output_type}. Usage: {self.description},"
        return func_summary


```

这段代码定义了一个名为 `ability` 的装饰器函数，它接受四个参数：`name`、`description`、`parameters` 和 `output_type`。这个装饰器函数的作用是在函数定义中添加 `Ability` 类的实例，这个实例可以通过调用 `ability` 函数来获取。

具体来说，装饰器函数首先通过 `inspect.signature(func).parameters` 获取函数定义中的参数列表，然后使用 `set()` 函数将所有参数按照名称顺序转换成字符串，并将其添加到 `param_names` 集合中。接着，它将 `param_names` 集合与 `func_param_names` 集合进行比较，如果它们不相等，就会抛出一个 `ValueError` 异常。

如果参数列表和函数定义中的参数列表不一致，装饰器函数会将错误信息打印出来，并返回原始函数。如果它们相等，装饰器函数会创建一个新的 `Ability` 类的实例，并将它设置为参数 `parameters` 的 `Ability` 类的实例。最后，装饰器函数返回 modified 函数，这个函数仍然可以被调用，但是它的名称已经被修改为 `ability`，而参数列表已经被转换成了 `param_names` 集合中列出的参数名称。


```py
def ability(
    name: str, description: str, parameters: List[AbilityParameter], output_type: str
):
    def decorator(func):
        func_params = inspect.signature(func).parameters
        param_names = set(
            [AbilityParameter.parse_obj(param).name for param in parameters]
        )
        param_names.add("agent")
        param_names.add("task_id")
        func_param_names = set(func_params.keys())
        if param_names != func_param_names:
            raise ValueError(
                f"Mismatch in parameter names. Ability Annotation includes {param_names}, but function acatually takes {func_param_names} in function {func.__name__} signature"
            )
        func.ability = Ability(
            name=name,
            description=description,
            parameters=parameters,
            method=func,
            output_type=output_type,
        )
        return func

    return decorator


```

This is a class that appears to be an implementation of the招募起始化技能的行为。它的主要想法是允许使用特殊技能，如购买商品或召唤恶魔，并在行为发生时将其使用。

具体来说，该类中包含一个`Abilities`属性，一个`AbilitiesDescription`属性，一个`List<Ability>`属性，一个`List<str>`属性，以及一些辅助方法，如`run_ability`方法。

`run_ability`方法的具体实现可以分为两步。第一步是获取指定的技能，第二步是使用技能处理其参数和 keyword arguments，并返回结果。

第二步，`run_ability`方法通过对技能进行调用，获取其返回的结果。这个结果可以是任何类型，但它必须是可读的。如果技能的参数或 keyword arguments不正确，则引发一个异常并返回。

如果返回值是未定义的，则默认地返回一个空字符串。

该类的方法还提供了一个`list_abilities`方法，用于打印或列出技能。

``list_abilities`方法通过迭代`Abilities`对象来获取技能，并打印或列出它们。

``list_abilities_for_prompt`方法尝试从`Abilities`对象中获取一些技能，并打印它们。

``abilities_description`方法根据`Abilities`对象中的技能，打印或描述它们。

这些方法的具体实现没有给出，但它们的名称和实现看起来表明它们可能是与技能相关的某种描述性输出。


```py
class AbilityRegister:
    def __init__(self, agent) -> None:
        self.abilities = {}
        self.register_abilities()
        self.agent = agent

    def register_abilities(self) -> None:
        for ability_path in glob.glob(
            os.path.join(os.path.dirname(__file__), "**/*.py"), recursive=True
        ):
            if not os.path.basename(ability_path) in [
                "__init__.py",
                "registry.py",
            ]:
                ability = os.path.relpath(
                    ability_path, os.path.dirname(__file__)
                ).replace("/", ".")
                try:
                    module = importlib.import_module(
                        f".{ability[:-3]}", package="forge.sdk.abilities"
                    )
                    for attr in dir(module):
                        func = getattr(module, attr)
                        if hasattr(func, "ability"):
                            ab = func.ability

                            ab.category = (
                                ability.split(".")[0].lower().replace("_", " ")
                                if len(ability.split(".")) > 1
                                else "general"
                            )
                            self.abilities[func.ability.name] = func.ability
                except Exception as e:
                    print(f"Error occurred while registering abilities: {str(e)}")

    def list_abilities(self) -> List[Ability]:
        return self.abilities

    def list_abilities_for_prompt(self) -> List[str]:
        return [str(ability) for ability in self.abilities.values()]

    def abilities_description(self) -> str:
        abilities_by_category = {}
        for ability in self.abilities.values():
            if ability.category not in abilities_by_category:
                abilities_by_category[ability.category] = []
            abilities_by_category[ability.category].append(str(ability))

        abilities_description = ""
        for category, abilities in abilities_by_category.items():
            if abilities_description != "":
                abilities_description += "\n"
            abilities_description += f"{category}:"
            for ability in abilities:
                abilities_description += f"  {ability}"

        return abilities_description

    async def run_ability(
        self, task_id: str, ability_name: str, *args: Any, **kwds: Any
    ) -> Any:
        """
        This method runs a specified ability with the provided arguments and keyword arguments.

        The agent is passed as the first argument to the ability. This allows the ability to access and manipulate
        the agent's state as needed.

        Args:
            task_id (str): The ID of the task that the ability is being run for.
            ability_name (str): The name of the ability to run.
            *args: Variable length argument list.
            **kwds: Arbitrary keyword arguments.

        Returns:
            Any: The result of the ability execution.

        Raises:
            Exception: If there is an error in running the ability.
        """
        try:
            ability = self.abilities[ability_name]
            return await ability(self.agent, task_id, *args, **kwds)
        except Exception:
            raise


```

这段代码是一个Python脚本，主要作用是注册一个AbilityRegister对象，并在其上运行一个名为"abc"的测试能力。以下是该代码的功能解释：

1. 导入sys模块的一个AbilityRegister类型，以便在脚本中使用AbilityRegister的函数。
2. 将脚本所在的目录添加到Python的sys.path，以便系统在运行脚本时可以正确查找和安装所需的Python包。
3. 创建一个名为AbilityRegister的类，该类实现了AbilityRegister的接口，并在其中初始化了一个与题目中指定的agent参数无关的能力注册器。
4. 通过register.abilities_description()函数打印出注册的能力的描述信息，其中包括该能力支持的所有功能和参数。
5. 通过register.run_ability()函数运行一个名为"abc"的测试能力，并将该能力的输出打印出来。


```py
if __name__ == "__main__":
    import sys

    sys.path.append("/Users/swifty/dev/forge/forge")
    register = AbilityRegister(agent=None)
    print(register.abilities_description())
    print(register.run_ability("abc", "list_files", "/Users/swifty/dev/forge/forge"))

```

# `autogpts/forge/forge/sdk/abilities/__init__.py`

我需要更具体的上下文来回答你的问题。可以请你提供一下代码或提供一些关于代码的上下文信息，这样我才能够更好地解释它的作用。


```py

```

# `autogpts/forge/forge/sdk/abilities/file_system/files.py`

这段代码使用了Python中的typing模块，它是一个用于提供类型声明的库。在这段代码中，我们定义了一个名为"list_files"的函数，它接受一个字符串参数"path"，描述为"List files in a directory"。这个函数使用了ability库，一个Python库，用于帮助函数重用和扩展Python功能。ability库是一个经过精心设计的Python库，它提供了许多有用的功能，包括文件和目录操作，授权管理，和更多的功能。

在这段代码中，我们使用了ability库的add_function()方法来定义一个函数行为。这个函数行为的参数是一个字典，其中包含两个键：name和description。这些键告诉ability库如何描述这个函数，以便使用者可以理解这个函数可以做什么。在这里，name键是"list_files"，description键是这个函数的描述。parameters键包含一个列表，其中包含一个{name: description}的元素。这个列表告诉ability库这个函数需要哪些参数，以及这些参数有什么要求。在这里，我们有一个{name: description}的元素，其中name是"path"，description是这个函数的描述。这个函数需要一个字符串参数，这个参数用于指定要列出文件的目录。

最后，我们定义了一个内部函数ability.list\_files()，这个函数实现了我们刚刚定义的函数行为。在这个函数中，我们使用了ability库的resolve_path()方法来获取用户提供的目录路径，并使用List库的sort()方法来对列表进行排序。排序后的结果被返回，这样我们可以使用List库的len()方法来获取排序后的列表的长度。


```py
from typing import List

from ..registry import ability

@ability(
    name="list_files",
    description="List files in a directory",
    parameters=[
        {
            "name": "path",
            "description": "Path to the directory",
            "type": "string",
            "required": True,
        }
    ],
    output_type="list[str]",
)
```

这段代码定义了一个名为 `list_files` 的异步函数，它接受一个 `agent` 对象、一个任务 ID 和一个文件夹路径参数。它的作用是返回 agent 对象的工作区目录中所有文件的列表。

函数的实现非常简单，直接调用了 `agent.workspace.list` 方法，传入参数 `task_id` 和 `path`，然后返回返回结果。这个函数可以在异步上下文中使用，比如使用 `asyncio` 库的 `list_files` 函数可以写成：
```pybash
import asyncio

async def list_files(agent, task_id: str, path: str) -> List[str]:
   """
   List files in a workspace directory
   """
   return agent.workspace.list(task_id=task_id, path=str(path))

async def main(agent):
   """
   主函数
   """
   files = await list_files(agent, "task_1", "path/to/directory")
   for file in files:
       print(file)

asyncio.run(main(agent))
```
这段代码会在 agent 对象上调用 `list_files` 函数，并返回结果。`main` 函数会阻塞等待 `list_files` 函数返回结果，然后遍历所有文件并输出它们。


```py
async def list_files(agent, task_id: str, path: str) -> List[str]:
    """
    List files in a workspace directory
    """
    return agent.workspace.list(task_id=task_id, path=str(path))


@ability(
    name="write_file",
    description="Write data to a file",
    parameters=[
        {
            "name": "file_path",
            "description": "Path to the file",
            "type": "string",
            "required": True,
        },
        {
            "name": "data",
            "description": "Data to write to the file",
            "type": "bytes",
            "required": True,
        },
    ],
    output_type="None",
)
```

这段代码定义了一个名为 `write_file` 的异步函数，它接受四个参数：`agent`、`task_id`、`file_path` 和 `data`。

函数的作用是将 `data` 中的数据写入到指定的文件路径下。函数首先检查 `data` 是否为字符串类型，如果是，则将其转换为字节序列。然后，使用 `agent.workspace` 方法将数据写入到 `task_id` 为 `task_id`、文件名为 `file_path` 的文档中。

函数还使用 `agent.db` 方法创建一个新的文档作为 artifacts，其中 `file_name` 参数指定文件名，`relative_path` 参数指定文件路径。这个新的文档使用了 `agent_created` 参数来指示它是机器人自动创建的，这个参数在文档中使用了 `relative_path` 变量来获取文件的相对路径。

最后，函数返回一个 `asyncio` 异步函数，这个函数使用了 `await` 和 `await agent.db.create_artifact` 方法来等待数据库写入和创建文档。


```py
async def write_file(agent, task_id: str, file_path: str, data: bytes):
    """
    Write data to a file
    """
    if isinstance(data, str):
        data = data.encode()

    agent.workspace.write(task_id=task_id, path=file_path, data=data)
    return await agent.db.create_artifact(
        task_id=task_id,
        file_name=file_path.split("/")[-1],
        relative_path=file_path,
        agent_created=True,
    )


```

这段代码是一个异步函数，名为 `read_file`，具有以下参数：

* `file_path`：文件路径，参数类型为 `string`，要求必须要有，类型为 `string`。
* 无参数。

函数的作用是：

* 读取数据并返回，数据类型为字节（`bytes`）。
* `agent` 是异步执行的上下文，类型为 `Any`。
* `task_id` 是任务ID，类型为 `str`。
* `file_path` 是文件路径，参数类型为 `string`，要求必须要有，类型为 `string`。


```py
@ability(
    name="read_file",
    description="Read data from a file",
    parameters=[
        {
            "name": "file_path",
            "description": "Path to the file",
            "type": "string",
            "required": True,
        },
    ],
    output_type="bytes",
)
async def read_file(agent, task_id: str, file_path: str) -> bytes:
    """
    Read data from a file
    """
    return agent.workspace.read(task_id=task_id, path=file_path)

```

# `autogpts/forge/forge/sdk/abilities/web/web_search.py`

这段代码是一个Python程序，它使用未来的函数特性从第三方库中获取了DDGS（DuckDuckGo搜索引擎）的API密钥。然后，它使用了一个自定义的搜索注册表，允许用户输入一个搜索查询，并使用该注册表执行网络搜索。

具体来说，这段代码的作用是执行以下操作：

1. 导入`__future__`，以便使用未来的函数特性。
2. 导入`json`，`time`和`itertools`库。
3. 从`DDGS`库中导入`DDGS`类。
4. 从自定义的搜索注册表中获取搜索查询。
5. 使用`islice`库遍历搜索结果。
6. 调用`DDGS.search`方法，传递搜索查询和搜索结果，并使用`max_attempts`参数限制最大尝试次数。
7. 将搜索结果存储在Python list中，并将程序设置为运行状态。


```py

from __future__ import annotations

import json
import time
from itertools import islice

from duckduckgo_search import DDGS

from ..registry import ability

DUCKDUCKGO_MAX_ATTEMPTS = 3


@ability(
    name="web_search",
    description="Searches the web",
    parameters=[
        {
            "name": "query",
            "description": "The search query",
            "type": "string",
            "required": True,
        }
    ],
    output_type="list[str]",
)
```

这段代码是一个 Python 函数，名为 `web_search`，用于执行 Google 搜索并返回搜索结果。函数需要一个参数 `agent`，一个参数 `task_id` 和一个参数 `query`。

函数内部使用了一个 while 循环来重复执行搜索操作，直到搜索结果完全加载完成了。在每次搜索期间，函数先检查搜索查询是否为空，如果是，则返回一个空字符串。否则，函数调用一个名为 `DDGS()` 的函数执行搜索，并将搜索结果存储在 `search_results` 列表中。

如果 `search_results` 列表中有搜索结果，函数就会退出 while 循环并返回它。否则，函数将等待 1 秒钟，并手动增加 `attempts` 计数器的值，以便在每次尝试搜索时都重新尝试执行相同的操作。

最后，函数将 `search_results` 列表转换为 JSON 格式并返回，同时使用一个名为 `safe_google_results()` 的函数来确保在将搜索结果返回给用户时不会包含敏感信息。


```py
async def web_search(agent, task_id: str, query: str) -> str:
    """Return the results of a Google search

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
    search_results = []
    attempts = 0
    num_results = 8

    while attempts < DUCKDUCKGO_MAX_ATTEMPTS:
        if not query:
            return json.dumps(search_results)

        results = DDGS().text(query)
        search_results = list(islice(results, num_results))

        if search_results:
            break

        time.sleep(1)
        attempts += 1

    results = json.dumps(search_results, ensure_ascii=False, indent=4)
    return safe_google_results(results)


```



该函数`safe_google_results`接受两个参数，一个是字符串类型(可以是空)，另一个是列表类型。它返回一个字符串，表示谷歌搜索的结果。

如果结果列表是空，函数会将列表中的所有字符串转义并解码，然后将它们连接成一个字符串，用 UTF-8 编码。这将确保在将结果返回给调用者时，即使结果列表为空，也将返回一个空字符串。

如果结果列表是一个字符串，函数会将该字符串直接用 UTF-8 编码。

对于搜索结果，函数会将搜索结果中的所有字符串转义并解码，这将确保在将结果返回给调用者时，即使结果字符串存在Unicode字符，也将正确处理。此外，由于搜索结果可能包含其他数据(如引号、URL等)，函数还从搜索结果中提取出这些数据，并将其转义和解码，以确保在将结果返回给调用者时，可以正确处理这些数据。


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

# `autogpts/forge/forge/sdk/abilities/web/web_selenium.py`

该代码定义了一个用于浏览网站的命令类，其中包括两个静态方法，分别是`web_browse`和`browse_url`。

`web_browse`方法使用了`bs4`库从网站中获取页面内容，并通过调用`logging.debug()`函数将内容输出到控制台。

`browse_url`方法使用了`re`库从用户输入的 URL 中提取网页内容，并使用`logging.debug()`函数将内容输出到控制台。

此外，该代码还导入了一个名为`WebDriverException`的异常类，它是从`selenium.common.exceptions`包中导入的。


```py
"""Commands for browsing a website"""

from __future__ import annotations

COMMAND_CATEGORY = "web_browse"
COMMAND_CATEGORY_TITLE = "Web Browsing"

import logging
import re
from pathlib import Path
from sys import platform
from typing import TYPE_CHECKING, Optional, Type, List, Tuple

from bs4 import BeautifulSoup
from selenium.common.exceptions import WebDriverException
```

这段代码的作用是设置 Chrome 和 Edge WebDriver 的选项和提供者，包括允许使用的浏览器、选项卡、网络设置、自动化组件、服务、驱动程序等。它们允许在测试中使用这些工具来运行 Selenium WebDriver。

具体来说，这段代码包括以下几个部分：

1. 从selenium.webdriver.chrome.options模块中导入ChromeOptions变量。这个模块定义了用于 Chrome浏览器的各种选项，包括允许使用的CSS、JavaScript、API等。
2. 从selenium.webdriver.chrome.service模块中导入ChromeDriverService变量。这个模块是一个用于管理Chrome浏览器的工具，它可以启动、停止、卸载Chrome浏览器。
3. 从selenium.webdriver.chrome.webdriver模块中导入ChromeDriver变量。这个模块定义了Chrome浏览器的WebDriver类，它是使浏览器运行selenium WebDriver 的主要部分。
4. 从selenium.webdriver.common.by模块中导入By变量。这个模块定义了在Chrome和Edge WebDriver中使用的各种By标签，用于定位元素。
5. 从selenium.webdriver.common.options模块中导入BrowserOptions变量。这个模块定义了在Chrome和Edge浏览器中使用的各种选项卡、网络设置等。
6. 从selenium.webdriver.edge.options模块中导入EdgeOptions变量。这个模块定义了在Edge浏览器中使用的各种选项卡、网络设置等。
7. 从selenium.webdriver.edge.service模块中导入EdgeDriverService变量。这个模块定义了在Edge浏览器中使用的各种服务，包括启动、停止、卸载EdgeDriver等。
8. 从selenium.webdriver.edge.webdriver模块中导入WebDriver变量。这个模块定义了在Edge浏览器中运行的WebDriver类。
9. 从selenium.webdriver.firefox.options模块中导入Options变量。这个模块定义了在Firefox浏览器中使用的各种选项卡、网络设置等。
10. 从selenium.webdriver.firefox.service模块中导入GeckoDriverService变量。这个模块定义了在Firefox浏览器中使用的各种服务，包括启动、停止、卸载GeckoDriver等。
11. 从selenium.webdriver.firefox.webdriver模块中导入WebDriver变量。这个模块定义了在Firefox浏览器中运行的WebDriver类。
12. 从selenium.webdriver.safari.options模块中导入Options变量。这个模块定义了在Safari浏览器中使用的各种选项卡、网络设置等。
13. 从selenium.webdriver.safari.webdriver模块中导入WebDriver变量。这个模块定义了在Safari浏览器中运行的WebDriver类。
14. 从selenium.webdriver.support.expected_conditions模块中导入EC变量。这个模块定义了在Chrome和Firefox浏览器中使用的各种预期的条件，例如窗口、按钮、链接、文本框等。


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

这段代码的作用是：

1. 导入所需的库：selenium.webdriver.support.wait,WebDriverWait,from selenium.webdriver.support import time,from selenium.webdriver.support.models import Motion,from selenium.webdriver.common.by import By,from selenium.webdriver.的安全性 import checkToolb完整性的函数，from selenium.webdriver.security import allowSecurity,from selenium.webdriver.exceptions import PermissionDenied,ActionNotFound，十一、使用 Selenium 库等待网页加载完成，并获取页面元素。
2. 导入另一个库：webdriver_manager.chrome,ChromeDriverManager,from selenium.webdriver.chrome import ChromeDriverManager，从指定版本号的主机上安装 ChromeDriver。
3. 导入：webdriver_manager.firefox,GeckoDriverManager，从指定版本号的主机上安装 Firefox 的 GeckoDriverManager。
4. 导入：webdriver_manager.microsoft,EdgeChromiumDriverManager，从指定版本号的主机上安装 Edge 的 EdgeChromiumDriverManager。
5. 定义一个名为 ability 的类，继承自能力类，并覆盖重写其中的三个方法：从而访问浏览器的能力，并注册相关的能力。
6. 从 --registry 参数中加载一个函数，并将其名为 ability。
7. 通过 for 循环，使用 which 函数获取待安装的Chrome、Firefox 和 Edge的 versions，并使用 those 版本回调能力类中的方法。
8. 通过 for 循环，获取所有的火狐和微知的长春 Chrome 版本。
9. 通过 for 循环，获取所有的微软 Edge 和寒春版本。
10. 通过重写第一个方法，实现等待 30 秒后，如果等待过程中没有发生错误，就继续执行下一步。
11. 通过检查浏览器是否可用，来防止出现 ActionNotFound 和 PermissionDenied 异常。
12. 通过将浏览器的安全性设置为允许访问 URL，来保证在等待过程中，可以访问被请求的 URL。

完整的作用是：

这段代码是一个异步编程的应用，用于自动执行浏览器加载网页的任务，并获取页面元素。通过使用 Selenium 库等待网页加载完成，并获取页面元素。另外，还通过引入不同的浏览器驱动程序，来使得代码能够对不同的浏览器进行支持。


```py
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager as EdgeDriverManager


from ..registry import ability
from forge.sdk.errors import *
import functools
import re
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

from requests.compat import urljoin


```

这段代码的作用是提取一个给定页面的所有链接，并返回它们的元组形式。

具体来说，它实现了以下两个函数：

- `extract_hyperlinks`：该函数接收一个BeautifulSoup对象和一个基础URL，并返回其中所有链接的元组。这个函数使用了Python的`requests`库来获取页面内容，并使用了`BeautifulSoup`库来解析HTML并提取链接。它通过使用`find_all`方法来查找所有的链接，然后遍历每个链接，提取出链接的文本和链接到的基础URL，并将它们组合成一个元组。最后，它返回了一个包含所有链接的列表。

- `urljoin`：该函数接收两个参数，一个是基础URL，另一个是一个或多个参数，并将它们连接起来形成一个新的URL。它实现了`urljoin.urljoin`函数，该函数可以方便地将多个URL参数连接起来形成一个新的URL。

总的来说，这段代码可以用于许多不同的应用，例如从给定的页面中提取所有链接，并将其存储到一个列表中，以便进行进一步的处理或分析。


```py
from bs4 import BeautifulSoup
from requests.compat import urljoin


def extract_hyperlinks(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    """Extract hyperlinks from a BeautifulSoup object

    Args:
        soup (BeautifulSoup): The BeautifulSoup object
        base_url (str): The base URL

    Returns:
        List[Tuple[str, str]]: The extracted hyperlinks
    """
    return [
        (link.text, urljoin(base_url, link["href"]))
        for link in soup.find_all("a", href=True)
    ]


```



该代码定义了两个函数，一个是`format_hyperlinks`，另一个是`validate_url`。

函数`format_hyperlinks`接收一个包含超链接的列表。该函数将每个超链接格式化为字符串，其中超链接包含`link_text`和`link_url`。函数返回一个包含格式化后的超链接的字符串列表。

函数`validate_url`接受一个函数作为参数，该函数需要一个`url`参数。该函数使用多种方式验证`url`的有效性，包括正则表达式验证、检查网络连接和本地文件访问检查。如果`url`不正确或未通过验证，函数将引发`ValueError`。

函数`validate_url`中使用了一个名为`is_valid_url`的函数，该函数接受一个`url`参数并返回一个布尔值，表示`url`是否有效。如果`url`无效，函数将引发`ValueError`。该函数将`validate_url`函数包装在一个内部，并使用该函数的`wrapper`函数来覆盖该内部函数。这个`wrapper`函数包含一个简单的比较，如果超链接有效，则将结果返回。如果超链接无效，则将引发`ValueError`并调用`is_valid_url`函数来检查`url`。

综合来看，这两个函数一起工作，使得函数调用时可以有一个带有超链接的参数，而函数会根据`url`参数的值来调用`validate_url`函数来验证`url`是否有效，或者根据`link_text`和`link_url`来格式化显示超链接。


```py
def format_hyperlinks(hyperlinks: list[tuple[str, str]]) -> list[str]:
    """Format hyperlinks to be displayed to the user

    Args:
        hyperlinks (List[Tuple[str, str]]): The hyperlinks to format

    Returns:
        List[str]: The formatted hyperlinks
    """
    return [f"{link_text} ({link_url})" for link_text, link_url in hyperlinks]



def validate_url(func: Callable[..., Any]) -> Any:
    """The method decorator validate_url is used to validate urls for any command that requires
    a url as an argument"""

    @functools.wraps(func)
    def wrapper(url: str, *args, **kwargs) -> Any:
        """Check if the URL is valid using a basic check, urllib check, and local file check

        Args:
            url (str): The URL to check

        Returns:
            the result of the wrapped function

        Raises:
            ValueError if the url fails any of the validation tests
        """
        # Most basic check if the URL is valid:
        if not re.match(r"^https?://", url):
            raise ValueError("Invalid URL format")
        if not is_valid_url(url):
            raise ValueError("Missing Scheme or Network location")
        # Restrict access to local files
        if check_local_file_access(url):
            raise ValueError("Access to local files is restricted")
        # Check URL length
        if len(url) > 2000:
            raise ValueError("URL is too long")

        return func(sanitize_url(url), *args, **kwargs)

    return wrapper


```

这段代码定义了一个名为 `is_valid_url` 的函数，用于检查给定的 URL 是否有效。函数接受一个字符串参数 `url`，返回一个布尔值，表示 URL 是否有效，其中有效的 URL 必须包含协议类型(如 HTTP、HTTPS、FTP 等)和域名。

函数首先使用 `urlparse` 函数解析 URL，获取协议类型和域名。然后，函数使用 Python 的所有非空类型中包含 `True` 的类型，即 `all()` 函数，确保 URL 中的协议类型和域名都为真。如果 `is_valid_url` 函数无法解析 URL，它将返回 `False`。


```py
def is_valid_url(url: str) -> bool:
    """Check if the URL is valid

    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


```

This function sanitizes a URL by removing query parameters and formatting the path and query components as a string. The function takes a URL string as input and returns the sanitized URL string.

This function can be used in the `requests` library to simplify the process of making requests to a URL. For example, you can use this function to sanitize the URL of a file before making an HTTP request to it:
```py
import requests

url = "file:///path/to/local/file.txt"

response = requests.get(url)

# The sanitized URL is stored in the response.text property
response.raise_for_status()
```
This will raise an exception if the file at the given URL does not exist.

This function can also be used to check if a URL is a local file. This can be done by checking if the URL starts with a local prefix. For example, this function can be used to check if an URL is local file if it starts with the string "file:///":
```py
from collections import cache

url = "file:///path/to/local/file.txt"

if "file:///" in url:
   # The URL is a local file
   pass
else:
   # The URL is not a local file
   pass
```
This function is useful for checking if a URL is local file before making an HTTP request to it.

This function can be used in the `urlparse` function from the `urllib` module to parse a URL into its components and then sanitize it.
```py
import urllib

url = "file:///path/to/local/file.txt"

parsed_url = urllib.parse(url)

# The sanitized URL is stored in the parsed_url.path property
print(parsed_url.path)
```
This will print the path of the file, which is the sanitized URL.


```py
def sanitize_url(url: str) -> str:
    """Sanitize the URL

    Args:
        url (str): The URL to sanitize

    Returns:
        str: The sanitized URL
    """
    parsed_url = urlparse(url)
    reconstructed_url = f"{parsed_url.path}{parsed_url.params}?{parsed_url.query}"
    return urljoin(url, reconstructed_url)


def check_local_file_access(url: str) -> bool:
    """Check if the URL is a local file

    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL is a local file, False otherwise
    """
    local_prefixes = [
        "file:///",
        "file://localhost/",
        "file://localhost",
        "http://localhost",
        "http://localhost/",
        "https://localhost",
        "https://localhost/",
        "http://2130706433",
        "http://2130706433/",
        "https://2130706433",
        "https://2130706433/",
        "http://127.0.0.1/",
        "http://127.0.0.1",
        "https://127.0.0.1/",
        "https://127.0.0.1",
        "https://0.0.0.0/",
        "https://0.0.0.0",
        "http://0.0.0.0/",
        "http://0.0.0.0",
        "http://0000",
        "http://0000/",
        "https://0000",
        "https://0000/",
    ]
    return any(url.startswith(prefix) for prefix in local_prefixes)




```

这段代码是一个Python程序，它实现了以下功能：

1. 获取一个名为`logger`的 logger 实例，这个 logger 将会输出 `__name__` 和 `__file__` 两个部分的名称。
2. 获取当前工作目录（FILE\_DIR）的父目录，并将其存储在名为 `FILE_DIR` 的变量中。
3. 定义一个名为 `TOKENS_TO_TRIGGER_SUMMARY` 的常量，值为 50。
4. 定义一个名为 `LINKS_TO_RETURN` 的常量，值为 20。
5. 定义一个名为 `BrowsingError` 的自定义异常类，它是 `CommandExecutionError` 的子类，这个异常类表示在尝试访问网页时发生了错误。
6. 在自定义异常类中，定义了两个参数 `url` 和 `question`，`url` 表示要访问的网页，`question` 表示要在网页上提出的问题。
7. 在 `BrowsingError` 的 `__str__` 方法中，打印出 `__file__` 和 `__name__` 两个部分的名称，以及自定义异常类的一个警告信息，警告信息通常是异常类的一个辅助信息，用于在没有给出完整错误信息时方便调试。
8. 在程序的其他部分中，将 `FILE_DIR` 和 `TOKENS_TO_TRIGGER_SUMMARY` 作为参数传递给 `BrowsingError` 的 `__init__` 方法，并将 `LINKS_TO_RETURN` 作为参数传递给 `BrowsingError` 的 `raise_` 方法。


```py
logger = logging.getLogger(__name__)

FILE_DIR = Path(__file__).parent.parent
TOKENS_TO_TRIGGER_SUMMARY = 50
LINKS_TO_RETURN = 20


class BrowsingError(CommandExecutionError):
    """An error occurred while trying to browse the page"""


@ability(
    name="read_webpage",
    description="Read a webpage, and extract specific information from it if a question is specified. If you are looking to extract specific information from the webpage, you should specify a question.",
    parameters=[
        {
            "name": "url",
            "description": "The URL to visit",
            "type": "string",
            "required": True,
        },
                {
            "name": "question",
            "description": "A question that you want to answer using the content of the webpage.",
            "type": "string",
            "required": False,
        }
    ],
    output_type="string",
)
```

这段代码是一个名为 `read_webpage` 的函数，它使用 Selenium 库浏览一个网站，并获取从网页中提取出来的文本和链接。

函数有两个参数，一个是表面的 `url` 参数，另一个是表面上的 `question` 参数，它的默认值是空字符串。

函数返回一个元组，包含网页中的文本和提取的链接。如果函数成功执行，它将返回一个字符串和一个列表，其中第一个元素是文本，第二个元素是链接。

函数内部包括以下步骤：

1. 尝试创建一个新的 WebDriver 实例并将其启动。
2. 打开一个网页，并使用 `scrape_text_with_selenium` 函数从 WebDriver 中捕获文本。
3. 使用 `scrape_links_with_selenium` 函数从 WebDriver 中捕获链接。
4. 如果 WebDriver 正常工作，从第一个链接开始捕获链接，并将链接添加到 `links` 列表中。
5. 如果 WebDriver 出现异常，使用 `raise` 语句将其记录下来。如果 `BrowsingError` 异常被抛出，它将捕获到异常中并提供错误消息。
6. 关闭 WebDriver。

函数最终返回 `text` 和 `links` 两个元素，其中 `text` 是网页中的文本，`links` 是提取的链接。


```py
@validate_url
async def read_webpage(agent, task_id: str, url: str, question: str = "") -> Tuple(str, List[str]):
    """Browse a website and return the answer and links to the user

    Args:
        url (str): The url of the website to browse
        question (str): The question to answer using the content of the webpage

    Returns:
        str: The answer and links to the user and the webdriver
    """
    driver = None
    try:
        driver = open_page_in_browser(url)

        text = scrape_text_with_selenium(driver)
        links = scrape_links_with_selenium(driver, url)

        if not text:
            return f"Website did not contain any text.\n\nLinks: {links}"

        # Limit links to LINKS_TO_RETURN
        if len(links) > LINKS_TO_RETURN:
            links = links[:LINKS_TO_RETURN]
        return (text, links)

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

这段代码是一个Python函数，名为`scrape_text_with_selenium`，它使用Selenium库从浏览器窗口中提取文本。

具体来说，这个函数接受一个参数`driver`，它是一个`WebDriver`对象，表示要操作的浏览器窗口。函数内部通过调用`execute_script`方法获取网页的DOM内容，并使用`BeautifulSoup`库将DOM内容转换为XML解析器可以处理的格式。然后函数遍历XML文档中的所有`script`和`style`标签，并提取出每个标签的`src`属性的值，最终返回文本内容。

函数的实现大致如下：
```pyscss
def scrape_text_with_selenium(driver: WebDriver) -> str:
   text = driver.execute_script("return document.body.outerHTML;")
   soup = BeautifulSoup(text, "html.parser")
   for script in soup(["script", "style"]):
       script.extract()
   text = soup.get_text()
   lines = (line.strip() for line in text.splitlines())
   chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
   text = "\n".join(chunk for chunk in chunks if chunk)
   return text
```
这个函数使用Selenium库从浏览器窗口中提取文本，最终返回被提取出来的文本。它接受一个参数`driver`，表示要操作的浏览器窗口，返回值为`text`，即提取出来的文本。


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

这段代码的作用是使用 Selenium 库从指定网站中 scraping 链接，并返回网站中所有的链接。

具体来说，代码中包含了以下步骤：

1. 使用 `WebDriver` 类从浏览器窗口中获取页面源代码，并将其保存到 `page_source` 变量中。
2. 使用 `BeautifulSoup` 类将页面源代码解析为 HTML 文档，并将其保存到 `soup` 变量中。
3. 使用 `soup` 的 `find_all` 方法找到所有 `script` 和 `style` 标签。
4. 使用 `extract` 方法从每个 `script` 和 `style` 标签中提取出链接。
5. 使用 `format_hyperlinks` 函数将提取出的链接格式化为字符串，以便输出。
6. 最后，返回所有链接，并用空括号括起。

整个函数的实现中，使用了 `WebDriver` 和 `BeautifulSoup` 两个库。其中，`WebDriver` 用于获取页面源代码，`BeautifulSoup` 用于解析 HTML 文档并提取链接。另外，函数还使用了 `extract_hyperlinks` 函数，该函数的具体实现不在本次代码中给出，但可以推测它可能是从 `soup` 变量中提取链接并拼接为字符串。


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

This is a script written in Python that uses the Selenium WebDriver to interact with a website. It checks the current WebDriver browser (Chrome, Firefox, Edge, or Safari) and sets the appropriate options for that browser. It also installs the required Gecko/Chrome driver, if the user doesn't already have it.

The script is then using the current WebDriver to navigate to the specified URL, wait for the page to load, and then extract the body element of the page using a CSS selector.

It assumes that the current WebDriver is being used on a Linux or Linux2 platform and the user has not configured the GPU to be used for development.

It also assumes that the user has not installed the Edge browser.

Note: This script is for educational and informational purposes only and should not be used in production environments, as it may cause your application to have security vulnerabilities.


```py
def open_page_in_browser(url: str) -> WebDriver:
    """Open a browser window and load a web page using Selenium

    Params:
        url (str): The URL of the page to load

    Returns:
        driver (WebDriver): A driver object representing the browser window to scrape
    """
    logging.getLogger("selenium").setLevel(logging.CRITICAL)
    selenium_web_browser = "chrome"
    selenium_headless = True
    options_available: dict[str, Type[BrowserOptions]] = {
        "chrome": ChromeOptions,
        "edge": EdgeOptions,
        "firefox": FirefoxOptions,
        "safari": SafariOptions,
    }

    options: BrowserOptions = options_available[selenium_web_browser]()
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36"
    )

    if selenium_web_browser == "firefox":
        if selenium_headless:
            options.headless = True
            options.add_argument("--disable-gpu")
        driver = FirefoxDriver(
            service=GeckoDriverService(GeckoDriverManager().install()), options=options
        )
    elif selenium_web_browser == "edge":
        driver = EdgeDriver(
            service=EdgeDriverService(EdgeDriverManager().install()), options=options
        )
    elif selenium_web_browser == "safari":
        # Requires a bit more setup on the users end
        # See https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari
        driver = SafariDriver(options=options)
    else:
        if platform == "linux" or platform == "linux2":
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--remote-debugging-port=9222")

        options.add_argument("--no-sandbox")
        if selenium_headless:
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

这段代码是一个 Python 函数，名为 `close_browser`，它接受一个参数 `driver`，这个参数是指一个 WebDriver，它可以从浏览器中关闭网络连接。

函数内部使用 `driver.quit()` 来关闭浏览器并停止所有的 WebDriver 对象的交互，这样当程序运行结束时，所有的 WebDriver 对象和浏览器都会被安全地关闭，避免了可能的资源泄漏和其他问题。

因此，这段代码的作用是关闭当前的 WebDriver 所属的浏览器，并确保所有相关的资源和组件都被安全地关闭，以避免潜在的安全隐患。


```py
def close_browser(driver: WebDriver) -> None:
    """Close the browser

    Args:
        driver (WebDriver): The webdriver to close

    Returns:
        None
    """
    driver.quit()


```