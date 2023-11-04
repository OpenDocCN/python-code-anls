# MetaGPT源码解析 2

# `metagpt/environment.py`

这段代码是一个Python脚本，使用了`#!/usr/bin/env python`作为文件元数据。它定义了一个名为`environment.py`的文件。

脚本的主要作用是定义了一个`Environment`类，这个类用于表示一个API环境。在这里，`@Time`表示脚本创建的时间，`@Author`表示脚本的作者，`@File`表示脚本所在的文件名。

接下来，从Python标准库中导入`asyncio`、`typing.Iterable`，并定义了一个名为`Memory`的类，这个类用于表示一个内存中的数据结构。同时，定义了一个名为`Role`的类，用于表示一个角色。最后，定义了一个名为`Message`的类，用于表示一个消息。

接下来，就是定义了一些变量，包括`loop`变量，用于创建一个asyncio循环，`case`变量，用于存储一个Message实例，`message`变量，用于存储一个Message实例，以及一个`counter`变量，用于存储一个计数器。

最重要的是，这个脚本定义了一个`Environment`类，这个类包含了很多与API有关的函数和方法，例如`create_role`函数，`add_message_to_history`函数等。这些函数和方法可以用于创建一个API环境，添加消息到历史，以及获取当前消息等等。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 22:12
@Author  : alexanderwu
@File    : environment.py
"""
import asyncio
from typing import Iterable

from pydantic import BaseModel, Field

from metagpt.memory import Memory
from metagpt.roles import Role
from metagpt.schema import Message


```



This is a Python `Role` class that defines an environment and a mechanism for adding, accessing, and running roles within that environment. The `Role` class has several private fields, including `roles: dict[str, Role]` which stores the roles of the current environment, `memory: Memory` which stores the memory for each role, and `history: str` which stores the history log. It also has public methods for adding roles, getting roles, and running the roles.

The `add_role` method adds a role to the current environment, setting the `roles` dictionary and the `memory` attribute to the newly added role.

The `add_roles` method adds a batch of roles to the current environment, setting the `roles` dictionary to the new roles and appending the new roles to the `roles` dictionary.

The `publish_message` method is a low-level method for publishing a message to the current environment. It adds the message to the `memory` attribute, which is a queue for storing messages. It also appends the message to the `history` attribute.

The `run` method is the main method for handling all roles in the current environment. It runs the roles one by one, updating the `message_queue` attribute with the output of each role.

The `get_roles` method returns the roles of the current environment.

The `get_role` method takes a role name as an argument and returns the role corresponding to that name. If the role is not found in the roles dictionary, it returns `None`.


```py
class Environment(BaseModel):
    """环境，承载一批角色，角色可以向环境发布消息，可以被其他角色观察到
       Environment, hosting a batch of roles, roles can publish messages to the environment, and can be observed by other roles
    
    """

    roles: dict[str, Role] = Field(default_factory=dict)
    memory: Memory = Field(default_factory=Memory)
    history: str = Field(default='')

    class Config:
        arbitrary_types_allowed = True

    def add_role(self, role: Role):
        """增加一个在当前环境的角色
           Add a role in the current environment
        """
        role.set_env(self)
        self.roles[role.profile] = role

    def add_roles(self, roles: Iterable[Role]):
        """增加一批在当前环境的角色
            Add a batch of characters in the current environment
        """
        for role in roles:
            self.add_role(role)

    def publish_message(self, message: Message):
        """向当前环境发布信息
          Post information to the current environment
        """
        # self.message_queue.put(message)
        self.memory.add(message)
        self.history += f"\n{message}"

    async def run(self, k=1):
        """处理一次所有信息的运行
        Process all Role runs at once
        """
        # while not self.message_queue.empty():
        # message = self.message_queue.get()
        # rsp = await self.manager.handle(message, self)
        # self.message_queue.put(rsp)
        for _ in range(k):
            futures = []
            for role in self.roles.values():
                future = role.run()
                futures.append(future)

            await asyncio.gather(*futures)

    def get_roles(self) -> dict[str, Role]:
        """获得环境内的所有角色
           Process all Role runs at once
        """
        return self.roles

    def get_role(self, name: str) -> Role:
        """获得环境内的指定角色
           get all the environment roles
        """
        return self.roles.get(name, None)

```

# `metagpt/inspect_module.py`

这段代码是一个Python脚本，用于打印模块中的类和函数。具体来说，它实现了以下功能：

1. 导入`inspect`模块以使用`inspect.getmembers()`函数获取模块中的成员（包括类和函数）。
2. 定义了一个`print_classes_and_functions()`函数，该函数接受一个`module`参数。
3. 在函数内部，使用`inspect.isclass()`和`inspect.isfunction()`函数来检查对象是否为类或函数。
4. 如果对象是类，函数将打印类的名称；如果对象是函数，函数将打印函数的名称。
5. 通过调用`print()`函数来打印模块的成员列表。
6. 最后，通过调用`dir()`函数获取模块的名称，并打印出来。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/28 14:54
@Author  : alexanderwu
@File    : inspect_module.py
"""

import inspect

import metagpt  # replace with your module


def print_classes_and_functions(module):
    """FIXME: NOT WORK.. """
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            print(f'Class: {name}')
        elif inspect.isfunction(obj):
            print(f'Function: {name}')
        else:
            print(name)

    print(dir(module))


```

这段代码是一个if语句，判断当前脚本是否为__main__。如果是，那么就会执行if语句内的内容。在这段if语句中，使用了一个函数print_classes_and_functions，它接受一个名为metagpt的参数。这个函数的功能是打印出所有类和函数的名称和定义。


```py
if __name__ == '__main__':
    print_classes_and_functions(metagpt)
```

# `metagpt/llm.py`

这段代码是一个Python脚本，用于从两个不同的API中获取人工智能生成的文本。具体来说，它实现了以下功能：

1. 从metagpt.provider.anthropic_api中的LLM类中获取人工智能生成的文本。
2. 从metagpt.provider.openai_api中的LLM类中获取人工智能生成的文本。
3. 将两个获取到的文本进行比较，以确定它们是否相同。
4. 如果两个文本相同，则返回生成的文本。

代码中定义了一个名为ai_func的函数，它接受一个字符串参数prompt，并使用LLM中的aask方法将其转换为人工智能生成的文本。函数返回生成的文本，如果两个文本相同，则返回生成的文本，否则返回None。

最后，在代码的底部，通过调用DEFAULT_LLM.aask("What is the meaning of life?")"来测试两个LLM的性能。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:45
@Author  : alexanderwu
@File    : llm.py
"""

from metagpt.provider.anthropic_api import Claude2 as Claude
from metagpt.provider.openai_api import OpenAIGPTAPI as LLM

DEFAULT_LLM = LLM()
CLAUDE_LLM = Claude()

async def ai_func(prompt):
    """使用LLM进行QA
       QA with LLMs
     """
    return await DEFAULT_LLM.aask(prompt)

```

# `metagpt/logs.py`

这段代码定义了一个名为 "logs.py" 的 Python 模块，包含了一个名为 "define_log_level" 的函数，以及一些导入的模块和函数。

函数 "define_log_level" 的作用是调整日志级别，将其设置为 "INFO" 或 "DEBUG" 级别之上。它通过调用另一个名为 "log


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/6/1 12:41
@Author  : alexanderwu
@File    : logs.py
"""

import sys

from loguru import logger as _logger

from metagpt.const import PROJECT_ROOT

def define_log_level(print_level="INFO", logfile_level="DEBUG"):
    """调整日志级别到level之上
       Adjust the log level to above level
    """
    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(PROJECT_ROOT / 'logs/log.txt', level=logfile_level)
    return _logger

```

这段代码创建了一个名为 "logger" 的变量，并使用 "define_log_level()" 函数将其设置为特定的日志级别。这个函数可以用来定义一个日志级别，告诉程序在哪些情况下应该输出日志，以及输出方式。输出结果通常会写入一个文件或控制台，并可以包含不同级别的信息，如错误信息、警告信息和普通日志信息。


```py
logger = define_log_level()

```

# `metagpt/manager.py`

以上代码实现了一个简单的意图处理引擎，接收一个消息并应用指定的角色处理逻辑。该引擎目前仅支持一个角色"Product Manager"，并且通过一个简单的字典根据当前的消息和角色信息匹配一个角色，然后将消息传递给该角色。

在代码中，首先定义了一个`handle`方法来处理消息，该方法首先获取当前环境中的所有角色，然后根据当前消息和这些角色的信息匹配一个角色，并使用该角色来处理消息。如果找到一个匹配的角色，则该角色将消息处理完毕，否则会输出一条错误消息。

接着定义了一个`prompt_template`变量，用于在消息和当前环境中的角色信息之间进行模板填充，生成一个询问角色应该如何处理消息的格式。

最后，在`__init__`方法中，将`Prompt`类实例化，并将其注册到系统中，以便在需要时可以服务。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:42
@Author  : alexanderwu
@File    : manager.py
"""
from metagpt.llm import LLM
from metagpt.logs import logger
from metagpt.schema import Message


class Manager:
    def __init__(self, llm: LLM = LLM()):
        self.llm = llm  # Large Language Model
        self.role_directions = {
            "BOSS": "Product Manager",
            "Product Manager": "Architect",
            "Architect": "Engineer",
            "Engineer": "QA Engineer",
            "QA Engineer": "Product Manager"
        }
        self.prompt_template = """
        Given the following message:
        {message}

        And the current status of roles:
        {roles}

        Which role should handle this message?
        """

    async def handle(self, message: Message, environment):
        """
        管理员处理信息，现在简单的将信息递交给下一个人
        The administrator processes the information, now simply passes the information on to the next person
        :param message:
        :param environment:
        :return:
        """
        # Get all roles from the environment
        roles = environment.get_roles()
        # logger.debug(f"{roles=}, {message=}")

        # Build a context for the LLM to understand the situation
        # context = {
        #     "message": str(message),
        #     "roles": {role.name: role.get_info() for role in roles},
        # }
        # Ask the LLM to decide which role should handle the message
        # chosen_role_name = self.llm.ask(self.prompt_template.format(context))

        # FIXME: 现在通过简单的字典决定流向，但之后还是应该有思考过程
        #The direction of flow is now determined by a simple dictionary, but there should still be a thought process afterwards
        next_role_profile = self.role_directions[message.role]
        # logger.debug(f"{next_role_profile}")
        for _, role in roles.items():
            if next_role_profile == role.profile:
                next_role = role
                break
        else:
            logger.error(f"No available role can handle message: {message}.")
            return

        # Find the chosen role and handle the message
        return await next_role.handle(message)

```

# `metagpt/schema.py`

这段代码定义了一个Python函数，名为`schema.py`，属于`env.py`包。函数内部定义了一些函数和方法，用于在定义数据结构和验证函数中使用。

具体来说，这段代码实现了一个自定义的`Annotated`类型，用于表示在Pydantic中定义的数据结构和验证规则中使用的字段和参数的类型。这个自定义类型实现了`Annotated`接口，其中包含了一些特殊的类型，如`Title`、`Description`和`Url`等，用于表示文档属性的类型。

此外，这段代码还定义了一个`BaseModel`类，用于定义一些通用的数据结构和验证规则。这个类实现了`BaseModel`接口，其中包含了一些通用的方法和属性，如`Field`、`URL`和`Deprecated']`等，用于表示数据结构和验证规则中使用的字段和参数的类型。

最后，这段代码还定义了一个`schema`函数，用于验证定义的数据结构和传递的参数是否符合定义的规则。这个函数使用了`metagpt.logs.logger`函数，用于在日志中输出验证失败的信息。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/8 22:12
@Author  : alexanderwu
@File    : schema.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type, TypedDict

from pydantic import BaseModel

from metagpt.logs import logger


```

这段代码定义了一个名为RawMessage的类，该类继承自TypedDict（类型字典）的类。RawMessage类有两个属性：内容（type="str"), 和角色（type="str")。

接着定义了一个名为Message的类，该类实现了BaseModel（类模型）的接口。Message类有两个属性：内容（type="str"), 指令内容（type="str"), 角色（type="str")。此外，还有两个属性：发送者（type="str"), 发送给的人（type="str") 和受限制的人（type="str")。

并且，RawMessage和Message类都使用了__init__()函数（初始化函数）。在__init__()函数中，内容参数为TypedDict的content类型，role参数为TypedDict的role类型。

另外，__str__()函数和__repr__()函数用于打印对象的字符串表示形式。

最后，还定义了一个方法to_dict()，该方法将对象的字典表示形式返回。


```py
class RawMessage(TypedDict):
    content: str
    role: str


@dataclass
class Message:
    """list[<role>: <content>]"""
    content: str
    instruct_content: BaseModel = field(default=None)
    role: str = field(default='user')  # system / user / assistant
    cause_by: Type["Action"] = field(default="")
    sent_from: str = field(default="")
    send_to: str = field(default="")
    restricted_to: str = field(default="")

    def __str__(self):
        # prefix = '-'.join([self.role, str(self.cause_by)])
        return f"{self.role}: {self.content}"

    def __repr__(self):
        return self.__str__()

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content
        }


```

这段代码定义了两个数据类，UserMessage 和 SystemMessage，它们继承自 Message 类。UserMessage 和 SystemMessage 都实现了 __init__ 方法，用于初始化消息内容并为消息添加发送者和接收者。

UserMessage 和 SystemMessage 的区别在于它们的发送者和接收者。UserMessage 的发送者是当前用户，接收者是系统，而 SystemMessage 的发送者是系统，接收者是当前用户。

在实际应用中，这段代码可以用于创建并发送系统或用户消息，例如通知用户有新消息、提醒系统有新消息等。


```py
@dataclass
class UserMessage(Message):
    """便于支持OpenAI的消息
       Facilitate support for OpenAI messages
    """
    def __init__(self, content: str):
        super().__init__(content, 'user')


@dataclass
class SystemMessage(Message):
    """便于支持OpenAI的消息
       Facilitate support for OpenAI messages
    """
    def __init__(self, content: str):
        super().__init__(content, 'system')


```

这段代码定义了一个名为 AIMessage 的类，该类继承自 Message 类，用于支持 OpenAI 的消息。在 AIMessage 类中，有一个构造函数，用于初始化该类的实例并设置其内容类型为 'assistant'。

在 if __name__ == '__main__' 语句下，代码创建了一个包含三个消息实例的列表，并将其打印到控制台。每个实例都由 UserMessage、SystemMessage 和 AIMessage 类创建，其中 UserMessage 和 SystemMessage 类分别用于表示测试内容和普通消息，而 AIMessage 类用于模拟 OpenAI 的消息。在打印这些消息后，代码将关闭控制台并停止执行。


```py
@dataclass
class AIMessage(Message):
    """便于支持OpenAI的消息
       Facilitate support for OpenAI messages
    """
    def __init__(self, content: str):
        super().__init__(content, 'assistant')


if __name__ == '__main__':
    test_content = 'test_message'
    msgs = [
        UserMessage(test_content),
        SystemMessage(test_content),
        AIMessage(test_content),
        Message(test_content, role='QA')
    ]
    logger.info(msgs)

```

# `metagpt/software_company.py`

这段代码定义了一个名为`software_company.py`的Python文件，用于定义一个公司软件产品的配置。

该文件使用了`pydantic`库，这是Python中一个用于定义API和数据结构的轻量级定义库。在这个例子中，使用`@pydantic.BaseModel`定义了一个名为`SoftwareCompany`的模型，该模型包含了`公司名称`、`公司描述`、`可运行环境`和`支持的语言`等属性。

接着，从`metagpt.actions`库中使用`BossRequirement`类来定义任务需求，从`metagpt.config`库中使用`CONFIG`类来定义配置文件中的参数，从`metagpt.environment`库中使用`Environment`类来定义当前的实验环境，从`metagpt.logs`库中使用`logger`类来输出日志信息，从`metagpt.roles`库中使用`Role`类来定义当前的角色，从`metagpt.schema`库中使用`Message`类来定义消息。

最后，导入了需要使用到的库，并定义了一些常量和函数，用于进行一些通用的操作。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 00:30
@Author  : alexanderwu
@File    : software_company.py
"""
from pydantic import BaseModel, Field

from metagpt.actions import BossRequirement
from metagpt.config import CONFIG
from metagpt.environment import Environment
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
```

这段代码定义了一个名为 "SoftwareCompany" 的类，它继承了 "BaseModel" 类，用于定义数据结构和操作。具体来说，这个类包含以下字段：

- `environment`：一个环境对象，可以使用 `Environment` 类进行创建。这个对象用于存储公司的信息，例如 maximum budget（最大预算）和 employees（员工）。
- `investment`：一个浮点数，用于存储公司的投资。如果投资超过最大预算，会引发 `NoMoneyException`。
- `idea`：一个字符串，用于存储公司的理念。

以及以下方法：

- `hire`：用于雇佣员工。
- `invest`：用于增加公司的投资。超过最大预算时会引发 `NoMoneyException`。
- `_check_balance`：用于检查公司的资金是否充足，如果超过最大预算，会引发 `NoMoneyException`。
- `start_project`：用于启动一个新项目，并发布一个消息给所有员工。
- `_save`：用于将公司状态保存到 JSON 格式。
- `run`：用于运行公司，直到达到预期的目标轮数或者公司的资金用光。


```py
from metagpt.utils.common import NoMoneyException


class SoftwareCompany(BaseModel):
    """
    Software Company: Possesses a team, SOP (Standard Operating Procedures), and a platform for instant messaging,
    dedicated to writing executable code.
    """
    environment: Environment = Field(default_factory=Environment)
    investment: float = Field(default=10.0)
    idea: str = Field(default="")

    class Config:
        arbitrary_types_allowed = True

    def hire(self, roles: list[Role]):
        """Hire roles to cooperate"""
        self.environment.add_roles(roles)

    def invest(self, investment: float):
        """Invest company. raise NoMoneyException when exceed max_budget."""
        self.investment = investment
        CONFIG.max_budget = investment
        logger.info(f'Investment: ${investment}.')

    def _check_balance(self):
        if CONFIG.total_cost > CONFIG.max_budget:
            raise NoMoneyException(CONFIG.total_cost, f'Insufficient funds: {CONFIG.max_budget}')

    def start_project(self, idea):
        """Start a project from publishing boss requirement."""
        self.idea = idea
        self.environment.publish_message(Message(role="BOSS", content=idea, cause_by=BossRequirement))

    def _save(self):
        logger.info(self.json())

    async def run(self, n_round=3):
        """Run company until target round or no money"""
        while n_round > 0:
            # self._save()
            n_round -= 1
            logger.debug(f"{n_round=}")
            self._check_balance()
            await self.environment.run()
        return self.environment.history
    
```

# `metagpt/_compat.py`

这段代码的作用是使用Python的`platform`、`sys`和`warnings`库，对Python 3.9进行一些升级和修改，以支持Windows操作系统。

首先，它检查了当前Python解释器的名称是否为`cpython`，如果是，则执行以下操作：

1. 导入`asyncio`库，以便在后续代码中使用异步编程。
2. 导入`warnings`库，以便在需要时捕获警告信息。
3. 如果当前Python版本前两个字符为`3.9`，那么执行以下操作：

4. 导入`asyncio.proactor_events`库的`_ProactorBasePipeTransport`类，以便在应用程序中使用`asyncio.proactor_events`库的异步传输。
5. 如果当前Python版本大于或等于3.9且大于2，那么执行以下操作：

6. 从`semantic_kernel.orchestration`库中导入`sk_function`，作为`asyncio`库中的一个异步函数。
7. 将当前事件循环策略设置为`asyncio.WindowsProactorEventLoopPolicy()`，以便在事件循环中使用Windows代理程序。


```py
import platform
import sys
import warnings

if sys.implementation.name == "cpython" and platform.system() == "Windows":
    import asyncio

    if sys.version_info[:2] == (3, 9):
        from asyncio.proactor_events import _ProactorBasePipeTransport

        # https://github.com/python/cpython/pull/92842
        def pacth_del(self, _warn=warnings.warn):
            if self._sock is not None:
                _warn(f"unclosed transport {self!r}", ResourceWarning, source=self)
                self._sock.close()

        _ProactorBasePipeTransport.__del__ = pacth_del

    if sys.version_info >= (3, 9, 0):
        from semantic_kernel.orchestration import sk_function as _  # noqa: F401

        # caused by https://github.com/microsoft/semantic-kernel/pull/1416
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

```

# `metagpt/__init__.py`

这段代码是一个Python脚本，使用了Python标准库中的`metagpt`库。具体来说，它是一个命令行工具，可以用来安装`metagpt`包。

具体来说，这个脚本做以下几件事情：

1. 导入`metagpt`库的`_compat`模块，以便使用其中的函数。
2. 定义一个名为`_`的常量，表示这是一个脚本，不需要保存文件。
3. 导入`os`库，以便使用其中的文件操作函数。
4. 创建一个`MetaGPT`类的实例，并将其赋值给`_`常量的`client`属性。
5. 创建一个名为`__init__`的函数，该函数在脚本启动时执行，不做任何其他事情。
6. 在`__init__`函数中，使用`os.system`函数创建一个新文件，并将其命名为`<metagpt_console>`，以便在命令行中使用。
7. 使用`print`函数输出一条消息，表明这是一个脚本，并告知用户如何使用它。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/24 22:26
# @Author  : alexanderwu
# @File    : __init__.py

from metagpt import _compat as _  # noqa: F401

```

# `metagpt/actions/action.py`

这段代码定义了一个 Python 函数 action.py，该函数接受一个 ABC 类型的参数，并返回一个 ActionOutput 类型的对象。函数的作用是帮助用户创建一个自定义的动作（action）。

具体来说，这段代码：

1. 导入了 required 的模块，包括 re、abc 和 typing。
2. 定义了一个名为 action.py 的函数。
3. 在函数内部，导入了 metagpt.actions.action_output 和 metagpt.llm 模块，这些模块可能与函数的作用有关，但具体是什么模块并不清楚。
4. 在 function 内部，定义了一个 ABC 类型的参数 action，该参数用于指定要执行的动作。
5. 定义了一个名为 retry 的函数，用于设置尝试执行操作的最小次数。
6. 定义了一个名为 stop_after_attempt 的函数，用于设置在每次尝试执行操作后停止尝试的最小时间（以秒为单位）。
7. 定义了一个名为 wait_fixed 的函数，用于设置每次尝试执行操作后等待的最长时间（以秒为单位）。
8. 在函数内部，定义了一个名为 ActionOutput 的类，该类可能与函数返回的对象的类型有关。
9. 在 function 内部，使用 ABC 和 typing 模块定义了参数 action 的类型，使其可以接受一个 ActionOutput 类型的参数。
10. 在 function 内部，使用 retry、stop_after_attempt 和 wait_fixed 函数来设置尝试执行操作的最小次数、停止尝试的时间以及等待的时间，以确保动作可以成功执行。
11. 在 function 内部，使用 metagpt.actions.action_output 和 metagpt.llm 模块来创建 ActionOutput 对象，并将其返回。
12. 在 function 内部，使用 action.py/ABC 和 action.py/action_output 模块创建一个自定义的动作，并将其返回。

这段代码的具体作用和使用方法取决于需要创建的动作类型和需要的其他功能。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : action.py
"""
import re
from abc import ABC
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from metagpt.actions.action_output import ActionOutput
from metagpt.llm import LLM
```

This is a class that inherits from the `Action` class provided by the `concurrent.futures` module. This class allows you to run an asynchronous action with a given prompt or query.

The class has a `prefix` attribute, which is a string that is appended to the prompt or query before it is sent to the action. Additionally, the class has an optional `system_msgs` parameter, which is a list of system messages to append to the prompt or query.

The class has a `_aask_v1` method which appends the prefix to the system messages and uses the `llm.aask` method to send the prompt or query to the action. The `format` parameter determines the format of the output, and the default format is "markdown".

The class also has a `run` method which should be implemented in a subclass, but it does not have any implementation in this class.

It's important to note that this class should be used in a way that ensures the action is runny, as `concurrent.futures` and `actionlib` do not threading.


```py
from metagpt.logs import logger
from metagpt.utils.common import OutputParser
from metagpt.utils.custom_decoder import CustomDecoder


class Action(ABC):
    def __init__(self, name: str = "", context=None, llm: LLM = None):
        self.name: str = name
        if llm is None:
            llm = LLM()
        self.llm = llm
        self.context = context
        self.prefix = ""
        self.profile = ""
        self.desc = ""
        self.content = ""
        self.instruct_content = None

    def set_prefix(self, prefix, profile):
        """Set prefix for later usage"""
        self.prefix = prefix
        self.profile = profile

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()

    async def _aask(self, prompt: str, system_msgs: Optional[list[str]] = None) -> str:
        """Append default prefix"""
        if not system_msgs:
            system_msgs = []
        system_msgs.append(self.prefix)
        return await self.llm.aask(prompt, system_msgs)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    async def _aask_v1(
        self,
        prompt: str,
        output_class_name: str,
        output_data_mapping: dict,
        system_msgs: Optional[list[str]] = None,
        format="markdown",  # compatible to original format
    ) -> ActionOutput:
        """Append default prefix"""
        if not system_msgs:
            system_msgs = []
        system_msgs.append(self.prefix)
        content = await self.llm.aask(prompt, system_msgs)
        logger.debug(content)
        output_class = ActionOutput.create_model_class(output_class_name, output_data_mapping)

        if format == "json":
            pattern = r"\[CONTENT\](\s*\{.*?\}\s*)\[/CONTENT\]"
            matches = re.findall(pattern, content, re.DOTALL)

            for match in matches:
                if match:
                    content = match
                    break

            parsed_data = CustomDecoder(strict=False).decode(content)

        else:  # using markdown parser
            parsed_data = OutputParser.parse_data_with_mapping(content, output_data_mapping)

        logger.debug(parsed_data)
        instruct_content = output_class(**parsed_data)
        return ActionOutput(content, instruct_content)

    async def run(self, *args, **kwargs):
        """Run action"""
        raise NotImplementedError("The run method should be implemented in a subclass.")

```

# `metagpt/actions/action_output.py`

该代码定义了一个名为`ActionOutput`的类，用于表示动作输出的内容以及附加的指令内容。

具体来说，该类有两个方法：`create_model_class`和`__validator_check_name__`，`__root_validator_check_missing_fields__`。

`create_model_class`方法用于创建一个类的实例，将`content`字段标记为`is_aligned=True`，将`instruct_content`字段标记为`validate=True`。这样，`ActionOutput`类就可以被作为`BaseModel`的子类使用。

`__validator_check_name__`方法用于验证给定的名称是否存在于映射中的键中。如果名称不在映射中，会抛出一个`ValueError`。

`__root_validator_check_missing_fields__`方法用于验证给定的值是否满足特定的条件。如果值不满足条件，会抛出一个`ValueError`。


```py
#!/usr/bin/env python
# coding: utf-8
"""
@Time    : 2023/7/11 10:03
@Author  : chengmaoyu
@File    : action_output
"""

from typing import Dict, Type

from pydantic import BaseModel, create_model, root_validator, validator


class ActionOutput:
    content: str
    instruct_content: BaseModel

    def __init__(self, content: str, instruct_content: BaseModel):
        self.content = content
        self.instruct_content = instruct_content

    @classmethod
    def create_model_class(cls, class_name: str, mapping: Dict[str, Type]):
        new_class = create_model(class_name, **mapping)

        @validator('*', allow_reuse=True)
        def check_name(v, field):
            if field.name not in mapping.keys():
                raise ValueError(f'Unrecognized block: {field.name}')
            return v

        @root_validator(pre=True, allow_reuse=True)
        def check_missing_fields(values):
            required_fields = set(mapping.keys())
            missing_fields = required_fields - set(values.keys())
            if missing_fields:
                raise ValueError(f'Missing fields: {missing_fields}')
            return values

        new_class.__validator_check_name = classmethod(check_name)
        new_class.__root_validator_check_missing_fields = classmethod(check_missing_fields)
        return new_class
    
```

# `metagpt/actions/add_requirement.py`

这段代码定义了一个名为 "add\_requirement.py" 的 Python 文件，并在其中定义了一个名为 "BossRequirement" 的类，该类继承自 "Action" 类。

具体来说，这个类有一个名为 "run" 的方法，该方法接受一个或多个参数，代表用户执行动作需要满足的要求。在这个方法中，使用到了 "metagpt.actions" 包中的 "Action" 和 "asyncio" 包中的 "await" 和 "raise" 关键字。

"metagpt.actions" 包提供了各种用于生成人类文章的 AI 动作，而 "asyncio" 包则提供了跨平台的并发编程支持。

通过 `import metagpt.actions` 导入 "Action" 类，可能是因为在代码中需要使用这些类的某些方法。而 `class BossRequirement(Action)` 则将 "Action" 类中的方法继承并重写了 "run" 方法，并添加了一个新的 "BossRequirement" 类，供满足某种特定要求时执行。

在 "run" 方法的实现中，使用了 `raise NotImplementedError` 来引发一个异常，这个异常将会在调用该方法时被捕捉，并导致程序崩溃。这是因为该方法中并没有实现 "run" 方法的具体逻辑，只是用来代表这个类需要满足的要求。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/20 17:46
@Author  : alexanderwu
@File    : add_requirement.py
"""
from metagpt.actions import Action


class BossRequirement(Action):
    """Boss Requirement without any implementation details"""
    async def run(self, *args, **kwargs):
        raise NotImplementedError

```

# `metagpt/actions/analyze_dep_libs.py`

这段代码是一个Python脚本，使用了#!/usr/bin/env python作为#号。这个脚本可以运行在Linux或类Unix的操作系统上。

脚本的作用是执行以下操作：

1. 从metagpt库中使用actions模块中的Action类创建一个Action对象。
2. 使用该Action对象的learning_ goal属性设置一个学习目标，其值为"分析依赖库"。
3. 使用该Action对象的result_str属性设置一个字符串，其中包含一个自定义的结果，这个结果将显示给用户。
4. 将执行该脚本的人（即脚本的作者）的信息存储在变量中，以便在结果中使用。
5. 输出当前日期和时间。

简而言之，这段代码的主要目的是创建一个根据用户意图分析依赖库的Python脚本，并将结果输出给用户。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/19 12:01
@Author  : alexanderwu
@File    : analyze_dep_libs.py
"""

from metagpt.actions import Action

PROMPT = """You are an AI developer, trying to write a program that generates code for users based on their intentions.

For the user's prompt:

---
```

这段代码是一个Python类，名为"AnalyzeDepLibs"，它实现了"Action"接口。这个类的使用方法是异步执行，需要一个参数"requirement"和一个参数"filepaths_string"。

这段代码的作用是分析程序的依赖关系，根据给定的依赖关系文件返回设计文件列表。主要步骤包括：

1. 将给定的prompt参数作为字符串，并将其与文件路径字符串拼接。
2. 通过调用"_aask"方法获取Prompt和设计文件列表。
3. 返回设计文件列表。


```py
The API is: {prompt}
---

We decide the generated files are: {filepaths_string}

Now that we have a file list, we need to understand the shared dependencies they have.
Please list and briefly describe the shared contents between the files we are generating, including exported variables, 
data patterns, id names of all DOM elements that javascript functions will use, message names and function names.
Focus only on the names of shared dependencies, do not add any other explanations.
"""


class AnalyzeDepLibs(Action):
    def __init__(self, name, context=None, llm=None):
        super().__init__(name, context, llm)
        self.desc = "Analyze the runtime dependencies of the program based on the context"

    async def run(self, requirement, filepaths_string):
        # prompt = f"Below is the product requirement document (PRD):\n\n{prd}\n\n{PROMPT}"
        prompt = PROMPT.format(prompt=requirement, filepaths_string=filepaths_string)
        design_filenames = await self._aask(prompt)
        return design_filenames

```

# `metagpt/actions/azure_tts.py`

This is a Python class that uses the Azure Text-to-Speech (TTS) service to synthesize speech from text files. The class has an `Action` implementation and requires a `Config` object to be passed in to configure the TTS settings.

The Azure TTS class has a `synthesize_speech` method which takes in the language code, the voice to use for the synthesis, the role of the synthesis, and the text to be synthesized. The method returns a `Synthesizer` object that is used to synthesize the speech.

The class also has a `Config` class that has a `get` method for the current configuration. This class is used to read the configuration from a file or a hard drive.

Overall, this class makes it easy to use Azure TTS to synthesize speech from text files in various languages and with various voices.


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/6/9 22:22
@Author  : Leo Xiao
@File    : azure_tts.py
"""
from azure.cognitiveservices.speech import AudioConfig, SpeechConfig, SpeechSynthesizer

from metagpt.actions.action import Action
from metagpt.config import Config


class AzureTTS(Action):
    def __init__(self, name, context=None, llm=None):
        super().__init__(name, context, llm)
        self.config = Config()

    # Parameters reference: https://learn.microsoft.com/zh-cn/azure/cognitive-services/speech-service/language-support?tabs=tts#voice-styles-and-roles
    def synthesize_speech(self, lang, voice, role, text, output_file):
        subscription_key = self.config.get('AZURE_TTS_SUBSCRIPTION_KEY')
        region = self.config.get('AZURE_TTS_REGION')
        speech_config = SpeechConfig(
            subscription=subscription_key, region=region)

        speech_config.speech_synthesis_voice_name = voice
        audio_config = AudioConfig(filename=output_file)
        synthesizer = SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config)

        # if voice=="zh-CN-YunxiNeural":
        ssml_string = f"""
            <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='{lang}' xmlns:mstts='http://www.w3.org/2001/mstts'>
                <voice name='{voice}'>
                    <mstts:express-as style='affectionate' role='{role}'>
                        {text}
                    </mstts:express-as>
                </voice>
            </speak>
            """

        synthesizer.speak_ssml_async(ssml_string).get()


```

这段代码的作用是使用 AzureTTS 库合成一段中文语音，并在听到合成语音时将其保存为 WAV 文件。

具体来说，代码首先定义了一个名为 __main__ 的全局变量，这个变量在 Python 程序中为 True。接着，代码使用 AzureTTS 函数分别从 "azure_tts" 和 "azure_tts_container" 两种不同的方式导入了 AzureTTS 库。其中，第一种方式将 AzureTTS 库的源代码直接复制到 Python 程序中，第二种方式则是将 AzureTTS 库的依赖项添加到 Python 程序的库中，并使用 Python 程序中的 "import" 语句来调用 AzureTTS 库中的函数。

接下来，代码通过 AzureTTS.synthesize_speech 函数合成了两段语音，一段是 "Hello, I am Kaka"，另一段是 "output.wav"。其中，第一个参数是目标语言和语音参数，第二个参数是输出音频文件的路径。在这里，合成好的两段语音都被转换为了 wave 格式 (wav)，并分别保存为 output.wav 和 output_2.wav 文件。


```py
if __name__ == "__main__":
    azure_tts = AzureTTS("azure_tts")
    azure_tts.synthesize_speech(
        "zh-CN",
        "zh-CN-YunxiNeural",
        "Boy",
        "Hello, I am Kaka",
        "output.wav")

```

# `metagpt/actions/clone_function.py`

这段代码的作用是将从头目录（pathlib）中获取一个路径对象（Path），然后将从这个路径对象中包含的函数代码（source_code）转换成函数格式（template_func）。转换后的函数代码将用于从给定的列表（请在题目中查看）中选择一个函数，并将其实现为单个函数。如果需要，将使用import语句（在给定列表中包含的函数中使用import语句）。实现后的函数将包含在使用WriteCode的动作中进行输出。


```py
from pathlib import Path
import traceback

from metagpt.actions.write_code import WriteCode
from metagpt.logs import logger
from metagpt.schema import Message
from metagpt.utils.highlight import highlight

CLONE_PROMPT = """
*context*
Please convert the function code ```{source_code}```py into the the function format: ```{template_func}```py.
*Please Write code based on the following list and context*
1. Write code start with ```, and end with ```py.
2. Please implement it in one function if possible, except for import statements. for exmaple:
```python
```py



This code defines a class `CloneFunction` that inherits from the `WriteCode` class. The `CloneFunction` class has an `__init__` method, which initializes the class with a name, a context, and an optional logging level. The `_save` method is a special method that saves the provided code to a file.

The `run` method is the main function of the `CloneFunction` class. It takes two arguments: a template function `template_func` and a source code string `source_code`. The method converts the source code to a format that is compatible with the `template_func`, and then returns the converted code.

The `template_func` should be a string that follows the format of `{template_func}`, where `{template_func}` is the name of the template function that should be applied to the input code. If the `template_func` is not found in the `CloneFunction` class, the conversion process will fail and an error will be raised.

Note that the input and output parameters of the `run` method should be the same as `template_func`, and the return value should be a string.


```
import pandas as pd
def run(*args) -> pd.DataFrame:
    ...
```py
3. Do not use public member functions that do not exist in your design.
4. The output function name, input parameters and return value must be the same as ```{template_func}```py.
5. Make sure the results before and after the code conversion are required to be exactly the same.
6. Don't repeat my context in your replies.
7. Return full results, for example, if the return value has df.head(), please return df.
8. If you must use a third-party package, use the most popular ones, for example: pandas, numpy, ta, ...
"""


class CloneFunction(WriteCode):
    def __init__(self, name="CloneFunction", context: list[Message] = None, llm=None):
        super().__init__(name, context, llm)

    def _save(self, code_path, code):
        if isinstance(code_path, str):
            code_path = Path(code_path)
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(code)
        logger.info(f"Saving Code to {code_path}")

    async def run(self, template_func: str, source_code: str) -> str:
        """将source_code转换成template_func一样的入参和返回类型"""
        prompt = CLONE_PROMPT.format(source_code=source_code, template_func=template_func)
        logger.info(f"query for CloneFunction: \n {prompt}")
        code = await self.write_code(prompt)
        logger.info(f'CloneFunction code is \n {highlight(code)}')
        return code


```

这段代码定义了一个名为 `run_function_code` 的函数，它接受一个字符串参数 `func_code`，一个参数 `func_name` 和三个额外的参数 `*args` 和 `**kwargs`。

`func_code` 参数是一个字符串，表示要运行的函数代码。`func_name` 参数是一个字符串，表示要给函数命名的名称。`*args` 和 `**kwargs` 参数是元组类型，表示 `func_code` 参数中的参数。这些参数将被传递给 `func` 函数，并返回一个元组，其中第一个元素是函数的返回值，第二个元素是错误信息。

函数的实现基本上是读取 `func_code` 中的代码，并使用 `exec` 函数执行该代码。执行成功后，函数将返回函数的名称，以及一个空字符串，表示函数没有返回任何值。如果执行过程中出现任何异常，函数将返回一个元组，其中第一个元素是异常信息，第二个元素是错误信息的格式。

`run_function_script` 函数是 `run_function_code` 的装饰函数，它接受一个字符串参数 `code_script_path`，一个参数 `func_name` 和三个额外的参数 `*args` 和 `**kwargs`。它首先检查 `code_script_path` 是否为字符串类型，如果是，函数将尝试读取该文件中的内容。如果不是字符串类型，函数将通过 `print` 函数将 `code_script_path` 中的内容打印出来。然后，函数将调用 `run_function_code` 函数，并传递 `code_script_path` 和 `func_name` 参数。


```py
def run_function_code(func_code: str, func_name: str, *args, **kwargs):
    """Run function code from string code."""
    try:
        locals_ = {}
        exec(func_code, locals_)
        func = locals_[func_name]
        return func(*args, **kwargs), ""
    except Exception:
        return "", traceback.format_exc()


def run_function_script(code_script_path: str, func_name: str, *args, **kwargs):
    """Run function code from script."""
    if isinstance(code_script_path, str):
        code_path = Path(code_script_path)
    code = code_path.read_text(encoding='utf-8')
    return run_function_code(code, func_name, *args, **kwargs)

```

# `metagpt/actions/debug_error.py`

这段代码是一个Python脚本，它的作用是解释如何输出错误信息。具体来说，它实现了以下功能：

1. 导入了`re`模块，这是一个Python标准库中的正则表达式模块，用于处理字符串的匹配和替换。
2. 在`import`语句中，引入了`logger`、`Action`和`CodeParser`三个模块，分别用于输出日志信息、执行具体操作和解析配置文件。
3. 在`logger.getLogger`函数中，创建了一个名为`debug`的logger实例，并将`console`输出级别设置为`debug`，这样所有输出信息都将发送到控制台。
4. 在`Action.performAction`函数中，定义了一个名为`output_error_message`的方法，该方法接受一个参数`message`，用于输出错误信息。
5. 在`CodeParser.parseConfigFile`函数中，解析了一个配置文件`debug_config.yaml`，并设置了一个名为`action_config`的配置参数。
6. 在`Action.executeAction`函数中，根据`action_config`参数，调用`output_error_message`方法输出错误信息，并返回结果。

总的来说，这段代码实现了一个简单的错误输出机制，可以在需要时将错误信息发送到控制台，并允许在配置文件中设置错误输出的行为。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:46
@Author  : alexanderwu
@File    : debug_error.py
"""
import re

from metagpt.logs import logger
from metagpt.actions.action import Action
from metagpt.utils.common import CodeParser

PROMPT_TEMPLATE = """
NOTICE
```

这段代码是一个 Python 类，名为 `DebugError`，属于 `Action` 类。它的作用是在开发或测试过程中，当其他开发或测试人员收到了一个错误时，提供一个用于修复代码的框架。

代码中包含一个 `Role` 变量，用于了解自己的角色（Engineer 或 QA Engineer），然后根据收到的信息，对代码进行开发或测试。

具体来说，当收到一个错误时，首先会根据 `Role` 判断自己的角色，然后使用一个名为 `file_name` 字段的字符串，来查找需要修复的代码文件。接着，代码会输出一个错误信息，并提示用户修复代码。如果修复后的代码通过了测试，代码将不会再次被运行。如果修复后的代码仍然无法通过测试，则说明代码存在严重的问题，需要进一步修复。


```py
1. Role: You are a Development Engineer or QA engineer;
2. Task: You received this message from another Development Engineer or QA engineer who ran or tested your code. 
Based on the message, first, figure out your own role, i.e. Engineer or QaEngineer,
then rewrite the development code or the test code based on your role, the error, and the summary, such that all bugs are fixed and the code performs well.
Attention: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the test case or script and triple quotes.
The message is as follows:
{context}
---
Now you should start rewriting the code:
## file name of the code to rewrite: Write code with triple quoto. Do your best to implement THIS IN ONLY ONE FILE.
"""
class DebugError(Action):
    def __init__(self, name="DebugError", context=None, llm=None):
        super().__init__(name, context, llm)

    # async def run(self, code, error):
    #     prompt = f"Here is a piece of Python code:\n\n{code}\n\nThe following error occurred during execution:" \
    #              f"\n\n{error}\n\nPlease try to fix the error in this code."
    #     fixed_code = await self._aask(prompt)
    #     return fixed_code
    
    async def run(self, context):
        if "PASS" in context:
            return "", "the original code works fine, no need to debug"
        
        file_name = re.search("## File To Rewrite:\s*(.+\\.py)", context).group(1)

        logger.info(f"Debug and rewrite {file_name}")

        prompt = PROMPT_TEMPLATE.format(context=context)
        
        rsp = await self._aask(prompt)

        code = CodeParser.parse_code(block="", text=rsp)

        return file_name, code

```

# `metagpt/actions/design_api.py`

这段代码是一个Python脚本，使用了`#!/usr/bin/env python`作为脚本解释器的行首。

该脚本的主要作用是定义了一个名为`design_api.py`的函数，该函数会在`WORKSPACE_ROOT`目录下创建一个名为`design_api.py`的文件。

具体来说，该脚本包含了以下操作：

1. 导入`shutil`、`pathlib`、`typing`和`metagpt.actions`、`metagpt.config`、`metagpt.const`和`metagpt.logs`模块。

2. 使用`Action`和`ActionOutput`类型定义了`design_api.py`函数的输入和输出。

3. 使用`CONFIG`类定义了一个名为`CONFIG`的配置文件。

4. 在`design_api.py`函数中，使用了`logger`类定义了一个名为`logger`的logger实例。

5. 在`design_api.py`函数中，创建了一个`Path`对象`WORKSPACE_ROOT`，并将其设置为工作区根目录。

6. 使用`shutil.copy`方法复制了`WORKSPACE_ROOT`目录下的`design_api.py`文件和`requirements.txt`文件。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 19:26
@Author  : alexanderwu
@File    : design_api.py
"""
import shutil
from pathlib import Path
from typing import List

from metagpt.actions import Action, ActionOutput
from metagpt.config import CONFIG
from metagpt.const import WORKSPACE_ROOT
from metagpt.logs import logger
```

```pycss
# 这段解释性的文本将作为答案
```python
   },
   "mermaid": {
       "PROMPT_TEMPLATE": "```pycss
# 这段解释性的文本将作为答案
```python
   },
}
```pysql
这段代码是一个自定义代码片段，定义了三种不同的模板：json、mermaid和格式化json。

json模板使用了Markdown的语法，可以轻松地将JSON格式的数据转换为Markdown格式的文本，常用于描述数据、回答等场景。

mermaid模板使用了Markdown的语法，可以轻松地将Markdown格式的文本转换为Mermaid格式的文本，常用于描述算法或者复杂的图表、流程等场景。

格式化json模板使用了Markdown的语法，可以轻松地将JSON格式的数据转换为格式化后的Markdown格式的文本，常用于回答一些需要强调文本格式的场景，如强调某个单词、短语或者句子。

这段代码还定义了一个名为templates的字典，其中包含上述三种模板，可以用来在需要使用这些模板时进行调用。


```
from metagpt.utils.common import CodeParser
from metagpt.utils.get_template import get_template
from metagpt.utils.json_to_markdown import json_to_markdown
from metagpt.utils.mermaid import mermaid_to_file

templates = {
    "json": {
        "PROMPT_TEMPLATE": """
# Context
{context}

## Format example
{format_example}
-----
Role: You are an architect; the goal is to design a SOTA PEP8-compliant python system; make the best use of good open source tools
```py

This code appears to be a Python package documentation, filled in with information about the package's requirements, implementation approach, and other details.

The `Requirement` section appears to be filled in with information about what the package needs to be done, such as filling in certain sections with missing information and specifying the maximum number of characters or tokens that can be used.

The `Implementation approach` section appears to be explaining the process that will be taken to fulfill the requirements specified in the `Requirement` section, but does not provide any details about the specific implementation.

The `Python package name` and `File list` sections appear to be straightforward requirements that need to be met, and specify the name of the package and the list of files that should be included in the package.

The `Data structures and interface definitions` section appears to be filled in with information about the data structures that will be used in the package and the interface definitions for those data structures.

The `Program call flow` section appears to be filled in with information about the flow of the program, including the CRUD (create, read, update, delete) operations that will be performed on objects.

Overall, this code appears to be providing information about the package and its requirements, as well as the specific implementation details that will be used to fulfill those requirements.


```
Requirement: Fill in the following missing information based on the context, each section name is a key in json
Max Output: 8192 chars or 2048 tokens. Try to use them up.

## Implementation approach: Provide as Plain text. Analyze the difficult points of the requirements, select the appropriate open-source framework.

## Python package name: Provide as Python str with python triple quoto, concise and clear, characters only use a combination of all lowercase and underscores

## File list: Provided as Python list[str], the list of ONLY REQUIRED files needed to write the program(LESS IS MORE!). Only need relative paths, comply with PEP8 standards. ALWAYS write a main.py or app.py here

## Data structures and interface definitions: Use mermaid classDiagram code syntax, including classes (INCLUDING __init__ method) and functions (with type annotations), CLEARLY MARK the RELATIONSHIPS between classes, and comply with PEP8 standards. The data structures SHOULD BE VERY DETAILED and the API should be comprehensive with a complete design. 

## Program call flow: Use sequenceDiagram code syntax, COMPLETE and VERY DETAILED, using CLASSES AND API DEFINED ABOVE accurately, covering the CRUD AND INIT of each object, SYNTAX MUST BE CORRECT.

## Anything UNCLEAR: Provide as Plain text. Make clear here.

```py

这段代码是一个Python代码，它将输出一个格式化好的JSON数据。这个JSON数据将被放入一个包含多个CONTENT标签的HTML元素中。

具体来说，这个代码将输出一个包含多个键值对的对象，其中键是Python中定义的一些数据结构的名称，值是相应的注释。这些键值对将用大括号{}包围，再在对象前面加上[CONTENT]标签。

例如，这个代码将输出一个包含以下内容的JSON对象：
```json
{
   "Implementation approach": "We will implement a game with scorekeeping functionality.",
   "Python package name": "snake_game",
   "File list": ["main.py"],
   "Data structures and interface definitions": "classGame, classGameScorekeeping, classGameState,\...",
   "Program call flow": "game calls gameLoop putting score up to scoreBoard",
   "Anything UNCLEAR": "The requirement is clear to me."
}
```py
这个JSON对象的格式可以根据[CONTENT]标签中的格式化字符串来控制。例如，如果要将JSON对象中的键值对之间的键值对对齐，可以使用`dict.items()`函数将Python中的数据结构名称和注释组合成一个字符串，并将其作为格式化字符串的一部分。


```
output a properly formatted JSON, wrapped inside [CONTENT][/CONTENT] like format example,
and only output the json inside this tag, nothing else
""",
        "FORMAT_EXAMPLE": """
[CONTENT]
{
    "Implementation approach": "We will ...",
    "Python package name": "snake_game",
    "File list": ["main.py"],
    "Data structures and interface definitions": '
    classDiagram
        class Game{
            +int score
        }
        ...
        Game "1" -- "1" Food: has
    ',
    "Program call flow": '
    sequenceDiagram
        participant M as Main
        ...
        G->>M: end game
    ',
    "Anything UNCLEAR": "The requirement is clear to me."
}
[/CONTENT]
```py

这段代码是一个 Python 语言的 `markdown` 类型，用于将一些Markdown格式的内容转化为格式化的语法，以便更易于阅读和理解。

具体来说，这段代码的作用是将一个复杂的 Markdown 文档转换为 Python 代码，从而使得这个文档以更加易于阅读和理解的方式展示出来。转换后的代码将包含一些注释，以帮助你更好地理解代码的作用和如何使用它。


```
""",
    },
    "markdown": {
        "PROMPT_TEMPLATE": """
# Context
{context}

## Format example
{format_example}
-----
Role: You are an architect; the goal is to design a SOTA PEP8-compliant python system; make the best use of good open source tools
Requirement: Fill in the following missing information based on the context, note that all sections are response with code form separately
Max Output: 8192 chars or 2048 tokens. Try to use them up.
Attention: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the code and triple quote.

```py

这段代码是一个用于解析需求规格描述的软件工程方法。它通过提供清晰的文本来描述如何实现一个软件系统。这个方法的目的是提供一个清晰和易于理解的指导，帮助开发人员更好地理解并实现所需的功能。

该方法首先要求提供一个Python软件包名称，以便在程序中使用。名称应该是一个Python字符串，使用所有小写字母和下划线。

接着，该方法要求提供一个文件列表，其中包含仅需要的文件。这个列表应该是一个Python列表，使用相对路径，符合PEP8标准。程序中应该有一个名为“main.py”或“app.py”的主文件，用于包含程序的入口点。

然后，该方法要求提供数据结构和接口定义，使用Mermaid类图代码语法。包括类的定义以及函数类型注释。明确定义类之间的关系，并符合PEP8标准。数据结构应该非常详细，接口应该完整且具有设计感。

接下来，该方法要求提供程序调用流程的序列 diagrams代码。使用正确的类和接口定义，覆盖创建、读取、修改和删除对象的CRUD操作。确保语法正确。

最后，如果有什么是不清楚的地方，应该使用纯文本进行说明。让读者清楚地理解软件系统的要求。


```
## Implementation approach: Provide as Plain text. Analyze the difficult points of the requirements, select the appropriate open-source framework.

## Python package name: Provide as Python str with python triple quoto, concise and clear, characters only use a combination of all lowercase and underscores

## File list: Provided as Python list[str], the list of ONLY REQUIRED files needed to write the program(LESS IS MORE!). Only need relative paths, comply with PEP8 standards. ALWAYS write a main.py or app.py here

## Data structures and interface definitions: Use mermaid classDiagram code syntax, including classes (INCLUDING __init__ method) and functions (with type annotations), CLEARLY MARK the RELATIONSHIPS between classes, and comply with PEP8 standards. The data structures SHOULD BE VERY DETAILED and the API should be comprehensive with a complete design. 

## Program call flow: Use sequenceDiagram code syntax, COMPLETE and VERY DETAILED, using CLASSES AND API DEFINED ABOVE accurately, covering the CRUD AND INIT of each object, SYNTAX MUST BE CORRECT.

## Anything UNCLEAR: Provide as Plain text. Make clear here.

""",
        "FORMAT_EXAMPLE": """
---
```py

这段代码是一个Python脚本，它定义了一个名为“snake_game”的Python包。

在注释中，代码解释了如何实现这个包。具体来说，它将包括以下步骤：

1. 将以下两个导入语句添加到脚本中：
```python
# 导入snake模块
import time
import random
```py

2. 定义一个名为“游戏主循环”函数，它是游戏的核心部分。在这个函数中，我们将创建游戏对象，处理游戏事件，并更新游戏状态。
```python
def game_loop():
   # 创建游戏对象
   game_object = game_manager()
   
   # 初始化游戏状态
   game_state = initialize_game_state()
   
   # 游戏循环的主要处理步骤
   while True:
       # 处理游戏事件
       for event in game_object.event_handler():
           # 如果是用户操作（如点击窗口）
           if event.type == 'user_action':
               # 根据用户操作更新游戏状态
               handle_user_action(game_state, event.data)
           
           # 游戏内部事件
           else:
               # 更新游戏状态
               update_game_state(game_state)
               
           # 游戏循环休眠一段时间，控制游戏速度
           time.sleep(0.1)
       
       # 游戏状态发生改变时，更新游戏画布
       update_game_display(game_state)
```py

3. 定义了一个名为“game_manager”函数，它将用于管理游戏对象。这个函数将创建一个游戏对象，包括创建游戏窗口、处理用户输入、更新游戏状态等。
```python
def game_manager():
   # 创建游戏窗口
   game_window = game_window()
   
   # 创建游戏对象
   game_object = game_from_window(game_window, game_state)
   
   # 游戏对象的状态，包括初始化、更新等
   game_object.state = 'initialized'
   
   # 将游戏对象添加到游戏窗口中
   game_window.add_game_object(game_object)
   
   # 游戏对象可能需要的一些方法，如获取游戏对象的属性
   # ...
   
   return game_object
```py

4. 定义了一个名为“initialize_game_state”函数，它将在游戏开始时初始化游戏状态。这个函数将创建一个游戏对象，并设置初始状态。
```python
def initialize_game_state():
   # 创建游戏对象
   game_object = game_from_window(game_window, game_state)
   
   # 根据初始状态设置游戏对象的属性
   game_object.state = 'initialized'
   
   # 设置游戏窗口的尺寸和游戏画布的大小
   game_window.set_size(800, 600)
   game_draw_buffer.set_size(800, 600)
   
   # 游戏开始时加载游戏元素
   game_element = load_game_element()
   game_object.load(game_element)
   
   # 创建游戏循环控制按钮
   control_button = create_control_button(game_object)
   
   # ...
   
   return game_object
```py

5. 定义了一个名为“update_game_state”函数，它将在每帧更新游戏状态。这个函数将更新游戏对象的状态，并更新游戏窗口。
```python
def update_game_state(game_state):
   # 更新游戏对象的状态
   game_object.state = 'updated'
   
   # 更新游戏窗口
   update_game_display(game_state)
```py

6. 定义了一个名为“update_game_display”函数


```
## Implementation approach
We will ...

## Python package name
```pypython
"snake_game"
```

## File list
```pypython
[
    "main.py",
]
```

```py

这段代码定义了一个游戏数据结构，包括一个名为Game的类和一个包含两个整数的属性score。在代码中，定义了一个名为"1"的类，该类包含一个食物对象，且该食物对象与名为"1"的类有关联，即继承自Game类。

这段代码的作用是创建一个游戏实例，并让用户可以玩该游戏。在游戏结束时，程序会打印出游戏得分。


```
## Data structures and interface definitions
```pymermaid
classDiagram
    class Game{
        +int score
    }
    ...
    Game "1" -- "1" Food: has
```

## Program call flow
```pymermaid
sequenceDiagram
    participant M as Main
    ...
    G->>M: end game
```

这段代码定义了一个名为 "Anything UNCLEAR" 的类，它具有以下属性和方法：

1. 属性：该类定义了一个 "Implementation approach" 属性，类型为字符串，但缺少具体内容。
2. 方法：该类定义了一个 "Implementation approach" 方法，类型为字符串，该方法没有实现任何功能。
3. 属性：该类定义了一个 "OUTPUT_MAPPING" 属性，类型为字典，包含以下键值对：
	* "Implementation approach": 类型为字符串，是一个字符串指导下的实现方法。
	* "Python package name": 类型为字符串，是该实现方法的包名称。
	* "File list": 类型为列表，是包含实现方法的文件列表。
	* "Data structures and interface definitions": 类型为字符串，是该实现方法中定义的数据结构和接口定义。
	* "Program call flow": 类型为字符串，是该实现方法中的程序调用流程。
4. 方法：该类定义了一个 " clear" 方法，类型为字符串，该方法没有实现任何功能。


```py
```

## Anything UNCLEAR
The requirement is clear to me.
---
""",
    },
}

OUTPUT_MAPPING = {
    "Implementation approach": (str, ...),
    "Python package name": (str, ...),
    "File list": (List[str], ...),
    "Data structures and interface definitions": (str, ...),
    "Program call flow": (str, ...),
    "Anything UNCLEAR": (str, ...),
}


```py

The code you provided is a Python class called `SystemDesignRunner` that inherits from the `ContextRunner` class. It appears to be responsible for running an asyncio program that creates, modifies, and saves a System Design file.

The `SystemDesignRunner` class has a `run` method that takes a `Context` object and an optional `format` parameter, which specifies the format of the Prompt to use. The `run` method starts by calling the `_aask` method, which asks the user to enter a prompt in the specified format. If the user enters an invalid format, an exception is raised. If the user enters a valid format, the `_aask_v1` method is called, which prompts the user to enter a System Design file. The System Design file is then modified and saved to the specified location. Finally, the modified System Design file is saved to the location specified by the `format` parameter.

The `SystemDesignRunner` also has a `_save` method that saves the System Design file to the specified location, as well as a `_recreate_workspace` method that recreates the workspace for the specified System Design file.


```
class WriteDesign(Action):
    def __init__(self, name, context=None, llm=None):
        super().__init__(name, context, llm)
        self.desc = (
            "Based on the PRD, think about the system design, and design the corresponding APIs, "
            "data structures, library tables, processes, and paths. Please provide your design, feedback "
            "clearly and in detail."
        )

    def recreate_workspace(self, workspace: Path):
        try:
            shutil.rmtree(workspace)
        except FileNotFoundError:
            pass  # Folder does not exist, but we don't care
        workspace.mkdir(parents=True, exist_ok=True)

    async def _save_prd(self, docs_path, resources_path, context):
        prd_file = docs_path / "prd.md"
        if context[-1].instruct_content and context[-1].instruct_content.dict()["Competitive Quadrant Chart"]:
            quadrant_chart = context[-1].instruct_content.dict()["Competitive Quadrant Chart"]
            await mermaid_to_file(quadrant_chart, resources_path / "competitive_analysis")

        if context[-1].instruct_content:
            logger.info(f"Saving PRD to {prd_file}")
            prd_file.write_text(json_to_markdown(context[-1].instruct_content.dict()))

    async def _save_system_design(self, docs_path, resources_path, system_design):
        data_api_design = system_design.instruct_content.dict()[
            "Data structures and interface definitions"
        ]  # CodeParser.parse_code(block="Data structures and interface definitions", text=content)
        seq_flow = system_design.instruct_content.dict()[
            "Program call flow"
        ]  # CodeParser.parse_code(block="Program call flow", text=content)
        await mermaid_to_file(data_api_design, resources_path / "data_api_design")
        await mermaid_to_file(seq_flow, resources_path / "seq_flow")
        system_design_file = docs_path / "system_design.md"
        logger.info(f"Saving System Designs to {system_design_file}")
        system_design_file.write_text((json_to_markdown(system_design.instruct_content.dict())))

    async def _save(self, context, system_design):
        if isinstance(system_design, ActionOutput):
            ws_name = system_design.instruct_content.dict()["Python package name"]
        else:
            ws_name = CodeParser.parse_str(block="Python package name", text=system_design)
        workspace = WORKSPACE_ROOT / ws_name
        self.recreate_workspace(workspace)
        docs_path = workspace / "docs"
        resources_path = workspace / "resources"
        docs_path.mkdir(parents=True, exist_ok=True)
        resources_path.mkdir(parents=True, exist_ok=True)
        await self._save_prd(docs_path, resources_path, context)
        await self._save_system_design(docs_path, resources_path, system_design)

    async def run(self, context, format=CONFIG.prompt_format):
        prompt_template, format_example = get_template(templates, format)
        prompt = prompt_template.format(context=context, format_example=format_example)
        # system_design = await self._aask(prompt)
        system_design = await self._aask_v1(prompt, "system_design", OUTPUT_MAPPING, format=format)
        # fix Python package name, we can't system_design.instruct_content.python_package_name = "xxx" since "Python package name" contain space, have to use setattr
        setattr(
            system_design.instruct_content,
            "Python package name",
            system_design.instruct_content.dict()["Python package name"].strip().strip("'").strip('"'),
        )
        await self._save(context, system_design)
        return system_design

```