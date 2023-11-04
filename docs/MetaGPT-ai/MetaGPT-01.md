# MetaGPT源码解析 1

# `examples/build_customized_agent.py`

这段代码是一个Python脚本，使用了 subprocess 模块来运行异步I/O操作。其主要目的是创建一个自定义的AI代理，以在 MetaGPT中执行各种任务。以下是代码的作用：

1. 导入所需的模块和函数：包括 `re`、`subprocess`、`asyncio`、`fire` 模块中的 `Action`、`Role`、`Message` 和 `logger` 函数。
2. 定义一个名为 `build_customized_agent` 的函数。
3. 在函数中，使用 `import fire` 导入 `fire` 模块，并使用 `ate` 方法创建一个 `fire.Resource` 对象，然后使用 `异步上下文管理器` 中的 `submit` 方法来提交该 `fire.Resource` 对象。
4. 在 `submit` 方法中，使用 `import asyncio` 导入 `asyncio` 模块，并使用 ` asyncio.subprocess.run` 方法运行一个异步I/O操作，该操作将使用 `Subprocess` 类从命令行中读取用户提供的参数并执行相应的命令。
5. 在 `build_customized_agent` 函数中，使用 `filer` 方法创建一个读取用户提供的配置文件内容的 `FILE` 对象，并使用 `submit` 方法提交一个 `asyncio.subprocess.run` 对象来读取用户提供的配置文件内容并将其存储为 `config` 变量。
6. 在 `submit` 方法中，使用 `asyncio.submit` 方法提交一个 `asyncio.Assuming干湿循环` 对象，该对象将异步执行 `build_customized_agent` 函数并将 `config` 变量作为参数传递给它。
7. 在 `build_customized_agent` 函数中，使用 `metagpt.actions` 模块中的 `Action` 类创建一个自定义的动作 `build_config`，并使用 `metagpt.roles` 模块中的 `Role` 类创建一个自定义的角色 `Customer`。
8. 在 `submit` 方法中，创建一个名为 `build_agent` 的 `asyncio.Assuming干湿循环` 对象，并使用 `submit` 方法提交一个 `asyncio.subprocess.run` 对象来读取用户提供的配置文件内容并将其存储为 `config` 变量，然后使用 `submit` 方法提交一个 `asyncio.Assuming干湿循环` 对象，该对象将异步执行 `build_config` 动作，并将 `config` 变量作为参数传递给它。
9. 在 `build_customized_agent` 函数中，使用 `asyncio` 模块中的 `Subprocess` 类创建一个 `asyncio.subprocess.run` 对象，该对象将使用 `submit` 方法提交一个 `asyncio.Assuming干湿循环` 对象，该对象将异步执行 `build_agent` 动作。
10. 在 `submit` 方法中，创建一个名为 `build_customized_agent` 的 `asyncio.Assuming干湿循环` 对象，并使用 `submit` 方法提交一个 `asyncio.subprocess.run` 对象来读取用户提供的配置文件内容并将其存储为 `config` 变量，然后使用 `submit` 方法提交一个 `asyncio.Assuming干湿循环` 对象，该对象将异步执行 `build_customized_agent` 函数并将 `config` 变量作为参数传递给它。
11. 在 `build_customized_agent` 函数中，创建一个名为 `submit_custom_agent` 的 `asyncio.Assuming干湿循环` 对象，并使用 `submit` 方法提交一个 `asyncio.subprocess.run` 对象来读取用户提供的配置文件内容并将其存储为 `config` 变量，然后使用 `submit` 方法提交一个 `asyncio.Assuming干湿循环` 对象，该对象将异步执行 `submit_custom_agent` 动作，并将 `config` 变量作为参数传递给它。
12. 在 `submit_custom_agent` 函数中，使用 `asyncio.submit` 方法提交一个 `asyncio.Assuming干湿循环` 对象，该对象将异步


```py
'''
Filename: MetaGPT/examples/build_customized_agent.py
Created Date: Tuesday, September 19th 2023, 6:52:25 pm
Author: garylin2099
'''
import re
import subprocess
import asyncio

import fire

from metagpt.actions import Action
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.logs import logger

```

这段代码定义了一个名为 `SimpleWriteCode` 的类，它实现了 `Action` 接口。这个类的实现包括一个名为 `run` 的方法，用于运行代码。

`run` 方法的参数是一个字符串 `instruction`，它会被格式化为一个用于输出代码的模板，然后使用 `_aask` 方法运行这个模板，并返回它的响应。

`_aask` 方法是一个方法，它的参数是一个字符串 `prompt`，它会被用来格式化输入，以作为模板的下一行。这个方法返回一个字符串 `rsp`，表示模板的响应。

如果 `instruction` 参数没有被提供，或者 `_aask` 方法返回的是一个非字符串类型，那么 `run` 方法会抛出一个异常。

模板字符串 `PROMPT_TEMPLATE` 被用来在 `run` 方法的 `instruction` 参数中插入两个测试用例。它包含一个字符串 `Write a python function that can {instruction}`，它告诉用户编写一个什么样的函数，以及两个可运行的测试用例。

最后，`SimpleWriteCode` 类的 `__init__` 方法设置了 `name`、`context` 和 `llm` 参数，使代码在实例化时可以指定。


```py
class SimpleWriteCode(Action):

    PROMPT_TEMPLATE = """
    Write a python function that can {instruction} and provide two runnnable test cases.
    Return ```python your_code_here ```py with NO other texts,
    example: 
    ```python
    # function
    def add(a, b):
        return a + b
    # test cases
    print(add(1, 2))
    print(add(3, 4))
    ```py
    your code:
    """

    def __init__(self, name="SimpleWriteCode", context=None, llm=None):
        super().__init__(name, context, llm)

    async def run(self, instruction: str):

        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)

        rsp = await self._aask(prompt)

        code_text = SimpleWriteCode.parse_code(rsp)

        return code_text

    @staticmethod
    def parse_code(rsp):
        pattern = r'```python(.*)```py'
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text

```

这段代码定义了一个名为SimpleRunCode的类，它实现了Action接口。SimpleRunCode类有一个初始化方法(__init__)，一个名为run的静态方法，和一个名为SimpleCoder的父类。

在SimpleRunCode的__init__方法中，代码继承自Action类并覆盖了__init__方法，从而实现了Action的初始化。在run方法中，使用subprocess.run方法运行一段Python代码，并获取其结果。结果保存在代码_result变量中，然后使用logger.info输出它。最后，使用Message类将结果返回。

在SimpleCoder类中，使用了一个初始化方法__init__，根据需要设置了一个name和一个profile。然后使用了一个名为_act的静态方法，它是SimpleCoder类的主要方法。

在_act方法中，首先从SimpleRunCode类中获取一个running_code参数。然后，从内存中读取最新的内存内容，并使用SimpleWriteCode类运行此代码。在执行完成后，将结果保存在代码_result变量中，并使用Message类将其返回。

SimpleRunCode类和SimpleCoder类一起组成了一个简单的命令行程序，可以接收一个运行的Python代码并输出结果。


```py
class SimpleRunCode(Action):
    def __init__(self, name="SimpleRunCode", context=None, llm=None):
        super().__init__(name, context, llm)

    async def run(self, code_text: str):
        result = subprocess.run(["python3", "-c", code_text], capture_output=True, text=True)
        code_result = result.stdout
        logger.info(f"{code_result=}")
        return code_result

class SimpleCoder(Role):
    def __init__(
        self,
        name: str = "Alice",
        profile: str = "SimpleCoder",
        **kwargs,
    ):
        super().__init__(name, profile, **kwargs)
        self._init_actions([SimpleWriteCode])

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self._rc.todo}")
        todo = self._rc.todo

        msg = self._rc.memory.get()[-1] # retrieve the latest memory
        instruction = msg.content

        code_text = await SimpleWriteCode().run(instruction)
        msg = Message(content=code_text, role=self.profile, cause_by=todo)

        return msg

```

这段代码定义了一个名为 `RunnableCoder` 的类，它实现了 `IJob` 和 `ISender` 接口，用于在 AMQP 系统中实现消息队列中的工作。

在 `__init__` 方法中，设置了 `name` 和 `profile` 两个参数，分别表示作业名称和开发人员签名，然后传递了 `**kwargs`，用于设置其他元数据。

在 `_init_actions` 方法中，定义了两个动作，分别是一个 `SimpleWriteCode` 和一个 `SimpleRunCode`，用于在 `_think` 方法中执行实际的作业操作。

在 `_act` 方法中，获取了 `todo` 参数，并实现了 `Message` 和 `Role` 两个接口，用于在作业执行时发送消息队列中的内容，并获取作业执行的结果。如果 `todo` 是 `SimpleWriteCode`，则使用 `SimpleWriteCode` 类发送消息，并将结果作为参数传递；如果 `todo` 是 `SimpleRunCode`，则使用 `SimpleRunCode` 类发送消息，并将结果作为参数传递。

在 `_react` 方法中，无限循环地执行 `_think` 和 `_act` 方法，并在 `todo` 变量为 `None` 时跳出循环，准备下一个作业执行。


```py
class RunnableCoder(Role):
    def __init__(
        self,
        name: str = "Alice",
        profile: str = "RunnableCoder",
        **kwargs,
    ):
        super().__init__(name, profile, **kwargs)
        self._init_actions([SimpleWriteCode, SimpleRunCode])

    async def _think(self) -> None:
        if self._rc.todo is None:
            self._set_state(0)
            return

        if self._rc.state + 1 < len(self._states):
            self._set_state(self._rc.state + 1)
        else:
            self._rc.todo = None

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self._rc.todo}")
        todo = self._rc.todo
        msg = self._rc.memory.get()[-1]

        if isinstance(todo, SimpleWriteCode):
            instruction = msg.content
            result = await SimpleWriteCode().run(instruction)

        elif isinstance(todo, SimpleRunCode):
            code_text = msg.content
            result = await SimpleRunCode().run(code_text)

        msg = Message(content=result, role=self.profile, cause_by=todo)
        self._rc.memory.add(msg)
        return msg

    async def _react(self) -> Message:
        while True:
            await self._think()
            if self._rc.todo is None:
                break
            await self._act()
        return Message(content="All job done", role=self.profile)

```

这段代码定义了一个名为`main`的函数，它接受一个字符串参数`msg`，其含义是“编写一个函数来计算一个列表的和”。

函数内部，首先创建了一个名为`SimpleCoder`的类实例，并创建了一个名为`RunnableCoder`的类实例。然后，使用`logger.info`函数输出一条日志信息，其中`msg`参数被设置为字符串`"write a function that calculates the sum of a list"`。

接着，定义一个名为`asyncio.run`的函数，该函数使用`asyncio`库的`run`方法，用于运行代码块并获取返回值。在代码块中，我们使用`role.run`方法来运行刚才创建的`RunnableCoder`实例，其中`msg`参数被设置为字符串`"write a function that calculates the sum of a list"`，作为参数传递给`run`方法。

最后，在`main`函数中，我们创建了一个`fire.Fire`实例，并将`asyncio.run`实例作为参数传递给`fire.Fire`的`main`函数。这样，当运行程序时，`asyncio.run`函数会运行`main`函数中的代码块，输出一条日志信息，表示程序成功地将字符串`"write a function that calculates the sum of a list"`传递给了`asyncio.run`函数，并获取到了它的返回值。


```py
def main(msg="write a function that calculates the sum of a list"):
    # role = SimpleCoder()
    role = RunnableCoder()
    logger.info(msg)
    result = asyncio.run(role.run(msg))
    logger.info(result)

if __name__ == '__main__':
    fire.Fire(main)

```

# `examples/debate.py`

这段代码是一个Python脚本，它使用asyncio、platform、fire和metagpt库来实现一个辩论系统。在这个系统中，用户可以创建辩论题目、开始辩论和查看其他参与者的得分。

具体来说，这段代码定义了一个Debate类，它包含以下方法：

- `__init__`：初始化函数，用于设置辩论题目的内容和格式，以及初始化其他对象的变量。
- `debate_page`：辩论页面的元数据，包括论题、双方角色和得分的格式。
- `boss_requirement`：老板需求，用于定义辩论题目所要求的资源和条件。
- `action`：动作，用于定义辩论过程中的行动。
- `role`：角色，用于定义辩论中各方的角色。
- `message_schema`：消息格式，用于定义辩论消息的格式。
- `logger`：日志记录器，用于记录辩论过程中发生的事情。
- `run`：辩论开始函数，用于启动辩论，并等待用户的行动。
- `on_action_response`：在动作响应事件中执行的函数，用于在收到一个复杂的反应时处理它的内容。
- `on_message`：在消息接收中执行的函数，用于处理到达的消息。

这段代码的主要目的是创建一个辩论系统，使用户可以创建新的辩论题目，开始辩论并对其他参与者的得分进行评价。


```py
'''
Filename: MetaGPT/examples/debate.py
Created Date: Tuesday, September 19th 2023, 6:52:25 pm
Author: garylin2099
'''
import asyncio
import platform
import fire

from metagpt.software_company import SoftwareCompany
from metagpt.actions import Action, BossRequirement
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.logs import logger

```

这段代码定义了一个名为 "ShoutOut" 的类，该类实现了 "Action" 接口。

在这个类的实现中，首先定义了一个名为 "PROMPT_TEMPLATE" 的字符串变量。这个模板用于生成一段关于当前debate主题的 prompt，可以在debate中使用。

然后，定义了一个名为 "run" 的方法，这个方法接收三个参数：上下文(context)、论点(position)和对手的名字(opponent_name)。这个方法首先创建一个 PROMPT_TEMPLATE 中的 prompt，然后使用 "断线下来的方式"(asyncio.coroutine)调用一个名为 "持会者" 的类中的 "讯号" 方法，将 prompt 发送到上下文中的 " Martin" 对象。

最后，在 "run" 方法的回调中，使用 "持有者" 类中的 "信息" 方法打印生成的 prompt。

总结一下，这段代码定义了一个可以生成debate中 prompt的对象，该对象可以用于发送一个持有者的debate。


```py
class ShoutOut(Action):
    """Action: Shout out loudly in a debate (quarrel)"""

    PROMPT_TEMPLATE = """
    ## BACKGROUND
    Suppose you are {name}, you are in a debate with {opponent_name}.
    ## DEBATE HISTORY
    Previous rounds:
    {context}
    ## YOUR TURN
    Now it's your turn, you should closely respond to your opponent's latest argument, state your position, defend your arguments, and attack your opponent's arguments,
    craft a strong and emotional response in 80 words, in {name}'s rhetoric and viewpoints, your will argue:
    """

    def __init__(self, name="ShoutOut", context=None, llm=None):
        super().__init__(name, context, llm)

    async def run(self, context: str, name: str, opponent_name: str):

        prompt = self.PROMPT_TEMPLATE.format(context=context, name=name, opponent_name=opponent_name)
        # logger.info(prompt)

        rsp = await self._aask(prompt)

        return rsp

```

这段代码定义了一个名为Trump的角色类，继承自Role类。这个角色类包含了一个初始化方法(__init__)，一个观察方法(_observe)，和一个执行方法(_act)。

在初始化方法(__init__)中，传递了两个参数name和profile，分别表示这个角色的名字和个人资料。还传递了两个未命名的参数kwargs，可能是用来进行其他设置。

在观察方法(_observe)中，重写了父类的方法_observe，这个方法会接收来自对手的消息，并返回消息的数量。

在执行方法(_act)中，打印了一些日志信息，表示这个角色已经准备好了。然后，使用self._rc.memory.get_by_actions([ShoutOut])获取了所有已经发送的消息，并把它们存储在一个context列表中。最后，使用ShoutOut.run方法来发送消息，并返回结果。消息的内容、角色的设置、发送消息的角色和发送消息的原因都包含在消息对象Message中。


```py
class Trump(Role):
    def __init__(
        self,
        name: str = "Trump",
        profile: str = "Republican",
        **kwargs,
    ):
        super().__init__(name, profile, **kwargs)
        self._init_actions([ShoutOut])
        self._watch([ShoutOut])
        self.name = "Trump"
        self.opponent_name = "Biden"

    async def _observe(self) -> int:
        await super()._observe()
        # accept messages sent (from opponent) to self, disregard own messages from the last round
        self._rc.news = [msg for msg in self._rc.news if msg.send_to == self.name]  
        return len(self._rc.news)

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self._rc.todo}")

        msg_history = self._rc.memory.get_by_actions([ShoutOut])
        context = []
        for m in msg_history:
            context.append(str(m))
        context = "\n".join(context)

        rsp = await ShoutOut().run(context=context, name=self.name, opponent_name=self.opponent_name)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=ShoutOut,
            sent_from=self.name,
            send_to=self.opponent_name,
        )

        return msg

```

这段代码定义了一个名为Biden的类，继承自Role类，用于模拟虚拟人类的对话。在这个类中，作者定义了一个初始化方法(__init__)，该方法接受一个名字参数，一个profile参数和一个**kwargs的参数。初始化方法首先调用父类的初始化方法，然后创建一个消息观察器(__observe)和一个消息处理器(__act)分别从消息历史和记忆中获取消息。接着，作者创建了一个设置类属性，如name和profile，然后覆盖在__init__中**kwargs的子方法。

在类中，作者定义了一个名为_observe的方法(__observe)，该方法接受一个变换函数并返回一个观察消息后剩余的消息数量。在_observe方法中，作者首先调用父类的_observe方法，然后创建一个空列表来存储消息。最后，将获取到的消息添加到存储的消息列表中并返回消息的数量。

接着，作者定义了一个名为_act的消息处理器(__act)，该方法在处理消息之前先输出当前正在做什么。然后创建一个空列表来存储消息，将消息的列表和消息处理程序的参数一起添加到消息列表中，并调用ShoutOut的消息处理函数来处理消息。最后，将消息列表中的所有消息返回给调用者。

总结起来，这段代码定义了一个虚拟人类，可以模拟接受辩论题目，也可以发送简单的问候，对话体验流畅。


```py
class Biden(Role):
    def __init__(
        self,
        name: str = "Biden",
        profile: str = "Democrat",
        **kwargs,
    ):
        super().__init__(name, profile, **kwargs)
        self._init_actions([ShoutOut])
        self._watch([BossRequirement, ShoutOut])
        self.name = "Biden"
        self.opponent_name = "Trump"

    async def _observe(self) -> int:
        await super()._observe()
        # accept the very first human instruction (the debate topic) or messages sent (from opponent) to self,
        # disregard own messages from the last round
        self._rc.news = [msg for msg in self._rc.news if msg.cause_by == BossRequirement or msg.send_to == self.name]
        return len(self._rc.news)

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self._rc.todo}")

        msg_history = self._rc.memory.get_by_actions([BossRequirement, ShoutOut])
        context = []
        for m in msg_history:
            context.append(str(m))
        context = "\n".join(context)

        rsp = await ShoutOut().run(context=context, name=self.name, opponent_name=self.opponent_name)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=ShoutOut,
            sent_from=self.name,
            send_to=self.opponent_name,
        )

        return msg

```

这段代码定义了一个名为`startup`的函数，它接受四个参数：`idea`、`investment`、`n_round`和`code_review`，并返回一个函数`main`。

函数`main`接受两个参数：`idea`和`investment`，它们用于启动一场辩论，并返回辩论的轮数。

函数`startup`的作用是为`SoftwareCompany`对象做初始化，并为该公司添加两位总统。然后，它调用`run`方法来运行公司的程序，并传递给程序的参数包括`idea`、`investment`和`n_round`。最后，它使用`run`方法来运行程序，并等待程序执行完。

函数`startup`的作用是为`SoftwareCompany`对象做初始化，并为该公司添加两位总统。然后，它调用`run`方法来运行公司的程序，并传递给程序的参数包括`idea`、`investment`和`n_round`。最后，它使用`run`方法来运行程序，并等待程序执行完。


```py
async def startup(idea: str, investment: float = 3.0, n_round: int = 5,
                  code_review: bool = False, run_tests: bool = False):
    """We reuse the startup paradigm for roles to interact with each other.
    Now we run a startup of presidents and watch they quarrel. :) """
    company = SoftwareCompany()
    company.hire([Biden(), Trump()])
    company.invest(investment)
    company.start_project(idea)
    await company.run(n_round=n_round)


def main(idea: str, investment: float = 3.0, n_round: int = 10):
    """
    :param idea: Debate topic, such as "Topic: The U.S. should commit more in climate change fighting" 
                 or "Trump: Climate change is a hoax"
    :param investment: contribute a certain dollar amount to watch the debate
    :param n_round: maximum rounds of the debate
    :return:
    """
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(startup(idea, investment, n_round))


```

这段代码是一个Python程序中的一个if语句，它的作用是在程序运行时判断程序是否作为主程序运行。如果程序作为主程序运行，那么程序会执行动器中的 `main` 函数。

具体来说，`if __name__ == '__main__':` 是一个特殊的环境标识，只在程序作为主程序运行时才有意义。如果程序作为子程序或交互式程序运行，则程序将跳过此行代码，并继续按照之前的逻辑执行。

如果`if __name__ == '__main__':` 语句为真，那么代码块中的代码会被执行。这个代码块中的 `fire.Fire(main)` 将调用定义在 `fire` 模块中的 `Fire` 函数，并将 `main` 参数传递给它，从而调用 `main` 函数并传递给它的参数将被打印出来。


```py
if __name__ == '__main__':
    fire.Fire(main)

```

# `examples/invoice_ocr.py`

这段代码是一个Python脚本，名为`invoice_ocr.py`。它使用`asyncio`库作为标准库，使用`typescript`库作为第三方依赖项。以下是对脚本的功能和用途的详细解释：

```pypython
# _*_ coding: utf-8 _*_
```

这是一个shebang line，表示该脚本使用UTF-8编码，并具有Python源代码压缩标志。它定义了一个`invoice_ocr_assistant`类，该类实现了`metagpt.roles.invoice_ocr_assistant`接口，该接口定义了OCR助理的角色。

```pypython
import asyncio
from pathlib import Path

from metagpt.roles.invoice_ocr_assistant import InvoiceOCRAssistant
from metagpt.schema import Message
```

这是脚本中导入的第三方依赖项，包括`asyncio`库、`typescript`库和`metagpt.roles.invoice_ocr_assistant`类。

```pypython
...
```

这是一个空白行，用于提供元数据。

```pypython
#!/usr/bin/env python3
```

这是一个 shebang line，表示该脚本使用类条件语句，并使用`/usr/bin/env python3`路径来执行脚本。

```pypython
...
```

这是一个空白行，用于提供元数据。

```pypython
from metagpt.roles.invoice_ocr_assistant import InvoiceOCRAssistant
from metagpt.schema import Message
```

这是脚本中导入的第三方依赖项，包括`metagpt.roles.invoice_ocr_assistant`类和`metagpt.schema`类，该类定义了`Message`类。

```pypython
...
```

这是一个空白行，用于提供元数据。

```pypython
# 在这里，脚本具体的实现将开始
```

这是一个空白行，用于提供元数据。

```pypython
# _ invite users to test invoice_ocr_assistant
# 在这里，脚本具体的实现将开始
```

这是一个空白行，用于提供元数据。

```pypython
import asyncio
from typing import Any, Text, Dict

# 在这里，定义了一些变量
```

这是一个空白行，用于提供元数据。

```pypython
def main(ctx: Any = None) -> Any:
   # 在这里，脚本具体的实现将开始
```

这是一个空白行，用于提供元数据。

```pypython
# 如果当前目录下的邀請文檔文件存在，则执行它们
# 如果不存在，则创建它们

# 将invoice_ocr_assistant实例设置为全局变量
invoice_ocr_assistant = InvoiceOCRAssistant()

# 将当前文件中的所有消息存储为映射
messages = {}

# 循环处理所有来信
for message in asyncio.get_event_loop().invoke_with(message):
   # 解析消息
   body: Dict[Text, Any] = await message.parse_body()
   # 获取消息类型
   kind: str = body['type']
   # 提取消息数据
   data: Any = body['data']
   # 根据消息类型将消息添加到消息映射中
   if kind == 'invoice_recognized':
       # 在这里，处理invoice_recognized类型的消息
       asyncio.get_event_loop().run_until_complete(
           invoice_ocr_assistant.recognize_invoice(data))
   elif kind == 'invoice_not_recognized':
       # 在这里，处理invoice_not_recognized类型的消息
       asyncio.get_event_loop().run_until_complete(
           invoice_ocr_assistant.notify_result(data))
   else:
       # 在这里，处理所有消息
       messages[body['id']] = {
           'text': body['text'],
           'type': kind,
           'data': data
       }
```

这是脚本的主要实现部分。它首先定义了一个`main`函数，该函数接受一个上下文对象`ctx`（在某些情况下可能没有意义）。

```pypython
...
```

这是一个空白行，用于提供元数据。

```pypython
# 在这里，定义了一些变量
```

这是一个空白行，用于提供元数据。

```pypython
def main(ctx: Any = None) -> Any:
   # 在这里，脚本具体的实现将开始
```

这是一个空白行，用于提供元数据。

```pypython
# 如果当前目录下的邀請文檔文件存在，则执行它们
# 如果不存在，则创建它们

# 将invoice_ocr_assistant实例设置为全局变量
invoice_ocr_assistant = InvoiceOCRAssistant()

# 将当前文件中的所有消息存储为映射
messages = {}

# 循环处理所有来信
for message in asyncio.get_event_loop().invoke_with(message):
   # 解析消息
   body: Dict[Text, Any] = await message.parse_body()
   # 获取消息类型
   kind: str = body['type']
   # 提取消息数据
   data: Any = body['data']
   # 根据消息类型将消息添加到消息映射中
   if kind == 'invoice_recognized':
       # 在这里，处理invoice_recognized类型的消息
       asyncio.get_event_loop().run_until_complete(
           invoice_ocr_assistant.recognize_invoice(data))
   elif kind == 'invoice_not_recognized':
       # 在这里，处理invoice_not_recognized类型的消息
       asyncio.get_event_loop().run_until_complete(
           invoice_ocr_assistant.notify_result(data))
   else:
       # 在这里，处理所有消息
       messages[body['id']] = {
           'text': body['text'],
           'type': kind,
           'data': data
       }
```

这是脚本的主要实现部分。它首先定义了一个`main`函数，该函数接受一个上下文对象`ctx`（在某些情况下可能没有意义）。

```pypython
...
```

这是一个空白行，用于提供元数据。

```pypython
# 在这里，定义了一些变量
```

这是一个空白行，用于提供元数据。

```pypython
def main(ctx: Any = None) -> Any:
   # 在这里，脚本具体的实现将开始
```

这是一个空白行，用于提供元数据。

```pypython
# 如果当前目录下的邀請文檔文件存在，则执行它们
# 如果不存在，则创建它们

# 将invoice_ocr_assistant实例设置为全局变量
invoice_ocr_assistant = InvoiceOCRAssistant()

# 将当前文件中的所有消息存储为映射
messages = {}

# 循环处理所有来信
for message in asyncio.get_event_loop().invoke_with(message):
   # 解析消息
   body: Dict[Text, Any] = await message.parse_body()
   # 获取消息类型
   kind: str = body['type']
   # 提取消息数据
   data: Any = body['data']
   # 根据消息类型将消息添加到消息映射中
   if kind == 'invoice_recognized':
       # 在这里，处理invoice_recognized类型的消息
       asyncio.get_event_loop().run_until_complete(
           invoice_ocr_assistant.recognize_invoice(data))
   elif kind == 'invoice_not_recognized':
       # 在这里，处理invoice_not_recognized类型的消息
       asyncio.get_event_loop().run_until_complete(
           invoice_ocr_assistant.notify_result(data))
   else:
       # 在这里，处理所有消息
       messages[body


```
#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
@Time    : 2023/9/21 21:40:57
@Author  : Stitch-z
@File    : invoice_ocr.py
"""

import asyncio
from pathlib import Path

from metagpt.roles.invoice_ocr_assistant import InvoiceOCRAssistant
from metagpt.schema import Message


```py

这段代码定义了一个名为 `main` 的函数，它使用了 Python 的 `asyncio` 库来实现异步操作。函数的作用是在一个程序中处理一系列测试数据的文件。

具体来说，这个程序会在当前工作目录下遍历一系列测试数据的文件，这些文件都是相对于当前工作目录的绝对路径。程序会将每个测试数据的文件路径存储在一个列表中，然后使用 `InvoiceOCRAssistant` 类来运行 `run` 方法，该方法的接收者是一个字符串类型的参数，表示要运行的消息的内容。这个消息将包含一个 `FilePath` 对象，它将调用程序中定义的 `relative_paths` 变量中的任意一个文件路径，并将该路径作为参数传递给 `run` 方法中。

由于使用了 `asyncio` 库，程序中的 `for` 循环可以使用 `await` 关键字来暂停执行，让程序在循环中的每个文件路径都能够正确运行 `run` 方法。


```
async def main():
    relative_paths = [
        Path("../tests/data/invoices/invoice-1.pdf"),
        Path("../tests/data/invoices/invoice-2.png"),
        Path("../tests/data/invoices/invoice-3.jpg"),
        Path("../tests/data/invoices/invoice-4.zip")
    ]
    # The absolute path of the file
    absolute_file_paths = [Path.cwd() / path for path in relative_paths]

    for path in absolute_file_paths:
        role = InvoiceOCRAssistant()
        await role.run(Message(
            content="Invoicing date",
            instruct_content={"file_path": path}
        ))


```py

这段代码使用了Python的异步编程库asyncio来 run一个名为“main”的函数。如果当前脚本目录（通常为“__main__”或“console”）是Python内置的“__main__”脚本，那么脚本会直接在此处运行。否则，它将运行“main”函数，该函数可能是定义在其他文件中的。在这个例子中，“main”函数未定义，因此脚本会报错。

具体来说，这段代码的作用是：如果当前脚本目录是Python内置的“__main__”脚本，那么运行“main”函数；否则，运行“main”函数并输出错误信息。


```
if __name__ == '__main__':
    asyncio.run(main())


```py

# `examples/llm_hello_world.py`

这段代码是一个Python脚本，运行在命令行#!/usr/bin/env python环境下。

它的主要作用是演示如何使用LLM(LLM相对应的API)进行自然语言处理。LLM是一个基于Python的LLM模型，可以用于构建自然语言处理应用程序，如问答系统等。

具体来说，这段代码实现了一个LLM实例化器(LLM)、一个Claude实例和一个日志记录器(Logger)。它使用这些组件来与用户进行交互，并输出一些信息。

具体实现步骤如下：

1. 导入所需的模块。
2. 使用LLM实例化器来创建一个LLM实例。
3. 使用Claude实例来获取LLM实例的输出。
4. 使用日志记录器来输出信息。
5. 使用LLM实例化器的方法来完成一些任务，如向用户提问并获取答案、输出自然语言处理结果等。
6. 输出一些结果，如用户的角色、内容等，并使用ACOMPletion和ACOMPletionBatchText和ACOMPletionText等方法。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/6 14:13
@Author  : alexanderwu
@File    : llm_hello_world.py
"""
import asyncio

from metagpt.llm import LLM, Claude
from metagpt.logs import logger


async def main():
    llm = LLM()
    claude = Claude()
    logger.info(await claude.aask('你好，请进行自我介绍'))
    logger.info(await llm.aask('hello world'))
    logger.info(await llm.aask_batch(['hi', 'write python hello world.']))

    hello_msg = [{'role': 'user', 'content': 'count from 1 to 10. split by newline.'}]
    logger.info(await llm.acompletion(hello_msg))
    logger.info(await llm.acompletion_batch([hello_msg]))
    logger.info(await llm.acompletion_batch_text([hello_msg]))

    logger.info(await llm.acompletion_text(hello_msg))
    await llm.acompletion_text(hello_msg, stream=True)


```py

这段代码使用了Python的异步编程库——asyncio。`__name__`是一个Python的保留字，用于防止Python脚本被意外的模块加载。

if __name__ == '__main__':
```是Python 3中的一个特殊语句，表示只有在脚本作为主程序运行时才会执行。

asyncio.run(main())
```py是运行主程序的函数，`asyncio.run`是asyncio库中的函数，用于运行异步函数。它接受一个函数作为参数，这个函数将会被提交给事件循环(事件循环即内存中的一个迭代器)来执行。在这个例子中，传递给它的函数是`main`，因此这个函数会被事件循环执行。


```
if __name__ == '__main__':
    asyncio.run(main())

```py

# `examples/research.py`

这段代码是一个Python脚本，它使用asyncio库作为事件循环。主要目的是执行名为“main”的函数。

1. 导入必要的库：
```python
import asyncio
from metagpt.roles.researcher import RESEARCH_PATH, Researcher
```py
这些库用于数据科学领域，提供了对主题和报告的搜索功能。

1. 定义名为“main”的函数：
```python
async def main():
```py
这个函数将会定义一个异步函数，也就是一个将来时函数，它会在异步环境中运行。

1. 导入元数据：
```python
   topic = "dataiku vs. datarobot"
```py
定义了一个名为“topic”的变量，它存储了要搜索的主题。

1. 创建一个“ Researcher”实例：
```python
   role = Researcher(language="en-us")
```py
将名为“researcher”的函数作为参数传入，并将其命名为“role”。

1. 使用异步上下文管理器（asyncio）：
```python
   async for root in ENV:
       yield root
```py
这是一个异步上下文管理器，用于在给定的环境中异步地迭代搜索根目录。

1. 调用“run”函数，并获取搜索结果：
```python
   await role.run(topic)
```py
将“topic”作为参数传递给“run”函数，并使用“async”关键字覆盖“run”函数的返回值。这将异步执行“run”函数，并等待其完成。

1. 打印结果：
```python
   print(f"save report to {RESEARCH_PATH / f'{topic}.md'}.")
```py
这段代码将会打印结果，并使用字符串格式化将结果中的“{RESEARCH_PATH / f'{topic}.md'}”格式化。

1. 使用Python内置的“run”函数来运行“main”函数：
```python
if __name__ == '__main__':
   asyncio.run(main())
```py
这段代码将会执行“main”函数，并在其完成后退出Python。


```
#!/usr/bin/env python

import asyncio

from metagpt.roles.researcher import RESEARCH_PATH, Researcher


async def main():
    topic = "dataiku vs. datarobot"
    role = Researcher(language="en-us")
    await role.run(topic)
    print(f"save report to {RESEARCH_PATH / f'{topic}.md'}.")


if __name__ == '__main__':
    asyncio.run(main())

```py

# `examples/search_google.py`

该代码是一个Python脚本，名为`search_google.py`。它在运行时使用`asyncio`库的`run`方法来运行一个任务。

具体来说，该脚本实现了一个搜索Google的建议引擎（Google Search Engine）的程序。您需要使用`-*- coding: utf-8 -*-`这行注释来指定输入编码为UTF-8。

接下来，该脚本导入了`asyncio`和`metagpt.roles`库。然后定义了一个名为`main`的函数，它使用`Searcher`类来搜索Google，并将其返回结果存储在变量`res`中。

`Searcher`类是一个`metagpt.roles`库中的类，它实现了一个用于搜索API的接口。在这里，我们可能需要根据需要进行一些更改，以使其符合您的特定需求。

最后，该脚本使用`asyncio.run`方法运行`main`函数，将其放在一个名为`asyncio_main`协程中。这个协程会等待`main`函数返回结果，并输出结果。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/7 18:32
@Author  : alexanderwu
@File    : search_google.py
"""

import asyncio

from metagpt.roles import Searcher


async def main():
    await Searcher().run("What are some good sun protection products?")


```py

这段代码使用了Python中的异步编程库——asyncio。`__name__`是一个Python特性，用于定义一个模块是否作为脚本文件运行。如果`__name__`等于'__main__'，那么这段代码将会被当作一个独立的Python脚本文件运行，而不是作为Python内置模块的函数或类来执行。

`asyncio.run`是一个来自asyncio库的函数，用于运行一个事件循环（或称为"协程"）并返回一个 Future 对象。在这个函数中，我们使用 `asyncio.run` 来运行一段代码块，并且可以传递一个函数作为参数，这个函数将会被运行。这段代码将会阻塞当前线程，直到指定的代码块完成为止，然后将结果返回给调用者。

如果这段代码中的 `main` 函数包含任何异步代码，那么 `asyncio.run` 将始终返回一个 Future 对象，因为这些代码块是在异步环境下执行的。这些 Future 对象可以用于调用 `asyncio.sleep`、`asyncio.wait` 或者 `asyncio. Instrumentation` 函数等Asyncio库中的函数。


```
if __name__ == '__main__':
    asyncio.run(main())

```py

# `examples/search_kb.py`

这段代码是一个Python脚本，它使用asyncio库作为事件循环。脚本的功能是搜索关于面部清洁剂的问答。

具体来说，它首先导入metagpt.const、metagpt.document_store和metagpt.roles模块，然后定义了一个名为search的函数。

search函数使用metagpt.document_store中的FaissStore类将面部清洁剂数据存储在文件系统中的某个路径。接下来，它使用metagpt.roles中的Sales类执行一个查询，即询问用户关于面部清洁剂的意见。然后，它将查询结果输出到控制台。

搜索函数内部使用asyncio库的run方法来执行查询。这个方法将异步操作转换为同步操作，因此可以确保查询结果在输出之前已经被计算出来了。

最后，search函数会在程序启动时一直运行，直到被手动停止。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : search_kb.py
"""
import asyncio

from metagpt.const import DATA_PATH
from metagpt.document_store import FaissStore
from metagpt.logs import logger
from metagpt.roles import Sales


async def search():
    store = FaissStore(DATA_PATH / 'example.json')
    role = Sales(profile="Sales", store=store)

    queries = ["Which facial cleanser is good for oily skin?", "Is L'Oreal good to use?"]
    for query in queries:
        logger.info(f"User: {query}")
        result = await role.run(query)
        logger.info(result)


```py

这段代码使用了Python的异步编程库——asyncio。它通过调用asyncio.run()函数来运行一个搜索操作。具体来说，该函数将在事件循环中异步执行搜索操作，并将结果打印到控制台。

在这段代码中，首先定义了一个名为"__name__"的属性，它用于保存当前脚本是否作为主程序运行。如果该属性为'__main__'，则说明该脚本作为主程序运行。在这种情况下，代码将使用asyncio.run()函数来运行搜索操作。

asyncio.run()函数是asyncio库中的一个函数，它接受一个函数作为参数，并将该函数在事件循环中异步执行。通过调用该函数，我们可以将搜索操作作为一个函数传递给它，并在搜索操作完成后将结果打印到控制台。

因此，这段代码的作用是运行一个搜索操作，并在搜索操作完成后将结果打印到控制台。


```
if __name__ == '__main__':
    asyncio.run(search())

```py

# `examples/search_with_specific_engine.py`

这段代码使用了Python的asyncio库来实现异步编程。asyncio是一个用于编写异步代码的标准库，它提供了许多异步编程的工具和函数。

具体来说，这段代码的作用是使用多个异步搜索引擎来获取有关不同主题的信息。具体来说，它使用Metagpt的角色(Searcher)和工具(SearchEngineType)来访问Google和SerpApi搜索引擎。在主函数中，它等待这些搜索引擎的运行结果，并将结果打印出来。

例如，对于第一个查询"What are some good sun protection products?"来说，它使用Searcher来访问Google和SerpApi搜索引擎，并让Searcher运行两个参数的查询。"What are some of the best skiing beaches?"。它还使用Searcher来运行另一个查询，这个查询将在未来返回结果。

这段代码还使用了异步函数来确保在等待搜索引擎结果的时候不会阻塞程序的运行。这个函数使用了异步编程的一般原则，即使用await关键字来等待异步操作的结果。


```
import asyncio

from metagpt.roles import Searcher
from metagpt.tools import SearchEngineType


async def main():
    # Serper API
    #await Searcher(engine = SearchEngineType.SERPER_GOOGLE).run(["What are some good sun protection products?","What are some of the best beaches?"])
    # SerpAPI
    #await Searcher(engine=SearchEngineType.SERPAPI_GOOGLE).run("What are the best ski brands for skiers?")
    # Google API
    await Searcher(engine=SearchEngineType.DIRECT_GOOGLE).run("What are the most interesting human facts?")

if __name__ == '__main__':
    asyncio.run(main())

```py

# `examples/sk_agent.py`

这段代码定义了一个Python脚本，名为`sk_agent.py`，使用了`asyncio`库进行异步编程，时间单位为秒。

该脚本导入了三个嵌套的`FileIOSkill`、`MathSkill`、`TextSkill`和`TimeSkill`，表明它可能与操作系统技能、数学技能、文本技能和时间技能有关。

接下来，该脚本使用这些技能定义了一个`SequentialPlanner`实例，一个`ActionPlanner`实例，并创建了一个名为`sk_agent`的类。

具体来说，这个脚本定义了一个`sk_agent.sk_agent_类`，其中包含以下方法：

* `FileIOSkill`：用于在文件系统中执行操作，例如打开、读取和关闭文件。
* `MathSkill`：用于数学计算，例如加法、减法、乘法和除法。
* `TextSkill`：用于处理文本，例如将文本转换为小写、大写或首字母大写，以及删除指定文本的前缀和后缀。
* `TimeSkill`：用于处理时间，例如获取当前时间、格式化时间和执行定时任务。
* `SequentialPlanner`：用于定义一系列任务，例如使用`FileIOSkill`和`TextSkill`在文件系统中读取和写入数据，以及使用`MathSkill`执行基本的算术计算。
* `ActionPlanner`：用于定义一系列动作，例如打开应用程序、关闭应用程序或发送电子邮件。
* `sk_agent_main`：作为`sk_agent`类的静态方法，用于启动命令行脚本并加载已知技能。

总之，这段代码定义了一个用于操作系统技能、数学技能、文本技能和时间技能的脚本，它创建了一个`sk_agent`类，用于定义和执行各种技能。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/13 12:36
@Author  : femto Zheng
@File    : sk_agent.py
"""
import asyncio

from semantic_kernel.core_skills import FileIOSkill, MathSkill, TextSkill, TimeSkill
from semantic_kernel.planning import SequentialPlanner

# from semantic_kernel.planning import SequentialPlanner
from semantic_kernel.planning.action_planner.action_planner import ActionPlanner

```py

这段代码的作用是使用 Metagpt 工具包中的几个函数和类，实现一个基本的机器人任务规划。

具体来说，它包括以下操作：

1. 从 metagpt.actions 包中导入 BossRequirement 类，定义了机器人需求。
2. 从 metagpt.const 包中导入 SKILL_DIRECTORY，定义了技能目录。
3. 从 metagpt.roles.sk_agent 包中导入 SkAgent 类，定义了机器人角色。
4. 从 metagpt.schema 包中导入 Message 类，定义了消息类型。
5. 从 metagpt.tools.search_engine 包中导入 SkSearchEngine 类，定义了搜索引擎。
6. 使用 main() 函数作为程序的入口点。
7. 在 main() 函数中，使用 await 关键字，等待基本规划器示例、动作规划器示例和序列规划器示例的结果。
8. 使用 asyncio 库中的异步函数，实现并发执行。


```
from metagpt.actions import BossRequirement
from metagpt.const import SKILL_DIRECTORY
from metagpt.roles.sk_agent import SkAgent
from metagpt.schema import Message
from metagpt.tools.search_engine import SkSearchEngine


async def main():
    # await basic_planner_example()
    # await action_planner_example()

    # await sequential_planner_example()
    await basic_planner_web_search_example()


```py

这段代码定义了两个异步函数：`basic_planner_example()` 和 `sequential_planner_example()`。这两个函数都是使用 `SKILL_DIRECTORY` 中定义的技能，使用 `SequentialPlanner` 类来生成连续的日期建议。

具体来说，`basic_planner_example()` 在作用域中创建了一个任务，该任务包含一个将文本转换为大写文本的请求和一个将技能导入到 `SkAgent` 的函数。然后，它使用 `run()` 方法将消息发送给 `role`，使它运行 `basic_planner_example()` 函数。

另一方面，`sequential_planner_example()` 复制 `basic_planner_example()` 的行为，但将其中的 `BasicPlanner` 类替换为 `SequentialPlanner` 类。它还使用 `import_skill()` 方法从 `SKILL_DIRECTORY` 中导入 `TextSkill`，并将其命名为 `TextSkill`。


```
async def basic_planner_example():
    task = """
    Tomorrow is Valentine's day. I need to come up with a few date ideas. She speaks French so write it in French.
    Convert the text to uppercase"""
    role = SkAgent()

    # let's give the agent some skills
    role.import_semantic_skill_from_directory(SKILL_DIRECTORY, "SummarizeSkill")
    role.import_semantic_skill_from_directory(SKILL_DIRECTORY, "WriterSkill")
    role.import_skill(TextSkill(), "TextSkill")
    # using BasicPlanner
    await role.run(Message(content=task, cause_by=BossRequirement))


async def sequential_planner_example():
    task = """
    Tomorrow is Valentine's day. I need to come up with a few date ideas. She speaks French so write it in French.
    Convert the text to uppercase"""
    role = SkAgent(planner_cls=SequentialPlanner)

    # let's give the agent some skills
    role.import_semantic_skill_from_directory(SKILL_DIRECTORY, "SummarizeSkill")
    role.import_semantic_skill_from_directory(SKILL_DIRECTORY, "WriterSkill")
    role.import_skill(TextSkill(), "TextSkill")
    # using BasicPlanner
    await role.run(Message(content=task, cause_by=BossRequirement))


```py

这两段代码是在使用Python的asyncio库实现的。第一个函数是一个异步函数，它使用BasicPlannerWebSearchExample类来执行一个任务。第二个函数是一个异步函数，它使用ActionPlanner类来执行一个任务。这两个函数都是使用 SkillfulAI代理执行的。

具体来说，第一个函数basic_planner_web_search_example()中，通过实现了SkSearchEngine和QASkill接口，让AI代理可以执行搜索任务。在这个函数中，我们通过代理的run()方法提交了一个任务，这个任务是一个字符串类型的参数，包含了一个问题，通过使用Role.import_skill()方法，将问题相关的技能导入代理中，最后通过代理的run()方法提交了这个问题。

第二个函数 action_planner_example()中，我们创建了一个ActionPlanner类，并让它的任务执行者SkAgent中，导入了一些技能，如数学技能、文件IO技能、时间技能和文本技能。然后，我们通过ActionPlanner的run()方法提交了一个任务，这个任务是一个字符串类型的参数，包含了一个问题，通过使用Role.import_skill()方法，将问题相关的技能导入代理中，最后通过代理的run()方法提交了这个问题。


```
async def basic_planner_web_search_example():
    task = """
    Question: Who made the 1989 comic book, the film version of which Jon Raymond Polito appeared in?"""
    role = SkAgent()

    role.import_skill(SkSearchEngine(), "WebSearchSkill")
    # role.import_semantic_skill_from_directory(skills_directory, "QASkill")

    await role.run(Message(content=task, cause_by=BossRequirement))


async def action_planner_example():
    role = SkAgent(planner_cls=ActionPlanner)
    # let's give the agent 4 skills
    role.import_skill(MathSkill(), "math")
    role.import_skill(FileIOSkill(), "fileIO")
    role.import_skill(TimeSkill(), "time")
    role.import_skill(TextSkill(), "text")
    task = "What is the sum of 110 and 990?"
    await role.run(Message(content=task, cause_by=BossRequirement))  # it will choose mathskill.Add


```py

这段代码使用了Python的异步编程库——asyncio来实现一个简单的异步main函数。

"__name__ == "__main__":", 是一个Python的特性，表示如果当前脚本被作为独立的主程序运行，而不是作为模块的导入，那么该代码就是该脚本的主函数，代码中的第一个if语句的条件判断就是在判断该脚本是否作为主函数运行。

"asyncio.run(main())" 是一个使用 asyncio 库的异步main函数的实现。asyncio 是一个用于 Python 3 的异步编程库，而 run 函数则是 asyncio 库中的一个用于运行异步函数的函数。该函数接受一个参数 main 参数，表示该函数运行的是主函数，main 参数就是该函数中要运行的异步函数。

整个代码的作用就是使用 asyncio 库的 run 函数来运行一个异步main函数，该函数中包含 main 函数和异步代码。


```
if __name__ == "__main__":
    asyncio.run(main())

```py

# `examples/use_off_the_shelf_agent.py`

这段代码的作用是编写一个Python程序，用于编写一个产品经理（Product Manager）应用程序，用于在游戏蛇（snake game）中执行自动脚本。具体来说，这个程序将编写一个PRD（产品需求文档，Product Requirements Document），描述游戏中的目标和规则。

具体实现包括：

1. 导入 asyncio：这是Python 3.7引入的异步编程库，用于编写异步程序。
2. 从 metagpt.roles.product_manager 导入 ProductManager：这是一个用于管理游戏角色（Product）的类，可能包含与游戏蛇游戏相关的功能。
3. 从 metagpt.logs import logger：这是用于输出信息到Logger的类，用于在日志中记录信息。
4. 定义一个名为 main 的函数：这个函数将作为程序的入口点，负责启动程序并执行 main 函数内包含的代码。
5. 在 main 函数内编写一个字符串 msg，表示要编写的 PRD 内容。
6. 使用ProductManager的 run 方法运行 msg，得到结果并将其存储在变量 result 中。
7. 使用 logger.info 方法将 result 的内容输出到控制台，以便在日志中查看。


```
'''
Filename: MetaGPT/examples/use_off_the_shelf_agent.py
Created Date: Tuesday, September 19th 2023, 6:52:25 pm
Author: garylin2099
'''
import asyncio

from metagpt.roles.product_manager import ProductManager
from metagpt.logs import logger

async def main():
    msg = "Write a PRD for a snake game"
    role = ProductManager()
    result = await role.run(msg)
    logger.info(result.content[:100])

```py

这段代码使用了Python的异步编程库asyncio来运行一个名为“main”的函数。

当程序运行时，首先检查是否运行了一个名为“__main__”的函数。如果是，则程序将使用异步代码块中的函数来运行该函数。

异步代码块中的函数使用asyncio.run()方法来运行一个异步任务。这个函数接受一个参数列表，其中包含函数、装饰器和事件处理程序。在这个例子中，我们没有传递函数本身作为参数，而是传递了函数的名称“main”。

如果“main”函数没有定义任何事件处理程序，则运行程序后，它将一直运行在主协程中，直到被手动停止。


```
if __name__ == '__main__':
    asyncio.run(main())

```py

# `examples/write_tutorial.py`

这段代码是一个Python脚本，它实现了异步函数main。该脚本使用了一个名为TutorialAssistant的类，该类实现了米饭功能，用于向用户提示编写关于MySQL教程。

具体来说，该脚本实现了以下功能：

1. 定义了一个名为main的函数，该函数包含使脚本运行的代码。
2. 导入名为asyncio的库，该库提供了异步编程的框架。
3. 导入名为TutorialAssistant的类，该类继承自米饭（model）函数，实现了米饭的功能。
4. 在main函数中定义了一个名为topic的变量，用于存储要向用户提供的教程主题。
5. 创建一个名为role的实例，并将其初始化为TutorialAssistant，其中参数language设置为"Chinese"，表示使用中文语言向用户提示。
6. 在role实例的run方法中包含一个带有参数topic的参数，用于向用户提示编写关于该主题的教程。
7. 使用asyncio库中的run函数，该函数用于运行指定的代码块，并返回一个变量，该变量表示使脚本运行的协程。
8. 将协程变量role.run（topic）与asyncio库中的submit函数（submit）结合，以便在协程变量使脚本保持运行的同时，执行函数并输出其结果。


```
#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
@Time    : 2023/9/4 21:40:57
@Author  : Stitch-z
@File    : tutorial_assistant.py
"""
import asyncio

from metagpt.roles.tutorial_assistant import TutorialAssistant


async def main():
    topic = "Write a tutorial about MySQL"
    role = TutorialAssistant(language="Chinese")
    await role.run(topic)


```py

这段代码是一个 Python 程序，它使用了 Python 的 asyncio 库来运行一个名为 "main" 的函数。具体来说，这段代码的作用是执行 "main" 函数，并且在程序运行时始终存在，即使程序不使用它也会运行。

asyncio 库是一个用于编写网络应用程序的库，它提供了异步编程的支持。在这个例子中，asyncio.run() 函数是一个异步编程函数，它可以运行一个函数 "main"，并且不阻塞程序的其他部分。这个函数会异步地执行 main 函数，并将结果返回给程序的其他部分。


```
if __name__ == '__main__':
    asyncio.run(main())


```py

# `metagpt/config.py`

这段代码是一个Python脚本，它提供了两个类的实例化，一个是`openai.Assets`类，另一个是`yaml.Loader`类。

`openai.Assets`类是一个自定义的`Assets`类，这个类被用来在训练和应用中加载各种不同的资产（如模型的权重、数据集等）。

`yaml.Loader`类是一个`yaml`库中的`Loader`类，这个类被用来读取和解析YAML格式的配置文件。

接下来的两行代码指定了脚本的解释器为`/usr/bin/env python`，这意味着脚本可以使用Python 3的执行环境。

在接下来的几行中，脚本导入了`os`库，这是Python标准库中的一个用于操作系统功能操作的库。

然后，脚本使用`import openai`导入了`openai`库，这是由OpenAI开发的一个基于API的自动化工具和库，用于构建和训练人工智能模型。

接下来，脚本使用`import yaml`导入了`yaml`库，这个库被用于解析和生成YAML格式的配置文件。

接着，脚本使用`logger`库中的`getLogger`函数创建了一个名为`my_logger`的logger实例，这个logger用于在日志中记录信息。

然后，脚本使用`metagpt.const`库中的`PROJECT_ROOT`常量，这个常量表示当前项目的根目录。

接下来，脚本使用`metagpt.logs`库中的`getLogger`函数创建了一个名为`my_logger`的logger实例，这个logger用于在日志中记录信息。

接着，脚本使用`metagpt.tools.SearchEngineType`和`metagpt.tools.WebBrowserEngineType`类，这些类被用于提供搜索引擎和Web浏览器引擎的选项。

最后，脚本使用`metagpt.utils.Singleton`类创建了一个名为`my_asset_manager`的单例实例，这个单例用于在应用程序中管理`openai.Assets`类的实例。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provide configuration, singleton
"""
import os

import openai
import yaml

from metagpt.const import PROJECT_ROOT
from metagpt.logs import logger
from metagpt.tools import SearchEngineType, WebBrowserEngineType
from metagpt.utils.singleton import Singleton


```py

This is a Python class that appears to be used for configuring a研究领域 using the research question and associated concepts.

It has methods for getting the usage and report of a model for researchers, as well as a method for generating a usage report of a given model.

It also has a method for initializing the class with a configuration file and an optional environment variable for a `mermaid_engine` that is used to render diagrams in the Mermaid format.

The class has a `_init_with_config_files_and_env` method that reads the configuration file (either `config/key.yaml` or `config/config.yaml`) and environment variables from the system, and a `_get` method that is used to look up values in the configuration file or environment variables.

Overall, it appears to be a useful class for researchers who want to use a `mermaid_engine` to render diagrams in their research, and who want to specify the initial configuration of the `mermaid_engine` using environment variables.


```
class NotConfiguredException(Exception):
    """Exception raised for errors in the configuration.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="The required configuration is not set"):
        self.message = message
        super().__init__(self.message)


class Config(metaclass=Singleton):
    """
    Regular usage method:
    config = Config("config.yaml")
    secret_key = config.get_key("MY_SECRET_KEY")
    print("Secret key:", secret_key)
    """

    _instance = None
    key_yaml_file = PROJECT_ROOT / "config/key.yaml"
    default_yaml_file = PROJECT_ROOT / "config/config.yaml"

    def __init__(self, yaml_file=default_yaml_file):
        self._configs = {}
        self._init_with_config_files_and_env(self._configs, yaml_file)
        logger.info("Config loading done.")
        self.global_proxy = self._get("GLOBAL_PROXY")
        self.openai_api_key = self._get("OPENAI_API_KEY")
        self.anthropic_api_key = self._get("Anthropic_API_KEY")
        if (not self.openai_api_key or "YOUR_API_KEY" == self.openai_api_key) and (
                not self.anthropic_api_key or "YOUR_API_KEY" == self.anthropic_api_key
        ):
            raise NotConfiguredException("Set OPENAI_API_KEY or Anthropic_API_KEY first")
        self.openai_api_base = self._get("OPENAI_API_BASE")
        openai_proxy = self._get("OPENAI_PROXY") or self.global_proxy
        if openai_proxy:
            openai.proxy = openai_proxy
            openai.api_base = self.openai_api_base
        self.openai_api_type = self._get("OPENAI_API_TYPE")
        self.openai_api_version = self._get("OPENAI_API_VERSION")
        self.openai_api_rpm = self._get("RPM", 3)
        self.openai_api_model = self._get("OPENAI_API_MODEL", "gpt-4")
        self.max_tokens_rsp = self._get("MAX_TOKENS", 2048)
        self.deployment_name = self._get("DEPLOYMENT_NAME")
        self.deployment_id = self._get("DEPLOYMENT_ID")

        self.spark_appid = self._get("SPARK_APPID")
        self.spark_api_secret = self._get("SPARK_API_SECRET")
        self.spark_api_key = self._get("SPARK_API_KEY")
        self.domain = self._get("DOMAIN")
        self.spark_url = self._get("SPARK_URL")

        self.claude_api_key = self._get("Anthropic_API_KEY")
        self.serpapi_api_key = self._get("SERPAPI_API_KEY")
        self.serper_api_key = self._get("SERPER_API_KEY")
        self.google_api_key = self._get("GOOGLE_API_KEY")
        self.google_cse_id = self._get("GOOGLE_CSE_ID")
        self.search_engine = SearchEngineType(self._get("SEARCH_ENGINE", SearchEngineType.SERPAPI_GOOGLE))
        self.web_browser_engine = WebBrowserEngineType(self._get("WEB_BROWSER_ENGINE", WebBrowserEngineType.PLAYWRIGHT))
        self.playwright_browser_type = self._get("PLAYWRIGHT_BROWSER_TYPE", "chromium")
        self.selenium_browser_type = self._get("SELENIUM_BROWSER_TYPE", "chrome")

        self.long_term_memory = self._get("LONG_TERM_MEMORY", False)
        if self.long_term_memory:
            logger.warning("LONG_TERM_MEMORY is True")
        self.max_budget = self._get("MAX_BUDGET", 10.0)
        self.total_cost = 0.0

        self.puppeteer_config = self._get("PUPPETEER_CONFIG", "")
        self.mmdc = self._get("MMDC", "mmdc")
        self.calc_usage = self._get("CALC_USAGE", True)
        self.model_for_researcher_summary = self._get("MODEL_FOR_RESEARCHER_SUMMARY")
        self.model_for_researcher_report = self._get("MODEL_FOR_RESEARCHER_REPORT")
        self.mermaid_engine = self._get("MERMAID_ENGINE", "nodejs")
        self.pyppeteer_executable_path = self._get("PYPPETEER_EXECUTABLE_PATH", "")

        self.prompt_format = self._get("PROMPT_FORMAT", "markdown")

    def _init_with_config_files_and_env(self, configs: dict, yaml_file):
        """Load from config/key.yaml, config/config.yaml, and env in decreasing order of priority"""
        configs.update(os.environ)

        for _yaml_file in [yaml_file, self.key_yaml_file]:
            if not _yaml_file.exists():
                continue

            # Load local YAML file
            with open(_yaml_file, "r", encoding="utf-8") as file:
                yaml_data = yaml.safe_load(file)
                if not yaml_data:
                    continue
                os.environ.update({k: v for k, v in yaml_data.items() if isinstance(v, str)})
                configs.update(yaml_data)

    def _get(self, *args, **kwargs):
        return self._configs.get(*args, **kwargs)

    def get(self, key, *args, **kwargs):
        """Search for a value in config/key.yaml, config/config.yaml, and env; raise an error if not found"""
        value = self._get(key, *args, **kwargs)
        if value is None:
            raise ValueError(f"Key '{key}' not found in environment variables or in the YAML file")
        return value


```py

这段代码是在创建一个名为 "CONFIG" 的配置对象，可能会用于配置程序的一些设置。

具体来说，CONFIG = Config()将会把一个名为 Config 的类实例化并赋值给对象变量 CONFIG，以便后续的使用。

这个 Config 类可能包含程序的一些设置，例如：

- 设置程序的一些选项，例如是否进行调试输出
- 设置是否在程序启动时加载配置文件
- 设置程序的一些行为，例如在程序启动时做些什么

通过创建 Config 类并将其赋值给 CONFIG，程序就可以使用这个配置对象来管理配置选项和行为。


```
CONFIG = Config()

```py

# `metagpt/const.py`

这段代码是一个Python脚本，主要作用是定义了一个名为`get_project_root`的函数，用于查找项目的根目录。函数实现的过程是：从当前目录开始向上遍历，直到找到项目的根目录或者`.git`、`.project_root`或`.gitignore`文件存在时停止遍历。如果停止遍历后仍然没有找到根目录，函数将异常并输出一条错误消息。

这个脚本的具体作用是，在项目开发过程中，为了保证代码的规范性和可维护性，需要定义一个项目的根目录。通过`get_project_root`函数，可以方便地获取到当前项目的根目录，从而进行项目的深度遍历和某些系统的集成。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/1 11:59
@Author  : alexanderwu
@File    : const.py
"""
from pathlib import Path


def get_project_root():
    """Search upwards to find the project root directory."""
    current_path = Path.cwd()
    while True:
        if (
            (current_path / ".git").exists()
            or (current_path / ".project_root").exists()
            or (current_path / ".gitignore").exists()
        ):
            return current_path
        parent_path = current_path.parent
        if parent_path == current_path:
            raise Exception("Project root not found.")
        current_path = parent_path


```py

这段代码定义了一些项目相关的路径，包括数据文件存储目录、工作区文件存储目录、metagpt提示文件存储目录、swagger文件存储目录等。其中，DATA_PATH 是数据文件存储目录，WORKSPACE_ROOT 是工作区文件存储目录，PROMPT_PATH 是metagpt提示文件存储目录，UT_PATH 是数据文件存储目录中的ut文件夹，SWAGGER_PATH 是swagger文件存储目录，UT_PY_PATH 是ut文件夹中的python文件，API_QUESTIONS_PATH 是数据文件存储目录中的问题文件，YAPI_URL 是来自Yapi.deepwisdomai.com的API接口地址，TMP 是项目根目录下的临时文件夹，RESEARCH_PATH 是数据文件存储目录中的研究文件夹，TUTORIAL_PATH 是数据文件存储目录中的tutorial_docx文件夹，INVOICE_OCR_TABLE_PATH 是数据文件存储目录中的invoice_table文件夹。此外，SKILL_DIRECTORY 是项目根目录下的metagpt技能文件夹。


```
PROJECT_ROOT = get_project_root()
DATA_PATH = PROJECT_ROOT / "data"
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"
PROMPT_PATH = PROJECT_ROOT / "metagpt/prompts"
UT_PATH = PROJECT_ROOT / "data/ut"
SWAGGER_PATH = UT_PATH / "files/api/"
UT_PY_PATH = UT_PATH / "files/ut/"
API_QUESTIONS_PATH = UT_PATH / "files/question/"
YAPI_URL = "http://yapi.deepwisdomai.com/"
TMP = PROJECT_ROOT / "tmp"
RESEARCH_PATH = DATA_PATH / "research"
TUTORIAL_PATH = DATA_PATH / "tutorial_docx"
INVOICE_OCR_TABLE_PATH = DATA_PATH / "invoice_table"

SKILL_DIRECTORY = PROJECT_ROOT / "metagpt/skills"

```py

这段代码定义了一个名为MEM_TTL的存储器变量，并将其赋值为24 * 30 * 3600。具体来说，这个代码将24个字节（每个字节8个位二进制数）组成一个30位的二进制数，然后将这个30位二进制数乘以3600（秒），得到一个表示毫秒级的浮点数，即24 * 30 * 3600 = 259200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000


```
MEM_TTL = 24 * 30 * 3600

```