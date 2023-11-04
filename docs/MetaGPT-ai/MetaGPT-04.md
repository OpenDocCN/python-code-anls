# MetaGPT源码解析 4

# `metagpt/actions/search_and_summarize.py`

这段代码是一个Python脚本，用于执行在Google搜索引擎中执行搜索操作。下面是解释作用域和部分的详细说明：

```py
# -*- coding: utf-8 -*-
```

这是Python注释的格式，告诉其在代码中使用的编码类型是utf-8。

```py
# -*- coding: utf-8 -*-
```

这是Python注释的格式，告诉其在代码中使用的编码类型是utf-8。

```py
import pydantic
```

这是Python中pydantic库的导入，用于定义API的文档对象模型。

```py
from metagpt.actions import Action
from metagpt.config import Config
from metagpt import logger
```

这是从metagpt库中导入的类，包括metagpt是一个机器学习API的文档，以及metagpt库和metagpt工具的类和函数。

```py
from metagpt.logs import logger
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.schema import Message
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.tools.search_engine import SearchEngine
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
import pydantic
```

这是Python中pydantic库的导入，用于定义API的文档对象模型。

```py
from metagpt.actions import Action
from metagpt.config import Config
from metagpt import logger
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.logs import logger
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.schema import Message
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.tools.search_engine import SearchEngine
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
import pydantic
```

这是Python中pydantic库的导入，用于定义API的文档对象模型。

```py
from metagpt.actions import Action
from metagpt.config import Config
from metagpt import logger
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.logs import logger
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.schema import Message
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.tools.search_engine import SearchEngine
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
import pydantic
```

这是Python中pydantic库的导入，用于定义API的文档对象模型。

```py
from metagpt.actions import Action
from metagpt.config import Config
from metagpt import logger
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.logs import logger
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.schema import Message
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.tools.search_engine import SearchEngine
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
import pydantic
```

这是Python中pydantic库的导入，用于定义API的文档对象模型。

```py
from metagpt.actions import Action
from metagpt.config import Config
from metagpt import logger
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.logs import logger
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.schema import Message
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.tools.search_engine import SearchEngine
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
import pydantic
```

这是Python中pydantic库的导入，用于定义API的文档对象模型。

```py
from metagpt.actions import Action
from metagpt.config import Config
from metagpt import logger
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.logs import logger
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.schema import Message
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.tools.search_engine import SearchEngine
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
import pydantic
```

这是Python中pydantic库的导入，用于定义API的文档对象模型。

```py
from metagpt.actions import Action
from metagpt.config import Config
from metagpt import logger
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.logs import logger
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.schema import Message
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.tools.search_engine import SearchEngine
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
import pydantic
```

这是Python中pydantic库的导入，用于定义API的文档对象模型。

```py
from metagpt.actions import Action
from metagpt.config import Config
from metagpt import logger
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.logs import logger
```

这是从metagpt库中导入的类，包括metagpt库和metagpt工具的类和函数。

```py
from metagpt.schema import Message
```



```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/23 17:26
@Author  : alexanderwu
@File    : search_google.py
"""
import pydantic

from metagpt.actions import Action
from metagpt.config import Config
from metagpt.logs import logger
from metagpt.schema import Message
from metagpt.tools.search_engine import SearchEngine

```

这段代码是一个人工智能助手，用于对参考信息和对话历史的最新对话进行摘要和总结。它根据对话内容，提取出与主题相关的内容，并尝试消除与对话无关的文本。此外，如果对话中包含可点击的链接，它将把链接标注在主要文本的格式中。整段代码的目的是帮助用户获得有用的信息，并提供清晰、简洁、非重复、适度长度的回复。


```py
SEARCH_AND_SUMMARIZE_SYSTEM = """### Requirements
1. Please summarize the latest dialogue based on the reference information (secondary) and dialogue history (primary). Do not include text that is irrelevant to the conversation.
- The context is for reference only. If it is irrelevant to the user's search request history, please reduce its reference and usage.
2. If there are citable links in the context, annotate them in the main text in the format [main text](citation link). If there are none in the context, do not write links.
3. The reply should be graceful, clear, non-repetitive, smoothly written, and of moderate length, in {LANG}.

### Dialogue History (For example)
A: MLOps competitors

### Current Question (For example)
A: MLOps competitors

### Current Reply (For example)
1. Alteryx Designer: <desc> etc. if any
2. Matlab: ditto
```

这段代码是一个字符串，它定义了一个名为 `SEARCH_AND_SUMMARIZE_PROMPT` 的变量。该变量的值为 `"""

SEARCH_AND_SUMMARIZE_SYSTEM_EN_US = SEARCH_AND_SUMMARIZE_SYSTEM.format(LANG="en-us")`。它的作用是定义了一个参数 `SEARCH_AND_SUMMARIZE_SYSTEM_EN_US`，该参数有一个 `SEARCH_AND_SUMMARIZE_SYSTEM` 和 `LANG` 参数，并将 `LANG` 参数设置为 `en-us`。

接着，该代码定义了一个名为 `SEARCH_AND_SUMMARIZE_PROMPT` 的变量，它的值为 `"""

SEARCH_AND_SUMMARIZE_PROMPT = """
### Reference Information
{CONTEXT}

### Dialogue History
{QUERY_HISTORY}
{QUERY}
"""`。它的作用是定义了一个参数 `SEARCH_AND_SUMMARIZE_PROMPT`，该参数包含了两个嵌套的引用信息 `{CONTEXT}` 和 `{QUERY_HISTORY}`，以及一个查询 `{QUERY}`。这些参数将在 RapidMiner Studio 中使用，用于搜索和分析数据。


```py
3. IBM SPSS Statistics
4. RapidMiner Studio
5. DataRobot AI Platform
6. Databricks Lakehouse Platform
7. Amazon SageMaker
8. Dataiku
"""

SEARCH_AND_SUMMARIZE_SYSTEM_EN_US = SEARCH_AND_SUMMARIZE_SYSTEM.format(LANG="en-us")

SEARCH_AND_SUMMARIZE_PROMPT = """
### Reference Information
{CONTEXT}

### Dialogue History
{QUERY_HISTORY}
{QUERY}

```

这段代码是一个人工智能助手，它有两个主要的功能：1）根据提供的参考信息和对话历史，对最新的对话进行摘要；2）在摘要中，确保内容与对话背景相关，不包含与对话无关的文本。此外，助手还负责在摘要中标注有可引用链接的文本，以确保用户可以方便地查阅相关资料。最后，助手将输出一个合适的、清晰、非重复、流畅的简体中文回复。


```py
### Current Question
{QUERY}

### Current Reply: Based on the information, please write the reply to the Question


"""


SEARCH_AND_SUMMARIZE_SALES_SYSTEM = """## Requirements
1. Please summarize the latest dialogue based on the reference information (secondary) and dialogue history (primary). Do not include text that is irrelevant to the conversation.
- The context is for reference only. If it is irrelevant to the user's search request history, please reduce its reference and usage.
2. If there are citable links in the context, annotate them in the main text in the format [main text](citation link). If there are none in the context, do not write links.
3. The reply should be graceful, clear, non-repetitive, smoothly written, and of moderate length, in Simplified Chinese.

```

这段代码是一个Example对话，展示了一个用户问和一个销售人员之间的对话，用户询问关于油性皮肤的洁面产品，销售人员提供了一些产品推荐，然后向用户推荐了L'Oreal公司的两个产品，一个是男性面部清洁剂，适用于油性皮肤，具有清洁、控制油脂分泌、平衡水和油、毛孔清洁、深层清洁毛孔等功效，另一个是年龄适当的保湿清洁剂，添加了两种主要成分，即 coconut oil 和 Centella Asiatica，具有更深层次的清洁、紧肤、温和以及不会让皮肤感到干燥的功效。


```py
# Example
## Reference Information
...

## Dialogue History
user: Which facial cleanser is good for oily skin?
Salesperson: Hello, for oily skin, it is suggested to choose a product that can deeply cleanse, control oil, and is gentle and skin-friendly. According to customer feedback and market reputation, the following facial cleansers are recommended:...
user: Do you have any by L'Oreal?
> Salesperson: ...

## Ideal Answer
Yes, I've selected the following for you:
1. L'Oreal Men's Facial Cleanser: Oil control, anti-acne, balance of water and oil, pore purification, effectively against blackheads, deep exfoliation, refuse oil shine. Dense foam, not tight after washing.
2. L'Oreal Age Perfect Hydrating Cleanser: Added with sodium cocoyl glycinate and Centella Asiatica, two effective ingredients, it can deeply cleanse, tighten the skin, gentle and not tight.
"""

```

This code appears to be a Python script that takes in some information about a user's search query for food in Xiamen and returns a list of top foods based on the input.

The script first defines a variable called `SEARCH_AND_SUMMARIZE_SALES_PROMPT` which contains some information about the search prompt. It then defines a variable called `SEARCH_FOOD` which will be used to store the user's search query.

The script then reads in the user's search query and displays it using the `print()` function. Next, it reads in the list of food options from a file specified as `INPUT_FILE`. It then loops through each option and adds it to a list called `food_options`.

Finally, the script returns the list of top foods based on the user's search query, by counting the frequency of each option in the input list. It then sorts the list of options based on the frequency count and returns the top options in a new list called `SORTED_OPTIONS`.


```py
SEARCH_AND_SUMMARIZE_SALES_PROMPT = """
## Reference Information
{CONTEXT}

## Dialogue History
{QUERY_HISTORY}
{QUERY}
> {ROLE}: 

"""

SEARCH_FOOD = """
# User Search Request
What are some delicious foods in Xiamen?

```

```pycss
class SearchAndSummarize(Action):
   def __init__(self, 
       name="Search and Summarize",
       context=None, 
       llm=None,
       engine=None,
       search_func=None
   ):
       self.config = Config()
       self.engine = engine or self.config.search_engine

       try:
           self.search_engine = SearchEngine(self.engine, run_func=search_func)
       except pydantic.ValidationError:
           self.search_engine = None

       self.result = ""
       super().__init__(name, context, llm)

   async def run(self, 
       context: list[Message], 
       system_text=SEARCH_AND_SUMMARIZE_SYSTEM
   ) -> str:
       if self.search_engine is None:
           logger.warning("Configure one of SERPAPI_API_KEY, SERPER_API_KEY, GOOGLE_API_KEY to unlock full feature")
           return ""

       query = context[-1].content
       # logger.debug(query)
       rsp = await self.search_engine.run(query)
       self.result = rsp
       if not rsp:
           logger.error("empty rsp...")
           return ""
       # logger.info(rsp)

       system_prompt = [system_text]

       prompt = SEARCH_AND_SUMMARIZE_PROMPT.format(
           # PREFIX = self.prefix,
           ROLE=self.profile,
           CONTEXT=rsp,
           QUERY_HISTORY="\n".join([str(i) for i in context[:-1]]),
           QUERY=str(context[-1]),
       )
       result = await self._aask(prompt, system_prompt)
       logger.debug(prompt)
       logger.debug(result)
       return result
   

# Example usage:
async def main(args):
   loop = asyncio.get_event_loop()
   result = loop.run_until_complete(SearchAndSummarize().run, args=args)
   print(result)

```



```py
# Requirements
You are a member of a professional butler team and will provide helpful suggestions:
1. Please summarize the user's search request based on the context and avoid including unrelated text.
2. Use [main text](reference link) in markdown format to **naturally annotate** 3-5 textual elements (such as product words or similar text sections) within the main text for easy navigation.
3. The response should be elegant, clear, **without any repetition of text**, smoothly written, and of moderate length.
"""


class SearchAndSummarize(Action):
    def __init__(self, name="", context=None, llm=None, engine=None, search_func=None):
        self.config = Config()
        self.engine = engine or self.config.search_engine

        try:
            self.search_engine = SearchEngine(self.engine, run_func=search_func)
        except pydantic.ValidationError:
            self.search_engine = None

        self.result = ""
        super().__init__(name, context, llm)

    async def run(self, context: list[Message], system_text=SEARCH_AND_SUMMARIZE_SYSTEM) -> str:
        if self.search_engine is None:
            logger.warning("Configure one of SERPAPI_API_KEY, SERPER_API_KEY, GOOGLE_API_KEY to unlock full feature")
            return ""

        query = context[-1].content
        # logger.debug(query)
        rsp = await self.search_engine.run(query)
        self.result = rsp
        if not rsp:
            logger.error("empty rsp...")
            return ""
        # logger.info(rsp)

        system_prompt = [system_text]

        prompt = SEARCH_AND_SUMMARIZE_PROMPT.format(
            # PREFIX = self.prefix,
            ROLE=self.profile,
            CONTEXT=rsp,
            QUERY_HISTORY="\n".join([str(i) for i in context[:-1]]),
            QUERY=str(context[-1]),
        )
        result = await self._aask(prompt, system_prompt)
        logger.debug(prompt)
        logger.debug(result)
        return result
    
```

# `metagpt/actions/write_code.py`

这段代码定义了一个Python脚本，名为`write_code.py`，环境设置为`usr/bin/env python`。

在脚本内部，首先导入了`metagpt.actions`、`metagpt.actions.action`、`metagpt.const`、`metagpt.logs`、`metagpt.schema`、`metagpt.utils.common`和`tenacity`模块。

接着，定义了一个名为`WriteDesign`的类，该类继承自`metagpt.actions.action.Action`类。

然后，通过调用`action.add_argument`方法，将`WriteDesign`类的参数添加到即将要执行的`Action`实例中。

接着，定义了一个`logger`实例，用于记录信息。

接着，定义了一个`CodeParser`类，用于解析输入的代码格式。

接着，定义了一个`retry`方法，用于设置重试的最大次数和超时时间。

接着，定义了一个`stop_after_attempt`方法，用于在达到一定尝试次数后停止尝试。

接着，定义了一个`wait_fixed`方法，用于等待固定时间。

最后，在脚本内部创建了一个工作区根目录`WORKSPACE_ROOT`，并将所有定义的类和函数都保存到该目录下。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:45
@Author  : alexanderwu
@File    : write_code.py
"""
from metagpt.actions import WriteDesign
from metagpt.actions.action import Action
from metagpt.const import WORKSPACE_ROOT
from metagpt.logs import logger
from metagpt.schema import Message
from metagpt.utils.common import CodeParser
from tenacity import retry, stop_after_attempt, wait_fixed

```

这段代码是一个Python Prompt的模板，用于指导开发人员编写符合PEP8规范、优雅、模块化、易于阅读和维护的Python 3.9代码。它包括了多个要求，包括使用单行代码实现仅有一个文件、使用现有的API、实现紧跟语法的代码片段、使用双井号进行分隔、有注释等。通过遵循这些要求，开发人员可以编写出高质量的、易于维护的Python代码。


```py
PROMPT_TEMPLATE = """
NOTICE
Role: You are a professional engineer; the main goal is to write PEP8 compliant, elegant, modular, easy to read and maintain Python 3.9 code (but you can also use other programming language)
ATTENTION: Use '##' to SPLIT SECTIONS, not '#'. Output format carefully referenced "Format example".

## Code: {filename} Write code with triple quoto, based on the following list and context.
1. Do your best to implement THIS ONLY ONE FILE. ONLY USE EXISTING API. IF NO API, IMPLEMENT IT.
2. Requirement: Based on the context, implement one following code file, note to return only in code form, your code will be part of the entire project, so please implement complete, reliable, reusable code snippets
3. Attention1: If there is any setting, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE.
4. Attention2: YOU MUST FOLLOW "Data structures and interface definitions". DONT CHANGE ANY DESIGN.
5. Think before writing: What should be implemented and provided in this document?
6. CAREFULLY CHECK THAT YOU DONT MISS ANY NECESSARY CLASS/FUNCTION IN THIS FILE.
7. Do not use public member functions that do not exist in your design.

-----
```

This is a Python class that appears to be designed to write code in a prompt and then save it to a file. Here is a summary of the class:

* It has a `write_code` method, which takes a prompt as an argument and returns a code string. This method uses the `_aask` method to prompt the user to enter the code, and then parses the code using `CodeParser`.
* It has a `run` method, which takes a context and a filename as arguments and returns the code that would be written to the file. This method uses the `_aask_v1` method to prompt the user to enter the code, and then calls the `write_code` method to save the code to the file.
* It has a `WriteCode` class, which appears to be the parent class for this class and defines a `__init__` method, a `write_code` method, and a `run` method.
* The `write_code` method takes a prompt and then returns the code that would be written to the file. The method uses the `_is_invalid` method to check if the file should be saved, and if the file is not in the expected format (e.g. .mp3 or .wav), the method returns without saving the code.
* The `run` method takes the code to save and the filename to write, and then calls the `write_code` method to save the code to the file.
* The `_aask` method is a dependency of this class and appears to be a class method for asking the user for input.
* The `CodeParser` class is a dependency of this class and appears to be responsible for parsing the code prompt.


```py
# Context
{context}
-----
## Format example
-----
## Code: {filename}
```python
## {filename}
...
```py
-----
"""


class WriteCode(Action):
    def __init__(self, name="WriteCode", context: list[Message] = None, llm=None):
        super().__init__(name, context, llm)

    def _is_invalid(self, filename):
        return any(i in filename for i in ["mp3", "wav"])

    def _save(self, context, filename, code):
        # logger.info(filename)
        # logger.info(code_rsp)
        if self._is_invalid(filename):
            return

        design = [i for i in context if i.cause_by == WriteDesign][0]

        ws_name = CodeParser.parse_str(block="Python package name", text=design.content)
        ws_path = WORKSPACE_ROOT / ws_name
        if f"{ws_name}/" not in filename and all(i not in filename for i in ["requirements.txt", ".md"]):
            ws_path = ws_path / ws_name
        code_path = ws_path / filename
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(code)
        logger.info(f"Saving Code to {code_path}")

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    async def write_code(self, prompt):
        code_rsp = await self._aask(prompt)
        code = CodeParser.parse_code(block="", text=code_rsp)
        return code

    async def run(self, context, filename):
        prompt = PROMPT_TEMPLATE.format(context=context, filename=filename)
        logger.info(f'Writing {filename}..')
        code = await self.write_code(prompt)
        # code_rsp = await self._aask_v1(prompt, "code_rsp", OUTPUT_MAPPING)
        # self._save(context, filename, code)
        return code
    
```

# `metagpt/actions/write_code_review.py`

这段代码定义了一个Python脚本，名为`write_code_review.py`。它使用`metagpt`库来执行一个行动（Action），这个行动可能与代码审查相关。

具体来说，这个脚本定义了一个名为`Action`的类，这个类使用`metagpt.actions.action`模块提供的功能。通过这个类的创建，可以定义一个行动计划，然后执行相应的操作。

接下来，它导入了两个模块：`metagpt.logs`和`metagpt.schema`，这两个模块用于在日志和上下文中使用`metagpt`库的输出。

接着，它定义了一个`Message`类，这个类的作用是接收一个`Message`对象，然后执行相应的操作。这个`Message`对象可以在`metagpt`库的日志中使用，这样就可以在日志中记录这个行动计划的结果。

此外，它还定义了一个名为`CodeParser`的类，这个类的作用是解析代码文件，将代码转换为另一个代码文件。这个操作可以在`metagpt`库的代码文件夹中执行，这样就可以将代码解析为另一个代码文件。

最后，它使用`tenacity`库来 retry、停止 Attempt 和等待固定时间。`retry`函数可以允许在一定次数失败后重试操作，而`stop_after_attempt`函数可以在连续失败一定次数后停止尝试，`wait_fixed`函数可以在指定的固定时间内等待结果。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:45
@Author  : alexanderwu
@File    : write_code_review.py
"""

from metagpt.actions.action import Action
from metagpt.logs import logger
from metagpt.schema import Message
from metagpt.utils.common import CodeParser
from tenacity import retry, stop_after_attempt, wait_fixed

PROMPT_TEMPLATE = """
```

This code is a NOTICE tool used for code review. It checks whether the code conforms to the PEP8 standards, is elegant and modularized, easy to read and maintain, and is written in Python 3.9 or another programming language.

The code consists of a series of checks to evaluate the code, with specific suggestions for improvement. These checks include:

1. Verification that the code is implemented as per the requirements.
2. Verification that there are no issues with the code logic.
3. Verification that the existing code follows the "Data structures and interface definitions".
4. Verification that there is a function in the code that is omitted or not fully implemented that needs to be implemented.
5. Verification that the code has unnecessary or lack dependencies.

The tool provides suggestions for improvement by outlining any issues that need to be addressed, and provides a rewritten code based on those suggestions. The suggestions are given in the format example, and the number of suggestions are limited to 5.


```py
NOTICE
Role: You are a professional software engineer, and your main task is to review the code. You need to ensure that the code conforms to the PEP8 standards, is elegantly designed and modularized, easy to read and maintain, and is written in Python 3.9 (or in another programming language).
ATTENTION: Use '##' to SPLIT SECTIONS, not '#'. Output format carefully referenced "Format example".

## Code Review: Based on the following context and code, and following the check list, Provide key, clear, concise, and specific code modification suggestions, up to 5.
```
1. Check 0: Is the code implemented as per the requirements?
2. Check 1: Are there any issues with the code logic?
3. Check 2: Does the existing code follow the "Data structures and interface definitions"?
4. Check 3: Is there a function in the code that is omitted or not fully implemented that needs to be implemented?
5. Check 4: Does the code have unnecessary or lack dependencies?
```py

## Rewrite Code: {filename} Base on "Code Review" and the source code, rewrite code with triple quotes. Do your utmost to optimize THIS SINGLE FILE. 
-----
```

这段代码是一个 PHP 代码片段，表示一个名为 `context` 的自定义context。通常情况下，上下文（context）是指在某个上下文中需要用到的信息或数据。这里，上下文是一个未知的值，需要根据具体情况进行解析和填写。
```pycode
{context}
```
在这里，`{context}` 表示一个占位符，将未来的值替代为它的内容。这个占位符在代码中可能会被用来输出提示信息、错误信息或者在计算中使用，使得代码更加可读、易懂。
```pyfile=filename.php
```
这是一个简单的PHP文件头，告诉服务器这个文件是php的文件名。当程序遇到这个文件时，服务器会知道要读取的PHP文件名是 `filename.php`。


```py
# Context
{context}

## Code: {filename}
```
{code}
```py
-----

## Format example
-----
{format_example}
-----

"""

```

这段代码是一个字符串，包含了一个format_example变量和一个可变参数的格式字符串。当变量format_example包含一个格式字符串和一些参数时，它将用format()函数来格式化这些参数，并将结果存储在format_example变量中。

具体来说，该代码的作用是定义了一个format_example变量，该变量包含了一个格式字符串和一个或多个可变参数。当变量format_example包含一个格式字符串和一些参数时，它会用format()函数来格式化这些参数，并将结果存储在format_example变量中。如果format_example变量包含的格式字符串是"..."，那么它会在需要格式化的参数列表和格式字符串之间插入缓冲区字符，将参数和格式字符串连接起来，并插入缓冲区字符。最终，format_example变量的结果将是一个格式化的字符串，可以根据需要进行格式化。


```py
FORMAT_EXAMPLE = """

## Code Review
1. The code ...
2. ...
3. ...
4. ...
5. ...

## Rewrite Code: {filename}
```python
## {filename}
...
```py
"""


```

这段代码定义了一个名为 "WriteCodeReview" 的类，它继承自自定义的 "Action" 类。这个类的初始化方法包含三个参数：一个名称（默认值为 "WriteCodeReview"）、一个上下文列表（可以是空列表）和一个逻辑消息（LLM，这里是可选的）。

在 `write_code` 方法中，使用 `super().__init__(name, context, llm)` 来调用父类的初始化方法。这里的 `super()` 调用表示尊重父类的行为，因为在自定义方法中可能需要设置父类的参数。

`write_code` 方法的主要作用是编写代码并返回。它使用 `asyncio` 库的 `run` 方法来加载代码并运行写入操作。`write_code` 方法的 `async` 特性使用 `await` 和 `await` 关键字来自动执行代码。

`write_code` 方法的 `write_code` 方法接受一个参数 `prompt`，它是一个字符串，表示要运行的命令或提示信息。这个参数是 `write_code` 方法的主要输入，用于从用户或外部获取代码输入。

`run` 方法包含一个带有 `context` 和 `code` 参数的 `async` 方法。`context` 参数是一个输入参数，表示当前上下文中的消息列表。`code` 参数是一个输入参数，表示要运行的代码。在这里，`context` 参数被设置为 `None`，意味着将要运行的代码将只从 `code` 参数中获取输入。

`run` 方法还包含一个带有 `filename` 参数的 `async` 方法。这个参数用于指定要保存的文件名。`async` 方法使用 `await` 关键字来获取代码，并将其保存到指定的位置。

`write_code` 和 `run` 方法的另一个重用 `asyncio` 库的 `format_example` 方法，用于将 `Context` 和 `code` 参数格式化为 `FORMAT_EXAMPLE` 和 `OUTPUT_MAPPING` 字典。


```py
class WriteCodeReview(Action):
    def __init__(self, name="WriteCodeReview", context: list[Message] = None, llm=None):
        super().__init__(name, context, llm)

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    async def write_code(self, prompt):
        code_rsp = await self._aask(prompt)
        code = CodeParser.parse_code(block="", text=code_rsp)
        return code

    async def run(self, context, code, filename):
        format_example = FORMAT_EXAMPLE.format(filename=filename)
        prompt = PROMPT_TEMPLATE.format(context=context, code=code, filename=filename, format_example=format_example)
        logger.info(f'Code review {filename}..')
        code = await self.write_code(prompt)
        # code_rsp = await self._aask_v1(prompt, "code_rsp", OUTPUT_MAPPING)
        # self._save(context, filename, code)
        return code
    
```

# `metagpt/actions/write_docstring.py`

这段代码是一个名为“Code Docstring Generator”的工具，用于自动生成Python代码的文档字符串。它通过指定的风格创建了指定代码的文档字符串。

具体来说，这段代码接受两个命令行选项：

- `--overwrite`：如果设置此选项并指定文档样式，则覆盖原始文件中的内容并生成新的文档字符串。
- `--style=<docstring_style>`：指定生成文档字符串的样式。有效的选项包括：
   - `'google'`
   - `'numpy'`
   - `'sphinx'`
   默认值为`'google'`。

使用方法：

1. 在命令行中运行`python3 -m metagpt.actions.write_docstring <filename> [--overwrite] [--style=<docstring_style>]`命令。
2. 如果设置`--overwrite`选项并指定文档样式，则这段代码将覆盖`<filename>`文件的内容并生成新的文档字符串。
3. 如果没有设置`--style`选项，则默认值为`'google'`。


```py
"""Code Docstring Generator.

This script provides a tool to automatically generate docstrings for Python code. It uses the specified style to create
docstrings for the given code and system text.

Usage:
    python3 -m metagpt.actions.write_docstring <filename> [--overwrite] [--style=<docstring_style>]

Arguments:
    filename           The path to the Python file for which you want to generate docstrings.

Options:
    --overwrite        If specified, overwrite the original file with the code containing docstrings.
    --style=<docstring_style>   Specify the style of the generated docstrings.
                                Valid values: 'google', 'numpy', or 'sphinx'.
                                Default: 'google'

```

This script uses the 'fire' library to create a command-line interface. It generates docstrings for the given Python code using the specified docstring style and adds them to the code.

The script takes two arguments:

- `startup.py`: The file from which to generate docstrings.
- `--overwrite`: A flag to indicate that the output file should be overwritten.
- `--style=numpy`: A style to use for the docstrings (default is `google`).

The script then reads the given file, generates the docstrings using the specified style, and adds them to the code. The resulting docstrings are added to the `startup.py` file.


```py
Example:
    python3 -m metagpt.actions.write_docstring startup.py --overwrite False --style=numpy

This script uses the 'fire' library to create a command-line interface. It generates docstrings for the given Python code using
the specified docstring style and adds them to the code.
"""
import ast
from typing import Literal

from metagpt.actions.action import Action
from metagpt.utils.common import OutputParser
from metagpt.utils.pycst import merge_docstring

PYTHON_DOCSTRING_SYSTEM = '''### Requirements
1. Add docstrings to the given code following the {style} style.
```

这段代码是一个文本，它提供了对一个Python函数和类的描述。它指导我们如何修改函数体，使其使用Ellipsis对象来减少输出。对于已经使用类型注释的函数或类，它建议我们不要在文档中包括它们。此外，它还告诉我们如何从给定的Python代码中提取出类、函数或文档字符串，以避免包含其他文本。

具体而言，第一行指导我们如何将函数体替换为一个Ellipsis对象。这样做的好处是，它可以让我们专注于函数或类本身的行为，而不是其定义，从而减少输出。第二行告诉我们，如果类型注释已经完整，我们就不需要包括它们在文档字符串中。第三行提醒我们，当我们自动类型注释函数或类时，不需要在文档中包含它们。最后一行建议我们，当我们提取函数、类或文档字符串时，应该只关注它们的文本内容，而不是它们的来源。


```py
2. Replace the function body with an Ellipsis object(...) to reduce output.
3. If the types are already annotated, there is no need to include them in the docstring.
4. Extract only class, function or the docstrings for the module parts from the given Python code, avoiding any other text.

### Input Example
```python
def function_with_pep484_type_annotations(param1: int) -> bool:
    return isinstance(param1, int)

class ExampleError(Exception):
    def __init__(self, msg: str):
        self.msg = msg
```py

### Output Example
```

这段代码定义了一个函数`function_with_pep484_type_annotations`，该函数具有PEP 484类型注释。这个函数接受一个整数参数`param1`，并返回一个布尔值。函数内部的具体实现你们可以在代码中查看。


```py
```python
{example}
```py
'''

# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

PYTHON_DOCSTRING_EXAMPLE_GOOGLE = '''
def function_with_pep484_type_annotations(param1: int) -> bool:
    """Example function with PEP 484 type annotations.

    Extended description of function.

    Args:
        param1: The first parameter.

    Returns:
        The return value. True for success, False otherwise.
    """
    ...

```

这段代码定义了一个名为ExampleError的类，该类继承自Python标准中的Exception类。在这个类中，重写了__init__方法，该方法的实现与该方法的文档类似。

ExampleError类的__init__方法接收一个名为msg的参数，这个参数用于描述异常的详细信息。当创建一个ExampleError实例时，将调用__init__方法，并且可以使用实例变量msg来获取该异常的详细信息。

由于该类的方法在类级文档中已经定义了，因此不需要在这里再次定义。用户在使用这个类时，可以通过使用 try-except 语句来捕获并处理ExampleError异常。


```py
class ExampleError(Exception):
    """Exceptions are documented in the same way as classes.

    The __init__ method was documented in the class level docstring.

    Args:
        msg: Human readable string describing the exception.

    Attributes:
        msg: Human readable string describing the exception.
    """
    ...
'''

PYTHON_DOCSTRING_EXAMPLE_NUMPY = '''
```

这段代码定义了一个函数`function_with_pep484_type_annotations`，它带有PEP 484类型注释。函数有一个参数`param1`，类型为整数。函数返回一个布尔值，表示成功或失败。具体实现中，可能还会有一些其他的逻辑。


```py
def function_with_pep484_type_annotations(param1: int) -> bool:
    """
    Example function with PEP 484 type annotations.

    Extended description of function.

    Parameters
    ----------
    param1
        The first parameter.

    Returns
    -------
    bool
        The return value. True for success, False otherwise.
    """
    ...

```

这段代码定义了一个名为ExampleError的异常类，继承自Exception类(自定义异常类)，并在该类中定义了一个__init__方法，用于初始化该异常类的实例。

在该类的__init__方法.__doc__中，对该方法的参数msg进行了详细的描述，指出该参数是一个字符串，用于描述异常的情况。

该异常类有一个名为msg的属性，用于存储描述异常情况的字符串。

由于该异常类是自定义的，因此该类中的__init__方法、__file__方法以及__name__方法都不会被自动继承。


```py
class ExampleError(Exception):
    """
    Exceptions are documented in the same way as classes.

    The __init__ method was documented in the class level docstring.

    Parameters
    ----------
    msg
        Human readable string describing the exception.

    Attributes
    ----------
    msg
        Human readable string describing the exception.
    """
    ...
```

这段代码定义了一个名为 `function_with_pep484_type_annotations` 的函数，它带有 PEP 484 类型注释。这个函数有一个参数 `param1`，它的类型是一个整数（`int`）。函数体内部包含了这个参数，但是没有定义返回值。

这个函数的作用是提供一个示例，展示了如何使用 PEP 484 类型注释来描述一个函数。类型注释可以帮助人们更好地理解函数的参数和返回值的数据类型。


```py
'''

PYTHON_DOCSTRING_EXAMPLE_SPHINX = '''
def function_with_pep484_type_annotations(param1: int) -> bool:
    """Example function with PEP 484 type annotations.

    Extended description of function.

    :param param1: The first parameter.
    :type param1: int

    :return: The return value. True for success, False otherwise.
    :rtype: bool
    """
    ...

```

这段代码定义了一个名为ExampleError的类，它继承自Python标准库中的Exception类。该类定义了一个__init__方法，用于初始化一个字符串参数"msg"，该参数用于描述异常。

在__init__方法的文档中，已经通过在同一个docstring中使用了%(docstring)语法来定义了一个人类可读的文档字符串，该文档字符串将包含在当ExampleError类被创建时产生的文档字符串。

此外，该类还定义了一个内部方法，该方法使用msg作为参数，来执行一些异常处理操作，例如记录异常信息到文件中，或发送邮件通知等。


```py
class ExampleError(Exception):
    """Exceptions are documented in the same way as classes.

    The __init__ method was documented in the class level docstring.

    :param msg: Human-readable string describing the exception.
    :type msg: str
    """
    ...
'''

_python_docstring_style = {
    "google": PYTHON_DOCSTRING_EXAMPLE_GOOGLE.strip(),
    "numpy": PYTHON_DOCSTRING_EXAMPLE_NUMPY.strip(),
    "sphinx": PYTHON_DOCSTRING_EXAMPLE_SPHINX.strip(),
}


```

这段代码定义了一个名为 `WriteDocstring` 的类，用于为代码编写文档字符串。

该类包含了一个 `__init__` 方法，用于初始化该类的实例并设置文档字符串。

该类定义了一个 `run` 方法，该方法接受三个参数：要运行的代码、系统文本和指定文档字符串的样式。该方法使用 `_simplify_python_code` 函数将传入的代码进行简化，然后使用 `aask` 函数将简化的代码和系统文本一起发送到 `__aask` 函数中，该函数将根据指定的样式返回文档字符串。

最后，该类定义了一个 `merge_docstring` 函数，该函数将两个或多个文档字符串合并为一个。


```py
class WriteDocstring(Action):
    """This class is used to write docstrings for code.

    Attributes:
        desc: A string describing the action.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.desc = "Write docstring for code."

    async def run(
        self, code: str,
        system_text: str = PYTHON_DOCSTRING_SYSTEM,
        style: Literal["google", "numpy", "sphinx"] = "google",
    ) -> str:
        """Writes docstrings for the given code and system text in the specified style.

        Args:
            code: A string of Python code.
            system_text: A string of system text.
            style: A string specifying the style of the docstring. Can be 'google', 'numpy', or 'sphinx'.

        Returns:
            The Python code with docstrings added.
        """
        system_text = system_text.format(style=style, example=_python_docstring_style[style])
        simplified_code = _simplify_python_code(code)
        documented_code = await self._aask(f"```python\n{simplified_code}\n```py", [system_text])
        documented_code = OutputParser.parse_python_code(documented_code)
        return merge_docstring(code, documented_code)


```

这段代码定义了一个名为 `_simplify_python_code` 的函数，它接受一个字符串参数 `code`。这个函数的作用是简化传入的 Python 代码，通过删除表达式和最后一个 if 语句，使得代码更加简洁易懂。

具体来说，这个函数接受一个 AST（抽象语法树）对象 `code_tree`，它代表了源代码文件中的代码结构。函数首先将 `code_tree` 中的 body 部分遍历，如果遍历到的是一个 if 语句，则将其删除；如果遍历到的不是 if 语句，就继续遍历。最后，函数返回的是 `code_tree` 对象经过简化处理后的结果。

通过调用 `_simplify_python_code` 函数，我们可以将一个复杂的 Python 代码变得更易于阅读和理解。


```py
def _simplify_python_code(code: str) -> None:
    """Simplifies the given Python code by removing expressions and the last if statement.

    Args:
        code: A string of Python code.

    Returns:
        The simplified Python code.
    """
    code_tree = ast.parse(code)
    code_tree.body = [i for i in code_tree.body if not isinstance(i, ast.Expr)]
    if isinstance(code_tree.body[-1], ast.If):
        code_tree.body.pop()
    return ast.unparse(code_tree)


```

这段代码是一个Python脚本，它的作用是定义了一个函数run，该函数会运行一个Python文件并将其内容写入同一文件，或者覆盖同一文件。通过观察代码，我们可以看到以下几点解释：

1. 函数run接受三个参数：filename、overwrite和style，分别代表要运行的文件、是否覆盖现有文件和要使用的编写格式的描述符。
2. 函数内部使用with open()语句打开文件并将其内容存储到变量code中。
3. 函数内部使用f.read()方法读取文件内容，并将其存储到变量code中。
4. 函数内部使用WriteDocstring().run()方法运行编写格式的代码，并将结果存储到变量 code 中。
5. 如果overwrite参数为True，函数将打开文件并将其内容替换为代码内容。
6. 函数返回代码内容。

通过这些解释，我们可以看到函数run的作用是运行一个Python文件并将其内容写入同一文件，或者覆盖同一文件。通过不同的参数，我们可以选择不同的方式来运行文件，比如使用不同的格式来描述文件的内容。


```py
if __name__ == "__main__":
    import fire

    async def run(filename: str, overwrite: bool = False, style: Literal["google", "numpy", "sphinx"] = "google"):
        with open(filename) as f:
            code = f.read()
        code = await WriteDocstring().run(code, style=style)
        if overwrite:
            with open(filename, "w") as f:
                f.write(code)
        return code

    fire.Fire(run)

```

# `metagpt/actions/write_prd.py`

这段代码定义了一个名为`write_prd.py`的Python文件，并在其中实现了以下功能：

1. 从`metagpt.actions`模块中导入`Action`和`ActionOutput`类型。
2. 从`metagpt.actions.search_and_summarize`模块中导入`SearchAndSummarize`。
3. 从`metagpt.config`模块中导入`CONFIG`。
4. 从`metagpt.logs`模块中导入`logger`。
5. 从`metagpt.utils.get_template`模块中导入`get_template`。
6. 定义了一个名为`write_prd`的函数，接受一个字符串参数`text`。
7. 在函数内部，首先创建一个包含`text`标签的内容列表`paragraphs`。
8. 然后使用`get_template`函数从`text`中获取模板，并将其存储为`template`。
9. 使用`SearchAndSummarize`函数搜索文本中的关键词，并将搜索结果存储为`summary_results`。
10. 在函数内部创建一个包含两个变量的列表`actions`，分别存储`Action`和`ActionOutput`对象。
11. 将`actions[0]`设置为`SearchAndSummarize`函数生成的`summary_results`。
12. 将`actions[1]`设置为`Action`对象，其中`text`为模板，`summary_results`为搜索结果。
13. 调用`logger.info(f'正在创建一个包含文章标题 {text} 的文档。')`来输出创建文档的信息。
14. 最后，返回`actions`列表，以便将其可以将文档的多个部分组合成一个完整的文档。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:45
@Author  : alexanderwu
@File    : write_prd.py
"""
from typing import List

from metagpt.actions import Action, ActionOutput
from metagpt.actions.search_and_summarize import SearchAndSummarize
from metagpt.config import CONFIG
from metagpt.logs import logger
from metagpt.utils.get_template import get_template

```

这段代码定义了一个名为 `templates` 的字典，其中包含一个名为 `json` 的键，它包含一个字符串 `PROMPT_TEMPLATE`。

的字符串 `PROMPT_TEMPLATE` 在模板中使用了 Merpeople 的语法，定义了一个名为 `requirements` 的列表，其中包含了一些 requirement。接着定义了一个名为 `search_information` 的列表，其中包含了一些搜索信息。最后，定义了一个名为 `quadrantChart` 的嵌套的列表，其中包含了一个嵌套的 `title` 元素，以及四个 `quadrant` 元素，每个元素下面包含了一个需求列表。

整个模板是为了描述一个关于如何改进一个产品在四个不同竞争对手环境中的表现。


```py
templates = {
    "json": {
        "PROMPT_TEMPLATE": """
# Context
## Original Requirements
{requirements}

## Search Information
{search_information}

## mermaid quadrantChart code syntax example. DONT USE QUOTO IN CODE DUE TO INVALID SYNTAX. Replace the <Campain X> with REAL COMPETITOR NAME
```mermaid
quadrantChart
    title Reach and engagement of campaigns
    x-axis Low Reach --> High Reach
    y-axis Low Engagement --> High Engagement
    quadrant-1 We should expand
    quadrant-2 Need to promote
    quadrant-3 Re-evaluate
    quadrant-4 May be improved
    "Campaign: A": [0.3, 0.6]
    "Campaign B": [0.45, 0.23]
    "Campaign C": [0.57, 0.69]
    "Campaign D": [0.78, 0.34]
    "Campaign E": [0.40, 0.34]
    "Campaign F": [0.35, 0.78]
    "Our Target Product": [0.5, 0.6]
```py

这段代码是一个Jira要求的示例，用于描述一个职业产品经理需要完成的任务。具体来说，这个产品经理需要根据上下文提供产品需求、产品目标、用户故事和竞争分析。这些信息将用于评估产品是否符合客户和公司的要求，从而确保产品的成功上市。


```
```py

## Format example
{format_example}
-----
Role: You are a professional product manager; the goal is to design a concise, usable, efficient product
Requirements: According to the context, fill in the following missing information, each section name is a key in json ,If the requirements are unclear, ensure minimum viability and avoid excessive design

## Original Requirements: Provide as Plain text, place the polished complete original requirements here

## Product Goals: Provided as Python list[str], up to 3 clear, orthogonal product goals. If the requirement itself is simple, the goal should also be simple

## User Stories: Provided as Python list[str], up to 5 scenario-based user stories, If the requirement itself is simple, the user stories should also be less

## Competitive Analysis: Provided as Python list[str], up to 7 competitive product analyses, consider as similar competitors as possible

```

{"FORMAT_EXAMPLE": "Original Requirements: , Search Information:, Requirements: , Product Goals: , User Stories: , Competitive Analysis: , Competitive Quadrant Chart: We should expand, Need to promote, Re-evaluate, May be improved,Campaign A: [0.3, 0.6], Campaign B: [0.45, 0.23], Campaign C: [0.57, 0.69], Campaign D: [0.78, 0.34], Campaign E: [0.40, 0.34], Campaign F: [0.35, 0.78]",
"UI Design draft": " UI Design draft: , Anything UNCLEAR: "
}


```py
## Competitive Quadrant Chart: Use mermaid quadrantChart code syntax. up to 14 competitive products. Translation: Distribute these competitor scores evenly between 0 and 1, trying to conform to a normal distribution centered around 0.5 as much as possible.

## Requirement Analysis: Provide as Plain text. Be simple. LESS IS MORE. Make your requirements less dumb. Delete the parts unnessasery.

## Requirement Pool: Provided as Python list[list[str], the parameters are requirement description, priority(P0/P1/P2), respectively, comply with PEP standards; no more than 5 requirements and consider to make its difficulty lower

## UI Design draft: Provide as Plain text. Be simple. Describe the elements and functions, also provide a simple style description and layout description.
## Anything UNCLEAR: Provide as Plain text. Make clear here.

output a properly formatted JSON, wrapped inside [CONTENT][/CONTENT] like format example,
and only output the json inside this tag, nothing else
""",
        "FORMAT_EXAMPLE": """
[CONTENT]
{
    "Original Requirements": "",
    "Search Information": "",
    "Requirements": "",
    "Product Goals": [],
    "User Stories": [],
    "Competitive Analysis": [],
    "Competitive Quadrant Chart": "quadrantChart
                title Reach and engagement of campaigns
                x-axis Low Reach --> High Reach
                y-axis Low Engagement --> High Engagement
                quadrant-1 We should expand
                quadrant-2 Need to promote
                quadrant-3 Re-evaluate
                quadrant-4 May be improved
                Campaign A: [0.3, 0.6]
                Campaign B: [0.45, 0.23]
                Campaign C: [0.57, 0.69]
                Campaign D: [0.78, 0.34]
                Campaign E: [0.40, 0.34]
                Campaign F: [0.35, 0.78]",
    "Requirement Analysis": "",
    "Requirement Pool": [["P0","P0 requirement"],["P1","P1 requirement"]],
    "UI Design draft": "",
    "Anything UNCLEAR": "",
}
[/CONTENT]
```

该代码是一个 Python 语言的 Python 包，包含了一个名为 "prompt" 的函数和一个名为 "markdown" 的字典。

"prompt" 函数是一个简单的两步指令函数，第一步用于创建一个字符串对象，第二步用于将其中的单引号'"'删除，并返回该字符串。其作用是用于在代码中引用一个字符串，使其在不需要单引号的情况下被作为其他程序或函数使用。

"markdown" 字典包含了 "PROMPT_TEMPLATE" 和 "mermaid" 两个键值对。其中 "PROMPT_TEMPLATE" 是一个字符串，用于存储在 "markdown" 字典中 "mermaid" 的模板数据。而 "mermaid" 是一个 Python 的函数模块，通过提供许多图表和数据可视化组件，帮助用户创建丰富多彩的文档和表格。在 "markdown" 字典中， "mermaid" 的模板数据被用来定义图表的样例数据，通过 "PROMPT_TEMPLATE" 中提供的模板数据，可以使得 "mermaid" 函数根据上下文环境的内容生成相应的图表。


```py
""",
    },
    "markdown": {
        "PROMPT_TEMPLATE": """
# Context
## Original Requirements
{requirements}

## Search Information
{search_information}

## mermaid quadrantChart code syntax example. DONT USE QUOTO IN CODE DUE TO INVALID SYNTAX. Replace the <Campain X> with REAL COMPETITOR NAME
```mermaid
quadrantChart
    title Reach and engagement of campaigns
    x-axis Low Reach --> High Reach
    y-axis Low Engagement --> High Engagement
    quadrant-1 We should expand
    quadrant-2 Need to promote
    quadrant-3 Re-evaluate
    quadrant-4 May be improved
    "Campaign: A": [0.3, 0.6]
    "Campaign B": [0.45, 0.23]
    "Campaign C": [0.57, 0.69]
    "Campaign D": [0.78, 0.34]
    "Campaign E": [0.40, 0.34]
    "Campaign F": [0.35, 0.78]
    "Our Target Product": [0.5, 0.6]
```py

这段代码是一个Python格式示例，用于描述如何根据特定的要求输出文本。它主要包括以下几部分：

1. `Format example`：提供了一个简单的例子，描述了如何根据特定的格式输出文本。
2. ` Role: You are a professional product manager; the goal is to design a concise, usable, efficient product`：说明了一个产品经理的角色和目标，即设计简洁、易用、高效的產品。
3. ` Requirements: According to the context, fill in the following missing information, note that each sections are returned in Python code triple quote form seperatedly. If the requirements are unclear, ensure minimum viability and avoid excessive design`：说明了在输入具体要求之前，需要了解产品经理的背景和上下文。如果要求不明确，需要确保产品具有最低的可用性和可维护性。
4. `ATTENTION: Use '##' to SPLIT SECTIONS, not '#'`：提醒开发者注意，使用`##`代替`#`来分割代码的各个部分。
5. `Original Requirements: Provide as Plain text, place the polished complete original requirements here`：提供了原始需求列表，这些列表应该使用纯文本格式。
6. `Product Goals: Provided as Python list[str], up to 3 clear, orthogonal product goals. If the requirement itself is simple, the goal should also be simple`：描述了产品目标，即至多三个清晰、互相平行的产品目标。如果目标本身很简单，那么目标也应该很简单。
7. `User Stories: Provided as Python list[str], up to 5 scenario-based user stories, If the requirement itself is simple, the user stories should also be less`：描述了用户故事，即最多五个基于场景的简短用户故事。如果要求本身很简单，那么用户故事也应该简短。

最后，该代码还提供了一个简单的示例，用于说明如何根据产品目标和用户故事来输出文本。


```
```py

## Format example
{format_example}
-----
Role: You are a professional product manager; the goal is to design a concise, usable, efficient product
Requirements: According to the context, fill in the following missing information, note that each sections are returned in Python code triple quote form seperatedly. If the requirements are unclear, ensure minimum viability and avoid excessive design
ATTENTION: Use '##' to SPLIT SECTIONS, not '#'. AND '## <SECTION_NAME>' SHOULD WRITE BEFORE the code and triple quote. Output carefully referenced "Format example" in format.

## Original Requirements: Provide as Plain text, place the polished complete original requirements here

## Product Goals: Provided as Python list[str], up to 3 clear, orthogonal product goals. If the requirement itself is simple, the goal should also be simple

## User Stories: Provided as Python list[str], up to 5 scenario-based user stories, If the requirement itself is simple, the user stories should also be less

```

This code appears to be a Python script that provides a competitive analysis and requirement analysis for up to 7 competitive products.

The competitive analysis is provided as a list of strings, with up to 7 competitive product analyses. The competitive quadrant chart is created using the mermaid quadrantChart code syntax, up to 14 competitive products. The chart is designed to distribute these competitor scores evenly between 0 and 1, with the aim of conforming to a normal distribution centered around 0.5.

The requirement analysis is provided as plain text and is simple. It includes up to 5 requirements, with their priority assigned (P0/P1/P2), and aims to make their difficulty lower.

The UI design draft is also provided as plain text and is simple. It describes the elements and functions, as well as provides a simple style description and layout description.

Overall, it appears that this script is designed to provide a competitive analysis and requirement analysis for up to 7 competitive products, with the aim of providing a simple and streamlined process for the user.


```py
## Competitive Analysis: Provided as Python list[str], up to 7 competitive product analyses, consider as similar competitors as possible

## Competitive Quadrant Chart: Use mermaid quadrantChart code syntax. up to 14 competitive products. Translation: Distribute these competitor scores evenly between 0 and 1, trying to conform to a normal distribution centered around 0.5 as much as possible.

## Requirement Analysis: Provide as Plain text. Be simple. LESS IS MORE. Make your requirements less dumb. Delete the parts unnessasery.

## Requirement Pool: Provided as Python list[list[str], the parameters are requirement description, priority(P0/P1/P2), respectively, comply with PEP standards; no more than 5 requirements and consider to make its difficulty lower

## UI Design draft: Provide as Plain text. Be simple. Describe the elements and functions, also provide a simple style description and layout description.
## Anything UNCLEAR: Provide as Plain text. Make clear here.
""",
        "FORMAT_EXAMPLE": """
---
## Original Requirements
The boss ... 

```

以上代码是一个敏捷开发环境中使用的通用模板，用于描述产品功能的可行性、目标、用户故事和竞争分析。具体来说：

1. `Product Goals`：产品目标。描述了产品的宏观目标，例如提高用户满意度、增加用户粘性等。
2. `User Stories`：用户故事。描述了用户在使用产品时需要满足的需求和期望，例如登录、添加商品、搜索商品等。
3. `Competitive Analysis`：竞争分析。描述了市场中类似产品的竞争格局，包括它们的优势、劣势和市场占有率等。

这个通用模板提供了一个结构化的方法来描述产品开发过程中的各个方面，帮助团队更好地理解、沟通和协作，以确保产品的成功开发。


```py
## Product Goals
```python
[
    "Create a ...",
]
```py

## User Stories
```python
[
    "As a user, ...",
]
```py

## Competitive Analysis
```

This code appears to be a part of a Python Snake game. It is using the `mermaid` library to create a chart called `quadrantChart`. The chart appears to be a bar chart that displays the reach and engagement of different campaigns for a product named "Our Target Product". The chart is broken down into two categories: "Google" and "Our Target Product". The values listed for each category are the estimated number of users who have seen or interacted with the campaign.


```py
```python
[
    "Python Snake Game: ...",
]
```py

## Competitive Quadrant Chart
```mermaid
quadrantChart
    title Reach and engagement of campaigns
    ...
    "Our Target Product": [0.6, 0.7]
```py

## Requirement Analysis
```

This code appears to be a part of a game or application that requires a product to be entered. The "End game..." and "P0" suggest that this could be a code snippet for a game or a script that prompts the user to input a product number, and the square brackets indicate that more than one line's worth of code is being provided.

The UI Design draft that comes with this code snippet is a simple description of the requirement. It does not provide any context or information about the game or application, but it does suggest that the product number will be required and that it should be entered by the user.


```py
The product should be a ...

## Requirement Pool
```python
[
    ["End game ...", "P0"]
]
```py

## UI Design draft
Give a basic function description, and a draft

## Anything UNCLEAR
There are no unclear points.
---
```

这段代码定义了一个名为 `OUTPUT_MAPPING` 的字典，它包含了多个键值对，每个键都是 `OUTPUT_MAPPING.`，类型都是 `List[str]`。

这个字典的作用是映射原始需求描述符（如 "Original Requirements", "Product Goals", "User Stories", etc.）到相应的输出图形，以便更好地理解和呈现这些需求。对于每个输出图形，都有一个对应的字符串表示，例如 "Original Requirements" 对应的字符串就是空字符串。


```py
""",
    },
}

OUTPUT_MAPPING = {
    "Original Requirements": (str, ...),
    "Product Goals": (List[str], ...),
    "User Stories": (List[str], ...),
    "Competitive Analysis": (List[str], ...),
    "Competitive Quadrant Chart": (str, ...),
    "Requirement Analysis": (str, ...),
    "Requirement Pool": (List[List[str]], ...),
    "UI Design draft": (str, ...),
    "Anything UNCLEAR": (str, ...),
}


```

这段代码定义了一个名为 "WritePRD" 的类，它实现了 Python 标准中的 "Action" 接口。这个类的父类是 "base.Action"，因此在创建 "WritePRD" 类的实例时，会继承自 "base.Action" 的构造函数。

"WritePRD" 类的构造函数包含三个参数，分别是一个字符串参数 "name"、一个指向 "WritePRD" 类的实例的引用、以及一个指向 "llm"(可能是日志管理器)的引用。这些参数将在实例创建时传递给父类的构造函数。

"WritePRD" 类包含一个名为 "run" 的方法，该方法实现了 "Action" 接口的 "run" 方法。这个方法的参数包括两个参数：一个包含 "requirements"(可能是用于搜索和摘要搜索的信息)的参数，以及一个格式字符串和一个或多个参数。"format"参数指定了格式字符串中使用的字段。

当 "run" 方法被调用时，它将执行以下操作：首先，创建一个名为 "sas" 的 "SearchAndSummarize" 类的实例。"sas" 将会在上下文中执行 "run" 方法，传递一个包含 "requirements" 的参数和一个字符串参数 "system_text"，这个参数指定了要搜索的系统文本。

然后，"sas" 将返回一个字符串 "rsp"，其中包含搜索结果和摘要。如果 "sas.result":

```py
rsp = "Search Results:\n[SEARCH_AND_SUMMARIZE_SYSTEM_EN_US system_text=\"System性能分析结果：\n{\\"system_performance_analysis_results\":[{\"system_performance_analysis_result\":\"success\",\"description\":\"This system has been successfully configured\"}]}](SEARCH_AND_SUMMARIZE_SYSTEM_EN_US%E3%80%82%5BSEARCH_AND_SUMMARIZE_SYSTEM_EN_US%5D%5BSEARCH_AND_SUMMARIZE_SYSTEM_EN_US%5D%5BSEARCH_AND_SUMMARIZE_SYSTEM_EN_US)
```

否则，它将返回一个空字符串。

最后，"write_prd" 类的实例将使用 "asana_sdk" 库中的 "WriteAPIDoc" 函数，将搜索结果和摘要写入一个 "APIDoc" 对象中，并将 "format" 参数作为参数传递。"


```py
class WritePRD(Action):
    def __init__(self, name="", context=None, llm=None):
        super().__init__(name, context, llm)

    async def run(self, requirements, format=CONFIG.prompt_format, *args, **kwargs) -> ActionOutput:
        sas = SearchAndSummarize()
        # rsp = await sas.run(context=requirements, system_text=SEARCH_AND_SUMMARIZE_SYSTEM_EN_US)
        rsp = ""
        info = f"### Search Results\n{sas.result}\n\n### Search Summary\n{rsp}"
        if sas.result:
            logger.info(sas.result)
            logger.info(rsp)

        prompt_template, format_example = get_template(templates, format)
        prompt = prompt_template.format(
            requirements=requirements, search_information=info, format_example=format_example
        )
        logger.debug(prompt)
        # prd = await self._aask_v1(prompt, "prd", OUTPUT_MAPPING)
        prd = await self._aask_v1(prompt, "prd", OUTPUT_MAPPING, format=format)
        return prd

```