# MetaGPT源码解析 5

# `metagpt/actions/write_prd_review.py`

该代码是一个Python脚本，用于编写PRD Review。它实现了MetagPT（一个自动元编程框架）中的一個Action。

具体来说，该脚本定义了一个名为WritePRDReview的类，继承自Action类。在WritePRDReview类中，初始化了一个名为self的类属性，包括一个指向产品需求文档（PRD）的引用self.prd，一个desc属性用于存储PRD Review的描述，以及一个prd_review_prompt_template属性用于存储PRD Review的提示模板。

在该脚本的run方法中，首先将传入的PRD实例化，然后使用该PRD创建一个review，最后返回review。在该review的实现中，调用了另一个方法_aask，这个方法似乎实现了与GPT模型的交互以获取用户的输入，并在review中进行了应用。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:45
@Author  : alexanderwu
@File    : write_prd_review.py
"""
from metagpt.actions.action import Action


class WritePRDReview(Action):
    def __init__(self, name, context=None, llm=None):
        super().__init__(name, context, llm)
        self.prd = None
        self.desc = "Based on the PRD, conduct a PRD Review, providing clear and detailed feedback"
        self.prd_review_prompt_template = """
        Given the following Product Requirement Document (PRD):
        {prd}

        As a project manager, please review it and provide your feedback and suggestions.
        """

    async def run(self, prd):
        self.prd = prd
        prompt = self.prd_review_prompt_template.format(prd=self.prd)
        review = await self._aask(prompt)
        return review
    
```

# `metagpt/actions/write_test.py`

这段代码是一个Python脚本，用于执行一个测试用例。它包含以下几个主要部分：

1. 导入所需的模块：通过导入metagpt.actions、metagpt.logs和metagpt.utils.common模块，可以实现对相关功能的支持。
2. 定义了一个名为Action的类，该类继承自metagpt.actions.action.Action。这个类将负责执行后续操作。
3. 定义了一个名为logger的类，来自metagpt.logs模块。通过创建一个logger实例，可以输出消息到日志文件。
4. 定义了一个名为CodeParser的类，来自metagpt.utils.common模块。通过创建一个CodeParser实例，可以解析特定的代码格式。
5. 在脚本的顶部，定义了一个名为PROMPT_TEMPLATE的常量。这个常量定义了一个模板，用于在执行测试用例时生成通知。
6. 在脚本中，使用logger.getLogger()方法创建了一个logger实例。然后，通过logger.initLogger()方法设置了日志的源。
7. 通过CodeParser.parseFile(文件路径，代码格式要求)方法，解析了指定文件夹中的代码。
8. 在角色和需求变量中，使用了格式化字符串，生成了一个通知。
9. 通过Action.run()方法，将生成的通知发送给了logger。
10. 最后，在脚本的结尾，使用logger.shutdown()方法关闭了logger实例。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 22:12
@Author  : alexanderwu
@File    : environment.py
"""
from metagpt.actions.action import Action
from metagpt.logs import logger
from metagpt.utils.common import CodeParser

PROMPT_TEMPLATE = """
NOTICE
1. Role: You are a QA engineer; the main goal is to design, develop, and execute PEP8 compliant, well-structured, maintainable test cases and scripts for Python 3.9. Your focus should be on ensuring the product quality of the entire project through systematic testing.
2. Requirement: Based on the context, develop a comprehensive test suite that adequately covers all relevant aspects of the code file under review. Your test suite will be part of the overall project QA, so please develop complete, robust, and reusable test cases.
```



This code provides some guidelines and best practices for writing test cases using the Python unittest framework.

The main attention1 instruction is to use `##` to split sections instead of `#`, and to write `## <SECTION_NAME>` before each test case or script.

Attention2 tells us to always set a default value for any settings in tests, and to use strong typing and explicit variables.

Attention3 reminds us to follow the "Data structures and interface definitions" guidelines, and not to change the existing design of the code in any way. Additionally, it asks to be mindful of any edge cases that may exist, and to make sure that the tests respect the existing design and ensure its validity.

The last instruction is to think before writing, and to carefully check that they don't miss any necessary test cases or scripts in the file.


```py
3. Attention1: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the test case or script.
4. Attention2: If there are any settings in your tests, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE.
5. Attention3: YOU MUST FOLLOW "Data structures and interface definitions". DO NOT CHANGE ANY DESIGN. Make sure your tests respect the existing design and ensure its validity.
6. Think before writing: What should be tested and validated in this document? What edge cases could exist? What might fail?
7. CAREFULLY CHECK THAT YOU DON'T MISS ANY NECESSARY TEST CASES/SCRIPTS IN THIS FILE.
Attention: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the test case or script and triple quotes.
-----
## Given the following code, please write appropriate test cases using Python's unittest framework to verify the correctness and robustness of this code:
```python
{code_to_test}
```py
Note that the code to test is at {source_file_path}, we will put your test code at {workspace}/tests/{test_file_name}, and run your test code from {workspace},
you should correctly import the necessary classes based on these file locations!
## {test_file_name}: Write test code with triple quoto. Do your best to implement THIS ONLY ONE FILE.
"""


```

该代码定义了一个名为 WriteTest 的动作(Action)，用于编写测试代码。该动作实现了两个方法：write_code 和 run。下面分别解释这两个方法的作用。

1. write_code 方法的作用是接收一个提示(prompt)，然后编写代码并返回。代码编写过程中可能会遇到一些问题，因此这个方法也包含了异常处理。如果代码编写成功，则会返回编写好的代码。如果出现异常，则会记录下来并返回之前的代码，这样就可以在需要时进行调试。

2. run 方法的作用是接收要测试的代码、测试文件名、源文件路径和一个工作区(workspace)。这个方法会将代码文件解析为可执行代码，并运行它。如果测试失败，则会记录下来。

该代码的实现是为了提供一个用于编写测试代码的类，方便在测试过程中编写和运行测试代码。


```py
class WriteTest(Action):
    def __init__(self, name="WriteTest", context=None, llm=None):
        super().__init__(name, context, llm)

    async def write_code(self, prompt):
        code_rsp = await self._aask(prompt)

        try:
            code = CodeParser.parse_code(block="", text=code_rsp)
        except Exception:
            # Handle the exception if needed
            logger.error(f"Can't parse the code: {code_rsp}")

            # Return code_rsp in case of an exception, assuming llm just returns code as it is and doesn't wrap it inside ```
            code = code_rsp
        return code

    async def run(self, code_to_test, test_file_name, source_file_path, workspace):
        prompt = PROMPT_TEMPLATE.format(
            code_to_test=code_to_test,
            test_file_name=test_file_name,
            source_file_path=source_file_path,
            workspace=workspace,
        )
        code = await self.write_code(prompt)
        return code

```py

# `metagpt/actions/write_tutorial.py`

这段代码是一个Python脚本，用于实现一个Tutorial Assistant。该Assistant具有创建文件夹和编辑文件的内容的功能。具体来说，该Assistant可以接受根目录和内容相关的Prompt，创建或编辑指定的目录和文件。

具体来说，该代码首先导入了所需的类和函数，包括Action、DIRECTORY_PROMPT和CONTENT_PROMPT以及OutputParser。然后，定义了Assistant类，该类实现了Action接口，实现了创建和编辑目录和文件的功能。

在Assistant类的__init__方法中，定义了要将所有内容存储在一个名为assistant_dir的元组中。这个元组包含了当前工作目录（即脚本所在的目录），以及一些工具和文件，用于处理Assistant的各种行为。

接着，定义了几个Prompt接口，分别用于创建和编辑目录和文件。这些Prompt类实现了Action接口，并实现了与用户交互的功能。

最重要的是，在脚本的最后部分，定义了一个单个的函数，用于创建一个新的目录。这个函数接受用户输入的目录名称作为参数，并将它添加到assistant_dir中。

总之，这段代码实现了一个简单的学生助手，可以帮助用户创建或编辑各种文件夹和文档，以完成各种学习任务。


```
#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
@Time    : 2023/9/4 15:40:40
@Author  : Stitch-z
@File    : tutorial_assistant.py
@Describe : Actions of the tutorial assistant, including writing directories and document content.
"""

from typing import Dict

from metagpt.actions import Action
from metagpt.prompts.tutorial_assistant import DIRECTORY_PROMPT, CONTENT_PROMPT
from metagpt.utils.common import OutputParser


```py

这段代码定义了一个名为 "WriteDirectory" 的类，它实现了 "Action" 类。这个类的参数包括一个字符串参数 "name" 和一个字符参数 "language"，它们用于指定这个动作要执行的名字和输出的语言。

在类的初始化方法 "__init__" 中，我们调用了父类 "Action" 的 "**init**" 方法，并传入了 "name" 和 "language" 参数。这样就实现了父类所定义的所有方法。

类的 "run" 方法是一个异步方法，它接收一个字符串参数 "topic"，并使用 "DIRECTORY_PROMPT" 和 "language" 变量来生成一个与主题相关的目录结构。这个目录结构通常是类似于这样的：
```css
{
   "title": "xxx",
   "directory": [{"dir 1": ["sub dir 1", "sub dir 2"]}]
}
```py
它的具体实现包括两个步骤：
1. 使用 `DIRECTORY_PROMPT` 函数将主题 "topic" 和语言 "language" 生成一个字符串，这个字符串会被用来生成目录结构。
2. 使用 `_aask` 函数发送这个主题和语言的字符串，获取一个 HTTP 响应。
3. 使用 `OutputParser` 类提取出这个响应中的数据。
4. 使用 `format` 方法将主题和语言插入到字符串中，生成一个与主题相关的目录结构。

最终，这个类的实例可以用 `action = WriteDirectory(name="你的名字", language="你的语言")` 来创建，它会执行 `run` 方法来生成一个符合主题 "你的名字" 和语言 "你的语言" 的目录结构。


```
class WriteDirectory(Action):
    """Action class for writing tutorial directories.

    Args:
        name: The name of the action.
        language: The language to output, default is "Chinese".
    """

    def __init__(self, name: str = "", language: str = "Chinese", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.language = language

    async def run(self, topic: str, *args, **kwargs) -> Dict:
        """Execute the action to generate a tutorial directory according to the topic.

        Args:
            topic: The tutorial topic.

        Returns:
            the tutorial directory information, including {"title": "xxx", "directory": [{"dir 1": ["sub dir 1", "sub dir 2"]}]}.
        """
        prompt = DIRECTORY_PROMPT.format(topic=topic, language=self.language)
        resp = await self._aask(prompt=prompt)
        return OutputParser.extract_struct(resp, dict)


```py

这段代码定义了一个名为 "WriteContent" 的动作类，用于编写教程内容。这个类包含了两个参数：名称（String）和内容目录（String）。默认情况下，语言设置为 "Chinese"。

在类的初始化方法 "__init__" 中，首先调用父类的初始化方法，然后设置自己的参数。接着，定义了 "run" 方法，这个方法会在给定的目录和主题下执行 WriteContent 类别的动作，并返回写作的内容。

"run" 方法的实现比较简单，直接使用了 "asyncio" 库中的 "Await" 和 "return" 方法。确保了 "write_doc" 方法可以正确地写入目录内容。


```
class WriteContent(Action):
    """Action class for writing tutorial content.

    Args:
        name: The name of the action.
        directory: The content to write.
        language: The language to output, default is "Chinese".
    """

    def __init__(self, name: str = "", directory: str = "", language: str = "Chinese", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.language = language
        self.directory = directory

    async def run(self, topic: str, *args, **kwargs) -> str:
        """Execute the action to write document content according to the directory and topic.

        Args:
            topic: The tutorial topic.

        Returns:
            The written tutorial content.
        """
        prompt = CONTENT_PROMPT.format(topic=topic, language=self.language, directory=self.directory)
        return await self._aask(prompt=prompt)


```py

# `metagpt/actions/__init__.py`

这段代码定义了一个 Enum 对象，名为 Metadata，包含了以下字段：

```
from typing import Any, Enum, Union
```py

```
@Time    : 2023/5/11 17:44
@Author  : alexanderwu
@File    : __init__.py
```py

定义了一个 MetadataDesign 类，属于 Metadata 类。在类内实现了以下方法：

```
from enum import Enum
from typing import Any, Enum, Union
class MetadataDesign(Enum):
   GENERATED_TESTS = Union[str, Enum[str]]
   GENERATED_COPY = Union[str, Enum[str]]
   TESTS_IN_PLACE = Union[str, Enum[str]]

   def __post_init__(self):
       """告貰ma告蒖楔？"`
       pass

   def __str__(self):
       """定义从这里，我主小康胶氦蠖箔"""
       pass
```py

创建了一个 Metadata 类，实现了 `__post_init__` 和 `__str__` 方法。

```
from metagpt.actions.action import Action
from metagpt.actions.action_output import ActionOutput
from metagpt.actions.add_requirement import BossRequirement
from metagpt.actions.debug_error import DebugError
from metagpt.actions.design_api import WriteDesign
from metagpt.actions.design_api_review import DesignReview

class Metadata:
   def __init__(self, name: str):
       self.name = name
       self.design_api: WriteDesign = DebugError(1)
       self.design_api_review: DesignReview = DebugError(2)
```py

```
   def add_requirement(self, requirement: BossRequirement):
       pass

   def test_in_place(self, alternative: str = "") -> None:
       pass
```py

从 `Metadata` 类来看，它实现了 `__post_init__` 和 `__str__` 方法。

```
   def __post_init__(self):
       pass

   def __str__(self):
       return f"Metadata({self.name}))"
```py

在这两个方法中，第一个方法是 `__post_init__`，用于自定义初始化，但是在这里并没有实现任何自定义逻辑。第二个方法 `__str__` 是 `__post_init__` 的别称，用于打印 `Metadata` 对象的字符串表示。

```
       self.design_api: WriteDesign = DebugError(1)
       self.design_api_review: DesignReview = DebugError(2)
```py

在 `Metadata` 创建者方法的实现中，通过 `WriteDesign` 和 `DesignReview` 对象实现了 `metagpt.actions.design_api` 和 `metagpt.actions.design_api_review`。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:44
@Author  : alexanderwu
@File    : __init__.py
"""
from enum import Enum

from metagpt.actions.action import Action
from metagpt.actions.action_output import ActionOutput
from metagpt.actions.add_requirement import BossRequirement
from metagpt.actions.debug_error import DebugError
from metagpt.actions.design_api import WriteDesign
from metagpt.actions.design_api_review import DesignReview
```py

这段代码是一个自动执行的Python代码，它的作用是执行一系列的任务，包括添加要求、创建产品文件夹、运行代码、收集链接、集成网页、处理数据、编写测试和创建产品需求等等。

具体来说，这段代码分为以下几个部分：

1. 导入需要的类和函数：从metagpt.actions.design_filenames开始导入DesignFilenames，然后从metagpt.actions.project_management开始导入AssignTasks和WriteTasks，从metagpt.actions.research开始导入CollectLinks、WebBrowseAndSummarize和ConductResearch，从metagpt.actions.run_code开始导入RunCode，从metagpt.actions.search_and_summarize开始导入SearchAndSummarize，从metagpt.actions.write_code开始导入WriteCode，从metagpt.actions.write_code_review开始导入WriteCodeReview，从metagpt.actions.write_prd开始导入WritePRD，从metagpt.actions.write_prd_review开始导入WritePRDReview，从metagpt.actions.write_test开始导入WriteTest。

2. 定义了一个ActionType类：用于区分不同的动作类型，例如ADD\_REQUIREMENT、WRITE\_PRD、DESIGN\_REVIEW等等。

3. 从metagpt.actions.DesignFilenames类中创建DesignFilenames实例。

4. 从metagpt.actions.ProjectManagement类中创建AssignTasks和WriteTasks实例。

5. 从metagpt.actions.Research类中创建CollectLinks、WebBrowseAndSummarize和ConductResearch实例。

6. 从metagpt.actions.RunCode类中创建RunCode实例。

7. 从metagpt.actions.SearchAndSummarize类中创建SearchAndSummarize实例。

8. 从metagpt.actions.WriteCode类中创建WriteCode实例。

9. 从metagpt.actions.WriteCodeReview类中创建WriteCodeReview实例。

10. 从metagpt.actions.WritePRD类中创建WritePRD实例。

11. 从metagpt.actions.WritePRDReview类中创建WritePRDReview实例。

12. 从metagpt.actions.WriteTest类中创建WriteTest实例。

13. 将这些实例的结果进行汇总，得到一个True或False的值，表示任务的完成情况。


```
from metagpt.actions.design_filenames import DesignFilenames
from metagpt.actions.project_management import AssignTasks, WriteTasks
from metagpt.actions.research import CollectLinks, WebBrowseAndSummarize, ConductResearch
from metagpt.actions.run_code import RunCode
from metagpt.actions.search_and_summarize import SearchAndSummarize
from metagpt.actions.write_code import WriteCode
from metagpt.actions.write_code_review import WriteCodeReview
from metagpt.actions.write_prd import WritePRD
from metagpt.actions.write_prd_review import WritePRDReview
from metagpt.actions.write_test import WriteTest


class ActionType(Enum):
    """All types of Actions, used for indexing."""

    ADD_REQUIREMENT = BossRequirement
    WRITE_PRD = WritePRD
    WRITE_PRD_REVIEW = WritePRDReview
    WRITE_DESIGN = WriteDesign
    DESIGN_REVIEW = DesignReview
    DESIGN_FILENAMES = DesignFilenames
    WRTIE_CODE = WriteCode
    WRITE_CODE_REVIEW = WriteCodeReview
    WRITE_TEST = WriteTest
    RUN_CODE = RunCode
    DEBUG_ERROR = DebugError
    WRITE_TASKS = WriteTasks
    ASSIGN_TASKS = AssignTasks
    SEARCH_AND_SUMMARIZE = SearchAndSummarize
    COLLECT_LINKS = CollectLinks
    WEB_BROWSE_AND_SUMMARIZE = WebBrowseAndSummarize
    CONDUCT_RESEARCH = ConductResearch


```py

这段代码定义了一个名为 `__all__` 的列表，包含了四个元素，分别是 `ActionType`, `Action`, `ActionOutput`。这个列表的作用是告诉 Python 执行时自动按照顺序加载这些变量，而不需要使用 `**` 或者 `isin` 方法。

换句话说，这段代码定义了一个临时的变量，可以让 Python 在运行时自动加载 `ActionType`, `Action`, `ActionOutput` 变量，而无需在运行时使用 `**` 或者 `isin` 方法。这个列表通常用于 ismap 或者自然的上下文中，以便于 Python 在运行时自动创建或者查找变量。


```
__all__ = [
    "ActionType",
    "Action",
    "ActionOutput",
]

```py

# `metagpt/document_store/base_store.py`

这段代码定义了一个名为 "BaseStore" 的抽象类，继承自 "ABC"（Abstract base class）和 "metagpt.config.Config"（metagpt 配置类）的类。这个抽象类包含三个抽象方法 "search"、"write" 和 "add"，用于在未实现具体方法的情况下定义了这些方法的行为。

具体来说，这段代码的作用是定义了一个抽象的基存储类，用于在将来的应用程序中提供搜索、写入和添加数据的功能。具体实现细节由子类负责，基类提供了一些通用的接口。这个抽象类可以被用于任何需要这些基存储服务的应用程序中。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/28 00:01
@Author  : alexanderwu
@File    : base_store.py
"""
from abc import ABC, abstractmethod
from pathlib import Path

from metagpt.config import Config


class BaseStore(ABC):
    """FIXME: consider add_index, set_index and think about granularity."""

    @abstractmethod
    def search(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def write(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def add(self, *args, **kwargs):
        raise NotImplementedError


```py

这段代码定义了一个名为 LocalStore 的类，继承自 BaseStore 和 ABC 类。LocalStore 类包含一个初始化方法(__init__)，该方法接受两个参数：raw_data 和 cache_dir。

在初始化方法中，首先检查 raw_data 是否为空，如果是，则抛出一个 FileNotFoundError。然后，初始化 Config 对象和 raw_data。接着，如果还没有指定 cache_dir，则将 cache_dir 设置为 raw_data 的父目录。最后，使用 _load() 和 _write() 方法来加载和保存数据。

_get_index_and_store_fname() 方法用于获取索引文件和存储文件路径，该方法首先将 fname 从 raw_data 中分离出来，然后将 fname 作为文件名，分别生成索引文件和存储文件路径。

_load() 和 _write() 方法都是 abstract 方法，这意味着它们的方法体内没有具体的实现。这些方法在 LocalStore 类中被声明为 abstract，这意味着它们必须在子类中实现。


```
class LocalStore(BaseStore, ABC):
    def __init__(self, raw_data: Path, cache_dir: Path = None):
        if not raw_data:
            raise FileNotFoundError
        self.config = Config()
        self.raw_data = raw_data
        if not cache_dir:
            cache_dir = raw_data.parent
        self.cache_dir = cache_dir
        self.store = self._load()
        if not self.store:
            self.store = self.write()

    def _get_index_and_store_fname(self):
        fname = self.raw_data.name.split('.')[0]
        index_file = self.cache_dir / f"{fname}.index"
        store_file = self.cache_dir / f"{fname}.pkl"
        return index_file, store_file

    @abstractmethod
    def _load(self):
        raise NotImplementedError

    @abstractmethod
    def _write(self, docs, metadatas):
        raise NotImplementedError
    
```py

# `metagpt/document_store/chromadb_store.py`

这段代码定义了一个名为`ChromaStore`的类，用于在Chromadb中存储和检索数据。该类包含以下方法：

- `__init__`方法：用于初始化ChromaStore对象并设置其名称。该方法使用Chromadb客户端创建一个存储桶对象，并将其保存到实例变量中。
- `search`方法：用于搜索给定查询语句下的文档。该方法可以进行 optional 过滤，例如使用`metadata_filter`和`document_filter`参数进行元数据和文档筛选。
- `persist`方法：该方法是Chroma存储桶的默认方法。它的作用类似于接口，但实际上会在运行时发生以下操作：打印错误消息并返回。这是因为Chroma存储桶是一个客户端API，它并不支持本地数据持久化。
- `write`方法：用于将多个文档、元数据和ID添加到存储桶中。该方法实现了一个类似于Chroma存储桶原子的功能，但该方法对于多个文档、元数据和ID的插入进行了限制，这意味着它无法在单个调用中同时更改它们。
- `add`方法：用于将单个文档、元数据和ID添加到存储桶中。该方法实现了一个类似于Chroma存储桶原子的功能，但该方法对于单个文档、元数据和ID的插入进行了限制，这意味着它无法在单个调用中同时更改它们。
- `delete`方法：用于从存储桶中删除文档、元数据和ID。该方法实现了一个类似于Chroma存储桶原子的功能，但该方法对于文档、元数据的删除进行了限制，这意味着它无法在单个调用中同时更改它们。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/29 14:46
@Author  : alexanderwu
@File    : chromadb_store.py
"""
import chromadb


class ChromaStore:
    """If inherited from BaseStore, or importing other modules from metagpt, a Python exception occurs, which is strange."""
    def __init__(self, name):
        client = chromadb.Client()
        collection = client.create_collection(name)
        self.client = client
        self.collection = collection

    def search(self, query, n_results=2, metadata_filter=None, document_filter=None):
        # kwargs can be used for optional filtering
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=metadata_filter,  # optional filter
            where_document=document_filter  # optional filter
        )
        return results

    def persist(self):
        """Chroma recommends using server mode and not persisting locally."""
        raise NotImplementedError

    def write(self, documents, metadatas, ids):
        # This function is similar to add(), but it's for more generalized updates
        # It assumes you're passing in lists of docs, metadatas, and ids
        return self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def add(self, document, metadata, _id):
        # This function is for adding individual documents
        # It assumes you're passing in a single doc, metadata, and id
        return self.collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[_id],
        )

    def delete(self, _id):
        return self.collection.delete([_id])

```py

# `metagpt/document_store/document.py`

该代码是一个Python脚本，用于执行以下操作：

1. 导入所需的库：pandas、 langchain.document_loaders。
2. 创建一个文档类 Document，其中包含一个文本加载器和一个文档加载器，分别用于从文件中读取文本和从 langchain 文档库中读取文档。
3. 定义 Document 类，继承自 langchain.document_loaders.Document类，实现 load_document 方法。
4. 在脚本中包含一个 from langchain.document_loaders import 的语句，用于从 langchain 文档库中导入 document_loaders 函数。
5. 在脚本中包含一个 from pathlib import 的语句，用于从 pathlib 库中导入路径函数。
6. 在脚本中包含一个 main 函数，其中包含一个 Document 实例化的操作，将其保存为 /path/to/output/file。

该脚本的主要目的是执行以下操作：

1. 读取文件中的文本并将其存储为 DataFrame。
2. 读取 langchain 文档库中的文档并将其存储为 Document 对象。
3. 将 Document 对象保存为 /path/to/output/file。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/6/8 14:03
@Author  : alexanderwu
@File    : document.py
"""
from pathlib import Path

import pandas as pd
from langchain.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
)
```py

这段代码的主要作用是读取数据并对其进行验证，如果读取失败则抛出异常。具体实现包括从不同文件格式中读取数据，例如从Excel、CSV、JSON、Word文档和PDF文件中读取数据。对于每种文件格式，代码会根据其特点对数据进行不同的处理，例如在读取Excel和CSV文件时，代码会使用pandas库读取并返回数据；在读取Word文档和PDF文件时，代码会使用UnstructuredWordDocumentLoader和UnstructuredPDFLoader对数据进行解析。在代码中，还定义了一个函数validate_cols，用于验证数据文件列是否与read_data函数返回的DataFrame列一一对应。


```
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm


def validate_cols(content_col: str, df: pd.DataFrame):
    if content_col not in df.columns:
        raise ValueError


def read_data(data_path: Path):
    suffix = data_path.suffix
    if '.xlsx' == suffix:
        data = pd.read_excel(data_path)
    elif '.csv' == suffix:
        data = pd.read_csv(data_path)
    elif '.json' == suffix:
        data = pd.read_json(data_path)
    elif suffix in ('.docx', '.doc'):
        data = UnstructuredWordDocumentLoader(str(data_path), mode='elements').load()
    elif '.txt' == suffix:
        data = TextLoader(str(data_path)).load()
        text_splitter = CharacterTextSplitter(separator='\n', chunk_size=256, chunk_overlap=0)
        texts = text_splitter.split_documents(data)
        data = texts
    elif '.pdf' == suffix:
        data = UnstructuredPDFLoader(str(data_path), mode="elements").load()
    else:
        raise NotImplementedError
    return data


```py

这段代码定义了一个名为 `Document` 的类，用于读取和处理文档和元数据。

在 `__init__` 方法中，首先调用 `read_data` 函数从指定的数据文件中读取数据，并将读取到的数据赋值给 `self.data` 变量。然后检查 `self.data` 是否为 pandas DataFrame，如果是，则调用 `validate_cols` 函数对内容列进行验证。接下来，将 `content_col` 和 `meta_col` 设置为数据中包含文档和元数据的列的名称。

接着，定义了两个辅助方法 `_get_docs_and_metadatas_by_df` 和 `_get_docs_and_metadatas_by_langchain`，用于获取文档和元数据。这两个方法都是异步方法，使用了 `tqdm` 库来跟踪进度。具体来说，`_get_docs_and_metadatas_by_df` 方法对数据中的每个文档，使用 `df[self.content_col].iloc[i]` 提取文档中的内容，并检查是否定义了 `self.meta_col`。如果是，则使用 `df[self.meta_col].iloc[i]` 提取元数据。否则，创建一个空字典。

`_get_docs_and_metadatas_by_langchain` 方法对数据中的每个文档，使用一个列表推导式来获取该文档的元数据。

最后，在 `get_docs_and_metadatas` 方法中，根据 `self.data` 的类型来调用 `_get_docs_and_metadatas_by_df` 或 `_get_docs_and_metadatas_by_langchain` 方法，根据返回的数据类型对它进行正确的处理。如果没有定义 `self.data`，则会引发 `NotImplementedError`。


```
class Document:

    def __init__(self, data_path, content_col='content', meta_col='metadata'):
        self.data = read_data(data_path)
        if isinstance(self.data, pd.DataFrame):
            validate_cols(content_col, self.data)
        self.content_col = content_col
        self.meta_col = meta_col

    def _get_docs_and_metadatas_by_df(self) -> (list, list):
        df = self.data
        docs = []
        metadatas = []
        for i in tqdm(range(len(df))):
            docs.append(df[self.content_col].iloc[i])
            if self.meta_col:
                metadatas.append({self.meta_col: df[self.meta_col].iloc[i]})
            else:
                metadatas.append({})

        return docs, metadatas

    def _get_docs_and_metadatas_by_langchain(self) -> (list, list):
        data = self.data
        docs = [i.page_content for i in data]
        metadatas = [i.metadata for i in data]
        return docs, metadatas

    def get_docs_and_metadatas(self) -> (list, list):
        if isinstance(self.data, pd.DataFrame):
            return self._get_docs_and_metadatas_by_df()
        elif isinstance(self.data, list):
            return self._get_docs_and_metadatas_by_langchain()
        else:
            raise NotImplementedError
        
```py

# `metagpt/document_store/faiss_store.py`

这段Python代码定义了一个名为"faiss_store.py"的文件，其中包含一个导入自'faiss'包的函数。这个函数的主要作用是加载一个已经训练好的词汇表(word embedding)，并将其存储在一个FAISS库中，以便后续的文本分析任务使用。

具体来说，这个函数接受两个参数。第一个参数是一个字符串'/path/to/trained/embeddings/vocab.txt'，这个文件中包含已经训练好的词汇表。第二个参数是一个Optional的类型，表示是否要加载词汇表中的词汇。

在函数内部，首先使用'import faiss'导入'faiss'包。然后使用'pickle.load'函数从文件中读取已经训练好的词汇表。接着，使用'langchain.embeddings'包中的'OpenAIEmbeddings'类将词汇表中的词汇转换为OpenAIE embeddings。最后，使用'langchain.vectorstores'包中的'FAISS'类将OpenAIE embeddings存储到FAISS库中。

整个函数的目的是提供一个简单的方法来加载已经训练好的词汇表，以便在需要进行文本分析任务时使用。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/25 10:20
@Author  : alexanderwu
@File    : faiss_store.py
"""
import pickle
from pathlib import Path
from typing import Optional

import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

```py

This is a class that implements the OpenAI-IE (Interactive Explainable AI) framework. It is based on the诺曼模式，即豹子模式下的产品经理-架构师模式。

The class is a product manager that manages an AI explainable system. It uses the OpenAI API to interact with the system, and provides methods to enhance, maintain, and retrieve the AI explainable system.

The `add` method is used to add new texts to the store. The `delete` method is not yet implemented.

The `write` method is responsible for persisting the store to a file, and updating the index file and index.

The `persist` method is used to update the store, and it will save any changes made to the store to disk.

The `search` method is used to search for similar documents based on a query.

The `metadatas` property is a dictionary that stores the metadata of the documents.

The class also has a property `query_engine` that is responsible for rendering the queries to the user.

The `use_storing_device` property controls whether the store should be used or not.


```
from metagpt.const import DATA_PATH
from metagpt.document_store.base_store import LocalStore
from metagpt.document_store.document import Document
from metagpt.logs import logger


class FaissStore(LocalStore):
    def __init__(self, raw_data: Path, cache_dir=None, meta_col='source', content_col='output'):
        self.meta_col = meta_col
        self.content_col = content_col
        super().__init__(raw_data, cache_dir)

    def _load(self) -> Optional["FaissStore"]:
        index_file, store_file = self._get_index_and_store_fname()
        if not (index_file.exists() and store_file.exists()):
            logger.info("Missing at least one of index_file/store_file, load failed and return None")
            return None
        index = faiss.read_index(str(index_file))
        with open(str(store_file), "rb") as f:
            store = pickle.load(f)
        store.index = index
        return store

    def _write(self, docs, metadatas):
        store = FAISS.from_texts(docs, OpenAIEmbeddings(openai_api_version="2020-11-07"), metadatas=metadatas)
        return store

    def persist(self):
        index_file, store_file = self._get_index_and_store_fname()
        store = self.store
        index = self.store.index
        faiss.write_index(store.index, str(index_file))
        store.index = None
        with open(store_file, "wb") as f:
            pickle.dump(store, f)
        store.index = index

    def search(self, query, expand_cols=False, sep='\n', *args, k=5, **kwargs):
        rsp = self.store.similarity_search(query, k=k, **kwargs)
        logger.debug(rsp)
        if expand_cols:
            return str(sep.join([f"{x.page_content}: {x.metadata}" for x in rsp]))
        else:
            return str(sep.join([f"{x.page_content}" for x in rsp]))

    def write(self):
        """Initialize the index and library based on the Document (JSON / XLSX, etc.) file provided by the user."""
        if not self.raw_data.exists():
            raise FileNotFoundError
        doc = Document(self.raw_data, self.content_col, self.meta_col)
        docs, metadatas = doc.get_docs_and_metadatas()

        self.store = self._write(docs, metadatas)
        self.persist()
        return self.store

    def add(self, texts: list[str], *args, **kwargs) -> list[str]:
        """FIXME: Currently, the store is not updated after adding."""
        return self.store.add_texts(texts)

    def delete(self, *args, **kwargs):
        """Currently, langchain does not provide a delete interface."""
        raise NotImplementedError


```py

这段代码使用了Python的Faiss库来执行文件操作。

首先，在代码中定义了一个名为'__main__'的模块，这是一个特殊的模块，表示当前文件已经被激活，可以执行一些其他任务。

如果当前文件是 __main__ 文件，那么就会执行以下代码。

1. 导入 FaissStore 类，并将其存储在名为 faiss_store 的变量中。
2. 获取 DATA_PATH 变量中的文件名，并将其除以'/'得到文件名。
3. 调用 FaissStore 类的 search 方法，以检索文件中包含 'Oily Skin Facial Cleanser' 的文档。
4. 将 'Oily Skin Facial Cleanser' 字符串转换为数字序列，从 0 到 2。
5. 将数字序列列表添加到 faiss_store 中。
6. 再次调用 FaissStore 类的 search 方法，以检索文件中包含所有词汇的文档，包括之前的文档。

这段代码的作用是执行一个面部清洁剂的检测，并在检测到包含特定词汇的文档时输出一条消息。


```
if __name__ == '__main__':
    faiss_store = FaissStore(DATA_PATH / 'qcs/qcs_4w.json')
    logger.info(faiss_store.search('Oily Skin Facial Cleanser'))
    faiss_store.add([f'Oily Skin Facial Cleanser-{i}' for i in range(3)])
    logger.info(faiss_store.search('Oily Skin Facial Cleanser'))

```py

# `metagpt/document_store/lancedb_store.py`

`Lance` is a library for storing and retrieving data using the Lane有限元数据库. It supports several operations including `create_table`, `insert`, `update`, and `delete`.

The `create_table` function is used to create a table in the Lane database. It takes in the name and metadata of the table, and returns a handle to that table.

The `insert` function is used to insert a new document into a table in the Lane database. It takes in the data for the document and the table name, and adds the document to the table.

The `update` function is used to update a document in a table in the Lane database. It takes in the document, the table name, and the new data, and updates the document accordingly.

The `delete` function is used to delete a document from a table in the Lane database. It takes in the id of the document to be deleted, and returns a boolean indicating whether the document was deleted successfully.

The `persist` function is used to persist data from the Lane database to disk. It should be implemented in order to persist the data to disk, such as using the `write` method.

The `write` method is used to write data from the Lane database to disk. It takes in the data to write, the metadatas to write to, and the id of the table in the Lane database. It will write the data to the specified table in the Lane database, or create it if the table doesn't exist.

The `add` method is used to add data to a table in the Lane database. It takes in the data, the metadata for the data, and the id of the table in the Lane database. It should be implemented in order to insert the data into the table in the Lane database.

The `delete` method is used to delete a document from a table in the Lane database. It takes in the id of the document to be deleted and returns a boolean indicating whether the document was deleted successfully.

The `open` method is used to open the Lane database file.

The `close` method is used to close the Lane database file.

The `abort` method is used to abort an operation in the Lane database.

The `lookup` method is used to look up a file in the Lane database by its path.

The `directory` property is used to get a list of all the files and directories in the Lane database.

The `is_directory` property is used to determine if an object is a directory or not.

The `listdir` method is used to list the contents of a directory in the Lane database.

The `subdir` method is used to list the contents of a subdirectory in the Lane database.

The `make_directory` method is used to create a directory in the Lane database.

The `rm_directory` method is used to remove a directory and all its contents in the Lane database.


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/9 15:42
@Author  : unkn-wn (Leon Yee)
@File    : lancedb_store.py
"""
import os
import shutil

import lancedb


class LanceStore:
    def __init__(self, name):
        db = lancedb.connect("./data/lancedb")
        self.db = db
        self.name = name
        self.table = None

    def search(self, query, n_results=2, metric="L2", nprobes=20, **kwargs):
        # This assumes query is a vector embedding
        # kwargs can be used for optional filtering
        # .select - only searches the specified columns
        # .where - SQL syntax filtering for metadata (e.g. where("price > 100"))
        # .metric - specifies the distance metric to use
        # .nprobes - values will yield better recall (more likely to find vectors if they exist) at the expense of latency.
        if self.table is None:
            raise Exception("Table not created yet, please add data first.")

        results = (
            self.table.search(query)
            .limit(n_results)
            .select(kwargs.get("select"))
            .where(kwargs.get("where"))
            .metric(metric)
            .nprobes(nprobes)
            .to_df()
        )
        return results

    def persist(self):
        raise NotImplementedError

    def write(self, data, metadatas, ids):
        # This function is similar to add(), but it's for more generalized updates
        # "data" is the list of embeddings
        # Inserts into table by expanding metadatas into a dataframe: [{'vector', 'id', 'meta', 'meta2'}, ...]

        documents = []
        for i in range(len(data)):
            row = {"vector": data[i], "id": ids[i]}
            row.update(metadatas[i])
            documents.append(row)

        if self.table is not None:
            self.table.add(documents)
        else:
            self.table = self.db.create_table(self.name, documents)

    def add(self, data, metadata, _id):
        # This function is for adding individual documents
        # It assumes you're passing in a single vector embedding, metadata, and id

        row = {"vector": data, "id": _id}
        row.update(metadata)

        if self.table is not None:
            self.table.add([row])
        else:
            self.table = self.db.create_table(self.name, [row])

    def delete(self, _id):
        # This function deletes a row by id.
        # LanceDB delete syntax uses SQL syntax, so you can use "in" or "="
        if self.table is None:
            raise Exception("Table not created yet, please add data first")

        if isinstance(_id, str):
            return self.table.delete(f"id = '{_id}'")
        else:
            return self.table.delete(f"id = {_id}")

    def drop(self, name):
        # This function drops a table, if it exists.

        path = os.path.join(self.db.uri, name + ".lance")
        if os.path.exists(path):
            shutil.rmtree(path)

```py

# `metagpt/document_store/milvus_store.py`

该代码是一个Python脚本，名为"milvus_store.py"。它实现了以下功能：

1. 从系统生成的元数据中定义了数据类型的映射，包括将整数映射为64位无符号整数类型，将字符串映射为无符号字符串类型，将浮点数映射为double类型，将numpy数组映射为float\_vector类型。
2. 引入了NumPy和Pymilvus库。
3. 创建了一个名为"milvus_store"的Collection对象，该对象使用了一个连接，连接名为"file"。
4. 使用对应该Collection对象的"file"连接读取文件中的数据，并将其存储为该Collection对象中的数据。
5. 创建了一个名为"type_mapping"的TypedDict，其中包含数据类型映射。
6. 在脚本的注释中提到了将来可能需要添加的函数和变量，但没有具体的实现。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/28 00:00
@Author  : alexanderwu
@File    : milvus_store.py
"""
from typing import TypedDict

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections

from metagpt.document_store.base_store import BaseStore

type_mapping = {
    int: DataType.INT64,
    str: DataType.VARCHAR,
    float: DataType.DOUBLE,
    np.ndarray: DataType.FLOAT_VECTOR
}


```py

这段代码定义了一个名为 `columns_to_milvus_schema` 的函数，用于将一个字典 `columns` 中包含的列转换为Milvus数据结构中的数据类型。

函数接收两个参数，一个是`primary_col_name`，另一个是`desc`。其中，`primary_col_name`参数指定了主列的名称，如果该参数为空，则该函数将主列为默认列。`desc`参数指定了描述性的字符串，用于在返回的DataFrame中显示列的名称。

函数内部，首先定义了一个空的列表 `fields`，用于存储每个列的数据类型和对应的FieldSchema对象。然后，遍历`columns`字典中的每个列，根据列的类型将相应的FieldSchema对象添加到 `fields`列表中。

接下来，定义了一个名为 `CollectionSchema` 的类，该类将`fields`列表中的每个FieldSchema对象组合成一个DataFrame，并设置该DataFrame的描述性字符串为`desc`参数。最后，函数返回一个名为 `schema`的DataFrame，其中包含已转换为Milvus数据结构的列。


```
def columns_to_milvus_schema(columns: dict, primary_col_name: str = "", desc: str = ""):
    """Assume the structure of columns is str: regular type"""
    fields = []
    for col, ctype in columns.items():
        if ctype == str:
            mcol = FieldSchema(name=col, dtype=type_mapping[ctype], max_length=100)
        elif ctype == np.ndarray:
            mcol = FieldSchema(name=col, dtype=type_mapping[ctype], dim=2)
        else:
            mcol = FieldSchema(name=col, dtype=type_mapping[ctype], is_primary=(col == primary_col_name))
        fields.append(mcol)
    schema = CollectionSchema(fields, description=desc)
    return schema


```py

This is a class that inherits from the ` milvus.search.SearchIndex` class. It has a search method that takes in a query and performs a linear search on the documents in the collection. It also has a write method that can be used to insert data into the collection.

The search method takes in a query as a list of lists of floating point numbers and performs a linear search on the documents in the collection. The search parameters include a metric type of "L2" and a parameter of "nprobe" which is the number of search queries to perform. The search method returns the results of the search and is marked as "isimilar" method.

The write method is used to insert data into the collection. It takes in the data to be inserted as a list and the name of the collection.

It is important to note that the current implementation of the search method is not correct for the current version of Milvus (v2.0.x) as it is using the deprecated ` search.SearchIndex` class, which is being replaced by the new ` search.SearchEngine` class.


```
class MilvusConnection(TypedDict):
    alias: str
    host: str
    port: str


class MilvusStore(BaseStore):
    """
    FIXME: ADD TESTS
    https://milvus.io/docs/v2.0.x/create_collection.md
    """

    def __init__(self, connection):
        connections.connect(**connection)
        self.collection = None

    def _create_collection(self, name, schema):
        collection = Collection(
            name=name,
            schema=schema,
            using='default',
            shards_num=2,
            consistency_level="Strong"
        )
        return collection

    def create_collection(self, name, columns):
        schema = columns_to_milvus_schema(columns, 'idx')
        self.collection = self._create_collection(name, schema)
        return self.collection

    def drop(self, name):
        Collection(name).drop()

    def load_collection(self):
        self.collection.load()

    def build_index(self, field='emb'):
        self.collection.create_index(field, {"index_type": "FLAT", "metric_type": "L2", "params": {}})

    def search(self, query: list[list[float]], *args, **kwargs):
        """
        FIXME: ADD TESTS
        https://milvus.io/docs/v2.0.x/search.md
        All search and query operations within Milvus are executed in memory. Load the collection to memory before conducting a vector similarity search.
        Note the above description, is this logic serious? This should take a long time, right?
        """
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=query,
            anns_field=kwargs.get('field', 'emb'),
            param=search_params,
            limit=10,
            expr=None,
            consistency_level="Strong"
        )
        # FIXME: results contain id, but to get the actual value from the id, we still need to call the query interface
        return results

    def write(self, name, schema, *args, **kwargs):
        """
        FIXME: ADD TESTS
        https://milvus.io/docs/v2.0.x/create_collection.md
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def add(self, data, *args, **kwargs):
        """
        FIXME: ADD TESTS
        https://milvus.io/docs/v2.0.x/insert_data.md
        import random
        data = [
          [i for i in range(2000)],
          [i for i in range(10000, 12000)],
          [[random.random() for _ in range(2)] for _ in range(2000)],
        ]

        :param args:
        :param kwargs:
        :return:
        """
        self.collection.insert(data)

```py

# `metagpt/document_store/qdrant_store.py`

这段代码定义了一个名为 `QdrantConnection` 的类，用于表示与 Qdrant 服务进行交互的数据连接。

```dataclasses
from dataclasses import dataclass
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, PointStruct, VectorParams
from metagpt.document_store.base_store import BaseStore
```py

首先，我们导入了 `dataclasses` 包以支持使用声明的数据类，以及来自 `typing` 包的 `List` 类。

```dataclasses
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, PointStruct, VectorParams
from metagpt.document_store.base_store import BaseStore
```py

接下来，我们定义了 `QdrantConnection` 类，其中包含用于表示与 Qdrant 服务进行交互的数据属性。

```dataclasses
@dataclass
class QdrantConnection:
   """
  Args:
      url: qdrant url
      host: qdrant host
      port: qdrant port
      memory: qdrant service use memory mode
      api_key: qdrant cloud api_key
  """
   url: str = None
   host: str = None
   port: int = None
   memory: bool = False
   api_key: str = None
```py

在这个类中，我们定义了五个属性：`url`、`host`、`port`、`memory` 和 `api_key`。这些属性都是整型或布尔值，用于表示与 Qdrant 服务进行交互所需的参数。

```dataclasses
   from qdrant_client import QdrantClient
   from qdrant_client.models import Filter, PointStruct, VectorParams
   from metagpt.document_store.base_store import BaseStore
```py

我们还从 `qdrant_client` 包中导入了 `QdrantClient` 类，用于在客户端与 Qdrant 服务进行交互。

```dataclasses
   from qdrant_client.models import Filter, PointStruct, VectorParams
```py

我们还从 `metagpt.document_store.base_store` 包中导入了 `BaseStore` 类，用于在客户端与 Qdrant 服务进行交互，管理 Qdrant 存储库的连接和操作。

```dataclasses
@dataclass
class QdrantConnection:
   """
  Args:
      url: qdrant url
      host: qdrant host
      port: qdrant port
      memory: qdrant service use memory mode
      api_key: qdrant cloud api_key
  """
   url: str = None
   host: str = None
   port: int = None
   memory: bool = False
   api_key: str = None
```py

最后，我们在类中添加了一个 `__post_init__` 方法，用于在实例化对象时初始化这些属性。

```dataclasses
   from dataclasses import dataclass
   from typing import List
   from qdrant_client import QdrantClient
   from qdrant_client.models import Filter, PointStruct, VectorParams
   from metagpt.document_store.base_store import BaseStore
   from dataclasses import field
   from typing import Any, Dict
   
   @dataclass
   class Config:
       field: bool = False
       field: str = False
       field: int = False
       field: bool = False
       field: str = False
       field: int = False
   
   @dataclass
   class ConfigResponse:
       field: Dict[str, Any] = field(default_factory=dict)
   
   @dataclass
   class SqlQuery:
       field: str = field(init=False, allow_init=True)
       field: List[str] = field(init=False, allow_init=True)
   
   @dataclass
   class SqlQueryResponse:
       field: Dict[str, Any] = field(default_factory=dict)
   
   @dataclass
   class AccessToken:
       field: str = field(init=False, allow_init=True)
   
   @dataclass
   class TokenResponse:
       field: str = field(init=False, allow_init=True)
```py

这段代码的 `__post_init__` 方法对 `QdrantConnection` 的所有属性都进行了初始化，并使用 `dataclasses` 提供的 `field` 语法定义了一个 `Config` 类，用于携带 Qdrant 服务的配置选项。

```dataclasses
   from dataclasses import dataclass
   from typing import List
   from qdrant_client import QdrantClient
   from qdrant_client.models import Filter, PointStruct, VectorParams
   from metagpt.document_store.base_store import BaseStore
   from dataclasses import field
   from typing import Any, Dict
   
   @dataclass
   class Config:
       field: bool = False
       field: str = False
       field: int = False
       field: bool = False
       field: str = False
       field: int = False
   
   @dataclass
   class ConfigResponse:
       field: Dict[str, Any] = field(default_factory=dict)
   
   @dataclass
   class SqlQuery:
       field: str = field(init=False, allow_init=True)
       field: List[str] = field(init=False, allow_init=True)
   
   @dataclass
   class SqlQueryResponse:
       field: Dict[str, Any] = field(default_factory=dict)
   
   @dataclass
   class AccessToken:
       field: str = field(init=False, allow_init=True)
   
   @dataclass
   class TokenResponse:
       field: str = field(init=False, allow_init=True)
```py

在这段代码中，我们还定义了两个类的子类，`QdrantConnection` 和 `ConfigResponse`，以及一个名为 `TokenResponse` 的枚举类型。

这段代码的作用是定义一个 Qdrant 服务连接类，用于与 Qdrant 服务进行交互，并管理 Qdrant 存储库的连接和操作。


```
from dataclasses import dataclass
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, PointStruct, VectorParams

from metagpt.document_store.base_store import BaseStore


@dataclass
class QdrantConnection:
    """
   Args:
       url: qdrant url
       host: qdrant host
       port: qdrant port
       memory: qdrant service use memory mode
       api_key: qdrant cloud api_key
   """
    url: str = None
    host: str = None
    port: int = None
    memory: bool = False
    api_key: str = None


```py

这是一个类型定义，定义了QDRANT client对象的行为。QDRANT client对象旨在使用QDRANT协议与QDRANT服务器通信。

```python
   def __init__(
       self,
       url: str = "https://api.qdrant.com/v1/",
       client_id: str = "your_client_id",
       client_secret: str = "your_client_secret",
       redis_base_url: str = "https://redis.您好：kMemJabiru7972.me/0/",
       redis_password: str = "your_redis_password",
       qdrant_base_url: str = "https://api.qdrant.com/v1/",
       qdrant_api_key: str = "your_qdrant_api_key",
       access_token: str = "your_access_token",
       access_token_expiry: int = 3600,  # 3600 seconds
       write_value: str = "your_write_value",
       write_value_expiry: int = 1800,  # 1800 seconds
       error_correction: str = "lZdDkIuPuJNz8",
   ):
       self.client = Cluster(
           redis_client=Redis(
               host=redis_base_url,
               port=6379,
               password=redis_password,
               haven=0,
               driver="为空函数",
           ),
           qdrant_client=AJV1(
               qdrant_base_url=qdrant_base_url,
               api_key=qdrant_api_key,
               write_app_key="为空函数",
               app_key="为空函数",
           ),
           # 初始化 own with your own token
       )
       self.client.initialize_属于自己的认证信息。
       self.vectors_config = VectorsConfig()

   def create_collection(
       self,
       collection_name: str = "your_collection_name",
       vectors_config: VectorsConfig = VectorsConfig(),
       **kwargs
   ) -> None:
       """
       create a new vector collection
       """
       return self.client.create_collection(
           collection_name=collection_name,
           vectors_config=vectors_config,
           **kwargs
       )

   def has_collection(self, collection_name: str):
       """
       检查一个特定的vector集合是否存在
       """
       return self.client.get_collection(collection_name)

   def delete_collection(self, collection_name: str, timeout: int = 60):
       res = self.client.delete_collection(collection_name, timeout=timeout)
       if not res:
           try:
               self.client.wait_for_deletion(collection_name, timeout=timeout)
           except:
               raise Exception(f"Delete collection {collection_name} failed.")
       return res

   def add(self, collection_name: str, points: List[PointStruct]):
       """
       向给定的向量集合中添加新的点
       """
       return self.client.upsert(
           collection_name="your_collection_name",
           points=points,
           "your_collection_name",
           炙
       )

   def search(
       self,
       collection_name: str,
       query: List[float],
       query_filter: Filter = None,
       k: int = 10,
       return_vector=False,
   ):
       """
       search for similar data
       """
       hits = self.client.search(
           collection_name=collection_name,
           query_vector=query,
           query_filter=query_filter,
           limit=k,
           with_vectors=return_vector,
       )
       return [hit.__dict__ for hit in hits]

   def write(self, *args, **kwargs):
       pass

```py
上述代码中，我们创建了一个名为`Client`的类。在这个类中，我们有以下方法：

- `__init__`：初始化QDRANT客户端的一些设置，包括API密钥、用户认证信息、自定义设置等。
- `create_collection`：创建一个新的向量集合。
- `has_collection`：检查给定的向量集合是否存在。
- `delete_collection`：删除给定向量集合。
- `add`：将给定向量集合中的点添加到集合中。
- `search`：在给定的向量集合中搜索与查询点相关的结果，并返回结果。
- `write`：向给定向量集合写入点数据。


```
class QdrantStore(BaseStore):
    def __init__(self, connect: QdrantConnection):
        if connect.memory:
            self.client = QdrantClient(":memory:")
        elif connect.url:
            self.client = QdrantClient(url=connect.url, api_key=connect.api_key)
        elif connect.host and connect.port:
            self.client = QdrantClient(
                host=connect.host, port=connect.port, api_key=connect.api_key
            )
        else:
            raise Exception("please check QdrantConnection.")

    def create_collection(
        self,
        collection_name: str,
        vectors_config: VectorParams,
        force_recreate=False,
        **kwargs,
    ):
        """
        create a collection
        Args:
            collection_name: collection name
            vectors_config: VectorParams object,detail in https://github.com/qdrant/qdrant-client
            force_recreate: default is False, if True, will delete exists collection,then create it
            **kwargs:

        Returns:

        """
        try:
            self.client.get_collection(collection_name)
            if force_recreate:
                res = self.client.recreate_collection(
                    collection_name, vectors_config=vectors_config, **kwargs
                )
                return res
            return True
        except:  # noqa: E722
            return self.client.recreate_collection(
                collection_name, vectors_config=vectors_config, **kwargs
            )

    def has_collection(self, collection_name: str):
        try:
            self.client.get_collection(collection_name)
            return True
        except:  # noqa: E722
            return False

    def delete_collection(self, collection_name: str, timeout=60):
        res = self.client.delete_collection(collection_name, timeout=timeout)
        if not res:
            raise Exception(f"Delete collection {collection_name} failed.")

    def add(self, collection_name: str, points: List[PointStruct]):
        """
        add some vector data to qdrant
        Args:
            collection_name: collection name
            points: list of PointStruct object, about PointStruct detail in https://github.com/qdrant/qdrant-client

        Returns: NoneX

        """
        # self.client.upload_records()
        self.client.upsert(
            collection_name,
            points,
        )

    def search(
        self,
        collection_name: str,
        query: List[float],
        query_filter: Filter = None,
        k=10,
        return_vector=False,
    ):
        """
        vector search
        Args:
            collection_name: qdrant collection name
            query: input vector
            query_filter: Filter object, detail in https://github.com/qdrant/qdrant-client
            k: return the most similar k pieces of data
            return_vector: whether return vector

        Returns: list of dict

        """
        hits = self.client.search(
            collection_name=collection_name,
            query_vector=query,
            query_filter=query_filter,
            limit=k,
            with_vectors=return_vector,
        )
        return [hit.__dict__ for hit in hits]

    def write(self, *args, **kwargs):
        pass

```