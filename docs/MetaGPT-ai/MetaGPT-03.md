# MetaGPT源码解析 3

# `metagpt/actions/design_api_review.py`

这段代码定义了一个名为 DesignReview 的类，该类继承自 Action 类，它实现了运行时检查产品需求文档（PRD）和设计规范的接口。

在 DesignReview 的构造函数中，初始化了超参数，包括名称、上下文和懒加载（LLM）。

run 方法是异步方法，它接收两个参数：产品需求文档（PRD）和设计规范。首先，它将 PRD 中的文本打印出来，然后使用 _aask 方法在 Askr 上下文中执行该文本，得到一个 JSON 响应，该响应表示 API 设计是否符合 PRD 的要求，是否符合良好的设计实践。

最终，将 API 设计的响应返回给调用该方法的行动对象。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 19:31
@Author  : alexanderwu
@File    : design_api_review.py
"""
from metagpt.actions.action import Action


class DesignReview(Action):
    def __init__(self, name, context=None, llm=None):
        super().__init__(name, context, llm)

    async def run(self, prd, api_design):
        prompt = f"Here is the Product Requirement Document (PRD):\n\n{prd}\n\nHere is the list of APIs designed " \
                 f"based on this PRD:\n\n{api_design}\n\nPlease review whether this API design meets the requirements" \
                 f" of the PRD, and whether it complies with good design practices."

        api_review = await self._aask(prompt)
        return api_review
    
```

# `metagpt/actions/design_filenames.py`

这段代码是一个Python脚本，使用了Python的`/usr/bin/env python`环境。以下是对脚本的功能和用途的解释：

1. `#!/usr/bin/env python` 是脚本的路径分隔符，表示脚本从 `/usr/bin/env python` 目录开始，并使用 Python 解释器执行。
2. `PROMPT` 是用于在运行脚本时显示的提示信息，说明了脚本的作用和用途。
3. `from metagpt.actions import Action` 是从 `metagpt.actions` 类中导入了一个名为 `Action` 的函数。这个函数可能在脚本中用来执行一些操作。
4. `from metagpt.logs import logger` 是从 `metagpt.logs` 类中导入了一个名为 `logger` 的函数。这个函数可能在脚本中用来输出调试信息。
5. `Action.generate_code(int(input.ence()), "intro_code")` 是脚本的入口函数。这个函数接收两个参数：一个整数和一个字符串，表示意图和代码的头部信息。函数的作用是生成指定意图的代码。
6. `logger.log(logger.LOG_INFO, msg=f"Request to generate code for intent {int(input.ence())}")` 是函数内部的代码。这个代码用来输出一条信息级别的日志，其中 `int(input.ence())` 是从用户输入中获取的意图。
7. `print(f"Intent: {int(input.ence())}")` 是在用户输入后立即执行的代码。它用来输出输入的意图。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/19 11:50
@Author  : alexanderwu
@File    : design_filenames.py
"""
from metagpt.actions import Action
from metagpt.logs import logger

PROMPT = """You are an AI developer, trying to write a program that generates code for users based on their intentions.
When given their intentions, provide a complete and exhaustive list of file paths needed to write the program for the user.
Only list the file paths you will write and return them as a Python string list.
Do not add any other explanations, just return a Python string list."""


```

这段代码定义了一个名为 "DesignFilenames" 的类，该类实现了 Action 接口。这个类的实现包括一个构造函数(__init__)，一个名为 "run" 的方法，以及一个默认构造函数(intrinsic function, but not implemented)。

在初始化函数(__init__)中，首先调用父类(即 Solution)的构造函数(__init__)，然后传递自己的名称、上下文对象和逻辑模型(llm)作为参数。然后定义了一个名为 "desc" 的字符串，用于在运行时显示输出的提示信息。

在 "run" 方法中，首先创建一个 PRD 的提示信息，然后使用 `_aask` 方法(即 self.ask)将提示信息转化为一个字符串，这个字符串将包含系统的设计信息。然后将设计文件名列表返回，并输出 "Based on the PRD, consider system design, and carry out the basic design of the corresponding APIs, data structures, and database tables. Please give your design, feedback clearly and in detail."。

DesignFilenames类的目的，根据 PRD 进行系统设计，并创建相应的 API、数据结构和数据库表。在这个类中，通过异步的方式，使用 ask 方法获取用户输入的设计文件名列表，然后执行系统设计并返回设计文件名列表。


```py
class DesignFilenames(Action):
    def __init__(self, name, context=None, llm=None):
        super().__init__(name, context, llm)
        self.desc = "Based on the PRD, consider system design, and carry out the basic design of the corresponding " \
                    "APIs, data structures, and database tables. Please give your design, feedback clearly and in detail."

    async def run(self, prd):
        prompt = f"The following is the Product Requirement Document (PRD):\n\n{prd}\n\n{PROMPT}"
        design_filenames = await self._aask(prompt)
        logger.debug(prompt)
        logger.debug(design_filenames)
        return design_filenames
    
```

# `metagpt/actions/detail_mining.py`

这段代码是一个Python脚本，用于从metagpt库中执行一个特定的动作。metagpt是一个人工智能系统，用于生成文本，可以用于多种用途，如自动写作、对话等。

具体来说，这段代码执行以下操作：

1. 导入需要使用的库：来自metagpt库的Action和ActionOutput，以及logger库，用于记录日志。
2. 定义一个名为ActionTemplate的类，该类将用于存储metagpt库中的动作模板。
3. 定义一个名为Time的函数，用于在动作中设置一个定时器，以在指定时间间隔后执行该动作。
4. 定义一个名为Logger的函数，用于在动作中记录日志信息。
5. 从metagpt库中获取特定主题的Action，并在Action模板中设置参数。
6. 调用Action模板中的函数，并将获取到的Action输出到ActionOutput中。
7. 调用Logger函数，并将日志信息添加到ActionOutput中。
8. 在程序结束时，关闭Logger函数。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/12 17:45
@Author  : fisherdeng
@File    : detail_mining.py
"""
from metagpt.actions import Action, ActionOutput
from metagpt.logs import logger

PROMPT_TEMPLATE = """
##TOPIC
{topic}

##RECORD
{record}

```

This code is a Python code snippet that generates output for a given input text. The output is limited to 150 words and is generated based on the topic provided in the input.

The code contains a for loop that reads the input text from the user. The input text is then formatted according to the specific details specified in the `...(Please provide the specific details you would like to inquire about here.)` section. The format of the output is pre-defined by the `FORMAT_EXAMPLE` variable and varies based on the input text.

The purpose of this code is to provide a simple way for the user to ask questions about a given topic, while limiting the output to 150 words to keep the focus on the topic.


```py
##Format example
{format_example}
-----

Task: Refer to the "##TOPIC" (discussion objectives) and "##RECORD" (discussion records) to further inquire about the details that interest you, within a word limit of 150 words.
Special Note 1: Your intention is solely to ask questions without endorsing or negating any individual's viewpoints.
Special Note 2: This output should only include the topic "##OUTPUT". Do not add, remove, or modify the topic. Begin the output with '##OUTPUT', followed by an immediate line break, and then proceed to provide the content in the specified format as outlined in the "##Format example" section.
Special Note 3: The output should be in the same language as the input.
"""
FORMAT_EXAMPLE = """

##

##OUTPUT
...(Please provide the specific details you would like to inquire about here.)

```

这段代码定义了一个名为 "DetailMining" 的类，用于实现 LLM(自然语言处理)模型进一步挖掘讨论中的有用信息。该类使用 Action 类来实现异步请求和处理输出。

具体来说，代码中定义了一个名为 "OUTPUT_MAPPING" 的字典，用于存储与给定主题和记录相关的输出格式。然后，代码创建了一个名为 "DetailMining" 的类，该类继承自 Action 类。

在 "DetailMining" 的构造函数中，代码创建了一个 "self" 对象，该对象具有父类的所有方法。然后，在 "run" 方法中，代码实现了异步请求的逻辑，该逻辑使用 "topic" 和 "record" 作为参数。代码还定义了一个 "PROMPT_TEMPLATE" 常量和一个 "FORMAT_EXAMPLE" 常量，这些常量将在下面被使用。

在 "run" 方法中，代码首先使用 "self._aask_v1" 方法创建一个异步请求，该请求使用 "topic" 和 "record" 将要发送到 OUTPUT_MAPPING 字典中键的值。然后，代码使用 "prompt" 参数将构建好的主题和记录参数化，并将结果存储在 "rsp" 变量中。最后，代码使用 "return" 语句将结果返回给调用方。


```py
##

##
"""
OUTPUT_MAPPING = {
    "OUTPUT": (str, ...),
}


class DetailMining(Action):
    """This class allows LLM to further mine noteworthy details based on specific "##TOPIC"(discussion topic) and "##RECORD" (discussion records), thereby deepening the discussion.
    """
    def __init__(self, name="", context=None, llm=None):
        super().__init__(name, context, llm)

    async def run(self, topic, record) -> ActionOutput:
        prompt = PROMPT_TEMPLATE.format(topic=topic, record=record, format_example=FORMAT_EXAMPLE)
        rsp = await self._aask_v1(prompt, "detail_mining", OUTPUT_MAPPING)
        return rsp

```

# `metagpt/actions/execute_task.py`

这段代码定义了一个名为 "execute_task.py" 的 Python 文件，并在其中实现了动作（Action）和消息（Message）的类。具体来说，这段代码定义了一个名为 "ExecuteTask" 的类，该类继承自名为 "Action" 的类，具有 "ExecuteTask" 的名称、一个未定义的 "name" 参数，以及一个未定义的 "context" 参数和一个未定义的 "llm" 参数。

ExecuteTask 类包含一个 "run" 方法，该方法接受一个或多个参数，并执行与该方法同名的操作。在运行方法中，action 类中的 `super().__init__(self, name, context, llm)` 方法会先执行父类的初始化方法，然后调用自身的方法来执行操作。因此，这段代码的具体作用是实现了一个命令行工具，接受用户输入的参数，并执行指定的任务。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/13 12:26
@Author  : femto Zheng
@File    : execute_task.py
"""
from metagpt.actions import Action
from metagpt.schema import Message


class ExecuteTask(Action):
    def __init__(self, name="ExecuteTask", context: list[Message] = None, llm=None):
        super().__init__(name, context, llm)

    def run(self, *args, **kwargs):
        pass

```

# `metagpt/actions/invoice_ocr.py`

这段代码是一个Python脚本，用于实现发票OCR助手的一些操作。以下是代码的主要功能和行为的解释：

1. 导入所需的Python模块和库：`os, zipfile, pathlib, datetime`。
2. 设置文件存储目录：`os.path.join(os.path.dirname(__file__), '..')`。
3. 检查当前工作目录是否为包含`invoice_ocr.py`文件的目录：`os.path.exists('invoice_ocr.py')`。
4. 如果当前工作目录包含`invoice_ocr.py`文件，则执行以下操作：`if os.path.exists('invoice_ocr.py'):`。
5. 创建一个名为`invoice_ocr_assistant.py`的新文件：`open('invoice_ocr_assistant.py', 'w')`。
6. 将上面创建的新文件的内容复制到当前工作目录下的新文件中：`print("ActionCopyright", datetime.now())`。
7. 将当前工作目录下的`invoice_ocr.py`文件中的所有内容复制到新创建的`invoice_ocr_assistant.py`文件中：`print("ActionCopyright", datetime.now())`。
8. 新创建的`invoice_ocr_assistant.py`文件中的内容如下：
```pypython
# -*- coding: utf-8 -*-

"""
@Time    : 2023/9/21 18:10:20
@Author  : Stitch-z
@File    : invoice_ocr_assistant.py
@Describe : Actions of the invoice ocr assistant.
"""

import os
import zipfile
from pathlib import Path
from datetime import datetime


def main():
   print("ActionStart")
   # 1. 选择要处理的发票文件
   # 2. 读取并解压选择的发票文件
   # 3. 设置工作目录
   # 4. 创建新文件并将文件内容复制到新文件中
   # 5. 将新文件保存为zip格式并输出


if __name__ == '__main__':
   main()
```
9. `main`函数是主要的入口点，负责执行 invoice OCR 辅助工具的所有操作。
10. `main`函数中的步骤：
	* 选择要处理的发票文件
	* 读取并解压选择的发票文件
	* 设置工作目录
	* 创建新文件并将文件内容复制到新文件中
	* 将新文件保存为zip格式并输出。


```py
#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
@Time    : 2023/9/21 18:10:20
@Author  : Stitch-z
@File    : invoice_ocr.py
@Describe : Actions of the invoice ocr assistant.
"""

import os
import zipfile
from pathlib import Path
from datetime import datetime

```

This is a Python class that performs an action to identify invoice files through Optical Character Recognition (OCR). The `run` method takes an input file path and other arguments, and returns a list of OCR results.

The `file_ext` property is used to determine the file extension. If the file extension is ".zip", the class will perform OCR on the zip file. If the file extension is one of the recognized extensions for OCR, such as ".pdf", ".png", or ".jpg", the class will perform OCR on the corresponding file. If the file extension is ".zip", ".pdf", ".png", or ".jpg", the class will perform OCR on the corresponding file.

The `_check_file_type` method is used to determine the file type based on its extension.

The `_unzip` method is used to perform the action to unzip the file if it is a batch file.

The `_ocr` method is used to perform OCR on the input file.

Note: This class uses the `PaddleOCR` library for OCR. This class is for informational purposes only, and should not be used for production purposes without additional checks and validation.


```py
import pandas as pd
from paddleocr import PaddleOCR

from metagpt.actions import Action
from metagpt.const import INVOICE_OCR_TABLE_PATH
from metagpt.logs import logger
from metagpt.prompts.invoice_ocr import EXTRACT_OCR_MAIN_INFO_PROMPT, REPLY_OCR_QUESTION_PROMPT
from metagpt.utils.common import OutputParser
from metagpt.utils.file import File


class InvoiceOCR(Action):
    """Action class for performing OCR on invoice files, including zip, PDF, png, and jpg files.

    Args:
        name: The name of the action. Defaults to an empty string.
        language: The language for OCR output. Defaults to "ch" (Chinese).

    """

    def __init__(self, name: str = "", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    @staticmethod
    async def _check_file_type(file_path: Path) -> str:
        """Check the file type of the given filename.

        Args:
            file_path: The path of the file.

        Returns:
            The file type based on FileExtensionType enum.

        Raises:
            Exception: If the file format is not zip, pdf, png, or jpg.
        """
        ext = file_path.suffix
        if ext not in [".zip", ".pdf", ".png", ".jpg"]:
            raise Exception("The invoice format is not zip, pdf, png, or jpg")

        return ext

    @staticmethod
    async def _unzip(file_path: Path) -> Path:
        """Unzip a file and return the path to the unzipped directory.

        Args:
            file_path: The path to the zip file.

        Returns:
            The path to the unzipped directory.
        """
        file_directory = file_path.parent / "unzip_invoices" / datetime.now().strftime("%Y%m%d%H%M%S")
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                # Use CP437 to encode the file name, and then use GBK decoding to prevent Chinese garbled code
                relative_name = Path(zip_info.filename.encode("cp437").decode("gbk"))
                if relative_name.suffix:
                    full_filename = file_directory / relative_name
                    await File.write(full_filename.parent, relative_name.name, zip_ref.read(zip_info.filename))

        logger.info(f"unzip_path: {file_directory}")
        return file_directory

    @staticmethod
    async def _ocr(invoice_file_path: Path):
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=1)
        ocr_result = ocr.ocr(str(invoice_file_path), cls=True)
        return ocr_result

    async def run(self, file_path: Path, *args, **kwargs) -> list:
        """Execute the action to identify invoice files through OCR.

        Args:
            file_path: The path to the input file.

        Returns:
            A list of OCR results.
        """
        file_ext = await self._check_file_type(file_path)

        if file_ext == ".zip":
            # OCR recognizes zip batch files
            unzip_path = await self._unzip(file_path)
            ocr_list = []
            for root, _, files in os.walk(unzip_path):
                for filename in files:
                    invoice_file_path = Path(root) / Path(filename)
                    # Identify files that match the type
                    if Path(filename).suffix in [".zip", ".pdf", ".png", ".jpg"]:
                        ocr_result = await self._ocr(str(invoice_file_path))
                        ocr_list.append(ocr_result)
            return ocr_list

        else:
            #  OCR identifies single file
            ocr_result = await self._ocr(file_path)
            return [ocr_result]


```

这段代码定义了一个名为 "GenerateTable" 的类，它实现了从 OCR 结果中生成表格的功能。这个类接受两个参数，一个是动作名称(例如，可以命名为 "GenerateTable")，另一个是生成表格所使用的语言(例如，可以命名为 "ch" 或 "en")。

当 "GenerateTable" 类被实例化时，它继承了父类 "Action" 的 "run" 方法，这个方法处理了生成表格的逻辑。

在 "run" 方法中，首先调用父类的 "run" 方法来处理 OCR 结果。然后，根据所选的语言，将结果保存为 Excel 文件。

对于每个 OCR 结果，类会将结果保存为一张表格。然后，对于每个 Excel 文件，类会将表格数据保存为一个新的 Excel 文件。

最后，生成 Excel 文件的函数使用了 pandas 库，将表格数据保存为 Excel 文件。同时，在类中还添加了一个 "name" 参数，用于指定动作名称，它的默认值为一个空字符串。


```py
class GenerateTable(Action):
    """Action class for generating tables from OCR results.

    Args:
        name: The name of the action. Defaults to an empty string.
        language: The language used for the generated table. Defaults to "ch" (Chinese).

    """

    def __init__(self, name: str = "", language: str = "ch", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.language = language

    async def run(self, ocr_results: list, filename: str, *args, **kwargs) -> dict[str, str]:
        """Processes OCR results, extracts invoice information, generates a table, and saves it as an Excel file.

        Args:
            ocr_results: A list of OCR results obtained from invoice processing.
            filename: The name of the output Excel file.

        Returns:
            A dictionary containing the invoice information.

        """
        table_data = []
        pathname = INVOICE_OCR_TABLE_PATH
        pathname.mkdir(parents=True, exist_ok=True)

        for ocr_result in ocr_results:
            # Extract invoice OCR main information
            prompt = EXTRACT_OCR_MAIN_INFO_PROMPT.format(ocr_result=ocr_result, language=self.language)
            ocr_info = await self._aask(prompt=prompt)
            invoice_data = OutputParser.extract_struct(ocr_info, dict)
            if invoice_data:
                table_data.append(invoice_data)

        # Generate Excel file
        filename = f"{filename.split('.')[0]}.xlsx"
        full_filename = f"{pathname}/{filename}"
        df = pd.DataFrame(table_data)
        df.to_excel(full_filename, index=False)
        return table_data


```

这段代码定义了一个名为 "ReplyQuestion" 的类，它实现了 Action 类，用于生成根据 OCR 结果回复问题的回复。类有两个参数，一个是字符串类型的 name，另一个是字符串类型的 language，它们分别用于指定回答的名称和生成回答所使用的语言。

在类的初始化方法 "__init__" 中，调用了父类 "Action" 的初始化方法，并使用参数星号 "*args, **kwargs" 传递给父类的初始化方法，以便覆盖默认值。

类的 "run" 方法是生成回答的核心部分，它接收两个参数，一个是问题(查询)和一个 OCR 结果列表。运行方法首先将问题和 OCR 结果存储在两个变量中，然后使用 "format" 方法将问题、OCR 结果和语言参数拼接成一个新的字符串，最后使用 "aask" 方法将该字符串发送到 AI 语言模型的预测中，并返回预测的结果。

类的实例可以通过创建一个 "ReplyQuestion" 类的实例并调用其 "run" 方法来生成回答。例如：

```py
reply_question = ReplyQuestion()

async def generate_reply(query: str, ocr_results: list) -> str:
   return reply_question.run(query, ocr_results)

async def main():
   query = "您的问题是什么？"
   ocr_results = [{"label": "斑点", "score": 0.85}, {"label": "文本", "score": 0.6}, {"label": "轮廓", "score": 0.70}]

   reply = generate_reply(query, ocr_results)
   print(reply)

if __name__ == "__main__":
   main()
```

这段代码会创建一个 "ReplyQuestion" 类的实例，并使用该实例的 "generate_reply" 函数来生成回答。在此示例中，问题为 "您的问题是什么？",OCR 结果已将文本、斑点轮廓和分数分别为 0.85, 0.60 和 0.70  scores。生成回答的结果将根据所给 OCR 结果


```py
class ReplyQuestion(Action):
    """Action class for generating replies to questions based on OCR results.

    Args:
        name: The name of the action. Defaults to an empty string.
        language: The language used for generating the reply. Defaults to "ch" (Chinese).

    """

    def __init__(self, name: str = "", language: str = "ch", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.language = language

    async def run(self, query: str, ocr_result: list, *args, **kwargs) -> str:
        """Reply to questions based on ocr results.

        Args:
            query: The question for which a reply is generated.
            ocr_result: A list of OCR results.

        Returns:
            A reply result of string type.
        """
        prompt = REPLY_OCR_QUESTION_PROMPT.format(query=query, ocr_result=ocr_result, language=self.language)
        resp = await self._aask(prompt=prompt)
        return resp


```

# `metagpt/actions/prepare_interview.py`

这段代码是一个Python脚本，用于准备面试。脚本中定义了一个名为"PROMPT_TEMPLATE"的常量，该常量是一个格式化模板，用于在面试中回答问题。

具体来说，这段代码的作用是定义了一个准备面试的模板，并使用该模板来生成回答。在模板中，{context}表示当前面试中的上下文信息，例如面试官的问题、自己的回答等等。生成好的回答将按照格式化模板的格式进行排版，并输出到控制台。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/19 15:02
@Author  : DevXiaolan
@File    : prepare_interview.py
"""
from metagpt.actions import Action

PROMPT_TEMPLATE = """
# Context
{context}

## Format example
---
```

This code appears to be a simple HTML document that introduces a question for the interviewer to ask the interviewee. The question is of the form `Q1: question 1 here` and `Q2: question 2 here...`, and the code includes two references to `point 1` and `point 2`.

It is likely that this code is intended to be included in an interview for a job that involves frontend or backend development, and the code provides a starting point for the interviewer to ask questions about the interviewee's experience and skills.

The code does not output anything, but it serves as a template for the interviewer to ask questions related to the interviewee's knowledge of frontend and backend development.


```py
Q1: question 1 here
References:
  - point 1
  - point 2

Q2: question 2 here...
---

-----
Role: You are an interviewer of our company who is well-knonwn in frontend or backend develop;
Requirement: Provide a list of questions for the interviewer to ask the interviewee, by reading the resume of the interviewee in the context.
Attention: Provide as markdown block as the format above, at least 10 questions.
"""

# prepare for a interview


```

这段代码定义了一个名为 "PrepareInterview" 的类，该类实现了 "Action" 接口。

在 "PrepareInterview" 的初始化方法 "__init__" 中，调用了父类 "Action" 的初始化方法，进而实现了类 "Action" 的初始化。同时，还在初始化方法中传入了一个 "name"、一个 "context" 和一个 "llm" 参数。

"run" 方法是 "PrepareInterview" 的核心方法，该方法异步执行，并在该方法中调用了 " PrepareInterview" 类的一个名为 " _aask_v1" 的方法。

由于没有提供 "prepare" 函数的定义，因此无法确定 "PrepareInterview" 的具体行为。但是，从 "run" 方法的实现来看，"PrepareInterview" 似乎是一个用于准备面试的工具类，它使用一个名为 "llm" 的参数来保存问答库，并在 "run" 方法中调用了 " _aask_v1" 方法来获取用户的问题列表。


```py
class PrepareInterview(Action):
    def __init__(self, name, context=None, llm=None):
        super().__init__(name, context, llm)

    async def run(self, context):
        prompt = PROMPT_TEMPLATE.format(context=context)
        question_list = await self._aask_v1(prompt)
        return question_list


```

# `metagpt/actions/project_management.py`

这段代码是一个Python脚本，使用了`#!/usr/bin/env python`作为Shell脚本的入口。

该脚本的主要作用是定义了一个名为`project_management.py`的函数式接口，用于在Metagpt工作空间中执行一系列操作。

具体来说，该脚本从`typing.List`类型中创建了一个名为`actions`的接口，这个接口可以定义一个执行动作的方法`execute`，该方法接受一个`Action`对象作为参数，并返回一个描述该动作的元数据`ExecutionResult`。

该脚本还从`metagpt.config`和`metagpt.const`命名空间中获取了`CONFIG`和`WORKSPACE_ROOT`变量，分别用于设置工作空间根目录和环境变量。

此外，该脚本还定义了一个名为`CodeParser`的类，用于将Markdown编码的文本转换为HTML格式的文本。

最后，该脚本使用`get_template`函数从工作空间根目录中获取一个模板文件，并将其内容写入一个名为`metadata.txt`的文件中。

总结起来，该脚本定义了一个简单的API，用于在Metagpt工作空间中执行一系列操作，包括创建、编辑和删除文档、设置工作空间根目录等。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 19:12
@Author  : alexanderwu
@File    : project_management.py
"""
from typing import List

from metagpt.actions.action import Action
from metagpt.config import CONFIG
from metagpt.const import WORKSPACE_ROOT
from metagpt.utils.common import CodeParser
from metagpt.utils.get_template import get_template
from metagpt.utils.json_to_markdown import json_to_markdown

```

这段代码定义了一个名为 `templates` 的字典，其中包含一个名为 `json` 的键，其值为一个包含两个子键的嵌套字典。

在 `json` 键中，第一个子键 `PROMPT_TEMPLATE` 是一个字符串，其中包含一个 Prompt(提示)模板，用于在回答问题之前提供问题的上下文信息。这个模板以 `#` 开头的格式包括一个 `Context` 字段，一个 `Format example` 字段，和一个 `Attention` 字段，用于在回答问题之前提供注意事項。

在 `json` 键的第二层子键中，包含一个名为 `requirements.txt` 的文件，其中包含一些 Python 第三方包的名称和版本信息，这些包可能需要在特定情况下使用。


```py
templates = {
    "json": {
        "PROMPT_TEMPLATE": """
# Context
{context}

## Format example
{format_example}
-----
Role: You are a project manager; the goal is to break down tasks according to PRD/technical design, give a task list, and analyze task dependencies to start with the prerequisite modules
Requirements: Based on the context, fill in the following missing information, each section name is a key in json. Here the granularity of the task is a file, if there are any missing files, you can supplement them
Attention: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the code and triple quote.

## Required Python third-party packages: Provided in requirements.txt format

```

这段代码是一个配置文件，其中包含了许多要求、API规格、依赖关系、任务列表、共享知识和UNCLEAR消息。它主要用于定义一个项目的结构和规则，使得开发团队能够遵循这些规则，并能够更有效地开发项目。

具体来说，该配置文件包含以下内容：

- 项目所需的其他第三方软件包，以 requirements.txt 文件格式提供。
- API 规格，以 OpenAPI 3.0 的格式描述了可能由前端和后端使用的所有 API。
- 依赖关系列表，以 Python list[list[str]] 格式提供，列出了开发过程中需要使用的所有文件和依赖关系。
- 任务列表，以 Python list[str] 格式提供，列出了开发过程中需要完成的任务和依赖关系。
- 共享知识，以 Python list[str] 格式提供，列出了开发过程中需要知道的一些公共知识和 UNCLEAR 消息。
- Anything UNCLEAR，以 plain text 提供，告诉开发团队需要知道的一些公共知识和 UNCLEAR 消息。

此外，该配置文件还提供了一个 JSON 格式例子，用于展示如何正确格式化 API 规格。


```py
## Required Other language third-party packages: Provided in requirements.txt format

## Full API spec: Use OpenAPI 3.0. Describe all APIs that may be used by both frontend and backend.

## Logic Analysis: Provided as a Python list[list[str]. the first is filename, the second is class/method/function should be implemented in this file. Analyze the dependencies between the files, which work should be done first

## Task list: Provided as Python list[str]. Each str is a filename, the more at the beginning, the more it is a prerequisite dependency, should be done first

## Shared Knowledge: Anything that should be public like utils' functions, config's variables details that should make clear first. 

## Anything UNCLEAR: Provide as Plain text. Make clear here. For example, don't forget a main entry. don't forget to init 3rd party libs.

output a properly formatted JSON, wrapped inside [CONTENT][/CONTENT] like format example,
and only output the json inside this tag, nothing else
""",
        "FORMAT_EXAMPLE": '''
{
    "Required Python third-party packages": [
        "flask==1.1.2",
        "bcrypt==3.2.0"
    ],
    "Required Other language third-party packages": [
        "No third-party ..."
    ],
    "Full API spec": """
        openapi: 3.0.0
        ...
        description: A JSON object ...
     """,
    "Logic Analysis": [
        ["game.py","Contains..."]
    ],
    "Task list": [
        "game.py"
    ],
    "Shared Knowledge": """
        'game.py' contains ...
    """,
    "Anything UNCLEAR": "We need ... how to start."
}
```

这段代码是一个 Python 语言的 `Markdown` 类，用于将一些Markdown样式的文本转换为HTML格式。下面是该类的作用说明：

1. `''`：这是一个单引号，用于表示一个字符串的引号。

2. `'Markdown':` 这是一个字典，指定了该类的名称。

3. `{'PROMPT_TEMPLATE': ''}`：这是一个字典，指定了 `PROMPT_TEMPLATE` 的键为空字符串。这个键用于存储一个Markdown模板，该模板将被应用于将Markdown文本转换为HTML格式的函数中。

4. `{'format_example': ''}`：这是一个字典，指定了 `format_example` 的键为空字符串。这个键用于存储一个Markdown格式示例，该示例将被应用于将Markdown文本转换为HTML格式的函数中。

5. `{'required_python_packages': ''}`：这是一个字典，指定了 `required_python_packages` 的键为空字符串。这个键用于存储一个Python第三方包的列表，该列表将被应用于将Markdown文本转换为HTML格式的函数中。

6. `{'task_list': ''}`：这是一个字典，指定了 `task_list` 的键为空字符串。这个键用于存储一个任务列表，该列表将被应用于将Markdown文本转换为HTML格式的函数中。

7. `{'format_example_ctx': ''}`：这是一个字典，指定了 `format_example_ctx` 的键为空字符串。这个键用于存储一个Markdown格式示例的上下文，该上下文将被应用于将Markdown文本转换为HTML格式的函数中。

8. `{'content': ''}`：这是一个字典，指定了 `content` 的键为空字符串。这个键用于存储一个Markdown文本内容，该内容将被应用于将Markdown文本转换为HTML格式的函数中。

9. `{'output_html': ''}`：这是一个字典，指定了 `output_html` 的键为空字符串。这个键用于存储一个将Markdown文本转换为HTML格式的函数的输出结果。

10. `{'description': ''}`：这是一个字典，指定了 `description` 的键为空字符串。这个键用于存储一个描述Markdown文本内容的字符串。

11. `'visual_content': ''}`：这是一个字典，指定了 `visual_content` 的键为空字符串。这个键用于存储一个描述Markdown文本内容的图像，该图像将被应用于将Markdown文本转换为HTML格式的函数中。

12. `'thumbnail_url': ''}`：这是一个字典，指定了 `thumbnail_url` 的键为空字符串。这个键用于存储一个描述Markdown文本内容的图像的URL，该图像将被应用于将Markdown文本转换为HTML格式的函数中。

13. `'authorization_message': ''}`：这是一个字典，指定了 `authorization_message` 的键为空字符串。这个键用于存储一个在将Markdown文本转换为HTML格式的过程中需要提供的授权信息。

14. `'json_variables': ''}`：这是一个字典，指定了 `json_variables` 的键为空字符串。这个键用于存储一个将Markdown文本转换为HTML格式的过程中需要使用的JSON变量列表。

15. `'render_template': ''}`：这是一个字典，指定了 `render_template` 的键为空字符串。这个键用于存储一个函数，该函数将被应用于将Markdown文本转换为HTML格式的函数中。

16. `'execute_command': ''}`：这是一个字典，指定了 `execute_command` 的键为空字符串。这个键用于存储一个函数，该函数将被应用于将Markdown文本转换为HTML格式的函数中。

17. `'concat_templates': ''}`：这是一个字典，指定了 `concat_templates` 的键为空字符串。这个键用于存储一个函数，该函数将被应用于将多个Markdown模板组合成单个HTML格式的函数中。

18. `'post_process_content': ''}`：这是一个字典，指定了 `post_process_content` 的键为空字符串。这个键用于存储一个函数，该函数将被应用于将多个Markdown模板组合成单个HTML格式的函数中。

19. `'variables': ''}`：这是一个字典，指定了 `variables` 的键为空字符串。这个键用于存储一个将Markdown文本转换为HTML格式的过程中需要使用的变量列表。

20. `'task_id': ''}`：这是一个字典，指定了 `task_id` 的键为空字符串。这个键用于存储一个唯一标识符，该标识符将被应用于将Markdown文本转换为HTML格式的函数中。


```py
''',
    },
    "markdown": {
        "PROMPT_TEMPLATE": """
# Context
{context}

## Format example
{format_example}
-----
Role: You are a project manager; the goal is to break down tasks according to PRD/technical design, give a task list, and analyze task dependencies to start with the prerequisite modules
Requirements: Based on the context, fill in the following missing information, note that all sections are returned in Python code triple quote form seperatedly. Here the granularity of the task is a file, if there are any missing files, you can supplement them
Attention: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the code and triple quote.

## Required Python third-party packages: Provided in requirements.txt format

```

这段代码是一个 Python 脚本，用于生成一个列表，其中包含一些 required的其他语言第三方包，以及一个清晰的 API 说明。

首先，它定义了一个 required 的列表，其中包含一些第三方包，这些包需要在 requirements.txt 文件中使用。

接下来，它定义了一个 API 说明，描述了可能由前后端使用的所有 API。

然后，它定义了一个逻辑分析，其中将文件之间的依赖关系分析为谁应该先做什么。

接着，它定义了一个待完成的任务列表，其中包含一些文件，这些文件是后续任务的先决条件，应该在完成其他任务之前先完成。

接下来，它定义了一个关于共同知识的部分，其中包含了一些应该公开共享的信息，比如函数和配置变量的详细说明。

最后，它定义了一个uncertainty，其中包含了一些不清楚的信息，需要通过其他渠道进行澄清。

此外，它定义了一个 FORMAT_EXAMPLE，其中包含了一些用于格式化说明的指令。


```py
## Required Other language third-party packages: Provided in requirements.txt format

## Full API spec: Use OpenAPI 3.0. Describe all APIs that may be used by both frontend and backend.

## Logic Analysis: Provided as a Python list[list[str]. the first is filename, the second is class/method/function should be implemented in this file. Analyze the dependencies between the files, which work should be done first

## Task list: Provided as Python list[str]. Each str is a filename, the more at the beginning, the more it is a prerequisite dependency, should be done first

## Shared Knowledge: Anything that should be public like utils' functions, config's variables details that should make clear first. 

## Anything UNCLEAR: Provide as Plain text. Make clear here. For example, don't forget a main entry. don't forget to init 3rd party libs.

""",
        "FORMAT_EXAMPLE": '''
---
```

这段代码是一个Python程序的requirements.txt文件。这个文件列出了程序必需的其他第三方包。

flask是一个流行的Python web框架，用于创建简单HTTP服务器和API。

bcrypt是一个安全的密码哈希库，用于将密码哈希为字节串并存储。

No third-party包是指不需要依赖其他第三方软件包。在某些情况下，程序可能不需要额外的软件包，因此可以填写这部分的空。


```py
## Required Python third-party packages
```python
"""
flask==1.1.2
bcrypt==3.2.0
"""
```py

## Required Other language third-party packages
```python
"""
No third-party ...
"""
```py

```

这段代码是一个Python文件，包含了一个描述性文本，描述了该文件所使用的API的详细信息。该文本是由Python标准库中的`full_api_spec`模块定义的。

具体来说，这段代码定义了一个名为`openapi`的常量，该常量保存在一个名为`__full_api_spec__`的属性中。该常量返回一个JSON对象，其中包含了该API的详细信息，如描述、术语、限制等等。这些信息描述了API的功能、用途、限制等，是开发人员理解和使用API的重要依据。

此外，该代码还定义了一个名为`description`的常量，该常量返回一个字符串，用于描述该API的用途和描述。

最后，该代码还定义了一个包含两个元素的列表`game_py_contains_dependencies`，该列表包含了两个Python模块的名称，分别是`game.py`和`Contains`。这两个模块可能是依赖于该API的模块，开发人员需要按照该API的规范来编写代码，以满足API的要求。


```py
## Full API spec
```python
"""
openapi: 3.0.0
...
description: A JSON object ...
"""
```py

## Logic Analysis
```python
[
    ["game.py", "Contains ..."],
]
```py

```

This code is a Python variable that contains a list of string literals (indented) representing the filenames of Python modules that should be included in a "game.txt" file.

The variable is a "task list" and it's used to store the list of modules that should be imported when the script "game.py" is executed. The list is divided by comma delimiter, which means that each element of the list is a separate module.

The contents of the task list is not directly accessible, but some of the elements of the list, like "game.py", are. This allows the code to be organized into separate, maintainable modules, and allows for easier modification and addition to the project.


```py
## Task list
```python
[
    "game.py",
]
```py

## Shared Knowledge
```python
"""
'game.py' contains ...
"""
```py

## Anything UNCLEAR
```

这段代码是一个 Javascript 对象，包含了多组配置参数。我们需要了解每个参数的作用，以便在需要时正确设置它们。

1. `We need ... how to start.` 是全局性的指导，告诉我们如何开始编写代码。

2. `... how to start.` 是 `We need ... how to start.` 的省略号，表示这是一个无限循环的列表。我们可以在需要时添加更多的 `how to start` 来获取更多的指导。

3. `We need Required Python third-party packages: ["package1", "package2"]` 和 `We need Received Other language third-party packages: ["package3", "package4"]` 是两个列表，每个列表都包含了所需的第三方包。这里 "We need" 是全局性的指导，告诉我们需要哪些第三方包。

4. `We need Full API spec: "{API_SPEC}"` 是全局性的指导，告诉我们需要一个完整的 API 规范。这里 "{API_SPEC}" 是接口规范的占位符，将会在需要时被替换为实际的 API 规范。

5. `We need Logic Analysis: ["{LABEL_ANALYSIS}", "{LABEL_ANALYSIS_EXPLAIN}"]` 是两个列表，每个列表都包含了需要的信息。这里 "We need" 是全局性的指导，告诉我们需要哪些信息。

6. `We need Task list: ["{TASK_LIST}"]` 是全局性的指导，告诉我们需要哪些任务列表。这里 "We need" 是全局性的指导，告诉我们需要哪些任务列表。

7. `We need Shared Knowledge: str, ...` 和 `We need Anything UNCLEAR: str, ...` 是两个全局性的指导，告诉我们需要哪些共享知识或清晰的说明。

8. `OUTPUT_MAPPING = {...}` 是全局性的对象，其中 `...` 是需要设置的键值对。

这里 `OUTPUT_MAPPING` 是告诉我们需要设置的键值对，它将会在我们编写代码时将相应的值（如列表、字符串等）替换到我们的代码中。


```py
We need ... how to start.
---
''',
    },
}
OUTPUT_MAPPING = {
    "Required Python third-party packages": (List[str], ...),
    "Required Other language third-party packages": (List[str], ...),
    "Full API spec": (str, ...),
    "Logic Analysis": (List[List[str]], ...),
    "Task list": (List[str], ...),
    "Shared Knowledge": (str, ...),
    "Anything UNCLEAR": (str, ...),
}


```

这段代码定义了一个名为 WriteTasks 的自定义 Action 类，用于创建或更新 tasks.txt 和 requirements.txt 文件，并将它们的内容保存到工作空间中。

在 WriteTasks 的初始化方法 "__init__" 中，代码继承自 Action 类，并覆盖了该类的 "**init**" 方法。在该方法的三个参数中，第一个参数 "name" 是一个字符串，用于指定该自定义 action 的名称。第二个参数 "context" 是一个参数对象，用于存储 action 上下文中的参数或数据。第三个参数 "llm" 是一个参数对象，用于存储 action 实例的本地逻辑模型。

在自定义的 "**save**" 方法中，代码调用了父类的 "**save**" 方法，并添加了一些自定义的逻辑。首先，代码检查 action 上下文中是否包含一个名为 "instruct_content" 的参数，如果存在，就从该参数中获取包名。否则，代码将根据内容从工作空间目录中提取包名，并将其写入到 workspace_root 目录下的 ws_name 目录下的 "docs/api_spec_and_tasks.md" 文件中。

接着，代码将 requests.post 方法用于向服务器发送 POST 请求，并在请求正文中包含了两个参数：格式和一个格式example。如果请求成功，则将返回的结果存储到 instance_variables 中。最后，代码调用 self._aask_v1 方法，该方法使用了 asksr256 库来处理请求和响应的 JSON 数据。在自定义的 "**run**" 方法中，代码调用了 _save 方法，该方法保存了 instance_variables 中存储的内容。


```py
class WriteTasks(Action):
    def __init__(self, name="CreateTasks", context=None, llm=None):
        super().__init__(name, context, llm)

    def _save(self, context, rsp):
        if context[-1].instruct_content:
            ws_name = context[-1].instruct_content.dict()["Python package name"]
        else:
            ws_name = CodeParser.parse_str(block="Python package name", text=context[-1].content)
        file_path = WORKSPACE_ROOT / ws_name / "docs/api_spec_and_tasks.md"
        file_path.write_text(json_to_markdown(rsp.instruct_content.dict()))

        # Write requirements.txt
        requirements_path = WORKSPACE_ROOT / ws_name / "requirements.txt"
        requirements_path.write_text("\n".join(rsp.instruct_content.dict().get("Required Python third-party packages")))

    async def run(self, context, format=CONFIG.prompt_format):
        prompt_template, format_example = get_template(templates, format)
        prompt = prompt_template.format(context=context, format_example=format_example)
        rsp = await self._aask_v1(prompt, "task", OUTPUT_MAPPING, format=format)
        self._save(context, rsp)
        return rsp


```



这段代码定义了一个名为AssignTasks的类，它实现了Action接口。这个类的定义了一个名为run的异步方法，该方法接受一个或多个参数，并使用关键字args和kwargs来获取这些参数。在这个方法中，你应该实现实际的操作。

具体来说，AssignTasks类中的run方法会在指定的参数被传递给它之后，等待事件的发生，然后执行操作并返回结果。这个操作可以被视为一个任务，它可以在运行时动态地分配给指定的应用程序上下文。

AssignTasks类继承自Action类，这意味着它也实现了Action接口的所有方法。具体来说，Action接口定义了以下方法：

- base.Action：返回一个异步任务对象，代表异步操作已经完成，可以用于调用run等方法。
- base.Action.run(action.Argument)：返回一个异步任务对象，代表运行指定的操作。
- base.Action.run_async(action.Argument)：返回一个异步任务对象，代表运行指定的操作，并返回一个 Future 对象。

因此，AssignTasks类中的run方法实际上是一个实现了Action接口的异步方法，用于执行指定的任务。


```py
class AssignTasks(Action):
    async def run(self, *args, **kwargs):
        # Here you should implement the actual action
        pass

```

# `metagpt/actions/research.py`

这段代码是一个Python脚本，使用了Python 2.7的语法。以下是对脚本的作用和部分的解释：

1. `#!/usr/bin/env python`：这是一个 shebang，指定了脚本的解释器为 `/usr/bin/env python`，即使用 Python 2.7 的解释器来运行此脚本。

2. `from __future__ import annotations`：这是一个 future_default_behavior，表示在未来的 Python 版本中，会自动采用该行为的设计。

3. `import asyncio`：引入了 asyncio 库，使得脚本能够使用异步编程。

4. `import json`：引入了 json 库，用于将数据以 JSON 格式存储。

5. `from typing import Callable`：引入了 typing 库，用于支持引用了来自不同预期的类型的变量。

6. `from metagpt.actions import Action`：引入了 Action 类，该类用于定义在 Metagpt 2.0 上下文中执行的操作。

7. `from metagpt.config import CONFIG`：引入了 CONFIG 类，该类用于将配置文件中的参数读取到应用程序中。

8. `from metagpt.logs import logger`：引入了 logger 类，用于在 Metagpt 2.0 中记录和输出日志信息。

9. `from metagpt.tools.search_engine import SearchEngine`：引入了 SearchEngine 类，该类用于在 Metagpt 2.0 中执行搜索操作。

10. `from metagpt.tools.web_browser_engine import WebBrowserEngine, WebBrowserEngineType`：引入了 WebBrowserEngine 类，用于在 Metagpt 2.0 中打开一个 Web 浏览器。

11. `Action.from_function`：将一个函数包装成 Action，该函数将在 Metagpt 2.0 中执行该操作。

12. `logger.bind`：将 logger 类中的 `__call__` 方法与一个函数绑定，使得可以在将来的日志记录中使用当前的上下文信息来调用该函数。

13. `logger.register_callback`：将 logger 类中的 `__call__` 方法与一个回调函数绑定，用于在将来的日志记录中执行该回调函数。

14. `asyncio.create_task_fn`：创建一个异步函数，该函数将在异步操作完成后返回结果。

15. `typing.cast`：从已知类型中创建一个对象，如果已知类型与给定的参数类型不匹配，将返回给定的参数类型。

16. `await`：在异步操作中使用 `await` 关键字，用于暂停执行并等待异步操作的结果。

17. `aiohttp`：导入了一个名为 `aiohttp` 的库，该库与 `asyncio` 一起用于处理 HTTP 请求。

18. `urlopen`：是一个 URL 打开函数，用于打开一个 URL并返回其内容。

19. `parse_obj_as`：是一个自定义的 `parse_obj_as` 函数，用于将 Pydantic 模型中的对象解析为 Python 对象。

20. `Action.wrap_with`：将一个函数包装成 Action，该函数将在 Metagpt 2.0 中执行该操作，并使用另一个函数作为它的参数。


```py
#!/usr/bin/env python

from __future__ import annotations

import asyncio
import json
from typing import Callable

from pydantic import parse_obj_as

from metagpt.actions import Action
from metagpt.config import CONFIG
from metagpt.logs import logger
from metagpt.tools.search_engine import SearchEngine
from metagpt.tools.web_browser_engine import WebBrowserEngine, WebBrowserEngineType
```

这段代码的作用是定义了几个常量和变量，以及一个函数和一个类，用于生成回答样例和解析研究主题。具体解释如下：

1. `LANG_PROMPT` 是一个变量，用于在回答中指定所处语言。
2. `RESEARCH_BASE_SYSTEM` 是一个字符串，定义了 AI 模型需要基于的研究基础系统。
3. `RESEARCH_TOPIC_SYSTEM` 是一个字符串，定义了 AI 模型需要研究的主题，通常是一个包含标签或话题的名称。
4. `SEARCH_TOPIC_PROMPT` 是一个字符串，定义了生成搜索主题的模板，用于在回答中搜索特定的关键词。
5. `SUMMARIZE_SEARCH_PROMPT` 是一个字符串，定义了如何简要描述搜索结果的信息。
6. `OutputParser` 是从 `metagpt.utils.common` 包中导入的输出解析器类，该类用于解析回答中的分段信息。
7. `generate_prompt_chunk` 是 `metagpt.utils.text.generate_prompt_chunk` 函数，用于生成回答中的分段信息。
8. `reduce_message_length` 是 `metagpt.utils.text.reduce_message_length` 函数，用于减少回答中的消息长度。


```py
from metagpt.utils.common import OutputParser
from metagpt.utils.text import generate_prompt_chunk, reduce_message_length

LANG_PROMPT = "Please respond in {language}."

RESEARCH_BASE_SYSTEM = """You are an AI critical thinker research assistant. Your sole purpose is to write well \
written, critically acclaimed, objective and structured reports on the given text."""

RESEARCH_TOPIC_SYSTEM = "You are an AI researcher assistant, and your research topic is:\n#TOPIC#\n{topic}"

SEARCH_TOPIC_PROMPT = """Please provide up to 2 necessary keywords related to your research topic for Google search. \
Your response must be in JSON format, for example: ["keyword1", "keyword2"]."""

SUMMARIZE_SEARCH_PROMPT = """### Requirements
1. The keywords related to your research topic and the search results are shown in the "Search Result Information" section.
```

这段代码是一个人工智能助手，它根据用户在研究主题上搜索的关键词，提供了多达{decomposition_nums}个相关的查询。

具体来说，这段代码可以执行以下操作：

1. 通过调用一个名为`SearchEngine`的类，使用其`searchUpTo`方法获取与研究主题相关的搜索结果。
2. 构造一个包含研究主题、查询和结果的JSON对象，并将此对象传递给`COLLECT_AND_RANKURLS_PROMPT`函数，这个函数会在对象中添加相应的内容。
3. 构造一个包含多个查询的JSON对象，并将此对象传递给`SearchEngine`的`search`方法，这个方法会使用上面获得的搜索结果，返回匹配查询的列表。
4. 将查询结果返回，以制表符分隔。

总之，这段代码的作用是帮助用户在研究主题上进行搜索，并提供相关的高质量的查询结果。


```py
2. Provide up to {decomposition_nums} queries related to your research topic base on the search results.
3. Please respond in the following JSON format: ["query1", "query2", "query3", ...].

### Search Result Information
{search_results}
"""

COLLECT_AND_RANKURLS_PROMPT = """### Topic
{topic}
### Query
{query}

### The online search results
{results}

```

这段代码是一个人工智能助手，它的目的是根据用户提供的搜索查询来提供相关的信息和答案。它分为两个主要部分：第一部分是要求用户提供的查询必须与搜索主题相关，第二部分是如果搜索查询与主题有关，则第二部分会执行以下操作：1. 通过参考信息中的文本回答问题。2. 如果问题无法直接从参考信息中回答，但文本与主题有关，则提供一份全面的总结。3. 如果文本与主题完全无关，则返回简单的“不相关”。

具体来说，这段代码需要用户提供的查询，然后会对查询进行处理，如果查询与搜索主题有关，则会执行以下操作：1. 从参考信息中的文本中提取相关信息并回答问题。2. 如果查询无法从参考信息中直接回答，但文本与主题有关，则会生成一份全面的总结，可能会包含一些事实信息、数据、统计数据等。3. 如果查询与主题完全无关，则返回一个简单的“不相关”。

总之，这段代码将根据用户提供的搜索查询提供相关的信息，以回答用户的问题或提供有用的帮助。


```py
### Requirements
Please remove irrelevant search results that are not related to the query or topic. Then, sort the remaining search results \
based on the link credibility. If two results have equal credibility, prioritize them based on the relevance. Provide the
ranked results' indices in JSON format, like [0, 1, 3, 4, ...], without including other words.
"""

WEB_BROWSE_AND_SUMMARIZE_PROMPT = '''### Requirements
1. Utilize the text in the "Reference Information" section to respond to the question "{query}".
2. If the question cannot be directly answered using the text, but the text is related to the research topic, please provide \
a comprehensive summary of the text.
3. If the text is entirely unrelated to the research topic, please reply with a simple text "Not relevant."
4. Include all relevant factual information, numbers, statistics, etc., if available.

### Reference Information
{content}
```

这段代码是一个人工智能助手，它提供了一段关于如何根据所给主题进行研究报告的说明。具体来说，这段代码要求用户提供一个关于所选主题的研究报告，报告需满足以下要求：

1. 直接针对所选主题进行深入研究，确保报告内容紧凑、结构清晰、详细、具有可读性。
2. 使用适当的图表、数据和信息来展示研究结果，以便于清晰地表达。
3. 将报告分为适当的段落，使用相应的格式和样式进行排版，使其易于阅读和理解。
4. 在报告的结尾列出所有来源的引用，使用 APA 格式。


```py
'''


CONDUCT_RESEARCH_PROMPT = '''### Reference Information
{content}

### Requirements
Please provide a detailed research report in response to the following topic: "{topic}", using the information provided \
above. The report must meet the following requirements:

- Focus on directly addressing the chosen topic.
- Ensure a well-structured and in-depth presentation, incorporating relevant facts and figures where available.
- Present data and findings in an intuitive manner, utilizing feature comparative tables, if applicable.
- The report should have a minimum word count of 2,000 and be formatted with Markdown syntax following APA style guidelines.
- Include all source URLs in APA format at the end of the report.
```

This is a class that uses the Aikassa AI研究引擎 to perform research on a given topic. It uses the\_aask function to send queries to the research engine, and it defines a custom的搜索结果解析函数. The search engine can search for up to `max_results` URLs and return them in a ranked order. The class also provides methods for logging, extracting information from the search results, and rank the results based on a defined function.


```py
'''


class CollectLinks(Action):
    """Action class to collect links from a search engine."""
    def __init__(
        self,
        name: str = "",
        *args,
        rank_func: Callable[[list[str]], None] | None = None,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)
        self.desc = "Collect links from a search engine."
        self.search_engine = SearchEngine()
        self.rank_func = rank_func

    async def run(
        self,
        topic: str,
        decomposition_nums: int = 4,
        url_per_query: int = 4,
        system_text: str | None = None,
    ) -> dict[str, list[str]]:
        """Run the action to collect links.

        Args:
            topic: The research topic.
            decomposition_nums: The number of search questions to generate.
            url_per_query: The number of URLs to collect per search question.
            system_text: The system text.

        Returns:
            A dictionary containing the search questions as keys and the collected URLs as values.
        """
        system_text = system_text if system_text else RESEARCH_TOPIC_SYSTEM.format(topic=topic)
        keywords = await self._aask(SEARCH_TOPIC_PROMPT, [system_text])
        try:
            keywords = OutputParser.extract_struct(keywords, list)
            keywords = parse_obj_as(list[str], keywords)
        except Exception as e:
            logger.exception(f"fail to get keywords related to the research topic \"{topic}\" for {e}")
            keywords = [topic]
        results = await asyncio.gather(*(self.search_engine.run(i, as_string=False) for i in keywords))

        def gen_msg():
            while True:
                search_results = "\n".join(f"#### Keyword: {i}\n Search Result: {j}\n" for (i, j) in zip(keywords, results))
                prompt = SUMMARIZE_SEARCH_PROMPT.format(decomposition_nums=decomposition_nums, search_results=search_results)
                yield prompt
                remove = max(results, key=len)
                remove.pop()
                if len(remove) == 0:
                    break
        prompt = reduce_message_length(gen_msg(), self.llm.model, system_text, CONFIG.max_tokens_rsp)
        logger.debug(prompt)
        queries = await self._aask(prompt, [system_text])
        try:
            queries = OutputParser.extract_struct(queries, list)
            queries = parse_obj_as(list[str], queries)
        except Exception as e:
            logger.exception(f"fail to break down the research question due to {e}")
            queries = keywords
        ret = {}
        for query in queries:
            ret[query] = await self._search_and_rank_urls(topic, query, url_per_query)
        return ret

    async def _search_and_rank_urls(self, topic: str, query: str, num_results: int = 4) -> list[str]:
        """Search and rank URLs based on a query.

        Args:
            topic: The research topic.
            query: The search query.
            num_results: The number of URLs to collect.

        Returns:
            A list of ranked URLs.
        """
        max_results = max(num_results * 2, 6)
        results = await self.search_engine.run(query, max_results=max_results, as_string=False)
        _results = "\n".join(f"{i}: {j}" for i, j in zip(range(max_results), results))
        prompt = COLLECT_AND_RANKURLS_PROMPT.format(topic=topic, query=query, results=_results)
        logger.debug(prompt)
        indices = await self._aask(prompt)
        try:
            indices = OutputParser.extract_struct(indices, list)
            assert all(isinstance(i, int) for i in indices)
        except Exception as e:
            logger.exception(f"fail to rank results for {e}")
            indices = list(range(max_results))
        results = [results[i] for i in indices]
        if self.rank_func:
            results = self.rank_func(results)
        return [i["link"] for i in results[:num_results]]


```



This is a class called `WebBrowserEngine` which appears to be responsible for running a web browser engine to search the web and provide summaries based on a given research question.

It has a method called `run` which takes in several arguments:

* `url`: The main URL to browse.
* `urls`: Additional URLs to browse.
* `query`: The research question.
* `system_text`: The system text.

It returns a dictionary containing the URLs as keys and their summaries as values.

It has another method called `generate_prompt_chunk` which appears to be used to generate chunked summaries from一段 text. It takes in a piece of text `content` and a prompt template `prompt_template`, and generates summaries by breaking up the text using the specified prompt template and then returning only the first summary.

It also has a method called `_aask` which appears to be used to Ask the AI to summarize a given text.

Overall, this class seems to be a part of a larger system for summarizing summaries based on research questions.


```py
class WebBrowseAndSummarize(Action):
    """Action class to explore the web and provide summaries of articles and webpages."""
    def __init__(
        self,
        *args,
        browse_func: Callable[[list[str]], None] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if CONFIG.model_for_researcher_summary:
            self.llm.model = CONFIG.model_for_researcher_summary
        self.web_browser_engine = WebBrowserEngine(
            engine=WebBrowserEngineType.CUSTOM if browse_func else None,
            run_func=browse_func,
        )
        self.desc = "Explore the web and provide summaries of articles and webpages."

    async def run(
        self,
        url: str,
        *urls: str,
        query: str,
        system_text: str = RESEARCH_BASE_SYSTEM,
    ) -> dict[str, str]:
        """Run the action to browse the web and provide summaries.

        Args:
            url: The main URL to browse.
            urls: Additional URLs to browse.
            query: The research question.
            system_text: The system text.

        Returns:
            A dictionary containing the URLs as keys and their summaries as values.
        """
        contents = await self.web_browser_engine.run(url, *urls)
        if not urls:
            contents = [contents]

        summaries = {}
        prompt_template = WEB_BROWSE_AND_SUMMARIZE_PROMPT.format(query=query, content="{}")
        for u, content in zip([url, *urls], contents):
            content = content.inner_text
            chunk_summaries = []
            for prompt in generate_prompt_chunk(content, prompt_template, self.llm.model, system_text, CONFIG.max_tokens_rsp):
                logger.debug(prompt)
                summary = await self._aask(prompt, [system_text])
                if summary == "Not relevant.":
                    continue
                chunk_summaries.append(summary)

            if not chunk_summaries:
                summaries[u] = None
                continue

            if len(chunk_summaries) == 1:
                summaries[u] = chunk_summaries[0]
                continue

            content = "\n".join(chunk_summaries)
            prompt = WEB_BROWSE_AND_SUMMARIZE_PROMPT.format(query=query, content=content)
            summary = await self._aask(prompt, [system_text])
            summaries[u] = summary
        return summaries


```

这段代码定义了一个名为 ConductResearch 的动作类，它继承自 Action 类（默认的异常类）。这个类的目的是进行研究和生成研究报告。在初始化方法 "__init__" 中，参数传递给了父类的构造函数，同时将 CONFIG.model_for_researcher_report 设置为模型的实例。

"run" 方法是这个类的行动方法。它接收三个参数：研究主题、研究和报告的内容以及系统文本。系统文本是一个字符串，可以提供一个默认的研究基础系统。这个方法返回生成的研究报告。

在这段代码中，还定义了一个名为 CONFIG 的类，用于存储一些 configurations，例如当前系统文本、模型实例等。不过，这段代码没有使用这些配置类，因为在这个类的行动方法中，我们直接使用了 CONFIG 中的常量。


```py
class ConductResearch(Action):
    """Action class to conduct research and generate a research report."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if CONFIG.model_for_researcher_report:
            self.llm.model = CONFIG.model_for_researcher_report

    async def run(
        self,
        topic: str,
        content: str,
        system_text: str = RESEARCH_BASE_SYSTEM,
    ) -> str:
        """Run the action to conduct research and generate a research report.

        Args:
            topic: The research topic.
            content: The content for research.
            system_text: The system text.

        Returns:
            The generated research report.
        """
        prompt = CONDUCT_RESEARCH_PROMPT.format(topic=topic, content=content)
        logger.debug(prompt)
        self.llm.auto_max_tokens = True
        return await self._aask(prompt, [system_text])


```

这段代码定义了一个名为 `get_research_system_text` 的函数，它接受两个参数，一个是研究主题 `topic`，另一个是用于系统文本输出的编程语言 `language`。

函数内部首先通过 `RESEARCH_TOPIC_SYSTEM` 和 `LANG_PROMPT` 函数获取研究主题和编程语言对应的系统文本，然后使用 `format` 函数将它们组合成一个字符串，并将结果返回。

具体来说，函数内部执行以下操作：

1. 调用 `RESEARCH_TOPIC_SYSTEM` 函数，其中 `topic` 是研究主题，`language` 是编程语言。该函数返回两个参数：`SEARCH_TOPIC_SYSTEM.format(topic=topic)` 和 `LANG_PROMPT.format(language=language)`。

2. 创建一个空字符串，然后使用 `join` 函数将两个参数与空字符串连接起来，得到一个字符串。

3. 调用 `format` 函数，将 `topic` 和 `language` 参数传递给 `LANG_PROMPT.format` 函数，得到一个字符串。

4. 将步骤 2 和 3 得到的结果字符串与空字符串连接起来，得到最终的系统文本字符串，并将其返回。


```py
def get_research_system_text(topic: str, language: str):
    """Get the system text for conducting research.

    Args:
        topic: The research topic.
        language: The language for the system text.

    Returns:
        The system text for conducting research.
    """
    return " ".join((RESEARCH_TOPIC_SYSTEM.format(topic=topic), LANG_PROMPT.format(language=language)))

```

# `metagpt/actions/run_code.py`

这段代码是一个Python脚本，用于运行名为“run_code.py”的程序。

具体来说，它执行以下操作：

1. 导入所需的模块和函数：os、subprocess、traceback、typing.Tuple、Action、logger。

2. 定义了一个名为“Action”的类，继承自Action类，用于执行具体的操作。

3. 定义了一个名为“run_code.py”的函数，该函数会执行以下操作：

a. 导入所需的模块：os、subprocess、traceback。

b. 定义一个名为“logger”的函数，用于记录日志信息。

c. 定义一个名为“Action”的类，继承自Action类，并在其中重写了“__init__”方法，用于初始化该行动。

d. 定义一个名为“MetagptAction”的类，继承自Action类，并重写了“run”方法，用于执行具体的操作。

e. 在脚本的顶部添加了以下代码：

```py
@Time    : 2023/5/11 17:46
@Author  : alexanderwu
@File    : run_code.py
```

表示该脚本是在2023年5月11日17点46分由Alexander Wu编写的，并保存在名为“run_code.py”的文件中。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:46
@Author  : alexanderwu
@File    : run_code.py
"""
import os
import subprocess
import traceback
from typing import Tuple

from metagpt.actions.action import Action
from metagpt.logs import logger

```

这段代码是一个 Prompt，用于提示开发人员在代码运行结果中报告结果。如果运行结果中没有错误，则需要明确批准结果。如果运行结果中指出了错误，则需要指出是开发代码还是测试代码导致了错误，并提供具体的修复建议。

具体来说，这段代码的作用是让开发人员分析代码运行结果，提供错误信息，以及指出导致错误的具体部分。如果运行结果中没有错误，则需要明确批准结果。如果运行结果中指出了错误，则需要给出具体的修复建议，指出是开发代码还是测试代码导致了错误，并提供具体的修复建议。


```py
PROMPT_TEMPLATE = """
Role: You are a senior development and qa engineer, your role is summarize the code running result.
If the running result does not include an error, you should explicitly approve the result.
On the other hand, if the running result indicates some error, you should point out which part, the development code or the test code, produces the error,
and give specific instructions on fixing the errors. Here is the code info:
{context}
Now you should begin your analysis
---
## instruction:
Please summarize the cause of the errors and give correction instruction
## File To Rewrite:
Determine the ONE file to rewrite in order to fix the error, for example, xyz.py, or test_xyz.py
## Status:
Determine if all of the code works fine, if so write PASS, else FAIL,
WRITE ONLY ONE WORD, PASS OR FAIL, IN THIS SECTION
```

这段代码是一个Python脚本，用于生成一个报告，报告的内容是关于软件开发或测试中存在的问题的指出。

第1行是一个调试信息，告诉开发人员或测试人员代码中存在问题。如果是问题出在开发代码中，则输出"Engineer"；如果是问题出在测试代码中，则输出"QaEngineer"；如果是普通情况，则输出"NoOne"。

第3行是一个提示，告诉开发人员或测试人员应该在相应的代码部分填写必要的信息、状态和发送信息。

第5至第8行是代码文件的相关信息，包括文件名、代码内容和注释。

第10至第12行是测试文件的相关信息，包括文件名、代码内容和注释。

第14行是一个空行，用于分离代码文件和测试文件。

第16至第19行是开发人员或测试人员需要填写的内容，包括问题和描述问题的组件。

第22至第25行是可选的内容，包括其他反馈或要求。


```py
## Send To:
Please write Engineer if the errors are due to problematic development codes, and QaEngineer to problematic test codes, and NoOne if there are no errors,
WRITE ONLY ONE WORD, Engineer OR QaEngineer OR NoOne, IN THIS SECTION.
---
You should fill in necessary instruction, status, send to, and finally return all content between the --- segment line.
"""

CONTEXT = """
## Development Code File Name
{code_file_name}
## Development Code
```python
{code}
```py
## Test File Name
{test_file_name}
```

This is a class that appears to be a simple command-line tool for running a Python script or a text-based command. It has several methods for running a script or a command, as well as additional options for running a script or a command.

The `run` method is the main method for running a command. It takes a series of arguments, such as the command to run, the mode of run (script or text), and various optional arguments for the command. It returns the output from the command in a decoded form.

The `run_script` method is specific to running a script. It takes a command and various optional arguments, such as the code file to read from or the name of the script. It returns the output from the command in a decoded form.

The `run_text` method is specific to running text-based commands. It takes a command and various optional arguments, such as the test code to run. It returns the output from the command in a decoded form.

The `_aask` method appears to be a higher-level method for running asyncio commands. It takes a prompt and returns a response. It can be used in a similar way to `run` or `run_script`.

The class also includes several helper methods, such as `additional_python_paths`, which appears to be a list of directories to add to the PATH environment variable.


```py
## Test Code
```python
{test_code}
```py
## Running Command
{command}
## Running Output
standard output: {outs};
standard errors: {errs};
"""


class RunCode(Action):
    def __init__(self, name="RunCode", context=None, llm=None):
        super().__init__(name, context, llm)

    @classmethod
    async def run_text(cls, code) -> Tuple[str, str]:
        try:
            # We will document_store the result in this dictionary
            namespace = {}
            exec(code, namespace)
            return namespace.get("result", ""), ""
        except Exception:
            # If there is an error in the code, return the error message
            return "", traceback.format_exc()

    @classmethod
    async def run_script(cls, working_directory, additional_python_paths=[], command=[]) -> Tuple[str, str]:
        working_directory = str(working_directory)
        additional_python_paths = [str(path) for path in additional_python_paths]

        # Copy the current environment variables
        env = os.environ.copy()

        # Modify the PYTHONPATH environment variable
        additional_python_paths = [working_directory] + additional_python_paths
        additional_python_paths = ":".join(additional_python_paths)
        env["PYTHONPATH"] = additional_python_paths + ":" + env.get("PYTHONPATH", "")

        # Start the subprocess
        process = subprocess.Popen(
            command, cwd=working_directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

        try:
            # Wait for the process to complete, with a timeout
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            logger.info("The command did not complete within the given timeout.")
            process.kill()  # Kill the process if it times out
            stdout, stderr = process.communicate()
        return stdout.decode("utf-8"), stderr.decode("utf-8")

    async def run(
        self, code, mode="script", code_file_name="", test_code="", test_file_name="", command=[], **kwargs
    ) -> str:
        logger.info(f"Running {' '.join(command)}")
        if mode == "script":
            outs, errs = await self.run_script(command=command, **kwargs)
        elif mode == "text":
            outs, errs = await self.run_text(code=code)

        logger.info(f"{outs=}")
        logger.info(f"{errs=}")

        context = CONTEXT.format(
            code=code,
            code_file_name=code_file_name,
            test_code=test_code,
            test_file_name=test_file_name,
            command=" ".join(command),
            outs=outs[:500],  # outs might be long but they are not important, truncate them to avoid token overflow
            errs=errs[:10000],  # truncate errors to avoid token overflow
        )

        prompt = PROMPT_TEMPLATE.format(context=context)
        rsp = await self._aask(prompt)

        result = context + rsp

        return result

```