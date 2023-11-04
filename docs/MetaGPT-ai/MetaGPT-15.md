# MetaGPT源码解析 15

# `tests/metagpt/roles/test_architect.py`

这段代码是一个Python脚本，使用了Python标准库中的pytest库来进行异步测试。以下是对脚本的解释：

1. `#!/usr/bin/env python` 是脚本的元数据，告诉操作系统如何解释脚本。在这里，它指定了脚本的路径和后缀，以便操作系统知道如何调用脚本中的代码。
2. `-*- coding: utf-8 -*-` 是编码元数据，告诉操作系统如何处理脚本中的字节序列。在这里，它指定了编码为UTF-8，以便能够正确地处理从标准库中导入的Python字节序列。
3. `@Time    : 2023/5/20 14:37` 和 `@Author  : alexanderwu` 是时间和作者信息，告诉操作系统脚本的时间和作者。这些信息通常是可选的，但如果不提供，脚本仍然可以正常运行。
4. `import pytest` 是导入pytest库，以便能够在脚本中使用pytest提供的异步测试框架。
5. `from metagpt.logs import logger` 是导入metagpt.logs库，以便使用该库中的logger函数。
6. `from metagpt.roles import Architect` 是导入metagpt.roles库，以便使用该库中的Architect类。
7. `from tests.metagpt.roles.mock import MockMessages` 是导入test.metagpt.roles.mock库，以便使用该库中的MockMessages类。
8. `@pytest.mark.asyncio` 是mark.asyncio元数据，告诉pytest运行脚本时使用异步编程。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/20 14:37
@Author  : alexanderwu
@File    : test_architect.py
"""
import pytest

from metagpt.logs import logger
from metagpt.roles import Architect
from tests.metagpt.roles.mock import MockMessages


@pytest.mark.asyncio
```

这段代码定义了一个名为 `test_architect` 的函数，它属于一个名为 `Architect` 的类。该函数的作用是测试该类中包含的收发消息功能是否可以正常工作。

具体来说，函数接收一个名为 `MockMessages.req` 的消息，然后将其传递给类中的 `recv` 方法。接着，函数使用一个名为 `MockMessages.prd` 的消息来调用类中的 `handle` 方法，并将 `handle` 方法返回的结果存储在名为 `rsp` 的变量中。

最后，函数使用一个名为 `logger.info` 的函数来打印出 `rsp` 的内容。根据函数内部的内容，可以推断出 `rsp` 变量中应该包含一个有多个元素的消息列表。


```py
async def test_architect():
    role = Architect()
    role.recv(MockMessages.req)
    rsp = await role.handle(MockMessages.prd)
    logger.info(rsp)
    assert len(rsp.content) > 0

```

# `tests/metagpt/roles/test_engineer.py`

这段代码是一个Python脚本，使用了Python的标准库中的pytest库进行测试驱动程序的编写。

该脚本的作用是测试一个名为"test_engineer.py"的函数，该函数可能对一个名为"test_engineer"的函数进行测试。

具体来说，该脚本使用metagpt库中的日志输出功能，通过在函数内输出日志信息，方便后续追踪和调试问题。

该脚本使用metagpt库中的Engineer类，可能对一个Engineer对象进行测试。

该脚本使用metagpt库中的CodeParser类，对一个字符串进行解析，以获取其中的元数据。

该脚本使用metagpt库中的TASK类和TASK_TOMATO_CLOCK类，可能代表一个测试任务，以及与该任务相关的时钟。

该脚本使用metagpt库中的MockMessages类，创建一些模拟的消息，用于模拟函数的输出和行为。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 10:14
@Author  : alexanderwu
@File    : test_engineer.py
"""
import pytest

from metagpt.logs import logger
from metagpt.roles.engineer import Engineer
from metagpt.utils.common import CodeParser
from tests.metagpt.roles.mock import (
    STRS_FOR_PARSING,
    TASKS,
    TASKS_TOMATO_CLOCK,
    MockMessages,
)


```

这段代码使用了Python的异步编程库——asyncio，并进行了一个单元测试来验证asyncio的作用。

测试1：EngineerTestCase
-----------------------------

```pypython
@pytest.mark.asyncio
async def test_engineer():
   engineer = Engineer()

   # 发送请求
   engineer.recv(MockMessages.req)
   engineer.recv(MockMessages.prd)
   engineer.recv(MockMessages.system_design)

   # 期望响应
   await engineer.handle(MockMessages.tasks)

   # 记录日志
   logger.info(await rsp.content for rsp in await engineer.handle(MockMessages.tasks))

   assert "all done." == rsp.content
```

这段代码的作用是测试异步编程中Engineer类的作用。首先创建一个Engineer对象，然后使用recv()方法接收请求、产品设计和系统设计消息。接着，使用handle()方法处理这些消息并获取期望的响应。最后，通过handle()方法将处理结果记录到logger中，并验证期望的响应是否与实际一致。

测试2：test_parse_str
---------------------------

```pypython
@pytest.mark.asyncio
def test_parse_str():
   for idx, i in enumerate(STRS_FOR_PARSING):
       text = CodeParser.parse_str(f"{idx+1}", i)
       # logger.info(text)
       assert text == 'a'
```

这段代码的作用是验证可以解析的文本是否与预期相同。首先，定义一个枚举STRS_FOR_PARSING，用于存储需要解析的文本编号。接着，定义一个CodeParser类，该类用于解析文本。然后，使用for循环遍历STRS_FOR_PARSING中的每个文本，并使用parse_str()方法解析该文本。最后，比较解析后的文本与预期的文本是否相同，并记录到logger中。


```py
@pytest.mark.asyncio
async def test_engineer():
    engineer = Engineer()

    engineer.recv(MockMessages.req)
    engineer.recv(MockMessages.prd)
    engineer.recv(MockMessages.system_design)
    rsp = await engineer.handle(MockMessages.tasks)

    logger.info(rsp)
    assert "all done." == rsp.content


def test_parse_str():
    for idx, i in enumerate(STRS_FOR_PARSING):
        text = CodeParser.parse_str(f"{idx+1}", i)
        # logger.info(text)
        assert text == 'a'


```

这段代码是一个 Python 函数，名为 `test_parse_blocks()`，它用于测试 `CodeParser.parse_blocks()` 函数的正确性。函数的主要目的是验证 `parse_blocks()` 函数能够正确地将给定的任务列表解析成一个或多个 Python 代码块。

具体来说，这段代码首先定义了一个名为 `tasks` 的变量，它可能是通过 `CodeParser.parse_blocks()` 函数获取的。然后，代码通过 `logger.info()` 函数输出 `tasks` 字典的键，以便观察解析出的代码块。接着，代码使用 `assert` 语句检查 `tasks` 键是否包含名为 `Task list` 的键，这意味着 `parse_blocks()` 函数正确地将所有的任务列表解析成了一个或多个 Python 代码块。

最后，代码将一个包含多个任务的列表（可能是从 `target_list` 变量中获取的）作为参数传递给 `parse_blocks()` 函数，然后观察它是否正确地解析出了这些代码块。


```py
def test_parse_blocks():
    tasks = CodeParser.parse_blocks(TASKS)
    logger.info(tasks.keys())
    assert 'Task list' in tasks.keys()


target_list = [
    "smart_search_engine/knowledge_base.py",
    "smart_search_engine/index.py",
    "smart_search_engine/ranking.py",
    "smart_search_engine/summary.py",
    "smart_search_engine/search.py",
    "smart_search_engine/main.py",
    "smart_search_engine/interface.py",
    "smart_search_engine/user_feedback.py",
    "smart_search_engine/security.py",
    "smart_search_engine/testing.py",
    "smart_search_engine/monitoring.py",
]


```

这段代码是一个函数 `test_parse_file_list()`，它测试了 `CodeParser.parse_file_list()` 函数的正确性。

具体来说，这段代码首先定义了一个名为 `tasks` 的列表变量，使用 `CodeParser.parse_file_list()` 函数将 "任务列表" 和 "Task list" 两个文件的内容读取到这个列表中。然后，代码输出这个列表，使用 `assert isinstance()` 断言，确保这个列表是一个有效的列表。接着，代码使用相同的 `CodeParser.parse_file_list()` 函数读取 "Task list" 文件的内容，并输出一个名为 `file_list` 的列表变量。使用 `assert isinstance()` 断言，确保这个列表是一个有效的列表。

接下来，代码将 `tasks` 和 `file_list` 进行比较，使用 `assert all()` 断言，确保这两个列表是等价的。如果 `tasks` 和 `file_list` 不等价，说明 `CodeParser.parse_file_list()` 函数在解析文件列表时出现了错误，从而导致这个测试失败。


```py
def test_parse_file_list():
    tasks = CodeParser.parse_file_list("任务列表", TASKS)
    logger.info(tasks)
    assert isinstance(tasks, list)
    assert target_list == tasks

    file_list = CodeParser.parse_file_list("Task list", TASKS_TOMATO_CLOCK, lang="python")
    logger.info(file_list)


target_code = """task_list = [
    "smart_search_engine/knowledge_base.py",
    "smart_search_engine/index.py",
    "smart_search_engine/ranking.py",
    "smart_search_engine/summary.py",
    "smart_search_engine/search.py",
    "smart_search_engine/main.py",
    "smart_search_engine/interface.py",
    "smart_search_engine/user_feedback.py",
    "smart_search_engine/security.py",
    "smart_search_engine/testing.py",
    "smart_search_engine/monitoring.py",
]
```

这段代码是一个测试函数，它的作用是测试 `CodeParser.parse_code` 函数的正确性。

具体来说，这段代码以下两部分：

1. `test_parse_code` 是函数名，它指定了要测试的内容。

2. `CodeParser.parse_code` 是函数实体，它包含了测试函数中要使用的功能。

3. `("任务列表", TASKS, lang="python")` 参数是一个字符串，它指定了输入数据的格式。其中，`"任务列表"` 是输入数据的标题，`TASK` 是输入数据的列数，`lang="python"` 是输入数据的语言。

4. `parse_code` 函数会使用 `TASK` 和 `lang="python"` 参数来解析输入数据，并返回解析后的代码。

5. `logger.info(code)` 是在调用 `parse_code` 函数之后，将解析后的代码输出到控制台。

6. `assert isinstance(code, str)` 是在输出解析后的代码之后，使用 `assert` 语句来确保输出的是一个字符串类型。

7. `assert target_code == code` 是在输出解析后的代码之后，使用 `assert` 语句来确保输出的是一个字符串类型。


```py
"""


def test_parse_code():
    code = CodeParser.parse_code("任务列表", TASKS, lang="python")
    logger.info(code)
    assert isinstance(code, str)
    assert target_code == code

```

# `tests/metagpt/roles/test_invoice_ocr_assistant.py`

这段代码是一个Python脚本，它实现了基于MetagPT OCR模型的发票自动识别功能。以下是代码的主要部分：

```py
#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
@Time    : 2023/9/21 23:11:27
@Author  : Stitch-z
@File    : test_invoice_ocr_assistant.py
"""

from pathlib import Path
import pytest
import pandas as pd
from metagpt.roles.invoice_ocr_assistant import InvoiceOCRAssistant

# 定义测试用例的输入文件夹路径
input_folder = Path("/path/to/your/test/input")
output_folder = Path("/path/to/output")

# 读取测试数据集
test_data = pd.read_csv(
   f"{input_folder}/input/data/test_data.csv",
   header=None,
   engine="pandas",
)

# 读取训练数据集
train_data = pd.read_csv(
   f"{input_folder}/input/data/train_data.csv",
   header=None,
   engine="pandas",
)

# 设置评估指标
评估指标 = "准确率"

# 发票OCR模型
invoice_ocr_assistant = InvoiceOCRAssistant()

# 对训练数据集进行预测
predictions = invoice_ocr_assistant.predict(train_data)

# 对测试数据集进行预测
test_predictions = invoice_ocr_assistant.predict(test_data)

# 输出预测结果
print(f"{评估指标}精度： {100:.2f}%")

# 输出预测结果
print(f"{评估指标}精度： {100:.2f}%")
```

这段代码首先定义了测试用例输入文件夹和输出文件夹，然后从输入数据集中读取测试数据和训练数据。接着，定义评估指标，然后创建发票OCR模型实例。接着使用predict方法对训练数据集进行预测，并对测试数据集进行预测。最后，输出预测结果。


```py
#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
@Time    : 2023/9/21 23:11:27
@Author  : Stitch-z
@File    : test_invoice_ocr_assistant.py
"""

from pathlib import Path

import pytest
import pandas as pd

from metagpt.roles.invoice_ocr_assistant import InvoiceOCRAssistant
```

This appears to be a description of an invoice with multiple lines. Each line corresponds to a different invoice and includes the following information:

1. Invoicing date: The date when the invoice was issued.
2. Invoicing number: An optional invoice number that can be used to identify the invoice.
3. Total amount/price: The total amount charged for the invoice, including taxes, fees, and any applicable discounts.
4. Accounting date: The date when the invoice should be recorded in the accounting system.
5. Client: The name or description of the client who received the invoice.
6. Address: The address to which the invoice should be sent.
7. Expiration date: The date on which the invoice expires or becomes due.
8. Download link: A link allowing the recipient to download the invoice file.

The invoice appears to be for a purchase made by the client for goods and services provided by a supplier. The invoice includes line items for the purchase and specifies the amounts and dates of each item.


```py
from metagpt.schema import Message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "invoice_path", "invoice_table_path", "expected_result"),
    [
        (
            "Invoicing date",
            Path("../../data/invoices/invoice-1.pdf"),
            Path("../../../data/invoice_table/invoice-1.xlsx"),
            [
                {
                    "收款人": "小明",
                    "城市": "深圳市",
                    "总费用/元": 412.00,
                    "开票日期": "2023年02月03日"
                }
            ]
        ),
        (
            "Invoicing date",
            Path("../../data/invoices/invoice-2.png"),
            Path("../../../data/invoice_table/invoice-2.xlsx"),
            [
                {
                    "收款人": "铁头",
                    "城市": "广州市",
                    "总费用/元": 898.00,
                    "开票日期": "2023年03月17日"
                }
            ]
        ),
        (
            "Invoicing date",
            Path("../../data/invoices/invoice-3.jpg"),
            Path("../../../data/invoice_table/invoice-3.xlsx"),
            [
                {
                    "收款人": "夏天",
                    "城市": "福州市",
                    "总费用/元": 2462.00,
                    "开票日期": "2023年08月26日"
                }
            ]
        ),
        (
            "Invoicing date",
            Path("../../data/invoices/invoice-4.zip"),
            Path("../../../data/invoice_table/invoice-4.xlsx"),
            [
                {
                    "收款人": "小明",
                    "城市": "深圳市",
                    "总费用/元": 412.00,
                    "开票日期": "2023年02月03日"
                },
                {
                    "收款人": "铁头",
                    "城市": "广州市",
                    "总费用/元": 898.00,
                    "开票日期": "2023年03月17日"
                },
                {
                    "收款人": "夏天",
                    "城市": "福州市",
                    "总费用/元": 2462.00,
                    "开票日期": "2023年08月26日"
                }
            ]
        ),
    ]
)
```

这段代码定义了一个名为 `test_invoice_ocr_assistant` 的函数，它接受四个参数：查询文本 `query`、发票文件路径 `invoice_path`、发票表格文件路径 `invoice_table_path` 和预期结果列表 `expected_result`。

函数首先将 `invoice_path` 设置为当前工作目录中的发票文件所在路径，然后创建一个名为 `InvoiceOCRAssistant` 的类，并使用 `run` 方法运行该类的一个实例，该实例接收一个包含查询文本和发票文件路径的 `Message` 对象作为参数。

运行 `InvoiceOCRAssistant` 实例后，发票表格文件将被读取并存储为 `df` 数据框。然后，函数将数据框 `df` 按行解析为字典 `dict_result`，并检查它是否与预期结果列表 `expected_result` 相同。

如果 `dict_result` 和 `expected_result` 不同，函数将输出一个错误消息并退出。


```py
async def test_invoice_ocr_assistant(
    query: str,
    invoice_path: Path,
    invoice_table_path: Path,
    expected_result: list[dict]
):
    invoice_path = Path.cwd() / invoice_path
    role = InvoiceOCRAssistant()
    await role.run(Message(
        content=query,
        instruct_content={"file_path": invoice_path}
    ))
    invoice_table_path = Path.cwd() / invoice_table_path
    df = pd.read_excel(invoice_table_path)
    dict_result = df.to_dict(orient='records')
    assert dict_result == expected_result


```

# `tests/metagpt/roles/test_product_manager.py`

这段代码是一个Python脚本，使用了Python的asyncio库，用于测试一个名为test_product_manager.py的函数。

具体来说，这段代码的作用是定义了一个名为ProductManager的类，该类使用了metagpt.roles库，用于模拟一个产品经理的角色。在测试中，使用了pytest库来运行测试，并使用了一个名为MockMessages的虚拟对象，用于模拟和测试函数的行为。

ProductManager类包括了一些静态方法，用于设置和清除对象的状态。其中最重要的是，有一个名为role的静态方法，该方法可以模拟产品经理的角色行为，包括创建、编辑、删除商品等操作。

此外，还定义了一个名为message的静态方法，用于模拟各种场景下的消息，以便在测试中测试ProductManager类的行为。

最后，定义了一个测试类，该类继承自pytest.mark.asyncio，并定义了测试函数的参数和返回值。该测试函数使用了一个准备工作的装饰器，用于在测试开始时加载metagpt.roles库。

总的来说，这段代码的作用是定义了一个ProductManager类的模拟对象，并使用pytest库和metagpt.roles库来模拟产品经理的角色行为，以便在测试中测试该类的方法和属性。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/16 14:50
@Author  : alexanderwu
@File    : test_product_manager.py
"""
import pytest

from metagpt.logs import logger
from metagpt.roles import ProductManager
from tests.metagpt.roles.mock import MockMessages


@pytest.mark.asyncio
```

这段代码定义了一个名为 `test_product_manager` 的异步函数，它使用 `ProductManager` 类来处理一个名为 `MockMessages.req` 的消息。

具体来说，这个函数创建了一个 `ProductManager` 对象，然后使用 `handle` 方法来处理消息。处理成功后，使用 `assert` 语句来检查消息的内容是否符合预期。

在这个例子中， `ProductManager` 类可能是一个用于管理产品目标（product goals）的 API 类。假设 `ProductManager` 类具有 `handle` 方法来接收和处理消息，那么这个函数可能会使用这个方法来处理收到的消息并执行相应的操作。

由于这个函数使用了 `asyncio` 库，因此它使用了 `await` 关键字来让函数在执行时等待结果。在这个例子中， `ProductManager` 类的 `handle` 方法使用 `asyncio.sleep` 函数来等待消息的到达。


```py
async def test_product_manager():
    product_manager = ProductManager()
    rsp = await product_manager.handle(MockMessages.req)
    logger.info(rsp)
    assert len(rsp.content) > 0
    assert "Product Goals" in rsp.content

```

# `tests/metagpt/roles/test_project_manager.py`

这段代码是一个Python脚本，使用了Python的asyncio库，主要用于测试一个名为`test_project_manager.py`的模块。

具体来说，这段代码以下是一个导入了`pytest`和`metagpt.logs`和`metagpt.roles`的模块：

```pypython
# 引入需要的模块
import pytest
import metagpt.logs as logs
import metagpt.roles as roles
from tests.metagpt.roles.mock import MockMessages
```

接下来定义了一个名为`test_project_manager.py`的模块，在其中导入了`pytest`和`metagpt.logs`和`metagpt.roles`，并定义了一个`MockMessages`类，继承自`metagpt.roles.MockMessages`，用于模拟输出日志信息。

接着在`test_project_manager.py`中定义了一个`test_project_manager_睑一本的函数：

```pypython
@pytest.mark.asyncio
async def test_project_manager(request):
   # 在测试函数中，将模拟输出日志信息的函数挂载到请求对象中，这里我们通过异步的方式从下文获取输出日志信息
   output_logger = MockMessages()
   项目经理 = roles.ProjectManager()
   output_logger.output_message.return_value = "输出日志信息"
   output_logger.get_logger.return_value = [output_logger]

   # 在异步函数中，使用项目经理的`output_log`方法，输出日志信息
   await project_manager.output_log("项目经理")

   # 通过异步的方式，获取输出日志信息的响应
   response = await request.text()
   output_log = logs.get_logger().output_message

   # 断言输出日志信息的来源是否为项目经理
   assert "项目经理" in output_log.source
   assert "输出日志信息" in output_log.content
```

最后，在`test_project_manager.py`中定义了`pytest`标记的测试函数，用于测试`output_log`函数是否可以正常工作：

```pypython
@pytest.mark.asyncio
def test_output_log(request):
   # 创建一个模拟输出日志信息的函数
   mock_output_logger = MockMessages()
   mock_output_logger.output_message.return_value = "输出日志信息"
   mock_get_logger.return_value = [mock_output_logger]

   # 使用项目经理的`output_log`方法，输出日志信息
   await project_manager.output_log("项目经理")

   # 通过异步的方式，获取输出日志信息的响应
   response = await request.text()
   output_log = logs.get_logger().output_message

   # 断言输出日志信息的来源是否为项目经理
   assert "项目经理" in output_log.source
   assert "输出日志信息" in output_log.content
```


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 10:23
@Author  : alexanderwu
@File    : test_project_manager.py
"""
import pytest

from metagpt.logs import logger
from metagpt.roles import ProjectManager
from tests.metagpt.roles.mock import MockMessages


@pytest.mark.asyncio
```

这段代码是一个异步函数，名为 `test_project_manager`，它使用 Python 的 `asyncio` 库创建了一个名为 `ProjectManager` 的类，并使用 `handle` 方法处理了一个名为 `MockMessages.system_design` 的消息。

具体来说，这段代码的作用是测试 `ProjectManager` 类的一个方法 `handle`。当 `test_project_manager` 函数运行时，它创建了一个新的 `ProjectManager` 实例，然后使用 `handle` 方法处理了一个消息 `MockMessages.system_design`。

处理消息之后，代码会输出一个 `INFO` 级别的消息，其中包括 `rsp` 对象。`rsp` 对象是 `ProjectManager` 类的一个响应，它可能是处理消息的结果，不过在这个例子中，我们没有进一步的检查。


```py
async def test_project_manager():
    project_manager = ProjectManager()
    rsp = await project_manager.handle(MockMessages.system_design)
    logger.info(rsp)

```

# `tests/metagpt/roles/test_qa_engineer.py`

这段代码是一个Python脚本，它使用`#!/usr/bin/env python`作为命令行参数，表示该脚本使用Python 3作为解释器。

该脚本定义了一个名为`test_qa_engineer.py`的文件。该文件表示该脚本的作用，即解释如何在测试质量引擎器（QA）中使用Python。

在该脚本中，定义了一个名为`test_qa_engineer.py`的文件，这个文件是Python的一个类，它实现了`test_qa_engineer`函数。

`test_qa_engineer`函数的功能是接收一些参数，包括`qa_engineer`对象，需要测试的测试套件列表，以及测试套件的测试计划。通过调用这个函数，可以测试QA引擎的性能。

在文件的最后，定义了一个`#`符号，用于注释，表示该部分内容为注释。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 12:01
@Author  : alexanderwu
@File    : test_qa_engineer.py
"""

```

# `tests/metagpt/roles/test_researcher.py`

这段代码是一个Python脚本，用于测试和模拟自然语言处理（NLP）应用程序中的对话功能。它使用了pytest和metagpt库，通过与系统消息进行交互，来模拟与智能助手或机器学习平台之间的对话。

具体来说，这段代码包括以下功能：

1. 定义了一个名为“mock_llm_ask”的函数，该函数使用Pathlib库的TemporaryDirectory对象创建一个临时文件，并从metagpt库中导入一个名为“researcher”的角色。
2. 使用pytest库的装饰器来运行该函数，作为测试套件中的一部分。
3. 在函数内部，使用random库的函数来生成一个0.5到1之间的随机整数，用于评估用户输入的关键词是否与提示相关。
4. 如果用户输入的关键词与提示相关，函数将返回一个字符串，其中包含与关键词相关的系统消息。
5. 如果用户输入的关键词与提示不相关，函数将返回一个字符串，其中包含“Not relevant”的消息。
6. 如果用户输入的关键词包括“Please provide up to 2 necessary keywords”和/或“Provide up to 4 queries related to your research topic”，函数将尝试返回与研究主题相关的系统消息。
7. 如果用户输入的关键词包括“sort the remaining search results”，函数将返回与剩余搜索结果相关的系统消息。
8. 如果用户输入的关键词包括“Not relevant”和/或“sort the remaining search results”，函数将尝试返回与研究主题相关的系统消息，并将其排序为（1,2）。

总的来说，这段代码通过模拟与机器学习平台或智能助手之间的对话，来测试和评估自然语言处理应用程序的功能和性能。


```py
from pathlib import Path
from random import random
from tempfile import TemporaryDirectory

import pytest

from metagpt.roles import researcher


async def mock_llm_ask(self, prompt: str, system_msgs):
    if "Please provide up to 2 necessary keywords" in prompt:
        return '["dataiku", "datarobot"]'
    elif "Provide up to 4 queries related to your research topic" in prompt:
        return '["Dataiku machine learning platform", "DataRobot AI platform comparison", ' \
            '"Dataiku vs DataRobot features", "Dataiku and DataRobot use cases"]'
    elif "sort the remaining search results" in prompt:
        return '[1,2]'
    elif "Not relevant." in prompt:
        return "Not relevant" if random() > 0.5 else prompt[-100:]
    elif "provide a detailed research report" in prompt:
        return f"# Research Report\n## Introduction\n{prompt}"
    return ""


```

这段代码是一个用于测试 "datarobot" 和 "dataiku" 两个数据科学工具库的Python函数。在这个函数中，使用了 "@pytest.mark.asyncio" 注解来声明一个名为 "asyncio" 的测试标记，这意味着该函数将作为一个异步函数来执行。

函数内部使用了 "with TemporaryDirectory() as dirname" 这一行来创建一个临时目录，并将其赋值给一个名为 "dirname" 的变量。这个目录将在函数运行期间被创建，并在函数退出时被删除。

接下来，使用了 "mocker.patch("metagpt.provider.base_gpt_api.BaseGPTAPI.aask", mock_llm_ask)" 这一行来模拟 "metagpt.provider.base_gpt_api.BaseGPTAPI.aask" 函数的行为。这个函数的作用是模拟向 "metagpt.provider.base_gpt_api.BaseGPTAPI" 类中的 "aask" 方法发送请求并获取响应的行为。注意，这里使用了 "mock_llm_ask" 函数来模拟实际请求的行为，而不是 "metagpt.provider.base_gpt_api.BaseGPTAPI.aask" 函数本身。

然后，在 "async researcher.Researcher().run(topic)" 这一行中，使用了 "asyncio" 中的 "Researcher" 类来创建一个 "asyncio" 任务，并使用了 "topic" 变量来指定要搜索的研究主题。

最后，使用了 "assert (researcher.RESEARCH_PATH / f"{topic}.md").read_text().startswith("# Research Report")" 这一行来验证搜索结果是否符合预期。具体来说，这段代码假设 "datarobot" 和 "dataiku" 两个数据科学工具库已经被安装在系统环境中，并且 "asyncio" 函数和 "Researcher" 类都已经被导入了相应的模块中。如果搜索结果符合预期，那么这段代码就不会输出任何错误信息，否则就会输出错误信息。


```py
@pytest.mark.asyncio
async def test_researcher(mocker):
    with TemporaryDirectory() as dirname:
        topic = "dataiku vs. datarobot"
        mocker.patch("metagpt.provider.base_gpt_api.BaseGPTAPI.aask", mock_llm_ask)
        researcher.RESEARCH_PATH = Path(dirname)
        await researcher.Researcher().run(topic)
        assert (researcher.RESEARCH_PATH / f"{topic}.md").read_text().startswith("# Research Report")

```

# `tests/metagpt/roles/test_tutorial_assistant.py`

该代码是一个Python脚本，使用了神秘的`#!`语法。它用于在测试脚本中定义一个函数`test_tutorial_assistant.py`。

函数内导入了两个模块：`aiofiles` 和 `pytest`。

然后，导入了另一个模块`metagpt.roles.tutorial_assistant`，并定义了一个名为`TutorialAssistant`的类。

接下来，定义了一个`test_tutorial_assistant.py`函数，它的作用是测试`metagpt.roles.tutorial_assistant`中的`TutorialAssistant`类。

该函数使用了`@pytest.mark.asyncio`和`@pytest.mark.parametrize`装饰，用于异步编程和参数枚举。

具体来说，`@pytest.mark.asyncio`表示该函数是一个异步函数，而`@pytest.mark.parametrize`表示该函数有两个参数，分别是`language`和`topic`，它们都是字符串类型。


```py
#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
@Time    : 2023/9/6 23:11:27
@Author  : Stitch-z
@File    : test_tutorial_assistant.py
"""
import aiofiles
import pytest

from metagpt.roles.tutorial_assistant import TutorialAssistant


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("language", "topic"),
    [("Chinese", "Write a tutorial about Python")]
)
```

这段代码定义了一个名为 `test_tutorial_assistant` 的异步函数，它接受两个参数 `language` 和 `topic`，分别表示学习语言和教程的主题。函数内部首先定义了一个常量 `topic`，表示要写的教程主题。接着，函数创建了一个名为 `TutorialAssistant` 的类，这个类可能是用于编写教程的虚拟助手。函数内部使用 `await` 关键字，让 `run` 方法等待 `topic` 参数的完成，然后返回 `msg` 对象。接着，函数尝试读取 `filename` 并返回其文件内容。然后，函数使用 `aiofiles` 库打开文件，并使用 `await` 关键字让文件内容以 `f"{title}"` 的形式读取。最后，函数使用 `assert` 检查文件内容是否以 `f"{title}"` 的形式开头。


```py
async def test_tutorial_assistant(language: str, topic: str):
    topic = "Write a tutorial about MySQL"
    role = TutorialAssistant(language=language)
    msg = await role.run(topic)
    filename = msg.content
    title = filename.split("/")[-1].split(".")[0]
    async with aiofiles.open(filename, mode="r") as reader:
        content = await reader.read()
        assert content.startswith(f"# {title}")
```

# `tests/metagpt/roles/test_ui.py`

这段代码的作用是测试一个名为"UI Design"的角色的属性和行为。首先，它从metagpt库中导入SoftwareCompany和ProductManager类，这两个类可能用于管理软件和产品。接着，它定义了一个名为"test_add_ui"的函数，这个函数没有具体的实现，只是起到测试的作用。最后，它导入了两个测试函数，一个用于测试"UI Design"角色的"profile"属性，另一个用于测试添加用户的能力。


```py
# -*- coding: utf-8 -*-
# @Date    : 2023/7/22 02:40
# @Author  : stellahong (stellahong@fuzhi.ai)
#
from metagpt.software_company import SoftwareCompany
from metagpt.roles import ProductManager

from tests.metagpt.roles.ui_role import UI


def test_add_ui():
    ui = UI()
    assert ui.profile == "UI Design"


```

这段代码定义了一个名为 `test_ui_role` 的函数，是一个异步函数，它接受一个字符串参数 `idea`，一个浮点数参数 `investment`，和一个整数参数 `n_round`。函数内部使用了一家名为 `SoftwareCompany` 的类来创建一个模拟公司，这个公司拥有一个名为 `hire` 的方法来招募员工，一个名为 `invest` 的方法来投资，一个名为 `start_project` 的方法来启动新项目，一个名为 `run` 的方法来运行项目。

具体来说，这段代码会创建一个名为 `SoftwareCompany` 的类，并分别向其中添加一个名为 `ProductManager` 的实例作为员工和另一个名为 `UI` 的无限循环实例。然后，它会对传入的 `investment` 参数创建一个 `float` 对象，并将其设置为 `3.0`.接下来，它会调用 `start_project` 方法，并将 `idea` 和 `n_round` 参数传递给该方法。

最后，它会等待 `n_round` 次运行周期结束，并依次调用 `run` 方法 `n_round` 次，以确保 `n_round` 次连续运行。每次运行 `run` 方法时，它都会创建一个新的 instance of `SoftwareCompany`，并添加 `ProductManager` 和 `UI` 实例，然后运行 `run` 方法 5 次，每次运行的时间间隔为 `n_round`。


```py
async def test_ui_role(idea: str, investment: float = 3.0, n_round: int = 5):
    """Run a startup. Be a boss."""
    company = SoftwareCompany()
    company.hire([ProductManager(), UI()])
    company.invest(investment)
    company.start_project(idea)
    await company.run(n_round=n_round)

```

# `tests/metagpt/roles/ui_role.py`

这段代码是一个Python脚本，它实现了使用富文本生成（Metagpt）API自动生成PRD（产品需求文档）的功能。以下是代码的主要功能和使用方法：

1. 导入必要的模块和函数：os、re、functools、importlib、metagpt.actions、metagpt.const、metagpt.logs、metagpt.roles、metagpt.schema、metagpt.tools.sd_engine。

2. 从metagpt.actions模块中导入Action、ActionOutput和WritePRD；从metagpt.const模块中导入WORKSPACE_ROOT；从metagpt.logs模块中导入logger；从metagpt.roles模块中导入Role；从metagpt.schema模块中导入Message；从metagpt.tools.sd_engine模块中导入SDEngine。

3. 使用functools库中的 wraps函数来创建一个装饰器，该装饰器将 Metagpt Action 和输出包装成函数式接口；使用metagpt.actions模块中的 Action 和 ActionOutput创建一个 Action 类，用于表示PRD的每个自然语言文本；使用metagpt.const模块中的 WORKSPACE_ROOT 变量设置工作区根目录；使用metagpt.logs模块中的 logger 函数来实现一个简单的日志记录功能；使用metagpt.roles模块中的 Role 类表示用户角色，根据用户角色可以生成不同的文本；使用metagpt.schema模块中的 Message 类表示PRD的每个自然语言文本，可以设置消息的优先级、是否需要回复、是否对所有用户可见等属性；使用metagpt.tools.sd_engine模块中的 SDEngine 类来实现对 Metagpt API 的请求发送。

4. 在主程序中调用 Action、ActionOutput 和 WritePRD 中提供的类和函数。具体来说，首先创建一个 WritePRD 类的实例，然后使用 WritePRD.generate 方法来生成 PRD。在生成过程中，可以根据需要设置参数，比如设置时间、需求类型、描述、优先级等。


```py
# -*- coding: utf-8 -*-
# @Date    : 2023/7/15 16:40
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
import os
import re
from functools import wraps
from importlib import import_module

from metagpt.actions import Action, ActionOutput, WritePRD
from metagpt.const import WORKSPACE_ROOT
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.tools.sd_engine import SDEngine

```

这段代码是一个Python Prompt模板，用于生成产品需求描述（PRD）的UI设计描述。它将一个字典作为参数传递给这个模板，并包含了一些提示用户在模板中填写空缺信息，例如设计目标，格式样本，所需元素和UI样式等。

具体来说，这段代码的作用是：

1. 定义了一个名为"PROMPT_TEMPLATE"的字典模板，其中包含了一些提示用户填写空缺信息的变量。
2. 通过调用这段代码，将指定的字典作为参数传递给模板，并在模板中填充这些变量。
3. 通过输出模板的格式样本，向用户展示了需要填写的字段以及模板的结构，以便用户更好地理解填写内容。
4. 通过输出指定的元素和UI样式，向用户展示了需要采用的样式，以便用户更好地了解需要设计的UI。
5. 通过提供一些提示信息，帮助用户更好地理解需求和格式，并提供了一个简单的方式来完成这个任务。


```py
PROMPT_TEMPLATE = """
# Context
{context}

## Format example
{format_example}
-----
Role: You are a UserInterface Designer; the goal is to finish a UI design according to PRD, give a design description, and select specified elements and UI style.
Requirements: Based on the context, fill in the following missing information, provide detailed HTML and CSS code
Attention: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the code and triple quote.

## UI Design Description:Provide as Plain text, place the design objective here
## Selected Elements:Provide as Plain text, up to 5 specified elements, clear and simple
## HTML Layout:Provide as Plain text, use standard HTML code
## CSS Styles (styles.css):Provide as Plain text,use standard css code
```

This code appears to be a Python script, and it is not providing any output or explanation of its own.

However, if we were to examine the code, we would see that it consists of two main blocks of code:

1. A `FORMAT_EXAMPLE` constant defined as follows:
```py
const FORMAT_EXAMPLE = """

   UI Design Description

   Snake games are classic and addictive games with simple yet engaging elements. Here are the main elements commonly found in snake games 

   Game Grid: The game grid is a rectangular array of elements that the player can use to block access to the Snake.

   Snake: The player controls a snake that moves across the grid.
```
This constant is likely used to emphasize the UI design description that it follows.

2. A `const selectedElements = ["Game Grid","Snake"]` variable defined as follows:
```py
const selectedElements = ["Game Grid","Snake"]
```
This variable is likely used to keep track of the elements that are currently selected by the player (or maybe to change their selection later on).


```py
## Anything UNCLEAR:Provide as Plain text. Make clear here.

"""

FORMAT_EXAMPLE = """

## UI Design Description
```Snake games are classic and addictive games with simple yet engaging elements. Here are the main elements commonly found in snake games ```py

## Selected Elements

Game Grid: The game grid is a rectangular...

Snake: The player controls a snake that moves across the grid...

```

这段代码是一个网页上的 Snake 游戏。它包括以下几个主要部分：

1. HTML 布局：定义了页面的基本结构，包括游戏区域、分数和游戏结束的提示。
2. 样式表：包含了样式定义，包括背景颜色、字体大小和游戏区域的大小和布局。
3. 游戏逻辑：包括 Snake 的移动、取食和得分功能，以及判断游戏是否结束。
4. 图形元素：绘制了 Snake 游戏中的各个元素，如食物、蛇和得分牌。

总体来说，这段代码实现了一个 Snake 游戏的基本框架，玩家可以通过交互来控制游戏中的 Snake，并获取分数。在游戏的结束部分，玩家可以通过点击重新开始游戏按钮来重新开始。


```py
Food: Food items (often represented as small objects or differently colored blocks)

Score: The player's score increases each time the snake eats a piece of food. The longer the snake becomes, the higher the score.

Game Over: The game ends when the snake collides with itself or an obstacle. At this point, the player's final score is displayed, and they are given the option to restart the game.


## HTML Layout
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Snake Game</title>
    <link rel="stylesheet" href="styles.css">
```

This HTML code creates a div with a class of "game-grid" that contains a div with a class of "Snake" that will be dynamically generated using JavaScript. Additionally, there is a div with a class of "food" that will be dynamically generated using JavaScript.

The CSS styles (styles.css) are used to style the page. The body is given a display of "flex" which aligns the content to the center, and makes it responsive to different screen sizes. The background color is set to a light gray.

The Snake and food divs are not currently being implemented in the code or CSS, they are just place holders for dynamically generated content.


```py
</head>
<body>
    <div class="game-grid">
        <!-- Snake will be dynamically generated here using JavaScript -->
    </div>
    <div class="food">
        <!-- Food will be dynamically generated here using JavaScript -->
    </div>
</body>
</html>

## CSS Styles (styles.css)
body {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
    background-color: #f0f0f0;
}

```

这段代码定义了一个游戏网格的布局。网格有四列，每列包含二十行，以使它有水平和垂直展开。网格之间的空白称为“gap”，目前设置为1像素。网格的背景颜色为灰色，边框颜色为蓝色。每个单元格元素的背景颜色为白色，以便在网格中显示。

网格中的每个单元格元素都被设置为一个完整的width=100%高度=100%包含内容的虚对象。这意味着它们会占据其父元素的全部宽度，并将其高度和内容大小填充满它们在网格中的位置。这使得我们可以在网格中创建一个自包含的交互式元素，就像一个完整的游戏网格一样。

通过调整grid-template-columns和grid-template-rows的值，可以自定义网格的大小和布局。目前，该网格的宽度为400像素，高度为400像素，间隙为1像素。


```py
.game-grid {
    width: 400px;
    height: 400px;
    display: grid;
    grid-template-columns: repeat(20, 1fr); /* Adjust to the desired grid size */
    grid-template-rows: repeat(20, 1fr);
    gap: 1px;
    background-color: #222;
    border: 1px solid #555;
}

.game-grid div {
    width: 100%;
    height: 100%;
    background-color: #444;
}

```

这段代码是一个CSS（层叠样式表）文件中的选择器，用于为网页元素添加样式。具体来说，这段代码的作用是：

1. 为名为"snake-segment"的元素添加背景颜色为"#00cc66"的样式。
2. 为名为"food"的元素添加宽度为100%、高度为100%、背景颜色为"#cc3300"的定位内置样式。同时，该元素在网页上被放置于绝对定位，相对于其父元素的左50%、右50%的位置。
3. 为名为"game-over"的元素添加位置为(-50%, -50%)、宽度为24像素、字体大小为24像素、颜色为"#ff0000"的样式，且设置为初始隐藏状态。该元素将被显示在屏幕上，当游戏结束时可见。


```py
.snake-segment {
    background-color: #00cc66; /* Snake color */
}

.food {
    width: 100%;
    height: 100%;
    background-color: #cc3300; /* Food color */
    position: absolute;
}

/* Optional styles for a simple game over message */
.game-over {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 24px;
    font-weight: bold;
    color: #ff0000;
    display: none;
}

```

这段代码定义了一个名为 `load_engine` 的装饰器函数，它接受一个函数作为参数，这个函数的参数是一个元组（两个参数）和一个关键字参数，装饰器会根据这个元组和关键字参数来加载对应的功能模块。

具体来说，这个装饰器首先通过元组 `OUTPUT_MAPPING` 来查找对应的功能描述，然后通过关键字参数 `file_name` 和 `engine_name` 来加载对应的函数模块。接着，通过 `import_module` 函数加载模块，并使用 `ip_module_cls` 来获取对应的类。最后，使用类的方法 `engine_file.ip_module_cls()` 加载对应的函数实例，并返回。

整个装饰器的作用就是将一个函数包装成对应模块的功能，如果加载失败或者函数本身有误，就返回一个 `None`。


```py
## Anything UNCLEAR
There are no unclear points.

"""

OUTPUT_MAPPING = {
    "UI Design Description": (str, ...),
    "Selected Elements": (str, ...),
    "HTML Layout": (str, ...),
    "CSS Styles (styles.css)": (str, ...),
    "Anything UNCLEAR": (str, ...),
}


def load_engine(func):
    """Decorator to load an engine by file name and engine name."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        file_name, engine_name = func(*args, **kwargs)
        engine_file = import_module(file_name, package="metagpt")
        ip_module_cls = getattr(engine_file, engine_name)
        try:
            engine = ip_module_cls()
        except:
            engine = None

        return engine

    return wrapper


```

这段代码定义了一个装饰器函数 `parse(func)`，用于解析使用正则表达式模式的数据。

具体来说，这个装饰器函数接受一个函数参数 `func`，在函数内部执行以下操作：

1. 通过调用 `func` 函数，并将传递给 `parse` 的第一个参数，来获取要解析的信息。

2. 将 `func` 函数返回的结果与传递给 `parse` 的第二个参数 `pattern` 合并，并使用正则表达式模式 `re.DOTALL` 进行匹配。

3. 如果 `pattern` 匹配 `context` 中的正则表达式，则从 `context` 中的匹配部分提取文本信息，并输出到 `logger.info` 函数。

4. 如果 `pattern` 不匹配 `context` 中的正则表达式，则输出一条警告信息。

5. 返回解析后的文本信息。



这个装饰器函数可以被用于需要解析字符串信息的函数中，通过将解析函数与原始函数 `func` 绑定来使用。例如，以下代码展示了如何使用 `parse` 装饰器来解析字符串中的电子邮件地址：

```py 
def extract_email(text):
   def parse(func):
       @wraps(func)
       def wrapper(*args, **kwargs):
           pattern = kwargs.get('pattern')
           context = args[0]
           if pattern:
               return func(*args, **kwargs)[0]
           else:
               return func(*args, **kwargs)[0]

       return wrapper

   return parse

email_pattern = re.compile(r'@\w+')

def extract_email_address(text):
   return extract_email(text).replace('@', ' ').replace('/', ' ')



email_address = extract_email_address('example@example.com')
print(email_address)  # 输出： 'example@example.com'
```

在这个例子中，`extract_email_address` 函数首先定义了一个新的函数 `extract_email`，并且在函数内部执行了装饰器 `parse` 的操作。

然后，`extract_email_address` 函数使用 `extract_email` 装饰器来提取字符串中的电子邮件地址。

最后，通过调用 `extract_email_address` 函数，并将传递给 `parse` 装饰器的第一个参数，来获取解析函数，并将解析后的电子邮件地址输出。


```py
def parse(func):
    """Decorator to parse information using regex pattern."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        context, pattern = func(*args, **kwargs)
        match = re.search(pattern, context, re.DOTALL)
        if match:
            text_info = match.group(1)
            logger.info(text_info)
        else:
            text_info = context
            logger.info("未找到匹配的内容")

        return text_info

    return wrapper


```

This is a Python class that defines an action called "ui\_design" in the stable diffusion game engine. The action expects a list of messages called `requirements`, each containing information about the desired UI design, such as the UI description, CSS code, and HTML code.

The `run` method takes the `requirements` list and other keyword arguments, and returns an `ActionOutput` object containing the UI description generated from the `ui_design_action` method.

The `ui_design_action` method generates the UI description by querying the game engine for the specified UI description format and then parses out the content. It then saves the generated CSS and HTML files.

Finally, the `run` method sends the `ui_design` action to the game engine, passing in the `requirements` list as the first message. It then returns an `ActionOutput` object containing the UI description generated by the action.


```py
class UIDesign(Action):
    """Class representing the UI Design action."""

    def __init__(self, name, context=None, llm=None):
        super().__init__(name, context, llm)  # 需要调用LLM进一步丰富UI设计的prompt

    @parse
    def parse_requirement(self, context: str):
        """Parse UI Design draft from the context using regex."""
        pattern = r"## UI Design draft.*?\n(.*?)## Anything UNCLEAR"
        return context, pattern

    @parse
    def parse_ui_elements(self, context: str):
        """Parse Selected Elements from the context using regex."""
        pattern = r"## Selected Elements.*?\n(.*?)## HTML Layout"
        return context, pattern

    @parse
    def parse_css_code(self, context: str):
        pattern = r"```css.*?\n(.*?)## Anything UNCLEAR"
        return context, pattern

    @parse
    def parse_html_code(self, context: str):
        pattern = r"```pyhtml.*?\n(.*?)```"
        return context, pattern

    async def draw_icons(self, context, *args, **kwargs):
        """Draw icons using SDEngine."""
        engine = SDEngine()
        icon_prompts = self.parse_ui_elements(context)
        icons = icon_prompts.split("\n")
        icons = [s for s in icons if len(s.strip()) > 0]
        prompts_batch = []
        for icon_prompt in icons:
            # fixme: 添加icon lora
            prompt = engine.construct_payload(icon_prompt + ".<lora:WZ0710_AW81e-3_30e3b128d64T32_goon0.5>")
            prompts_batch.append(prompt)
        await engine.run_t2i(prompts_batch)
        logger.info("Finish icon design using StableDiffusion API")

    async def _save(self, css_content, html_content):
        save_dir = WORKSPACE_ROOT / "resources" / "codes"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        # Save CSS and HTML content to files
        css_file_path = save_dir / "ui_design.css"
        html_file_path = save_dir / "ui_design.html"

        with open(css_file_path, "w") as css_file:
            css_file.write(css_content)
        with open(html_file_path, "w") as html_file:
            html_file.write(html_content)

    async def run(self, requirements: list[Message], *args, **kwargs) -> ActionOutput:
        """Run the UI Design action."""
        # fixme: update prompt (根据需求细化prompt）
        context = requirements[-1].content
        ui_design_draft = self.parse_requirement(context=context)
        # todo: parse requirements str
        prompt = PROMPT_TEMPLATE.format(context=ui_design_draft, format_example=FORMAT_EXAMPLE)
        logger.info(prompt)
        ui_describe = await self._aask_v1(prompt, "ui_design", OUTPUT_MAPPING)
        logger.info(ui_describe.content)
        logger.info(ui_describe.instruct_content)
        css = self.parse_css_code(context=ui_describe.content)
        html = self.parse_html_code(context=ui_describe.content)
        await self._save(css_content=css, html_content=html)
        await self.draw_icons(ui_describe.content)
        return ui_describe


```py

这段代码定义了一个名为 "UI" 的类，代表 UI 角色。这个类继承自 "Role" 类，继承了 "name"、"profile"、"goal" 和 "constraints" 属性，同时包含了一个 "skills" 属性。

在 "UI" 类的初始化方法 "__init__" 中，首先调用父类的 "**init**" 方法，传递 "name"、"profile" 和 "goal" 参数，同时传递约束条件 "constraints"。然后，调用 "super__init__" 方法，传递本身实例的 "name"、"profile" 和 "goal" 参数，以及约束条件。然后，加载技能 "SD"，并将 "SD_ENGINE" 属性设置为从 "skills" 列表中获得的技能 engine。最后，将 "UI" 类实例加入到 "UI_ROLES" 列表中。

在 "load_skills" 方法中，加载 "SD" 技能。


```
class UI(Role):
    """Class representing the UI Role."""

    def __init__(
        self,
        name="Catherine",
        profile="UI Design",
        goal="Finish a workable and good User Interface design based on a product design",
        constraints="Give clear layout description and use standard icons to finish the design",
        skills=["SD"],
    ):
        super().__init__(name, profile, goal, constraints)
        self.load_skills(skills)
        self._init_actions([UIDesign])
        self._watch([WritePRD])

    @load_engine
    def load_sd_engine(self):
        """Load the SDEngine."""
        file_name = ".tools.sd_engine"
        engine_name = "SDEngine"
        return file_name, engine_name

    def load_skills(self, skills):
        """Load skills for the UI Role."""
        # todo: 添加其他出图engine
        for skill in skills:
            if skill == "SD":
                self.sd_engine = self.load_sd_engine()
                logger.info(f"load skill engine {self.sd_engine}")

```py

# `tests/metagpt/roles/__init__.py`

这段代码是一个Python脚本，它使用了#号来标识一个密的行开始。它包含一个@Time和@Author元数据，表示在脚本中时间戳和作者信息。此外，它还包含一个@File数据，表示这个脚本是在名为"__init__.py"的文件中定义的。

总的来说，这段代码定义了一个函数，但目前没有对其进行定义使用。也就是说，当脚本被运行时，它不会产生任何实际的输出结果。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 10:14
@Author  : alexanderwu
@File    : __init__.py
"""

```py

# `tests/metagpt/tools/test_code_interpreter.py`

这段代码的作用是测试一个机器学习算法的实现，其中包括：

1. 引入pytest库，用于测试；
2. 引入pandas库，用于数据读取和处理；
3. 引入pathlib库，用于文件路径操作；
4. 从pathlib库中定义了一个名为"sales_desc"的函数，它可能是用于读取sales.csv文件中的描述性信息；
5. 从pathlib库中定义了一个名为"store_desc"的函数，它可能是用于读取store.csv文件中的描述性信息；
6. 从metagpt库中定义了一个名为Action的函数，它可能是用于执行测试用例中的操作；
7. 从metagpt库中定义了一个名为logger的函数，它可能是用于记录日志信息。

这段代码中的函数和类都使用了metagpt库中的函数和类，因此可以推断出这是一段用于测试metagpt库中算法的实现。同时，由于该代码中没有定义任何函数参数，因此也可以推断出该代码是一个单元测试。


```
import pytest
import pandas as pd
from pathlib import Path

from tests.data import sales_desc, store_desc
from metagpt.tools.code_interpreter import OpenCodeInterpreter, OpenInterpreterDecorator
from metagpt.actions import Action
from metagpt.logs import logger


logger.add('./tests/data/test_ci.log')
stock = "./tests/data/baba_stock.csv"


# TODO: 需要一种表格数据格式，能够支持schame管理的，标注字段类型和字段含义。
```py

这段代码定义了一个名为 `CreateStockIndicators` 的类，该类实现了异步方法 `run`。该方法接收两个参数：`stock_path` 和 `indicators`，分别表示股票数据文件和指标列表。

方法实现中，首先从 `indicators` 列表中选择一个或多个指标，然后使用 `pandas` 和 `ta` 库计算这些指标。接下来，将计算得到的指标数据与 `stock_path` 中的股票数据进行合并，得到一个新的 `DataFrame` 对象。最后，将合并后的 `DataFrame` 对象中包含指标的数据保存为文件，并使用 `OpenCodeInterpreter` 对计算得到的指标进行可视化，将 `Date` 列转换为日期类型，并使用合适的颜色填充颜色。

整段代码的主要作用是对给定的股票数据文件和指标列表进行计算，然后生成一个新的 `DataFrame` 对象，并将计算得到的指标数据保存为文件。同时，使用 `OpenCodeInterpreter` 对计算得到的指标进行可视化，以便更好地观察指标的变化。


```
class CreateStockIndicators(Action):
    @OpenInterpreterDecorator(save_code=True, code_file_path="./tests/data/stock_indicators.py")
    async def run(self, stock_path: str, indicators=['Simple Moving Average', 'BollingerBands']) -> pd.DataFrame:
        """对stock_path中的股票数据, 使用pandas和ta计算indicators中的技术指标, 返回带有技术指标的股票数据，不需要去除空值, 不需要安装任何包；
            指标生成对应的三列: SMA, BB_upper, BB_lower
        """
        ...


@pytest.mark.asyncio
async def test_actions():
    # 计算指标
    indicators = ['Simple Moving Average', 'BollingerBands']
    stocker = CreateStockIndicators()
    df, msg = await stocker.run(stock, indicators=indicators)
    assert isinstance(df, pd.DataFrame)
    assert 'Close' in df.columns
    assert 'Date' in df.columns
    # 将df保存为文件，将文件路径传入到下一个action
    df_path = './tests/data/stock_indicators.csv'
    df.to_csv(df_path)
    assert Path(df_path).is_file()
    # 可视化指标结果
    figure_path = './tests/data/figure_ci.png'
    ci_ploter = OpenCodeInterpreter()
    ci_ploter.chat(f"使用seaborn对{df_path}中与股票布林带有关的数据列的Date, Close, SMA, BB_upper（布林带上界）, BB_lower（布林带下界）进行可视化, 可视化图片保存在{figure_path}中。不需要任何指标计算，把Date列转换为日期类型。要求图片优美，BB_upper, BB_lower之间使用合适的颜色填充。")
    assert Path(figure_path).is_file()

```py

# `tests/metagpt/tools/test_moderation.py`

这段代码是一个Python脚本，用于测试moderation库中的 Moderation 类。具体来说，它使用pytest库进行测试，使用了名为 `test_moderation.py` 的文件名。

在脚本中，首先导入了 `metagpt.tools.moderation` 包，这是moderation库的包装类。接着定义了一个名为 `@pytest.mark.parametrize` 的装饰器，用于对测试用例中的参数进行枚举。

在装饰器的帮助下，定义了两个参数，分别是 `content` 和 `content_sub`。这里的 `content` 指的是要测试的正文内容，而 `content_sub` 则是一个子类的参数，用于测试一个在正文中插入的评论。

最后，在 `test_moderation.py` 文件中，定义了一个名为 `test_moderation` 的函数，接受两个参数，一个是 `content`，另一个是 `content_sub`。函数内部使用 Moderation 类对传入的正文内容进行处理，然后输出处理后的结果。

这个脚本的作用是测试 moderation库中的 Moderation 类，以确认它是否能够对不同类型的测试用例进行有效的处理。通过对 `content` 和 `content_sub` 这两个参数的传入，可以测试不同类型的测试用例，例如在正文中插入一个评论，以及从正文中提取一个子内容等。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/26 14:46
@Author  : zhanglei
@File    : test_moderation.py
"""

import pytest

from metagpt.tools.moderation import Moderation


@pytest.mark.parametrize(
    ("content",),
    [
        [
            ["I will kill you", "The weather is really nice today", "I want to hit you"],
        ]
    ],
)
```py

该代码定义了一个名为 `test_moderation` 的函数，该函数接受一个名为 `content` 的参数。

函数内部定义了一个名为 `moderation` 的类 `Moderation`，以及一个名为 `moderation.moderation` 的方法。

`moderation.moderation` 方法接受一个名为 `content` 的参数，并返回一个被称为 `results` 的列表。

该函数还使用 `assert` 语句来验证 `results` 是否为列表类型，以及验证返回值的数量是否等于 `content` 的长度。


```
def test_moderation(content):
    moderation = Moderation()
    results = moderation.moderation(content=content)
    assert isinstance(results, list)
    assert len(results) == len(content)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content",),
    [
        [
            ["I will kill you", "The weather is really nice today", "I want to hit you"],
        ]
    ],
)
```py

这段代码定义了一个名为 `test_amoderation` 的函数，它接受一个参数 `content`，代表要测试的内容。函数内部使用 `asyncio` 库的 ` Moderation` 类来实现对内容进行 `amoderation` 操作，并将结果保存在一个名为 `results` 的列表中。

具体来说，函数内部创建了一个 ` Moderation` 对象 `moderation`，并使用 `amoderation` 方法对传入的内容进行 `amoderation` 操作，将结果存储在 `results` 中。然后使用 `isinstance` 函数判断 `results` 是否为列表类型，如果是，再使用 `len` 函数获取 `results` 中的元素个数，与传入的内容的长度进行比较，最后使用 `assert` 语句确保两个结果相符合。


```
async def test_amoderation(content):
    moderation = Moderation()
    results = await moderation.amoderation(content=content)
    assert isinstance(results, list)
    assert len(results) == len(content)

```py

# `tests/metagpt/tools/test_prompt_generator.py`

该代码是一个Python脚本，用于测试PromptGenerator模块的功能。具体来说，它做了以下几件事情：

1. 导入pytest库
2. 导入了metagpt库中的logger、Beagercube模板、Enron模板、GPTPromptGenerator和WikiHow模板
3. 在PromptGenerator模块中定义了logger，以便在测试中记录信息
4. 在测试函数中，实例化了PromptGenerator对象，并使用它们分别生成了Beagercube、Enron和GPT模板的Prompt
5. 没有做其他事情，目前不知道还有什么作用

PromptGenerator模块是一个用于生成Prompt的库，它可以根据不同的模板和语法规则，生成各种类型的Prompt，以满足测试和评估的需求。在这段注释中，作者说明了该代码是在2023年5月27日17:46创建的，以及它的作者是Alexander Wu。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/2 17:46
@Author  : alexanderwu
@File    : test_prompt_generator.py
"""

import pytest

from metagpt.logs import logger
from metagpt.tools.prompt_writer import (
    BEAGECTemplate,
    EnronTemplate,
    GPTPromptGenerator,
    WikiHowTemplate,
)


```py

这部分代码使用了Python的pytest库进行测试，并且使用了LLM AI API。

有两个测试方法，一个是`test_gpt_prompt_generator`，另一个是`test_wikihow_template`。

1. `test_gpt_prompt_generator`测试了GPT Prompt Generator的功能。这个测试方法接收一个LLM AI API实例，并创建了一个GPT Prompt Generator实例。然后，它使用这个生成器生成一个示例问题，并使用LLM AI API进行模拟测试。最后，它检查生成的结果是否为空，如果没有，就表示成功。

2. `test_wikihow_template`测试了WikiHow模板的功能。这个测试方法接收一个LLM AI API实例，并创建了一个WikiHow模板实例。然后，它使用这个模板生成一个示例问题，并使用LLM AI API进行模拟测试。最后，它检查生成的结果是否为空，如果没有，就表示成功。


```
@pytest.mark.usefixtures("llm_api")
def test_gpt_prompt_generator(llm_api):
    generator = GPTPromptGenerator()
    example = "商品名称:WonderLab 新肌果味代餐奶昔 小胖瓶 胶原蛋白升级版 饱腹代餐粉6瓶 75g/瓶(6瓶/盒) 店铺名称:金力宁食品专营店 " \
              "品牌:WonderLab 保质期:1年 产地:中国 净含量:450g"

    results = llm_api.ask_batch(generator.gen(example))
    logger.info(results)
    assert len(results) > 0


@pytest.mark.usefixtures("llm_api")
def test_wikihow_template(llm_api):
    template = WikiHowTemplate()
    question = "learn Python"
    step = 5

    results = template.gen(question, step)
    assert len(results) > 0
    assert any("Give me 5 steps to learn Python." in r for r in results)


```py

这段代码使用了pytest的mark.usefixtures功能，用于在测试函数中使用fixture。fixture是用于控制测试中依赖对象的工具，它们允许您在测试中修改对象的行为而不需要在每个测试都创建自己的实例。

在这个例子中，@pytest.mark.usefixtures("llm_api")确保在测试函数enron_template和beagec_template中使用了llm_api fixture。这个fixture在测试函数运行时创建并返回一个EnronTemplate或BEAGECTemplate对象，这样您可以在测试中使用它们。

在测试函数中，使用模板类（EnronTemplate和BEAGECTemplate）创建的实例来生成测试数据。然后，使用generated()方法生成测试结果。

最后，使用pytest的assert功能来验证测试结果。在这个例子中，使用了len()函数来验证结果的数量是否大于0，以及assert any()函数来验证任何生成结果中是否包含"Write an email with the subject \"Meeting Agenda\""。


```
@pytest.mark.usefixtures("llm_api")
def test_enron_template(llm_api):
    template = EnronTemplate()
    subj = "Meeting Agenda"

    results = template.gen(subj)
    assert len(results) > 0
    assert any("Write an email with the subject \"Meeting Agenda\"." in r for r in results)


def test_beagec_template():
    template = BEAGECTemplate()

    results = template.gen()
    assert len(results) > 0
    assert any("Edit and revise this document to improve its grammar, vocabulary, spelling, and style."
               in r for r in results)

```