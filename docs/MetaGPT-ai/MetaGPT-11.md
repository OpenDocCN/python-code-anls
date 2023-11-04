# MetaGPT源码解析 11

# `metagpt/utils/json_to_markdown.py`

该代码是一个Python脚本，名为`json_to_markdown.py`，其目的是将JSON数据转换为Markdown格式。

具体来说，该脚本接受一个JSON对象（dictionary）作为输入参数，并返回一个以Markdown格式编写的字符串。JSON对象可以是嵌套的列表和字典。

在该脚本中，首先检查输入的JSON对象是否为字典类型。如果是，该脚本遍历Key-Value对，并将它们转换为Markdown格式。如果是列表，该脚本在Markdown中使用“#”作为头号，然后是逗号和换行符，接着是数字和空格，以及Markdown中的列表符号“-”。最后，如果输入的JSON对象是字典类型，该脚本将其递归地转换为Markdown格式，并在需要时将JSON对象的嵌套层数与当前深度相加。

如果输入的JSON对象不是字典类型，该脚本将直接将JSON对象的字符串转换为Markdown格式。

该脚本在通过Markdown的方式来将JSON数据转换为易于阅读的格式。通过该脚本，可以将较为复杂的JSON数据转换为易于理解的Markdown格式。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/11 11:50
@Author  : femto Zheng
@File    : json_to_markdown.py
"""


# since we original write docs/*.md in markdown format, so I convert json back to markdown
def json_to_markdown(data, depth=2):
    """
    Convert a JSON object to Markdown with headings for keys and lists for arrays, supporting nested objects.

    Args:
        data: JSON object (dictionary) or value.
        depth (int): Current depth level for Markdown headings.

    Returns:
        str: Markdown representation of the JSON data.
    """
    markdown = ""

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                # Handle JSON arrays
                markdown += "#" * depth + f" {key}\n\n"
                items = [str(item) for item in value]
                markdown += "- " + "\n- ".join(items) + "\n\n"
            elif isinstance(value, dict):
                # Handle nested JSON objects
                markdown += "#" * depth + f" {key}\n\n"
                markdown += json_to_markdown(value, depth + 1)
            else:
                # Handle other values
                markdown += "#" * depth + f" {key}\n\n{value}\n\n"
    else:
        # Handle non-dictionary JSON data
        markdown = str(data)

    return markdown

```

# `metagpt/utils/make_sk_kernel.py`

这段代码是一个Python脚本，使用了`#!/usr/bin/env python`作为脚本路径的命令行参数。以下是对脚本的功能和部分输出的解释：

1. 导入`semantic_kernel`和`open_ai`库，以及`open_ai_chat_completion`类。
2. 通过`import semantic_kernel as sk`导入`semantic_kernel`库，但未定义`sk`库；通过`from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import ( AzureChatCompletion, )`导入`open_ai`库中关于`azure_chat_completion`服务的类；通过`from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import ( OpenAIChatCompletion, )`导入`open_ai`库中关于`open_ai_chat_completion`服务的类。
3. 创建两个实例：`AzureChatCompletion`实例和`OpenAIChatCompletion`实例。
4. 通过`AzureChatCompletion.start()`启动`AzureChatCompletion`实例的对话。
5. 通过`OpenAIChatCompletion.post_message(text='你好，人工智能助手')`向`open_ai_chat_completion`服务发送一条消息，并要求服务在消息被系统接收后进行响应。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/13 12:29
@Author  : femto Zheng
@File    : make_sk_kernel.py
"""
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import (
    AzureChatCompletion,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import (
    OpenAIChatCompletion,
)

```

这段代码定义了一个名为 `make_sk_kernel()` 的函数，它使用 MetagPT 的 `sk` 库创建了一个自定义的语言模型。

函数中首先引入了 `CONFIG` 变量，然后定义了一个名为 `kernel` 的变量。接着，函数判断了 `CONFIG` 中指定的 OpenAI API 类型是否为 "azure"，如果是，则添加了一个名为 "chat_completion" 的聊天服务，其相关参数如下：

- `"chat_completion"`：这是一个聊天的服务名称。
- `AzureChatCompletion`：这是一个自定义的聊天服务类，它继承了 `sk.Service` 类，负责创建和处理聊天请求。
- `CONFIG.deployment_name`：这是部署这个聊天服务时的部署名称。
- `CONFIG.openai_api_base`：这是 OpenAI API 的基本 URL。
- `CONFIG.openai_api_key`：这是用于身份验证的 API 密钥。

如果 `CONFIG.openai_api_type` 不是 "azure"，则添加另一个名为 "chat_completion" 的聊天服务，其参数与上面类似。这个服务使用的是 OpenAI 训练的模型，并且需要输入组织的 ID。

最后，函数返回了一个包含 `kernel` 变量的引用，这个引用可以在函数外部使用，例如：

```py
kernel = make_sk_kernel()
```


```py
from metagpt.config import CONFIG


def make_sk_kernel():
    kernel = sk.Kernel()
    if CONFIG.openai_api_type == "azure":
        kernel.add_chat_service(
            "chat_completion",
            AzureChatCompletion(CONFIG.deployment_name, CONFIG.openai_api_base, CONFIG.openai_api_key),
        )
    else:
        kernel.add_chat_service(
            "chat_completion",
            OpenAIChatCompletion(
                CONFIG.openai_api_model, CONFIG.openai_api_key, org_id=None, endpoint=CONFIG.openai_api_base
            ),
        )

    return kernel

```

# `metagpt/utils/mermaid.py`

这段代码是一个Python脚本，使用了 `#!/usr/bin/env python` 作为Shell脚本的路径模板，表示该脚本使用Python 3作为运行时环境。接下来是脚本的一些信息：

1. 导入 `asyncio`、`os` 和 `pathlib` 库，这些库用于异步编程、操作系统操作和文件操作；
2. 通过 `import asyncio` 导入 `asyncio` 库，该库提供了异步编程的基础；
3. 通过 `import os` 导入 `os` 库，该库提供了文件和目录操作的功能；
4. 通过 `import pathlib` 导入 `pathlib` 库，该库提供了对路径和目录操作的支持；
5. 通过 `CONFIG.init()` 初始化 `metagpt` 配置文件，`CONFIG` 是 `metagpt` 项目的配置类，`init()` 方法用于初始化配置文件；
6. 通过 `logger.init(穿越要下载的 API 需先安装 "aiohttp" 和 "requests" 库，请先运行 "pip install aiohttp 和 requests"，否则会抛出离线验证失败)` 初始化 `metagpt` 的日志输出，`logger` 是 `metagpt` 的日志输出类，`init()` 方法用于初始化日志输出；
7. 通过 `check_cmd_exists` 函数检查 `cmd` 是否存在于系统环境变量中，若不存在则执行安装命令，`cmd` 是 `安装命令`，需要根据实际情况修改；
8. 导入 `metagpt` 项目的一些类和函数，用于执行 `metagpt` 项目的相关任务。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/7/4 10:53
@Author  : alexanderwu alitrack
@File    : mermaid.py
"""
import asyncio
import os
from pathlib import Path

from metagpt.config import CONFIG
from metagpt.const import PROJECT_ROOT
from metagpt.logs import logger
from metagpt.utils.common import check_cmd_exists


```

To convert Mermaid code to a PNG, you can use the following steps:

1. Install the required Python packages by running the following command in your terminal:
```pyarduino
pip install pymmo-imageio px4-api
```
2. Create a Python script by running the following command:
```pycss
cat > ${PROJECT_DIR}/generate_image.py <<-'EOF'
from PIL import Image
import numpy as np
import subprocess
import re

CONFIG = {
   'puppeteer_config': False,
   'mmdc_config': False,
   'playwright_config': False,
   'pyppeteer_config': False,
   'ink_config': False,
}

def convert_mermaid_to_image(mermaid_code, output_file, width, height):
   # Convert the Mermaid code to a PNG image
   png_cmd = f" converts支配层的{mermaid_code}为图片并保存在{output_file}"
   png_output = subprocess.Popen(png_cmd, shell=True, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

   while True:
       line = await png_output.readline()
       if line.startswith('Error'):
           png_output.break_model()
           break

   return png_output.read().decode()

def main():
   # Set the input Mermaid code
   input_file = "input_mermaid.txt"
   output_file = "output_image.png"

   # Convert the Mermaid code to an image and save it as a PNG file
   png_image = convert_mermaid_to_image(input_file, output_file, 800, 800)

   # Display the image
   display = (800, 800)
   image = np.array(png_image, dtype=np.uint8, endpoint="L")
   image = (1, 1, image.shape[2], image.shape[3])
   顯示.show(image)

if __name__ == "__main__":
   main()
```
3. Run the script by running the following command:
```pyarduino
python generate_image.py
```
4. The script will read the Mermaid code from the specified input file, convert it to an image, and save it as a PNG file in the same directory as the script.

Note: This script assumes that you have the required Python packages installed. If you don't have the required packages, you can install them by running the following command:
```py
pip install pymmo-imageio px4-api
```


```py
async def mermaid_to_file(mermaid_code, output_file_without_suffix, width=2048, height=2048) -> int:
    """suffix: png/svg/pdf

    :param mermaid_code: mermaid code
    :param output_file_without_suffix: output filename
    :param width:
    :param height:
    :return: 0 if succeed, -1 if failed
    """
    # Write the Mermaid code to a temporary file
    dir_name = os.path.dirname(output_file_without_suffix)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    tmp = Path(f"{output_file_without_suffix}.mmd")
    tmp.write_text(mermaid_code, encoding="utf-8")

    engine = CONFIG.mermaid_engine.lower()
    if engine == "nodejs":
        if check_cmd_exists(CONFIG.mmdc) != 0:
            logger.warning("RUN `npm install -g @mermaid-js/mermaid-cli` to install mmdc")
            return -1

        for suffix in ["pdf", "svg", "png"]:
            output_file = f"{output_file_without_suffix}.{suffix}"
            # Call the `mmdc` command to convert the Mermaid code to a PNG
            logger.info(f"Generating {output_file}..")

            if CONFIG.puppeteer_config:
                commands = [
                    CONFIG.mmdc,
                    "-p",
                    CONFIG.puppeteer_config,
                    "-i",
                    str(tmp),
                    "-o",
                    output_file,
                    "-w",
                    str(width),
                    "-H",
                    str(height),
                ]
            else:
                commands = [CONFIG.mmdc, "-i", str(tmp), "-o", output_file, "-w", str(width), "-H", str(height)]
            process = await asyncio.create_subprocess_shell(
                " ".join(commands), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()
            if stdout:
                logger.info(stdout.decode())
            if stderr:
                logger.error(stderr.decode())
    else:
        if engine == "playwright":
            from metagpt.utils.mmdc_playwright import mermaid_to_file

            return await mermaid_to_file(mermaid_code, output_file_without_suffix, width, height)
        elif engine == "pyppeteer":
            from metagpt.utils.mmdc_pyppeteer import mermaid_to_file

            return await mermaid_to_file(mermaid_code, output_file_without_suffix, width, height)
        elif engine == "ink":
            from metagpt.utils.mmdc_ink import mermaid_to_file

            return await mermaid_to_file(mermaid_code, output_file_without_suffix)
        else:
            logger.warning(f"Unsupported mermaid engine: {engine}")
    return 0


```

这段代码定义了一个类MMC1，它描述了一个用于搜索和排序的系统中各个组件之间的关系和交互。

MMC1中的类Main是一个类，它定义了系统的主要组件以及它们的交互方式。其中，search_engine和main()是Main类中的两个方法，分别用于搜索查询内容和运行主程序。

MMC1中的类SearchEngine是一个类，它定义了搜索引擎的各个组件以及它们的交互方式。其中，index、ranking和summary是SearchEngine类中的三个方法，分别用于索引、排序和生成摘要。而search()方法则用于接收查询字符串并返回搜索结果。

MMC1中的类Index是一个类，它定义了索引的各个组件以及它们的交互方式。其中，knowledge_base是Index类中的一个属性，用于存储知识库，而create_index()和query_index()方法分别用于创建索引和在查询时获取索引。

MMC1中的类Ranking是一个类，它定义了评分的各个组件以及它们的交互方式。其中，rank_results()方法用于生成评分结果。

MMC1中的类Summary是一个类，它定义了摘要的各个组件以及它们的交互方式。其中，summarize_results()方法用于生成摘要。

MMC1中的类KnowledgeBase是一个类，它定义了知识库的各个组件以及它们的交互方式。其中，update()和fetch_data()方法用于更新知识库和获取数据，而search()方法用于在知识库中搜索查询字符串并返回结果。


```py
MMC1 = """classDiagram
    class Main {
        -SearchEngine search_engine
        +main() str
    }
    class SearchEngine {
        -Index index
        -Ranking ranking
        -Summary summary
        +search(query: str) str
    }
    class Index {
        -KnowledgeBase knowledge_base
        +create_index(data: dict)
        +query_index(query: str) list
    }
    class Ranking {
        +rank_results(results: list) list
    }
    class Summary {
        +summarize_results(results: list) str
    }
    class KnowledgeBase {
        +update(data: dict)
        +fetch_data(query: str) dict
    }
    Main --> SearchEngine
    SearchEngine --> Index
    SearchEngine --> Ranking
    SearchEngine --> Summary
    Index --> KnowledgeBase"""

```

此代码是一个UML序列图，表示了SE(SearchEngine)、I(Index)、R(Ranking)、KB(KnowledgeBase)、M(Main)之间的交互和通信。

在该序列图中，SE(SearchEngine)接收一个查询(query)，然后向索引(Index)发送请求，获取该查询在知识库(KnowledgeBase)中是否可得分。如果成功，SE(SearchEngine)将返回查询结果(results)，然后将结果(results)转发给排名(Ranking)模块，以便对结果进行排名并返回排名结果(ranked_results)。

如果失败，SE(SearchEngine)将向I(Index)发送查询请求，并将查询请求传递给KB(KnowledgeBase)以获取查询数据。KB(KnowledgeBase)将返回查询数据(data)，然后将数据(data)传递给I(Index)，以便将数据传递给SE(SearchEngine)以继续查询。

如此循环交互，SE(SearchEngine)最终将返回一个包含查询结果(results)的集合，该集合将被传递给排名(Ranking)模块以进行排名并返回排名结果(ranked_results)。最终，排名结果(ranked_results)将被返回给SE(SearchEngine)，该结果将显示在页面上以提供用户一个更好的用户体验。


```py
MMC2 = """sequenceDiagram
    participant M as Main
    participant SE as SearchEngine
    participant I as Index
    participant R as Ranking
    participant S as Summary
    participant KB as KnowledgeBase
    M->>SE: search(query)
    SE->>I: query_index(query)
    I->>KB: fetch_data(query)
    KB-->>I: return data
    I-->>SE: return results
    SE->>R: rank_results(results)
    R-->>SE: return ranked_results
    SE->>S: summarize_results(ranked_results)
    S-->>SE: return summary
    SE-->>M: return summary"""


```

这段代码的作用是执行两个事件循环，并为它们中的每个循环运行一个`mermaid_to_file`函数。这个函数将Mercator格式的数据存储到一个指定的文件中。

首先，代码创建一个名为`loop`的事件循环对象。然后，代码使用`asyncio.new_event_loop()`方法从事件循环上下文中获取这个事件循环对象。

接下来，代码使用`loop.run_until_complete()`方法运行第一个事件循环，直到指定的函数完成。这个函数使用`mermaid_to_file()`函数将Mercator格式的数据存储到第一个文件中。

在完成第一个事件循环后，代码使用相同的`run_until_complete()`方法运行第二个事件循环，直到指定的函数完成。这个函数使用`mermaid_to_file()`函数将Mercator格式的数据存储到第二个文件中。

最后，代码使用`loop.close()`方法关闭事件循环对象。


```py
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(mermaid_to_file(MMC1, PROJECT_ROOT / f"{CONFIG.mermaid_engine}/1"))
    result = loop.run_until_complete(mermaid_to_file(MMC2, PROJECT_ROOT / f"{CONFIG.mermaid_engine}/1"))
    loop.close()

```

# `metagpt/utils/mmdc_ink.py`

该代码是一个 Python 脚本，实现了将 Mermaid 语法中的文本内容转换为图像文件的功能。该脚本的主要作用是为 Mermaid 语法提供一种将文本转换为图像的方法，以便用户可以将 Mermaid 文本内容保存为图片。

具体来说，该脚本实现了以下功能：

1. 读取输入的 Mermaid 代码并编码为字节序列。
2. 将字节序列转换为 Base64 编码的字符串。
3. 遍历给定的文件类型（png 和 svg）。
4. 对于每种文件类型，使用 aiohttp 库发送 HTTP GET 请求获取一个图像文件，并将其存储为本地文件。
5. 将生成的图像文件保存为给定的输出文件名（不带文件后缀）。

该脚本使用了一个辅助函数 `mermaid_to_file` 来执行实际的文件写入操作。该函数将输入的 Mermaid 代码转换为 Base64 编码的字符串，并使用 `ClientSession` 和 `ClientError` 类从网络上获取图像文件。函数的参数包括两个带有 `output_file_without_suffix` 参数，用于指定生成的图像文件的文件名，以及一个用于包含图像文件的目录。

该脚本需要导入 `base64`、`os` 和 `metagpt.logs` 库。需要使用 `ClientSession` 和 `ClientError` 类，因此需要导入 `aiohttp`。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/4 16:12
@Author  : alitrack
@File    : mermaid.py
"""
import base64
import os

from aiohttp import ClientSession,ClientError
from metagpt.logs import logger


async def mermaid_to_file(mermaid_code, output_file_without_suffix):
    """suffix: png/svg
    :param mermaid_code: mermaid code
    :param output_file_without_suffix: output filename without suffix
    :return: 0 if succeed, -1 if failed
    """
    encoded_string = base64.b64encode(mermaid_code.encode()).decode()

    for suffix in ["svg", "png"]:
        output_file = f"{output_file_without_suffix}.{suffix}"
        path_type = "svg" if suffix == "svg" else "img"
        url = f"https://mermaid.ink/{path_type}/{encoded_string}"
        async with ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        text = await response.content.read()
                        with open(output_file, 'wb') as f:
                            f.write(text)
                        logger.info(f"Generating {output_file}..")
                    else:
                        logger.error(f"Failed to generate {output_file}")
                        return -1
            except ClientError as e:
                logger.error(f"network error: {e}")
                return -1
    return 0

```

# `metagpt/utils/mmdc_playwright.py`

This is a JavaScript script that generates a PNG image of a SVG representation of a graph. The script is using the Chrome DevTools Protocol to interact with the SVG element and the PDF.js library to handle the PDF file generated.

It takes an input SVG file and an output file name without the .svg and .pdf suffixes. The script first checks the input file and then sets the viewport size of the browser to capture the screenshot of the SVG. Then, it generates the PNG image, sets the file name and writes the image to the output directory. If the file is a PDF, it uses the PDF.js library to generate the PDF image and write it to the output file.

It also handles the error and finally close the browser.


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/4 16:12
@Author  : Steven Lee
@File    : mmdc_playwright.py
"""

import os
from urllib.parse import urljoin
from playwright.async_api import async_playwright
from metagpt.logs import logger

async def mermaid_to_file(mermaid_code, output_file_without_suffix, width=2048, height=2048)-> int:
    """
    Converts the given Mermaid code to various output formats and saves them to files.

    Args:
        mermaid_code (str): The Mermaid code to convert.
        output_file_without_suffix (str): The output file name without the file extension.
        width (int, optional): The width of the output image in pixels. Defaults to 2048.
        height (int, optional): The height of the output image in pixels. Defaults to 2048.

    Returns:
        int: Returns 1 if the conversion and saving were successful, -1 otherwise.
    """
    suffixes=['png', 'svg', 'pdf']
    __dirname = os.path.dirname(os.path.abspath(__file__))

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        device_scale_factor = 1.0
        context = await browser.new_context(
                viewport={'width': width, 'height': height},
                device_scale_factor=device_scale_factor,
            )
        page = await context.new_page()

        async def console_message(msg):
            logger.info(msg.text)
        page.on('console', console_message)

        try:
            await page.set_viewport_size({'width': width, 'height': height})

            mermaid_html_path = os.path.abspath(
                os.path.join(__dirname, 'index.html'))
            mermaid_html_url = urljoin('file:', mermaid_html_path)
            await page.goto(mermaid_html_url)
            await page.wait_for_load_state("networkidle")

            await page.wait_for_selector("div#container", state="attached")
            mermaid_config = {}
            background_color = "#ffffff"
            my_css = ""
            await page.evaluate(f'document.body.style.background = "{background_color}";')

            metadata = await page.evaluate('''async ([definition, mermaidConfig, myCSS, backgroundColor]) => {
                const { mermaid, zenuml } = globalThis;
                await mermaid.registerExternalDiagrams([zenuml]);
                mermaid.initialize({ startOnLoad: false, ...mermaidConfig });
                const { svg } = await mermaid.render('my-svg', definition, document.getElementById('container'));
                document.getElementById('container').innerHTML = svg;
                const svgElement = document.querySelector('svg');
                svgElement.style.backgroundColor = backgroundColor;

                if (myCSS) {
                    const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
                    style.appendChild(document.createTextNode(myCSS));
                    svgElement.appendChild(style);
                }

            }''', [mermaid_code, mermaid_config, my_css, background_color])

            if 'svg' in suffixes :
                svg_xml = await page.evaluate('''() => {
                    const svg = document.querySelector('svg');
                    const xmlSerializer = new XMLSerializer();
                    return xmlSerializer.serializeToString(svg);
                }''')
                logger.info(f"Generating {output_file_without_suffix}.svg..")
                with open(f'{output_file_without_suffix}.svg', 'wb') as f:
                    f.write(svg_xml.encode('utf-8'))

            if  'png' in suffixes:
                clip = await page.evaluate('''() => {
                    const svg = document.querySelector('svg');
                    const rect = svg.getBoundingClientRect();
                    return {
                        x: Math.floor(rect.left),
                        y: Math.floor(rect.top),
                        width: Math.ceil(rect.width),
                        height: Math.ceil(rect.height)
                    };
                }''')
                await page.set_viewport_size({'width': clip['x'] + clip['width'], 'height': clip['y'] + clip['height']})
                screenshot = await page.screenshot(clip=clip, omit_background=True, scale='device')
                logger.info(f"Generating {output_file_without_suffix}.png..")
                with open(f'{output_file_without_suffix}.png', 'wb') as f:
                    f.write(screenshot)
            if 'pdf' in suffixes:
                pdf_data = await page.pdf(scale=device_scale_factor)
                logger.info(f"Generating {output_file_without_suffix}.pdf..")
                with open(f'{output_file_without_suffix}.pdf', 'wb') as f:
                    f.write(pdf_data)
            return 0
        except Exception as e:
            logger.error(e)
            return -1
        finally:
            await browser.close()

```

# `metagpt/utils/mmdc_pyppeteer.py`

This is a JavaScript function that generates a SVG image of a specified XML data and saves it as a PNG file. It takes in a SVG XML data string and a number of file suffixes that the SVG image will be saved with. It also takes into account the device scale factor and the default device scale factor for the browser. It returns 0 if the process is successful or -1 if an error occurs.


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/4 16:12
@Author  : alitrack
@File    : mmdc_pyppeteer.py
"""
import os
from urllib.parse import urljoin
from pyppeteer import launch
from metagpt.logs import logger
from metagpt.config import CONFIG

async def mermaid_to_file(mermaid_code, output_file_without_suffix, width=2048, height=2048)-> int:
    """
    Converts the given Mermaid code to various output formats and saves them to files.

    Args:
        mermaid_code (str): The Mermaid code to convert.
        output_file_without_suffix (str): The output file name without the file extension.
        width (int, optional): The width of the output image in pixels. Defaults to 2048.
        height (int, optional): The height of the output image in pixels. Defaults to 2048.

    Returns:
        int: Returns 1 if the conversion and saving were successful, -1 otherwise.
    """
    suffixes = ['png', 'svg', 'pdf']   
    __dirname = os.path.dirname(os.path.abspath(__file__))

    
    if CONFIG.pyppeteer_executable_path:
        browser = await launch(headless=True,
                            executablePath=CONFIG.pyppeteer_executable_path,
                            args=['--disable-extensions',"--no-sandbox"] 
                            )
    else:
        logger.error("Please set the environment variable:PYPPETEER_EXECUTABLE_PATH.")
        return -1
    page = await browser.newPage()
    device_scale_factor = 1.0

    async def console_message(msg):
        logger.info(msg.text)
    page.on('console', console_message)

    try:
        await page.setViewport(viewport={'width': width, 'height': height, 'deviceScaleFactor': device_scale_factor})

        mermaid_html_path = os.path.abspath(
            os.path.join(__dirname, 'index.html'))
        mermaid_html_url = urljoin('file:', mermaid_html_path)
        await page.goto(mermaid_html_url)

        await page.querySelector("div#container")
        mermaid_config = {}
        background_color = "#ffffff"
        my_css = ""
        await page.evaluate(f'document.body.style.background = "{background_color}";')

        metadata = await page.evaluate('''async ([definition, mermaidConfig, myCSS, backgroundColor]) => {
            const { mermaid, zenuml } = globalThis;
            await mermaid.registerExternalDiagrams([zenuml]);
            mermaid.initialize({ startOnLoad: false, ...mermaidConfig });
            const { svg } = await mermaid.render('my-svg', definition, document.getElementById('container'));
            document.getElementById('container').innerHTML = svg;
            const svgElement = document.querySelector('svg');
            svgElement.style.backgroundColor = backgroundColor;

            if (myCSS) {
                const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
                style.appendChild(document.createTextNode(myCSS));
                svgElement.appendChild(style);
            }
        }''', [mermaid_code, mermaid_config, my_css, background_color])

        if 'svg' in suffixes :
            svg_xml = await page.evaluate('''() => {
                const svg = document.querySelector('svg');
                const xmlSerializer = new XMLSerializer();
                return xmlSerializer.serializeToString(svg);
            }''')
            logger.info(f"Generating {output_file_without_suffix}.svg..")
            with open(f'{output_file_without_suffix}.svg', 'wb') as f:
                f.write(svg_xml.encode('utf-8'))

        if  'png' in suffixes:
            clip = await page.evaluate('''() => {
                const svg = document.querySelector('svg');
                const rect = svg.getBoundingClientRect();
                return {
                    x: Math.floor(rect.left),
                    y: Math.floor(rect.top),
                    width: Math.ceil(rect.width),
                    height: Math.ceil(rect.height)
                };
            }''')
            await page.setViewport({'width': clip['x'] + clip['width'], 'height': clip['y'] + clip['height'], 'deviceScaleFactor': device_scale_factor})
            screenshot = await page.screenshot(clip=clip, omit_background=True, scale='device')
            logger.info(f"Generating {output_file_without_suffix}.png..")
            with open(f'{output_file_without_suffix}.png', 'wb') as f:
                f.write(screenshot)
        if 'pdf' in suffixes:
            pdf_data = await page.pdf(scale=device_scale_factor)
            logger.info(f"Generating {output_file_without_suffix}.pdf..")
            with open(f'{output_file_without_suffix}.pdf', 'wb') as f:
                f.write(pdf_data)
        return 0
    except Exception as e:
        logger.error(e)
        return -1
    finally:
        await browser.close()


```

# `metagpt/utils/parse_html.py`

该代码是一个Python脚本，主要目的是从BSD授权协议中获得，并实现了HTTP GET请求获取网页内容的功能。具体实现包括：

1. 从输入页面内容中提取出HTML代码，使用BeautifulSoup库对HTML代码进行解析，并从解析后的HTML代码中提取出<title>标签的文本内容，存储在WebPage类的`inner_text`属性中。
2. 如果HTML代码解析失败，或者提取出的<title>标签为空字符串，那么`inner_text`属性将设置为空字符串。
3. 定义了一个`WebPage`类，该类实现了BaseModel类，包括`__post__`、`__repr__`、`__len__`、`__get__`方法，以及一个特殊的`__get__`方法，用于从BSD授权协议中获得链接。
4. 在`__get__`方法中，使用`urlparse`和`urljoin`方法将获取到的链接进行解析，并使用`self.url`属性将链接加入到了`self.url`属性中，以便将链接与HTML页面内容进行匹配。
5. 在`get_links`方法中，使用BeautifulSoup库获取页面中的所有链接，并使用`urljoin`方法将链接进行组合，以便与`self.url`属性中的链接进行匹配。


```py
#!/usr/bin/env python
from __future__ import annotations

from typing import Generator, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from pydantic import BaseModel


class WebPage(BaseModel):
    inner_text: str
    html: str
    url: str

    class Config:
        underscore_attrs_are_private = True

    _soup : Optional[BeautifulSoup] = None
    _title: Optional[str] = None

    @property
    def soup(self) -> BeautifulSoup:
        if self._soup is None:
            self._soup = BeautifulSoup(self.html, "html.parser")
        return self._soup
    
    @property
    def title(self):
        if self._title is None:
            title_tag = self.soup.find("title")
            self._title = title_tag.text.strip() if title_tag is not None else ""
        return self._title

    def get_links(self) -> Generator[str, None, None]:
        for i in self.soup.find_all("a", href=True):
            url = i["href"]
            result = urlparse(url)
            if not result.scheme and result.path:
                yield urljoin(self.url, url)
            elif url.startswith(("http://", "https://")):
                yield urljoin(self.url, url)


```



该函数的作用是获取指定页面的HTML内容，并返回一个BeautifulSoup对象。函数的实现主要分为两个步骤：

1. 通过调用BeautifulSoup的`_get_soup`函数，获取指定页面的内容并将其存储在`soup`变量中。
2. 通过调用`soup.get_text(strip=True)`方法，提取页面中的文本内容，并将其存储在`text`变量中。

具体来说，`_get_soup`函数的作用是将页面中的HTML代码转化为BeautifulSoup对象，而`soup.get_text`方法则从BeautifulSoup对象中选择所有的文本内容，并将其存储在`text`变量中。在这里，函数使用了Python标准库中的`BeautifulSoup`类，该类可以轻松地从HTML页面中提取出文本内容。


```py
def get_html_content(page: str, base: str):
    soup = _get_soup(page)

    return soup.get_text(strip=True)


def _get_soup(page: str):
    soup = BeautifulSoup(page, "html.parser")
    # https://stackoverflow.com/questions/1936466/how-to-scrape-only-visible-webpage-text-with-beautifulsoup
    for s in soup(["style", "script", "[document]", "head", "title"]):
        s.extract()

    return soup

```

# `metagpt/utils/pycst.py`

这段代码是一个用于提取Python类或函数的文档字符串的函数。其作用是帮助用户在不需要编写额外代码的情况下，自动生成docstring注释。

具体来说，代码首先定义了一个名为DocstringNode的联合类，代表Python中的类定义、函数定义或模块定义。

接着，代码定义了一个名为get_docstring_statement的函数，该函数接收一个DocstringNode作为参数，返回其中的docstring声明的最简单的语句行。

函数的核心部分是对于每个 DocstringNode，首先检查其所属的类型是否为Module，如果不是，则递归地检查其body节点。如果body为空，则直接返回None。如果body包含一个SimpleStatementLine，则将其返回。如果body包含一个Expr对象，并且其value为SimpleString或ConcatenatedString类型，则返回该Expr对象的evaluated_value。

这里需要注意，对于Expr对象，其evaluated_value是经过计算生成的，而不是简单的字符串。

最后，代码还定义了一个名为Module作为其唯一的子类。在示例代码中，Module被用作如下：

```py
from libcst import Module

def test_example import get_docstring

def test_get_docstring import module_with_docstring

def get_docstring(body: Union[cst.Module, cst.ClassDef, cst.FunctionDef]):
   """Extracts the docstring from the body of a node."""
   if isinstance(body, cst.Module):
       body = body.body
   else:
       body = body.body.body

   if not body:
       return None

   statement = body[0]
   if not isinstance(statement, cst.SimpleStatementLine):
       return None

   expr = statement
   while isinstance(expr, (cst.BaseSuite, cst.SimpleStatementLine)):
       if len(expr.body) == 0:
           return None
       expr = expr.body[0]

   if not isinstance(expr, cst.Expr):
       return None
   
   evaluated_value = expr.value
   if not isinstance(evaluated_value, (cst.SimpleString, cst.ConcatenatedString)):
       return None
   
   return statement

example_module = Module()
example_module.add_example()
example_example = example_module.example
example_example.defs = [
   ("my_example_function", {"my_example_function": "returns": cst.Expr(str(example_example))})
]
```

以上代码的目的是测试get_docstring函数的正确性。在上述示例代码中，它定义了一个测试用例my_example_function，该函数在example_example模块中定义，并返回了一个示例输出。


```py
from __future__ import annotations

from typing import Union

import libcst as cst
from libcst._nodes.module import Module

DocstringNode = Union[cst.Module, cst.ClassDef, cst.FunctionDef]


def get_docstring_statement(body: DocstringNode) -> cst.SimpleStatementLine:
    """Extracts the docstring from the body of a node.

    Args:
        body: The body of a node.

    Returns:
        The docstring statement if it exists, None otherwise.
    """
    if isinstance(body, cst.Module):
        body = body.body
    else:
        body = body.body.body

    if not body:
        return

    statement = body[0]
    if not isinstance(statement, cst.SimpleStatementLine):
        return

    expr = statement
    while isinstance(expr, (cst.BaseSuite, cst.SimpleStatementLine)):
        if len(expr.body) == 0:
            return None
        expr = expr.body[0]

    if not isinstance(expr, cst.Expr):
        return None
    
    val = expr.value
    if not isinstance(val, (cst.SimpleString, cst.ConcatenatedString)):
        return None
    
    evaluated_value = val.evaluated_value    
    if isinstance(evaluated_value, bytes):
        return None

    return statement


```

这段代码定义了一个名为 DocstringCollector 的类，用于从给定的 CST（脚本上下文）中收集文档字符串。该类包含以下方法：

- `__init__`：构造函数，用于初始化栈和文档字符串字典。
- `visit_Module`：访问者方法，用于访问模块节点。
- `leave_Module`：访问者方法，用于访问模块节点。
- `visit_ClassDef`：访问者方法，用于访问类定义节点。
- `leave_ClassDef`：访问者方法，用于访问类定义节点。
- `visit_FunctionDef`：访问者方法，用于访问函数定义节点。
- `leave_FunctionDef`：访问者方法，用于访问函数定义节点。
- `_leave`：私有方法，用于从栈中弹出文档字符串并将其添加到文档字符串字典中。
- `get_docstring_statement`：私有方法，用于获取文档字符串语句。

该类通过在给定的 CST 中遍历类、函数和模块等节点，并收集它们所产生的文档字符串，从而实现在给定 CST 中自动生成的文档字符串的收集。


```py
class DocstringCollector(cst.CSTVisitor):
    """A visitor class for collecting docstrings from a CST.

    Attributes:
        stack: A list to keep track of the current path in the CST.
        docstrings: A dictionary mapping paths in the CST to their corresponding docstrings.
    """
    def __init__(self):
        self.stack: list[str] = []
        self.docstrings: dict[tuple[str, ...], cst.SimpleStatementLine] = {}

    def visit_Module(self, node: cst.Module) -> bool | None:
        self.stack.append("")

    def leave_Module(self, node: cst.Module) -> None:
        return self._leave(node)

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self.stack.append(node.name.value)

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        return self._leave(node)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self.stack.append(node.name.value)

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        return self._leave(node)

    def _leave(self, node: DocstringNode) -> None:
        key = tuple(self.stack)
        self.stack.pop()
        if hasattr(node, "decorators") and any(i.decorator.value == "overload" for i in node.decorators):
            return

        statement = get_docstring_statement(node)
        if statement:
            self.docstrings[key] = statement


```



This appears to be a simple implementation of a custom leave method for a `cst.DocstringNode` class that customizes the behavior of the `leave` method of `DocstringNode` objects.

The `DocstringNode` class seems to be part of the `cst` module and has a `with_changes` method for sharing changes between its parent `ModuleNode` and its children `FunctionDef` and `MethodDef` objects.

The `visit_ClassDef`, `visit_FunctionDef`, and `leave_ClassDef` methods appear to specialize the behavior of the `leave` method by adding the name of the class or function to the stack of visited nodes.

The `_leave` method appears to be responsible for removing the name of the class or function from the stack and returning the `DocstringNode` object with the changes made to its body.

Overall, this implementation appears to be a simple way to customize the behavior of the `leave` method of `DocstringNode` objects.


```py
class DocstringTransformer(cst.CSTTransformer):
    """A transformer class for replacing docstrings in a CST.

    Attributes:
        stack: A list to keep track of the current path in the CST.
        docstrings: A dictionary mapping paths in the CST to their corresponding docstrings.
    """
    def __init__(
        self,
        docstrings: dict[tuple[str, ...], cst.SimpleStatementLine],
    ):
        self.stack: list[str] = []
        self.docstrings = docstrings

    def visit_Module(self, node: cst.Module) -> bool | None:
        self.stack.append("")

    def leave_Module(self, original_node: Module, updated_node: Module) -> Module:
        return self._leave(original_node, updated_node)

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self.stack.append(node.name.value)

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
        return self._leave(original_node, updated_node)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self.stack.append(node.name.value)

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
        return self._leave(original_node, updated_node)

    def _leave(self, original_node: DocstringNode, updated_node: DocstringNode) -> DocstringNode:
        key = tuple(self.stack)
        self.stack.pop()

        if hasattr(updated_node, "decorators") and any((i.decorator.value == "overload") for i in updated_node.decorators):
            return updated_node

        statement = self.docstrings.get(key)
        if not statement:
            return updated_node

        original_statement = get_docstring_statement(original_node)

        if isinstance(updated_node, cst.Module):
            body = updated_node.body
            if original_statement:
                return updated_node.with_changes(body=(statement, *body[1:]))
            else:
                updated_node = updated_node.with_changes(body=(statement, cst.EmptyLine(), *body))
                return updated_node

        body = updated_node.body.body[1:] if original_statement else updated_node.body.body
        return updated_node.with_changes(body=updated_node.body.with_changes(body=(statement, *body)))


```

该函数的作用是合并原始代码中的文档字符串，使得每个函数都有相应的文档字符串。具体来说，它将原始代码和文档代码作为参数，解析出它们所包含的C编程语言的字符串对象。然后，它创建了一个代码树和一个文档代码树，接着，对代码树进行访问，按照DocstringCollector类中的文档字符串收集函数，对文档代码树进行访问，按照DocstringTransformer类中的文档字符串变换函数，最后，将修改后的代码树返回。


```py
def merge_docstring(code: str, documented_code: str) -> str:
    """Merges the docstrings from the documented code into the original code.

    Args:
        code: The original code.
        documented_code: The documented code.

    Returns:
        The original code with the docstrings from the documented code.
    """
    code_tree = cst.parse_module(code)
    documented_code_tree = cst.parse_module(documented_code)

    visitor = DocstringCollector()
    documented_code_tree.visit(visitor)
    transformer = DocstringTransformer(visitor.docstrings)
    modified_tree = code_tree.visit(transformer)
    return modified_tree.code

```

# `metagpt/utils/read_document.py`

这段代码是一个Python脚本，主要作用是读取一个docx格式的文件并将其中的段落内容存储在一个列表中。脚本中首先导入了docx库，然后定义了一个名为`read_docx`的函数，该函数接受一个文件路径作为参数，返回一个包含docx文档中所有段落内容的列表。

具体来说，脚本中首先使用`import docx`语句导入了docx库，然后定义了`read_docx`函数，函数内部使用`docx.Document`类打开一个docx文件，并定义了一个`paragraphs_list`变量来存储所有的段落内容，该变量在循环中从doc中获取所有段落并将其内容添加到`paragraphs_list`中。

最后，脚本中没有做其他事情，直接导入了自定义库`get_document`，该库可能是一个用于读取docx文件的库，但并不包含在脚本中使用。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 15:45
@Author  : alexanderwu
@File    : read_document.py
"""

import docx

def read_docx(file_path: str) -> list:
    """Open a docx file"""
    doc = docx.Document(file_path)

    # Create an empty list to store paragraph contents
    paragraphs_list = []

    # Iterate through the paragraphs in the document and add their content to the list
    for paragraph in doc.paragraphs:
        paragraphs_list.append(paragraph.text)

    return paragraphs_list

```

# `metagpt/utils/serialize.py`

这段代码是一个Python脚本，实现了序列化和反序列化Message对象的功能。

具体来说，这段代码定义了一个名为actionoutout_schema_to_mapping的函数，它接收一个Message对象的schema作为参数，返回一个实现了Message对象序列化和反序列化功能的字典。

函数内部首先定义了一个名为mapping的 dictionary，用于存储Message对象的结构信息。然后，对于传入的schema，函数遍历其中 properties的层次结构，根据其类型进行相应的处理。

如果传入的property是字符串类型，则将其转换为元组类型，以便在序列化和反序列化时能够正确地相互转换。

如果传入的property是列表类型，并且其元素也属于字符串类型，则将其转换为元组类型。

如果传入的property是列表类型，并且其元素也属于列表类型，则将其转换为一个元组类型的列表。

这里需要注意的是，对于列表类型的序列化和反序列化，函数的处理方式与实际需要不符。在这里，我们将其转化为一个列表，以便在需要时能够方便地进行扩展。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : the implement of serialization and deserialization

import copy
import pickle
from typing import Dict, List

from metagpt.actions.action_output import ActionOutput
from metagpt.schema import Message


def actionoutout_schema_to_mapping(schema: Dict) -> Dict:
    """
    directly traverse the `properties` in the first level.
    schema structure likes
    ```
    {
        "title":"prd",
        "type":"object",
        "properties":{
            "Original Requirements":{
                "title":"Original Requirements",
                "type":"string"
            },
        },
        "required":[
            "Original Requirements",
        ]
    }
    ```py
    """
    mapping = dict()
    for field, property in schema["properties"].items():
        if property["type"] == "string":
            mapping[field] = (str, ...)
        elif property["type"] == "array" and property["items"]["type"] == "string":
            mapping[field] = (List[str], ...)
        elif property["type"] == "array" and property["items"]["type"] == "array":
            # here only consider the `List[List[str]]` situation
            mapping[field] = (List[List[str]], ...)
    return mapping


```

这段代码定义了两个函数 `serialize_message` 和 `deserialize_message`，它们分别对消息对象 `Message` 进行序列化和反序列化操作。

`serialize_message` 函数接收一个消息对象 `message`，对其进行复制并深拷贝，然后将其 `instruct_content` 属性也复制一份。接下来，如果 `instruct_content` 存在，将其进行序列化并返回。

`deserialize_message` 函数接收一个消息序列化后的字符串 `message_ser`，将其反序列化并返回。

具体来说，这两个函数的核心操作是：

1. 对 `Message` 对象进行复制和深拷贝，以便在序列化和反序列化时避免对原始对象的直接修改。
2. 对 `instruct_content` 属性进行序列化和反序列化，以便在序列化时确保其类型在反序列化时能够正确匹配。
3. 如果序列化成功，对 `instruct_content` 属性进行创建模型和 mapping，以便在反序列化时将模型和 mapping 加载进消息对象中。
4. 如果反序列化成功，对 `Message` 对象中的 `instruct_content` 属性进行更新，以便在下一个循环中对原始对象进行操作。


```py
def serialize_message(message: Message):
    message_cp = copy.deepcopy(message)  # avoid `instruct_content` value update by reference
    ic = message_cp.instruct_content
    if ic:
        # model create by pydantic create_model like `pydantic.main.prd`, can't pickle.dump directly
        schema = ic.schema()
        mapping = actionoutout_schema_to_mapping(schema)

        message_cp.instruct_content = {"class": schema["title"], "mapping": mapping, "value": ic.dict()}
    msg_ser = pickle.dumps(message_cp)

    return msg_ser


def deserialize_message(message_ser: str) -> Message:
    message = pickle.loads(message_ser)
    if message.instruct_content:
        ic = message.instruct_content
        ic_obj = ActionOutput.create_model_class(class_name=ic["class"], mapping=ic["mapping"])
        ic_new = ic_obj(**ic["value"])
        message.instruct_content = ic_new

    return message

```

# `metagpt/utils/singleton.py`

这段代码定义了一个名为`Singleton`的单例模式类。这个模式类的`__call__`方法保证同一时刻只有一个实例被创建。

具体来说，这段代码做以下几件事情：

1. 导入`abc`模块。
2. 定义一个名为`Singleton`的单例模式类，该类继承自`abc.ABCMeta`（抽象中间件）和`type`类型。
3. 在`Singleton`类中定义了一个名为`_instances`的私有变量，用于存储该类的实例。
4. 在`__call__`方法中，首先检查是否已经创建过该类的实例。如果是，则直接返回实例；如果不是，则创建一个新的实例，并将它存储在`_instances`中。
5. 由于`__call__`方法是单例模式类的`__call__`方法，因此它可以确保同一时刻只有一个实例被创建。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 16:15
@Author  : alexanderwu
@File    : singleton.py
"""
import abc


class Singleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
```

# `metagpt/utils/special_tokens.py`

这段代码的作用是定义了两个字符串常量# msg_sep 和 #filename_code_sep，用于在写代码信息中区分不同的代码块。

#token to separate different code messages in a WriteCode Message content
MSG_SEP = "#*000*#"  

这个代码块定义了一个字符串常量# msg_sep，表示代码块之间的分隔符为"#*000*#"。

#token to seperate file name and the actual code text in a code message
FILENAME_CODE_SEP = "#*001*#"  

这个代码块定义了一个字符串常量# filename_code_sep，表示代码块之间的分隔符为"#*001*#"。这个分隔符将文件名和代码文本区分开来，使得代码和文档可以同时存储在一个项目中。


```py
# token to separate different code messages in a WriteCode Message content
MSG_SEP = "#*000*#" 
# token to seperate file name and the actual code text in a code message
FILENAME_CODE_SEP = "#*001*#"

```

# `metagpt/utils/text.py`

这段代码定义了一个名为 `reduce_message_length` 的函数，它接受一个生成器（`msgs`）、一个字符串（`system_text`）和一个整数（`reserved`）。这个函数的作用是减少输入字符串（`system_text`和`msgs`）中连接起来的消息段的长度，使其不超过指定的最大字符数（`max_token`）。

具体来说，函数首先获取一个名为 `model_name` 的字符串，然后使用一个名为 `count_string_tokens` 的函数来计算给定字符串中可用的字符数。接下来，函数遍历输入的每个消息段（`msgs`），如果给定消息段中的字符数小于最大字符数，函数就返回这个消息段。如果最大字符数已经用完了，函数就会 raise 一个 `RuntimeError`。


```py
from typing import Generator, Sequence

from metagpt.utils.token_counter import TOKEN_MAX, count_string_tokens


def reduce_message_length(msgs: Generator[str, None, None], model_name: str, system_text: str, reserved: int = 0,) -> str:
    """Reduce the length of concatenated message segments to fit within the maximum token size.

    Args:
        msgs: A generator of strings representing progressively shorter valid prompts.
        model_name: The name of the encoding to use. (e.g., "gpt-3.5-turbo")
        system_text: The system prompts.
        reserved: The number of reserved tokens.

    Returns:
        The concatenated message segments reduced to fit within the maximum token size.

    Raises:
        RuntimeError: If it fails to reduce the concatenated message length.
    """
    max_token = TOKEN_MAX.get(model_name, 2048) - count_string_tokens(system_text, model_name) - reserved
    for msg in msgs:
        if count_string_tokens(msg, model_name) < max_token:
            return msg

    raise RuntimeError("fail to reduce message length")


```

这段代码定义了一个名为 `generate_prompt_chunk` 的函数，它接受一个文本字符串 `text`，一个用于提示的模板字符串 `prompt_template`，以及一个要使用的模型名称 `model_name`。它的功能是将传入的文本拆分成具有最大关键词长度的片段，并将这些片段生成作为一个生成器。

函数首先使用 `splitlines` 方法将文本拆分成行，然后遍历每个行，计算关键词计数并将关键词添加到当前行中。接下来，它将比较当前可用的关键词数量与最大关键词数量。如果当前可用的关键词数量小于最大关键词数量，函数将生成指定的片段并继续迭代。如果当前可用的关键词数量大于最大关键词数量，函数将拼接一个新的片段并继续迭代。

在生成片段后，如果还有保留的关键词未使用完，函数将生成片段并使用保留的关键词。最后，如果仍然有保留的关键词未使用完，函数将生成一个包含所有关键词的片段并继续迭代。


```py
def generate_prompt_chunk(
    text: str,
    prompt_template: str,
    model_name: str,
    system_text: str,
    reserved: int = 0,
) -> Generator[str, None, None]:
    """Split the text into chunks of a maximum token size.

    Args:
        text: The text to split.
        prompt_template: The template for the prompt, containing a single `{}` placeholder. For example, "### Reference\n{}".
        model_name: The name of the encoding to use. (e.g., "gpt-3.5-turbo")
        system_text: The system prompts.
        reserved: The number of reserved tokens.

    Yields:
        The chunk of text.
    """
    paragraphs = text.splitlines(keepends=True)
    current_token = 0
    current_lines = []

    reserved = reserved + count_string_tokens(prompt_template+system_text, model_name)
    # 100 is a magic number to ensure the maximum context length is not exceeded
    max_token = TOKEN_MAX.get(model_name, 2048) - reserved - 100  

    while paragraphs:
        paragraph = paragraphs.pop(0)
        token = count_string_tokens(paragraph, model_name)
        if current_token + token <= max_token:
            current_lines.append(paragraph)
            current_token += token
        elif token > max_token:
            paragraphs = split_paragraph(paragraph) + paragraphs
            continue
        else:
            yield prompt_template.format("".join(current_lines))
            current_lines = [paragraph]
            current_token = token

    if current_lines:
        yield prompt_template.format("".join(current_lines))


```



该函数 `split_paragraph` 用于将一段文本分解为多个部分，每个部分的边界由指定的分隔符指定。函数接受两个参数：一段文本 `paragraph` 和一个字符串 `sep`，表示每个部分之间的分隔符。函数还接受一个可选参数 `count`，表示要分割成多少部分。

函数内部首先调用一个辅助函数 `_split_text_with_ends`，该函数将文本 `paragraph` 分割成每个部分，并返回每个部分的列表。如果 `paragraph` 的长度小于 `count`，则 `_split_text_with_ends` 可能会返回一个空列表。

然后，函数内部调用另一个辅助函数 `_split_by_count`，该函数将每个部分的分隔符数量固定为 `count`，并将每个部分的分割成子部分。该函数的输出是每个部分的文本列表。

最后，函数内部使用分治法将 `paragraph` 分割成 `count` 个部分。具体来说，对于每个子部分，函数首先创建一个空的列表，然后在列表中复制 `count` 个子部分。然后，函数将这些子部分中的每个部分复制到一个新列表中，并返回该新列表。

函数的返回值是使用 `_split_by_count` 函数分割出来的所有部分的文本列表。


```py
def split_paragraph(paragraph: str, sep: str = ".,", count: int = 2) -> list[str]:
    """Split a paragraph into multiple parts.

    Args:
        paragraph: The paragraph to split.
        sep: The separator character.
        count: The number of parts to split the paragraph into.

    Returns:
        A list of split parts of the paragraph.
    """
    for i in sep:
        sentences = list(_split_text_with_ends(paragraph, i))
        if len(sentences) <= 1:
            continue
        ret = ["".join(j) for j in _split_by_count(sentences, count)]
        return ret
    return _split_by_count(paragraph, count)


```

这段代码定义了一个名为 `decode_unicode_escape` 的函数，用于将给定的文本中的 Unicode 转义序列解码为普通字符。函数的参数是一个字符串 `text`，函数返回解码后的文本。

函数内部首先通过 `encode_utf8` 函数将文本编码为 UTF-8 字符编码，然后使用 `decode_unicode_escape` 函数将编码后的文本解码为 Unicode 转义序列。

接下来是另一个名为 `_split_by_count` 的函数，该函数将给定的列表 `lst` 分割成具有指定长度的子列表，并在分割点处只返回子列表的一部分。函数的参数是一个序列 `lst` 和一个整数 `count`，参数用于分割子列表的阈值。函数返回一个生成器，可以使用 for 循环遍历分割点 `i` 和对应的子列表 `y`。

最后，该代码将所有功能组合在一起，定义了一个函数 `decode_unicode_escape`，它可以将给定的文本中的 Unicode 转义序列解码为普通字符并返回解码后的文本。


```py
def decode_unicode_escape(text: str) -> str:
    """Decode a text with unicode escape sequences.

    Args:
        text: The text to decode.

    Returns:
        The decoded text.
    """
    return text.encode("utf-8").decode("unicode_escape", "ignore")


def _split_by_count(lst: Sequence , count: int):
    avg = len(lst) // count
    remainder = len(lst) % count
    start = 0
    for i in range(count):
        end = start + avg + (1 if i < remainder else 0)
        yield lst[start:end]
        start = end


```

这段代码定义了一个名为 `_split_text_with_ends` 的函数，它接受一个字符串参数 `text` 和一个字符参数 `sep`，并返回一个迭代器 `Parts` 类型。

函数的作用是 Split 一个给定的文本字符串 `text` 按照给定字符 `sep` 划分成多个子字符串，并将这些子字符串组合成一个完整的新字符串，然后返回这个新的字符串。

函数的实现主要分为以下几个步骤：

1. 初始化两个空的字符列表 `parts` 和 `yield`，用于存储 Split 后的结果。

2. 遍历给定的字符串 `text` 和分隔符 `sep`，并将其添加到列表 `parts` 中。

3. 当遍历到的字符正好是分隔符 `sep` 时，使用 `yield` 语句将 `parts` 列表中的所有字符串组合成一个新的字符串，并将 `parts` 列表重置为空。

4. 如果 `parts` 列表不为空，使用 `yield` 语句将 `parts` 列表中的所有字符串组合成一个新的字符串，并将 `parts` 列表重置为空。

5. 在函数的最后，如果 `parts` 列表不为空，使用 `yield` 语句将 `parts` 列表中的所有字符串组合成一个新的字符串，并将 `parts` 列表重置为空。

函数的返回类型为 `Parts`，它代表一个只包含 Split 后的子字符串的列表。


```py
def _split_text_with_ends(text: str, sep: str = "."):
    parts = []
    for i in text:
        parts.append(i)
        if i == sep:
            yield "".join(parts)
            parts = []
    if parts:
        yield "".join(parts)

```

# `metagpt/utils/token_counter.py`

Based on the information provided, it appears that the `token.ipynb` file is a TikTok video with a script that uses the auto-generated text prompts generated by OpenAI GPT models to interact with the user. The `ref1` and `ref2` files are GitHub links to the original code and documentation for the chat model used by the TikTok video.

The `token_counter.py` file is part of the Auto-GPT project by `Significant-Gravitas`. It appears to be a Python file that manages the counter of tokens used by the Auto-GPT model. The `llm` module is responsible for managing the token counter.

The chat model used by the TikTok video is based on the GPT-3.5 model and uses a pre-trained language model with attention based on the input text. The model is trained to generate human-like text responses to the user's input and can generate text in a variety of formats including text embeddings.

The `auto_generated_text_template.py` file is also part of the Auto-GPT project by `Significant-Gravitas`. It appears to be a Python file that contains the pre-defined text templates used by the chat model. The templates are designed to encourage the model to generate human-like text responses to the user's input.

The `AutoGPT.py` file is part of the Chat Models GitHub repository created by `hwchase17`. It appears to be a Python file that contains the code for the Auto-GPT model used by the chat. The model is a pre-trained language model based on the GPT-3.5 model and uses attention based on the input text to generate human-like text responses to the user's input.


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/18 00:40
@Author  : alexanderwu
@File    : token_counter.py
ref1: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
ref2: https://github.com/Significant-Gravitas/Auto-GPT/blob/master/autogpt/llm/token_counter.py
ref3: https://github.com/hwchase17/langchain/blob/master/langchain/chat_models/openai.py
"""
import tiktoken

TOKEN_COSTS = {
    "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
    "gpt-3.5-turbo-0301": {"prompt": 0.0015, "completion": 0.002},
    "gpt-3.5-turbo-0613": {"prompt": 0.0015, "completion": 0.002},
    "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
    "gpt-3.5-turbo-16k-0613": {"prompt": 0.003, "completion": 0.004},
    "gpt-4-0314": {"prompt": 0.03, "completion": 0.06},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
    "gpt-4-32k-0314": {"prompt": 0.06, "completion": 0.12},
    "gpt-4-0613": {"prompt": 0.06, "completion": 0.12},
    "text-embedding-ada-002": {"prompt": 0.0004, "completion": 0.0},
}


```

这段代码是一个Python中的字典，其中包含了一些GPT模型的 token 预处理相关的参数。具体来说，这个字典是一个token max park，它存储了不同GPT模型在不同的条件下可以处理的最大token数。每个键都是一个GPT模型和对应的token数，而每个键中的值则是一个数字，表示这个预处理参数在对应GPT模型下的最大token数。这个字典的作用是提供一些关于GPT模型性能的指标，可以帮助人们了解如何对GPT模型进行预处理。


```py
TOKEN_MAX = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-3.5-turbo-16k-0613": 16384,
    "gpt-4-0314": 8192,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-4-0613": 8192,
    "text-embedding-ada-002": 8192,
}


```

It looks like you are implementing a function `count_message_tokens()` that takes in a list of messages and a model name, and returns the number of tokens in each message. The number of tokens in each message depends on the specific model you are using.

The `gpt-3.5-turbo` model has a default configuration that expects every message to follow the format `<|start|>{role/name}\n{content}<|end|>`. This means that for each message, you need to parse the `<|start|>` token and the role/name from the `{content}` token. The role token is often not included in the message, so you need to extract it from the `{content}` token. The number of tokens in each message is determined by the `tokens_per_message` parameter, which specifies how many tokens should be allocated to each message.

The `gpt-4` model has a default configuration that期望 every message follows the format `<|start|>assistant<|message|>`. This means that for each message, you need to parse the `<|start|>` and `assistant<|message|>` tokens from the `{message}` token. The number of tokens in each message is determined by the `tokens_per_message` parameter, which specifies how many tokens should be allocated to each message.

If the `gpt-4` model is not the one you are using, it is likely that the `count_message_tokens()` function will raise a `NotImplementedError` because the default implementation for the `gpt-4` model is not provided. It is important to check the documentation for the specific model you are using to understand how to properly implement the `count_message_tokens()` function.


```py
def count_message_tokens(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return count_message_tokens(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return count_message_tokens(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


```

这段代码定义了一个名为 `count_string_tokens` 的函数，它接受一个文本字符串 `string` 和一个模型名称 `model_name` 作为参数，然后返回文本字符串中 token 的数量。

函数首先使用 `tiktoken.encoding_for_model` 函数获取一个预训练的模型，该模型使用指定的模型名称。然后，它将接收的文本字符串编码为该模型的字节表示，并返回编码中的字节数。

最后，函数将返回文本字符串中的 token 数量，这个数量将直接作为整数返回。


```py
def count_string_tokens(string: str, model_name: str) -> int:
    """
    Returns the number of tokens in a text string.

    Args:
        string (str): The text string.
        model_name (str): The name of the encoding to use. (e.g., "gpt-3.5-turbo")

    Returns:
        int: The number of tokens in the text string.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))


```

这段代码定义了一个名为 `get_max_completion_tokens` 的函数，它接受两个参数：一个包含消息的列表 `messages` 和一个模型名称 `model`，以及一个默认值 `default`。函数返回一个整数，表示给定模型和消息列表的情况下，能够计算出的最大完成单词的数量。

函数首先检查给定的模型是否在 `TOKEN_MAX` 字典中，如果不存在，则返回预设值 `default`。否则，函数使用 `TOKEN_MAX` 中的模型的值减去消息列表中所有消息的标记词数量，再减去 1，这样得到的结果就是能够计算出的最大完成单词的数量。

函数的实现主要使用了两个辅助函数：`count_message_tokens` 和 `TOKEN_MAX`。`count_message_tokens` 函数用于计算给定消息列表中所有消息的标记词数量，它接收一个消息列表作为输入，并返回其中的标记词数量。`TOKEN_MAX` 是一个字典，其中包含所有给定的模型名称和对应的标记词数量。函数首先尝试从 `TOKEN_MAX` 中查找给定的模型，如果找不到，就返回预设值 `default`。


```py
def get_max_completion_tokens(messages: list[dict], model: str, default: int) -> int:
    """Calculate the maximum number of completion tokens for a given model and list of messages.

    Args:
        messages: A list of messages.
        model: The model name.

    Returns:
        The maximum number of completion tokens.
    """
    if model not in TOKEN_MAX:
        return default
    return TOKEN_MAX[model] - count_message_tokens(messages) - 1

```