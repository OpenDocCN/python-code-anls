# MetaGPT源码解析 9

# `metagpt/tools/moderation.py`

这段代码定义了一个名为 "moderation.py" 的 Python 文件，它使用了环境变量来指定使用的 Python 解释器。

在文件内部，首先定义了一个名为 "moderation" 的类，它包含了一个名为 "moderation" 的方法，该方法接受一个字符串或一个包含字符串的列表作为参数，并返回一个包含每个内容中受到 AMOR(情感管理机器人) 审核标记的结果的列表。

"moderation" 类包含了一个内部方法 "amoderation"，该方法也接受一个字符串或一个包含字符串的列表作为参数，并使用 "llm" 类从 AMOR 模型中获取相应的结果。如果内容中包含标记，该方法返回包含标记的列表。

在 "moderation" 类的 "moderation" 和 "amoderation" 方法之间，存在一个明显的差异：一个是使用 "llm.moderation" 方法，另一个是使用 "llm.amoderation" 方法。这是因为 "moderation" 方法需要返回一个包含标记的列表，而 "amoderation" 方法需要返回一个包含结果的列表。

该代码的主要目的是定义一个可以对内容进行 AMOR 审核的模单个例，并允许在需要时从 AMOR 模型中获取审核结果。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/26 14:27
@Author  : zhanglei
@File    : moderation.py
"""
from typing import Union

from metagpt.llm import LLM


class Moderation:
    def __init__(self):
        self.llm = LLM()

    def moderation(self, content: Union[str, list[str]]):
        resp = []
        if content:
            moderation_results = self.llm.moderation(content=content)
            results = moderation_results.results
            for item in results:
                resp.append(item.flagged)

        return resp

    async def amoderation(self, content: Union[str, list[str]]):
        resp = []
        if content:
            moderation_results = await self.llm.amoderation(content=content)
            results = moderation_results.results
            for item in results:
                resp.append(item.flagged)

        return resp


```

这段代码是一个if语句，判断当前脚本是否为__main__.__name__。如果是，那么执行if语句块内的内容。if语句块中定义了一个名为moderation的变量，并调用了其的moderation函数。这个函数接收一个参数content，该参数是一个列表，包含了一些用户评论。函数的作用是打印出这些评论，不过具体打印哪些评论并没有在if语句块中明确说明。


```py
if __name__ == "__main__":
    moderation = Moderation()
    print(moderation.moderation(content=["I will kill you", "The weather is really nice today", "I want to hit you"]))

```

# `metagpt/tools/prompt_writer.py`

This code defines a class `GPTPromptGenerator` that uses LLM, given an output, request LLM to provide input in different styles (instruction, chatbot, and query). The `__init__` method initializes the generators for each style, which are obtained by calling `getattr` with the argument `f"{style}_style"`.

The `gen_instruction_style` method is an example of how the class can be used to generate instructions. It takes an example output and returns a prompt for the instruction style. The prompt is of the form `X`, where `X` is a variable that represents the example output. The LLM model is then requested to generate an instruction-style prompt for the given example.

Overall, this code provides a simple way to generate prompts for different styles using LLM, which can be useful for various tasks like generating chatbot responses or prompting users for queries.


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/2 16:03
@Author  : alexanderwu
@File    : prompt_writer.py
"""
from typing import Union


class GPTPromptGenerator:
    """Using LLM, given an output, request LLM to provide input (supporting instruction, chatbot, and query styles)"""
    def __init__(self):
        self._generators = {i: getattr(self, f"gen_{i}_style") for i in ['instruction', 'chatbot', 'query']}

    def gen_instruction_style(self, example):
        """Instruction style: Given an output, request LLM to provide input"""
        return f"""Instruction: X
```

This code appears to be a simple Python class that contains two methods for generating outputs in different styles.

The `gen_chatbot_style` method takes an `example` argument and returns a chatbot-style output string. It uses the `f` string method to format the output string, which includes some Chatbot-specific information such as the message from the user, the message's reply, and a Chatbot response.

The `gen_query_style` method takes an `example` argument and returns a query-style output string. It also uses the `f` string method to format the output string, but for a query instead of a chatbot conversation.

In both styles, the `example` argument is passed as an additional argument to the method and used in the output string.


```py
Output: {example}
What kind of instruction might this output come from?
X:"""

    def gen_chatbot_style(self, example):
        """Chatbot style: Given an output, request LLM to provide input"""
        return f"""You are a chatbot. A user sent you an informal message, and you replied as follows.
Message: X
Reply: {example}
What could the informal message X be?
X:"""

    def gen_query_style(self, example):
        """Query style: Given an output, request LLM to provide input"""
        return f"""You are a search engine. Someone made a detailed query, and the most relevant document to this query is as follows.
```

这段代码是一个Python编写的类，名为`QueryX`。它用于生成基于给定example和风格的文本输出。

具体来说，这段代码定义了一个名为`gen`的内部方法，该方法接受两个参数，一个是`example`，表示LLM预期的输出样本，另一个是`style`，表示输出的风格，可以是`all`（获取所有可能的输出）、`instruction`（获取指令风格的输出）、`chatbot`（获取聊天机器人风格的输出）或者`query`（获取查询风格的输出）。

如果`style`不等于`'all'`，那么`gen`方法会使用`self._generators[style]`生成指定风格的输出，如果`style`等于`'all'`，那么`gen`方法会将所有可能的输出生成并返回一个列表。

总之，这段代码的作用是定义了一个用于生成不同风格的文本输出的方法，可以接受一个example和一个style参数，返回一个或多个可能的输出样本。


```py
Query: X
Document: {example} What is the detailed query X?
X:"""

    def gen(self, example: str, style: str = 'all') -> Union[list[str], str]:
        """
        Generate one or multiple outputs using the example, allowing LLM to reply with the corresponding input

        :param example: Expected LLM output sample
        :param style: (all|instruction|chatbot|query)
        :return: Expected LLM input sample (one or multiple)
        """
        if style != 'all':
            return self._generators[style](example)
        return [f(example) for f in self._generators.values()]


```

这段代码定义了一个名为WikiHowTemplate的类。在这个类中，有一个构造函数，用于初始化该类的实例。构造函数中包含一个字符串变量_prompts，它会在创建实例时用于显示一些询问用户的问题的模板。

该类的实例可以通过以下方式调用构造函数：

```py
WikiHowTemplate template;
template.template = "Give me {step} steps to {question}. How to {question}?"
                                 "Do you know how can I {question}? List {step} instructions to {question}?"
                                                 "What are some tips to {question}? What are some steps to {question}?"
                                                 "Can you provide {step} clear and concise instructions on how to {question}? "
                                                 "I'm interested in learning how to {question}. Could you break it down into {step} easy-to-follow steps?"
                                                 "For someone who is new to {question}, what would be {step} key steps to get started? "
                                                 "What is the most efficient way to {question}? Could you provide a list of {step} steps?"
                                                 "Do you have any advice on how to {question} successfully? Maybe a step-by-step guide with {step} steps?"
                                                 "I'm trying to accomplish {question}. Could you walk me through the process with {step} detailed instructions?"
                                                 "What are the essential {step} steps to {question}?"
```

该类的实例也可以通过以下方式调用其中的一个方法：

```py
template.step1 = "First, gather all necessary materials."
template.step2 = "Next, create a detailed itinerary."
template.step3 = "Then, make sure you are prepared."
```

这个类的实例可以帮助用户回答一些常见的问题，例如如何做某件事情，或者列举一些简单的步骤。通过初始化该类的实例，用户可以方便地回答他们感兴趣的问题，并且可以轻松地创建一个清晰的计划。


```py
class WikiHowTemplate:
    def __init__(self):
        self._prompts = """Give me {step} steps to {question}.
How to {question}?
Do you know how can I {question}?
List {step} instructions to {question}.
What are some tips to {question}?
What are some steps to {question}?
Can you provide {step} clear and concise instructions on how to {question}?
I'm interested in learning how to {question}. Could you break it down into {step} easy-to-follow steps?
For someone who is new to {question}, what would be {step} key steps to get started?
What is the most efficient way to {question}? Could you provide a list of {step} steps?
Do you have any advice on how to {question} successfully? Maybe a step-by-step guide with {step} steps?
I'm trying to accomplish {question}. Could you walk me through the process with {step} detailed instructions?
What are the essential {step} steps to {question}?
```

This code defines a class called `EnronTemplate` that generates prompts for the user to answer. The class has an `__init__` method that initializes the `_prompts` variable with a prompt for composing an email with the specified subject and a step-by-step guide for achieving the goal.

The `gen` method takes two arguments, a `question` and a `step`. It returns a list of prompts for achieving the goal. The method uses the `_prompts` template to generate the prompts using the `format` method, passing in the `question` and `step` arguments.

The `EnronTemplate` class has a `__prompts` method that generates a comprehensive guide for achieving the user's goal. The method takes a `question` and a `step` argument and returns a list of prompts, each one more specific than the last. The method uses the `write_email` method to generate a template email with the specified subject and a step-by-step guide for achieving the goal.


```py
I need to {question}, but I'm not sure where to start. Can you give me {step} actionable steps?
As a beginner in {question}, what are the {step} basic steps I should take?
I'm looking for a comprehensive guide on how to {question}. Can you provide {step} detailed steps?
Could you outline {step} practical steps to achieve {question}?
What are the {step} fundamental steps to consider when attempting to {question}?"""

    def gen(self, question: str, step: str) -> list[str]:
        return self._prompts.format(question=question, step=step).splitlines()


class EnronTemplate:
    def __init__(self):
        self._prompts = """Write an email with the subject "{subj}".
Can you craft an email with the subject {subj}?
Would you be able to compose an email and use {subj} as the subject?
```

This code defines a class called `BEAGECTemplate` which generates email templates using a `gen` method. The `gen` method takes a subject `subj` and returns a list of prompts that can be used to construct the email.

The `BEAGECTemplate` class has an `__init__` method which initializes the email templates with a few default prompts for improving the grammar, vocabulary, spelling, and style of the emails.

The `BEAGECTemplate` class also has a `generate_email` method which takes a subject `subj` and returns a list of prompts that can be used to construct the email. The returned prompts include a prompt for shooting the email, another prompt for generating the email, and another prompt for writing the email with the subject of `subj`.

In summary, this code defines a class called `BEAGECTemplate` which generates email templates for a given subject `subj`. The class has methods for generating and constructing the email templates.


```py
Create an email about {subj}.
Draft an email and include the subject "{subj}".
Generate an email about {subj}.
Hey, can you shoot me an email about {subj}?
Do you mind crafting an email for me with {subj} as the subject?
Can you whip up an email with the subject of "{subj}"?
Hey, can you write an email and use "{subj}" as the subject?
Can you send me an email about {subj}?"""

    def gen(self, subj):
        return self._prompts.format(subj=subj).splitlines()


class BEAGECTemplate:
    def __init__(self):
        self._prompts = """Edit and revise this document to improve its grammar, vocabulary, spelling, and style.
```

This code appears to be a script for a document editing tool. It is designed to analyze a text document for errors related to grammar, spelling, and style, and provide suggestions for improvement.

The script starts by generating a list of all errors by splitting the input text into lines. It then defines a `gen` method that returns this list of errors.

Next, the script defines a `refine` method that takes the input text and eliminates all errors related to grammar, vocabulary, and style. It then defines a `polish` method that does the same, but with a focus on improving the writing style.

The script also defines an `enhance` method that goes through the text and fixes all grammar errors and style issues, while improving the overall quality.

Finally, the script defines a `rewrite` method that fixes all grammar errors and style issues, and a `cleanup` method that takes the input text and cleans it up by removing any grammar or spelling errors.

Overall, this code appears to be a sophisticated tool for improving the quality of a text document.


```py
Revise this document to correct all the errors related to grammar, spelling, and style.
Refine this document by eliminating all grammatical, lexical, and orthographic errors and improving its writing style.
Polish this document by rectifying all errors related to grammar, vocabulary, and writing style.
Enhance this document by correcting all the grammar errors and style issues, and improving its overall quality.
Rewrite this document by fixing all grammatical, lexical and orthographic errors.
Fix all grammar errors and style issues and rewrite this document.
Take a stab at fixing all the mistakes in this document and make it sound better.
Give this document a once-over and clean up any grammar or spelling errors.
Tweak this document to make it read smoother and fix any mistakes you see.
Make this document sound better by fixing all the grammar, spelling, and style issues.
Proofread this document and fix any errors that make it sound weird or confusing."""

    def gen(self):
        return self._prompts.splitlines()

```

# `metagpt/tools/sd_engine.py`

这段代码是一个 Python 程序，使用了 `asyncio`、`base64`、`io`、`json`、`os` 和 `typing` 等模块。它的主要作用是获取一个图片的链接，将图片保存为本地文件，并提取图片的 tarball（tar.gz）文件。

具体来说，它实现了以下步骤：

1. 使用 `aiohttp` 库创建一个客户端会话，用于向目标网站发送请求。
2. 使用 `base64` 库将图片的 base64 编码数据转换成字符串并保存到 `base64_图片.txt` 文件中。
3. 使用 `io` 库的 `ReadCloser` 类将图片的 tarball 文件读取并保存到 `tar_图片.txt` 文件中。
4. 使用 `json` 库将图片的信息存储到 `img_info.json` 文件中。
5. 创建一个名为 `images` 的文件夹，并将 `base64_图片.txt` 和 `tar_图片.txt` 文件复制到 `images` 文件夹中。
6. 如果 `images` 文件夹不存在，则创建它。

这段代码的目的是获取一个图片的链接，并下载和保存图片，以便于后续使用。


```py
# -*- coding: utf-8 -*-
# @Date    : 2023/7/19 16:28
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
import asyncio
import base64
import io
import json
import os
from os.path import join
from typing import List

from aiohttp import ClientSession
from PIL import Image, PngImagePlugin

```

这段代码的作用是创建一个Metagpt的配置对象，并设置了一些参数的值，用于控制Prompt、SD模型检查点、批次大小等。

具体来说，它定义了一个名为“payload”的字典，其中包含了一些Prompt相关的参数。它还定义了一个名为“negative_prompt”的参数，指定了当前Prompt的否定提示，这里使用了元启发式否定(easynegative)算法，其值为0.8。

此外，定义了一个名为“override_settings”的参数，其中包含了一些设置，用于在SD模型训练过程中覆盖其他设置，比如将GTM模型中的一部分设置为像素级模型。

该代码还定义了一些批次相关的参数，比如批次大小、迭代次数等。同时，定义了一个名为“alwayson_scripts”的字典，其中包含了一些脚本，这些脚本将在Metagpt训练过程中被执行。

最后，还定义了一个名为“do_not_save_samples”的参数，指定了是否在训练过程中保存样本。


```py
from metagpt.config import Config
from metagpt.const import WORKSPACE_ROOT
from metagpt.logs import logger

config = Config()

payload = {
    "prompt": "",
    "negative_prompt": "(easynegative:0.8),black, dark,Low resolution",
    "override_settings": {"sd_model_checkpoint": "galaxytimemachinesGTM_photoV20"},
    "seed": -1,
    "batch_size": 1,
    "n_iter": 1,
    "steps": 20,
    "cfg_scale": 7,
    "width": 512,
    "height": 768,
    "restore_faces": False,
    "tiling": False,
    "do_not_save_samples": False,
    "do_not_save_grid": False,
    "enable_hr": False,
    "hr_scale": 2,
    "hr_upscaler": "Latent",
    "hr_second_pass_steps": 0,
    "hr_resize_x": 0,
    "hr_resize_y": 0,
    "hr_upscale_to_x": 0,
    "hr_upscale_to_y": 0,
    "truncate_x": 0,
    "truncate_y": 0,
    "applied_old_hires_behavior_to": None,
    "eta": None,
    "sampler_index": "DPM++ SDE Karras",
    "alwayson_scripts": {},
}

```

This is a Python class that appears to implement an SD card output. It has methods for running the SD card API in two different modes (i2i and i2f), as well as methods for generating prompts andnegatives for an image.

The SD card API is accessed through the `sd_t2i_url` property, which is a URL for the SD card API endpoint. The `run` method sends an HTTP POST request to this endpoint with the provided payload. The payload can be configured by setting the `payload` property, which is a dictionary containing the image data.

The `run_i2i` method is not implemented in this class, but appears to be a method for converting an image to an i2i format, which is a cryptocurrency payment format.

The `run_sam` method is not implemented in this class, but appears to be a method for converting an image to a SAM format, which is a file format for samples.

The `run_t2i` method is responsible for running the SD card API for multiple prompts. It takes a list of prompts and returns the generated images. The `run_t2i` method sends a POST request to the SD card API with the provided prompts in the payload.

The `run_i2f` method is not implemented in this class, but appears to be a method for converting an image to an i2f format.

The `run_t2f` method is not implemented in this class, but appears to be a method for converting an image to an i2f format.


```py
default_negative_prompt = "(easynegative:0.8),black, dark,Low resolution"


class SDEngine:
    def __init__(self):
        # Initialize the SDEngine with configuration
        self.config = Config()
        self.sd_url = self.config.get("SD_URL")
        self.sd_t2i_url = f"{self.sd_url}{self.config.get('SD_T2I_API')}"
        # Define default payload settings for SD API
        self.payload = payload
        logger.info(self.sd_t2i_url)

    def construct_payload(
        self,
        prompt,
        negtive_prompt=default_negative_prompt,
        width=512,
        height=512,
        sd_model="galaxytimemachinesGTM_photoV20",
    ):
        # Configure the payload with provided inputs
        self.payload["prompt"] = prompt
        self.payload["negtive_prompt"] = negtive_prompt
        self.payload["width"] = width
        self.payload["height"] = height
        self.payload["override_settings"]["sd_model_checkpoint"] = sd_model
        logger.info(f"call sd payload is {self.payload}")
        return self.payload

    def _save(self, imgs, save_name=""):
        save_dir = WORKSPACE_ROOT / "resources" / "SD_Output"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        batch_decode_base64_to_image(imgs, save_dir, save_name=save_name)

    async def run_t2i(self, prompts: List):
        # Asynchronously run the SD API for multiple prompts
        session = ClientSession()
        for payload_idx, payload in enumerate(prompts):
            results = await self.run(url=self.sd_t2i_url, payload=payload, session=session)
            self._save(results, save_name=f"output_{payload_idx}")
        await session.close()

    async def run(self, url, payload, session):
        # Perform the HTTP POST request to the SD API
        async with session.post(url, json=payload, timeout=600) as rsp:
            data = await rsp.read()

        rsp_json = json.loads(data)
        imgs = rsp_json["images"]
        logger.info(f"callback rsp json is {rsp_json.keys()}")
        return imgs

    async def run_i2i(self):
        # todo: 添加图生图接口调用
        raise NotImplementedError

    async def run_sam(self):
        # todo：添加SAM接口调用
        raise NotImplementedError


```

这段代码的作用是使用Python的Pillow库将Base64编码的图片解码成一张图片，并将解码后的图片保存为指定名称的PNG文件。同时，还支持对多个图片进行批量处理，将解码后的图片保存到指定的文件夹中。

具体来说，代码中定义了两个函数：`decode_base64_to_image` 和 `batch_decode_base64_to_image`。其中，`decode_base64_to_image` 函数接收一个图片对象（通常是从系统中读取的图片），将图片的Base64编码值作为参数，再以该参数值解码图片，最后将解码后的图片保存为指定名称的PNG文件。而 `batch_decode_base64_to_image` 函数则是一个协程函数，将多个图片（通常是从用户输入中读取的图片）批量处理，并将解码后的图片保存到指定文件夹中。

在主函数部分，首先创建了一个SDEngine实例，然后设置了一个提示信息，让用户输入一个图像和文件名。接着使用`event_loop` 循环引擎，将用户输入的提示信息运行至一个`asyncio` 事件循环中。在循环中，调用 `engine.run_t2i` 方法获取用户输入的图像，然后等待事件循环直到 `SDEngine` 实例收到用户输入。最后，调用 `decode_base64_to_image` 或 `batch_decode_base64_to_image` 函数处理用户输入的图片，并将结果保存为PNG文件。


```py
def decode_base64_to_image(img, save_name):
    image = Image.open(io.BytesIO(base64.b64decode(img.split(",", 1)[0])))
    pnginfo = PngImagePlugin.PngInfo()
    logger.info(save_name)
    image.save(f"{save_name}.png", pnginfo=pnginfo)
    return pnginfo, image

def batch_decode_base64_to_image(imgs, save_dir="", save_name=""):
    for idx, _img in enumerate(imgs):
        save_name = join(save_dir, save_name)
        decode_base64_to_image(_img, save_name=save_name)

if __name__ == "__main__":
    engine = SDEngine()
    prompt = "pixel style, game design, a game interface should be minimalistic and intuitive with the score and high score displayed at the top. The snake and its food should be easily distinguishable. The game should have a simple color scheme, with a contrasting color for the snake and its food. Complete interface boundary"

    engine.construct_payload(prompt)

    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(engine.run_t2i(prompt))

```

# `metagpt/tools/search_engine.py`

这段代码是一个Python脚本，用于实现一个搜索引擎。它包含以下功能：

1. 导入必要的模块和函数：通过使用importlib库，它导入了一个函数式编程的接口，这使得我们可以使用Python标准库中的函数来编写代码。同时，它还导入了一个名为sk_function的函数，用于与Semantic Kernel库进行交互。

2. 设置环境变量：设置了一个名为ENV_VAR的的环境变量，这个变量可以用来存储值，例如IPFS和关谷AI的URL。

3. 导入从搜索引擎配置文件中获得的配置：通过使用metagpt.config和metagpt.tools库，从搜索引擎配置文件中读取了一个名为CONFIG的配置对象。

4. 导入SearchEngineType：用于在我们的搜索引擎中使用一个名为SearchEngineType的类型。

5. 通过函数式编程调用sk_function：在搜索引擎配置文件中定义了一个函数式编程的接口，通过这个接口可以调用一个名为sk_function的函数。这个函数接受一个URL参数，用于在Semantic Kernel中查找与该URL相关的技能定义。

6. 使用类型提示和overload：通过使用类型提示和overload，可以定义一个接口类型，并将其类型应用于搜索引擎类型。这样，就可以在定义搜索引擎时使用sk_function函数，同时又可以支持不同搜索引擎的类型。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/6 20:15
@Author  : alexanderwu
@File    : search_engine.py
"""
import importlib
from typing import Callable, Coroutine, Literal, overload, Optional, Union

from semantic_kernel.skill_definition import sk_function

from metagpt.config import CONFIG
from metagpt.tools import SearchEngineType


```

这段代码定义了一个名为 SkSearchEngine 的类，该类包含一个名为 __init__ 的方法，该方法用于创建一个 SearchEngine 对象。

在 SkSearchEngine 的 `__init__` 方法中，创建了一个空的 SearchEngine 对象，并将其存储在 `self.search_engine` 变量中。

该类还定义了一个名为 `searchAsync` 的方法，该方法使用了 `@sk_function` 装饰器来定义搜索操作的说明。

`searchAsync` 方法的作用是在不抛出任何异常的情况下，异步地执行搜索操作并返回结果。它调用了 `self.search_engine.run(query)` 方法来执行实际的搜索操作，其中 `query` 是传入的搜索查询字符串。

由于 `run(query)` 方法的实现在方法中，所以它的实现可能因具体实现而异。但通常情况下，该方法将搜索引擎中的结果进行处理并返回。最后，由于 `async` 关键字，`run(query)` 方法的实现使用了 Python 的 `asyncio` 库，以便异步地执行操作。


```py
class SkSearchEngine:
    def __init__(self):
        self.search_engine = SearchEngine()

    @sk_function(
        description="searches results from Google. Useful when you need to find short "
        "and succinct answers about a specific topic. Input should be a search query.",
        name="searchAsync",
        input_description="search",
    )
    async def run(self, query: str) -> str:
        result = await self.search_engine.run(query)
        return result


```

This is a class called `SearchEngine` that implements the `SearchEngine` interface from the `metagpt.tools.search_engine_serper` module. It has a `run` method that takes a search query and an optional number of results to return, and returns the search results as a string or a list of dictionaries.

The class has several overloads for the `run` method, including one that accepts a `context` argument instead of a `search_engine` object, and another that accepts a `max_results` argument instead of a `max_results` parameter.

The `run` method uses the `serper_google` engine by default, but can be customized by passing an instance of the `SearchEngineType.DIRECT_GOOGLE` class to the `SearchEngine` constructor.

Note that the `run` method is marked as `async` and has an `await` before the `run` method, indicating that it is an asynchronous method.


```py
class SearchEngine:
    """Class representing a search engine.

    Args:
        engine: The search engine type. Defaults to the search engine specified in the config.
        run_func: The function to run the search. Defaults to None.

    Attributes:
        run_func: The function to run the search.
        engine: The search engine type.
    """

    def __init__(
        self,
            engine: Optional[SearchEngineType] = None,
            run_func: Callable[[str, int, bool], Coroutine[None, None, Union[str, list[str]]]] = None,
    ):
        engine = engine or CONFIG.search_engine
        if engine == SearchEngineType.SERPAPI_GOOGLE:
            module = "metagpt.tools.search_engine_serpapi"
            run_func = importlib.import_module(module).SerpAPIWrapper().run
        elif engine == SearchEngineType.SERPER_GOOGLE:
            module = "metagpt.tools.search_engine_serper"
            run_func = importlib.import_module(module).SerperWrapper().run
        elif engine == SearchEngineType.DIRECT_GOOGLE:
            module = "metagpt.tools.search_engine_googleapi"
            run_func = importlib.import_module(module).GoogleAPIWrapper().run
        elif engine == SearchEngineType.DUCK_DUCK_GO:
            module = "metagpt.tools.search_engine_ddg"
            run_func = importlib.import_module(module).DDGAPIWrapper().run
        elif engine == SearchEngineType.CUSTOM_ENGINE:
            pass  # run_func = run_func
        else:
            raise NotImplementedError
        self.engine = engine
        self.run_func = run_func

    @overload
    def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: Literal[True] = True,
    ) -> str:
        ...

    @overload
    def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: Literal[False] = False,
    ) -> list[dict[str, str]]:
        ...

    async def run(self, query: str, max_results: int = 8, as_string: bool = True) -> Union[str, list[dict[str, str]]]:
        """Run a search query.

        Args:
            query: The search query.
            max_results: The maximum number of results to return. Defaults to 8.
            as_string: Whether to return the results as a string or a list of dictionaries. Defaults to True.

        Returns:
            The search results as a string or a list of dictionaries.
        """
        return await self.run_func(query, max_results=max_results, as_string=as_string)

```

# `metagpt/tools/search_engine_ddg.py`

这段代码是一个Python脚本，旨在搜索在线信息的同时尊重未来的功能。它使用`duckduckgo_search`库进行搜索，但首先会尝试从本地安装库或互联网上查找该库。如果本地安装库成功，则行将跳过该行。

具体来说，该代码以下几个主要部分：

1. `import asyncio`：引入了Python 3.7中的异步库`asyncio`，使脚本能够使用异步编程。
2. `import json`：引入了Python标准库中的`json`库，用于将搜索结果以JSON格式输出。
3. `from typing import Literal, overload`：引入了两个`typing`库中的`Literal`类型和`overload`类型，用于定义输入和输出参数的类型。通过这种方式，可以确保脚本在使用`DDGS`库时始终得到正确的参数。
4. `try:`：这是一个尝试块，用于在发生任何异常时回滚操作并打印错误消息。`from duckduckgo_search import DDGS`这一行是导入`duckduckgo_search`库，如果失败，将会抛出`ImportError`异常。
5. `except ImportError:`：这是一个捕获`ImportError`异常的行，用于在发生上述异常时恢复脚本。这一行将会打印错误消息，然后离开脚本。
6. `import asyncio`：再次导入`asyncio`库，以便在脚本中使用异步编程。
7. `from json import dumps`：导入了`json`库的`dumps`函数，以便将搜索结果以JSON格式输出。
8. `DDGS`：导入了`duckduckgo_search`库，它是搜索在线信息的API。
9. `搜索相关信息`：`try`块中的两行代码用于将搜索结果存储到`DDGS`库中。这两行代码将搜索结果的`url`、`描述`和`价格`等信息存储为元组（一种轻量级数据结构，用于在Python中存储可变数量的值）。
10. `print(json.dumps(result, indent=4))`：这一行将搜索结果以JSON格式输出，并使用`indent`参数指定输出的JSON字符串的缩进。`indent`参数是一个字符参数，用于指定输出中的字符数。它也可以根据需要进行调整以适应输出的长度。
11. `print(search_result)`：这一行将搜索结果的原始信息打印出来。它包括搜索结果的`url`、`描述`、`价格`、`搜索时间`和`收藏数`。


```py
#!/usr/bin/env python

from __future__ import annotations

import asyncio
import json
from concurrent import futures
from typing import Literal, overload

try:
    from duckduckgo_search import DDGS
except ImportError:
    raise ImportError(
        "To use this module, you should have the `duckduckgo_search` Python package installed. "
        "You can install it by running the command: `pip install -e.[search-ddg]`"
    )

```



This is a class that defines a Google search API client that can be used to run searches for specific queries. The class has an `async` method for running the search asynchronously, which uses the `run_in_executor` method to execute the search on a coroutine.

The `run` method can take four arguments:

- `query`: The search query (up to 2,000 characters).
- `max_results`: The number of search results to return (up to 100).
- `as_string`: A flag indicating whether the search results should be returned as a formatted string or as a list of dictionaries.
- `focus`: A list of focus terms to include in the search (up to 1,000).

The search results are returned in the format defined by the `as_string` argument. If `as_string` is `True`, the search results are returned in the following format:
```pyjson
[
 {
   "link": "https://www.google.com/search?q=" + query,
   "snippet": "",
   "title": "Search results for " + query
 }
]
```
If `as_string` is `False`, the search results are returned as a list of dictionaries, where each dictionary contains information about each search result, including its `link`, `snippet`, and `title`.

Note that this class uses the `google-api-python-client` library to interact with the Google Search API. This library is not included in the default Python packages and must be installed separately using `pip`.


```py
from metagpt.config import CONFIG


class DDGAPIWrapper:
    """Wrapper around duckduckgo_search API.

    To use this module, you should have the `duckduckgo_search` Python package installed.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        executor: futures.Executor | None = None,
    ):
        kwargs = {}
        if CONFIG.global_proxy:
            kwargs["proxies"] = CONFIG.global_proxy
        self.loop = loop
        self.executor = executor
        self.ddgs = DDGS(**kwargs)

    @overload
    def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: Literal[True] = True,
        focus: list[str] | None = None,
    ) -> str:
        ...

    @overload
    def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: Literal[False] = False,
        focus: list[str] | None = None,
    ) -> list[dict[str, str]]:
        ...

    async def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: bool = True,
    ) -> str | list[dict]:
        """Return the results of a Google search using the official Google API

        Args:
            query: The search query.
            max_results: The number of results to return.
            as_string: A boolean flag to determine the return type of the results. If True, the function will
                return a formatted string with the search results. If False, it will return a list of dictionaries
                containing detailed information about each search result.

        Returns:
            The results of the search.
        """
        loop = self.loop or asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.executor,
            self._search_from_ddgs,
            query,
            max_results,
        )
        search_results = await future

        # Return the list of search result URLs
        if as_string:
            return json.dumps(search_results, ensure_ascii=False)
        return search_results

    def _search_from_ddgs(self, query: str, max_results: int):
        return [
            {"link": i["href"], "snippet": i["body"], "title": i["title"]}
            for (_, i) in zip(range(max_results), self.ddgs.text(query))
        ]


```

这段代码是一个Python脚本，它导入了Python标准库中的fire模块，然后使用fire.Fire函数来启动一个Fire应用程序。

Fire是一个Python库，可以轻松地创建一个交互式的Fire应用程序。它提供了一些方便的功能，如可以在控制台中输出变量、将函数和类作为参数传递给Fire应用程序等。

具体地说，这段代码的作用是创建一个Fire应用程序，通过调用DDGAPIWrapper().run方法来运行应用程序。由于此代码是在if __name__ == "__main__"这个条件下运行的，因此只会执行应用程序的代码部分，而不会加载整个应用程序。


```py
if __name__ == "__main__":
    import fire

    fire.Fire(DDGAPIWrapper().run)

```

# `metagpt/tools/search_engine_googleapi.py`

这段代码是一个Python脚本，使用了`asyncio`、`json`、`typing`、`urllib.parse`、`httplib2`、`pydantic`、`metagpt.config`、`metagpt.logs`模块。它的目的是执行以下任务：

1. 从标准输入读取JSON数据
2. 解析请求URL
3. 使用`httplib2`库的`get`方法下载响应，并将其存储在Python对象中
4. 将JSON数据转换为字典
5. 下载失败时，打印日志信息，并记录到配置文件中

它是一个辅助脚本，用于在遇到相关问题时能够方便地打印日志信息，并下载一个JSON数据。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
from concurrent import futures
from typing import Optional
from urllib.parse import urlparse

import httplib2
from pydantic import BaseModel, validator

from metagpt.config import CONFIG
from metagpt.logs import logger

```

This is a class that defines a Google search API client that can be used to perform search queries and returns the search results in a specific format. The class has methods to run a search with different parameters and return the search results in a JSON format.

The `run` method handles the search query and returns the search results. The method accepts the search query, maximum number of results to return, and options for returning the results in the specified format. The search results are obtained from the Google search API call using the `google_api_client.list()` method.

If any errors occur during the API call, the search results are reset to an empty list. Focus parameters can be used to filter the search results by specific terms, and the search results are returned in a formatted JSON string if the `as_string` parameter is set to `True`.

The `safe_google_results` function is used to sanitize the search results, which is not explicitly defined in the class documentation.


```py
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    raise ImportError(
        "To use this module, you should have the `google-api-python-client` Python package installed. "
        "You can install it by running the command: `pip install -e.[search-google]`"
    )


class GoogleAPIWrapper(BaseModel):
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None
    loop: Optional[asyncio.AbstractEventLoop] = None
    executor: Optional[futures.Executor] = None

    class Config:
        arbitrary_types_allowed = True

    @validator("google_api_key", always=True)
    @classmethod
    def check_google_api_key(cls, val: str):
        val = val or CONFIG.google_api_key
        if not val:
            raise ValueError(
                "To use, make sure you provide the google_api_key when constructing an object. Alternatively, "
                "ensure that the environment variable GOOGLE_API_KEY is set with your API key. You can obtain "
                "an API key from https://console.cloud.google.com/apis/credentials."
            )
        return val

    @validator("google_cse_id", always=True)
    @classmethod
    def check_google_cse_id(cls, val: str):
        val = val or CONFIG.google_cse_id
        if not val:
            raise ValueError(
                "To use, make sure you provide the google_cse_id when constructing an object. Alternatively, "
                "ensure that the environment variable GOOGLE_CSE_ID is set with your API key. You can obtain "
                "an API key from https://programmablesearchengine.google.com/controlpanel/create."
            )
        return val

    @property
    def google_api_client(self):
        build_kwargs = {"developerKey": self.google_api_key}
        if CONFIG.global_proxy:
            parse_result = urlparse(CONFIG.global_proxy)
            proxy_type = parse_result.scheme
            if proxy_type == "https":
                proxy_type = "http"
            build_kwargs["http"] = httplib2.Http(
                proxy_info=httplib2.ProxyInfo(
                    getattr(httplib2.socks, f"PROXY_TYPE_{proxy_type.upper()}"),
                    parse_result.hostname,
                    parse_result.port,
                ),
            )
        service = build("customsearch", "v1", **build_kwargs)
        return service.cse()

    async def run(
        self,
        query: str,
        max_results: int = 8,
        as_string: bool = True,
        focus: list[str] | None = None,
    ) -> str | list[dict]:
        """Return the results of a Google search using the official Google API.

        Args:
            query: The search query.
            max_results: The number of results to return.
            as_string: A boolean flag to determine the return type of the results. If True, the function will
                return a formatted string with the search results. If False, it will return a list of dictionaries
                containing detailed information about each search result.
            focus: Specific information to be focused on from each search result.

        Returns:
            The results of the search.
        """
        loop = self.loop or asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.executor, self.google_api_client.list(q=query, num=max_results, cx=self.google_cse_id).execute
        )
        try:
            result = await future
            # Extract the search result items from the response
            search_results = result.get("items", [])

        except HttpError as e:
            # Handle errors in the API call
            logger.exception(f"fail to search {query} for {e}")
            search_results = []

        focus = focus or ["snippet", "link", "title"]
        details = [{i: j for i, j in item_dict.items() if i in focus} for item_dict in search_results]
        # Return the list of search result URLs
        if as_string:
            return safe_google_results(details)

        return details


```

这段代码定义了一个名为`safe_google_results`的函数，它接受一个字符串参数`results`，并返回一个字符串类型的结果。

函数内部的逻辑如下：

1. 如果`results`是列表，那么遍历列表中的每个结果，并将它们作为一个参数传递给一个内部函数。这个内部函数将使用`json.dumps()`将列表中的每个结果序列化并转换为JSON格式。
2. 如果`results`是一个字符串，那么将其解码为`utf-8`编码，并使用`decode()`将其转换为`utf-8`编码的字符串。
3. 最后，函数返回`safe_message`，它是搜索结果的JSON编码形式。

这个函数的作用是确保在搜索Google时，即使返回的结果不是Python代码可以处理的类型，也能以一种安全的方式进行处理。


```py
def safe_google_results(results: str | list) -> str:
    """Return the results of a google search in a safe format.

    Args:
        results: The search results.

    Returns:
        The results of the search.
    """
    if isinstance(results, list):
        safe_message = json.dumps([result for result in results])
    else:
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")
    return safe_message


```

这段代码是一个Python脚本，它导入了Python标准库中的fire库，然后使用fire库中的🔥函数创建了一个GoogleAPIWrapper对象，并调用该对象的run方法。

具体来说，这段代码的作用是运行一个使用Google API的程序，并在程序运行时启动该程序。Google API是一个Python库，提供了访问Google服务的接口，包括Gmail、Google Drive等。通过🔥函数，该库可以方便地创建一个GoogleAPIWrapper对象，并调用其中的run方法来运行程序。

在程序运行时，fire库中的🔥函数会将所有使用该库的Python脚本和当前工作目录下的Python库都上传到Google Cloud Platform（GCP）的 runners 服务中，并在该服务中运行这些脚本。因此，这段代码会在运行程序时启动Google API，并将其与Google APIWrapper对象一起使用。


```py
if __name__ == "__main__":
    import fire

    fire.Fire(GoogleAPIWrapper().run)

```

# `metagpt/tools/search_engine_meilisearch.py`

这段代码定义了一个名为 `DataSource` 的类，用于存储从网站或其他数据源获取的数据。在这个类中，有两个方法：`__init__` 和 `__str__`。

`__init__` 方法接受两个参数 `name` 和 `url`，分别用于存储数据源名称和URL。在创建新实例时，该方法会执行 `self.name = name` 和 `self.url = url`，使数据源得到初始化。

`__str__` 方法返回一个字符串，通常用于打印或格式化数据源名称。在这里，它使用了 `type(self).__str__` 的语法来获取数据源名称的类型（因为 `self` 是在 `DataSource` 类中创建的实例），然后返回 `type(self).__str__` 的结果。

注意，这段代码没有定义任何函数或类，因此不会产生任何函数调用或类成员。它只是一个简单的数据源定义类，用于在程序运行时创建和初始化数据源对象。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/22 21:33
@Author  : alexanderwu
@File    : search_engine_meilisearch.py
"""

from typing import List

import meilisearch
from meilisearch.index import Index


class DataSource:
    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url


```

该代码是一个名为MeilisearchEngine的类，用于实现MeiliSearch搜索引擎。它包含两个方法：`__init__`和`search`。下面分别解释这两个方法的作用。

1. `__init__`方法：

该方法接受两个参数：`url`和`token`。它使用MeiliSearch提供的`Client`类来创建一个MeiliSearch客户端实例，并将其存储在`self`变量中。同时，它还创建了一个名为`_index`的索引对象，用于将搜索结果存储到该索引中。

2. `search`方法：

该方法接受一个参数：`query`。它使用MeiliSearch提供的`Index`对象来搜索MeiliSearch索引中包含查询的文档。如果查询包含有效关键词，它将返回搜索结果。如果发生错误，例如MeiliSearch API的响应不正确，它将捕获并打印错误信息。

总之，该代码是一个简单的MeiliSearch搜索引擎，用于在MeiliSearch索引中搜索包含有效关键词的文档。


```py
class MeilisearchEngine:
    def __init__(self, url, token):
        self.client = meilisearch.Client(url, token)
        self._index: Index = None

    def set_index(self, index):
        self._index = index

    def add_documents(self, data_source: DataSource, documents: List[dict]):
        index_name = f"{data_source.name}_index"
        if index_name not in self.client.get_indexes():
            self.client.create_index(uid=index_name, options={'primaryKey': 'id'})
        index = self.client.get_index(index_name)
        index.add_documents(documents)
        self.set_index(index)

    def search(self, query):
        try:
            search_results = self._index.search(query)
            return search_results['hits']
        except Exception as e:
            # Handle MeiliSearch API errors
            print(f"MeiliSearch API error: {e}")
            return []

```

# `metagpt/tools/search_engine_serpapi.py`

该代码是一个Python脚本，名为`search_engine_serpapi.py`，使用Python 3.8类型注释。

该脚本的主要作用是实现一个搜索引擎服务，使用Google搜索引擎，并将结果返回给用户。以下是该脚本的功能和结构：

1. 从`requests`库导入一些常用的函数和类，包括`requests`库用于发送HTTP请求、`BeautifulSoup`库用于解析HTML文档等。

2. 通过`pydantic`库定义一个`SearchEngineSchema`类，该类用于定义搜索查询的结构，包括`query`字段、`url`字段、`project_id`字段等。

3. 通过`asyncio`库的`run`函数来启动一个独立的协程，该协程负责处理搜索请求并获取结果。

4. 在协程中，使用`aiohttp`库发送HTTP请求，获取搜索结果，并使用`beautifulsoup4`库解析HTML文档。

5. 将结果存储为Python字典或元组，并返回给用户。

6. 在文件级别使用`#!/usr/bin/env python`语句，说明该脚本是一个Python脚本，应该使用`python`解释器来运行。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/23 18:27
@Author  : alexanderwu
@File    : search_engine_serpapi.py
"""
from typing import Any, Dict, Optional, Tuple

import aiohttp
from pydantic import BaseModel, Field, validator

from metagpt.config import CONFIG


```

This is a Python implementation of a function `get_search_result_snippet` that takes a SerpAPI response object and returns the snippet of the answer box in the format of `"<answer_box.snippet>"`.

It uses the `get_focused` function to get the words in the answer box that match the given focus, and then joins them together into a single string.

If the `answer_box` key is not found in the response, the function returns the string `"No good search result found"`.

It also handles the case when the `sports_results` or `knowledge_graph` key is present in the response, in which case the function returns the `game_spotlight` or `description` respectively.


```py
class SerpAPIWrapper(BaseModel):
    search_engine: Any  #: :meta private:
    params: dict = Field(
        default={
            "engine": "google",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
        }
    )
    serpapi_api_key: Optional[str] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        arbitrary_types_allowed = True

    @validator("serpapi_api_key", always=True)
    @classmethod
    def check_serpapi_api_key(cls, val: str):
        val = val or CONFIG.serpapi_api_key
        if not val:
            raise ValueError(
                "To use, make sure you provide the serpapi_api_key when constructing an object. Alternatively, "
                "ensure that the environment variable SERPAPI_API_KEY is set with your API key. You can obtain "
                "an API key from https://serpapi.com/."
            )
        return val

    async def run(self, query, max_results: int = 8, as_string: bool = True, **kwargs: Any) -> str:
        """Run query through SerpAPI and parse result async."""
        return self._process_response(await self.results(query, max_results), as_string=as_string)

    async def results(self, query: str, max_results: int) -> dict:
        """Use aiohttp to run query through SerpAPI and return the results async."""

        def construct_url_and_params() -> Tuple[str, Dict[str, str]]:
            params = self.get_params(query)
            params["source"] = "python"
            params["num"] = max_results
            params["output"] = "json"
            url = "https://serpapi.com/search"
            return url, params

        url, params = construct_url_and_params()
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    res = await response.json()
        else:
            async with self.aiosession.get(url, params=params) as response:
                res = await response.json()

        return res

    def get_params(self, query: str) -> Dict[str, str]:
        """Get parameters for SerpAPI."""
        _params = {
            "api_key": self.serpapi_api_key,
            "q": query,
        }
        params = {**self.params, **_params}
        return params

    @staticmethod
    def _process_response(res: dict, as_string: bool) -> str:
        """Process response from SerpAPI."""
        # logger.debug(res)
        focus = ["title", "snippet", "link"]
        get_focused = lambda x: {i: j for i, j in x.items() if i in focus}

        if "error" in res.keys():
            raise ValueError(f"Got error from SerpAPI: {res['error']}")
        if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
            toret = res["answer_box"]["answer"]
        elif "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            toret = res["answer_box"]["snippet"]
        elif "answer_box" in res.keys() and "snippet_highlighted_words" in res["answer_box"].keys():
            toret = res["answer_box"]["snippet_highlighted_words"][0]
        elif "sports_results" in res.keys() and "game_spotlight" in res["sports_results"].keys():
            toret = res["sports_results"]["game_spotlight"]
        elif "knowledge_graph" in res.keys() and "description" in res["knowledge_graph"].keys():
            toret = res["knowledge_graph"]["description"]
        elif "snippet" in res["organic_results"][0].keys():
            toret = res["organic_results"][0]["snippet"]
        else:
            toret = "No good search result found"

        toret_l = []
        if "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            toret_l += [get_focused(res["answer_box"])]
        if res.get("organic_results"):
            toret_l += [get_focused(i) for i in res.get("organic_results")]

        return str(toret) + "\n" + str(toret_l) if as_string else toret_l


```

这段代码是一个Python脚本，它导入了Python标准库中的fire库，然后使用fire库中的🔥函数创建了一个火堆。接下来，将Fire函数的输入参数设置为调用SerpAPIWrapper().run方法的结果，即将这个方法传递给fire库中的🔥函数。最后，通过在if语句中检查当前脚本是否作为主程序运行，如果是，则执行Fire函数中的内容，否则不执行。


```py
if __name__ == "__main__":
    import fire

    fire.Fire(SerpAPIWrapper().run)

```

# `metagpt/tools/search_engine_serper.py`

该代码是一个Python脚本，用于搜索Google搜索引擎上的SERP API数据。它包括以下几个主要部分：

1. 导入一些必要的模块和函数：`json` 用于读取和写入JSON数据，`aiohttp` 用于发送HTTP请求，`typing` 用于😉


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/23 18:27
@Author  : alexanderwu
@File    : search_engine_serpapi.py
"""
import json
from typing import Any, Dict, Optional, Tuple

import aiohttp
from pydantic import BaseModel, Field, validator

from metagpt.config import CONFIG


```

This is a Python implementation of a simple web scraper that retrieves a search result from the SerpAPI and displays the relevant information. The scraper focuses on the title, snippet, and link of the search result, and also extracts information from the answer box if available. It also filters out results from the `sports_results` and `knowledge_graph` sections.

The scraper uses the `requests` and `beautifulsoup4` libraries for making HTTP requests and parsing HTML, respectively. It also defines a `get_focused` function to retrieve the search result with the specified focus.

The scraper also checks for errors and returns a meaningful message if one exists. If the `answer_box` and `snippet` keys are not found in the response, the scraper returns a meaningful message.


```py
class SerperWrapper(BaseModel):
    search_engine: Any  #: :meta private:
    payload: dict = Field(default={"page": 1, "num": 10})
    serper_api_key: Optional[str] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        arbitrary_types_allowed = True

    @validator("serper_api_key", always=True)
    @classmethod
    def check_serper_api_key(cls, val: str):
        val = val or CONFIG.serper_api_key
        if not val:
            raise ValueError(
                "To use, make sure you provide the serper_api_key when constructing an object. Alternatively, "
                "ensure that the environment variable SERPER_API_KEY is set with your API key. You can obtain "
                "an API key from https://serper.dev/."
            )
        return val

    async def run(self, query: str, max_results: int = 8, as_string: bool = True, **kwargs: Any) -> str:
        """Run query through Serper and parse result async."""
        if isinstance(query, str):
            return self._process_response((await self.results([query], max_results))[0], as_string=as_string)
        else:
            results = [self._process_response(res, as_string) for res in await self.results(query, max_results)]
        return "\n".join(results) if as_string else results

    async def results(self, queries: list[str], max_results: int = 8) -> dict:
        """Use aiohttp to run query through Serper and return the results async."""

        def construct_url_and_payload_and_headers() -> Tuple[str, Dict[str, str]]:
            payloads = self.get_payloads(queries, max_results)
            url = "https://google.serper.dev/search"
            headers = self.get_headers()
            return url, payloads, headers

        url, payloads, headers = construct_url_and_payload_and_headers()
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=payloads, headers=headers) as response:
                    res = await response.json()
        else:
            async with self.aiosession.get.post(url, data=payloads, headers=headers) as response:
                res = await response.json()

        return res

    def get_payloads(self, queries: list[str], max_results: int) -> Dict[str, str]:
        """Get payloads for Serper."""
        payloads = []
        for query in queries:
            _payload = {
                "q": query,
                "num": max_results,
            }
            payloads.append({**self.payload, **_payload})
        return json.dumps(payloads, sort_keys=True)

    def get_headers(self) -> Dict[str, str]:
        headers = {"X-API-KEY": self.serper_api_key, "Content-Type": "application/json"}
        return headers

    @staticmethod
    def _process_response(res: dict, as_string: bool = False) -> str:
        """Process response from SerpAPI."""
        # logger.debug(res)
        focus = ["title", "snippet", "link"]

        def get_focused(x):
            return {i: j for i, j in x.items() if i in focus}

        if "error" in res.keys():
            raise ValueError(f"Got error from SerpAPI: {res['error']}")
        if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
            toret = res["answer_box"]["answer"]
        elif "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            toret = res["answer_box"]["snippet"]
        elif "answer_box" in res.keys() and "snippet_highlighted_words" in res["answer_box"].keys():
            toret = res["answer_box"]["snippet_highlighted_words"][0]
        elif "sports_results" in res.keys() and "game_spotlight" in res["sports_results"].keys():
            toret = res["sports_results"]["game_spotlight"]
        elif "knowledge_graph" in res.keys() and "description" in res["knowledge_graph"].keys():
            toret = res["knowledge_graph"]["description"]
        elif "snippet" in res["organic"][0].keys():
            toret = res["organic"][0]["snippet"]
        else:
            toret = "No good search result found"

        toret_l = []
        if "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
            toret_l += [get_focused(res["answer_box"])]
        if res.get("organic"):
            toret_l += [get_focused(i) for i in res.get("organic")]

        return str(toret) + "\n" + str(toret_l) if as_string else toret_l


```

这段代码使用了Python的 `__name__` 特性，用来判断当前脚本是否作为主程序运行。如果是，那么就会执行 `fire.Fire` 函数，将 `SerperWrapper` 类的实例调用并传入 `run` 方法，从而引发一场火灾。

具体来说，`fire.Fire` 是一个Python标准库中的函数，可以将一个或多个参数传入并触发火焰，起到激发热情、启发灵感、激发创意等作用。在这个例子中，它接收一个参数 `SerperWrapper`，代表一个包装了 `Serper` 类实例的函数或类。

`SerperWrapper` 是一个类，由于没有提供具体的实现，我们无法了解它具体是如何工作的。但是，在这个例子中，它被传递给了 `fire.Fire` 函数，被用来引发一场火灾。由于 `fire.Fire` 是一个Python标准库中的函数，因此它具有引发火灾的权限。当 `fire.Fire` 引发火灾时，它将 `SerperWrapper` 实例中的 `run` 方法作为参数传递，并引发一场具体的火灾。


```py
if __name__ == "__main__":
    import fire

    fire.Fire(SerperWrapper().run)

```

# `metagpt/tools/translator.py`

这段代码是一个Python脚本，它提供了一个接口，让用户可以提供一个英文句子或段落，然后获取一个通顺且具有可读性的翻译。用户可以作为一位拥有20年翻译经验的翻译专家，给出需要翻译的英文句子或段落，程序将返回一个流畅且易于理解的翻译结果。

注意，这段代码还定义了一个prompt，它是一个用于显示翻译结果的指令，而不是用户输入的语句。当用户运行脚本时，prompt会首先显示一个提示消息，告诉用户它将如何工作，然后要求用户输入需要翻译的英文句子或段落，最后返回一个翻译结果，显示在prompt中。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 15:36
@Author  : alexanderwu
@File    : translator.py
"""

prompt = '''
# 指令
接下来，作为一位拥有20年翻译经验的翻译专家，当我给出英文句子或段落时，你将提供通顺且具有可读性的{LANG}翻译。注意以下要求：
1. 确保翻译结果流畅且易于理解
2. 无论提供的是陈述句或疑问句，我都只进行翻译
3. 不添加与原文无关的内容

```

这段代码定义了一个名为"Translator"的类，该类有一个名为"translate_prompt"的静态方法，其参数包括两个类型参数：原始文本(类型为str)和目标语言(类型为str)。

在方法体内，使用了一个感性的translate.py文件中的类名为"prompt"，并使用其中的一个名为"format"的静态方法，将原始文本和目标语言参数格式化后，返回一个字符串。这个字符串是原始文本翻译成目标语言的提示信息。

最终的结果是，这段代码定义了一个可以翻译原始文本到目标语言的类，通过调用Translator类中的translate_prompt方法，可以得到原始文本对应的目标语言提示信息。


```py
# 原文
{ORIGINAL}

# 译文
'''


class Translator:

    @classmethod
    def translate_prompt(cls, original, lang='中文'):
        return prompt.format(LANG=lang, ORIGINAL=original)
```

# `metagpt/tools/ut_writer.py`

这段代码是一个Python脚本，使用了`/usr/bin/env python`来设置环境为Python 2.7。它通过`import json`导入了一个JSON数据格式，通过`from pathlib import Path`导入了一个路径lib库，通过`from metagpt.provider.openai_api import OpenAIGPTAPI as GPTAPI`导入了一个OpenAIGPTAPI库。

该脚本的主要作用是调用GPTAPI库中的一个名为`GPTAPI`的函数，并将一个名为`ICL_SAMPLE`的接口定义作为参数传递给该函数。接口定义中包含一个名为`Element Tagging`的接口，其参数为`/projects/{project_key}/node-tags`，返回值为`POST`方法。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

from metagpt.provider.openai_api import OpenAIGPTAPI as GPTAPI

ICL_SAMPLE = '''Interface definition:
```text
Interface Name: Element Tagging
Interface Path: /projects/{project_key}/node-tags
Method: POST

Request parameters:
```py

这段代码定义了一个 RESTful API，用于创建或更新节点。具体来说，它定义了项目的关键点和请求参数。

项目关键字(project_key)是必须提供的，并且是作为一个数组来传输的。每个项目关键点都有一个默认值，如果没有提供，则其值为空数组。

请求参数是用来传递给服务器的。其中，nodes参数是必需的，它是作为一个数组来传输的。节点键(node_key)和标签(tags)都是必需的，并且是字符串和数组类型。节点类型(node_type)必须是Dataset或Recipe中的一个，而模式(mode)必须是ADD或DELETE中的一个。

运行时参数operations是一个数组，用于传递给服务器，以定义要执行的操作。标签(tags)也是一个数组，用于指定要附加的标签。

服务器将使用这些参数来创建或更新节点。一旦节点已经被创建或更新，将返回包含项目关键字的响应，其中包含节点和操作类型。


```
Path parameters:
project_key

Body parameters:
Name	Type	Required	Default Value	Remarks
nodes	array	Yes		Nodes
	node_key	string	No		Node key
	tags	array	No		Original node tag list
	node_type	string	No		Node type DATASET / RECIPE
operations	array	Yes		
	tags	array	No		Operation tag list
	mode	string	No		Operation type ADD / DELETE

Return data:
Name	Type	Required	Default Value	Remarks
```py

这段代码定义了一个测试套件，通过传入不同的参数来测试不同情况下的返回值。在这个测试套件中，有以下几个参数：

* `project_key`：项目编号，用于指定输入的节点列表。
* `nodes`：节点列表，用于指定每个项目的节点数据。
* `operations`：期望的操作，用于指定每个操作的操作类型。
* `expected_msg`：期望的消息，用于指定每个测试的预期输出。

通过传入不同的参数，可以测试以下情况：

* `project_key` 和 `nodes` 都正确，但没有进行任何操作，预期输出为空字符串 `""`。
* `project_key` 正确，`nodes` 中有一个节点，但该节点没有标签，预期输出为包含标签信息的节点对象 `None`。
* `project_key` 正确，`nodes` 中有两个节点，分别对应标签为 "tag1" 和 "tag2"，预期输出为包含标签信息的节点对象 `{'dataset_001': {'标签': ['tag1', 'tag2']}, 'dataset_002': {'标签': ['tag1']}}`。
* `project_key` 错误，例如传入 `"abc"`，预期输出为包含标签信息的节点对象 `None`。
* `nodes` 中有一个节点，该节点对应标签为 "new_tag1"， `mode` 为 "ADD"，预期输出为包含标签信息的节点对象 `{'dataset_001': {'标签': ['new_tag1']}, 'dataset_002': {'标签': ['new_tag1']}}`。
* `nodes` 中有一个节点，该节点对应标签为 "new_tag1"， `mode` 为 "DELETE"，预期输出为空字符串 `""`。
* `nodes` 中有一个节点，该节点对应标签为空， `mode` 为 "ADD"，预期输出为包含标签信息的节点对象 `{'dataset_001': {'标签': ['new_tag1']}, 'dataset_002': {'标签': []}}`。
* `operations` 为 `None`，预期输出为空字符串 `""`。
* `operations` 为 `"ADD"`，预期输出为空字符串 `""`。
* `operations` 为 `"DELETE"`，预期输出为空字符串 `""`。
* `operations` 为 `"MAX_NUMS"`，预期输出为空字符串 `""`。

测试套件使用了 `pytest` 库，通过调用 `pytest.mark.parametrize` 函数可以传入不同的参数，从而组成一组测试用例。在调用 `pytest.mark.parametrize` 函数时，可以传入一个或多个参数，如 `parametrize("project_key, nodes, operations, expected_msg", [...])`，也可以只传入一个参数，如 `parametrize("project_key, nodes, operations", [...])`。


```
code	integer	Yes		Status code
msg	string	Yes		Prompt message
data	object	Yes		Returned data
list	array	No		Node list true / false
node_type	string	No		Node type DATASET / RECIPE
node_key	string	No		Node key
```py

Unit test：
```python
@pytest.mark.parametrize(
"project_key, nodes, operations, expected_msg",
[
("project_key", [{"node_key": "dataset_001", "tags": ["tag1", "tag2"], "node_type": "DATASET"}], [{"tags": ["new_tag1"], "mode": "ADD"}], "success"),
("project_key", [{"node_key": "dataset_002", "tags": ["tag1", "tag2"], "node_type": "DATASET"}], [{"tags": ["tag1"], "mode": "DELETE"}], "success"),
("", [{"node_key": "dataset_001", "tags": ["tag1", "tag2"], "node_type": "DATASET"}], [{"tags": ["new_tag1"], "mode": "ADD"}], "Missing the required parameter project_key"),
(123, [{"node_key": "dataset_001", "tags": ["tag1", "tag2"], "node_type": "DATASET"}], [{"tags": ["new_tag1"], "mode": "ADD"}], "Incorrect parameter type"),
("project_key", [{"node_key": "a"*201, "tags": ["tag1", "tag2"], "node_type": "DATASET"}], [{"tags": ["new_tag1"], "mode": "ADD"}], "Request parameter exceeds field boundary")
]
)
```py

这段代码定义了一个名为 `test_node_tags` 的函数，属于接口定义和单元测试示例。它接受一个名为 `project_key` 的项目键，四个参数 `nodes`、`operations` 和 `expected_msg`，分别表示测试节点、测试操作和预期结果。

具体来说，这段代码的作用是测试一个名为 `nodes` 的参数，验证其是否符合预期。通过调用该函数，可以得到一个测试套件，其中包括一些测试函数，如 `test_request_params`、`test_field_boundary_verification` 等。


```
def test_node_tags(project_key, nodes, operations, expected_msg):
    pass

# The above is an interface definition and a unit test example.
# Next, please play the role of an expert test manager with 20 years of experience at Google. When I give the interface definition, 
# reply to me with a unit test. There are several requirements:
# 1. Only output one `@pytest.mark.parametrize` and the corresponding test_<interface name> function (inside pass, do not implement).
# -- The function parameter contains expected_msg for result verification.
# 2. The generated test cases use shorter text or numbers and are as compact as possible.
# 3. If comments are needed, use Chinese.

# If you understand, please wait for me to give the interface definition and just answer "Understood" to save tokens.
'''

ACT_PROMPT_PREFIX = '''Refer to the test types: such as missing request parameters, field boundary verification, incorrect field type.
```py

This code is a part of a `pytest.mark.parametrize` scope, which allows for the specification of test fixtures (parametrized tests) within one test method.

The given code defines a test fixture that generates 10 parameterized test cases for the OCR (On-Centity Recognition) API, specifically for the task of Contract Treaty Task OCR. The tests will be executed within the `@pytest.mark.parametrize` scope, meaning that each test case will be independent of the others.

The `OCR_API_DOC` documentation provides additional context for the API, indicating its purpose and usage.


```
Please output 10 test cases within one `@pytest.mark.parametrize` scope.
```pytext
'''

YFT_PROMPT_PREFIX = '''Refer to the test types: such as SQL injection, cross-site scripting (XSS), unauthorized access and privilege escalation, 
authentication and authorization, parameter verification, exception handling, file upload and download.
Please output 10 test cases within one `@pytest.mark.parametrize` scope.
```text
'''

OCR_API_DOC = '''```pytext
Interface Name: OCR recognition
Interface Path: /api/v1/contract/treaty/task/ocr
Method: POST

```

这段代码定义了一个 RESTful API，用于在不同文档之间传递参数。具体来说，它实现了以下功能：

1. 定义了输入参数：包括路径参数、查询参数和请求体参数。
2. 定义了响应数据：包括只读参数和读写参数。
3. 在代码中定义了请求参数，包括文件 ID、输入合同 ID、开始和结束时间以及提取类型等。
4. 支持路径参数，可以通过 URL 或 HTTP 状态码传递文件 ID。
5. 支持查询参数，可以通过 URL 或 HTTP 状态码传递开始和结束时间。
6. 支持请求体参数，可以通过 HTTP POST 或 HTTP PUT 请求传递。
7. 提供了用于解析 JSON 和 XML 的库。
8. 通过 `exports` 导出了应用程序模块，方便与其他模块使用。


```py
Request Parameters:
Path Parameters:

Body Parameters:
Name	Type	Required	Default Value	Remarks
file_id	string	Yes		
box	array	Yes		
contract_id	number	Yes		Contract id
start_time	string	No		yyyy-mm-dd
end_time	string	No		yyyy-mm-dd
extract_type	number	No		Recognition type 1- During import 2- After import Default 1

Response Data:
Name	Type	Required	Default Value	Remarks
code	integer	Yes		
```

This is a class definition that uses the Black Lagoon API to build and train language models. The class provides methods to generate human-generated questions and answers, as well as save the generated data to files.

The `__init__` method is used to store the questions and answers data in memory, and the class method `_generate_ut` is used to process the under-the-table data.

The `_ask_gpt_and_save` method is used to generate questions and save both the questions and answers to files.

The `generate_questions` method is a utility method that asks the Black Lagoon API to generate the questions and save them to a file.

The `gpt_msgs_to_code` method is a utility method that chooses the appropriate method to call the Black Lagoon API based on the `chatgpt_method` value and generates the code to ask the questions.

The `get_file_path` method is a utility method that generates the file path based on the base path and the file name.


```py
message	string	Yes		
data	object	Yes		
```
'''


class UTGenerator:
    """UT Generator: Construct UT through API documentation"""

    def __init__(self, swagger_file: str, ut_py_path: str, questions_path: str,
                 chatgpt_method: str = "API", template_prefix=YFT_PROMPT_PREFIX) -> None:
        """Initialize UT Generator

        Args:
            swagger_file: path to the swagger file
            ut_py_path: path to store test cases
            questions_path: path to store the template, facilitating subsequent checks
            chatgpt_method: API method
            template_prefix: use the template, default is YFT_UT_PROMPT
        """
        self.swagger_file = swagger_file
        self.ut_py_path = ut_py_path
        self.questions_path = questions_path
        assert chatgpt_method in ["API"], "Invalid chatgpt_method"
        self.chatgpt_method = chatgpt_method

        # ICL: In-Context Learning, provide an example here for GPT to mimic
        self.icl_sample = ICL_SAMPLE
        self.template_prefix = template_prefix

    def get_swagger_json(self) -> dict:
        """Load Swagger JSON from a local file"""
        with open(self.swagger_file, "r", encoding="utf-8") as file:
            swagger_json = json.load(file)
        return swagger_json

    def __para_to_str(self, prop, required, name=""):
        name = name or prop["name"]
        ptype = prop["type"]
        title = prop.get("title", "")
        desc = prop.get("description", "")
        return f'{name}\t{ptype}\t{"Yes" if required else "No"}\t{title}\t{desc}'

    def _para_to_str(self, prop):
        required = prop.get("required", False)
        return self.__para_to_str(prop, required)

    def para_to_str(self, name, prop, prop_object_required):
        required = name in prop_object_required
        return self.__para_to_str(prop, required, name)

    def build_object_properties(self, node, prop_object_required, level: int = 0) -> str:
        """Recursively output properties of object and array[object] types

        Args:
            node (_type_): value of the child item
            prop_object_required (_type_): whether it's a required field
            level: current recursion depth
        """

        doc = ""

        def dive_into_object(node):
            """If it's an object type, recursively output its properties"""
            if node.get("type") == "object":
                sub_properties = node.get("properties", {})
                return self.build_object_properties(sub_properties, prop_object_required, level=level + 1)
            return ""

        if node.get("in", "") in ["query", "header", "formData"]:
            doc += f'{"	" * level}{self._para_to_str(node)}\n'
            doc += dive_into_object(node)
            return doc

        for name, prop in node.items():
            doc += f'{"	" * level}{self.para_to_str(name, prop, prop_object_required)}\n'
            doc += dive_into_object(prop)
            if prop["type"] == "array":
                items = prop.get("items", {})
                doc += dive_into_object(items)
        return doc

    def get_tags_mapping(self) -> dict:
        """Process tag and path mappings

        Returns:
            Dict: mapping of tag to path
        """
        swagger_data = self.get_swagger_json()
        paths = swagger_data["paths"]
        tags = {}

        for path, path_obj in paths.items():
            for method, method_obj in path_obj.items():
                for tag in method_obj["tags"]:
                    if tag not in tags:
                        tags[tag] = {}
                    if path not in tags[tag]:
                        tags[tag][path] = {}
                    tags[tag][path][method] = method_obj

        return tags

    def generate_ut(self, include_tags) -> bool:
        """Generate test case files"""
        tags = self.get_tags_mapping()
        for tag, paths in tags.items():
            if include_tags is None or tag in include_tags:
                self._generate_ut(tag, paths)
        return True

    def build_api_doc(self, node: dict, path: str, method: str) -> str:
        summary = node["summary"]

        doc = f"API Name: {summary}\nAPI Path: {path}\nMethod: {method.upper()}\n"
        doc += "\nRequest Parameters:\n"
        if "parameters" in node:
            parameters = node["parameters"]
            doc += "Path Parameters:\n"

            # param["in"]: path / formData / body / query / header
            for param in parameters:
                if param["in"] == "path":
                    doc += f'{param["name"]} \n'

            doc += "\nBody Parameters:\n"
            doc += "Name\tType\tRequired\tDefault Value\tRemarks\n"
            for param in parameters:
                if param["in"] == "body":
                    schema = param.get("schema", {})
                    prop_properties = schema.get("properties", {})
                    prop_required = schema.get("required", [])
                    doc += self.build_object_properties(prop_properties, prop_required)
                else:
                    doc += self.build_object_properties(param, [])

        # Display response data information
        doc += "\nResponse Data:\n"
        doc += "Name\tType\tRequired\tDefault Value\tRemarks\n"
        responses = node["responses"]
        response = responses.get("200", {})
        schema = response.get("schema", {})
        properties = schema.get("properties", {})
        required = schema.get("required", {})

        doc += self.build_object_properties(properties, required)
        doc += "\n"
        doc += "```py"

        return doc

    def _store(self, data, base, folder, fname):
        """Store data in a file."""
        file_path = self.get_file_path(Path(base) / folder, fname)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(data)

    def ask_gpt_and_save(self, question: str, tag: str, fname: str):
        """Generate questions and store both questions and answers"""
        messages = [self.icl_sample, question]
        result = self.gpt_msgs_to_code(messages=messages)

        self._store(question, self.questions_path, tag, f"{fname}.txt")
        self._store(result, self.ut_py_path, tag, f"{fname}.py")

    def _generate_ut(self, tag, paths):
        """Process the structure under a data path

        Args:
            tag (_type_): module name
            paths (_type_): Path Object
        """
        for path, path_obj in paths.items():
            for method, node in path_obj.items():
                summary = node["summary"]
                question = self.template_prefix
                question += self.build_api_doc(node, path, method)
                self.ask_gpt_and_save(question, tag, summary)

    def gpt_msgs_to_code(self, messages: list) -> str:
        """Choose based on different calling methods"""
        result = ''
        if self.chatgpt_method == "API":
            result = GPTAPI().ask_code(msgs=messages)

        return result

    def get_file_path(self, base: Path, fname: str):
        """Save different file paths

        Args:
            base (str): Path
            fname (str): File name
        """
        path = Path(base)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / fname
        return str(file_path)

```