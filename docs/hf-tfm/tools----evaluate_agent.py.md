# `.\tools\evaluate_agent.py`

```py
### Fake tools for test
# 定义一系列用于测试目的的假工具函数

def classifier(text, labels):
    # 返回一个文本分类的结果字符串
    return f"This is the classification of {text} along {labels}."

def translator(text, src_lang, tgt_lang):
    # 返回一个文本翻译的结果字符串
    return f"This is the translation of {text} from {src_lang} to {tgt_lang}."

def speaker(text):
    # 返回一个将文本朗读成声音的结果字符串
    return f"This is actually a sound reading {text}."

def transcriber(audio):
    # 如果输入不是声音，则抛出值错误异常
    if "sound" not in audio:
        raise ValueError(f"`audio` ({audio}) is not a sound.")
    # 返回从声音转录出的文本结果字符串
    return f"This is the transcribed text from {audio}."

def image_generator(prompt):
    # 返回一个表示给定提示的图像的结果字符串
    return f"This is actually an image representing {prompt}."

def image_captioner(image):
    # 如果输入不是图像，则抛出值错误异常
    if "image" not in image:
        raise ValueError(f"`image` ({image}) is not an image.")
    # 返回给定图像的描述结果字符串
    return f"This is a description of {image}."

def image_transformer(image, prompt):
    # 如果输入不是图像，则抛出值错误异常
    if "image" not in image:
        raise ValueError(f"`image` ({image}) is not an image.")
    # 返回根据给定提示对图像进行转换的结果字符串
    return f"This is a transformation of {image} according to {prompt}."

def question_answerer(text, question):
    # 返回对给定问题从文本中得到的答案结果字符串
    return f"This is the answer to {question} from {text}."

def image_qa(image, question):
    # 如果输入不是图像，则抛出值错误异常
    if "image" not in image:
        raise ValueError(f"`image` ({image}) is not an image.")
    # 返回对给定问题从图像中得到的答案结果字符串
    return f"This is the answer to {question} from {image}."

def text_downloader(url):
    # 返回从给定 URL 下载的内容结果字符串
    return f"This is the content of {url}."

def summarizer(text):
    # 返回给定文本的摘要结果字符串
    return f"This is a summary of {text}."

def video_generator(prompt, seconds=2):
    # 返回一个包含给定提示的视频结果字符串
    return f"A video of {prompt}"

def document_qa(image, question):
    # 返回对给定问题从文档图像中得到的答案结果字符串
    return f"This is the answer to {question} from the document {image}."

def image_segmenter(image, prompt):
    # 返回在给定图像中对给定提示的分割结果字符串
    return f"This is the mask of {prompt} in {image}"

TEST_TOOLS = {
    "text_classifier": classifier,
    "translator": translator,
    "text_reader": speaker,
    "summarizer": summarizer,
    "transcriber": transcriber,
    "image_generator": image_generator,
    "image_captioner": image_captioner,
    "image_transformer": image_transformer,
    "text_qa": question_answerer,
    "text_downloader": text_downloader,
    "image_qa": image_qa,
    "video_generator": video_generator,
    "document_qa": document_qa,
    "image_segmenter": image_segmenter,
}

class Problem:
    """
    占位符类，暂时没有定义任何内容
    """
    # 一个类，用于组织解决问题所需的所有信息，以便评估代理程序。
    
    Args:
        task (`str` 或 `list[str]`):
            要执行任务的一个或多个描述。如果是列表，则应包含相同任务的不同表达方式。
        inputs (`list[str]` 或 `dict[str, str]`):
            将提供给工具的输入。在这个测试环境中，只接受字符串作为值。当你想要指定每个输入的值时，请传递一个字典；或者直接传递期望的输入列表（在这种情况下，使用 `<<input_name>>` 作为值）。
        answer (`str` 或 `list[str]`):
            问题的理论答案（或可能的有效答案列表），作为代码。
    """
    
    # 初始化方法，用于设置实例的属性
    def __init__(self, task, inputs, answer):
        self.task = task      # 将传入的任务描述存储在实例的属性中
        self.inputs = inputs  # 将传入的输入数据存储在实例的属性中
        self.answer = answer  # 将传入的答案数据存储在实例的属性中
### 定义一个评估任务列表，包含多个问题实例

EVALUATION_TASKS = [
    # 定义一个问题实例，任务是判断给定的 `text`（西班牙语）是积极还是消极的
    Problem(
        task=[
            "Is the following `text` (in Spanish) positive or negative?",
            "Is the text in the variable `text` (in Spanish) positive or negative?",
            "Translate the following `text` from Spanish to English then tell me if its positive or negative.",
        ],
        inputs=["text"],
        # 答案是一个字符串表达式，调用了多个函数来进行文本处理和分类
        answer="""text_classifier(translator(text, src_lang="Spanish", tgt_lang="English"), labels=["positive", "negative"])""",
    ),

    # 定义一个问题实例，任务是描述给定的 `image` 包含的内容
    Problem(
        task=[
            "Tell me out loud what the `image` contains.",
            "Describe the following `image` out loud.",
            "Find what is in the picture stored in `image` then read it out loud.",
        ],
        inputs=["image"],
        # 答案是一个列表，包含了两种描述图片内容的方法
        answer=[
            "text_reader(image_captioner(image))",
            "text_reader(image_qa(image, question='What is in the image?'))",
        ],
    ),

    # 定义一个问题实例，任务是根据 `text_input` 生成图片，然后根据 `prompt` 进行变换
    Problem(
        task=[
            "Generate an image from the text given in `text_input`. Then transform it according to the text in `prompt`.",
            "Use the following `text_input` to generate an image, then transform it by using the text in `prompt`.",
        ],
        inputs=["text_input", "prompt"],
        # 答案是一个字符串，调用了多个函数来生成并变换图片
        answer="image_transformer(image_generator(text_input), prompt)",
    ),

    # 定义一个问题实例，任务是根据 `url` 下载内容，进行摘要并生成一张图片
    Problem(
        task=[
            "Download the content of `url`, summarize it then generate an image from its content.",
            "Use a summary of the web page at `url` to generate an image.",
            "Summarize the content of the web page at `url`, and use the result to generate an image.",
        ],
        inputs=["url"],
        # 答案是一个字符串，调用了多个函数来下载、摘要并生成图片
        answer="image_generator(summarizer(text_downloader(url)))",
    ),

    # 定义一个问题实例，任务是根据 `text` 和 `image` 进行图片的文本提示变换
    Problem(
        task=[
            "Transform the following `image` using the prompt in `text`. The prompt is in Spanish.",
            "Use the text prompt in `text` (in Spanish) to transform the following `image`.",
            "Translate the `text` from Spanish to English then use it to transform the picture in `image`.",
        ],
        inputs=["text", "image"],
        # 答案是一个字符串，调用了多个函数来进行图片的文本提示变换
        answer="image_transformer(image, translator(text, src_lang='Spanish', tgt_lang='English'))",
    ),

    # 定义一个问题实例，任务是根据 `url` 下载内容，进行摘要并朗读摘要
    Problem(
        task=[
            "Download the content of `url`, summarize it then read it out loud to me.",
            "Read me a summary of the web page at `url`.",
        ],
        inputs=["url"],
        # 答案是一个字符串，调用了多个函数来下载、摘要并朗读摘要
        answer="text_reader(summarizer(text_downloader(url)))",
    ),

    # 定义一个问题实例，任务是根据 `text_input` 生成一张图片
    Problem(
        task=[
            "Generate an image from the text given in `text_input`.",
        ],
        inputs=["text_input"],
        # 答案是一个字符串，调用了一个函数来生成图片
        answer="image_generator(text_input)",
    ),
]
    Problem(
        task=[
            "Replace the beaver in the `image` by the `prompt`.",
            "Transform the `image` so that it contains the `prompt`.",
            "Use `prompt` to transform this `image`.",
        ],
        inputs=["image", "prompt"],
        answer="image_transformer(image, prompt)",
    ),
    Problem(
        task=[
            "Provide me the summary of the `text`, then read it to me before transcribing it and translating it in French.",
            "Summarize `text`, read it out loud then transcribe the audio and translate it in French.",
            "Read me a summary of the `text` out loud. Transcribe this and translate it in French.",
        ],
        inputs=["text"],
        answer="translator(transcriber(text_reader(summarizer(text))), src_lang='English', tgt_lang='French')",
    ),
    Problem(
        task=["Generate a video of the `prompt`", "Animate a `prompt`", "Make me a short video using `prompt`."],
        inputs={"prompt": "A lobster swimming"},
        answer="video_generator('A lobster swimming')",
    ),
    Problem(
        task=[
            "Download the following file `url`, summarize it in a few words and generate a video from it."
            "Fetch the file at this `url`, summarize it, and create an animation out of it."
        ],
        inputs=["url"],
        answer="video_generator(summarizer(text_downloader(url)))",
    ),



    Problem(
        task=[
            "Replace the beaver in the `image` by the `prompt`.",
            "Transform the `image` so that it contains the `prompt`.",
            "Use `prompt` to transform this `image`.",
        ],
        inputs=["image", "prompt"],
        answer="image_transformer(image, prompt)",
    ),
    # 创建一个 Problem 对象，包含了替换图片中的某物体和转换图片的任务，输入为图片和提示，答案为调用 image_transformer 函数
    Problem(
        task=[
            "Provide me the summary of the `text`, then read it to me before transcribing it and translating it in French.",
            "Summarize `text`, read it out loud then transcribe the audio and translate it in French.",
            "Read me a summary of the `text` out loud. Transcribe this and translate it in French.",
        ],
        inputs=["text"],
        answer="translator(transcriber(text_reader(summarizer(text))), src_lang='English', tgt_lang='French')",
    ),
    # 创建一个 Problem 对象，包含了对文本进行摘要、朗读、转录和翻译任务，输入为文本，答案为复合函数调用
    Problem(
        task=["Generate a video of the `prompt`", "Animate a `prompt`", "Make me a short video using `prompt`."],
        inputs={"prompt": "A lobster swimming"},
        answer="video_generator('A lobster swimming')",
    ),
    # 创建一个 Problem 对象，包含了根据提示生成视频的任务，输入为提示，答案为调用 video_generator 函数
    Problem(
        task=[
            "Download the following file `url`, summarize it in a few words and generate a video from it."
            "Fetch the file at this `url`, summarize it, and create an animation out of it."
        ],
        inputs=["url"],
        answer="video_generator(summarizer(text_downloader(url)))",
    ),
    # 创建一个 Problem 对象，包含了从 URL 下载文件、摘要并生成视频的任务，输入为 URL，答案为调用 video_generator 函数
EVALUATION_CHATS = [
    [  # 开始一个列表，包含多个问题对象
        Problem(  # 创建第一个问题对象
            task=[  # 问题描述列表
                "Translate the following `text` from Spanish to English.",  # 翻译从西班牙语到英语的文本
                "Translate the following `text` from Spanish to English.",  # 同上，重复描述
            ],
            inputs=["text"],  # 输入参数为一个文本字符串
            answer="translated_text=translator(text, src_lang='Spanish', tgt_lang='English')",  # 答案是调用翻译器函数进行翻译
        ),
        Problem(  # 创建第二个问题对象
            task=[  # 问题描述列表
                "Is it positive or negative?",  # 判断文本情感是积极还是消极
                "Tell me if its positive or negative.",  # 同上，重复描述
            ],
            inputs=[],  # 无输入参数
            answer="text_classifier(translated_text, labels=['positive', 'negative'])",  # 使用文本分类器判断文本情感
        ),
    ],
    [  # 开始第二个问题列表
        Problem(  # 创建第一个问题对象
            task=[  # 问题描述列表
                "What does this `image` contain?",  # 描述图像包含的内容
                "Describe the following `image`.",  # 描述以下的图像
                "Find what is in the picture stored in `image`",  # 找出存储在 `image` 中图片的内容
            ],
            inputs=["image"],  # 输入参数为一个图像
            answer=[  # 答案是一个包含两个动作的列表
                "description=image_captioner(image)",  # 生成图像描述
                "description=image_qa(image, question='What is in the image?')",  # 使用图像问答系统找出图像中的内容
            ],
        ),
        Problem(  # 创建第二个问题对象
            task=[  # 问题描述列表
                "Now, read the description out loud.",  # 现在大声朗读描述
                "Great! Can you read it out loud?",  # 太棒了！你能大声朗读吗？
                "Read it out loud.",  # 大声朗读
            ],
            inputs=[],  # 无输入参数
            answer=["audio=text_reader(description)", "audio=text_reader(description)"],  # 生成描述的语音输出
        ),
    ],
    [  # 开始第三个问题列表
        Problem(  # 创建第一个问题对象
            task=[  # 问题描述列表
                "Generate an image from the text given in `text_input`.",  # 使用 `text_input` 中的文本生成图像
                "Use the following `text_input` to generate an image",  # 使用以下 `text_input` 生成图像
            ],
            inputs=["text_input"],  # 输入参数为一个文本输入
            answer="image = image_generator(text_input)",  # 生成图像的操作
        ),
        Problem(  # 创建第二个问题对象
            task=[  # 问题描述列表
                "Transform it according to the text in `prompt`.",  # 根据 `prompt` 中的文本对图像进行转换
                "Transform it by using the text in `prompt`.",  # 使用 `prompt` 中的文本进行转换
            ],
            inputs=["prompt"],  # 输入参数为一个提示文本
            answer="image_transformer(image, prompt)",  # 对图像进行转换的操作
        ),
    ],
    [  # 开始第四个问题列表
        Problem(  # 创建第一个问题对象
            task=[  # 问题描述列表
                "Download the content of `url` and summarize it.",  # 下载 `url` 的内容并进行摘要
                "Summarize the content of the web page at `url`.",  # 总结位于 `url` 的网页内容
            ],
            inputs=["url"],  # 输入参数为一个 URL
            answer="summary = summarizer(text_downloader(url))",  # 使用文本下载器下载内容并进行摘要生成
        ),
        Problem(  # 创建第二个问题对象
            task=[  # 问题描述列表
                "Generate an image from its content.",  # 从其内容生成一幅图像
                "Use the previous result to generate an image.",  # 使用上述结果生成图像
            ],
            inputs=[],  # 无输入参数
            answer="image_generator(summary)",  # 根据摘要内容生成图像
        ),
    ],
]
    [
        # 第一个问题组
        Problem(
            # 任务描述：将这段西班牙文`text`翻译成英文。
            task=[
                "Translate this Spanish `text` in English.",
                "Translate the `text` from Spanish to English.",
            ],
            # 输入参数：text，需要翻译的文本
            inputs=["text"],
            # 答案：调用translator函数进行翻译，从西班牙语到英语
            answer="translated_text = translator(text, src_lang='Spanish', tgt_lang='English')",
        ),
        Problem(
            # 任务描述：使用翻译后的`text`来转换以下的`image`。
            task=[
                "Transform the following `image` using the translated `text`.",
                "Use the previous result to transform the following `image`.",
            ],
            # 输入参数：image，需要进行转换的图像；translated_text，已翻译的文本
            inputs=["image"],
            # 答案：调用image_transformer函数，使用翻译后的文本来转换图像
            answer="image_transformer(image, translated_text)",
        ),
    ],
    [
        # 第二个问题组
        Problem(
            # 任务描述：下载`url`的内容。
            task=["Download the content of `url`.", "Get me the text on the web page `url`."],
            # 输入参数：url，需要下载内容的网址
            inputs=["url"],
            # 答案：调用text_downloader函数下载网页内容
            answer="text = text_downloader(url)",
        ),
        Problem(
            # 任务描述：对文本进行总结。
            task=["Summarize this text.", "Summarize this text."],
            # 输入参数：无（使用前面下载的文本）
            inputs=[],
            # 答案：调用summarizer函数对文本进行总结
            answer="summary = summarizer(text)",
        ),
        Problem(
            # 任务描述：朗读给我听。
            task=["Read it out loud to me.", "Read me the previous result."],
            # 输入参数：无（使用前面生成的总结文本）
            inputs=[],
            # 答案：调用text_reader函数朗读总结文本
            answer="text_reader(summary)",
        ),
    ],
    [
        # 第三个问题组
        Problem(
            # 任务描述：根据给定的`text_input`生成一张图像。
            task=["Generate an image from the text given in `text_input`."],
            # 输入参数：text_input，用于生成图像的文本输入
            inputs=["text_input"],
            # 答案：调用image_generator函数生成图像
            answer="image_generator(text_input)",
        ),
    ],
    [
        # 第四个问题组
        Problem(
            # 任务描述：用`prompt`替换`image`中的海狸。
            task=[
                "Replace the beaver in the `image` by the `prompt`.",
                "Transform the `image` so that it contains the `prompt`.",
                "Use `prompt` to transform this `image`.",
            ],
            # 输入参数：image，需要进行转换的图像；prompt，用于替换的提示
            inputs=["image", "prompt"],
            # 答案：调用image_transformer函数，使用prompt来转换图像
            answer="image_transformer(image, prompt)",
        ),
    ],
    [
        # 第五个问题组
        Problem(
            # 任务描述：提供`text`的摘要。
            task=["Provide me the summary of the `text`.", "Summarize `text`."],
            # 输入参数：text，需要进行总结的文本
            inputs=["text"],
            # 答案：调用summarizer函数对文本进行总结
            answer="summary = summarizer(text)",
        ),
        Problem(
            # 任务描述：将摘要朗读给我听。
            task=["Read this summary to me.", "Read it out loud."],
            # 输入参数：无（使用前面生成的总结文本）
            inputs=[],
            # 答案：调用text_reader函数朗读总结文本
            answer="audio = text_reader(summarizer(text))",
        ),
        Problem(
            # 任务描述：将上一结果转录成文本。
            task=["Transcribing the previous result back in text.", "Transcribe the audio."],
            # 输入参数：无（使用前面生成的音频）
            inputs=[],
            # 答案：调用transcriber函数将音频转录成文本
            answer="text = transcriber(audio)",
        ),
        Problem(
            # 任务描述：将上一结果翻译成法语。
            task=["Translating the last result in French.", "Translate this in French."],
            # 输入参数：无（使用前面生成的文本）
            inputs=[],
            # 答案：调用translator函数将文本从英语翻译成法语
            answer="translator(text, src_lang='English', tgt_lang='French')",
        ),
    ],
    [
        # 第六个问题组
        Problem(
            # 任务描述：根据`prompt`生成一个视频。
            task=[
                "Generate a video of the `prompt`",
                "Animate a `prompt`",
                "Make me a short video using `prompt`.",
            ],
            # 输入参数：prompt，用于生成视频的提示文本
            inputs={"prompt": "A lobster swimming"},
            # 答案：调用video_generator函数生成视频
            answer="video_generator('A lobster swimming')",
        ),
    ],
    [
        # 创建一个包含两个问题的列表，每个问题包括任务描述、输入要求和答案方法
        Problem(
            # 第一个问题的任务描述
            task=[
                "Download the content of `url` and summarize it.",
                "Summarize the content of the web page at `url`."
            ],
            # 第一个问题的输入要求，需要一个参数 `url`
            inputs=["url"],
            # 第一个问题的答案方法，使用 `text_downloader` 下载 `url` 的内容，然后使用 `summarizer` 进行总结
            answer="summary = summarizer(text_downloader(url))"
        ),
        # 第二个问题的问题描述
        Problem(
            task=["generate a video from it.", "Create an animation from the last result."],
            # 第二个问题没有输入要求，所以是一个空列表
            inputs=[],
            # 第二个问题的答案方法，使用上一个问题中生成的 `summary` 来生成视频
            answer="video_generator(summary)"
        ),
    ],
# 定义函数，用于获取理论工具集和代码中实际使用的工具集的比较结果
def get_theoretical_tools(agent_answer, theoretical_answer, code_answer):
    # 如果理论答案不是列表，则返回代码中的测试工具集合
    if not isinstance(theoretical_answer, list):
        return {name for name in TEST_TOOLS if name in code_answer}

    # 如果代理答案是字典类型，则逐个比较理论答案和代码答案
    if isinstance(agent_answer, dict):
        for one_answer, one_code in zip(theoretical_answer, code_answer):
            # 如果代理答案的值包含在理论答案中，则返回在代码中使用的测试工具集合
            if one_answer in agent_answer.values():
                return {name for name in TEST_TOOLS if name in one_code}

    # 逐个比较理论答案和代码答案
    for one_answer, one_code in zip(theoretical_answer, code_answer):
        # 如果代理答案等于理论答案之一，则返回在代码中使用的测试工具集合
        if agent_answer == one_answer:
            return {name for name in TEST_TOOLS if name in one_code}

    # 返回代码中使用的第一个测试工具集合
    return {name for name in TEST_TOOLS if name in code_answer[0]}


# 定义函数，评估给定的代码
def evaluate_code(code, inputs=None, state=None, verbose=False, return_interpretor_error=False):
    # 复制基本的 Python 工具集到当前工具集中
    tools = BASE_PYTHON_TOOLS.copy()
    
    # 遍历测试工具集合，将代码中使用的工具添加到当前工具集中
    for name, tool in TEST_TOOLS.items():
        if name not in code:
            continue
        tools[name] = tool

    # 如果输入是字典类型，则复制一份输入
    if isinstance(inputs, dict):
        inputs = inputs.copy()
    # 如果输入不为空，则将每个输入映射到特定的占位符格式
    elif inputs is not None:
        inputs = {inp: f"<<{inp}>>" for inp in inputs}

    # 如果状态不为空，则更新状态信息，否则使用输入作为状态信息
    if state is not None:
        state.update(inputs)
    else:
        state = inputs

    try:
        # 尝试评估代码，使用当前工具集和状态
        return evaluate(code, tools, state)
    except InterpretorError as e:
        # 如果发生解释器错误，则返回错误消息字符串
        return str(e)
    except Exception as e:
        # 如果发生其他异常，根据 verbose 参数决定是否打印异常信息，并返回 None
        if verbose:
            print(e)
        return None


# 定义函数，评分给定的代码答案
def score_code(agent_answer, theoretical_answer, verbose: bool = False):
    # 如果 verbose 为 True，则打印代理答案和理论答案
    if verbose:
        print(agent_answer, theoretical_answer)
    
    # 如果理论答案不是列表，则将其转换为列表形式
    theoretical_answer = theoretical_answer if isinstance(theoretical_answer, list) else [theoretical_answer]

    # 如果代理答案包含在理论答案中，则返回完美匹配的分数 1.0
    if agent_answer in theoretical_answer:
        if verbose:
            print("Perfect!")
        return 1
    # 如果代理答案是字典类型，并且其值在理论答案中，则返回部分匹配的分数 0.75
    elif isinstance(agent_answer, dict) and any(v in theoretical_answer for v in agent_answer.values()):
        if verbose:
            print("Almost perfect, result in state!")
        return 0.75
    # 否则，返回未完全匹配的分数 0.3
    else:
        if verbose:
            print("Result is not the right one but code executed.")
        return 0.3


# 定义函数，评估单个结果的解释
def evaluate_one_result(explanation, code, agent_answer, theoretical_answer, answer, verbose=False):
    # 提取解释中使用的工具集合
    tools_in_explanation = {name for name in TEST_TOOLS if f"`{name}`" in explanation}
    
    # 获取理论工具集和代码实际使用的工具集的比较结果
    theoretical_tools = get_theoretical_tools(agent_answer, theoretical_answer, answer)
    
    # 如果解释中使用的工具集与理论工具集完全匹配，则设置工具选择分数为 1.0，无工具选择错误
    if tools_in_explanation == theoretical_tools:
        tool_selection_score = 1.0
        tool_selection_errors = None
    else:
        # 否则，计算缺失的工具和意外的工具数量，并计算工具选择分数
        missing_tools = len(theoretical_tools - tools_in_explanation)
        unexpected_tools = len(tools_in_explanation - theoretical_tools)
        tool_selection_score = max(0, 1.0 - 0.25 * missing_tools - 0.25 * unexpected_tools)

        # 设置工具选择错误信息
        tool_selection_errors = {
            "selected_tools": tools_in_explanation,
            "theoretical_tools": theoretical_tools,
        }

    # 提取代码中使用的工具集合
    tools_in_code = {name for name in TEST_TOOLS if name in code}
    # 如果代码中使用的工具与理论工具相匹配
    if tools_in_code == theoretical_tools:
        # 工具使用得分为满分 1.0
        tool_used_score = 1.0
        # 错误信息为空
        tool_used_errors = None
    else:
        # 计算缺失的工具数量
        missing_tools = len(theoretical_tools - tools_in_code)
        # 计算多余的工具数量
        unexpected_tools = len(tools_in_code - theoretical_tools)
        # 计算工具使用得分，考虑缺失工具和多余工具的惩罚
        tool_used_score = max(0, 1.0 - 0.25 * missing_tools - 0.25 * unexpected_tools)

        # 生成工具使用错误信息，包含选中的工具和理论上应有的工具
        tool_used_errors = {
            "selected_tools": tools_in_explanation,
            "theoretical_tools": theoretical_tools,
        }

    # 对代码进行评分，返回评分结果
    score = score_code(agent_answer, theoretical_answer, verbose=verbose)
    # 如果评分小于 1.0
    if score < 1.0:
        # 生成代码错误信息，包含生成的代码、评估结果和理论答案
        code_errors = {
            "code_produced": code,
            "evaluation": agent_answer,
            "theoretical_answer": theoretical_answer,
        }
    else:
        # 如果评分为满分，错误信息为空
        code_errors = None

    # 返回工具选择得分、工具使用得分、代码评分以及相应的错误信息
    return (tool_selection_score, tool_used_score, score), (tool_selection_errors, tool_used_errors, code_errors)
# 对代理工具进行一致性检查，确保包含所有必需的测试工具
agent_tools = set(agent.toolbox.keys())
if agent_tools != set(TEST_TOOLS):
    # 计算缺失的工具和多余的工具，并引发值错误
    missing_tools = set(TEST_TOOLS) - agent_tools
    unexpected_tools = agent_tools - set(TEST_TOOLS)
    raise ValueError(
        f"Fix the test tools in the evaluate_agent module. Tools missing: {missing_tools}. Extra tools: {unexpected_tools}."
    )

# 初始化评估任务列表和其对应的索引列表
eval_tasks = []
eval_idx = []
for idx, pb in enumerate(EVALUATION_TASKS):
    if isinstance(pb.task, list):
        # 将任务列表展开，并更新索引列表
        eval_tasks.extend(pb.task)
        eval_idx.extend([idx] * len(pb.task))
    else:
        # 添加单个任务及其索引
        eval_tasks.append(pb.task)
        eval_idx.append(idx)

# 初始化评分变量
tool_selection_score = 0
tool_used_score = 0
code_score = 0

# 如果需要返回错误信息，则初始化错误字典
if return_errors:
    tool_selection_errors = {}
    tool_used_errors = {}
    code_errors = {}

# 分批次处理评估任务
for start_idx in range(0, len(eval_tasks), batch_size):
    end_idx = min(start_idx + batch_size, len(eval_tasks))
    batch_tasks = eval_tasks[start_idx:end_idx]

    # 根据任务生成相应的提示语句
    prompts = [agent.format_prompt(task) for task in batch_tasks]
    # 代理执行生成代码任务，停止条件为 "Task:"
    results = agent.generate_many(prompts, stop=["Task:"])

    # 遍历每个任务结果
    for idx, result in enumerate(results):
        # 获取当前任务的问题和答案
        problem = EVALUATION_TASKS[eval_idx[start_idx + idx]]
        if verbose:
            # 如果启用了详细输出，打印任务内容
            print(f"====Task {start_idx + idx}====\n{batch_tasks[idx]}\n")

        # 清理生成的代码并准备执行
        explanation, code = agent.clean_code_for_run(result)

        # 评估代理的答案和生成的代码答案
        agent_answer = evaluate_code(code, problem.inputs, verbose=verbose)
        if isinstance(problem.answer, list):
            theoretical_answer = [evaluate_code(answer, problem.inputs) for answer in problem.answer]
        else:
            theoretical_answer = evaluate_code(problem.answer, problem.inputs)

        # 调用评估函数，获取得分和可能的错误
        scores, errors = evaluate_one_result(
            explanation, code, agent_answer, theoretical_answer, problem.answer, verbose=verbose
        )

        # 累加各项得分
        tool_selection_score += scores[0]
        tool_used_score += scores[1]
        code_score += scores[2]

        # 如果需要记录错误信息，则将其添加到相应的错误字典中
        if return_errors:
            if errors[0] is not None:
                tool_selection_errors[batch_tasks[idx]] = errors[0]
            if errors[1] is not None:
                tool_used_errors[batch_tasks[idx]] = errors[1]
            if errors[2] is not None:
                code_errors[batch_tasks[idx]] = errors[2]
    # 计算并构建评分字典，包括工具选择、工具使用和代码评分，每项分数都是相对于评估任务数量的百分比
    scores = {
        "tool selection score": 100 * (tool_selection_score / len(eval_tasks)),
        "tool used score": 100 * (tool_used_score / len(eval_tasks)),
        "code score": 100 * (code_score / len(eval_tasks)),
    }

    # 如果需要返回错误信息，则返回评分字典和各类错误列表；否则，仅返回评分字典
    if return_errors:
        return scores, tool_selection_errors, tool_used_errors, code_errors
    else:
        return scores
# 对给定的代理程序进行评估，检查其是否具备正确的工具集
def evaluate_chat_agent(agent, verbose=False, return_errors=False):
    """
    Evaluates a new agent on all `EVALUATION_CHATS`.

    Example:

    ```
    agent = NewOpenAiAgent(model="text-davinci-003", api_key=your_api_key)
    bads = new_evaluate_agent(agent)
    for bad in bads:
        print(bad)
    ```
    """
    # 检查代理程序的工具集合是否与预期的测试工具集合一致
    agent_tools = set(agent.toolbox.keys())
    if agent_tools != set(TEST_TOOLS):
        # 计算缺失的工具和多余的工具
        missing_tools = set(TEST_TOOLS) - agent_tools
        unexpected_tools = agent_tools - set(TEST_TOOLS)
        # 抛出数值错误，指示需要修复评估模块中的测试工具
        raise ValueError(
            f"Fix the test tools in the evaluate_agent module. Tools mising: {missing_tools}. Extra tools: {unexpected_tools}."
        )

    # 初始化评分变量
    tool_selection_score = 0
    tool_used_score = 0
    code_score = 0
    total_steps = 0

    # 如果需要返回错误信息，初始化错误字典
    if return_errors:
        tool_selection_errors = {}
        tool_used_errors = {}
        code_errors = {}
    # 遍历评估对话中的每个问题
    for chat_problem in EVALUATION_CHATS:
        # 检查第一个任务是否为字符串，若是则标记为已解决的问题列表
        if isinstance(chat_problem[0].task, str):
            resolved_problems = [chat_problem]
        else:
            # 否则，根据每个任务生成一个新的Problem对象列表
            resolved_problems = [
                [Problem(task=pb.task[i], inputs=pb.inputs, answer=pb.answer) for pb in chat_problem]
                for i in range(len(chat_problem[0].task))
            ]
        
        # 遍历解决的问题列表
        for problem in resolved_problems:
            # 准备Agent进行新对话的准备工作
            agent.prepare_for_new_chat()
            agent_state = {}  # 重置Agent的状态
            # 根据第一个答案是否为列表，确定理论状态的初始化方式
            theoretical_state = (
                [{} for _ in range(len(problem[0].answer))] if isinstance(problem[0].answer, list) else {}
            )
            
            # 遍历每个问题中的每个步骤
            for step, step_problem in enumerate(problem):
                # 如果设定了详细输出模式，打印当前任务描述
                if verbose:
                    print(step_problem.task)
                
                total_steps += 1  # 总步数加一
                # 格式化Agent的提示信息，准备生成一条对话
                prompt = agent.format_prompt(step_problem.task, chat_mode=True)
                # 生成Agent的回答，同时设定停止词以防止过长输出
                result = agent.generate_one(prompt, stop=["Human:", "====="])
                # 将生成的对话历史记录保存到Agent的聊天历史中
                agent.chat_history = prompt + result + "\n"

                # 清理生成的代码，获取解释和代码本身
                explanation, code = clean_code_for_chat(result)

                # 如果设定了详细输出模式，打印Agent生成的解释和代码
                if verbose:
                    print(f"==Explanation from the agent==\n{explanation}")
                    print(f"\n==Code generated by the agent==\n{code}")

                # 评估Agent的回答和生成的代码
                agent_answer = evaluate_code(code, step_problem.inputs, state=agent_state, verbose=verbose)

                answer = step_problem.answer
                if isinstance(answer, list):
                    # 若答案为列表，计算每个理论答案对应的状态
                    theoretical_answer = [
                        evaluate_code(a, step_problem.inputs, state=state)
                        for a, state in zip(answer, theoretical_state)
                    ]
                else:
                    # 否则，直接计算理论答案
                    theoretical_answer = evaluate_code(answer, step_problem.inputs, state=theoretical_state)

                # 评估一次结果，获取分数和可能的错误信息
                scores, errors = evaluate_one_result(
                    explanation, code, agent_answer, theoretical_answer, answer, verbose=verbose
                )

                # 累加工具选择得分、工具使用得分和代码得分
                tool_selection_score += scores[0]
                tool_used_score += scores[1]
                code_score += scores[2]

                # 如果需要返回错误信息，记录工具选择、工具使用和代码错误
                if return_errors:
                    if errors[0] is not None:
                        tool_selection_errors[step_problem.task] = errors[0]
                    if errors[1] is not None:
                        tool_used_errors[step_problem.task] = errors[1]
                    if errors[2] is not None:
                        code_errors[step_problem.task] = errors[2]

    # 计算并返回总体得分，根据需要返回错误信息
    scores = {
        "tool selection score": 100 * (tool_selection_score / total_steps),
        "tool used score": 100 * (tool_used_score / total_steps),
        "code score": 100 * (code_score / total_steps),
    }

    if return_errors:
        return scores, tool_selection_errors, tool_used_errors, code_errors
    else:
        return scores
```