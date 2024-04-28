# `.\transformers\tools\evaluate_agent.py`

```
#!/usr/bin/env python
# coding=utf-8

# 版权声明
# 版权声明说明了对代码的使用限制和要求
# 在这里给出了 Apache License, Version 2.0 的说明和获取方式

# 导入相关的模块和函数
from .agents import BASE_PYTHON_TOOLS, clean_code_for_chat, clean_code_for_run
from .python_interpreter import InterpretorError, evaluate

# 定义一些用于测试的虚假工具函数
def classifier(text, labels):
    return f"This is the classification of {text} along {labels}."
# ...

# 用于测试的虚假工具函数的集合
TEST_TOOLS = {
    "text_classifier": classifier,
    "translator": translator,
    # ...
}

# 定义名为 Problem 的类
# 这是一个占位符类，没有具体的实现
class Problem:
    """
    这是一个占位符类
    """
    # 重新组织所有信息来解决问题，以便评估代理
    # 构造函数，接受任务描述、输入和答案作为参数
    class Problem:
        
        def __init__(self, task, inputs, answer):
            # 初始化任务描述
            self.task = task
            # 初始化输入数据
            self.inputs = inputs
            # 初始化答案数据
            self.answer = answer
### 代理将要评估的问题列表。
EVALUATION_TASKS = [
    # 创建一个 Problem 对象
    Problem(
        # 指定任务描述列表
        task=[
            "Is the following `text` (in Spanish) positive or negative?",
            "Is the text in the variable `text` (in Spanish) positive or negative?",
            "Translate the following `text` from Spanish to English then tell me if its positive or negative.",
        ],
        # 指定输入参数列表
        inputs=["text"],
        # 指定答案
        answer="""text_classifier(translator(text, src_lang="Spanish", tgt_lang="English"), labels=["positive", "negative"])""",
    ),
    Problem(
        task=[
            "Tell me out loud what the `image` contains.",
            "Describe the following `image` out loud.",
            "Find what is in the picture stored in `image` then read it out loud.",
        ],
        inputs=["image"],
        # 指定答案列表
        answer=[
            "text_reader(image_captioner(image))",
            "text_reader(image_qa(image, question='What is in the image?'))",
        ],
    ),
    Problem(
        task=[
            "Generate an image from the text given in `text_input`. Then transform it according to the text in `prompt`.",
            "Use the following `text_input` to generate an image, then transform it by using the text in `prompt`.",
        ],
        inputs=["text_input", "prompt"],
        answer="image_transformer(image_generator(text_input), prompt)",
    ),
    Problem(
        task=[
            "Download the content of `url`, summarize it then generate an image from its content.",
            "Use a summary of the web page at `url` to generate an image.",
            "Summarize the content of the web page at `url`, and use the result to generate an image.",
        ],
        inputs=["url"],
        answer="image_generator(summarizer(text_downloader(url)))",
    ),
    Problem(
        task=[
            "Transform the following `image` using the prompt in `text`. The prompt is in Spanish.",
            "Use the text prompt in `text` (in Spanish) to transform the following `image`.",
            "Translate the `text` from Spanish to English then use it to transform the picture in `image`.",
        ],
        inputs=["text", "image"],
        answer="image_transformer(image, translator(text, src_lang='Spanish', tgt_lang='English'))",
    ),
    Problem(
        task=[
            "Download the content of `url`, summarize it then read it out loud to me.",
            "Read me a summary of the web page at `url`.",
        ],
        inputs=["url"],
        answer="text_reader(summarizer(text_downloader(url)))",
    ),
    Problem(
        task=[
            "Generate an image from the text given in `text_input`.",
        ],
        inputs=["text_input"],
        answer="image_generator(text_input)",
    ),
    # 定义一个问题对象，包括三个任务描述和输入参数
    Problem(
        task=[
            "Replace the beaver in the `image` by the `prompt`.",
            "Transform the `image` so that it contains the `prompt`.",
            "Use `prompt` to transform this `image`.",
        ],
        inputs=["image", "prompt"],
        answer="image_transformer(image, prompt)",
    ),
    # 定义另一个问题对象，包括三个任务描述和输入参数
    Problem(
        task=[
            "Provide me the summary of the `text`, then read it to me before transcribing it and translating it in French.",
            "Summarize `text`, read it out loud then transcribe the audio and translate it in French.",
            "Read me a summary of the `text` out loud. Transcribe this and translate it in French.",
        ],
        inputs=["text"],
        answer="translator(transcriber(text_reader(summarizer(text))), src_lang='English', tgt_lang='French')",
    ),
    # 定义另一个问题对象，包括三个任务描述和输入参数
    Problem(
        task=["Generate a video of the `prompt`", "Animate a `prompt`", "Make me a short video using `prompt`."],
        inputs={"prompt": "A lobster swimming"},
        answer="video_generator('A lobster swimming')",
    ),
    # 定义最后一个问题对象，包括两个任务描述和输入参数
    Problem(
        task=[
            "Download the following file `url`, summarize it in a few words and generate a video from it."
            "Fetch the file at this `url`, summarize it, and create an animation out of it."
        ],
        inputs=["url"],
        answer="video_generator(summarizer(text_downloader(url)))",
    ),
# 定义一个名为EVALUATION_CHATS的列表，其中包含了多组问题和答案
EVALUATION_CHATS = [
    # 第一组问题和答案
    [
        # 定义一个名为Problem的对象，其中包含要翻译的文本和翻译后的结果
        Problem(
            task=[
                "Translate the following `text` from Spanish to English.",
                "Translate the following `text` from Spanish to English.",
            ],
            inputs=["text"],
            answer="translated_text=translator(text, src_lang='Spanish', tgt_lang='English')",
        ),
        # 定义一个名为Problem的对象，用于进行情感分类
        Problem(
            task=[
                "Is it positive or negative?",
                "Tell me if its positive or negative.",
            ],
            inputs=[],
            answer="text_classifier(translated_text, labels=['positive', 'negative'])",
        ),
    ],
    # 第二组问题和答案
    [
        # 定义一个名为Problem的对象，用于描述给定图像
        Problem(
            task=[
                "What does this `image` contain?",
                "Describe the following `image`.",
                "Find what is in the picture stored in `image`",
            ],
            inputs=["image"],
            answer=[
                "description=image_captioner(image)",
                "description=image_qa(image, question='What is in the image?')",
            ],
        ),
        # 定义一个名为Problem的对象，用于将描述文本转化为音频
        Problem(
            task=["Now, read the description out loud.", "Great! Can you read it out loud?", "Read it out loud."],
            inputs=[],
            answer=["audio=text_reader(description)", "audio=text_reader(description)"],
        ),
    ],
    # 第三组问题和答案
    [
        # 定义一个名为Problem的对象，用于从文本生成图像
        Problem(
            task=[
                "Generate an image from the text given in `text_input`.",
                "Use the following `text_input` to generate an image",
            ],
            inputs=["text_input"],
            answer="image = image_generator(text_input)",
        ),
        # 定义一个名为Problem的对象，用于根据提示文本对图像进行变换
        Problem(
            task=[
                "Transform it according to the text in `prompt`.",
                "Transform it by using the text in `prompt`.",
            ],
            inputs=["prompt"],
            answer="image_transformer(image, prompt)",
        ),
    ],
    # 第四组问题和答案
    [
        # 定义一个名为Problem的对象，用于下载并总结网页内容
        Problem(
            task=[
                "Download the content of `url` and summarize it.",
                "Summarize the content of the web page at `url`.",
            ],
            inputs=["url"],
            answer="summary = summarizer(text_downloader(url))",
        ),
        # 定义一个名为Problem的对象，用于从总结内容生成图像
        Problem(
            task=[
                "Generate an image from its content.",
                "Use the previous result to generate an image.",
            ],
            inputs=[],
            answer="image_generator(summary)",
        ),
    ],
]
    [
        Problem(
            task=[
                "Translate this Spanish `text` in English.", 
                "Translate the `text` from Spanish to English.",
            ],
            inputs=["text"],
            answer="translated_text = translator(text, src_lang='Spanish', tgt_lang='English')",
        ),
        # 定义一个问题，将传入的西班牙文文本`text`翻译成英文
        # 输入参数包含文本`text`
        # 调用翻译函数translator，将`text`从西班牙文翻译为英文，并存储在变量translated_text中
    
        Problem(
            task=[
                "Transform the following `image` using the translated `text`.",
                "Use the previous result to transform the following `image`.",
            ],
            inputs=["image"],
            answer="image_transformer(image, translated_text)",
        ),
        # 定义一个问题，利用之前翻译后的文本`translated_text`来转换传入的图像`image`
        # 输入参数包含图像`image`
        # 调用转换图像函数image_transformer，使用`translated_text`对`image`进行转换
    ],
    
    [
        Problem(
            task=["Download the content of `url`.", "Get me the text on the website `url`."],
            inputs=["url"],
            answer="text = text_downloader(url)",
        ),
        # 定义一个问题，下载指定URL地址`url`的内容
        # 输入参数包含URL地址`url`
        # 调用文本下载函数text_downloader，将URL地址`url`的内容下载并存储在变量text中
    
        Problem(
            task=["Summarize this text.", "Summarize this text."],
            inputs=[],
            answer="summary = summarizer(text)",
        ),
        # 定义一个问题，对所给文本进行总结
        # 无输入参数
        # 调用总结函数summarizer，对文本进行总结并存储在变量summary中
    
        Problem(
            task=["Read it out loud to me.", "Read me the previous result."],
            inputs=[],
            answer="text_reader(summary)",
        ),
        # 定义一个问题，将前一个结果阅读出来
        # 无输入参数
        # 调用文本阅读函数text_reader，将summary的内容读出
    ],
    
    [
        Problem(
            task=[
                "Generate an image from the text given in `text_input`.",
            ],
            inputs=["text_input"],
            answer="image_generator(text_input)",
        ),
        # 定义一个问题，根据给定的`text_input`文本生成一个图像
        # 输入参数包含文本输入`text_input`
        # 调用图像生成器函数image_generator，根据`text_input`生成图像
    ],
    
    [
        Problem(
            task=[
                "Replace the beaver in the `image` by the `prompt`.",
                "Transform the `image` so that it contains the `prompt`.",
                "Use `prompt` to transform this `image`.",
            ],
            inputs=["image", "prompt"],
            answer="image_transformer(image, prompt)",
        ),
        # 定义一个问题，用`prompt`替换图像`image`中的海狸
        # 输入参数包含图像`image`和替换文本`prompt`
        # 调用图像转换函数image_transformer，将`prompt`用于转换`image`
    ],
    
    [
        Problem(
            task=["Provide me the summary of the `text`.", "Summarize `text`."],
            inputs=["text"],
            answer="summary = summarizer(text)",
        ),
        # 定义一个问题，为我提供文本`text`的摘要
        # 输入参数为文本`text`
        # 调用总结函数summarizer，对文本`text`进行总结并存储在变量summary中
    
        Problem(
            task=["Read this summary to me.", "Read it out loud."],
            inputs=[],
            answer="audio = text_reader(summarizer(text))",
        ),
        # 定义一个问题，将之前的摘要朗读出来
        # 无输入参数
        # 调用文本阅读函数text_reader，将摘要内容读出并存储在变量audio中
    
        Problem(
            task=["Transcribing the previous result back in text.", "Transcribe the audio."],
            inputs=[],
            answer="text = transcriber(audio)",
        ),
        # 定义一个问题，将之前的结果转录回文本
        # 无输入参数
        # 调用音频转录函数transcriber，将音频内容转录为文本并存储在变量text中
    
        Problem(
            task=["Translating the last result in French.", "Translate this in French."],
            inputs=[],
            answer="translator(text, src_lang='English', tgt_lang='French')",
        ),
        # 定义一个问题，将最后的结果翻译成法语
        # 无输入参数
        # 调用翻译函数translator，将文本内容从英语翻译成法语
    ],
    
    [
        Problem(
            task=["Generate a video of the `prompt`", "Animate a `prompt`", "Make me a short video using `prompt`."],
            inputs={"prompt": "A lobster swimming"},
            answer="video_generator('A lobster swimming')",
        ),
        # 定义一个问题，根据`prompt`生成一个视频
        # 输入参数为`prompt`，值为“A lobster swimming”
        # 调用视频生成器函数video_generator，根据`prompt`生成视频
    ],
    [
        # 定义一个包含两个任务的问题对象
        Problem(
            task=[
                "Download the content of `url` and summarize it.",  # 下载`url`的内容并进行总结
                "Summarize the content of the web page at `url`.",  # 对`url`的网页内容进行总结
            ],
            inputs=["url"],  # 输入参数为`url`
            answer="summary = summarizer(text_downloader(url))",  # 答案为对下载内容进行总结
        ),
        # 定义一个生成视频的问题对象
        Problem(
            task=["generate a video from it.",  # 从上一个结果生成一个视频
                  "Create an animation from the last result."],  # 从上一个结果创建一个动画
            inputs=[],  # 输入参数为空
            answer="video_generator(summary)",  # 答案为生成视频
        ),
    ],
# 定义一个函数，用于获取理论工具集
def get_theoretical_tools(agent_answer, theoretical_answer, code_answer):
    # 如果理论答案不是列表，则返回测试工具集中与代码答案匹配的工具名集合
    if not isinstance(theoretical_answer, list):
        return {name for name in TEST_TOOLS if name in code_answer}

    # 如果代理答案是字典类型
    if isinstance(agent_answer, dict):
        # 遍历理论答案和代码答案，如果理论答案中的值在代理答案的值中，则返回测试工具集中与代码答案匹配的工具名集合
        for one_answer, one_code in zip(theoretical_answer, code_answer):
            if one_answer in agent_answer.values():
                return {name for name in TEST_TOOLS if name in one_code}

    # 遍历理论答案和代码答案，如果代理答案等于理论答案，则返回测试工具集中与代码答案匹配的工具名集合
    for one_answer, one_code in zip(theoretical_answer, code_answer):
        if agent_answer == one_answer:
            return {name for name in TEST_TOOLS if name in one_code}

    # 返回测试工具集中与代码答案匹配的工具名集合
    return {name for name in TEST_TOOLS if name in code_answer[0]}


# 定义一个函数，用于评估代码
def evaluate_code(code, inputs=None, state=None, verbose=False, return_interpretor_error=False):
    # 复制基本 Python 工具集
    tools = BASE_PYTHON_TOOLS.copy()
    # 遍历测试工具集，如果工具名不在代码中，则继续下一个工具
    for name, tool in TEST_TOOLS.items():
        if name not in code:
            continue
        tools[name] = tool

    # 如果输入是字典类型，则复制一份
    if isinstance(inputs, dict):
        inputs = inputs.copy()
    # 如果输入不为空
    elif inputs is not None:
        # 将输入转换为字典，键为输入，值为输入的占位符
        inputs = {inp: f"<<{inp}>>" for inp in inputs}

    # 如果状态不为空，则更新状态
    if state is not None:
        state.update(inputs)
    else:
        state = inputs

    try:
        # 尝试评估代码
        return evaluate(code, tools, state)
    except InterpretorError as e:
        return str(e)
    except Exception as e:
        # 如果 verbose 为真，则打印异常信息
        if verbose:
            print(e)
        return None


# 定义一个函数，用于评分代码
def score_code(agent_answer, theoretical_answer, verbose: bool = False):
    # 如果 verbose 为真，则打印代理答案和理论答案
    if verbose:
        print(agent_answer, theoretical_answer)
    # 如果理论答案不是列表，则将其转换为列表
    theoretical_answer = theoretical_answer if isinstance(theoretical_answer, list) else [theoretical_answer]

    # 如果代理答案在理论答案中
    if agent_answer in theoretical_answer:
        # 如果 verbose 为真，则打印 "Perfect!"
        if verbose:
            print("Perfect!")
        return 1
    # 如果代理答案是字典类型，并且代理答案的值在理论答案中
    elif isinstance(agent_answer, dict) and any(v in theoretical_answer for v in agent_answer.values()):
        # 如果 verbose 为真，则打印 "Almsot perfect, result in state!"
        if verbose:
            print("Almsot perfect, result in state!")
        return 0.75
    else:
        # 如果 verbose 为真，则打印 "Result is not the right one but code executed."
        if verbose:
            print("Result is not the right one but code executed.")
        return 0.3


# 定义一个函数，用于评估一个结果
def evaluate_one_result(explanation, code, agent_answer, theoretical_answer, answer, verbose=False):
    # 在解释中提到的工具集
    tools_in_explanation = {name for name in TEST_TOOLS if f"`{name}`" in explanation}
    # 获取理论工具集
    theoretical_tools = get_theoretical_tools(agent_answer, theoretical_answer, answer)
    # 如果解释中的工具集与理论工具集相同
    if tools_in_explanation == theoretical_tools:
        tool_selection_score = 1.0
        tool_selection_errors = None
    else:
        # 计算工具选择得分和错误
        missing_tools = len(theoretical_tools - tools_in_explanation)
        unexpected_tools = len(tools_in_explanation - theoretical_tools)
        tool_selection_score = max(0, 1.0 - 0.25 * missing_tools - 0.25 * unexpected_tools)

        tool_selection_errors = {
            "selected_tools": tools_in_explanation,
            "theoretical_tools": theoretical_tools,
        }

    # 在代码中出现的工具集
    tools_in_code = {name for name in TEST_TOOLS if name in code}
    # 如果代码中使用的工具与理论工具相同
    if tools_in_code == theoretical_tools:
        # 工具使用得分为1.0
        tool_used_score = 1.0
        # 没有工具使用错误
        tool_used_errors = None
    else:
        # 计算缺失的工具数量
        missing_tools = len(theoretical_tools - tools_in_code)
        # 计算多余的工具数量
        unexpected_tools = len(tools_in_code - theoretical_tools)
        # 计算工具使用得分，最小为0
        tool_used_score = max(0, 1.0 - 0.25 * missing_tools - 0.25 * unexpected_tools)

        # 记录工具使用错误信息
        tool_used_errors = {
            "selected_tools": tools_in_explanation,
            "theoretical_tools": theoretical_tools,
        }

    # 计算代码得分
    score = score_code(agent_answer, theoretical_answer, verbose=verbose)
    # 如果得分小于1.0
    if score < 1.0:
        # 记录代码错误信息
        code_errors = {
            "code_produced": code,
            "evaluation": agent_answer,
            "theoretical_answer": theoretical_answer,
        }
    else:
        # 没有代码错误
        code_errors = None

    # 返回工具选择得分、工具使用得分、代码得分以及相关错误信息
    return (tool_selection_score, tool_used_score, score), (tool_selection_errors, tool_used_errors, code_errors)
def evaluate_agent(agent, batch_size=8, verbose=False, return_errors=False):
    """
    Evaluates a new agent on all `EVALUATION_TASKS`.

    Example:

    ```py
    agent = NewOpenAiAgent(model="text-davinci-003", api_key=your_api_key)
    bads = new_evaluate_agent(agent)
    for bad in bads:
        print(bad)
    ```
    """
    # 对 agent 的工具进行检查，确保包含所有测试工具
    agent_tools = set(agent.toolbox.keys())
    if agent_tools != set(TEST_TOOLS):
        missing_tools = set(TEST_TOOLS) - agent_tools
        unexpected_tools = set(agent_tools) - TEST_TOOLS
        raise ValueError(
            f"Fix the test tools in the evaluate_agent module. Tools mising: {missing_tools}. Extra tools: {unexpected_tools}."
        )

    eval_tasks = []
    eval_idx = []
    for idx, pb in enumerate(EVALUATION_TASKS):
        if isinstance(pb.task, list):
            eval_tasks.extend(pb.task)
            eval_idx.extend([idx] * len(pb.task)
        else:
            eval_tasks.append(pb.task)
            eval_idx.append(idx)

    tool_selection_score = 0
    tool_used_score = 0
    code_score = 0

    if return_errors:
        tool_selection_errors = {}
        tool_used_errors = {}
        code_errors = {}

    for start_idx in range(0, len(eval_tasks), batch_size):
        end_idx = min(start_idx + batch_size, len(eval_tasks)
        batch_tasks = eval_tasks[start_idx:end_idx]

        prompts = [agent.format_prompt(task) for task in batch_tasks]
        results = agent.generate_many(prompts, stop=["Task:"])

        for idx, result in enumerate(results):
            problem = EVALUATION_TASKS[eval_idx[start_idx + idx]]
            if verbose:
                print(f"====Task {start_idx + idx}====\n{batch_tasks[idx]}\n")
            explanation, code = clean_code_for_run(result)

            # 评估 agent 的答案和代码答案
            agent_answer = evaluate_code(code, problem.inputs, verbose=verbose)
            if isinstance(problem.answer, list):
                theoretical_answer = [evaluate_code(answer, problem.inputs) for answer in problem.answer]
            else:
                theoretical_answer = evaluate_code(problem.answer, problem.inputs)

            scores, errors = evaluate_one_result(
                explanation, code, agent_answer, theoretical_answer, problem.answer, verbose=verbose
            )

            tool_selection_score += scores[0]
            tool_used_score += scores[1]
            code_score += scores[2]

            if return_errors:
                if errors[0] is not None:
                    tool_selection_errors[batch_tasks[idx]] = errors[0]
                if errors[1] is not None:
                    tool_used_errors[batch_tasks[idx]] = errors[1]
                if errors[2] is not None:
                    code_errors[batch_tasks[idx]] = errors[2]
    # 计算工具选择得分，工具使用得分和代码得分，并将它们放入字典中
    scores = {
        "tool selection score": 100 * (tool_selection_score / len(eval_tasks)),
        "tool used score": 100 * (tool_used_score / len(eval_tasks)),
        "code score": 100 * (code_score / len(eval_tasks)),
    }

    # 如果需要返回错误信息，则返回得分和各种错误信息
    if return_errors:
        return scores, tool_selection_errors, tool_used_errors, code_errors
    # 否则，只返回得分
    else:
        return scores
def evaluate_chat_agent(agent, verbose=False, return_errors=False):
    """
    Evaluates a new agent on all `EVALUATION_CHATS`.

    Example:

    ```py
    agent = NewOpenAiAgent(model="text-davinci-003", api_key=your_api_key)
    bads = new_evaluate_agent(agent)
    for bad in bads:
        print(bad)
    ```
    """
    # 对代理工具进行检查
    agent_tools = set(agent.toolbox.keys())
    # 如果代理工具不符合测试工具的要求，抛出异常
    if agent_tools != set(TEST_TOOLS):
        missing_tools = set(TEST_TOOLS) - agent_tools
        unexpected_tools = agent_tools - set(TEST_TOOLS)
        raise ValueError(
            f"Fix the test tools in the evaluate_agent module. Tools mising: {missing_tools}. Extra tools: {unexpected_tools}."
        )

    # 初始化得分和步骤计数
    tool_selection_score = 0
    tool_used_score = 0
    code_score = 0
    total_steps = 0

    # 如果需要返回错误信息，初始化错误字典
    if return_errors:
        tool_selection_errors = {}
        tool_used_errors = {}
        code_errors = {}
    # 遍历评估聊天中的每个问题
    for chat_problem in EVALUATION_CHATS:
        # 检查问题的任务是否为字符串类型，如果是则将其放入resolved_problems列表中
        if isinstance(chat_problem[0].task, str):
            resolved_problems = [chat_problem]
        else:
            # 如果任务不是字符串类型，则根据任务的长度创建新的问题列表
            resolved_problems = [
                [Problem(task=pb.task[i], inputs=pb.inputs, answer=pb.answer) for pb in chat_problem]
                for i in range(len(chat_problem[0].task))
            ]
        # 遍历解决后的问题列表
        for problem in resolved_problems:
            # 准备新的聊天
            agent.prepare_for_new_chat()
            agent_state = {}  # 初始化代理状态
            theoretical_state = (
                [{} for _ in range(len(problem[0].answer))] if isinstance(problem[0].answer, list) else {}
            )  # 初始化理论状态

            # 遍历问题的每个步骤
            for step, step_problem in enumerate(problem):
                # 如果设置了详细输出，打印问题任务
                if verbose:
                    print(step_problem.task)
                total_steps += 1  # 总步数加一
                prompt = agent.format_prompt(step_problem.task, chat_mode=True)  # 格式化提示信息
                result = agent.generate_one(prompt, stop=["Human:", "====="])  # 生成一个结果
                agent.chat_history = prompt + result + "\n"  # 更新聊天历史记录

                explanation, code = clean_code_for_chat(result)  # 清理生成的代码

                if verbose:
                    # 如果设置了详细输出，打印代理的解释和生成的代码
                    print(f"==Explanation from the agent==\n{explanation}")
                    print(f"\n==Code generated by the agent==\n{code}")

                # 评估代理的答案和代码答案
                agent_answer = evaluate_code(code, step_problem.inputs, state=agent_state, verbose=verbose)

                answer = step_problem.answer
                if isinstance(answer, list):
                    # 如果答案是列表，则根据理论状态评估每个答案
                    theoretical_answer = [
                        evaluate_code(a, step_problem.inputs, state=state)
                        for a, state in zip(answer, theoretical_state)
                    ]
                else:
                    theoretical_answer = evaluate_code(answer, step_problem.inputs, state=theoretical_state)

                # 评估单个结果的得分和错误
                scores, errors = evaluate_one_result(
                    explanation, code, agent_answer, theoretical_answer, answer, verbose=verbose
                )

                tool_selection_score += scores[0]  # 工具选择得分
                tool_used_score += scores[1]  # 工具使用得分
                code_score += scores[2]  # 代码得分

                if return_errors:
                    # 如果需要返回错误信息，则将错误信息存储在相应的字典中
                    if errors[0] is not None:
                        tool_selection_errors[step_problem.task] = errors[0]
                    if errors[1] is not None:
                        tool_used_errors[step_problem.task] = errors[1]
                    if errors[2] is not None:
                        code_errors[step_problem.task] = errors[2]

    # 计算总体得分
    scores = {
        "tool selection score": 100 * (tool_selection_score / total_steps),
        "tool used score": 100 * (tool_used_score / total_steps),
        "code score": 100 * (code_score / total_steps),
    }

    # 如果需要返回错误信息，则返回得分和错误信息字典
    if return_errors:
        return scores, tool_selection_errors, tool_used_errors, code_errors
    else:
        # 否则只返回得分
        return scores
```