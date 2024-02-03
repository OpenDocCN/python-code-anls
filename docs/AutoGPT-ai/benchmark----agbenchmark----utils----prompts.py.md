# `.\AutoGPT\benchmark\agbenchmark\utils\prompts.py`

```py
# 定义一个字典，用于存储不同评分方式的说明
SCORING_MAP = {
    "percentage": "assign a float score that will represent a percentage out of 100. Use decimal points to be even more accurate. 0 represents the worst possible generation, while 100 represents the ideal generation",
    "scale": "assign an integer score from a scale of 1-10. 1 represents a really bad generation, while 10 represents an ideal generation",
    "binary": "assign a binary score of either 0 or 1. 0 represents a failure, while 1 represents a success",
}

# 定义一个字符串模板，用于评估机器生成的回答与人类答案的接近程度
REFERENCE_PROMPT = """Ignore previous directions. You are now an expert at evaluating how close machine generated responses are to human answers. You essentially act as a hyper advanced BLEU score.
In order to score the machine generated response you will {scoring}. Make sure to factor in the distance to the ideal response into your thinking, deliberation, and final result regarding scoring. Return nothing but a float score.

Here is the given task for you to evaluate:
{task}

Here is the ideal response you're comparing to based on the task:
{answer}

Here is the current machine generated response to the task that you need to evaluate:
{response}

"""

# 定义一个字符串模板，用于评估机器生成的回答与给定任务的接近程度
RUBRIC_PROMPT = """Ignore previous directions. You are now an expert at evaluating machine generated responses to given tasks.
In order to score the generated texts you will {scoring}. Make sure to factor in rubric into your thinking, deliberation, and final result regarding scoring. Return nothing but a float score.

Here is the given task for you to evaluate:
{task}

Use the below rubric to guide your thinking about scoring:
{answer}

Here is the current machine generated response to the task that you need to evaluate:
{response}

"""

# 定义一个字符串模板，用于评估机器生成的回答与给定任务的接近程度
QUESTION_PROMPT = """Ignore previous directions. You are now an expert at evaluating machine generated responses to given tasks.
# 为了评分生成的文本，需要进行评分。确保考虑生成的回答是否很好地回答问题，以便准确评分。返回一个浮点数分数。

# 给定的任务如下：
{task}

# 这是一个检查任务是否正确完成的问题：
{answer}

# 这是需要评估的任务的当前机器生成的响应：
{response}

"""

# 以下是如何根据上述内容评分机器生成的响应的一些示例：
{examples}

"""

# 自定义提示信息
{custom}
{scoring}

"""

# 提示信息映射，根据不同的提示类型选择相应的提示信息
PROMPT_MAP = {
    "rubric": RUBRIC_PROMPT,
    "reference": REFERENCE_PROMPT,
    "question": QUESTION_PROMPT,
    "custom": CUSTOM_PROMPT,
}

# 结束提示信息，提醒始终以仅包含浮点分数的形式结束您的响应。
Float score:"""
```