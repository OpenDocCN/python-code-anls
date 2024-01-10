# `MetaGPT\tests\metagpt\strategy\prompt_templates\creative_writing.py`

```

# 定义一个标准的提示文本，包含一个占位符 {input}，用于生成一段连贯的短文
standard_prompt = """
Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: {input}
"""

# 定义一个创作任务的提示文本，包含一个占位符 {input}，用于生成一段连贯的短文，并要求提供一个计划和输出
cot_prompt = """
Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: {input}

Make a plan then write. Your output should be like:

Plan:
<Your plan here with json format>

Passage:
<Your passage here with json format>
"""

# 定义一个投票任务的提示文本，要求根据给定的指令和选择，分析每个选择的优劣，并在最后一行得出结论
vote_prompt = """Given an instruction and several choices, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where s the integer id of the choice.
"""

# 定义一个比较任务的提示文本，要求简要分析两个段落的连贯性，并在最后一行得出结论
compare_prompt = """Briefly analyze the coherency of the following two passages. Conclude in the last line "The more coherent passage is 1", "The more coherent passage is 2", or "The two passages are similarly coherent".
"""

# 定义一个评分任务的提示文本，要求分析给定的段落，并在最后一行给出连贯性评分
score_prompt = """Analyze the following passage, then at the last line conclude "Thus the coherency score is {s}", where s is an integer from 1 to 10.

```