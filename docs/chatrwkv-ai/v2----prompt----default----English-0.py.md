# `ChatRWKV\v2\prompt\default\English-0.py`

```py
# 定义用户界面的符号
interface = ":"
# 定义用户的名称
user = "Question"
# 定义机器人的名称
bot = "Answer"
# 初始化对话的提示信息，包括用户名称、用户界面符号、机器人名称和机器人的欢迎语
init_prompt = f'''{user}{interface} hi

{bot}{interface} Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.
'''
```