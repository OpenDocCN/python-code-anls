# `.\graphrag\graphrag\prompt_tune\prompt\persona.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Fine-tuning prompts for persona generation."""

# 定义生成角色描述的提示文本模板，用于生成专家角色描述
GENERATE_PERSONA_PROMPT = """
You are an intelligent assistant that helps a human to analyze the information in a text document.
Given a specific type of task and sample text, help the user by generating a 3 to 4 sentence description of an expert who could help solve the problem.
Use a format similar to the following:
You are an expert {{role}}. You are skilled at {{relevant skills}}. You are adept at helping people with {{specific task}}.

task: {sample_task}
persona description:"""


这段代码定义了一个用于生成专家角色描述的提示文本模板，包含了角色描述的格式和样例任务。
```