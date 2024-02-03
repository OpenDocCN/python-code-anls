# `.\AutoGPT\autogpts\autogpt\autogpt\core\planning\templates.py`

```py
# Rules of thumb:
# - 模板不会在字符串末尾添加新行。这是模板的责任或消费模板的责任。

####################
# 计划者默认值 #
####################

# 用户目标
USER_OBJECTIVE = (
    "Write a wikipedia style article about the project: "
    "https://github.com/significant-gravitas/AutoGPT"
)

# 计划提示
# -----------

# 计划提示的限制条件
PLAN_PROMPT_CONSTRAINTS = (
    "~4000 word limit for short term memory. Your short term memory is short, so "
    "immediately save important information to files.",
    "If you are unsure how you previously did something or want to recall past "
    "events, thinking about similar events will help you remember.",
    "No user assistance",
    "Exclusively use the commands listed below e.g. command_name",
)

# 计划提示的资源
PLAN_PROMPT_RESOURCES = (
    "Internet access for searches and information gathering.",
    "Long-term memory management.",
    "File output.",
)

# 计划提示的绩效评估
PLAN_PROMPT_PERFORMANCE_EVALUATIONS = (
    "Continuously review and analyze your actions to ensure you are performing to"
    " the best of your abilities.",
    "Constructively self-criticize your big-picture behavior constantly.",
    "Reflect on past decisions and strategies to refine your approach.",
    "Every command has a cost, so be smart and efficient. Aim to complete tasks in"
    " the least number of steps.",
    "Write all code to a file",
)

# 计划提示的响应字典
PLAN_PROMPT_RESPONSE_DICT = {
    "thoughts": {
        "text": "thought",
        "reasoning": "reasoning",
        "plan": "- short bulleted\n- list that conveys\n- long-term plan",
        "criticism": "constructive self-criticism",
        "speak": "thoughts summary to say to user",
    },
    "command": {"name": "command name", "args": {"arg name": "value"}},
}

# 计划提示的响应格式
PLAN_PROMPT_RESPONSE_FORMAT = (
    "You should only respond in JSON format as described below\n"
    "Response Format:\n"
    "{response_json_structure}\n"
    "Ensure the response can be parsed by Python json.loads"
)
# 触发计划的提示信息，指示用户使用指定格式确定下一个命令
PLAN_TRIGGERING_PROMPT = (
    "Determine which next command to use, and respond using the format specified above:"
)

# 主要计划提示信息模板，包含目标、信息、约束、命令、资源、性能评估等内容
PLAN_PROMPT_MAIN = (
    "{header}\n\n"
    "GOALS:\n\n{goals}\n\n"
    "Info:\n{info}\n\n"
    "Constraints:\n{constraints}\n\n"
    "Commands:\n{commands}\n\n"
    "Resources:\n{resources}\n\n"
    "Performance Evaluations:\n{performance_evaluations}\n\n"
    "You should only respond in JSON format as described below\n"
    "Response Format:\n{response_json_structure}\n"
    "Ensure the response can be parsed by Python json.loads"
)

# 参数化模板
###########################
# Parameterized templates #
###########################
```