# `.\AutoGPT\autogpts\autogpt\autogpt\core\runner\client_lib\parser.py`

```py
# 解析代理人姓名和目标，返回格式化后的字符串
def parse_agent_name_and_goals(name_and_goals: dict) -> str:
    # 根据字典中的键值拼接代理人姓名和角色信息
    parsed_response = f"Agent Name: {name_and_goals['agent_name']}\n"
    parsed_response += f"Agent Role: {name_and_goals['agent_role']}\n"
    parsed_response += "Agent Goals:\n"
    # 遍历代理人目标列表，将序号和目标内容拼接到字符串中
    for i, goal in enumerate(name_and_goals["agent_goals"]):
        parsed_response += f"{i+1}. {goal}\n"
    return parsed_response


# 解析代理人计划，返回格式化后的字符串
def parse_agent_plan(plan: dict) -> str:
    parsed_response = "Agent Plan:\n"
    # 遍历任务列表，将任务信息拼接到字符串中
    for i, task in enumerate(plan["task_list"]):
        parsed_response += f"{i+1}. {task['objective']}\n"
        parsed_response += f"Task type: {task['type']}  "
        parsed_response += f"Priority: {task['priority']}\n"
        parsed_response += "Ready Criteria:\n"
        # 遍历准备条件列表，将准备条件信息拼接到字符串中
        for j, criteria in enumerate(task["ready_criteria"]):
            parsed_response += f"    {j+1}. {criteria}\n"
        parsed_response += "Acceptance Criteria:\n"
        # 遍历接受条件列表，将接受条件信息拼接到字符串中
        for j, criteria in enumerate(task["acceptance_criteria"]):
            parsed_response += f"    {j+1}. {criteria}\n"
        parsed_response += "\n"

    return parsed_response


# 解析下一个能力，返回格式化后的字符串
def parse_next_ability(current_task, next_ability: dict) -> str:
    # 拼接当前任务的目标信息
    parsed_response = f"Current Task: {current_task.objective}\n"
    # 将能力参数字典中的键值对拼接成字符串
    ability_args = ", ".join(
        f"{k}={v}" for k, v in next_ability["ability_arguments"].items()
    )
    parsed_response += f"Next Ability: {next_ability['next_ability']}({ability_args})\n"
    parsed_response += f"Motivation: {next_ability['motivation']}\n"
    parsed_response += f"Self-criticism: {next_ability['self_criticism']}\n"
    parsed_response += f"Reasoning: {next_ability['reasoning']}\n"
    return parsed_response


# 解析能力结果，返回格式化后的字符串
def parse_ability_result(ability_result) -> str:
    parsed_response = f"Ability: {ability_result['ability_name']}\n"
    parsed_response += f"Ability Arguments: {ability_result['ability_args']}\n"
    parsed_response += f"Ability Result: {ability_result['success']}\n"
    parsed_response += f"Message: {ability_result['message']}\n"
    # 将 ability_result 字典中的 'new_knowledge' 值添加到 parsed_response 字符串中
    parsed_response += f"Data: {ability_result['new_knowledge']}\n"
    # 返回拼接后的 parsed_response 字符串
    return parsed_response
```