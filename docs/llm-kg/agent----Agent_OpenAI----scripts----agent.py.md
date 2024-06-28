# `.\agent\Agent_OpenAI\scripts\agent.py`

```
import colorama  # 导入colorama库，用于在控制台输出彩色文本
from colorama import Fore  # 从colorama库中导入Fore，用于设置文本颜色
from langchain_openai import OpenAI  # 导入langchain_openai库中的OpenAI类
from pydantic.v1 import BaseModel  # 导入pydantic.v1库中的BaseModel类
from scripts.tool import Tool, ToolResult  # 从scripts.tool模块中导入Tool和ToolResult类
from scripts.utils import parse_function_args, run_tool_from_response  # 从scripts.utils模块中导入parse_function_args和run_tool_from_response函数

# 定义StepResult类，继承自BaseModel类，表示步骤执行结果
class StepResult(BaseModel):
    event: str  # 表示事件名称，为字符串类型
    content: str  # 表示事件内容，为字符串类型
    success: bool  # 表示事件执行成功与否，为布尔类型


# 定义系统消息常量，描述任务和工具的使用情况
SYSTEM_MESSAGE = """You are tasked with completing specific objectives and must report the outcomes. At your disposal, you have a variety of tools, each specialized in performing a distinct type of task.

For successful task completion:
Thought: Consider the task at hand and determine which tool is best suited based on its capabilities and the nature of the work.

Use the report_tool with an instruction detailing the results of your work.
If you encounter an issue and cannot complete the task:

Use the report_tool to communicate the challenge or reason for the task's incompletion.
You will receive feedback based on the outcomes of each tool's task execution or explanations for any tasks that couldn't be completed. This feedback loop is crucial for addressing and resolving any issues by strategically deploying the available tools.
"""

# 定义OpenAIAgent类
class OpenAIAgent:

    def __init__(
            self,
            tools: list[Tool],  # 工具列表，每个元素为Tool类的实例
            client: OpenAI,  # OpenAI客户端，使用OpenAI类的实例
            system_message: str = SYSTEM_MESSAGE,  # 系统消息，默认为SYSTEM_MESSAGE常量
            model_name: str = "gpt-3.5-turbo-0125",  # 模型名称，默认为"gpt-3.5-turbo-0125"
            max_steps: int = 5,  # 最大步骤数，默认为5
            verbose: bool = True  # 是否显示详细输出，默认为True
    ):
        self.tools = tools  # 初始化工具列表
        self.client = client  # 初始化OpenAI客户端
        self.model_name = model_name  # 初始化模型名称
        self.system_message = system_message  # 初始化系统消息
        self.memory = []  # 初始化记忆列表，暂时为空
        self.step_history = []  # 初始化步骤历史记录，暂时为空
        self.max_steps = max_steps  # 初始化最大步骤数
        self.verbose = verbose  # 初始化是否显示详细输出标志

    def to_console(self, tag: str, message: str, color: str = "green"):
        # 在控制台输出消息
        if self.verbose:
            color_prefix = Fore.__dict__[color.upper()]  # 根据color参数设置颜色前缀
            print(color_prefix + f"{tag}: {message}{colorama.Style.RESET_ALL}")

    def run(self, user_input: str):
        # 运行Agent，处理用户输入user_input
        self.to_console("START", f"Starting Agent with Input: {user_input}")  # 输出Agent启动信息和用户输入内容
        openai_tools = [tool.openai_tool_schema for tool in self.tools]  # 获取所有工具的OpenAI工具模式

        # 初始化步骤历史记录，包括系统消息和用户输入
        self.step_history = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_input}
        ]

        step_result = None  # 初始化步骤结果变量
        i = 0  # 初始化步骤计数器

        # 执行步骤循环，直到达到最大步骤数或任务完成
        while i < self.max_steps:
            step_result = self.run_step(self.step_history, openai_tools)  # 执行当前步骤
            if step_result.event == "finish":
                break  # 如果步骤事件为"finish"，则任务完成，跳出循环
            elif step_result.event == "error":
                self.to_console(step_result.event, step_result.content, "red")  # 输出错误消息
            else:
                self.to_console(step_result.event, step_result.content, "yellow")  # 输出警告消息

            i += 1  # 更新步骤计数器

        self.to_console("Final Result", step_result.content, "green")  # 输出最终结果
        return step_result.content  # 返回步骤结果内容
    #`
    def run_step(self, messages: list[dict], tools):

        # plan the next step
        response = self.client.chat.completions.create(
            model=self.model_name,  # 使用指定的模型名创建聊天完成
            messages=messages,      # 提供聊天消息列表
            tools=tools              # 提供工具列表
        )

        # add message to history
        self.step_history.append(response.choices[0].message)  # 将返回的消息添加到步骤历史中
        # check if tool call is present
        if not response.choices[0].message.tool_calls:  # 检查是否有工具调用
            step_result = StepResult(event="Error", content="No tool calls were returned.", success=False)
            return step_result

        tool_name = response.choices[0].message.tool_calls[0].function.name  # 获取工具调用的名称
        tool_kwargs = parse_function_args(response)  # 解析函数参数

        # execute the tool call
        self.to_console("Tool Call", f"Name: {tool_name}\nArgs: {tool_kwargs}", "magenta")  # 输出工具调用信息到控制台
        tool_result = run_tool_from_response(response, tools=self.tools)  # 执行工具调用
        tool_result_msg = self.tool_call_message(response, tool_result)  # 生成工具调用消息
        self.step_history.append(tool_result_msg)  # 将工具调用结果消息添加到步骤历史

        if tool_result.success:  # 检查工具调用是否成功
            step_result = StepResult(
                event="tool_result",
                content=tool_result.content,
                success=True)
        else:
            step_result = StepResult(
                event="error",
                content=tool_result.content,
                success=False
            )

        return step_result

    def tool_call_message(self, response, tool_result: ToolResult):
        tool_call = response.choices[0].message.tool_calls[0]  # 获取工具调用信息
        return {
            "tool_call_id": tool_call.id,  # 工具调用ID
            "role": "tool",  # 角色标识为工具
            "name": tool_call.function.name,  # 工具名称
            "content": tool_result.content,  # 工具调用返回内容
        }
```