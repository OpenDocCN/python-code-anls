# `.\Langchain-Chatchat\server\agent\custom_template.py`

```
# 导入未来版本的注解功能
from __future__ import annotations
# 导入自定义模块
from langchain.agents import Tool, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from typing import List
from langchain.schema import AgentAction, AgentFinish

# 从配置文件中导入支持的代理模型
from configs import SUPPORT_AGENT_MODEL
# 从服务器代理模块中导入模型容器
from server.agent import model_container

# 自定义字符串提示模板类，继承自StringPromptTemplate类
class CustomPromptTemplate(StringPromptTemplate):
    # 模板字符串
    template: str
    # 工具列表
    tools: List[Tool]

    # 格式化方法，返回格式化后的字符串
    def format(self, **kwargs) -> str:
        # 获取中间步骤
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        # 遍历中间步骤，拼接日志和观察结果
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        # 拼接工具描述
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # 拼接工具名称
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        # 格式化模板字符串
        return self.template.format(**kwargs)

# 自定义代理输出解析器类，继承自AgentOutputParser类
class CustomOutputParser(AgentOutputParser):
    # 开始标志
    begin: bool = False

    # 初始化方法
    def __init__(self):
        # 调用父类初始化方法
        super().__init__()
        # 设置开始标志为True
        self.begin = True
    # 解析LLM输出，根据不同情况返回AgentFinish对象、包含字典和字符串的元组或AgentAction对象
    def parse(self, llm_output: str) -> AgentFinish | tuple[dict[str, str], str] | AgentAction:
        # 如果模型不在支持的模型列表中，并且开始标志为True，则执行以下操作
        if not any(agent in model_container.MODEL for agent in SUPPORT_AGENT_MODEL) and self.begin:
            # 将开始标志设为False
            self.begin = False
            # 定义停止词列表
            stop_words = ["Observation:"]
            # 初始化最小索引为LLM输出的长度
            min_index = len(llm_output)
            # 遍历停止词列表
            for stop_word in stop_words:
                # 查找停止词在LLM输出中的索引
                index = llm_output.find(stop_word)
                # 如果找到停止词并且索引小于最小索引，则更新最小索引
                if index != -1 and index < min_index:
                    min_index = index
                # 截取LLM输出至最小索引处
                llm_output = llm_output[:min_index]

        # 如果LLM输出中包含"Final Answer:"，则执行以下操作
        if "Final Answer:" in llm_output:
            # 将开始标志设为True
            self.begin = True
            # 返回AgentFinish对象，包含输出和日志信息
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:", 1)[-1].strip()},
                log=llm_output,
            )
        
        # 将LLM输出按"Action:"分割为两部分
        parts = llm_output.split("Action:")
        # 如果分割后的部分小于2，则返回AgentFinish对象，包含失败信息和日志信息
        if len(parts) < 2:
            return AgentFinish(
                return_values={"output": f"调用agent工具失败，该回答为大模型自身能力的回答:\n\n `{llm_output}`"},
                log=llm_output,
            )

        # 获取动作和动作输入
        action = parts[1].split("Action Input:")[0].strip()
        action_input = parts[1].split("Action Input:")[1].strip()
        try:
            # 尝试创建AgentAction对象
            ans = AgentAction(
                tool=action,
                tool_input=action_input.strip(" ").strip('"'),
                log=llm_output
            )
            return ans
        except:
            # 创建AgentFinish对象，包含失败信息和日志信息
            return AgentFinish(
                return_values={"output": f"调用agent失败: `{llm_output}`"},
                log=llm_output,
            )
```