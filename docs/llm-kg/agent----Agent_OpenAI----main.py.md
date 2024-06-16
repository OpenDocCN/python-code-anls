# `.\agent\Agent_OpenAI\main.py`

```
# 导入所需模块和库
from datetime import datetime  # 从 datetime 模块导入 datetime 类
import os  # 导入 os 模块

from openai import OpenAI  # 从 openai 模块导入 OpenAI 类
from pydantic.v1 import BaseModel  # 从 pydantic.v1 模块导入 BaseModel 类

from scripts.agent import OpenAIAgent  # 从 scripts.agent 模块导入 OpenAIAgent 类
from scripts.tool import Tool  # 从 scripts.tool 模块导入 Tool 类

# === TOOL DEFINITIONS ===

# 定义一个名为 Expense 的数据模型，包含描述、净金额、总金额、税率和日期字段
class Expense(BaseModel):
    description: str  # 描述字段，字符串类型
    net_amount: float  # 净金额字段，浮点数类型
    gross_amount: float  # 总金额字段，浮点数类型
    tax_rate: float  # 税率字段，浮点数类型
    date: datetime  # 日期字段，datetime 类型


# 定义一个函数用于添加开支
def add_expense_func(**kwargs):
    return f"Added expense: {kwargs} to the database."  # 返回已添加开支的信息


# 创建一个名为 add_expense_tool 的工具对象，包含名称、模型和函数
add_expense_tool = Tool(
    name="add_expense_tool",  # 工具名称
    model=Expense,  # 工具模型为 Expense 类
    function=add_expense_func  # 工具函数为 add_expense_func 函数
)


# 定义一个名为 ReportTool 的数据模型，包含报告字段
class ReportTool(BaseModel):
    report: str = None  # 报告字段，默认为 None


# 定义一个函数用于生成报告
def report_func(report: str = None):
    return f"Reported: {report}"  # 返回报告信息


# 创建一个名为 report_tool 的工具对象，包含名称、模型和函数
report_tool = Tool(
    name="report_tool",  # 工具名称
    model=ReportTool,  # 工具模型为 ReportTool 类
    function=report_func  # 工具函数为 report_func 函数
)


# 定义一个名为 DateTool 的数据模型，包含日期字段
class DateTool(BaseModel):
    x: str = None  # 日期字段，默认为 None


# 创建一个名为 get_date_tool 的工具对象，包含名称、模型、函数和验证缺失项的选项
get_date_tool = Tool(
    name="get_current_date",  # 工具名称
    model=DateTool,  # 工具模型为 DateTool 类
    function=lambda: datetime.now().strftime("%Y-%m-%d"),  # 工具函数为获取当前日期的 lambda 函数
    validate_missing=False  # 不验证缺失项
)

# 创建一个包含所有工具对象的列表
tools = [
    add_expense_tool,  # 添加开支工具对象
    report_tool,  # 报告工具对象
    get_date_tool  # 获取日期工具对象
]


# === RUN AGENT ===

# 创建一个 OpenAI 客户端对象，并使用环境变量中的 API 密钥
client = OpenAI(api_key=os.getenv("API_KEY"))
# 指定 GPT 模型名称
model_name = "gpt-3.5-turbo-0125"
# 创建一个 OpenAIAgent 实例，传入工具列表、OpenAI 客户端和模型名称
agent = OpenAIAgent(tools, client, model_name=model_name, verbose=True)

# 定义用户输入
user_input = "I have spend 5$ on a coffee today please track my expense. The tax rate is 0.2"

# 运行代理程序，传入用户输入
agent.run(user_input)
```