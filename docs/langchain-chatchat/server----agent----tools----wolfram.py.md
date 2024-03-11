# `.\Langchain-Chatchat\server\agent\tools\wolfram.py`

```py
# 导入 Langchain 自带的 Wolfram Alpha API 封装模块
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
# 导入 Pydantic 模块中的 BaseModel 和 Field 类
from pydantic import BaseModel, Field
# 设置 Wolfram Alpha API 的应用密钥
wolfram_alpha_appid = "your key"

# 定义一个函数 wolfram，用于向 Wolfram Alpha API 发送查询并返回结果
def wolfram(query: str):
    # 创建 WolframAlphaAPIWrapper 对象，传入应用密钥
    wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid=wolfram_alpha_appid)
    # 调用 WolframAlphaAPIWrapper 对象的 run 方法，发送查询并获取结果
    ans = wolfram.run(query)
    # 返回查询结果
    return ans

# 定义一个 Pydantic 模块中的 BaseModel 类的子类 WolframInput
class WolframInput(BaseModel):
    # 定义 WolframInput 类的属性 location，类型为 str，描述为"需要运算的具体问题"
    location: str = Field(description="需要运算的具体问题")
```