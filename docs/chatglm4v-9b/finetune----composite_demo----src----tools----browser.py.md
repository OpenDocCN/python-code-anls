# `.\chatglm4-finetune\composite_demo\src\tools\browser.py`

```py
# 简单的浏览器工具说明
"""
Simple browser tool.

# Usage

Please start the backend browser server according to the instructions in the README.
"""

# 导入用于格式化输出的模块
from pprint import pprint
# 导入正则表达式模块
import re
# 导入请求模块，用于发送 HTTP 请求
import requests
# 导入 Streamlit 库，用于创建 Web 应用
import streamlit as st
# 导入数据类模块，用于定义数据结构
from dataclasses import dataclass

# 导入浏览器服务器的 URL 配置
from .config import BROWSER_SERVER_URL
# 导入工具观察接口
from .interface import ToolObservation

# 定义正则表达式用于匹配引用格式
QUOTE_REGEX = re.compile(r"\[(\d+)†(.+?)\]")

# 定义引用数据类，用于存储标题和 URL
@dataclass
class Quote:
    title: str
    url: str

# 检查会话状态中是否包含引用信息，如果没有则初始化为空字典
if "quotes" not in st.session_state:
    st.session_state.quotes = {}

# 获取会话状态中的引用字典
quotes: dict[str, Quote] = st.session_state.quotes

# 定义映射响应的函数，将响应转换为工具观察对象
def map_response(response: dict) -> ToolObservation:
    # 打印浏览器响应以供调试
    print('===BROWSER_RESPONSE===')
    pprint(response)
    # 获取角色元数据
    role_metadata = response.get("roleMetadata")
    # 获取其他元数据
    metadata = response.get("metadata")
    
    # 处理引用结果
    if role_metadata.split()[0] == 'quote_result' and metadata:
        # 提取引用 ID
        quote_id = QUOTE_REGEX.search(role_metadata.split()[1]).group(1)
        # 获取引用的元数据
        quote: dict[str, str] = metadata['metadata_list'][0]
        # 将引用添加到引用字典
        quotes[quote_id] = Quote(quote['title'], quote['url'])
    # 处理浏览器结果
    elif role_metadata == 'browser_result' and metadata:
        # 遍历元数据列表，将每个引用添加到字典
        for i, quote in enumerate(metadata['metadata_list']):
            quotes[str(i)] = Quote(quote['title'], quote['url'])

    # 返回工具观察对象，包含内容类型、文本、角色元数据和元数据
    return ToolObservation(
        content_type=response.get("contentType"),
        text=response.get("result"),
        role_metadata=role_metadata,
        metadata=metadata,
    )

# 定义工具调用函数，接受代码和会话 ID 作为参数
def tool_call(code: str, session_id: str) -> list[ToolObservation]:
    # 构建请求字典，包含会话 ID 和操作代码
    request = {
        "session_id": session_id,
        "action": code,
    }
    # 发送 POST 请求到浏览器服务器并获取响应
    response = requests.post(BROWSER_SERVER_URL, json=request).json()
    # 将响应映射为工具观察对象的列表并返回
    return list(map(map_response, response))
```