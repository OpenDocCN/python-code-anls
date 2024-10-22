# `.\chatglm4-finetune\composite_demo\src\tools\cogview.py`

```py
# 导入 Streamlit 库，用于构建 Web 应用
import streamlit as st
# 导入 ZhipuAI 类，作为 AI 客户端
from zhipuai import ZhipuAI
# 从 ZhipuAI 库导入生成的图像类型
from zhipuai.types.image import GeneratedImage

# 从本地配置文件导入模型名称和 API 密钥
from .config import COGVIEW_MODEL, ZHIPU_AI_KEY
# 从本地接口模块导入工具观察类
from .interface import ToolObservation

# 使用 Streamlit 的缓存机制缓存 ZhipuAI 客户端
@st.cache_resource
def get_zhipu_client():
    # 创建并返回一个 ZhipuAI 客户端实例，使用 API 密钥
    return ZhipuAI(api_key=ZHIPU_AI_KEY)

# 定义映射响应的函数，接收生成的图像作为参数
def map_response(img: GeneratedImage):
    # 返回工具观察对象，包含图像的相关信息
    return ToolObservation(
        content_type='image',  # 设置内容类型为图像
        text='CogView 已经生成并向用户展示了生成的图片。',  # 返回的文本信息
        image_url=img.url,  # 图像的 URL 地址
        role_metadata='cogview_result'  # 角色元数据，标识来源
    )

# 定义工具调用函数，接收提示文本和会话 ID
def tool_call(prompt: str, session_id: str) -> list[ToolObservation]:
    # 获取 ZhipuAI 客户端实例
    client = get_zhipu_client()
    # 调用生成图像的 API，返回响应数据
    response = client.images.generations(model=COGVIEW_MODEL, prompt=prompt).data
    # 将响应映射为工具观察列表并返回
    return list(map(map_response, response))
```