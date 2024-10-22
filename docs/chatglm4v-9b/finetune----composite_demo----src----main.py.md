# `.\chatglm4-finetune\composite_demo\src\main.py`

```
# 这个文档演示 GLM-4 的所有工具和长上下文聊天能力
"""
This demo show the All tools and Long Context chat Capabilities of GLM-4.
Please follow the Readme.md to run the demo.
"""

# 导入操作系统模块
import os
# 导入 traceback 模块，用于调试时打印异常信息
import traceback
# 导入枚举类
from enum import Enum
# 导入字节流操作类
from io import BytesIO
# 导入生成唯一标识符的函数
from uuid import uuid4

# 导入 Streamlit 库，用于创建网页应用
import streamlit as st
# 从 Streamlit 的 delta_generator 模块导入 DeltaGenerator 类
from streamlit.delta_generator import DeltaGenerator

# 导入处理图像的库
from PIL import Image

# 导入客户端相关的类和函数
from client import Client, ClientType, get_client
# 从 conversation 模块导入相关的常量和类
from conversation import (
    FILE_TEMPLATE,
    Conversation,
    Role,
    postprocess_text,
    response_to_str,
)
# 从工具注册模块导入调度工具和获取工具的函数
from tools.tool_registry import dispatch_tool, get_tools
# 导入文本提取相关的实用函数
from utils import extract_pdf, extract_docx, extract_pptx, extract_text

# 获取聊天模型路径，如果未设置则使用默认值
CHAT_MODEL_PATH = os.environ.get("CHAT_MODEL_PATH", "THUDM/glm-4-9b-chat")
# 获取多模态模型路径，如果未设置则使用默认值
VLM_MODEL_PATH = os.environ.get("VLM_MODEL_PATH", "THUDM/glm-4v-9b")

# 判断是否使用 VLLM，根据环境变量进行设置
USE_VLLM = os.environ.get("USE_VLLM", "0") == "1"
# 判断是否使用 API，根据环境变量进行设置
USE_API = os.environ.get("USE_API", "0") == "1"

# 定义模式枚举类
class Mode(str, Enum):
    # 所有工具模式的标识
    ALL_TOOLS = "🛠️ All Tools"
    # 长上下文模式的标识
    LONG_CTX = "📝 文档解读"
    # 多模态模式的标识
    VLM = "🖼️ 多模态"

# 定义一个函数用于向对话历史中追加对话
def append_conversation(
    conversation: Conversation,  # 当前对话
    history: list[Conversation],  # 对话历史
    placeholder: DeltaGenerator | None = None,  # 可选的占位符
) -> None:
    """
    将一段对话追加到历史中，同时在新的 markdown 块中显示
    """
    # 将当前对话添加到历史列表中
    history.append(conversation)
    # 显示当前对话内容
    conversation.show(placeholder)

# 设置 Streamlit 页面的配置
st.set_page_config(
    # 页面标题
    page_title="GLM-4 Demo",
    # 页面图标
    page_icon=":robot:",
    # 页面布局方式
    layout="centered",
    # 初始侧边栏状态
    initial_sidebar_state="expanded",
)

# 设置页面标题
st.title("GLM-4 Demo")
# 显示 markdown 文本，包含技术文档链接
st.markdown(
    "<sub>智谱AI 公开在线技术文档: https://zhipu-ai.feishu.cn/wiki/RuMswanpkiRh3Ok4z5acOABBnjf </sub> \n\n <sub> 更多 GLM-4 开源模型的使用方法请参考文档。</sub>",
    unsafe_allow_html=True,
)

# 在侧边栏中创建用户输入组件
with st.sidebar:
    # 创建 slider 组件用于调整 top_p 参数
    top_p = st.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
    # 创建 slider 组件用于调整 top_k 参数
    top_k = st.slider("top_k", 1, 20, 10, step=1, key="top_k")
    # 创建 slider 组件用于调整温度参数
    temperature = st.slider("temperature", 0.0, 1.5, 0.95, step=0.01)
    # 创建 slider 组件用于调整重复惩罚参数
    repetition_penalty = st.slider("repetition_penalty", 0.0, 2.0, 1.0, step=0.01)
    # 创建 slider 组件用于调整最大新令牌数
    max_new_tokens = st.slider("max_new_tokens", 1, 4096, 2048, step=1)
    # 创建两列布局
    cols = st.columns(2)
    # 创建导出按钮
    export_btn = cols[0]
    # 创建清除历史记录的按钮
    clear_history = cols[1].button("Clear", use_container_width=True)
    # 创建重试按钮
    retry = export_btn.button("Retry", use_container_width=True)

# 如果用户点击清除历史记录按钮
if clear_history:
    # 保存当前页和客户端状态
    page = st.session_state.page
    client = st.session_state.client
    # 清除会话状态
    st.session_state.clear()
    # 恢复当前页和客户端状态
    st.session_state.page = page
    st.session_state.client = client
    # 重置文件上传状态
    st.session_state.files_uploaded = False
    # 重置上传文本
    st.session_state.uploaded_texts = ""
    # 重置上传文件数量
    st.session_state.uploaded_file_nums = 0
    # 重置对话历史
    st.session_state.history = []

# 检查文件上传状态，如果未定义则初始化为 False
if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = False

# 检查会话 ID，如果未定义则生成一个新的 UUID
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid4()

# 检查对话历史，如果未定义则初始化为空列表
if "history" not in st.session_state:
    st.session_state.history = []

# 检查是否为首次对话
first_round = len(st.session_state.history) == 0

# 定义构建客户端的函数
def build_client(mode: Mode) -> Client:
    # 根据不同的模式进行处理
        match mode:
            # 如果模式是 ALL_TOOLS
            case Mode.ALL_TOOLS:
                # 设置会话状态中的 top_k 为 10
                st.session_state.top_k = 10
                # 根据是否使用 VLLM 选择客户端类型
                typ = ClientType.VLLM if USE_VLLM else ClientType.HF
                # 根据是否使用 API 更新客户端类型
                typ = ClientType.API if USE_API else typ
                # 返回与指定模型路径和客户端类型相关的客户端
                return get_client(CHAT_MODEL_PATH, typ)
            # 如果模式是 LONG_CTX
            case Mode.LONG_CTX:
                # 设置会话状态中的 top_k 为 10
                st.session_state.top_k = 10
                # 根据是否使用 VLLM 选择客户端类型
                typ = ClientType.VLLM if USE_VLLM else ClientType.HF
                # 返回与指定模型路径和客户端类型相关的客户端
                return get_client(CHAT_MODEL_PATH, typ)
            # 如果模式是 VLM
            case Mode.VLM:
                # 设置会话状态中的 top_k 为 1
                st.session_state.top_k = 1
                # vLLM 不适用于 VLM 模式
                return get_client(VLM_MODEL_PATH, ClientType.HF)
# 页面变化的回调函数
def page_changed() -> None:
    # 声明全局变量 client
    global client
    # 获取当前会话状态中的页面名称
    new_page: str = st.session_state.page
    # 清空会话历史记录
    st.session_state.history.clear()
    # 根据新页面构建客户端并更新会话状态
    st.session_state.client = build_client(Mode(new_page))


# 创建单选框供用户选择功能
page = st.radio(
    # 提示用户选择功能
    "选择功能",
    # 从模式中提取功能值
    [mode.value for mode in Mode],
    # 会话状态中的键
    key="page",
    # 横向显示选项
    horizontal=True,
    # 默认选中项
    index=None,
    # 隐藏标签
    label_visibility="hidden",
    # 功能改变时调用的回调函数
    on_change=page_changed,
)

# 帮助信息的文本
HELP = """
### 🎉 欢迎使用 GLM-4!

请在上方选取一个功能。每次切换功能时，将会重新加载模型并清空对话历史。

文档解读模式与 VLM 模式仅支持在第一轮传入文档或图像。
""".strip()

# 如果未选择页面，则显示帮助信息并退出
if page is None:
    st.markdown(HELP)
    exit()

# 如果选择了长上下文模式
if page == Mode.LONG_CTX:
    # 如果是第一轮
    if first_round:
        # 文件上传控件，允许多文件上传
        uploaded_files = st.file_uploader(
            "上传文件",
            # 支持的文件类型
            type=["pdf", "txt", "py", "docx", "pptx", "json", "cpp", "md"],
            # 允许上传多个文件
            accept_multiple_files=True,
        )
        # 如果有上传文件且之前未上传过
        if uploaded_files and not st.session_state.files_uploaded:
            # 存储上传文本的列表
            uploaded_texts = []
            # 遍历每个上传的文件
            for uploaded_file in uploaded_files:
                # 获取文件名
                file_name: str = uploaded_file.name
                # 生成随机文件名
                random_file_name = str(uuid4())
                # 获取文件扩展名
                file_extension = os.path.splitext(file_name)[1]
                # 创建临时文件路径
                file_path = os.path.join("/tmp", random_file_name + file_extension)
                # 写入文件数据到临时路径
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # 根据文件扩展名提取内容
                if file_name.endswith(".pdf"):
                    content = extract_pdf(file_path)
                elif file_name.endswith(".docx"):
                    content = extract_docx(file_path)
                elif file_name.endswith(".pptx"):
                    content = extract_pptx(file_path)
                else:
                    content = extract_text(file_path)
                # 格式化并存储提取的内容
                uploaded_texts.append(
                    FILE_TEMPLATE.format(file_name=file_name, file_content=content)
                )
                # 删除临时文件
                os.remove(file_path)
            # 将上传的文本存储到会话状态
            st.session_state.uploaded_texts = "\n\n".join(uploaded_texts)
            # 记录上传文件数量
            st.session_state.uploaded_file_nums = len(uploaded_files)
        else:
            # 如果没有上传文件，则清空文本和计数
            st.session_state.uploaded_texts = ""
            st.session_state.uploaded_file_nums = 0
# 如果选择了 VLM 模式
elif page == Mode.VLM:
    # 如果是第一轮
    if first_round:
        # 单文件上传控件，支持的图片类型
        uploaded_image = st.file_uploader(
            "上传图片",
            # 支持的图片类型
            type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            # 仅允许上传一个文件
            accept_multiple_files=False,
        )
        # 如果上传了图片
        if uploaded_image:
            # 读取图片数据
            data: bytes = uploaded_image.read()
            # 打开图片并转换为 RGB 格式
            image = Image.open(BytesIO(data)).convert("RGB")
            # 将上传的图片存储到会话状态
            st.session_state.uploaded_image = image
        else:
            # 如果没有上传图片，则将状态设置为 None
            st.session_state.uploaded_image = None

# 创建用户输入聊天的文本框
prompt_text = st.chat_input("Chat with GLM-4!", key="chat_input")

# 如果输入为空且未重试
if prompt_text == "" and retry == False:
    # 打印清理信息
    print("\n== Clean ==\n")
    # 清空会话历史记录
    st.session_state.history = []
    # 退出程序
    exit()

# 从会话状态获取历史记录
history: list[Conversation] = st.session_state.history

# 如果进行了重试
if retry:
    # 打印重试信息
    print("\n== Retry ==\n")
    # 初始化用户最后一次对话索引为 None
    last_user_conversation_idx = None
    # 遍历历史对话，获取每个对话的索引和内容
        for idx, conversation in enumerate(history):
            # 检查对话角色是否为用户
            if conversation.role.value == Role.USER.value:
                # 记录最后一个用户对话的索引
                last_user_conversation_idx = idx
        # 如果找到最后一个用户对话的索引
        if last_user_conversation_idx is not None:
            # 获取最后一个用户对话的内容作为提示文本
            prompt_text = history[last_user_conversation_idx].content
            # 打印新的提示文本和对应的索引
            print(f"New prompt: {prompt_text}, idx = {last_user_conversation_idx}")
            # 删除从最后一个用户对话索引到历史的所有对话
            del history[last_user_conversation_idx:]
# 遍历历史对话记录
for conversation in history:
    # 显示每个对话的内容
    conversation.show()

# 根据页面模式获取工具列表，如果模式为 ALL_TOOLS，则获取工具，否则返回空列表
tools = get_tools() if page == Mode.ALL_TOOLS else []

# 从会话状态中获取客户端实例，并指定类型为 Client
client: Client = st.session_state.client

# 主函数，接收用户输入的提示文本
def main(prompt_text: str):
    # 声明使用全局变量 client
    global client
    # 确保客户端实例不为空
    assert client is not None

# 调用主函数，传入提示文本
main(prompt_text)
```