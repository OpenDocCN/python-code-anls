# `.\Langchain-Chatchat\webui_pages\dialogue\dialogue.py`

```
# 导入streamlit库
import streamlit as st
# 从webui_pages.utils模块中导入所有内容
from webui_pages.utils import *
# 导入streamlit_chatbox库
from streamlit_chatbox import *
# 导入streamlit_modal库
from streamlit_modal import Modal
# 导入datetime模块中的datetime类
from datetime import datetime
# 导入os模块
import os
# 导入re模块
import re
# 导入time模块
import time
# 从configs模块中导入TEMPERATURE, HISTORY_LEN, PROMPT_TEMPLATES, LLM_MODELS, DEFAULT_KNOWLEDGE_BASE, DEFAULT_SEARCH_ENGINE, SUPPORT_AGENT_MODEL
from configs import (TEMPERATURE, HISTORY_LEN, PROMPT_TEMPLATES, LLM_MODELS,
                     DEFAULT_KNOWLEDGE_BASE, DEFAULT_SEARCH_ENGINE, SUPPORT_AGENT_MODEL)
# 从server.knowledge_base.utils模块中导入LOADER_DICT
from server.knowledge_base.utils import LOADER_DICT
# 导入uuid模块
import uuid
# 从typing模块中导入List, Dict
from typing import List, Dict

# 创建一个聊天框对象
chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "chatchat_icon_blue_square_v2.png"
    )
)


def get_messages_history(history_len: int, content_in_expander: bool = False) -> List[Dict]:
    '''
    返回消息历史。
    content_in_expander控制是否返回expander元素中的内容，一般导出的时候可以选上，传入LLM的history不需要
    '''

    def filter(msg):
        # 过滤消息中的内容
        content = [x for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    return chat_box.filter_history(history_len=history_len, filter=filter)


@st.cache_data
def upload_temp_docs(files, _api: ApiRequest) -> str:
    '''
    将文件上传到临时目录，用于文件对话
    返回临时向量库ID
    '''
    return _api.upload_temp_docs(files).get("data", {}).get("id")


def parse_command(text: str, modal: Modal) -> bool:
    '''
    检查用户是否输入了自定义命令，当前支持：
    /new {session_name}。如果未提供名称，默认为“会话X”
    /del {session_name}。如果未提供名称，在会话数量>1的情况下，删除当前会话。
    /clear {session_name}。如果未提供名称，默认清除当前会话
    /help。查看命令帮助
    返回值：输入的是命令返回True，否则返回False
    '''
    # 使用正则表达式匹配输入文本中的命令和名称
    if m := re.match(r"/([^\s]+)\s*(.*)", text):
        # 提取匹配结果中的命令和名称
        cmd, name = m.groups()
        # 去除名称两端的空格
        name = name.strip()
        # 获取当前聊天框中的所有会话名称
        conv_names = chat_box.get_chat_names()
        # 根据命令执行相应操作
        if cmd == "help":
            # 打开帮助模态框
            modal.open()
        elif cmd == "new":
            # 如果名称为空，则自动生成一个会话名称
            if not name:
                i = 1
                while True:
                    name = f"会话{i}"
                    if name not in conv_names:
                        break
                    i += 1
            # 如果会话名称已存在，则提示错误
            if name in st.session_state["conversation_ids"]:
                st.error(f"该会话名称 “{name}” 已存在")
                time.sleep(1)
            else:
                # 生成唯一的会话 ID，并保存会话名称和当前会话名称
                st.session_state["conversation_ids"][name] = uuid.uuid4().hex
                st.session_state["cur_conv_name"] = name
        elif cmd == "del":
            # 如果名称为空，则使用当前会话名称
            name = name or st.session_state.get("cur_conv_name")
            # 如果只剩下一个会话，则无法删除
            if len(conv_names) == 1:
                st.error("这是最后一个会话，无法删除")
                time.sleep(1)
            # 如果名称为空或者名称不存在，则提示错误
            elif not name or name not in st.session_state["conversation_ids"]:
                st.error(f"无效的会话名称：“{name}”")
                time.sleep(1)
            else:
                # 删除会话 ID，删除聊天框中的会话名称，清空当前会话名称
                st.session_state["conversation_ids"].pop(name, None)
                chat_box.del_chat_name(name)
                st.session_state["cur_conv_name"] = ""
        elif cmd == "clear":
            # 清空聊天记录
            chat_box.reset_history(name=name or None)
        # 返回 True 表示成功执行命令
        return True
    # 返回 False 表示未匹配到有效命令
    return False
# 定义对话页面函数，接受 ApiRequest 对象和是否为轻量级模式的布尔值作为参数
def dialogue_page(api: ApiRequest, is_lite: bool = False):
    # 设置会话状态中的对话 ID 字典，如果不存在则创建
    st.session_state.setdefault("conversation_ids", {})
    # 设置当前对话的对话 ID，如果不存在则创建一个新的 UUID
    st.session_state["conversation_ids"].setdefault(chat_box.cur_chat_name, uuid.uuid4().hex)
    # 设置会话状态中的文件对话 ID，默认为 None
    st.session_state.setdefault("file_chat_id", None)
    # 获取默认的 LLM 模型
    default_model = api.get_default_llm_model()[0]

    # 如果对话框未初始化，则显示欢迎信息并初始化对话框
    if not chat_box.chat_inited:
        st.toast(
            f"欢迎使用 [`Langchain-Chatchat`](https://github.com/chatchat-space/Langchain-Chatchat) ! \n\n"
            f"当前运行的模型`{default_model}`, 您可以开始提问了."
        )
        chat_box.init_session()

    # 弹出自定义命令帮助信息的模态框
    modal = Modal("自定义命令", key="cmd_help", max_width="500")
    if modal.is_open():
        # 获取自定义命令的帮助信息并显示
        cmds = [x for x in parse_command.__doc__.split("\n") if x.strip().startswith("/")]
        st.write("\n\n".join(cmds))

    # 在应用程序重新运行时显示历史聊天消息
    chat_box.output_messages()

    # 设置聊天输入框的占位符文本
    chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter。输入/help查看自定义命令 "

    # 定义反馈函数，用于处理用户的反馈
    def on_feedback(
            feedback,
            message_id: str = "",
            history_index: int = -1,
    ):
        reason = feedback["text"]
        score_int = chat_box.set_feedback(feedback=feedback, history_index=history_index)
        # 将反馈信息发送给 API
        api.chat_feedback(message_id=message_id,
                          score=score_int,
                          reason=reason)
        st.session_state["need_rerun"] = True

    # 反馈参数设置
    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "欢迎反馈您打分的理由",
    }

    # 如果需要重新运行应用程序，则重新运行
    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()

    # 获取当前时间
    now = datetime.now()
    
    # 在侧边栏中显示内容
    with st.sidebar:
        # 创建两列布局
        cols = st.columns(2)
        export_btn = cols[0]
        # 如果按钮被点击，则清空对话历史
        if cols[1].button(
                "清空对话",
                use_container_width=True,
        ):
            chat_box.reset_history()
            st.rerun()
    # 创建一个下载按钮，用于导出对话记录
    export_btn.download_button(
        # 按钮显示的文本为“导出记录”
        "导出记录",
        # 将对话框中导出的内容转换为 Markdown 格式，并拼接成字符串
        "".join(chat_box.export2md()),
        # 设置导出文件的名称，使用当前时间作为文件名
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        # 设置导出文件的 MIME 类型为 text/markdown
        mime="text/markdown",
        # 使用容器的宽度作为按钮的宽度
        use_container_width=True,
    )
```