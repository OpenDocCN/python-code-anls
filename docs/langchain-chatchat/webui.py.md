# `.\Langchain-Chatchat\webui.py`

```
# 导入 streamlit 库
import streamlit as st
# 导入自定义工具函数
from webui_pages.utils import *
# 导入自定义选项菜单组件
from streamlit_option_menu import option_menu
# 导入对话页面和聊天框组件
from webui_pages.dialogue.dialogue import dialogue_page, chat_box
# 导入知识库管理页面组件
from webui_pages.knowledge_base.knowledge_base import knowledge_base_page
# 导入操作系统相关库
import os
# 导入系统相关库
import sys
# 导入版本号
from configs import VERSION
# 导入 API 地址
from server.utils import api_address

# 创建 API 请求对象
api = ApiRequest(base_url=api_address())

# 主程序入口
if __name__ == "__main__":
    # 检查是否为 lite 模式
    is_lite = "lite" in sys.argv

    # 设置页面配置
    st.set_page_config(
        "Langchain-Chatchat WebUI",
        os.path.join("img", "chatchat_icon_blue_square_v2.png"),
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/chatchat-space/Langchain-Chatchat',
            'Report a bug': "https://github.com/chatchat-space/Langchain-Chatchat/issues",
            'About': f"""欢迎使用 Langchain-Chatchat WebUI {VERSION}！"""
        }
    )

    # 定义页面字典
    pages = {
        "对话": {
            "icon": "chat",
            "func": dialogue_page,
        },
        "知识库管理": {
            "icon": "hdd-stack",
            "func": knowledge_base_page,
        },
    }

    # 在侧边栏中显示 logo 和当前版本信息
    with st.sidebar:
        st.image(
            os.path.join(
                "img",
                "logo-long-chatchat-trans-v2.png"
            ),
            use_column_width=True
        )
        st.caption(
            f"""<p align="right">当前版本：{VERSION}</p>""",
            unsafe_allow_html=True,
        )
        # 获取页面选项和图标
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        # 显示选项菜单
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            # menu_icon="chat-quote",
            default_index=default_index,
        )

    # 根据选中的页面执行相应的功能
    if selected_page in pages:
        pages[selected_page]["func"](api=api, is_lite=is_lite)
```