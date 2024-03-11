# `.\Langchain-Chatchat\webui_pages\knowledge_base\knowledge_base.py`

```
# 导入 streamlit 库
import streamlit as st
# 从 webui_pages.utils 模块中导入所有内容
from webui_pages.utils import *
# 导入 AgGrid 和 JsCode 类
from st_aggrid import AgGrid, JsCode
# 从 st_aggrid.grid_options_builder 模块中导入 GridOptionsBuilder 类
from st_aggrid.grid_options_builder import GridOptionsBuilder
# 导入 pandas 库并重命名为 pd
import pandas as pd
# 从 server.knowledge_base.utils 模块中导入 get_file_path 和 LOADER_DICT
from server.knowledge_base.utils import get_file_path, LOADER_DICT
# 从 server.knowledge_base.kb_service.base 模块中导入 get_kb_details 和 get_kb_file_details
from server.knowledge_base.kb_service.base import get_kb_details, get_kb_file_details
# 从 typing 模块中导入 Literal, Dict, Tuple 类型
from typing import Literal, Dict, Tuple
# 从 configs 模块中导入多个常量
from configs import (kbs_config,
                     EMBEDDING_MODEL, DEFAULT_VS_TYPE,
                     CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE)
# 从 server.utils 模块中导入 list_embed_models 和 list_online_embed_models
from server.utils import list_embed_models, list_online_embed_models
# 导入 os 模块
import os
# 导入 time 模块

# 定义一个 JavaScript 代码，用于渲染单元格内容
cell_renderer = JsCode("""function(params) {if(params.value==true){return '✓'}else{return '×'}}""")

# 定义一个函数，用于配置 AgGrid 表格
def config_aggrid(
        df: pd.DataFrame,
        columns: Dict[Tuple[str, str], Dict] = {},
        selection_mode: Literal["single", "multiple", "disabled"] = "single",
        use_checkbox: bool = False,
) -> GridOptionsBuilder:
    # 从 DataFrame 创建 GridOptionsBuilder 对象
    gb = GridOptionsBuilder.from_dataframe(df)
    # 配置 "No" 列的宽度为 40
    gb.configure_column("No", width=40)
    # 遍历传入的列配置信息，设置列的属性
    for (col, header), kw in columns.items():
        gb.configure_column(col, header, wrapHeaderText=True, **kw)
    # 配置选择模式、复选框等
    gb.configure_selection(
        selection_mode=selection_mode,
        use_checkbox=use_checkbox,
        pre_selected_rows=st.session_state.get("selected_rows", [0]),
    )
    # 配置分页功能
    gb.configure_pagination(
        enabled=True,
        paginationAutoPageSize=False,
        paginationPageSize=10
    )
    return gb

# 定义一个函数，用于检查文件是否存在于本地知识库文件夹中
def file_exists(kb: str, selected_rows: List) -> Tuple[str, str]:
    """
    check whether a doc file exists in local knowledge base folder.
    return the file's name and path if it exists.
    """
    # 如果有选中的行
    if selected_rows:
        # 获取选中行的文件名
        file_name = selected_rows[0]["file_name"]
        # 获取文件路径
        file_path = get_file_path(kb, file_name)
        # 如果文件存在
        if os.path.isfile(file_path):
            return file_name, file_path
    return "", ""

# 定义一个函数，用于展示知识库页面
def knowledge_base_page(api: ApiRequest, is_lite: bool = None):
    # 尝试获取知识库详情列表，并将其转换为以知识库名称为键的字典
    try:
        kb_list = {x["kb_name"]: x for x in get_kb_details()}
    except Exception as e:
        # 如果出现异常，显示错误信息并停止程序执行
        st.error(
            "获取知识库信息错误，请检查是否已按照 `README.md` 中 `4 知识库初始化与迁移` 步骤完成初始化或迁移，或是否为数据库连接错误。")
        st.stop()
    
    # 获取知识库名称列表
    kb_names = list(kb_list.keys())

    # 检查会话状态中是否存在已选择的知识库名称，并确定其在知识库名称列表中的索引
    if "selected_kb_name" in st.session_state and st.session_state["selected_kb_name"] in kb_names:
        selected_kb_index = kb_names.index(st.session_state["selected_kb_name"])
    else:
        selected_kb_index = 0

    # 如果会话状态中不存在已选择的知识库信息，则初始化为空字符串
    if "selected_kb_info" not in st.session_state:
        st.session_state["selected_kb_info"] = ""

    # 格式化选定的知识库名称，包括知识库类型和嵌入模型信息
    def format_selected_kb(kb_name: str) -> str:
        if kb := kb_list.get(kb_name):
            return f"{kb_name} ({kb['vs_type']} @ {kb['embed_model']})"
        else:
            return kb_name

    # 创建下拉选择框，用于选择或新建知识库，并设置默认选项和格式化函数
    selected_kb = st.selectbox(
        "请选择或新建知识库：",
        kb_names + ["新建知识库"],
        format_func=format_selected_kb,
        index=selected_kb_index
    )
    # 如果选择的知识库是"新建知识库"
    if selected_kb == "新建知识库":
        # 创建一个表单
        with st.form("新建知识库"):

            # 输入新建知识库的名称
            kb_name = st.text_input(
                "新建知识库名称",
                placeholder="新知识库名称，不支持中文命名",
                key="kb_name",
            )
            # 输入知识库的简介
            kb_info = st.text_input(
                "知识库简介",
                placeholder="知识库简介，方便Agent查找",
                key="kb_info",
            )

            # 创建两列
            cols = st.columns(2)

            # 获取向量库类型列表
            vs_types = list(kbs_config.keys())
            # 选择向量库类型
            vs_type = cols[0].selectbox(
                "向量库类型",
                vs_types,
                index=vs_types.index(DEFAULT_VS_TYPE),
                key="vs_type",
            )

            # 获取嵌入模型列表
            if is_lite:
                embed_models = list_online_embed_models()
            else:
                embed_models = list_embed_models() + list_online_embed_models()

            # 选择嵌入模型
            embed_model = cols[1].selectbox(
                "Embedding 模型",
                embed_models,
                index=embed_models.index(EMBEDDING_MODEL),
                key="embed_model",
            )

            # 创建新建按钮
            submit_create_kb = st.form_submit_button(
                "新建",
                # disabled=not bool(kb_name),
                use_container_width=True,
            )

        # 如果点击了新建按钮
        if submit_create_kb:
            # 检查知识库名称是否为空
            if not kb_name or not kb_name.strip():
                st.error(f"知识库名称不能为空！")
            # 检查知识库名称是否已存在
            elif kb_name in kb_list:
                st.error(f"名为 {kb_name} 的知识库已经存在！")
            else:
                # 调用API创建知识库
                ret = api.create_knowledge_base(
                    knowledge_base_name=kb_name,
                    vector_store_type=vs_type,
                    embed_model=embed_model,
                )
                # 显示返回信息
                st.toast(ret.get("msg", " "))
                # 更新会话状态中的知识库名称和简介
                st.session_state["selected_kb_name"] = kb_name
                st.session_state["selected_kb_info"] = kb_info
                # 重新运行应用
                st.rerun()
```