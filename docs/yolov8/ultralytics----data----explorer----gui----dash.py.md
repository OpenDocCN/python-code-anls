# `.\yolov8\ultralytics\data\explorer\gui\dash.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

import sys
import time
from threading import Thread

from ultralytics import Explorer  # 导入Explorer类，用于数据探索
from ultralytics.utils import ROOT, SETTINGS  # 导入项目根目录和设置模块
from ultralytics.utils.checks import check_requirements  # 导入检查依赖的函数

check_requirements(("streamlit>=1.29.0", "streamlit-select>=0.3"))  # 检查必要的Streamlit版本和扩展

import streamlit as st  # 导入Streamlit库
from streamlit_select import image_select  # 导入图像选择组件


def _get_explorer():
    """Initializes and returns an instance of the Explorer class."""
    # 初始化Explorer实例，使用当前会话状态中的数据集和模型
    exp = Explorer(data=st.session_state.get("dataset"), model=st.session_state.get("model"))
    
    # 创建一个线程，用于生成嵌入表格，接受force和split参数
    thread = Thread(
        target=exp.create_embeddings_table,
        kwargs={"force": st.session_state.get("force_recreate_embeddings"), "split": st.session_state.get("split")},
    )
    thread.start()  # 启动线程
    
    # 创建进度条，显示生成嵌入表格的进度
    progress_bar = st.progress(0, text="Creating embeddings table...")
    while exp.progress < 1:
        time.sleep(0.1)
        progress_bar.progress(exp.progress, text=f"Progress: {exp.progress * 100}%")
    thread.join()  # 等待线程完成
    st.session_state["explorer"] = exp  # 将Explorer实例存储在会话状态中
    progress_bar.empty()  # 清空进度条


def init_explorer_form(data=None, model=None):
    """Initializes an Explorer instance and creates embeddings table with progress tracking."""
    if data is None:
        # 如果未提供数据集，则从配置文件夹中加载所有数据集的名称
        datasets = ROOT / "cfg" / "datasets"
        ds = [d.name for d in datasets.glob("*.yaml")]
    else:
        ds = [data]

    if model is None:
        # 如果未提供模型，则使用默认的YoloV8模型列表
        models = [
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
            "yolov8n-seg.pt",
            "yolov8s-seg.pt",
            "yolov8m-seg.pt",
            "yolov8l-seg.pt",
            "yolov8x-seg.pt",
            "yolov8n-pose.pt",
            "yolov8s-pose.pt",
            "yolov8m-pose.pt",
            "yolov8l-pose.pt",
            "yolov8x-pose.pt",
        ]
    else:
        models = [model]

    splits = ["train", "val", "test"]

    # 在Streamlit中创建表单，用于初始化Explorer实例
    with st.form(key="explorer_init_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.selectbox("Select dataset", ds, key="dataset")  # 数据集选择框
        with col2:
            st.selectbox("Select model", models, key="model")  # 模型选择框
        with col3:
            st.selectbox("Select split", splits, key="split")  # 数据集划分选择框
        st.checkbox("Force recreate embeddings", key="force_recreate_embeddings")  # 复选框，用于强制重新生成嵌入表
        st.form_submit_button("Explore", on_click=_get_explorer)  # 提交按钮，点击后触发_get_explorer函数


def query_form():
    """Sets up a form in Streamlit to initialize Explorer with dataset and model selection."""
    # 创建一个表单，用于初始化Explorer实例，并选择数据集和模型
    with st.form("query_form"):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.text_input(
                "Query",
                "WHERE labels LIKE '%person%' AND labels LIKE '%dog%'",
                label_visibility="collapsed",
                key="query",
            )  # 文本输入框，用于输入查询条件
        with col2:
            st.form_submit_button("Query", on_click=run_sql_query)  # 查询按钮，点击后执行run_sql_query函数


def ai_query_form():
    # 此函数尚未实现，预留用于将来扩展
    pass
    """Sets up a Streamlit form for user input to initialize Explorer with dataset and model selection."""
    # 使用 Streamlit 创建一个表单，用于用户输入以初始化 Explorer，包括数据集和模型选择
    
    with st.form("ai_query_form"):
        # 创建一个名为 "ai_query_form" 的表单
    
        col1, col2 = st.columns([0.8, 0.2])
        # 在界面上创建两列，比例为 0.8 和 0.2
    
        with col1:
            # 在第一列显示以下内容
            st.text_input("Query", "Show images with 1 person and 1 dog", label_visibility="collapsed", key="ai_query")
            # 创建一个文本输入框，用于输入查询内容，默认显示文本为 "Show images with 1 person and 1 dog"，标签不可见，键值为 "ai_query"
    
        with col2:
            # 在第二列显示以下内容
            st.form_submit_button("Ask AI", on_click=run_ai_query)
            # 创建一个提交按钮，显示文本为 "Ask AI"，点击按钮会触发名为 run_ai_query 的函数
# 初始化一个 Streamlit 表单，用于基于自定义输入进行 AI 图像查询
def find_similar_imgs(imgs):
    # 从会话状态中获取名为 "explorer" 的对象
    exp = st.session_state["explorer"]
    # 调用 explorer 对象的 get_similar 方法，使用 imgs 参数进行图像相似性查询，限制查询数量为会话状态中的 "limit"，返回类型为 "arrow"
    similar = exp.get_similar(img=imgs, limit=st.session_state.get("limit"), return_type="arrow")
    # 从查询结果中获取图像文件路径列表
    paths = similar.to_pydict()["im_file"]
    # 将查询结果的图像文件路径列表存储在会话状态中的 "imgs" 键下
    st.session_state["imgs"] = paths
    # 将查询结果对象存储在会话状态中的 "res" 键下
    st.session_state["res"] = similar


# 初始化一个 Streamlit 表单，用于基于自定义输入进行 AI 图像查询
def similarity_form(selected_imgs):
    # 输出表单标题
    st.write("Similarity Search")
    # 创建名为 "similarity_form" 的表单
    with st.form("similarity_form"):
        # 将表单分成两列，比例为 1:1
        subcol1, subcol2 = st.columns([1, 1])
        with subcol1:
            # 在第一列中添加一个数字输入框，用于设置查询结果的限制数量，初始值为 25
            st.number_input(
                "limit", min_value=None, max_value=None, value=25, label_visibility="collapsed", key="limit"
            )

        with subcol2:
            # 禁用按钮的状态取决于是否选择了至少一张图像
            disabled = not len(selected_imgs)
            # 显示当前选择的图像数量
            st.write("Selected: ", len(selected_imgs))
            # 添加提交按钮 "Search"，点击时调用 find_similar_imgs 函数，传入 selected_imgs 参数
            st.form_submit_button(
                "Search",
                disabled=disabled,
                on_click=find_similar_imgs,
                args=(selected_imgs,),
            )
        # 如果未选择任何图像，则显示错误消息
        if disabled:
            st.error("Select at least one image to search.")


# 未注释代码段
# def persist_reset_form():
#    with st.form("persist_reset"):
#        col1, col2 = st.columns([1, 1])
#        with col1:
#            st.form_submit_button("Reset", on_click=reset)
#
#        with col2:
#            st.form_submit_button("Persist", on_click=update_state, args=("PERSISTING", True))


# 执行 SQL 查询并返回结果
def run_sql_query():
    # 清除会话状态中的错误信息
    st.session_state["error"] = None
    # 获取会话状态中的查询字符串
    query = st.session_state.get("query")
    # 如果查询字符串非空
    if query.rstrip().lstrip():
        # 从会话状态中获取名为 "explorer" 的对象
        exp = st.session_state["explorer"]
        # 调用 explorer 对象的 sql_query 方法执行 SQL 查询，返回类型为 "arrow"
        res = exp.sql_query(query, return_type="arrow")
        # 将查询结果的图像文件路径列表存储在会话状态中的 "imgs" 键下
        st.session_state["imgs"] = res.to_pydict()["im_file"]
        # 将查询结果对象存储在会话状态中的 "res" 键下
        st.session_state["res"] = res


# 执行 AI 查询并更新会话状态中的查询结果
def run_ai_query():
    # 如果未设置 SETTINGS 中的 "openai_api_key"，则设置错误信息并返回
    if not SETTINGS["openai_api_key"]:
        st.session_state["error"] = (
            'OpenAI API key not found in settings. Please run yolo settings openai_api_key="..."'
        )
        return
    # 导入 pandas 库，以便更快地导入 'import ultralytics'
    import pandas  # scope for faster 'import ultralytics'

    # 清除会话状态中的错误信息
    st.session_state["error"] = None
    # 获取会话状态中的 AI 查询字符串
    query = st.session_state.get("ai_query")
    # 如果查询字符串非空
    if query.rstrip().lstrip():
        # 从会话状态中获取名为 "explorer" 的对象
        exp = st.session_state["explorer"]
        # 调用 explorer 对象的 ask_ai 方法执行 AI 查询
        res = exp.ask_ai(query)
        # 如果返回的结果不是 pandas.DataFrame 或结果为空，则设置错误信息并返回
        if not isinstance(res, pandas.DataFrame) or res.empty:
            st.session_state["error"] = "No results found using AI generated query. Try another query or rerun it."
            return
        # 将查询结果中的图像文件路径列表存储在会话状态中的 "imgs" 键下
        st.session_state["imgs"] = res["im_file"].to_list()
        # 将查询结果对象存储在会话状态中的 "res" 键下
        st.session_state["res"] = res


# 重置探索器的初始状态，清除会话变量
def reset_explorer():
    # 清除会话状态中的 "explorer" 对象
    st.session_state["explorer"] = None
    # 清除会话状态中的 "imgs" 键
    st.session_state["imgs"] = None
    # 清除会话状态中的错误信息
    st.session_state["error"] = None


# 未注释代码段
# def utralytics_explorer_docs_callback():
    """Resets the explorer to its initial state by clearing session variables."""
    # 使用 streamlit 库的 container 组件创建一个带边框的容器
    with st.container(border=True):
        # 在容器中显示图片，图片来源于指定的 URL，设置宽度为 100 像素
        st.image(
            "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg",
            width=100,
        )
        # 在容器中使用 Markdown 格式显示文本，文本包含 HTML 元素
        st.markdown(
            "<p>This demo is built using Ultralytics Explorer API. Visit <a href='https://docs.ultralytics.com/datasets/explorer/'>API docs</a> to try examples & learn more</p>",
            unsafe_allow_html=True,
            help=None,
        )
        # 在容器中添加一个链接按钮，链接到 Ultralytics Explorer API 文档页面
        st.link_button("Ultralytics Explorer API", "https://docs.ultralytics.com/datasets/explorer/")
def layout(data=None, model=None):
    """Resets explorer session variables and provides documentation with a link to API docs."""
    # 设置页面配置为宽布局，侧边栏初始状态为折叠
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    # 在页面中心显示标题，支持HTML标记
    st.markdown("<h1 style='text-align: center;'>Ultralytics Explorer Demo</h1>", unsafe_allow_html=True)

    # 如果会话状态中不存在"explorer"变量，初始化探索器表单并返回
    if st.session_state.get("explorer") is None:
        init_explorer_form(data, model)
        return

    # 显示返回到选择数据集的按钮，点击时调用reset_explorer函数
    st.button(":arrow_backward: Select Dataset", on_click=reset_explorer)
    
    # 获取会话状态中的"explorer"对象
    exp = st.session_state.get("explorer")
    
    # 创建两列布局，比例为0.75和0.25，列之间的间隙为"small"
    col1, col2 = st.columns([0.75, 0.25], gap="small")
    
    # 初始化一个空列表imgs，用于存储图像数据
    imgs = []
    
    # 如果会话状态中存在"error"变量，显示错误信息
    if st.session_state.get("error"):
        st.error(st.session_state["error"])
    
    # 如果会话状态中存在"imgs"变量，将imgs设置为该变量的值
    elif st.session_state.get("imgs"):
        imgs = st.session_state.get("imgs")
    
    # 否则，从exp对象的表中获取图像文件列表并存储到imgs中
    else:
        imgs = exp.table.to_lance().to_table(columns=["im_file"]).to_pydict()["im_file"]
        # 将结果表存储到会话状态的"res"变量中
        st.session_state["res"] = exp.table.to_arrow()
    
    # 计算总图像数量和已选择的图像数量，初始化selected_imgs为空列表
    total_imgs, selected_imgs = len(imgs), []
    with col1:
        # 列1的内容

        # 拆分子列，共5列
        subcol1, subcol2, subcol3, subcol4, subcol5 = st.columns(5)
        
        with subcol1:
            # 在子列1中显示文本
            st.write("Max Images Displayed:")

        with subcol2:
            # 在子列2中获取用户输入的最大显示图片数量
            num = st.number_input(
                "Max Images Displayed",
                min_value=0,
                max_value=total_imgs,
                value=min(500, total_imgs),
                key="num_imgs_displayed",
                label_visibility="collapsed",
            )
        
        with subcol3:
            # 在子列3中显示文本
            st.write("Start Index:")

        with subcol4:
            # 在子列4中获取用户输入的起始索引
            start_idx = st.number_input(
                "Start Index",
                min_value=0,
                max_value=total_imgs,
                value=0,
                key="start_index",
                label_visibility="collapsed",
            )
        
        with subcol5:
            # 在子列5中创建一个重置按钮，并在点击时执行重置操作
            reset = st.button("Reset", use_container_width=False, key="reset")
            if reset:
                # 重置图像数据的会话状态
                st.session_state["imgs"] = None
                # 实验性重新运行应用以应用更改
                st.experimental_rerun()

        # 显示查询表单和AI查询表单
        query_form()
        ai_query_form()

        if total_imgs:
            # 初始化变量
            labels, boxes, masks, kpts, classes = None, None, None, None, None
            # 获取当前任务类型
            task = exp.model.task
            
            # 如果用户选择显示标签
            if st.session_state.get("display_labels"):
                # 从结果中获取标签、边界框、掩模、关键点和类别信息
                labels = st.session_state.get("res").to_pydict()["labels"][start_idx : start_idx + num]
                boxes = st.session_state.get("res").to_pydict()["bboxes"][start_idx : start_idx + num]
                masks = st.session_state.get("res").to_pydict()["masks"][start_idx : start_idx + num]
                kpts = st.session_state.get("res").to_pydict()["keypoints"][start_idx : start_idx + num]
                classes = st.session_state.get("res").to_pydict()["cls"][start_idx : start_idx + num]
            
            # 获取显示的图像
            imgs_displayed = imgs[start_idx : start_idx + num]
            
            # 显示选定的图像，包括相关信息
            selected_imgs = image_select(
                f"Total samples: {total_imgs}",
                images=imgs_displayed,
                use_container_width=False,
                labels=labels,
                classes=classes,
                bboxes=boxes,
                masks=masks if task == "segment" else None,
                kpts=kpts if task == "pose" else None,
            )

    with col2:
        # 在列2中显示相似性表单
        similarity_form(selected_imgs)
        
        # 显示一个复选框，控制是否显示标签
        st.checkbox("Labels", value=False, key="display_labels")
        
        # 调用用于用户行为分析的文档回调函数
        utralytics_explorer_docs_callback()
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 使用命令行参数构建一个字典，键值对为偶数索引的参数作为键，奇数索引的参数作为对应的值
    kwargs = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    # 将构建的参数字典作为关键字参数传递给名为 layout 的函数
    layout(**kwargs)
```