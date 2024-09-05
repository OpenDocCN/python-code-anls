# `.\yolov8\ultralytics\solutions\streamlit_inference.py`

```py
# 导入所需的库
import io  # 用于处理字节流
import time  # 用于时间相关操作

import cv2  # OpenCV库，用于图像处理
import torch  # PyTorch深度学习库

# 导入自定义函数和变量
from ultralytics.utils.checks import check_requirements  # 导入检查依赖的函数
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS  # 导入下载相关的变量


def inference(model=None):
    """使用Ultralytics YOLOv8在Streamlit应用中进行实时目标检测。"""
    
    # 检查并确保Streamlit版本符合要求，以加快Ultralytics包的加载速度
    check_requirements("streamlit>=1.29.0")  
    
    # 导入Streamlit库，仅在需要时进行导入以减少加载时间
    import streamlit as st  

    # 导入YOLOv8模型
    from ultralytics import YOLO  

    # 定义样式配置：隐藏主菜单
    menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""

    # 定义主标题配置：Ultralytics YOLOv8 Streamlit应用的标题
    main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; 
                             font-family: 'Archivo', sans-serif; margin-top:-50px;margin-bottom:20px;">
                    Ultralytics YOLOv8 Streamlit Application
                    </h1></div>"""

    # 定义副标题配置：展示实时目标检测的描述
    sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; 
                    font-family: 'Archivo', sans-serif; margin-top:-15px; margin-bottom:50px;">
                    Experience real-time object detection on your webcam with the power of Ultralytics YOLOv8! </h4>
                    </div>"""

    # 设置Streamlit页面配置：页面标题、布局、侧边栏状态
    st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide", initial_sidebar_state="auto")

    # 在页面中添加自定义的HTML样式和标题
    st.markdown(menu_style_cfg, unsafe_allow_html=True)
    st.markdown(main_title_cfg, unsafe_allow_html=True)
    st.markdown(sub_title_cfg, unsafe_allow_html=True)

    # 在侧边栏添加Ultralytics的Logo图标
    with st.sidebar:
        logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
        st.image(logo, width=250)

    # 在侧边栏添加标题：“用户配置”
    st.sidebar.title("User Configuration")

    # 添加视频源选择下拉菜单：webcam 或 video
    source = st.sidebar.selectbox(
        "Video",
        ("webcam", "video"),
    )

    vid_file_name = ""
    if source == "video":
        # 如果选择上传视频文件，则显示上传按钮
        vid_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
        if vid_file is not None:
            g = io.BytesIO(vid_file.read())  # 将上传的视频文件读取为字节流对象
            vid_location = "ultralytics.mp4"
            with open(vid_location, "wb") as out:  # 打开临时文件以写入字节
                out.write(g.read())  # 将读取的字节写入文件
            vid_file_name = "ultralytics.mp4"
    elif source == "webcam":
        vid_file_name = 0  # 如果选择使用摄像头，则设置视频源为默认摄像头

    # 添加模型选择下拉菜单：从GITHUB_ASSETS_STEMS中选择以yolov8开头的模型
    available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolov8")]
    if model:
        available_models.insert(0, model.split(".pt")[0])  # 插入模型名称（去除.pt后缀）作为选项之一

    selected_model = st.sidebar.selectbox("Model", available_models)  # 选择所需的模型
    with st.spinner("Model is downloading..."):
        model = YOLO(f"{selected_model.lower()}.pt")  # 加载 YOLO 模型
        class_names = list(model.names.values())  # 将类名字典转换为类名列表
    st.success("Model loaded successfully!")  # 在界面上显示模型加载成功的消息

    # 多选框，显示类名并获取所选类的索引
    selected_classes = st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
    selected_ind = [class_names.index(option) for option in selected_classes]

    if not isinstance(selected_ind, list):  # 确保 selected_ind 是一个列表
        selected_ind = list(selected_ind)

    enable_trk = st.sidebar.radio("Enable Tracking", ("Yes", "No"))  # 在侧边栏提供选择是否启用跟踪
    conf = float(st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01))  # 设置置信度阈值的滑块
    iou = float(st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01))  # 设置IoU阈值的滑块

    col1, col2 = st.columns(2)
    org_frame = col1.empty()  # 创建一个空白的列，用于显示原始帧
    ann_frame = col2.empty()  # 创建一个空白的列，用于显示带有注释的帧

    fps_display = st.sidebar.empty()  # 用于显示FPS的占位符

    if st.sidebar.button("Start"):  # 如果点击了“Start”按钮
        videocapture = cv2.VideoCapture(vid_file_name)  # 捕获视频

        if not videocapture.isOpened():
            st.error("Could not open webcam.")  # 如果无法打开摄像头，则显示错误消息

        stop_button = st.button("Stop")  # 停止推断的按钮

        while videocapture.isOpened():
            success, frame = videocapture.read()  # 读取视频帧
            if not success:
                st.warning("Failed to read frame from webcam. Please make sure the webcam is connected properly.")
                break  # 如果读取失败，则显示警告消息并退出循环

            prev_time = time.time()  # 记录当前时间

            # 存储模型预测结果
            if enable_trk == "Yes":
                results = model.track(frame, conf=conf, iou=iou, classes=selected_ind, persist=True)  # 调用模型进行跟踪
            else:
                results = model(frame, conf=conf, iou=iou, classes=selected_ind)  # 调用模型进行推断
            annotated_frame = results[0].plot()  # 在帧上添加注释

            # 计算模型的FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # 在界面上显示原始帧和带注释的帧
            org_frame.image(frame, channels="BGR")
            ann_frame.image(annotated_frame, channels="BGR")

            if stop_button:
                videocapture.release()  # 释放视频捕获资源
                torch.cuda.empty_cache()  # 清空CUDA内存
                st.stop()  # 停止Streamlit应用程序

            # 在侧边栏显示FPS
            fps_display.metric("FPS", f"{fps:.2f}")

        videocapture.release()  # 释放视频捕获资源

    torch.cuda.empty_cache()  # 清空CUDA内存

    cv2.destroyAllWindows()  # 销毁窗口
# 如果这个脚本被作为主程序执行（而不是被导入到其他脚本中），则执行以下代码
if __name__ == "__main__":
    # 调用名为inference的函数来进行推断任务
    inference()
```