# `.\MinerU\app.py`

```
# 版权信息，声明版权归 Opendatalab 所有
# 导入 base64 模块，用于处理 base64 编码和解码
import base64
# 导入 os 模块，用于与操作系统交互
import os
# 导入 time 模块，用于时间相关的功能
import time
# 导入 zipfile 模块，用于 ZIP 文件的读写
import zipfile
# 导入 Path 类，用于处理文件路径
from pathlib import Path
# 导入 re 模块，用于正则表达式操作
import re

# 从 loguru 库导入 logger，用于日志记录
from loguru import logger

# 从 hash_utils 模块导入 compute_sha256 函数，用于计算文件的 SHA-256 哈希值
from magic_pdf.libs.hash_utils import compute_sha256
# 从 AbsReaderWriter 导入抽象类，作为文件读写的基类
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
# 从 DiskReaderWriter 导入具体实现类，用于磁盘上的读写操作
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
# 从 common 模块导入 do_parse 和 prepare_env 函数，用于解析和环境准备
from magic_pdf.tools.common import do_parse, prepare_env

# 在系统中执行命令安装 gradio 库
os.system("pip install gradio")
# 在系统中执行命令安装 gradio-pdf 库
os.system("pip install gradio-pdf")
# 导入 gradio 库，用于构建用户界面
import gradio as gr
# 从 gradio_pdf 导入 PDF 类，用于处理 PDF 文件
from gradio_pdf import PDF


# 定义读取文件的函数，接收文件路径作为参数
def read_fn(path):
    # 创建 DiskReaderWriter 实例，指定文件的目录
    disk_rw = DiskReaderWriter(os.path.dirname(path))
    # 读取指定文件并返回其内容，以二进制模式
    return disk_rw.read(os.path.basename(path), AbsReaderWriter.MODE_BIN)


# 定义解析 PDF 文件的函数，接收文档路径、输出目录和结束页 ID
def parse_pdf(doc_path, output_dir, end_page_id):
    # 创建输出目录，如果已经存在则不抛出异常
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 根据文档名和当前时间生成文件名
        file_name = f"{str(Path(doc_path).stem)}_{time.time()}"
        # 读取 PDF 文件的内容
        pdf_data = read_fn(doc_path)
        # 设置解析方法为自动
        parse_method = "auto"
        # 准备环境，包括生成本地图片和 Markdown 文件的目录
        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, parse_method)
        # 调用解析函数，处理 PDF 数据并生成输出
        do_parse(
            output_dir,
            file_name,
            pdf_data,
            [],
            parse_method,
            False,
            end_page_id=end_page_id,
        )
        # 返回生成的 Markdown 目录和文件名
        return local_md_dir, file_name
    except Exception as e:
        # 记录异常信息
        logger.exception(e)


# 定义将目录压缩为 ZIP 文件的函数，接收目录路径和输出 ZIP 文件路径
def compress_directory_to_zip(directory_path, output_zip_path):
    """
    压缩指定目录到一个 ZIP 文件。

    :param directory_path: 要压缩的目录路径
    :param output_zip_path: 输出的 ZIP 文件路径
    """
    try:
        # 创建 ZIP 文件，指定写入模式和压缩格式
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

            # 遍历目录中的所有文件和子目录
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    # 构建完整的文件路径
                    file_path = os.path.join(root, file)
                    # 计算相对路径，以便在 ZIP 中使用
                    arcname = os.path.relpath(file_path, directory_path)
                    # 将文件添加到 ZIP 文件中
                    zipf.write(file_path, arcname)
        # 返回成功标志
        return 0
    except Exception as e:
        # 记录异常信息
        logger.exception(e)
        # 返回失败标志
        return -1


# 定义将图片转换为 base64 格式的函数，接收图片路径
def image_to_base64(image_path):
    # 以二进制模式打开图片文件
    with open(image_path, "rb") as image_file:
        # 读取文件内容并转换为 base64 编码，返回为 UTF-8 字符串
        return base64.b64encode(image_file.read()).decode('utf-8')


# 定义替换 Markdown 文本中的图片链接为 base64 格式的函数
def replace_image_with_base64(markdown_text, image_dir_path):
    # 匹配 Markdown 中的图片标签
    pattern = r'\!\[(?:[^\]]*)\]\(([^)]+)\)'

    # 定义替换函数，处理匹配到的图片链接
    def replace(match):
        # 获取相对路径
        relative_path = match.group(1)
        # 构建完整的图片路径
        full_path = os.path.join(image_dir_path, relative_path)
        # 将图片转换为 base64 格式
        base64_image = image_to_base64(full_path)
        # 返回替换后的 Markdown 图片标签
        return f"![{relative_path}](data:image/jpeg;base64,{base64_image})"

    # 使用正则表达式替换 Markdown 文本中的图片链接
    return re.sub(pattern, replace, markdown_text)


# 定义将 PDF 文件转换为 Markdown 的函数，接收文件路径和结束页
def to_markdown(file_path, end_pages):
    # 获取解析生成的 Markdown 文件及其压缩包路径
    local_md_dir, file_name = parse_pdf(file_path, './output', end_pages - 1)
    # 根据生成的 Markdown 目录计算 SHA-256 哈希值，并构建输出 ZIP 文件路径
    archive_zip_path = os.path.join("./output", compute_sha256(local_md_dir) + ".zip")
    # 调用压缩函数，将目录压缩为 ZIP 文件
    zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
    # 检查压缩是否成功
        if zip_archive_success == 0:
            # 记录压缩成功的日志信息
            logger.info("压缩成功")
        else:
            # 记录压缩失败的日志信息
            logger.error("压缩失败")
        # 生成 Markdown 文件的完整路径
        md_path = os.path.join(local_md_dir, file_name + ".md")
        # 打开 Markdown 文件并读取其内容
        with open(md_path, 'r', encoding='utf-8') as f:
            # 将读取到的内容存储到变量
            txt_content = f.read()
        # 将 Markdown 内容中的图片替换为 Base64 编码
        md_content = replace_image_with_base64(txt_content, local_md_dir)
        # 生成转换后的 PDF 文件的完整路径
        new_pdf_path = os.path.join(local_md_dir, file_name + "_layout.pdf")
    
        # 返回转换后的 Markdown 内容、原始文本内容、压缩文件路径和新 PDF 文件路径
        return md_content, txt_content, archive_zip_path, new_pdf_path
# 定义包含 LaTeX 定界符的列表，分别用于展示和行内模式
latex_delimiters = [{"left": "$$", "right": "$$", "display": True},
                    {"left": '$', "right": '$', "display": False}]


# 初始化模型的函数
def init_model():
    # 从自定义模型模块导入 ModelSingleton 类
    from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton
    try:
        # 创建模型管理器的单例实例
        model_manager = ModelSingleton()
        # 获取文本模型，参数为 False 表示不使用预训练模型
        txt_model = model_manager.get_model(False, False)
        # 记录文本模型初始化完成的日志
        logger.info(f"txt_model init final")
        # 获取 OCR 模型，参数为 True 表示使用预训练模型
        ocr_model = model_manager.get_model(True, False)
        # 记录 OCR 模型初始化完成的日志
        logger.info(f"ocr_model init final")
        # 返回成功状态
        return 0
    except Exception as e:
        # 捕获并记录异常
        logger.exception(e)
        # 返回失败状态
        return -1


# 调用初始化模型的函数，并将返回值存储在 model_init 中
model_init = init_model()
# 记录模型初始化状态的日志
logger.info(f"model_init: {model_init}")


# 主程序入口
if __name__ == "__main__":
    # 使用 gr.Blocks 创建应用界面
    with gr.Blocks() as demo:
        # 创建一行布局
        with gr.Row():
            # 创建一个列布局，用于显示 PDF 和转换选项
            with gr.Column(variant='panel', scale=5):
                # 创建一个 Markdown 组件用于显示 PDF
                pdf_show = gr.Markdown()
                # 创建一个滑块，用于选择最大转换页数
                max_pages = gr.Slider(1, 10, 5, step=1, label="Max convert pages")
                # 创建一个行布局用于按钮
                with gr.Row() as bu_flow:
                    # 创建一个转换按钮
                    change_bu = gr.Button("Convert")
                    # 创建一个清除按钮，清除 pdf_show 的内容
                    clear_bu = gr.ClearButton([pdf_show], value="Clear")
                # 创建一个 PDF 组件，允许用户上传 PDF
                pdf_show = PDF(label="Please upload pdf", interactive=True, height=800)

            # 创建另一个列布局，用于显示转换结果
            with gr.Column(variant='panel', scale=5):
                # 创建一个文件组件，用于显示转换结果文件
                output_file = gr.File(label="convert result", interactive=False)
                # 创建选项卡组件
                with gr.Tabs():
                    # 创建一个 Markdown 渲染选项卡
                    with gr.Tab("Markdown rendering"):
                        # 创建一个 Markdown 组件用于渲染，支持 LaTeX 定界符
                        md = gr.Markdown(label="Markdown rendering", height=900, show_copy_button=True,
                                         latex_delimiters=latex_delimiters, line_breaks=True)
                    # 创建一个文本选项卡
                    with gr.Tab("Markdown text"):
                        # 创建一个文本区域组件，用于显示 Markdown 文本
                        md_text = gr.TextArea(lines=45, show_copy_button=True)
            # 绑定转换按钮的点击事件，调用 to_markdown 函数
            change_bu.click(fn=to_markdown, inputs=[pdf_show, max_pages], outputs=[md, md_text, output_file, pdf_show])
            # 绑定清除按钮的事件，清除指定组件的内容
            clear_bu.add([md, pdf_show, md_text, output_file])

    # 启动应用
    demo.launch()
```