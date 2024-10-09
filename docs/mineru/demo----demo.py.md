# `.\MinerU\demo\demo.py`

```
# 导入操作系统相关的模块
import os
# 导入 JSON 处理模块
import json

# 从 loguru 导入日志记录器
from loguru import logger

# 从 magic_pdf.pipe 导入 UNIPipe 类
from magic_pdf.pipe.UNIPipe import UNIPipe
# 从 magic_pdf.rw 导入 DiskReaderWriter 类
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

# 导入 magic_pdf.model 模块作为模型配置
import magic_pdf.model as model_config 
# 设置使用内置模型解析
model_config.__use_inside_model__ = True

# 尝试执行以下代码块，如果出现异常则捕获
try:
    # 获取当前脚本所在的目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 定义示例名称
    demo_name = "demo1"
    # 拼接 PDF 文件的完整路径
    pdf_path = os.path.join(current_script_dir, f"{demo_name}.pdf")
    # 拼接模型配置 JSON 文件的完整路径
    model_path = os.path.join(current_script_dir, f"{demo_name}.json")
    # 以二进制模式读取 PDF 文件的内容
    pdf_bytes = open(pdf_path, "rb").read()
    # model_json = json.loads(open(model_path, "r", encoding="utf-8").read())
    # 初始化 model_json 为一个空列表，表示使用内置模型解析
    model_json = []  # model_json传空list使用内置模型解析
    # 定义一个包含 PDF 类型和模型列表的字典
    jso_useful_key = {"_pdf_type": "", "model_list": model_json}
    # 拼接本地图像目录的完整路径
    local_image_dir = os.path.join(current_script_dir, 'images')
    # 获取图像目录的基本名称
    image_dir = str(os.path.basename(local_image_dir))
    # 创建 DiskReaderWriter 实例以处理图像目录
    image_writer = DiskReaderWriter(local_image_dir)
    # 创建 UNIPipe 实例以处理 PDF 数据和其他参数
    pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
    # 调用分类处理方法
    pipe.pipe_classify()
    """如果没有传入有效的模型数据，则使用内置model解析"""
    # 检查 model_json 是否为空
    if len(model_json) == 0:
        # 如果设置使用内置模型
        if model_config.__use_inside_model__:
            # 调用分析处理方法
            pipe.pipe_analyze()
        else:
            # 记录错误信息并退出
            logger.error("need model list input")
            exit(1)
    # 调用解析处理方法
    pipe.pipe_parse()
    # 生成 Markdown 内容
    md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
    # 以 UTF-8 编码写入 Markdown 文件
    with open(f"{demo_name}.md", "w", encoding="utf-8") as f:
        f.write(md_content)
# 捕获并记录异常
except Exception as e:
    logger.exception(e)
```