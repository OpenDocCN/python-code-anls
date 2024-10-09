# `.\MinerU\demo\magic_pdf_parse_main.py`

```
# 导入操作系统模块
import os
# 导入 JSON 处理模块
import json
# 导入深拷贝模块
import copy

# 从 loguru 导入日志记录器
from loguru import logger

# 从 UNIPipe 模块导入 UNIPipe 类
from magic_pdf.pipe.UNIPipe import UNIPipe
# 从 OCRPipe 模块导入 OCRPipe 类
from magic_pdf.pipe.OCRPipe import OCRPipe
# 从 TXTPipe 模块导入 TXTPipe 类
from magic_pdf.pipe.TXTPipe import TXTPipe
# 从 DiskReaderWriter 模块导入 DiskReaderWriter 类
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
# 导入魔法 PDF 配置模型
import magic_pdf.model as model_config

# 设置使用内部模型
model_config.__use_inside_model__ = True

# todo: 设备类型选择 （？）

# 定义 json_md_dump 函数，用于将模型结果和数据写入文件
def json_md_dump(
        pipe,  # 处理管道对象
        md_writer,  # 文件写入器对象
        pdf_name,  # PDF 文件名
        content_list,  # 文本内容列表
        md_content,  # Markdown 内容
):
    # 写入模型结果到 model.json
    orig_model_list = copy.deepcopy(pipe.model_list)  # 深拷贝管道中的模型列表
    md_writer.write(  # 调用写入器写入模型数据
        content=json.dumps(orig_model_list, ensure_ascii=False, indent=4),  # 转换为 JSON 格式
        path=f"{pdf_name}_model.json"  # 指定输出文件名
    )

    # 写入中间结果到 middle.json
    md_writer.write(  # 调用写入器写入中间数据
        content=json.dumps(pipe.pdf_mid_data, ensure_ascii=False, indent=4),  # 转换为 JSON 格式
        path=f"{pdf_name}_middle.json"  # 指定输出文件名
    )

    # text文本结果写入到 conent_list.json
    md_writer.write(  # 调用写入器写入文本内容
        content=json.dumps(content_list, ensure_ascii=False, indent=4),  # 转换为 JSON 格式
        path=f"{pdf_name}_content_list.json"  # 指定输出文件名
    )

    # 写入结果到 .md 文件中
    md_writer.write(  # 调用写入器写入 Markdown 内容
        content=md_content,  # 直接写入 Markdown 内容
        path=f"{pdf_name}.md"  # 指定输出文件名
    )


# 定义 pdf_parse_main 函数，执行 PDF 转换过程
def pdf_parse_main(
        pdf_path: str,  # PDF 文件路径
        parse_method: str = 'auto',  # 解析方法，默认为 auto
        model_json_path: str = None,  # 可选模型数据文件路径
        is_json_md_dump: bool = True,  # 是否导出 JSON 和 Markdown 数据
        output_dir: str = None  # 输出结果目录
):
    """
    执行从 pdf 转换到 json、md 的过程，输出 md 和 json 文件到 pdf 文件所在的目录

    :param pdf_path: .pdf 文件的路径，可以是相对路径，也可以是绝对路径
    :param parse_method: 解析方法， 共 auto、ocr、txt 三种，默认 auto，如果效果不好，可以尝试 ocr
    :param model_json_path: 已经存在的模型数据文件，如果为空则使用内置模型，pdf 和 model_json 务必对应
    :param is_json_md_dump: 是否将解析后的数据写入到 .json 和 .md 文件中，默认 True，会将不同阶段的数据写入到不同的 .json 文件中（共3个.json文件），md内容会保存到 .md 文件中
    :param output_dir: 输出结果的目录地址，会生成一个以 pdf 文件名命名的文件夹并保存所有结果
    """
    try:
        # 获取 PDF 文件名，去掉扩展名
        pdf_name = os.path.basename(pdf_path).split(".")[0]
        # 获取 PDF 文件的父路径
        pdf_path_parent = os.path.dirname(pdf_path)

        # 如果提供了输出目录，则将输出路径与 PDF 文件名连接
        if output_dir:
            output_path = os.path.join(output_dir, pdf_name)
        else:
            # 否则，使用 PDF 文件的父路径
            output_path = os.path.join(pdf_path_parent, pdf_name)

        # 设置保存图片的路径
        output_image_path = os.path.join(output_path, 'images')

        # 获取图片的父路径，以便在 .md 和 content_list.json 中使用相对路径
        image_path_parent = os.path.basename(output_image_path)

        # 读取 PDF 文件的二进制数据
        pdf_bytes = open(pdf_path, "rb").read()

        # 如果提供了模型 JSON 路径，读取模型解析后的 PDF 文件的 JSON 数据
        if model_json_path:
            model_json = json.loads(open(model_json_path, "r", encoding="utf-8").read())
        else:
            # 如果没有提供，初始化为空列表
            model_json = []

        # 执行解析步骤
        # 创建用于写入图片和 markdown 的 DiskReaderWriter 对象
        image_writer, md_writer = DiskReaderWriter(output_image_path), DiskReaderWriter(output_path)

        # 根据解析方法选择不同的解析方式
        if parse_method == "auto":
            # 构造用于解析的 JSON 关键字
            jso_useful_key = {"_pdf_type": "", "model_list": model_json}
            # 创建自动解析管道
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
        elif parse_method == "txt":
            # 创建文本解析管道
            pipe = TXTPipe(pdf_bytes, model_json, image_writer)
        elif parse_method == "ocr":
            # 创建 OCR 解析管道
            pipe = OCRPipe(pdf_bytes, model_json, image_writer)
        else:
            # 处理未知的解析方法，记录错误并退出
            logger.error("unknown parse method, only auto, ocr, txt allowed")
            exit(1)

        # 执行分类步骤
        pipe.pipe_classify()

        # 如果没有传入模型数据，则使用内置模型进行解析
        if not model_json:
            if model_config.__use_inside_model__:
                # 调用解析方法
                pipe.pipe_analyze()
            else:
                # 如果需要模型列表却没有提供，记录错误并退出
                logger.error("need model list input")
                exit(1)

        # 执行解析步骤
        pipe.pipe_parse()

        # 保存文本和 markdown 格式的结果
        content_list = pipe.pipe_mk_uni_format(image_path_parent, drop_mode="none")
        md_content = pipe.pipe_mk_markdown(image_path_parent, drop_mode="none")

        # 如果需要 JSON 格式的 markdown 输出，则调用相应函数
        if is_json_md_dump:
            json_md_dump(pipe, md_writer, pdf_name, content_list, md_content)

    except Exception as e:
        # 捕获异常并记录错误信息
        logger.exception(e)
# 测试代码块，确保以下代码在直接运行时执行
if __name__ == '__main__':
    # 定义 PDF 文件的路径，使用原始字符串避免转义字符
    pdf_path = r"C:\Users\XYTK2\Desktop\2024-2016-gb-cd-300.pdf"
    # 调用主解析函数，传入 PDF 文件路径进行处理
    pdf_parse_main(pdf_path)
```