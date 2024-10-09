# `.\MinerU\tests\test_cli\test_cli_sdk.py`

```
# 这是一个测试 CLI 和 SDK 的模块
"""test cli and sdk."""
# 导入日志模块
import logging
# 导入操作系统模块
import os

# 导入 pytest 测试框架
import pytest
# 从配置模块中导入配置
from conf import conf
# 从公共库导入通用功能
from lib import common

# 导入模型配置
import magic_pdf.model as model_config
# 导入 UNIPipe 管道类
from magic_pdf.pipe.UNIPipe import UNIPipe
# 导入磁盘读写器类
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

# 设置内部模型的使用状态为 True
model_config.__use_inside_model__ = True
# 获取 PDF 资源路径配置
pdf_res_path = conf.conf['pdf_res_path']
# 获取代码路径配置
code_path = conf.conf['code_path']
# 获取 PDF 开发路径配置
pdf_dev_path = conf.conf['pdf_dev_path']


class TestCli:
    """测试 CLI 的类."""

    @pytest.mark.P0
    def test_pdf_auto_sdk(self):
        """自动测试 PDF SDK 的方法."""
        # 初始化一个空列表，用于存放示例名称
        demo_names = list()
        # 构建 PDF 文件夹的路径
        pdf_path = os.path.join(pdf_dev_path, 'pdf')
        # 遍历 PDF 文件夹中的所有文件
        for pdf_file in os.listdir(pdf_path):
            # 如果文件以 .pdf 结尾，则添加到示例名称列表
            if pdf_file.endswith('.pdf'):
                demo_names.append(pdf_file.split('.')[0])
        # 遍历示例名称列表
        for demo_name in demo_names:
            # 构建完整的 PDF 文件路径
            pdf_path = os.path.join(pdf_dev_path, 'pdf', f'{demo_name}.pdf')
            # 打印 PDF 文件路径
            print(pdf_path)
            # 读取 PDF 文件的字节内容
            pdf_bytes = open(pdf_path, 'rb').read()
            # 构建本地图像目录的路径
            local_image_dir = os.path.join(pdf_dev_path, 'pdf', 'images')
            # 获取图像目录的基本名称
            image_dir = str(os.path.basename(local_image_dir))
            # 创建磁盘读写器实例
            image_writer = DiskReaderWriter(local_image_dir)
            # 初始化一个空列表，用于存放模型数据
            model_json = list()
            # 定义有用的 JSON 键
            jso_useful_key = {'_pdf_type': '', 'model_list': model_json}
            # 创建 UNIPipe 管道实例
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
            # 调用分类方法
            pipe.pipe_classify()
            # 如果模型数据为空
            if len(model_json) == 0:
                # 如果使用内部模型
                if model_config.__use_inside_model__:
                    # 调用分析方法
                    pipe.pipe_analyze()
                else:
                    # 退出程序
                    exit(1)
            # 调用解析方法
            pipe.pipe_parse()
            # 创建 Markdown 内容
            md_content = pipe.pipe_mk_markdown(image_dir, drop_mode='none')
            # 构建结果路径
            dir_path = os.path.join(pdf_dev_path, 'mineru')
            # 如果目录不存在，则创建它
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            # 构建 Markdown 文件的路径
            res_path = os.path.join(dir_path, f'{demo_name}.md')
            # 删除已存在的结果文件
            common.delete_file(res_path)
            # 以写模式打开结果文件并写入内容
            with open(res_path, 'w+', encoding='utf-8') as f:
                f.write(md_content)
            # 统计文件夹内容并检查
            common.sdk_count_folders_and_check_contents(res_path)

    @pytest.mark.P0
    # 定义测试 PDF OCR SDK 的函数
        def test_pdf_ocr_sdk(self):
            """pdf sdk ocr test."""  # 文档字符串，说明这是一个 PDF SDK OCR 测试
            demo_names = list()  # 初始化一个空列表，用于存储演示文件名
            pdf_path = os.path.join(pdf_dev_path, 'pdf')  # 构建 PDF 文件夹的路径
            for pdf_file in os.listdir(pdf_path):  # 遍历 PDF 文件夹中的所有文件
                if pdf_file.endswith('.pdf'):  # 检查文件是否以 .pdf 结尾
                    demo_names.append(pdf_file.split('.')[0])  # 将文件名（去掉扩展名）添加到 demo_names 列表中
            for demo_name in demo_names:  # 遍历所有演示文件名
                pdf_path = os.path.join(pdf_dev_path, 'pdf', f'{demo_name}.pdf')  # 构建当前 PDF 文件的完整路径
                print(pdf_path)  # 输出当前 PDF 文件路径
                pdf_bytes = open(pdf_path, 'rb').read()  # 以二进制模式打开 PDF 文件并读取其内容
                local_image_dir = os.path.join(pdf_dev_path, 'pdf', 'images')  # 构建本地图像目录的路径
                image_dir = str(os.path.basename(local_image_dir))  # 获取图像目录的基本名称
                image_writer = DiskReaderWriter(local_image_dir)  # 初始化图像写入器，用于写入图像到指定目录
                model_json = list()  # 初始化空列表，用于存储模型信息
                jso_useful_key = {'_pdf_type': 'ocr', 'model_list': model_json}  # 创建一个字典，包含 PDF 类型和模型列表
                pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)  # 创建 UNIPipe 对象，用于处理 PDF
                pipe.pipe_classify()  # 调用管道分类方法
                if len(model_json) == 0:  # 检查模型列表是否为空
                    if model_config.__use_inside_model__:  # 如果配置中使用内部模型
                        pipe.pipe_analyze()  # 调用分析方法
                    else:  # 如果不使用内部模型
                        exit(1)  # 退出程序，返回代码 1
                pipe.pipe_parse()  # 调用解析方法
                md_content = pipe.pipe_mk_markdown(image_dir, drop_mode='none')  # 创建 Markdown 内容
                dir_path = os.path.join(pdf_dev_path, 'mineru')  # 构建结果存储目录的路径
                if not os.path.exists(dir_path):  # 检查结果目录是否存在
                    os.makedirs(dir_path, exist_ok=True)  # 如果不存在，则创建结果目录
                res_path = os.path.join(dir_path, f'{demo_name}.md')  # 构建结果文件的完整路径
                common.delete_file(res_path)  # 删除已存在的结果文件
                with open(res_path, 'w+', encoding='utf-8') as f:  # 以写入模式打开结果文件
                    f.write(md_content)  # 将 Markdown 内容写入结果文件
                common.sdk_count_folders_and_check_contents(res_path)  # 检查结果文件的内容和文件夹数量
    
        @pytest.mark.P0  # 标记该测试为 P0 优先级
    # 定义测试 PDF 文本 SDK 的方法
    def test_pdf_txt_sdk(self):
        # 方法文档字符串，说明这是 PDF SDK 文本测试
        """pdf sdk txt test."""
        # 创建一个空列表以存储 demo 文件名
        demo_names = list()
        # 构建 PDF 文件所在路径
        pdf_path = os.path.join(pdf_dev_path, 'pdf')
        # 遍历 PDF 文件夹中的所有文件
        for pdf_file in os.listdir(pdf_path):
            # 检查文件是否以 .pdf 结尾
            if pdf_file.endswith('.pdf'):
                # 将去掉扩展名的文件名添加到 demo_names 列表中
                demo_names.append(pdf_file.split('.')[0])
        # 遍历所有 demo 文件名
        for demo_name in demo_names:
            # 构建当前 PDF 文件的完整路径
            pdf_path = os.path.join(pdf_dev_path, 'pdf', f'{demo_name}.pdf')
            # 打印 PDF 文件路径
            print(pdf_path)
            # 以二进制模式读取 PDF 文件内容
            pdf_bytes = open(pdf_path, 'rb').read()
            # 构建本地图片目录路径
            local_image_dir = os.path.join(pdf_dev_path, 'pdf', 'images')
            # 获取图片目录的基本名称
            image_dir = str(os.path.basename(local_image_dir))
            # 创建 DiskReaderWriter 实例以处理图像写入
            image_writer = DiskReaderWriter(local_image_dir)
            # 初始化一个空的模型 JSON 列表
            model_json = list()
            # 创建一个包含 PDF 类型和模型列表的字典
            jso_useful_key = {'_pdf_type': 'txt', 'model_list': model_json}
            # 创建 UNIPipe 实例以处理 PDF 数据
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
            # 调用 classify 方法以分类 PDF 数据
            pipe.pipe_classify()
            # 检查模型 JSON 是否为空
            if len(model_json) == 0:
                # 根据配置决定是否调用分析方法
                if model_config.__use_inside_model__:
                    pipe.pipe_analyze()
                else:
                    exit(1)  # 退出程序
            # 解析 PDF 数据
            pipe.pipe_parse()
            # 创建 Markdown 内容
            md_content = pipe.pipe_mk_markdown(image_dir, drop_mode='none')
            # 构建结果目录的路径
            dir_path = os.path.join(pdf_dev_path, 'mineru')
            # 检查结果目录是否存在，若不存在则创建
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            # 构建 Markdown 结果文件的路径
            res_path = os.path.join(dir_path, f'{demo_name}.md')
            # 删除已存在的结果文件
            common.delete_file(res_path)
            # 以写入模式打开结果文件，使用 UTF-8 编码
            with open(res_path, 'w+', encoding='utf-8') as f:
                # 将 Markdown 内容写入结果文件
                f.write(md_content)
            # 检查结果文件夹及其内容
            common.sdk_count_folders_and_check_contents(res_path)
    
    # 定义测试 PDF CLI 自动化的方法
    @pytest.mark.P0
    def test_pdf_cli_auto(self):
        # 方法文档字符串，说明这是 magic_pdf CLI 自动测试
        """magic_pdf cli test auto."""
        # 创建一个空列表以存储 demo 文件名
        demo_names = []
        # 构建 PDF 文件所在路径
        pdf_path = os.path.join(pdf_dev_path, 'pdf')
        # 遍历 PDF 文件夹中的所有文件
        for pdf_file in os.listdir(pdf_path):
            # 检查文件是否以 .pdf 结尾
            if pdf_file.endswith('.pdf'):
                # 将去掉扩展名的文件名添加到 demo_names 列表中
                demo_names.append(pdf_file.split('.')[0])
        # 遍历所有 demo 文件名
        for demo_name in demo_names:
            # 构建结果路径
            res_path = os.path.join(pdf_dev_path, 'mineru')
            # 删除已存在的结果文件夹
            common.delete_file(res_path)
            # 构建执行 magic-pdf 命令的字符串
            cmd = 'magic-pdf -p %s -o %s -m %s' % (os.path.join(
                pdf_path, f'{demo_name}.pdf'), res_path, 'auto')
            # 记录命令到日志
            logging.info(cmd)
            # 使用系统命令执行 magic-pdf
            os.system(cmd)
            # 检查结果文件夹及其内容
            common.cli_count_folders_and_check_contents(
                os.path.join(res_path, demo_name, 'auto'))
    
    # 标记为优先级 P0 的测试方法
    @pytest.mark.P0
    # 定义一个测试函数，用于测试 magic_pdf CLI 的文本处理功能
    def test_pdf_clit_txt(self):
        """magic_pdf cli test txt."""  # 文档字符串，说明该测试的目的
        demo_names = []  # 初始化一个空列表，用于存储 PDF 文件的基本名称
        # 构造 PDF 文件所在目录的路径
        pdf_path = os.path.join(pdf_dev_path, 'pdf')
        # 遍历 PDF 目录中的所有文件
        for pdf_file in os.listdir(pdf_path):
            # 检查文件是否以 .pdf 结尾
            if pdf_file.endswith('.pdf'):
                # 将文件名（不包含扩展名）添加到 demo_names 列表
                demo_names.append(pdf_file.split('.')[0])
        # 遍历 demo_names 列表中的每个文件名
        for demo_name in demo_names:
            # 构造结果文件的路径
            res_path = os.path.join(pdf_dev_path, 'mineru')
            # 删除结果路径中的所有文件
            common.delete_file(res_path)
            # 构造执行 magic-pdf 命令的字符串
            cmd = 'magic-pdf -p %s -o %s -m %s' % (os.path.join(
                pdf_path, f'{demo_name}.pdf'), res_path, 'txt')
            # 记录命令到日志
            logging.info(cmd)
            # 执行命令
            os.system(cmd)
            # 检查结果文件夹的内容
            common.cli_count_folders_and_check_contents(
                os.path.join(res_path, demo_name, 'txt'))

    # 使用 pytest 标记该测试为优先级 0
    @pytest.mark.P0
    # 定义一个测试函数，用于测试 magic_pdf CLI 的 OCR 功能
    def test_pdf_clit_ocr(self):
        """magic_pdf cli test ocr."""  # 文档字符串，说明该测试的目的
        demo_names = []  # 初始化一个空列表，用于存储 PDF 文件的基本名称
        # 构造 PDF 文件所在目录的路径
        pdf_path = os.path.join(pdf_dev_path, 'pdf')
        # 遍历 PDF 目录中的所有文件
        for pdf_file in os.listdir(pdf_path):
            # 检查文件是否以 .pdf 结尾
            if pdf_file.endswith('.pdf'):
                # 将文件名（不包含扩展名）添加到 demo_names 列表
                demo_names.append(pdf_file.split('.')[0])
        # 遍历 demo_names 列表中的每个文件名
        for demo_name in demo_names:
            # 构造结果文件的路径
            res_path = os.path.join(pdf_dev_path, 'mineru')
            # 删除结果路径中的所有文件
            common.delete_file(res_path)
            # 构造执行 magic-pdf 命令的字符串
            cmd = 'magic-pdf -p %s -o %s -m %s' % (os.path.join(
                pdf_path, f'{demo_name}.pdf'), res_path, 'ocr')
            # 记录命令到日志
            logging.info(cmd)
            # 执行命令
            os.system(cmd)
            # 检查结果文件夹的内容
            common.cli_count_folders_and_check_contents(
                os.path.join(res_path, demo_name, 'ocr'))
# 判断当前模块是否是主程序运行
if __name__ == '__main__':
    # 调用 pytest 运行测试
    pytest.main()
```