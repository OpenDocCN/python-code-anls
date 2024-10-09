# `.\MinerU\tests\test_cli\lib\pre_clean.py`

```
"""
clean data
"""  # 文档字符串，描述该脚本的功能
import argparse  # 导入 argparse 模块，用于解析命令行参数
import os  # 导入 os 模块，用于与操作系统交互
import re  # 导入 re 模块，用于正则表达式操作
import htmltabletomd # type: ignore  # 导入 htmltabletomd 模块，忽略类型检查
import pypandoc  # 导入 pypandoc 模块，用于文档格式转换
import argparse  # 再次导入 argparse 模块，可能是多余的

# 创建命令行参数解析器，描述工具类型
parser = argparse.ArgumentParser(description="get tool type")
# 添加一个必须的字符串参数，用于输入工具名称
parser.add_argument(
    "--tool_name",
    type=str,
    required=True,
    help="input tool name",
)
# 添加一个必须的字符串参数，用于输入下载目录
parser.add_argument(
    "--download_dir",
    type=str,
    required=True,
    help="input download dir",
)
# 解析命令行参数，并将结果存储在 args 中
args = parser.parse_args()

def clean_markdown_images(content):
    """
    clean markdown images  # 文档字符串，描述该函数的功能
    """
    # 编译正则表达式，用于匹配 Markdown 图像格式
    pattern = re.compile(r'!\[[^\]]*\]\([^)]*\)', re.IGNORECASE)  
    # 使用正则表达式替换匹配的图像部分为空字符串
    cleaned_content = pattern.sub('', content)   
    # 返回清理后的内容
    return cleaned_content
   
def clean_ocrmath_photo(content):
    """
    clean ocrmath photo  # 文档字符串，描述该函数的功能
    """
    # 编译正则表达式，用于匹配 OCRMath 图片格式
    pattern = re.compile(r'\\includegraphics\[.*?\]\{.*?\}', re.IGNORECASE)  
    # 使用正则表达式替换匹配的图片部分为空字符串
    cleaned_content = pattern.sub('', content)   
    # 返回清理后的内容
    return cleaned_content

def convert_html_table_to_md(html_table):
    """
    convert html table to markdown table  # 文档字符串，描述该函数的功能
    """
    # 将 HTML 表格字符串按行分割
    lines = html_table.strip().split('\n')  
    md_table = ''  # 初始化 Markdown 表格字符串
    # 检查第一行是否存在表头标签
    if lines and '<tr>' in lines[0]:  
        in_thead = True  # 初始化表头标志
        for line in lines:  # 遍历每一行
            if '<th>' in line:  # 如果该行包含表头标签
                # 提取表头单元格内容
                cells = re.findall(r'<th>(.*?)</th>', line)  
                # 构建 Markdown 格式的表头行
                md_table += '| ' + ' | '.join(cells) + ' |\n'  
                in_thead = False  # 设置表头标志为 False
            elif '<td>' in line and not in_thead:  # 如果该行是表格内容且不是表头
                # 提取表格单元格内容
                cells = re.findall(r'<td>(.*?)</td>', line)  
                # 构建 Markdown 格式的内容行
                md_table += '| ' + ' | '.join(cells) + ' |\n'  
        # 去掉末尾的空格并添加换行符
        md_table = md_table.rstrip() + '\n'    
    # 返回构建的 Markdown 表格字符串
    return md_table  
 
def convert_latext_to_md(content):
    """
    convert latex table to markdown table  # 文档字符串，描述该函数的功能
    """
    # 查找所有 LaTeX 表格的内容
    tables = re.findall(r'\\begin\{tabular\}(.*?)\\end\{tabular\}', content, re.DOTALL)  
    placeholders = []  # 初始化占位符列表
    for table in tables:  # 遍历每个找到的表格
        # 创建占位符
        placeholder = f"<!-- TABLE_PLACEHOLDER_{len(placeholders)} -->"  
        # 创建替换字符串
        replace_str = f"\\begin{{tabular}}{table}cl\\end{{tabular}}"
        # 将原始内容中的表格替换为占位符
        content = content.replace(replace_str, placeholder)  
        try:
            # 尝试将 LaTeX 表格转换为 Markdown 格式
            pypandoc.convert_text(replace_str, format="latex", to="md", outputfile="output.md", encoding="utf-8")
        except:  # 捕获异常
            markdown_string = replace_str  # 如果转换失败，使用原始字符串
        else:  # 如果转换成功
            # 从输出文件中读取 Markdown 字符串
            markdown_string = open('output.md', 'r', encoding='utf-8').read()
        # 将占位符和转换后的 Markdown 字符串添加到列表
        placeholders.append((placeholder, markdown_string)) 
    new_content = content  # 初始化新内容为原内容
    for placeholder, md_table in placeholders:  # 遍历占位符和对应的 Markdown 表格
        # 将占位符替换为 Markdown 表格
        new_content = new_content.replace(placeholder, md_table)  
        # 写入文件  
    # 返回替换后的新内容
    return new_content

 
def convert_htmltale_to_md(content):
    """
    convert html table to markdown table  # 文档字符串，描述该函数的功能
    """
    # 查找所有 HTML 表格的内容
    tables = re.findall(r'<table>(.*?)</table>', content, re.DOTALL)  
    placeholders = []  # 初始化占位符列表
    # 遍历所有表格
        for table in tables:
            # 创建一个占位符，用于替代 HTML 表格，格式为 TABLE_PLACEHOLDER_加上当前占位符数量
            placeholder = f"<!-- TABLE_PLACEHOLDER_{len(placeholders)} -->"  
            # 将 HTML 表格替换为对应的占位符
            content = content.replace(f"<table>{table}</table>", placeholder)  
            try:
                # 尝试将 HTML 表格转换为 Markdown 表格
                convert_table = htmltabletomd.convert_table(table)
            except:
                # 如果转换失败，则保留原表格
                convert_table = table
            # 将占位符和转换后的表格添加到占位符列表中
            placeholders.append((placeholder,convert_table)) 
        # 将内容保存到新的变量中
        new_content = content  
        # 遍历所有占位符和对应的 Markdown 表格
        for placeholder, md_table in placeholders:  
            # 用 Markdown 表格替换占位符
            new_content = new_content.replace(placeholder, md_table)  
            # 写入文件  
        # 返回最终的新内容
        return new_content
# 定义一个清理数据的函数，接受产品类型和下载目录作为参数
def clean_data(prod_type, download_dir):
    """
    clean data
    """
    # 构建目标目录路径，用于存放清理后的文件
    tgt_dir = os.path.join(download_dir, prod_type, "cleaned")
    # 如果目标目录不存在，则创建该目录
    if not os.path.exists(tgt_dir):  
        os.makedirs(tgt_dir) 
    # 构建源目录路径，包含需要清理的文件
    source_dir = os.path.join(download_dir, prod_type)
    # 列出源目录中的所有文件名
    filenames = os.listdir(source_dir)
    # 遍历文件名列表
    for filename in filenames:
        # 只处理以 '.md' 结尾的文件
        if filename.endswith('.md'):
            # 构建输入文件的完整路径
            input_file = os.path.join(source_dir, filename)
            # 构建输出文件的完整路径，并添加前缀 'cleaned_'
            output_file = os.path.join(tgt_dir, "cleaned_" + filename)
            # 以只读模式打开输入文件，使用 UTF-8 编码
            with open(input_file, 'r', encoding='utf-8') as fr:
                # 读取输入文件的内容
                content = fr.read()
                # 清理 Markdown 图片内容
                new_content = clean_markdown_images(content)
                # 以写入模式打开输出文件，使用 UTF-8 编码
                with open(output_file, 'w', encoding='utf-8') as fw:
                    # 将清理后的内容写入输出文件
                    fw.write(new_content)


# 主程序入口，检查是否为主模块执行
if __name__ == '__main__':
    # 获取工具类型参数
    tool_type = args.tool_name
    # 获取下载目录参数
    download_dir = args.download_dir
    # 调用清理数据函数
    clean_data(tool_type, download_dir)
```