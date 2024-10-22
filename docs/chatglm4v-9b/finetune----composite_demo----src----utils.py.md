# `.\chatglm4-finetune\composite_demo\src\utils.py`

```
# 从 langchain_community 的文档加载器导入 PyMuPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
# 导入处理 Word 文档的库
import docx
# 导入处理 PowerPoint 演示文稿的库
from pptx import Presentation

# 定义提取文本的函数，接收文件路径作为参数
def extract_text(path):
    # 打开文件并读取其内容
    return open(path, 'r').read()

# 定义提取 PDF 内容的函数，接收文件路径作为参数
def extract_pdf(path):
    # 使用 PyMuPDFLoader 加载 PDF 文件
    loader = PyMuPDFLoader(path)
    # 从加载器中提取数据
    data = loader.load()
    # 提取每个页面的内容并存入列表
    data = [x.page_content for x in data]
    # 将所有页面内容合并为一个字符串
    content = '\n\n'.join(data)
    # 返回合并后的内容
    return content

# 定义提取 DOCX 内容的函数，接收文件路径作为参数
def extract_docx(path):
    # 使用 docx 库打开 DOCX 文件
    doc = docx.Document(path)
    # 初始化一个空列表以存储段落内容
    data = []
    # 遍历文档中的每个段落
    for paragraph in doc.paragraphs:
        # 将段落文本添加到列表中
        data.append(paragraph.text)
    # 将所有段落内容合并为一个字符串
    content = '\n\n'.join(data)
    # 返回合并后的内容
    return content

# 定义提取 PPTX 内容的函数，接收文件路径作为参数
def extract_pptx(path):
    # 使用 Presentation 类打开 PPTX 文件
    prs = Presentation(path)
    # 初始化一个空字符串以存储文本
    text = ""
    # 遍历每个幻灯片
    for slide in prs.slides:
        # 遍历幻灯片中的每个形状
        for shape in slide.shapes:
            # 检查形状是否包含文本属性
            if hasattr(shape, "text"):
                # 将形状的文本添加到字符串中，并换行
                text += shape.text + "\n"
    # 返回收集的文本
    return text
```