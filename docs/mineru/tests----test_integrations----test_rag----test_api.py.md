# `.\MinerU\tests\test_integrations\test_rag\test_api.py`

```
# 导入所需的库和模块
import json  # 用于处理 JSON 数据
import os  # 用于与操作系统交互，如文件和目录操作
import shutil  # 用于高级文件操作，如复制和删除文件
import tempfile  # 用于创建临时文件和目录

from magic_pdf.integrations.rag.api import DataReader, RagDocumentReader  # 导入 RAG API 中的数据读取器
from magic_pdf.integrations.rag.type import CategoryType  # 导入类别类型
from magic_pdf.integrations.rag.utils import \
    convert_middle_json_to_layout_elements  # 导入用于转换 JSON 的工具函数


def test_rag_document_reader():
    # setup
    unitest_dir = '/tmp/magic_pdf/unittest/integrations/rag'  # 定义单元测试目录
    os.makedirs(unitest_dir, exist_ok=True)  # 创建单元测试目录，如果已存在则不报错
    temp_output_dir = tempfile.mkdtemp(dir=unitest_dir)  # 创建临时输出目录
    os.makedirs(temp_output_dir, exist_ok=True)  # 确保临时输出目录存在

    # test
    with open('tests/test_integrations/test_rag/assets/middle.json') as f:  # 打开 JSON 文件
        json_data = json.load(f)  # 读取 JSON 数据
    res = convert_middle_json_to_layout_elements(json_data, temp_output_dir)  # 将 JSON 数据转换为布局元素

    doc = RagDocumentReader(res)  # 创建 RagDocumentReader 实例，传入转换结果
    assert len(list(iter(doc))) == 1  # 断言文档数量为 1

    page = list(iter(doc))[0]  # 获取第一个文档页面
    assert len(list(iter(page))) == 10  # 断言页面元素数量为 10
    assert len(page.get_rel_map()) == 3  # 断言关联映射的长度为 3

    item = list(iter(page))[0]  # 获取页面中的第一个元素
    assert item.category_type == CategoryType.text  # 断言该元素的类别类型为文本

    # teardown
    shutil.rmtree(temp_output_dir)  # 删除临时输出目录


def test_data_reader():
    # setup
    unitest_dir = '/tmp/magic_pdf/unittest/integrations/rag'  # 定义单元测试目录
    os.makedirs(unitest_dir, exist_ok=True)  # 创建单元测试目录，如果已存在则不报错
    temp_output_dir = tempfile.mkdtemp(dir=unitest_dir)  # 创建临时输出目录
    os.makedirs(temp_output_dir, exist_ok=True)  # 确保临时输出目录存在

    # test
    data_reader = DataReader('tests/test_integrations/test_rag/assets', 'ocr',  # 创建 DataReader 实例，指定路径和类型
                             temp_output_dir)  # 传入临时输出目录

    assert data_reader.get_documents_count() == 2  # 断言文档数量为 2
    for idx in range(data_reader.get_documents_count()):  # 遍历每个文档索引
        document = data_reader.get_document_result(idx)  # 获取指定索引的文档结果
        assert document is not None  # 断言文档结果不为 None

    # teardown
    shutil.rmtree(temp_output_dir)  # 删除临时输出目录
```