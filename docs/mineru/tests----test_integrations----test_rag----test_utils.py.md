# `.\MinerU\tests\test_integrations\test_rag\test_utils.py`

```
# 导入所需的模块
import json  # 处理 JSON 数据
import os  # 处理文件和目录操作
import shutil  # 提供文件操作的高级接口
import tempfile  # 生成临时文件和目录

# 从其他模块导入特定的类型和函数
from magic_pdf.integrations.rag.type import CategoryType  # 导入分类类型
from magic_pdf.integrations.rag.utils import (
    convert_middle_json_to_layout_elements, inference)  # 导入转换和推理函数


# 测试函数：将中间 JSON 转换为布局元素
def test_convert_middle_json_to_layout_elements():
    # setup
    unitest_dir = '/tmp/magic_pdf/unittest/integrations/rag'  # 设置单元测试目录
    os.makedirs(unitest_dir, exist_ok=True)  # 如果目录不存在，则创建
    temp_output_dir = tempfile.mkdtemp(dir=unitest_dir)  # 创建临时输出目录
    os.makedirs(temp_output_dir, exist_ok=True)  # 确保临时输出目录存在

    # test
    with open('tests/test_integrations/test_rag/assets/middle.json') as f:  # 打开 JSON 文件
        json_data = json.load(f)  # 读取 JSON 数据
    res = convert_middle_json_to_layout_elements(json_data, temp_output_dir)  # 调用转换函数

    assert len(res) == 1  # 验证结果长度是否为 1
    assert len(res[0].layout_dets) == 10  # 验证布局细节的数量是否为 10
    assert res[0].layout_dets[0].anno_id == 0  # 验证第一个布局细节的注释 ID 是否为 0
    assert res[0].layout_dets[0].category_type == CategoryType.text  # 验证第一个布局细节的类别类型是否为文本
    assert len(res[0].extra.element_relation) == 3  # 验证额外元素关系的数量是否为 3

    # teardown
    shutil.rmtree(temp_output_dir)  # 删除临时输出目录


# 测试函数：推理功能
def test_inference():
    
    asset_dir = 'tests/test_integrations/test_rag/assets'  # 资源目录
    # setup
    unitest_dir = '/tmp/magic_pdf/unittest/integrations/rag'  # 设置单元测试目录
    os.makedirs(unitest_dir, exist_ok=True)  # 如果目录不存在，则创建
    temp_output_dir = tempfile.mkdtemp(dir=unitest_dir)  # 创建临时输出目录
    os.makedirs(temp_output_dir, exist_ok=True)  # 确保临时输出目录存在

    # test
    res = inference(  # 调用推理函数
        asset_dir + '/one_page_with_table_image.pdf',  # 输入 PDF 文件路径
        temp_output_dir,  # 临时输出目录
        'ocr',  # 使用 OCR 模式
    )

    assert res is not None  # 验证结果不为空
    assert len(res) == 1  # 验证结果长度是否为 1
    assert len(res[0].layout_dets) == 10  # 验证布局细节的数量是否为 10
    assert res[0].layout_dets[0].anno_id == 0  # 验证第一个布局细节的注释 ID 是否为 0
    assert res[0].layout_dets[0].category_type == CategoryType.text  # 验证第一个布局细节的类别类型是否为文本
    assert len(res[0].extra.element_relation) == 3  # 验证额外元素关系的数量是否为 3

    # teardown
    shutil.rmtree(temp_output_dir)  # 删除临时输出目录
```