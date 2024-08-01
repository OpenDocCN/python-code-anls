# `.\DB-GPT-src\dbgpt\rag\knowledge\tests\test_csv.py`

```py
# 导入需要的模块和函数
from unittest.mock import MagicMock, mock_open, patch
import pytest
# 从项目的dbgpt.rag.knowledge.csv模块中导入CSVKnowledge类
from dbgpt.rag.knowledge.csv import CSVKnowledge

# 模拟的CSV数据，包括id、name和age字段的数据
MOCK_CSV_DATA = "id,name,age\n1,John Doe,30\n2,Jane Smith,25\n3,Bob Johnson,40"

# 使用pytest的fixture装饰器定义mock_file_open fixture，模拟文件打开操作
@pytest.fixture
def mock_file_open():
    # 使用patch函数替换内置的open函数，并模拟打开的文件内容为MOCK_CSV_DATA
    with patch("builtins.open", mock_open(read_data=MOCK_CSV_DATA)) as mock_file:
        yield mock_file

# 使用pytest的fixture装饰器定义mock_csv_dict_reader fixture，模拟CSV的字典读取器
@pytest.fixture
def mock_csv_dict_reader():
    # 使用patch和MagicMock函数创建一个模拟的csv.DictReader对象
    with patch("csv.DictReader", MagicMock()) as mock_csv:
        # 设置模拟对象的返回值为一个迭代器，模拟CSV文件的内容
        mock_csv.return_value = iter(
            [
                {"id": "1", "name": "John Doe", "age": "30"},
                {"id": "2", "name": "Jane Smith", "age": "25"},
                {"id": "3", "name": "Bob Johnson", "age": "40"},
            ]
        )
        yield mock_csv

# 定义测试函数test_load_from_csv，测试从CSV文件加载数据的功能
def test_load_from_csv(mock_file_open, mock_csv_dict_reader):
    # 创建CSVKnowledge对象，指定文件路径为"test_data.csv"，指定源列为"name"
    knowledge = CSVKnowledge(file_path="test_data.csv", source_column="name")
    # 调用CSVKnowledge对象的_load方法加载数据
    documents = knowledge._load()
    # 断言加载的文档数量为3
    assert len(documents) == 3
```