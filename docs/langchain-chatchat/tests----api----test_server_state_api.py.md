# `.\Langchain-Chatchat\tests\api\test_server_state_api.py`

```
# 导入必要的模块
import sys
from pathlib import Path
# 获取当前文件的父目录的父目录的父目录作为根路径
root_path = Path(__file__).parent.parent.parent
# 将根路径添加到系统路径中
sys.path.append(str(root_path))

# 导入自定义模块
from webui_pages.utils import ApiRequest

# 导入 pytest 模块
import pytest
# 导入 pprint 模块
from pprint import pprint
# 导入 List 类型提示
from typing import List

# 创建 ApiRequest 实例
api = ApiRequest()

# 测试获取默认的 LLM 模型
def test_get_default_llm():
    # 调用 ApiRequest 实例的方法获取默认的 LLM 模型
    llm = api.get_default_llm_model()
    
    # 打印获取到的 LLM 模型
    print(llm)
    # 断言获取到的 LLM 模型是元组类型
    assert isinstance(llm, tuple)
    # 断言元组的第一个元素是字符串类型，第二个元素是布尔类型
    assert isinstance(llm[0], str) and isinstance(llm[1], bool)

# 测试获取服务器配置信息
def test_server_configs():
    # 调用 ApiRequest 实例的方法获取服务器配置信息
    configs = api.get_server_configs()
    # 使用 pprint 打印服务器配置信息，限制打印深度为2
    pprint(configs, depth=2)

    # 断言获取到的配置信息是字典类型
    assert isinstance(configs, dict)
    # 断言配置信息字典不为空
    assert len(configs) > 0

# 测试列出搜索引擎
def test_list_search_engines():
    # 调用 ApiRequest 实例的方法列出搜索引擎
    engines = api.list_search_engines()
    # 使用 pprint 打印搜索引擎列表
    pprint(engines)

    # 断言获取到的搜索引擎列表是列表类型
    assert isinstance(engines, list)
    # 断言搜索引擎列表不为空
    assert len(engines) > 0

# 使用 pytest 的参数化装饰器，测试获取提示模板
@pytest.mark.parametrize("type", ["llm_chat", "agent_chat"])
def test_get_prompt_template(type):
    # 打印提示模板类型
    print(f"prompt template for: {type}")
    # 调用 ApiRequest 实例的方法获取提示模板
    template = api.get_prompt_template(type=type)

    # 打印获取到的提示模板
    print(template)
    # 断言获取到的提示模板是字符串类型
    assert isinstance(template, str)
    # 断言提示模板不为空
    assert len(template) > 0
```