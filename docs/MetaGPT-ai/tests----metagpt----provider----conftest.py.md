# `MetaGPT\tests\metagpt\provider\conftest.py`

```

# 导入 pytest 模块
import pytest

# 定义一个自动使用的装饰器，用于模拟 llm_mock
@pytest.fixture(autouse=True)
def llm_mock(rsp_cache, mocker, request):
    # 一个空的装置，用于覆盖全局的 llm_mock 装置
    # 因为在 provider 文件夹中，我们想要测试特定模型的 aask 和 aask 函数
    pass

```