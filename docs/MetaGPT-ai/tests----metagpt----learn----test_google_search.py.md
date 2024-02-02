# `MetaGPT\tests\metagpt\learn\test_google_search.py`

```py

# 导入异步 I/O 模块
import asyncio

# 导入数据验证模块
from pydantic import BaseModel

# 从 metagpt.learn.google_search 模块中导入 google_search 函数
from metagpt.learn.google_search import google_search

# 定义异步函数 mock_google_search
async def mock_google_search():
    # 定义输入数据模型
    class Input(BaseModel):
        input: str

    # 定义输入数据列表
    inputs = [{"input": "ai agent"}]

    # 遍历输入数据列表
    for i in inputs:
        # 根据输入数据模型创建输入对象
        seed = Input(**i)
        # 调用 google_search 函数进行搜索，并等待结果返回
        result = await google_search(seed.input)
        # 断言搜索结果不为空
        assert result != ""

# 定义测试套件函数
def test_suite():
    # 获取事件循环
    loop = asyncio.get_event_loop()
    # 创建任务并将其加入事件循环
    task = loop.create_task(mock_google_search())
    loop.run_until_complete(task)

# 如果当前脚本为主程序
if __name__ == "__main__":
    # 运行测试套件
    test_suite()

```