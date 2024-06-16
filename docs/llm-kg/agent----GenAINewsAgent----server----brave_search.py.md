# `.\agent\GenAINewsAgent\server\brave_search.py`

```
# 引入AsyncBrave类，这是一个异步搜索客户端
# 从brave模块中导入AsyncBrave类
from brave import AsyncBrave
# 从configs模块中导入BRAVE_API_KEY常量
from configs import BRAVE_API_KEY
# 导入类型提示模块，用于声明函数参数和返回类型
from typing import List, Dict, Union

# 定义BraveSearch类
class BraveSearch:

    # 初始化方法，设置Brave客户端对象
    def __init__(self) -> None:
        self.brave_client = AsyncBrave(BRAVE_API_KEY)

    # 解析搜索结果的私有方法
    def __parse_results__(self, search_results: Dict):
        # 指定要处理的结果类型
        result_keys = ["web", "news"]
        results = []
        # 遍历结果类型
        for key in result_keys:
            # 检查search_results对象是否有指定的属性
            if hasattr(search_results, key):
                # 获取指定类型（web或news）的结果集合
                for result in getattr(getattr(search_results, key), "results"):
                    # 提取每个搜索结果的标题、URL、描述、页面年龄、年龄和是否为突发新闻等信息
                    results += [{
                        "title": result.title,
                        "url": result.url,
                        "description": result.description,
                        "page_age": result.page_age,
                        "age": result.age,
                        "is_breaking_news": getattr(result, "breaking", False)
                    }]
        # 返回处理后的结果列表
        return results

    # 异步调用方法，执行搜索操作并解析结果
    async def __call__(self, query: str, **kwargs):
        # 使用Brave客户端执行异步搜索
        search_results = await self.brave_client.search(query,
                                                        count=3,
                                                        **kwargs)
        # 调用内部方法解析搜索结果
        results = self.__parse_results__(search_results)
        # 返回解析后的结果列表
        return results

# 主程序入口
if __name__ == "__main__":
    import asyncio
    # 创建BraveSearch对象
    bs = BraveSearch()
    # 运行异步事件循环，并执行搜索操作
    asyncio.run(bs("lok sabha elections 2024"))
```