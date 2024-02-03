# `.\AutoGPT\benchmark\agbenchmark\challenges\verticals\code\4_url_shortener\artifacts_out\test.py`

```py
# 导入单元测试模块
import unittest

# 从 url_shortener 模块中导入 retrieve_url 和 shorten_url 函数
from url_shortener import retrieve_url, shorten_url


# 定义 TestURLShortener 类，继承自 unittest.TestCase
class TestURLShortener(unittest.TestCase):
    
    # 定义测试方法 test_url_retrieval
    def test_url_retrieval(self):
        # 缩短 URL 以获取其缩短形式
        shortened_url = shorten_url("https://www.example.com")

        # 直接使用缩短后的 URL 检索原始 URL
        retrieved_url = retrieve_url(shortened_url)

        # 断言检查检索到的 URL 是否与原始 URL 匹配
        self.assertEqual(
            retrieved_url,
            "https://www.example.com",
            "Retrieved URL does not match the original!",
        )


# 如果当前脚本被直接执行，则运行单元测试
if __name__ == "__main__":
    unittest.main()
```