# `ZeroNet\src\Test\TestTranslate.py`

```py
# 导入 Translate 模块
from Translate import Translate

# 定义 TestTranslate 类
class TestTranslate:
    # 定义 testTranslateStrict 方法
    def testTranslateStrict(self):
        # 创建 Translate 对象
        translate = Translate()
        # 定义包含原始和翻译字符串的数据
        data = """
            translated = _("original")
            not_translated = "original"
        """
        # 调用 translateData 方法翻译数据
        data_translated = translate.translateData(data, {"_(original)": "translated"})
        # 断言翻译后的数据包含特定字符串
        assert 'translated = _("translated")' in data_translated
        assert 'not_translated = "original"' in data_translated

    # 定义 testTranslateStrictNamed 方法
    def testTranslateStrictNamed(self):
        # 创建 Translate 对象
        translate = Translate()
        # 定义包含原始和翻译字符串的数据
        data = """
            translated = _("original", "original named")
            translated_other = _("original", "original other named")
            not_translated = "original"
        """
        # 调用 translateData 方法翻译数据
        data_translated = translate.translateData(data, {"_(original, original named)": "translated"})
        # 断言翻译后的数据包含特定字符串
        assert 'translated = _("translated")' in data_translated
        assert 'not_translated = "original"' in data_translated

    # 定义 testTranslateUtf8 方法
    def testTranslateUtf8(self):
        # 创建 Translate 对象
        translate = Translate()
        # 定义包含需要翻译的 UTF-8 字符串的数据
        data = """
            greeting = "Hi again árvztűrőtökörfúrógép!"
        """
        # 调用 translateData 方法翻译数据
        data_translated = translate.translateData(data, {"Hi again árvztűrőtökörfúrógép!": "Üdv újra árvztűrőtökörfúrógép!"})
        # 断言翻译后的数据与预期结果相等
        assert data_translated == """
            greeting = "Üdv újra árvztűrőtökörfúrógép!"
        """
    # 定义一个测试函数，用于测试Translate类的转义功能
    def testTranslateEscape(self):
        # 创建一个Translate对象
        _ = Translate()
        # 将"Hello"翻译为"Szia"
        _["Hello"] = "Szia"

        # 简单的转义
        # 定义包含转义内容的字符串
        data = "{_[Hello]} {username}!"
        # 定义包含特殊字符的用户名
        username = "Hacker<script>alert('boom')</script>"
        # 对字符串进行翻译
        data_translated = _(data)
        # 断言"Szia"在翻译后的字符串中
        assert 'Szia' in data_translated
        # 断言"<"不在翻译后的字符串中
        assert '<' not in data_translated
        # 断言翻译后的字符串符合预期
        assert data_translated == "Szia Hacker&lt;script&gt;alert(&#x27;boom&#x27;)&lt;/script&gt;!"

        # 转义字典
        # 定义包含转义内容的字典
        user = {"username": "Hacker<script>alert('boom')</script>"}
        # 定义包含转义内容的字符串
        data = "{_[Hello]} {user[username]}!"
        # 对字符串进行翻译
        data_translated = _(data)
        # 断言"Szia"在翻译后的字符串中
        assert 'Szia' in data_translated
        # 断言"<"不在翻译后的字符串中
        assert '<' not in data_translated
        # 断言翻译后的字符串符合预期
        assert data_translated == "Szia Hacker&lt;script&gt;alert(&#x27;boom&#x27;)&lt;/script&gt;!"

        # 转义列表
        # 定义包含转义内容的列表
        users = [{"username": "Hacker<script>alert('boom')</script>"}]
        # 定义包含转义内容的字符串
        data = "{_[Hello]} {users[0][username]}!"
        # 对字符串进行翻译
        data_translated = _(data)
        # 断言"Szia"在翻译后的字符串中
        assert 'Szia' in data_translated
        # 断言"<"不在翻译后的字符串中
        assert '<' not in data_translated
        # 断言翻译后的字符串符合预期
        assert data_translated == "Szia Hacker&lt;script&gt;alert(&#x27;boom&#x27;)&lt;/script&gt;!"
```