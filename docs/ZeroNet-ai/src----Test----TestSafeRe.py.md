# `ZeroNet\src\Test\TestSafeRe.py`

```
# 导入 SafeRe 模块
from util import SafeRe
# 导入 pytest 模块
import pytest

# 定义 TestSafeRe 类
class TestSafeRe:
    # 定义 testSafeMatch 方法
    def testSafeMatch(self):
        # 断言 SafeRe 模块中的 match 方法匹配成功
        assert SafeRe.match(
            "((js|css)/(?!all.(js|css))|data/users/.*db|data/users/.*/.*|data/archived|.*.py)",
            "js/ZeroTalk.coffee"
        )
        # 断言 SafeRe 模块中的 match 方法匹配成功
        assert SafeRe.match(".+/data.json", "data/users/1J3rJ8ecnwH2EPYa6MrgZttBNc61ACFiCj/data.json")

    # 使用 pytest.mark.parametrize 装饰器定义参数化测试
    @pytest.mark.parametrize("pattern", ["([a-zA-Z]+)*", "(a|aa)+*", "(a|a?)+", "(.*a){10}", "((?!json).)*$", r"(\w+\d+)+C"])
    # 定义 testUnsafeMatch 方法
    def testUnsafeMatch(self, pattern):
        # 使用 pytest.raises 检查 SafeRe 模块中的 match 方法是否抛出 UnsafePatternError 异常
        with pytest.raises(SafeRe.UnsafePatternError) as err:
            SafeRe.match(pattern, "aaaaaaaaaaaaaaaaaaaaaaaa!")
        # 断言异常信息中包含 "Potentially unsafe"
        assert "Potentially unsafe" in str(err.value)

    # 使用 pytest.mark.parametrize 装饰器定义参数化测试
    @pytest.mark.parametrize("pattern", ["^(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)(.*a)$"])
    # 定义 testUnsafeRepetition 方法
    def testUnsafeRepetition(self, pattern):
        # 使用 pytest.raises 检查 SafeRe 模块中的 match 方法是否抛出 UnsafePatternError 异常
        with pytest.raises(SafeRe.UnsafePatternError) as err:
            SafeRe.match(pattern, "aaaaaaaaaaaaaaaaaaaaaaaa!")
        # 断言异常信息中包含 "More than"
        assert "More than" in str(err.value)
```