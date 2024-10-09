# `.\MinerU\tests\test_para\test_hyphen_at_line_end.py`

```
# 从 magic_pdf.dict2md.ocr_mkcontent 模块导入 __is_hyphen_at_line_end 函数
from magic_pdf.dict2md.ocr_mkcontent import __is_hyphen_at_line_end

# 定义测试函数，用于检查行尾是否是连字符
def test_hyphen_at_line_end():
    """
    测试行尾是不是一个连字符
    """
    # 定义测试用例，预期为正的情况
    test_cases_ok = [
        "I am zhang-",          # 行尾是连字符，应该通过测试
        "you are zhang- ",      # 行尾是连字符，后面有空格，应该通过测试
        "math-",                # 行尾是连字符，应该通过测试
        "This is a TEST-",      # 行尾是连字符，应该通过测试
        "This is a TESTing-",   # 行尾是连字符，应该通过测试
        "美国人 hello-",        # 行尾是连字符，应该通过测试
    ]
    # 定义测试用例，预期为负的情况
    test_cases_bad = [
        "This is a TEST$-",     # 行尾是特殊字符，不是连字符，应该不通过测试
        "This is a TEST21-",    # 行尾是数字，不是连字符，应该不通过测试
        "中国人-",              # 行尾是中文，不是连字符，应该不通过测试
        "美国人 hello人-",      # 行尾是中文，不是连字符，应该不通过测试
        "this is 123-",         # 行尾是数字，不是连字符，应该不通过测试
    ]
    # 遍历预期为正的测试用例，逐一进行断言
    for test_case in test_cases_ok:
        assert __is_hyphen_at_line_end(test_case)  # 验证行尾是连字符的断言

    # 遍历预期为负的测试用例，逐一进行断言
    for test_case in test_cases_bad:
        assert not __is_hyphen_at_line_end(test_case)  # 验证行尾不是连字符的断言
```