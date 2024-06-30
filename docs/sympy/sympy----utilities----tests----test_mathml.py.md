# `D:\src\scipysrc\sympy\sympy\utilities\tests\test_mathml.py`

```
import os  # 导入操作系统相关模块
from textwrap import dedent  # 导入文本格式化模块中的dedent函数
from sympy.external import import_module  # 导入外部模块导入函数
from sympy.testing.pytest import skip  # 导入测试框架中的跳过函数
from sympy.utilities.mathml import apply_xsl  # 导入应用XSL转换函数

# 尝试导入lxml模块
lxml = import_module('lxml')

# 获取当前文件路径并拼接上特定文件名，返回绝对路径
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_xxe.py"))

# 定义测试函数test_xxe
def test_xxe():
    # 断言确保文件存在
    assert os.path.isfile(path)
    # 如果lxml模块不存在，则跳过测试
    if not lxml:
        skip("lxml not installed.")

    # 构建包含XXE漏洞的XML字符串
    mml = dedent(
        rf"""
        <!--?xml version="1.0" ?-->
        <!DOCTYPE replace [<!ENTITY ent SYSTEM "file://{path}"> ]>
        <userInfo>
        <firstName>John</firstName>
        <lastName>&ent;</lastName>
        </userInfo>
        """
    )
    
    # 指定XSL文件路径
    xsl = 'mathml/data/simple_mmlctop.xsl'

    # 应用XSL转换到XML字符串上
    res = apply_xsl(mml, xsl)

    # 断言转换后的结果符合预期格式
    assert res == \
        '<?xml version="1.0"?>\n<userInfo>\n<firstName>John</firstName>\n<lastName/>\n</userInfo>\n'
```