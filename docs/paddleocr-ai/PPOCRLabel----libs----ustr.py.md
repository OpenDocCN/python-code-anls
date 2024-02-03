# `.\PaddleOCR\PPOCRLabel\libs\ustr.py`

```py
# 版权声明，允许任何人免费获取并使用该软件，需包含版权声明和许可声明
# 导入sys模块和DEFAULT_ENCODING常量
import sys
from libs.constants import DEFAULT_ENCODING

# 定义一个函数ustr，用于处理unicode字符串在Python2和Python3之间的兼容性
def ustr(x):
    '''py2/py3 unicode helper'''

    # 如果Python版本低于3.0.0
    if sys.version_info < (3, 0, 0):
        # 导入PyQt4.QtCore模块中的QString类
        from PyQt4.QtCore import QString
        # 如果输入参数x的类型为str，则将其解码为unicode
        if type(x) == str:
            return x.decode(DEFAULT_ENCODING)
        # 如果输入参数x的类型为QString，则将其转换为unicode字符串
        if type(x) == QString:
            # 使用toUtf8()方法将QString转换为字节流，再转换为unicode字符串
            return unicode(x.toUtf8(), DEFAULT_ENCODING, 'ignore')
        # 其他情况直接返回输入参数x
        return x
    else:
        # 如果Python版本为3.0.0及以上，直接返回输入参数x
        return x
```