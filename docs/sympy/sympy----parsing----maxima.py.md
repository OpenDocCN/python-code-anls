# `D:\src\scipysrc\sympy\sympy\parsing\maxima.py`

```
import re  # 导入正则表达式模块
from sympy.concrete.products import product  # 导入 sympy 中的 product 函数
from sympy.concrete.summations import Sum  # 导入 sympy 中的 Sum 函数
from sympy.core.sympify import sympify  # 导入 sympy 中的 sympify 函数
from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入 sympy 中的 cos 和 sin 函数


class MaximaHelpers:
    def maxima_expand(expr):
        return expr.expand()  # 执行 sympy 表达式的展开操作

    def maxima_float(expr):
        return expr.evalf()  # 计算 sympy 表达式的数值结果

    def maxima_trigexpand(expr):
        return expr.expand(trig=True)  # 对 sympy 表达式进行三角函数展开

    def maxima_sum(a1, a2, a3, a4):
        return Sum(a1, (a2, a3, a4)).doit()  # 计算 sympy 表达式的求和结果

    def maxima_product(a1, a2, a3, a4):
        return product(a1, (a2, a3, a4))  # 计算 sympy 表达式的乘积结果

    def maxima_csc(expr):
        return 1/sin(expr)  # 计算 csc 函数（余割函数）的值

    def maxima_sec(expr):
        return 1/cos(expr)  # 计算 sec 函数（正割函数）的值


sub_dict = {
    'pi': re.compile(r'%pi'),  # 将 %pi 替换为 'pi' 的正则表达式对象
    'E': re.compile(r'%e'),  # 将 %e 替换为 'E' 的正则表达式对象
    'I': re.compile(r'%i'),  # 将 %i 替换为 'I' 的正则表达式对象
    '**': re.compile(r'\^'),  # 将 ^ 替换为 '**' 的正则表达式对象
    'oo': re.compile(r'\binf\b'),  # 将 \binf\b 替换为 'oo' 的正则表达式对象
    '-oo': re.compile(r'\bminf\b'),  # 将 \bminf\b 替换为 '-oo' 的正则表达式对象
    "'-'": re.compile(r'\bminus\b'),  # 将 \bminus\b 替换为 ''-' 的正则表达式对象
    'maxima_expand': re.compile(r'\bexpand\b'),  # 将 \bexpand\b 替换为 'maxima_expand' 的正则表达式对象
    'maxima_float': re.compile(r'\bfloat\b'),  # 将 \bfloat\b 替换为 'maxima_float' 的正则表达式对象
    'maxima_trigexpand': re.compile(r'\btrigexpand'),  # 将 \btrigexpand 替换为 'maxima_trigexpand' 的正则表达式对象
    'maxima_sum': re.compile(r'\bsum\b'),  # 将 \bsum\b 替换为 'maxima_sum' 的正则表达式对象
    'maxima_product': re.compile(r'\bproduct\b'),  # 将 \bproduct\b 替换为 'maxima_product' 的正则表达式对象
    'cancel': re.compile(r'\bratsimp\b'),  # 将 \bratsimp\b 替换为 'cancel' 的正则表达式对象
    'maxima_csc': re.compile(r'\bcsc\b'),  # 将 \bcsc\b 替换为 'maxima_csc' 的正则表达式对象
    'maxima_sec': re.compile(r'\bsec\b')  # 将 \bsec\b 替换为 'maxima_sec' 的正则表达式对象
}

var_name = re.compile(r'^\s*(\w+)\s*:')  # 匹配变量名的正则表达式对象


def parse_maxima(str, globals=None, name_dict={}):
    str = str.strip()  # 去除字符串两端的空白字符
    str = str.rstrip('; ')  # 去除字符串末尾的分号和空格

    for k, v in sub_dict.items():  # 遍历替换字典中的每一对键值对
        str = v.sub(k, str)  # 使用正则表达式对象 v 将字符串中的 k 替换为 v

    assign_var = None
    var_match = var_name.search(str)  # 在字符串中搜索匹配变量名的结果
    if var_match:
        assign_var = var_match.group(1)  # 提取匹配到的变量名
        str = str[var_match.end():].strip()  # 去除变量名部分后的字符串部分

    dct = MaximaHelpers.__dict__.copy()  # 复制 MaximaHelpers 类的字典
    dct.update(name_dict)  # 更新字典 dct，添加额外的名字映射
    obj = sympify(str, locals=dct)  # 使用 sympy 中的 sympify 函数解析字符串 str，并在 locals 中使用 dct 中的定义

    if assign_var and globals:
        globals[assign_var] = obj  # 如果存在全局变量字典并且存在赋值变量名，则将结果存入全局变量字典

    return obj  # 返回解析后的 sympy 表达式对象
```