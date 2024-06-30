# `D:\src\scipysrc\sympy\sympy\parsing\autolev\_parse_autolev_antlr.py`

```
# 导入版本函数从 importlib.metadata 模块
from importlib.metadata import version
# 从 sympy.external 中导入 import_module 函数
from sympy.external import import_module

# 导入 AutolevParser 类型的模块 autolevparser
autolevparser = import_module('sympy.parsing.autolev._antlr.autolevparser',
                              import_kwargs={'fromlist': ['AutolevParser']})
# 导入 AutolevLexer 类型的模块 autolevlexer
autolevlexer = import_module('sympy.parsing.autolev._antlr.autolevlexer',
                             import_kwargs={'fromlist': ['AutolevLexer']})
# 导入 AutolevListener 类型的模块 autolevlistener
autolevlistener = import_module('sympy.parsing.autolev._antlr.autolevlistener',
                                import_kwargs={'fromlist': ['AutolevListener']})

# 从 autolevparser 模块获取 AutolevParser 类，如果找不到返回 None
AutolevParser = getattr(autolevparser, 'AutolevParser', None)
# 从 autolevlexer 模块获取 AutolevLexer 类，如果找不到返回 None
AutolevLexer = getattr(autolevlexer, 'AutolevLexer', None)
# 从 autolevlistener 模块获取 AutolevListener 类，如果找不到返回 None
AutolevListener = getattr(autolevlistener, 'AutolevListener', None)

# 定义解析 Autolev 代码的函数，autolev_code 是 Autolev 代码的输入，include_numeric 是一个布尔值
def parse_autolev(autolev_code, include_numeric):
    # 导入 antlr4 模块
    antlr4 = import_module('antlr4')
    # 检查 antlr4 是否导入成功并且版本是否为 4.11 开头
    if not antlr4 or not version('antlr4-python3-runtime').startswith('4.11'):
        # 如果未导入或版本不匹配，则抛出 ImportError 异常
        raise ImportError("Autolev parsing requires the antlr4 Python package,"
                          " provided by pip (antlr4-python3-runtime)"
                          " conda (antlr-python-runtime), version 4.11")
    
    try:
        # 尝试将 autolev_code 作为可读行列表读取
        l = autolev_code.readlines()
        input_stream = antlr4.InputStream("".join(l))
    except Exception:
        # 如果读取失败，则将 autolev_code 视为字符串输入流
        input_stream = antlr4.InputStream(autolev_code)

    # 如果 AutolevListener 类存在
    if AutolevListener:
        # 从内部模块 _listener_autolev_antlr 中导入 MyListener 类
        from ._listener_autolev_antlr import MyListener
        # 使用 input_stream 创建 AutolevLexer 对象
        lexer = AutolevLexer(input_stream)
        # 创建通用的 token 流
        token_stream = antlr4.CommonTokenStream(lexer)
        # 使用 token 流创建 AutolevParser 对象
        parser = AutolevParser(token_stream)
        # 解析代码生成语法树
        tree = parser.prog()
        # 创建 MyListener 实例，传入 include_numeric 参数
        my_listener = MyListener(include_numeric)
        # 创建解析树遍历器
        walker = antlr4.ParseTreeWalker()
        # 遍历语法树，并调用监听器处理节点
        walker.walk(my_listener, tree)
        # 返回监听器输出的代码的字符串表示
        return "".join(my_listener.output_code)
```