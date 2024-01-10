# `MetaGPT\metagpt\utils\highlight.py`

```

# 导入需要的模块
from pygments import highlight as highlight_
from pygments.formatters import HtmlFormatter, TerminalFormatter
from pygments.lexers import PythonLexer, SqlLexer

# 定义函数，用于高亮显示代码
def highlight(code: str, language: str = "python", formatter: str = "terminal"):
    # 根据指定的语言选择相应的词法分析器
    if language.lower() == "python":
        lexer = PythonLexer()
    elif language.lower() == "sql":
        lexer = SqlLexer()
    else:
        raise ValueError(f"Unsupported language: {language}")

    # 根据指定的格式选择相应的输出格式
    if formatter.lower() == "terminal":
        formatter = TerminalFormatter()
    elif formatter.lower() == "html":
        formatter = HtmlFormatter()
    else:
        raise ValueError(f"Unsupported formatter: {formatter}")

    # 使用 Pygments 库对代码进行高亮处理
    return highlight_(code, lexer, formatter)

```