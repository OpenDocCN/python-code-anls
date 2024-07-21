# `.\pytorch\torch\_dynamo\funcname_cache.py`

```py
import tokenize  # 导入 tokenize 模块，用于对 Python 源文件进行词法分析

from typing import Dict, List, Optional  # 导入必要的类型声明

cache: Dict[str, Dict[int, str]] = {}  # 定义一个全局变量 cache，用于缓存文件名到行号和函数名的映射字典


def clearcache() -> None:
    cache.clear()  # 清空缓存，即清空全局变量 cache


def _add_file(filename: str) -> None:
    try:
        with tokenize.open(filename) as f:  # 打开指定文件进行读取
            tokens = list(tokenize.generate_tokens(f.readline))  # 生成文件的 token 流
    except OSError:
        cache[filename] = {}  # 如果打开文件失败，将文件名加入缓存并初始化为空字典
        return

    # NOTE: undefined behavior if file is not valid Python source,
    # since tokenize will have undefined behavior.
    result: Dict[int, str] = {}  # 存储行号到函数名的映射关系的字典
    # current full funcname, e.g. xxx.yyy.zzz
    cur_name = ""  # 当前完整的函数名，例如 xxx.yyy.zzz
    cur_indent = 0  # 当前的缩进级别
    significant_indents: List[int] = []  # 存储重要的缩进级别的列表

    for i, token in enumerate(tokens):
        if token.type == tokenize.INDENT:  # 如果是缩进 token
            cur_indent += 1  # 缩进级别加一
        elif token.type == tokenize.DEDENT:  # 如果是取消缩进 token
            cur_indent -= 1  # 缩进级别减一
            # 可能是函数或类的结束
            if significant_indents and cur_indent == significant_indents[-1]:
                significant_indents.pop()  # 弹出最后一个重要的缩进级别
                # 删除最后一个名称部分
                cur_name = cur_name.rpartition(".")[0]
        elif (
            token.type == tokenize.NAME
            and i + 1 < len(tokens)
            and tokens[i + 1].type == tokenize.NAME
            and (token.string == "class" or token.string == "def")
        ):
            # 类或函数名紧跟在 class/def 标记之后
            significant_indents.append(cur_indent)  # 将当前缩进级别加入重要级别列表
            if cur_name:
                cur_name += "."
            cur_name += tokens[i + 1].string  # 添加当前类或函数名
        result[token.start[0]] = cur_name  # 将当前行的函数名映射存入结果字典中

    cache[filename] = result  # 将结果字典存入全局缓存中


def get_funcname(filename: str, lineno: int) -> Optional[str]:
    if filename not in cache:
        _add_file(filename)  # 如果文件名不在缓存中，调用 _add_file 函数处理该文件
    return cache[filename].get(lineno, None)  # 返回指定行号的函数名，如果不存在则返回 None
```