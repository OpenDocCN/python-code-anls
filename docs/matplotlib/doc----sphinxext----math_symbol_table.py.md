# `D:\src\scipysrc\matplotlib\doc\sphinxext\math_symbol_table.py`

```py
# 导入正则表达式模块
import re
# 导入docutils中的Directive类
from docutils.parsers.rst import Directive

# 导入_mathtext和_mathtext_data模块
from matplotlib import _mathtext, _mathtext_data

# 正则表达式模式，匹配黑板粗体字符
bb_pattern = re.compile("Bbb[A-Z]")
# 正则表达式模式，匹配手写体字符
scr_pattern = re.compile("scr[a-zA-Z]")
# 正则表达式模式，匹配哥特体字符
frak_pattern = re.compile("frak[A-Z]")

# 定义符号的列表，包含各类数学符号和字符
symbols = [
    # 小写希腊字母
    ["Lower-case Greek",
     4,
     (r"\alpha", r"\beta", r"\gamma",  r"\chi", r"\delta", r"\epsilon",
      r"\eta", r"\iota",  r"\kappa", r"\lambda", r"\mu", r"\nu",  r"\omega",
      r"\phi",  r"\pi", r"\psi", r"\rho",  r"\sigma",  r"\tau", r"\theta",
      r"\upsilon", r"\xi", r"\zeta",  r"\digamma", r"\varepsilon", r"\varkappa",
      r"\varphi", r"\varpi", r"\varrho", r"\varsigma",  r"\vartheta")],
    # 大写希腊字母
    ["Upper-case Greek",
     4,
     (r"\Delta", r"\Gamma", r"\Lambda", r"\Omega", r"\Phi", r"\Pi", r"\Psi",
      r"\Sigma", r"\Theta", r"\Upsilon", r"\Xi")],
    # 希伯来字母
    ["Hebrew",
     6,
     (r"\aleph", r"\beth", r"\gimel", r"\daleth")],
    # 拉丁命名字符
    ["Latin named characters",
     6,
     r"""\aa \AA \ae \AE \oe \OE \O \o \thorn \Thorn \ss \eth \dh \DH""".split()],
    # 定界符
    ["Delimiters",
     5,
     _mathtext.Parser._delims],
    # 大型符号
    ["Big symbols",
     5,
     _mathtext.Parser._overunder_symbols | _mathtext.Parser._dropsub_symbols],
    # 标准函数名
    ["Standard function names",
     5,
     {fr"\{fn}" for fn in _mathtext.Parser._function_names}],
    # 二元操作符号
    ["Binary operation symbols",
     4,
     _mathtext.Parser._binary_operators],
    # 关系符号
    ["Relation symbols",
     4,
     _mathtext.Parser._relation_symbols],
    # 箭头符号
    ["Arrow symbols",
     4,
     _mathtext.Parser._arrow_symbols],
    # 点符号
    ["Dot symbols",
     4,
     r"""\cdots \vdots \ldots \ddots \adots \Colon \therefore \because""".split()],
    # 黑板粗体字符
    ["Black-board characters",
     6,
     [fr"\{symbol}" for symbol in _mathtext_data.tex2uni
      if re.match(bb_pattern, symbol)]],
    # 手写体字符
    ["Script characters",
     6,
     [fr"\{symbol}" for symbol in _mathtext_data.tex2uni
      if re.match(scr_pattern, symbol)]],
    # 哥特体字符
    ["Fraktur characters",
     6,
     [fr"\{symbol}" for symbol in _mathtext_data.tex2uni
      if re.match(frak_pattern, symbol)]],
    # 杂项符号
    ["Miscellaneous symbols",
     4,
     r"""\neg \infty \forall \wp \exists \bigstar \angle \partial
     \nexists \measuredangle \emptyset \sphericalangle \clubsuit
     \varnothing \complement \diamondsuit \imath \Finv \triangledown
     \heartsuit \jmath \Game \spadesuit \ell \hbar \vartriangle
     \hslash \blacksquare \blacktriangle \sharp \increment
     \prime \blacktriangledown \Im \flat \backprime \Re \natural
     \circledS \P \copyright \circledR \S \yen \checkmark \$
     \cent \triangle \QED \sinewave \dag \ddag \perthousand \ac
     \lambdabar \L \l \degree \danger \maltese \clubsuitopen
     \i \hermitmatrix \sterling \nabla \mho""".split()],
]

# 定义函数，运行状态机
def run(state_machine):
    # 定义渲染符号的函数，将符号处理成 LaTeX 格式，如果忽略变体则处理特定情况
    def render_symbol(sym, ignore_variant=False):
        # 如果忽略变体并且符号不是特定的 "\varnothing" 或 "\varlrtriangle"，则替换成普通格式
        if ignore_variant and sym not in (r"\varnothing", r"\varlrtriangle"):
            sym = sym.replace(r"\var", "\\")
        # 如果符号以反斜杠开头
        if sym.startswith("\\"):
            # 去掉开头的反斜杠
            sym = sym.lstrip("\\")
            # 如果符号不在数学文本解析器的上下函数或函数名集合中
            if sym not in (_mathtext.Parser._overunder_functions |
                           _mathtext.Parser._function_names):
                # 将符号转换为对应的 Unicode 字符
                sym = chr(_mathtext_data.tex2uni[sym])
        # 如果符号在 ('\\', '|', '+', '-', '*') 中，则返回转义后的 LaTeX 格式
        return f'\\{sym}' if sym in ('\\', '|', '+', '-', '*') else sym

    # 初始化空列表 lines，用于存储生成的表格行
    lines = []

    # 遍历每个符号类别、列数、符号列表的元组
    for category, columns, syms in symbols:
        # 对符号列表进行排序，按照渲染后的符号和是否为变体的顺序
        syms = sorted(syms,
                      key=lambda sym: (render_symbol(sym, ignore_variant=True),
                                       sym.startswith(r"\var")),
                      reverse=(category == "Hebrew"))  # 如果是希伯来语，按逆序排列（右到左）

        # 生成渲染后的符号字符串列表，包含原始符号和渲染后的符号
        rendered_syms = [f"{render_symbol(sym)} ``{sym}``" for sym in syms]

        # 确定实际显示的列数，取列数和符号列表长度的较小值
        columns = min(columns, len(syms))

        # 添加符号类别作为标题
        lines.append("**%s**" % category)
        lines.append('')

        # 计算渲染后符号的最大宽度
        max_width = max(map(len, rendered_syms))

        # 生成表头，每列都有等宽的分隔线
        header = (('=' * max_width) + ' ') * columns
        lines.append(header.rstrip())

        # 按列数分段，生成每行的内容
        for part in range(0, len(rendered_syms), columns):
            # 每个部分的行内容，右对齐以符合表格格式
            row = " ".join(
                sym.rjust(max_width) for sym in rendered_syms[part:part + columns])
            lines.append(row)

        # 添加表格底部的分隔线
        lines.append(header.rstrip())
        lines.append('')

    # 将生成的符号表格插入到状态机中，标记为 "Symbol table"
    state_machine.insert_input(lines, "Symbol table")

    # 返回空列表，表示没有额外的输出
    return []
class MathSymbolTableDirective(Directive):
    has_content = False  # 指示指令是否包含内容，这里为False，表示没有内容
    required_arguments = 0  # 指令需要的必选参数个数，这里为0个
    optional_arguments = 0  # 指令可以有的可选参数个数，这里为0个
    final_argument_whitespace = False  # 指示最后一个参数是否允许空白，这里为False，表示不允许
    option_spec = {}  # 指令支持的选项，这里为空字典，表示没有额外选项

    def run(self):
        # 执行指令的主要逻辑，这里调用了run函数并传入state_machine参数
        return run(self.state_machine)


def setup(app):
    # 将MathSymbolTableDirective指令添加到app中
    app.add_directive("math_symbol_table", MathSymbolTableDirective)

    # 设置app的元数据，表明其在并行读取和写入时是安全的
    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata


if __name__ == "__main__":
    # 进行表格的验证工作

    print("SYMBOLS NOT IN STIX:")
    all_symbols = {}
    for category, columns, syms in symbols:
        if category == "Standard Function Names":
            continue
        for sym in syms:
            if len(sym) > 1:
                # 将符号添加到all_symbols字典中
                all_symbols[sym[1:]] = None
                # 如果符号不在_tex2uni中，则打印符号
                if sym[1:] not in _mathtext_data.tex2uni:
                    print(sym)

    # 添加重音符号
    all_symbols.update({v[1:]: k for k, v in _mathtext.Parser._accent_map.items()})
    all_symbols.update({v: v for v in _mathtext.Parser._wide_accents})
    print("SYMBOLS NOT IN TABLE:")
    for sym, val in _mathtext_data.tex2uni.items():
        # 如果符号不在all_symbols中，则打印符号及其Unicode值
        if sym not in all_symbols:
            print(f"{sym} = {chr(val)}")
```