# `D:\src\scipysrc\sympy\sympy\printing\tableform.py`

```
# 导入需要的类和函数：Tuple、S、Symbol、SympifyError从sympy.core中导入，FunctionType从types中导入
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError

# 导入TableForm类，用于创建数据的漂亮表格表示形式
from types import FunctionType

class TableForm:
    r"""
    Create a nice table representation of data.

    Examples
    ========

    >>> from sympy import TableForm
    >>> t = TableForm([[5, 7], [4, 2], [10, 3]])
    >>> print(t)
    5  7
    4  2
    10 3

    You can use the SymPy's printing system to produce tables in any
    format (ascii, latex, html, ...).

    >>> print(t.as_latex())
    \begin{tabular}{l l}
    $5$ & $7$ \\
    $4$ & $2$ \\
    $10$ & $3$ \\
    \end{tabular}

    """

    def __repr__(self):
        # 使用sstr函数从.str模块中导入，返回实例的字符串表示形式
        from .str import sstr
        return sstr(self, order=None)

    def __str__(self):
        # 使用sstr函数从.str模块中导入，返回实例的字符串表示形式
        from .str import sstr
        return sstr(self, order=None)

    def as_matrix(self):
        """Returns the data of the table in Matrix form.

        Examples
        ========

        >>> from sympy import TableForm
        >>> t = TableForm([[5, 7], [4, 2], [10, 3]], headings='automatic')
        >>> t
          | 1  2
        --------
        1 | 5  7
        2 | 4  2
        3 | 10 3
        >>> t.as_matrix()
        Matrix([
        [ 5, 7],
        [ 4, 2],
        [10, 3]])
        """
        # 导入Matrix类从sympy.matrices.dense模块中，将数据转换为Matrix形式并返回
        from sympy.matrices.dense import Matrix
        return Matrix(self._lines)

    def as_str(self):
        # XXX obsolete ?
        # 返回实例的字符串表示形式，可能已经过时
        return str(self)

    def as_latex(self):
        # 使用latex函数从.latex模块中导入，返回实例的LaTeX格式表示形式
        from .latex import latex
        return latex(self)
    # 返回对象的字符串表示形式
    def _sympystr(self, p):
        """
        Returns the string representation of 'self'.

        Examples
        ========

        >>> from sympy import TableForm
        >>> t = TableForm([[5, 7], [4, 2], [10, 3]])
        >>> s = t.as_str()

        """
        # 初始化每列的宽度为零
        column_widths = [0] * self._w
        # 初始化存储输出行的列表
        lines = []
        
        # 遍历所有行
        for line in self._lines:
            new_line = []
            # 遍历当前行的每个元素
            for i in range(self._w):
                # 将元素转换为字符串
                s = str(line[i])
                # 如果开启了去除零操作且当前字符串为 "0"，则替换为空格
                if self._wipe_zeros and (s == "0"):
                    s = " "
                # 获取当前字符串的长度
                w = len(s)
                # 更新当前列的最大宽度
                if w > column_widths[i]:
                    column_widths[i] = w
                # 将处理后的字符串添加到新行中
                new_line.append(s)
            # 将新行添加到输出行列表中
            lines.append(new_line)

        # 检查表头第一行是否存在
        if self._headings[0]:
            # 将表头第一行的元素转换为字符串并获取最大宽度
            self._headings[0] = [str(x) for x in self._headings[0]]
            _head_width = max(len(x) for x in self._headings[0])

        # 检查表头第二行是否存在
        if self._headings[1]:
            new_line = []
            # 遍历表头第二行的每个元素
            for i in range(self._w):
                # 将元素转换为字符串
                s = str(self._headings[1][i])
                # 获取当前字符串的长度
                w = len(s)
                # 更新当前列的最大宽度
                if w > column_widths[i]:
                    column_widths[i] = w
                # 将处理后的字符串添加到新行中
                new_line.append(s)
            # 更新表头第二行为新行
            self._headings[1] = new_line

        # 定义格式化字符串列表
        format_str = []

        # 定义对齐函数
        def _align(align, w):
            return '%%%s%ss' % (
                ("-" if align == "l" else ""),
                str(w))
        
        # 生成每列的格式化字符串
        format_str = [_align(align, w) for align, w in zip(self._alignments, column_widths)]
        
        # 如果存在表头第一行，则插入对齐信息和分隔符
        if self._headings[0]:
            format_str.insert(0, _align(self._head_align, _head_width))
            format_str.insert(1, '|')
        
        # 将格式化字符串连接为最终的格式化字符串
        format_str = ' '.join(format_str) + '\n'

        # 初始化字符串列表
        s = []
        
        # 如果存在表头第二行，则处理其输出格式
        if self._headings[1]:
            d = self._headings[1]
            if self._headings[0]:
                d = [""] + d
            first_line = format_str % tuple(d)
            s.append(first_line)
            s.append("-" * (len(first_line) - 1) + "\n")
        
        # 遍历处理后的每行数据并格式化输出
        for i, line in enumerate(lines):
            d = [l if self._alignments[j] != 'c' else
                 l.center(column_widths[j]) for j, l in enumerate(line)]
            if self._headings[0]:
                l = self._headings[0][i]
                l = (l if self._head_align != 'c' else
                     l.center(_head_width))
                d = [l] + d
            s.append(format_str % tuple(d))
        
        # 返回格式化后的字符串表示形式，去除末尾的换行符
        return ''.join(s)[:-1]
    def _latex(self, printer):
        """
        Returns the LaTeX string representation of the table.

        Args:
            printer: The printer object used for converting elements to LaTeX.

        Returns:
            A LaTeX string representing the table.
        """
        # Check heading:
        if self._headings[1]:
            # Prepare new_line to hold string representations of headings[1]
            new_line = []
            for i in range(self._w):
                # Convert each element of headings[1] to string and append to new_line
                new_line.append(str(self._headings[1][i]))
            # Update self._headings[1] to store the formatted strings
            self._headings[1] = new_line

        alignments = []
        if self._headings[0]:
            # Convert elements of headings[0] to strings
            self._headings[0] = [str(x) for x in self._headings[0]]
            # Initialize alignments list with self._head_align if headings[0] exists
            alignments = [self._head_align]
        # Extend alignments list with self._alignments
        alignments.extend(self._alignments)

        # Construct the beginning of the LaTeX tabular environment with alignments
        s = r"\begin{tabular}{" + " ".join(alignments) + "}\n"

        if self._headings[1]:
            d = self._headings[1]
            if self._headings[0]:
                # If headings[0] exists, prepend an empty string to d
                d = [""] + d
            # Construct the first line of the table with headings
            first_line = " & ".join(d) + r" \\" + "\n"
            s += first_line
            s += r"\hline" + "\n"

        # Iterate through each line in self._lines
        for i, line in enumerate(self._lines):
            d = []
            for j, x in enumerate(line):
                if self._wipe_zeros and (x in (0, "0")):
                    # If self._wipe_zeros is True and x is 0 or "0", replace with empty string
                    d.append(" ")
                    continue
                f = self._column_formats[j]
                if f:
                    if isinstance(f, FunctionType):
                        # Apply the function f to element x, i, j
                        v = f(x, i, j)
                        if v is None:
                            # If function returns None, use printer._print to convert x to string
                            v = printer._print(x)
                    else:
                        # Use format string f to format element x
                        v = f % x
                    d.append(v)
                else:
                    # Use printer._print to convert x to LaTeX string enclosed in $
                    v = printer._print(x)
                    d.append("$%s$" % v)
            if self._headings[0]:
                # If headings[0] exists, prepend the ith element of headings[0] to d
                d = [self._headings[0][i]] + d
            # Construct each line of the table and append to s
            s += " & ".join(d) + r" \\" + "\n"

        # Close the LaTeX tabular environment
        s += r"\end{tabular}"
        return s
```