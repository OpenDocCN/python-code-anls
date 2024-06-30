# `D:\src\scipysrc\sympy\sympy\printing\pretty\stringpict.py`

```
"""Prettyprinter by Jurjen Bos.
(I hate spammers: mail me at pietjepuk314 at the reverse of ku.oc.oohay).
All objects have a method that create a "stringPict",
that can be used in the str method for pretty printing.

Updates by Jason Gedge (email <my last name> at cs mun ca)
    - terminal_string() method
    - minor fixes and changes (mostly to prettyForm)

TODO:
    - Allow left/center/right alignment options for above/below and
      top/center/bottom alignment options for left/right
"""

# 导入 shutil 库
import shutil

# 导入 pretty_symbology 模块中的特定符号和函数
from .pretty_symbology import hobj, vobj, xsym, xobj, pretty_use_unicode, line_width, center

# 导入 sympy.utilities.exceptions 模块中的异常处理函数
from sympy.utilities.exceptions import sympy_deprecation_warning

# 全局变量，用于包装行
_GLOBAL_WRAP_LINE = None

# 定义 stringPict 类
class stringPict:
    """An ASCII picture.
    The pictures are represented as a list of equal length strings.
    """
    # 特殊值，用于 stringPict.below
    LINE = 'line'

    def __init__(self, s, baseline=0):
        """Initialize from string.
        Multiline strings are centered.
        """
        self.s = s
        # 将字符串 s 按行分割，并确保每行等长后存入 picture 属性
        self.picture = stringPict.equalLengths(s.splitlines())
        # baseline 是基线的行号
        self.baseline = baseline
        self.binding = None

    @staticmethod
    def equalLengths(lines):
        # 处理空行的情况
        if not lines:
            return ['']

        # 计算所有行中的最大宽度
        width = max(line_width(line) for line in lines)
        # 将每行文字居中对齐后返回
        return [center(line, width) for line in lines]

    def height(self):
        """The height of the picture in characters."""
        # 返回图片的高度，即 picture 列表的长度
        return len(self.picture)

    def width(self):
        """The width of the picture in characters."""
        # 返回图片的宽度，即第一行的字符数
        return line_width(self.picture[0])

    @staticmethod
    def next(*args):
        """Put a string of stringPicts next to each other.
        Returns string, baseline arguments for stringPict.
        """
        # 将所有参数转换为 stringPict 对象
        objects = []
        for arg in args:
            if isinstance(arg, str):
                arg = stringPict(arg)
            objects.append(arg)

        # 计算合并后的基线和高度
        newBaseline = max(obj.baseline for obj in objects)
        newHeightBelowBaseline = max(
            obj.height() - obj.baseline
            for obj in objects)
        newHeight = newBaseline + newHeightBelowBaseline

        # 合并所有对象的图片数据
        pictures = []
        for obj in objects:
            oneEmptyLine = [' ' * obj.width()]
            basePadding = newBaseline - obj.baseline
            totalPadding = newHeight - obj.height()
            pictures.append(
                oneEmptyLine * basePadding +
                obj.picture +
                oneEmptyLine * (totalPadding - basePadding))

        # 将合并后的结果拼接为字符串并返回
        result = [''.join(lines) for lines in zip(*pictures)]
        return '\n'.join(result), newBaseline
    def right(self, *args):
        r"""Put pictures next to this one.
        Returns string, baseline arguments for stringPict.
        (Multiline) strings are allowed, and are given a baseline of 0.

        Examples
        ========

        >>> from sympy.printing.pretty.stringpict import stringPict
        >>> print(stringPict("10").right(" + ",stringPict("1\r-\r2",1))[0])
             1
        10 + -
             2

        """
        # 调用 stringPict 类的 next 方法，将当前对象和参数 args 传递进去
        return stringPict.next(self, *args)

    def left(self, *args):
        """Put pictures (left to right) at left.
        Returns string, baseline arguments for stringPict.
        """
        # 调用 stringPict 类的 next 方法，将当前对象添加到参数 args 的末尾，然后传递进去
        return stringPict.next(*(args + (self,)))

    @staticmethod
    def stack(*args):
        """Put pictures on top of each other,
        from top to bottom.
        Returns string, baseline arguments for stringPict.
        The baseline is the baseline of the second picture.
        Everything is centered.
        Baseline is the baseline of the second picture.
        Strings are allowed.
        The special value stringPict.LINE is a row of '-' extended to the width.
        """
        # 将参数 args 中的每个元素转换为 stringPict 对象（除了 stringPict.LINE），如果是字符串则转换为 stringPict 对象
        objects = []
        for arg in args:
            if arg is not stringPict.LINE and isinstance(arg, str):
                arg = stringPict(arg)
            objects.append(arg)

        # 计算新的宽度，以最大的对象宽度为准（不包括 stringPict.LINE）
        newWidth = max(
            obj.width()
            for obj in objects
            if obj is not stringPict.LINE)

        # 创建一个 stringPict 对象，内容为一个水平的 '-'，宽度为 newWidth
        lineObj = stringPict(hobj('-', newWidth))

        # 将 args 中的 stringPict.LINE 替换为正确的行对象（lineObj）
        for i, obj in enumerate(objects):
            if obj is stringPict.LINE:
                objects[i] = lineObj

        # 堆叠图片，并将结果居中
        newPicture = [center(line, newWidth) for obj in objects for line in obj.picture]
        newBaseline = objects[0].height() + objects[1].baseline
        return '\n'.join(newPicture), newBaseline

    def below(self, *args):
        """Put pictures under this picture.
        Returns string, baseline arguments for stringPict.
        Baseline is baseline of top picture

        Examples
        ========

        >>> from sympy.printing.pretty.stringpict import stringPict
        >>> print(stringPict("x+3").below(
        ...       stringPict.LINE, '3')[0]) #doctest: +NORMALIZE_WHITESPACE
        x+3
        ---
         3

        """
        # 调用 stack 方法，将当前对象和参数 args 合并堆叠，然后返回堆叠后的字符串和当前对象的基线
        s, baseline = stringPict.stack(self, *args)
        return s, self.baseline

    def above(self, *args):
        """Put pictures above this picture.
        Returns string, baseline arguments for stringPict.
        Baseline is baseline of bottom picture.
        """
        # 调用 stack 方法，将参数 args 和当前对象合并堆叠，计算出堆叠后的字符串和基线
        string, baseline = stringPict.stack(*(args + (self,)))
        # 重新计算基线，确保基线与底部图片的对齐
        baseline = len(string.splitlines()) - self.height() + self.baseline
        return string, baseline
    def parens(self, left='(', right=')', ifascii_nougly=False):
        """Put parentheses around self.
        Returns string, baseline arguments for stringPict.

        left or right can be None or empty string which means 'no paren from
        that side'
        """
        h = self.height()  # 获取当前对象的高度
        b = self.baseline  # 获取当前对象的基线位置

        # XXX this is a hack -- ascii parens are ugly!
        if ifascii_nougly and not pretty_use_unicode():
            h = 1  # 如果需要使用 ASCII 格式且未启用 Unicode 美化，则设置高度为1
            b = 0  # 基线位置为0

        res = self  # 将当前对象赋给 res 变量

        if left:
            lparen = stringPict(vobj(left, h), baseline=b)  # 使用指定的左括号创建 stringPict 对象
            res = stringPict(*lparen.right(self))  # 将当前对象用左括号包裹并赋给 res

        if right:
            rparen = stringPict(vobj(right, h), baseline=b)  # 使用指定的右括号创建 stringPict 对象
            res = stringPict(*res.right(rparen))  # 将 res 右侧加上右括号并赋给 res

        return ('\n'.join(res.picture), res.baseline)  # 返回由 res 对象的图片和基线组成的元组

    def leftslash(self):
        """Precede object by a slash of the proper size.
        """
        # XXX not used anywhere ?
        height = max(
            self.baseline,  # 使用当前对象的基线位置和
            self.height() - 1 - self.baseline) * 2 + 1  # 高度减去基线位置再乘以2加1，得到斜杠的高度
        slash = '\n'.join(
            ' '*(height - i - 1) + xobj('/', 1) + ' '*i  # 构造斜杠的字符串表示，i 控制空格的数量
            for i in range(height)
        )
        return self.left(stringPict(slash, height // 2))  # 将斜杠加到当前对象左侧并返回

    def root(self, n=None):
        """Produce a nice root symbol.
        Produces ugly results for big n inserts.
        """
        # XXX not used anywhere
        # XXX duplicate of root drawing in pretty.py
        #put line over expression
        result = self.above('_'*self.width())  # 在当前对象上方加上下划线

        #construct right half of root symbol
        height = self.height()  # 获取当前对象的高度
        slash = '\n'.join(
            ' ' * (height - i - 1) + '/' + ' ' * i  # 构造根号的右半部分
            for i in range(height)
        )
        slash = stringPict(slash, height - 1)  # 创建 stringPict 对象表示根号右半部分

        #left half of root symbol
        if height > 2:
            downline = stringPict('\\ \n \\', 1)  # 构造根号左半部分的下行线
        else:
            downline = stringPict('\\')  # 若高度小于等于2，只有一条反斜杠

        #put n on top, as low as possible
        if n is not None and n.width() > downline.width():
            downline = downline.left(' ' * (n.width() - downline.width()))  # 将 n 放在最上方，尽可能低
            downline = downline.above(n)  # 将 n 放在 downline 上方

        #build root symbol
        root = downline.right(slash)  # 将左右两部分拼接成根号符号

        #glue it on at the proper height
        #normally, the root symbel is as high as self
        #which is one less than result
        #this moves the root symbol one down
        #if the root became higher, the baseline has to grow too
        root.baseline = result.baseline - result.height() + root.height()  # 调整根号符号的基线位置

        return result.left(root)  # 将根号符号加到当前对象左侧并返回

    def terminal_width(self):
        """Return the terminal width if possible, otherwise return 0.
        """
        size = shutil.get_terminal_size(fallback=(0, 0))  # 获取终端的宽度
        return size.columns  # 返回终端的列数

    def __eq__(self, o):
        if isinstance(o, str):
            return '\n'.join(self.picture) == o  # 如果 o 是字符串，则比较当前对象的图片和 o 是否相等
        elif isinstance(o, stringPict):
            return o.picture == self.picture  # 如果 o 是 stringPict 对象，则比较图片是否相等
        return False  # 其它情况下返回 False
    # 返回当前对象的哈希值，调用父类的哈希函数
    def __hash__(self):
        return super().__hash__()

    # 返回对象的字符串表示，将图片内容连接成一个字符串，用换行符分隔
    def __str__(self):
        return '\n'.join(self.picture)

    # 返回对象的规范字符串表示，包括图片内容和基线信息
    def __repr__(self):
        return "stringPict(%r,%d)" % ('\n'.join(self.picture), self.baseline)

    # 实现对象的索引访问，返回图片内容的指定索引位置的值
    def __getitem__(self, index):
        return self.picture[index]

    # 返回对象表示的字符串长度，即字符串s的长度
    def __len__(self):
        return len(self.s)
class prettyForm(stringPict):
    """
    Extension of the stringPict class that knows about basic math applications,
    optimizing double minus signs.

    "Binding" is interpreted as follows::

        ATOM this is an atom: never needs to be parenthesized
        FUNC this is a function application: parenthesize if added (?)
        DIV  this is a division: make wider division if divided
        POW  this is a power: only parenthesize if exponent
        MUL  this is a multiplication: parenthesize if powered
        ADD  this is an addition: parenthesize if multiplied or powered
        NEG  this is a negative number: optimize if added, parenthesize if
             multiplied or powered
        OPEN this is an open object: parenthesize if added, multiplied, or
             powered (example: Piecewise)
    """

    ATOM, FUNC, DIV, POW, MUL, ADD, NEG, OPEN = range(8)

    def __init__(self, s, baseline=0, binding=0, unicode=None):
        """
        Initialize a prettyForm object.

        Args:
            s (str): The string representation.
            baseline (int): Baseline for rendering.
            binding (int): Binding power for operator precedence.
            unicode (str, optional): Deprecated Unicode representation.

        """
        stringPict.__init__(self, s, baseline)
        self.binding = binding
        if unicode is not None:
            sympy_deprecation_warning(
                """
                The unicode argument to prettyForm is deprecated. Only the s
                argument (the first positional argument) should be passed.
                """,
                deprecated_since_version="1.7",
                active_deprecations_target="deprecated-pretty-printing-functions")
        self._unicode = unicode or s

    @property
    def unicode(self):
        """
        Deprecated property to retrieve Unicode representation.
        
        Returns:
            str: The Unicode representation.
            
        """
        sympy_deprecation_warning(
            """
            The prettyForm.unicode attribute is deprecated. Use the
            prettyForm.s attribute instead.
            """,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-pretty-printing-functions")
        return self._unicode

    # Note: code to handle subtraction is in _print_Add

    def __add__(self, *others):
        """
        Overloaded addition operator for prettyForm objects.
        
        Args:
            *others: Variable number of prettyForm objects to add.
        
        Returns:
            prettyForm: A new prettyForm object representing the addition.

        """
        arg = self
        if arg.binding > prettyForm.NEG:
            arg = stringPict(*arg.parens())
        result = [arg]
        for arg in others:
            # add parentheses for weak binders
            if arg.binding > prettyForm.NEG:
                arg = stringPict(*arg.parens())
            # use existing minus sign if available
            if arg.binding != prettyForm.NEG:
                result.append(' + ')
            result.append(arg)
        return prettyForm(binding=prettyForm.ADD, *stringPict.next(*result))
    # 定义特殊方法 __truediv__，实现对象的除法操作，支持美化显示为堆叠或斜线形式
    def __truediv__(self, den, slashed=False):
        """Make a pretty division; stacked or slashed.
        """
        # 如果需要斜线形式，则抛出未实现的错误
        if slashed:
            raise NotImplementedError("Can't do slashed fraction yet")
        
        # 复制被除数
        num = self
        
        # 如果被除数的绑定为 DIV 类型，则将其转换为字符串图片形式
        if num.binding == prettyForm.DIV:
            num = stringPict(*num.parens())
        
        # 如果除数的绑定为 DIV 类型，则将其转换为字符串图片形式
        if den.binding == prettyForm.DIV:
            den = stringPict(*den.parens())

        # 如果被除数的绑定为 NEG 类型，则提取右侧的内容
        if num.binding == prettyForm.NEG:
            num = num.right(" ")[0]

        # 返回一个 DIV 类型的 prettyForm 对象，堆叠显示被除数、横线、除数
        return prettyForm(binding=prettyForm.DIV, *stringPict.stack(
            num,
            stringPict.LINE,
            den))

    # 定义特殊方法 __mul__，实现对象的乘法操作，支持美化显示为乘法形式
    def __mul__(self, *others):
        """Make a pretty multiplication.
        Parentheses are needed around +, - and neg.
        """
        # 数量单位的字典
        quantity = {
            'degree': "\N{DEGREE SIGN}"
        }

        # 如果没有其他参数，则直接返回 self
        if len(others) == 0:
            return self  # We aren't actually multiplying... So nothing to do here.

        # 添加需要括号的参数
        arg = self
        if arg.binding > prettyForm.MUL and arg.binding != prettyForm.NEG:
            arg = stringPict(*arg.parens())
        result = [arg]
        
        # 遍历其他参数
        for arg in others:
            # 如果参数不在数量单位字典中，则添加乘号
            if arg.picture[0] not in quantity.values():
                result.append(xsym('*'))
            
            # 如果参数的绑定大于 MUL 并且不是 NEG，则将其转换为字符串图片形式
            if arg.binding > prettyForm.MUL and arg.binding != prettyForm.NEG:
                arg = stringPict(*arg.parens())
            
            # 添加参数到结果列表中
            result.append(arg)

        # 处理特殊情况，例如 -1 * x 转换为 -x
        len_res = len(result)
        for i in range(len_res):
            if i < len_res - 1 and result[i] == '-1' and result[i + 1] == xsym('*'):
                result.pop(i)
                result.pop(i)
                result.insert(i, '-')
        
        # 如果结果的第一个元素以 '-' 开头，则设置绑定为 NEG；否则设置为 MUL
        if result[0][0] == '-':
            bin = prettyForm.NEG
            if result[0] == '-':
                right = result[1]
                if right.picture[right.baseline][0] == '-':
                    result[0] = '- '
        else:
            bin = prettyForm.MUL
        
        # 返回一个对应绑定类型的 prettyForm 对象
        return prettyForm(binding=bin, *stringPict.next(*result))

    # 定义特殊方法 __repr__，返回对象的字符串表示，包括图片列表、基线和绑定
    def __repr__(self):
        return "prettyForm(%r,%d,%d)" % (
            '\n'.join(self.picture),
            self.baseline,
            self.binding)
    # 定义一个特殊方法 __pow__，用于实现自定义的幂运算行为
    def __pow__(self, b):
        """Make a pretty power.
        """
        # 将当前对象赋值给变量 a
        a = self
        # 初始化一个标志，用于指示是否使用内联函数形式
        use_inline_func_form = False
        # 如果 b 的绑定是幂运算
        if b.binding == prettyForm.POW:
            # 调整 b 的格式为字符串图片对象
            b = stringPict(*b.parens())
        # 如果 a 的绑定大于函数的绑定等级
        if a.binding > prettyForm.FUNC:
            # 将 a 调整为字符串图片对象
            a = stringPict(*a.parens())
        # 如果 a 的绑定是函数
        elif a.binding == prettyForm.FUNC:
            # 启发式判断是否使用内联幂运算
            if b.height() > 1:
                # 将 a 调整为字符串图片对象
                a = stringPict(*a.parens())
            else:
                # 设置标志以使用内联函数形式
                use_inline_func_form = True

        # 如果使用内联函数形式
        if use_inline_func_form:
            # 调整 b 的基线位置
            b.baseline = a.prettyFunc.baseline + b.height()
            # 创建函数部分的字符串图片对象
            func = stringPict(*a.prettyFunc.right(b))
            # 返回处理后的字符串图片对象
            return prettyForm(*func.right(a.prettyArgs))
        else:
            # 创建顶部和底部的字符串图片对象
            top = stringPict(*b.left(' ' * a.width()))
            bot = stringPict(*a.right(' ' * b.width()))

        # 返回幂运算的字符串图片对象
        return prettyForm(binding=prettyForm.POW, *bot.above(top))

    # 简单函数列表
    simpleFunctions = ["sin", "cos", "tan"]

    @staticmethod
    def apply(function, *args):
        """Functions of one or more variables.
        """
        # 如果函数在简单函数列表中
        if function in prettyForm.simpleFunctions:
            # 简单函数：尽可能只使用空格
            assert len(args) == 1, "Simple function %s must have 1 argument" % function
            # 获取参数的漂亮表示
            arg = args[0].__pretty__()
            # 如果参数的绑定小于等于除法的绑定等级
            if arg.binding <= prettyForm.DIV:
                # 优化：无需添加括号
                return prettyForm(binding=prettyForm.FUNC, *arg.left(function + ' '))
        
        # 初始化参数列表
        argumentList = []
        # 遍历参数列表
        for arg in args:
            # 添加逗号
            argumentList.append(',')
            # 获取参数的漂亮表示，并添加到参数列表中
            argumentList.append(arg.__pretty__())
        # 创建参数列表的字符串图片对象
        argumentList = stringPict(*stringPict.next(*argumentList[1:]))
        # 将参数列表添加括号
        argumentList = stringPict(*argumentList.parens())
        # 返回函数应用的字符串图片对象
        return prettyForm(binding=prettyForm.ATOM, *argumentList.left(function))
```