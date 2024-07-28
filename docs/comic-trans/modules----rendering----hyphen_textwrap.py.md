# `.\comic-translate\modules\rendering\hyphen_textwrap.py`

```py
# 导入正则表达式模块
import re

# 定义模块的公共接口，包括文本包装和填充功能
__all__ = ['TextWrapper', 'wrap', 'fill', 'dedent', 'indent', 'shorten']

# 硬编码识别的空白字符为美国ASCII字符集中的空白字符
_whitespace = '\t\n\x0b\x0c\r '

class TextWrapper:
    """
    文本包装和填充对象。公共接口包括wrap()和fill()方法；其他方法只是为了
    允许子类重写以调整默认行为。如果想要完全替换主要的包装算法，
    可能需要重写_wrap_chunks()方法。

    几个实例属性控制包装的各个方面:
      width（默认：70）
        包装行的最大宽度（除非break_long_words为false）
      initial_indent（默认：""）
        将添加到包装输出的第一行的字符串。计入行的宽度。
      subsequent_indent（默认：""）
        将添加到所有包装输出的行（除了第一行）。也计入每行的宽度。
      expand_tabs（默认：true）
        在进一步处理之前将输入文本中的制表符扩展为空格。
        每个制表符将根据其位置变成0..'tabsize'个空格。
        如果为false，则每个制表符被视为单个字符。
      tabsize（默认：8）
        将输入文本中的制表符扩展为0..'tabsize'个空格，除非
        'expand_tabs'为false。
      replace_whitespace（默认：true）
        在制表符扩展后，将输入文本中的所有空白字符替换为空格。
        注意，如果expand_tabs为false且replace_whitespace为true，
        每个制表符将转换为单个空格！
      fix_sentence_endings（默认：false）
        确保句子结束标点后始终跟着两个空格。默认关闭，因为算法
        （不可避免地）是不完美的。
      break_long_words（默认：true）
        打破长于'width'的单词。如果为false，这些单词将不会被打破，
        并且某些行可能会比'width'更长。
      break_on_hyphens（默认：true）
        允许打破连字符单词。如果为true，包装优先发生在空格和复合词中的连字符的右侧。
      drop_whitespace（默认：true）
        从行中删除前导和尾随空白字符。
      max_lines（默认：None）
        截断包装后的行数。
      placeholder（默认：' [...]'）
        添加到截断文本的最后一行。

    """
    # 创建一个字典，用于将 Unicode 中的空白字符映射为普通空格字符
    unicode_whitespace_trans = dict.fromkeys(map(ord, _whitespace), ord(' '))

    # 这个复杂的正则表达式用于将文本分割成可包含在单词中断行的块。
    # 例如："Hello there -- you goof-ball, use the -b option!"
    # 被分割为："Hello/ /there/ /--/ /you/ /goof-/ball,/ /use/ /the/ /-b/ /option!
    # （在剔除空字符串后）。
    word_punct = r'[\w!"\'&.,?]'      # 匹配单词字符和常见标点
    letter = r'[^\d\W]'              # 匹配非数字非标点非空白字符
    whitespace = r'[%s]' % re.escape(_whitespace)  # 匹配所有空白字符
    nowhitespace = '[^' + whitespace[1:]  # 匹配非空白字符
    # 定义一个正则表达式，用于分割单词
    wordsep_re = re.compile(r'''
        ( # 任何空白字符
          %(ws)s+
        | # 两个单词之间的破折号
          (?<=%(wp)s) -{2,} (?=\w)
        | # 单词，可能带连字符
          %(nws)s+? (?:
            # 带连字符的单词
              -(?: (?<=%(lt)s{2}-) | (?<=%(lt)s-%(lt)s-))
              (?= %(lt)s -? %(lt)s)
            | # 单词结束
              (?=%(ws)s|\Z)
            | # 破折号
              (?<=%(wp)s) (?=-{2,}\w)
            )
        )''' % {'wp': word_punct, 'lt': letter,
                'ws': whitespace, 'nws': nowhitespace},
        re.VERBOSE)
    del word_punct, letter, nowhitespace

    # 这个简单一些的正则表达式只是在识别的空格处分割。例如：
    # "Hello there -- you goof-ball, use the -b option!"
    # 被分割为："Hello/ /there/ /--/ /you/ /goof-ball,/ /use/ /the/ /-b/ /option!/
    wordsep_simple_re = re.compile(r'(%s+)' % whitespace)
    del whitespace

    # XXX 这不是地域或字符集感知的 —— string.lowercase
    # 仅支持 US-ASCII（因此仅限于英文）
    # 正则表达式，用于识别句子结束符号（小写字母 + 句子结束标点符号 + 可选的引号结束）
    sentence_end_re = re.compile(r'[a-z]'             # 小写字母
                                 r'[\.\!\?]'          # 句子结束标点符号
                                 r'[\"\']?'           # 可选的引号结束
                                 r'\Z')               # 块的结尾
    def __init__(self,
                 width=70,
                 initial_indent="",
                 subsequent_indent="",
                 expand_tabs=True,
                 replace_whitespace=True,
                 fix_sentence_endings=False,
                 break_long_words=True,
                 drop_whitespace=True,
                 break_on_hyphens=True,
                 hyphenate_broken_words=True,
                 tabsize=8,
                 *,
                 max_lines=None,
                 placeholder=' [...]'):
        """
        Initialize the TextWrapper object with specified parameters.

        Args:
        width -- The maximum width of wrapped lines (default 70).
        initial_indent -- Prefix for the first line of wrapped output (default "").
        subsequent_indent -- Prefix for subsequent lines of wrapped output (default "").
        expand_tabs -- If True, expand tabs in input text (default True).
        replace_whitespace -- If True, replace all other whitespace characters with spaces (default True).
        fix_sentence_endings -- If True, attempt to fix sentence endings (default False).
        break_long_words -- If True, break long words (default True).
        drop_whitespace -- If True, drop leading and trailing whitespace from lines (default True).
        break_on_hyphens -- If True, break words on hyphens (default True).
        hyphenate_broken_words -- If True, hyphenate broken words (default True).
        tabsize -- Number of spaces per tab character when expand_tabs is True (default 8).
        max_lines -- Maximum number of lines to output (default None).
        placeholder -- Placeholder string for omitted text in output (default ' [...]').
        """
        self.width = width
        self.initial_indent = initial_indent
        self.subsequent_indent = subsequent_indent
        self.expand_tabs = expand_tabs
        self.replace_whitespace = replace_whitespace
        self.fix_sentence_endings = fix_sentence_endings
        self.break_long_words = break_long_words
        self.drop_whitespace = drop_whitespace
        self.break_on_hyphens = break_on_hyphens
        self.hyphenate_broken_words = hyphenate_broken_words
        self.tabsize = tabsize
        self.max_lines = max_lines
        self.placeholder = placeholder


    # -- Private methods -----------------------------------------------
    # (possibly useful for subclasses to override)

    def _munge_whitespace(self, text):
        """_munge_whitespace(text : string) -> string

        Munge whitespace in text: expand tabs and convert all other
        whitespace characters to spaces.  Eg. " foo\tbar\n\nbaz"
        becomes " foo    bar  baz".
        """
        if self.expand_tabs:
            text = text.expandtabs(self.tabsize)
        if self.replace_whitespace:
            text = text.translate(self.unicode_whitespace_trans)
        return text


    def _split(self, text):
        """_split(text : string) -> [string]

        Split the text to wrap into indivisible chunks.  Chunks are
        not quite the same as words; see _wrap_chunks() for full
        details.  As an example, the text
          Look, goof-ball -- use the -b option!
        breaks into the following chunks:
          'Look,', ' ', 'goof-', 'ball', ' ', '--', ' ',
          'use', ' ', 'the', ' ', '-b', ' ', 'option!'
        if break_on_hyphens is True, or in:
          'Look,', ' ', 'goof-ball', ' ', '--', ' ',
          'use', ' ', 'the', ' ', '-b', ' ', option!'
        otherwise.
        """
        if self.break_on_hyphens is True:
            chunks = self.wordsep_re.split(text)
        else:
            chunks = self.wordsep_simple_re.split(text)
        chunks = [c for c in chunks if c]

        return chunks
    # 修正分块文本中嵌入的句子结尾问题。例如，当原始文本包含 "... foo.\nBar ..." 时，
    # munge_whitespace() 和 split() 将其转换为 [..., "foo.", " ", "Bar", ...]，
    # 其中空格少了一个；此方法简单地将一个空格改为两个空格。
    def _fix_sentence_endings(self, chunks):
        i = 0
        patsearch = self.sentence_end_re.search
        
        # 遍历 chunks 列表直到倒数第二个元素
        while i < len(chunks)-1:
            # 如果下一个元素是空格并且当前元素符合句子结尾的正则表达式条件
            if chunks[i+1] == " " and patsearch(chunks[i]):
                # 将下一个空格替换为两个空格
                chunks[i+1] = "  "
                i += 2
            else:
                i += 1

    # 处理长单词（或者最可能是单词而不是空白的文本块），该单词太长而无法适应任何行。
    def _handle_long_word(self, reversed_chunks, cur_line, cur_len, width):
        # 如果指定的宽度小于1，则至少留下一个空间
        if width < 1:
            space_left = 1
        else:
            space_left = width - cur_len

        # 如果允许分割长单词，则执行以下操作：将下一个文本块的尽可能多的部分放入当前行。
        if self.break_long_words:
            end = space_left
            chunk = reversed_chunks[-1]
            
            # 如果允许在连字符处分割，并且文本块的长度大于剩余空间
            if self.break_on_hyphens and len(chunk) > space_left:
                # 在最后一个连字符后分割，但仅当其前面存在非连字符时
                hyphen = chunk.rfind('-', 0, space_left)
                if hyphen > 0 and any(c != '-' for c in chunk[:hyphen]):
                    end = hyphen + 1
                    
            # 将chunk的前end部分添加到当前行
            if chunk[:end]:
                cur_line.append(chunk[:end])
                # 如果打破的单词需要连字符，并且分割位置不是 ['-',' ','.',','] 中的特定字符
                if self.hyphenate_broken_words and chunk[:end][-1] not in ['-',' ','.',',']:
                    cur_line.append('-')
            reversed_chunks[-1] = chunk[end:]

        # 否则，必须保留长单词的完整性。如果当前行上没有任何文本，则将其添加到当前行。
        elif not cur_line:
            cur_line.append(reversed_chunks.pop())

        # 如果不允许分割长单词，并且当前行已经有文本，则什么也不做。
        # 下次在 _wrap_chunks() 的主循环中，cur_len 将为零，因此下一行将完全专注于我们现在无法处理的长单词。
    # 将文本中的空白字符处理为单个空格，并返回处理后的文本
    def _split_chunks(self, text):
        text = self._munge_whitespace(text)
        # 使用处理后的文本进行分块处理
        return self._split(text)

    # -- Public interface ----------------------------------------------

    def wrap(self, text):
        """wrap(text : string) -> [string]

        Reformat the single paragraph in 'text' so it fits in lines of
        no more than 'self.width' columns, and return a list of wrapped
        lines.  Tabs in 'text' are expanded with string.expandtabs(),
        and all other whitespace characters (including newline) are
        converted to space.
        """
        # 将文本分成块
        chunks = self._split_chunks(text)
        # 如果设置了修正句子结尾标点的选项，修正块中句子的结尾标点
        if self.fix_sentence_endings:
            self._fix_sentence_endings(chunks)
        # 对分块后的文本进行包裹处理，并返回包裹后的行列表
        return self._wrap_chunks(chunks)

    def fill(self, text):
        """fill(text : string) -> string

        Reformat the single paragraph in 'text' to fit in lines of no
        more than 'self.width' columns, and return a new string
        containing the entire wrapped paragraph.
        """
        # 调用wrap方法将文本进行包裹，并用换行符连接成一个新的字符串返回
        return "\n".join(self.wrap(text))
# -- Convenience interface ---------------------------------------------

def wrap(text, width=70, **kwargs):
    """Wrap a single paragraph of text, returning a list of wrapped lines.

    Reformat the single paragraph in 'text' so it fits in lines of no
    more than 'width' columns, and return a list of wrapped lines.  By
    default, tabs in 'text' are expanded with string.expandtabs(), and
    all other whitespace characters (including newline) are converted to
    space.  See TextWrapper class for available keyword args to customize
    wrapping behaviour.
    """
    # 创建 TextWrapper 对象，用于文本包装
    w = TextWrapper(width=width, **kwargs)
    # 使用 TextWrapper 对象的 wrap 方法对文本进行包装
    return w.wrap(text)

def fill(text, width=70, **kwargs):
    """Fill a single paragraph of text, returning a new string.

    Reformat the single paragraph in 'text' to fit in lines of no more
    than 'width' columns, and return a new string containing the entire
    wrapped paragraph.  As with wrap(), tabs are expanded and other
    whitespace characters converted to space.  See TextWrapper class for
    available keyword args to customize wrapping behaviour.
    """
    # 创建 TextWrapper 对象，用于文本填充
    w = TextWrapper(width=width, **kwargs)
    # 使用 TextWrapper 对象的 fill 方法对文本进行填充
    return w.fill(text)

def shorten(text, width, **kwargs):
    """Collapse and truncate the given text to fit in the given width.

    The text first has its whitespace collapsed.  If it then fits in
    the *width*, it is returned as is.  Otherwise, as many words
    as possible are joined and then the placeholder is appended::

        >>> textwrap.shorten("Hello  world!", width=12)
        'Hello world!'
        >>> textwrap.shorten("Hello  world!", width=11)
        'Hello [...]'
    """
    # 创建 TextWrapper 对象，用于文本缩短
    w = TextWrapper(width=width, max_lines=1, **kwargs)
    # 使用 TextWrapper 对象的 fill 方法对文本进行缩短处理
    return w.fill(' '.join(text.strip().split()))


# -- Loosely related functionality -------------------------------------

# 正则表达式，用于匹配全为空白字符（空格或制表符）的行
_whitespace_only_re = re.compile('^[ \t]+$', re.MULTILINE)
# 正则表达式，用于匹配每行非空白字符之前的空白字符（空格或制表符）
_leading_whitespace_re = re.compile('(^[ \t]*)(?:[^ \t\n])', re.MULTILINE)

def dedent(text):
    """Remove any common leading whitespace from every line in `text`.

    This can be used to make triple-quoted strings line up with the left
    edge of the display, while still presenting them in the source code
    in indented form.

    Note that tabs and spaces are both treated as whitespace, but they
    are not equal: the lines "  hello" and "\\thello" are
    considered to have no common leading whitespace.

    Entirely blank lines are normalized to a newline character.
    """
    # 移除每行文本的共同前导空白字符，使其在显示时左对齐
    # 空白行将被规范化为一个换行符
    text = _whitespace_only_re.sub('', text)
    # 找到所有行的最长公共前导空白字符串
    indents = _leading_whitespace_re.findall(text)
    # 遍历所有缩进级别列表中的缩进字符串
    for indent in indents:
        # 如果当前的边距（margin）还未设置，则将其设置为当前的缩进字符串
        if margin is None:
            margin = indent

        # 如果当前行比前面的边距更深：
        # 不进行任何更改（前面的边距仍然保持在顶部）。
        elif indent.startswith(margin):
            pass

        # 如果当前行与前面的边距一致且没有更深的缩进：
        # 这行成为新的边距（margin）的候选。
        elif margin.startswith(indent):
            margin = indent

        # 找到当前行与前面边距之间最大的共同空白部分。
        else:
            for i, (x, y) in enumerate(zip(margin, indent)):
                if x != y:
                    margin = margin[:i]
                    break

    # 健全性检查（仅用于测试和调试）
    if 0 and margin:
        # 检查每一行是否以当前边距（margin）开头，若不是则抛出异常。
        for line in text.split("\n"):
            assert not line or line.startswith(margin), \
                   "line = %r, margin = %r" % (line, margin)

    # 如果存在边距（margin），则将文本中每行开头的该边距去除。
    if margin:
        text = re.sub(r'(?m)^' + margin, '', text)
    # 返回处理后的文本
    return text
# 定义一个函数 `indent`，用于在文本 `text` 的每一行开头添加前缀 `prefix`。
def indent(text, prefix, predicate=None):
    """Adds 'prefix' to the beginning of selected lines in 'text'.
    
    如果提供了 'predicate' 函数，则只对满足 'predicate(line)' 条件的行添加前缀。
    如果未提供 'predicate'，则默认添加前缀到所有非空且不仅包含空白字符的行。
    """
    # 如果未提供 `predicate`，则使用默认的 `predicate` 函数
    if predicate is None:
        # `str.splitlines(True)` 不会生成空字符串。
        # ''.splitlines(True) => []
        # 'foo\n'.splitlines(True) => ['foo\n']
        # 因此我们可以直接使用 `not s.isspace()` 进行判断。
        predicate = lambda s: not s.isspace()

    # 初始化一个列表，用于存储处理后的带有前缀的行
    prefixed_lines = []
    # 遍历文本的每一行
    for line in text.splitlines(True):
        # 如果满足 `predicate(line)` 条件，则在该行前添加 `prefix`
        if predicate(line):
            prefixed_lines.append(prefix)
        # 将当前行加入处理后的列表
        prefixed_lines.append(line)

    # 将列表中的所有行连接成一个字符串，并返回结果
    return ''.join(prefixed_lines)


# 如果脚本作为主程序运行
if __name__ == "__main__":
    # 调用 `dedent` 函数并打印结果
    print(dedent("Hello there.\n  This is indented."))
```