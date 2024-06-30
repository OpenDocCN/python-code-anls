# `D:\src\scipysrc\sympy\doc\src\_pygments\styles.py`

```
"""
Pygments styles used for syntax highlighting.

These are based on the Sphinx style (see
https://github.com/sphinx-doc/sphinx/blob/master/sphinx/pygments_styles.py)
for light mode and the Friendly style for dark mode.

The styles here have been adjusted so that they are WCAG AA compatible. The
tool at https://github.com/mpchadwick/pygments-high-contrast-stylesheets was
used to identify colors that should be adjusted.

"""
# 导入需要的 Pygments 类和模块
from pygments.style import Style
from pygments.styles.friendly import FriendlyStyle
from pygments.styles.native import NativeStyle
from pygments.token import Comment, Generic, Literal, Name, Number, Text

# 定义自定义的高对比度 Sphinx 风格样式类
class SphinxHighContrastStyle(Style):
    """
    Like Sphinx (which is like friendly, but a bit darker to enhance contrast
    on the green background) but with higher contrast colors.

    """
    
    # 自定义行间距样式，用于多行 Unicode 输出
    @property
    def _pre_style(self):
        # This is used instead of the default 125% so that multiline Unicode
        # pprint output looks good
        return 'line-height: 120%;'

    # 设定背景颜色和默认样式
    background_color = '#eeffcc'
    default_style = ''

    # 继承自 FriendlyStyle 的样式，并修改其中的部分样式以提高对比度
    styles = FriendlyStyle.styles
    styles.update({
        # Sphinx 修改的部分，与 "friendly" 风格类似
        Generic.Output: '#333',
        Number: '#208050',

        # 从 "friendly" 风格调整过来的注释颜色，以在背景上有更好的对比度
        Comment: 'italic #3c7a88',
        Comment.Hashbang: 'italic #3c7a88',
        Comment.Multiline: 'italic #3c7a88',
        Comment.PreprocFile: 'italic #3c7a88',
        Comment.Single: 'italic #3c7a88',
        Comment.Special: '#3a7784 bg:#fff0f0',
        Generic.Error: '#e60000',
        Generic.Inserted: '#008200',
        Generic.Prompt: 'bold #b75709',
        Name.Class: 'bold #0e7ba6',
        Name.Constant: '#2b79a1',
        Name.Entity: 'bold #c54629',
        Name.Namespace: 'bold #0e7ba6',
        Name.Variable: '#ab40cd',
        Text.Whitespace: '#707070',
        Literal.String.Interpol: 'italic #3973b7',
        Literal.String.Other: '#b75709',
        Name.Variable.Class: '#ab40cd',
        Name.Variable.Global: '#ab40cd',
        Name.Variable.Instance: '#ab40cd',
        Name.Variable.Magic: '#ab40cd',
    })


# 定义自定义的高对比度 Native 风格样式类
class NativeHighContrastStyle(NativeStyle):
    """
    Like native, but with higher contrast colors.
    """
    
    # 自定义行间距样式，用于多行 Unicode 输出
    @property
    def _pre_style(self):
        # This is used instead of the default 125% so that multiline Unicode
        # pprint output looks good
        return 'line-height: 120%;'

    # 继承自 NativeStyle 的样式，并调整以提高对比度的部分样式
    styles = NativeStyle.styles

    # 这里可以添加更多的样式调整，以实现更高的对比度
    # 更新样式字典，将预处理指令的注释样式设为粗体红色
    styles.update({
        Comment.Preproc: 'bold #e15a5a',
        # 将特殊注释的样式设为粗体红色，前景为亮红色，背景为深红色
        Comment.Special: 'bold #f75050 bg:#520000',
        # 将删除内容的通用样式设为亮红色
        Generic.Deleted: '#e75959',
        # 将错误信息的通用样式设为亮红色
        Generic.Error: '#e75959',
        # 将回溯跟踪的通用样式设为亮红色
        Generic.Traceback: '#e75959',
        # 将数字文字的字面值样式设为浅蓝色
        Literal.Number: '#438dc4',
        # 将内置名称的样式设为深青色
        Name.Builtin: '#2594a1',
        # 将类名的样式设为深青色
        Name.Class: '#548bd3',
        # 将函数名的样式设为深青色
        Name.Function: '#548bd3',
        # 将命名空间的样式设为深青色
        Name.Namespace: '#548bd3',
        # 将空白字符的文本样式设为灰色
        Text.Whitespace: '#878787',
        # 将二进制数字文字的字面值样式设为浅蓝色
        Literal.Number.Bin: '#438dc4',
        # 将浮点数的数字文字的字面值样式设为浅蓝色
        Literal.Number.Float: '#438dc4',
        # 将十六进制数字文字的字面值样式设为浅蓝色
        Literal.Number.Hex: '#438dc4',
        # 将整数的数字文字的字面值样式设为浅蓝色
        Literal.Number.Integer: '#438dc4',
        # 将八进制数字文字的字面值样式设为浅蓝色
        Literal.Number.Oct: '#438dc4',
        # 将伪内置名称的样式设为深青色
        Name.Builtin.Pseudo: '#2594a1',
        # 将魔术函数的函数名样式设为深青色
        Name.Function.Magic: '#548bd3',
        # 将长整型数字文字的字面值样式设为浅蓝色
        Literal.Number.Integer.Long: '#438dc4',
    })
```