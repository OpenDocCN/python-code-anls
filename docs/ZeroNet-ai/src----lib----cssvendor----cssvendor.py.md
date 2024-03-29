# `ZeroNet\src\lib\cssvendor\cssvendor.py`

```py
# 导入正则表达式模块
import re

# 定义函数，用于处理 CSS 内容，为其中的属性添加浏览器前缀
def prefix(content):
    # 替换 CSS 内容中的 @keyframes 规则，为其添加浏览器前缀
    content = re.sub(
        b"@keyframes (.*? {.*?}\s*})", b"@keyframes \\1\n@-webkit-keyframes \\1\n@-moz-keyframes \\1\n",
        content, flags=re.DOTALL
    )
    # 替换 CSS 内容中的部分属性，为其添加浏览器前缀
    content = re.sub(
        b'([^-\*])(border-radius|box-shadow|appearance|transition|animation|box-sizing|' +
        b'backface-visibility|transform|filter|perspective|animation-[a-z-]+): (.*?)([;}])',
        b'\\1-webkit-\\2: \\3; -moz-\\2: \\3; -o-\\2: \\3; -ms-\\2: \\3; \\2: \\3 \\4', content
    )
    # 替换 CSS 内容中的 linear-gradient 属性，为其添加浏览器前缀
    content = re.sub(
        b'(?<=[^a-zA-Z0-9-])([a-zA-Z0-9-]+): {0,1}(linear-gradient)\((.*?)(\)[;\n])',
        b'\\1: -webkit-\\2(\\3);' +
        b'\\1: -moz-\\2(\\3);' +
        b'\\1: -o-\\2(\\3);' +
        b'\\1: -ms-\\2(\\3);' +
        b'\\1: \\2(\\3);', content
    )
    # 返回处理后的 CSS 内容
    return content

# 主函数，用于测试 prefix 函数
if __name__ == "__main__":
    # 打印处理后的 CSS 内容
    print(prefix(b"""
    .test {
        border-radius: 5px;
        background: linear-gradient(red, blue);
    }


    @keyframes flip {
      0%   { transform: perspective(120px) rotateX(0deg) rotateY(0deg); }
      50%  { transform: perspective(120px) rotateX(-180.1deg) rotateY(0deg) }
      100% { transform: perspective(120px) rotateX(-180deg) rotateY(-179.9deg); }
    }


    """).decode("utf8"))
```