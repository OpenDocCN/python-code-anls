# `D:\src\scipysrc\matplotlib\lib\matplotlib\_mathtext_data.py`

```
"""
font data tables for truetype and afm computer modern fonts
"""

# 导入必要的模块和类型声明
from __future__ import annotations
from typing import overload, Union

# LaTeX符号到字体数据的映射字典
latex_to_bakoma = {
    '\\__sqrt__'                 : ('cmex10', 0x70),  # 平方根符号
    '\\bigcap'                   : ('cmex10', 0x5c),  # 大交集符号
    '\\bigcup'                   : ('cmex10', 0x5b),  # 大并集符号
    '\\bigodot'                  : ('cmex10', 0x4b),  # 大圆点运算符号
    '\\bigoplus'                 : ('cmex10', 0x4d),  # 大加运算符号
    '\\bigotimes'                : ('cmex10', 0x4f),  # 大乘运算符号
    '\\biguplus'                 : ('cmex10', 0x5d),  # 大并运算符号
    '\\bigvee'                   : ('cmex10', 0x5f),  # 大逻辑或符号
    '\\bigwedge'                 : ('cmex10', 0x5e),  # 大逻辑与符号
    '\\coprod'                   : ('cmex10', 0x61),  # 调和乘积符号
    '\\int'                      : ('cmex10', 0x5a),  # 积分符号
    '\\langle'                   : ('cmex10', 0xad),  # 左尖括号
    '\\leftangle'                : ('cmex10', 0xad),  # 左尖括号（别名）
    '\\leftbrace'                : ('cmex10', 0xa9),  # 左大括号
    '\\oint'                     : ('cmex10', 0x49),  # 闭合环积分符号
    '\\prod'                     : ('cmex10', 0x59),  # 乘积符号
    '\\rangle'                   : ('cmex10', 0xae),  # 右尖括号
    '\\rightangle'               : ('cmex10', 0xae),  # 右尖括号（别名）
    '\\rightbrace'               : ('cmex10', 0xaa),  # 右大括号
    '\\sum'                      : ('cmex10', 0x58),  # 总和符号
    '\\widehat'                  : ('cmex10', 0x62),  # 宽帽符号
    '\\widetilde'                : ('cmex10', 0x65),  # 宽波浪符号
    '\\{'                        : ('cmex10', 0xa9),  # 左大括号（直接字符）
    '\\}'                        : ('cmex10', 0xaa),  # 右大括号（直接字符）
    '{'                          : ('cmex10', 0xa9),  # 左大括号（简略）
    '}'                          : ('cmex10', 0xaa),  # 右大括号（简略）

    ','                          : ('cmmi10', 0x3b),  # 逗号
    '.'                          : ('cmmi10', 0x3a),  # 句号
    '/'                          : ('cmmi10', 0x3d),  # 斜杠
    '<'                          : ('cmmi10', 0x3c),  # 小于号
    '>'                          : ('cmmi10', 0x3e),  # 大于号
    '\\alpha'                    : ('cmmi10', 0xae),  # α符号
    '\\beta'                     : ('cmmi10', 0xaf),  # β符号
    '\\chi'                      : ('cmmi10', 0xc2),  # χ符号
    '\\combiningrightarrowabove' : ('cmmi10', 0x7e),  # 上箭头
    '\\delta'                    : ('cmmi10', 0xb1),  # δ符号
    '\\ell'                      : ('cmmi10', 0x60),  # ℓ符号
    '\\epsilon'                  : ('cmmi10', 0xb2),  # ε符号
    '\\eta'                      : ('cmmi10', 0xb4),  # η符号
    '\\flat'                     : ('cmmi10', 0x5b),  # ♭符号
    '\\frown'                    : ('cmmi10', 0x5f),  # 垂直符号
    '\\gamma'                    : ('cmmi10', 0xb0),  # γ符号
    '\\imath'                    : ('cmmi10', 0x7b),  # 数学斜体i符号
    '\\iota'                     : ('cmmi10', 0xb6),  # ι符号
    '\\jmath'                    : ('cmmi10', 0x7c),  # 数学斜体j符号
    '\\kappa'                    : ('cmmi10', 0x2219),  # κ符号
    '\\lambda'                   : ('cmmi10', 0xb8),  # λ符号
    '\\leftharpoondown'          : ('cmmi10', 0x29),  # 向下箭头
    '\\leftharpoonup'            : ('cmmi10', 0x28),  # 向上箭头
    '\\mu'                       : ('cmmi10', 0xb9),  # μ符号
    '\\natural'                  : ('cmmi10', 0x5c),  # 自然符号
    '\\nu'                       : ('cmmi10', 0xba),  # ν符号
    '\\omega'                    : ('cmmi10', 0x21),  # ω符号
    '\\phi'                      : ('cmmi10', 0xc1),
    '\\pi'                       : ('cmmi10', 0xbc),
    '\\psi'                      : ('cmmi10', 0xc3),
    '\\rho'                      : ('cmmi10', 0xbd),
    '\\rightharpoondown'         : ('cmmi10', 0x2b),
    '\\rightharpoonup'           : ('cmmi10', 0x2a),
    '\\sharp'                    : ('cmmi10', 0x5d),
    '\\sigma'                    : ('cmmi10', 0xbe),
    '\\smile'                    : ('cmmi10', 0x5e),
    '\\tau'                      : ('cmmi10', 0xbf),
    '\\theta'                    : ('cmmi10', 0xb5),
    '\\triangleleft'             : ('cmmi10', 0x2f),
    '\\triangleright'            : ('cmmi10', 0x2e),
    '\\upsilon'                  : ('cmmi10', 0xc0),
    '\\varepsilon'               : ('cmmi10', 0x22),
    '\\varphi'                   : ('cmmi10', 0x27),
    '\\varrho'                   : ('cmmi10', 0x25),
    '\\varsigma'                 : ('cmmi10', 0x26),
    '\\vartheta'                 : ('cmmi10', 0x23),
    '\\wp'                       : ('cmmi10', 0x7d),
    '\\xi'                       : ('cmmi10', 0xbb),
    '\\zeta'                     : ('cmmi10', 0xb3),




    '!'                          : ('cmr10', 0x21),
    '%'                          : ('cmr10', 0x25),
    '&'                          : ('cmr10', 0x26),
    '('                          : ('cmr10', 0x28),
    ')'                          : ('cmr10', 0x29),
    '+'                          : ('cmr10', 0x2b),
    '0'                          : ('cmr10', 0x30),
    '1'                          : ('cmr10', 0x31),
    '2'                          : ('cmr10', 0x32),
    '3'                          : ('cmr10', 0x33),
    '4'                          : ('cmr10', 0x34),
    '5'                          : ('cmr10', 0x35),
    '6'                          : ('cmr10', 0x36),
    '7'                          : ('cmr10', 0x37),
    '8'                          : ('cmr10', 0x38),
    '9'                          : ('cmr10', 0x39),
    ':'                          : ('cmr10', 0x3a),
    ';'                          : ('cmr10', 0x3b),
    '='                          : ('cmr10', 0x3d),
    '?'                          : ('cmr10', 0x3f),
    '@'                          : ('cmr10', 0x40),
    '['                          : ('cmr10', 0x5b),



    '\\#'                        : ('cmr10', 0x23),
    '\\$'                        : ('cmr10', 0x24),
    '\\%'                        : ('cmr10', 0x25),
    '\\Delta'                    : ('cmr10', 0xa2),
    '\\Gamma'                    : ('cmr10', 0xa1),
    '\\Lambda'                   : ('cmr10', 0xa4),
    '\\Omega'                    : ('cmr10', 0xad),
    '\\Phi'                      : ('cmr10', 0xa9),
    '\\Pi'                       : ('cmr10', 0xa6),
    '\\Psi'                      : ('cmr10', 0xaa),
    '\\Sigma'                    : ('cmr10', 0xa7),
    '\\Theta'                    : ('cmr10', 0xa3),
    '\\Upsilon'                  : ('cmr10', 0xa8),


注释：

    # 符号对应的字体和编码
    '\\phi'                      : ('cmmi10', 0xc1),
    '\\pi'                       : ('cmmi10', 0xbc),
    '\\psi'                      : ('cmmi10', 0xc3),
    '\\rho'                      : ('cmmi10', 0xbd),
    '\\rightharpoondown'         : ('cmmi10', 0x2b),
    '\\rightharpoonup'           : ('cmmi10', 0x2a),
    '\\sharp'                    : ('cmmi10', 0x5d),
    '\\sigma'                    : ('cmmi10', 0xbe),
    '\\smile'                    : ('cmmi10', 0x5e),
    '\\tau'                      : ('cmmi10', 0xbf),
    '\\theta'                    : ('cmmi10', 0xb5),
    '\\triangleleft'             : ('cmmi10', 0x2f),
    '\\triangleright'            : ('cmmi10', 0x2e),
    '\\upsilon'                  : ('cmmi10', 0xc0),
    '\\varepsilon'               : ('cmmi10', 0x22),
    '\\varphi'                   : ('cmmi10', 0x27),
    '\\varrho'                   : ('cmmi10', 0x25),
    '\\varsigma'                 : ('cmmi10', 0x26),
    '\\vartheta'                 : ('cmmi10', 0x23),
    '\\wp'                       : ('cmmi10', 0x7d),
    '\\xi'                       : ('cmmi10', 0xbb),
    '\\zeta'                     : ('cmmi10', 0xb3),

    # 数字和特殊字符对应的字体和编码
    '!'                          : ('cmr10', 0x21),
    '%'                          : ('cmr10', 0x25),
    '&'                          : ('cmr10', 0x26),
    '('                          : ('cmr10', 0x28),
    ')'                          : ('cmr10', 0x29),
    '+'                          : ('cmr10', 0x2b),
    '0'                          : ('cmr10', 0x30),
    '1'                          : ('cmr10', 0x31),
    '2'                          : ('cmr10', 0x32),
    '3'                          : ('cmr10', 0x33),
    '4'                          : ('cmr10', 0x34),
    '5'                          : ('cmr10', 0x35),
    '6'                          : ('cmr10', 0x36),
    '7'                          : ('cmr10', 0x37),
    '8'                          : ('cmr10', 0x38),
    '9'                          : ('cmr10', 0x39),
    ':'                          : ('cmr10', 0x3a),
    ';'                          : ('cmr10', 0x3b),
    '='                          : ('cmr10', 0x3d),
    '?'                          : ('cmr10', 0x3f),
    '@'                          : ('cmr10', 0x40),
    '['                          : ('cmr10', 0x5b),

    # 特殊符号对应的字体和编码
    '\\#'                        : ('cmr10', 0x23),
    '\\$'                        : ('cmr10', 0x24),
    '\\%'                        : ('cmr10', 0x25),
    '\\Delta'                    : ('cmr10', 0xa2),
    '\\Gamma'                    : ('cmr10', 0xa1),
    '\\Lambda'                   : ('cmr10', 0xa4),
    '\\Omega'                    : ('cmr10', 0xad),
    '\\Phi'                      : ('cmr10', 0xa9),
    '\\Pi'                       : ('cmr10', 0xa6),
    '\\Psi'                      : ('cmr10', 0xaa),
    '\\Sigma'                    : ('cmr10', 0xa7),
    '\\Theta'                    : ('cmr10', 0xa3),
    '\\Upsilon'                  : ('cmr10', 0xa8),
    '\\Xi'                       : ('cmr10', 0xa5),  # 符号 '\\Xi' 对应的字体为 'cmr10'，Unicode 码点为 0xa5
    '\\circumflexaccent'         : ('cmr10', 0x5e),  # 符号 '\\circumflexaccent' 对应的字体为 'cmr10'，Unicode 码点为 0x5e
    '\\combiningacuteaccent'     : ('cmr10', 0xb6),  # 符号 '\\combiningacuteaccent' 对应的字体为 'cmr10'，Unicode 码点为 0xb6
    '\\combiningbreve'           : ('cmr10', 0xb8),  # 符号 '\\combiningbreve' 对应的字体为 'cmr10'，Unicode 码点为 0xb8
    '\\combiningdiaeresis'       : ('cmr10', 0xc4),  # 符号 '\\combiningdiaeresis' 对应的字体为 'cmr10'，Unicode 码点为 0xc4
    '\\combiningdotabove'        : ('cmr10', 0x5f),  # 符号 '\\combiningdotabove' 对应的字体为 'cmr10'，Unicode 码点为 0x5f
    '\\combininggraveaccent'     : ('cmr10', 0xb5),  # 符号 '\\combininggraveaccent' 对应的字体为 'cmr10'，Unicode 码点为 0xb5
    '\\combiningoverline'        : ('cmr10', 0xb9),  # 符号 '\\combiningoverline' 对应的字体为 'cmr10'，Unicode 码点为 0xb9
    '\\combiningtilde'           : ('cmr10', 0x7e),  # 符号 '\\combiningtilde' 对应的字体为 'cmr10'，Unicode 码点为 0x7e
    '\\leftbracket'              : ('cmr10', 0x5b),  # 符号 '\\leftbracket' 对应的字体为 'cmr10'，Unicode 码点为 0x5b
    '\\leftparen'                : ('cmr10', 0x28),  # 符号 '\\leftparen' 对应的字体为 'cmr10'，Unicode 码点为 0x28
    '\\rightbracket'             : ('cmr10', 0x5d),  # 符号 '\\rightbracket' 对应的字体为 'cmr10'，Unicode 码点为 0x5d
    '\\rightparen'               : ('cmr10', 0x29),  # 符号 '\\rightparen' 对应的字体为 'cmr10'，Unicode 码点为 0x29
    '\\widebar'                  : ('cmr10', 0xb9),  # 符号 '\\widebar' 对应的字体为 'cmr10'，Unicode 码点为 0xb9
    ']'                          : ('cmr10', 0x5d),  # 符号 ']' 对应的字体为 'cmr10'，Unicode 码点为 0x5d

    '*'                          : ('cmsy10', 0xa4),  # 符号 '*' 对应的字体为 'cmsy10'，Unicode 码点为 0xa4
    '\N{MINUS SIGN}'             : ('cmsy10', 0xa1),  # 符号 '\N{MINUS SIGN}' 对应的字体为 'cmsy10'，Unicode 码点为 0xa1
    '\\Downarrow'                : ('cmsy10', 0x2b),  # 符号 '\\Downarrow' 对应的字体为 'cmsy10'，Unicode 码点为 0x2b
    '\\Im'                       : ('cmsy10', 0x3d),  # 符号 '\\Im' 对应的字体为 'cmsy10'，Unicode 码点为 0x3d
    '\\Leftarrow'                : ('cmsy10', 0x28),  # 符号 '\\Leftarrow' 对应的字体为 'cmsy10'，Unicode 码点为 0x28
    '\\Leftrightarrow'           : ('cmsy10', 0x2c),  # 符号 '\\Leftrightarrow' 对应的字体为 'cmsy10'，Unicode 码点为 0x2c
    '\\P'                        : ('cmsy10', 0x7b),  # 符号 '\\P' 对应的字体为 'cmsy10'，Unicode 码点为 0x7b
    '\\Re'                       : ('cmsy10', 0x3c),  # 符号 '\\Re' 对应的字体为 'cmsy10'，Unicode 码点为 0x3c
    '\\Rightarrow'               : ('cmsy10', 0x29),  # 符号 '\\Rightarrow' 对应的字体为 'cmsy10'，Unicode 码点为 0x29
    '\\S'                        : ('cmsy10', 0x78),  # 符号 '\\S' 对应的字体为 'cmsy10'，Unicode 码点为 0x78
    '\\Uparrow'                  : ('cmsy10', 0x2a),  # 符号 '\\Uparrow' 对应的字体为 'cmsy10'，Unicode 码点为 0x2a
    '\\Updownarrow'              : ('cmsy10', 0x6d),  # 符号 '\\Updownarrow' 对应的字体为 'cmsy10'，Unicode 码点为 0x6d
    '\\Vert'                     : ('cmsy10', 0x6b),  # 符号 '\\Vert' 对应的字体为 'cmsy10'，Unicode 码点为 0x6b
    '\\aleph'                    : ('cmsy10', 0x40),  # 符号 '\\aleph' 对应的字体为 'cmsy10'，Unicode 码点为 0x40
    '\\approx'                   : ('cmsy10', 0xbc),  # 符号 '\\approx' 对应的字体为 'cmsy10'，Unicode 码点为 0xbc
    '\\ast'                      : ('cmsy10', 0xa4),  # 符号 '\\ast' 对应的字体为 'cmsy10'，Unicode 码点为 0xa4
    '\\asymp'                    : ('cmsy10', 0xb3),  # 符号 '\\asymp' 对应的字体为 'cmsy10'，Unicode 码点为 0xb3
    '\\backslash'                : ('cmsy10', 0x6e),  # 符号 '\\backslash' 对应的字体为 'cmsy10'，Unicode 码点为 0x6e
    '\\bigcirc'                  : ('cmsy10', 0xb0),  # 符号 '\\bigcirc' 对应的字体为 'cmsy10'，Unicode 码点为 0xb0
    '\\bigtriangledown'          : ('cmsy10', 0x35),  # 符号 '\\bigtriangledown' 对应的字体为 'cmsy10'，Unicode 码点为 0x35
    '\\bigtriangleup'            : ('cmsy10', 0x34),  # 符号 '\\bigtriangleup' 对应的字体为 'cmsy10'，Unicode 码点为 0x34
    '\\bot'                      : ('cmsy10', 0x3f),  # 符号 '\\bot' 对应的字体为 'cmsy10'，Unicode 码点为 0x3f
    '\\bullet'                   : ('cmsy10', 0xb2),  # 符号 '\\bullet' 对应的字体为 'cmsy10'，Unicode 码点为 0xb2
    '\\cap'                      : ('cmsy10', 0x5c),  # 符号 '\\cap' 对应的字体为 'cmsy10'，Unicode 码点为 0x5c
    '\\cdot'                     : ('cmsy10', 0xa2),  # 符号 '\\cdot' 对应的字体为 'cmsy10'，Unicode 码点为 0xa2
    '\\circ'                     : ('cmsy10', 0xb1),  # 符号 '\\circ' 对应的字体为 'cmsy10'，Unicode 码点为 0xb1
    '\\clubsuit'                 : ('cmsy10', 0x7c),  # 符号 '\\clubs
    '\\heartsuit'                : ('cmsy10', 0x7e),
    # 符号 '\\heartsuit' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x7e
    '\\in'                       : ('cmsy10', 0x32),
    # 符号 '\\in' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x32
    '\\infty'                    : ('cmsy10', 0x31),
    # 符号 '\\infty' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x31
    '\\lbrace'                   : ('cmsy10', 0x66),
    # 符号 '\\lbrace' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x66
    '\\lceil'                    : ('cmsy10', 0x64),
    # 符号 '\\lceil' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x64
    '\\leftarrow'                : ('cmsy10', 0xc3),
    # 符号 '\\leftarrow' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0xc3
    '\\leftrightarrow'           : ('cmsy10', 0x24),
    # 符号 '\\leftrightarrow' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x24
    '\\leq'                      : ('cmsy10', 0x2219),
    # 符号 '\\leq' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x2219
    '\\lfloor'                   : ('cmsy10', 0x62),
    # 符号 '\\lfloor' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x62
    '\\ll'                       : ('cmsy10', 0xbf),
    # 符号 '\\ll' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0xbf
    '\\mid'                      : ('cmsy10', 0x6a),
    # 符号 '\\mid' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x6a
    '\\mp'                       : ('cmsy10', 0xa8),
    # 符号 '\\mp' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0xa8
    '\\nabla'                    : ('cmsy10', 0x72),
    # 符号 '\\nabla' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x72
    '\\nearrow'                  : ('cmsy10', 0x25),
    # 符号 '\\nearrow' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x25
    '\\neg'                      : ('cmsy10', 0x3a),
    # 符号 '\\neg' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x3a
    '\\ni'                       : ('cmsy10', 0x33),
    # 符号 '\\ni' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x33
    '\\nwarrow'                  : ('cmsy10', 0x2d),
    # 符号 '\\nwarrow' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x2d
    '\\odot'                     : ('cmsy10', 0xaf),
    # 符号 '\\odot' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0xaf
    '\\ominus'                   : ('cmsy10', 0xaa),
    # 符号 '\\ominus' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0xaa
    '\\oplus'                    : ('cmsy10', 0xa9),
    # 符号 '\\oplus' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0xa9
    '\\oslash'                   : ('cmsy10', 0xae),
    # 符号 '\\oslash' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0xae
    '\\otimes'                   : ('cmsy10', 0xad),
    # 符号 '\\otimes' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0xad
    '\\pm'                       : ('cmsy10', 0xa7),
    # 符号 '\\pm' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0xa7
    '\\prec'                     : ('cmsy10', 0xc1),
    # 符号 '\\prec' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0xc1
    '\\preceq'                   : ('cmsy10', 0xb9),
    # 符号 '\\preceq' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0xb9
    '\\prime'                    : ('cmsy10', 0x30),
    # 符号 '\\prime' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x30
    '\\propto'                   : ('cmsy10', 0x2f),
    # 符号 '\\propto' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x2f
    '\\rbrace'                   : ('cmsy10', 0x67),
    # 符号 '\\rbrace' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x67
    '\\rceil'                    : ('cmsy10', 0x65),
    # 符号 '\\rceil' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x65
    '\\rfloor'                   : ('cmsy10', 0x63),
    # 符号 '\\rfloor' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x63
    '\\rightarrow'               : ('cmsy10', 0x21),
    # 符号 '\\rightarrow' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x21
    '\\searrow'                  : ('cmsy10', 0x26),
    # 符号 '\\searrow' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x26
    '\\sim'                      : ('cmsy10', 0xbb),
    # 符号 '\\sim' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0xbb
    '\\simeq'                    : ('cmsy10', 0x27),
    # 符号 '\\simeq' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x27
    '\\slash'                    : ('cmsy10', 0x36),
    # 符号 '\\slash' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x36
    '\\spadesuit'                : ('cmsy10', 0xc4),
    # 符号 '\\spadesuit' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0xc4
    '\\sqcap'                    : ('cmsy10', 0x75),
    # 符号 '\\sqcap' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x75
    '\\sqcup'                    : ('cmsy10', 0x74),
    # 符号 '\\sqcup' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x74
    '\\sqsubseteq'               : ('cmsy10', 0x76),
    # 符号 '\\sqsubseteq' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x76
    '\\sqsupseteq'               : ('cmsy10', 0x77),
    # 符号 '\\sqsupseteq' 的元组表示，包含字体名称 'cmsy10' 和 Unicode 编码 0x77
    '\\subset'                   : ('cmsy10', 0xbd),
    # 符号 '\\
    '\\wedge'                    : ('cmsy10', 0x5e),
    '\\wr'                       : ('cmsy10', 0x6f),
    '\\|'                        : ('cmsy10', 0x6b),
    '|'                          : ('cmsy10', 0x6a),

    '\\_'                        : ('cmtt10', 0x5f)
# 自动化生成的字典，将字符串键映射到对应的Unicode码值
type12uni = {
    'aring'          : 229,
    'quotedblright'  : 8221,
    'V'              : 86,
    'dollar'         : 36,
    'four'           : 52,
    'Yacute'         : 221,
    'P'              : 80,
    'underscore'     : 95,
    'p'              : 112,
    'Otilde'         : 213,
    'perthousand'    : 8240,
    'zero'           : 48,
    'dotlessi'       : 305,
    'Scaron'         : 352,
    'zcaron'         : 382,
    'egrave'         : 232,
    'section'        : 167,
    'Icircumflex'    : 206,
    'ntilde'         : 241,
    'ampersand'      : 38,
    'dotaccent'      : 729,
    'degree'         : 176,
    'K'              : 75,
    'acircumflex'    : 226,
    'Aring'          : 197,
    'k'              : 107,
    'smalltilde'     : 732,
    'Agrave'         : 192,
    'divide'         : 247,
    'ocircumflex'    : 244,
    'asciitilde'     : 126,
    'two'            : 50,
    'E'              : 69,
    'scaron'         : 353,
    'F'              : 70,
    'bracketleft'    : 91,
    'asciicircum'    : 94,
    'f'              : 102,
    'ordmasculine'   : 186,
    'mu'             : 181,
    'paragraph'      : 182,
    'nine'           : 57,
    'v'              : 118,
    'guilsinglleft'  : 8249,
    'backslash'      : 92,
    'six'            : 54,
    'A'              : 65,
    'icircumflex'    : 238,
    'a'              : 97,
    'ogonek'         : 731,
    'q'              : 113,
    'oacute'         : 243,
    'ograve'         : 242,
    'edieresis'      : 235,
    'comma'          : 44,
    'otilde'         : 245,
    'guillemotright' : 187,
    'ecircumflex'    : 234,
    'greater'        : 62,
    'uacute'         : 250,
    'L'              : 76,
    'bullet'         : 8226,
    'cedilla'        : 184,
    'ydieresis'      : 255,
    'l'              : 108,
    'logicalnot'     : 172,
    'exclamdown'     : 161,
    'endash'         : 8211,
    'agrave'         : 224,
    'Adieresis'      : 196,
    'germandbls'     : 223,
    'Odieresis'      : 214,
    'space'          : 32,
    'quoteright'     : 8217,
    'ucircumflex'    : 251,
    'G'              : 71,
    'quoteleft'      : 8216,
    'W'              : 87,
    'Q'              : 81,
    'g'              : 103,
    'w'              : 119,
    'question'       : 63,
    'one'            : 49,
    'ring'           : 730,
    'figuredash'     : 8210,
    'B'              : 66,
    'iacute'         : 237,
    'Ydieresis'      : 376,
    'R'              : 82,
    'b'              : 98,
    'r'              : 114,
    'Ccedilla'       : 199,
    'minus'          : 8722,
    'Lslash'         : 321,
    'Uacute'         : 218,
    'yacute'         : 253,
    'Ucircumflex'    : 219,
    'quotedbl'       : 34,
    'onehalf'        : 189,
    'Thorn'          : 222,
    'M'              : 77,
    'eight'          : 56,
    'multiply'       : 215,
    'grave'          : 96,
    'Ocircumflex'    : 212,
    'm'              : 109,
}
    'Ugrave'         : 217,
    'guilsinglright' : 8250,
    'Ntilde'         : 209,
    'questiondown'   : 191,
    'Atilde'         : 195,
    'ccedilla'       : 231,
    'Z'              : 90,
    'copyright'      : 169,
    'yen'            : 165,
    'Eacute'         : 201,
    'H'              : 72,
    'X'              : 88,
    'Idieresis'      : 207,
    'bar'            : 124,
    'h'              : 104,
    'x'              : 120,
    'udieresis'      : 252,
    'ordfeminine'    : 170,
    'braceleft'      : 123,
    'macron'         : 175,
    'atilde'         : 227,
    'Acircumflex'    : 194,
    'Oslash'         : 216,
    'C'              : 67,
    'quotedblleft'   : 8220,
    'S'              : 83,
    'exclam'         : 33,
    'Zcaron'         : 381,
    'equal'          : 61,
    's'              : 115,
    'eth'            : 240,
    'Egrave'         : 200,
    'hyphen'         : 45,
    'period'         : 46,
    'igrave'         : 236,
    'colon'          : 58,
    'Ecircumflex'    : 202,
    'trademark'      : 8482,
    'Aacute'         : 193,
    'cent'           : 162,
    'lslash'         : 322,
    'c'              : 99,
    'N'              : 78,
    'breve'          : 728,
    'Oacute'         : 211,
    'guillemotleft'  : 171,
    'n'              : 110,
    'idieresis'      : 239,
    'braceright'     : 125,
    'seven'          : 55,
    'brokenbar'      : 166,
    'ugrave'         : 249,
    'periodcentered' : 183,
    'sterling'       : 163,
    'I'              : 73,
    'Y'              : 89,
    'Eth'            : 208,
    'emdash'         : 8212,
    'i'              : 105,
    'daggerdbl'      : 8225,
    'y'              : 121,
    'plusminus'      : 177,
    'less'           : 60,
    'Udieresis'      : 220,
    'D'              : 68,
    'five'           : 53,
    'T'              : 84,
    'oslash'         : 248,
    'acute'          : 180,
    'd'              : 100,
    'OE'             : 338,
    'Igrave'         : 204,
    't'              : 116,
    'parenright'     : 41,
    'adieresis'      : 228,
    'quotesingle'    : 39,
    'twodotenleader' : 8229,
    'slash'          : 47,
    'ellipsis'       : 8230,
    'numbersign'     : 35,
    'odieresis'      : 246,
    'O'              : 79,
    'oe'             : 339,
    'o'              : 111,
    'Edieresis'      : 203,
    'plus'           : 43,
    'dagger'         : 8224,
    'three'          : 51,
    'hungarumlaut'   : 733,
    'parenleft'      : 40,
    'fraction'       : 8260,
    'registered'     : 174,
    'J'              : 74,
    'dieresis'       : 168,
    'Ograve'         : 210,
    'j'              : 106,
    'z'              : 122,
    'ae'             : 230,
    'semicolon'      : 59,
    'at'             : 64,
    'Iacute'         : 205,
    'percent'        : 37,
    'bracketright'   : 93,
    'AE'             : 198,
    'asterisk'       : 42,
    'aacute'         : 225,
    'U'              : 85,
    'eacute'         : 233,
    'e'              : 101,  # 定义键为 'e'，对应的值为整数 101
    'thorn'          : 254,  # 定义键为 'thorn'，对应的值为整数 254
    'u'              : 117,  # 定义键为 'u'，对应的值为整数 117
}

# 创建一个新的字典uni2type1，将type12uni字典的键值对颠倒存储
uni2type1 = {v: k for k, v in type12uni.items()}

# 下面的脚本用于对tex2uni字典进行排序和格式化

## 对于十进制数值：int(hex(v), 16)
# newtex = {k: hex(v) for k, v in tex2uni.items()}
# sd = dict(sorted(newtex.items(), key=lambda item: item[0]))

## 用于对排序后的字典进行格式化，确保适当的间距
## 值 '24' 是根据newtex键中最长字符串的长度来确定的
# for key in sd:
#     print("{0:24} : {1: <s},".format("'" + key + "'", sd[key]))

# 定义一个名为tex2uni的字典，将特定的字符映射到Unicode十六进制数值
tex2uni = {
    '#'                      : 0x23,
    '$'                      : 0x24,
    '%'                      : 0x25,
    'AA'                     : 0xc5,
    'AE'                     : 0xc6,
    'BbbC'                   : 0x2102,
    'BbbN'                   : 0x2115,
    'BbbP'                   : 0x2119,
    'BbbQ'                   : 0x211a,
    'BbbR'                   : 0x211d,
    'BbbZ'                   : 0x2124,
    'Bumpeq'                 : 0x224e,
    'Cap'                    : 0x22d2,
    'Colon'                  : 0x2237,
    'Cup'                    : 0x22d3,
    'DH'                     : 0xd0,
    'Delta'                  : 0x394,
    'Doteq'                  : 0x2251,
    'Downarrow'              : 0x21d3,
    'Equiv'                  : 0x2263,
    'Finv'                   : 0x2132,
    'Game'                   : 0x2141,
    'Gamma'                  : 0x393,
    'H'                      : 0x30b,
    'Im'                     : 0x2111,
    'Join'                   : 0x2a1d,
    'L'                      : 0x141,
    'Lambda'                 : 0x39b,
    'Ldsh'                   : 0x21b2,
    'Leftarrow'              : 0x21d0,
    'Leftrightarrow'         : 0x21d4,
    'Lleftarrow'             : 0x21da,
    'Longleftarrow'          : 0x27f8,
    'Longleftrightarrow'     : 0x27fa,
    'Longrightarrow'         : 0x27f9,
    'Lsh'                    : 0x21b0,
    'Nearrow'                : 0x21d7,
    'Nwarrow'                : 0x21d6,
    'O'                      : 0xd8,
    'OE'                     : 0x152,
    'Omega'                  : 0x3a9,
    'P'                      : 0xb6,
    'Phi'                    : 0x3a6,
    'Pi'                     : 0x3a0,
    'Psi'                    : 0x3a8,
    'QED'                    : 0x220e,
    'Rdsh'                   : 0x21b3,
    'Re'                     : 0x211c,
    'Rightarrow'             : 0x21d2,
    'Rrightarrow'            : 0x21db,
    'Rsh'                    : 0x21b1,
    'S'                      : 0xa7,
    'Searrow'                : 0x21d8,
    'Sigma'                  : 0x3a3,
    'Subset'                 : 0x22d0,
    'Supset'                 : 0x22d1,
    'Swarrow'                : 0x21d9,
    'Theta'                  : 0x398,
    'Thorn'                  : 0xde,
    'Uparrow'                : 0x21d1,
    'Updownarrow'            : 0x21d5,
    'Upsilon'                : 0x3a5,
    'Vdash'                  : 0x22a9,
    'Vert'                   : 0x2016,
}
    {
        'Vvdash'                 : 0x22aa,  # 双垂直线，Unicode码点为0x22aa
        'Xi'                     : 0x39e,   # 希腊字母大写 Xi，Unicode码点为0x39e
        '_'                      : 0x5f,    # 下划线，Unicode码点为0x5f
        '__sqrt__'               : 0x221a,  # 平方根符号，Unicode码点为0x221a
        'aa'                     : 0xe5,    # 拉丁字母 a 与环音符号，Unicode码点为0xe5
        'ac'                     : 0x223e,  # 波浪运算符，Unicode码点为0x223e
        'acute'                  : 0x301,   # 音调符号，Unicode码点为0x301
        'acwopencirclearrow'     : 0x21ba,  # 逆时针环形箭头，Unicode码点为0x21ba
        'adots'                  : 0x22f0,  # 等间隔省略号，Unicode码点为0x22f0
        'ae'                     : 0xe6,    # 拉丁字母组合 ae，Unicode码点为0xe6
        'aleph'                  : 0x2135,  # 阿列夫符号，Unicode码点为0x2135
        'alpha'                  : 0x3b1,   # 希腊字母小写 alpha，Unicode码点为0x3b1
        'amalg'                  : 0x2a3f,  # 集合并运算符，Unicode码点为0x2a3f
        'angle'                  : 0x2220,  # 角度符号，Unicode码点为0x2220
        'approx'                 : 0x2248,  # 近似等于号，Unicode码点为0x2248
        'approxeq'               : 0x224a,  # 近似等于或等于号，Unicode码点为0x224a
        'approxident'            : 0x224b,  # 近似等于且不等于号，Unicode码点为0x224b
        'arceq'                  : 0x2258,  # 弧度等于号，Unicode码点为0x2258
        'ast'                    : 0x2217,  # 星号运算符，Unicode码点为0x2217
        'asterisk'               : 0x2a,    # 星号，Unicode码点为0x2a
        'asymp'                  : 0x224d,  # 近似等于与反相等于号，Unicode码点为0x224d
        'backcong'               : 0x224c,  # 反向同等于号，Unicode码点为0x224c
        'backepsilon'            : 0x3f6,   # 反向 epsilon 符号，Unicode码点为0x3f6
        'backprime'              : 0x2035,  # 反向 prime 符号，Unicode码点为0x2035
        'backsim'                : 0x223d,  # 反向同等于号，Unicode码点为0x223d
        'backsimeq'              : 0x22cd,  # 反向同等于或等于号，Unicode码点为0x22cd
        'backslash'              : 0x5c,    # 反斜线，Unicode码点为0x5c
        'bagmember'              : 0x22ff,  # 袋子成员符号，Unicode码点为0x22ff
        'bar'                    : 0x304,   # 竖线符号，Unicode码点为0x304
        'barleftarrow'           : 0x21e4,  # 带箭头的竖线，Unicode码点为0x21e4
        'barvee'                 : 0x22bd,  # 竖或运算符，Unicode码点为0x22bd
        'barwedge'               : 0x22bc,  # 竖与运算符，Unicode码点为0x22bc
        'because'                : 0x2235,  # 因为符号，Unicode码点为0x2235
        'beta'                   : 0x3b2,   # 希腊字母小写 beta，Unicode码点为0x3b2
        'beth'                   : 0x2136,  # 贝塔符号，Unicode码点为0x2136
        'between'                : 0x226c,  # 在...之间符号，Unicode码点为0x226c
        'bigcap'                 : 0x22c2,  # 大交集符号，Unicode码点为0x22c2
        'bigcirc'                : 0x25cb,  # 大圆圈符号，Unicode码点为0x25cb
        'bigcup'                 : 0x22c3,  # 大并集符号，Unicode码点为0x22c3
        'bigodot'                : 0x2a00,  # 大圆点乘符号，Unicode码点为0x2a00
        'bigoplus'               : 0x2a01,  # 大加号符号，Unicode码点为0x2a01
        'bigotimes'              : 0x2a02,  # 大乘号符号，Unicode码点为0x2a02
        'bigsqcup'               : 0x2a06,  # 大方并集符号，Unicode码点为0x2a06
        'bigstar'                : 0x2605,  # 大星号符号，Unicode码点为0x2605
        'bigtriangledown'        : 0x25bd,  # 大下三角符号，Unicode码点为0x25bd
        'bigtriangleup'          : 0x25b3,  # 大上三角符号，Unicode码点为0x25b3
        'biguplus'               : 0x2a04,  # 大加交符号，Unicode码点为0x2a04
        'bigvee'                 : 0x22c1,  # 大或运算符，Unicode码点为0x22c1
        'bigwedge'               : 0x22c0,  # 大与运算符，Unicode码点为0x22c0
        'blacksquare'            : 0x25a0,  # 黑色方块符号，Unicode码点为0x25a0
        'blacktriangle'          : 0x25b4,  # 黑色上三角符号，Unicode码点为0x25b4
        'blacktriangledown'      : 0x25be,  # 黑色下三角符号，Unicode码点为0x25be
        'blacktriangleleft'      : 0x25c0,  # 黑色左三角符号，Unicode码点为0x25c0
        'blacktriangleright'     : 0x25b6,  # 黑色右三角符号，Unicode码点为0x25b6
        'bot'                    : 0x22a5,  # 底部符号，Unicode码点为0x22a5
        'bowtie'                 : 0x22c8,  # 蝴蝶结符号，Unicode码点为0x22c8
        'boxbar'                 : 0x25eb,  # 方框与竖线符号，Unicode码点为0x25eb
        'boxdot'                 : 0x22a1,  # 方框与小圆点符号，Unicode码点为0x22a1
        'boxminus'               : 0x229f,  # 方框与减号符号，Unicode码点为0x229f
        'boxplus'                : 0x229e,  # 方框与加号符号，Unicode码点为0x229e
        'boxtimes'               : 0x22a0,  # 方框与乘号符号
    'circlearrowleft'        : 0x21ba,  # 左循环箭头的 Unicode 编码
    'circlearrowright'       : 0x21bb,  # 右循环箭头的 Unicode 编码
    'circledR'               : 0xae,    # 注册商标符号的 Unicode 编码
    'circledS'               : 0x24c8,  # 包围 S 字母的圆圈符号的 Unicode 编码
    'circledast'             : 0x229b,  # 包围星号的圆圈符号的 Unicode 编码
    'circledcirc'            : 0x229a,  # 包围圆圈的圆圈符号的 Unicode 编码
    'circleddash'            : 0x229d,  # 包围短横线的圆圈符号的 Unicode 编码
    'circumflexaccent'       : 0x302,   # 圆顶音符号的 Unicode 编码
    'clubsuit'               : 0x2663,  # 黑桃符号的 Unicode 编码
    'clubsuitopen'           : 0x2667,  # 空心黑桃符号的 Unicode 编码
    'colon'                  : 0x3a,    # 冒号的 Unicode 编码
    'coloneq'                : 0x2254,  # 定义等于符号的 Unicode 编码
    'combiningacuteaccent'   : 0x301,   # 组合式锐音符号的 Unicode 编码
    'combiningbreve'         : 0x306,   # 组合式弯音符号的 Unicode 编码
    'combiningdiaeresis'     : 0x308,   # 组合式分音符号的 Unicode 编码
    'combiningdotabove'      : 0x307,   # 组合式上点符号的 Unicode 编码
    'combiningfourdotsabove' : 0x20dc,  # 组合式上四点符号的 Unicode 编码
    'combininggraveaccent'   : 0x300,   # 组合式重音符号的 Unicode 编码
    'combiningoverline'      : 0x304,   # 组合式上划线符号的 Unicode 编码
    'combiningrightarrowabove' : 0x20d7, # 组合式上右箭头符号的 Unicode 编码
    'combiningthreedotsabove' : 0x20db, # 组合式上三点符号的 Unicode 编码
    'combiningtilde'         : 0x303,   # 组合式颚音符号的 Unicode 编码
    'complement'             : 0x2201,  # 补集符号的 Unicode 编码
    'cong'                   : 0x2245,  # 渐近等于符号的 Unicode 编码
    'coprod'                 : 0x2210,  # 余积运算符号的 Unicode 编码
    'copyright'              : 0xa9,    # 版权符号的 Unicode 编码
    'cup'                    : 0x222a,  # 并集符号的 Unicode 编码
    'cupdot'                 : 0x228d,  # 包含点的并集符号的 Unicode 编码
    'cupleftarrow'           : 0x228c,  # 包含左箭头的并集符号的 Unicode 编码
    'curlyeqprec'            : 0x22de,  # 曲线等于前导符号的 Unicode 编码
    'curlyeqsucc'            : 0x22df,  # 曲线等于后继符号的 Unicode 编码
    'curlyvee'               : 0x22ce,  # 曲线 V 符号的 Unicode 编码
    'curlywedge'             : 0x22cf,  # 曲线 W 符号的 Unicode 编码
    'curvearrowleft'         : 0x21b6,  # 曲线左箭头符号的 Unicode 编码
    'curvearrowright'        : 0x21b7,  # 曲线右箭头符号的 Unicode 编码
    'cwopencirclearrow'      : 0x21bb,  # 顺时针打开的圆形箭头符号的 Unicode 编码
    'd'                      : 0x323,   # 带逗号的 D 符号的 Unicode 编码
    'dag'                    : 0x2020,  # 活力 D 符号的 Unicode 编码
    'dagger'                 : 0x2020,  # 匕首符号的 Unicode 编码
    'daleth'                 : 0x2138,  # D 符号的 Unicode 编码
    'danger'                 : 0x2621,  # 危险符号的 Unicode 编码
    'dashleftarrow'          : 0x290e,  # 短横线左箭头符号的 Unicode 编码
    'dashrightarrow'         : 0x290f,  # 短横线右箭头符号的 Unicode 编码
    'dashv'                  : 0x22a3,  # 短横线 V 符号的 Unicode 编码
    'ddag'                   : 0x2021,  # 活力 D 符号的 Unicode 编码
    'ddagger'                : 0x2021,  # 双刺符号的 Unicode 编码
    'ddddot'                 : 0x20dc,  # 下四点符号的 Unicode 编码
    'dddot'                  : 0x20db,  # 下三点符号的 Unicode 编码
    'ddot'                   : 0x308,   # 上点符号的 Unicode 编码
    'ddots'                  : 0x22f1,  # 上点符号的 Unicode 编码
    'degree'                 : 0xb0,    # 度数符号的 Unicode 编码
    'delta'                  : 0x3b4,   # 德尔塔符号的 Unicode 编码
    'dh'                     : 0xf0,    # D H 符号的 Unicode 编码
    'diamond'                : 0x22c4,  # 菱形符号的 Unicode 编码
    'diamondsuit'            : 0x2662,  # 钻石套符号的 Unicode 编码
    'digamma'                : 0x3dd,   # D 符号的 Unicode 编码
    'disin'                  : 0x22f2,  # 反包含符号的 Unicode 编码
    'div'                    : 0xf7,    # 除法符号的 Unicode 编码
    'divideontimes'          : 0x22c7,  # 乘法除法符号的 Unicode 编码
    'dot'                    : 0x307,   # 点符号的 Unicode 编码
    'doteq'                  : 0x2250,  # 等于点符号的 Unicode 编码
    'doteqdot'               : 0x2251,  # 点等于点符号的 Unicode 编码
    'dotminus'               : 0x2238,  # 点减符号的 Unicode 编码
    'dotplus'                : 0x2214,  # 点加符号的 Unicode 编码
    'dots'                   : 0x2026,  # 省略号的 Unicode 编码
    'dotsminusdots'          : 0x223a,  # 点减省略号的点符号的 Unicode 编码
    'doublebarwedge'         : 0x2306,  # 双杠楔形符号的 Unicode 编码
    'downarrow'              : 0x2193,  # 向下箭头符号的 Unicode 编码
    'downdownarrows'         : 0x21ca,  # 双向下箭头符号的 Unicode 编码
    'down
    'eqcirc'                 : 0x2256,
    # 定义键名 'eqcirc'，对应的值是十六进制数 0x2256
    'eqcolon'                : 0x2255,
    # 定义键名 'eqcolon'，对应的值是十六进制数 0x2255
    'eqdef'                  : 0x225d,
    # 定义键名 'eqdef'，对应的值是十六进制数 0x225d
    'eqgtr'                  : 0x22dd,
    # 定义键名 'eqgtr'，对应的值是十六进制数 0x22dd
    'eqless'                 : 0x22dc,
    # 定义键名 'eqless'，对应的值是十六进制数 0x22dc
    'eqsim'                  : 0x2242,
    # 定义键名 'eqsim'，对应的值是十六进制数 0x2242
    'eqslantgtr'             : 0x2a96,
    # 定义键名 'eqslantgtr'，对应的值是十六进制数 0x2a96
    'eqslantless'            : 0x2a95,
    # 定义键名 'eqslantless'，对应的值是十六进制数 0x2a95
    'equal'                  : 0x3d,
    # 定义键名 'equal'，对应的值是十六进制数 0x3d
    'equalparallel'          : 0x22d5,
    # 定义键名 'equalparallel'，对应的值是十六进制数 0x22d5
    'equiv'                  : 0x2261,
    # 定义键名 'equiv'，对应的值是十六进制数 0x2261
    'eta'                    : 0x3b7,
    # 定义键名 'eta'，对应的值是十六进制数 0x3b7
    'eth'                    : 0xf0,
    # 定义键名 'eth'，对应的值是十六进制数 0xf0
    'exists'                 : 0x2203,
    # 定义键名 'exists'，对应的值是十六进制数 0x2203
    'fallingdotseq'          : 0x2252,
    # 定义键名 'fallingdotseq'，对应的值是十六进制数 0x2252
    'flat'                   : 0x266d,
    # 定义键名 'flat'，对应的值是十六进制数 0x266d
    'forall'                 : 0x2200,
    # 定义键名 'forall'，对应的值是十六进制数 0x2200
    'frakC'                  : 0x212d,
    # 定义键名 'frakC'，对应的值是十六进制数 0x212d
    'frakZ'                  : 0x2128,
    # 定义键名 'frakZ'，对应的值是十六进制数 0x2128
    'frown'                  : 0x2322,
    # 定义键名 'frown'，对应的值是十六进制数 0x2322
    'gamma'                  : 0x3b3,
    # 定义键名 'gamma'，对应的值是十六进制数 0x3b3
    'geq'                    : 0x2265,
    # 定义键名 'geq'，对应的值是十六进制数 0x2265
    'geqq'                   : 0x2267,
    # 定义键名 'geqq'，对应的值是十六进制数 0x2267
    'geqslant'               : 0x2a7e,
    # 定义键名 'geqslant'，对应的值是十六进制数 0x2a7e
    'gg'                     : 0x226b,
    # 定义键名 'gg'，对应的值是十六进制数 0x226b
    'ggg'                    : 0x22d9,
    # 定义键名 'ggg'，对应的值是十六进制数 0x22d9
    'gimel'                  : 0x2137,
    # 定义键名 'gimel'，对应的值是十六进制数 0x2137
    'gnapprox'               : 0x2a8a,
    # 定义键名 'gnapprox'，对应的值是十六进制数 0x2a8a
    'gneqq'                  : 0x2269,
    # 定义键名 'gneqq'，对应的值是十六进制数 0x2269
    'gnsim'                  : 0x22e7,
    # 定义键名 'gnsim'，对应的值是十六进制数 0x22e7
    'grave'                  : 0x300,
    # 定义键名 'grave'，对应的值是十六进制数 0x300
    'greater'                : 0x3e,
    # 定义键名 'greater'，对应的值是十六进制数 0x3e
    'gtrapprox'              : 0x2a86,
    # 定义键名 'gtrapprox'，对应的值是十六进制数 0x2a86
    'gtrdot'                 : 0x22d7,
    # 定义键名 'gtrdot'，对应的值是十六进制数 0x22d7
    'gtreqless'              : 0x22db,
    # 定义键名 'gtreqless'，对应的值是十六进制数 0x22db
    'gtreqqless'             : 0x2a8c,
    # 定义键名 'gtreqqless'，对应的值是十六进制数 0x2a8c
    'gtrless'                : 0x2277,
    # 定义键名 'gtrless'，对应的值是十六进制数 0x2277
    'gtrsim'                 : 0x2273,
    # 定义键名 'gtrsim'，对应的值是十六进制数 0x2273
    'guillemotleft'          : 0xab,
    # 定义键名 'guillemotleft'，对应的值是十六进制数 0xab
    'guillemotright'         : 0xbb,
    # 定义键名 'guillemotright'，对应的值是十六进制数 0xbb
    'guilsinglleft'          : 0x2039,
    # 定义键名 'guilsinglleft'，对应的值是十六进制数 0x2039
    'guilsinglright'         : 0x203a,
    # 定义键名 'guilsinglright'，对应的值是十六进制数 0x203a
    'hat'                    : 0x302,
    # 定义键名 'hat'，对应的值是十六进制数 0x302
    'hbar'                   : 0x127,
    # 定义键名 'hbar'，对应的值是十六进制数 0x127
    'heartsuit'              : 0x2661,
    # 定义键名 'heartsuit'，对应的值是十六进制数 0x2661
    'hermitmatrix'           : 0x22b9,
    # 定义键名 'hermitmatrix'，对应的值是十六进制数 0x22b9
    'hookleftarrow'          : 0x21a9,
    # 定义键名 'hookleftarrow'，对应的值是十六进制数 0x21a9
    'hookrightarrow'
    'lbrack'                 : 0x5b,          # 定义键 'lbrack'，对应值为十六进制数 0x5b
    'lceil'                  : 0x2308,       # 定义键 'lceil'，对应值为十六进制数 0x2308
    'ldots'                  : 0x2026,       # 定义键 'ldots'，对应值为十六进制数 0x2026
    'leadsto'                : 0x21dd,       # 定义键 'leadsto'，对应值为十六进制数 0x21dd
    'leftarrow'              : 0x2190,       # 定义键 'leftarrow'，对应值为十六进制数 0x2190
    'leftarrowtail'          : 0x21a2,       # 定义键 'leftarrowtail'，对应值为十六进制数 0x21a2
    'leftbrace'              : 0x7b,         # 定义键 'leftbrace'，对应值为十六进制数 0x7b
    'leftharpoonaccent'      : 0x20d0,       # 定义键 'leftharpoonaccent'，对应值为十六进制数 0x20d0
    'leftharpoondown'        : 0x21bd,       # 定义键 'leftharpoondown'，对应值为十六进制数 0x21bd
    'leftharpoonup'          : 0x21bc,       # 定义键 'leftharpoonup'，对应值为十六进制数 0x21bc
    'leftleftarrows'         : 0x21c7,       # 定义键 'leftleftarrows'，对应值为十六进制数 0x21c7
    'leftparen'              : 0x28,         # 定义键 'leftparen'，对应值为十六进制数 0x28
    'leftrightarrow'         : 0x2194,       # 定义键 'leftrightarrow'，对应值为十六进制数 0x2194
    'leftrightarrows'        : 0x21c6,       # 定义键 'leftrightarrows'，对应值为十六进制数 0x21c6
    'leftrightharpoons'      : 0x21cb,       # 定义键 'leftrightharpoons'，对应值为十六进制数 0x21cb
    'leftrightsquigarrow'    : 0x21ad,       # 定义键 'leftrightsquigarrow'，对应值为十六进制数 0x21ad
    'leftsquigarrow'         : 0x219c,       # 定义键 'leftsquigarrow'，对应值为十六进制数 0x219c
    'leftthreetimes'         : 0x22cb,       # 定义键 'leftthreetimes'，对应值为十六进制数 0x22cb
    'leq'                    : 0x2264,       # 定义键 'leq'，对应值为十六进制数 0x2264
    'leqq'                   : 0x2266,       # 定义键 'leqq'，对应值为十六进制数 0x2266
    'leqslant'               : 0x2a7d,       # 定义键 'leqslant'，对应值为十六进制数 0x2a7d
    'less'                   : 0x3c,         # 定义键 'less'，对应值为十六进制数 0x3c
    'lessapprox'             : 0x2a85,       # 定义键 'lessapprox'，对应值为十六进制数 0x2a85
    'lessdot'                : 0x22d6,       # 定义键 'lessdot'，对应值为十六进制数 0x22d6
    'lesseqgtr'              : 0x22da,       # 定义键 'lesseqgtr'，对应值为十六进制数 0x22da
    'lesseqqgtr'             : 0x2a8b,       # 定义键 'lesseqqgtr'，对应值为十六进制数 0x2a8b
    'lessgtr'                : 0x2276,       # 定义键 'lessgtr'，对应值为十六进制数 0x2276
    'lesssim'                : 0x2272,       # 定义键 'lesssim'，对应值为十六进制数 0x2272
    'lfloor'                 : 0x230a,       # 定义键 'lfloor'，对应值为十六进制数 0x230a
    'lgroup'                 : 0x27ee,       # 定义键 'lgroup'，对应值为十六进制数 0x27ee
    'lhd'                    : 0x25c1,       # 定义键 'lhd'，对应值为十六进制数 0x25c1
    'll'                     : 0x226a,       # 定义键 'll'，对应值为十六进制数 0x226a
    'llcorner'               : 0x231e,       # 定义键 'llcorner'，对应值为十六进制数 0x231e
    'lll'                    : 0x22d8,       # 定义键 'lll'，对应值为十六进制数 0x22d8
    'lnapprox'               : 0x2a89,       # 定义键 'lnapprox'，对应值为十六进制数 0x2a89
    'lneqq'                  : 0x2268,       # 定义键 'lneqq'，对应值为十六进制数 0x2268
    'lnsim'                  : 0x22e6,       # 定义键 'lnsim'，对应值为十六进制数 0x22e6
    'longleftarrow'          : 0x27f5,       # 定义键 'longleftarrow'，对应值为十六进制数 0x27f5
    'longleftrightarrow'     : 0x27f7,       # 定义键 'longleftrightarrow'，对应值为十六进制数 0x27f7
    'longmapsto'             : 0x27fc,       # 定义键 'longmapsto'，对应值为十六进制数 0x27fc
    'longrightarrow'         : 0x27f6,       # 定义键 'longrightarrow'，对应值为十六进制数 0x27f6
    'looparrowleft'          : 0x21ab,       # 定义键 'looparrowleft'，对应值为十六进制数 0x21ab
    'looparrowright'         : 0x21ac,       # 定义键 'looparrowright'，对应值为十六进制数 0x21ac
    'lq'                     : 0x2018,       # 定义键 'lq'，对应值为十六进制数 0x2018
    'lrcorner'               : 0x231f,       # 定义键 'lrcorner'，对应值为十六进制数 0x231f
    'ltimes'                 : 0x22c9,       # 定义键 'ltimes'，对应值为十六进制数 0x22c9
    'macron'                 : 0xaf,         # 定义键 'macron'，对应值为十六进制数 0xaf
    'maltese'                : 0x2720,       # 定义键 'maltese'，对应值为十六进制数 0x2720
    'mapsdown'               : 0x21a7,       # 定义
    'nequiv'                 : 0x2262,
    'nexists'                : 0x2204,
    'ngeq'                   : 0x2271,
    'ngtr'                   : 0x226f,
    'ngtrless'               : 0x2279,
    'ngtrsim'                : 0x2275,
    'ni'                     : 0x220b,
    'niobar'                 : 0x22fe,
    'nis'                    : 0x22fc,
    'nisd'                   : 0x22fa,
    'nleftarrow'             : 0x219a,
    'nleftrightarrow'        : 0x21ae,
    'nleq'                   : 0x2270,
    'nless'                  : 0x226e,
    'nlessgtr'               : 0x2278,
    'nlesssim'               : 0x2274,
    'nmid'                   : 0x2224,
    'not'                    : 0x338,
    'notin'                  : 0x2209,
    'notsmallowns'           : 0x220c,
    'nparallel'              : 0x2226,
    'nprec'                  : 0x2280,
    'npreccurlyeq'           : 0x22e0,
    'nrightarrow'            : 0x219b,
    'nsim'                   : 0x2241,
    'nsimeq'                 : 0x2244,
    'nsqsubseteq'            : 0x22e2,
    'nsqsupseteq'            : 0x22e3,
    'nsubset'                : 0x2284,
    'nsubseteq'              : 0x2288,
    'nsucc'                  : 0x2281,
    'nsucccurlyeq'           : 0x22e1,
    'nsupset'                : 0x2285,
    'nsupseteq'              : 0x2289,
    'ntriangleleft'          : 0x22ea,
    'ntrianglelefteq'        : 0x22ec,
    'ntriangleright'         : 0x22eb,
    'ntrianglerighteq'       : 0x22ed,
    'nu'                     : 0x3bd,
    'nvDash'                 : 0x22ad,
    'nvdash'                 : 0x22ac,
    'nwarrow'                : 0x2196,
    'o'                      : 0xf8,
    'obar'                   : 0x233d,
    'ocirc'                  : 0x30a,
    'odot'                   : 0x2299,
    'oe'                     : 0x153,
    'oequal'                 : 0x229c,
    'oiiint'                 : 0x2230,
    'oiint'                  : 0x222f,
    'oint'                   : 0x222e,
    'omega'                  : 0x3c9,
    'ominus'                 : 0x2296,
    'oplus'                  : 0x2295,
    'origof'                 : 0x22b6,
    'oslash'                 : 0x2298,
    'otimes'                 : 0x2297,
    'overarc'                : 0x311,
    'overleftarrow'          : 0x20d6,
    'overleftrightarrow'     : 0x20e1,
    'parallel'               : 0x2225,
    'partial'                : 0x2202,
    'perp'                   : 0x27c2,
    'perthousand'            : 0x2030,
    'phi'                    : 0x3d5,
    'pi'                     : 0x3c0,
    'pitchfork'              : 0x22d4,
    'plus'                   : 0x2b,
    'pm'                     : 0xb1,
    'prec'                   : 0x227a,
    'precapprox'             : 0x2ab7,
    'preccurlyeq'            : 0x227c,
    'preceq'                 : 0x227c,
    'precnapprox'            : 0x2ab9,
    'precnsim'               : 0x22e8,
    'precsim'                : 0x227e,
    'prime'                  : 0x2032,
    {
        'prod'                   : 0x220f,  # Unicode代码点，表示数学符号“∏”（乘积）
        'propto'                 : 0x221d,  # Unicode代码点，表示数学符号“∝”（比例）
        'prurel'                 : 0x22b0,  # Unicode代码点，表示数学符号“⊰”（关系符号）
        'psi'                    : 0x3c8,   # Unicode代码点，表示希腊字母“ψ”
        'quad'                   : 0x2003,  # Unicode代码点，表示空格符号“ ”
        'questeq'                : 0x225f,  # Unicode代码点，表示数学符号“≟”（等号与问号重叠）
        'rangle'                 : 0x27e9,  # Unicode代码点，表示数学符号“⟩”（右尖括号）
        'rasp'                   : 0x2bc,   # Unicode代码点，表示音标符号“ʼ”（标音符号）
        'ratio'                  : 0x2236,  # Unicode代码点，表示数学符号“∶”（比率）
        'rbrace'                 : 0x7d,    # Unicode代码点，表示字符“}”（右大括号）
        'rbrack'                 : 0x5d,    # Unicode代码点，表示字符“]”（右方括号）
        'rceil'                  : 0x2309,  # Unicode代码点，表示数学符号“⌉”（右上角括号）
        'rfloor'                 : 0x230b,  # Unicode代码点，表示数学符号“⌋”（右下角括号）
        'rgroup'                 : 0x27ef,  # Unicode代码点，表示数学符号“⟯”（右分组括号）
        'rhd'                    : 0x25b7,  # Unicode代码点，表示数学符号“▷”（右向尖角）
        'rho'                    : 0x3c1,   # Unicode代码点，表示希腊字母“ρ”
        'rightModels'            : 0x22ab,  # Unicode代码点，表示数学符号“⊫”（右模型）
        'rightangle'             : 0x221f,  # Unicode代码点，表示数学符号“∟”（直角）
        'rightarrow'             : 0x2192,  # Unicode代码点，表示字符“→”（右箭头）
        'rightarrowbar'          : 0x21e5,  # Unicode代码点，表示数学符号“⇥”（带尾箭头向右）
        'rightarrowtail'         : 0x21a3,  # Unicode代码点，表示数学符号“↣”（右尾箭头）
        'rightassert'            : 0x22a6,  # Unicode代码点，表示数学符号“⊦”（右断言）
        'rightbrace'             : 0x7d,    # Unicode代码点，表示字符“}”（右大括号）
        'rightharpoonaccent'     : 0x20d1,  # Unicode代码点，表示数学符号“⃑”（右哈希箭头上方）
        'rightharpoondown'       : 0x21c1,  # Unicode代码点，表示数学符号“⇁”（右哈希箭头向下）
        'rightharpoonup'         : 0x21c0,  # Unicode代码点，表示数学符号“↼”（右哈希箭头向上）
        'rightleftarrows'        : 0x21c4,  # Unicode代码点，表示数学符号“⇄”（右左箭头）
        'rightleftharpoons'      : 0x21cc,  # Unicode代码点，表示数学符号“⇌”（右左哈希箭头）
        'rightparen'             : 0x29,    # Unicode代码点，表示字符“)”（右括号）
        'rightrightarrows'       : 0x21c9,  # Unicode代码点，表示数学符号“⇉”（右双箭头）
        'rightsquigarrow'        : 0x219d,  # Unicode代码点，表示数学符号“↝”（右波浪箭头）
        'rightthreetimes'        : 0x22cc,  # Unicode代码点，表示数学符号“⋌”（右三次乘积）
        'rightzigzagarrow'       : 0x21dd,  # Unicode代码点，表示数学符号“⇝”（右波浪箭头）
        'ring'                   : 0x2da,   # Unicode代码点，表示字符“˚”（环上标）
        'risingdotseq'           : 0x2253,  # Unicode代码点，表示数学符号“≓”（上升点序）
        'rq'                     : 0x2019,  # Unicode代码点，表示字符“’”（右单引号）
        'rtimes'                 : 0x22ca,  # Unicode代码点，表示数学符号“⋊”（右乘）
        'scrB'                   : 0x212c,  # Unicode代码点，表示数学花体字母“ℬ”
        'scrE'                   : 0x2130,  # Unicode代码点，表示数学花体字母“ℰ”
        'scrF'                   : 0x2131,  # Unicode代码点，表示数学花体字母“ℱ”
        'scrH'                   : 0x210b,  # Unicode代码点，表示数学花体字母“ℋ”
        'scrI'                   : 0x2110,  # Unicode代码点，表示数学花体字母“ℐ”
        'scrL'                   : 0x2112,  # Unicode代码点，表示数学花体字母“ℒ”
        'scrM'                   : 0x2133,  # Unicode代码点，表示数学花体字母“ℳ”
        'scrR'                   : 0x211b,  # Unicode代码点，表示数学花体字母“ℛ”
        'scre'                   : 0x212f,  # Unicode代码点，表示数学花体字母“ℯ”
        'scrg'                   : 0x210a,  # Unicode代码点，表示数学花体字母“ℊ”
        'scro'                   : 0x2134,  # Unicode代码点，表示数学花体字母“ℴ”
        'scurel'                 : 0x22b1,  # Unicode代码点，表示数学符号“⊱”（关系符号）
        'searrow'                : 0x2198,  # Unicode代码点，表示数学符号“↘”（南东箭头）
        'setminus'               : 0x2216,  # Unicode代码点，表示数学符号“∖”（集合差）
        'sharp'                  : 0x266f,  # Unicode代码点，表示音符号“♯”（升号）
        'sigma'                  : 0x3c3,   # Unicode代码点，表示希腊字母“σ”
        'sim'                    : 0x223c,  # Unicode代码点，表示数学符号“∼”（相似）
        'simeq'                  : 0x2243,  # Unicode代码点，表示数学符号“≃”（近似）
        'simneqq'                : 0x2246,  # Unicode代码点，表示数学符号“≆”（非等于的相似）
        'sinewave'               : 0x223f,  # Unicode代码点，表示数学符号“∿”（正弦波）
        'slash'                  : 0x2215,  # Unicode代码点，表示字符“∕”（斜杠）
        'smallin'                : 0x220a,  # Unicode代码点，表示数学符号“∊”（属于）
        'smallintclockwise'      : 0x
    {
        'ss'                     : 0xdf,
        'star'                   : 0x22c6,
        'stareq'                 : 0x225b,
        'sterling'               : 0xa3,
        'subset'                 : 0x2282,
        'subseteq'               : 0x2286,
        'subseteqq'              : 0x2ac5,
        'subsetneq'              : 0x228a,
        'subsetneqq'             : 0x2acb,
        'succ'                   : 0x227b,
        'succapprox'             : 0x2ab8,
        'succcurlyeq'            : 0x227d,
        'succeq'                 : 0x227d,
        'succnapprox'            : 0x2aba,
        'succnsim'               : 0x22e9,
        'succsim'                : 0x227f,
        'sum'                    : 0x2211,
        'supset'                 : 0x2283,
        'supseteq'               : 0x2287,
        'supseteqq'              : 0x2ac6,
        'supsetneq'              : 0x228b,
        'supsetneqq'             : 0x2acc,
        'swarrow'                : 0x2199,
        't'                      : 0x361,
        'tau'                    : 0x3c4,
        'textasciiacute'         : 0xb4,
        'textasciicircum'        : 0x5e,
        'textasciigrave'         : 0x60,
        'textasciitilde'         : 0x7e,
        'textexclamdown'         : 0xa1,
        'textquestiondown'       : 0xbf,
        'textquotedblleft'       : 0x201c,
        'textquotedblright'      : 0x201d,
        'therefore'              : 0x2234,
        'theta'                  : 0x3b8,
        'thickspace'             : 0x2005,
        'thorn'                  : 0xfe,
        'tilde'                  : 0x303,
        'times'                  : 0xd7,
        'to'                     : 0x2192,
        'top'                    : 0x22a4,
        'triangle'               : 0x25b3,
        'triangledown'           : 0x25bf,
        'triangleeq'             : 0x225c,
        'triangleleft'           : 0x25c1,
        'trianglelefteq'         : 0x22b4,
        'triangleq'              : 0x225c,
        'triangleright'          : 0x25b7,
        'trianglerighteq'        : 0x22b5,
        'turnednot'              : 0x2319,
        'twoheaddownarrow'       : 0x21a1,
        'twoheadleftarrow'       : 0x219e,
        'twoheadrightarrow'      : 0x21a0,
        'twoheaduparrow'         : 0x219f,
        'ulcorner'               : 0x231c,
        'underbar'               : 0x331,
        'unlhd'                  : 0x22b4,
        'unrhd'                  : 0x22b5,
        'uparrow'                : 0x2191,
        'updownarrow'            : 0x2195,
        'updownarrowbar'         : 0x21a8,
        'updownarrows'           : 0x21c5,
        'upharpoonleft'          : 0x21bf,
        'upharpoonright'         : 0x21be,
        'uplus'                  : 0x228e,
        'upsilon'                : 0x3c5,
        'upuparrows'             : 0x21c8,
        'urcorner'               : 0x231d,
        'vDash'                  : 0x22a8,
        'varepsilon'             : 0x3b5,
        'varisinobar'            : 0x22f6,
        'varisins'               : 0x22f3,
        'varkappa'               : 0x3f0,
        'varlrtriangle'          : 0x22bf,
        'varniobar'              : 0x22fd,
        'varnis'                 : 0x22fb,
        'varnothing'             : 0x2205,
    }
    
    
    注释：
    
    {
        'ss'                     : 0xdf,           # 字符 'ss' 的 Unicode 码点
        'star'                   : 0x22c6,         # 星号符的 Unicode 码点
        'stareq'                 : 0x225b,         # 星等于符的 Unicode 码点
        'sterling'               : 0xa3,           # 英镑符的 Unicode 码点
        'subset'                 : 0x2282,         # 子集符的 Unicode 码点
        'subseteq'               : 0x2286,         # 子集等于符的 Unicode 码点
        'subseteqq'              : 0x2ac5,         # 子集等于或等于符的 Unicode 码点
        'subsetneq'              : 0x228a,         # 子集不等于符的 Unicode 码点
        'subsetneqq'             : 0x2acb,         # 子集不等于或等于符的 Unicode 码点
        'succ'                   : 0x227b,         # 成功符的 Unicode 码点
        'succapprox'             : 0x2ab8,         # 成功近似符的 Unicode 码点
        'succcurlyeq'            : 0x227d,         # 成功花括号等于符的 Unicode 码点
        'succeq'                 : 0x227d,         # 成功等于符的 Unicode 码点
        'succnapprox'            : 0x2aba,         # 成功不近似符的 Unicode 码点
        'succnsim'               : 0x22e9,         # 成功不相似符的 Unicode 码点
        'succsim'                : 0x227f,         # 成功相似符的 Unicode 码点
        'sum'                    : 0x2211,         # 总和符的 Unicode 码点
        'supset'                 : 0x2283,         # 超集符的 Unicode 码点
        'supseteq'               : 0x2287,         # 超集等于符的 Unicode 码点
        'supseteqq'              : 0x2ac6,         # 超集等于或等于符的 Unicode 码点
        'supsetneq'              : 0x228b,         # 超集不等于符的 Unicode 码点
        'supsetneqq'             : 0x2acc,         # 超集不等于或等于符的 Unicode 码点
        'swarrow'                : 0x2199,         # 西南箭头符的 Unicode 码点
        't'                      : 0x361,          # 字符 't' 的 Unicode 码点
        'tau'                    : 0x3c4,          # 希腊字母 τ (tau) 的 Unicode 码点
        'textasciiacute'         : 0xb4,           # ASCII 尖音符的 Unicode 码点
        'textasciicircum'        : 0x5e,           # ASCII 圆形符的 Unicode 码点
        'textasciigrave'         : 0x60,           # ASCII 重音符的 Unicode 码点
        'textasciitilde'         : 0x7e,           # ASCII 波浪符的 Unicode 码点
        'textexclamdown'         : 0xa1,           # 倒置感叹符的 Unicode 码点
        'textquestiondown'       : 0xbf,           # 倒置问号符的 Unicode 码点
        'textquotedblleft'       : 0x201c,         # 左双引号的 Unicode 码点
        'textquotedblright'      : 0x201d,         # 右双引号的 Unicode 码点
        'therefore'              : 0x2234,         # 因此符的 Unicode 码点
        'theta'
    {
        'varphi'                 : 0x3c6,     # 符号 'varphi' 对应的 Unicode 编码
        'varpi'                  : 0x3d6,     # 符号 'varpi' 对应的 Unicode 编码
        'varpropto'              : 0x221d,    # 符号 'varpropto' 对应的 Unicode 编码
        'varrho'                 : 0x3f1,     # 符号 'varrho' 对应的 Unicode 编码
        'varsigma'               : 0x3c2,     # 符号 'varsigma' 对应的 Unicode 编码
        'vartheta'               : 0x3d1,     # 符号 'vartheta' 对应的 Unicode 编码
        'vartriangle'            : 0x25b5,    # 符号 'vartriangle' 对应的 Unicode 编码
        'vartriangleleft'        : 0x22b2,    # 符号 'vartriangleleft' 对应的 Unicode 编码
        'vartriangleright'       : 0x22b3,    # 符号 'vartriangleright' 对应的 Unicode 编码
        'vdash'                  : 0x22a2,    # 符号 'vdash' 对应的 Unicode 编码
        'vdots'                  : 0x22ee,    # 符号 'vdots' 对应的 Unicode 编码
        'vec'                    : 0x20d7,    # 符号 'vec' 对应的 Unicode 编码
        'vee'                    : 0x2228,    # 符号 'vee' 对应的 Unicode 编码
        'veebar'                 : 0x22bb,    # 符号 'veebar' 对应的 Unicode 编码
        'veeeq'                  : 0x225a,    # 符号 'veeeq' 对应的 Unicode 编码
        'vert'                   : 0x7c,      # 符号 'vert' 对应的 Unicode 编码
        'wedge'                  : 0x2227,    # 符号 'wedge' 对应的 Unicode 编码
        'wedgeq'                 : 0x2259,    # 符号 'wedgeq' 对应的 Unicode 编码
        'widebar'                : 0x305,     # 符号 'widebar' 对应的 Unicode 编码
        'widehat'                : 0x302,     # 符号 'widehat' 对应的 Unicode 编码
        'widetilde'              : 0x303,     # 符号 'widetilde' 对应的 Unicode 编码
        'wp'                     : 0x2118,    # 符号 'wp' 对应的 Unicode 编码
        'wr'                     : 0x2240,    # 符号 'wr' 对应的 Unicode 编码
        'xi'                     : 0x3be,     # 符号 'xi' 对应的 Unicode 编码
        'yen'                    : 0xa5,      # 符号 'yen' 对应的 Unicode 编码
        'zeta'                   : 0x3b6,     # 符号 'zeta' 对应的 Unicode 编码
        '{'                      : 0x7b,      # 符号 '{' 对应的 Unicode 编码
        '|'                      : 0x2016,    # 符号 '|' 对应的 Unicode 编码
        '}'                      : 0x7d,      # 符号 '}' 对应的 Unicode 编码
    }
# _stix_virtual_fonts 是一个字典，用于存储虚拟字体的信息，键是字符串类型的字体名称，值可以是一个字典或者列表。
_stix_virtual_fonts: dict[str, Union[dict[
    str, list[_EntryTypeIn]], list[_EntryTypeIn]]] = {
    # 'cal' 字体的映射，包含了一组元组，每个元组代表一个映射条目，包括起始字符、结束字符、目标字体和起始位置。
    'cal': [
        ("\N{LATIN CAPITAL LETTER A}",  # 起始字符为拉丁大写字母A
         "\N{LATIN CAPITAL LETTER Z}",  # 结束字符为拉丁大写字母Z
         "it",                          # 目标字体为'it'
         0xe22d),                       # 起始位置为0xe22d
    ],
    # 'frak' 字体的映射，包含了多个字体到映射条目的字典，每个字体有对应的元组列表。
    'frak': {
        "rm": [
            ("\N{LATIN CAPITAL LETTER A}",    # 起始字符为拉丁大写字母A
             "\N{LATIN CAPITAL LETTER B}",    # 结束字符为拉丁大写字母B
             "rm",                            # 目标字体为'rm'
             "\N{MATHEMATICAL FRAKTUR CAPITAL A}"),  # 起始位置为数学分数体大写字母A
            # 后续类似地描述了其它映射条目
            ("\N{LATIN CAPITAL LETTER C}", "\N{LATIN CAPITAL LETTER C}", "rm", "\N{BLACK-LETTER CAPITAL C}"),
            ("\N{LATIN CAPITAL LETTER D}", "\N{LATIN CAPITAL LETTER G}", "rm", "\N{MATHEMATICAL FRAKTUR CAPITAL D}"),
            ("\N{LATIN CAPITAL LETTER H}", "\N{LATIN CAPITAL LETTER H}", "rm", "\N{BLACK-LETTER CAPITAL H}"),
            ("\N{LATIN CAPITAL LETTER I}", "\N{LATIN CAPITAL LETTER I}", "rm", "\N{BLACK-LETTER CAPITAL I}"),
            ("\N{LATIN CAPITAL LETTER J}", "\N{LATIN CAPITAL LETTER Q}", "rm", "\N{MATHEMATICAL FRAKTUR CAPITAL J}"),
            ("\N{LATIN CAPITAL LETTER R}", "\N{LATIN CAPITAL LETTER R}", "rm", "\N{BLACK-LETTER CAPITAL R}"),
            ("\N{LATIN CAPITAL LETTER S}", "\N{LATIN CAPITAL LETTER Y}", "rm", "\N{MATHEMATICAL FRAKTUR CAPITAL S}"),
            ("\N{LATIN CAPITAL LETTER Z}", "\N{LATIN CAPITAL LETTER Z}", "rm", "\N{BLACK-LETTER CAPITAL Z}"),
            ("\N{LATIN SMALL LETTER A}", "\N{LATIN SMALL LETTER Z}", "rm", "\N{MATHEMATICAL FRAKTUR SMALL A}"),
        ],
        # 'bf' 字体的映射，包含了一组元组，每个元组代表一个映射条目，包括起始字符、结束字符、目标字体和起始位置。
        "bf": [
            ("\N{LATIN CAPITAL LETTER A}", "\N{LATIN CAPITAL LETTER Z}", "bf", "\N{MATHEMATICAL BOLD FRAKTUR CAPITAL A}"),
            ("\N{LATIN SMALL LETTER A}", "\N{LATIN SMALL LETTER Z}", "bf", "\N{MATHEMATICAL BOLD FRAKTUR SMALL A}"),
        ],
    },
}
    'scr': [
        # 列表的每个元素是一个元组，包含四个字符串，分别代表不同的字符形式
        ("\N{LATIN CAPITAL LETTER A}",
         "\N{LATIN CAPITAL LETTER A}",
         "it",
         "\N{MATHEMATICAL SCRIPT CAPITAL A}"),
        ("\N{LATIN CAPITAL LETTER B}",
         "\N{LATIN CAPITAL LETTER B}",
         "it",
         "\N{SCRIPT CAPITAL B}"),
        ("\N{LATIN CAPITAL LETTER C}",
         "\N{LATIN CAPITAL LETTER D}",
         "it",
         "\N{MATHEMATICAL SCRIPT CAPITAL C}"),
        ("\N{LATIN CAPITAL LETTER E}",
         "\N{LATIN CAPITAL LETTER F}",
         "it",
         "\N{SCRIPT CAPITAL E}"),
        ("\N{LATIN CAPITAL LETTER G}",
         "\N{LATIN CAPITAL LETTER G}",
         "it",
         "\N{MATHEMATICAL SCRIPT CAPITAL G}"),
        ("\N{LATIN CAPITAL LETTER H}",
         "\N{LATIN CAPITAL LETTER H}",
         "it",
         "\N{SCRIPT CAPITAL H}"),
        ("\N{LATIN CAPITAL LETTER I}",
         "\N{LATIN CAPITAL LETTER I}",
         "it",
         "\N{SCRIPT CAPITAL I}"),
        ("\N{LATIN CAPITAL LETTER J}",
         "\N{LATIN CAPITAL LETTER K}",
         "it",
         "\N{MATHEMATICAL SCRIPT CAPITAL J}"),
        ("\N{LATIN CAPITAL LETTER L}",
         "\N{LATIN CAPITAL LETTER L}",
         "it",
         "\N{SCRIPT CAPITAL L}"),
        ("\N{LATIN CAPITAL LETTER M}",
         "\N{LATIN CAPITAL LETTER M}",
         "it",
         "\N{SCRIPT CAPITAL M}"),
        ("\N{LATIN CAPITAL LETTER N}",
         "\N{LATIN CAPITAL LETTER Q}",
         "it",
         "\N{MATHEMATICAL SCRIPT CAPITAL N}"),
        ("\N{LATIN CAPITAL LETTER R}",
         "\N{LATIN CAPITAL LETTER R}",
         "it",
         "\N{SCRIPT CAPITAL R}"),
        ("\N{LATIN CAPITAL LETTER S}",
         "\N{LATIN CAPITAL LETTER Z}",
         "it",
         "\N{MATHEMATICAL SCRIPT CAPITAL S}"),
        ("\N{LATIN SMALL LETTER A}",
         "\N{LATIN SMALL LETTER D}",
         "it",
         "\N{MATHEMATICAL SCRIPT SMALL A}"),
        ("\N{LATIN SMALL LETTER E}",
         "\N{LATIN SMALL LETTER E}",
         "it",
         "\N{SCRIPT SMALL E}"),
        ("\N{LATIN SMALL LETTER F}",
         "\N{LATIN SMALL LETTER F}",
         "it",
         "\N{MATHEMATICAL SCRIPT SMALL F}"),
        ("\N{LATIN SMALL LETTER G}",
         "\N{LATIN SMALL LETTER G}",
         "it",
         "\N{SCRIPT SMALL G}"),
        ("\N{LATIN SMALL LETTER H}",
         "\N{LATIN SMALL LETTER N}",
         "it",
         "\N{MATHEMATICAL SCRIPT SMALL H}"),
        ("\N{LATIN SMALL LETTER O}",
         "\N{LATIN SMALL LETTER O}",
         "it",
         "\N{SCRIPT SMALL O}"),
        ("\N{LATIN SMALL LETTER P}",
         "\N{LATIN SMALL LETTER Z}",
         "it",
         "\N{MATHEMATICAL SCRIPT SMALL P}"),
    ],
    },
    # 定义一个包含三个元组的列表 'tt'
    'tt': [
        # 第一个元组，包含特殊字符 '\N{DIGIT ZERO}', '\N{DIGIT NINE}', 'rm', 和数学单间距数字 '0'
        ("\N{DIGIT ZERO}",
         "\N{DIGIT NINE}",
         "rm",
         "\N{MATHEMATICAL MONOSPACE DIGIT ZERO}"),
        # 第二个元组，包含大写拉丁字母 'A', 'Z', 'rm', 和数学单间距大写字母 'A'
        ("\N{LATIN CAPITAL LETTER A}",
         "\N{LATIN CAPITAL LETTER Z}",
         "rm",
         "\N{MATHEMATICAL MONOSPACE CAPITAL A}"),
        # 第三个元组，包含小写拉丁字母 'a', 'z', 'rm', 和数学单间距小写字母 'a'
        ("\N{LATIN SMALL LETTER A}",
         "\N{LATIN SMALL LETTER Z}",
         "rm",
         "\N{MATHEMATICAL MONOSPACE SMALL A}")
    ],
# 定义函数重载，处理单个输入，返回单个输出
@overload
def _normalize_stix_fontcodes(d: _EntryTypeIn) -> _EntryTypeOut: ...

# 定义函数重载，处理列表输入，返回列表输出
@overload
def _normalize_stix_fontcodes(d: list[_EntryTypeIn]) -> list[_EntryTypeOut]: ...

# 定义函数重载，处理字典输入，返回字典输出
@overload
def _normalize_stix_fontcodes(d: dict[str, list[_EntryTypeIn] |
                                      dict[str, list[_EntryTypeIn]]]
                              ) -> dict[str, list[_EntryTypeOut] |
                                        dict[str, list[_EntryTypeOut]]]: ...

# 定义函数_normalize_stix_fontcodes，根据输入类型进行不同的归一化处理
def _normalize_stix_fontcodes(d):
    # 如果输入是元组，将其中的字符串转换为对应的 Unicode 码点，保持其他不变
    if isinstance(d, tuple):
        return tuple(ord(x) if isinstance(x, str) and len(x) == 1 else x for x in d)
    # 如果输入是列表，对列表中的每个元素递归调用_normalize_stix_fontcodes进行归一化处理
    elif isinstance(d, list):
        return [_normalize_stix_fontcodes(x) for x in d]
    # 如果输入是字典，对字典中的每个键值对递归调用_normalize_stix_fontcodes进行归一化处理
    elif isinstance(d, dict):
        return {k: _normalize_stix_fontcodes(v) for k, v in d.items()}

# 定义字典stix_virtual_fonts，用于存储归一化后的STIX虚拟字体信息，可能是字典或列表
stix_virtual_fonts: dict[str, Union[dict[str, list[_EntryTypeOut]],
                                    list[_EntryTypeOut]]]
# 使用_normalize_stix_fontcodes函数对_stix_virtual_fonts进行归一化处理，赋值给stix_virtual_fonts
stix_virtual_fonts = _normalize_stix_fontcodes(_stix_virtual_fonts)

# 归一化完成后，删除不再需要的_stix_virtual_fonts列表，释放内存
del _stix_virtual_fonts

# 定义字典stix_glyph_fixes，用于存储需要修正的STIX字形映射关系
stix_glyph_fixes = {
    # 修正错误的字形映射：将0x22d2映射到0x22d3，将0x22d3映射到0x22d2
    0x22d2: 0x22d3,
    0x22d3: 0x22d2,
}
```