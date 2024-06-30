# `D:\src\scipysrc\sympy\sympy\parsing\autolev\_antlr\autolevlexer.py`

```
# 由 `setup.py antlr` 自动生成的代码，不要手动编辑
#
# 导入必要的模块和库
from antlr4 import *                # 导入整个antlr4模块
from io import StringIO             # 导入StringIO类，用于处理字符串IO操作
import sys                          # 导入sys模块，用于系统相关操作
if sys.version_info[1] > 5:
    from typing import TextIO       # 导入TextIO类型，用于文本IO操作
else:
    from typing.io import TextIO    # 在Python版本小于等于5时，导入TextIO类型

# 定义一个函数，用于序列化ATN（自动机转移网络）
def serializedATN():
    # 初始化ATN序列化器，并反序列化ATN数据
    atn = ATNDeserializer().deserialize(serializedATN())
    
    # 创建决策到DFA的映射列表，每个决策对应一个DFA
    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]
    
    # 返回ATN对象
    return atn

# 定义一个词法分析器类 AutolevLexer，继承自 Lexer
class AutolevLexer(Lexer):
    
    # 初始化ATN对象为反序列化后的ATN对象
    atn = ATNDeserializer().deserialize(serializedATN())
    
    # 创建决策到DFA的映射列表，每个决策对应一个DFA
    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    # 定义词法单元的整数标记，每个标记对应一个词法单元类型
    T__0 = 1
    T__1 = 2
    T__2 = 3
    T__3 = 4
    T__4 = 5
    T__5 = 6
    T__6 = 7
    T__7 = 8
    T__8 = 9
    T__9 = 10
    T__10 = 11
    T__11 = 12
    T__12 = 13
    T__13 = 14
    T__14 = 15
    T__15 = 16
    T__16 = 17
    T__17 = 18
    T__18 = 19
    T__19 = 20
    T__20 = 21
    T__21 = 22
    T__22 = 23
    T__23 = 24
    T__24 = 25
    T__25 = 26
    Mass = 27
    Inertia = 28
    Input = 29
    Output = 30
    Save = 31
    UnitSystem = 32
    Encode = 33
    Newtonian = 34
    Frames = 35
    Bodies = 36
    Particles = 37
    Points = 38
    Constants = 39
    Specifieds = 40
    Imaginary = 41
    Variables = 42
    MotionVariables = 43
    INT = 44
    FLOAT = 45
    EXP = 46
    LINE_COMMENT = 47
    ID = 48
    WS = 49

    # 定义通道名称列表，包含默认通道和隐藏通道
    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    # 定义模式名称列表，只有一个默认模式
    modeNames = [ "DEFAULT_MODE" ]

    # 定义字面量名称列表，包含所有词法单元的字面量名称
    literalNames = [ "<INVALID>",
            "'['", "']'", "'='", "'+='", "'-='", "':='", "'*='", "'/='",
            "'^='", "','", "'''", "'('", "')'", "'{'", "'}'", "':'", "'+'",
            "'-'", "';'", "'.'", "'>'", "'0>'", "'1>>'", "'^'", "'*'", "'/'" ]

    # 定义符号名称列表，包含所有词法单元的符号名称
    symbolicNames = [ "<INVALID>",
            "Mass", "Inertia", "Input", "Output", "Save", "UnitSystem",
            "Encode", "Newtonian", "Frames", "Bodies", "Particles", "Points",
            "Constants", "Specifieds", "Imaginary", "Variables", "MotionVariables",
            "INT", "FLOAT", "EXP", "LINE_COMMENT", "ID", "WS" ]

    # 定义规则名称列表，包含所有词法规则的规则名称
    ruleNames = [ "T__0", "T__1", "T__2", "T__3", "T__4", "T__5", "T__6",
                  "T__7", "T__8", "T__9", "T__10", "T__11", "T__12", "T__13",
                  "T__14", "T__15", "T__16", "T__17", "T__18", "T__19",
                  "T__20", "T__21", "T__22", "T__23", "T__24", "T__25",
                  "Mass", "Inertia", "Input", "Output", "Save", "UnitSystem",
                  "Encode", "Newtonian", "Frames", "Bodies", "Particles",
                  "Points", "Constants", "Specifieds", "Imaginary", "Variables",
                  "MotionVariables", "DIFF", "DIGIT", "INT", "FLOAT", "EXP",
                  "LINE_COMMENT", "ID", "WS" ]

    # 定义语法文件名称
    grammarFileName = "Autolev.g4"
    # 初始化函数，用于实例化对象
    def __init__(self, input=None, output:TextIO = sys.stdout):
        # 调用父类的初始化方法，传入输入和输出参数
        super().__init__(input, output)
        # 检查版本号是否为 "4.11.1"，并进行必要的处理
        self.checkVersion("4.11.1")
        # 创建 LexerATNSimulator 实例，用于词法分析
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        # 初始化动作（actions）为 None
        self._actions = None
        # 初始化断言（predicates）为 None
        self._predicates = None
```