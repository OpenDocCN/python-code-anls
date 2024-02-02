# `ZeroNet\plugins\UiFileManager\media\codemirror\mode\python.js`

```py
// CodeMirror, 版权归 Marijn Haverbeke 和其他人所有
// 根据 MIT 许可证分发：https://codemirror.net/LICENSE

// 匿名函数，接受一个 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 生成包含指定单词的正则表达式
  function wordRegexp(words) {
    return new RegExp("^((" + words.join(")|(") + "))\\b");
  }

  // 包含逻辑运算符的正则表达式
  var wordOperators = wordRegexp(["and", "or", "not", "is"]);
  // 常见关键字
  var commonKeywords = ["as", "assert", "break", "class", "continue",
                        "def", "del", "elif", "else", "except", "finally",
                        "for", "from", "global", "if", "import",
                        "lambda", "pass", "raise", "return",
                        "try", "while", "with", "yield", "in"];
  // 常见内置函数
  var commonBuiltins = ["abs", "all", "any", "bin", "bool", "bytearray", "callable", "chr",
                        "classmethod", "compile", "complex", "delattr", "dict", "dir", "divmod",
                        "enumerate", "eval", "filter", "float", "format", "frozenset",
                        "getattr", "globals", "hasattr", "hash", "help", "hex", "id",
                        "input", "int", "isinstance", "issubclass", "iter", "len",
                        "list", "locals", "map", "max", "memoryview", "min", "next",
                        "object", "oct", "open", "ord", "pow", "property", "range",
                        "repr", "reversed", "round", "set", "setattr", "slice",
                        "sorted", "staticmethod", "str", "sum", "super", "tuple",
                        "type", "vars", "zip", "__import__", "NotImplemented",
                        "Ellipsis", "__debug__"];
  // 注册 Python 语言的关键字和内置函数
  CodeMirror.registerHelper("hintWords", "python", commonKeywords.concat(commonBuiltins));

  // 定义顶层状态
  function top(state) {
    # 返回当前作用域的最后一个状态
    return state.scopes[state.scopes.length - 1];
  }

  # 定义 Python 语言的模式
  CodeMirror.defineMode("python", function(conf, parserConf) {
    # 定义错误类名
    var ERRORCLASS = "error";

    # 定义分隔符
    var delimiters = parserConf.delimiters || parserConf.singleDelimiters || /^[\(\)\[\]\{\}@,:`=;\.\\]/;
    # 与旧的配置系统向后兼容
    var operators = [parserConf.singleOperators, parserConf.doubleOperators, parserConf.doubleDelimiters, parserConf.tripleDelimiters,
                     parserConf.operators || /^([-+*/%\/&|^]=?|[<>=]+|\/\/=?|\*\*=?|!=|[~!@]|\.\.\.)/]
    for (var i = 0; i < operators.length; i++) if (!operators[i]) operators.splice(i--, 1)

    # 定义悬挂缩进
    var hangingIndent = parserConf.hangingIndent || conf.indentUnit;

    # 定义关键字和内置函数
    var myKeywords = commonKeywords, myBuiltins = commonBuiltins;
    if (parserConf.extra_keywords != undefined)
      myKeywords = myKeywords.concat(parserConf.extra_keywords);

    if (parserConf.extra_builtins != undefined)
      myBuiltins = myBuiltins.concat(parserConf.extra_builtins);

    # 判断是否为 Python 3
    var py3 = !(parserConf.version && Number(parserConf.version) < 3)
    if (py3) {
      # 根据 PEP-0465，@ 也是一个操作符
      var identifiers = parserConf.identifiers|| /^[_A-Za-z\u00A1-\uFFFF][_A-Za-z0-9\u00A1-\uFFFF]*/;
      myKeywords = myKeywords.concat(["nonlocal", "False", "True", "None", "async", "await"]);
      myBuiltins = myBuiltins.concat(["ascii", "bytes", "exec", "print"]);
      # 定义字符串前缀的正则表达式
      var stringPrefixes = new RegExp("^(([rbuf]|(br)|(fr))?('{3}|\"{3}|['\"]))", "i");
    } else {
      // 如果不是以换行符开头，则使用默认的标识符正则表达式
      var identifiers = parserConf.identifiers|| /^[_A-Za-z][_A-Za-z0-9]*/;
      // 将自定义关键字添加到默认关键字列表中
      myKeywords = myKeywords.concat(["exec", "print"]);
      // 将自定义内置函数添加到默认内置函数列表中
      myBuiltins = myBuiltins.concat(["apply", "basestring", "buffer", "cmp", "coerce", "execfile",
                                      "file", "intern", "long", "raw_input", "reduce", "reload",
                                      "unichr", "unicode", "xrange", "False", "True", "None"]);
      // 创建字符串前缀的正则表达式
      var stringPrefixes = new RegExp("^(([rubf]|(ur)|(br))?('{3}|\"{3}|['\"]))", "i");
    }
    // 创建关键字的正则表达式
    var keywords = wordRegexp(myKeywords);
    // 创建内置函数的正则表达式
    var builtins = wordRegexp(myBuiltins);

    // tokenizers
    function tokenBase(stream, state) {
      // 判断是否是行首，并且上一个标记不是转义符
      var sol = stream.sol() && state.lastToken != "\\"
      if (sol) state.indent = stream.indentation()
      // 处理作用域变化
      if (sol && top(state).type == "py") {
        var scopeOffset = top(state).offset;
        if (stream.eatSpace()) {
          var lineOffset = stream.indentation();
          if (lineOffset > scopeOffset)
            pushPyScope(state);
          else if (lineOffset < scopeOffset && dedent(stream, state) && stream.peek() != "#")
            state.errorToken = true;
          return null;
        } else {
          var style = tokenBaseInner(stream, state);
          if (scopeOffset > 0 && dedent(stream, state))
            style += " " + ERRORCLASS;
          return style;
        }
      }
      return tokenBaseInner(stream, state);
    }

    }

    }
    # 创建一个函数，用于生成字符串标记
    def tokenStringFactory(delimiter, tokenOuter):
      # 去除分隔符开头的可能的空格字符
      while ("rubf".indexOf(delimiter.charAt(0).toLowerCase()) >= 0)
        delimiter = delimiter.substr(1);

      # 检查分隔符是否为单个字符
      var singleline = delimiter.length == 1;
      var OUTCLASS = "string";

      # 定义字符串标记函数
      def tokenString(stream, state):
        # 在未到行尾的情况下循环
        while (!stream.eol()):
          # 读取非特定字符的内容
          stream.eatWhile(/[^'"\\]/);
          # 如果遇到转义字符
          if (stream.eat("\\")):
            stream.next();
            # 如果是单行字符串且已到行尾，则返回字符串标记
            if (singleline && stream.eol())
              return OUTCLASS;
          # 如果匹配到分隔符，则设置状态为外部标记，并返回字符串标记
          else if (stream.match(delimiter)):
            state.tokenize = tokenOuter;
            return OUTCLASS;
          # 否则继续读取
          else:
            stream.eat(/['"]/);
        # 如果是单行字符串
        if (singleline):
          # 如果启用了单行字符串错误检查，则返回错误标记，否则设置状态为外部标记
          if (parserConf.singleLineStringErrors)
            return ERRORCLASS;
          else
            state.tokenize = tokenOuter;
        return OUTCLASS;
      tokenString.isString = true;
      return tokenString;

    # 定义一个函数，用于推入 Python 作用域
    def pushPyScope(state):
      # 循环直到栈顶为 Python 作用域
      while (top(state).type != "py") state.scopes.pop()
      # 将新的 Python 作用域推入栈中
      state.scopes.push({offset: top(state).offset + conf.indentUnit,
                         type: "py",
                         align: null})

    # 定义一个函数，用于推入括号作用域
    def pushBracketScope(stream, state, type):
      # 检查是否需要对齐
      var align = stream.match(/^([\s\[\{\(]|#.*)*$/, false) ? null : stream.column() + 1
      # 将新的括号作用域推入栈中
      state.scopes.push({offset: state.indent + hangingIndent,
                         type: type,
                         align: align})

    # 定义一个函数，用于缩进
    def dedent(stream, state):
      # 获取当前缩进值
      var indented = stream.indentation();
      # 循环直到栈顶作用域的偏移量小于当前缩进值
      while (state.scopes.length > 1 && top(state).offset > indented):
        # 如果栈顶不是 Python 作用域，则返回 True
        if (top(state).type != "py") return True;
        # 弹出栈顶作用域
        state.scopes.pop();
      # 返回栈顶作用域的偏移量是否不等于当前缩进值
      return top(state).offset != indented;
    // 定义一个函数，用于对输入的流进行词法分析
    function tokenLexer(stream, state) {
      // 如果流的位置在行首，设置状态为行首
      if (stream.sol()) state.beginningOfLine = true;

      // 调用状态中的tokenize方法对流进行词法分析，获取样式
      var style = state.tokenize(stream, state);
      // 获取当前流中的内容
      var current = stream.current();

      // 处理装饰器
      if (state.beginningOfLine && current == "@")
        // 如果在行首且当前字符为@，则判断后续是否为标识符，是则返回"meta"，否则根据py3返回"operator"或ERRORCLASS
        return stream.match(identifiers, false) ? "meta" : py3 ? "operator" : ERRORCLASS;

      // 如果当前字符不为空白字符，设置状态为非行首
      if (/\S/.test(current)) state.beginningOfLine = false;

      // 如果样式为"variable"或"builtin"，且上一个标记为"meta"，则样式设置为"meta"
      if ((style == "variable" || style == "builtin")
          && state.lastToken == "meta")
        style = "meta";

      // 处理作用域变化
      if (current == "pass" || current == "return")
        state.dedent += 1;

      // 如果当前字符为"lambda"，设置状态中的lambda为true
      if (current == "lambda") state.lambda = true;
      // 如果当前字符为":"且不是lambda表达式且当前作用域为"py"，则推入新的Python作用域
      if (current == ":" && !state.lambda && top(state).type == "py")
        pushPyScope(state);

      // 如果当前字符长度为1且样式不为"string"或"comment"
      if (current.length == 1 && !/string|comment/.test(style)) {
        // 获取当前字符在"[({"中的索引
        var delimiter_index = "[({".indexOf(current);
        // 如果索引不为-1，根据索引推入相应的括号作用域
        if (delimiter_index != -1)
          pushBracketScope(stream, state, "])}".slice(delimiter_index, delimiter_index+1));

        // 获取当前字符在"])}"中的索引
        delimiter_index = "])}".indexOf(current);
        // 如果索引不为-1
        if (delimiter_index != -1) {
          // 如果当前作用域类型与当前字符相同，设置缩进为作用域出栈的偏移量减去悬挂缩进
          if (top(state).type == current) state.indent = state.scopes.pop().offset - hangingIndent
          // 否则返回ERRORCLASS
          else return ERRORCLASS;
        }
      }
      // 如果需要减少缩进且流在行尾且当前作用域为"py"
      if (state.dedent > 0 && stream.eol() && top(state).type == "py") {
        // 如果作用域栈长度大于1，出栈
        if (state.scopes.length > 1) state.scopes.pop();
        // 减少缩进
        state.dedent -= 1;
      }

      // 返回样式
      return style;
    }
    # 定义名为 external 的对象
    var external = {
      # 定义 startState 方法，接受 basecolumn 参数
      startState: function(basecolumn) {
        # 返回包含初始状态信息的对象
        return {
          tokenize: tokenBase,  # 定义 tokenize 属性为 tokenBase 函数
          scopes: [{offset: basecolumn || 0, type: "py", align: null}],  # 定义 scopes 属性为包含初始状态信息的数组
          indent: basecolumn || 0,  # 定义 indent 属性为 basecolumn 或 0
          lastToken: null,  # 定义 lastToken 属性为 null
          lambda: false,  # 定义 lambda 属性为 false
          dedent: 0  # 定义 dedent 属性为 0
        };
      },

      # 定义 token 方法，接受 stream 和 state 参数
      token: function(stream, state) {
        var addErr = state.errorToken;  # 定义 addErr 变量为 state.errorToken
        if (addErr) state.errorToken = false;  # 如果 addErr 为真，则将 state.errorToken 设置为 false
        var style = tokenLexer(stream, state);  # 定义 style 变量为 tokenLexer 函数的返回值

        if (style && style != "comment")  # 如果 style 存在且不为 "comment"
          state.lastToken = (style == "keyword" || style == "punctuation") ? stream.current() : style;  # 如果 style 为 "keyword" 或 "punctuation"，则将 stream.current() 赋值给 state.lastToken，否则将 style 赋值给 state.lastToken
        if (style == "punctuation") style = null;  # 如果 style 为 "punctuation"，则将 style 设置为 null

        if (stream.eol() && state.lambda)  # 如果到达行尾且 state.lambda 为真
          state.lambda = false;  # 将 state.lambda 设置为 false
        return addErr ? style + " " + ERRORCLASS : style;  # 如果 addErr 为真，则返回 style + " " + ERRORCLASS，否则返回 style
      },

      # 定义 indent 方法，接受 state 和 textAfter 参数
      indent: function(state, textAfter) {
        if (state.tokenize != tokenBase)  # 如果 state.tokenize 不等于 tokenBase
          return state.tokenize.isString ? CodeMirror.Pass : 0;  # 如果 state.tokenize.isString 为真，则返回 CodeMirror.Pass，否则返回 0

        var scope = top(state), closing = scope.type == textAfter.charAt(0)  # 定义 scope 变量为 state 的顶部元素，定义 closing 变量为 scope.type 是否等于 textAfter 的第一个字符
        if (scope.align != null)  # 如果 scope.align 不为 null
          return scope.align - (closing ? 1 : 0)  # 返回 scope.align 减去 (closing 为真时的 1，否则为 0)
        else
          return scope.offset - (closing ? hangingIndent : 0)  # 返回 scope.offset 减去 (closing 为真时的 hangingIndent，否则为 0)
      },

      electricInput: /^\s*[\}\]\)]$/,  # 定义 electricInput 属性为正则表达式 /^\s*[\}\]\)]$/
      closeBrackets: {triples: "'\""},  # 定义 closeBrackets 属性为包含 triples 属性的对象
      lineComment: "#",  # 定义 lineComment 属性为 "#"
      fold: "indent"  # 定义 fold 属性为 "indent"
    };
    return external;  # 返回 external 对象
  });

  # 定义 MIME 类型为 "text/x-python" 的语法高亮
  CodeMirror.defineMIME("text/x-python", "python");

  # 定义名为 words 的函数，接受 str 参数，返回 str 以空格分割的数组
  var words = function(str) { return str.split(" "); };

  # 定义 MIME 类型为 "text/x-cython" 的语法高亮
  CodeMirror.defineMIME("text/x-cython", {
    name: "python",  # 定义 name 属性为 "python"
    extra_keywords: words("by cdef cimport cpdef ctypedef enum except "+
                          "extern gil include nogil property public "+
                          "readonly struct union DEF IF ELIF ELSE")  # 定义 extra_keywords 属性为 words 函数的返回值
  });
# 闭合了一个代码块或者函数的结束
```