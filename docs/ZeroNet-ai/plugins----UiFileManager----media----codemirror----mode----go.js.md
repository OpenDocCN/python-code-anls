# `ZeroNet\plugins\UiFileManager\media\codemirror\mode\go.js`

```py
// 使用立即调用函数表达式（IIFE）来定义模块
(function(mod) {
  // 如果是 CommonJS 环境，使用 require 引入 CodeMirror
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 引入 CodeMirror
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接引入 CodeMirror
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义 Go 语言的模式
  CodeMirror.defineMode("go", function(config) {
    // 获取缩进单位
    var indentUnit = config.indentUnit;

    // 定义关键字
    var keywords = {
      "break":true, "case":true, "chan":true, "const":true, "continue":true,
      "default":true, "defer":true, "else":true, "fallthrough":true, "for":true,
      "func":true, "go":true, "goto":true, "if":true, "import":true,
      "interface":true, "map":true, "package":true, "range":true, "return":true,
      "select":true, "struct":true, "switch":true, "type":true, "var":true,
      "bool":true, "byte":true, "complex64":true, "complex128":true,
      "float32":true, "float64":true, "int8":true, "int16":true, "int32":true,
      "int64":true, "string":true, "uint8":true, "uint16":true, "uint32":true,
      "uint64":true, "int":true, "uint":true, "uintptr":true, "error": true,
      "rune":true
    };

    // 定义原子
    var atoms = {
      "true":true, "false":true, "iota":true, "nil":true, "append":true,
      "cap":true, "close":true, "complex":true, "copy":true, "delete":true, "imag":true,
      "len":true, "make":true, "new":true, "panic":true, "print":true,
      "println":true, "real":true, "recover":true
    };

    // 定义操作符字符
    var isOperatorChar = /[+\-*&^%:=<>!|\/]/;

    var curPunc;

    // 定义基本的 token 处理函数
    function tokenBase(stream, state) {
      var ch = stream.next();
      // 如果是引号，进入字符串 token 处理函数
      if (ch == '"' || ch == "'" || ch == "`") {
        state.tokenize = tokenString(ch);
        return state.tokenize(stream, state);
      }
    # 如果字符是数字或者小数点
    if (/[\d\.]/.test(ch)) {
      # 如果字符是小数点
      if (ch == ".") {
        # 匹配小数点后面的数字，包括科学计数法
        stream.match(/^[0-9]+([eE][\-+]?[0-9]+)?/);
      } else if (ch == "0") {
        # 如果字符是0，匹配16进制或者8进制数字
        stream.match(/^[xX][0-9a-fA-F]+/) || stream.match(/^0[0-7]+/);
      } else {
        # 匹配普通的数字，包括小数和科学计数法
        stream.match(/^[0-9]*\.?[0-9]*([eE][\-+]?[0-9]+)?/);
      }
      # 返回数字类型
      return "number";
    }
    # 如果字符是特殊符号
    if (/[\[\]{}\(\),;\:\.]/.test(ch)) {
      # 记录当前的特殊符号
      curPunc = ch;
      # 返回空
      return null;
    }
    # 如果字符是斜杠
    if (ch == "/") {
      # 如果斜杠后面是星号，进入注释状态
      if (stream.eat("*")) {
        state.tokenize = tokenComment;
        return tokenComment(stream, state);
      }
      # 如果斜杠后面是斜杠，跳过整行
      if (stream.eat("/")) {
        stream.skipToEnd();
        return "comment";
      }
    }
    # 如果字符是操作符
    if (isOperatorChar.test(ch)) {
      # 匹配整个操作符
      stream.eatWhile(isOperatorChar);
      # 返回操作符类型
      return "operator";
    }
    # 匹配变量名
    stream.eatWhile(/[\w\$_\xa1-\uffff]/);
    # 获取当前匹配的字符串
    var cur = stream.current();
    # 如果是关键字，返回关键字类型
    if (keywords.propertyIsEnumerable(cur)) {
      if (cur == "case" || cur == "default") curPunc = "case";
      return "keyword";
    }
    # 如果是原子类型，返回原子类型
    if (atoms.propertyIsEnumerable(cur)) return "atom";
    # 否则返回变量类型
    return "variable";
  }

  # 处理字符串的函数
  function tokenString(quote) {
    return function(stream, state) {
      var escaped = false, next, end = false;
      while ((next = stream.next()) != null) {
        if (next == quote && !escaped) {end = true; break;}
        escaped = !escaped && quote != "`" && next == "\\";
      }
      if (end || !(escaped || quote == "`"))
        state.tokenize = tokenBase;
      return "string";
    };
  }

  # 处理注释的函数
  function tokenComment(stream, state) {
    var maybeEnd = false, ch;
    while (ch = stream.next()) {
      if (ch == "/" && maybeEnd) {
        state.tokenize = tokenBase;
        break;
      }
      maybeEnd = (ch == "*");
    }
    return "comment";
  }

  # 上下文对象的构造函数
  function Context(indented, column, type, align, prev) {
    this.indented = indented;
    this.column = column;
    this.type = type;
    this.align = align;
    this.prev = prev;
  }
  # 压入新的上下文
  function pushContext(state, col, type) {
  // 返回一个新的上下文对象，用于表示代码的上下文环境
  return state.context = new Context(state.indented, col, type, null, state.context);
  }
  // 弹出当前上下文对象，返回上一个上下文对象
  function popContext(state) {
    if (!state.context.prev) return;
    var t = state.context.type;
    if (t == ")" || t == "]" || t == "}")
      state.indented = state.context.indented;
    return state.context = state.context.prev;
  }

  // 接口

  return {
    // 初始化编辑器状态
    startState: function(basecolumn) {
      return {
        tokenize: null,
        context: new Context((basecolumn || 0) - indentUnit, 0, "top", false),
        indented: 0,
        startOfLine: true
      };
    },

    // 对输入的流进行标记化处理
    token: function(stream, state) {
      var ctx = state.context;
      if (stream.sol()) {
        if (ctx.align == null) ctx.align = false;
        state.indented = stream.indentation();
        state.startOfLine = true;
        if (ctx.type == "case") ctx.type = "}";
      }
      if (stream.eatSpace()) return null;
      curPunc = null;
      // 获取标记的样式
      var style = (state.tokenize || tokenBase)(stream, state);
      if (style == "comment") return style;
      if (ctx.align == null) ctx.align = true;

      // 根据不同的标点符号进行上下文的推入和弹出
      if (curPunc == "{") pushContext(state, stream.column(), "}");
      else if (curPunc == "[") pushContext(state, stream.column(), "]");
      else if (curPunc == "(") pushContext(state, stream.column(), ")");
      else if (curPunc == "case") ctx.type = "case";
      else if (curPunc == "}" && ctx.type == "}") popContext(state);
      else if (curPunc == ctx.type) popContext(state);
      state.startOfLine = false;
      return style;
    },
    # 定义一个函数，用于处理缩进
    indent: function(state, textAfter) {
      # 如果当前状态不是在基本标记或者不是空的，则返回 CodeMirror.Pass
      if (state.tokenize != tokenBase && state.tokenize != null) return CodeMirror.Pass;
      # 获取当前上下文和文本后的第一个字符
      var ctx = state.context, firstChar = textAfter && textAfter.charAt(0);
      # 如果上下文类型是 "case" 并且文本后的字符是 "case" 或 "default"，则修改上下文类型为 "}"
      if (ctx.type == "case" && /^(?:case|default)\b/.test(textAfter)) {
        state.context.type = "}";
        return ctx.indented;
      }
      # 判断是否是闭合字符
      var closing = firstChar == ctx.type;
      # 如果上下文有对齐属性，则返回上下文列数加上（如果是闭合字符则加0，否则加1）
      if (ctx.align) return ctx.column + (closing ? 0 : 1);
      # 否则返回上下文缩进加上（如果是闭合字符则加0，否则加缩进单位）
      else return ctx.indented + (closing ? 0 : indentUnit);
    },

    # 定义电气字符
    electricChars: "{}):",
    # 定义闭合括号
    closeBrackets: "()[]{}''\"\"``",
    # 定义折叠方式
    fold: "brace",
    # 定义块注释的起始标记
    blockCommentStart: "/*",
    # 定义块注释的结束标记
    blockCommentEnd: "*/",
    # 定义行注释的标记
    lineComment: "//"
  };
# 结束对 CodeMirror 的定义，闭合大括号
});

# 定义 MIME 类型为 "text/x-go" 的语言为 Go 语言
CodeMirror.defineMIME("text/x-go", "go");

# 闭合对 CodeMirror 的定义
});
```