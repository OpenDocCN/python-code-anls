# `ZeroNet\plugins\UiFileManager\media\codemirror\mode\coffeescript.js`

```
// CodeMirror, 版权由 Marijn Haverbeke 和其他人所有
// 根据 MIT 许可证分发：https://codemirror.net/LICENSE

/**
 * 链接到项目的 GitHub 页面：
 * https://github.com/pickhardt/coffeescript-codemirror-mode
 */
(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
"use strict";

CodeMirror.defineMode("coffeescript", function(conf, parserConf) {
  var ERRORCLASS = "error";

  function wordRegexp(words) {
    return new RegExp("^((" + words.join(")|(") + "))\\b");
  }

  var operators = /^(?:->|=>|\+[+=]?|-[\-=]?|\*[\*=]?|\/[\/=]?|[=!]=|<[><]?=?|>>?=?|%=?|&=?|\|=?|\^=?|\~|!|\?|(or|and|\|\||&&|\?)=)/;
  var delimiters = /^(?:[()\[\]{},:`=;]|\.\.?\.?)/;
  var identifiers = /^[_A-Za-z$][_A-Za-z$0-9]*/;
  var atProp = /^@[_A-Za-z$][_A-Za-z$0-9]*/;

  var wordOperators = wordRegexp(["and", "or", "not",
                                  "is", "isnt", "in",
                                  "instanceof", "typeof"]);
  var indentKeywords = ["for", "while", "loop", "if", "unless", "else",
                        "switch", "try", "catch", "finally", "class"];
  var commonKeywords = ["break", "by", "continue", "debugger", "delete",
                        "do", "in", "of", "new", "return", "then",
                        "this", "@", "throw", "when", "until", "extends"];

  var keywords = wordRegexp(indentKeywords.concat(commonKeywords));

  indentKeywords = wordRegexp(indentKeywords);


  var stringPrefixes = /^('{3}|\"{3}|['\"])/;
  var regexPrefixes = /^(\/{3}|\/)/;
  var commonConstants = ["Infinity", "NaN", "undefined", "null", "true", "false", "on", "off", "yes", "no"];
  var constants = wordRegexp(commonConstants);

  // Tokenizers
  function tokenBase(stream, state) {
    // 处理作用域变化
    if (stream.sol()) {
      // 如果当前行是起始行，检查作用域对齐情况并设置偏移量
      if (state.scope.align === null) state.scope.align = false;
      var scopeOffset = state.scope.offset;
      // 如果当前行有空格，则检查缩进情况
      if (stream.eatSpace()) {
        var lineOffset = stream.indentation();
        // 如果当前行缩进大于作用域偏移量且作用域类型为 "coffee"，则返回缩进
        if (lineOffset > scopeOffset && state.scope.type == "coffee") {
          return "indent";
        } else if (lineOffset < scopeOffset) {
          return "dedent";
        }
        return null;
      } else {
        // 如果当前行没有空格且作用域偏移量大于 0，则执行取消缩进操作
        if (scopeOffset > 0) {
          dedent(stream, state);
        }
      }
    }
    // 如果当前行有空格，则返回空
    if (stream.eatSpace()) {
      return null;
    }

    var ch = stream.peek();

    // 处理 docco 标题注释（单行）
    if (stream.match("####")) {
      // 跳过当前行剩余内容
      stream.skipToEnd();
      return "comment";
    }

    // 处理多行注释
    if (stream.match("###")) {
      // 设置状态的 tokenize 属性为 longComment 函数，并执行该函数
      state.tokenize = longComment;
      return state.tokenize(stream, state);
    }

    // 单行注释
    if (ch === "#") {
      // 跳过当前行剩余内容
      stream.skipToEnd();
      return "comment";
    }

    // 处理数字字面量
    # 如果流匹配到以数字或小数点开头的字符
    if (stream.match(/^-?[0-9\.]/, false)) {
      # 声明一个变量来标记是否为浮点数
      var floatLiteral = false;
      # 匹配浮点数
      if (stream.match(/^-?\d*\.\d+(e[\+\-]?\d+)?/i)) {
        floatLiteral = true;
      }
      if (stream.match(/^-?\d+\.\d*/)) {
        floatLiteral = true;
      }
      if (stream.match(/^-?\.\d+/)) {
        floatLiteral = true;
      }

      if (floatLiteral) {
        # 防止出现额外的小数点，例如 1..
        if (stream.peek() == "."){
          stream.backUp(1);
        }
        return "number";
      }
      # 声明一个变量来标记是否为整数
      var intLiteral = false;
      # 匹配十六进制数
      if (stream.match(/^-?0x[0-9a-f]+/i)) {
        intLiteral = true;
      }
      # 匹配十进制数
      if (stream.match(/^-?[1-9]\d*(e[\+\-]?\d+)?/)) {
        intLiteral = true;
      }
      # 匹配零
      if (stream.match(/^-?0(?![\dx])/i)) {
        intLiteral = true;
      }
      if (intLiteral) {
        return "number";
      }
    }

    # 处理字符串
    if (stream.match(stringPrefixes)) {
      state.tokenize = tokenFactory(stream.current(), false, "string");
      return state.tokenize(stream, state);
    }
    # 处理正则表达式
    if (stream.match(regexPrefixes)) {
      if (stream.current() != "/" || stream.match(/^.*\//, false)) { # 防止除法运算符被高亮显示
        state.tokenize = tokenFactory(stream.current(), true, "string-2");
        return state.tokenize(stream, state);
      } else {
        stream.backUp(1);
      }
    }

    # 处理运算符和分隔符
    if (stream.match(operators) || stream.match(wordOperators)) {
      return "operator";
    }
    if (stream.match(delimiters)) {
      return "punctuation";
    }

    # 匹配常量
    if (stream.match(constants)) {
      return "atom";
    }

    # 匹配属性
    if (stream.match(atProp) || state.prop && stream.match(identifiers)) {
      return "property";
    }

    # 匹配关键字
    if (stream.match(keywords)) {
      return "keyword";
    }
    // 如果流匹配标识符，则返回"variable"
    if (stream.match(identifiers)) {
      return "variable";
    }

    // 处理未检测到的项
    stream.next();
    return ERRORCLASS;
  }

  // 创建一个函数，用于生成特定类型的标记
  function tokenFactory(delimiter, singleline, outclass) {
    return function(stream, state) {
      while (!stream.eol()) {
        // 吃掉除了特定字符之外的所有字符
        stream.eatWhile(/[^'"\/\\]/);
        if (stream.eat("\\")) {
          stream.next();
          if (singleline && stream.eol()) {
            return outclass;
          }
        } else if (stream.match(delimiter)) {
          state.tokenize = tokenBase;
          return outclass;
        } else {
          stream.eat(/['"\/]/);
        }
      }
      if (singleline) {
        if (parserConf.singleLineStringErrors) {
          outclass = ERRORCLASS;
        } else {
          state.tokenize = tokenBase;
        }
      }
      return outclass;
    };
  }

  // 处理长注释
  function longComment(stream, state) {
    while (!stream.eol()) {
      // 吃掉除了"#"之外的所有字符
      stream.eatWhile(/[^#]/);
      if (stream.match("###")) {
        state.tokenize = tokenBase;
        break;
      }
      stream.eatWhile("#");
    }
    return "comment";
  }

  // 缩进处理
  function indent(stream, state, type) {
    type = type || "coffee";
    var offset = 0, align = false, alignOffset = null;
    for (var scope = state.scope; scope; scope = scope.prev) {
      if (scope.type === "coffee" || scope.type == "}") {
        offset = scope.offset + conf.indentUnit;
        break;
      }
    }
    if (type !== "coffee") {
      align = null;
      alignOffset = stream.column() + stream.current().length;
    } else if (state.scope.align) {
      state.scope.align = false;
    }
    state.scope = {
      offset: offset,
      type: type,
      prev: state.scope,
      align: align,
      alignOffset: alignOffset
    };
  }

  // 减少缩进处理
  function dedent(stream, state) {
    if (!state.scope.prev) return;
    // 如果当前作用域类型为咖啡因（coffee），则执行以下操作
    if (state.scope.type === "coffee") {
      // 获取当前缩进值
      var _indent = stream.indentation();
      // 初始化匹配标志
      var matched = false;
      // 遍历作用域链
      for (var scope = state.scope; scope; scope = scope.prev) {
        // 如果当前缩进值等于作用域偏移量，则设置匹配标志为true，并跳出循环
        if (_indent === scope.offset) {
          matched = true;
          break;
        }
      }
      // 如果没有匹配到作用域，则返回true
      if (!matched) {
        return true;
      }
      // 循环直到作用域链中的前一个作用域的偏移量等于当前缩进值
      while (state.scope.prev && state.scope.offset !== _indent) {
        state.scope = state.scope.prev;
      }
      // 返回false
      return false;
    } else {
      // 如果当前作用域类型不是咖啡因（coffee），则将作用域设置为前一个作用域
      state.scope = state.scope.prev;
      // 返回false
      return false;
    }
  }

  // tokenLexer函数用于处理token
  function tokenLexer(stream, state) {
    // 调用tokenize方法获取样式
    var style = state.tokenize(stream, state);
    // 获取当前token
    var current = stream.current();

    // 处理作用域变化
    if (current === "return") {
      state.dedent = true;
    }
    if (((current === "->" || current === "=>") && stream.eol())
        || style === "indent") {
      // 调用indent函数处理缩进
      indent(stream, state);
    }
    // 获取当前token在"[({"中的索引
    var delimiter_index = "[({".indexOf(current);
    if (delimiter_index !== -1) {
      // 调用indent函数处理缩进
      indent(stream, state, "])}".slice(delimiter_index, delimiter_index+1));
    }
    // 如果当前token匹配indentKeywords的正则表达式，则调用indent函数处理缩进
    if (indentKeywords.exec(current)){
      indent(stream, state);
    }
    // 如果当前token为"then"，则调用dedent函数处理减少缩进
    if (current == "then"){
      dedent(stream, state);
    }

    // 如果样式为"dedent"，则调用dedent函数处理减少缩进
    if (style === "dedent") {
      if (dedent(stream, state)) {
        return ERRORCLASS;
      }
    }
    // 获取当前token在"()}"中的索引
    delimiter_index = "])}".indexOf(current);
    if (delimiter_index !== -1) {
      // 循环直到作用域类型为咖啡因（coffee）且前一个作用域存在
      while (state.scope.type == "coffee" && state.scope.prev)
        state.scope = state.scope.prev;
      // 如果当前作用域类型等于当前token，则将作用域设置为前一个作用域
      if (state.scope.type == current)
        state.scope = state.scope.prev;
    }
    // 如果需要减少缩进且当前行结束，则执行以下操作
    if (state.dedent && stream.eol()) {
      // 如果作用域类型为咖啡因（coffee）且前一个作用域存在，则将作用域设置为前一个作用域
      if (state.scope.type == "coffee" && state.scope.prev)
        state.scope = state.scope.prev;
      // 将减少缩进标志设置为false
      state.dedent = false;
    }

    // 返回样式
    return style;
  }

  var external = {
    # 定义起始状态的函数，接受一个基础列参数
    startState: function(basecolumn) {
      # 返回一个对象，包含tokenize、scope、prop和dedent属性
      return {
        tokenize: tokenBase,  # 设置tokenize属性为tokenBase函数
        scope: {offset:basecolumn || 0, type:"coffee", prev: null, align: false},  # 设置scope属性为包含offset、type、prev和align属性的对象
        prop: false,  # 设置prop属性为false
        dedent: 0  # 设置dedent属性为0
      };
    },

    # 定义token函数，接受stream和state两个参数
    token: function(stream, state) {
      # 声明并初始化fillAlign变量，根据scope.align是否为null和scope的值
      var fillAlign = state.scope.align === null && state.scope;
      # 如果fillAlign为true并且stream在行首，则将fillAlign.align设置为false
      if (fillAlign && stream.sol()) fillAlign.align = false;

      # 调用tokenLexer函数，传入stream和state参数，返回style样式
      var style = tokenLexer(stream, state);
      # 如果style存在并且不是"comment"样式
      if (style && style != "comment") {
        # 如果fillAlign存在，则将fillAlign.align设置为true
        if (fillAlign) fillAlign.align = true;
        # 如果style为"punctuation"并且stream当前字符为"."，则将state.prop设置为true
        state.prop = style == "punctuation" && stream.current() == "."
      }

      # 返回style样式
      return style;
    },

    # 定义indent函数，接受state和text两个参数
    indent: function(state, text) {
      # 如果state.tokenize不等于tokenBase函数，则返回0
      if (state.tokenize != tokenBase) return 0;
      # 声明并初始化scope变量为state.scope
      var scope = state.scope;
      # 声明并初始化closer变量，根据text是否存在以及text的第一个字符
      var closer = text && "])}".indexOf(text.charAt(0)) > -1;
      # 如果closer为true并且scope.type为"coffee"并且scope.prev存在，则循环更新scope为scope.prev
      if (closer) while (scope.type == "coffee" && scope.prev) scope = scope.prev;
      # 声明并初始化closes变量，根据closer是否为true以及scope.type是否等于text的第一个字符
      var closes = closer && scope.type === text.charAt(0);
      # 如果scope.align存在，则返回scope.alignOffset减去（如果closes为true则减1，否则减0）
      if (scope.align)
        return scope.alignOffset - (closes ? 1 : 0);
      # 否则返回（如果closes为true则返回scope.prev，否则返回scope）的offset属性
      else
        return (closes ? scope.prev : scope).offset;
    },

    # 设置lineComment属性为"#"
    lineComment: "#",
    # 设置fold属性为"indent"
    fold: "indent"
  };
  # 返回external变量
  return external;
// 定义 MIME 类型为 "application/vnd.coffeescript" 的媒体类型为 "coffeescript"
CodeMirror.defineMIME("application/vnd.coffeescript", "coffeescript");

// 定义 MIME 类型为 "text/x-coffeescript" 的媒体类型为 "coffeescript"
CodeMirror.defineMIME("text/x-coffeescript", "coffeescript");

// 定义 MIME 类型为 "text/coffeescript" 的媒体类型为 "coffeescript"
CodeMirror.defineMIME("text/coffeescript", "coffeescript");
```