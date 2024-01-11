# `ZeroNet\plugins\UiFileManager\media\codemirror\mode\css.js`

```
// CodeMirror, 版权归 Marijn Haverbeke 和其他人所有
// 根据 MIT 许可证分发：https://codemirror.net/LICENSE

(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
"use strict";

CodeMirror.defineMode("css", function(config, parserConfig) {
  var inline = parserConfig.inline
  if (!parserConfig.propertyKeywords) parserConfig = CodeMirror.resolveMode("text/css");

  var indentUnit = config.indentUnit,
      tokenHooks = parserConfig.tokenHooks,
      documentTypes = parserConfig.documentTypes || {},
      mediaTypes = parserConfig.mediaTypes || {},
      mediaFeatures = parserConfig.mediaFeatures || {},
      mediaValueKeywords = parserConfig.mediaValueKeywords || {},
      propertyKeywords = parserConfig.propertyKeywords || {},
      nonStandardPropertyKeywords = parserConfig.nonStandardPropertyKeywords || {},
      fontProperties = parserConfig.fontProperties || {},
      counterDescriptors = parserConfig.counterDescriptors || {},
      colorKeywords = parserConfig.colorKeywords || {},
      valueKeywords = parserConfig.valueKeywords || {},
      allowNested = parserConfig.allowNested,
      lineComment = parserConfig.lineComment,
      supportsAtComponent = parserConfig.supportsAtComponent === true;

  var type, override;
  function ret(style, tp) { type = tp; return style; }

  // Tokenizers

  function tokenBase(stream, state) {
    var ch = stream.next();
    if (tokenHooks[ch]) {
      var result = tokenHooks[ch](stream, state);
      if (result !== false) return result;
    }
    if (ch == "@") {
      stream.eatWhile(/[\w\\\-]/);
      return ret("def", stream.current());
    } else if (ch == "=" || (ch == "~" || ch == "|") && stream.eat("=")) {
      return ret(null, "compare");
    } else if (ch == "\"" || ch == "'") {
      // 如果当前字符是双引号或单引号，则设置状态为字符串标记，并返回相应的标记结果
      state.tokenize = tokenString(ch);
      return state.tokenize(stream, state);
    } else if (ch == "#") {
      // 如果当前字符是井号，则吃掉所有的字母、反斜杠和连字符，并返回哈希标记结果
      stream.eatWhile(/[\w\\\-]/);
      return ret("atom", "hash");
    } else if (ch == "!") {
      // 如果当前字符是感叹号，则匹配零个或多个空白字符和字母，并返回重要关键字标记结果
      stream.match(/^\s*\w*/);
      return ret("keyword", "important");
    } else if (/\d/.test(ch) || ch == "." && stream.eat(/\d/)) {
      // 如果当前字符是数字或者是小数点并且后面紧跟着数字，则吃掉所有的字母、数字和百分号，并返回数字单元标记结果
      stream.eatWhile(/[\w.%]/);
      return ret("number", "unit");
    } else if (ch === "-") {
      // 如果当前字符是减号
      if (/[\d.]/.test(stream.peek())) {
        // 如果下一个字符是数字或者小数点，则吃掉所有的字母、数字和百分号，并返回数字单元标记结果
        stream.eatWhile(/[\w.%]/);
        return ret("number", "unit");
      } else if (stream.match(/^-[\w\\\-]*/)) {
        // 如果下一个字符是连字符，匹配零个或多个字母、反斜杠和连字符，并返回变量定义或变量标记结果
        stream.eatWhile(/[\w\\\-]/);
        if (stream.match(/^\s*:/, false))
          return ret("variable-2", "variable-definition");
        return ret("variable-2", "variable");
      } else if (stream.match(/^\w+-/)) {
        // 如果下一个字符是字母和连字符，则返回元标记结果
        return ret("meta", "meta");
      }
    } else if (/[,+>*\/]/.test(ch)) {
      // 如果当前字符是逗号、加号、大于号、乘号或斜杠，则返回选择操作标记结果
      return ret(null, "select-op");
    } else if (ch == "." && stream.match(/^-?[_a-z][_a-z0-9-]*/i)) {
      // 如果当前字符是小数点并且匹配字母、数字、下划线和连字符的组合，则返回限定符标记结果
      return ret("qualifier", "qualifier");
    } else if (/[:;{}\[\]\(\)]/.test(ch)) {
      // 如果当前字符是冒号、分号、大括号、中括号或小括号，则返回相应的标记结果
      return ret(null, ch);
    } else if (stream.match(/[\w-.]+(?=\()/)) {
      // 如果匹配字母、数字、连字符和句点的组合并且后面紧跟着左括号，则设置状态为括号标记
      if (/^(url(-prefix)?|domain|regexp)$/.test(stream.current().toLowerCase())) {
        state.tokenize = tokenParenthesized;
      }
      return ret("variable callee", "variable");
    } else if (/[\w\\\-]/.test(ch)) {
      // 如果当前字符是字母、数字、反斜杠或连字符，则吃掉所有的字母、反斜杠和连字符，并返回属性单词标记结果
      stream.eatWhile(/[\w\\\-]/);
      return ret("property", "word");
    } else {
      // 其他情况返回空标记结果
      return ret(null, null);
    }
  }

  function tokenString(quote) {
    // 返回一个函数，用于处理字符串标记
    return function(stream, state) {
      var escaped = false, ch;
      while ((ch = stream.next()) != null) {
        if (ch == quote && !escaped) {
          if (quote == ")") stream.backUp(1);
          break;
        }
        escaped = !escaped && ch == "\\";
      }
      if (ch == quote || !escaped && quote != ")") state.tokenize = null;
      return ret("string", "string");
    }
  }
  // 结束当前函数
  };
  // 定义函数 tokenParenthesized，接受 stream 和 state 两个参数
  function tokenParenthesized(stream, state) {
    // 获取下一个字符
    stream.next(); // Must be '('
    // 如果下一个字符不是空格、双引号或右括号
    if (!stream.match(/\s*[\"\')]/, false))
      // 设置状态为 tokenString(")")
      state.tokenize = tokenString(")");
    else
      // 否则状态置空
      state.tokenize = null;
    // 返回结果
    return ret(null, "(");
  }

  // 上下文管理

  // 定义 Context 类
  function Context(type, indent, prev) {
    this.type = type;
    this.indent = indent;
    this.prev = prev;
  }

  // 推入上下文
  function pushContext(state, stream, type, indent) {
    // 创建新的上下文对象
    state.context = new Context(type, stream.indentation() + (indent === false ? 0 : indentUnit), state.context);
    // 返回类型
    return type;
  }

  // 弹出上下文
  function popContext(state) {
    if (state.context.prev)
      state.context = state.context.prev;
    return state.context.type;
  }

  // 传递
  function pass(type, stream, state) {
    return states[state.context.type](type, stream, state);
  }
  // 弹出并传递
  function popAndPass(type, stream, state, n) {
    for (var i = n || 1; i > 0; i--)
      state.context = state.context.prev;
    return pass(type, stream, state);
  }

  // 解析器

  // 将单词作为值
  function wordAsValue(stream) {
    var word = stream.current().toLowerCase();
    if (valueKeywords.hasOwnProperty(word))
      override = "atom";
    else if (colorKeywords.hasOwnProperty(word))
      override = "keyword";
    else
      override = "variable";
  }

  // 定义 states 对象
  var states = {};

  // states 对象的 top 属性
  states.top = function(type, stream, state) {
    // 如果类型为 "{"
    if (type == "{") {
      // 推入上下文，类型为 "block"
      return pushContext(state, stream, "block");
    } else if (type == "}" && state.context.prev) {
      // 弹出上下文
      return popContext(state);
    } else if (supportsAtComponent && /@component/i.test(type)) {
      // 如果支持 @component 并且类型匹配
      return pushContext(state, stream, "atComponentBlock");
    } else if (/^@(-moz-)?document$/i.test(type)) {
      // 如果类型匹配
      return pushContext(state, stream, "documentTypes");
    } else if (/^@(media|supports|(-moz-)?document|import)$/i.test(type)) {
      // 如果类型匹配
      return pushContext(state, stream, "atBlock");
    } else if (/^@(font-face|counter-style)/i.test(type)) {
      // 如果类型匹配
      state.stateArg = type;
      return "restricted_atBlock_before";
    }
  } else if (/^@(-(moz|ms|o|webkit)-)?keyframes$/i.test(type)) {
    // 如果类型是以 @ 开头，且符合 keyframes 格式，则返回 "keyframes"
    return "keyframes";
  } else if (type && type.charAt(0) == "@") {
    // 如果类型以 @ 开头，则将当前状态推入上下文栈，并返回 "at"
    return pushContext(state, stream, "at");
  } else if (type == "hash") {
    // 如果类型是 "hash"，则将 override 设置为 "builtin"
    override = "builtin";
  } else if (type == "word") {
    // 如果类型是 "word"，则将 override 设置为 "tag"
    override = "tag";
  } else if (type == "variable-definition") {
    // 如果类型是 "variable-definition"，则返回 "maybeprop"
    return "maybeprop";
  } else if (type == "interpolation") {
    // 如果类型是 "interpolation"，则将当前状态推入上下文栈，并返回 "interpolation"
    return pushContext(state, stream, "interpolation");
  } else if (type == ":") {
    // 如果类型是 ":"，则返回 "pseudo"
    return "pseudo";
  } else if (allowNested && type == "(") {
    // 如果允许嵌套且类型是 "("，则将当前状态推入上下文栈，并返回 "parens"
    return pushContext(state, stream, "parens");
  }
  // 返回当前状态的类型
  return state.context.type;
};

states.block = function(type, stream, state) {
  if (type == "word") {
    // 如果类型是 "word"
    var word = stream.current().toLowerCase();
    if (propertyKeywords.hasOwnProperty(word)) {
      // 如果 propertyKeywords 包含当前单词，则将 override 设置为 "property"，并返回 "maybeprop"
      override = "property";
      return "maybeprop";
    } else if (nonStandardPropertyKeywords.hasOwnProperty(word)) {
      // 如果 nonStandardPropertyKeywords 包含当前单词，则将 override 设置为 "string-2"，并返回 "maybeprop"
      override = "string-2";
      return "maybeprop";
    } else if (allowNested) {
      // 如果允许嵌套
      override = stream.match(/^\s*:(?:\s|$)/, false) ? "property" : "tag";
      return "block";
    } else {
      // 否则将 override 添加 "error"，并返回 "maybeprop"
      override += " error";
      return "maybeprop";
    }
  } else if (type == "meta") {
    // 如果类型是 "meta"，则返回 "block"
    return "block";
  } else if (!allowNested && (type == "hash" || type == "qualifier")) {
    // 如果不允许嵌套且类型是 "hash" 或 "qualifier"，则将 override 设置为 "error"，并返回 "block"
    override = "error";
    return "block";
  } else {
    // 其他情况下，调用 states.top 方法
    return states.top(type, stream, state);
  }
};

states.maybeprop = function(type, stream, state) {
  if (type == ":") 
    // 如果类型是 ":"，则将当前状态推入上下文栈，并返回 "prop"
    return pushContext(state, stream, "prop");
  // 其他情况下，调用 pass 方法
  return pass(type, stream, state);
};

states.prop = function(type, stream, state) {
  if (type == ";") 
    // 如果类型是 ";"，则弹出上下文栈
    return popContext(state);
  if (type == "{" && allowNested) 
    // 如果类型是 "{" 且允许嵌套，则将当前状态推入上下文栈，并返回 "propBlock"
    return pushContext(state, stream, "propBlock");
  if (type == "}" || type == "{") 
    // 如果类型是 "}" 或 "{"，则弹出上下文栈并调用 pass 方法
    return popAndPass(type, stream, state);
  if (type == "(") 
    // 如果类型是 "("，则将当前状态推入上下文栈，并返回 "parens"
    return pushContext(state, stream, "parens");
};
    # 如果类型为 "hash" 并且当前流的内容不符合哈希格式，则将 override 变量添加 " error"，表示错误
    if (type == "hash" && !/^#([0-9a-fA-f]{3,4}|[0-9a-fA-f]{6}|[0-9a-fA-f]{8})$/.test(stream.current())) {
      override += " error";
    } 
    # 如果类型为 "word"，则调用 wordAsValue 函数处理流
    else if (type == "word") {
      wordAsValue(stream);
    } 
    # 如果类型为 "interpolation"，则将当前状态切换为 "interpolation"，并返回上下文
    else if (type == "interpolation") {
      return pushContext(state, stream, "interpolation");
    }
    # 返回 "prop" 表示属性
    return "prop";
    };
    
    # 定义状态处理函数 propBlock
    states.propBlock = function(type, _stream, state) {
      # 如果类型为 "}"，则弹出上下文
      if (type == "}") return popContext(state);
      # 如果类型为 "word"，则将 override 设置为 "property"，并返回 "maybeprop"
      if (type == "word") { override = "property"; return "maybeprop"; }
      # 返回当前上下文的类型
      return state.context.type;
    };
    
    # 定义状态处理函数 parens
    states.parens = function(type, stream, state) {
      # 如果类型为 "{" 或 "}"，则弹出上下文并传递类型和流
      if (type == "{" || type == "}") return popAndPass(type, stream, state);
      # 如果类型为 ")"，则弹出上下文
      if (type == ")") return popContext(state);
      # 如果类型为 "("，则将当前状态切换为 "parens"，并推入上下文
      if (type == "(") return pushContext(state, stream, "parens");
      # 如果类型为 "interpolation"，则将当前状态切换为 "interpolation"，并推入上下文
      if (type == "interpolation") return pushContext(state, stream, "interpolation");
      # 如果类型为 "word"，则调用 wordAsValue 函数处理流
      if (type == "word") wordAsValue(stream);
      # 返回 "parens" 表示括号
      return "parens";
    };
    
    # 定义状态处理函数 pseudo
    states.pseudo = function(type, stream, state) {
      # 如果类型为 "meta"，则返回 "pseudo"
      if (type == "meta") return "pseudo";
      # 如果类型为 "word"，则将 override 设置为 "variable-3"，并返回当前上下文的类型
      if (type == "word") {
        override = "variable-3";
        return state.context.type;
      }
      # 否则传递类型、流和状态
      return pass(type, stream, state);
    };
    
    # 定义状态处理函数 documentTypes
    states.documentTypes = function(type, stream, state) {
      # 如果类型为 "word" 并且 documentTypes 中包含当前流的内容，则将 override 设置为 "tag"，并返回当前上下文的类型
      if (type == "word" && documentTypes.hasOwnProperty(stream.current())) {
        override = "tag";
        return state.context.type;
      } 
      # 否则调用 atBlock 函数处理类型、流和状态
      else {
        return states.atBlock(type, stream, state);
      }
    };
    
    # 定义状态处理函数 atBlock
    states.atBlock = function(type, stream, state) {
      # 如果类型为 "("，则将当前状态切换为 "atBlock_parens"，并推入上下文
      if (type == "(") return pushContext(state, stream, "atBlock_parens");
      # 如果类型为 "}" 或 ";"，则弹出上下文并传递类型、流和状态
      if (type == "}" || type == ";") return popAndPass(type, stream, state);
      # 如果类型为 "{"，则弹出上下文并推入新的上下文，类型为 "block" 或 "top"
      if (type == "{") return popContext(state) && pushContext(state, stream, allowNested ? "block" : "top");
      # 如果类型为 "interpolation"，则将当前状态切换为 "interpolation"，并推入上下文
      if (type == "interpolation") return pushContext(state, stream, "interpolation");
    # 如果类型为 "word"
    if (type == "word") {
      # 获取当前单词并转换为小写
      var word = stream.current().toLowerCase();
      # 如果单词是 "only"、"not"、"and"、"or"，则覆盖为 "keyword"
      if (word == "only" || word == "not" || word == "and" || word == "or")
        override = "keyword";
      # 如果单词在 mediaTypes 中，则覆盖为 "attribute"
      else if (mediaTypes.hasOwnProperty(word))
        override = "attribute";
      # 如果单词在 mediaFeatures 中，则覆盖为 "property"
      else if (mediaFeatures.hasOwnProperty(word))
        override = "property";
      # 如果单词在 mediaValueKeywords 中，则覆盖为 "keyword"
      else if (mediaValueKeywords.hasOwnProperty(word))
        override = "keyword";
      # 如果单词在 propertyKeywords 中，则覆盖为 "property"
      else if (propertyKeywords.hasOwnProperty(word))
        override = "property";
      # 如果单词在 nonStandardPropertyKeywords 中，则覆盖为 "string-2"
      else if (nonStandardPropertyKeywords.hasOwnProperty(word))
        override = "string-2";
      # 如果单词在 valueKeywords 中，则覆盖为 "atom"
      else if (valueKeywords.hasOwnProperty(word))
        override = "atom";
      # 如果单词在 colorKeywords 中，则覆盖为 "keyword"
      else if (colorKeywords.hasOwnProperty(word))
        override = "keyword";
      # 否则覆盖为 "error"
      else
        override = "error";
    }
    # 返回状态上下文的类型
    return state.context.type;
  };

  # 处理 @component 块
  states.atComponentBlock = function(type, stream, state) {
    # 如果类型为 "}"
    if (type == "}")
      # 弹出并传递
      return popAndPass(type, stream, state);
    # 如果类型为 "{"
    if (type == "{")
      # 弹出上下文并推入新上下文
      return popContext(state) && pushContext(state, stream, allowNested ? "block" : "top", false);
    # 如果类型为 "word"
    if (type == "word")
      # 覆盖为 "error"
      override = "error";
    # 返回状态上下文的类型
    return state.context.type;
  };

  # 处理块括号
  states.atBlock_parens = function(type, stream, state) {
    # 如果类型为 ")"
    if (type == ")") return popContext(state);
    # 如果类型为 "{" 或 "}"
    if (type == "{" || type == "}") return popAndPass(type, stream, state, 2);
    # 否则调用 atBlock 处理
    return states.atBlock(type, stream, state);
  };

  # 处理受限制的块之前
  states.restricted_atBlock_before = function(type, stream, state) {
    # 如果类型为 "{"
    if (type == "{")
      # 推入新上下文
      return pushContext(state, stream, "restricted_atBlock");
    # 如果类型为 "word" 并且状态参数为 "@counter-style"
    if (type == "word" && state.stateArg == "@counter-style") {
      # 覆盖为 "variable"
      override = "variable";
      return "restricted_atBlock_before";
    }
    # 否则传递
    return pass(type, stream, state);
  };

  # 处理受限制的块
  states.restricted_atBlock = function(type, stream, state) {
    # 如果类型为 "}"
    if (type == "}") {
      # 状态参数设为 null
      state.stateArg = null;
      # 弹出上下文
      return popContext(state);
    }
    # 如果类型为 "word"
    if (type == "word") {
      # 如果状态参数为 "@font-face" 并且字体属性中不包含当前单词，或者状态参数为 "@counter-style" 并且计数器描述中不包含当前单词
      if ((state.stateArg == "@font-face" && !fontProperties.hasOwnProperty(stream.current().toLowerCase())) ||
          (state.stateArg == "@counter-style" && !counterDescriptors.hasOwnProperty(stream.current().toLowerCase())))
        # 覆盖为 "error"
        override = "error";
      else
        # 否则覆盖为 "property"
        override = "property";
      # 返回 "maybeprop"
      return "maybeprop";
    }
    # 返回 "restricted_atBlock"
    return "restricted_atBlock";
  };

  # 状态为 keyframes
  states.keyframes = function(type, stream, state) {
    # 如果类型为 "word"，覆盖为 "variable"，返回 "keyframes"
    if (type == "word") { override = "variable"; return "keyframes"; }
    # 如果类型为 "{"，推入上下文为 "top"
    if (type == "{") return pushContext(state, stream, "top");
    # 否则传递类型、流和状态
    return pass(type, stream, state);
  };

  # 状态为 at
  states.at = function(type, stream, state) {
    # 如果类型为 ";"，弹出上下文
    if (type == ";") return popContext(state);
    # 如果类型为 "{" 或 "}"，弹出并传递类型、流和状态
    if (type == "{" || type == "}") return popAndPass(type, stream, state);
    # 如果类型为 "word"，覆盖为 "tag"
    if (type == "word") override = "tag";
    # 如果类型为 "hash"，覆盖为 "builtin"
    else if (type == "hash") override = "builtin";
    # 返回 "at"
    return "at";
  };

  # 状态为 interpolation
  states.interpolation = function(type, stream, state) {
    # 如果类型为 "}"，弹出上下文
    if (type == "}") return popContext(state);
    # 如果类型为 "{" 或 ";"，弹出并传递类型、流和状态
    if (type == "{" || type == ";") return popAndPass(type, stream, state);
    # 如果类型为 "word"，覆盖为 "variable"
    if (type == "word") override = "variable";
    # 如果类型不是 "variable"、"(" 或 ")"，覆盖为 "error"
    else if (type != "variable" && type != "(" && type != ")") override = "error";
    # 返回 "interpolation"
    return "interpolation";
  };

  # 返回对象
  return {
    # 开始状态
    startState: function(base) {
      # 返回对象，tokenize 为 null，状态为 "block" 或 "top"，状态参数为 null，上下文为新的 Context 对象
      return {tokenize: null,
              state: inline ? "block" : "top",
              stateArg: null,
              context: new Context(inline ? "block" : "top", base || 0, null)};
    },

    # 标记
    token: function(stream, state) {
      # 如果没有状态标记并且流吃掉空格，返回 null
      if (!state.tokenize && stream.eatSpace()) return null;
      # 样式为状态标记或 tokenBase 处理后的样式
      var style = (state.tokenize || tokenBase)(stream, state);
      # 如果样式存在并且为对象
      if (style && typeof style == "object") {
        # 类型为样式的第二个元素，样式为样式的第一个元素
        type = style[1];
        style = style[0];
      }
      # 覆盖为样式
      override = style;
      # 如果类型不为 "comment"
      if (type != "comment")
        # 状态为 states[state.state] 处理后的状态
        state.state = states[state.state](type, stream, state);
      # 返回覆盖
      return override;
    },
    # 定义一个函数，用于处理代码缩进
    indent: function(state, textAfter) {
      # 获取当前上下文和下一个字符
      var cx = state.context, ch = textAfter && textAfter.charAt(0);
      # 获取缩进值
      var indent = cx.indent;
      # 如果上下文类型是属性，并且下一个字符是"}"或")"，则将上下文切换到前一个上下文
      if (cx.type == "prop" && (ch == "}" || ch == ")")) cx = cx.prev;
      # 如果存在前一个上下文
      if (cx.prev) {
        # 如果下一个字符是"}"，并且上下文类型是"block"、"top"、"interpolation"或"restricted_atBlock"，则从父上下文恢复缩进
        if (ch == "}" && (cx.type == "block" || cx.type == "top" ||
                          cx.type == "interpolation" || cx.type == "restricted_atBlock")) {
          cx = cx.prev;
          indent = cx.indent;
        # 如果下一个字符是")"，并且上下文类型是"parens"、"atBlock_parens"，或者下一个字符是"{"，并且上下文类型是"at"或"atBlock"，则相对于当前上下文减少缩进
        } else if (ch == ")" && (cx.type == "parens" || cx.type == "atBlock_parens") ||
            ch == "{" && (cx.type == "at" || cx.type == "atBlock")) {
          indent = Math.max(0, cx.indent - indentUnit);
        }
      }
      # 返回缩进值
      return indent;
    },

    # 定义可触发自动缩进的字符
    electricChars: "}",
    # 定义块注释的起始标记
    blockCommentStart: "/*",
    # 定义块注释的结束标记
    blockCommentEnd: "*/",
    # 定义块注释的续行标记
    blockCommentContinue: " * ",
    # 定义行注释的标记
    lineComment: lineComment,
    # 定义代码折叠的标记
    fold: "brace"
  };
  // 定义一个函数，用于将数组转换为键值对，键为数组元素的小写形式，值为 true
  function keySet(array) {
    var keys = {};
    for (var i = 0; i < array.length; ++i) {
      keys[array[i].toLowerCase()] = true;
    }
    return keys;
  }

  // 定义文档类型数组和对应的键值对
  var documentTypes_ = [
    "domain", "regexp", "url", "url-prefix"
  ], documentTypes = keySet(documentTypes_);

  // 定义媒体类型数组和对应的键值对
  var mediaTypes_ = [
    "all", "aural", "braille", "handheld", "print", "projection", "screen",
    "tty", "tv", "embossed"
  ], mediaTypes = keySet(mediaTypes_);

  // 定义媒体特性数组和对应的键值对
  var mediaFeatures_ = [
    "width", "min-width", "max-width", "height", "min-height", "max-height",
    "device-width", "min-device-width", "max-device-width", "device-height",
    "min-device-height", "max-device-height", "aspect-ratio",
    "min-aspect-ratio", "max-aspect-ratio", "device-aspect-ratio",
    "min-device-aspect-ratio", "max-device-aspect-ratio", "color", "min-color",
    "max-color", "color-index", "min-color-index", "max-color-index",
    "monochrome", "min-monochrome", "max-monochrome", "resolution",
    "min-resolution", "max-resolution", "scan", "grid", "orientation",
    "device-pixel-ratio", "min-device-pixel-ratio", "max-device-pixel-ratio",
    "pointer", "any-pointer", "hover", "any-hover"
  ], mediaFeatures = keySet(mediaFeatures_);

  // 定义媒体值关键词数组和对应的键值对
  var mediaValueKeywords_ = [
    "landscape", "portrait", "none", "coarse", "fine", "on-demand", "hover",
    "interlace", "progressive"
  ], mediaValueKeywords = keySet(mediaValueKeywords_);

  // 定义属性关键词数组和对应的键值对
  var propertyKeywords_ = [
    "align-content", "align-items", "align-self", "alignment-adjust",
    "alignment-baseline", "anchor-point", "animation", "animation-delay",
    "animation-direction", "animation-duration", "animation-fill-mode",
    "animation-iteration-count", "animation-name", "animation-play-state",
    "animation-timing-function", "appearance", "azimuth", "backdrop-filter",
    "backface-visibility", "background", "background-attachment",
    "background-blend-mode", "background-clip", "background-color",
    // ... 其他属性关键词
  ];
    # CSS 属性列表，包含各种样式属性
    "background-image", "background-origin", "background-position",
    "background-position-x", "background-position-y", "background-repeat",
    "background-size", "baseline-shift", "binding", "bleed", "block-size",
    "bookmark-label", "bookmark-level", "bookmark-state", "bookmark-target",
    "border", "border-bottom", "border-bottom-color", "border-bottom-left-radius",
    "border-bottom-right-radius", "border-bottom-style", "border-bottom-width",
    "border-collapse", "border-color", "border-image", "border-image-outset",
    "border-image-repeat", "border-image-slice", "border-image-source",
    "border-image-width", "border-left", "border-left-color", "border-left-style",
    "border-left-width", "border-radius", "border-right", "border-right-color",
    "border-right-style", "border-right-width", "border-spacing", "border-style",
    "border-top", "border-top-color", "border-top-left-radius",
    "border-top-right-radius", "border-top-style", "border-top-width",
    "border-width", "bottom", "box-decoration-break", "box-shadow", "box-sizing",
    "break-after", "break-before", "break-inside", "caption-side", "caret-color",
    "clear", "clip", "color", "color-profile", "column-count", "column-fill",
    "column-gap", "column-rule", "column-rule-color", "column-rule-style",
    "column-rule-width", "column-span", "column-width", "columns", "contain",
    "content", "counter-increment", "counter-reset", "crop", "cue", "cue-after",
    "cue-before", "cursor", "direction", "display", "dominant-baseline",
    "drop-initial-after-adjust", "drop-initial-after-align",
    "drop-initial-before-adjust", "drop-initial-before-align", "drop-initial-size",
    "drop-initial-value", "elevation", "empty-cells", "fit", "fit-position",
    "flex", "flex-basis", "flex-direction", "flex-flow", "flex-grow",
    "flex-shrink", "flex-wrap", "float", "float-offset", "flow-from", "flow-into",
    "font", "font-family", "font-feature-settings", "font-kerning",
    # 定义一系列 CSS 属性名称
    "font-language-override", "font-optical-sizing", "font-size",
    "font-size-adjust", "font-stretch", "font-style", "font-synthesis",
    "font-variant", "font-variant-alternates", "font-variant-caps",
    "font-variant-east-asian", "font-variant-ligatures", "font-variant-numeric",
    "font-variant-position", "font-variation-settings", "font-weight", "gap",
    "grid", "grid-area", "grid-auto-columns", "grid-auto-flow", "grid-auto-rows",
    "grid-column", "grid-column-end", "grid-column-gap", "grid-column-start",
    "grid-gap", "grid-row", "grid-row-end", "grid-row-gap", "grid-row-start",
    "grid-template", "grid-template-areas", "grid-template-columns",
    "grid-template-rows", "hanging-punctuation", "height", "hyphens", "icon",
    "image-orientation", "image-rendering", "image-resolution", "inline-box-align",
    "inset", "inset-block", "inset-block-end", "inset-block-start", "inset-inline",
    "inset-inline-end", "inset-inline-start", "isolation", "justify-content",
    "justify-items", "justify-self", "left", "letter-spacing", "line-break",
    "line-height", "line-height-step", "line-stacking", "line-stacking-ruby",
    "line-stacking-shift", "line-stacking-strategy", "list-style",
    "list-style-image", "list-style-position", "list-style-type", "margin",
    "margin-bottom", "margin-left", "margin-right", "margin-top", "marks",
    "marquee-direction", "marquee-loop", "marquee-play-count", "marquee-speed",
    "marquee-style", "max-block-size", "max-height", "max-inline-size",
    "max-width", "min-block-size", "min-height", "min-inline-size", "min-width",
    "mix-blend-mode", "move-to", "nav-down", "nav-index", "nav-left", "nav-right",
    "nav-up", "object-fit", "object-position", "offset", "offset-anchor",
    "offset-distance", "offset-path", "offset-position", "offset-rotate",
    "opacity", "order", "orphans", "outline", "outline-color", "outline-offset",
    "outline-style", "outline-width", "overflow", "overflow-style",
    # CSS 属性列表，包括各种样式属性
    "overflow-wrap", "overflow-x", "overflow-y", "padding", "padding-bottom",
    "padding-left", "padding-right", "padding-top", "page", "page-break-after",
    "page-break-before", "page-break-inside", "page-policy", "pause",
    "pause-after", "pause-before", "perspective", "perspective-origin", "pitch",
    "pitch-range", "place-content", "place-items", "place-self", "play-during",
    "position", "presentation-level", "punctuation-trim", "quotes",
    "region-break-after", "region-break-before", "region-break-inside",
    "region-fragment", "rendering-intent", "resize", "rest", "rest-after",
    "rest-before", "richness", "right", "rotate", "rotation", "rotation-point",
    "row-gap", "ruby-align", "ruby-overhang", "ruby-position", "ruby-span",
    "scale", "scroll-behavior", "scroll-margin", "scroll-margin-block",
    "scroll-margin-block-end", "scroll-margin-block-start", "scroll-margin-bottom",
    "scroll-margin-inline", "scroll-margin-inline-end",
    "scroll-margin-inline-start", "scroll-margin-left", "scroll-margin-right",
    "scroll-margin-top", "scroll-padding", "scroll-padding-block",
    "scroll-padding-block-end", "scroll-padding-block-start",
    "scroll-padding-bottom", "scroll-padding-inline", "scroll-padding-inline-end",
    "scroll-padding-inline-start", "scroll-padding-left", "scroll-padding-right",
    "scroll-padding-top", "scroll-snap-align", "scroll-snap-type",
    "shape-image-threshold", "shape-inside", "shape-margin", "shape-outside",
    "size", "speak", "speak-as", "speak-header", "speak-numeral",
    "speak-punctuation", "speech-rate", "stress", "string-set", "tab-size",
    "table-layout", "target", "target-name", "target-new", "target-position",
    "text-align", "text-align-last", "text-combine-upright", "text-decoration",
    "text-decoration-color", "text-decoration-line", "text-decoration-skip",
    "text-decoration-skip-ink", "text-decoration-style", "text-emphasis",
    # 定义了一系列 CSS 属性关键字，包括文本样式、布局、动画等
    "text-emphasis-color", "text-emphasis-position", "text-emphasis-style",
    "text-height", "text-indent", "text-justify", "text-orientation",
    "text-outline", "text-overflow", "text-rendering", "text-shadow",
    "text-size-adjust", "text-space-collapse", "text-transform",
    "text-underline-position", "text-wrap", "top", "transform", "transform-origin",
    "transform-style", "transition", "transition-delay", "transition-duration",
    "transition-property", "transition-timing-function", "translate",
    "unicode-bidi", "user-select", "vertical-align", "visibility", "voice-balance",
    "voice-duration", "voice-family", "voice-pitch", "voice-range", "voice-rate",
    "voice-stress", "voice-volume", "volume", "white-space", "widows", "width",
    "will-change", "word-break", "word-spacing", "word-wrap", "writing-mode", "z-index",
    # SVG 特定的属性关键字
    "clip-path", "clip-rule", "mask", "enable-background", "filter", "flood-color",
    "flood-opacity", "lighting-color", "stop-color", "stop-opacity", "pointer-events",
    "color-interpolation", "color-interpolation-filters",
    "color-rendering", "fill", "fill-opacity", "fill-rule", "image-rendering",
    "marker", "marker-end", "marker-mid", "marker-start", "shape-rendering", "stroke",
    "stroke-dasharray", "stroke-dashoffset", "stroke-linecap", "stroke-linejoin",
    "stroke-miterlimit", "stroke-opacity", "stroke-width", "text-rendering",
    "baseline-shift", "dominant-baseline", "glyph-orientation-horizontal",
    "glyph-orientation-vertical", "text-anchor", "writing-mode"
    # 定义了一系列非标准的 CSS 属性关键字
    ], propertyKeywords = keySet(propertyKeywords_);
    
    var nonStandardPropertyKeywords_ = [
    "border-block", "border-block-color", "border-block-end",
    "border-block-end-color", "border-block-end-style", "border-block-end-width",
    "border-block-start", "border-block-start-color", "border-block-start-style",
    "border-block-start-width", "border-block-style", "border-block-width",
    # 定义包含 CSS 属性关键字的数组
    var cssPropertyKeywords_ = [
        "border-inline", "border-inline-color", "border-inline-end",
        "border-inline-end-color", "border-inline-end-style",
        "border-inline-end-width", "border-inline-start", "border-inline-start-color",
        "border-inline-start-style", "border-inline-start-width",
        "border-inline-style", "border-inline-width", "margin-block",
        "margin-block-end", "margin-block-start", "margin-inline", "margin-inline-end",
        "margin-inline-start", "padding-block", "padding-block-end",
        "padding-block-start", "padding-inline", "padding-inline-end",
        "padding-inline-start", "scroll-snap-stop", "scrollbar-3d-light-color",
        "scrollbar-arrow-color", "scrollbar-base-color", "scrollbar-dark-shadow-color",
        "scrollbar-face-color", "scrollbar-highlight-color", "scrollbar-shadow-color",
        "scrollbar-track-color", "searchfield-cancel-button", "searchfield-decoration",
        "searchfield-results-button", "searchfield-results-decoration", "shape-inside", "zoom"
      ], nonStandardPropertyKeywords = keySet(nonStandardPropertyKeywords_);
    
    # 将 CSS 属性关键字数组转换为关键字集合
    var cssPropertyKeywords = keySet(cssPropertyKeywords_);
    
    # 定义包含字体属性关键字的数组
    var fontProperties_ = [
        "font-display", "font-family", "src", "unicode-range", "font-variant",
         "font-feature-settings", "font-stretch", "font-weight", "font-style"
      ], fontProperties = keySet(fontProperties_);
    
    # 定义包含计数器描述符关键字的数组
    var counterDescriptors_ = [
        "additive-symbols", "fallback", "negative", "pad", "prefix", "range",
        "speak-as", "suffix", "symbols", "system"
      ], counterDescriptors = keySet(counterDescriptors_);
    
    # 定义包含颜色关键字的数组
    var colorKeywords_ = [
        "aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige",
        "bisque", "black", "blanchedalmond", "blue", "blueviolet", "brown",
        "burlywood", "cadetblue", "chartreuse", "chocolate", "coral", "cornflowerblue",
        "cornsilk", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod",
        "darkgray", "darkgreen", "darkkhaki", "darkmagenta", "darkolivegreen",
        "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen",
        # ...（此处省略了部分颜色关键字）
      ];
    # 颜色关键词列表
    [
        "darkslateblue", "darkslategray", "darkturquoise", "darkviolet",
        "deeppink", "deepskyblue", "dimgray", "dodgerblue", "firebrick",
        "floralwhite", "forestgreen", "fuchsia", "gainsboro", "ghostwhite",
        "gold", "goldenrod", "gray", "grey", "green", "greenyellow", "honeydew",
        "hotpink", "indianred", "indigo", "ivory", "khaki", "lavender",
        "lavenderblush", "lawngreen", "lemonchiffon", "lightblue", "lightcoral",
        "lightcyan", "lightgoldenrodyellow", "lightgray", "lightgreen", "lightpink",
        "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray",
        "lightsteelblue", "lightyellow", "lime", "limegreen", "linen", "magenta",
        "maroon", "mediumaquamarine", "mediumblue", "mediumorchid", "mediumpurple",
        "mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise",
        "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin",
        "navajowhite", "navy", "oldlace", "olive", "olivedrab", "orange", "orangered",
        "orchid", "palegoldenrod", "palegreen", "paleturquoise", "palevioletred",
        "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue",
        "purple", "rebeccapurple", "red", "rosybrown", "royalblue", "saddlebrown",
        "salmon", "sandybrown", "seagreen", "seashell", "sienna", "silver", "skyblue",
        "slateblue", "slategray", "snow", "springgreen", "steelblue", "tan",
        "teal", "thistle", "tomato", "turquoise", "violet", "wheat", "white",
        "whitesmoke", "yellow", "yellowgreen"
      ], 
      # 颜色关键词集合
      colorKeywords = keySet(colorKeywords_);
    
      # 值关键词列表
      var valueKeywords_ = [
        "above", "absolute", "activeborder", "additive", "activecaption", "afar",
        "after-white-space", "ahead", "alias", "all", "all-scroll", "alphabetic", "alternate",
        "always", "amharic", "amharic-abegede", "antialiased", "appworkspace",
        "arabic-indic", "armenian", "asterisks", "attr", "auto", "auto-flow", "avoid", "avoid-column", "avoid-page",
    # 一系列 CSS 属性值，用于设置元素的样式和布局
    # 以下是一些常见的 CSS 属性值，包括字体、颜色、布局等
    "avoid-region", "background", "backwards", "baseline", "below", "bidi-override", "binary",
    "bengali", "blink", "block", "block-axis", "bold", "bolder", "border", "border-box",
    "both", "bottom", "break", "break-all", "break-word", "bullets", "button", "button-bevel",
    "buttonface", "buttonhighlight", "buttonshadow", "buttontext", "calc", "cambodian",
    "capitalize", "caps-lock-indicator", "caption", "captiontext", "caret",
    "cell", "center", "checkbox", "circle", "cjk-decimal", "cjk-earthly-branch",
    "cjk-heavenly-stem", "cjk-ideographic", "clear", "clip", "close-quote",
    "col-resize", "collapse", "color", "color-burn", "color-dodge", "column", "column-reverse",
    "compact", "condensed", "contain", "content", "contents",
    "content-box", "context-menu", "continuous", "copy", "counter", "counters", "cover", "crop",
    "cross", "crosshair", "currentcolor", "cursive", "cyclic", "darken", "dashed", "decimal",
    "decimal-leading-zero", "default", "default-button", "dense", "destination-atop",
    "destination-in", "destination-out", "destination-over", "devanagari", "difference",
    "disc", "discard", "disclosure-closed", "disclosure-open", "document",
    "dot-dash", "dot-dot-dash",
    "dotted", "double", "down", "e-resize", "ease", "ease-in", "ease-in-out", "ease-out",
    "element", "ellipse", "ellipsis", "embed", "end", "ethiopic", "ethiopic-abegede",
    "ethiopic-abegede-am-et", "ethiopic-abegede-gez", "ethiopic-abegede-ti-er",
    "ethiopic-abegede-ti-et", "ethiopic-halehame-aa-er",
    "ethiopic-halehame-aa-et", "ethiopic-halehame-am-et",
    "ethiopic-halehame-gez", "ethiopic-halehame-om-et",
    "ethiopic-halehame-sid-et", "ethiopic-halehame-so-et",
    "ethiopic-halehame-ti-er", "ethiopic-halehame-ti-et", "ethiopic-halehame-tig",
    "ethiopic-numeric", "ew-resize", "exclusion", "expanded", "extends", "extra-condensed",
    # 这是一个长字符串，包含了大量的样式属性值
    # 该字符串可能用于定义 CSS 样式或其他类型的配置
    # 由于没有上下文，无法确定具体用途，只能简单描述其内容
    # 以下是一系列的 CSS 样式属性值，包括字体、颜色、布局等
    # 这些值可以用于定义 HTML 元素的外观和行为
    "mix", "mongolian", "monospace", "move", "multiple", "multiply", "myanmar", "n-resize",
    "narrower", "ne-resize", "nesw-resize", "no-close-quote", "no-drop",
    "no-open-quote", "no-repeat", "none", "normal", "not-allowed", "nowrap",
    "ns-resize", "numbers", "numeric", "nw-resize", "nwse-resize", "oblique", "octal", "opacity", "open-quote",
    "optimizeLegibility", "optimizeSpeed", "oriya", "oromo", "outset",
    "outside", "outside-shape", "overlay", "overline", "padding", "padding-box",
    "painted", "page", "paused", "persian", "perspective", "plus-darker", "plus-lighter",
    "pointer", "polygon", "portrait", "pre", "pre-line", "pre-wrap", "preserve-3d",
    "progress", "push-button", "radial-gradient", "radio", "read-only",
    "read-write", "read-write-plaintext-only", "rectangle", "region",
    "relative", "repeat", "repeating-linear-gradient",
    "repeating-radial-gradient", "repeat-x", "repeat-y", "reset", "reverse",
    "rgb", "rgba", "ridge", "right", "rotate", "rotate3d", "rotateX", "rotateY",
    "rotateZ", "round", "row", "row-resize", "row-reverse", "rtl", "run-in", "running",
    "s-resize", "sans-serif", "saturation", "scale", "scale3d", "scaleX", "scaleY", "scaleZ", "screen",
    "scroll", "scrollbar", "scroll-position", "se-resize", "searchfield",
    "searchfield-cancel-button", "searchfield-decoration",
    "searchfield-results-button", "searchfield-results-decoration", "self-start", "self-end",
    "semi-condensed", "semi-expanded", "separate", "serif", "show", "sidama",
    "simp-chinese-formal", "simp-chinese-informal", "single",
    "skew", "skewX", "skewY", "skip-white-space", "slide", "slider-horizontal",
    "slider-vertical", "sliderthumb-horizontal", "sliderthumb-vertical", "slow",
    "small", "small-caps", "small-caption", "smaller", "soft-light", "solid", "somali",
    "source-atop", "source-in", "source-out", "source-over", "space", "space-around", "space-between", "space-evenly", "spell-out", "square",
    # 定义一系列 CSS 的关键字和值
    "square-button", "start", "static", "status-bar", "stretch", "stroke", "sub",
    "subpixel-antialiased", "super", "sw-resize", "symbolic", "symbols", "system-ui", "table",
    "table-caption", "table-cell", "table-column", "table-column-group",
    "table-footer-group", "table-header-group", "table-row", "table-row-group",
    "tamil",
    "telugu", "text", "text-bottom", "text-top", "textarea", "textfield", "thai",
    "thick", "thin", "threeddarkshadow", "threedface", "threedhighlight",
    "threedlightshadow", "threedshadow", "tibetan", "tigre", "tigrinya-er",
    "tigrinya-er-abegede", "tigrinya-et", "tigrinya-et-abegede", "to", "top",
    "trad-chinese-formal", "trad-chinese-informal", "transform",
    "translate", "translate3d", "translateX", "translateY", "translateZ",
    "transparent", "ultra-condensed", "ultra-expanded", "underline", "unset", "up",
    "upper-alpha", "upper-armenian", "upper-greek", "upper-hexadecimal",
    "upper-latin", "upper-norwegian", "upper-roman", "uppercase", "urdu", "url",
    "var", "vertical", "vertical-text", "visible", "visibleFill", "visiblePainted",
    "visibleStroke", "visual", "w-resize", "wait", "wave", "wider",
    "window", "windowframe", "windowtext", "words", "wrap", "wrap-reverse", "x-large", "x-small", "xor",
    "xx-large", "xx-small"
  ], valueKeywords = keySet(valueKeywords_);

  # 将所有关键字合并成一个列表
  var allWords = documentTypes_.concat(mediaTypes_).concat(mediaFeatures_).concat(mediaValueKeywords_)
    .concat(propertyKeywords_).concat(nonStandardPropertyKeywords_).concat(colorKeywords_)
    .concat(valueKeywords_);
  # 在 CodeMirror 中注册 CSS 的关键字
  CodeMirror.registerHelper("hintWords", "css", allWords);

  # 定义处理 CSS 注释的函数
  function tokenCComment(stream, state) {
    var maybeEnd = false, ch;
    while ((ch = stream.next()) != null) {
      if (maybeEnd && ch == "/") {
        state.tokenize = null;
        break;
      }
      maybeEnd = (ch == "*");
    }
    return ["comment", "comment"];
  }

  # 定义 CSS 的 MIME 类型
  CodeMirror.defineMIME("text/css", {
    documentTypes: documentTypes,
    mediaTypes: mediaTypes,  # 媒体类型
    mediaFeatures: mediaFeatures,  # 媒体特性
    mediaValueKeywords: mediaValueKeywords,  # 媒体值关键词
    propertyKeywords: propertyKeywords,  # 属性关键词
    nonStandardPropertyKeywords: nonStandardPropertyKeywords,  # 非标准属性关键词
    fontProperties: fontProperties,  # 字体属性
    counterDescriptors: counterDescriptors,  # 计数器描述符
    colorKeywords: colorKeywords,  # 颜色关键词
    valueKeywords: valueKeywords,  # 值关键词
    tokenHooks: {  # 标记钩子
      "/": function(stream, state) {  # 当遇到 "/" 时的处理函数
        if (!stream.eat("*")) return false;  # 如果下一个字符不是 "*"，则返回 false
        state.tokenize = tokenCComment;  # 设置状态的标记为 tokenCComment
        return tokenCComment(stream, state);  # 调用 tokenCComment 处理函数
      }
    },
    name: "css"  # 名称为 "css"
  });

  CodeMirror.defineMIME("text/x-scss", {  # 定义 MIME 类型为 "text/x-scss"
    mediaTypes: mediaTypes,  # 媒体类型
    mediaFeatures: mediaFeatures,  # 媒体特性
    mediaValueKeywords: mediaValueKeywords,  # 媒体值关键词
    propertyKeywords: propertyKeywords,  # 属性关键词
    nonStandardPropertyKeywords: nonStandardPropertyKeywords,  # 非标准属性关键词
    colorKeywords: colorKeywords,  # 颜色关键词
    valueKeywords: valueKeywords,  # 值关键词
    fontProperties: fontProperties,  # 字体属性
    allowNested: true,  # 允许嵌套
    lineComment: "//",  # 行注释为 "//"
    tokenHooks: {  # 标记钩子
      "/": function(stream, state) {  # 当遇到 "/" 时的处理函数
        if (stream.eat("/")) {  # 如果下一个字符是 "/"
          stream.skipToEnd();  # 跳过直到行尾
          return ["comment", "comment"];  # 返回注释类型
        } else if (stream.eat("*")) {  # 如果下一个字符是 "*"
          state.tokenize = tokenCComment;  # 设置状态的标记为 tokenCComment
          return tokenCComment(stream, state);  # 调用 tokenCComment 处理函数
        } else {  # 其他情况
          return ["operator", "operator"];  # 返回操作符类型
        }
      },
      ":": function(stream) {  # 当遇到 ":" 时的处理函数
        if (stream.match(/\s*\{/, false))  # 如果匹配到空格和 "{"
          return [null, null]  # 返回空
        return false;  # 返回 false
      },
      "$": function(stream) {  # 当遇到 "$" 时的处理函数
        stream.match(/^[\w-]+/);  # 匹配字母、数字、下划线、连字符
        if (stream.match(/^\s*:/, false))  # 如果匹配到空格和 ":"
          return ["variable-2", "variable-definition"];  # 返回变量定义类型
        return ["variable-2", "variable"];  # 返回变量类型
      },
      "#": function(stream) {  # 当遇到 "#" 时的处理函数
        if (!stream.eat("{")) return false;  # 如果下一个字符不是 "{"
        return [null, "interpolation"];  # 返回空和插值类型
      }
    },
    name: "css",  # 名称为 "css"
    helperType: "scss"  # 帮助类型为 "scss"
  });

  CodeMirror.defineMIME("text/x-less", {  # 定义 MIME 类型为 "text/x-less"
    mediaTypes: mediaTypes,  # 媒体类型
    mediaFeatures: mediaFeatures,  # 媒体特性
    mediaValueKeywords: mediaValueKeywords,  // 媒体值关键词
    propertyKeywords: propertyKeywords,  // 属性关键词
    nonStandardPropertyKeywords: nonStandardPropertyKeywords,  // 非标准属性关键词
    colorKeywords: colorKeywords,  // 颜色关键词
    valueKeywords: valueKeywords,  // 值关键词
    fontProperties: fontProperties,  // 字体属性
    allowNested: true,  // 允许嵌套
    lineComment: "//",  // 单行注释符号
    tokenHooks: {  // 标记钩子
      "/": function(stream, state) {  // 当遇到 "/" 时执行的函数
        if (stream.eat("/")) {  // 如果下一个字符是 "/"，则为单行注释
          stream.skipToEnd();  // 跳过注释内容
          return ["comment", "comment"];  // 返回注释类型
        } else if (stream.eat("*")) {  // 如果下一个字符是 "*"，则为多行注释
          state.tokenize = tokenCComment;  // 设置状态的标记为多行注释
          return tokenCComment(stream, state);  // 执行多行注释的函数
        } else {
          return ["operator", "operator"];  // 返回操作符类型
        }
      },
      "@": function(stream) {  // 当遇到 "@" 时执行的函数
        if (stream.eat("{")) return [null, "interpolation"];  // 如果下一个字符是 "{"，则为插值
        if (stream.match(/^(charset|document|font-face|import|(-(moz|ms|o|webkit)-)?keyframes|media|namespace|page|supports)\b/i, false)) return false;  // 匹配关键字
        stream.eatWhile(/[\w\\\-]/);  // 匹配字母、数字、下划线和连字符
        if (stream.match(/^\s*:/, false))  // 如果匹配到冒号
          return ["variable-2", "variable-definition"];  // 返回变量定义类型
        return ["variable-2", "variable"];  // 返回变量类型
      },
      "&": function() {  // 当遇到 "&" 时执行的函数
        return ["atom", "atom"];  // 返回原子类型
      }
    },
    name: "css",  // 设置名称为 CSS
    helperType: "less"  // 设置帮助类型为 less
  });

  CodeMirror.defineMIME("text/x-gss", {  // 定义 MIME 类型为 text/x-gss
    documentTypes: documentTypes,  // 文档类型
    mediaTypes: mediaTypes,  // 媒体类型
    mediaFeatures: mediaFeatures,  // 媒体特性
    propertyKeywords: propertyKeywords,  // 属性关键词
    nonStandardPropertyKeywords: nonStandardPropertyKeywords,  // 非标准属性关键词
    fontProperties: fontProperties,  // 字体属性
    counterDescriptors: counterDescriptors,  // 计数器描述
    colorKeywords: colorKeywords,  // 颜色关键词
    valueKeywords: valueKeywords,  // 值关键词
    supportsAtComponent: true,  // 支持 at 组件
    tokenHooks: {  // 标记钩子
      "/": function(stream, state) {  // 当遇到 "/" 时执行的函数
        if (!stream.eat("*")) return false;  // 如果下一个字符不是 "*"，则返回 false
        state.tokenize = tokenCComment;  // 设置状态的标记为多行注释
        return tokenCComment(stream, state);  // 执行多行注释的函数
      }
    },
    name: "css",  // 设置名称为 CSS
    helperType: "gss"  // 设置帮助类型为 gss
  });
# 闭合了一个代码块或者函数的结束
```