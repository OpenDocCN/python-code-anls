# `ZeroNet\plugins\UiFileManager\media\codemirror\mode\xml.js`

```
// CodeMirror, 版权由 Marijn Haverbeke 和其他人持有
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

var htmlConfig = {
  autoSelfClosers: {'area': true, 'base': true, 'br': true, 'col': true, 'command': true,
                    'embed': true, 'frame': true, 'hr': true, 'img': true, 'input': true,
                    'keygen': true, 'link': true, 'meta': true, 'param': true, 'source': true,
                    'track': true, 'wbr': true, 'menuitem': true},
  implicitlyClosed: {'dd': true, 'li': true, 'optgroup': true, 'option': true, 'p': true,
                     'rp': true, 'rt': true, 'tbody': true, 'td': true, 'tfoot': true,
                     'th': true, 'tr': true},
  contextGrabbers: {
    'dd': {'dd': true, 'dt': true},
    'dt': {'dd': true, 'dt': true},
    'li': {'li': true},
    'option': {'option': true, 'optgroup': true},
    'optgroup': {'optgroup': true},
    'p': {'address': true, 'article': true, 'aside': true, 'blockquote': true, 'dir': true,
          'div': true, 'dl': true, 'fieldset': true, 'footer': true, 'form': true,
          'h1': true, 'h2': true, 'h3': true, 'h4': true, 'h5': true, 'h6': true,
          'header': true, 'hgroup': true, 'hr': true, 'menu': true, 'nav': true, 'ol': true,
          'p': true, 'pre': true, 'section': true, 'table': true, 'ul': true},
    'rp': {'rp': true, 'rt': true},
    'rt': {'rp': true, 'rt': true},
    'tbody': {'tbody': true, 'tfoot': true},
    'td': {'td': true, 'th': true},
    'tfoot': {'tbody': true},
    'th': {'td': true, 'th': true},
    'thead': {'tbody': true, 'tfoot': true},
  # 定义一个包含'tr'键和值为{'tr': true}的对象
  'tr': {'tr': true}
},
# 定义一个包含"pre"键和值为true的对象
doNotIndent: {"pre": true},
# 允许未加引号的属性名
allowUnquoted: true,
# 允许缺失属性值
allowMissing: true,
# 忽略属性名的大小写
caseFold: true
}

// 定义 XML 配置对象
var xmlConfig = {
  autoSelfClosers: {},  // 自动关闭的标签
  implicitlyClosed: {},  // 隐式关闭的标签
  contextGrabbers: {},  // 上下文抓取器
  doNotIndent: {},  // 不缩进的标签
  allowUnquoted: false,  // 允许不带引号的属性值
  allowMissing: false,  // 允许缺失的标签
  allowMissingTagName: false,  // 允许缺失的标签名
  caseFold: false  // 是否忽略大小写
}

// 定义 XML 模式
CodeMirror.defineMode("xml", function(editorConf, config_) {
  var indentUnit = editorConf.indentUnit  // 缩进单位
  var config = {}  // 配置对象
  var defaults = config_.htmlMode ? htmlConfig : xmlConfig  // 默认配置
  for (var prop in defaults) config[prop] = defaults[prop]  // 将默认配置复制到配置对象
  for (var prop in config_) config[prop] = config_[prop]  // 将传入的配置参数复制到配置对象

  // 为标记器返回变量
  var type, setStyle;

  // 处于文本状态的标记器
  function inText(stream, state) {
    function chain(parser) {
      state.tokenize = parser;
      return parser(stream, state);
    }

    var ch = stream.next();
    if (ch == "<") {
      if (stream.eat("!")) {
        if (stream.eat("[")) {
          if (stream.match("CDATA[")) return chain(inBlock("atom", "]]>"));
          else return null;
        } else if (stream.match("--")) {
          return chain(inBlock("comment", "-->"));
        } else if (stream.match("DOCTYPE", true, true)) {
          stream.eatWhile(/[\w\._\-]/);
          return chain(doctype(1));
        } else {
          return null;
        }
      } else if (stream.eat("?")) {
        stream.eatWhile(/[\w\._\-]/);
        state.tokenize = inBlock("meta", "?>");
        return "meta";
      } else {
        type = stream.eat("/") ? "closeTag" : "openTag";
        state.tokenize = inTag;
        return "tag bracket";
      }
    } else if (ch == "&") {
      var ok;
      if (stream.eat("#")) {
        if (stream.eat("x")) {
          ok = stream.eatWhile(/[a-fA-F\d]/) && stream.eat(";");
        } else {
          ok = stream.eatWhile(/[\d]/) && stream.eat(";");
        }
      } else {
        ok = stream.eatWhile(/[\w\.\-:]/) && stream.eat(";");
      }
      return ok ? "atom" : "error";
    } else {
      stream.eatWhile(/[^&<]/);
      return null;
    }
  }
  inText.isInText = true;

  // 处于标签状态的标记器
  function inTag(stream, state) {
    # 从流中获取下一个字符
    var ch = stream.next();
    # 如果字符是">"或者("/"并且下一个字符是">")，则设置tokenize为inText，设置type为"endTag"或者"selfcloseTag"，返回"tag bracket"
    if (ch == ">" || (ch == "/" && stream.eat(">"))) {
      state.tokenize = inText;
      type = ch == ">" ? "endTag" : "selfcloseTag";
      return "tag bracket";
    } 
    # 如果字符是"="，则设置type为"equals"，返回null
    else if (ch == "=") {
      type = "equals";
      return null;
    } 
    # 如果字符是"<"，则设置tokenize为inText，state.state为baseState，state.tagName和state.tagStart为null，调用state.tokenize并返回结果
    else if (ch == "<") {
      state.tokenize = inText;
      state.state = baseState;
      state.tagName = state.tagStart = null;
      var next = state.tokenize(stream, state);
      return next ? next + " tag error" : "tag error";
    } 
    # 如果字符是单引号或双引号，则设置tokenize为inAttribute，并设置stringStartCol为当前列数，调用state.tokenize并返回结果
    else if (/[\'\"]/.test(ch)) {
      state.tokenize = inAttribute(ch);
      state.stringStartCol = stream.column();
      return state.tokenize(stream, state);
    } 
    # 如果字符不是空格、非断行空格、=、<、>、单引号、双引号、/，则匹配非空白字符并返回"word"
    else {
      stream.match(/^[^\s\u00a0=<>\"\']*[^\s\u00a0=<>\"\'\/]/);
      return "word";
    }
  }

  # 返回一个函数，该函数用于处理属性值的tokenize
  function inAttribute(quote) {
    var closure = function(stream, state) {
      while (!stream.eol()) {
        if (stream.next() == quote) {
          state.tokenize = inTag;
          break;
        }
      }
      return "string";
    };
    closure.isInAttribute = true;
    return closure;
  }

  # 返回一个函数，该函数用于处理块级元素的tokenize
  function inBlock(style, terminator) {
    return function(stream, state) {
      while (!stream.eol()) {
        if (stream.match(terminator)) {
          state.tokenize = inText;
          break;
        }
        stream.next();
      }
      return style;
    }
  }

  # 返回一个函数，该函数用于处理文档类型声明的tokenize
  function doctype(depth) {
    return function(stream, state) {
      var ch;
      while ((ch = stream.next()) != null) {
        if (ch == "<") {
          state.tokenize = doctype(depth + 1);
          return state.tokenize(stream, state);
        } else if (ch == ">") {
          if (depth == 1) {
            state.tokenize = inText;
            break;
          } else {
            state.tokenize = doctype(depth - 1);
            return state.tokenize(stream, state);
          }
        }
      }
      return "meta";
    };
  }

  # 定义Context类，用于存储状态上下文
  function Context(state, tagName, startOfLine) {
    this.prev = state.context;
    this.tagName = tagName;
    # 设置缩进等级为当前状态的缩进等级
    this.indent = state.indented;
    # 设置行首标记为给定的行首标记
    this.startOfLine = startOfLine;
    # 如果配置中不需要缩进的标签存在，或者当前上下文存在且不需要缩进，则设置不缩进标志为真
    if (config.doNotIndent.hasOwnProperty(tagName) || (state.context && state.context.noIndent))
      this.noIndent = true;
  }
  # 弹出当前上下文
  function popContext(state) {
    if (state.context) state.context = state.context.prev;
  }
  # 如果可能，弹出当前上下文
  function maybePopContext(state, nextTagName) {
    var parentTagName;
    while (true) {
      if (!state.context) {
        return;
      }
      parentTagName = state.context.tagName;
      # 如果当前上下文不在上下文抓取器中，或者下一个标签不在当前上下文的上下文抓取器中，则返回
      if (!config.contextGrabbers.hasOwnProperty(parentTagName) ||
          !config.contextGrabbers[parentTagName].hasOwnProperty(nextTagName)) {
        return;
      }
      # 弹出当前上下文
      popContext(state);
    }
  }

  # 基础状态函数，根据类型返回不同的状态函数
  function baseState(type, stream, state) {
    if (type == "openTag") {
      # 设置标签起始位置为当前流的列位置
      state.tagStart = stream.column();
      return tagNameState;
    } else if (type == "closeTag") {
      return closeTagNameState;
    } else {
      return baseState;
    }
  }
  # 标签名状态函数，根据类型返回不同的状态函数
  function tagNameState(type, stream, state) {
    if (type == "word") {
      # 设置当前标签名为流的当前内容，设置样式为"tag"，并返回属性状态函数
      state.tagName = stream.current();
      setStyle = "tag";
      return attrState;
    } else if (config.allowMissingTagName && type == "endTag") {
      # 如果允许缺少标签名，并且类型为"endTag"，则设置样式为"tag bracket"，并返回属性状态函数
      setStyle = "tag bracket";
      return attrState(type, stream, state);
    } else {
      # 设置样式为"error"，并返回标签名状态函数
      setStyle = "error";
      return tagNameState;
    }
  }
  # 关闭标签名状态函数，根据类型返回不同的状态函数
  function closeTagNameState(type, stream, state) {
    if (type == "word") {
      # 获取当前标签名
      var tagName = stream.current();
      # 如果当前上下文存在且当前上下文的标签名不等于当前标签名，并且当前上下文的标签名在隐式关闭配置中，则弹出当前上下文
      if (state.context && state.context.tagName != tagName &&
          config.implicitlyClosed.hasOwnProperty(state.context.tagName))
        popContext(state);
      # 如果当前上下文存在且当前上下文的标签名等于当前标签名，或者匹配关闭标签为假，则设置样式为"tag"，并返回关闭状态函数
      if ((state.context && state.context.tagName == tagName) || config.matchClosing === false) {
        setStyle = "tag";
        return closeState;
      } else {
        # 设置样式为"tag error"，并返回关闭状态错误函数
        setStyle = "tag error";
        return closeStateErr;
      }
    } else if (config.allowMissingTagName && type == "endTag") {
      # 如果允许缺少标签名，并且类型为"endTag"，则设置样式为"tag bracket"，并返回关闭状态函数
      setStyle = "tag bracket";
      return closeState(type, stream, state);
  } else {
    // 如果不是开始标签，则设置样式为错误，并返回关闭状态错误
    setStyle = "error";
    return closeStateErr;
  }
}

function closeState(type, _stream, state) {
  // 如果类型不是结束标签，则设置样式为错误，并返回关闭状态
  if (type != "endTag") {
    setStyle = "error";
    return closeState;
  }
  // 弹出当前上下文
  popContext(state);
  return baseState;
}
function closeStateErr(type, stream, state) {
  // 设置样式为错误，并返回关闭状态
  setStyle = "error";
  return closeState(type, stream, state);
}

function attrState(type, _stream, state) {
  // 如果类型是单词，则设置样式为属性，并返回属性等号状态
  if (type == "word") {
    setStyle = "attribute";
    return attrEqState;
  } else if (type == "endTag" || type == "selfcloseTag") {
    // 如果类型是结束标签或自闭合标签
    var tagName = state.tagName, tagStart = state.tagStart;
    state.tagName = state.tagStart = null;
    // 如果是自闭合标签或者自动自闭合标签，则可能弹出当前上下文
    if (type == "selfcloseTag" ||
        config.autoSelfClosers.hasOwnProperty(tagName)) {
      maybePopContext(state, tagName);
    } else {
      // 否则，可能弹出当前上下文，并创建新的上下文
      maybePopContext(state, tagName);
      state.context = new Context(state, tagName, tagStart == state.indented);
    }
    return baseState;
  }
  // 设置样式为错误，并返回属性状态
  setStyle = "error";
  return attrState;
}
function attrEqState(type, stream, state) {
  // 如果类型是等号，则返回属性值状态，否则根据配置决定是否设置样式为错误，并返回属性状态
  if (type == "equals") return attrValueState;
  if (!config.allowMissing) setStyle = "error";
  return attrState(type, stream, state);
}
function attrValueState(type, stream, state) {
  // 如果类型是字符串，则返回属性继续状态，如果类型是单词且允许未引用，则设置样式为字符串，并返回属性状态，否则设置样式为错误，并返回属性状态
  if (type == "string") return attrContinuedState;
  if (type == "word" && config.allowUnquoted) {setStyle = "string"; return attrState;}
  setStyle = "error";
  return attrState(type, stream, state);
}
function attrContinuedState(type, stream, state) {
  // 如果类型是字符串，则返回属性继续状态，否则返回属性状态
  if (type == "string") return attrContinuedState;
  return attrState(type, stream, state);
}

return {
  startState: function(baseIndent) {
    // 返回初始状态对象
    var state = {tokenize: inText,
                 state: baseState,
                 indented: baseIndent || 0,
                 tagName: null, tagStart: null,
                 context: null}
    if (baseIndent != null) state.baseIndent = baseIndent
    return state
  },
    # 定义 token 函数，用于对代码进行词法分析
    token: function(stream, state) {
      # 如果当前行没有标签名，并且是行首，则记录缩进值
      if (!state.tagName && stream.sol())
        state.indented = stream.indentation();

      # 如果当前位置是空白字符，则返回空
      if (stream.eatSpace()) return null;
      type = null;
      # 调用 state.tokenize 方法对代码进行词法分析，并返回样式
      var style = state.tokenize(stream, state);
      # 如果返回样式或者类型，并且样式不是注释
      if ((style || type) && style != "comment") {
        setStyle = null;
        # 根据返回的样式或类型，更新状态
        state.state = state.state(type || style, stream, state);
        # 如果设置了样式，则更新样式
        if (setStyle)
          style = setStyle == "error" ? style + " error" : setStyle;
      }
      # 返回样式
      return style;
    },
    # 定义一个函数，用于处理代码缩进
    indent: function(state, textAfter, fullLine) {
      var context = state.context;
      # 如果当前在属性中，则缩进到字符串开始的位置
      if (state.tokenize.isInAttribute) {
        if (state.tagStart == state.indented)
          return state.stringStartCol + 1;
        else
          return state.indented + indentUnit;
      }
      # 如果上下文存在且不需要缩进，则返回 CodeMirror.Pass
      if (context && context.noIndent) return CodeMirror.Pass;
      # 如果不在标签或文本中，则返回当前行的缩进长度，否则返回 0
      if (state.tokenize != inTag && state.tokenize != inText)
        return fullLine ? fullLine.match(/^(\s*)/)[0].length : 0;
      # 如果存在标签名，则根据配置进行缩进
      if (state.tagName) {
        if (config.multilineTagIndentPastTag !== false)
          return state.tagStart + state.tagName.length + 2;
        else
          return state.tagStart + indentUnit * (config.multilineTagIndentFactor || 1);
      }
      # 如果配置了 alignCDATA 并且匹配到 CDATA，则返回 0
      if (config.alignCDATA && /<!\[CDATA\[/.test(textAfter)) return 0;
      # 匹配到闭合标签，则根据上下文进行缩进
      var tagAfter = textAfter && /^<(\/)?([\w_:\.-]*)/.exec(textAfter);
      if (tagAfter && tagAfter[1]) { // Closing tag spotted
        while (context) {
          if (context.tagName == tagAfter[2]) {
            context = context.prev;
            break;
          } else if (config.implicitlyClosed.hasOwnProperty(context.tagName)) {
            context = context.prev;
          } else {
            break;
          }
        }
      } else if (tagAfter) { // Opening tag spotted
        while (context) {
          var grabbers = config.contextGrabbers[context.tagName];
          if (grabbers && grabbers.hasOwnProperty(tagAfter[2]))
            context = context.prev;
          else
            break;
        }
      }
      # 寻找上一个起始行并返回其缩进加上一个缩进单位
      while (context && context.prev && !context.startOfLine)
        context = context.prev;
      if (context) return context.indent + indentUnit;
      else return state.baseIndent || 0;
    },

    # 定义一个正则表达式，用于检测输入是否为闭合标签
    electricInput: /<\/[\s\w:]+>$/,
    # 定义块注释的起始标记
    blockCommentStart: "<!--",
    # 定义块注释的结束标记
    blockCommentEnd: "-->",

    # 根据配置选择处理模式
    configuration: config.htmlMode ? "html" : "xml",
    # 根据配置文件中的htmlMode属性确定helperType的取值，如果为true则为"html"，否则为"xml"
    helperType: config.htmlMode ? "html" : "xml",

    # 跳过属性值的处理，如果当前状态为attrValueState，则将状态改为attrState
    skipAttribute: function(state) {
      if (state.state == attrValueState)
        state.state = attrState
    },

    # 返回当前的XML标签信息，如果存在tagName则返回{name: state.tagName, close: state.type == "closeTag"}，否则返回null
    xmlCurrentTag: function(state) {
      return state.tagName ? {name: state.tagName, close: state.type == "closeTag"} : null
    },

    # 返回当前的XML上下文信息，遍历state.context，将tagName存在的上下文信息添加到context数组中，然后将数组反转并返回
    xmlCurrentContext: function(state) {
      var context = []
      for (var cx = state.context; cx; cx = cx.prev)
        if (cx.tagName) context.push(cx.tagName)
      return context.reverse()
    }
  };
# 定义 MIME 类型为 "text/xml" 的语法高亮为 XML
CodeMirror.defineMIME("text/xml", "xml");
# 定义 MIME 类型为 "application/xml" 的语法高亮为 XML
CodeMirror.defineMIME("application/xml", "xml");
# 如果 CodeMirror.mimeModes 对象中不包含 "text/html" 的属性
if (!CodeMirror.mimeModes.hasOwnProperty("text/html"))
  # 定义 MIME 类型为 "text/html" 的语法高亮为 XML，并启用 HTML 模式
  CodeMirror.defineMIME("text/html", {name: "xml", htmlMode: true});
```